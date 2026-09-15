# Author: even
"""Deterministic lowering from FFT stage IR to Ascend C source.

emit validates the contract and realization, constructs the stage IR, then
generates butterfly kernels, their host launch sequence and CMake configuration.
The serialized ir.json records the exact stages used for generation. Native
compilation and target measurement are separate responsibilities of the runners.
"""

from .config import workspace, remote_root, npu_environment, connect_ssh
import json
import math
from pathlib import Path
from .fft_ir import Contract, identity, validate_action


def emit(c, a, directory, mode="pipeline", catalog_steps=None):
    """Emit one realization, or the internally constructed shared kernel catalog."""
    validate_action(c, a)
    if mode not in ("pipeline", "forward", "inverse"):
        raise ValueError("invalid FFT mode")
    if min(c.shape) < 4 or math.prod(c.shape) * c.batch % 32:
        raise ValueError("DMA lowering requires axes >=4 and total divisible by 32")
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if catalog_steps is None:
        digest, ir = identity(c, a, mode)
    else:
        from dataclasses import asdict
        import hashlib

        ir = dict(contract=asdict(c), mode="kernel_catalog", stages=catalog_steps)
        digest = hashlib.sha256(json.dumps(ir, sort_keys=True).encode()).hexdigest()
    (d / "ir.json").write_text(json.dumps(dict(sha256=digest, **ir), indent=2))
    total = math.prod(c.shape) * c.batch
    prefix = '#include "kernel_operator.h"\nusing namespace AscendC;\n'
    kernels = []
    launches = []
    includes = []
    for idx, step in enumerate(ir["stages"]):
        name = "fft_step_%d" % idx
        includes.append('#include "aclrtlaunch_%s.h"' % name)
        launches.append(
            'ck(ACLRT_LAUNCH_KERNEL(%s)(%d,s,src,h,dst),"launch"); src=dst; dst=(dst==b?c:b);'
            % (name, a["cores"])
        )
        start = (
            'extern "C" __global__ __aicore__ void %s(GM_ADDR input,GM_ADDR weight,GM_ADDR output){\n' % name
        )
        start += "GlobalTensor<float> x,h,y; x.SetGlobalBuffer((__gm__ float*)input); h.SetGlobalBuffer((__gm__ float*)weight); y.SetGlobalBuffer((__gm__ float*)output);\n"
        if step["kind"] != "fft":
            body = "TPipe pipe; TBuf<TPosition::VECCALC> vb,hb; pipe.InitBuffer(vb,256);pipe.InitBuffer(hb,256); auto v=vb.Get<float>();auto w=hb.Get<float>();\n"
            body += (
                "for(int base=GetBlockIdx()*32;base<%d;base+=GetBlockNum()*32){DataCopy(v,x[2*base],64);"
                % total
            )
            if step["kind"] == "multiply":
                body += "DataCopy(w,h[2*base],64);"
            body += (
                "PipeBarrier<PIPE_ALL>();for(int k=0;k<32;k++){float r=v.GetValue(2*k),i=v.GetValue(2*k+1);"
            )
            if step["kind"] == "multiply":
                body += "float hr=w.GetValue(2*k),hi=w.GetValue(2*k+1);float rr=r*hr-i*hi;i=r*hi+i*hr;r=rr;"
            else:
                body += "r*=%.12ef;i*=%.12ef;" % (1 / math.prod(c.shape), 1 / math.prod(c.shape))
            body += "v.SetValue(2*k,r);v.SetValue(2*k+1,i);}PipeBarrier<PIPE_ALL>();DataCopy(y[2*base],v,64);PipeBarrier<PIPE_ALL>();}}\n"
            kernels.append(start + body)
            continue
        axis = step["axis"]
        n = c.shape[axis]
        stride = math.prod(c.shape[axis + 1 :])
        lines = total // n
        bits = int(math.log2(n))
        lanes = min(stride, step.get("pack", a.get("pack", 8)))
        count = 2 * n * lanes
        body = (
            "TPipe pipe; TBuf<TPosition::VECCALC> vb,rb,hb;pipe.InitBuffer(vb,%d);pipe.InitBuffer(rb,%d);pipe.InitBuffer(hb,%d);auto v=vb.Get<float>();auto raw=rb.Get<float>();auto w=hb.Get<float>();\n"
            % (count * 4, count * 4, count * 4)
        )
        body += (
            "for(int line=GetBlockIdx()*%d;line<%d;line+=GetBlockNum()*%d){int base=(line/%d)*%d+line%%%d;\n"
            % (lanes, lines, lanes, stride, n * stride, stride)
        )
        if stride == 1:
            body += "DataCopy(raw,x[2*base],%d);" % count
            if step["multiply"]:
                body += "DataCopy(w,h[2*base],%d);" % count
        else:
            body += "for(int k=0;k<%d;k++){DataCopy(raw[k*%d],x[2*(base+k*%d)],%d);" % (
                n,
                2 * lanes,
                stride,
                2 * lanes,
            )
            if step["multiply"]:
                body += "DataCopy(w[k*%d],h[2*(base+k*%d)],%d);" % (2 * lanes, stride, 2 * lanes)
            body += "}"
        body += "PipeBarrier<PIPE_ALL>();for(int k=0;k<%d;k++){int p=k;" % n
        if step["first"] == 0:
            body += "int rev=0;for(int b=0;b<%d;b++){rev=(rev<<1)|(p&1);p>>=1;}p=rev;" % bits
        body += (
            "for(int l=0;l<%d;l++){v.SetValue(2*(k*%d+l),raw.GetValue(2*(p*%d+l)));v.SetValue(2*(k*%d+l)+1,raw.GetValue(2*(p*%d+l)+1));}}\n"
            % (lanes, lanes, lanes, lanes, lanes)
        )
        for level in range(step["first"], step["last"]):
            m = 2 ** (level + 1)
            half = m // 2
            for j in range(half):
                angle = (1 if step["inverse"] else -1) * 2 * math.pi * j / m

                def f(v):
                    return "(" + ("%.12e" % v) + "f)"

                wr = f(math.cos(angle))
                wi = f(math.sin(angle))
                body += (
                    "for(int b=0;b<%d;b+=%d)for(int l=0;l<%d;l++){int u=2*((b+%d)*%d+l),q=u+%d;float ar=v.GetValue(u),ai=v.GetValue(u+1),br=v.GetValue(q),bi=v.GetValue(q+1);float tr=%s*br-%s*bi,ti=%s*bi+%s*br;v.SetValue(u,ar+tr);v.SetValue(u+1,ai+ti);v.SetValue(q,ar-tr);v.SetValue(q+1,ai-ti);}\n"
                    % (n, m, lanes, j, lanes, 2 * half * lanes, wr, wi, wr, wi)
                )
        if step["multiply"] or step["scale"]:
            body += "for(int k=0;k<%d;k++){float rr=v.GetValue(2*k),ii=v.GetValue(2*k+1);" % (n * lanes)
            if step["multiply"]:
                body += "float hr=w.GetValue(2*k),hi=w.GetValue(2*k+1),tmp=rr*hr-ii*hi;ii=rr*hi+ii*hr;rr=tmp;"
            if step["scale"]:
                body += "rr*=%.12ef;ii*=%.12ef;" % (1 / math.prod(c.shape), 1 / math.prod(c.shape))
            body += "v.SetValue(2*k,rr);v.SetValue(2*k+1,ii);}"
        body += "PipeBarrier<PIPE_ALL>();"
        if stride == 1:
            body += "DataCopy(y[2*base],v,%d);" % count
        else:
            body += "for(int k=0;k<%d;k++){DataCopy(y[2*(base+k*%d)],v[k*%d],%d);}" % (
                n,
                stride,
                2 * lanes,
                2 * lanes,
            )
        body += "PipeBarrier<PIPE_ALL>();}}\n"
        kernels.append(start + body)
    (d / "kernels.cpp").write_text(prefix + "\n".join(kernels))
    host = (
        HOST.replace("@INCLUDES@", "\n".join(includes))
        .replace("@TOTAL@", str(total))
        .replace("@LAUNCHES@", "\n".join(launches))
    )
    (d / "main.cpp").write_text(host)
    cmake = Path(__file__).with_name("CMakeLists.txt").read_text()
    (d / "CMakeLists.txt").write_text(cmake)
    return ir


HOST = r"""
#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
@INCLUDES@
void ck(aclError e,const char* s){if(e){fprintf(stderr,"ACL %d %s\n",e,s);exit(2);}}
std::vector<float> read(const std::string& path){std::vector<float> v(@TOTAL@*2); std::ifstream f(path,std::ios::binary);if(!f.read((char*)v.data(),v.size()*4)){fprintf(stderr,"bad input %s\n",path.c_str());exit(3);}return v;}
int main(int argc,char**argv){if(argc<3)return 1;
auto xh=read(std::string(argv[1])+"/x.bin"),hh=read(std::string(argv[1])+"/h.bin");
ck(aclInit(nullptr),"init");ck(aclrtSetDevice(0),"device");aclrtStream s;ck(aclrtCreateStream(&s),"stream");
const size_t bytes=@TOTAL@*8;void *x,*h,*b,*c;
ck(aclrtMalloc(&x,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc");ck(aclrtMalloc(&h,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc");ck(aclrtMalloc(&b,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc");ck(aclrtMalloc(&c,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc");
ck(aclrtMemcpy(x,bytes,xh.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE),"copy");ck(aclrtMemcpy(h,bytes,hh.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE),"copy");
void* result=nullptr;
auto run=[&](){void* src=x;void* dst=b;
// Ping-pong scratch buffers; the input remains immutable between repetitions.
@LAUNCHES@
result=src;
};
run();ck(aclrtSynchronizeStream(s),"check sync");std::vector<float> out(@TOTAL@*2);ck(aclrtMemcpy(out.data(),bytes,result,bytes,ACL_MEMCPY_DEVICE_TO_HOST),"output");std::ofstream f(argv[2],std::ios::binary);f.write((char*)out.data(),bytes);f.close();
for(int w=0;w<3;w++)run();ck(aclrtSynchronizeStream(s),"warmup");
printf("{\"samples_us\":[");const int groups=10,iters=3;
for(int g=0;g<groups;g++){auto t=std::chrono::steady_clock::now();for(int j=0;j<iters;j++)run();ck(aclrtSynchronizeStream(s),"timing sync");double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count()/iters;printf("%s%.9f",g?",":"",us);}printf("]}\n");
aclrtFree(x);aclrtFree(h);aclrtFree(b);aclrtFree(c);aclrtDestroyStream(s);aclrtResetDevice(0);aclFinalize();return 0;}
"""
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--shape", default="8,8,8")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--group", type=int, default=2)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--fusion", action="store_true")
    p.add_argument("--mode", default="pipeline", choices=["pipeline", "forward", "inverse"])
    args = p.parse_args()
    emit(
        Contract(tuple(map(int, args.shape.split(","))), args.batch),
        dict(group=args.group, cores=args.cores, fusion=args.fusion),
        args.directory,
        args.mode,
    )
