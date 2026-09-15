# Author: even
"""Shared Ascend C kernel catalog and target execution plans.

Catalog entries cover supported axis-stage ranges and terminal epilogues.
generate sends their IR through the common source emitter. A candidate becomes
an ordered list of catalog IDs; the runner builds the shared executable and
collects correctness outputs and timing samples on the configured NPU.
"""

from .config import workspace, remote_root, npu_environment, connect_ssh
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
import time
import numpy as np
from .campaign import Runner, ENV, ROOT
from .cuda_source import source
from .fft_ir import metrics
from .lower import emit
from .mask_ir import WORKLOADS, stages, initial, features
from .validate_semantics import inputs, reference

REMOTE = remote_root("catalog")


def verified_batch(folder, contract, realizations, seed, samples, mode, shuffle_seed, binary_sha256):
    """Reuse a batch only when its request, executable and saved outputs agree."""
    folder = Path(folder)
    rows = json.loads((folder / "results.json").read_text())
    plan = json.loads((folder / "plans.json").read_text())
    order = list(range(len(realizations)))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    ranks = {index: rank for rank, index in enumerate(order)}

    def require(condition):
        if not condition:
            raise RuntimeError("existing catalog batch does not verify; choose a new tag: " + str(folder))

    require(
        plan["input_seed"] == seed
        and plan["order"] == order
        and plan["catalog_binary_sha256"] == binary_sha256
    )
    require(len(rows) == len(realizations) == len(plan["plans"]))
    for index, (row, realization, entry) in enumerate(zip(rows, realizations, plan["plans"])):
        require(
            row["index"] == index
            and row["input_seed"] == seed
            and row["mode"] == mode
            and row["realization"] == realization
            and entry["realization"] == realization
            and entry["mode"] == mode
            and row["catalog_binary_sha256"] == binary_sha256
            and row["pass_correctness"]
            and row["execution_order"] == ranks[index]
        )
        values = np.asarray(row["samples_us"], dtype=float)
        require(
            values.shape == (samples,)
            and bool(np.all(np.isfinite(values) & (values > 0)))
            and float(np.median(values)) == row["p50_us"]
        )
    x, h = inputs(contract, seed)
    require((folder / "x.bin").read_bytes() == x.tobytes() and (folder / "h.bin").read_bytes() == h.tobytes())
    outputs = np.fromfile(folder / "outputs.bin", dtype="<c8").reshape(len(rows), *x.shape)
    expected = reference(contract, x, h, mode)
    for output in outputs:
        require(metrics(output, expected)["pass_correctness"])
    return rows


def canonical_step(c, s):
    s = dict(s)
    if s["kind"] == "fft":
        s["pack"] = min(math.prod(c.shape[s["axis"] + 1 :]), s["pack"])
    return json.dumps(s, sort_keys=True, separators=(",", ":"))


def catalog_steps(c, packs):
    out = []
    seen = set()
    for inv in (False, True):
        for axis, n in enumerate(c.shape):
            bits = n.bit_length() - 1
            for first in range(bits):
                for last in range(first + 1, bits + 1):
                    for fused in (False, True) if axis == 0 and last == bits else (False,):
                        for pack in packs:
                            s = dict(
                                kind="fft",
                                axis=axis,
                                first=first,
                                last=last,
                                inverse=inv,
                                multiply=fused and not inv,
                                scale=fused and inv,
                                pack=pack,
                            )
                            k = canonical_step(c, s)
                            if k not in seen:
                                seen.add(k)
                                out.append(json.loads(k))
    out.extend([dict(kind="multiply"), dict(kind="scale")])
    return out


HOST = r"""
#include <acl/acl.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
@INCLUDES@
void ck(aclError e,const char* x){if(e){fprintf(stderr,"ACL %d %s\n",e,x);exit(2);}}
std::vector<float> read(const std::string& p){std::vector<float> v(@TOTAL@*2);std::ifstream f(p,std::ios::binary);if(!f.read((char*)v.data(),v.size()*4)){fprintf(stderr,"input error\n");exit(3);}return v;}
struct Plan {std::string id;int cores;std::vector<int> kernels;};
int main(int argc,char**argv){if(argc<5)return 1;
auto xh=read(std::string(argv[1])+"/x.bin"),hh=read(std::string(argv[1])+"/h.bin");
std::ifstream pf(argv[2]);std::vector<Plan> plans;std::string id;int cores,n,k;
while(pf>>id>>cores>>n){if(n<1||n>100||(cores!=1&&cores!=4&&cores!=8))return 4;Plan p;p.id=id;p.cores=cores;for(int i=0;i<n;i++){if(!(pf>>k)||k<0||k>=@KERNELS@)return 4;p.kernels.push_back(k);}plans.push_back(p);}
if(plans.empty())return 4;const int samples=atoi(argv[4]),iters=3;if(samples<1||samples>50)return 4;
ck(aclInit(nullptr),"init");ck(aclrtSetDevice(0),"device");aclrtStream s;ck(aclrtCreateStream(&s),"stream");
const size_t bytes=@TOTAL@*8;void *x,*h,*b,*c;
ck(aclrtMalloc(&x,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc");ck(aclrtMalloc(&h,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc");ck(aclrtMalloc(&b,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc");ck(aclrtMalloc(&c,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc");
ck(aclrtMemcpy(x,bytes,xh.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE),"copy");ck(aclrtMemcpy(h,bytes,hh.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE),"copy");
std::ofstream out(argv[3],std::ios::binary);std::vector<float> yh(@TOTAL@*2);
for(const auto& plan:plans){void* result=nullptr;int blocks=plan.cores;
auto run=[&](){void* src=x;void* dst=b;for(int kernel:plan.kernels){switch(kernel){
@DISPATCH@
default:exit(5);}
src=dst;dst=(dst==b?c:b);}result=src;};
run();ck(aclrtSynchronizeStream(s),"check sync");ck(aclrtMemcpy(yh.data(),bytes,result,bytes,ACL_MEMCPY_DEVICE_TO_HOST),"output");out.write((char*)yh.data(),bytes);
for(int w=0;w<3;w++)run();ck(aclrtSynchronizeStream(s),"warmup");
printf("{\"id\":\"%s\",\"samples_us\":[",plan.id.c_str());
for(int j=0;j<samples;j++){auto t=std::chrono::steady_clock::now();for(int it=0;it<iters;it++)run();ck(aclrtSynchronizeStream(s),"timing sync");double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count()/iters;printf("%s%.9f",j?",":"",us);}printf("]}\n");fflush(stdout);
}
aclrtFree(x);aclrtFree(h);aclrtFree(b);aclrtFree(c);aclrtDestroyStream(s);aclrtResetDevice(0);aclFinalize();return 0;}
"""


def generate(name, d):
    c, packs, _ = WORKLOADS[name]
    d = Path(d)
    ss = catalog_steps(c, packs)
    ir = emit(c, initial(c), d, catalog_steps=ss)
    src, contract = source(c.shape, c.batch)
    assert c == contract
    (d / "source.cu").write_text(src, encoding="utf-8", newline="\n")
    includes = "\n".join('#include "aclrtlaunch_fft_step_%d.h"' % i for i in range(len(ss)))
    dispatch = "\n".join(
        'case %d:ck(ACLRT_LAUNCH_KERNEL(fft_step_%d)(blocks,s,src,h,dst),"launch");break;' % (i, i)
        for i in range(len(ss))
    )
    host = (
        HOST.replace("@TOTAL@", str(math.prod(c.shape) * c.batch))
        .replace("@KERNELS@", str(len(ss)))
        .replace("@INCLUDES@", includes)
        .replace("@DISPATCH@", dispatch)
    )
    (d / "main.cpp").write_text(host, encoding="utf-8", newline="\n")
    return ir


class CatalogRunner(Runner):
    def __init__(self):
        super().__init__()
        self.root = ROOT / "experiments" / "v46"
        self.root.mkdir(exist_ok=True)
        self.journal = ROOT / "evidence" / "v46_target_records.jsonl"
        self.catalogs = {}

    def prepare(self, name):
        d = self.root / name / "catalog"
        d.mkdir(parents=True, exist_ok=True)
        metadata = d / "build_metadata.json"
        c, packs, _ = WORKLOADS[name]
        if metadata.exists():
            r = json.loads(metadata.read_text())
            assert r["build_ok"]
            self.catalogs[name] = r
            return r
        ir = generate(name, d)
        remote = REMOTE + "/" + name + "/catalog"
        self.command("mkdir -p " + shlex.quote(remote))
        for p in d.iterdir():
            if p.is_file():
                self.s.put(str(p), remote + "/" + p.name)
        cmd = (
            ENV
            + "set -e\ncd "
            + shlex.quote(remote)
            + """
cmake -S . -B build -DCMAKE_CXX_COMPILER="${CXX:-g++}" -DCMAKE_BUILD_TYPE=Release > configure.log 2>&1
cmake --build build -j2 > build.log 2>&1
sha256sum build/sage_fft
"""
        )
        (d / "build.sh").write_text(cmd)
        t = time.perf_counter()
        rc, out, err = self.command(cmd, 1800)
        for f in ("configure.log", "build.log"):
            try:
                self.s.get(remote + "/" + f, str(d / f))
            except FileNotFoundError:
                pass
        r = dict(
            workload=name,
            contract=asdict(c),
            build_ok=rc == 0,
            build_seconds=time.perf_counter() - t,
            exit_code=rc,
            stderr=err,
            kernels=len(ir["stages"]),
            hashes={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in d.iterdir()
                if p.name in ("source.cu", "kernels.cpp", "main.cpp", "ir.json", "CMakeLists.txt")
            },
        )
        if rc:
            raise RuntimeError(json.dumps(r) + "\n" + (d / "build.log").read_text(errors="replace")[-2000:])
        r["binary_sha256"] = out.strip().split()[0]
        self.s.get(remote + "/build/sage_fft", str(d / "sage_fft"))
        metadata.write_text(json.dumps(r, indent=2))
        self.catalogs[name] = r
        print(json.dumps(dict(event="catalog_built", **r)), flush=True)
        return r

    def evaluate_many(self, name, realizations, tag, seed=1, samples=5, mode="pipeline", shuffle_seed=None):
        c, packs, _ = WORKLOADS[name]
        meta = self.prepare(name)
        d = self.root / name / tag
        result_path = d / "results.json"
        if result_path.exists():
            return verified_batch(
                d, c, realizations, seed, samples, mode, shuffle_seed, meta["binary_sha256"]
            )
        d.mkdir(parents=True, exist_ok=True)
        remote = REMOTE + "/" + name + "/" + tag
        ss = catalog_steps(c, packs)
        ids = {canonical_step(c, s): i for i, s in enumerate(ss)}
        plans = []
        for i, r in enumerate(realizations):
            steps = stages(c, r, mode)
            plan = [ids[canonical_step(c, s)] for s in steps]
            plans.append(dict(id="p%06d" % i, realization=r, kernel_ids=plan, mode=mode))
        order = list(range(len(plans)))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(order)
        lines = [
            "%s %d %d %s"
            % (
                plans[i]["id"],
                realizations[i]["cores"],
                len(plans[i]["kernel_ids"]),
                " ".join(map(str, plans[i]["kernel_ids"])),
            )
            for i in order
        ]
        (d / "plans.txt").write_text("\n".join(lines) + "\n")
        (d / "plans.json").write_text(
            json.dumps(
                dict(input_seed=seed, order=order, plans=plans, catalog_binary_sha256=meta["binary_sha256"]),
                indent=2,
            )
        )
        x, h = inputs(c, seed)
        x.tofile(d / "x.bin")
        h.tofile(d / "h.bin")
        self.command("mkdir -p " + shlex.quote(remote))
        for f in ("plans.txt", "x.bin", "h.bin"):
            self.s.put(str(d / f), remote + "/" + f)
        cmd = (
            ENV
            + "cd "
            + shlex.quote(remote)
            + "\n../catalog/build/sage_fft . plans.txt outputs.bin "
            + str(samples)
        )
        (d / "run.sh").write_text(cmd)
        t = time.perf_counter()
        rc, out, err = self.command(cmd, 1800)
        wall = time.perf_counter() - t
        (d / "run.log").write_text(out + err)
        if rc:
            raise RuntimeError("target execution failed " + str(rc) + " " + err[-1000:])
        self.s.get(remote + "/outputs.bin", str(d / "outputs.bin"))
        ys = np.fromfile(d / "outputs.bin", np.complex64).reshape(len(plans), *x.shape)
        ref = reference(c, x, h, mode)
        parsed = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
        assert len(parsed) == len(plans)
        rows = []
        for rank, (i, row, y) in enumerate(zip(order, parsed, ys)):
            assert row["id"] == plans[i]["id"]
            row.update(metrics(y, ref))
            row.update(
                workload=name,
                tag=tag,
                index=i,
                input_seed=seed,
                realization=realizations[i],
                mode=mode,
                p50_us=float(np.median(row["samples_us"])),
                features=features(c, realizations[i]),
                catalog_binary_sha256=meta["binary_sha256"],
                execution_order=rank,
            )
            rows.append(row)
        rows.sort(key=lambda r: r["index"])
        result_path.write_text(json.dumps(rows, indent=2))
        # Per-batch results are authoritative. Build the aggregate journal after
        # the campaign; a sync client can briefly lock a frequently appended file.
        print(
            json.dumps(
                dict(
                    event="target_batch",
                    workload=name,
                    tag=tag,
                    checked=len(rows),
                    passed=sum(r["pass_correctness"] for r in rows),
                    seconds=round(wall, 3),
                    minimum_us=min(r["p50_us"] for r in rows),
                )
            ),
            flush=True,
        )
        if not all(r["pass_correctness"] for r in rows):
            raise RuntimeError("target correctness failure: " + str(d))
        return rows


if __name__ == "__main__":
    import argparse
    from .mask_ir import all_realizations, random_realization

    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["build", "smoke", "oracle"])
    p.add_argument("--workloads", default="M1,M2,M3")
    args = p.parse_args()
    r = CatalogRunner()
    try:
        for name in args.workloads.split(","):
            r.prepare(name)
            if args.phase == "smoke":
                rng = random.Random(20260911)
                cs = [initial(WORKLOADS[name][0])] + [random_realization(name, rng) for _ in range(8)]
                for mode in ("pipeline", "forward", "inverse"):
                    r.evaluate_many(name, cs, "smoke_" + mode, mode=mode)
            if args.phase == "oracle":
                assert name in ("M1", "M2")
                r.evaluate_many(name, list(all_realizations(name)), "oracle_b1", shuffle_seed=9101)
    finally:
        r.s.close()
        r.c.close()
