# Author: even
"""Regenerate all six native FP16 variants using radix-2 vector butterflies.

No input, source, or hardware work occurs on import. Outputs are explicit CLI paths.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from benchmarks.source_support import write_support


def op(s):
    return s + "PipeBarrier<PIPE_V>();\n"


def fft(first, last, inverse, optimized=False):
    s = ""
    for stage in range(first, last):
        m = 2 ** (stage + 1)
        for base in range(0, 8, m):
            for j in range(m // 2):
                u = (base + j) * 16
                q = (base + j + m // 2) * 16
                w = (1 if inverse else -1) * 2 * math.pi * j / m
                c = round(math.cos(w), 12)
                d = round(math.sin(w), 12)
                if optimized and c == 1 and d == 0:
                    for var in ["r", "i"]:
                        s += (
                            op(f"Add(t,{var}[{u}],{var}[{q}],16);")
                            + op(f"Sub({var}[{q}],{var}[{u}],{var}[{q}],16);")
                            + op(f"Adds({var}[{u}],t,(half)0,16);")
                        )
                    continue
                if optimized and c == 0:
                    s += op(f"Muls(t,i[{q}],(half){-d:.1f}f,16);") + op(
                        f"Muls(t[16],r[{q}],(half){d:.1f}f,16);"
                    )
                    s += op(f"Sub(r[{q}],r[{u}],t,16);") + op(f"Add(r[{u}],r[{u}],t,16);")
                    s += op(f"Sub(i[{q}],i[{u}],t[16],16);") + op(f"Add(i[{u}],i[{u}],t[16],16);")
                    continue
                s += op(f"Muls(t,r[{q}],(half){c:.12f}f,16);")
                s += op(f"Muls(t[32],i[{q}],(half){d:.12f}f,16);")
                s += op("Sub(t,t,t[32],16);")
                s += op(f"Muls(t[16],i[{q}],(half){c:.12f}f,16);")
                s += op(f"Muls(t[48],r[{q}],(half){d:.12f}f,16);")
                s += op("Add(t[16],t[16],t[48],16);")
                s += op(f"Sub(r[{q}],r[{u}],t,16);") + op(f"Add(r[{u}],r[{u}],t,16);")
                s += op(f"Sub(i[{q}],i[{u}],t[16],16);") + op(f"Add(i[{u}],i[{u}],t[16],16);")
    return s


def multiply():
    s = "PipeBarrier<PIPE_ALL>();\n"
    s += "for(int k=0;k<8;k++)for(int l=0;l<8;l++){hr.SetValue(k*16+l,wh.GetValue(2*(k*8+l)));hi.SetValue(k*16+l,wh.GetValue(2*(k*8+l)+1));}\nPipeBarrier<PIPE_ALL>();\n"
    for k in range(8):
        u = k * 16
        s += (
            op(f"Mul(t,r[{u}],hr[{u}],16);") + op(f"Mul(t[32],i[{u}],hi[{u}],16);") + op("Sub(t,t,t[32],16);")
        )
        s += (
            op(f"Mul(t[16],r[{u}],hi[{u}],16);")
            + op(f"Mul(t[48],i[{u}],hr[{u}],16);")
            + op(f"Add(i[{u}],t[16],t[48],16);")
            + op(f"Adds(r[{u}],t,(half)0,16);")
        )
    return s


def kernel(
    idx,
    axis=0,
    first=0,
    last=3,
    inverse=False,
    mul=False,
    scale=False,
    second=None,
    pointwise=None,
    optimized=False,
):
    # Each tile contains eight length-8 lines. Native vector operations use
    # 16 half lanes with eight data lanes and eight initialized padding lanes.
    s = f'extern "C" __global__ __aicore__ void fft_step_{idx}(GM_ADDR input,GM_ADDR weight,GM_ADDR output){{\n'
    s += "GlobalTensor<half> x,h,y;x.SetGlobalBuffer((__gm__ half*)input);h.SetGlobalBuffer((__gm__ half*)weight);y.SetGlobalBuffer((__gm__ half*)output);\nTPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,2560);auto raw=buf.Get<half>();auto wh=raw[128];auto r=raw[256];auto i=raw[384];auto r2=raw[512];auto i2=raw[640];auto hr=raw[768];auto hi=raw[896];auto t=raw[1024];\n"
    s += "for(int tile=GetBlockIdx();tile<8;tile+=GetBlockNum()){\nDuplicate(r,(half)0,768);PipeBarrier<PIPE_ALL>();\n"
    if axis == 0 and not pointwise:
        s += "for(int k=0;k<8;k++){DataCopy(raw[k*16],x[2*(k*64+tile*8)],16);"
        if mul:
            s += "DataCopy(wh[k*16],h[2*(k*64+tile*8)],16);"
        s += "}\n"
    else:
        s += "DataCopy(raw,x[tile*128],128);"
        if mul or pointwise == "multiply":
            s += "DataCopy(wh,h[tile*128],128);"
    s += "PipeBarrier<PIPE_ALL>();\n"
    reverse = first == 0 and not pointwise
    p = "((k&1)*4+(k&2)+((k&4)/4))" if reverse else "k"
    raw_index = f"l*8+{p}" if axis == 2 and not pointwise else f"({p})*8+l"
    s += f"for(int k=0;k<8;k++)for(int l=0;l<8;l++){{int p={raw_index};r.SetValue(k*16+l,raw.GetValue(2*p));i.SetValue(k*16+l,raw.GetValue(2*p+1));}}\nPipeBarrier<PIPE_ALL>();\n"
    if not pointwise:
        s += fft(first, last, inverse, optimized)
    if mul or pointwise == "multiply":
        s += multiply()
    if second:
        reverse = "((k&1)*4+(k&2)+((k&4)/4))"
        loc = f"l*16+{reverse}" if second == "transpose" else f"({reverse})*16+l"
        s += f"PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++)for(int l=0;l<8;l++){{r2.SetValue(k*16+l,r.GetValue({loc}));i2.SetValue(k*16+l,i.GetValue({loc}));}}\nPipeBarrier<PIPE_ALL>();"
        s += op("Adds(r,r2,(half)0,128);") + op("Adds(i,i2,(half)0,128);")
        s += fft(0, 3, True if second == "inverse" else inverse, optimized)
    if scale or pointwise == "scale":
        s += op("Muls(r,r,(half)0.001953125f,128);") + op("Muls(i,i,(half)0.001953125f,128);")
    out_axis = (1 if axis == 2 else 2) if second == "transpose" else axis
    out_index = "l*8+k" if out_axis == 2 and not pointwise else "k*8+l"
    s += f"PipeBarrier<PIPE_ALL>();for(int k=0;k<8;k++)for(int l=0;l<8;l++){{int p={out_index};raw.SetValue(2*p,r.GetValue(k*16+l));raw.SetValue(2*p+1,i.GetValue(k*16+l));}}\nPipeBarrier<PIPE_ALL>();\n"
    if axis == 0 and not pointwise:
        s += "for(int k=0;k<8;k++)DataCopy(y[2*(k*64+tile*8)],raw[k*16],16);"
    else:
        s += "DataCopy(y[tile*128],raw,128);"
    return s + "PipeBarrier<PIPE_ALL>();}}\n"


def plans():
    staged = []
    for inverse in (False, True):
        for axis in (2, 1, 0) if not inverse else (0, 1, 2):
            staged.extend(dict(axis=axis, first=j, last=j + 1, inverse=inverse) for j in range(3))
        staged.append(dict(pointwise="scale" if inverse else "multiply"))
    grouped = [
        dict(axis=2),
        dict(axis=1),
        dict(axis=0, mul=True),
        dict(axis=0, inverse=True),
        dict(axis=1, inverse=True),
        dict(axis=2, inverse=True, scale=True),
    ]
    fused5 = [
        dict(axis=2),
        dict(axis=1),
        dict(axis=0, mul=True, second="inverse"),
        dict(axis=1, inverse=True),
        dict(axis=2, inverse=True, scale=True),
    ]
    fused3 = [
        dict(axis=2, second="transpose"),
        dict(axis=0, mul=True, second="inverse"),
        dict(axis=1, inverse=True, second="transpose", scale=True),
    ]
    return copy.deepcopy(
        {
            "staged20": staged,
            "grouped6": grouped,
            "fused5": fused5,
            "fused3": fused3,
            "grouped6_opt": grouped,
            "fused3_opt": fused3,
        }
    )


def generate(output, variants=None):
    output = Path(output)
    available = plans()
    selected = list(variants if variants is not None else available)
    if not selected or any(name not in available for name in selected):
        raise ValueError("select one or more known FP16 variants")
    manifest = {}
    for name in selected:
        folder = output / name
        ops = available[name]
        optimized = name.endswith("_opt")
        code = '#include "kernel_operator.h"\nusing namespace AscendC;\n'
        code += "".join(kernel(j, optimized=optimized, **op) for j, op in enumerate(ops))
        write_support(folder, len(ops), "fp16")
        (folder / "kernels.cpp").write_text(code, encoding="utf-8", newline="\n")
        plan = dict(
            name=name,
            dtype="FP16",
            shape=[8, 8, 8],
            launches=len(ops),
            ub_bytes_per_block=2560,
            blocks=[1, 4, 8],
            twiddle_specialized=optimized,
            arithmetic="native half vector Add/Sub/Mul/Muls",
            ops=ops,
        )
        (folder / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n")
        manifest[name] = hashlib.sha256((folder / "kernels.cpp").read_bytes()).hexdigest()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", choices=list(plans()), nargs="+")
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.variants), indent=2))


if __name__ == "__main__":
    main()
