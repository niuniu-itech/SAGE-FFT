# Author: even
"""Run the FP16 tuning, confirmation and held-out numerical controls."""

import argparse
import json
from pathlib import Path
import random

from benchmarks.run_native import run

VARIANTS = ("staged20", "grouped6", "fused5", "fused3", "grouped6_opt", "fused3_opt")


def matrix(backend):
    if backend == "npu":
        jobs = [("tune", v, b, 1, 1000) for v in VARIANTS for b in (1, 4, 8)]
        jobs += [("confirm", v, b, s, 10000) for v in VARIANTS for b in (4, 8) for s in (1, 2, 3)]
        jobs += [("heldout", v, 4, s, 1) for v in VARIANTS for s in (4, 5)]
    elif backend == "cuda":
        modes = ("direct", "graph1", "graph100")
        jobs = [("tune", m, 4, 1, 1000) for m in modes]
        jobs += [("confirm", m, 4, s, 10000) for m in modes for s in (1, 2, 3)]
        jobs += [("heldout", "graph100", 4, s, 100) for s in (4, 5)]
    else:
        raise ValueError("backend must be npu or cuda")
    random.Random(439).shuffle(jobs)
    return jobs


def run_matrix(backend, inputs, source, output, binary=None, enable_hardware=False, cann_env=None):
    if not enable_hardware:
        raise ValueError("hardware execution requires --enable-hardware")
    inputs, source, output = map(Path, (inputs, source, output))
    if backend == "cuda" and binary is None:
        raise ValueError("CUDA requires --binary")
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    for phase, variant, blocks, seed, iterations in matrix(backend):
        src = source / variant if backend == "npu" else source
        exe = src / "build/sage_fft" if backend == "npu" else Path(binary)
        tag = f"{phase}_{variant}_b{blocks}_s{seed}"
        result = run(
            exe,
            src,
            inputs / f"input_{seed}",
            output / (tag + ".bin"),
            "fp16",
            backend,
            enable_hardware=True,
            iterations=iterations,
            blocks=blocks,
            mode=variant if backend == "cuda" else "direct",
            cann_env=cann_env,
        )
        completed.append(
            dict(
                id=tag,
                phase=phase,
                variant=variant,
                blocks=blocks,
                seed=seed,
                accepted=result["accepted_timing"],
                median_us=result["median_us"],
            )
        )
        print(json.dumps(completed[-1]), flush=True)
    summary = dict(complete=True, backend=backend, cases=len(completed), runs=completed)
    (output / "completion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("npu", "cuda"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--enable-hardware", action="store_true")
    parser.add_argument("--cann-env", type=Path)
    run_matrix(**vars(parser.parse_args()))
