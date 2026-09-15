# Author: even
"""Measure matched resident-data pipelines and retain every correctness-gated sample."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess

import numpy as np

from benchmarks.build_native import cann_environment


def check_output(path, reference):
    output = np.fromfile(path, dtype="<c8")
    if output.size != reference.size:
        raise ValueError("output size mismatch")
    delta = np.abs(output.astype(np.complex128) - reference)
    return bool(np.isfinite(output).all() and np.all(delta <= 1e-4 + 1e-4 * np.abs(reference)))


def run_matrix(
    prepared,
    native_build,
    output,
    enable_hardware=False,
    cann_env=None,
    iterations=1000,
    warmups=1000,
    groups=11,
    sessions=3,
):
    if not enable_hardware:
        raise ValueError("hardware execution requires --enable-hardware")
    if min(iterations, groups, sessions) < 1 or warmups < 0:
        raise ValueError("invalid repetition counts")
    prepared, native_build, output = map(Path, (prepared, native_build, output))
    manifest = json.loads((prepared / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((prepared / name).read_bytes()).hexdigest() != expected:
            raise ValueError("prepared artifact hash mismatch: " + name)
    output.mkdir(parents=True, exist_ok=False)
    env = cann_environment(cann_env)
    runs = []
    jobs = [
        (n, v, seed, session)
        for n in manifest["shape_sizes"]
        for v in ("native", "native_builtin_norm", "staged", "sage")
        for seed in (1, 2, 3)
        for session in range(1, sessions + 1)
    ]
    random.Random(45901).shuffle(jobs)
    for n, variant, seed, session in jobs:
        binary = native_build / ("native_fft_pipeline" if variant == "native" else "native_fft_builtin_norm")
        if variant in ("staged", "sage"):
            binary = prepared / "controls" / f"n{n}_{variant}" / "build" / "sage_fft"
        ip = prepared / "inputs" / f"n{n}" / f"input_{seed}"
        prefix = output / f"n{n}_{variant}_s{seed}_r{session}"
        command = [
            str(binary.resolve()),
            str(ip.resolve()),
            str(prefix.resolve()),
            str(n),
            str(iterations),
            str(warmups),
            str(groups),
        ]
        if variant in ("staged", "sage"):
            command.append("8")
        run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1800)
        Path(str(prefix) + ".stdout.txt").write_text(run.stdout, encoding="utf-8")
        Path(str(prefix) + ".stderr.txt").write_text(run.stderr, encoding="utf-8")
        if run.returncode:
            raise RuntimeError(f"{prefix.name}: device exit {run.returncode}")
        if variant in ("staged", "sage"):
            raw = json.loads(next(line for line in run.stdout.splitlines() if line.startswith("{")))
        else:
            raw = json.loads(Path(str(prefix) + ".json").read_text())
        samples = raw["samples_us"]
        if len(samples) != groups or any(not np.isfinite(x) or x <= 0 for x in samples):
            raise ValueError("invalid timing samples")
        reference = np.fromfile(ip / "reference.bin", dtype="<c16")
        correct = all(
            check_output(Path(str(prefix) + suffix), reference) for suffix in (".before.bin", ".after.bin")
        )
        if variant.startswith("native"):
            correct &= check_output(
                Path(str(prefix) + ".forward.bin"), np.fromfile(ip / "reference_fft.bin", dtype="<c16")
            )
        record = dict(
            n=n,
            variant=variant,
            seed=seed,
            session=session,
            iterations=iterations,
            warmups=warmups,
            groups=groups,
            samples_us=samples,
            median_us=statistics.median(samples),
            correctness_passed=correct,
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256((prepared / "manifest.json").read_bytes()).hexdigest(),
            inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ip.glob("*.bin")},
            outputs={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.glob(prefix.name + ".*.bin")
            },
            timing_method="host_batch_submit_sync",
            raw=raw,
        )
        Path(str(prefix) + ".result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        if not correct:
            raise ValueError(f"{prefix.name}: correctness failed")
        runs.append(record)
        print(
            json.dumps({k: record[k] for k in ("n", "variant", "seed", "session", "median_us")}), flush=True
        )
    result = {"complete": True, "runs": len(runs), "rows": []}
    for n in manifest["shape_sizes"]:
        for variant in ("native", "native_builtin_norm", "staged", "sage"):
            medians = [
                statistics.median(
                    r["median_us"] for r in runs if (r["n"], r["variant"], r["seed"]) == (n, variant, seed)
                )
                for seed in (1, 2, 3)
            ]
            result["rows"].append(
                dict(n=n, variant=variant, latency_us=statistics.median(medians), seed_medians_us=medians)
            )
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True, type=Path)
    p.add_argument("--native-build", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--enable-hardware", action="store_true")
    p.add_argument("--cann-env", type=Path)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--warmups", type=int, default=1000)
    p.add_argument("--groups", type=int, default=11)
    p.add_argument("--sessions", type=int, default=3)
    print(json.dumps(run_matrix(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
