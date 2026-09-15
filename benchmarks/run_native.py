# Author: even
"""Opt-in LOCAL hardware run with correctness gates and artifact hashes."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from .build_native import cann_environment
from .verify_outputs import read_complex, verify


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_timing(record, iterations, precision, backend, mode, blocks, device):
    samples = record.get("samples_us", [])
    expected = {
        "timing_method": "host_batch_submit_sync",
        "unit": "us_per_pipeline",
        "groups": 11,
        "warmup": 1000,
        "iterations": iterations,
        "precision": precision,
        "device_index": device,
    }
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("native timing metadata disagrees with the requested protocol")
    if len(samples) != 11 or any(
        not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in samples
    ):
        raise ValueError("expected 11 finite positive timing samples")
    if backend == "cuda" and record.get("mode") != mode:
        raise ValueError("CUDA mode mismatch")
    if backend == "npu" and record.get("blocks") != blocks:
        raise ValueError("NPU block count mismatch")
    return samples


def run(
    binary,
    source,
    inputs,
    output,
    precision,
    backend,
    enable_hardware=False,
    iterations=10000,
    blocks=4,
    mode="direct",
    device=0,
    cann_env=None,
    timeout=1800,
):
    if not enable_hardware:
        raise ValueError("hardware execution requires the explicit --enable-hardware flag")
    if backend not in ("npu", "cuda") or precision not in ("fp32", "fp16"):
        raise ValueError("invalid backend/precision")
    if iterations not in range(1, 100001) or blocks not in (1, 4, 8) or device < 0 or timeout <= 0:
        raise ValueError("invalid iterations, blocks, device, or timeout")
    if mode not in ("direct", "graph1", "graph100", "empty"):
        raise ValueError("unsupported mode")
    if backend == "npu" and mode != "direct":
        raise ValueError("CUDA modes cannot be applied to NPU variants")
    if backend == "cuda" and mode == "graph100" and iterations % 100:
        raise ValueError("graph100 requires iterations divisible by 100")
    binary, source = Path(binary).resolve(strict=True), Path(source).resolve(strict=True)
    inputs, output = Path(inputs).resolve(strict=True), Path(output).resolve()
    # Validate file sizes, quantized dtype and finiteness before opening a device.
    for name in ("x.bin", "h.bin"):
        read_complex(inputs / name, precision)
    outputs = [
        Path(str(output) + suffix) for suffix in ("", ".after", ".stdout.txt", ".stderr.txt", ".result.json")
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("choose a new --output path; existing run artifacts are preserved")
    sources = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and (path.suffix in (".cpp", ".cu", ".h") or path.name == "CMakeLists.txt")
        and "build" not in path.relative_to(source).parts
    )
    if not sources:
        raise ValueError("--source contains no native source files")
    env = cann_environment(cann_env if backend == "npu" else None)
    command = [
        str(binary),
        str(inputs),
        str(output),
        str(iterations),
        str(blocks) if backend == "npu" else mode,
        str(device),
    ]
    record = {
        "backend": backend,
        "precision": precision,
        "command": command,
        "source_sha256": {p.relative_to(source).as_posix(): sha256(p) for p in sources},
        "binary_sha256": sha256(binary),
        "input_sha256": {name: sha256(inputs / name) for name in ("x.bin", "h.bin")},
        "started_unix": time.time(),
        "hardware_executed": False,
        "accepted_timing": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        record["hardware_executed"] = True
        process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
        outputs[2].write_text(process.stdout, encoding="utf-8")
        outputs[3].write_text(process.stderr, encoding="utf-8")
        record["returncode"] = process.returncode
        process.check_returncode()
        timing = json.loads(process.stdout)
        samples = validate_timing(timing, iterations, precision, backend, mode, blocks, device)
        correctness = verify(inputs, output, precision, empty_control=mode == "empty")
        record.update(
            timing=timing,
            correctness=correctness,
            output_sha256={p.name: sha256(p) for p in outputs[:2]},
            accepted_timing=correctness["passed"] and mode != "empty",
        )
        if correctness["passed"]:
            record["median_us"] = statistics.median(samples)
        if not correctness["passed"]:
            raise ValueError("numeric correctness gate failed; timing cannot be used")
    except Exception as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record["ended_unix"] = time.time()
        outputs[4].write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-hardware", action="store_true")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--backend", choices=("npu", "cuda"), required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--blocks", type=int, choices=(1, 4, 8), default=4)
    parser.add_argument("--mode", choices=("direct", "graph1", "graph100", "empty"), default="direct")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cann-env", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if not args.enable_hardware:
        parser.error("hardware execution requires --enable-hardware")
    print(json.dumps(run(**vars(args)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
