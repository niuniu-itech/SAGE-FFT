# Author: even
"""Recheck saved static FFT measurements without contacting either device."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from sage_fft.fft_ir import Contract, identity, metrics
from sage_fft.validate_semantics import inputs, reference


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timing(row):
    samples = np.asarray(row["samples_us"], dtype=float)
    require(samples.ndim == 1 and samples.size > 0, "missing timing samples")
    require(bool(np.all(np.isfinite(samples) & (samples > 0))), "invalid timing samples")
    require(statistics.median(samples) == row["p50_us"], "timing median disagrees with samples")


def verify_static(folder, row):
    folder = Path(folder)
    contract_data = dict(row["contract"], shape=tuple(row["contract"]["shape"]))
    contract = Contract(**contract_data)
    require(asdict(contract) == contract_data, "contract was not retained exactly")
    require(row["build_ok"] and row["exit_code"] == 0, "unsuccessful device execution")
    require(row["ir_sha256"] == identity(contract, row["action"], row["mode"])[0], "IR identity mismatch")
    for name, field in (("kernels.cpp", "target_source_sha256"), ("sage_fft", "binary_sha256")):
        require(sha256(folder / name) == row[field], "artifact hash mismatch: " + name)
    cuda = (folder / "source.cu").read_text(encoding="utf-8").replace("\r\n", "\n")
    require(hashlib.sha256(cuda.encode()).hexdigest() == row["source_sha256"], "CUDA source hash mismatch")
    x, h = inputs(contract, row["seed"])
    input_dir = folder / ("input_%d" % row["seed"])
    for name, value in (("x.bin", x), ("h.bin", h)):
        require((input_dir / name).read_bytes() == value.tobytes(), "input data mismatch: " + name)
        if "input_sha256" in row:
            require(sha256(input_dir / name) == row["input_sha256"][name], "recorded input hash mismatch")
    output_file = folder / ("out_%d.bin" % row["seed"])
    output = np.fromfile(output_file, dtype="<c8").reshape(x.shape)
    result = metrics(output, reference(contract, x, h, row["mode"]))
    require(result["pass_correctness"], "saved output fails the independent reference")
    if "output_sha256" in row:
        require(sha256(output_file) == row["output_sha256"], "recorded output hash mismatch")
    timing(row)
    return dict(
        id=row["id"],
        seed=row["seed"],
        phase=row["phase"],
        output_sha256=sha256(output_file),
        relative_l2=result["relative_l2"],
    )


def audit(workspace):
    workspace = Path(workspace)
    artifacts = workspace / "experiments/artifacts"
    verified, retained_failures = [], []
    sources = {}
    for journal_name in ("target_records.jsonl", "scaling_records.jsonl"):
        journal = workspace / "evidence" / journal_name
        if not journal.exists():
            continue
        rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        seen = set()
        for row in rows:
            if not row.get("pass_correctness"):
                retained_failures.append(dict(journal=journal_name, id=row["id"], error=row.get("error")))
                continue
            marker = (row["id"], row["seed"])
            require(marker not in seen, "duplicate measurement identity")
            seen.add(marker)
            verified.append(verify_static(artifacts / row["id"], row))
            sources[row["id"]] = row
    require(bool(verified), "no saved static measurements found")

    calibration_counts = {}
    for study in ("calibration", "oracle"):
        journal = workspace / "evidence" / (study + "_records.jsonl")
        if not journal.exists():
            continue
        folder = workspace / "experiments" / study
        rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            original = sources[row["source_measurement_id"]]
            require(row["pass_correctness"], "failed calibration output")
            for field in ("action", "binary_sha256", "target_source_sha256"):
                require(row[field] == original[field], "calibration changed selected executable")
            contract = Contract(**dict(original["contract"], shape=tuple(original["contract"]["shape"])))
            x, h = inputs(contract, row["input_seed"])
            for name, value in (("x.bin", x), ("h.bin", h)):
                require(
                    (folder / ("input_%d" % row["input_seed"]) / name).read_bytes() == value.tobytes(),
                    "calibration input mismatch",
                )
            output = np.fromfile(folder / row["output_file"], dtype="<c8").reshape(x.shape)
            require(
                metrics(output, reference(contract, x, h))["pass_correctness"], "calibration output mismatch"
            )
            timing(row)
        calibration_counts[study] = len(rows)
    return dict(
        passed=True,
        static_outputs=len(verified),
        phases=dict(Counter(row["phase"] for row in verified)),
        calibration_outputs=calibration_counts,
        retained_failures=retained_failures,
        max_relative_l2=max(row["relative_l2"] for row in verified),
        outputs=verified,
        scope="Saved artifacts only; protocol completeness must be checked separately.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
