# Author: even
"""Derive matched-block FP16 speedups from the complete confirmation matrix."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.fusion_study import VARIANTS
from benchmarks.run_native import validate_timing


def summarize(directory):
    directory = Path(directory)
    rows = []
    records = {}
    artifacts = []
    identities = {}
    for variant in VARIANTS:
        for blocks in (4, 8):
            values = []
            for seed in (1, 2, 3):
                path = directory / f"confirm_{variant}_b{blocks}_s{seed}.bin.result.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                if not record.get("hardware_executed") or not record.get("accepted_timing"):
                    raise ValueError("unaccepted hardware result: " + path.name)
                if record.get("backend") != "npu" or record.get("precision") != "fp16":
                    raise ValueError("unexpected device or precision: " + path.name)
                if not record["correctness"]["passed"]:
                    raise ValueError("failed correctness result: " + path.name)
                samples = validate_timing(record["timing"], 10000, "fp16", "npu", "direct", blocks, 0)
                value = statistics.median(samples)
                if value != record["median_us"]:
                    raise ValueError("saved median disagrees with samples: " + path.name)
                identity = (record["binary_sha256"], record["source_sha256"])
                if variant in identities and identities[variant] != identity:
                    raise ValueError("variant mixes different source or binaries: " + variant)
                identities[variant] = identity
                for output, expected in record["output_sha256"].items():
                    if hashlib.sha256((directory / output).read_bytes()).hexdigest() != expected:
                        raise ValueError("output hash mismatch: " + output)
                records[variant, blocks, seed] = record
                values.append(value)
                artifacts.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            rows.append(
                dict(
                    variant=variant,
                    blocks=blocks,
                    seed_medians_us=values,
                    latency_us=statistics.median(values),
                )
            )
    for seed in (1, 2, 3):
        inputs = [r["input_sha256"] for (v, b, s), r in records.items() if s == seed]
        if any(value != inputs[0] for value in inputs):
            raise ValueError("input identity differs across paired implementations")
    baseline = {row["blocks"]: row["latency_us"] for row in rows if row["variant"] == "staged20"}
    by_variant = {(row["variant"], row["blocks"]): row["latency_us"] for row in rows}
    for row in rows:
        row["speedup"] = baseline[row["blocks"]] / row["latency_us"]
    return dict(
        complete=True,
        confirmation_runs=len(records),
        rows=rows,
        matched_twiddle_fusion_gain={
            str(b): by_variant["grouped6_opt", b] / by_variant["fused3_opt", b] for b in (4, 8)
        },
        formula="median_seed(T_staged) / median_seed(T_variant), at the same block count",
        artifacts=artifacts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps({k: result[k] for k in ("complete", "confirmation_runs", "matched_twiddle_fusion_gain")})
    )
