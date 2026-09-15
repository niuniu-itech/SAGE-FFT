# Author: even
"""Check paired aggregation and reject altered or failed measurement artifacts."""

import hashlib
import json

import pytest

from benchmarks.fusion_study import VARIANTS
from benchmarks.summarize_fusion import summarize


@pytest.fixture
def confirmation(tmp_path):
    # Synthetic times exercise the reporting formula, not accelerator performance.
    for index, variant in enumerate(VARIANTS):
        for blocks, baseline in ((4, 200), (8, 120)):
            for seed, jitter in ((1, -1), (2, 0), (3, 1)):
                prefix = f"confirm_{variant}_b{blocks}_s{seed}.bin"
                output = tmp_path / prefix
                output.write_bytes(b"synthetic-output")
                samples = [baseline / (index + 1) + jitter] * 11
                row = dict(
                    hardware_executed=True,
                    accepted_timing=True,
                    backend="npu",
                    precision="fp16",
                    correctness=dict(passed=True),
                    binary_sha256="fixture-" + variant,
                    source_sha256={"kernels.cpp": "fixture-" + variant},
                    input_sha256={"x.bin": "seed-" + str(seed)},
                    median_us=samples[0],
                    output_sha256={prefix: hashlib.sha256(output.read_bytes()).hexdigest()},
                    timing=dict(
                        timing_method="host_batch_submit_sync",
                        unit="us_per_pipeline",
                        groups=11,
                        warmup=1000,
                        iterations=10000,
                        precision="fp16",
                        device_index=0,
                        blocks=blocks,
                        samples_us=samples,
                    ),
                )
                (tmp_path / (prefix + ".result.json")).write_text(json.dumps(row), encoding="utf-8")
    return tmp_path


def test_speedups_use_the_matching_block_baseline(confirmation):
    result = summarize(confirmation)
    assert result["confirmation_runs"] == 36
    for row in result["rows"]:
        assert row["speedup"] == pytest.approx(VARIANTS.index(row["variant"]) + 1)
    assert result["matched_twiddle_fusion_gain"] == pytest.approx({"4": 1.2, "8": 1.2})


def test_modified_output_cannot_support_a_speedup(confirmation):
    (confirmation / "confirm_fused3_opt_b8_s3.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="output hash mismatch"):
        summarize(confirmation)


def test_failed_case_cannot_be_dropped_from_aggregation(confirmation):
    path = confirmation / "confirm_staged20_b4_s1.bin.result.json"
    row = json.loads(path.read_text())
    row["correctness"]["passed"] = False
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="failed correctness"):
        summarize(confirmation)
