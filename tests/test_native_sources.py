# Author: even
"""CPU-only input, fusion-plan, generator, and verification regression tests."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks.fp16.generate import generate as generate_sources, plans
from benchmarks.generate_inputs import generate, make_inputs
from benchmarks.run_native import run, validate_timing
from benchmarks.verify_outputs import error_metrics, oracle, read_complex, verify

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


@pytest.mark.parametrize("precision,seeds", [("fp32", range(1, 4)), ("fp16", range(1, 6))])
def test_generated_inputs_match_reference_hashes_and_layout(tmp_path, precision, seeds):
    expected = json.loads((BENCHMARKS / "input_reference_sha256.json").read_text())[precision]
    manifest = generate(tmp_path / "first", precision)
    second = generate(tmp_path / "second", precision)
    assert manifest["sha256"] == expected == second["sha256"]
    for seed in seeds:
        x, h = make_inputs(seed, precision)
        assert x.shape == h.shape == (1024,)
        assert x.dtype.itemsize == (2 if precision == "fp16" else 4)
        decoded = read_complex(tmp_path / f"first/input_{seed}/x.bin", precision).reshape(-1)
        np.testing.assert_array_equal(decoded.real, x[::2].astype(np.float64))
        np.testing.assert_array_equal(decoded.imag, x[1::2].astype(np.float64))
    assert len(set(expected.values())) == len(expected)


def test_seed_quantization_and_holdout_rule():
    for seed in (1, 2, 3):
        for full, half in zip(make_inputs(seed, "fp32"), make_inputs(seed, "fp16")):
            np.testing.assert_array_equal(half, full.astype(np.float16))
    with pytest.raises(ValueError):
        make_inputs(4, "fp32")
    # Independent first draw verifies the holdout distribution is not the train rule.
    rng = np.random.Generator(np.random.PCG64(4))
    x, h = make_inputs(4, "fp16")
    np.testing.assert_array_equal(x, rng.normal(0, 0.25, 1024).astype(np.float16))
    np.testing.assert_array_equal(h, rng.uniform(-0.5, 0.5, 1024).astype(np.float16))


def radix_stages(value, axis, first, last, inverse):
    """Independent CPU radix-2 evaluator for the exported partial-stage plans."""
    value = np.moveaxis(value, axis, -1).copy()
    if first == 0:
        value = value[..., [0, 4, 2, 6, 1, 5, 3, 7]].copy()
    for stage in range(first, last):
        width = 2 ** (stage + 1)
        for base in range(0, 8, width):
            for j in range(width // 2):
                lo, hi = base + j, base + j + width // 2
                phase = (1 if inverse else -1) * 2j * np.pi * j / width
                a, b = value[..., lo].copy(), value[..., hi].copy() * np.exp(phase)
                value[..., lo], value[..., hi] = a + b, a - b
    return np.moveaxis(value, -1, axis)


def execute_plan(ops, x, h):
    value = x.astype(np.complex128)
    for op in ops:
        pointwise = op.get("pointwise")
        axis, inverse = op.get("axis", 0), op.get("inverse", False)
        if pointwise is None:
            value = radix_stages(value, axis, op.get("first", 0), op.get("last", 3), inverse)
        if op.get("mul") or pointwise == "multiply":
            value = value * h
        if op.get("second") == "inverse":
            value = radix_stages(value, axis, 0, 3, True)
        elif op.get("second") == "transpose":
            value = radix_stages(value, 1 if axis == 2 else 2, 0, 3, inverse)
        if op.get("scale") or pointwise == "scale":
            value = value / 512
    return value


@pytest.mark.parametrize("seed", (1, 4))
def test_all_six_fusion_plans_preserve_weighted_transform(seed):
    scalars = make_inputs(seed, "fp16")
    x, h = ((a[::2].astype(np.float64) + 1j * a[1::2].astype(np.float64)).reshape(8, 8, 8) for a in scalars)
    reference = np.fft.ifftn(np.fft.fftn(x) * h)
    expected_counts = {
        "staged20": 20,
        "grouped6": 6,
        "fused5": 5,
        "fused3": 3,
        "grouped6_opt": 6,
        "fused3_opt": 3,
    }
    for name, ops in plans().items():
        assert len(ops) == expected_counts[name]
        np.testing.assert_allclose(execute_plan(ops, x, h), reference, atol=1e-12, rtol=1e-12)
    # A weight multiplication moved across an axis transform must be detected.
    broken = plans()["grouped6"]
    broken[0]["mul"] = True
    broken[2]["mul"] = False
    assert not np.allclose(execute_plan(broken, x, h), reference, atol=1e-4, rtol=1e-4)


def test_generator_recreates_all_sources_in_new_directory(tmp_path):
    hashes = generate_sources(tmp_path)
    provenance = json.loads((BENCHMARKS / "SOURCE_PROVENANCE.json").read_text())
    for name, digest in hashes.items():
        assert digest == provenance[f"fp16/{name}/kernels.cpp"]["release_sha256"]
        for filename in ("kernels.cpp", "main.cpp", "CMakeLists.txt", "plan.json"):
            assert (tmp_path / name / filename).read_bytes() == (
                BENCHMARKS / "fp16" / name / filename
            ).read_bytes()
    # Twiddle-specialized controls emit genuinely different, shorter programs.
    for plain, optimized in (("grouped6", "grouped6_opt"), ("fused3", "fused3_opt")):
        assert (tmp_path / optimized / "kernels.cpp").stat().st_size < (
            tmp_path / plain / "kernels.cpp"
        ).stat().st_size


def test_importing_generators_cannot_create_outputs(tmp_path):
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
def forbidden(*args, **kwargs):
    raise AssertionError('generator performed import-time filesystem mutation')
Path.mkdir = forbidden
Path.write_text = forbidden
Path.write_bytes = forbidden
import benchmarks.generate_inputs
import benchmarks.fp16.generate
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(ROOT)], cwd=tmp_path, check=True)
    assert not list(tmp_path.iterdir())


def test_cpu_oracle_and_before_after_file_verification(tmp_path):
    generate(tmp_path, "fp16", [4])
    ref = oracle(tmp_path / "input_4", "fp16")
    output = tmp_path / "result.bin"
    pairs = np.stack((ref.real, ref.imag), axis=-1).astype("<f2")
    pairs.tofile(output)
    pairs.tofile(str(output) + ".after")
    assert verify(tmp_path / "input_4", output, "fp16")["passed"]
    pairs.flat[0] += 1
    pairs.tofile(str(output) + ".after")
    assert not verify(tmp_path / "input_4", output, "fp16")["passed"]
    with output.open("ab") as file:
        file.write(b"trailing data")
    with pytest.raises(ValueError, match="exactly"):
        read_complex(output, "fp16")


def test_precision_gates_and_nonfinite_rejection():
    ref = np.ones((8, 8, 8), dtype=np.complex128)
    assert error_metrics(ref + 0.002, ref, "fp16")["passed"]
    assert not error_metrics(ref + 0.002, ref, "fp32")["passed"]
    # Within the FP16 per-element allowance, outside the relative-L2 allowance.
    assert not error_metrics(ref + 0.004, ref, "fp16")["passed"]
    outlier = ref.copy()
    outlier.flat[0] += 0.008
    assert not error_metrics(outlier, ref, "fp16")["passed"]
    assert not error_metrics(ref * np.nan, ref, "fp16")["passed"]
    zero = np.zeros_like(ref)
    assert error_metrics(zero, zero, "fp16")["passed"]
    assert not error_metrics(zero + 1e-6, zero, "fp16")["passed"]


def test_hardware_runner_requires_opt_in_before_file_or_device_access():
    with pytest.raises(ValueError, match="enable-hardware"):
        run("missing", "missing", "missing", "missing", "fp16", "npu")


def test_timing_metadata_gate():
    record = dict(
        samples_us=[1.0] * 11,
        timing_method="host_batch_submit_sync",
        unit="us_per_pipeline",
        groups=11,
        warmup=1000,
        iterations=10000,
        precision="fp16",
        device_index=0,
        blocks=4,
    )
    assert len(validate_timing(record, 10000, "fp16", "npu", "direct", 4, 0)) == 11
    record["timing_method"] = "device_event"
    with pytest.raises(ValueError, match="metadata"):
        validate_timing(record, 10000, "fp16", "npu", "direct", 4, 0)


def test_checked_sources_match_release_provenance():
    provenance = json.loads((BENCHMARKS / "SOURCE_PROVENANCE.json").read_text())
    for relative, entry in provenance.items():
        assert hashlib.sha256((BENCHMARKS / relative).read_bytes()).hexdigest() == entry["release_sha256"]
