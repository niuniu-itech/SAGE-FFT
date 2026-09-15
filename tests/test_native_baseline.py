# Author: even
"""Check the native baseline's input semantics, compiler path and numeric gates."""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.native_fft.generate import generate
from benchmarks.native_fft.run import check_output, run_matrix


def test_native_controls_match_bundled_arithmetic_and_oracle(tmp_path):
    manifest = generate(tmp_path / "prepared", (8,))
    assert [r["launches"] for r in manifest["controls"]] == [20, 6]
    root = Path(__file__).resolve().parents[1]
    for label, variant in (("staged", "staged20"), ("sage", "sage6")):
        current = tmp_path / "prepared/controls" / ("n8_" + label)
        assert (current / "kernels.cpp").read_text() == (
            root / "benchmarks/fp32" / variant / "kernels.cpp"
        ).read_text()
        assert (current / "ir.json").is_file()
    folder = tmp_path / "prepared/inputs/n8/input_3"
    x = np.fromfile(folder / "x.bin", dtype="<c8").reshape(8, 8, 8)
    h = np.fromfile(folder / "h.bin", dtype="<c8").reshape(8, 8, 8)
    expected = np.fft.ifftn(np.fft.fftn(x.astype(np.complex128)) * h.astype(np.complex128))
    saved = np.fromfile(folder / "reference.bin", dtype="<c16").reshape(8, 8, 8)
    np.testing.assert_array_equal(saved, expected)


def test_native_baseline_rejects_corruption_and_requires_explicit_execution(tmp_path):
    ref = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    path = tmp_path / "output.bin"
    ref.astype("<c8").tofile(path)
    assert check_output(path, ref)
    np.array([complex(float("nan"), 2), 3 - 4j], dtype="<c8").tofile(path)
    assert not check_output(path, ref)
    np.array([1], dtype="<c8").tofile(path)
    with pytest.raises(ValueError, match="size"):
        check_output(path, ref)
    with pytest.raises(ValueError, match="enable-hardware"):
        run_matrix(tmp_path, tmp_path, tmp_path / "run")


def test_modified_reference_is_rejected_before_device_execution(tmp_path):
    prepared = tmp_path / "prepared"
    generate(prepared, (8,))
    (prepared / "inputs/n8/input_1/reference.bin").write_bytes(b"altered reference")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_matrix(prepared, tmp_path / "no-binary", tmp_path / "run", enable_hardware=True)
    assert not (tmp_path / "run").exists()
