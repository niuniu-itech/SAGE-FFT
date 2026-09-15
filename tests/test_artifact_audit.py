# Author: even
"""Recorded success cannot hide modified data, code, or timing summaries."""

from dataclasses import asdict
import hashlib

import numpy as np
import pytest

from benchmarks.audit_artifacts import verify_static
from sage_fft.fft_ir import Contract, identity
from sage_fft.validate_semantics import inputs, reference


@pytest.fixture
def saved_case(tmp_path):
    contract = Contract((32,), 1)
    action = dict(group=1, cores=1, fusion=False)
    content = b"synthetic-test-artifact"
    digest = hashlib.sha256(content).hexdigest()
    for name in ("kernels.cpp", "sage_fft", "source.cu"):
        (tmp_path / name).write_bytes(content)
    ip = tmp_path / "input_1"
    ip.mkdir()
    x, h = inputs(contract, 1)
    x.tofile(ip / "x.bin")
    h.tofile(ip / "h.bin")
    reference(contract, x, h).astype(np.complex64).tofile(tmp_path / "out_1.bin")
    row = dict(
        id="case",
        seed=1,
        phase="test",
        contract=asdict(contract),
        action=action,
        build_ok=True,
        exit_code=0,
        pass_correctness=True,
        mode="pipeline",
        ir_sha256=identity(contract, action)[0],
        target_source_sha256=digest,
        binary_sha256=digest,
        source_sha256=digest,
        samples_us=[1, 2, 3],
        p50_us=2,
    )
    return tmp_path, row


def test_saved_case_checks_without_hardware(saved_case):
    folder, row = saved_case
    assert verify_static(folder, row)["relative_l2"] < 1e-6


@pytest.mark.parametrize("change", ["output", "binary", "median"])
def test_changed_artifacts_cannot_pass(saved_case, change):
    folder, row = saved_case
    if change == "output":
        np.full((1, 32), np.nan, np.complex64).tofile(folder / "out_1.bin")
    elif change == "binary":
        (folder / "sage_fft").write_bytes(b"different-test-artifact")
    else:
        row["p50_us"] = 0.5
    with pytest.raises(ValueError):
        verify_static(folder, row)
