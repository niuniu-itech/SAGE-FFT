# Author: even
"""An interrupted hardware campaign must never count an empty directory as success."""

from unittest.mock import Mock
import hashlib
import json

import numpy as np

import pytest

from sage_fft.campaign import verified_case
from sage_fft.fft_ir import Contract, identity
from sage_fft.validate_semantics import inputs, reference


def test_existing_directory_without_records_is_not_a_completed_case(tmp_path):
    runner = Mock(root=tmp_path, journal=tmp_path / "journal.jsonl")
    (tmp_path / "partial").mkdir()
    with pytest.raises(RuntimeError, match="no measurement journal"):
        verified_case(runner, Contract((8, 8, 8), 1), {}, "partial")
    runner.evaluate.assert_not_called()


def test_failed_build_does_not_complete_campaign_case(tmp_path):
    runner = Mock(root=tmp_path, journal=tmp_path / "journal.jsonl")
    runner.evaluate.return_value = {"id": "failed", "build_ok": False}
    with pytest.raises(RuntimeError, match="three verified"):
        verified_case(runner, Contract((8, 8, 8), 1), {}, "failed")


def test_saved_success_flag_does_not_override_corrupted_output(tmp_path):
    contract = Contract((32,), 1)
    action = dict(group=1, cores=1, fusion=False)
    folder = tmp_path / "case"
    folder.mkdir()
    digest = hashlib.sha256(b"synthetic-test-artifact").hexdigest()
    for name in ("kernels.cpp", "sage_fft"):
        (folder / name).write_bytes(b"synthetic-test-artifact")
    rows = []
    for seed in (1, 2, 3):
        ip = folder / f"input_{seed}"
        ip.mkdir()
        x, h = inputs(contract, seed)
        x.tofile(ip / "x.bin")
        h.tofile(ip / "h.bin")
        reference(contract, x, h).astype(np.complex64).tofile(folder / f"out_{seed}.bin")
        rows.append(
            dict(
                id="case",
                seed=seed,
                build_ok=True,
                exit_code=0,
                pass_correctness=True,
                action=action,
                mode="pipeline",
                target_source_sha256=digest,
                binary_sha256=digest,
                ir_sha256=identity(contract, action)[0],
            )
        )
    journal = tmp_path / "journal.jsonl"
    journal.write_text("\n".join(map(json.dumps, rows)))
    runner = Mock(root=tmp_path, journal=journal)
    assert len(verified_case(runner, contract, action, "case")) == 3
    np.full((1, 32), np.nan, np.complex64).tofile(folder / "out_2.bin")
    with pytest.raises(RuntimeError, match="output fails correctness"):
        verified_case(runner, contract, action, "case")
    runner.evaluate.assert_not_called()
