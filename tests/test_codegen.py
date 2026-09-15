# Author: even
"""Deterministic source generation, with native compilation explicitly disabled."""

import hashlib
import json
import re
import subprocess

import pytest

from sage_fft import Contract
from sage_fft.fft_ir import identity
from sage_fft.lower import emit
from sage_fft.mask_ir import apply, initial


@pytest.fixture(autouse=True)
def forbid_native_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Source generation must not launch a native compiler or subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def test_emission_is_deterministic_and_matches_legal_launch_plan(tmp_path):
    contract = Contract((8, 8, 8), 1)
    state = apply(contract, initial(contract), dict(level="fft", region="all", op="merge", boundaries="all"))
    state = apply(contract, state, dict(level="pipeline", region="both", op="fuse"))
    state = apply(
        contract, state, dict(level="schedule", region="all", op="reschedule", schedule=dict(cores=4, pack=8))
    )
    first, second = tmp_path / "one", tmp_path / "two"
    ir = emit(contract, state, first)
    assert emit(contract, state, second) == ir
    files = {"kernels.cpp", "main.cpp", "ir.json", "CMakeLists.txt"}
    assert files <= {path.name for path in first.iterdir()}
    for name in files:
        assert (first / name).read_bytes() == (second / name).read_bytes()

    manifest = json.loads((first / "ir.json").read_text())
    assert manifest["sha256"] == identity(contract, state)[0]
    assert len(manifest["stages"]) == 6
    host = (first / "main.cpp").read_text()
    kernels = (first / "kernels.cpp").read_text()
    launch_ids = re.findall(r"ACLRT_LAUNCH_KERNEL\((fft_step_\d+)\)\(4,s,src,h,dst\)", host)
    kernel_ids = re.findall(r"void (fft_step_\d+)\(GM_ADDR", kernels)
    assert launch_ids == kernel_ids and len(launch_ids) == 6
    assert not re.search(r"@[A-Z_]+@", host + kernels)
    assert "void* src=x;void* dst=b;" in host
    assert "result=src;" in host
    assert "ACL_MEMCPY_DEVICE_TO_HOST" in host
    assert "DataCopy" in kernels


def test_emitted_ir_identity_changes_with_independent_inverse_mask(tmp_path):
    contract = Contract((8, 8), 1)
    before = initial(contract)
    after = apply(contract, before, dict(level="fft", region="inverse:0", op="merge", boundaries=[1]))
    emit(contract, before, tmp_path / "before")
    emit(contract, after, tmp_path / "after")
    first = json.loads((tmp_path / "before/ir.json").read_text())
    second = json.loads((tmp_path / "after/ir.json").read_text())
    assert first["sha256"] != second["sha256"]
    assert len(first["stages"]) == len(second["stages"]) + 1
    assert first["action"]["forward_masks"] == second["action"]["forward_masks"]


def test_generation_rejects_dma_tail_before_writing_sources(tmp_path):
    contract = Contract((4,), 1)
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError):
        emit(contract, initial(contract), destination)
    assert not destination.exists()


def test_catalog_source_generation_requires_no_npu_or_cuda_toolchain(tmp_path):
    from sage_fft.catalog import generate

    ir = generate("M2", tmp_path)
    source = (tmp_path / "source.cu").read_text()
    assert "cufftPlan2d(&plan, 8, 8, CUFFT_C2C)" in source
    assert ir["mode"] == "kernel_catalog"
    assert ir["stages"]
    assert len(re.findall(r"case \d+:ck\(ACLRT_LAUNCH_KERNEL", (tmp_path / "main.cpp").read_text())) == len(
        ir["stages"]
    )
    digest = json.loads((tmp_path / "ir.json").read_text())["sha256"]
    expected = hashlib.sha256(json.dumps(ir, sort_keys=True).encode()).hexdigest()
    assert digest == expected
