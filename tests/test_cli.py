# Author: even
"""Exercise the public installed entry point without devices or model services."""

import importlib
import json
import pkgutil
import subprocess
import sys

import sage_fft


def run(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "sage_fft", *args], cwd=cwd, text=True, capture_output=True, check=True
    )


def test_cli_emits_roundtrip_and_native_sources(tmp_path):
    source = tmp_path / "reference.cu"
    run("emit-cuda", "--output", str(source), cwd=tmp_path)
    imported = json.loads(run("import-cuda", str(source), cwd=tmp_path).stdout)
    assert imported["shape"] == [8, 8, 8]
    result = json.loads(run("validate-cpu", cwd=tmp_path).stdout)
    assert result["pass_correctness"]
    emitted = tmp_path / "npu"
    run("emit-npu", "--output", str(emitted), cwd=tmp_path)
    assert (emitted / "CMakeLists.txt").is_file()
    assert (emitted / "kernels.cpp").is_file()
    migrated = tmp_path / "from_cuda"
    result = json.loads(
        run("migrate", str(source), "--blocks", "8", "--output", str(migrated), cwd=tmp_path).stdout
    )
    ir = json.loads((migrated / "ir.json").read_text())
    assert result["launches"] == 6 and ir["action"]["cores"] == 8
    assert (migrated / "source.cu").read_bytes() == source.read_bytes()


def test_cli_variants_have_different_boundaries(tmp_path):
    baseline = json.loads(run("inspect", "--variant", "staged", cwd=tmp_path).stdout)
    grouped = json.loads(run("inspect", "--variant", "grouped", cwd=tmp_path).stdout)
    assert baseline["features"]["launches"] == 20
    assert grouped["features"]["launches"] == 6
    assert grouped["features"]["logical_KiB"] < baseline["features"]["logical_KiB"]


def test_invalid_cli_contract_fails_cleanly(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "sage_fft", "inspect", "--shape", "7,8"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "Traceback" not in proc.stderr


def test_imports_do_not_run_experiments(tmp_path, monkeypatch):
    monkeypatch.setenv("SAGE_WORKDIR", str(tmp_path / "unexpected"))
    # All configuration remains lazy; this also catches legacy absolute imports.
    for module in pkgutil.iter_modules(sage_fft.__path__):
        if (
            module.name in {"analyze_results", "analyze_quality"}
            and importlib.util.find_spec("matplotlib") is None
        ):
            continue  # Plot-only modules belong to the optional plots extra.
        if module.name != "__main__":
            importlib.import_module("sage_fft." + module.name)
    assert not (tmp_path / "unexpected").exists()
