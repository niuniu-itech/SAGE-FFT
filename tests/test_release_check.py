# Author: even
"""Synthetic release-audit regressions; fixtures contain no live credentials."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.check_repository import MIB, audit, text_findings


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_secret_findings_never_include_matched_values(tmp_path):
    key = "sk-" + "syntheticA1" * 6
    private_address = ".".join(("192", "168", "3", "7"))
    path = "D:" + "/private/project"
    fixture = "\n".join(
        (
            key,
            private_address,
            path,
            "password = '" + "fixture-pass" + "'",
            "-----BEGIN " + "PRIVATE KEY-----",
        )
    )
    write(tmp_path, "candidate.txt", fixture)
    result = audit(tmp_path)
    assert {item["category"] for item in result["findings"]} == {
        "likely_api_key",
        "private_ip",
        "private_absolute_path",
        "hardcoded_secret",
        "private_key",
    }
    serialized = json.dumps(result)
    for secret in (key, private_address, path, "fixture-pass"):
        assert secret not in serialized
    assert all(set(item) == {"file", "line", "category", "severity"} for item in result["findings"])


def test_scanner_handles_existing_source_patterns_and_narrow_fixture_exception():
    source = 'password=os.environ.get(prefix + "PASSWORD")\n'
    source += "\\\\centering\\n\\\\caption{example}"
    assert not text_findings("src/config.py", source)
    network_path = chr(92) * 2 + "privatehost" + chr(92) + "share"
    assert text_findings("src/config.py", network_path)[0].category == "private_absolute_path"
    malformed = "https://" + "user:synthetic-url-password@model.example.invalid/v1"
    assert not text_findings("tests/test_model.py", malformed)
    assert text_findings("src/config.py", malformed)[0].category == "credential_url"
    changed = malformed.replace("synthetic-url-password", "another-fixture")
    assert text_findings("tests/test_model.py", changed)[0].category == "credential_url"


def test_shell_secrets_are_checked_but_environment_expansion_is_allowed():
    for prefix in ("export ", "$env:", ""):
        line = prefix + "SAGE_PASSWORD=" + repr("synthetic-shell-fixture")
        assert text_findings("setup.sh", line)[0].category == "hardcoded_secret"
    assert not text_findings("setup.sh", 'export SAGE_PASSWORD="${SAGE_PASSWORD}"')


def test_fallback_honors_ignores_negation_and_nested_rules(tmp_path):
    write(tmp_path, ".gitignore", "*.tmp\n.env*\n!.env.example\n/runs/\n")
    secret = "api_key = '" + "syntheticSecret" + "'"
    write(tmp_path, ".env", secret)
    write(tmp_path, ".env.example", "SAGE_LLM_API_KEY=\n")
    write(tmp_path, "runs/hidden.txt", secret)
    write(tmp_path, "throwaway.tmp", secret)
    write(tmp_path, "nested/.gitignore", "/hidden.txt\n")
    write(tmp_path, "nested/hidden.txt", secret)
    write(tmp_path, "nested/keep.txt", "ordinary input\n")
    result = audit(tmp_path)
    assert result["mode"] == "filtered_walk"
    assert result["candidate_files"] == 4
    assert result["findings"] == []


@pytest.mark.skipif(shutil.which("git") is None, reason="requires local Git only")
def test_git_checks_untracked_candidates_and_forced_tracked_artifacts(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    write(tmp_path, ".gitignore", "*.exe\n.env\n/build/\n")
    write(tmp_path, ".env", "password = '" + "ignored-fixture" + "'\n")
    write(tmp_path, "untracked.txt", "api_key = '" + "visible-fixture" + "'\n")
    executable = tmp_path / "native.exe"
    executable.write_bytes(b"MZ" + b"synthetic")
    write(tmp_path, "build/cache.txt", "generated cache\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", "native.exe", "build/cache.txt"], check=True)
    result = audit(tmp_path)
    assert result["mode"] == "git"
    assert {(item["file"], item["category"]) for item in result["findings"]} == {
        ("native.exe", "compiled_artifact"),
        ("build/cache.txt", "tracked_excluded_artifact"),
        ("untracked.txt", "hardcoded_secret"),
    }


def test_size_limits_are_mebibytes_and_do_not_echo_contents(tmp_path):
    # Sparse temporary files exercise thresholds without allocating large buffers.
    for name, size in (("warning.dat", 50 * MIB + 1), ("failure.dat", 100 * MIB + 1)):
        with (tmp_path / name).open("wb") as file:
            file.truncate(size)
    result = audit(tmp_path)
    assert result["error_count"] == result["warning_count"] == 1
    assert {item["category"] for item in result["findings"]} == {"file_over_50_mib", "file_over_100_mib"}


def test_markdown_links_must_exist_in_release_and_fences_are_ignored(tmp_path):
    write(tmp_path, ".gitignore", "/local/\n")
    write(tmp_path, "present.md", "# Existing target\n")
    write(tmp_path, "local/private.md", "local only\n")
    write(
        tmp_path,
        "README.md",
        "\n".join(
            (
                "[ok](present.md#section)",
                "[missing](absent.md)",
                "[excluded](local/private.md)",
                "[external](https://example.invalid/page)",
                "```bash",
                "[example](not-a-real-file)",
                "```",
                "[reference]: another-missing.md",
            )
        ),
    )
    result = audit(tmp_path)
    assert [(item["line"], item["category"]) for item in result["findings"]] == [
        (2, "missing_markdown_target"),
        (3, "markdown_target_not_in_release"),
        (8, "missing_markdown_target"),
    ]


def test_json_cli_fails_safely_without_echoing_secret(tmp_path):
    secret = "ghp_" + "syntheticA1" * 5
    write(tmp_path, "candidate.txt", secret)
    script = Path(__file__).resolve().parents[1] / "scripts/check_repository.py"
    process = subprocess.run(
        [sys.executable, "-B", str(script), "--root", str(tmp_path), "--json"], capture_output=True, text=True
    )
    assert process.returncode == 1
    assert secret not in process.stdout + process.stderr
    assert json.loads(process.stdout)["error_count"] == 1
