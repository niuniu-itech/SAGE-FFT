#!/usr/bin/env python3
# Author: even
"""Audit public-release candidates without printing matched content or secrets."""

import argparse
from dataclasses import asdict, dataclass
import fnmatch
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

MIB = 1024 * 1024
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "runs",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "CMakeFiles",
    "node_modules",
}
COMPILED_SUFFIXES = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".o",
    ".obj",
    ".a",
    ".lib",
    ".pyc",
    ".pyo",
    ".class",
    ".whl",
}
RAW_SUFFIXES = {".bin", ".npy", ".npz", ".pt", ".pth", ".onnx", ".safetensors"}
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".woff", ".woff2"}
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
IPV6 = re.compile(r"(?<![\w:])(?:fc|fd)[0-9a-f]{2}:[0-9a-f:]+", re.IGNORECASE)
WINDOWS_PATH = re.compile(r"(?<![\w])[A-Za-z]:[/\\](?![/\\])[^\s\"'<>`]*")
PRIVATE_PATH = re.compile(r"(?<![\w])/(?:home|Users|data\d*|root|mnt/[a-z]/Users)/[^\s\"'<>`]+").search
UNC_PATH = re.compile(
    r"(?<![^\s\"'`=])\\{2,4}[A-Za-z0-9_.-]+\\{1,2}(?![nrt](?:\\|[\"'\s]|$))[A-Za-z0-9_$.-]+"
)
KEY_PATTERN = re.compile(
    r"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|"
    r"github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{35}|"
    r"xox[baprs]-[A-Za-z0-9-]{20,})\b"
)
PEM_PATTERN = re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----")
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|password|passwd|access[_-]?token|secret[_-]?key|client[_-]?secret|token)"
    r"\b[\"']?\s*[:=]\s*[\"']([^\"'\n]{3,})[\"']"
)
ENV_SECRET = re.compile(
    r"(?i)^\s*(?:export\s+|\$env:)?[A-Z_]*(?:API_KEY|PASSWORD|ACCESS_TOKEN|SECRET_KEY)\s*=\s*"
    r"(?:\"[^\"$()]{3,}\"|'[^']{3,}'|[A-Za-z0-9_/+=.-]{3,})\s*(?:#.*)?$"
)
USERINFO_URL = re.compile(r"https?://[^\s/\"'<>`]+@[^\s/\"'<>`]+")
FILE_URL = re.compile(r"file://[^\s\"'<>`]+")  # release-audit: rule-definition
MD_INLINE = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[\"'][^\n]*?[\"'])?\s*\)")
MD_REFERENCE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)")

# Exact, named malformed-URL fixtures in the intercepted model transport test.
# No entire file or category is exempt; any changed/extra literal is checked.
SAFE_MATCHES = {
    ("tests/test_model.py", "credential_url", "https://user:synthetic-url-password@model.example.invalid"),
    ("tests/test_model.py", "credential_url", "http://user@model.example.invalid"),
    ("tests/test_model.py", "local_file_url", "file:///tmp/model-test"),
}


@dataclass(frozen=True, order=True)
class Finding:
    file: str
    line: int
    category: str
    severity: str = "error"


def excluded(relative):
    return any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in Path(relative).parts)


def git_candidates(root):
    """Include tracked paths even when ignored, plus nonignored untracked paths."""

    def git(args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout

    try:
        git(["rev-parse", "--show-toplevel"])
        paths = git(["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "."])
        tracked = git(["ls-files", "-z", "--cached", "--", "."])
    except (OSError, subprocess.CalledProcessError):
        return None
    decode = lambda value: {os.fsdecode(p).replace("\\", "/") for p in value.split(b"\0") if p}
    return sorted(decode(paths)), decode(tracked)


def ignore_rules(folder, root):
    path = folder / ".gitignore"
    if not path.is_file():
        return []
    base = folder.relative_to(root).as_posix()
    base = "" if base == "." else base + "/"
    try:
        return [
            (base, line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except (OSError, UnicodeError):
        return []


def ignored(relative, is_dir, rules):
    """Fallback supports this repository's ordered globs, anchors and negations."""
    state = False
    for base, pattern in rules:
        if not relative.startswith(base):
            continue
        local = relative[len(base) :]
        negate = pattern.startswith("!")
        pattern = pattern[1:] if negate else pattern
        directory_only = pattern.endswith("/")
        pattern = pattern.rstrip("/")
        anchored = pattern.startswith("/")
        pattern = pattern.lstrip("/")
        if not pattern:
            continue
        if directory_only:
            parts = local.split("/")
            directories = parts if is_dir else parts[:-1]
            choices = ["/".join(directories[: i + 1]) for i in range(len(directories))]
            if not anchored and "/" not in pattern:
                choices += directories
        else:
            choices = [local] if anchored or "/" in pattern else local.split("/")
        if any(fnmatch.fnmatchcase(choice, pattern) for choice in choices):
            state = not negate
    return state


def fallback_candidates(root):
    paths = []

    def visit(folder, inherited):
        rules = inherited + ignore_rules(folder, root)
        for path in sorted(folder.iterdir()):
            relative = path.relative_to(root).as_posix()
            is_dir = path.is_dir() and not path.is_symlink()
            if excluded(relative) or ignored(relative, is_dir, rules):
                continue
            if is_dir:
                visit(path, rules)
            else:
                paths.append(relative)

    visit(root, [])
    return paths, set()


def private_ip(literal):
    try:
        address = ipaddress.ip_address(literal.rstrip(":"))
    except ValueError:
        return False
    if address.version == 6:
        return address in ipaddress.ip_network((int("fc" + "0" * 30, 16), 7))
    a, b, _, _ = (int(part) for part in literal.split("."))
    return a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168)


def text_findings(relative, text):
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        if relative == "scripts/check_repository.py" and line.endswith("# release-audit: rule-definition"):
            continue
        categories = set()
        if any(private_ip(match.group()) for pattern in (IPV4, IPV6) for match in pattern.finditer(line)):
            categories.add("private_ip")
        if WINDOWS_PATH.search(line) or PRIVATE_PATH(line) or UNC_PATH.search(line):
            categories.add("private_absolute_path")
        if KEY_PATTERN.search(line):
            categories.add("likely_api_key")
        if PEM_PATTERN.search(line):
            categories.add("private_key")
        if SECRET_ASSIGNMENT.search(line) or ENV_SECRET.search(line):
            categories.add("hardcoded_secret")
        for category, pattern in (("credential_url", USERINFO_URL), ("local_file_url", FILE_URL)):
            if any(
                (relative, category, match.group()) not in SAFE_MATCHES for match in pattern.finditer(line)
            ):
                categories.add(category)
        findings.extend(Finding(relative, number, category) for category in categories)
    return findings


def markdown_findings(root, relative, text, candidates):
    findings = []
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            delimiter = marker.group(1)
            if fence is None:
                fence = delimiter
            elif delimiter[0] == fence[0] and len(delimiter) >= len(fence):
                fence = None
            continue
        if fence:
            continue
        targets = [m.group(1) for m in MD_INLINE.finditer(line)]
        reference = MD_REFERENCE.match(line)
        if reference:
            targets.append(reference.group(1))
        for target in targets:
            target = target.strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            local = unquote(parsed.path)
            destination = (
                (root / local.lstrip("/")) if local.startswith("/") else (root / relative).parent / local
            )
            resolved = destination.resolve()
            if not resolved.is_relative_to(root):
                findings.append(Finding(relative, number, "markdown_link_outside_repository"))
                continue
            key = resolved.relative_to(root).as_posix()
            if not resolved.exists():
                findings.append(Finding(relative, number, "missing_markdown_target"))
            elif resolved.is_file() and key not in candidates:
                findings.append(Finding(relative, number, "markdown_target_not_in_release"))
    return findings


def audit(root):
    root = Path(root).resolve(strict=True)
    result = git_candidates(root)
    mode = "git" if result is not None else "filtered_walk"
    paths, tracked = result if result is not None else fallback_candidates(root)
    candidate_set = {p for p in paths if not excluded(p)}
    findings, scanned = [], 0
    for relative in paths:
        if excluded(relative):
            if relative in tracked:
                findings.append(Finding(relative, 0, "tracked_excluded_artifact"))
            continue
        path = root / relative
        if path.is_symlink():
            if not path.resolve().is_relative_to(root):
                findings.append(Finding(relative, 0, "symlink_outside_repository"))
            continue
        if not path.exists():  # Tracked files deleted in the working tree are not release contents.
            continue
        if not path.is_file():
            continue
        scanned += 1
        try:
            size = path.stat().st_size
            if size > 100 * MIB:
                findings.append(Finding(relative, 0, "file_over_100_mib"))
                continue
            if size > 50 * MIB:
                findings.append(Finding(relative, 0, "file_over_50_mib", "warning"))
            suffix = path.suffix.lower()
            with path.open("rb") as stream:
                header = stream.read(4096)
            if (
                suffix in COMPILED_SUFFIXES
                or path.name in {"executable", "sage_fft", "sage_cuda_fft"}
                or header.startswith(
                    (b"\x7fELF", b"MZ", b"!<arch>\n", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf")
                )
            ):
                findings.append(Finding(relative, 0, "compiled_artifact"))
                continue
            if suffix in RAW_SUFFIXES or path.name.endswith(".bin.after"):
                findings.append(Finding(relative, 0, "raw_binary_or_dataset"))
                continue
            if suffix in MEDIA_SUFFIXES or b"\0" in header:
                continue
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeError:
                findings.append(Finding(relative, 0, "unscanned_non_utf8_file", "warning"))
                continue
            # The scanner's exact safe-fixture declarations are data, not endpoints.
            if relative == "scripts/check_repository.py":
                start = text.index("SAFE_MATCHES = {")
                end = text.index("\n}", start) + 2
                text = text[:start] + "\n" * text[start:end].count("\n") + text[end:]
            findings.extend(text_findings(relative, text))
            if suffix.lower() in (".md", ".markdown"):
                findings.extend(markdown_findings(root, relative, text, candidate_set))
        except OSError:
            findings.append(Finding(relative, 0, "unreadable_file"))
    findings = sorted(set(findings))
    return {
        "mode": mode,
        "candidate_files": len(paths),
        "scanned_files": scanned,
        "error_count": sum(item.severity == "error" for item in findings),
        "warning_count": sum(item.severity == "warning" for item in findings),
        "findings": [asdict(item) for item in findings],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="print JSON with locations/categories only")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument("--strict-warnings", action="store_true")
    args = parser.parse_args()
    try:
        result = audit(args.root)
    except (OSError, ValueError):
        result = {
            "mode": "unavailable",
            "candidate_files": 0,
            "scanned_files": 0,
            "error_count": 1,
            "warning_count": 0,
            "findings": [asdict(Finding(".", 0, "repository_unavailable"))],
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for item in result["findings"]:
            print(f"{item['severity'].upper()} {item['file']}:{item['line']} {item['category']}")
        print(
            f"Audit {result['mode']}: {result['scanned_files']} files; "
            f"{result['error_count']} errors; {result['warning_count']} warnings"
        )
    return int(bool(result["error_count"] or (args.strict_warnings and result["warning_count"])))


if __name__ == "__main__":
    raise SystemExit(main())
