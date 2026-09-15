# Author: even
"""Put new CMake build trees on configured scratch storage while keeping their logical paths."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys


def run(cmake, args):
    scratch = os.environ.get("SAGE_CMAKE_SCRATCH_ROOT")
    if scratch and "-S" in args and "-B" in args:
        logical = Path(args[args.index("-B") + 1]).absolute()
        if not logical.exists() and not logical.is_symlink():
            root = Path(scratch).resolve()
            digest = hashlib.sha256(str(logical).encode()).hexdigest()[:24]
            target = root / digest
            target.mkdir(parents=True, exist_ok=False)
            logical.parent.mkdir(parents=True, exist_ok=True)
            logical.symlink_to(target, target_is_directory=True)
    return subprocess.call([cmake, *args])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: cmake_scratch.py CMAKE [ARGS...]")
    raise SystemExit(run(sys.argv[1], sys.argv[2:]))
