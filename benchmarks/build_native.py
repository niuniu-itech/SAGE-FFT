# Author: even
"""Build native sources locally using an explicitly selected SDK/toolchain."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def cann_environment(script=None):
    """Optionally source a caller-supplied CANN set_env.sh in a child Bash."""
    env = os.environ.copy()
    if script is None:
        return env
    script = Path(script).resolve(strict=True)
    bash = shutil.which("bash")
    if not bash:
        raise RuntimeError("sourcing CANN requires Bash; alternatively configure your shell first")
    process = subprocess.run(
        [bash, "-c", 'set -e; source "$1" >/dev/null; env -0', "sage-cann-env", str(script)],
        env=env,
        capture_output=True,
        check=True,
    )
    return dict(item.decode().split("=", 1) for item in process.stdout.split(b"\0") if item)


def build(
    source,
    build_dir,
    backend,
    cann_root=None,
    cann_env=None,
    soc="Ascend310P1",
    cuda_arch="80",
    compiler=None,
    jobs=2,
):
    source, build_dir = Path(source).resolve(strict=True), Path(build_dir).resolve()
    if backend not in ("npu", "cuda"):
        raise ValueError("backend must be npu or cuda")
    if not (source / "CMakeLists.txt").is_file():
        raise ValueError("--source must name a native variant directory containing CMakeLists.txt")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    env = cann_environment(cann_env if backend == "npu" else None)
    command = ["cmake", "-S", str(source), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"]
    if backend == "npu":
        root = cann_root or env.get("ASCEND_CANN_PACKAGE_PATH") or env.get("ASCEND_HOME_PATH")
        if not root:
            raise ValueError("set ASCEND_CANN_PACKAGE_PATH/ASCEND_HOME_PATH or pass --cann-root")
        command.extend([f"-DASCEND_CANN_PACKAGE_PATH={Path(root).resolve()}", f"-DSOC_VERSION={soc}"])
    else:
        command.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
    if compiler:
        command.append(f"-DCMAKE_CXX_COMPILER={compiler}")
    subprocess.run(command, env=env, check=True)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release", "--parallel", str(jobs)],
        env=env,
        check=True,
    )
    manifest = {
        "backend": backend,
        "source": str(source),
        "build_dir": str(build_dir),
        "configure_command": command,
        "hardware_executed": False,
    }
    (build_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("npu", "cuda"), required=True)
    parser.add_argument("--cann-root", type=Path)
    parser.add_argument("--cann-env", type=Path, help="optional local CANN set_env.sh")
    parser.add_argument("--soc", default="Ascend310P1")
    parser.add_argument(
        "--cuda-arch", default="80", help="CMake CUDA architectures, measured default 80 (A100)"
    )
    parser.add_argument(
        "--compiler", help="optional C++ compiler path; otherwise CMake uses the active environment"
    )
    parser.add_argument("--jobs", type=int, default=2)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
