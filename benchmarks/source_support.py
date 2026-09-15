# Author: even
"""Shared source templates for the fixed-size native benchmarks."""

from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent / "templates"


def host_source(launches, precision):
    if launches not in (3, 5, 6, 20) or precision not in ("fp16", "fp32"):
        raise ValueError("unsupported launch count or precision")
    text = (TEMPLATES / "npu_main.cpp.in").read_text(encoding="utf-8")
    values = {
        "LAUNCH_INCLUDES": "\n".join(f'#include "aclrtlaunch_fft_step_{j}.h"' for j in range(launches)),
        "SCALAR": "uint16_t" if precision == "fp16" else "float",
        "LAUNCHES": "\n".join(
            f'        ck(ACLRT_LAUNCH_KERNEL(fft_step_{j})(blocks, stream, src, h, dst), "launch {j}");\n'
            "        src = dst; dst = (dst == b ? c : b);"
            for j in range(launches)
        ),
        "PRECISION": precision,
        "COUNT": str(launches),
    }
    for key, value in values.items():
        text = text.replace(f"@{key}@", value)
    return text


def write_support(folder, launches, precision):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "main.cpp").write_text(host_source(launches, precision), encoding="utf-8", newline="\n")
    (folder / "CMakeLists.txt").write_text(
        (TEMPLATES / "CMakeLists.txt").read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
