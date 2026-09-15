# Author: even
"""Generate native-baseline inputs and IR-derived controls for cubic FP32 pipelines."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from sage_fft.cuda_source import source
from sage_fft.fft_ir import Contract
from sage_fft.lower import emit
from sage_fft.validate_semantics import inputs


def generate(output, sizes=(8, 16)):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new output directory")
    template = Path(__file__).with_name("matched_main.cpp.in").read_text(encoding="utf-8")
    records = []
    for n in sizes:
        if n not in (8, 16, 32, 64):
            raise ValueError("supported comparison sizes are 8, 16, 32 and 64")
        c = Contract((n, n, n), 1)
        for seed in (1, 2, 3):
            folder = output / "inputs" / f"n{n}" / f"input_{seed}"
            folder.mkdir(parents=True)
            x, h = inputs(c, seed)
            x.astype("<c8").tofile(folder / "x.bin")
            h.astype("<c8").tofile(folder / "h.bin")
            forward = np.fft.fftn(x.astype(np.complex128), axes=(1, 2, 3))
            y = np.fft.ifftn(forward * h.astype(np.complex128), axes=(1, 2, 3))
            forward.astype("<c16").tofile(folder / "reference_fft.bin")
            y.astype("<c16").tofile(folder / "reference.bin")
        for variant, action in (
            ("staged", dict(group=1, cores=8, fusion=False)),
            ("sage", dict(group=16, cores=8, fusion=True)),
        ):
            folder = output / "controls" / f"n{n}_{variant}"
            ir = emit(c, action, folder)
            text, _ = source(c.shape, c.batch)
            (folder / "source.cu").write_text(text, encoding="utf-8", newline="\n")
            count = len(ir["stages"])
            kernel_hash = hashlib.sha256((folder / "kernels.cpp").read_bytes()).hexdigest()
            host = template
            replacements = {
                "@INCLUDES@": "\n".join(f'#include "aclrtlaunch_fft_step_{i}.h"' for i in range(count)),
                "@N@": str(n),
                "@LAUNCHES@": str(count),
                "@VARIANT@": variant,
                "@KERNEL_SHA@": kernel_hash,
                "@CALLS@": "\n".join(
                    f'ck(ACLRT_LAUNCH_KERNEL(fft_step_{i})(blocks, stream, src, h, dst), "launch {i}");\n'
                    "src = dst; dst = (dst == b ? c : b);"
                    for i in range(count)
                ),
            }
            for old, new in replacements.items():
                host = host.replace(old, new)
            if re.search(r"@[A-Z_]+@", host):
                raise ValueError("unresolved host placeholder")
            (folder / "main.cpp").write_text(host, encoding="utf-8", newline="\n")
            records.append(dict(n=n, variant=variant, launches=count, kernel_sha256=kernel_hash))
    manifest = {
        "shape_sizes": list(sizes),
        "seeds": [1, 2, 3],
        "controls": records,
        "numpy_version": np.__version__,
        "files": {
            p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", default="8,16")
    args = parser.parse_args()
    result = generate(args.output, tuple(map(int, args.sizes.split(","))))
    print(json.dumps({"controls": len(result["controls"]), "files": len(result["files"])}))


if __name__ == "__main__":
    main()
