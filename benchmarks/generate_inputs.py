# Author: even
"""Generate deterministic 8 x 8 x 8 interleaved complex inputs without hardware."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SHAPE = (8, 8, 8)
DTYPES = {"fp32": np.dtype("<f4"), "fp16": np.dtype("<f2")}


def make_inputs(seed, precision="fp32"):
    """Return 1024 real scalars per array in real/imaginary interleaved order.

    Seeds 1--3 reproduce validate_semantics.inputs(random), including its FP32
    rounding before FP16 conversion. Seeds 4--5 retain the FP16 holdout rule.
    """
    if precision not in DTYPES or seed not in range(1, 6):
        raise ValueError("precision must be fp32/fp16; seed must be 1 through 5")
    if precision == "fp32" and seed > 3:
        raise ValueError("the FP32 input generator has only seeds 1 through 3")
    rng = np.random.Generator(np.random.PCG64(seed))
    if seed <= 3:
        shape = (1,) + SHAPE
        x = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype("<c8")
        h = (rng.uniform(0.25, 1, size=shape) * np.exp(1j * rng.uniform(-np.pi, np.pi, size=shape))).astype(
            "<c8"
        )
        return tuple(z.view("<f4").reshape(-1).astype(DTYPES[precision]) for z in (x, h))
    x = rng.normal(0, 0.25, 1024).astype(DTYPES[precision])
    h = rng.uniform(-0.5, 0.5, 1024).astype(DTYPES[precision])
    return x, h


def generate(output, precision="fp32", seeds=None):
    output = Path(output)
    seeds = list(seeds if seeds is not None else range(1, 4 if precision == "fp32" else 6))
    arrays = [(seed, make_inputs(seed, precision)) for seed in seeds]
    hashes = {}
    for seed, values in arrays:
        folder = output / f"input_{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, array in zip(("x", "h"), values):
            path = folder / f"{name}.bin"
            array.tofile(path)
            hashes[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "precision": precision,
        "shape": list(SHAPE),
        "batch": 1,
        "layout": "C-order, little-endian, interleaved real/imaginary",
        "rng": "numpy.random.Generator(PCG64(seed))",
        "numpy_version": np.__version__,
        "seeds": seeds,
        "sha256": hashes,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=DTYPES, default="fp32")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.precision, args.seeds), indent=2))


if __name__ == "__main__":
    main()
