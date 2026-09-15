# Author: even
"""Independent double-precision DFT and FFT oracles; no target timing simulation."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import json
import hashlib
from pathlib import Path
import numpy as np
from .fft_ir import Contract, candidates, execute, metrics

WORKLOADS = [
    ((64,), 4),
    ((256,), 4),
    ((8, 16), 1),
    ((16, 32), 1),
    ((4, 8, 8), 1),
    ((8, 8, 8), 1),
    ((8, 16, 16), 1),
]


def inputs(c, seed, kind="random"):
    rng = np.random.default_rng(seed)
    shape = (c.batch,) + c.shape
    x = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    if kind == "impulse":
        x.fill(0)
        x[(slice(None),) + (0,) * len(c.shape)] = 1
    if kind == "constant":
        x.fill(1 + 0.25j)
    if kind == "tone":
        coords = np.indices(c.shape)
        phase = sum((i + 1) * coords[i] / n for i, n in enumerate(c.shape))
        x[:] = np.exp(2j * np.pi * phase).astype(np.complex64)
    h = (rng.uniform(0.25, 1, size=shape) * np.exp(1j * rng.uniform(-np.pi, np.pi, size=shape))).astype(
        np.complex64
    )
    return x, h


def reference(c, x, h, mode="pipeline"):
    axes = tuple(range(1, x.ndim))
    if mode == "forward":
        return np.fft.fftn(x.astype(np.complex128), axes=axes)
    if mode == "inverse":
        return np.fft.ifftn(x.astype(np.complex128), axes=axes)
    return np.fft.ifftn(np.fft.fftn(x.astype(np.complex128), axes=axes) * h, axes=axes)


def main():
    prepare_workspace()
    records = []
    input_manifest = []
    for shape, batch in WORKLOADS:
        c = Contract(shape, batch)
        for seed in (1, 2, 3):
            for kind in ("random", "impulse", "constant", "tone"):
                x, h = inputs(c, seed, kind)
                input_manifest.append(
                    dict(
                        shape=shape,
                        batch=batch,
                        seed=seed,
                        input_kind=kind,
                        x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                        h_sha256=hashlib.sha256(h.tobytes()).hexdigest(),
                    )
                )
                for a in candidates():
                    for mode in ("forward", "inverse", "pipeline"):
                        r = metrics(execute(c, a, x, h, mode), reference(c, x, h, mode))
                        records.append(
                            dict(
                                shape=shape, batch=batch, seed=seed, input_kind=kind, action=a, mode=mode, **r
                            )
                        )
    # Direct dense DFT for an additional independent small multidimensional oracle.
    c = Contract((4, 8), 1)
    x, h = inputs(c, 9)
    z = x.astype(np.complex128)
    for ax, n in enumerate(c.shape, 1):
        matrix = np.exp(-2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n)
        z = np.moveaxis(np.moveaxis(z, ax, -1) @ matrix.T, -1, ax)
    direct = metrics(execute(c, candidates()[0], x, h, "forward"), z)
    out = workspace() / "evidence"
    out.mkdir(exist_ok=True)
    report = dict(
        backend="CPU IR interpreter, not NPU execution",
        numpy_version=np.__version__,
        arithmetic="explicit FP32 component products followed by FP32 addition/subtraction",
        criterion="abs(error) <= 1e-4 + 1e-4 * abs(reference), every element",
        implementation_sha256=hashlib.sha256(Path(execute.__code__.co_filename).read_bytes()).hexdigest(),
        input_manifest=input_manifest,
        records=records,
        direct_dft=direct,
    )
    (out / "cpu_semantics.json").write_text(json.dumps(report, indent=2))
    passed = sum(r["pass_correctness"] for r in records)
    print(
        json.dumps(
            dict(
                passed=passed,
                total=len(records),
                max_relative_l2=max(r["relative_l2"] for r in records),
                direct_dft=direct,
            )
        )
    )
    assert passed == len(records) and direct["pass_correctness"]


if __name__ == "__main__":
    main()
