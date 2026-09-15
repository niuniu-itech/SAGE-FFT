# Author: even
"""CPU semantic counterexamples: invertibility alone is not FFT compatibility."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import json
import math
from pathlib import Path
import numpy as np
from .fft_ir import Contract, metrics, import_cuda
from .cuda_source import source
from .validate_semantics import inputs


def main():
    prepare_workspace()
    c = Contract((8, 16), 1)
    axes = (1, 2)
    M = math.prod(c.shape)
    f = lambda x: np.fft.fftn(x, axes=axes)
    g = lambda x: np.fft.ifftn(x, axes=axes)
    # Inverse-paired errors deliberately pass round-trip tests on arbitrary x.
    transforms = {
        "correct": (f, g),
        "paired_twiddle_sign": (lambda x: g(x) * M, lambda x: f(x) / M),
        "paired_amplitude": (lambda x: 2 * f(x), lambda x: g(x) / 2),
        "bin_shift": (lambda x: np.roll(f(x), 1, axis=2), lambda x: g(np.roll(x, -1, axis=2))),
        "layout_permutation": (
            lambda x: np.swapaxes(f(x), 1, 2).reshape(x.shape),
            lambda x: g(np.swapaxes(x.reshape(1, 16, 8), 1, 2)),
        ),
        "missing_inverse_scale": (f, lambda x: g(x) * M),
        "double_inverse_scale": (f, lambda x: g(x) / M),
        "valid_axis_reordering": (
            lambda x: np.fft.fftn(x, axes=(2, 1)),
            lambda x: np.fft.ifftn(x, axes=(2, 1)),
        ),
    }
    rows = []
    for seed in (1, 2, 3):
        for kind in ("random", "impulse", "constant", "tone"):
            x, h = inputs(c, seed, kind)
            for name, (ff, gg) in transforms.items():
                checks = {
                    "forward": metrics(ff(x), f(x))["pass_correctness"],
                    "inverse": metrics(gg(x), g(x))["pass_correctness"],
                    "weighted_pipeline": metrics(gg(ff(x) * h), g(f(x) * h))["pass_correctness"],
                    "round_trip": metrics(gg(ff(x)), x)["pass_correctness"],
                }
                rows.append(dict(mutant=name, seed=seed, input=kind, checks=checks))
    summary = {
        k: {
            test: sum(r["checks"][test] for r in rows if r["mutant"] == k)
            for test in ("forward", "inverse", "weighted_pipeline", "round_trip")
        }
        for k in transforms
    }
    assert all(
        summary["correct"][x] == 12 and summary["valid_axis_reordering"][x] == 12 for x in summary["correct"]
    )
    assert summary["bin_shift"]["round_trip"] == 12 and summary["paired_amplitude"]["weighted_pipeline"] == 12
    out = workspace() / "evidence" / "v46_semantic_mutants.json"
    out.write_text(
        json.dumps(
            dict(
                scope="CPU semantic counterexamples, not NPU measurements or compiler rejection rates",
                tests_per_mutant=12,
                summary=summary,
                cases=rows,
            ),
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
