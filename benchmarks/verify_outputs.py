# Author: even
"""Validate native output files against a complex128 NumPy CPU oracle."""

import argparse
import json
from pathlib import Path

import numpy as np

from .generate_inputs import DTYPES, SHAPE


def read_complex(path, precision):
    dtype = DTYPES[precision]
    path = Path(path)
    expected = 1024 * dtype.itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected exactly {expected} bytes")
    scalars = np.fromfile(path, dtype=dtype).astype(np.float64).reshape(-1, 2)
    result = (scalars[:, 0] + 1j * scalars[:, 1]).reshape(SHAPE)
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: contains non-finite data")
    return result


def oracle(input_dir, precision, empty_control=False):
    x = read_complex(Path(input_dir) / "x.bin", precision)
    h = read_complex(Path(input_dir) / "h.bin", precision)
    return x if empty_control else np.fft.ifftn(np.fft.fftn(x) * h)


def error_metrics(actual, reference, precision):
    if precision not in DTYPES:
        raise ValueError("precision must be fp32 or fp16")
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or actual.size == 0:
        raise ValueError("actual/reference shapes must match and be nonempty")
    if not np.isfinite(reference).all():
        raise ValueError("reference contains non-finite values")
    finite = bool(np.isfinite(actual).all())
    if not finite:
        return {"passed": False, "finite": False, "relative_l2": None, "max_abs": None}
    error = np.abs(actual - reference)
    norm = float(np.linalg.norm(reference))
    err_norm = float(np.linalg.norm(actual - reference))
    relative = err_norm / norm if norm else (0.0 if not err_norm else None)
    atol = rtol = 0.003 if precision == "fp16" else 1e-4
    elementwise = bool(np.all(error <= atol + rtol * np.abs(reference)))
    relative_pass = precision == "fp32" or (relative is not None and relative <= 0.003)
    return {
        "passed": elementwise and relative_pass,
        "finite": True,
        "relative_l2": relative,
        "max_abs": float(np.max(error)),
        "elementwise_passed": elementwise,
        "atol": atol,
        "rtol": rtol,
        "relative_l2_limit": 0.003 if precision == "fp16" else None,
    }


def verify(input_dir, output, precision, empty_control=False):
    reference = oracle(input_dir, precision, empty_control)
    checks = []
    for suffix in ("", ".after"):
        path = Path(str(output) + suffix)
        checks.append(
            {
                "phase": "before" if not suffix else "after",
                **error_metrics(read_complex(path, precision), reference, precision),
            }
        )
    return {
        "precision": precision,
        "oracle": "numpy.complex128 FFT weighted inverse FFT",
        "empty_control": empty_control,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="also requires OUTPUT.after")
    parser.add_argument("--precision", choices=DTYPES, required=True)
    parser.add_argument("--empty-control", action="store_true")
    args = parser.parse_args()
    result = verify(args.inputs, args.output, args.precision, args.empty_control)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
