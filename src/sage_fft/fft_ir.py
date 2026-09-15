# Author: even
"""FFT contract, registered CUDA adapter and executable stage semantics.

The adapter extracts a Contract from the supported literal cuFFT source family.
Realizations lower to ordered stage records describing axis, butterfly range,
direction and epilogues. The same records drive source emission and the CPU
interpreter used for semantic checks; CPU runtime is not NPU performance.
"""

from .config import workspace, remote_root, npu_environment, connect_ssh
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import re
import numpy as np


@dataclass(frozen=True)
class Contract:
    shape: tuple
    batch: int = 4
    dtype: str = "complex64"
    layout: str = "interleaved_row_major"
    bin_order: str = "natural"
    inverse_normalization: str = "1/product(shape)"
    atol: float = 1e-4
    rtol: float = 1e-4

    def validate(self):
        if not isinstance(self.shape, tuple) or not 1 <= len(self.shape) <= 3:
            raise ValueError("shape must be a tuple of one to three axes")
        if any(type(n) is not int or not 2 <= n <= 1024 or n & (n - 1) for n in self.shape):
            raise ValueError("axes must be integer powers of two from 2 to 1024")
        if type(self.batch) is not int or self.batch <= 0:
            raise ValueError("batch must be a positive integer")
        if self.dtype != "complex64" or self.layout != "interleaved_row_major":
            raise ValueError("only interleaved row-major complex64 is supported")
        if self.bin_order != "natural" or self.inverse_normalization != "1/product(shape)":
            raise ValueError("natural bins and normalized inverse are required")
        for value in (self.atol, self.rtol):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("tolerances must be finite nonnegative numbers")
        return self


def import_cuda(source):
    """Accept only the emitted, literal plan and tagged pipeline, reject other sources.

    This adapter does not parse arbitrary CUDA or recover opaque GPU kernels.
    """
    plans = re.findall(r"cufftPlan([123])d\(&plan,\s*([\d, ]+),\s*CUFFT_C2C(?:,\s*(\d+))?\)", source)
    if len(plans) != 1:
        raise ValueError("expected one literal cufftPlan1d/2d/3d C2C plan")
    rank, dimensions, batch = plans[0]
    shape = tuple(int(x.strip()) for x in dimensions.split(","))
    if len(shape) != int(rank):
        raise ValueError("plan rank mismatch")
    tags = re.findall(r"SAGE_CONTRACT (\{[^\n]+\})", source)
    if len(tags) != 1:
        raise ValueError("explicit pipeline contract required")
    tag = json.loads(tags[0])
    if tag != {"pipeline": "fft_complexmul_ifft", "normalization": "1/N", "layout": "interleaved_row_major"}:
        raise ValueError("unsupported pipeline contract")
    if not re.search(r"cufftExecC2C\(plan,\s*x,\s*z,\s*CUFFT_FORWARD\)", source):
        raise ValueError("forward call missing")
    if not re.search(r"cufftExecC2C\(plan,\s*z,\s*y,\s*CUFFT_INVERSE\)", source):
        raise ValueError("inverse call missing")
    c = Contract(shape, int(batch) if batch else 1)
    c.validate()
    # The registered source family includes the pointwise and scale semantics.
    # A matching plan/tag alone must not authorize an arbitrary CUDA kernel body.
    from .cuda_source import render

    if re.sub(r"\s+", "", source) != re.sub(r"\s+", "", render(shape, c.batch)):
        raise ValueError("source is outside the registered CUDA pipeline family")
    return c


def candidates():
    return [dict(group=g, cores=c, fusion=f) for g in (1, 2, 16) for c in (1, 4, 8) for f in (False, True)]


def validate_action(c, a):
    if not isinstance(a, dict):
        raise ValueError("realization must be an object")
    if "forward_masks" in a:
        from .mask_ir import validate

        return validate(c, a)
    c.validate()
    if (
        set(a) != {"group", "cores", "fusion"}
        or type(a.get("group")) is not int
        or type(a.get("cores")) is not int
        or type(a.get("fusion")) is not bool
        or a not in candidates()
    ):
        raise ValueError("action outside compiler-enumerated space")
    # Standalone pointwise kernels process 32 complex elements per DMA chunk.
    # The lowering has no padded tail path, so reject unsupported tails.
    if math.prod(c.shape) * c.batch % 32:
        raise ValueError("pointwise DMA tail is unsupported")
    # Three interleaved buffers: raw input, butterfly workspace, and weights.
    for axis, n in enumerate(c.shape):
        stride = math.prod(c.shape[axis + 1 :])
        lanes = min(stride, 8)
        dma_complex = n if stride == 1 else lanes
        if dma_complex % 4:
            raise ValueError("axis DMA transfer must be a multiple of 32 bytes")
        if 3 * 2 * n * lanes * 4 > 256 * 1024:
            raise ValueError("UB capacity exceeded")
        lines = math.prod(c.shape) * c.batch // n
        if lines % lanes:
            raise ValueError("incomplete output ownership group")


def stages(c, a, mode="pipeline"):
    if mode not in ("pipeline", "forward", "inverse"):
        raise ValueError("mode must be pipeline, forward or inverse")
    if not isinstance(a, dict):
        raise ValueError("realization must be an object")
    if "forward_masks" in a:
        from .mask_ir import stages as boundary_stages

        return boundary_stages(c, a, mode)
    validate_action(c, a)
    result = []
    for inverse in (False, True) if mode == "pipeline" else (mode == "inverse",):
        for axis in reversed(range(len(c.shape))):
            bits = int(math.log2(c.shape[axis]))
            for first in range(0, bits, a["group"]):
                last = min(bits, first + a["group"])
                tail = axis == 0 and last == bits
                result.append(
                    dict(
                        kind="fft",
                        axis=axis,
                        first=first,
                        last=last,
                        inverse=inverse,
                        multiply=bool(mode == "pipeline" and not inverse and tail and a["fusion"]),
                        scale=bool(inverse and tail and a["fusion"]),
                    )
                )
        if mode == "pipeline" and not inverse and not a["fusion"]:
            result.append(dict(kind="multiply"))
        if inverse and not a["fusion"]:
            result.append(dict(kind="scale"))
    return result


def identity(c, a, mode="pipeline"):
    value = dict(contract=asdict(c), action=a, mode=mode, stages=stages(c, a, mode))
    raw = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest(), value


def _multiply_complex64(a, b):
    """Round each scalar FP32 product before the component add/subtract.

    NumPy's complex multiplication ufunc can use a different rounding path
    across versions or CPU dispatch implementations. Keeping the real scalar
    operations explicit makes this interpreter's arithmetic stable; it is not
    a claim of bitwise agreement with compiled GPU or NPU instructions.
    """
    a = np.asarray(a, dtype=np.complex64)
    b = np.asarray(b, dtype=np.complex64)
    result = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.complex64)
    result.real = a.real * b.real - a.imag * b.imag
    result.imag = a.real * b.imag + a.imag * b.real
    return result


def execute(c, a, x, h, mode="pipeline"):
    c.validate()
    expected = (c.batch,) + c.shape
    if np.shape(x) != expected or (mode == "pipeline" and np.shape(h) != expected):
        raise ValueError("inputs must match the contract batch and axes")
    z = x.copy().astype(np.complex64)
    for step in stages(c, a, mode):
        if step["kind"] == "multiply":
            z = _multiply_complex64(z, h)
            continue
        if step["kind"] == "scale":
            z = (z / math.prod(c.shape)).astype(np.complex64)
            continue
        ax = step["axis"] + 1
        n = c.shape[step["axis"]]
        v = np.moveaxis(z, ax, -1).copy()
        if step["first"] == 0:
            bits = int(math.log2(n))
            rev = [int(format(k, "0%db" % bits)[::-1], 2) for k in range(n)]
            v = v[..., rev].copy()
        for level in range(step["first"], step["last"]):
            m = 2 ** (level + 1)
            half = m // 2
            w = np.exp((1 if step["inverse"] else -1) * 2j * np.pi * np.arange(half) / m).astype(np.complex64)
            for base in range(0, n, m):
                u = v[..., base : base + half].copy()
                t = _multiply_complex64(v[..., base + half : base + m], w)
                v[..., base : base + half] = u + t
                v[..., base + half : base + m] = u - t
        z = np.moveaxis(v, -1, ax)
        if step["multiply"]:
            z = _multiply_complex64(z, h)
        if step["scale"]:
            z = (z / math.prod(c.shape)).astype(np.complex64)
    return z


def metrics(y, ref):
    delta = np.abs(y.astype(np.complex128) - ref)
    return dict(
        pass_correctness=bool(np.all(delta <= 1e-4 + 1e-4 * np.abs(ref))),
        max_abs=float(delta.max()),
        relative_l2=float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(ref.ravel()), 1e-30)),
    )
