# Author: even
"""Hierarchical FFT realizations and their legal transformation menu.

Forward and inverse masks describe stage boundaries independently. Fusion flags
attach pointwise epilogues, while cores and pack specify the shared schedule.
Actions produce validated realizations; stages converts them to the ordered IR
consumed by the backend. A model selects an action, never kernel source code.
"""

from .config import workspace, remote_root, npu_environment, connect_ssh
import copy
import itertools
import json
import math
from .fft_ir import Contract

WORKLOADS = {
    "M1": (Contract((64,), 4), (8,), True),
    "M2": (Contract((8, 8), 1), (4, 8), False),
    "M3": (Contract((8, 8, 8), 1), (4, 8, 16), False),
}


def initial(c):
    masks = [[1] * (n.bit_length() - 2) for n in c.shape]
    return dict(
        forward_masks=copy.deepcopy(masks),
        inverse_masks=copy.deepcopy(masks),
        fuse_multiply=False,
        fuse_scale=False,
        cores=1,
        pack=8,
    )


def validate(c, r):
    c.validate()
    if not isinstance(r, dict):
        raise ValueError("realization must be an object")
    if set(r) != set(initial(c)):
        raise ValueError("unexpected realization fields")
    if type(r["cores"]) is not int or r["cores"] not in (1, 4, 8):
        raise ValueError("core count")
    if type(r["pack"]) is not int or r["pack"] not in (4, 8, 16):
        raise ValueError("packing")
    for k in ("fuse_multiply", "fuse_scale"):
        if type(r[k]) is not bool:
            raise ValueError("fusion must be Boolean")
    for k in ("forward_masks", "inverse_masks"):
        if not isinstance(r[k], (list, tuple)):
            raise ValueError("masks must be sequences")
        if len(r[k]) != len(c.shape):
            raise ValueError("mask rank")
        for n, mask in zip(c.shape, r[k]):
            if not isinstance(mask, (list, tuple)):
                raise ValueError("mask must be a sequence")
            if len(mask) != n.bit_length() - 2:
                raise ValueError("mask length")
            if any(type(v) is not int or v not in (0, 1) for v in mask):
                raise ValueError("mask bit")
    if min(c.shape) < 4 or math.prod(c.shape) * c.batch % 32:
        raise ValueError("DMA tail")
    for axis, n in enumerate(c.shape):
        stride = math.prod(c.shape[axis + 1 :])
        lanes = min(stride, r["pack"])
        if (n if stride == 1 else lanes) % 4:
            raise ValueError("DMA alignment")
        if math.prod(c.shape) * c.batch // n % lanes:
            raise ValueError("line ownership")
        if 24 * n * lanes > 256 * 1024:
            raise ValueError("UB capacity")


def stages(c, r, mode="pipeline"):
    if mode not in ("pipeline", "forward", "inverse"):
        raise ValueError("invalid FFT mode")
    validate(c, r)
    out = []
    for inv in (False, True) if mode == "pipeline" else (mode == "inverse",):
        for axis in reversed(range(len(c.shape))):
            mask = r["inverse_masks" if inv else "forward_masks"][axis]
            ends = [i + 1 for i, v in enumerate(mask) if v] + [len(mask) + 1]
            first = 0
            for last in ends:
                tail = axis == 0 and last == len(mask) + 1
                out.append(
                    dict(
                        kind="fft",
                        axis=axis,
                        first=first,
                        last=last,
                        inverse=inv,
                        multiply=bool(mode == "pipeline" and not inv and tail and r["fuse_multiply"]),
                        scale=bool(inv and tail and r["fuse_scale"]),
                        pack=r["pack"],
                    )
                )
                first = last
        if mode == "pipeline" and not inv and not r["fuse_multiply"]:
            out.append(dict(kind="multiply"))
        if inv and not r["fuse_scale"]:
            out.append(dict(kind="scale"))
    return out


def key(r):
    return json.dumps(r, sort_keys=True, separators=(",", ":"))


def features(c, r):
    ss = stages(c, r)
    elements = math.prod(c.shape) * c.batch
    lanes = [min(math.prod(c.shape[i + 1 :]), r["pack"]) for i in range(len(c.shape))]
    groups = [elements // n // p for n, p in zip(c.shape, lanes)]
    return dict(
        launches=len(ss),
        logical_KiB=(16 * elements * len(ss) + 8 * elements) / 1024,
        UB_KiB=max(24 * n * p for n, p in zip(c.shape, lanes)) / 1024,
        line_groups=groups,
        useful_blocks=[min(r["cores"], g) for g in groups],
        cores=r["cores"],
        pack=r["pack"],
    )


def all_realizations(name):
    c, packs, paired = WORKLOADS[name]
    widths = [n.bit_length() - 2 for n in c.shape]
    bits = sum(widths)
    count = bits if paired else 2 * bits
    for value in range(1 << count):
        flat = [(value >> i) & 1 for i in range(count)]
        offset = 0
        parts = []
        for _ in range(1 if paired else 2):
            masks = []
            for w in widths:
                masks.append(flat[offset : offset + w])
                offset += w
            parts.append(masks)
        if paired:
            parts.append(copy.deepcopy(parts[0]))
        for fusion, cores, pack in itertools.product(range(4), (1, 4, 8), packs):
            yield dict(
                forward_masks=parts[0],
                inverse_masks=parts[1],
                fuse_multiply=bool(fusion & 1),
                fuse_scale=bool(fusion & 2),
                cores=cores,
                pack=pack,
            )


def apply(c, r, a):
    """Apply the model's explicit action, then validate the resulting graph."""
    if not isinstance(a, dict):
        raise ValueError("action must be an object")
    validate(c, r)
    z = copy.deepcopy(r)
    level = a.get("level")
    region = a.get("region")
    op = a.get("op")
    if level == "fft":
        if set(a) != {"level", "region", "op", "boundaries"}:
            raise ValueError("FFT action schema")
        if op not in ("merge", "split"):
            raise ValueError("FFT operation")
        regions = [(d, i) for d in ("forward", "inverse") for i in range(len(c.shape))]
        if region != "all":
            if region not in [d + ":" + str(i) for d, i in regions]:
                raise ValueError("FFT region")
            d, i = region.split(":")
            regions = [(d, int(i))]
        for d, i in regions:
            mask = z[d + "_masks"][i]
            bs = a["boundaries"]
            if bs == "all":
                bs = list(range(1, len(mask) + 1))
            if (
                not isinstance(bs, list)
                or not bs
                or any(type(v) is not int for v in bs)
                or len(set(bs)) != len(bs)
            ):
                raise ValueError("boundary list")
            for b in bs:
                if type(b) is not int or not 1 <= b <= len(mask):
                    raise ValueError("boundary index")
                mask[b - 1] = int(op == "split")
    elif level == "pipeline":
        if set(a) != {"level", "region", "op"}:
            raise ValueError("pipeline action schema")
        if region not in ("multiply", "scale", "both") or op not in ("fuse", "unfuse"):
            raise ValueError("pipeline action")
        for name in ("multiply", "scale"):
            if region in (name, "both"):
                z["fuse_" + name] = op == "fuse"
    elif level == "schedule":
        if set(a) != {"level", "region", "op", "schedule"} or region != "all" or op != "reschedule":
            raise ValueError("schedule action")
        if not isinstance(a["schedule"], dict) or set(a["schedule"]) != {"cores", "pack"}:
            raise ValueError("schedule fields")
        z.update(a["schedule"])
    else:
        raise ValueError("unknown action level")
    validate(c, z)
    if z == r:
        raise ValueError("no change")
    return z


def actions(c, r, packs=(4, 8, 16)):
    """Equivalent typed actions and flat IDs are derived from this same menu."""
    proposed = []
    for region in ["all"] + [d + ":" + str(i) for d in ("forward", "inverse") for i in range(len(c.shape))]:
        for op in ("merge", "split"):
            proposed.append(dict(level="fft", region=region, op=op, boundaries="all"))
        if region != "all":
            d, i = region.split(":")
            mask = r[d + "_masks"][int(i)]
            for b, v in enumerate(mask):
                proposed.append(
                    dict(level="fft", region=region, op="merge" if v else "split", boundaries=[b + 1])
                )
    for region in ("multiply", "scale", "both"):
        for op in ("fuse", "unfuse"):
            proposed.append(dict(level="pipeline", region=region, op=op))
    for cores, pack in itertools.product((1, 4, 8), packs):
        proposed.append(
            dict(level="schedule", region="all", op="reschedule", schedule=dict(cores=cores, pack=pack))
        )
    seen = set()
    out = []
    for a in proposed:
        try:
            z = apply(c, r, a)
        except ValueError:
            continue
        k = key(z)
        if k in seen:
            continue
        seen.add(k)
        out.append((a, z))
    return out


def random_realization(name, rng):
    c, packs, paired = WORKLOADS[name]
    r = initial(c)
    for d in ("forward", "inverse"):
        for mask in r[d + "_masks"]:
            for i in range(len(mask)):
                mask[i] = rng.randrange(2)
    if paired:
        r["inverse_masks"] = copy.deepcopy(r["forward_masks"])
    r.update(
        fuse_multiply=bool(rng.randrange(2)),
        fuse_scale=bool(rng.randrange(2)),
        cores=rng.choice((1, 4, 8)),
        pack=rng.choice(packs),
    )
    validate(c, r)
    return r
