# Author: even
"""Independent timing blocks and static-binary checks after all policy runs."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import copy, json, random, hashlib, os
from pathlib import Path
import numpy as np
from .transfer_study import WORKLOADS
from .mask_ir import initial, key, all_realizations, random_realization
from .catalog import CatalogRunner, ROOT
from .fft_ir import metrics
from .campaign import Runner


def main():
    prepare_workspace()
    root = ROOT / "experiments" / "v46"
    target = CatalogRunner()
    rng = random.Random(4611)
    try:
        # Standalone transforms test semantics that a weighted round trip can hide.
        c = WORKLOADS["M4"][0]
        smoke = [initial(c)] + [random_realization("M4", rng) for _ in range(8)]
        for mode in ("pipeline", "forward", "inverse"):
            target.evaluate_many("M4", smoke, "final_smoke_" + mode, seed=1, samples=5, mode=mode)
        # Repeating the complete small controls avoids selecting a unique optimum
        # from a single noisy screen. None of these outcomes reach the policies.
        for name in ("M1", "M2"):
            for block in (2, 3):
                target.evaluate_many(
                    name,
                    list(all_realizations(name)),
                    "oracle_b%d" % block,
                    seed=block,
                    samples=5,
                    shuffle_seed=9100 + block,
                )
        for name in ("M2", "M3", "M4"):
            source = root / name / "search"
            rows = [json.loads(p.read_text()) for p in source.glob("*/*/step_[0-9][0-9].json")]
            expected = 252 if name in ("M2", "M3") else 324
            assert len(rows) == expected, (name, len(rows), expected)
            zs = {key(initial(WORKLOADS[name][0])): initial(WORKLOADS[name][0])}
            for r in rows:
                if r.get("accepted"):
                    zs[key(r["realization"])] = r["realization"]
            c, packs, _ = WORKLOADS[name]
            # Explicit schedule/fusion anchors augment the best-observed reference.
            for bit in (0, 1):
                for cores in (1, 4, 8):
                    for pack in packs:
                        for fusion in range(4):
                            r = initial(c)
                            r.update(
                                cores=cores,
                                pack=pack,
                                fuse_multiply=bool(fusion & 1),
                                fuse_scale=bool(fusion & 2),
                            )
                            r["forward_masks"] = [[bit] * len(m) for m in r["forward_masks"]]
                            r["inverse_masks"] = copy.deepcopy(r["forward_masks"])
                            zs[key(r)] = r
            if name == "M2":
                full = {}
                for block in (1, 2, 3):
                    for r in json.loads((root / name / ("oracle_b%d" % block) / "results.json").read_text()):
                        full.setdefault(key(r["realization"]), []).append(r)
                med = {k: float(np.median([r["p50_us"] for r in v])) for k, v in full.items()}
                best = min(med.values())
                for k, v in full.items():
                    if med[k] <= best * 1.02:
                        zs[k] = v[0]["realization"]
            candidates = list(zs.values())
            (root / name / "heldout_candidates.json").write_text(json.dumps(candidates, indent=2))
            for block in range(1, 6):
                target.evaluate_many(
                    name,
                    candidates,
                    "heldout_b%d" % block,
                    seed=2 + (block % 2),
                    samples=10,
                    shuffle_seed=5000 + block,
                )
        # A separate static executable checks each catalog's composition semantics.
        from . import campaign

        campaign.REMOTE = remote_root("static-controls")
        static = Runner()
        record_dir = root / "static_checks"
        record_dir.mkdir(exist_ok=True)

        def save(row):
            p = record_dir / (row["id"] + "_s" + str(row.get("seed", 0)) + ".json")
            p.write_text(json.dumps(row, indent=2))
            print(
                json.dumps(dict(event="static_check", id=row["id"], correct=row.get("pass_correctness"))),
                flush=True,
            )

        static.record = save
        try:
            comparisons = []
            for name in ("M1", "M2", "M3", "M4"):
                c = WORKLOADS[name][0]
                r = random_realization(name, random.Random(46000 + int(name[-1])))
                label = "v46_static_" + name
                record = record_dir / (label + "_s2.json")
                if record.exists():
                    result = json.loads(record.read_text())
                else:
                    result = static.evaluate(c, r, label, seeds=(2,), phase="catalog_static_crosscheck")
                cat = target.evaluate_many(name, [r], "static_crosscheck", seed=2, samples=10)[0]
                static_y = np.fromfile(static.root / label / "out_2.bin", np.complex64)
                catalog_y = np.fromfile(root / name / "static_crosscheck" / "outputs.bin", np.complex64)
                comparisons.append(
                    dict(
                        workload=name,
                        static_pass=result["pass_correctness"],
                        catalog_pass=cat["pass_correctness"],
                        bitwise_equal=bool(np.array_equal(static_y, catalog_y)),
                        max_difference=float(np.max(np.abs(static_y - catalog_y))),
                        static_binary_sha256=result["binary_sha256"],
                        catalog_binary_sha256=cat["catalog_binary_sha256"],
                    )
                )
            (ROOT / "evidence" / "v46_catalog_static_checks.json").write_text(
                json.dumps(comparisons, indent=2)
            )
            assert all(r["bitwise_equal"] for r in comparisons)
        finally:
            static.s.close()
            static.c.close()
    finally:
        target.s.close()
        target.c.close()


if __name__ == "__main__":
    main()
