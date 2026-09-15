# Author: even
"""Run final search conditions and independent calibration on the configured NPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

from sage_fft.config import prepare_workspace
from sage_fft.catalog import CatalogRunner
from sage_fft.hosted_schema import SchemaController
from sage_fft.hosted_parallel import SerializedTarget
from sage_fft.live_target import LiveRunner
from sage_fft.mask_ir import key
from sage_fft.structured_search import LLM, POLICIES, run_search


def run(kind, seeds=(41, 42, 43)):
    root = prepare_workspace()
    target = SerializedTarget() if kind == "hosted" else LiveRunner()
    if kind == "hosted":
        target.root = root / "experiments/hosted_v489"
        controller = SchemaController()
        jobs = [
            ("M3", policy, label, seed)
            for policy, label in (("flat_llm", "schema_flat"), ("sage", "schema_typed"), ("greedy", "greedy"))
            for seed in seeds
        ]
    else:
        controller = LLM()
        jobs = [
            (name, policy, policy, seed) for name in ("M2", "M3") for policy in POLICIES for seed in seeds
        ]
    random.Random(46104).shuffle(jobs)
    selected = {}
    try:

        def execute(job):
            name, policy, label, seed = job
            realization = run_search(
                target, controller, name, policy, seed, label=label, output_root=target.root
            )
            return (name, label, seed), realization

        if kind == "hosted":
            with ThreadPoolExecutor(max_workers=3) as pool:
                for identity, realization in pool.map(execute, jobs):
                    selected[identity] = realization
        else:
            for job in jobs:
                identity, realization = execute(job)
                selected[identity] = realization
        summaries = {}
        for name in sorted({j[0] for j in jobs}):
            unique = {key(r): r for (n, _, _), r in selected.items() if n == name}
            ordered = [unique[k] for k in sorted(unique)]
            rows = []
            for block in range(1, 6):
                rows.append(
                    CatalogRunner.evaluate_many(
                        target,
                        name,
                        ordered,
                        f"acceptance_heldout_b{block}",
                        seed=2 + (block % 2),
                        samples=10,
                        shuffle_seed=6100 + block,
                    )
                )
            summaries[name] = {
                "unique_realizations": len(ordered),
                "timing_rounds": 5,
                "samples_per_round": 10,
                "measurements": rows,
            }
        result = {
            "completed": True,
            "kind": kind,
            "search_runs": len(jobs),
            "proposals_per_run": 12,
            "seeds": list(seeds),
            "calibration": summaries,
        }
        target.root.mkdir(parents=True, exist_ok=True)
        (target.root / "acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        target.close_live()
        target.s.close()
        target.c.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("hosted", "local"), required=True)
    parser.add_argument("--seeds", default="41,42,43")
    args = parser.parse_args()
    result = run(args.kind, tuple(map(int, args.seeds.split(","))))
    print(json.dumps({k: result[k] for k in ("completed", "kind", "search_runs")}))
