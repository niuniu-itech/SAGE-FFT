# Author: even
"""Descriptive coordinate replay on measured M2; not an online search trial."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import json, itertools, collections
import numpy as np
from .catalog import ROOT
from .mask_ir import key, initial, WORKLOADS


def main():
    prepare_workspace()
    data = collections.defaultdict(list)
    realizations = {}
    for block in (1, 2, 3):
        for row in json.loads(
            (ROOT / "experiments/v46/M2" / ("oracle_b%d" % block) / "results.json").read_text()
        ):
            k = key(row["realization"])
            data[k].append(row["p50_us"])
            realizations[k] = row["realization"]
    latency = {k: float(np.median(v)) for k, v in data.items()}
    assert len(latency) == 6144
    best = min(latency, key=latency.get)
    fields = {
        "pipeline": ("fuse_multiply", "fuse_scale"),
        "boundaries": ("forward_masks", "inverse_masks"),
        "schedule": ("cores", "pack"),
    }
    results = []
    for order in itertools.permutations(fields):
        current = initial(WORKLOADS["M2"][0])
        queried = {key(current)}
        trace = []
        for level in order:
            fixed = set(current) - set(fields[level])
            candidates = [k for k, r in realizations.items() if all(r[f] == current[f] for f in fixed)]
            queried.update(candidates)
            winner = min(candidates, key=lambda k: (latency[k], k))
            current = realizations[winner]
            trace.append(
                dict(level=level, candidates=len(candidates), latency_us=latency[winner], realization=current)
            )
        results.append(
            dict(
                order=order,
                unique_oracle_queries=len(queried),
                ratio=latency[key(current)] / latency[best],
                trace=trace,
            )
        )
    out = dict(
        scope="Post-measurement coordinate replay; each level can query its entire coordinate subspace. Not equal-budget online policy evidence.",
        global_median_us=latency[best],
        global_realization=realizations[best],
        orders=results,
    )
    (ROOT / "evidence/v46_coordinate_replay.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({",".join(r["order"]): round(r["ratio"], 4) for r in results}, indent=2))


if __name__ == "__main__":
    main()
