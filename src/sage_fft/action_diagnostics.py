# Author: even
"""Blind next-action comparison on prespecified states; held-out NPU scoring."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import copy
import json
from pathlib import Path
import random
from .catalog import CatalogRunner, ROOT
from .mask_ir import WORKLOADS, initial, key, actions
from .structured_search import LLM, make_request, parse_response, feedback


def states():
    c = WORKLOADS["M2"][0]
    base = initial(c)
    out = []
    r = copy.deepcopy(base)
    r.update(cores=8, pack=4)
    out.append(r)
    r = copy.deepcopy(base)
    r["forward_masks"] = r["inverse_masks"] = [[0, 0], [0, 0]]
    out.append(r)
    r = copy.deepcopy(base)
    r.update(forward_masks=[[1, 0], [0, 1]], inverse_masks=[[0, 1], [1, 0]], cores=4, fuse_multiply=True)
    out.append(r)
    r = copy.deepcopy(base)
    r.update(forward_masks=[[0, 0], [0, 0]], cores=8, fuse_scale=True)
    out.append(r)
    r = copy.deepcopy(out[1])
    r.update(fuse_multiply=True, fuse_scale=True)
    out.append(r)
    r = copy.deepcopy(base)
    r.update(forward_masks=[[0, 1], [0, 1]], inverse_masks=[[0, 1], [0, 1]], cores=4, pack=4)
    out.append(r)
    return out


def main():
    prepare_workspace()
    d = ROOT / "experiments" / "v46" / "action_diagnostics"
    d.mkdir(exist_ok=True)
    c, packs, _ = WORKLOADS["M2"]
    ss = states()
    (d / "prespecified_states.json").write_text(json.dumps(ss, indent=2))
    oracle = json.loads((ROOT / "experiments" / "v46" / "M2" / "oracle_b1" / "results.json").read_text())
    lookup = {key(r["realization"]): r for r in oracle}
    init = initial(c)
    llm = LLM()
    records = []
    neighbors = {}
    for i, r in enumerate(ss):
        history = [feedback(c, lookup[key(init)]), feedback(c, lookup[key(r)])]
        seen = {key(init), key(r)}
        neighbors[i] = [z for a, z in actions(c, r, packs) if key(z) not in seen]
        for seed in (41, 42, 43):
            for policy in ("flat_llm", "sage"):
                p = d / ("state%d_%s_s%d.json" % (i, policy, seed))
                if p.exists():
                    records.append(json.loads(p.read_text()))
                    continue
                request, menu = make_request("M2", r, history, policy, seed, 1, seen)
                raw = llm.call(request)
                row = dict(state=i, policy=policy, seed=seed, raw=raw, valid=False)
                try:
                    action, z, usage, timings = parse_response(raw, policy, menu, c, r)
                    row.update(valid=True, action=action, realization=z, usage=usage, timings=timings)
                except Exception as ex:
                    row["error"] = str(ex)
                p.write_text(json.dumps(row, indent=2))
                records.append(row)
                print(
                    json.dumps(
                        {k: row.get(k) for k in ("state", "policy", "seed", "valid", "action", "error")}
                    ),
                    flush=True,
                )
    if llm.s:
        llm.s.close()
    target = CatalogRunner()
    try:
        for i, zs in neighbors.items():
            measured = target.evaluate_many(
                "M2", zs, "action_neighbors_s%d" % i, seed=2, samples=10, shuffle_seed=1200 + i
            )
            scores = {key(z["realization"]): z["p50_us"] for z in measured}
            best = min(scores.values())
            for row in records:
                if row["state"] == i and row["valid"]:
                    latency = scores[key(row["realization"])]
                    row.update(
                        heldout_latency_us=latency,
                        neighbor_best_us=best,
                        ratio=latency / best,
                        near_best=latency <= 1.02 * best,
                        rank=1 + sum(v < latency for v in scores.values()),
                        neighbors=len(scores),
                    )
        (d / "scored.json").write_text(json.dumps(records, indent=2))
    finally:
        target.s.close()
        target.c.close()


if __name__ == "__main__":
    main()
