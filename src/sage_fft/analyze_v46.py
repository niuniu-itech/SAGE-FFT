# Author: even
"""Build a claim ledger from raw, independently calibrated target records."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import json, collections, hashlib
from pathlib import Path
import numpy as np
from .transfer_study import WORKLOADS
from .mask_ir import key, initial
from .fft_ir import metrics
from .catalog import ROOT


def distribution(xs):
    xs = list(map(float, xs))
    return dict(n=len(xs), median=float(np.median(xs)), min=min(xs), max=max(xs), values=xs)


def main():
    prepare_workspace()
    root = ROOT / "experiments" / "v46"
    summary = dict(
        workloads={},
        scope="M1/M2 complete controls; M3/M4 best observed among visited candidates and prespecified anchors",
    )
    for name in ("M2", "M3", "M4"):
        batches = [
            json.loads((root / name / ("heldout_b%d" % b) / "results.json").read_text()) for b in range(1, 6)
        ]
        points = collections.defaultdict(list)
        for batch in batches:
            for row in batch:
                points[key(row["realization"])].append(row["p50_us"])
        calibration = {k: float(np.median(v)) for k, v in points.items()}
        reference = min(calibration.values())
        init = json.loads((root / name / "search_initial" / "results.json").read_text())[0]
        results = {}
        trajectories = {}
        for policy_dir in sorted((root / name / "search").iterdir()):
            if not policy_dir.is_dir():
                continue
            policy = policy_dir.name
            traces = []
            lat = []
            ratio = []
            costs = []
            valid = []
            input_tokens = []
            output_tokens = []
            first_hits = []
            for seed in (41, 42, 43):
                rows = [
                    json.loads((policy_dir / str(seed) / ("step_%02d.json" % i)).read_text())
                    for i in range(1, 13)
                ]
                incumbent = init["realization"]
                online = init["p50_us"]
                trace = []
                cost = 0
                accepted = 0
                for i, row in enumerate(rows, 1):
                    cost += row.get("model_seconds", 0)
                    if row["accepted"]:
                        accepted += 1
                        if row["feedback"]["latency_us"] < online:
                            online = row["feedback"]["latency_us"]
                            incumbent = row["realization"]
                    trace.append(calibration[key(incumbent)] / reference)
                    usage = row.get("usage", {})
                    if usage.get("prompt_tokens") is not None:
                        input_tokens.append(usage["prompt_tokens"])
                    if usage.get("completion_tokens") is not None:
                        output_tokens.append(usage["completion_tokens"])
                traces.append(trace)
                lat.append(calibration[key(incumbent)])
                ratio.append(trace[-1])
                costs.append(cost)
                valid.append(accepted)
                first_hits.append(next((i + 1 for i, v in enumerate(trace) if v <= 1.02), None))
            results[policy] = dict(
                latency_us=distribution(lat),
                ratio=distribution(ratio),
                model_seconds=distribution(costs),
                accepted=valid,
                near_optimal_first_hits=first_hits,
            )
            if input_tokens:
                results[policy].update(
                    input_tokens=distribution(input_tokens), output_tokens=distribution(output_tokens)
                )
            trajectories[policy] = traces
        summary["workloads"][name] = dict(
            shape=WORKLOADS[name][0].shape,
            reference_us=reference,
            reference_kind="best independently remeasured candidate; exhaustive contenders included"
            if name == "M2"
            else "best observed; not a global optimum",
            calibrated_candidates=len(calibration),
            policies=results,
            trajectories=trajectories,
        )
    total = 0
    bad = 0
    paired = 0
    paired_bad = 0
    max_error = 0
    cuda = root / "cuda"
    for name in WORKLOADS:
        d = root / name
        if not d.exists():
            continue
        for p in d.glob("*/results.json"):
            if p.parent.name.startswith("session_"):
                continue
            rows = json.loads(p.read_text())
            total += len(rows)
            bad += sum(not r["pass_correctness"] for r in rows)
            if rows:
                max_error = max(max_error, max(r["max_abs"] for r in rows))
            if not rows or rows[0].get("mode") != "pipeline":
                continue
            reference_path = cuda / name / ("out_%d.bin" % rows[0]["input_seed"])
            if not reference_path.exists():
                continue
            ref = np.fromfile(reference_path, np.complex64)
            values = np.fromfile(p.parent / "outputs.bin", np.complex64).reshape(len(rows), len(ref))
            # Batch binary output follows execution_order, while JSON rows follow index.
            order = sorted(rows, key=lambda r: r.get("execution_order", r["index"]))
            for row, y in zip(order, values):
                m = metrics(y, ref.astype(np.complex128))
                paired += 1
                paired_bad += not m["pass_correctness"]
    summary["correctness"] = dict(
        target_outputs=total,
        failed=bad,
        paired_cuda_outputs=paired,
        paired_cuda_failed=paired_bad,
        max_abs=max_error,
        excludes="records outside the configured indexed-catalog study",
    )
    diag = json.loads((root / "action_diagnostics" / "scored.json").read_text())
    summary["action_diagnostic"] = {}
    for policy in ("flat_llm", "sage"):
        rows = [r for r in diag if r["policy"] == policy]
        summary["action_diagnostic"][policy] = dict(
            total=len(rows),
            valid=sum(r["valid"] for r in rows),
            near_best=sum(r.get("near_best", False) for r in rows),
            ratio=distribution([r["ratio"] for r in rows if r["valid"]]),
        )
    (ROOT / "evidence" / "v46_results_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            dict(
                correctness=summary["correctness"],
                policies={
                    name: {
                        p: dict(
                            ratio=round(x["ratio"]["median"], 3),
                            latency_us=round(x["latency_us"]["median"], 3),
                            model_s=round(x["model_seconds"]["median"], 2),
                        )
                        for p, x in info["policies"].items()
                    }
                    for name, info in summary["workloads"].items()
                },
            ),
            indent=2,
        )
    )
    assert not bad and not paired_bad


if __name__ == "__main__":
    main()
