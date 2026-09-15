# Author: even
"""Independent common calibration after the hosted-model search finishes."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import json, collections
from pathlib import Path
import numpy as np
from . import catalog
from .hosted_study import ROOT, NEW, REMOTE
from .catalog import CatalogRunner
from .mask_ir import key
from .fft_ir import metrics
from .analyze_v46 import distribution


def main(expected_records=252, result_root=None, model=None, summary_path=None, catalog_remote=None):
    prepare_workspace()
    NEW = Path(result_root) if result_root is not None else globals()["NEW"]
    records = [json.loads(p.read_text()) for p in (NEW / "M3/search").glob("*/*/step_[0-9][0-9].json")]
    assert len(records) == expected_records, (len(records), expected_records)
    candidates = {
        key(r): r for r in json.loads((ROOT / "experiments/v46/M3/heldout_candidates.json").read_text())
    }
    for r in records:
        if r["accepted"]:
            candidates[key(r["realization"])] = r["realization"]
    zs = list(candidates.values())
    (NEW / "M3/heldout_candidates.json").write_text(json.dumps(zs, indent=2))
    catalog.REMOTE = catalog_remote if catalog_remote is not None else REMOTE
    target = CatalogRunner()
    target.root = NEW
    try:
        for b in range(1, 6):
            target.evaluate_many(
                "M3", zs, "heldout_b%d" % b, seed=2 + b % 2, samples=10, shuffle_seed=46100 + b
            )
    finally:
        target.s.close()
        target.c.close()
    values = collections.defaultdict(list)
    for b in range(1, 6):
        for r in json.loads((NEW / "M3" / ("heldout_b%d" % b) / "results.json").read_text()):
            values[key(r["realization"])].append(r["p50_us"])
    calibration = {k: float(np.median(v)) for k, v in values.items()}
    ref = min(calibration.values())
    init = json.loads((NEW / "M3/search_initial/results.json").read_text())[0]
    policies = {}
    trajectories = {}
    for folder in sorted((NEW / "M3/search").iterdir()):
        lat = []
        ratios = []
        cost = []
        valid = []
        hits1 = []
        hits5 = []
        improvements = []
        traces = []
        usage = []
        for seed in (41, 42, 43):
            best = init["realization"]
            online = init["p50_us"]
            trace = []
            seconds = 0
            accepted = 0
            improved = 0
            for step in range(1, 13):
                p = folder / str(seed) / ("step_%02d.json" % step)
                r = json.loads(p.read_text())
                seconds += r.get("model_seconds", 0)
                if r["accepted"]:
                    accepted += 1
                    if r["feedback"]["latency_us"] < online:
                        best = r["realization"]
                        online = r["feedback"]["latency_us"]
                        improved += 1
                trace.append(calibration[key(best)] / ref)
                prop = p.with_name("step_%02d.proposal.json" % step)
                if prop.exists():
                    raw = json.loads(prop.read_text())
                    if raw.get("response"):
                        usage.append(json.loads(raw["response"]).get("usage", {}))
            lat.append(calibration[key(best)])
            ratios.append(trace[-1])
            cost.append(seconds)
            valid.append(accepted)
            improvements.append(improved)
            traces.append(trace)
            hits1.append(next((i + 1 for i, v in enumerate(trace) if v <= 1.01), None))
            hits5.append(next((i + 1 for i, v in enumerate(trace) if v <= 1.05), None))
        policies[folder.name] = dict(
            latency_us=distribution(lat),
            ratio=distribution(ratios),
            model_seconds=distribution(cost),
            accepted=valid,
            incumbent_improvements=improvements,
            first_1pct=hits1,
            first_5pct=hits5,
            auc_regret=distribution([np.mean(np.array(t) - 1) for t in traces]),
            tokens={
                k: sum(u.get(k, 0) for u in usage)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
        )
        trajectories[folder.name] = traces
    total = bad = paired_bad = 0
    for p in (NEW / "M3").glob("*/results.json"):
        rows = json.loads(p.read_text())
        total += len(rows)
        bad += sum(not r["pass_correctness"] for r in rows)
        reference = np.fromfile(
            ROOT / "experiments/v46/cuda/M3" / ("out_%d.bin" % rows[0]["input_seed"]), np.complex64
        )
        outputs = np.fromfile(p.parent / "outputs.bin", np.complex64).reshape(len(rows), len(reference))
        order = sorted(rows, key=lambda r: r.get("execution_order", r["index"]))
        for row, y in zip(order, outputs):
            paired_bad += not metrics(y, reference.astype(np.complex128))["pass_correctness"]
    summary = dict(
        workload="M3",
        model=model or json.loads((ROOT / "evidence/v46_hosted_preregistration.json").read_text())["model"],
        reference_us=ref,
        reference_kind="best observed among union of hosted candidates and initial M3 calibrated candidates",
        calibrated_candidates=len(zs),
        policies=policies,
        trajectories=trajectories,
        correctness=dict(
            target_outputs=total, failed=bad, paired_cuda_outputs=total, paired_cuda_failed=paired_bad
        ),
    )
    destination = (
        Path(summary_path) if summary_path is not None else ROOT / "evidence/v46_hosted_summary.json"
    )
    destination.write_text(json.dumps(summary, indent=2))
    assert not bad and not paired_bad
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
