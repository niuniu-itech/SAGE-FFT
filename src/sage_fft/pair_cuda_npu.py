# Author: even
"""Pair retained NPU pipeline outputs with actually executed cuFFT outputs."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import csv, json
from pathlib import Path
import numpy as np
from .fft_ir import metrics
from .validate_semantics import WORKLOADS

ROOT = workspace()


def main():
    prepare_workspace()
    cuda = [
        json.loads(s) for s in (ROOT / "evidence/cuda_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(cuda) == 21 and all(r["pass_correctness"] for r in cuda)
    lookup = {(tuple(shape), batch): wi for wi, (shape, batch) in enumerate(WORKLOADS)}
    rows = [
        json.loads(s)
        for s in (ROOT / "evidence/target_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    pairs = []
    for r in rows:
        if not r.get("pass_correctness") or r.get("mode") != "pipeline":
            continue
        c = r["contract"]
        shape = (c["batch"],) + tuple(c["shape"])
        wi = lookup[(tuple(c["shape"]), c["batch"])]
        seed = r["seed"]
        nd = ROOT / "experiments/artifacts" / r["id"]
        gd = ROOT / "experiments/cuda_artifacts" / ("w%d" % wi)
        for name in ("x.bin", "h.bin"):
            assert (nd / ("input_%d" % seed) / name).read_bytes() == (
                gd / ("input_%d" % seed) / name
            ).read_bytes()
        npu = np.fromfile(nd / ("out_%d.bin" % seed), np.complex64).reshape(shape)
        gpu = np.fromfile(gd / ("out_%d.bin" % seed), np.complex64).reshape(shape)
        m = metrics(npu, gpu.astype(np.complex128))
        assert m["pass_correctness"], r["id"]
        pairs.append(
            dict(
                npu_id=r["id"],
                phase=r["phase"],
                workload=wi + 1,
                shape=c["shape"],
                batch=c["batch"],
                seed=seed,
                **m,
            )
        )
    (ROOT / "evidence/cuda_npu_pairs.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    summary = []
    for wi in range(1, 8):
        rs = [r for r in pairs if r["workload"] == wi and r["phase"] == "ablation"]
        if not rs:
            continue
        g = [r for r in cuda if tuple(r["shape"]) == WORKLOADS[wi - 1][0]]
        summary.append(
            dict(
                workload=wi,
                comparisons=len(rs),
                max_relative_l2=max(r["relative_l2"] for r in rs),
                max_abs=max(r["max_abs"] for r in rs),
                cufft_median_us=float(np.median([np.median(r["samples_us"]) for r in g])),
            )
        )
    with (ROOT / "evidence/cuda_npu_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(
        json.dumps(
            dict(
                cuda_executions=len(cuda),
                npu_pairs=len(pairs),
                pass_all=True,
                max_relative_l2=max(r["relative_l2"] for r in pairs),
                max_abs=max(r["max_abs"] for r in pairs),
            )
        )
    )


if __name__ == "__main__":
    main()
