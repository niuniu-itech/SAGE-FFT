# Author: even
"""Audit new measurements and separate configuration quality from timing noise."""

import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import workspace, prepare_workspace

ROOT = workspace()
P = ROOT / "report"
E = ROOT / "evidence"
from .fft_ir import Contract, import_cuda, identity, metrics, candidates
from .validate_semantics import reference
from .analyze_results import save, writecsv, COLORS, PURPLE


def load(name):
    return [json.loads(s) for s in (E / name).read_text().splitlines()]


def main():
    prepare_workspace()
    (P / "figures").mkdir(parents=True, exist_ok=True)
    rows = load("scaling_records.jsonl")
    cuda = load("scaling_cuda_records.jsonl")
    cal = load("calibration_records.jsonl")
    assert len(rows) == 27 and len(cuda) == 9 and len(cal) == 60
    pairs = []
    for r in rows:
        d = ROOT / "experiments/artifacts" / r["id"]
        n = r["contract"]["shape"][0]
        c = Contract(tuple(r["contract"]["shape"]), 1)
        assert import_cuda((d / "source.cu").read_text()) == c
        assert identity(c, r["action"])[0] == r["ir_sha256"]
        for name, key in [("kernels.cpp", "target_source_sha256"), ("sage_fft", "binary_sha256")]:
            assert hashlib.sha256((d / name).read_bytes()).hexdigest() == r[key]
        assert hashlib.sha256((d / "source.cu").read_text().encode()).hexdigest() == r["source_sha256"]
        seed = r["seed"]
        ip = d / ("input_%d" % seed)
        shape = (1,) + c.shape
        x = np.fromfile(ip / "x.bin", np.complex64).reshape(shape)
        h = np.fromfile(ip / "h.bin", np.complex64).reshape(shape)
        y = np.fromfile(d / ("out_%d.bin" % seed), np.complex64).reshape(shape)
        direct = metrics(y, reference(c, x, h))
        assert direct["pass_correctness"]
        assert abs(direct["relative_l2"] - r["relative_l2"]) < 1e-12
        assert np.median(r["samples_us"]) == r["p50_us"]
        gd = ROOT / "experiments/artifacts" / ("scaling_cuda_n%d" % n)
        for name in ("x.bin", "h.bin"):
            assert (ip / name).read_bytes() == (gd / ip.name / name).read_bytes()
        gpu = np.fromfile(gd / ("out_%d.bin" % seed), np.complex64).reshape(shape)
        assert metrics(gpu, reference(c, x, h))["pass_correctness"]
        cr = next(v for v in cuda if v["shape"] == list(c.shape) and v["seed"] == seed)
        assert hashlib.sha256((gd / "reference").read_bytes()).hexdigest() == cr["binary_sha256"]
        assert hashlib.sha256((gd / "source.cu").read_bytes()).hexdigest() == cr["source_sha256"]
        pair = metrics(y, gpu.astype(np.complex128))
        assert pair["pass_correctness"]
        pairs.append(dict(id=r["id"], n=n, seed=seed, **pair))
    writecsv(E / "scaling_cuda_pairs.csv", pairs)
    summaries = []
    for n in (16, 32, 64):
        v = {
            label: float(np.median([r["p50_us"] for r in rows if r["id"] == "scaling_n%d_%s" % (n, label)]))
            for label in ("staged", "intra", "full")
        }
        summaries.append(
            dict(
                n=n,
                **v,
                speedup=v["staged"] / v["full"],
                fusion_speedup=v["intra"] / v["full"],
                max_relative_l2=max(r["relative_l2"] for r in pairs if r["n"] == n),
            )
        )
    writecsv(E / "scaling_summary.csv", summaries)
    tab = r"""\begin{table}[t]\centering
\caption{Larger 3D pipelines at eight fixed blocks. S: separated stages; I: whole-axis grouping; H: grouping plus inter-operator fusion. Latencies are medians over three inputs.}
\label{tab:scaling}\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrr@{}}\toprule
Shape & S (ms) & I (ms) & H (ms) & S/H\\\midrule
"""
    for s in summaries:
        tab += "$%d^3$ & %.3f & %.3f & %.3f & %.2f$\\times$\\\\\n" % (
            s["n"],
            s["staged"] / 1000,
            s["intra"] / 1000,
            s["full"] / 1000,
            s["speedup"],
        )
    tab += r"\bottomrule\end{tabular}\end{table}" + "\n"
    (P / "scaling_table.tex").write_text(tab)
    base = ROOT / "experiments/calibration"
    c = Contract((8, 8, 8), 1)
    x = np.fromfile(base / "input_4/x.bin", np.complex64).reshape((1, 8, 8, 8))
    h = np.fromfile(base / "input_4/h.bin", np.complex64).reshape(x.shape)
    source_rows = {r["id"]: r for r in load("target_records.jsonl")}
    for r in cal:
        original = source_rows[r["source_measurement_id"]]
        assert r["action"] == original["action"] and r["binary_sha256"] == original["binary_sha256"]
        y = np.fromfile(base / r["output_file"], np.complex64).reshape(x.shape)
        assert metrics(y, reference(c, x, h))["pass_correctness"]
        assert np.median(r["samples_us"]) == r["p50_us"]
    props = [
        json.loads(p.read_text()) for p in (ROOT / "experiments/artifacts").glob("policy_*.proposal.json")
    ]
    oracle = load("oracle_records.jsonl")
    assert len(oracle) == 54 and {(r["block"], r["candidate_id"]) for r in oracle} == {
        (b, i) for b in range(3) for i in range(18)
    }
    assert all(r["pass_correctness"] and np.isfinite(r["p50_us"]) and r["p50_us"] > 0 for r in oracle)
    oracle_latency = {
        idx: float(np.median([r["p50_us"] for r in oracle if r["candidate_id"] == idx])) for idx in range(18)
    }
    optimum = min(oracle_latency.values())
    optimum_ids = {idx for idx, v in oracle_latency.items() if v == optimum}
    common_id = min(optimum_ids)
    common = candidates()[common_id]
    names = ["fixed_heuristic", "uniform_random", "llm_no_history", "sage"]
    labels = ["Heuristic", "Random", "LLM w/o H", "SAGE"]
    colors = [COLORS[0], COLORS[1], COLORS[2], PURPLE]
    hits = []
    final = []
    with (E / "policy_best_by_budget.csv").open() as f:
        incumbents = [v for v in csv.DictReader(f) if v["budget"] == "8"]
    for name in names:
        for seed in (41, 42, 43):
            ps = [v for v in props if v["policy"] == name and v["search_seed"] == seed]
            b = [v["budget"] for v in ps if v.get("chosen_id") in optimum_ids]
            hits.append(dict(policy=name, seed=seed, first_hit=min(b) if b else None))
            w = next(v for v in incumbents if v["policy"] == name and int(v["seed"]) == seed)
            original = source_rows[w["winner"]]
            final.append(
                dict(
                    policy=name,
                    seed=seed,
                    action=original["action"],
                    target_source_sha256=original["target_source_sha256"],
                )
            )
    nonrandom = [v for v in final if v["policy"] != "uniform_random"]
    same_configuration = len({json.dumps(v["action"], sort_keys=True) for v in nonrandom}) == 1
    same_source = len({v["target_source_sha256"] for v in nonrandom}) == 1
    (E / "policy_configuration_identity.json").write_text(
        json.dumps(
            {
                "reference_config": common,
                "optimal_candidate_ids": sorted(optimum_ids),
                "nonrandom_incumbents_share_configuration": same_configuration,
                "nonrandom_incumbents_share_source": same_source,
                "first_hits": hits,
                "final_incumbents": final,
            },
            indent=2,
        )
    )
    calibration = []
    for name in names:
        b = [
            float(np.median([r["p50_us"] for r in cal if r["policy"] == name and r["block"] == block]))
            for block in range(5)
        ]
        calibration.append(
            dict(policy=name, median_us=float(np.median(b)), min_us=min(b), max_us=max(b), block_medians=b)
        )
    (E / "calibration_summary.json").write_text(json.dumps(calibration, indent=2))
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    trajectories = []
    outcomes = []
    for name in names:
        ps = [v for v in props if v["policy"] == name]
        outcomes.append(
            dict(
                policy=name,
                evaluated=sum("measurement_id" in v for v in ps),
                rejected_model_proposals=sum(v.get("error") == "invalid_model_proposal" for v in ps),
            )
        )
        for seed in (41, 42, 43):
            visited = set()
            for b in range(1, 9):
                v = next(v for v in ps if v["search_seed"] == seed and v["budget"] == b)
                if "measurement_id" in v:
                    visited.add(v["chosen_id"])
                trajectories.append(
                    dict(
                        policy=name,
                        seed=seed,
                        budget=b,
                        best_calibrated_ratio=min(oracle_latency[i] for i in visited) / optimum
                        if visited
                        else None,
                        unique_evaluated=len(visited),
                    )
                )
    writecsv(E / "policy_calibrated_progress.csv", trajectories)
    writecsv(E / "policy_proposal_use.csv", outcomes)
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.02))
    ax, bx, cx, dx = axes.flat
    markers = ["s", "^", "D", "o"]
    styles = ["-", "--", "-.", ":"]
    for name, label, color, marker, style in zip(names, labels, colors, markers, styles):
        z = np.array(
            [
                [
                    v["best_calibrated_ratio"]
                    for v in trajectories
                    if v["policy"] == name and v["seed"] == seed
                ]
                for seed in (41, 42, 43)
            ],
            dtype=float,
        )
        med = np.median(z, axis=0)
        ax.fill_between(range(1, 9), z.min(axis=0), z.max(axis=0), step="post", color=color, alpha=0.09, lw=0)
        ax.step(range(1, 9), med, where="post", color=color, lw=1, ls=style)
    ax.axhline(1, color="#777777", lw=0.5, ls=":")
    ax.set(xlim=(0.8, 8.2), xticks=[1, 2, 4, 6, 8], xlabel="Proposal budget", ylabel="Best / optimum")
    ax.grid(alpha=0.15)
    matrix = np.array(
        [
            [
                next(v["first_hit"] for v in hits if v["policy"] == name and v["seed"] == seed) or 9
                for seed in (41, 42, 43)
            ]
            for name in names
        ]
    )
    hit_cmap = ListedColormap(["#EEEAF8", "#DBD3EF", "#C7BAE5", "#B39FDB", "#9A80CC", "#8363BD", "#ECECEC"])
    hit_norm = Normalize(2, 9)
    for j in range(4):
        for i in range(3):
            bx.add_patch(
                Rectangle(
                    (i - 0.5, j - 0.5),
                    1,
                    1,
                    facecolor=hit_cmap(hit_norm(matrix[j, i])),
                    edgecolor="white",
                    lw=0.3,
                )
            )
    bx.set(xlim=(-0.5, 2.5), ylim=(3.5, -0.5))
    for j in range(4):
        for i in range(3):
            bx.text(
                i,
                j,
                str(matrix[j, i]) if matrix[j, i] < 9 else ">8",
                ha="center",
                va="center",
                fontsize=7,
                color="#182237",
            )
    bx.set(
        xticks=[0, 1, 2],
        xticklabels=["41", "42", "43"],
        yticks=range(4),
        yticklabels=["H", "R", "N", "S"],
        xlabel="Search seed",
    )
    bx.tick_params(length=0)
    for i, (v, color) in enumerate(zip(outcomes, colors)):
        cx.barh(i, v["evaluated"], color=color, height=0.62)
        if v["rejected_model_proposals"]:
            cx.barh(
                i,
                v["rejected_model_proposals"],
                left=v["evaluated"],
                color="#F0F0F0",
                edgecolor="#888888",
                hatch="///",
                height=0.62,
                lw=0.4,
            )
        cx.text(
            v["evaluated"] / 2, i, str(v["evaluated"]), ha="center", va="center", fontsize=6.4, color="white"
        )
        if v["rejected_model_proposals"]:
            cx.text(
                v["evaluated"] + v["rejected_model_proposals"] / 2,
                i,
                str(v["rejected_model_proposals"]),
                ha="center",
                va="center",
                fontsize=6.4,
            )
    cx.set(
        xlim=(0, 24.5),
        xticks=[0, 8, 16, 24],
        yticks=range(4),
        yticklabels=["H", "R", "N", "S"],
        xlabel="Proposals",
    )
    cx.invert_yaxis()
    cx.grid(axis="x", alpha=0.12)
    cx.set_axisbelow(True)
    seed_summaries = []
    for i, (name, color) in enumerate(zip(names, colors)):
        for seed, offset in zip((41, 42, 43), (-0.24, 0, 0.24)):
            runs = sorted(
                [r for r in cal if r["policy"] == name and r["search_seed"] == seed], key=lambda r: r["block"]
            )
            vals = np.array([r["p50_us"] for r in runs])
            assert len(vals) == 5
            action = runs[0]["action"]
            assert all(r["action"] == action for r in runs)
            center = i + offset
            med = float(np.median(vals))
            dx.vlines(center, vals.min(), vals.max(), color=color, lw=0.55, alpha=0.6, zorder=2)
            dx.scatter(
                center + np.linspace(-0.055, 0.055, 5), vals, s=5, c=color, alpha=0.7, linewidths=0, zorder=3
            )
            dx.hlines(med, center - 0.095, center + 0.095, color=color, lw=1.3, zorder=4)
            seed_summaries.append(
                dict(
                    policy=name,
                    search_seed=seed,
                    group=action["group"],
                    cores=action["cores"],
                    fusion=action["fusion"],
                    n=5,
                    median_us=med,
                    min_us=float(vals.min()),
                    max_us=float(vals.max()),
                )
            )
    writecsv(E / "policy_incumbent_remeasurements.csv", seed_summaries)
    dx.set(
        xlim=(-0.55, 3.55),
        xticks=range(4),
        xticklabels=["H", "R", "N", "S"],
        ylabel="Latency ($\\mu$s)",
        xlabel="Policy",
    )
    dx.grid(axis="y", alpha=0.15)
    panel_labels = [
        "(a) Candidate quality",
        "(b) First optimum hit",
        "(c) Evaluated / rejected",
        "(d) Final-incumbent latency",
    ]
    for a, label in zip(axes.flat, panel_labels):
        a.tick_params(labelsize=6.3)
        a.xaxis.label.set_size(6.6)
        a.yaxis.label.set_size(6.6)
        a.xaxis.labelpad = 2
        a.yaxis.labelpad = 2
        a.text(0.5, -0.49, label, transform=a.transAxes, ha="center", va="top", fontsize=7)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.92, wspace=0.5, hspace=1.08)
    handles = [Line2D([0], [0], color=c, lw=1.1, ls=ls) for c, ls in zip(colors, styles)]
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        fontsize=6.3,
        frameon=False,
        columnspacing=0.65,
        handlelength=1.0,
    )
    save(fig, "policy")
    (P / "policy_figure.tex").write_text(
        "\\begin{figure}[t]\\centering\n\\includegraphics[width=\\columnwidth]{figures/policy.pdf}\n\\caption{Policy diagnostics on $8^3$. (a) Best visited latency relative to the independently calibrated 18-configuration minimum; lines show seed medians and bands seed ranges. Gaps denote no accepted candidate. (b) First proposal reaching a minimum. (c) Evaluated proposals and rejected model proposals. (d) Five remeasurements per incumbent; bars mark medians. Within each policy, seeds 41/42/43 run left to right. H/R/N/S follow legend order.}\n\\label{fig:policy}\\end{figure}\n",
        encoding="utf-8",
    )
    allsame = [r["p50_us"] for r in cal if r["policy"] != "uniform_random"]
    summary = dict(
        scale_outputs=27,
        cuda_runs=9,
        calibration_outputs=60,
        all_correct=True,
        scale_speedup_range=[min(v["speedup"] for v in summaries), max(v["speedup"] for v in summaries)],
        fusion_speedup_range=[
            min(v["fusion_speedup"] for v in summaries),
            max(v["fusion_speedup"] for v in summaries),
        ],
        max_scale_pair_relative_l2=max(v["relative_l2"] for v in pairs),
        same_configuration_source_hash=same_configuration and same_source,
        reference_config=common,
        optimal_candidate_ids=sorted(optimum_ids),
        first_hits=hits,
        calibration=calibration,
        nonrandom_incumbent_run_range_us=[min(allsame), max(allsame)],
    )
    (E / "quality_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
