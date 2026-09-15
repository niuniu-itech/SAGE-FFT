# Author: even
"""Derive every numerical table/plot from retained target records."""

from pathlib import Path
import csv, json, math, statistics, re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import workspace, prepare_workspace

ROOT = workspace()
P = ROOT / "report"
E = ROOT / "evidence"
F = P / "figures"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
    }
)
COLORS = ["#326FA8", "#B87630", "#398255", "#B64759"]
PURPLE = "#7353AA"


def save(fig, name):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(F / (name + "." + suffix), bbox_inches="tight", pad_inches=0.03, dpi=220)
    plt.close(fig)


def median(rows):
    return float(np.median([r["p50_us"] for r in rows]))


def sci(x):
    if not x:
        return "$0$"
    exp = int(math.floor(math.log10(x)))
    return "$%.2f\\!\\times\\!10^{%d}$" % (x / 10**exp, exp)


def writecsv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    prepare_workspace()
    (P / "figures").mkdir(parents=True, exist_ok=True)
    allrows = [json.loads(s) for s in (E / "target_records.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in allrows if r.get("pass_correctness") and r.get("build_ok")]
    groups = {}
    for r in rows:
        case_id = r["id"]
        if r.get("phase") == "ablation":
            case_id = re.sub(r"_retry[1-9][0-9]*$", "", case_id)
        groups.setdefault(case_id, []).append(r)
    summary = []
    selected = {}
    plot = []
    for wi in range(7):
        ids = ["ablation_w%d_a%d" % (wi, a) for a in range(11)]
        if not all(len(groups.get(i, [])) == 3 for i in ids):
            continue
        best = []
        for tier, subset in enumerate((ids[:1], ids[:3], ids[:7], ids)):
            chosen = min(subset, key=lambda i: median(groups[i]))
            best.append(chosen)
        selected[wi] = best
        baseline = {r["seed"]: r["p50_us"] for r in groups[best[0]]}
        vals = []
        for tier, i in enumerate(best):
            ratios = [baseline[r["seed"]] / r["p50_us"] for r in groups[i]]
            vals.append(ratios)
            plot.append(
                dict(
                    workload=wi + 1,
                    tier=tier,
                    selected_id=groups[i][0]["id"],
                    median_us=median(groups[i]),
                    speedup=median(groups[best[0]]) / median(groups[i]),
                    seed_min=min(ratios),
                    seed_max=max(ratios),
                )
            )
        rr = [r for i in ids for r in groups[i]]
        sem = [r for r in rows if r["id"].startswith("semantics_w%d_" % wi)]
        shape = rr[0]["contract"]["shape"]
        batch = rr[0]["contract"]["batch"]
        full = groups[best[-1]][0]
        summary.append(
            dict(
                workload=wi + 1,
                shape="x".join(map(str, shape)),
                batch=batch,
                records=len(rr) + len(sem),
                max_relative_l2=max(r["relative_l2"] for r in rr + sem),
                max_abs=max(r["max_abs"] for r in rr + sem),
                baseline_us=median(groups[best[0]]),
                full_us=median(groups[best[-1]]),
                speedup=median(groups[best[0]]) / median(groups[best[-1]]),
                launches_staged=rr[0]["kernel_launches"],
                launches_full=full["kernel_launches"],
                logical_bytes_staged=rr[0]["logical_gm_bytes"],
                logical_bytes_full=full["logical_gm_bytes"],
            )
        )
    writecsv(E / "workload_summary.csv", summary)
    writecsv(E / "ablation_summary.csv", plot)
    if len(summary) == 7:
        paired = {}
        paired_path = E / "cuda_npu_summary.csv"
        if paired_path.exists():
            with paired_path.open(encoding="utf-8") as f:
                paired_rows = list(csv.DictReader(f))
            paired = {int(r["workload"]): r for r in paired_rows}
            if len(paired_rows) != 7 or set(paired) != set(range(1, 8)):
                raise ValueError("CUDA pairing summary must contain all seven unique workloads")
            if any(
                not math.isfinite(float(r["max_relative_l2"])) or float(r["max_relative_l2"]) < 0
                for r in paired_rows
            ):
                raise ValueError("CUDA pairing errors must be finite and nonnegative")
        fig, ax = plt.subplots(figsize=(3.35, 1.65))
        x = np.arange(7)
        w = 0.19
        labels = ["Staged", "+ Schedule", "+ Grouping", "+ Fusion"]
        for tier in range(4):
            v = [next(p for p in plot if p["workload"] == wi + 1 and p["tier"] == tier) for wi in range(7)]
            m = np.array([r["speedup"] for r in v])
            lo = np.array([r["seed_min"] for r in v])
            hi = np.array([r["seed_max"] for r in v])
            ax.bar(
                x + (tier - 1.5) * w,
                m,
                w,
                color=COLORS[tier],
                edgecolor="white",
                linewidth=0.3,
                label=labels[tier],
                zorder=3,
            )
            ax.errorbar(
                x + (tier - 1.5) * w,
                m,
                yerr=[m - lo, hi - m],
                fmt="none",
                ecolor="#283247",
                elinewidth=0.55,
                capsize=1.4,
                zorder=4,
            )
        ax.axvspan(3.5, 6.5, color="#F0F8F1", zorder=0)
        ax.set_xticks(x, ["W%d" % (i + 1) for i in x])
        ax.set_ylabel("End-to-end speedup")
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.6, 6.6)
        ax.grid(axis="y", alpha=0.18, zorder=0)
        ax.legend(
            ncol=4,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            fontsize=6.6,
            frameon=False,
            columnspacing=0.65,
            handletextpad=0.35,
            handlelength=0.9,
        )
        save(fig, "ablation")
        (P / "ablation_figure.tex").write_text(
            r"""\begin{figure}[t]\centering
\includegraphics[width=\columnwidth]{figures/ablation.pdf}
\caption{Best observed implementation in each nested tier, normalized to the staged port. Bars are ratios of seed-median latencies; whiskers span the seed-paired ratios. W5--W7 (shaded) are 3D cases.}
\label{fig:ablation}\end{figure}
""",
            encoding="utf-8",
        )
        tab = r"""\begin{table}[t]
\caption{Workloads, maximum CUDA--NPU relative $\ell_2$ error across pipeline realizations, and best observed full-space NPU latency.}
\label{tab:correctness}\centering
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrr@{}}\toprule
Case & Shape (batch) & CUDA--NPU $\ell_2$ & $\mu$s\\\midrule
"""
        if paired:
            for r in summary:
                shape = r["shape"].replace("x", r"\!\times\!")
                tab += "W%d & $%s$ (%d) & %s & %.1f\\\\\n" % (
                    r["workload"],
                    shape,
                    r["batch"],
                    sci(float(paired[r["workload"]]["max_relative_l2"])),
                    r["full_us"],
                )
            tab += r"\bottomrule\end{tabular}\end{table}" + "\n"
            (P / "correctness_table.tex").write_text(tab, encoding="utf-8")
        fusion = []
        for wi in (4, 5, 6):
            for a, label in [(1, "Separate"), (5, "Intra-FFT"), (9, "Hierarchical")]:
                rs = groups["ablation_w%d_a%d" % (wi, a)]
                r = rs[0]
                fusion.append(
                    dict(
                        workload=wi + 1,
                        variant=label,
                        median_us=median(rs),
                        launches=r["kernel_launches"],
                        logical_gm_bytes=r["logical_gm_bytes"],
                        cores=4,
                    )
                )
        writecsv(E / "fusion_summary.csv", fusion)
        tab = r"""\begin{table}[t]\centering
\caption{Matched four-block 3D pipelines: separate stages (S), intra-FFT grouping (I), and hierarchical fusion (H). Latencies are medians over three input seeds. $K$ is the pipeline launch count.}
\label{tab:fusion}\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrl@{}}\toprule
Case & S ($\mu$s) & I ($\mu$s) & H ($\mu$s) & $K$: S/I/H\\\midrule
"""
        for wi in (5, 6, 7):
            v = [r for r in fusion if r["workload"] == wi]
            tab += "W%d & %.1f & %.1f & %.1f & %d/%d/%d\\\\\n" % (
                wi,
                *(r["median_us"] for r in v),
                *(r["launches"] for r in v),
            )
        tab += r"\bottomrule\end{tabular}\end{table}" + "\n"
        (P / "fusion_table.tex").write_text(tab, encoding="utf-8")
    policy = []
    proposals = []
    for p in sorted((ROOT / "experiments" / "artifacts").glob("policy_*.proposal.json")):
        prop = json.loads(p.read_text(encoding="utf-8"))
        proposals.append(prop)
    for name in ("fixed_heuristic", "uniform_random", "llm_no_history", "sage"):
        for seed in (41, 42, 43):
            incumbent = math.inf
            winner = None
            for b in range(1, 9):
                key = "policy_%s_s%d_b%d" % (name, seed, b)
                rr = groups.get(key, [])
                if rr and rr[0]["p50_us"] < incumbent:
                    incumbent = rr[0]["p50_us"]
                    winner = key
                if math.isfinite(incumbent):
                    policy.append(dict(policy=name, seed=seed, budget=b, best_us=incumbent, winner=winner))
    writecsv(E / "policy_best_by_budget.csv", policy)
    if len(proposals) == 96:
        fig, ax = plt.subplots(figsize=(3.35, 1.55))
        names = ["fixed_heuristic", "uniform_random", "llm_no_history", "sage"]
        labels = ["Fixed heuristic", "Uniform random", "LLM w/o history", "SAGE"]
        colors = [COLORS[0], COLORS[1], COLORS[2], PURPLE]
        polsum = []
        for name, label, color, marker in zip(names, labels, colors, ["s", "^", "D", "o"]):
            budgets = [1, 2, 4, 8]
            v = [[r["best_us"] for r in policy if r["policy"] == name and r["budget"] == b] for b in budgets]
            med = np.array([np.median(x) if x else np.nan for x in v])
            lo = np.array([min(x) if x else np.nan for x in v])
            hi = np.array([max(x) if x else np.nan for x in v])
            ax.plot(budgets, med, color=color, marker=marker, markersize=3, lw=1, label=label)
            ax.fill_between(budgets, lo, hi, color=color, alpha=0.09)
            polsum.append(
                dict(
                    policy=name,
                    median_best_b8_us=float(med[-1]),
                    min_best_b8_us=float(lo[-1]),
                    max_best_b8_us=float(hi[-1]),
                    valid_seeds_b1=len(v[0]),
                    valid_seeds_b8=len(v[-1]),
                )
            )
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xlabel("Charged proposals")
        ax.set_ylabel("Best valid latency ($\\mu$s)")
        ax.grid(alpha=0.18)
        ax.legend(
            ncol=2,
            loc="lower left",
            bbox_to_anchor=(-0.1, 1.01),
            fontsize=6.7,
            frameon=False,
            handlelength=1.2,
            columnspacing=1,
        )
        save(fig, "policy")
        writecsv(E / "policy_summary.csv", polsum)
        (P / "policy_figure.tex").write_text(
            r"""\begin{figure}[t]\centering
\includegraphics[width=\columnwidth]{figures/policy.pdf}
\caption{Same-model search on the $8\times8\times8$ pipeline. Lines show median best valid latency; shading spans three search seeds. Each accepted proposal generates and compiles fresh target source. Lower is better.}
\label{fig:policy}\end{figure}
""",
            encoding="utf-8",
        )
    audit = dict(
        valid_records=len(rows),
        failed_records=len(allrows) - len(rows),
        fresh_builds=len(groups),
        completed_ablation_workloads=len(summary),
        policy_proposals=len(proposals),
        cuda_pairing_available=(E / "cuda_npu_summary.csv").exists(),
        source="evidence/target_records.jsonl",
    )
    (E / "analysis_status.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
