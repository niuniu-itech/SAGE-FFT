# Author: even
"""Reasoned typed control and measured cross-workload memory on M4."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import copy, json, random, hashlib
from pathlib import Path
from .fft_ir import Contract
from .mask_ir import WORKLOADS, initial, key, features
from .catalog import ROOT
from .live_target import LiveRunner
from .structured_search import LLM, run_search

WORKLOADS["M4"] = (Contract((8, 16), 1), (4, 8, 16), False)

CONDITIONS = {
    "direct": ("sage", False, None),
    "reasoned": ("sage", True, None),
    "flat_reasoned": ("flat_llm", True, None),
    "related": ("sage", True, "related"),
    "unrelated": ("sage", True, "unrelated"),
    "flat_related": ("flat_llm", True, "related"),
    "greedy": ("greedy", False, None),
    "random": ("random", False, None),
    "evolution": ("evolution", False, None),
}


def memories():
    root = ROOT / "experiments" / "v46"
    sources = {}
    paths = sorted((root / "M2" / "search" / "greedy" / "41").glob("step_[0-9][0-9].json"))
    rows = [json.loads(p.read_text())["feedback"] for p in paths]
    assert len(rows) == 12
    baseline = json.loads((root / "M2" / "search_initial" / "results.json").read_text())[0]["p50_us"]
    sources["related"] = dict(source="M2", acquisition_measurements=12, baseline_us=baseline, records=rows)
    c = WORKLOADS["M1"][0]
    oracle = json.loads((root / "M1" / "oracle_b1" / "results.json").read_text())
    lookup = {key(r["realization"]): r for r in oracle}
    zs = []
    for bit in (0, 1):
        for cores in (1, 4, 8):
            for fused in (False, True):
                r = initial(c)
                r.update(
                    forward_masks=[[bit] * 5],
                    inverse_masks=[[bit] * 5],
                    cores=cores,
                    fuse_multiply=fused,
                    fuse_scale=fused,
                )
                zs.append(r)
    rows = [dict(realization=r, resources=features(c, r), latency_us=lookup[key(r)]["p50_us"]) for r in zs]
    sources["unrelated"] = dict(
        source="M1", acquisition_measurements=12, baseline_us=lookup[key(initial(c))]["p50_us"], records=rows
    )
    out = {}
    for label, src in sources.items():
        c = WORKLOADS[src["source"]][0]
        best = sorted(src["records"], key=lambda r: r["latency_us"])[:4]
        out[label] = [
            dict(
                shape=c.shape,
                batch=c.batch,
                realization=r["realization"],
                resources=r["resources"],
                latency_us=r["latency_us"],
                source_speedup=src["baseline_us"] / r["latency_us"],
            )
            for r in best
        ]
    (ROOT / "evidence" / "v46_transfer_sources.json").write_text(
        json.dumps(dict(sources=sources, prompt_memories=out), indent=2)
    )
    return out


class Controller(LLM):
    def __init__(self, reasoned, prior):
        super().__init__()
        self.reasoned = reasoned
        self.prior = prior

    def call(self, request):
        req = copy.deepcopy(request)
        req["max_tokens"] = 512
        state = json.loads(req["messages"][1]["content"])
        if self.prior:
            state["prior_workloads"] = self.prior
            state["prior_scope"] = (
                "Measured records from other FFT contracts; their latency is not a measurement of the current target shape."
            )
        req["messages"][1]["content"] = json.dumps(state, separators=(",", ":"))
        if self.reasoned:
            flat = "id" in req["output_schema"].get("properties", {})
            req["messages"][0]["content"] = req["messages"][0]["content"].split("Return JSON only")[0] + (
                "First give a short diagnosis comparing the largest launch reduction, useful block parallelism and supplied measured latency. Then choose a decision. "
                "Use actual resource or latency differences in the diagnosis; unmeasured latency is unknown. "
                + (
                    "The decision selects one listed candidate ID."
                    if flat
                    else "The decision gives one legal typed action, with level, region, operation and parameters."
                )
                + " Return JSON with diagnosis and decision fields only."
            )
            req["output_schema"] = dict(
                type="object",
                properties=dict(diagnosis=dict(type="string", maxLength=500), decision=req["output_schema"]),
                required=["diagnosis", "decision"],
                additionalProperties=False,
            )
        raw = super().call(req)
        raw["controller"] = dict(reasoned=self.reasoned, has_prior=bool(self.prior))
        return raw


def main():
    prepare_workspace()
    memo = memories()
    root = ROOT / "experiments" / "v46"
    (root / "M4").mkdir(exist_ok=True)
    protocol = Path(__file__).with_name("PROTOCOL_REASONING_TRANSFER.md")
    (ROOT / "evidence" / "v46_transfer_preregistration.json").write_text(
        json.dumps(
            dict(
                protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                conditions=CONDITIONS,
                shape=[8, 16],
                space=36864,
                seeds=[41, 42, 43],
                budget=12,
            ),
            indent=2,
        )
    )
    target = LiveRunner()
    try:
        for seed in (41, 42, 43):
            order = list(CONDITIONS)
            random.Random(seed + 4600).shuffle(order)
            for label in order:
                base, reasoned, prior = CONDITIONS[label]
                llm = (
                    Controller(reasoned, memo.get(prior))
                    if base not in ("greedy", "random", "evolution")
                    else None
                )
                run_search(target, llm, "M4", base, seed, label=label)
                if llm and llm.s:
                    llm.s.close()
    finally:
        target.close_live()
        target.s.close()
        target.c.close()


if __name__ == "__main__":
    main()
