# Author: even
"""Frozen hosted-model comparison using real NPU feedback; credentials stay in env."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import copy, json, random, hashlib, os, time, urllib.request, shutil
from pathlib import Path
from . import catalog, live_target
from .catalog import ROOT
from .live_target import LiveRunner
from .structured_search import run_search
from .transfer_study import memories

NEW = ROOT / "experiments/v46_api"
REMOTE = remote_root("hosted")
CONDITIONS = {
    "flat": ("flat_llm", False, None),
    "typed": ("sage", False, None),
    "no_history": ("no_history", False, None),
    "prior": ("sage", True, "related"),
    "greedy": ("greedy", False, None),
}


class HostedController:
    s = None

    def __init__(self, prior):
        self.prior = prior

    def call(self, request):
        req = copy.deepcopy(request)
        state = json.loads(req["messages"][1]["content"])
        if self.prior:
            state["prior_workloads"] = self.prior
            state["prior_scope"] = (
                "Measured records from other FFT contracts; their latency is not a measurement of the current target shape."
            )
        req["messages"][1]["content"] = json.dumps(state, separators=(",", ":"))
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
        native = dict(
            model=os.environ.get("SAGE_LLM_MODEL", ""),
            messages=req["messages"],
            temperature=0.4,
            max_tokens=512,
            enable_thinking=False,
            response_format={"type": "json_object"},
        )
        from .llm import ModelClient

        return ModelClient().call(native)


def main():
    prepare_workspace()
    protocol = Path(__file__).with_name("PROTOCOL_HOSTED_MODEL.md")
    NEW.mkdir(exist_ok=True)
    reg = ROOT / "evidence/v46_hosted_preregistration.json"
    identity = dict(
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        conditions=CONDITIONS,
        model=os.environ.get("SAGE_LLM_MODEL", ""),
        budget=12,
        seeds=[41, 42, 43],
        workload="M3",
    )
    if reg.exists():
        assert json.loads(reg.read_text()) == identity, "Frozen protocol/source changed"
    else:
        reg.write_text(json.dumps(identity, indent=2))
    memo = memories()
    catalog.REMOTE = REMOTE
    live_target.REMOTE = REMOTE
    target = LiveRunner()
    target.root = NEW
    try:
        target.prepare("M3")
        for seed in (41, 42, 43):
            order = list(CONDITIONS)
            random.Random(seed + 46100).shuffle(order)
            for label in order:
                base, _, prior = CONDITIONS[label]
                controller = HostedController(memo.get(prior)) if base != "greedy" else None
                run_search(target, controller, "M3", base, seed, label=label, output_root=NEW)
    finally:
        target.close_live()
        target.s.close()
        target.c.close()


if __name__ == "__main__":
    main()
