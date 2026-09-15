# Author: even
"""Post-inspection decoder controls; original prompts and failures stay intact."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import copy, json, random, hashlib, os, time, urllib.request, concurrent.futures
from pathlib import Path
from . import catalog, live_target
from .hosted_study import ROOT, NEW, REMOTE
from .hosted_parallel import SerializedTarget
from .structured_search import run_search


class SchemaController:
    s = None

    def call(self, request):
        req = copy.deepcopy(request)
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
        # HostedController parses and serializes this state in exactly this way.
        req["messages"][1]["content"] = json.dumps(
            json.loads(req["messages"][1]["content"]), separators=(",", ":")
        )
        schema = dict(
            type="object",
            properties=dict(diagnosis=dict(type="string"), decision=req["output_schema"]),
            required=["diagnosis", "decision"],
            additionalProperties=False,
        )
        native = dict(
            model=os.environ.get("SAGE_LLM_MODEL", ""),
            messages=req["messages"],
            temperature=0.4,
            max_tokens=512,
            enable_thinking=False,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "fft_decision", "strict": True, "schema": schema},
            },
        )
        from .llm import ModelClient

        return ModelClient().call(native)


def main():
    prepare_workspace()
    old = [
        p
        for p in (NEW / "M3/search").glob("*/*/step_[0-9][0-9].json")
        if not p.parent.parent.name.startswith("schema_")
    ]
    assert len(old) == 180
    protocol = Path(__file__).with_name("PROTOCOL_HOSTED_SCHEMA.md")
    reg = ROOT / "evidence/v46_hosted_schema_preregistration.json"
    identity = dict(
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        conditions=["schema_flat", "schema_typed"],
        budget=12,
        seeds=[41, 42, 43],
        workload="M3",
        status="explicitly post-inspection decoder control",
    )
    if reg.exists():
        assert json.loads(reg.read_text()) == identity
    else:
        reg.write_text(json.dumps(identity, indent=2))
    catalog.REMOTE = REMOTE
    live_target.REMOTE = REMOTE
    target = SerializedTarget()
    target.root = NEW
    jobs = [(label, seed) for label in ("schema_flat", "schema_typed") for seed in (41, 42, 43)]
    random.Random(46104).shuffle(jobs)

    def execute(job):
        label, seed = job
        run_search(
            target,
            SchemaController(),
            "M3",
            "flat_llm" if label == "schema_flat" else "sage",
            seed,
            label=label,
            output_root=NEW,
        )
        return dict(policy=label, seed=seed)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for result in pool.map(execute, jobs):
                print(json.dumps(dict(completed_trial=result)), flush=True)
    finally:
        target.close_live()
        target.s.close()
        target.c.close()


if __name__ == "__main__":
    main()
