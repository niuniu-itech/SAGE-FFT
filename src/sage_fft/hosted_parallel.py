# Author: even
"""Run independent policy trials concurrently and serialize NPU measurements."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
from pathlib import Path
import json, random, threading, concurrent.futures, hashlib, time
from . import catalog, live_target
from .hosted_study import ROOT, NEW, REMOTE, CONDITIONS, HostedController, memories, run_search
from .live_target import LiveRunner


class SerializedTarget(LiveRunner):
    def __init__(self):
        super().__init__()
        self.mutex = threading.RLock()

    def evaluate_many(self, *args, **kwargs):
        with self.mutex:
            return super().evaluate_many(*args, **kwargs)


def main():
    prepare_workspace()
    old = Path(__file__).with_name("hosted_study.py")
    reg = json.loads((ROOT / "evidence/v46_hosted_preregistration.json").read_text())
    assert hashlib.sha256(old.read_bytes()).hexdigest() == reg["source_sha256"]
    amendment = ROOT / "evidence/v46_hosted_execution_amendment.json"
    if not amendment.exists():
        completed = list((NEW / "M3/search").glob("*/*/step_[0-9][0-9].json"))
        amendment.write_text(
            json.dumps(
                dict(
                    reason="Reduce remote inference wall time by overlapping independent trials; NPU measurements remain serialized.",
                    completed_at_change=len(completed),
                    workers=3,
                    unchanged=[
                        "model",
                        "prompts",
                        "per-trial order",
                        "proposal budgets",
                        "candidate menu",
                        "incumbent rule",
                        "saved decisions",
                    ],
                    cost_scope="HTTP response time as observed; includes provider queueing; initial requests serial, later requests concurrent. Not controlled cross-model latency evidence.",
                    interrupted_call="One in-flight request may have been interrupted before a response was saved; all saved answers are retained.",
                ),
                indent=2,
            )
        )
    catalog.REMOTE = REMOTE
    live_target.REMOTE = REMOTE
    target = SerializedTarget()
    target.root = NEW
    memo = memories()
    jobs = [(seed, label) for seed in (41, 42, 43) for label in CONDITIONS]
    random.Random(46103).shuffle(jobs)

    def execute(job):
        seed, label = job
        base, _, prior = CONDITIONS[label]
        controller = HostedController(memo.get(prior)) if base != "greedy" else None
        run_search(target, controller, "M3", base, seed, label=label, output_root=NEW)
        return dict(seed=seed, policy=label)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(execute, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                print(json.dumps(dict(completed_trial=future.result())), flush=True)
    finally:
        target.close_live()
        target.s.close()
        target.c.close()


if __name__ == "__main__":
    main()
