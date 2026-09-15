# Author: even
"""Run the additional finite-space policy controls and held-out calibration."""

import argparse
import json
import runpy

from sage_fft.config import prepare_workspace
from sage_fft.campaign import Runner, policies, fft_only
from sage_fft.fft_ir import Contract, candidates
from sage_fft.quality_experiments import calibration


def run():
    workspace = prepare_workspace()
    for name, count in (("scaling_records.jsonl", 27), ("scaling_cuda_records.jsonl", 9)):
        path = workspace / "evidence" / name
        if not path.exists() or len(path.read_text(encoding="utf-8").splitlines()) != count:
            raise ValueError("complete the matched scaling measurements first: " + name)
    runner = Runner()
    try:
        policies(runner)
        fft_only(runner)
        runpy.run_module("sage_fft.analyze_results", run_name="__main__")
        calibration(runner)
        rows = [json.loads(line) for line in runner.journal.read_text(encoding="utf-8").splitlines()]
        covered = [
            r["action"]
            for r in rows
            if r.get("pass_correctness")
            and r.get("mode") == "pipeline"
            and r["contract"]["shape"] == [8, 8, 8]
        ]
        for index, action in enumerate(candidates()):
            if action not in covered:
                result = runner.evaluate(
                    Contract((8, 8, 8), 1),
                    action,
                    "oracle_fill_" + str(index),
                    seeds=(4,),
                    phase="oracle_coverage",
                )
                if not result.get("pass_correctness"):
                    raise RuntimeError("finite-space reference preparation failed")
    finally:
        runner.s.close()
        runner.c.close()
    runpy.run_module("sage_fft.oracle_calibration", run_name="__main__")
    runpy.run_module("sage_fft.analyze_quality", run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-hardware", action="store_true")
    args = parser.parse_args()
    if not args.enable_hardware:
        parser.error("pass --enable-hardware to run device and model experiments")
    run()
