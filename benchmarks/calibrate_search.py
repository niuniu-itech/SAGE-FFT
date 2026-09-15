# Author: even
"""Calibrate final hosted interfaces against the local and hosted candidate union."""

import argparse
import json
from pathlib import Path

from sage_fft.config import prepare_workspace, remote_root


def calibrate(result_root=None):
    workspace = prepare_workspace()
    root = Path(result_root) if result_root is not None else workspace / "experiments/hosted_v489"
    completed = json.loads((root / "acceptance.json").read_text(encoding="utf-8"))
    if not completed.get("completed") or completed.get("kind") != "hosted":
        raise ValueError("hosted search and its selected-state checks must finish first")
    proposals = list((root / "M3/search").glob("*/*/step_*.proposal.json"))
    models = {json.loads(p.read_text(encoding="utf-8"))["request"]["model"] for p in proposals}
    if len(models) != 1:
        raise ValueError("hosted records must identify one measured model")
    if not (workspace / "experiments/v46/M3/heldout_candidates.json").exists():
        raise ValueError("complete local final_calibration before the common hosted calibration")
    from sage_fft.calibrate_hosted import main

    main(
        expected_records=completed["search_runs"] * completed["proposals_per_run"],
        result_root=root,
        model=next(iter(models)),
        summary_path=workspace / "evidence/hosted_v489_summary.json",
        catalog_remote=remote_root("catalog"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path)
    calibrate(parser.parse_args().result_root)
