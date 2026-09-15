# Author: even
"""Summarize retained search decisions, including transport and token-limit failures."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def rejection_kind(row, proposal):
    if row.get("accepted"):
        return "accepted"
    if proposal and proposal.get("exit_code") != 0:
        return "transport"
    if proposal and proposal.get("response"):
        try:
            reply = json.loads(proposal["response"])
            finish = [choice.get("finish_reason") for choice in reply.get("choices", [])]
            if "length" in finish or reply.get("done_reason") == "length":
                return "token_limit"
        except json.JSONDecodeError:
            return "invalid_response"
    if "no unseen" in row.get("error", ""):
        return "menu_exhausted"
    return "rejected_decision"


def audit(root, seeds=(41, 42, 43), steps=12):
    root = Path(root)
    policies = {}
    artifacts = []
    models = set()
    complete = True
    for policy in sorted(p for p in root.iterdir() if p.is_dir()):
        actual_seeds = sorted(int(p.name) for p in policy.iterdir() if p.is_dir())
        complete &= actual_seeds == sorted(seeds)
        seed_rows = []
        reasons = Counter()
        for seed in seeds:
            records = sorted((policy / str(seed)).glob("step_[0-9][0-9].json"))
            complete &= [int(p.stem.rsplit("_", 1)[1]) for p in records] == list(range(1, steps + 1))
            accepted = 0
            seconds = 0.0
            for path in records:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("search_seed") != seed or row.get("policy") != policy.name:
                    raise ValueError("trace identity mismatch: " + str(path))
                if row.get("accepted") and row["feedback"]["correctness"] is not True:
                    raise ValueError("accepted search step has invalid correctness")
                proposal_path = path.with_name(path.stem + ".proposal.json")
                proposal = (
                    json.loads(proposal_path.read_text(encoding="utf-8")) if proposal_path.exists() else None
                )
                for artifact in (path, proposal_path):
                    if artifact.exists():
                        artifacts.append(
                            dict(
                                path=artifact.relative_to(root).as_posix(),
                                sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                            )
                        )
                if proposal and proposal.get("request", {}).get("model"):
                    models.add(proposal["request"]["model"])
                reasons[rejection_kind(row, proposal)] += 1
                accepted += bool(row.get("accepted"))
                seconds += row.get("model_seconds", 0)
            seed_rows.append(
                dict(seed=seed, proposals=len(records), accepted=accepted, model_seconds=seconds)
            )
        count = sum(row["proposals"] for row in seed_rows)
        accepted = sum(row["accepted"] for row in seed_rows)
        policies[policy.name] = dict(
            proposals=count,
            accepted=accepted,
            acceptance_percent=100 * accepted / count if count else None,
            median_inference_seconds=statistics.median(row["model_seconds"] for row in seed_rows),
            outcome_counts=dict(reasons),
            seeds=seed_rows,
        )
    if not policies:
        complete = False
    return dict(
        complete=bool(complete),
        models=sorted(models),
        policies=policies,
        artifacts=artifacts,
        scope="Retained search traces; device-output verification is a separate acceptance check.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--steps", default=12, type=int)
    args = parser.parse_args()
    result = audit(args.root, tuple(map(int, args.seeds.split(","))), args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("complete", "models", "policies")}, indent=2))
    if not result["complete"]:
        raise SystemExit(1)
