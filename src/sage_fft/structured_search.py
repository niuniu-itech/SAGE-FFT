# Author: even
"""Offline FFT exploration with indexed actions and verified target feedback.

run_search implements Algorithm 1: build the legal menu, decode a decision,
measure its realization, and retain valid history. Current state may move to a
slower valid candidate; the returned incumbent is the fastest verified state.
Candidate failures consume budget without replacing valid history. Saved steps
support resume, and the resulting native execution path contains no model call.
"""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
from .catalog import CatalogRunner, ROOT
from .live_target import LiveRunner
from .mask_ir import WORKLOADS, initial, actions, apply, key, features, random_realization
from .search_errors import CandidateRejected, CandidateCorrectnessError

POLICIES = ("greedy", "random", "evolution", "flat_llm", "no_history", "sage", "shuffled")


def analytic(c, r):
    f = features(c, r)
    return (
        f["launches"],
        -min(f["useful_blocks"]),
        -sum(f["useful_blocks"]),
        f["UB_KiB"],
        r["cores"],
        r["pack"],
    )


class LLM:
    s = None

    def __init__(self):
        from .llm import ModelClient

        self.client = ModelClient()

    def call(self, request):
        return self.client.call(request)


def make_request(name, current, history, policy, seed, step, seen):
    c, packs, _ = WORKLOADS[name]
    menu = [(a, z) for a, z in actions(c, current, packs) if key(z) not in seen]
    random.Random(seed * 100 + step).shuffle(menu)
    state = dict(
        shape=c.shape,
        batch=c.batch,
        budget_remaining=13 - step,
        current=current,
        current_resources=features(c, current),
        target="Ascend 310P1",
        semantics="complex64, natural bins, normalized inverse, FFT -> complex multiply -> inverse FFT",
        controls="Masks: 1 materializes after that radix-2 stage, 0 keeps the boundary inside a kernel. FFT axes execute last-to-first. All-axis merge/split is legal. Epilogues are independent. Cores are launch blocks; pack is adjacent strided FFT lines per block. Full-axis buffers: UB bytes depend on pack, not boundary count. Resources are analytical, latency is measured.",
        candidates=[
            dict(**({"id": i} if policy == "flat_llm" else {}), action=a, result_resources=features(c, z))
            for i, (a, z) in enumerate(menu)
        ],
    )
    if policy != "no_history":
        visible = copy.deepcopy(history[-6:])
        best = min(history, key=lambda r: r["latency_us"])
        if not any(v["realization"] == best["realization"] for v in visible):
            visible = [copy.deepcopy(best)] + visible
        if policy == "shuffled":
            vals = [h["latency_us"] for h in visible]
            random.Random(seed * 10000 + step).shuffle(vals)
            for h, lat in zip(visible, vals):
                h["latency_us"] = lat
        current_latency = next(h["latency_us"] for h in reversed(visible) if h["realization"] == current)
        best_latency = min(h["latency_us"] for h in visible)
        incumbent = min(visible, key=lambda h: h["latency_us"])
        current_features = features(c, current)
        for h in visible:
            h["delta_current_us"] = round(h["latency_us"] - current_latency, 6)
            h["delta_incumbent_us"] = round(h["latency_us"] - best_latency, 6)
            h["delta_resources_current"] = {
                k: h["resources"][k] - current_features[k] for k in ("launches", "logical_KiB", "UB_KiB")
            }
            h["delta_resources_incumbent"] = {
                k: h["resources"][k] - incumbent["resources"][k]
                for k in ("launches", "logical_KiB", "UB_KiB")
            }
        state["feedback"] = visible
    output = (
        'Return JSON only, exactly {"id": integer} selecting one candidate ID.'
        if policy == "flat_llm"
        else "Return JSON only: copy the typed action object of one candidate (level, region, op and its parameters). Do not return an ID or invent an action."
    )
    system = (
        "You control compiler-checked FFT migration. Choose the next legal transformation to minimize complete target pipeline latency within the remaining budget. Use measured outcomes when supplied; account for launches, memory materialization and useful parallelism. "
        + output
    )
    request = dict(
        model=os.environ.get("SAGE_LLM_MODEL", ""),
        messages=[
            dict(role="system", content=system),
            dict(role="user", content=json.dumps(state, separators=(",", ":"))),
        ],
        temperature=0.4,
        seed=seed * 100 + step,
        max_tokens=192,
    )
    if policy == "flat_llm":
        schema = dict(
            type="object",
            properties=dict(id=dict(type="integer", minimum=0, maximum=len(menu) - 1)),
            required=["id"],
            additionalProperties=False,
        )
    else:

        def obj(properties):
            return dict(
                type="object", properties=properties, required=list(properties), additionalProperties=False
            )

        regions = ["all"] + [d + ":" + str(i) for d in ("forward", "inverse") for i in range(len(c.shape))]
        schema = {
            "oneOf": [
                obj(
                    dict(
                        level={"const": "fft"},
                        region={"enum": regions},
                        op={"enum": ["merge", "split"]},
                        boundaries={
                            "anyOf": [
                                {"const": "all"},
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": max(c.shape).bit_length() - 2,
                                    },
                                    "minItems": 1,
                                },
                            ]
                        },
                    )
                ),
                obj(
                    dict(
                        level={"const": "pipeline"},
                        region={"enum": ["multiply", "scale", "both"]},
                        op={"enum": ["fuse", "unfuse"]},
                    )
                ),
                obj(
                    dict(
                        level={"const": "schedule"},
                        region={"const": "all"},
                        op={"const": "reschedule"},
                        schedule=obj(dict(cores={"enum": [1, 4, 8]}, pack={"enum": list(packs)})),
                    )
                ),
            ]
        }
    request["output_schema"] = schema
    return request, menu


def parse_response(raw, policy, menu, c, current):
    response = json.loads(raw["response"])
    content = (
        response["choices"][0]["message"]["content"]
        if "choices" in response
        else response["message"]["content"]
    )
    parsed = json.JSONDecoder().raw_decode(content[content.index("{") :])[0]
    if set(parsed) == {"diagnosis", "decision"}:
        parsed = parsed["decision"]
    if policy == "flat_llm":
        if set(parsed) != {"id"} or type(parsed["id"]) is not int or not 0 <= parsed["id"] < len(menu):
            raise ValueError("invalid candidate ID")
        action, z = menu[parsed["id"]]
    else:
        action = parsed
        z = apply(c, current, action)
        if key(z) not in {key(v) for _, v in menu}:
            raise ValueError("action outside common legal menu")
    usage = response.get(
        "usage",
        dict(prompt_tokens=response.get("prompt_eval_count"), completion_tokens=response.get("eval_count")),
    )
    timings = response.get(
        "timings",
        {
            k: response[k]
            for k in (
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
                "done_reason",
            )
            if k in response
        },
    )
    return action, z, usage, timings


def feedback(c, row):
    return dict(
        realization=row["realization"],
        resources=features(c, row["realization"]),
        latency_us=row["p50_us"],
        correctness=row["pass_correctness"],
        max_abs=row["max_abs"],
    )


def run_search(target, llm, name, policy, seed, label=None, output_root=None):
    """Run twelve proposals and return the best verified realization, including on resume."""
    label = label or policy
    c, packs, _ = WORKLOADS[name]
    current = initial(c)
    init = target.evaluate_many(name, [current], "search_initial", samples=10)[0]
    if not init["pass_correctness"]:
        raise CandidateCorrectnessError("initial realization must pass correctness before search", init)
    history = [feedback(c, init)]
    seen = {key(current)}
    rng = random.Random(seed)
    d = (
        (Path(output_root) if output_root else ROOT / "experiments" / "v46")
        / name
        / "search"
        / label
        / str(seed)
    )
    d.mkdir(parents=True, exist_ok=True)
    for step in range(1, 13):
        path = d / ("step_%02d.json" % step)
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("accepted"):
                if old["feedback"]["correctness"] is not True:
                    raise ValueError("accepted cached search step has invalid correctness")
                current = old["realization"]
                seen.add(key(current))
                history.append(old["feedback"])
            # Restore random state exactly, including previously rejected draws.
            if "rng_state" in old:

                def tup(v):
                    return tuple(tup(x) for x in v) if isinstance(v, list) else v

                rng.setstate(tup(old["rng_state"]))
            continue
        row = dict(
            workload=name,
            policy=label,
            interface=policy,
            search_seed=seed,
            step=step,
            parent=current,
            accepted=False,
        )
        if policy in ("greedy", "evolution"):
            parent = current
            if policy == "evolution":
                parent = rng.choice(sorted(history, key=lambda h: h["latency_us"])[:4])["realization"]
            opts = [(a, z) for a, z in actions(c, parent, packs) if key(z) not in seen]
            if not opts:
                z = random_realization(name, rng)
                action = None
            elif policy == "greedy":
                action, z = min(opts, key=lambda v: analytic(c, v[1]))
            else:
                action, z = rng.choice(opts)
            row["proposal_parent"] = parent
        elif policy == "random":
            z = random_realization(name, rng)
            while key(z) in seen:
                z = random_realization(name, rng)
            action = None
        else:
            request, menu = make_request(name, current, history, policy, seed, step, seen)
            proposal_path = d / ("step_%02d.proposal.json" % step)
            if proposal_path.exists():
                raw = json.loads(proposal_path.read_text())
            else:
                raw = llm.call(request)
                proposal_path.write_text(json.dumps(raw, indent=2))
            row["model_seconds"] = raw["model_seconds"]
            row["prompt_sha256"] = hashlib.sha256(
                json.dumps(raw["request"], sort_keys=True).encode()
            ).hexdigest()
            try:
                if raw["exit_code"]:
                    raise ValueError("inference transport failed")
                action, z, usage, timings = parse_response(raw, policy, menu, c, current)
                row.update(usage=usage, inference_timings=timings)
            except Exception as exc:
                row["error"] = str(exc)
                path.write_text(json.dumps(row, indent=2))
                print(json.dumps(row), flush=True)
                continue
        row.update(action=action, realization=z, rng_state=rng.getstate())
        if key(z) in seen:
            row["error"] = "duplicate"
            path.write_text(json.dumps(row, indent=2))
            continue
        measurement_tag = "search_%s_s%d_b%02d" % (label, seed, step)
        try:
            measured = target.evaluate_many(name, [z], measurement_tag, samples=5)[0]
        except CandidateRejected as exc:
            row.update(
                error=str(exc),
                failure_stage=exc.phase,
                measurement_tag=measurement_tag,
                incumbent_us=min(h["latency_us"] for h in history),
            )
            if exc.measurement is not None:
                row["feedback"] = feedback(c, exc.measurement)
            path.write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
            continue
        row.update(
            accepted=measured["pass_correctness"],
            feedback=feedback(c, measured),
            measurement_tag=measured["tag"],
        )
        if row["accepted"]:
            current = z
            seen.add(key(z))
            history.append(row["feedback"])
        else:
            row.update(error="candidate correctness failure", failure_stage="correctness")
        row["incumbent_us"] = min(h["latency_us"] for h in history)
        path.write_text(json.dumps(row, indent=2))
        print(
            json.dumps(
                {
                    k: row.get(k)
                    for k in (
                        "workload",
                        "policy",
                        "search_seed",
                        "step",
                        "accepted",
                        "incumbent_us",
                        "model_seconds",
                    )
                }
            ),
            flush=True,
        )
    return min(history, key=lambda h: h["latency_us"])["realization"]


def main():
    prepare_workspace()
    p = argparse.ArgumentParser()
    p.add_argument("--workloads", default="M2,M3")
    p.add_argument("--policies", default=",".join(POLICIES))
    p.add_argument("--seeds", default="41,42,43")
    a = p.parse_args()
    policies = a.policies.split(",")
    r = LiveRunner()
    llm = LLM() if any(p not in ("greedy", "random", "evolution") for p in policies) else None
    try:
        for seed in map(int, a.seeds.split(",")):
            for name in a.workloads.split(","):
                # Fixed premeasurement shuffle reduces order/temperature bias.
                order = policies.copy()
                random.Random(seed).shuffle(order)
                for policy in order:
                    run_search(r, llm, name, policy, seed)
    finally:
        r.close_live()
        r.s.close()
        r.c.close()
        if llm and llm.s:
            llm.s.close()


if __name__ == "__main__":
    main()
