# Author: even
"""Build and measure fresh static FFT realizations on Ascend."""

from .config import workspace, remote_root, npu_environment, connect_ssh
import argparse
import base64
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import time
import numpy as np
from .fft_ir import Contract, candidates, identity, metrics
from .cuda_source import source
from .lower import emit
from .validate_semantics import WORKLOADS, inputs, reference

ROOT = workspace()
REMOTE = remote_root("base")
ENV = npu_environment()


class Runner:
    def __init__(self):
        self.c = connect_ssh("NPU")
        self.s = self.c.open_sftp()
        self.root = ROOT / "experiments" / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        (ROOT / "evidence").mkdir(parents=True, exist_ok=True)
        self.journal = ROOT / "evidence" / "target_records.jsonl"

    def command(self, cmd, timeout=300):
        _, o, e = self.c.exec_command("bash -s", timeout=timeout)
        o.channel.sendall(cmd.encode())
        o.channel.shutdown_write()
        out = o.read().decode(errors="replace")
        err = e.read().decode(errors="replace")
        return o.channel.recv_exit_status(), out, err

    def record(self, r):
        with self.journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        print(
            json.dumps(
                {
                    k: r.get(k)
                    for k in ("id", "phase", "seed", "build_ok", "pass_correctness", "p50_us", "error")
                }
            ),
            flush=True,
        )

    def evaluate(self, c, a, tag, seeds=(1, 2, 3), mode="pipeline", phase="ablation"):
        d = self.root / tag
        remote = REMOTE + "/" + tag
        if d.exists():
            raise RuntimeError("fresh-build directory already exists: " + tag)
        t = time.perf_counter()
        ir = emit(c, a, d, mode)
        cuda, cuda_contract = source(c.shape, c.batch)
        assert cuda_contract == c
        (d / "source.cu").write_text(cuda, encoding="utf-8", newline="\n")
        rid, full = identity(c, a, mode)
        r = dict(
            id=tag,
            phase=phase,
            contract=asdict(c),
            action=a,
            mode=mode,
            ir_sha256=rid,
            source_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
            source_hash_format="UTF-8 text normalized to LF line endings",
            target_source_sha256=hashlib.sha256((d / "kernels.cpp").read_bytes()).hexdigest(),
            timestamp=time.time(),
            build_ok=False,
        )
        r["lower_seconds"] = time.perf_counter() - t
        self.command("mkdir -p " + shlex.quote(remote))
        for p in d.iterdir():
            if p.is_file():
                self.s.put(str(p), remote + "/" + p.name)
        build = (
            ENV
            + "set -e\ncd "
            + shlex.quote(remote)
            + """\ncmake -S . -B build -DCMAKE_CXX_COMPILER="${CXX:-g++}" -DCMAKE_BUILD_TYPE=Release >configure.log 2>&1
cmake --build build -j2 >build.log 2>&1
sha256sum build/sage_fft
"""
        )
        (d / "build.sh").write_text(build)
        t = time.perf_counter()
        rc, out, err = self.command(build)
        r["build_seconds"] = time.perf_counter() - t
        for name in ("configure.log", "build.log"):
            try:
                self.s.get(remote + "/" + name, str(d / name))
            except FileNotFoundError:
                pass
        if rc:
            r["error"] = "build_failed"
            r["stderr"] = err
            r["exit_code"] = rc
            self.record(r)
            return r
        r["build_ok"] = True
        r["binary_sha256"] = out.strip().split()[0]
        self.s.get(remote + "/build/sage_fft", str(d / "sage_fft"))
        result = []
        for seed in seeds:
            ip = d / ("input_%d" % seed)
            ip.mkdir()
            x, h = inputs(c, seed)
            x.tofile(ip / "x.bin")
            h.tofile(ip / "h.bin")
            rr = remote + "/" + ip.name
            self.command("mkdir -p " + shlex.quote(rr))
            for name in ("x.bin", "h.bin"):
                self.s.put(str(ip / name), rr + "/" + name)
            cmd = ENV + "cd " + shlex.quote(remote) + "\n./build/sage_fft " + ip.name + " out_%d.bin" % seed
            rc, out, err = self.command(cmd, 120)
            (d / ("run_%d.log" % seed)).write_text(out + err)
            row = dict(r, seed=seed, exit_code=rc)
            if rc:
                row["error"] = "run_failed"
                self.record(row)
                result.append(row)
                continue
            self.s.get(remote + "/out_%d.bin" % seed, str(d / ("out_%d.bin" % seed)))
            y = np.fromfile(d / ("out_%d.bin" % seed), np.complex64).reshape(x.shape)
            row.update(metrics(y, reference(c, x, h, mode)))
            row["input_sha256"] = {
                name: hashlib.sha256((ip / name).read_bytes()).hexdigest() for name in ("x.bin", "h.bin")
            }
            row["output_sha256"] = hashlib.sha256((d / ("out_%d.bin" % seed)).read_bytes()).hexdigest()
            timing = json.loads(next(v for v in out.splitlines() if v.startswith("{")))
            row.update(timing)
            row["p50_us"] = float(np.median(timing["samples_us"]))
            row["timing_scope"] = (
                "host batch submit + final stream synchronization; 10 groups x 3 pipeline executions"
            )
            row["kernel_launches"] = len(ir["stages"])
            row["logical_gm_bytes"] = 16 * math.prod(c.shape) * c.batch * len(ir["stages"]) + (
                8 * math.prod(c.shape) * c.batch if mode == "pipeline" else 0
            )
            row["traffic_kind"] = "analytical tensor reads/writes; not profiler hardware counters"
            self.record(row)
            result.append(row)
        return result[0] if len(result) == 1 else result

    def llm(self, prompt, seed):
        from .llm import ModelClient

        request = dict(
            messages=[
                dict(
                    role="system",
                    content='Select a legal FFT implementation ID. Return JSON only: {"id":integer}. Use supplied measured latency when available; lower is better.',
                ),
                dict(role="user", content=prompt),
            ],
            temperature=0.4,
            seed=seed,
            max_tokens=40,
        )
        result = ModelClient().call(request)
        return request, result["exit_code"], result["response"], result["stderr"], result["model_seconds"]


def verified_case(r, c, action, tag, mode="pipeline", phase="ablation"):
    """Resume only complete, matching runs; retain failures for inspection."""
    folder = r.root / tag
    if folder.exists():
        if not r.journal.is_file():
            raise RuntimeError("existing case has no measurement journal: " + tag)
        rows = [json.loads(line) for line in r.journal.read_text(encoding="utf-8").splitlines()]
        rows = [row for row in rows if row.get("id") == tag]
    else:
        result = r.evaluate(c, action, tag, mode=mode, phase=phase)
        rows = result if isinstance(result, list) else [result]
    if (
        len(rows) != 3
        or sorted(row.get("seed", -1) for row in rows) != [1, 2, 3]
        or not all(
            row.get("build_ok")
            and row.get("exit_code") == 0
            and row.get("pass_correctness")
            and row.get("action") == action
            and row.get("mode") == mode
            for row in rows
        )
    ):
        raise RuntimeError("case lacks three verified input seeds: " + tag)
    for row in rows:
        if row.get("ir_sha256") != identity(c, action, mode)[0]:
            raise RuntimeError("case IR disagrees with the requested contract: " + tag)
        for name, field in (("kernels.cpp", "target_source_sha256"), ("sage_fft", "binary_sha256")):
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != row[field]:
                raise RuntimeError("case artifact identity changed: " + tag)
        x, h = inputs(c, row["seed"])
        ip = folder / ("input_%d" % row["seed"])
        if (ip / "x.bin").read_bytes() != x.tobytes() or (ip / "h.bin").read_bytes() != h.tobytes():
            raise RuntimeError("case input bytes changed: " + tag)
        output = folder / ("out_%d.bin" % row["seed"])
        y = np.fromfile(output, np.complex64).reshape(x.shape)
        if not metrics(y, reference(c, x, h, mode))["pass_correctness"]:
            raise RuntimeError("retained case output fails correctness: " + tag)
    return rows


def ablation(r):
    configs = [dict(group=1, cores=1, fusion=False)]
    configs += [dict(group=1, cores=c, fusion=False) for c in (4, 8)]
    configs += [dict(group=g, cores=c, fusion=f) for f in (False, True) for g in (2, 16) for c in (4, 8)]
    for wi, (shape, batch) in enumerate(WLOADS):
        c = Contract(shape, batch)
        for ai, a in enumerate(configs):
            tag = "ablation_w%d_a%d" % (wi, ai)
            verified_case(r, c, a, tag)
    # Full transform-only checks plus diagnostic signal patterns are separate runs.
    for wi, (shape, batch) in enumerate(WLOADS):
        for mode in ("forward", "inverse"):
            tag = "semantics_w%d_%s" % (wi, mode)
            verified_case(
                r,
                Contract(shape, batch),
                dict(group=16, cores=4, fusion=True),
                tag,
                mode=mode,
                phase="semantics",
            )


WLOADS = WORKLOADS


def fft_only(r):
    for g in (1, 2, 16):
        tag = "fft_only_g%d_c4" % g
        if (r.root / tag).exists():
            continue
        r.evaluate(
            Contract((8, 8, 8), 1),
            dict(group=g, cores=4, fusion=False),
            tag,
            seeds=(1, 2, 3),
            mode="forward",
            phase="fft_only",
        )


def policies(r):
    pool = candidates()
    c = Contract((8, 8, 8), 1)
    order = sorted(
        range(len(pool)), key=lambda i: (-pool[i]["group"], -int(pool[i]["fusion"]), -pool[i]["cores"])
    )
    for seed in (41, 42, 43):
        for policy in ("fixed_heuristic", "uniform_random", "llm_no_history", "sage"):
            history = []
            seen = set()
            rng = random.Random(seed)
            current = dict(group=1, cores=1, fusion=False)
            for budget in range(1, 9):
                tag = "policy_%s_s%d_b%d" % (policy, seed, budget)
                if (r.root / tag).exists():
                    raise RuntimeError("policy resumptions require full history, choose a clean run")
                legal = [
                    i
                    for i in range(len(pool))
                    if i not in seen
                    and pool[i] != current
                    and not (pool[i]["group"] != current["group"] and pool[i]["fusion"] != current["fusion"])
                ]
                proposal = dict(policy=policy, search_seed=seed, budget=budget, legal_ids=legal)
                if policy == "fixed_heuristic":
                    idx = next(i for i in order if i in legal)
                elif policy == "uniform_random":
                    idx = rng.choice(legal)
                else:
                    presentation = legal.copy()
                    random.Random(seed + budget).shuffle(presentation)
                    state = dict(
                        shape=c.shape,
                        dtype=c.dtype,
                        current_realization=current,
                        goal="minimize complete FFT -> complex multiply -> normalized inverse FFT latency",
                        target="Ascend310P1, generated axis FFT kernels; group=16 means all stages of an axis in UB; fusion merges multiply and normalization; cores are AI Core blocks",
                        candidates=[dict(id=i, **pool[i]) for i in presentation],
                        budget_remaining=9 - budget,
                    )
                    if policy == "sage":
                        state["history"] = history
                    prompt = json.dumps(state, separators=(",", ":"))
                    req, rc, out, err, secs = r.llm(prompt, seed * 100 + budget)
                    proposal.update(request=req, response=out, model_seconds=secs)
                    try:
                        response = json.loads(out)
                        content = response["choices"][0]["message"]["content"]
                        import re

                        parsed = json.loads(re.search(r"\{[^{}]*\}", content).group())
                        assert isinstance(parsed["id"], int) and not isinstance(parsed["id"], bool)
                        idx = parsed["id"]
                        assert idx in legal and not rc
                        proposal["response_model"] = response.get("model")
                    except Exception:
                        proposal.update(error="invalid_model_proposal", stderr=err)
                        (r.root / (tag + ".proposal.json")).write_text(json.dumps(proposal, indent=2))
                        r.record(dict(id=tag, phase="policy", **proposal))
                        history.append(dict(budget=budget, error="invalid_model_proposal"))
                        continue
                proposal["chosen_id"] = idx
                seen.add(idx)
                action = pool[idx]
                level = (
                    "pipeline"
                    if action["fusion"] != current["fusion"]
                    else ("FFT-structure" if action["group"] != current["group"] else "schedule")
                )
                operation = {
                    "pipeline": ("FuseEpilogues" if action["fusion"] else "SeparateEpilogues"),
                    "FFT-structure": "GroupButterflies",
                    "schedule": "AllocateCores",
                }[level]
                operand = {
                    "pipeline": "forward multiply and inverse scale",
                    "FFT-structure": "axis butterfly stages",
                    "schedule": "current FFT graph",
                }[level]
                proposal["parent_realization"] = current.copy()
                proposal["hierarchical_action"] = dict(
                    level=level,
                    operation=operation,
                    operand=operand,
                    schedule={"cores": action["cores"]},
                    group=action["group"],
                    fusion=action["fusion"],
                )
                result = r.evaluate(c, action, tag, seeds=(1,), phase="policy")
                proposal["measurement_id"] = tag
                (r.root / (tag + ".proposal.json")).write_text(json.dumps(proposal, indent=2))
                feedback = dict(
                    id=idx,
                    action=action,
                    pass_correctness=result.get("pass_correctness"),
                    latency_us=result.get("p50_us"),
                    build_ok=result["build_ok"],
                    max_abs=result.get("max_abs"),
                    relative_l2=result.get("relative_l2"),
                    kernel_launches=result.get("kernel_launches"),
                    logical_gm_bytes=result.get("logical_gm_bytes"),
                )
                if not result["build_ok"]:
                    log = r.root / tag / "build.log"
                    feedback["build_diagnostics"] = (
                        log.read_text(encoding="utf-8", errors="replace")[-800:]
                        if log.exists()
                        else result.get("error")
                    )
                history.append(feedback)
                if result.get("pass_correctness"):
                    current = action.copy()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["ablation", "policy", "fft_only"])
    args = p.parse_args()
    r = Runner()
    try:
        if args.phase == "ablation":
            ablation(r)
        elif args.phase == "policy":
            policies(r)
            fft_only(r)
        else:
            fft_only(r)
    finally:
        r.s.close()
        r.c.close()
