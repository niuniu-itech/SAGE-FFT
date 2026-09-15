# Author: even
"""Persistent NPU runner for compiler-checked FFT launch plans.

Validated stage IR is mapped to cached kernels and submitted to a shared target
process. Results include output correctness and raw timing samples. Candidate
lowering or numerical failures are recoverable; shared build, transport and
process failures propagate to the caller. Cached results use the same gate.
"""

from .config import workspace, remote_root, npu_environment, connect_ssh
import hashlib
import json
import math
import shlex
import time
import numpy as np
from .catalog import CatalogRunner, REMOTE, ENV, generate, catalog_steps, canonical_step
from .mask_ir import WORKLOADS, stages, features
from .validate_semantics import inputs, reference
from .fft_ir import metrics
from .search_errors import CandidateLoweringError, CandidateCorrectnessError


def checked_measurements(rows, tag):
    """Apply the same numerical gate to fresh and cached target records."""
    for row in rows:
        if not row["pass_correctness"]:
            raise CandidateCorrectnessError("live target correctness failure " + tag, row)
    return rows


def live_host(old):
    start = old.index("std::ifstream pf(argv[2]);")
    end = old.index("const int samples=", start)
    old = old[:start] + "std::string id;int cores,n,k;\n" + old[end:]
    old = old.replace(
        "for(const auto& plan:plans){void* result=nullptr;int blocks=plan.cores;",
        'printf("READY\\n");fflush(stdout);\nwhile(std::cin>>id>>cores>>n){Plan plan;plan.id=id;plan.cores=cores;'
        "if(n<1||n>100||(cores!=1&&cores!=4&&cores!=8))return 4;"
        "for(int i=0;i<n;i++){if(!(std::cin>>k))return 4;plan.kernels.push_back(k);}"
        "void* result=nullptr;int blocks=plan.cores;",
    )
    old = old.replace("out.write((char*)yh.data(),bytes);", "out.write((char*)yh.data(),bytes);out.flush();")
    return old.replace("#include <fstream>", "#include <fstream>\n#include <iostream>")


class LiveRunner(CatalogRunner):
    def __init__(self):
        super().__init__()
        self.live = {}

    def start_live(self, name, seed=1, samples=5):
        cache = (name, seed, samples)
        if cache in self.live:
            return self.live[cache]
        c, packs, _ = WORKLOADS[name]
        d = self.root / name / "live_catalog"
        d.mkdir(parents=True, exist_ok=True)
        remote = REMOTE + "/" + name + "/live_catalog"
        mp = d / "build_metadata.json"
        if not mp.exists():
            generate(name, d)
            host = live_host((d / "main.cpp").read_text())
            (d / "main.cpp").write_text(host)
            self.command("mkdir -p " + shlex.quote(remote))
            for p in d.iterdir():
                if p.is_file():
                    self.s.put(str(p), remote + "/" + p.name)
            cmd = (
                ENV
                + "set -e\ncd "
                + shlex.quote(remote)
                + """
cmake -S . -B build -DCMAKE_CXX_COMPILER="${CXX:-g++}" -DCMAKE_BUILD_TYPE=Release >configure.log 2>&1
cmake --build build -j2 >build.log 2>&1
sha256sum build/sage_fft
"""
            )
            (d / "build.sh").write_text(cmd)
            t = time.perf_counter()
            rc, out, err = self.command(cmd, 1800)
            for f in ("configure.log", "build.log"):
                self.s.get(remote + "/" + f, str(d / f))
            if rc:
                raise RuntimeError((d / "build.log").read_text(errors="replace")[-3000:])
            self.s.get(remote + "/build/sage_fft", str(d / "sage_fft"))
            meta = dict(
                binary_sha256=out.split()[0],
                build_seconds=time.perf_counter() - t,
                build_ok=True,
                source_hashes={
                    f: hashlib.sha256((d / f).read_bytes()).hexdigest()
                    for f in ("main.cpp", "kernels.cpp", "source.cu", "ir.json")
                },
            )
            mp.write_text(json.dumps(meta, indent=2))
        meta = json.loads(mp.read_text())
        sid = "session_" + str(time.time_ns())
        sd = self.root / name / sid
        sd.mkdir()
        sr = REMOTE + "/" + name + "/" + sid
        x, h = inputs(c, seed)
        x.tofile(sd / "x.bin")
        h.tofile(sd / "h.bin")
        self.command("mkdir -p " + shlex.quote(sr))
        for f in ("x.bin", "h.bin"):
            self.s.put(str(sd / f), sr + "/" + f)
        cmd = (
            ENV
            + "cd "
            + shlex.quote(sr)
            + "\nexec ../live_catalog/build/sage_fft . - outputs.bin "
            + str(samples)
        )
        i, o, e = self.c.exec_command("bash -c " + shlex.quote(cmd), timeout=1800)
        first = o.readline()
        if first.strip() != "READY":
            raise RuntimeError("live target failed " + first + e.read().decode())
        ctx = dict(
            stdin=i,
            stdout=o,
            stderr=e,
            remote=sr,
            directory=sd,
            outputs=0,
            meta=meta,
            x=x,
            h=h,
            seed=seed,
            samples=samples,
        )
        self.live[cache] = ctx
        return ctx

    def evaluate_many(self, name, realizations, tag, seed=1, samples=5, mode="pipeline", shuffle_seed=None):
        if len(realizations) != 1 or mode != "pipeline":
            return super().evaluate_many(name, realizations, tag, seed, samples, mode, shuffle_seed)
        d = self.root / name / tag
        rp = d / "results.json"
        if rp.exists():
            return checked_measurements(json.loads(rp.read_text()), tag)
        c, packs, _ = WORKLOADS[name]
        r = realizations[0]
        ss = catalog_steps(c, packs)
        ids = {canonical_step(c, s): i for i, s in enumerate(ss)}
        try:
            candidate_steps = stages(c, r)
        except ValueError as exc:
            raise CandidateLoweringError("candidate plan validation failed: " + str(exc)) from exc
        plan = []
        for step in candidate_steps:
            step_key = canonical_step(c, step)
            try:
                plan.append(ids[step_key])
            except KeyError as exc:
                raise CandidateLoweringError(
                    "candidate stage is absent from the cached kernel catalog: " + step_key
                ) from exc
        ctx = self.start_live(name, seed, samples)
        index = ctx["outputs"]
        pid = "p%06d" % index
        line = "%s %d %d %s\n" % (pid, r["cores"], len(plan), " ".join(map(str, plan)))
        d.mkdir(parents=True, exist_ok=True)
        (d / "plans.txt").write_text(line)
        t = time.perf_counter()
        ctx["stdin"].write(line)
        ctx["stdin"].flush()
        raw = ctx["stdout"].readline()
        wall = time.perf_counter() - t
        if not raw:
            raise RuntimeError("live target exited " + ctx["stderr"].read().decode())
        row = json.loads(raw)
        assert row["id"] == pid
        count = math.prod(c.shape) * c.batch * 8
        with self.s.open(ctx["remote"] + "/outputs.bin", "rb") as f:
            f.seek(index * count)
            data = f.read(count)
        assert len(data) == count
        ctx["outputs"] += 1
        (d / "outputs.bin").write_bytes(data)
        (d / "run.log").write_text(raw)
        y = np.frombuffer(data, np.complex64).reshape(ctx["x"].shape)
        row.update(metrics(y, reference(c, ctx["x"], ctx["h"])))
        row.update(
            workload=name,
            tag=tag,
            index=0,
            input_seed=seed,
            realization=r,
            mode=mode,
            p50_us=float(np.median(row["samples_us"])),
            features=features(c, r),
            catalog_binary_sha256=ctx["meta"]["binary_sha256"],
            target_session=ctx["directory"].name,
            session_output_index=index,
            RPC_seconds=wall,
        )
        (d / "plans.json").write_text(
            json.dumps(
                dict(realization=r, kernel_ids=plan, catalog_binary_sha256=ctx["meta"]["binary_sha256"]),
                indent=2,
            )
        )
        rp.write_text(json.dumps([row], indent=2))
        # A numerical rejection leaves the shared host alive. Its output index
        # has advanced; the next plan starts from immutable inputs again.
        return checked_measurements([row], tag)

    def close_live(self):
        for ctx in self.live.values():
            ctx["stdin"].channel.shutdown_write()
            ctx["stdout"].read()
            err = ctx["stderr"].read()
            rc = ctx["stdout"].channel.recv_exit_status()
            (ctx["directory"] / "session.json").write_text(
                json.dumps(
                    dict(
                        outputs=ctx["outputs"],
                        catalog=ctx["meta"],
                        exit_code=rc,
                        stderr=err.decode(),
                        input_seed=ctx["seed"],
                    ),
                    indent=2,
                )
            )
        self.live = {}
