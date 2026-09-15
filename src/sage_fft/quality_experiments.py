# Author: even
"""Prespecified scale extension and held-out, randomized timing calibration."""

from .config import workspace, remote_root, npu_environment, connect_ssh, cuda_environment
import argparse, csv, hashlib, json, os, random, shlex, time
from pathlib import Path
import numpy as np
from .campaign import Runner, ENV, ROOT, REMOTE
from .fft_ir import Contract, metrics
from .cuda_source import source
from .validate_semantics import inputs, reference

SHAPES = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
CONFIGS = [
    ("staged", dict(group=1, cores=8, fusion=False)),
    ("intra", dict(group=16, cores=8, fusion=False)),
    ("full", dict(group=16, cores=8, fusion=True)),
]


def scaling(r):
    r.journal = ROOT / "evidence/scaling_records.jsonl"
    for shape in SHAPES:
        order = CONFIGS.copy()
        random.Random(20260911 + shape[0]).shuffle(order)
        for label, action in order:
            tag = "scaling_n%d_%s" % (shape[0], label)
            if (r.root / tag).exists():
                prior = [json.loads(s) for s in r.journal.read_text().splitlines()]
                done = [v for v in prior if v["id"] == tag and v.get("pass_correctness")]
                if len(done) == 3:
                    continue
                raise RuntimeError("incomplete existing scaling artifact: " + tag)
            r.evaluate(Contract(shape, 1), action, tag, seeds=(1, 2, 3), phase="scaling")


def calibration(r):
    base = ROOT / "experiments/calibration"
    base.mkdir(exist_ok=True)
    journal = ROOT / "evidence/calibration_records.jsonl"
    if journal.exists():
        raise RuntimeError("calibration already started; preserve the original blocks")
    records = [json.loads(s) for s in (ROOT / "evidence/target_records.jsonl").read_text().splitlines()]
    byid = {v["id"]: v for v in records}
    with (ROOT / "evidence/policy_best_by_budget.csv").open() as f:
        winners = [v for v in csv.DictReader(f) if v["budget"] == "8"]
    assert len(winners) == 12
    c = Contract((8, 8, 8), 1)
    x, h = inputs(c, 4)
    ip = base / "input_4"
    ip.mkdir(exist_ok=True)
    x.tofile(ip / "x.bin")
    h.tofile(ip / "h.bin")
    remote = REMOTE + "/quality_calibration"
    r.command("mkdir -p " + shlex.quote(remote + "/input_4"))
    for name in ("x.bin", "h.bin"):
        r.s.put(str(ip / name), remote + "/input_4/" + name)
    for block in range(5):
        order = winners.copy()
        random.Random(7300 + block).shuffle(order)
        for position, w in enumerate(order):
            original = byid[w["winner"]]
            tag = "b%d_%s_s%s" % (block, w["policy"], w["seed"])
            outpath = remote + "/" + tag + ".bin"
            binary = REMOTE + "/" + w["winner"] + "/build/sage_fft"
            command = (
                ENV
                + "sha256sum "
                + shlex.quote(binary)
                + "\n"
                + shlex.quote(binary)
                + " "
                + shlex.quote(remote + "/input_4")
                + " "
                + shlex.quote(outpath)
            )
            rc, out, err = r.command(command, 120)
            (base / (tag + ".log")).write_text(out + err, encoding="utf-8")
            assert rc == 0, (tag, err)
            assert out.splitlines()[0].split()[0] == original["binary_sha256"]
            r.s.get(outpath, str(base / (tag + ".bin")))
            y = np.fromfile(base / (tag + ".bin"), np.complex64).reshape(x.shape)
            result = metrics(y, reference(c, x, h))
            assert result["pass_correctness"]
            timing = json.loads(next(v for v in out.splitlines() if v.startswith("{")))
            row = dict(
                block=block,
                position=position,
                policy=w["policy"],
                search_seed=int(w["seed"]),
                input_seed=4,
                source_measurement_id=w["winner"],
                action=original["action"],
                binary_sha256=original["binary_sha256"],
                target_source_sha256=original["target_source_sha256"],
                timestamp=time.time(),
                output_file=tag + ".bin",
                p50_us=float(np.median(timing["samples_us"])),
                **timing,
                **result,
            )
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(json.dumps({"calibration_block_complete": block, "runs": len(order)}), flush=True)


def cuda():
    c = connect_ssh("CUDA")

    def run(cmd):
        _, o, e = c.exec_command("bash -lc " + shlex.quote(cuda_environment() + cmd), timeout=180)
        out = o.read().decode()
        err = e.read().decode()
        return o.channel.recv_exit_status(), out, err

    try:
        rc, out, err = run("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader")
        (ROOT / "evidence/scaling_cuda_environment.txt").write_text(out + err, encoding="utf-8")
        assert rc == 0 and int(out.splitlines()[0].split(",")[2].strip().split()[0]) < 500, (
            "GPU 0 unavailable"
        )
        journal = ROOT / "evidence/scaling_cuda_records.jsonl"
        with c.open_sftp() as s:
            for shape in SHAPES:
                tag = "scaling_cuda_n%d" % shape[0]
                d = ROOT / "experiments/artifacts" / tag
                if d.exists():
                    raise RuntimeError("existing CUDA scale artifact: " + tag)
                d.mkdir()
                remote = remote_root("cuda-scale") + "/" + tag
                src, contract = source(shape, 1)
                (d / "source.cu").write_text(src, encoding="utf-8", newline="\n")
                run("mkdir -p " + shlex.quote(remote))
                s.put(str(d / "source.cu"), remote + "/source.cu")
                rc, out, err = run(
                    "cd " + shlex.quote(remote) + " && nvcc -O3 source.cu -lcufft -o reference"
                )
                (d / "build.log").write_text(out + err)
                assert rc == 0, err
                s.get(remote + "/reference", str(d / "reference"))
                for seed in (1, 2, 3):
                    ip = d / ("input_%d" % seed)
                    ip.mkdir()
                    x, h = inputs(contract, seed)
                    x.tofile(ip / "x.bin")
                    h.tofile(ip / "h.bin")
                    run("mkdir -p " + shlex.quote(remote + "/" + ip.name))
                    for name in ("x.bin", "h.bin"):
                        s.put(str(ip / name), remote + "/" + ip.name + "/" + name)
                    rc, out, err = run(
                        "cd " + shlex.quote(remote) + " && ./reference " + ip.name + " out_%d.bin" % seed
                    )
                    (d / ("run_%d.log" % seed)).write_text(out + err)
                    assert rc == 0, err
                    s.get(remote + "/out_%d.bin" % seed, str(d / ("out_%d.bin" % seed)))
                    y = np.fromfile(d / ("out_%d.bin" % seed), np.complex64).reshape(x.shape)
                    result = metrics(y, reference(contract, x, h))
                    assert result["pass_correctness"]
                    row = dict(
                        id=tag,
                        shape=shape,
                        batch=1,
                        seed=seed,
                        source_sha256=hashlib.sha256(src.encode()).hexdigest(),
                        binary_sha256=hashlib.sha256((d / "reference").read_bytes()).hexdigest(),
                        **json.loads(out),
                        **result,
                    )
                    with journal.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps({"cuda_shape": shape, "seed": seed, "pass": True}), flush=True)
    finally:
        c.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["npu", "cuda"])
    a = p.parse_args()
    if a.phase == "cuda":
        cuda()
    else:
        r = Runner()
        try:
            rc, out, err = r.command('ps -eo comm,args | grep -E "[s]age_fft|[n]pu-smi"')
            (ROOT / "evidence/quality_npu_preflight.txt").write_text(out + err)
            if "build/sage_fft" in out:
                raise RuntimeError("another FFT measurement is running")
            scaling(r)
            calibration(r)
        finally:
            r.s.close()
            r.c.close()
