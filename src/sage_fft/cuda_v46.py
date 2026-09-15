# Author: even
"""Fresh CUDA reference executions for the three new workload contracts."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace, cuda_environment
import hashlib, json, os, shlex, time
from pathlib import Path
import numpy as np
from .mask_ir import WORKLOADS
from .cuda_source import source
from .validate_semantics import inputs, reference
from .fft_ir import metrics

ROOT = workspace()
REMOTE = remote_root("cuda")


def main():
    prepare_workspace()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", default="M1,M2,M3")
    args = parser.parse_args()
    names = args.workloads.split(",")
    if "M4" in names:
        from .fft_ir import Contract

        WORKLOADS["M4"] = (Contract((8, 16), 1), (4, 8, 16), False)
    s = connect_ssh("CUDA")
    sf = s.open_sftp()

    def cmd(text):
        _, o, e = s.exec_command("bash -s", timeout=180)
        o.channel.sendall((cuda_environment() + text).encode())
        o.channel.shutdown_write()
        out = o.read().decode()
        err = e.read().decode()
        return o.channel.recv_exit_status(), out, err

    d = ROOT / "experiments" / "v46" / "cuda"
    d.mkdir(parents=True, exist_ok=True)
    try:
        rc, out, err = cmd(
            "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader\nnvcc --version"
        )
        (d / "environment.txt").write_text(out + err)
        used = int(out.splitlines()[0].split(",")[1].split()[0])
        assert used < 500, "GPU busy"
        old = d / "records.json"
        rows = [r for r in json.loads(old.read_text()) if r["workload"] not in names] if old.exists() else []
        for name, (c, packs, paired) in WORKLOADS.items():
            if name not in names:
                continue
            wd = d / name
            wd.mkdir(parents=True, exist_ok=True)
            rr = REMOTE + "/" + name
            cmd("mkdir -p " + shlex.quote(rr))
            src, cc = source(c.shape, c.batch)
            assert c == cc
            (wd / "source.cu").write_text(src, encoding="utf-8", newline="\n")
            sf.put(str(wd / "source.cu"), rr + "/source.cu")
            rc, out, err = cmd("cd " + shlex.quote(rr) + "\nnvcc -O3 source.cu -lcufft -o reference\n")
            (wd / "build.log").write_text(out + err)
            assert rc == 0, err
            sf.get(rr + "/reference", str(wd / "reference"))
            for seed in (1, 2, 3):
                ip = wd / ("input_%d" % seed)
                ip.mkdir(exist_ok=True)
                x, h = inputs(c, seed)
                x.tofile(ip / "x.bin")
                h.tofile(ip / "h.bin")
                for f in ("x.bin", "h.bin"):
                    sf.put(str(ip / f), rr + "/" + f)
                rc, out, err = cmd("cd " + shlex.quote(rr) + "\n./reference . out.bin")
                (wd / ("run_%d.log" % seed)).write_text(out + err)
                assert rc == 0, err
                sf.get(rr + "/out.bin", str(wd / ("out_%d.bin" % seed)))
                y = np.fromfile(wd / ("out_%d.bin" % seed), np.complex64).reshape(x.shape)
                row = dict(
                    workload=name,
                    seed=seed,
                    source_sha256=hashlib.sha256((wd / "source.cu").read_bytes()).hexdigest(),
                    binary_sha256=hashlib.sha256((wd / "reference").read_bytes()).hexdigest(),
                    **metrics(y, reference(c, x, h)),
                    **json.loads(out),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
        (d / "records.json").write_text(json.dumps(rows, indent=2))
        assert all(r["pass_correctness"] for r in rows)
    finally:
        sf.close()
        s.close()


if __name__ == "__main__":
    main()
