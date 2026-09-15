# Author: even
"""Run the saved cuFFT source references when the authorized CUDA host is reachable."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace, cuda_environment
import json, os, shlex, time
from pathlib import Path
import numpy as np
from .cuda_source import source
from .validate_semantics import WORKLOADS, inputs, reference
from .fft_ir import metrics

ROOT = workspace()
REMOTE = remote_root("cuda")


def main():
    prepare_workspace()
    c = connect_ssh("CUDA")

    def run(cmd):
        _, o, e = c.exec_command("bash -lc " + shlex.quote(cuda_environment() + cmd), timeout=120)
        out = o.read().decode(errors="replace")
        err = e.read().decode(errors="replace")
        return o.channel.recv_exit_status(), out, err

    outdir = ROOT / "experiments" / "cuda_artifacts"
    outdir.mkdir(exist_ok=True)
    try:
        rc, out, err = run("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader")
        (outdir / "environment.txt").write_text(out + err, encoding="utf-8")
        if rc:
            raise RuntimeError("GPU preflight failed")
        # Do not start on a busy default GPU.
        first = out.splitlines()[0].split(",")
        if int(first[2].strip().split()[0]) >= 500:
            raise RuntimeError("GPU 0 is busy; choose an idle GPU before running")
        run("mkdir -p " + shlex.quote(REMOTE))
        with c.open_sftp() as s:
            for wi, (shape, batch) in enumerate(WORKLOADS):
                d = outdir / ("w%d" % wi)
                d.mkdir(exist_ok=True)
                remote = REMOTE + "/w%d" % wi
                run("mkdir -p " + shlex.quote(remote))
                src, contract = source(shape, batch)
                (d / "reference.cu").write_text(src, encoding="utf-8")
                s.put(str(d / "reference.cu"), remote + "/reference.cu")
                rc, out, err = run(
                    "cd " + shlex.quote(remote) + " && nvcc -O3 reference.cu -lcufft -o reference"
                )
                (d / "build.log").write_text(out + err, encoding="utf-8")
                if rc:
                    raise RuntimeError("cuFFT build failed: " + str(d))
                for seed in (1, 2, 3):
                    x, h = inputs(contract, seed)
                    x.tofile(d / "x.bin")
                    h.tofile(d / "h.bin")
                    for name in ("x.bin", "h.bin"):
                        s.put(str(d / name), remote + "/" + name)
                    rc, out, err = run("cd " + shlex.quote(remote) + " && ./reference . out.bin")
                    (d / ("run_%d.log" % seed)).write_text(out + err, encoding="utf-8")
                    if rc:
                        raise RuntimeError("CUDA execution failed")
                    s.get(remote + "/out.bin", str(d / ("out_%d.bin" % seed)))
                    y = np.fromfile(d / ("out_%d.bin" % seed), np.complex64).reshape(x.shape)
                    row = dict(
                        shape=shape,
                        batch=batch,
                        seed=seed,
                        backend="CUDA cuFFT",
                        timestamp=time.time(),
                        **metrics(y, reference(contract, x, h)),
                        **json.loads(out),
                    )
                    # Compare the identical source input with the retained full target realization.
                    npu = (
                        ROOT / "experiments" / "artifacts" / ("ablation_w%d_a9" % wi) / ("out_%d.bin" % seed)
                    )
                    if npu.exists():
                        row["paired_npu_vs_cuda"] = metrics(
                            np.fromfile(npu, np.complex64).reshape(x.shape), y.astype(np.complex128)
                        )
                    with (ROOT / "evidence" / "cuda_records.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
    finally:
        c.close()


if __name__ == "__main__":
    main()
