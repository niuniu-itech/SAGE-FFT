# Author: even
from .config import workspace, remote_root, connect_ssh, prepare_workspace, cuda_environment
import shlex
import os, json, hashlib, sys
from pathlib import Path
from .validate_semantics import WORKLOADS, inputs
from .fft_ir import Contract


def main():
    prepare_workspace()
    p = workspace()
    c = connect_ssh("CUDA")
    _, o, e = c.exec_command(
        "bash -lc "
        + shlex.quote(
            cuda_environment()
            + "nvcc --version\nnvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
        )
    )
    out = o.read().decode() + e.read().decode()
    (p / "evidence/cuda_environment.txt").write_text(out, encoding="utf-8")
    print(out)
    manifest = []
    with c.open_sftp() as s:
        for wi, (shape, batch) in enumerate(WORKLOADS):
            d = p / "experiments/cuda_artifacts" / ("w%d" % wi)
            d.mkdir(parents=True, exist_ok=True)
            s.get(remote_root("cuda") + "/w%d/reference" % wi, str(d / "reference"))
            for seed in (1, 2, 3):
                x, h = inputs(Contract(shape, batch), seed)
                ip = d / ("input_%d" % seed)
                ip.mkdir(exist_ok=True)
                x.tofile(ip / "x.bin")
                h.tofile(ip / "h.bin")
                npu = p / "experiments/artifacts" / ("ablation_w%d_a0" % wi) / ("input_%d" % seed)
                assert (ip / "x.bin").read_bytes() == (npu / "x.bin").read_bytes()
                assert (ip / "h.bin").read_bytes() == (npu / "h.bin").read_bytes()
                manifest.append(
                    dict(
                        workload=wi,
                        seed=seed,
                        x_sha256=hashlib.sha256((ip / "x.bin").read_bytes()).hexdigest(),
                        h_sha256=hashlib.sha256((ip / "h.bin").read_bytes()).hexdigest(),
                        input_provenance="re-materialized from the unchanged deterministic generator and checked against retained NPU input bytes",
                    )
                )
    (p / "evidence/cuda_input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    c.close()


if __name__ == "__main__":
    main()
