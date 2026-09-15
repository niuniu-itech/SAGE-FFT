# Author: even
"""Complete finite-space reference, independent of the original search timings."""

from .config import workspace, remote_root, npu_environment, connect_ssh, prepare_workspace
import hashlib, json, random, shlex, time
import numpy as np
from .campaign import Runner, ROOT, REMOTE, ENV
from .fft_ir import Contract, candidates, metrics
from .validate_semantics import inputs, reference


def main():
    prepare_workspace()
    r = Runner()
    base = ROOT / "experiments/oracle"
    base.mkdir(exist_ok=True)
    journal = ROOT / "evidence/oracle_records.jsonl"
    if journal.exists():
        raise RuntimeError("preserve existing oracle experiment")
    original = [json.loads(s) for s in (ROOT / "evidence/target_records.jsonl").read_text().splitlines()]
    selected = {}
    for v in original:
        if v.get("pass_correctness") and v.get("mode") == "pipeline" and v["contract"]["shape"] == [8, 8, 8]:
            selected.setdefault(candidates().index(v["action"]), v)
    assert len(selected) == 18
    c = Contract((8, 8, 8), 1)
    x, h = inputs(c, 4)
    ip = base / "input_4"
    ip.mkdir()
    x.tofile(ip / "x.bin")
    h.tofile(ip / "h.bin")
    remote = REMOTE + "/oracle_calibration"
    r.command("mkdir -p " + shlex.quote(remote + "/input_4"))
    for name in ("x.bin", "h.bin"):
        r.s.put(str(ip / name), remote + "/input_4/" + name)
    try:
        for block in range(3):
            order = list(range(18))
            random.Random(7400 + block).shuffle(order)
            for position, idx in enumerate(order):
                v = selected[idx]
                tag = "b%d_c%d" % (block, idx)
                binary = REMOTE + "/" + v["id"] + "/build/sage_fft"
                cmd = (
                    ENV
                    + "sha256sum "
                    + shlex.quote(binary)
                    + "\n"
                    + shlex.quote(binary)
                    + " "
                    + shlex.quote(remote + "/input_4")
                    + " "
                    + shlex.quote(remote + "/" + tag + ".bin")
                )
                rc, out, err = r.command(cmd, 120)
                (base / (tag + ".log")).write_text(out + err)
                assert rc == 0, err
                assert out.splitlines()[0].split()[0] == v["binary_sha256"]
                r.s.get(remote + "/" + tag + ".bin", str(base / (tag + ".bin")))
                y = np.fromfile(base / (tag + ".bin"), np.complex64).reshape(x.shape)
                m = metrics(y, reference(c, x, h))
                assert m["pass_correctness"]
                timing = json.loads(next(s for s in out.splitlines() if s.startswith("{")))
                row = dict(
                    block=block,
                    position=position,
                    candidate_id=idx,
                    action=v["action"],
                    source_measurement_id=v["id"],
                    binary_sha256=v["binary_sha256"],
                    target_source_sha256=v["target_source_sha256"],
                    input_seed=4,
                    timestamp=time.time(),
                    output_file=tag + ".bin",
                    p50_us=float(np.median(timing["samples_us"])),
                    **timing,
                    **m,
                )
                with journal.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            print(json.dumps({"oracle_block_complete": block, "configs": 18}), flush=True)
    finally:
        r.s.close()
        r.c.close()


if __name__ == "__main__":
    main()
