# Author: even
"""Public command line for inspection, source emission and explicit search.

migrate consumes a registered CUDA source and emits its FFT IR and NPU sources.
search runs offline tuning against the configured target. Inspection, semantic
checks and source emission work locally without device or model services.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .fft_ir import Contract, execute, import_cuda, metrics
from .mask_ir import initial, apply, features, stages
from .validate_semantics import inputs, reference


def _contract(args):
    try:
        c = Contract(tuple(int(n) for n in args.shape.split(",")), args.batch)
        return c.validate()
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "--shape must list one to three power-of-two axes; --batch must be positive"
        ) from exc


def _realization(c, args):
    r = initial(c)
    if args.variant == "grouped":
        r = apply(c, r, dict(level="fft", region="all", op="merge", boundaries="all"))
        r = apply(c, r, dict(level="pipeline", region="both", op="fuse"))
    r.update(cores=args.blocks, pack=args.pack)
    features(c, r)
    return r


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sage", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "validate-cpu", "emit-npu", "emit-cuda", "make-inputs"):
        p = commands.add_parser(command)
        p.add_argument("--shape", default="8,8,8")
        p.add_argument("--batch", type=int, default=1)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--variant", choices=("staged", "grouped"), default="grouped")
        p.add_argument("--blocks", type=int, choices=(1, 4, 8), default=4)
        p.add_argument("--pack", type=int, choices=(4, 8, 16), default=8)
        p.add_argument("--mode", choices=("pipeline", "forward", "inverse"), default="pipeline")
        if command in ("emit-npu", "emit-cuda", "make-inputs"):
            p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("import-cuda", help="validate a registered CUDA source")
    p.add_argument("source", type=Path)
    p = commands.add_parser("emit-catalog", help="emit cached Ascend kernel catalog sources")
    p.add_argument("--workload", choices=("M1", "M2", "M3"), default="M3")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("migrate", help="compile registered CUDA source through the FFT IR")
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=("staged", "grouped"), default="grouped")
    p.add_argument("--blocks", type=int, choices=(1, 4, 8), default=4)
    p.add_argument("--pack", type=int, choices=(4, 8, 16), default=8)
    p = commands.add_parser("search", help="run offline Algorithm 1 on the configured NPU and model")
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--policy", choices=("index", "greedy"), default="index")
    args = parser.parse_args(argv)
    try:
        if args.command in ("migrate", "search"):
            from .migration import compile_cuda, search_cuda

            source = args.source.read_text(encoding="utf-8")
            if args.command == "migrate":
                ir = compile_cuda(
                    source, args.output, variant=args.variant, blocks=args.blocks, pack=args.pack
                )
                print(json.dumps(dict(output=str(args.output), launches=len(ir["stages"]))))
            else:
                result = search_cuda(source, args.output, seed=args.seed, policy=args.policy)
                print(json.dumps(result, indent=2))
            return
        if args.command == "import-cuda":
            print(json.dumps(asdict(import_cuda(args.source.read_text(encoding="utf-8"))), indent=2))
            return
        if args.command == "emit-catalog":
            from .catalog import generate

            ir = generate(args.workload, args.output)
            print(json.dumps(dict(output=str(args.output), kernels=len(ir["stages"]))))
            return
        c = _contract(args)
        if args.command == "emit-cuda":
            from .cuda_source import source

            src, _ = source(c.shape, c.batch)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(src, encoding="utf-8", newline="\n")
            print(json.dumps(dict(output=str(args.output))))
            return
        x, h = inputs(c, args.seed)
        if args.command == "make-inputs":
            args.output.mkdir(parents=True, exist_ok=True)
            x.tofile(args.output / "x.bin")
            h.tofile(args.output / "h.bin")
            (args.output / "contract.json").write_text(json.dumps(asdict(c), indent=2), encoding="utf-8")
            print(json.dumps(dict(output=str(args.output), dtype="complex64", seed=args.seed)))
            return
        r = _realization(c, args)
        if args.command == "inspect":
            print(
                json.dumps(
                    dict(
                        contract=asdict(c),
                        realization=r,
                        features=features(c, r),
                        steps=stages(c, r, args.mode),
                    ),
                    indent=2,
                )
            )
        elif args.command == "validate-cpu":
            actual = execute(c, r, x, h, args.mode)
            result = metrics(actual, reference(c, x, h, args.mode))
            print(json.dumps(dict(backend="CPU semantic interpreter; not NPU timing", **result), indent=2))
            if not result["pass_correctness"]:
                raise SystemExit(1)
        else:
            from .lower import emit

            ir = emit(c, r, args.output, args.mode)
            print(json.dumps(dict(output=str(args.output), launches=len(ir["stages"]))))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
