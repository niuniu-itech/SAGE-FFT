# Author: even
"""Source-driven entry points for FFT migration and offline exploration.

compile_cuda connects the registered source adapter to IR-based source emission.
search_cuda runs the indexed or analytic policy and exports its best verified
catalog plan with matching static sources. Source emission alone is CPU-only;
default search execution requires the explicitly configured NPU and model.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .fft_ir import import_cuda
from .mask_ir import WORKLOADS, initial, apply, validate, stages
from .lower import emit


def compile_cuda(source, output, *, realization=None, variant="grouped", blocks=4, pack=8):
    """Compile the contract extracted from source, without a device or model call.

    emit constructs the stage IR in memory and consumes those same stages when
    emitting butterfly code. ir.json is its serialized audit artifact.
    """
    contract = import_cuda(source)
    if realization is None:
        if variant not in ("staged", "grouped"):
            raise ValueError("variant must be staged or grouped")
        realization = initial(contract)
        if variant == "grouped":
            realization = apply(
                contract, realization, dict(level="fft", region="all", op="merge", boundaries="all")
            )
            realization = apply(contract, realization, dict(level="pipeline", region="both", op="fuse"))
        realization.update(cores=blocks, pack=pack)
    validate(contract, realization)
    output = Path(output)
    ir = emit(contract, realization, output)
    (output / "source.cu").write_text(source, encoding="utf-8", newline="\n")
    return ir


def search_cuda(source, output, *, seed=41, policy="index", target=None, controller=None):
    """Execute Algorithm 1 for a registered catalog workload.

    The default target uses CANN over SSH; index selection uses the configured
    model with structured diagnosis. Injected targets/controllers are useful for
    integration tests, whose timing must never be labelled hardware evidence.
    A fresh output directory prevents mixing different searches or binaries.
    """
    from .catalog import catalog_steps, canonical_step
    from .structured_search import run_search

    contract = import_cuda(source)
    names = [name for name, (c, _, _) in WORKLOADS.items() if c == contract]
    if len(names) != 1:
        raise ValueError("online catalog search supports registered workloads M1, M2 and M3 only")
    if policy not in ("index", "greedy"):
        raise ValueError("policy must be index or greedy")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("search output must be a new or empty directory")
    name = names[0]
    owned_target = target is None
    if controller is None and policy == "index":
        from .hosted_schema import SchemaController

        controller = SchemaController()
    if owned_target:
        from .live_target import LiveRunner

        target = LiveRunner()
    try:
        output.mkdir(parents=True, exist_ok=True)
        if owned_target:
            target.root = output / "target"
            target.root.mkdir()
            (output / "evidence").mkdir()
            target.journal = output / "evidence" / "target_records.jsonl"
        (output / "source.cu").write_text(source, encoding="utf-8", newline="\n")
        best = run_search(
            target,
            controller,
            name,
            "flat_llm" if policy == "index" else "greedy",
            seed,
            label=policy,
            output_root=output,
        )
        validate(contract, best)
        # Static export uses the same validated stage IR as cached execution.
        ir = compile_cuda(source, output / "deployment", realization=best)
        packs = WORKLOADS[name][1]
        entries = catalog_steps(contract, packs)
        ids = {canonical_step(contract, step): i for i, step in enumerate(entries)}
        selected = stages(contract, best)
        plan = dict(
            contract=asdict(contract),
            realization=best,
            kernel_ids=[ids[canonical_step(contract, step)] for step in selected],
            stages=selected,
            blocks=best["cores"],
        )
        (output / "launch_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        result = dict(
            workload=name,
            seed=seed,
            policy=policy,
            budget=12,
            realization=best,
            launches=len(ir["stages"]),
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            verification_scope="Selected catalog realization passed the supplied target correctness gate. "
            "The exported static source requires native compilation and validation before deployment.",
        )
        (output / "best.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        if owned_target:
            try:
                target.close_live()
            finally:
                try:
                    target.s.close()
                finally:
                    target.c.close()
