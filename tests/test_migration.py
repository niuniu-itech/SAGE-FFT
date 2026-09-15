# Author: even
"""Source-to-IR integration and Algorithm 1 with an independent CPU oracle."""

import copy
import json
import re
from unittest.mock import Mock, patch

import numpy as np
import pytest

from sage_fft import fft_ir, lower
from sage_fft.cuda_source import render
from sage_fft.mask_ir import WORKLOADS, initial, stages
from sage_fft.migration import compile_cuda, search_cuda
from sage_fft.search_errors import CandidateLoweringError
from sage_fft.structured_search import run_search


def test_source_contract_drives_ir_and_emitted_fft(tmp_path):
    source = render((4, 8), 1)
    with patch.object(lower, "identity", wraps=fft_ir.identity) as construct:
        baseline = compile_cuda(source, tmp_path / "staged", variant="staged", blocks=8)
        grouped = compile_cuda(source, tmp_path / "grouped", blocks=8)
    assert construct.call_count == 2
    assert baseline["contract"]["shape"] == grouped["contract"]["shape"] == (4, 8)
    assert [len(ir["stages"]) for ir in (baseline, grouped)] == [12, 4]
    for name, ir in (("staged", baseline), ("grouped", grouped)):
        folder = tmp_path / name
        saved = json.loads((folder / "ir.json").read_text())
        assert saved["stages"] == ir["stages"]
        code = (folder / "kernels.cpp").read_text()
        host = (folder / "main.cpp").read_text()
        definitions = re.findall(r"void (fft_step_\d+)\(GM_ADDR", code)
        launches = re.findall(r"ACLRT_LAUNCH_KERNEL\((fft_step_\d+)\)", host)
        assert definitions == launches and len(definitions) == len(ir["stages"])
        assert (folder / "source.cu").read_text() == source
    assert (tmp_path / "staged/kernels.cpp").read_bytes() != (tmp_path / "grouped/kernels.cpp").read_bytes()


class IndexController:
    def call(self, request):
        # Exercise the real decoder and legal action menu; do not supply code.
        return dict(
            request=request,
            response=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({"diagnosis": "test only", "decision": {"id": 0}})
                            }
                        }
                    ]
                }
            ),
            exit_code=0,
            model_seconds=0.0,
        )


class SemanticTarget:
    """CPU semantic checks with synthetic ranking values, never NPU timings."""

    def __init__(self):
        self.calls = []
        self.correct = []

    def evaluate_many(self, name, candidates, tag, **kwargs):
        c = WORKLOADS[name][0]
        r = copy.deepcopy(candidates[0])
        self.calls.append(r)
        number = len(self.calls)
        if number == 3:
            raise CandidateLoweringError("injected candidate mapping rejection")
        rng = np.random.default_rng(91)
        shape = (c.batch,) + c.shape
        x = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
        h = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
        axes = tuple(range(1, len(shape)))
        reference = np.fft.ifftn(np.fft.fftn(x.astype(np.complex128), axes=axes) * h, axes=axes)
        actual = fft_ir.execute(c, r, x, h)
        if number == 4:
            actual = actual + 10  # A nominally fast but incorrect result cannot win.
        error = fft_ir.metrics(actual, reference)
        rank = 1.0 if number == 4 else 50.0 if number == 2 else 100.0 + number
        row = dict(realization=r, p50_us=rank, tag=tag, **error)
        if row["pass_correctness"]:
            self.correct.append(row)
        return [row]


def test_algorithm_returns_best_verified_plan_and_exports_ir(tmp_path):
    target = SemanticTarget()
    result = search_cuda(render((8, 8), 1), tmp_path / "search", target=target, controller=IndexController())
    expected = min(target.correct, key=lambda row: row["p50_us"])["realization"]
    assert result["realization"] == expected
    assert len(target.calls) == 13
    trace = tmp_path / "search/M2/search/index/41"
    assert len(list(trace.glob("step_[0-9][0-9].json"))) == 12
    rejected = [json.loads(path.read_text()) for path in trace.glob("step_[0-9][0-9].json")]
    assert {row.get("failure_stage") for row in rejected if not row["accepted"]} == {
        "lowering",
        "correctness",
    }
    assert all(row["incumbent_us"] >= 50 for row in rejected)
    plan = json.loads((tmp_path / "search/launch_plan.json").read_text())
    ir = json.loads((tmp_path / "search/deployment/ir.json").read_text())
    assert plan["realization"] == ir["action"] == expected
    assert plan["stages"] == ir["stages"] == stages(WORKLOADS["M2"][0], expected)
    assert len(plan["kernel_ids"]) == result["launches"]


def test_resume_rejects_a_falsely_accepted_incorrect_record(tmp_path):
    c = WORKLOADS["M2"][0]
    r = initial(c)
    path = tmp_path / "M2/search/flat_llm/41"
    path.mkdir(parents=True)
    (path / "step_01.json").write_text(
        json.dumps(dict(accepted=True, realization=r, feedback=dict(correctness=False)))
    )
    target = Mock()
    target.evaluate_many.return_value = [dict(realization=r, p50_us=100.0, pass_correctness=True, max_abs=0)]
    controller = Mock()
    with pytest.raises(ValueError, match="invalid correctness"):
        run_search(target, controller, "M2", "flat_llm", 41, output_root=tmp_path)
    controller.call.assert_not_called()


def test_unsupported_source_and_existing_runs_rejected_before_target_access(tmp_path):
    target, controller = Mock(), Mock()
    with pytest.raises(ValueError, match="registered workloads"):
        search_cuda(render((16, 16), 1), tmp_path / "unsupported", target=target, controller=controller)
    (tmp_path / "retained.json").write_text("original")
    with pytest.raises(ValueError, match="new or empty"):
        search_cuda(render((8, 8), 1), tmp_path, target=target, controller=controller)
    target.evaluate_many.assert_not_called()
    controller.call.assert_not_called()
    assert (tmp_path / "retained.json").read_text() == "original"
