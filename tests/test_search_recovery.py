# Author: even
"""Fault-injection checks for search recovery, without a model or target device.

Run from the repository root: python -m pytest tests/test_search_recovery.py
These tests exercise the real search loop, response decoder, target result reader,
and numerical check. Only candidate menus and external execution are replaced.
All generated records live in TemporaryDirectory instances.
"""

import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

try:
    import paramiko
except ImportError:
    # CPU fault injection also runs without the optional SSH transport package.
    paramiko = SimpleNamespace(SSHException=type("SSHException", (Exception,), {}))

from sage_fft import live_target
from sage_fft import structured_search as search
from sage_fft.search_errors import CandidateCorrectnessError, CandidateLoweringError


def realization(name):
    return {"name": name, "cores": 1}


def measurement(candidate, latency, tag, correct=True):
    return {
        "realization": copy.deepcopy(candidate),
        "p50_us": latency,
        "pass_correctness": correct,
        "max_abs": 0.0 if correct else 1.0,
        "tag": tag,
    }


class FakeController:
    """Return an actual hosted response selecting index zero in each fake menu."""

    def __init__(self, successful_steps):
        self.successful_steps = set(successful_steps)
        self.calls = []

    def call(self, request):
        self.calls.append(copy.deepcopy(request))
        successful = request["step"] in self.successful_steps
        content = json.dumps({"diagnosis": "controlled test", "decision": {"id": 0}})
        return {
            "request": copy.deepcopy(request),
            "response": json.dumps({"choices": [{"message": {"content": content}}]}),
            "exit_code": 0 if successful else 1,
            "model_seconds": 0.01,
        }


class FakeTarget:
    def __init__(self, outcomes, initial_correct=True, initial_exception=None):
        self.outcomes = outcomes
        self.initial_correct = initial_correct
        self.initial_exception = initial_exception
        self.calls = []

    def evaluate_many(self, name, candidates, tag, **kwargs):
        candidate = candidates[0]
        self.calls.append((candidate["name"], tag))
        if tag == "search_initial":
            if self.initial_exception is not None:
                raise self.initial_exception
            return [measurement(candidate, 100.0, tag, self.initial_correct)]
        outcome = self.outcomes[candidate["name"]]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, tuple):
            kind, latency = outcome
            if kind != "correctness":
                raise AssertionError("unknown test outcome")
            failed = measurement(candidate, latency, tag, False)
            raise CandidateCorrectnessError("injected numerical mismatch", failed)
        return [measurement(candidate, outcome, tag)]


class SearchRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.prompts = []
        self.stack.enter_context(patch.object(search, "WORKLOADS", {"TEST": (object(), (), None)}))
        self.stack.enter_context(
            patch.object(search, "initial", side_effect=lambda c: realization("initial"))
        )
        self.stack.enter_context(patch.object(search, "key", side_effect=lambda r: r["name"]))
        self.stack.enter_context(
            patch.object(search, "features", side_effect=lambda c, r: {"candidate": r["name"]})
        )
        self.stack.enter_context(patch.object(search, "make_request", side_effect=self.make_request))

    def make_request(self, name, current, history, policy, seed, step, seen):
        self.prompts.append(
            {
                "step": step,
                "current": copy.deepcopy(current),
                "history": copy.deepcopy(history),
                "seen": set(seen),
            }
        )
        request = {"step": step, "seed": seed, "messages": []}
        return request, [({"test_step": step}, realization("candidate_%02d" % step))]

    def run_search(self, target, controller):
        with redirect_stdout(io.StringIO()):
            return search.run_search(
                target, controller, "TEST", "flat_llm", 41, label="fault_test", output_root=self.root
            )

    def step_path(self, step):
        return self.root / "TEST" / "search" / "fault_test" / "41" / ("step_%02d.json" % step)

    def read_step(self, step):
        return json.loads(self.step_path(step).read_text(encoding="utf-8"))

    def test_candidate_failures_preserve_state_and_failed_fast_result_cannot_win(self):
        target = FakeTarget(
            {
                "candidate_01": CandidateLoweringError("injected missing kernel mapping"),
                "candidate_02": ("correctness", 0.001),
                "candidate_03": 140.0,
                "candidate_04": 60.0,
            }
        )
        controller = FakeController(range(1, 5))
        result = self.run_search(target, controller)

        self.assertEqual(result, realization("candidate_04"))
        self.assertEqual(len(controller.calls), 12)
        self.assertEqual(
            [name for name, _ in target.calls],
            ["initial", "candidate_01", "candidate_02", "candidate_03", "candidate_04"],
        )
        self.assertEqual(len(list(self.step_path(1).parent.glob("step_[0-9][0-9].json"))), 12)
        lower, numerical = self.read_step(1), self.read_step(2)
        self.assertFalse(lower["accepted"])
        self.assertEqual(lower["failure_stage"], "lowering")
        self.assertIn("missing kernel mapping", lower["error"])
        self.assertFalse(numerical["accepted"])
        self.assertEqual(numerical["failure_stage"], "correctness")
        self.assertFalse(numerical["feedback"]["correctness"])
        self.assertEqual(numerical["feedback"]["latency_us"], 0.001)
        for prompt in self.prompts[:3]:
            self.assertEqual(prompt["current"], realization("initial"))
            self.assertEqual([h["realization"]["name"] for h in prompt["history"]], ["initial"])
            self.assertEqual(prompt["seen"], {"initial"})
        self.assertEqual(self.prompts[3]["current"], realization("candidate_03"))
        self.assertEqual([h["latency_us"] for h in self.prompts[3]["history"]], [100.0, 140.0])
        self.assertTrue(self.read_step(3)["accepted"])
        self.assertEqual(self.read_step(3)["incumbent_us"], 100.0)
        self.assertEqual(self.read_step(4)["incumbent_us"], 60.0)
        for prompt in self.prompts[4:]:
            self.assertEqual(prompt["current"], realization("candidate_04"))
            self.assertEqual(
                [h["realization"]["name"] for h in prompt["history"]],
                ["initial", "candidate_03", "candidate_04"],
            )
            self.assertTrue(all(h["correctness"] for h in prompt["history"]))
            self.assertEqual(prompt["seen"], {"initial", "candidate_03", "candidate_04"})
        self.prompts.clear()
        resumed_target, resumed_controller = FakeTarget({}), FakeController([])
        self.assertEqual(self.run_search(resumed_target, resumed_controller), result)
        self.assertEqual(resumed_target.calls, [("initial", "search_initial")])
        self.assertEqual(resumed_controller.calls, [])
        self.assertEqual(self.prompts, [])

    def test_infrastructure_errors_propagate_without_becoming_rejected_steps(self):
        errors = [
            RuntimeError("injected shared target process failure"),
            OSError("injected connection failure"),
            paramiko.SSHException("injected SSH failure"),
        ]
        original_root = self.root
        for i, error in enumerate(errors):
            with self.subTest(error=type(error).__name__):
                self.root = original_root / ("fatal_%d" % i)
                target = FakeTarget({"candidate_01": error})
                controller = FakeController(range(1, 13))
                with self.assertRaises(type(error)) as caught:
                    self.run_search(target, controller)
                self.assertIs(caught.exception, error)
                self.assertEqual(len(controller.calls), 1)
                self.assertFalse(self.step_path(1).exists())

    def test_incorrect_initial_row_is_a_fatal_precondition(self):
        controller = FakeController(range(1, 13))
        with self.assertRaises(CandidateCorrectnessError):
            self.run_search(FakeTarget({}, initial_correct=False), controller)
        self.assertEqual(controller.calls, [])
        self.assertEqual(self.prompts, [])
        self.assertFalse(self.step_path(1).exists())

    def test_initial_candidate_exception_is_not_consumed_by_proposal_recovery(self):
        error = CandidateLoweringError("initial lowering failed")
        controller = FakeController(range(1, 13))
        with self.assertRaises(CandidateLoweringError) as caught:
            self.run_search(FakeTarget({}, initial_exception=error), controller)
        self.assertIs(caught.exception, error)
        self.assertEqual(controller.calls, [])

    def test_all_success_preserves_current_trajectory_and_returns_minimum(self):
        latencies = [110.0, 90.0, 120.0, 70.0, 80.0, 60.0, 65.0, 66.0, 67.0, 68.0, 69.0, 150.0]
        target = FakeTarget({"candidate_%02d" % i: v for i, v in enumerate(latencies, 1)})
        result = self.run_search(target, FakeController(range(1, 13)))
        self.assertEqual(result, realization("candidate_06"))
        for i, prompt in enumerate(self.prompts, 1):
            prior = "initial" if i == 1 else "candidate_%02d" % (i - 1)
            self.assertEqual(prompt["current"], realization(prior))
            self.assertEqual(len(prompt["history"]), i)
            row = self.read_step(i)
            self.assertTrue(row["accepted"])
            self.assertNotIn("failure_stage", row)
            self.assertEqual(row["incumbent_us"], min([100.0] + latencies[:i]))
        self.assertEqual(self.read_step(12)["realization"], realization("candidate_12"))

    def test_no_improvement_returns_initial_even_when_current_changes(self):
        result = self.run_search(FakeTarget({"candidate_01": 150.0}), FakeController([1]))
        self.assertEqual(result, realization("initial"))
        self.assertTrue(self.read_step(1)["accepted"])
        self.assertEqual(self.prompts[-1]["current"], realization("candidate_01"))

    def test_resume_restores_successful_current_history_and_prior_best(self):
        first = {
            "candidate_01": 90.0,
            "candidate_02": 130.0,
            "candidate_03": 70.0,
            "candidate_04": 110.0,
            "candidate_05": RuntimeError("interrupted before result"),
        }
        with self.assertRaises(RuntimeError):
            self.run_search(FakeTarget(first), FakeController(range(1, 13)))
        saved = [self.step_path(i).read_bytes() for i in range(1, 5)]
        self.prompts.clear()
        target = FakeTarget({"candidate_%02d" % i: 150.0 + i for i in range(5, 13)})
        controller = FakeController(range(1, 13))
        result = self.run_search(target, controller)

        self.assertEqual(result, realization("candidate_03"))
        self.assertEqual(
            [name for name, _ in target.calls], ["initial"] + ["candidate_%02d" % i for i in range(5, 13)]
        )
        self.assertEqual([r["step"] for r in controller.calls], list(range(6, 13)))
        resumed = self.prompts[0]
        self.assertEqual(resumed["step"], 5)
        self.assertEqual(resumed["current"], realization("candidate_04"))
        self.assertEqual([h["latency_us"] for h in resumed["history"]], [100.0, 90.0, 130.0, 70.0, 110.0])
        self.assertEqual(
            resumed["seen"], {"initial", "candidate_01", "candidate_02", "candidate_03", "candidate_04"}
        )
        self.assertEqual(saved, [self.step_path(i).read_bytes() for i in range(1, 5)])
        self.assertEqual(self.read_step(12)["incumbent_us"], 70.0)


class LiveTargetFailureTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        # Bypass constructors, which otherwise establish SSH connections.
        self.runner = live_target.LiveRunner.__new__(live_target.LiveRunner)
        self.runner.root = self.root
        self.runner.live = {}
        self.contract = SimpleNamespace(shape=(4,), batch=1)
        self.stack.enter_context(patch.object(live_target, "WORKLOADS", {"TEST": (self.contract, (), None)}))
        self.stack.enter_context(patch.object(live_target, "catalog_steps", return_value=["known"]))
        self.stack.enter_context(patch.object(live_target, "canonical_step", side_effect=lambda c, s: s))
        self.stages = self.stack.enter_context(patch.object(live_target, "stages", return_value=["known"]))
        self.stack.enter_context(patch.object(live_target, "features", return_value={}))
        self.stack.enter_context(
            patch.object(live_target, "reference", return_value=np.zeros(4, np.complex64))
        )

    def test_fresh_failure_then_success_advances_output_and_cached_failure_stays_rejected(self):
        failed_data = np.ones(4, dtype=np.complex64).tobytes()
        successful_data = np.zeros(4, dtype=np.complex64).tobytes()
        data = failed_data + successful_data
        rows = [
            {"id": "p000000", "samples_us": [0.001, 0.002, 0.003]},
            {"id": "p000001", "samples_us": [40.0, 50.0, 60.0]},
        ]
        context = {
            "stdin": io.StringIO(),
            "stdout": io.StringIO("".join(json.dumps(row) + "\n" for row in rows)),
            "stderr": io.BytesIO(),
            "remote": "mock_remote",
            "directory": self.root / "session_mock",
            "outputs": 0,
            "meta": {"binary_sha256": "0" * 64},
            "x": np.zeros(4, np.complex64),
            "h": np.zeros(4, np.complex64),
        }
        self.runner.s = SimpleNamespace(open=Mock(side_effect=lambda *args: io.BytesIO(data)))
        self.runner.start_live = Mock(return_value=context)
        failed = realization("failed")
        with self.assertRaises(CandidateCorrectnessError) as first:
            self.runner.evaluate_many("TEST", [failed], "failed_case")
        self.assertEqual(first.exception.phase, "correctness")
        self.assertFalse(first.exception.measurement["pass_correctness"])
        self.assertEqual(context["outputs"], 1)
        result = self.runner.evaluate_many("TEST", [realization("successful")], "successful_case")
        self.assertTrue(result[0]["pass_correctness"])
        self.assertEqual(result[0]["session_output_index"], 1)
        self.assertEqual(result[0]["p50_us"], 50.0)
        self.assertEqual(context["outputs"], 2)
        self.assertEqual((self.root / "TEST" / "failed_case" / "outputs.bin").read_bytes(), failed_data)
        self.assertEqual(
            (self.root / "TEST" / "successful_case" / "outputs.bin").read_bytes(), successful_data
        )
        for _ in range(2):
            with self.assertRaises(CandidateCorrectnessError) as cached:
                self.runner.evaluate_many("TEST", [failed], "failed_case")
            self.assertEqual(cached.exception.measurement, first.exception.measurement)
        self.assertEqual(self.runner.start_live.call_count, 2)
        self.assertEqual(self.runner.s.open.call_count, 2)
        self.assertEqual(context["outputs"], 2)

    def test_missing_kernel_mapping_is_a_candidate_lowering_rejection(self):
        self.stages.return_value = ["missing"]
        self.runner.start_live = Mock(side_effect=AssertionError("target must not start"))
        with self.assertRaises(CandidateLoweringError) as caught:
            self.runner.evaluate_many("TEST", [realization("bad_mapping")], "bad_mapping")
        self.assertEqual(caught.exception.phase, "lowering")
        self.runner.start_live.assert_not_called()

    def test_invalid_stages_are_a_candidate_lowering_rejection(self):
        self.stages.side_effect = ValueError("injected invalid stage ownership")
        self.runner.start_live = Mock(side_effect=AssertionError("target must not start"))
        with self.assertRaises(CandidateLoweringError):
            self.runner.evaluate_many("TEST", [realization("bad_stages")], "bad_stages")
        self.runner.start_live.assert_not_called()

    def test_shared_build_and_connection_errors_propagate(self):
        errors = [
            RuntimeError("shared catalog build failed"),
            OSError("SSH connection failed"),
            paramiko.SSHException("SSH transport failed"),
        ]
        for i, error in enumerate(errors):
            with self.subTest(error=type(error).__name__):
                self.runner.start_live = Mock(side_effect=error)
                with self.assertRaises(type(error)) as caught:
                    self.runner.evaluate_many("TEST", [realization("valid")], "fatal_%d" % i)
                self.assertIs(caught.exception, error)

    def test_target_eof_is_fatal_and_has_no_candidate_result(self):
        context = {
            "outputs": 0,
            "stdin": io.StringIO(),
            "stdout": io.StringIO(),
            "stderr": io.BytesIO(b"injected process exit 7"),
        }
        self.runner.start_live = Mock(return_value=context)
        self.runner.s = Mock()
        with self.assertRaisesRegex(RuntimeError, "live target exited.*process exit 7"):
            self.runner.evaluate_many("TEST", [realization("valid")], "target_exit")
        self.assertEqual(context["outputs"], 0)
        self.runner.s.open.assert_not_called()
        self.assertFalse((self.root / "TEST" / "target_exit" / "results.json").exists())


class SyntheticReplayTests(unittest.TestCase):
    def test_nine_synthetic_runs_restore_incumbents_without_external_calls(self):
        import random
        from sage_fft.mask_ir import WORKLOADS, initial, random_realization

        contract = WORKLOADS["M3"][0]
        initial_row = measurement(initial(contract), 1000.0, "synthetic_initial")
        checked_runs, checked_steps = 0, 0
        with tempfile.TemporaryDirectory() as directory:
            replay_root = Path(directory)
            for policy in ("greedy", "flat_llm", "sage"):
                for seed in (101, 102, 103):
                    with self.subTest(policy=policy, seed=seed):
                        destination = replay_root / "M3" / "search" / policy / str(seed)
                        destination.mkdir(parents=True)
                        rng = random.Random(seed)
                        expected = initial_row["realization"]
                        best = initial_row["p50_us"]
                        saved = {}
                        for step in range(1, 13):
                            accepted = step % 4 != 0
                            candidate = random_realization("M3", rng)
                            # Deliberately non-monotonic values test best vs. last state.
                            latency = float(200 + (step * 37 + seed) % 130)
                            row = dict(
                                accepted=accepted,
                                realization=candidate,
                                rng_state=rng.getstate(),
                                synthetic=True,
                            )
                            if accepted:
                                row["feedback"] = search.feedback(
                                    contract, measurement(candidate, latency, "synthetic")
                                )
                                if latency < best:
                                    best, expected = latency, candidate
                            else:
                                row["error"] = "synthetic rejected decision"
                            path = destination / ("step_%02d.json" % step)
                            saved[path] = json.dumps(row).encode()
                            path.write_bytes(saved[path])
                        target = Mock()
                        target.evaluate_many.return_value = [copy.deepcopy(initial_row)]
                        controller = Mock()
                        controller.call.side_effect = AssertionError("replay must not invoke a model")
                        with redirect_stdout(io.StringIO()):
                            restored = search.run_search(
                                target, controller, "M3", policy, seed, output_root=replay_root
                            )
                        self.assertEqual(restored, expected)
                        controller.call.assert_not_called()
                        target.evaluate_many.assert_called_once()
                        self.assertEqual(set(destination.iterdir()), set(saved))
                        for path, original in saved.items():
                            self.assertEqual(path.read_bytes(), original)
                        checked_runs += 1
                        checked_steps += len(saved)
        self.assertEqual((checked_runs, checked_steps), (9, 108))


if __name__ == "__main__":
    unittest.main()
