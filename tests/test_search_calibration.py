# Author: even
"""The final hosted study calibrates the catalog it actually compiled."""

import json
from unittest.mock import Mock

from benchmarks import calibrate_search
from sage_fft import calibrate_hosted


def test_final_study_uses_shared_catalog_location(tmp_path, monkeypatch):
    monkeypatch.setenv("SAGE_REMOTE_ROOT", "/tmp/sage-calibration-test")
    monkeypatch.setattr(calibrate_search, "prepare_workspace", lambda: tmp_path)
    root = tmp_path / "experiments/hosted_v489"
    step = root / "M3/search/schema_flat/41/step_01.proposal.json"
    step.parent.mkdir(parents=True)
    step.write_text(json.dumps(dict(request=dict(model="test-model"))))
    (root / "acceptance.json").write_text(
        json.dumps(dict(completed=True, kind="hosted", search_runs=9, proposals_per_run=12))
    )
    local = tmp_path / "experiments/v46/M3/heldout_candidates.json"
    local.parent.mkdir(parents=True)
    local.write_text("[]")
    run = Mock()
    monkeypatch.setattr(calibrate_hosted, "main", run)
    calibrate_search.calibrate()
    assert run.call_args.kwargs["catalog_remote"] == "/tmp/sage-calibration-test/catalog"
    assert run.call_args.kwargs["expected_records"] == 108
