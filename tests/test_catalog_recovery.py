# Author: even
"""A retained catalog batch must match the next requested measurement exactly."""

import json

import numpy as np
import pytest

from sage_fft.catalog import verified_batch
from sage_fft.fft_ir import Contract
from sage_fft.mask_ir import initial
from sage_fft.validate_semantics import inputs, reference


@pytest.fixture
def batch(tmp_path):
    contract = Contract((32,), 1)
    realization = initial(contract)
    plan = dict(
        input_seed=1,
        order=[0],
        catalog_binary_sha256="synthetic-digest",
        plans=[dict(realization=realization, mode="pipeline")],
    )
    row = dict(
        index=0,
        input_seed=1,
        mode="pipeline",
        realization=realization,
        catalog_binary_sha256="synthetic-digest",
        pass_correctness=True,
        execution_order=0,
        samples_us=[1, 2, 3],
        p50_us=2,
    )
    (tmp_path / "plans.json").write_text(json.dumps(plan))
    (tmp_path / "results.json").write_text(json.dumps([row]))
    x, h = inputs(contract, 1)
    x.tofile(tmp_path / "x.bin")
    h.tofile(tmp_path / "h.bin")
    reference(contract, x, h).astype(np.complex64).tofile(tmp_path / "outputs.bin")
    return dict(
        folder=tmp_path,
        contract=contract,
        realizations=[realization],
        seed=1,
        samples=3,
        mode="pipeline",
        shuffle_seed=None,
        binary_sha256="synthetic-digest",
    )


def test_matching_retained_batch_is_rechecked(batch):
    assert len(verified_batch(**batch)) == 1


@pytest.mark.parametrize("change", ["seed", "samples", "realization", "output", "binary"])
def test_stale_or_changed_batch_is_rejected(batch, change):
    if change in ("seed", "samples"):
        batch[change] += 1
    elif change == "realization":
        batch["realizations"][0]["cores"] = 4
    elif change == "binary":
        batch["binary_sha256"] = "different-digest"
    else:
        np.full((1, 32), np.nan, np.complex64).tofile(batch["folder"] / "outputs.bin")
    with pytest.raises(RuntimeError, match="does not verify"):
        verified_batch(**batch)
