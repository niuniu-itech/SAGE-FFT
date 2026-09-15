# Author: even
"""CPU numerical references and the restricted CUDA source boundary."""

import copy

import numpy as np
import pytest

from sage_fft import Contract, execute, import_cuda
from sage_fft.cuda_source import render, source
from sage_fft.fft_ir import metrics, stages, validate_action
from sage_fft.mask_ir import apply, initial


def _input(contract, kind):
    rng = np.random.default_rng(732)
    shape = (contract.batch, *contract.shape)
    x = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    if kind == "impulse":
        x.fill(0)
        x[(slice(None), *(n // 2 for n in contract.shape))] = 1 + 0.5j
    elif kind == "tone":
        coordinates = np.indices(contract.shape)
        phase = sum((axis + 1) * coordinates[axis] / n for axis, n in enumerate(contract.shape))
        x[:] = np.exp(2j * np.pi * phase)
    h = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    return x, h


def _realization(contract, variant):
    realization = initial(contract)
    if variant in ("grouped", "fused"):
        realization = apply(
            contract, realization, dict(level="fft", region="all", op="merge", boundaries="all")
        )
    if variant == "fused":
        realization = apply(contract, realization, dict(level="pipeline", region="both", op="fuse"))
    if variant == "asymmetric":
        realization = apply(
            contract, realization, dict(level="fft", region="forward:0", op="merge", boundaries="all")
        )
        realization = apply(contract, realization, dict(level="pipeline", region="multiply", op="fuse"))
    return realization


@pytest.mark.parametrize("shape,batch", [((8,), 4), ((4, 8), 1), ((8, 8, 8), 1)])
@pytest.mark.parametrize("kind", ["random", "impulse", "tone"])
@pytest.mark.parametrize("mode", ["forward", "inverse", "pipeline"])
@pytest.mark.parametrize("variant", ["staged", "grouped", "fused", "asymmetric"])
def test_cpu_realizations_match_independent_numpy_fft(shape, batch, kind, mode, variant):
    contract = Contract(shape, batch)
    realization = _realization(contract, variant)
    x, h = _input(contract, kind)
    original_x, original_h = x.copy(), h.copy()
    original_realization = copy.deepcopy(realization)
    axes = tuple(range(1, len(shape) + 1))
    if mode == "forward":
        reference = np.fft.fftn(x.astype(np.complex128), axes=axes)
    elif mode == "inverse":
        reference = np.fft.ifftn(x.astype(np.complex128), axes=axes)
    else:
        reference = np.fft.ifftn(np.fft.fftn(x.astype(np.complex128), axes=axes) * h, axes=axes)

    result = execute(contract, realization, x, h, mode)

    np.testing.assert_allclose(result, reference, atol=1e-4, rtol=1e-4)
    assert result.dtype == np.complex64
    assert result.shape == x.shape
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(h, original_h)
    assert realization == original_realization


@pytest.mark.parametrize("mode", ["forward", "inverse", "pipeline"])
def test_small_transform_matches_dense_dft_without_fft_library(mode):
    contract = Contract((8,), 4)
    x, h = _input(contract, "random")
    indices = np.arange(8)
    dft = np.exp(-2j * np.pi * np.outer(indices, indices) / 8)
    inverse = dft.conj() / 8
    if mode == "forward":
        reference = x @ dft.T
    elif mode == "inverse":
        reference = x @ inverse.T
    else:
        reference = ((x @ dft.T) * h) @ inverse.T
    result = execute(contract, _realization(contract, "fused"), x, h, mode)
    np.testing.assert_allclose(result, reference, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("variant", ["staged", "fused", "asymmetric"])
def test_large_tone_keeps_nominally_zero_bins_within_existing_tolerance(variant):
    """NumPy 2 complex ufunc rounding formerly leaked >1e-4 into a zero bin."""
    from sage_fft.validate_semantics import inputs

    contract = Contract((8, 16, 16), 1)
    x, h = inputs(contract, 1, "tone")
    reference = np.fft.fftn(x.astype(np.complex128), axes=(1, 2, 3))
    result = execute(contract, _realization(contract, variant), x, h, "forward")
    assert abs(reference[0, 5, 10, 3]) < 1e-8
    np.testing.assert_allclose(result, reference, atol=1e-4, rtol=1e-4)


def test_forward_and_inverse_checks_detect_paired_sign_errors():
    contract = Contract((8,), 4)
    x, _ = _input(contract, "random")
    forward = np.fft.fft(x, axis=1)
    wrong_forward = np.fft.ifft(x, axis=1) * 8
    wrong_inverse = np.fft.fft(wrong_forward, axis=1) / 8
    assert metrics(wrong_inverse, x)["pass_correctness"]
    assert not metrics(wrong_forward, forward)["pass_correctness"]


def test_metrics_reject_material_error_at_zero_reference():
    reference = np.zeros((1, 8), dtype=np.complex128)
    result = np.zeros_like(reference, dtype=np.complex64)
    result[0, 0] = 0.01
    report = metrics(result, reference)
    assert not report["pass_correctness"]
    assert np.isfinite(report["relative_l2"])
    assert report["max_abs"] == pytest.approx(0.01)


@pytest.mark.parametrize("shape,batch", [((32,), 4), ((8, 16), 1), ((8, 8, 8), 1)])
def test_registered_cuda_source_imports_its_literal_contract(shape, batch):
    text, contract = source(shape, batch)
    assert contract == Contract(shape, batch)
    assert import_cuda("\n\n" + text + "\n") == contract


@pytest.mark.parametrize(
    "old,new",
    [
        ('"normalization":"1/N"', '"normalization":"none"'),
        ('"layout":"interleaved_row_major"', '"layout":"planar"'),
        ("a.x*b.x-a.y*b.y", "a.x*b.x+a.y*b.y"),
        ("y[k].x/=8", "y[k].x/=16"),
        ("CUFFT_FORWARD", "CUFFT_INVERSE"),
        ("cufftPlan1d(&plan, 8, CUFFT_C2C, 4)", "cufftPlan1d(&plan, 16, CUFFT_C2C, 4)"),
    ],
)
def test_source_adapter_rejects_semantic_mutation_even_with_plan_and_tag(old, new):
    text = render((8,), 4)
    assert old in text
    with pytest.raises(ValueError):
        import_cuda(text.replace(old, new, 1))


def test_source_adapter_rejects_duplicate_plan_and_missing_contract():
    text = render((8,), 4)
    with pytest.raises(ValueError):
        import_cuda(text + "\n// cufftPlan1d(&plan, 8, CUFFT_C2C, 4)")
    with pytest.raises(ValueError):
        import_cuda(text.replace("SAGE_CONTRACT", "UNREGISTERED_CONTRACT"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": ()},
        {"shape": (8, 8, 8, 8)},
        {"shape": (3,)},
        {"shape": (2048,)},
        {"shape": (True,)},
        {"shape": (8.0,)},
        {"shape": (8,), "batch": 0},
        {"shape": (8,), "batch": True},
        {"shape": (8,), "batch": 1.5},
        {"shape": (8,), "dtype": "complex128"},
        {"shape": (8,), "bin_order": "bit_reversed"},
        {"shape": (8,), "inverse_normalization": "none"},
        {"shape": (8,), "atol": -1},
        {"shape": (8,), "rtol": float("nan")},
        {"shape": (8,), "atol": float("inf")},
    ],
)
def test_invalid_contracts_raise_explicit_value_error(kwargs):
    with pytest.raises(ValueError):
        Contract(**kwargs).validate()


@pytest.mark.parametrize("mode", ["typo", "", None, True])
def test_unknown_execution_mode_is_rejected(mode):
    contract = Contract((8,), 4)
    with pytest.raises(ValueError):
        stages(contract, initial(contract), mode)


@pytest.mark.parametrize(
    "action",
    [
        dict(group=True, cores=1, fusion=False),
        dict(group=1, cores=True, fusion=False),
        dict(group=1, cores=1, fusion=1),
        dict(group=1.0, cores=1, fusion=False),
    ],
)
def test_legacy_actions_do_not_confuse_booleans_floats_and_integers(action):
    with pytest.raises(ValueError):
        validate_action(Contract((8,), 4), action)
