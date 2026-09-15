# Author: even
"""Transformation composition, legal menus, and malformed action boundaries."""

import copy
import json

import pytest

from sage_fft import Contract
from sage_fft.mask_ir import actions, apply, features, initial, key, stages, validate


def test_levels_compose_twenty_to_eight_to_six_without_mutating_parents():
    contract = Contract((8, 8, 8), 1)
    staged = initial(contract)
    untouched = copy.deepcopy(staged)
    grouped = apply(contract, staged, dict(level="fft", region="all", op="merge", boundaries="all"))
    fused = apply(contract, grouped, dict(level="pipeline", region="both", op="fuse"))
    scheduled = apply(
        contract, fused, dict(level="schedule", region="all", op="reschedule", schedule=dict(cores=4, pack=8))
    )

    assert [features(contract, state)["launches"] for state in (staged, grouped, fused, scheduled)] == [
        20,
        8,
        6,
        6,
    ]
    assert staged == untouched
    assert not grouped["fuse_multiply"] and not grouped["fuse_scale"]
    assert fused["cores"] == 1 and scheduled["cores"] == 4
    assert len([step for step in stages(contract, fused) if step["kind"] == "fft"]) == 6
    assert all(step["kind"] == "fft" for step in stages(contract, fused))


def test_forward_and_inverse_masks_change_independently():
    contract = Contract((8, 8, 8), 1)
    before = initial(contract)
    after = apply(contract, before, dict(level="fft", region="forward:1", op="merge", boundaries=[1]))
    assert after["forward_masks"][1] == [0, 1]
    assert after["inverse_masks"] == before["inverse_masks"]
    assert after["forward_masks"][0] == before["forward_masks"][0]
    assert after["forward_masks"][2] == before["forward_masks"][2]
    assert len(stages(contract, after)) == 19
    restored = apply(contract, after, dict(level="fft", region="forward:1", op="split", boundaries=[1]))
    assert restored == before


def test_epilogue_fusion_removes_only_its_own_materialization_boundary():
    contract = Contract((8, 8, 8), 1)
    before = initial(contract)
    multiply = apply(contract, before, dict(level="pipeline", region="multiply", op="fuse"))
    steps = stages(contract, multiply)
    assert len(steps) == 19
    assert not any(step["kind"] == "multiply" for step in steps)
    assert sum(step["kind"] == "scale" for step in steps) == 1
    attachments = [step for step in steps if step.get("multiply")]
    assert len(attachments) == 1
    assert attachments[0]["axis"] == 0 and not attachments[0]["inverse"]
    assert attachments[0]["last"] == 3


def test_action_menu_is_unique_valid_and_excludes_noops():
    contract = Contract((8, 8), 1)
    current = initial(contract)
    menu = actions(contract, current)
    assert menu
    identities = [key(state) for _, state in menu]
    assert len(identities) == len(set(identities))
    assert key(current) not in identities
    assert {action["level"] for action, _ in menu} == {"fft", "pipeline", "schedule"}
    for action, state in menu:
        validate(contract, state)
        assert apply(contract, current, action) == state


@pytest.mark.parametrize(
    "field,value",
    [
        ("cores", True),
        ("cores", 2),
        ("cores", 4.0),
        ("pack", True),
        ("pack", 3),
        ("pack", 8.0),
        ("fuse_multiply", 1),
        ("fuse_scale", "false"),
        ("forward_masks", [[True, 1], [1, 1]]),
        ("forward_masks", [[0.0, 1], [1, 1]]),
        ("inverse_masks", [[2, 1], [1, 1]]),
        ("inverse_masks", [[1], [1, 1]]),
        ("forward_masks", [[1, 1]]),
        ("forward_masks", None),
        ("inverse_masks", [None, [1, 1]]),
    ],
)
def test_invalid_realizations_fail_validation(field, value):
    contract = Contract((8, 8), 1)
    realization = initial(contract)
    realization[field] = value
    with pytest.raises(ValueError):
        validate(contract, realization)


@pytest.mark.parametrize("contract", [Contract((4,), 1), Contract((2, 16), 1), Contract((1024, 16), 1)])
def test_dma_tail_axis_alignment_and_ub_overflow_are_rejected(contract):
    realization = initial(contract)
    realization["pack"] = 16
    with pytest.raises(ValueError):
        validate(contract, realization)


@pytest.mark.parametrize(
    "action",
    [
        None,
        [],
        "merge",
        dict(level="fft", region="all", op="merge", boundaries=[]),
        dict(level="fft", region="forward:0", op="merge", boundaries=[0]),
        dict(level="fft", region="forward:0", op="merge", boundaries=[3]),
        dict(level="fft", region="forward:0", op="merge", boundaries=[True]),
        dict(level="fft", region="forward:0", op="merge", boundaries=[1, 1]),
        dict(level="fft", region="forward:0", op="merge", boundaries=[[1]]),
        dict(level="fft", region="forward:9", op="merge", boundaries="all"),
        dict(level="pipeline", region="both", op="unfuse"),
        dict(level="pipeline", region="both", op="fuse", extra=True),
        dict(level="schedule", region="all", op="reschedule", schedule=None),
        dict(level="schedule", region="all", op="reschedule", schedule=dict(cores=4, pack=8, untrusted=1)),
    ],
)
def test_malformed_actions_are_rejected_without_mutating_current(action):
    contract = Contract((8, 8), 1)
    current = initial(contract)
    original = copy.deepcopy(current)
    with pytest.raises(ValueError):
        apply(contract, current, action)
    assert current == original


def _model_response(decision, backend="openai"):
    content = json.dumps(decision)
    if backend == "openai":
        response = {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
    else:
        response = {"message": {"content": content}, "prompt_eval_count": 7, "eval_count": 3}
    return {"response": json.dumps(response)}


@pytest.mark.parametrize("backend", ["openai", "ollama"])
def test_indexed_and_direct_interfaces_resolve_the_same_menu_entry(backend):
    from sage_fft.structured_search import parse_response

    contract = Contract((8, 8), 1)
    current = initial(contract)
    menu = actions(contract, current)
    selected = len(menu) // 2
    indexed = parse_response(_model_response({"id": selected}, backend), "flat_llm", menu, contract, current)
    direct = parse_response(_model_response(menu[selected][0], backend), "sage", menu, contract, current)
    assert indexed[:2] == direct[:2] == menu[selected]
    assert indexed[2] == direct[2] == {"prompt_tokens": 7, "completion_tokens": 3}
    assert current == initial(contract)


@pytest.mark.parametrize("candidate_id", [-1, 10**9, True, False, "0", 0.5, None, [0]])
def test_index_parser_rejects_invalid_ids(candidate_id):
    from sage_fft.structured_search import parse_response

    contract = Contract((8, 8), 1)
    current = initial(contract)
    with pytest.raises(ValueError):
        parse_response(
            _model_response({"id": candidate_id}), "flat_llm", actions(contract, current), contract, current
        )


def test_index_parser_rejects_id_equal_to_menu_length_and_extra_fields():
    from sage_fft.structured_search import parse_response

    contract = Contract((8, 8), 1)
    current = initial(contract)
    menu = actions(contract, current)
    for decision in ({"id": len(menu)}, {"id": 0, "action": "arbitrary"}):
        with pytest.raises(ValueError):
            parse_response(_model_response(decision), "flat_llm", menu, contract, current)


def test_direct_parser_rejects_an_id_and_a_valid_action_outside_shared_menu():
    from sage_fft.structured_search import parse_response

    contract = Contract((8, 8), 1)
    current = initial(contract)
    menu = actions(contract, current)
    for decision, allowed in (({"id": 0}, menu), (menu[0][0], menu[1:])):
        with pytest.raises(ValueError):
            parse_response(_model_response(decision), "sage", allowed, contract, current)


def test_optional_diagnosis_does_not_change_the_selected_action():
    from sage_fft.structured_search import parse_response

    contract = Contract((8, 8), 1)
    current = initial(contract)
    menu = actions(contract, current)
    raw = _model_response(
        {"diagnosis": "This action removes a materialization boundary.", "decision": {"id": 0}}
    )
    assert parse_response(raw, "flat_llm", menu, contract, current)[:2] == menu[0]


def test_request_interfaces_share_candidates_and_honor_seen_states():
    from sage_fft.structured_search import make_request

    contract = Contract((8, 8), 1)
    current = initial(contract)
    excluded = actions(contract, current, (4, 8))[0][1]
    seen = {key(current), key(excluded)}
    history = [{"realization": current, "latency_us": 1.0, "resources": features(contract, current)}]
    original_history = copy.deepcopy(history)
    indexed, indexed_menu = make_request("M2", current, history, "flat_llm", 41, 1, seen)
    direct, direct_menu = make_request("M2", current, history, "sage", 41, 1, seen)
    assert indexed_menu == direct_menu
    assert all(key(state) not in seen for _, state in indexed_menu)
    indexed_state = json.loads(indexed["messages"][1]["content"])
    direct_state = json.loads(direct["messages"][1]["content"])
    assert [entry["id"] for entry in indexed_state["candidates"]] == list(range(len(indexed_menu)))
    assert all("id" not in entry for entry in direct_state["candidates"])
    assert history == original_history
