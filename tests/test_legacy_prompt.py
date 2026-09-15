# Author: even
"""The finite-space policy must request the ID format its parser consumes."""

from unittest.mock import Mock

from sage_fft.campaign import Runner
from sage_fft.llm import ModelClient


def test_legacy_prompt_retains_the_original_output_instruction(monkeypatch):
    call = Mock(return_value=dict(exit_code=0, response="{}", stderr="", model_seconds=0.1))
    monkeypatch.setattr(ModelClient, "__init__", lambda self: None)
    monkeypatch.setattr(ModelClient, "call", call)
    request, *_ = Runner.__new__(Runner).llm("test-state", 4101)
    assert request["messages"][0] == dict(
        role="system",
        content='Select a legal FFT implementation ID. Return JSON only: {"id":integer}. Use supplied measured latency when available; lower is better.',
    )
    assert request["messages"][1] == dict(role="user", content="test-state")
    assert request["max_tokens"] == 40 and request["temperature"] == 0.4
