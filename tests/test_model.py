# Author: even
"""CPU-only contracts for model transport; every HTTP call is intercepted."""

import copy
import io
import json
import math
import urllib.error

import pytest

from sage_fft import llm


FAKE_KEY = "synthetic-model-test-secret-never-a-real-key"
BASE_URL = "https://model.example.invalid/v1"
SCHEMA = {
    "type": "object",
    "properties": {"index": {"type": "integer", "minimum": 0}},
    "required": ["index"],
    "additionalProperties": False,
}


@pytest.fixture(autouse=True)
def isolated_model_environment(monkeypatch):
    """Do not inherit credentials or permit accidental live model traffic."""
    for name in (
        "SAGE_LLM_BACKEND",
        "SAGE_LLM_MODEL",
        "SAGE_LLM_BASE_URL",
        "SAGE_LLM_TIMEOUT",
        "SAGE_LLM_API_KEY",
        "SAGE_LLM_DISABLE_THINKING",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SAGE_LLM_MODEL", "synthetic-test-model")
    monkeypatch.setenv("SAGE_LLM_BASE_URL", BASE_URL)

    def forbidden_transport(*args, **kwargs):
        pytest.fail("Model tests must intercept urllib transport")

    monkeypatch.setattr(llm.urllib.request, "urlopen", forbidden_transport)


@pytest.fixture
def decision_request():
    return {
        "messages": [{"role": "user", "content": "Select a legal action."}],
        "temperature": 0.25,
        "seed": 43,
        "max_tokens": 128,
        "output_schema": copy.deepcopy(SCHEMA),
    }


def capture_transport(monkeypatch, response_payload=None):
    observed = []
    payload = response_payload or {"choices": [{"message": {"content": '{"index":0}'}}]}
    encoded = json.dumps(payload).encode("utf-8")

    def respond(request, timeout):
        observed.append(
            {
                "url": request.full_url,
                "headers": {key.lower(): value for key, value in request.header_items()},
                "payload": json.loads(request.data),
                "timeout": timeout,
            }
        )
        return io.BytesIO(encoded)

    monkeypatch.setattr(llm.urllib.request, "urlopen", respond)
    return observed, payload


@pytest.mark.parametrize("backend", ["openai", "ollama"])
@pytest.mark.parametrize("schema_location", ["output_schema", "response_format"])
def test_schema_payload_and_input_immutability(monkeypatch, decision_request, backend, schema_location):
    if schema_location == "response_format":
        schema = decision_request.pop("output_schema")
        decision_request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "fft_decision", "strict": True, "schema": schema},
        }
    monkeypatch.setenv("SAGE_LLM_BACKEND", backend)
    monkeypatch.setenv("SAGE_LLM_TIMEOUT", "2.5")
    monkeypatch.setenv("SAGE_LLM_BASE_URL", BASE_URL + "/")
    original = copy.deepcopy(decision_request)
    observed, response = capture_transport(monkeypatch)

    result = llm.ModelClient().call(decision_request)

    assert decision_request == original
    assert len(observed) == 1
    sent = observed[0]
    assert sent["timeout"] == 2.5
    assert sent["headers"]["content-type"] == "application/json"
    assert "authorization" not in sent["headers"]
    assert sent["payload"]["model"] == "synthetic-test-model"
    assert sent["payload"]["messages"] == original["messages"]
    assert "output_schema" not in sent["payload"]
    assert result["request"] == sent["payload"]
    assert result["backend"] == backend
    assert result["search_seed"] == 43
    assert result["exit_code"] == 0
    assert result["stderr"] == ""
    assert json.loads(result["response"]) == response
    assert math.isfinite(result["model_seconds"]) and result["model_seconds"] >= 0
    if backend == "openai":
        assert sent["url"] == BASE_URL + "/chat/completions"
        assert sent["payload"]["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "fft_decision", "strict": True, "schema": SCHEMA},
        }
        assert sent["payload"]["temperature"] == 0.25
        assert sent["payload"]["seed"] == 43
        assert sent["payload"]["max_tokens"] == 128
    else:
        assert sent["url"] == BASE_URL + "/api/chat"
        assert sent["payload"]["format"] == SCHEMA
        assert sent["payload"]["stream"] is False
        assert sent["payload"]["options"] == {
            "temperature": 0.25,
            "seed": 43,
            "num_predict": 128,
            "num_ctx": 6144,
        }
        assert "response_format" not in sent["payload"]


@pytest.mark.parametrize("backend", ["openai", "ollama"])
def test_saved_request_does_not_alias_caller_messages(monkeypatch, decision_request, backend):
    monkeypatch.setenv("SAGE_LLM_BACKEND", backend)
    capture_transport(monkeypatch)
    original = copy.deepcopy(decision_request)
    result = llm.ModelClient().call(decision_request)
    result["request"]["messages"][0]["content"] = "A saved-result annotation"
    assert decision_request == original


def test_openai_thinking_setting_and_header_only_auth(monkeypatch, decision_request, capsys, caplog):
    monkeypatch.setenv("SAGE_LLM_API_KEY", FAKE_KEY)
    monkeypatch.setenv("SAGE_LLM_DISABLE_THINKING", "1")
    observed, _ = capture_transport(monkeypatch)
    result = llm.ModelClient().call(decision_request)
    assert observed[0]["headers"]["authorization"] == "Bearer " + FAKE_KEY
    assert observed[0]["payload"]["enable_thinking"] is False
    assert FAKE_KEY not in json.dumps(result)
    assert "Authorization" not in json.dumps(result)
    output = capsys.readouterr()
    assert FAKE_KEY not in output.out + output.err + caplog.text


@pytest.mark.parametrize("kind", ["url", "http", "timeout", "os"])
def test_transport_failures_redact_error_messages_and_bodies(
    monkeypatch, decision_request, capsys, caplog, kind
):
    monkeypatch.setenv("SAGE_LLM_API_KEY", FAKE_KEY)
    errors = {
        "url": urllib.error.URLError("Transport leaked " + FAKE_KEY),
        "http": urllib.error.HTTPError(
            BASE_URL + "/" + FAKE_KEY,
            401,
            "Denied " + FAKE_KEY,
            {"Authorization": "Bearer " + FAKE_KEY},
            io.BytesIO(FAKE_KEY.encode()),
        ),
        "timeout": TimeoutError("Timed out with " + FAKE_KEY),
        "os": OSError("Socket error " + FAKE_KEY),
    }

    def raise_error(request, timeout):
        raise errors[kind]

    monkeypatch.setattr(llm.urllib.request, "urlopen", raise_error)
    result = llm.ModelClient().call(decision_request)
    assert result["exit_code"] != 0
    assert result["response"] == ""
    assert result["stderr"]
    assert FAKE_KEY not in json.dumps(result)
    output = capsys.readouterr()
    assert FAKE_KEY not in output.out + output.err + caplog.text


@pytest.mark.parametrize("backend", ["unsupported", "", "OPENAI"])
def test_invalid_backend_is_rejected(monkeypatch, backend):
    monkeypatch.setenv("SAGE_LLM_BACKEND", backend)
    with pytest.raises(ValueError):
        llm.ModelClient()


@pytest.mark.parametrize("name", ["SAGE_LLM_MODEL", "SAGE_LLM_BASE_URL"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_required_configuration_is_nonempty(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        llm.ModelClient()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "", "not-a-number"])
def test_timeout_must_be_positive_and_finite(monkeypatch, value):
    monkeypatch.setenv("SAGE_LLM_TIMEOUT", value)
    with pytest.raises(ValueError):
        llm.ModelClient()


@pytest.mark.parametrize("value", ["0.25", "12", "180"])
def test_valid_timeout_is_accepted(monkeypatch, value):
    monkeypatch.setenv("SAGE_LLM_TIMEOUT", value)
    assert llm.ModelClient().timeout == float(value)


@pytest.mark.parametrize(
    "base_url",
    [
        "file:///tmp/model-test",
        "ftp://model.example.invalid",
        "data:text/plain,test",
        "httpsx://model.example.invalid",
        "model.example.invalid",
        "https:///missing-host",
        "https://user:synthetic-url-password@model.example.invalid/v1",
        "http://user@model.example.invalid",
    ],
)
def test_endpoint_requires_http_host_without_userinfo(monkeypatch, base_url):
    monkeypatch.setenv("SAGE_LLM_BASE_URL", base_url)
    with pytest.raises(ValueError):
        llm.ModelClient()
