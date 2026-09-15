# Author: even
"""Configurable model transport; authentication is never part of saved requests."""

import copy
import json
import math
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class ModelClient:
    def __init__(self):
        self.backend = os.environ.get("SAGE_LLM_BACKEND", "openai")
        if self.backend not in ("openai", "ollama"):
            raise ValueError("SAGE_LLM_BACKEND must be openai or ollama")
        self.model = os.environ.get("SAGE_LLM_MODEL", "").strip()
        self.base_url = os.environ.get("SAGE_LLM_BASE_URL", "").strip().rstrip("/")
        if not self.model or not self.base_url:
            raise ValueError("Set SAGE_LLM_MODEL and SAGE_LLM_BASE_URL for model calls")
        self.timeout = float(os.environ.get("SAGE_LLM_TIMEOUT", "180"))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("SAGE_LLM_TIMEOUT must be finite and positive")
        url = urlsplit(self.base_url)
        if (
            url.scheme not in ("https", "http")
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or any(c.isspace() for c in self.base_url)
        ):
            raise ValueError(
                "SAGE_LLM_BASE_URL must be an HTTP(S) URL without credentials or query parameters"
            )
        try:
            url.port
        except ValueError as exc:
            raise ValueError("SAGE_LLM_BASE_URL has an invalid port") from exc

    def call(self, request):
        native = copy.deepcopy(request)
        schema = native.pop("output_schema", None)
        native["model"] = self.model
        if self.backend == "ollama":
            response_format = native.get("response_format", {})
            if schema is None and response_format.get("type") == "json_schema":
                schema = response_format["json_schema"]["schema"]
            native = dict(
                model=self.model,
                messages=copy.deepcopy(request["messages"]),
                stream=False,
                options=dict(
                    temperature=request.get("temperature", 0.4),
                    seed=request.get("seed", 0),
                    num_predict=request.get("max_tokens", 192),
                    num_ctx=6144,
                ),
            )
            if schema:
                native["format"] = schema
            endpoint = self.base_url + "/api/chat"
        else:
            if schema:
                native["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "fft_decision", "strict": True, "schema": schema},
                }
            if os.environ.get("SAGE_LLM_DISABLE_THINKING") == "1":
                native["enable_thinking"] = False
            endpoint = self.base_url + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("SAGE_LLM_API_KEY")
        if token:
            headers["Authorization"] = "Bearer " + token
        outgoing = urllib.request.Request(endpoint, data=json.dumps(native).encode(), headers=headers)
        start = time.perf_counter()
        result = dict(request=native, backend=self.backend, search_seed=request.get("seed"))
        try:
            with urllib.request.urlopen(outgoing, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
            result.update(response=payload, exit_code=0, stderr="")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Do not serialize URLs, headers or provider error bodies containing credentials.
            result.update(response="", exit_code=1, stderr=type(exc).__name__)
        result["model_seconds"] = time.perf_counter() - start
        return result
