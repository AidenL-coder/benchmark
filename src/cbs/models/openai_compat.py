"""OpenAI-compatible backend: vLLM, Ollama, or a hosted provider.

All three speak the same wire format, so the frozen model is chosen by
`base_url` + `model` in config rather than by code (docs/DECISIONS.md D-01).

Cost note (brief section 10, "ask before live spend"): this client can spend real
money when pointed at a hosted provider. It refuses to run unless the config
either sets `usd_per_1k_prompt/completion_tokens` (so the accountant can enforce
a dollar cap) or explicitly declares the endpoint free via `local: true`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from cbs.budget import Usage
from cbs.models.base import (
    Completion,
    CompletionRequest,
    ModelClient,
    ModelUnavailable,
    estimate_tokens,
)

__all__ = ["OpenAICompatConfig", "OpenAICompatClient"]


@dataclass
class OpenAICompatConfig:
    base_url: str
    model: str
    api_key: str | None = None
    #: True for a local vLLM/Ollama server (no per-token cost).
    local: bool = True
    usd_per_1k_prompt_tokens: float = 0.0
    usd_per_1k_completion_tokens: float = 0.0
    timeout_s: float = 120.0
    max_retries: int = 3
    retry_backoff_s: float = 2.0
    #: Chat endpoint by default; some vLLM deployments prefer raw completions.
    endpoint: str = "chat"  # "chat" | "completions"

    def __post_init__(self) -> None:
        if self.endpoint not in ("chat", "completions"):
            raise ValueError(f"unknown endpoint {self.endpoint!r}")
        if not self.local and (
            self.usd_per_1k_prompt_tokens <= 0
            and self.usd_per_1k_completion_tokens <= 0
        ):
            raise ValueError(
                "a non-local endpoint may cost money: set usd_per_1k_* pricing so "
                "the budget accountant can enforce a dollar cap, or set local=true "
                "to declare the endpoint free (brief section 10)"
            )


class OpenAICompatClient(ModelClient):
    def __init__(self, config: OpenAICompatConfig):
        self.config = config
        self.model_id = config.model
        self._client = None

    def _http(self):
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ModelUnavailable(
                    "httpx is required for the OpenAI-compatible backend: "
                    "pip install -e .[serving]"
                ) from exc
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.Client(
                base_url=self.config.base_url.rstrip("/"),
                headers=headers,
                timeout=self.config.timeout_s,
            )
        return self._client

    def _payload(self, request: CompletionRequest) -> tuple[str, dict]:
        common = {
            "model": self.config.model,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            common["seed"] = request.seed
        if request.stop:
            common["stop"] = list(request.stop)

        if self.config.endpoint == "chat":
            messages = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": request.prompt})
            return "/v1/chat/completions", {**common, "messages": messages}

        prompt = request.prompt
        if request.system:
            prompt = f"{request.system}\n\n{prompt}"
        return "/v1/completions", {**common, "prompt": prompt}

    @staticmethod
    def _extract_text(endpoint: str, data: dict) -> tuple[str, str]:
        choice = (data.get("choices") or [{}])[0]
        finish = choice.get("finish_reason") or "stop"
        if endpoint == "chat":
            return (choice.get("message") or {}).get("content") or "", finish
        return choice.get("text") or "", finish

    def _usage(self, data: dict, request: CompletionRequest, text: str) -> tuple[Usage, bool]:
        raw = data.get("usage") or {}
        reported = "prompt_tokens" in raw and "completion_tokens" in raw
        if reported:
            pt = int(raw["prompt_tokens"])
            ct = int(raw["completion_tokens"])
        else:
            pt = estimate_tokens(request.prompt)
            ct = estimate_tokens(text)
        usd = (
            pt / 1000.0 * self.config.usd_per_1k_prompt_tokens
            + ct / 1000.0 * self.config.usd_per_1k_completion_tokens
        )
        return Usage(calls=1, prompt_tokens=pt, completion_tokens=ct, usd=usd), reported

    def _generate(self, request: CompletionRequest) -> Completion:
        path, payload = self._payload(request)
        client = self._http()

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = client.post(path, json=payload)
                if response.status_code >= 500 or response.status_code == 429:
                    raise ModelUnavailable(
                        f"{response.status_code} from {path}: {response.text[:200]}"
                    )
                response.raise_for_status()
                data = response.json()
                break
            except Exception as exc:  # noqa: BLE001 - retried below
                last_exc = exc
                if attempt == self.config.max_retries - 1:
                    raise ModelUnavailable(
                        f"{self.config.base_url} failed after "
                        f"{self.config.max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(self.config.retry_backoff_s * (2**attempt))
        else:  # pragma: no cover - loop always breaks or raises
            raise ModelUnavailable(str(last_exc))

        text, finish = self._extract_text(self.config.endpoint, data)
        usage, reported = self._usage(data, request, text)
        return Completion(
            text=text,
            usage=usage,
            finish_reason=finish,
            model_id=self.model_id,
            seed=request.seed,
            meta={
                "usage_reported": reported,
                "endpoint": self.config.endpoint,
                "task_id": request.meta.get("task_id"),
            },
        )

    def health_check(self) -> dict:
        """Confirm the endpoint is live. Used by `cbs env`."""
        try:
            response = self._http().get("/v1/models")
            response.raise_for_status()
            return {"ok": True, "models": response.json()}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "backend": "OpenAICompatClient",
            "base_url": self.config.base_url,
            "endpoint": self.config.endpoint,
            "local": self.config.local,
        }
