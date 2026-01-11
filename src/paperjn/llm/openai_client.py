from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Literal

from paperjn.llm.json_utils import extract_json_object


def _require_requests():
    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: requests. Install with `pip install -e '.[literature]'`.") from exc
    return requests


def _env_or_none(name: str | None) -> str | None:
    if not name:
        return None
    val = os.environ.get(name)
    return val if val else None


def _extract_text_from_response_json(resp_json: dict[str, Any]) -> str:
    # New Responses API sometimes provides `output_text` directly.
    if isinstance(resp_json.get("output_text"), str):
        return str(resp_json["output_text"])

    # Responses API: `output` array with `content` chunks.
    out = resp_json.get("output")
    if isinstance(out, list):
        chunks: list[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str):
                chunks.append(str(item["text"]))
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                if isinstance(c.get("text"), str):
                    chunks.append(str(c["text"]))
        if chunks:
            return "\n".join(chunks)

    # Chat Completions fallback: `choices[0].message.content`
    choices = resp_json.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return str(msg["content"])

    raise ValueError("Could not extract text from OpenAI response JSON.")


@dataclass(frozen=True)
class OpenAIClient:
    api_key: str | None
    base_url: str = "https://api.openai.com/v1"
    timeout_s: int = 90
    max_retries: int = 6
    request_delay_s: float = 0.0
    user_agent: str = "paperjn/0.1 (LLM extraction)"

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
    ) -> "OpenAIClient":
        return cls(api_key=_env_or_none(api_key_env), base_url=_env_or_none(base_url_env) or cls.base_url)

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        requests = _require_requests()

        url = self.base_url.rstrip("/") + path
        backoff_s = 1.0
        last_error: str | None = None

        for attempt in range(int(self.max_retries)):
            if self.request_delay_s:
                time.sleep(max(float(self.request_delay_s), 0.0))
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=int(self.timeout_s))

            if resp.status_code in {429} or resp.status_code >= 500:
                try:
                    retry_after = float(resp.headers.get("Retry-After")) if resp.headers else None
                except Exception:
                    retry_after = None
                time.sleep(max(backoff_s, retry_after or 0.0))
                backoff_s *= 1.7
                last_error = f"HTTP {resp.status_code}"
                continue

            if resp.status_code >= 400:
                try:
                    err = resp.json()
                except Exception:
                    err = {"error": resp.text}
                raise RuntimeError(f"OpenAI API error (HTTP {resp.status_code}): {err}")

            try:
                return resp.json()
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"OpenAI response was not JSON: {exc}") from exc

        raise RuntimeError(f"OpenAI API repeatedly failed: {last_error or 'unknown error'}")

    def call_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        reasoning_effort: str | None = "auto",
        max_output_tokens: int = 1200,
        endpoint: Literal["responses", "chat_completions"] = "responses",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Call a model and parse the returned JSON object (prompt must enforce JSON-only output)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if endpoint == "responses":
            base_payload = {
                "model": model,
                "input": messages,
                "max_output_tokens": int(max_output_tokens),
                "text": {"format": {"type": "json_object"}},
            }
            if temperature is not None:
                base_payload["temperature"] = float(temperature)

            if reasoning_effort is None:
                efforts_to_try: list[str | None] = [None]
            elif reasoning_effort == "auto":
                # gpt-5-nano supports `minimal`; gpt-5.2 supports `none`.
                if "nano" in model:
                    efforts_to_try = ["minimal", "low", None]
                else:
                    efforts_to_try = ["none", "low", None]
            else:
                efforts_to_try = [str(reasoning_effort), None]

            raw: dict[str, Any] | None = None
            last_exc: RuntimeError | None = None
            for eff in efforts_to_try:
                payload = dict(base_payload)
                if eff is not None:
                    payload["reasoning"] = {"effort": eff}
                try:
                    raw = self._post_json("/responses", payload)
                    break
                except RuntimeError as exc:
                    msg = str(exc)
                    last_exc = exc
                    if "reasoning" in msg and ("Unsupported parameter" in msg or "Unsupported value" in msg):
                        continue
                    raise
            if raw is None:
                raise last_exc or RuntimeError("OpenAI request failed.")
        else:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": int(max_output_tokens),
                "response_format": {"type": "json_object"},
            }
            if temperature is not None:
                payload["temperature"] = float(temperature)
            raw = self._post_json("/chat/completions", payload)

        status = raw.get("status")
        if status and status != "completed":
            raise RuntimeError(f"OpenAI response status not completed: {status} ({raw.get('incomplete_details')})")

        text = _extract_text_from_response_json(raw)
        obj = extract_json_object(text)
        return obj, raw
