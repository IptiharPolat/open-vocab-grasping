from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from open_vocab_grasping.agent.prompts import build_system_prompt
from open_vocab_grasping.agent.schemas import validate_grasp_plan


JsonRequester = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


@dataclass(frozen=True)
class PlannerResponse:
    provider: str
    model: str
    plan: dict[str, Any]
    request: dict[str, Any]
    response: dict[str, Any]
    latency_s: float
    usage: dict[str, Any]


class DeepSeekPlanner:
    def __init__(
        self,
        settings: dict[str, Any],
        requester: JsonRequester | None = None,
    ) -> None:
        self.model = str(settings.get("model", "deepseek-v4-flash"))
        self.base_url = str(settings.get("base_url", "https://api.deepseek.com")).rstrip("/")
        self.api_key_env = str(settings.get("api_key_env", "DEEPSEEK_API_KEY"))
        self.timeout_s = float(settings.get("timeout_s", 60.0))
        self.max_tokens = int(settings.get("max_tokens", 800))
        self.thinking = str(settings.get("thinking", "disabled"))
        self.execution_mode = str(settings.get("execution_mode", "open-vocab-simple"))
        if self.execution_mode not in {"open-vocab-simple", "open-vocab-graspnet"}:
            raise ValueError(
                f"Unsupported agent.execution_mode {self.execution_mode!r}"
            )
        self.requester = requester or self._post_json

    @staticmethod
    def _post_json(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("DeepSeek mode requires the 'agent' dependency: pip install -e '.[agent]'") from exc
        try:
            with httpx.Client(timeout=timeout_s, trust_env=False) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return dict(response.json())
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise RuntimeError(
                f"DeepSeek API returned HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"DeepSeek API request failed: {type(exc).__name__}: {exc}") from exc

    def plan(self, instruction: str, available_targets: list[str]) -> PlannerResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not set. "
                "Set it in this terminal without writing it to project files."
            )
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": build_system_prompt(
                        available_targets, execution_mode=self.execution_mode
                    ),
                },
                {"role": "user", "content": instruction},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": self.thinking},
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        started = perf_counter()
        response = self.requester(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            request_payload,
            self.timeout_s,
        )
        latency_s = perf_counter() - started
        try:
            message = response["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek response is missing choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek returned empty plan content")
        try:
            raw_plan = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"DeepSeek content is not valid JSON: {exc}") from exc
        if not isinstance(raw_plan, dict):
            raise RuntimeError("DeepSeek plan must be one JSON object")
        plan = validate_grasp_plan(raw_plan, available_targets)
        return PlannerResponse(
            provider="deepseek",
            model=str(response.get("model", self.model)),
            plan=plan,
            request=request_payload,
            response=response,
            latency_s=latency_s,
            usage=dict(response.get("usage") or {}),
        )
