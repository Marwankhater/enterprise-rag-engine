"""Chat model client for any OpenAI-compatible endpoint (Groq free tier by default, Ollama, vLLM, OpenAI)."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from app.core.config import Settings
from app.core.observability import record_llm_usage, span

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    def json(self) -> dict[str, Any]:
        """Parse a JSON object reply, tolerating code fences around it."""
        raw = self.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise LLMError(f"Expected a JSON object, got: {self.text[:200]!r}")
        try:
            value = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Invalid JSON from model: {exc}") from exc
        if not isinstance(value, dict):
            raise LLMError("Expected a JSON object")
        return value


class LLM(Protocol):
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        name: str = "llm",
    ) -> LLMResult: ...

    def stream(
        self, messages: list[dict[str, str]], *, model: str | None = None, max_tokens: int | None = None
    ) -> AsyncIterator[str]: ...


class OpenAICompatibleLLM:
    RETRYABLE: ClassVar[frozenset[int]] = frozenset({408, 409, 429, 500, 502, 503, 504})

    def __init__(self, settings: Settings, max_retries: int = 4) -> None:
        from openai import AsyncOpenAI

        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else "not-needed"
        self.client = AsyncOpenAI(
            api_key=key, base_url=settings.llm_base_url, timeout=settings.llm_timeout_s, max_retries=0
        )
        self.default_model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_retries = max_retries

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        name: str = "llm",
    ) -> LLMResult:
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        model = model or self.default_model
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": self.temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        with span(f"llm.{name}", model=model) as current:
            for attempt in range(self.max_retries + 1):
                started = time.perf_counter()
                try:
                    response = await self.client.chat.completions.create(**kwargs)
                except (APIConnectionError, APITimeoutError) as exc:
                    error: Exception = exc
                except APIStatusError as exc:
                    if exc.status_code not in self.RETRYABLE:
                        raise LLMError(f"{model} returned {exc.status_code}: {exc.message}") from exc
                    error = exc
                else:
                    record_llm_usage(current, model, response.usage, time.perf_counter() - started)
                    usage = response.usage
                    return LLMResult(
                        text=response.choices[0].message.content or "",
                        model=model,
                        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    )
                if attempt == self.max_retries:
                    raise LLMError(f"{model} failed after {attempt + 1} attempts: {error}") from error
                delay = min(20.0, 2**attempt) + random.uniform(0, 0.5)
                log.warning(
                    "llm retry",
                    extra={
                        "model": model,
                        "attempt": attempt + 1,
                        "delay_s": round(delay, 2),
                        "error": type(error).__name__,
                    },
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def stream(
        self, messages: list[dict[str, str]], *, model: str | None = None, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        model = model or self.default_model
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": self.temperature, "stream": True}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        with span("llm.stream", model=model):
            response = await self.client.chat.completions.create(**kwargs)
            async for chunk in response:
                if chunk.choices and (delta := chunk.choices[0].delta.content):
                    yield delta
