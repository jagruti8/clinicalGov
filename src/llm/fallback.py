"""Retry and fallback across models, plus output cleanup."""
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from src.config import ModelSpec

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_RETRYABLE_HINTS = (
    "429", "rate", "quota", "resource_exhausted", "timeout", "timed out", "connection",
    "temporarily", "overloaded", "unavailable", "502", "503", "504", "reset by peer",
)


class AllModelsFailed(RuntimeError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("All models failed: " + " | ".join(errors))


@dataclass
class CallResult:
    text: str
    model: str
    attempts: int
    latency_s: float
    errors: list[str] = field(default_factory=list)
    fell_back: bool = False


def is_retryable(exc: Exception) -> bool:
    """Transient failures (rate limits, timeouts, connection drops) are worth retrying."""
    message = f"{type(exc).__name__} {exc}".lower()
    return any(hint in message for hint in _RETRYABLE_HINTS)


def content_to_text(content) -> str:
    """Chat responses may be a string or a list of parts (Gemini does this)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content or "")


def clean_output(text: str, strip_think: bool) -> str:
    if strip_think:
        text = _THINK_RE.sub("", text)
    return text.strip()


def call_with_fallback(
    chain: list[ModelSpec],
    get_llm: Callable[[str], object],
    messages: list,
    retries: int = 2,
    backoff_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> CallResult:
    """Try each model in `chain` in order. Retry transient errors with backoff,
    move to the next model on a hard error or an empty reply."""
    errors: list[str] = []
    attempts = 0
    start = time.perf_counter()

    for index, spec in enumerate(chain):
        for attempt in range(retries + 1):
            attempts += 1
            try:
                response = get_llm(spec.name).invoke(messages)
                text = clean_output(content_to_text(response.content), spec.strip_think)
                if not text:
                    raise ValueError("empty response")
                return CallResult(
                    text=text, model=spec.name, attempts=attempts,
                    latency_s=time.perf_counter() - start, errors=errors,
                    fell_back=index > 0,
                )
            except Exception as exc:
                errors.append(f"{spec.name}: {type(exc).__name__}: {str(exc)[:200]}")
                if is_retryable(exc) and attempt < retries:
                    sleep(backoff_s * (2 ** attempt))
                    continue
                break
    raise AllModelsFailed(errors)
