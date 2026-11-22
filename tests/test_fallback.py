import pytest

from src.config import ModelSpec
from src.llm.fallback import AllModelsFailed, call_with_fallback, clean_output, content_to_text, is_retryable

from tests.conftest import ScriptedLLM


def run(llms, specs, **kw):
    sleeps = []
    result = call_with_fallback(specs, lambda n: llms[n], ["hi"], sleep=sleeps.append, **kw)
    return result, sleeps


def spec(name, **kw):
    return ModelSpec(name=name, model=name, **kw)


def test_success_first_try():
    result, sleeps = run({"a": ScriptedLLM(lambda t: "ok")}, [spec("a")])
    assert result.text == "ok" and result.model == "a" and not result.fell_back and sleeps == []


def test_retries_rate_limit_with_exponential_backoff():
    replies = iter([Exception("429 rate limit"), Exception("429 rate limit"), "fine"])
    llm = ScriptedLLM(lambda t: next(replies))
    result, sleeps = run({"a": llm}, [spec("a")], retries=2, backoff_s=2.0)
    assert result.text == "fine" and result.attempts == 3 and sleeps == [2.0, 4.0]


def test_hard_error_skips_retries_and_falls_back():
    llms = {"a": ScriptedLLM(lambda t: Exception("model 'a' not found")), "b": ScriptedLLM(lambda t: "from b")}
    result, sleeps = run(llms, [spec("a"), spec("b")])
    assert result.model == "b" and result.fell_back and sleeps == [] and len(llms["a"].calls) == 1


def test_exhausted_retries_then_fallback():
    llms = {"a": ScriptedLLM(lambda t: Exception("timeout")), "b": ScriptedLLM(lambda t: "from b")}
    result, _ = run(llms, [spec("a"), spec("b")], retries=1)
    assert result.model == "b" and len(llms["a"].calls) == 2


def test_empty_reply_counts_as_failure():
    llms = {"a": ScriptedLLM(lambda t: "<think>only thoughts</think>"), "b": ScriptedLLM(lambda t: "real")}
    result, _ = run(llms, [spec("a", strip_think=True), spec("b")])
    assert result.model == "b"


def test_all_fail_raises_with_every_error():
    llms = {"a": ScriptedLLM(lambda t: Exception("boom a")), "b": ScriptedLLM(lambda t: Exception("boom b"))}
    with pytest.raises(AllModelsFailed) as info:
        run(llms, [spec("a"), spec("b")])
    assert len(info.value.errors) == 2


def test_clean_output_strips_think_blocks():
    assert clean_output("<think>x\ny</think> answer ", True) == "answer"
    assert clean_output("<think>never closed", True) == ""
    assert clean_output("<think>keep</think>a", False) == "<think>keep</think>a"


def test_content_to_text_handles_part_lists():
    assert content_to_text([{"type": "text", "text": "a"}, "b"]) == "ab"


def test_is_retryable():
    assert is_retryable(Exception("HTTP 429 Too Many Requests"))
    assert is_retryable(TimeoutError("timed out"))
    assert not is_retryable(Exception("model not found"))
