"""Chat model construction and the model registry.

Imports of provider libraries are lazy, so only the providers a config actually
uses need to be installed.
"""
import os
import time
from typing import Callable, Optional

from src.config import AppConfig, ModelSpec
from src.llm.fallback import CallResult, call_with_fallback


def _api_key(spec: ModelSpec, default_env: str) -> str:
    env_name = spec.api_key_env or default_env
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(f"Environment variable {env_name} is not set (needed by model '{spec.name}').")
    return key


def build_chat_model(spec: ModelSpec):
    """Create a LangChain chat model for one model spec."""
    if spec.provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": spec.model,
            "temperature": spec.temperature,
            "client_kwargs": {"timeout": spec.timeout_s},
        }
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        if spec.num_ctx:
            kwargs["num_ctx"] = spec.num_ctx
        return ChatOllama(**kwargs)

    if spec.provider == "openai_compatible":
        from langchain_openai import ChatOpenAI

        key = os.environ.get(spec.api_key_env or "", "") or "not-needed"
        return ChatOpenAI(
            model=spec.model, base_url=spec.base_url, api_key=key,
            temperature=spec.temperature, timeout=spec.timeout_s,
        )

    if spec.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=spec.model, temperature=spec.temperature,
                        api_key=_api_key(spec, "GROQ_API_KEY"))

    if spec.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=spec.model, temperature=spec.temperature,
                                      google_api_key=_api_key(spec, "GEMINI_API_KEY"))

    raise ValueError(f"Unsupported provider: {spec.provider}")


class ModelRegistry:
    """Builds chat models lazily from config and calls them with retry and fallback."""

    def __init__(self, config: AppConfig,
                 builder: Callable[[ModelSpec], object] = build_chat_model,
                 sleep: Callable[[float], None] = time.sleep):
        self.config = config
        self._builder = builder
        self._sleep = sleep
        self._cache: dict[str, object] = {}

    def spec(self, name: str) -> ModelSpec:
        return self.config.models[name]

    def get_llm(self, name: str):
        if name not in self._cache:
            self._cache[name] = self._builder(self.config.models[name])
        return self._cache[name]

    def call(self, name: str, messages: list, chain: Optional[list[str]] = None) -> CallResult:
        """Call `name`; on failure after retries, try the fallback models in order."""
        names = chain or self.config.chain_for(name)
        specs = [self.config.models[n] for n in names]
        return call_with_fallback(
            specs, self.get_llm, messages,
            retries=self.config.pipeline.llm_retries,
            backoff_s=self.config.pipeline.retry_backoff_s,
            sleep=self._sleep,
        )
