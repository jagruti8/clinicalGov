"""Configuration loading: model specs, role assignment, and pipeline settings.

Everything model-related lives in a YAML file (config/models.yaml by default).
Swapping, adding, or removing a model is a config edit, never a code change.
"""
import os
import re
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "models.yaml"
SUPPORTED_PROVIDERS = {"ollama", "openai_compatible", "groq", "gemini"}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """Raised when the model configuration is missing or invalid."""


class ModelSpec(BaseModel):
    name: str = ""
    provider: str = "ollama"
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0
    timeout_s: float = 120.0
    num_ctx: Optional[int] = None
    tool_calling: Literal["json", "native"] = "json"
    strip_think: bool = False

    @model_validator(mode="after")
    def _check(self):
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"unsupported provider '{self.provider}' (use one of {sorted(SUPPORTED_PROVIDERS)})"
            )
        if not self.model.strip():
            raise ValueError("model tag must not be empty")
        return self


class EmbeddingSettings(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"


class Roles(BaseModel):
    planner: str
    synthesizers: list[str] = Field(min_length=1)
    evidence: str
    critic: str
    judge: str


class PipelineSettings(BaseModel):
    max_reformulations: int = Field(2, ge=0, le=5)
    relevance_threshold: float = 0.4
    retrieval_k: int = Field(5, ge=1, le=50)
    fetch_page_size: int = Field(25, ge=1, le=100)
    fetch_max_results: int = Field(50, ge=1, le=500)
    eligibility_chars: int = Field(1500, ge=100)
    max_context_chars: int = Field(12000, ge=1000)
    max_revisions: int = Field(1, ge=0, le=3)
    critique_mode: Literal["full", "code_only", "off"] = "full"
    nct_routing_override: bool = True
    plan_repair_retries: int = Field(1, ge=0, le=3)
    llm_retries: int = Field(2, ge=0, le=10)
    retry_backoff_s: float = Field(2.0, ge=0)


class AppConfig(BaseModel):
    embedding: EmbeddingSettings = EmbeddingSettings()
    models: dict[str, ModelSpec]
    roles: Roles
    fallback: list[str] = []
    pipeline: PipelineSettings = PipelineSettings()

    @model_validator(mode="after")
    def _check_references(self):
        for key, spec in self.models.items():
            spec.name = key
        referenced = [
            self.roles.planner, self.roles.evidence, self.roles.critic, self.roles.judge,
            *self.roles.synthesizers, *self.fallback,
        ]
        missing = sorted({m for m in referenced if m not in self.models})
        if missing:
            raise ValueError(f"roles/fallback reference unknown models: {missing}")
        if len(set(self.roles.synthesizers)) != len(self.roles.synthesizers):
            raise ValueError("roles.synthesizers must not repeat a model")
        return self

    def chain_for(self, model_name: str) -> list[str]:
        """The model itself first, then the configured fallbacks (never itself twice)."""
        return [model_name] + [m for m in self.fallback if m != model_name]

    def describe(self) -> dict:
        """Active roles with provider/tag/endpoint. Never includes secrets."""
        def info(name):
            spec = self.models[name]
            return {"name": name, "provider": spec.provider, "model": spec.model,
                    "base_url": spec.base_url}
        return {
            "planner": info(self.roles.planner),
            "synthesizers": [info(n) for n in self.roles.synthesizers],
            "evidence": info(self.roles.evidence),
            "critic": info(self.roles.critic),
            "judge": info(self.roles.judge),
            "fallback": list(self.fallback),
            "pipeline": self.pipeline.model_dump(),
        }


def _expand_env(text: str) -> str:
    def repl(match):
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(match.group(1), default)
    return _ENV_PATTERN.sub(repl, text)


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load config from `path`, else $MODELS_CONFIG, else config/models.yaml."""
    resolved = Path(path or os.environ.get("MODELS_CONFIG") or DEFAULT_CONFIG_PATH)
    if not resolved.is_absolute() and not resolved.exists():
        resolved = REPO_ROOT / resolved
    if not resolved.exists():
        raise ConfigError(f"Config file not found: {resolved}")

    try:
        raw = yaml.safe_load(_expand_env(resolved.read_text())) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file {resolved} is not valid YAML: {exc}") from exc

    defaults = raw.pop("defaults", {}) or {}
    models = raw.get("models") or {}
    raw["models"] = {name: {**defaults, **(spec or {})} for name, spec in models.items()}

    try:
        return AppConfig(**raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"Invalid config {resolved}: {problems}") from exc
