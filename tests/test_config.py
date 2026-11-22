import pytest

from src.config import ConfigError, load_config

VALID = """
defaults:
  base_url: ${OLLAMA_BASE_URL:-http://localhost:11434}
  num_ctx: 4096
models:
  a: {model: tag-a}
  b: {model: tag-b, strip_think: true}
roles: {planner: a, synthesizers: [a, b], evidence: a, critic: b, judge: b}
fallback: [a, b]
"""


def write(tmp_path, text):
    path = tmp_path / "models.yaml"
    path.write_text(text)
    return str(path)


def test_loads_and_applies_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    config = load_config(write(tmp_path, VALID))
    assert config.models["a"].base_url == "http://localhost:11434"
    assert config.models["b"].num_ctx == 4096
    assert config.models["b"].strip_think is True
    assert config.models["a"].name == "a"


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://cluster:11434")
    assert load_config(write(tmp_path, VALID)).models["a"].base_url == "http://cluster:11434"


def test_env_var_selects_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_CONFIG", write(tmp_path, VALID))
    assert set(load_config().models) == {"a", "b"}


def test_reduce_to_two_models_is_a_config_edit(tmp_path):
    text = VALID.replace("[a, b]", "[a]", 1)
    config = load_config(write(tmp_path, text))
    assert config.roles.synthesizers == ["a"]


def test_unknown_role_reference_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown models"):
        load_config(write(tmp_path, VALID.replace("judge: b", "judge: zzz")))


def test_duplicate_synthesizers_rejected(tmp_path):
    with pytest.raises(ConfigError, match="repeat"):
        load_config(write(tmp_path, VALID.replace("[a, b], evidence", "[a, a], evidence")))


def test_bad_provider_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unsupported provider"):
        load_config(write(tmp_path, VALID.replace("{model: tag-a}", "{model: tag-a, provider: nope}")))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "missing.yaml"))


def test_chain_never_repeats_model(tmp_path):
    config = load_config(write(tmp_path, VALID))
    assert config.chain_for("a") == ["a", "b"]
    assert config.chain_for("b") == ["b", "a"]


def test_describe_has_no_secrets(tmp_path):
    info = load_config(write(tmp_path, VALID)).describe()
    assert "api_key" not in str(info).lower()
    assert info["planner"]["model"] == "tag-a"


def test_shipped_configs_load():
    for name in ("config/models.yaml", "config/models.cloud.yaml"):
        config = load_config(name)
        assert config.roles.synthesizers
