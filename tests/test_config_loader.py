"""Tests for the YAML-driven agent and world loader."""

from __future__ import annotations

import os

import pytest

from src.agent_spec import (
    AgentSpec,
    attach_file_logger,
    resolve_secret,
    spec_from_dict,
    validate_base_url,
)
from src.config_loader import load_config


# -------- secret resolution --------


def test_resolve_literal_string():
    assert resolve_secret("sk-literal") == "sk-literal"


def test_resolve_env_substitution(monkeypatch):
    monkeypatch.setenv("MY_FAKE_KEY", "from-env")
    assert resolve_secret("${MY_FAKE_KEY}") == "from-env"


def test_resolve_env_substitution_missing_raises(monkeypatch):
    monkeypatch.delenv("DOES_NOT_EXIST_KEY", raising=False)
    with pytest.raises(ValueError, match="DOES_NOT_EXIST_KEY"):
        resolve_secret("${DOES_NOT_EXIST_KEY}")


def test_resolve_env_key_fallback(monkeypatch):
    monkeypatch.setenv("FALLBACK_KEY", "fallback-value")
    assert resolve_secret(None, env_key="FALLBACK_KEY") == "fallback-value"


def test_resolve_requires_some_source():
    with pytest.raises(ValueError):
        resolve_secret(None)


# -------- base_url validation --------


def test_validate_base_url_accepts_public_https():
    validate_base_url("https://api.openai.com/v1")


def test_validate_base_url_rejects_http_by_default():
    with pytest.raises(ValueError, match="HTTPS"):
        validate_base_url("http://api.example.com")


def test_validate_base_url_rejects_private_hosts():
    for url in [
        "https://localhost:8000",
        "https://127.0.0.1/api",
        "https://10.0.0.5/v1",
        "https://192.168.1.1/v1",
        "https://169.254.169.254/v1",
        "https://172.17.0.1/v1",
    ]:
        with pytest.raises(ValueError):
            validate_base_url(url)


def test_validate_base_url_allows_insecure_when_opted_in():
    validate_base_url("http://localhost:11434", allow_insecure=True)


def test_validate_base_url_allows_none():
    validate_base_url(None)


# -------- spec_from_dict --------


def test_spec_from_dict_minimal(monkeypatch):
    monkeypatch.setenv("X_KEY", "x")
    spec = spec_from_dict(
        {
            "id": "alice",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "${X_KEY}",
        }
    )
    assert spec.agent_id == "alice"
    assert spec.display_name == "alice"  # defaults to id
    assert spec.provider == "openai"
    assert spec.model == "gpt-4o-mini"
    assert spec.api_key == "x"
    assert spec.temperature == pytest.approx(0.4)


def test_spec_from_dict_repr_masks_key(monkeypatch):
    monkeypatch.setenv("X_KEY", "super-secret-key-do-not-leak")
    spec = spec_from_dict(
        {
            "id": "alice",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "${X_KEY}",
        }
    )
    text = repr(spec)
    assert "super-secret-key" not in text
    assert "***" in text


def test_spec_from_dict_missing_required():
    with pytest.raises(ValueError, match="id"):
        spec_from_dict({"provider": "openai", "model": "x", "api_key": "k"})


# -------- end-to-end YAML loader --------


def test_load_config_parses_world_and_agents(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY_A", "ka")
    monkeypatch.setenv("FAKE_KEY_B", "kb")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        """
world:
  rounds: 25
  matching: amm
  initial_coin: 50
  initial_shares: 5
  amm:
    coin_reserve: 500
    share_reserve: 50

agents:
  - id: a1
    name: "Agent A"
    provider: openai
    model: gpt-4o-mini
    api_key: ${FAKE_KEY_A}
    log_file: logs/a1.log
  - id: a2
    provider: anthropic
    model: claude-haiku-4-5-20251001
    api_key: ${FAKE_KEY_B}
""",
        encoding="utf-8",
    )
    world, specs = load_config(cfg)
    assert world.total_rounds == 25
    assert world.matching_mode == "amm"
    assert world.initial_coin == 50
    assert world.amm_coin_reserve == 500
    assert world.amm_share_reserve == 50
    assert len(specs) == 2
    assert specs[0].agent_id == "a1"
    assert specs[0].display_name == "Agent A"
    assert specs[0].log_file == "logs/a1.log"
    assert specs[0].api_key == "ka"
    assert specs[1].agent_id == "a2"
    assert specs[1].display_name == "a2"  # falls back to id
    assert specs[1].api_key == "kb"


def test_load_config_rejects_duplicate_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE", "x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        """
agents:
  - id: dup
    provider: openai
    model: gpt
    api_key: ${FAKE}
  - id: dup
    provider: anthropic
    model: claude
    api_key: ${FAKE}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_config(cfg)


def test_load_config_requires_at_least_one_agent(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("world: {rounds: 5}\nagents: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one agent"):
        load_config(cfg)


def test_load_config_rejects_unsafe_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE", "x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        """
agents:
  - id: a
    provider: openai
    model: gpt
    api_key: ${FAKE}
    base_url: https://169.254.169.254/v1
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="private"):
        load_config(cfg)


def test_load_config_allow_insecure_unblocks_localhost(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE", "x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        """
security:
  allow_insecure_base_url: true
agents:
  - id: a
    provider: openai_compatible
    model: gpt
    api_key: ${FAKE}
    base_url: http://localhost:8080/v1
""",
        encoding="utf-8",
    )
    _, specs = load_config(cfg)
    assert specs[0].base_url == "http://localhost:8080/v1"


# -------- per-agent file logger --------


def test_attach_file_logger_writes_to_per_agent_file(tmp_path):
    import logging

    log_path = tmp_path / "agent-x.log"
    attach_file_logger("agent-x", str(log_path))
    logging.getLogger("agent.agent-x").info("hello world")
    # Flush all handlers attached to this logger so the file is visible.
    for h in logging.getLogger("agent.agent-x").handlers:
        h.flush()
    assert "hello world" in log_path.read_text(encoding="utf-8")
