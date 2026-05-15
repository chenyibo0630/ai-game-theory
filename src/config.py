"""Configuration loader. Reads env vars (or .env) to construct LLM agents."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .agents.llm.llm_base import LLMConfig


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    env_key: str
    env_model: str
    default_model: str


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec("openai", "OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini"),
    ProviderSpec("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
    ProviderSpec("google", "GOOGLE_API_KEY", "GOOGLE_MODEL", "gemini-1.5-flash"),
    ProviderSpec("deepseek", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-chat"),
    ProviderSpec("qwen", "DASHSCOPE_API_KEY", "QWEN_MODEL", "qwen-plus"),
)


def load_env() -> None:
    """Lazy-import dotenv so the module is importable without it installed."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def llm_config_for(
    spec: ProviderSpec,
    *,
    matching_mode: str = "amm",
    num_traders: int = 5,
    initial_coin: float = 100.0,
    initial_shares: int = 10,
    total_rounds: int = 100,
    amm_coin_reserve: float = 1000.0,
    amm_share_reserve: int = 100,
) -> LLMConfig | None:
    api_key = os.environ.get(spec.env_key, "").strip()
    if not api_key:
        return None
    model = os.environ.get(spec.env_model, "").strip() or spec.default_model
    return LLMConfig(
        model=model,
        api_key=api_key,
        matching_mode=matching_mode,
        num_traders=num_traders,
        initial_coin=initial_coin,
        initial_shares=initial_shares,
        total_rounds=total_rounds,
        amm_coin_reserve=amm_coin_reserve,
        amm_share_reserve=amm_share_reserve,
    )
