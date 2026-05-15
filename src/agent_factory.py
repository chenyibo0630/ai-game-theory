"""Construct live :class:`LLMAgent` instances from :class:`AgentSpec`.

Provider-class registry is data-driven. To support a new provider just add
an entry to ``PROVIDER_CLASSES`` (or pass ``provider="openai_compatible"``
plus a custom ``base_url`` for any OpenAI-API-compatible endpoint).
"""

from __future__ import annotations

from typing import Mapping

from .agent_spec import AgentSpec, attach_file_logger
from .agents.llm.anthropic_agent import AnthropicAgent
from .agents.llm.deepseek_agent import DeepSeekAgent
from .agents.llm.google_agent import GoogleAgent
from .agents.llm.llm_base import LLMAgent, LLMConfig
from .agents.llm.openai_agent import OpenAIAgent
from .agents.llm.qwen_agent import QwenAgent


PROVIDER_CLASSES: dict[str, type[LLMAgent]] = {
    "openai": OpenAIAgent,
    "openai_compatible": OpenAIAgent,
    "anthropic": AnthropicAgent,
    "google": GoogleAgent,
    "deepseek": DeepSeekAgent,
    "qwen": QwenAgent,
}


def build_agent(spec: AgentSpec, *, world_facts: Mapping[str, object]) -> LLMAgent:
    """Turn one AgentSpec into a configured LLMAgent."""
    cls = PROVIDER_CLASSES.get(spec.provider)
    if cls is None:
        raise ValueError(
            f"unknown provider {spec.provider!r}. "
            f"Known: {sorted(PROVIDER_CLASSES)}"
        )

    config = LLMConfig(
        model=spec.model,
        api_key=spec.api_key,
        base_url=spec.base_url,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        timeout=spec.timeout,
        **world_facts,  # matching_mode, num_traders, initial_*, amm_*, total_rounds
    )

    agent = cls(spec.agent_id, config, display_name=spec.display_name)
    if spec.log_file:
        attach_file_logger(spec.agent_id, spec.log_file)
    return agent
