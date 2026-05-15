"""Construct live agents from :class:`AgentSpec`.

Provider-class registry is data-driven. Two registry families:

- ``LLM_PROVIDERS`` — providers that talk to a remote model endpoint.
- ``LOCAL_PROVIDERS`` — baseline agents (random, momentum, …) that need no
  api_key and no LLM config; they're built directly from the spec.

To add a new LLM provider, register it in ``LLM_PROVIDERS``. To add a new
local baseline, register it in ``LOCAL_PROVIDERS`` and (in
``agent_spec._LOCAL_PROVIDERS``) so the YAML loader allows missing
api_key/model.
"""

from __future__ import annotations

from typing import Callable, Mapping

from .agent_spec import AgentSpec, attach_file_logger
from .agents.base import Agent
from .agents.baseline import (
    BuyAndHoldAgent,
    MeanReversionAgent,
    MockAgent,
    MomentumAgent,
    RandomAgent,
)
from .agents.llm.anthropic_agent import AnthropicAgent
from .agents.llm.deepseek_agent import DeepSeekAgent
from .agents.llm.google_agent import GoogleAgent
from .agents.llm.llm_base import LLMAgent, LLMConfig
from .agents.llm.openai_agent import OpenAIAgent
from .agents.llm.qwen_agent import QwenAgent


LLM_PROVIDERS: dict[str, type[LLMAgent]] = {
    "openai": OpenAIAgent,
    "openai_compatible": OpenAIAgent,
    "anthropic": AnthropicAgent,
    "google": GoogleAgent,
    "deepseek": DeepSeekAgent,
    "qwen": QwenAgent,
}


# Each builder receives the AgentSpec and returns a ready-to-use Agent.
# Kept simple — the local agents take very few options.
def _build_mock(spec: AgentSpec) -> Agent:
    return MockAgent(spec.agent_id, display_name=spec.display_name)


def _build_random(spec: AgentSpec) -> Agent:
    return RandomAgent(spec.agent_id, display_name=spec.display_name)


def _build_momentum(spec: AgentSpec) -> Agent:
    return MomentumAgent(spec.agent_id, display_name=spec.display_name)


def _build_mean_reversion(spec: AgentSpec) -> Agent:
    return MeanReversionAgent(spec.agent_id, display_name=spec.display_name)


def _build_buy_and_hold(spec: AgentSpec) -> Agent:
    return BuyAndHoldAgent(spec.agent_id, display_name=spec.display_name)


LOCAL_PROVIDERS: dict[str, Callable[[AgentSpec], Agent]] = {
    "mock": _build_mock,
    "random": _build_random,
    "momentum": _build_momentum,
    "mean_reversion": _build_mean_reversion,
    "buy_and_hold": _build_buy_and_hold,
}


def build_agent(spec: AgentSpec, *, world_facts: Mapping[str, object]) -> Agent:
    """Turn one AgentSpec into a configured agent (LLM or local baseline)."""
    local_builder = LOCAL_PROVIDERS.get(spec.provider)
    if local_builder is not None:
        return local_builder(spec)

    cls = LLM_PROVIDERS.get(spec.provider)
    if cls is None:
        raise ValueError(
            f"unknown provider {spec.provider!r}. Known LLM providers: "
            f"{sorted(LLM_PROVIDERS)}; local: {sorted(LOCAL_PROVIDERS)}"
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
