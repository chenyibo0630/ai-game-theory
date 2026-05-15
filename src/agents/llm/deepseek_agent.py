"""DeepSeek agent — uses OpenAI-compatible API at api.deepseek.com."""

from __future__ import annotations

from dataclasses import replace

from .llm_base import LLMConfig
from .openai_agent import OpenAIAgent


class DeepSeekAgent(OpenAIAgent):
    provider = "deepseek"

    def __init__(
        self,
        agent_id: str,
        config: LLMConfig,
        *,
        display_name: str | None = None,
    ) -> None:
        if not config.base_url:
            # replace() preserves all the world-state fields (matching_mode,
            # initial_coin, etc.) — they're needed by the system-prompt builder.
            config = replace(config, base_url="https://api.deepseek.com")
        super().__init__(agent_id, config, display_name=display_name or "DeepSeek")
