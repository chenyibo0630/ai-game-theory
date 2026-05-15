"""Qwen (Alibaba DashScope) agent — uses OpenAI-compatible mode."""

from __future__ import annotations

from dataclasses import replace

from .llm_base import LLMConfig
from .openai_agent import OpenAIAgent


class QwenAgent(OpenAIAgent):
    provider = "qwen"

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        agent_id: str,
        config: LLMConfig,
        *,
        display_name: str | None = None,
    ) -> None:
        if not config.base_url:
            config = replace(config, base_url=self.DEFAULT_BASE_URL)
        super().__init__(agent_id, config, display_name=display_name or "Qwen")
