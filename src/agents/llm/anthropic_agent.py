"""Anthropic Claude agent."""

from __future__ import annotations

from .llm_base import LLMAgent, LLMConfig


class AnthropicAgent(LLMAgent):
    provider = "anthropic"

    def __init__(
        self,
        agent_id: str,
        config: LLMConfig,
        *,
        display_name: str | None = None,
    ) -> None:
        super().__init__(agent_id, config, display_name=display_name or "Anthropic")
        from anthropic import Anthropic

        kwargs = {"api_key": config.api_key, "timeout": config.timeout}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = Anthropic(**kwargs)

    def _call_provider(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.config.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        chunks = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                chunks.append(block.text)
        return "".join(chunks).strip()
