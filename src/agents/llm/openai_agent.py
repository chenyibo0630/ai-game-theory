"""OpenAI-API-compatible agent (default: gpt-4o-mini).

Reused by DeepSeek and any other provider that speaks the OpenAI chat-completions
protocol — they just override `base_url` and `model`.
"""

from __future__ import annotations

from .llm_base import LLMAgent, LLMConfig


class OpenAIAgent(LLMAgent):
    provider = "openai"

    def __init__(
        self,
        agent_id: str,
        config: LLMConfig,
        *,
        display_name: str | None = None,
    ) -> None:
        super().__init__(agent_id, config, display_name=display_name or "OpenAI")
        from openai import OpenAI  # local import keeps cold-start cheap

        kwargs = {"api_key": config.api_key, "timeout": config.timeout}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = OpenAI(**kwargs)

    def _call_provider(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        return content.strip()
