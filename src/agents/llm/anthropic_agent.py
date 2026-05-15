"""Anthropic Claude agent.

Auto-detects two credential forms:
- ``sk-ant-api03-...`` → standard API key, uses ``x-api-key`` header.
- ``sk-ant-oat01-...`` → OAuth session token (Claude Code / Claude.ai),
  uses ``Authorization: Bearer`` + the ``oauth-2025-04-20`` beta header,
  and prepends the Claude Code identity block to the system prompt.

OAuth tokens are issued for first-party Anthropic clients. Using them with
custom agents is a ToS gray area — only enable for personal lightweight
testing, and rotate by re-logging into Claude Code if revoked.
"""

from __future__ import annotations

from .llm_base import LLMAgent, LLMConfig

_OAUTH_PREFIX = "sk-ant-oat"
_OAUTH_BETA = "oauth-2025-04-20"
_CLAUDE_CODE_IDENTITY = (
    "You are Claude Code, Anthropic's official CLI for Claude."
)
# Model families that have deprecated the `temperature` request parameter.
# Sending it returns 400 invalid_request_error. Match by substring so any
# minor revision (e.g. dated suffix) is covered.
_NO_TEMPERATURE_MODELS: tuple[str, ...] = ("opus-4-7",)


def _model_rejects_temperature(model: str) -> bool:
    return any(tag in model for tag in _NO_TEMPERATURE_MODELS)


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
        from anthropic import Anthropic  # local import keeps cold-start cheap

        self._is_oauth = (config.api_key or "").startswith(_OAUTH_PREFIX)

        kwargs: dict = {"timeout": config.timeout}
        if self._is_oauth:
            # Bearer auth + the OAuth beta header. The SDK accepts auth_token
            # as the alternative to api_key and emits the right header for us.
            kwargs["auth_token"] = config.api_key
            kwargs["default_headers"] = {"anthropic-beta": _OAUTH_BETA}
        else:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = Anthropic(**kwargs)

    def _call_provider(self, system: str, user: str) -> str:
        if self._is_oauth:
            # OAuth tokens require the request to identify itself as Claude
            # Code. Multi-block system arrays carry the identity claim first
            # and the actual game prompt second.
            system_arg = [
                {"type": "text", "text": _CLAUDE_CODE_IDENTITY},
                {"type": "text", "text": system},
            ]
        else:
            system_arg = system

        kwargs: dict = {
            "model": self.config.model,
            "system": system_arg,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": self.config.max_tokens,
        }
        # Newer Claude models (Opus 4.7+) deprecate the temperature parameter
        # and reject requests that include it. Only forward it for older models.
        if not _model_rejects_temperature(self.config.model):
            kwargs["temperature"] = self.config.temperature

        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            body = getattr(exc, "body", None) or getattr(exc, "message", "")
            self.log.error("anthropic call failed: %s | body=%s", exc, body)
            raise
        chunks = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                chunks.append(block.text)
        return "".join(chunks).strip()
