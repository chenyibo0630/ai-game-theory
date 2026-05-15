"""Google Gemini agent (google-generativeai SDK)."""

from __future__ import annotations

from .llm_base import LLMAgent, LLMConfig


class GoogleAgent(LLMAgent):
    provider = "google"

    def __init__(
        self,
        agent_id: str,
        config: LLMConfig,
        *,
        display_name: str | None = None,
    ) -> None:
        super().__init__(agent_id, config, display_name=display_name or "Google")
        import google.generativeai as genai

        genai.configure(api_key=config.api_key)
        self._model = genai.GenerativeModel(
            config.model,
            generation_config={
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
                "response_mime_type": "application/json",
            },
        )

    def _call_provider(self, system: str, user: str) -> str:
        # Gemini doesn't have a native system role pre-1.5 — prepend it.
        prompt = f"{system}\n\n{user}"
        resp = self._model.generate_content(prompt, request_options={"timeout": self.config.timeout})
        return (resp.text or "").strip()
