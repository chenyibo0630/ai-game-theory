"""Identity + provider config for one trader.

This dataclass decouples *who an agent is* (id, display name, log destination)
from *what provider it uses* (api_key, base_url, model). Multiple agents may
share the same provider — e.g. two different OpenAI agents at different
temperatures — by simply giving them different `agent_id`s.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


@dataclass(frozen=True)
class AgentSpec:
    """All the per-agent knobs config.yaml can set."""

    agent_id: str
    display_name: str
    provider: str
    model: str
    api_key: str  # already resolved from env / literal
    base_url: str | None = None
    temperature: float = 0.4
    max_tokens: int = 256
    timeout: float = 30.0
    log_file: str | None = None

    def __repr__(self) -> str:  # never reveal the api_key
        return (
            f"AgentSpec(id={self.agent_id!r}, name={self.display_name!r}, "
            f"provider={self.provider!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key='***')"
        )


# -------- secret + URL resolution --------

_ENV_REF = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")
# Block obvious internal-network targets to limit SSRF blast radius when
# config.yaml is shared / committed. Public LLM providers always live on
# routable IPs, so this is rarely a usability obstacle.
_PRIVATE_HOST = re.compile(
    r"^(localhost|127\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.)",
    re.IGNORECASE,
)


def resolve_secret(raw: str | None, *, env_key: str | None = None) -> str:
    """Resolve an api_key value.

    Accepts:
    - a literal string (used as-is — note this is plaintext in config.yaml)
    - "${ENV_VAR}" syntax → looked up in os.environ
    - `env_key` fallback if the literal field is missing
    """
    if raw:
        match = _ENV_REF.match(raw)
        if match:
            env_name = match.group(1)
            value = os.environ.get(env_name, "").strip()
            if not value:
                raise ValueError(f"env var {env_name} required but not set")
            return value
        return raw
    if env_key:
        value = os.environ.get(env_key, "").strip()
        if not value:
            raise ValueError(f"env var {env_key} required but not set")
        return value
    raise ValueError("no api_key value or env reference provided")


def validate_base_url(url: str | None, *, allow_insecure: bool = False) -> None:
    """Reject blatantly unsafe base_urls before they reach the SDK.

    Set `allow_insecure=True` if you intentionally point at a localhost dev
    server. Default refuses any private/loopback target.
    """
    if url is None:
        return
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"}:
        raise ValueError(f"base_url scheme must be http(s), got {url!r}")
    if parsed.scheme == "http" and not allow_insecure:
        raise ValueError(f"base_url must use HTTPS unless allow_insecure=true: {url!r}")
    host = parsed.hostname or ""
    if not allow_insecure and _PRIVATE_HOST.match(host):
        raise ValueError(f"base_url points at a private/internal host: {url!r}")


# -------- per-agent file logger --------

_attached_handlers: set[tuple[str, str]] = set()
"""(logger_name, abs_log_path) pairs we've already attached, to keep idempotent
calls (e.g. tournament re-runs) from duplicating handlers."""


def attach_file_logger(agent_id: str, log_file: str) -> None:
    """Attach a per-agent file handler so prompts/responses/decisions for this
    agent flow to its own file. Safe to call multiple times."""
    log_path = Path(log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger_name = f"agent.{agent_id}"
    key = (logger_name, str(log_path))
    if key in _attached_handlers:
        return

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(handler)
    _attached_handlers.add(key)


# -------- entrypoint used by the YAML loader --------


# Providers that do not call any external LLM endpoint and therefore need
# no api_key / model. Adding "mock" here tells spec_from_dict to relax the
# normal credential-required validation for these baseline agents.
_LOCAL_PROVIDERS: frozenset[str] = frozenset({"mock", "random", "momentum",
                                              "mean_reversion", "buy_and_hold"})


def spec_from_dict(entry: Mapping[str, Any], *, allow_insecure_url: bool = False) -> AgentSpec:
    """Build an AgentSpec from one YAML entry. Raises ValueError on bad input."""
    try:
        agent_id = str(entry["id"])
        provider = str(entry["provider"])
    except KeyError as missing:
        raise ValueError(f"agent entry missing required key: {missing}") from None

    is_local = provider in _LOCAL_PROVIDERS
    # Local baseline agents have no model concept — fall back to the provider
    # name so AgentSpec.model still has a non-empty string for logging/UI.
    model = str(entry.get("model", provider if is_local else None) or "")
    if not model:
        raise ValueError(f"agent entry missing required key: 'model'")

    raw_key = entry.get("api_key")
    env_key = entry.get("api_key_env")
    if is_local and not raw_key and not env_key:
        api_key = ""
    else:
        api_key = resolve_secret(raw_key if isinstance(raw_key, str) else None, env_key=env_key)

    base_url = entry.get("base_url")
    if base_url is not None:
        validate_base_url(base_url, allow_insecure=allow_insecure_url)

    return AgentSpec(
        agent_id=agent_id,
        display_name=str(entry.get("name", agent_id)),
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(entry.get("temperature", 0.4)),
        max_tokens=int(entry.get("max_tokens", 256)),
        timeout=float(entry.get("timeout", 30.0)),
        log_file=entry.get("log_file"),
    )
