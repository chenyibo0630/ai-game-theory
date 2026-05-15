"""Load a complete simulation from a single YAML file.

The YAML schema is documented in `config.yaml.example`. This module exposes
one main entry point: :func:`load_config` which returns a fully-constructed
``WorldConfig`` plus a list of ``AgentSpec`` objects.

Only :func:`yaml.safe_load` is used — arbitrary Python object construction is
NEVER allowed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .agent_spec import AgentSpec, spec_from_dict
from .arena.world import WorldConfig


def load_config(path: str | Path) -> tuple[WorldConfig, list[AgentSpec]]:
    """Parse `path` as YAML and return (WorldConfig, [AgentSpec, ...])."""
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"config file not found: {raw_path}")

    data = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a YAML mapping, got {type(data).__name__}")

    world = _parse_world(data.get("world", {}))
    allow_insecure = bool((data.get("security") or {}).get("allow_insecure_base_url", False))
    agents = _parse_agents(data.get("agents", []), allow_insecure_url=allow_insecure)

    if not agents:
        raise ValueError("config.yaml must define at least one agent")

    return world, agents


def _parse_world(section: Any) -> WorldConfig:
    if not isinstance(section, dict):
        raise ValueError("`world` section must be a mapping")

    base = WorldConfig()
    overrides: dict[str, Any] = {}

    field_map = {
        "rounds": "total_rounds",
        "total_rounds": "total_rounds",
        "initial_coin": "initial_coin",
        "initial_shares": "initial_shares",
        "initial_price": "initial_price",
        "matching": "matching_mode",
        "matching_mode": "matching_mode",
        "parallel_decisions": "parallel_decisions",
        "decision_workers": "decision_workers",
    }
    for yaml_key, world_key in field_map.items():
        if yaml_key in section:
            overrides[world_key] = section[yaml_key]

    amm_section = section.get("amm") or {}
    if isinstance(amm_section, dict):
        if "coin_reserve" in amm_section:
            overrides["amm_coin_reserve"] = amm_section["coin_reserve"]
        if "share_reserve" in amm_section:
            overrides["amm_share_reserve"] = amm_section["share_reserve"]

    return replace(base, **overrides)


def _parse_agents(section: Any, *, allow_insecure_url: bool) -> list[AgentSpec]:
    if not isinstance(section, list):
        raise ValueError("`agents` must be a list")
    specs: list[AgentSpec] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(section):
        if not isinstance(entry, dict):
            raise ValueError(f"agent #{i} must be a mapping")
        spec = spec_from_dict(entry, allow_insecure_url=allow_insecure_url)
        if spec.agent_id in seen_ids:
            raise ValueError(f"duplicate agent id: {spec.agent_id!r}")
        seen_ids.add(spec.agent_id)
        specs.append(spec)
    return specs
