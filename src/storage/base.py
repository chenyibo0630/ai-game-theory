"""Sink protocol — what `World` calls into to persist a run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..arena.types import RoundReport


@dataclass(frozen=True)
class RunStart:
    """Metadata recorded once at the start of a simulation."""

    run_id: str
    total_rounds: int
    initial_coin: float
    initial_shares: int
    matching_mode: str
    amm_coin_reserve: float | None
    amm_share_reserve: int | None
    prompt_sha256: str | None = None
    seed: int | None = None
    notes: str | None = None


@dataclass(frozen=True)
class RunAgent:
    """Agent metadata recorded once per run, before any rounds."""

    run_id: str
    agent_id: str
    display_name: str
    provider: str = "unknown"
    model: str | None = None


class RunSink(Protocol):
    """A persistence target for one or more runs.

    Implementations may buffer internally but must flush by the time
    :meth:`finish_run` returns. Calls are NOT thread-safe — `World` invokes
    them on its own thread between rounds.
    """

    def start_run(self, meta: RunStart, agents: list[RunAgent]) -> None: ...

    def record_round(self, run_id: str, report: RoundReport) -> None: ...

    def finish_run(self, run_id: str, final_price: float) -> None: ...

    def close(self) -> None: ...


class NullSink:
    """No-op sink. Useful for tests and dry runs."""

    def start_run(self, meta: RunStart, agents: list[RunAgent]) -> None:
        return None

    def record_round(self, run_id: str, report: RoundReport) -> None:
        return None

    def finish_run(self, run_id: str, final_price: float) -> None:
        return None

    def close(self) -> None:
        return None
