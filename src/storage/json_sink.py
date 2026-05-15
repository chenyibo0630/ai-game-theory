"""JSON-file run sink. One file per run under `runs/`."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..arena.types import RoundReport
from .base import RunAgent, RunStart


@dataclass
class _Buffer:
    meta: RunStart
    agents: list[RunAgent]
    rounds: list[RoundReport] = field(default_factory=list)
    final_price: float | None = None


class JsonFileSink:
    """Writes one ``run-<id>.json`` per run on :meth:`finish_run`."""

    def __init__(self, out_dir: str | Path = "runs") -> None:
        self.out_dir = Path(out_dir)
        self._buffers: dict[str, _Buffer] = {}

    def start_run(self, meta: RunStart, agents: list[RunAgent]) -> None:
        self._buffers[meta.run_id] = _Buffer(meta=meta, agents=list(agents))

    def record_round(self, run_id: str, report: RoundReport) -> None:
        buf = self._buffers.get(run_id)
        if buf is None:
            raise RuntimeError(f"record_round before start_run for {run_id!r}")
        buf.rounds.append(report)

    def finish_run(self, run_id: str, final_price: float) -> None:
        buf = self._buffers.get(run_id)
        if buf is None:
            raise RuntimeError(f"finish_run before start_run for {run_id!r}")
        buf.final_price = final_price
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"run-{run_id}.json"
        payload = {
            "config": asdict(buf.meta),
            "agents": [asdict(a) for a in buf.agents],
            "final_price": final_price,
            "rounds": [r.model_dump() for r in buf.rounds],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        del self._buffers[run_id]

    def close(self) -> None:
        self._buffers.clear()
