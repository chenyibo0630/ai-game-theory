"""Tests for the storage abstraction. MySQL is not exercised here — it's
covered end-to-end by the docker-compose stack."""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.baseline import RandomAgent, BuyAndHoldAgent
from src.arena.world import World, WorldConfig
from src.storage import JsonFileSink, NullSink, RunAgent, RunStart


def _meta() -> RunStart:
    return RunStart(
        run_id="abc-123",
        total_rounds=5,
        initial_coin=100.0,
        initial_shares=10,
        matching_mode="amm",
        amm_coin_reserve=1000.0,
        amm_share_reserve=100,
    )


def _agents() -> list[RunAgent]:
    return [
        RunAgent(run_id="abc-123", agent_id="a", display_name="Alice", provider="baseline"),
        RunAgent(run_id="abc-123", agent_id="b", display_name="Bob", provider="baseline"),
    ]


def test_null_sink_is_silent():
    from src.arena.types import RoundReport

    sink = NullSink()
    sink.start_run(_meta(), _agents())
    sink.record_round(
        "abc-123",
        RoundReport(round_index=0, opening_price=10.0, clearing_price=10.0, cleared_volume=0),
    )
    sink.finish_run("abc-123", 10.5)
    sink.close()  # no exceptions, no side-effects


def test_json_sink_writes_one_file_per_run(tmp_path: Path):
    sink = JsonFileSink(out_dir=tmp_path)
    agents = [
        RandomAgent("a", seed=1),
        BuyAndHoldAgent("b"),
    ]
    config = WorldConfig(total_rounds=3, matching_mode="amm")
    world = World(config, agents)
    result = world.run()

    sink.start_run(_meta(), _agents())
    for report in result.round_reports:
        sink.record_round("abc-123", report)
    sink.finish_run("abc-123", result.final_price)
    sink.close()

    out = tmp_path / "run-abc-123.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["config"]["run_id"] == "abc-123"
    assert len(payload["rounds"]) == 3
    assert len(payload["agents"]) == 2
    assert payload["final_price"] == result.final_price


def test_json_sink_record_round_before_start_raises(tmp_path: Path):
    sink = JsonFileSink(out_dir=tmp_path)
    import pytest

    with pytest.raises(RuntimeError, match="before start_run"):
        from src.arena.types import RoundReport

        report = RoundReport(round_index=0, opening_price=10.0,
                             clearing_price=10.0, cleared_volume=0)
        sink.record_round("never-started", report)


def test_json_sink_finish_run_before_start_raises(tmp_path: Path):
    sink = JsonFileSink(out_dir=tmp_path)
    import pytest

    with pytest.raises(RuntimeError, match="before start_run"):
        sink.finish_run("never-started", 10.0)


def test_mysql_sink_import_available():
    """The MySQL sink should at least be importable in the test environment."""
    from src.storage import MySQLSink

    sink = MySQLSink(host="x", user="x", password="x", database="x")
    # We don't connect — just verify the object constructs without touching net.
    assert sink._conn is None
    sink.close()  # idempotent on un-opened connection
