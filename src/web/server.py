"""FastAPI app that serves the run-explorer UI.

Endpoints
---------
- ``GET /``                              → static index.html
- ``GET /api/runs``                      → list of run summaries
- ``GET /api/runs/{run_id}``             → run metadata + agents
- ``GET /api/runs/{run_id}/prices``      → [{round, opening_price, clearing_price, volume}]
- ``GET /api/runs/{run_id}/equity``      → {agent_id: [{round, equity, cash, shares}]}
- ``GET /api/runs/{run_id}/decisions``   → all decisions, all rounds
- ``GET /api/runs/{run_id}/trades``      → all trades, all rounds

Connects to MySQL via env vars (`MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`,
`MYSQL_DATABASE`). Returns 503 if MySQL is unreachable.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pymysql
import pymysql.cursors
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("web.server")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AI Game Theory — Run Explorer", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- connection management ----------
#
# Single shared connection guarded by a lock + ping(reconnect=True) on every
# acquire. This avoids the "new connection per HTTP request" anti-pattern the
# database reviewer flagged, without pulling in DBUtils.PooledDB. Sufficient
# for a single-worker local-dev FastAPI; promote to a real pool when we
# horizontally scale uvicorn workers.

_conn_lock = threading.Lock()
_conn: pymysql.connections.Connection | None = None


def _connect_params() -> dict[str, Any]:
    return dict(
        host=os.environ.get("MYSQL_HOST", "mysql"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER", "arena"),
        password=os.environ.get("MYSQL_PASSWORD", "arena"),
        database=os.environ.get("MYSQL_DATABASE", "ai_game_theory"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5,
    )


def _get_connection() -> pymysql.connections.Connection:
    global _conn
    if _conn is not None:
        try:
            _conn.ping(reconnect=True)
            return _conn
        except pymysql.MySQLError:
            try:
                _conn.close()
            except Exception:  # pragma: no cover
                pass
            _conn = None
    _conn = pymysql.connect(**_connect_params())
    return _conn


@contextmanager
def _query() -> Iterator:
    try:
        with _conn_lock:
            conn = _get_connection()
            with conn.cursor() as cur:
                yield cur
    except pymysql.MySQLError as exc:
        log.warning("MySQL query failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="Service temporarily unavailable"
        ) from None


def _rows_to_lists(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise Decimal → float for JSON serialisation."""
    out: list[dict[str, Any]] = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if hasattr(v, "as_tuple"):  # Decimal
                clean[k] = float(v)
            else:
                clean[k] = v
        out.append(clean)
    return out


# ---------- routes ----------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    with _query() as cur:
        cur.execute(
            """SELECT run_id, created_at, finished_at, total_rounds,
                      matching_mode, prompt_sha256
                 FROM runs
             ORDER BY created_at DESC
                LIMIT 100"""
        )
        return _rows_to_lists(cur.fetchall())


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    with _query() as cur:
        # Explicit columns — adding fields to `runs` (e.g. internal notes or
        # eventual secrets) must NOT auto-leak through the public API.
        cur.execute(
            """SELECT run_id, created_at, finished_at, total_rounds,
                      initial_coin, initial_shares, matching_mode,
                      amm_coin_reserve, amm_share_reserve, prompt_sha256,
                      seed, notes
                 FROM runs WHERE run_id = %s""",
            (run_id,),
        )
        run = cur.fetchone()
        if not run:
            raise HTTPException(status_code=404, detail="run not found")
        cur.execute(
            "SELECT agent_id, display_name, provider, model FROM run_agents WHERE run_id = %s",
            (run_id,),
        )
        agents = cur.fetchall()
        return {"run": _rows_to_lists([run])[0], "agents": _rows_to_lists(agents)}


@app.get("/api/runs/{run_id}/prices")
def run_prices(run_id: str) -> list[dict[str, Any]]:
    with _query() as cur:
        cur.execute(
            """SELECT round_index, opening_price, clearing_price, cleared_volume,
                      pool_coin, pool_shares
                 FROM rounds
                WHERE run_id = %s
             ORDER BY round_index""",
            (run_id,),
        )
        return _rows_to_lists(cur.fetchall())


@app.get("/api/runs/{run_id}/equity")
def run_equity(run_id: str) -> dict[str, list[dict[str, Any]]]:
    with _query() as cur:
        cur.execute(
            """SELECT round_index, agent_id, cash, shares, equity
                 FROM portfolios
                WHERE run_id = %s
             ORDER BY agent_id, round_index""",
            (run_id,),
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for row in _rows_to_lists(cur.fetchall()):
            out.setdefault(row["agent_id"], []).append(row)
        return out


@app.get("/api/runs/{run_id}/decisions")
def run_decisions(run_id: str) -> list[dict[str, Any]]:
    with _query() as cur:
        cur.execute(
            """SELECT round_index, agent_id, action, quantity, limit_price, rationale
                 FROM decisions
                WHERE run_id = %s
             ORDER BY round_index, agent_id""",
            (run_id,),
        )
        return _rows_to_lists(cur.fetchall())


@app.get("/api/runs/{run_id}/trades")
def run_trades(run_id: str) -> list[dict[str, Any]]:
    with _query() as cur:
        cur.execute(
            """SELECT round_index, buyer_id, seller_id, quantity, price
                 FROM trades
                WHERE run_id = %s
             ORDER BY round_index""",
            (run_id,),
        )
        return _rows_to_lists(cur.fetchall())
