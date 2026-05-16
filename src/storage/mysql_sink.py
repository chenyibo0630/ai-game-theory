"""MySQL run sink. Mirrors `db/schema.sql`.

Lazily imports PyMySQL so the module remains importable when the optional
dependency isn't installed.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from ..arena.types import RoundReport
from .base import RunAgent, RunStart

log = logging.getLogger("storage.mysql")


class MySQLSink:
    """Persist a run to MySQL row-by-row, using batched INSERTs.

    Connection parameters follow the standard PyMySQL kwargs. The connection
    is opened lazily on first use and held open until :meth:`close`.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int = 3306,
        user: str,
        password: str,
        database: str,
        charset: str = "utf8mb4",
    ) -> None:
        self._connect_kwargs = dict(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset=charset,
            autocommit=False,
        )
        self._conn = None

    # ---- protocol methods ----

    def start_run(self, meta: RunStart, agents: list[RunAgent]) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO runs (run_id, total_rounds, initial_coin, initial_shares,
                                     matching_mode, amm_coin_reserve, amm_share_reserve,
                                     prompt_sha256, seed, notes)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    meta.run_id,
                    meta.total_rounds,
                    meta.initial_coin,
                    meta.initial_shares,
                    meta.matching_mode,
                    meta.amm_coin_reserve,
                    meta.amm_share_reserve,
                    meta.prompt_sha256,
                    meta.seed,
                    meta.notes,
                ),
            )
            if agents:
                cur.executemany(
                    """INSERT INTO run_agents (run_id, agent_id, display_name, provider, model)
                       VALUES (%s, %s, %s, %s, %s)""",
                    [
                        (a.run_id, a.agent_id, a.display_name, a.provider, a.model)
                        for a in agents
                    ],
                )

    def record_round(self, run_id: str, report: RoundReport) -> None:
        pool_coin = report.pool_coin
        pool_shares = report.pool_shares

        # Build a single agent_id → Order lookup so the decisions write is O(N)
        # rather than O(N²) per round, and so an agent that somehow submitted
        # two orders in the same round won't be silently truncated.
        order_by_agent: dict[str, list] = {}
        for order in report.orders:
            order_by_agent.setdefault(order.agent_id, []).append(order)

        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO rounds (run_id, round_index, opening_price,
                                       clearing_price, cleared_volume,
                                       pool_coin, pool_shares)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    run_id,
                    report.round_index,
                    report.opening_price,
                    report.clearing_price,
                    report.cleared_volume,
                    pool_coin,
                    pool_shares,
                ),
            )
            if report.orders:
                cur.executemany(
                    """INSERT INTO orders (run_id, round_index, agent_id, side,
                                            quantity, limit_price)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    [
                        (run_id, o.round_index, o.agent_id, o.side.value,
                         o.quantity, o.limit_price)
                        for o in report.orders
                    ],
                )
            if report.trades:
                cur.executemany(
                    """INSERT INTO trades (run_id, round_index, buyer_id,
                                            seller_id, quantity, price)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    [
                        (run_id, t.round_index, t.buyer_id, t.seller_id,
                         t.quantity, t.price)
                        for t in report.trades
                    ],
                )
            if report.portfolios:
                cur.executemany(
                    """INSERT INTO portfolios (run_id, round_index, agent_id,
                                                cash, shares, equity)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    [
                        (run_id, report.round_index, p.agent_id, p.coin,
                         p.shares, p.equity(report.clearing_price))
                        for p in report.portfolios
                    ],
                )
            if report.rationales:
                decision_rows = []
                for agent_id, rationale in report.rationales.items():
                    orders = order_by_agent.get(agent_id, [])
                    if orders:
                        order = orders[0]
                        action, qty, limit = order.side.value, order.quantity, order.limit_price
                    else:
                        action, qty, limit = "HOLD", 0, 0.0
                    price_before, price_after = report.decision_prices.get(
                        agent_id, (report.opening_price, report.opening_price)
                    )
                    decision_rows.append(
                        (
                            run_id,
                            report.round_index,
                            agent_id,
                            action,
                            qty,
                            limit,
                            rationale,
                            price_before,
                            price_after,
                        )
                    )
                cur.executemany(
                    """INSERT INTO decisions (run_id, round_index, agent_id,
                                               action, quantity, limit_price,
                                               rationale, price_before,
                                               price_after)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                           rationale = VALUES(rationale),
                           price_before = VALUES(price_before),
                           price_after = VALUES(price_after)""",
                    decision_rows,
                )

    def finish_run(self, run_id: str, final_price: float) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE runs SET finished_at = CURRENT_TIMESTAMP WHERE run_id = %s",
                (run_id,),
            )

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    # ---- internals ----

    @contextmanager
    def _cursor(self) -> Iterator:
        conn = self._connect()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _connect(self):
        import pymysql  # lazy import — optional dep

        if self._conn is not None:
            try:
                # ping(reconnect=True) recovers from server-side timeouts
                # (e.g. wait_timeout) that leave us with a dead socket.
                self._conn.ping(reconnect=True)
                return self._conn
            except pymysql.MySQLError:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

        # Log host + database only — never the password.
        self._conn = pymysql.connect(**self._connect_kwargs)
        log.info(
            "MySQL connected to %s/%s",
            self._connect_kwargs["host"],
            self._connect_kwargs["database"],
        )
        return self._conn
