-- AI Game Theory — MySQL schema (InnoDB, utf8mb4)
--
-- Six tables, all keyed by `run_id` (UUID, generated at start of each
-- simulation). The frontend issues read-only queries by `run_id` to render
-- price/equity curves and per-round detail. The backend appends rows once
-- per round (one INSERT per table per round).
--
-- Schema is intentionally denormalised on string identifiers (agent_id /
-- run_id are short opaque IDs, not foreign keys) so a single MySQL instance
-- can hold many tournaments side by side without cross-run joins.

SET NAMES utf8mb4;
SET time_zone = '+00:00';

-- ---------------------------------------------------------------------------
-- One row per simulation run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    run_id            VARCHAR(64) NOT NULL PRIMARY KEY,
    created_at        DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at       DATETIME    NULL,
    total_rounds      INT         NOT NULL,
    initial_coin      DECIMAL(18, 6) NOT NULL,
    initial_shares    INT         NOT NULL,
    matching_mode     VARCHAR(32) NOT NULL,
    amm_coin_reserve  DECIMAL(18, 6) NULL,
    amm_share_reserve INT         NULL,
    prompt_sha256     CHAR(64)    NULL,
    seed              BIGINT      NULL,
    notes             TEXT        NULL,
    INDEX idx_runs_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

-- ---------------------------------------------------------------------------
-- One row per round per run. `clearing_price` is post-round mark price; for
-- AMM mode the engine reserves are recorded too.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rounds (
    run_id          VARCHAR(64) NOT NULL,
    round_index     INT         NOT NULL,
    opening_price   DECIMAL(18, 6) NOT NULL,
    clearing_price  DECIMAL(18, 6) NOT NULL,
    cleared_volume  INT         NOT NULL,
    pool_coin       DECIMAL(18, 6) NULL,
    pool_shares     INT         NULL,
    PRIMARY KEY (run_id, round_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

-- ---------------------------------------------------------------------------
-- Every order an agent submitted. Side is 'BUY' or 'SELL'.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    id            BIGINT      NOT NULL AUTO_INCREMENT,
    run_id        VARCHAR(64) NOT NULL,
    round_index   INT         NOT NULL,
    agent_id      VARCHAR(64) NOT NULL,
    side          VARCHAR(8)  NOT NULL,
    quantity      INT         NOT NULL,
    limit_price   DECIMAL(18, 6) NOT NULL,
    PRIMARY KEY (id),
    INDEX idx_orders_run_round (run_id, round_index),
    INDEX idx_orders_run_agent (run_id, agent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

-- ---------------------------------------------------------------------------
-- Every fill produced by the matching engine. For AMM, one side is 'POOL'.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trades (
    id            BIGINT      NOT NULL AUTO_INCREMENT,
    run_id        VARCHAR(64) NOT NULL,
    round_index   INT         NOT NULL,
    buyer_id      VARCHAR(64) NOT NULL,
    seller_id     VARCHAR(64) NOT NULL,
    quantity      INT         NOT NULL,
    price         DECIMAL(18, 6) NOT NULL,
    PRIMARY KEY (id),
    INDEX idx_trades_run_round (run_id, round_index),
    INDEX idx_trades_run_buyer (run_id, buyer_id),
    INDEX idx_trades_run_seller (run_id, seller_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

-- ---------------------------------------------------------------------------
-- Portfolio snapshot at the end of every round, every agent.
-- Drives the equity curve in the frontend.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolios (
    run_id      VARCHAR(64) NOT NULL,
    round_index INT         NOT NULL,
    agent_id    VARCHAR(64) NOT NULL,
    cash        DECIMAL(18, 6) NOT NULL,
    shares      INT         NOT NULL,
    equity      DECIMAL(18, 6) NOT NULL,  -- cash + shares * clearing_price
    PRIMARY KEY (run_id, round_index, agent_id),
    INDEX idx_portfolios_run_agent (run_id, agent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

-- ---------------------------------------------------------------------------
-- Agent decisions including the raw rationale (already sanitised by the
-- LLM parser). Used for prompt-version comparison and qualitative review.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS decisions (
    run_id        VARCHAR(64) NOT NULL,
    round_index   INT         NOT NULL,
    agent_id      VARCHAR(64) NOT NULL,
    action        VARCHAR(8)  NOT NULL,
    quantity      INT         NOT NULL,
    limit_price   DECIMAL(18, 6) NOT NULL,
    -- TEXT (not VARCHAR) — the LLM parser caps at 240 chars but model output
    -- could legitimately exceed VARCHAR limits during prompt iteration.
    rationale     TEXT        NOT NULL,
    -- Market price the engine saw immediately BEFORE and AFTER applying this
    -- agent's order. Under AMM these reflect the pool's spot at the agent's
    -- lex-ordered slot; under call_auction both rows of an agent share the
    -- uniform clearing price. HOLD / no-order agents have before == after ==
    -- that round's opening price. Nullable so older runs replay cleanly.
    price_before  DECIMAL(18, 6) NULL,
    price_after   DECIMAL(18, 6) NULL,
    PRIMARY KEY (run_id, round_index, agent_id),
    INDEX idx_decisions_run_agent (run_id, agent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

-- ---------------------------------------------------------------------------
-- Per-run agent registry — display names and providers for the frontend.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS run_agents (
    run_id        VARCHAR(64) NOT NULL,
    agent_id      VARCHAR(64) NOT NULL,
    display_name  VARCHAR(255) NOT NULL,
    provider      VARCHAR(64) NOT NULL DEFAULT 'unknown',
    model         VARCHAR(128) NULL,
    PRIMARY KEY (run_id, agent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
