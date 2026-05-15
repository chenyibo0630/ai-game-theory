"""Pluggable matching engines.

Two engines are bundled:

- :class:`CallAuctionExchange` (default) — batches all orders in a round and
  clears them at a single price that maximizes traded volume. Fair, with no
  time priority within a round.

- :class:`AMMEngine` — Uniswap-V2-style constant-product AMM. Each agent
  trades against a shared pool (`POOL` pseudo-counterparty). No fees. Pool
  reserves are configured up front and persist across rounds.

The :class:`MatchingEngine` ABC defines the contract both implementations
satisfy, so the World can be configured with either at runtime.
"""

from .base import POOL_ID, ClearingResult, MatchingEngine
from .call_auction import CallAuctionExchange
from .amm import AMMEngine

__all__ = [
    "AMMEngine",
    "CallAuctionExchange",
    "ClearingResult",
    "MatchingEngine",
    "POOL_ID",
]
