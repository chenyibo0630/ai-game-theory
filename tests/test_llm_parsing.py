"""Tests for the LLM response parser and budget clamp.

These tests do not hit a real provider — they exercise the pure-logic parts of
`llm_base` that decide what to do with model text and how to keep the model
honest about its own portfolio.
"""

from __future__ import annotations

import pytest

from src.agents.base import MarketView
from src.agents.llm.llm_base import LLMAgent, LLMConfig, _parse_decision
from src.arena.types import AgentDecision, PortfolioState


class _StubLLM(LLMAgent):
    """LLMAgent that returns a fixed string instead of calling the network."""

    def __init__(self, agent_id: str, response: str) -> None:
        super().__init__(agent_id, LLMConfig(model="stub", api_key="stub"))
        self._response = response

    def _call_provider(self, system: str, user: str) -> str:  # noqa: D401
        return self._response


def _view(*, coin: float = 100.0, shares: int = 10, price: float = 10.0) -> MarketView:
    return MarketView(
        round_index=0,
        total_rounds=10,
        current_price=price,
        price_history=(price,),
        portfolio=PortfolioState(agent_id="a", coin=coin, shares=shares),
    )


def test_parse_clean_json():
    decision = _parse_decision(
        '{"action": "BUY", "quantity": 3, "limit_price": 11.0, "rationale": "go"}'
    )
    assert decision == AgentDecision(action="BUY", quantity=3, limit_price=11.0, rationale="go")


def test_parse_markdown_wrapped_json():
    text = """Sure! Here's my answer:
    ```json
    {"action": "SELL", "quantity": 2, "limit_price": 9.5, "rationale": "trim"}
    ```
    Hope that helps."""
    decision = _parse_decision(text)
    assert decision.action == "SELL"
    assert decision.quantity == 2
    assert decision.limit_price == 9.5


def test_parse_hold_normalizes_quantity():
    decision = _parse_decision(
        '{"action": "HOLD", "quantity": 5, "limit_price": 10.0, "rationale": "wait"}'
    )
    assert decision.action == "HOLD"
    assert decision.quantity == 0
    assert decision.limit_price == 0.0


def test_parse_zero_quantity_buy_becomes_hold():
    decision = _parse_decision(
        '{"action": "BUY", "quantity": 0, "limit_price": 10.0, "rationale": "skip"}'
    )
    assert decision.action == "HOLD"


def test_parse_invalid_action_raises():
    with pytest.raises(ValueError):
        _parse_decision('{"action": "WIBBLE", "quantity": 1, "limit_price": 1.0}')


def test_parse_no_json_raises():
    with pytest.raises(ValueError):
        _parse_decision("I refuse to comply, sorry.")


def test_clamp_buy_to_budget():
    agent = _StubLLM(
        "a",
        '{"action": "BUY", "quantity": 100, "limit_price": 5.0, "rationale": "greedy"}',
    )
    decision = agent.decide(_view(coin=20.0, shares=0, price=5.0))
    # max affordable at limit 5 is 20/5 = 4
    assert decision.action == "BUY"
    assert decision.quantity == 4


def test_clamp_sell_to_holdings():
    agent = _StubLLM(
        "a",
        '{"action": "SELL", "quantity": 999, "limit_price": 5.0, "rationale": "dump"}',
    )
    decision = agent.decide(_view(coin=0.0, shares=3, price=5.0))
    assert decision.action == "SELL"
    assert decision.quantity == 3


def test_unparseable_response_falls_back_to_hold():
    agent = _StubLLM("a", "I refuse.")
    decision = agent.decide(_view())
    assert decision.action == "HOLD"
    assert "fallback" in decision.rationale


def test_buy_with_no_budget_becomes_hold():
    agent = _StubLLM(
        "a",
        '{"action": "BUY", "quantity": 5, "limit_price": 50.0, "rationale": "broke"}',
    )
    decision = agent.decide(_view(coin=0.1, shares=0, price=50.0))
    assert decision.action == "HOLD"


def test_sell_with_no_shares_becomes_hold():
    agent = _StubLLM(
        "a",
        '{"action": "SELL", "quantity": 5, "limit_price": 10.0, "rationale": "naked"}',
    )
    decision = agent.decide(_view(coin=100.0, shares=0, price=10.0))
    assert decision.action == "HOLD"


def test_prompt_contains_round_cash_shares_and_price():
    """Every per-round prompt must contain the four spec-required fields."""

    captured: list[str] = []

    class _CapturingStub(_StubLLM):
        def _call_provider(self, system: str, user: str) -> str:
            captured.append(user)
            return self._response

    agent = _CapturingStub(
        "a",
        '{"action": "HOLD", "quantity": 0, "limit_price": 0, "rationale": "n/a"}',
    )
    agent.decide(_view(coin=42.5, shares=7, price=11.25))
    prompt = captured[0].lower()
    assert "round:" in prompt
    assert "your cash" in prompt
    assert "42.5" in captured[0]
    assert "your shares" in prompt
    assert "7" in captured[0]
    assert "current market price" in prompt
    assert "11.25" in captured[0]


def test_prompt_contains_public_price_history():
    """Price history is the only past-rounds context shown to the agent."""

    captured: list[str] = []

    class _CapturingStub(_StubLLM):
        def _call_provider(self, system: str, user: str) -> str:
            captured.append(user)
            return self._response

    history = (10.0, 10.5, 10.2, 10.7)
    view = MarketView(
        round_index=4,
        total_rounds=10,
        current_price=10.7,
        price_history=history,
        portfolio=PortfolioState(agent_id="a", coin=50.0, shares=5),
    )
    agent = _CapturingStub(
        "a",
        '{"action": "HOLD", "quantity": 0, "limit_price": 0, "rationale": "n/a"}',
    )
    agent.decide(view)
    prompt = captured[0]
    assert "public price history" in prompt.lower()
    for p in history:
        assert f"{p:.4f}" in prompt


def test_system_prompt_includes_matching_mode_rules():
    """AMM-mode system prompt must mention the AMM mechanics."""

    from src.agents.llm.llm_base import build_system_prompt

    amm_prompt = build_system_prompt("amm")
    assert "amm" in amm_prompt.lower() or "automated market maker" in amm_prompt.lower()
    assert "slippage" in amm_prompt.lower()
    assert "r_c" in amm_prompt.lower() or "constant-product" in amm_prompt.lower()

    auction_prompt = build_system_prompt("call_auction")
    assert "call auction" in auction_prompt.lower()
    assert "clearing" in auction_prompt.lower()
