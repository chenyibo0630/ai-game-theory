"""Shared scaffolding for LLM-backed trading agents.

Every LLM agent receives the SAME engine-specific system prompt (world rules
+ win goal + allowed operations + matching-engine details). Each round the
user message tells the agent:

- the current round number (and total rounds)
- its own cash holdings
- its own share holdings
- the current market price
- the public price history of the game so far

The agent's current portfolio already encodes the result of its past trades,
so we deliberately do NOT pass per-trade memory: the current state plus the
public price trajectory contains all the information needed to act. No other
agent's state is ever revealed.

Subclasses implement `_call_provider(system, user) -> str` returning raw model
text; the base class handles prompt construction, JSON parsing with a regex
fallback, decision validation, and graceful HOLD fallback on any failure.
"""

from __future__ import annotations

import json
import logging
import re
from abc import abstractmethod
from dataclasses import dataclass

from ...arena.types import AgentDecision, Trade
from ..base import Agent, MarketView

log = logging.getLogger("agents.llm")


def _format_rules(
    *,
    num_traders: int,
    initial_coin: float,
    initial_shares: int,
    total_rounds: int,  # accepted but intentionally NOT shown to the agent
) -> str:
    del total_rounds  # 故意不告诉 agent 总轮数 —— 见下方"长周期"说明
    return f"""你是一个博弈论专家。

【世界规则】
- 共有 {num_traders} 名交易员相互竞争。市场中只有一只可交易股票；只有一种货币，名为 GameCoin（GC）。
- 首先需要深入理解交易的核心机制和算法。
- 每名交易员初始持有 {initial_coin:g} GC 和 {initial_shares} 股
  （因此初始个人净值 = 初始价格 × 持股数 + 现金）。
- 每一轮每名交易员独立地从以下三种动作中选择一个：
    BUY  —— 提交买入限价单（数量 + 可接受的最高价）
    SELL —— 提交卖出限价单（数量 + 可接受的最低价）
    HOLD —— 不操作
- 所有订单在每轮结束时统一揭示并撮合。你可以看到价格的变化，但是看不到其他人的订单、持仓、
  决策或身份。
- 一个订单要么按你提交的数量全额成交，要么本轮 0 成交（不存在部分成交）。

【你的唯一目标 —— 通过巧妙的交易成为这个世界中财富最多的人】
- 你的目标是理解交易的规则，利用低价买入高价卖出，在游戏中持续快速增长你的总财富，成为博弈论世界的首富（现金 + 持股市值）。把每一轮都看作一盘长棋中的一手。

【输出格式（严格 JSON，单个对象，不要 markdown、不要任何额外说明）】
{{
  "action": "BUY" | "SELL" | "HOLD",
  "quantity": <整数，HOLD 时为 0>,
  "limit_price": <数字，HOLD 时为 0>,
  "rationale": "<一句简短的中文理由>"
}}

【约束】
- BUY：quantity * limit_price 必须 ≤ 你当前的现金。
- SELL：quantity 必须 ≤ 你当前的持股数。
- quantity 必须是非负整数；limit_price 必须为正数。
"""


def _format_amm(
    *,
    num_traders: int,
    initial_coin: float,
    initial_shares: int,
    amm_coin_reserve: float,
    amm_share_reserve: int,
) -> str:
    spot = amm_coin_reserve / amm_share_reserve
    total_cash = num_traders * initial_coin + amm_coin_reserve
    total_shares = num_traders * initial_shares + amm_share_reserve
    return f"""
【撮合引擎】Uniswap V2 风格的自动做市商（AMM），同轮内逐单撮合、顺序伪随机。零手续费。

【初始状态】
- 流动池初始储备：R_c = {amm_coin_reserve:g} GC，R_s = {amm_share_reserve} 股。
- 初始现货价 = R_c / R_s = {spot:.4f} GC/股。
- 整局游戏的系统总量恒定：
    现金：{total_cash:g} GC = {num_traders} 名交易员 × {initial_coin:g} + 流动池 {amm_coin_reserve:g}
    股票：{total_shares}    = {num_traders} 名交易员 × {initial_shares} + 流动池 {amm_share_reserve}

【核心机制】
- 所有交易对手都是共享流动池 —— 交易员之间不存在直接撮合，流动池是每一笔
  成交的隐含对手方。
- 当前市场价 = R_c / R_s。
- 恒定乘积不变量：每一笔成交后 R_c · R_s 都保持不变（仅有整数股的微小取整）。
- 同一轮内的所有订单**按 SHA-256(round_index, agent_id) 派生的顺序**逐单对
  池子撮合 —— 这是按轮次种子的伪随机置换：可复盘、跨轮不偏。前一单会更新
  池子状态、影响后一单的有效价格，但所有订单同时密封提交，没人能事先知道
  自己这一轮排在第几位，无法据此抢跑。

【单笔成交公式】对当前 (R_c, R_s)：
    BUY  q 股的有效均价 = R_c / (R_s − q)
    SELL q 股的有效均价 = R_c / (R_s + q)
成交后：
    BUY:  R_c ← R_c + q · R_c/(R_s−q)，R_s ← R_s − q
    SELL: R_c ← R_c − q · R_c/(R_s+q)，R_s ← R_s + q

【limit_price 与部分成交】
- limit_price 是你能接受的最差有效均价。
- 引擎在限价约束下计算最大可成交数量 q_max：
    BUY:  q ≤ R_s − R_c / limit_price
    SELL: q ≤ R_c / limit_price − R_s
- 若 q_max ≥ 你的下单量：全额成交。
- 若 0 < q_max < 你的下单量：**部分成交** q_max 股，剩余作废。
- 若 q_max ≤ 0：本单完全不成交。
- 池子上限保护：BUY 还会被限制为池子留至少 1 股。

【范例 A —— 通过 BUY 抓上涨获利】
假设某轮池子 R_c=500，R_s=50，spot=10.00。你判断接下来几轮会有净买盘推高价格。
- 提交 BUY 5 股，limit_price=11.5
  - q_max = 50 − 500/11.5 = 6.52 → 6，≥ 5 → 全额成交
  - 有效买入均价 = R_c/(R_s−q) = 500/(50−5) = **11.111 GC/股**
  - 付出 5 × 11.111 = 55.56 GC，得到 5 股；池子 → (555.56, 45)，spot=12.35
- 之后 3 轮其他人继续净买入，池子漂到 R_c=600，R_s=42，spot=14.29
- 提交 SELL 5 股，limit_price=12.5
  - q_max = 600/12.5 − 42 = 6 ≥ 5 → 全额成交
  - 有效卖出均价 = R_c/(R_s+q) = 600/(42+5) = **12.766 GC/股**
  - 收回 5 × 12.766 = 63.83 GC

净利 = 63.83 − 55.56 = **+8.27 GC（持仓回到原状）**。
关键：5 股的体量把 spot 的 4.29 GC 上移**真正变成** 1.66 GC/股的实际进出价差；
若只下 1 股，单笔进出仅吃到约 ~0.4 GC，5 次小动作的累计利润远小于单次大单。

【范例 B —— 通过 SELL 抓下跌获利】
假设某轮池子 R_c=700，R_s=40，spot=17.50。你判断价格被推得过高、即将回落。
- 提交 SELL 4 股，limit_price=15
  - q_max = 700/15 − 40 = 6.67 → 6，≥ 4 → 全额成交
  - 有效卖出均价 = R_c/(R_s+q) = 700/(40+4) = **15.909 GC/股**
  - 收回 4 × 15.909 = 63.64 GC，让出 4 股；池子 → (636.36, 44)，spot=14.46
- 之后几轮其他人净卖出，池子漂到 R_c=560，R_s=48，spot=11.67
- 提交 BUY 4 股，limit_price=13
  - q_max = 48 − 560/13 = 4.92 → 4，≥ 4 → 全额成交
  - 有效买入均价 = R_c/(R_s−q) = 560/(48−4) = **12.727 GC/股**
  - 付出 4 × 12.727 = 50.91 GC，拿回 4 股

净利 = 63.64 − 50.91 = **+12.73 GC（持仓回到原状）**。
同样的逻辑：单笔够大，5.83 GC 的 spot 回落才被转化成 3.18 GC/股的实际价差利润。

【从范例中读出的关键】
- **滑点是确定的、可精确计算**：拿当前 R_c, R_s, q 套公式就能算出有效价、付/收金额。
  下单前**必须**先估算："这笔的 GC 成本 vs 我预期的 spot 移动够不够 cover？"
- **单笔下手过小是隐形亏损**：滑点对小单的比例确实更小（凸函数），
  但同样的方向判断下，5 股能拿到 ~8 GC 利润，5 次 1 股加起来只有 ~2 GC ——
  因为大单同时建立了更大的方向性敞口。规模本身就是利润。
- **限价是滑点保护、不是定价目标**：BUY 的 limit 越宽松成交越确定，
  实际付出的有效价由公式决定，通常远好于限价。设过紧会被部分成交或全单作废。

"""


def _format_call_auction(*, num_traders: int) -> str:
    return f"""
【撮合引擎】集合竞价（批量撮合，全 or 不成）。

【核心机制】
- 每轮收集全部 {num_traders} 名交易员的订单，按"使成交量最大化"的统一
  出清价 P* 一次性出清。
- 所有 limit_price ≥ P* 的 BUY 订单**全额**按 P* 成交。
- 所有 limit_price ≤ P* 的 SELL 订单**全额**按 P* 成交。
- 不存在"部分成交"——若某一侧总量超出对手侧总量，按优先级（买价从高到
  低、卖价从低到高，再按 agent_id）逐单剔除最低优先级订单，直到两侧总
  量相等为止。被剔除的订单本轮 0 成交。
- 同一轮内时间先后没有任何优先级。
- 未成交订单在本轮结束时撤销（不留挂单簿）。

【破平规则】
- 若多个价格产生相同的最大成交量，引擎优先选 |需求 − 供给| 最小的那个，
  再优先选与上一轮出清价最接近的那个。

【策略含义】
- 你的 limit_price 是"是否参与 + 优先级排序"的依据：买价越高、卖价越低，
  越不容易被剔除。
- 一旦入选，就按 P* 全额成交；不存在"成交一半"的情况。
- 当供需大体平衡时，出清价会锚定在上一轮价格附近 —— 想打破这个锚需要
  明显的方向性单边流量。
"""


def build_system_prompt(
    matching_mode: str,
    *,
    num_traders: int = 5,
    initial_coin: float = 100.0,
    initial_shares: int = 10,
    total_rounds: int = 100,
    amm_coin_reserve: float = 1000.0,
    amm_share_reserve: int = 100,
) -> str:
    """Compose the engine-specific system prompt with concrete world values."""
    rules = _format_rules(
        num_traders=num_traders,
        initial_coin=initial_coin,
        initial_shares=initial_shares,
        total_rounds=total_rounds,
    )
    mode = matching_mode.lower()
    if mode == "amm":
        return rules + _format_amm(
            num_traders=num_traders,
            initial_coin=initial_coin,
            initial_shares=initial_shares,
            amm_coin_reserve=amm_coin_reserve,
            amm_share_reserve=amm_share_reserve,
        )
    if mode == "call_auction":
        return rules + _format_call_auction(num_traders=num_traders)
    raise ValueError(f"unknown matching_mode: {matching_mode}")


# Convenience default for ad-hoc tests / smoke checks that don't go through
# LLMConfig — uses the default AMM world parameters.
SYSTEM_PROMPT = build_system_prompt("amm")


def _fmt_price_history(history: tuple[float, ...], window: int = 30) -> str:
    """Format the recent slice of public clearing prices for the prompt.

    `view.price_history` always starts with the engine's initial price (a
    PriceTick recorded before round 0). We show up to `window` most-recent
    entries, oldest first.
    """
    if not history:
        return "（暂无历史轮次）"
    tail = history[-window:]
    return ", ".join(f"{p:.4f}" for p in tail)


def _build_user_prompt(view: MarketView) -> str:
    history_str = _fmt_price_history(view.price_history)
    pool_line = ""
    if view.pool_reserves is not None:
        r_c, r_s = view.pool_reserves
        pool_line = f"流动池：R_c = {r_c:.4f} GC，R_s = {r_s} 股\n"

    return (
        # 每轮用户消息：轮次序号（只是计数器 —— 总轮数故意隐藏）、自己的现金、
        # 自己的持股、当前价格，以及 AMM 引擎下的实时流动池状态。
        f"=== 本轮信息 ===\n"
        f"轮次序号：{view.round_index + 1}\n"
        f"你的现金：{view.portfolio.coin:.4f} GC\n"
        f"你的持股：{view.portfolio.shares} 股\n"
        f"当前市场价：{view.current_price:.4f} GC\n"
        f"{pool_line}"
        "\n"
        f"=== 公开价格历史（由旧到新）===\n"
        f"{history_str}\n\n"
        "请按规定输出一个 JSON 对象作为回复。"
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _sanitize_rationale(raw: object) -> str:
    """Strip control characters and clamp length so untrusted model output
    can't break terminal rendering or downstream parsers."""
    return _CTRL_RE.sub(" ", str(raw))[:240]


def _parse_decision(text: str) -> AgentDecision:
    """Extract a JSON object from a model response and validate it."""
    if not text:
        raise ValueError("empty response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if not match:
            raise ValueError(f"no JSON object found in response: {text!r}")
        payload = json.loads(match.group(0))

    action = str(payload.get("action", "HOLD")).upper()
    if action not in {"BUY", "SELL", "HOLD"}:
        raise ValueError(f"invalid action: {action!r}")

    quantity = int(payload.get("quantity", 0) or 0)
    limit_price = float(payload.get("limit_price", 0) or 0)
    rationale = _sanitize_rationale(payload.get("rationale", ""))

    if action == "HOLD":
        quantity = 0
        limit_price = 0.0
    if quantity <= 0 or limit_price <= 0:
        action = "HOLD"
        quantity = 0
        limit_price = 0.0

    return AgentDecision(
        action=action,  # type: ignore[arg-type]
        quantity=quantity,
        limit_price=limit_price,
        rationale=rationale,
    )


@dataclass
class LLMConfig:
    """Per-agent provider config plus the world parameters needed to write a
    well-grounded system prompt (initial endowments, pool reserves, etc.).

    `api_key` is intentionally masked in __repr__ so accidental logging or
    serialisation of this dataclass cannot leak credentials.
    """

    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.4
    max_tokens: int = 256
    timeout: float = 30.0

    # World facts that go into the system prompt.
    matching_mode: str = "amm"
    num_traders: int = 5
    initial_coin: float = 100.0
    initial_shares: int = 10
    total_rounds: int = 100
    amm_coin_reserve: float = 1000.0
    amm_share_reserve: int = 100

    def __repr__(self) -> str:  # security: never reveal the api_key
        return (
            "LLMConfig("
            f"model={self.model!r}, base_url={self.base_url!r}, "
            f"matching_mode={self.matching_mode!r}, api_key='***')"
        )


class LLMAgent(Agent):
    """Base for any LLM-backed agent.

    Stateless across rounds — every decision is built solely from the
    MarketView (current portfolio + public price history). This matches the
    world spec: an agent's prior decisions are not given back to it explicitly,
    but the resulting portfolio state and the price trajectory are.
    """

    provider: str = "llm"

    def __init__(
        self,
        agent_id: str,
        config: LLMConfig,
        *,
        display_name: str | None = None,
    ) -> None:
        super().__init__(agent_id, display_name)
        self.config = config
        self._system_prompt = build_system_prompt(
            config.matching_mode,
            num_traders=config.num_traders,
            initial_coin=config.initial_coin,
            initial_shares=config.initial_shares,
            total_rounds=config.total_rounds,
            amm_coin_reserve=config.amm_coin_reserve,
            amm_share_reserve=config.amm_share_reserve,
        )
        # Per-agent logger so file handlers attached externally
        # (via config.yaml `log_file:` field) emit only this agent's prompts,
        # responses, and decisions.
        self.log = logging.getLogger(f"agent.{agent_id}")

    def decide(self, view: MarketView) -> AgentDecision:
        user_prompt = _build_user_prompt(view)
        self.log.debug("round=%d user_prompt:\n%s", view.round_index, user_prompt)
        try:
            raw = self._call_provider(self._system_prompt, user_prompt)
            self.log.debug("round=%d raw_response: %r", view.round_index, raw[:500])
            decision = _parse_decision(raw)
        except Exception as exc:
            # Only the exception class is recorded — `exc` strings can contain
            # the raw model response, which we don't want bleeding into logs.
            self.log.warning(
                "round=%d decision failed (%s) — fallback HOLD",
                view.round_index,
                exc.__class__.__name__,
            )
            log.warning(
                "%s (%s) failed at round %d: %s — HOLD",
                self.agent_id,
                self.provider,
                view.round_index,
                exc.__class__.__name__,
            )
            return AgentDecision(action="HOLD", rationale=f"fallback ({exc.__class__.__name__})")

        clamped = self._clamp_to_portfolio(view, decision)
        self.log.info(
            "round=%d decision=%s qty=%d limit=%.4f rationale=%s",
            view.round_index,
            clamped.action,
            clamped.quantity,
            clamped.limit_price,
            clamped.rationale,
        )
        return clamped

    @abstractmethod
    def _call_provider(self, system: str, user: str) -> str:
        """Return raw model text."""

    @staticmethod
    def _clamp_to_portfolio(view: MarketView, decision: AgentDecision) -> AgentDecision:
        """Bring an over-ambitious order back inside the agent's budget."""
        if decision.action == "BUY":
            max_qty_by_coin = int(view.portfolio.coin // decision.limit_price)
            qty = min(decision.quantity, max_qty_by_coin)
            if qty <= 0:
                return AgentDecision(action="HOLD", rationale="clamped: no budget")
            return decision.model_copy(update={"quantity": qty})

        if decision.action == "SELL":
            qty = min(decision.quantity, view.portfolio.shares)
            if qty <= 0:
                return AgentDecision(action="HOLD", rationale="clamped: no shares")
            return decision.model_copy(update={"quantity": qty})

        return decision
