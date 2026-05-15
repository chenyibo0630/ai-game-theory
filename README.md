# AI Game Theory — 多 Agent 博弈交易场

一个封闭的经济模拟世界。来自不同 LLM 供应商的 5 个 agent 在同一套规则下进行 100 轮交易博弈，最终资产价值最高者获胜。

## 一、世界规则

### 1.1 参与者

- 共 **5 名交易者**（agent），分别由不同的 LLM 供应商驱动：
  - OpenAI
  - Anthropic
  - Google (Gemini)
  - DeepSeek
  - 阿里通义千问 (Qwen)
- 也可以使用 `baseline` 模式：用 4 个启发式策略（随机、动量、均值回归、买入持有）代替 LLM，便于离线快速跑测。

### 1.2 初始禀赋

每个 agent 起步条件**完全相同**：

| 资产 | 数量 |
| --- | --- |
| 现金 GameCoin (GC) | 100 |
| 股票 WORLD | 10 股 |
| 起始权益（按初始价计算） | 200 GC |

> 注：之所以让每位 agent 同时持有现金与股票，是因为如果初始没人持有股票，第一轮就没有人能挂卖单，市场无法启动。给一个统一的初始持仓既保证公平，也保证流动性。

### 1.3 游戏长度

- 默认 **100 轮**，可通过命令行参数 `--rounds` 修改。

### 1.4 唯一标的

- 整个世界**只有一支股票** `WORLD`，一种货币 `GameCoin`。
- agent 的所有操作都围绕这一支股票展开。

---

## 二、每轮流程

每一轮按以下时序执行：

1. **系统向每个 agent 推送当轮信息**（详见第三节）。
2. agent **独立、并发**地决策，提交以下三种动作之一：
   - `BUY n shares` ：限价买入 n 股，附最高可接受单价（limit_price）。
   - `SELL n shares`：限价卖出 n 股，附最低可接受单价。
   - `HOLD`         ：本轮不动。
3. 所有订单收齐后，交给**撮合引擎**（详见第四节）成交。
4. 系统根据成交结果更新每个 agent 的现金和持股，并把本轮成交后的市场价作为下一轮的开盘价。

### 2.1 并发与公平

- agent 之间**互相不可见**：看不到别人的下单、持仓、决策、甚至身份。
- agent 提交订单的先后顺序**不影响**撮合结果（所有订单在轮末统一处理）。
- 每轮的 agent 决策在线程池中**并发**执行，仅用于压缩 LLM API 调用墙钟时间，不影响逻辑确定性。

---

## 三、agent 可见信息

### 3.1 系统提示词（System Prompt，全局一致 + 撮合规则定制）

每个 agent 都收到**完全相同**的系统提示词，由两部分拼接：

**A. 世界规则部分**（所有 agent 通用）：

- 世界规则：5 个交易者、同样的初始禀赋、固定轮数。
- 每轮可选的动作：BUY / SELL / HOLD。
- 输出格式规范（严格 JSON）。
- 约束：BUY 总额不得超过现金、SELL 数量不得超过持股。
- 获胜目标：最终轮结束时 `cash + shares * final_price` 最高者胜。
- 明确告知："你永远看不到其他 agent 的订单、持仓、决策或身份"。

**B. 撮合引擎部分**（按当前选用的引擎不同而不同）：

- 若使用 **AMM 模式**：系统提示词会说明恒定乘积公式 `R_c * R_s ≈ k`、买卖的有效价格公式 `R_c / (R_s ∓ q)`、滑点机制、以及限价单作为滑点保护的策略含义。
- 若使用 **集合竞价模式**：说明所有订单按统一清算价成交、超过/低于清算价的单完全成交、等于清算价的按比例分配等。

撮合规则的不同直接影响 agent 的最优策略，因此必须告诉它。

### 3.2 每轮用户消息（System 告知给 agent）

每轮系统向 agent 推送的内容**只包含**：

```
=== This round ===
Round: 5 of 100 (95 remaining)        ← 当前轮次号、总轮数、剩余轮数
Your cash: 100.0000 GC                ← 自己持有的现金
Your shares: 10                       ← 自己持有的股票
Current market price: 10.0000 GC      ← 当前市场价

=== Public price history (oldest → newest) ===
10.0000, 10.2000, 10.5300, ...        ← 公共价格历史（最近 30 轮的清算价）
```

### 3.3 不可见信息

- 其他 agent 的现金、持股、订单、决策、理由——**全部不可见**。
- 自己历史回合的下单/成交明细——**不直接喂给 LLM**。理由：当前的 `cash` 和 `shares` 已经累计反映了所有过去交易的结果，再额外灌输 fills 列表是冗余的；价格轨迹本身就给了足够的市场上下文。

---

## 四、撮合引擎（可插拔）

撮合逻辑通过抽象基类 `MatchingEngine` 解耦，可在配置中选择不同实现：

```python
# src/arena/matching/base.py
class MatchingEngine(ABC):
    def initial_price(self) -> float | None: ...
    def clear(self, orders, opening_price) -> ClearingResult: ...
```

目前内置两种实现，**默认使用 AMM**。

### 4.1 Uniswap V2 风格 AMM（默认，零手续费）

灵感来自以太坊上的 Uniswap V2。系统维护一个共享流动性池 `POOL`，持有两种资产的储备：

- `R_c` ：池子中的 GameCoin 储备
- `R_s` ：池子中的 WORLD 股票储备

#### 4.1.1 恒定乘积不变量

```
R_c * R_s ≈ k  (constant)
```

> 由于股票数量必须是整数，实际会有微小的离散漂移，但不影响整体语义。

#### 4.1.2 当前市场价

```
spot_price = R_c / R_s
```

#### 4.1.3 买入 q 股的执行公式

向池子注入现金 `Δ_c`，从池子取出 `q` 股，需满足：

```
R_c * R_s = (R_c + Δ_c) * (R_s - q)
   ⟹  Δ_c = q * R_c / (R_s - q)
   ⟹  有效均价 = Δ_c / q = R_c / (R_s - q)
```

给定限价 `limit_price`，最大可成交数量：

```
R_c / (R_s - q) ≤ limit  ⟺  q ≤ R_s - R_c / limit
```

#### 4.1.4 卖出 q 股的执行公式

向池子注入 `q` 股，取出现金 `Δ_c`：

```
R_c * R_s = (R_c - Δ_c) * (R_s + q)
   ⟹  Δ_c = q * R_c / (R_s + q)
   ⟹  有效均价 = R_c / (R_s + q)
```

给定限价 `limit_price`，最大可成交数量：

```
R_c / (R_s + q) ≥ limit  ⟺  q ≤ R_c / limit - R_s
```

#### 4.1.5 滑点与策略含义

- **滑点是真实存在的**：单笔越大，对池子状态的扰动越剧烈，每一股的边际价格越差。把大单分拆到多轮通常优于一次性出手。
- **限价是你的滑点保护**：如果在限价内池子撑不了你的全部需求量，会**部分成交**（剩余部分作废）。
- **买入推高、卖出压低**：如果多个 agent 同轮一齐买入，价格会被推得更高，后执行的买家承担更大成本。

#### 4.1.6 同轮多单的执行顺序

同一轮的多个订单按 `(agent_id, side)` 字典序**确定性顺序**逐个对池子撮合。顺序与提交时间无关——所有订单在轮末同时揭晓，没人能"抢跑"。

#### 4.1.7 公平性

- 同一轮内不同 agent 可能拿到**不同的有效价格**（因为池子状态在每笔成交后变化）。
- 这种"先撮合者占便宜"的现象在真实链上同样存在（受出块顺序影响），但游戏中是确定性、可复盘的。
- agent 之间在出价时**完全无信息**，无法事先知道谁会先撮合，因此不能利用顺序作弊。

### 4.2 集合竞价（可选，默认不启用）

每轮所有订单批量收集，求出一个使成交量最大化的**统一清算价** `P*`：

```
P* = argmax_P  min( demand(P), supply(P) )
```

平局裁决依次按：成交量最大 → `|demand - supply|` 最小 → 与开盘价最接近。

- `BUY limit > P*` 与 `SELL limit < P*` 的订单**全额成交**
- `limit == P*` 的边际订单按数量**比例分配**剩余容量（agent_id 字典序确定性 tie-break）
- 所有成交都以同一个 `P*` 结算
- 未成交订单**轮末作废**

集合竞价模式下没有滑点，所有成交者价格一致，整套交易系统的 agent-level 现金与持股之和守恒。

### 4.3 选择撮合模式

```bash
# CLI 选择
python -m src.main --matching amm           # 默认
python -m src.main --matching call_auction
```

```python
# 代码中通过 WorldConfig 选择
WorldConfig(
    matching_mode="amm",        # 或 "call_auction"
    amm_coin_reserve=1000.0,    # AMM 模式专用
    amm_share_reserve=100,
)
```

---

## 五、获胜条件

- 第 100 轮结束后，每个 agent 计算最终权益：

  ```
  equity = cash + shares * final_market_price
  ```

- 最终权益最高的 agent 获胜。

---

## 六、容错与鲁棒性

- **LLM 输出异常**：模型返回非 JSON、缺字段、字段类型错误等都会被解析器宽容地降级处理。`json.loads` 失败时尝试用正则截取 JSON 块。
- **超预算订单**：agent 报的 quantity × limit_price 超过现金、或卖出数量超过持股，自动裁剪到上限；裁剪后若无可执行内容则降级为 HOLD。
- **API 调用失败**：网络错误 / 超时 / API 报错 → 该 agent 本轮 HOLD，日志记录原因，游戏继续。
- **agent 抛异常**：捕获并降级为 HOLD，单个 agent 的崩溃不会让整局崩盘。

---

## 七、命令行 & 项目结构

### 7.1 Docker 一键启动（推荐）

```bash
cp .env.example .env                       # API keys
cp config.yaml.example config.yaml         # agent 配置

docker compose up --build                  # 启动 mysql + frontend
docker compose run --rm backend            # 跑一局，数据写入 mysql

open http://localhost:8000                 # 浏览资产曲线 / 排行榜
```

三个服务：
- **mysql** (port 3306) — 自动跑 `db/schema.sql` 建表
- **frontend** (port 8000) — FastAPI + Chart.js，查询 mysql 渲染价格曲线、equity 曲线、排行榜
- **backend** — 按 `config.yaml` 跑一局，结果落 mysql；可重复 `docker compose run --rm backend` 跑多局

### 7.2 本地 Python 快速开始（不用 docker）

```bash
pip install -r requirements.txt
cp .env.example .env       # 填入对应供应商的 API key
cp config.yaml.example config.yaml   # 编辑 agent 名字 / 模型 / 日志路径

# 用 config.yaml 跑（推荐）：每个 agent 自己的 id/名字/api_key/base_url/log
python -m src.main --config config.yaml

# 或者用 CLI 默认 roster（无 config.yaml 时）
python -m src.main --rounds 100 --mode baseline     # 无需 key
python -m src.main --rounds 100 --mode llm          # 用 .env 里的 key

# 切换撮合引擎
python -m src.main --matching amm --amm-coin-reserve 1000 --amm-share-reserve 100
python -m src.main --matching call_auction

# 多场比赛求胜率（锦标赛）
python -m src.tournament --runs 20 --rounds 50 --mode baseline

# 回放保存的对局
python -m src.replay runs/run-<timestamp>.json

# 跑测试
python -m pytest tests/
```

### 7.2 config.yaml 结构

每个 agent 用一个独立条目配置——`id` 必须唯一，名字、模型、API key、base_url、日志路径都各自指定。同一 provider 可以同时跑多个 agent（例如不同温度的两个 OpenAI agent）。完整字段见 `config.yaml.example`，简要：

```yaml
world:
  rounds: 100
  matching: amm
  amm:
    coin_reserve: 1000
    share_reserve: 100

security:
  allow_insecure_base_url: false   # 设 true 才允许 http:// 或本地地址

agents:
  - id: openai-alice
    name: "Alice (OpenAI)"
    provider: openai
    model: gpt-4o-mini
    api_key: ${OPENAI_API_KEY}     # 也支持直接写字面量
    temperature: 0.4
    log_file: runs/agents/alice.log

  - id: groq-frank
    provider: openai_compatible    # 任意 OpenAI 兼容端点
    model: llama-3.1-70b-versatile
    api_key: ${GROQ_API_KEY}
    base_url: https://api.groq.com/openai/v1
    log_file: runs/agents/frank.log
```

每个 `log_file` 路径会被 attached 一个 Python 的 `FileHandler`，agent 自己的 prompt / 原始 model 响应 / 解析后的决策 / 限价裁剪都写到这个文件里，方便单独复盘。

### 7.2 目录结构

```
src/
├── arena/
│   ├── types.py              # Order / Trade / PortfolioState / RoundReport（pydantic v2 不可变）
│   ├── matching/
│   │   ├── base.py           # MatchingEngine ABC + ClearingResult + POOL_ID 标记
│   │   ├── call_auction.py   # 集合竞价
│   │   └── amm.py            # Uniswap V2 风格 AMM
│   ├── world.py              # 游戏状态 + 轮次主循环 + 并发决策调度
│   └── visualize.py          # ASCII 价格图
├── agents/
│   ├── base.py               # Agent 抽象 + MarketView
│   ├── baseline/             # 离线启发式 baseline：random / momentum / mean-reversion / buy-and-hold
│   └── llm/                  # 5 个 LLM 供应商 agent；llm_base.py 处理提示词、JSON 解析、降级
├── config.py                 # .env 读取，按 provider 构造 LLMConfig
├── main.py                   # 单局 CLI 入口
├── replay.py                 # 回放工具
└── tournament.py             # 多局锦标赛
tests/                        # 46 个单测，覆盖撮合、世界循环、LLM 解析、AMM 数学等
runs/                         # 每局自动写入的 JSON 详细记录
```

### 7.3 运行日志

每局结束后会在 `runs/run-<timestamp>.json` 写入一份完整记录，包括：

- 世界配置
- 最终排行榜
- 每轮的开盘价、清算价、成交量、所有订单、所有成交、所有 portfolio 快照、每个 agent 的 rationale 字段

用 `python -m src.replay runs/<file>.json` 可以可读地查看。

---

## 八、扩展指南

### 8.1 加新 baseline agent

继承 `Agent` 实现 `decide(view)` 即可：

```python
from src.agents.base import Agent, MarketView
from src.arena.types import AgentDecision

class MyStrategyAgent(Agent):
    def decide(self, view: MarketView) -> AgentDecision:
        # ... your strategy here
        return AgentDecision(action="HOLD")
```

在 `src/main.py:_baseline_roster()` 中注册。

### 8.2 加新 LLM 供应商

若供应商兼容 OpenAI Chat Completions API，继承 `OpenAIAgent` 改 `base_url` 即可：

```python
class MyProviderAgent(OpenAIAgent):
    DEFAULT_BASE_URL = "https://api.myprovider.com/v1"
```

否则继承 `LLMAgent` 实现 `_call_provider(system, user) -> str`。

### 8.3 加新撮合引擎

继承 `MatchingEngine`，实现 `clear(orders, opening_price) -> ClearingResult`：

```python
class MyEngine(MatchingEngine):
    name = "my_engine"
    def clear(self, orders, opening_price):
        # ... return ClearingResult(...)
```

在 `src/arena/world.py:_build_engine()` 中注册，并在 `WorldConfig.matching_mode` 增加对应字面量。如果需要让 LLM 知晓新引擎的规则，在 `src/agents/llm/llm_base.py:build_system_prompt()` 中加分支。

---

## 九、设计取舍备忘

- **为什么默认用 AMM 而不是集合竞价**：AMM 的价格发现机制让市场价**真正会动**（滑点 + 顺序撮合 → 同轮可有不同 fill 价格），博弈性更强；集合竞价容易让价格"卡死"在初始水平附近，对启发式 baseline 而言信息量较少。
- **为什么不直接告诉 LLM 它自己的历史成交**：LLM 通过 `cash / shares` 已能完整推断自己累计的交易结果；额外塞一份 fills 列表只增加 token 成本、不增加决策信息。
- **为什么提示词包含撮合规则**：策略与撮合机制强耦合（AMM 关心滑点和切单，集合竞价关心限价档位与流动性平衡），不讲清规则等于让 agent 蒙眼下棋。
- **为什么 AMM 没有手续费**：游戏内不收 fee 是设计选择，目的是把"零和博弈"的性质保留得更纯粹（call_auction 模式下agent 之间是纯零和；AMM 模式下 agent 与池子之间仍然零和，但池子充当一个被动对手）。
- **为什么用 call auction 而不是 continuous double auction（CDA）作为可选传统模式**：LLM agent 是回合制决策，没法毫秒级挂改单；call auction 自然契合"同时出价"的语义。
