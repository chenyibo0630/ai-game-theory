# AI Game Theory — 多 Agent 博弈交易场

一个封闭的经济模拟世界。来自不同 LLM 供应商的 5 个 agent 在同一套规则下进行多轮交易博弈，最终资产价值最高者获胜。

---

## 一、世界规则

### 1.1 参与者

- 共 **5 名交易者**，可由不同 LLM 供应商驱动（OpenAI / Anthropic / Google Gemini / DeepSeek / Qwen），或换成离线 baseline（mock / random / momentum / mean-reversion / buy-and-hold）。
- 一个常用配置：**4 个 LLM + 1 个 mock**——mock 充当确定性噪声源，强制每轮注入流动性，避免市场停滞。

### 1.2 初始禀赋

每个 agent 起步条件完全相同：

| 资产 | 数量 |
| --- | --- |
| 现金 GameCoin (GC) | 100 |
| 股票 WORLD | 10 股 |

> 注：同时持有现金与股票是为了保证第一轮就有人能挂卖单，市场能启动。

### 1.3 游戏长度

- 默认 **100 轮**，`config.yaml` 里 `world.rounds` 任意调整。

### 1.4 唯一标的

- 整个世界**只有一支股票** `WORLD`、一种货币 `GameCoin`。

---

## 二、每轮流程

1. 系统向每个 agent 推送当轮信息（详见第三节）。
2. agent **独立、并发**地从 BUY / SELL / HOLD 中选一个，提交订单或 HOLD。
3. 收齐订单后，**撮合引擎**按确定性顺序逐单成交（详见第四节）。
4. 更新各 agent 的现金、持股、池子状态，把当前轮的撮合结束后的池子现货价作为下一轮的开盘价。

### 2.1 并发与公平

- agent 之间**互相不可见**：看不到别人的下单、持仓、决策、身份。
- 订单提交时间不影响撮合结果——所有订单在轮末同时揭晓，没人能"先提交"抢跑。
- 决策在线程池中**并发**执行，只用于压缩 LLM API 墙钟时间，不影响逻辑确定性。

---

## 三、Agent 可见信息

### 3.1 System Prompt（每个 agent 完全相同）

- 世界规则、初始禀赋、可选动作、输出 JSON 格式、约束。
- **撮合引擎细节**（AMM 公式 / call_auction 规则）。
- AMM 下额外有**两个 worked example**：演示一笔成功的 BUY 抓上涨、一笔成功的 SELL 抓下跌，包括有效价计算、付/收金额、池子状态变化、滑点 vs 预期收益的权衡。
- 不告知总轮数（隐藏游戏视野，模型不知道还剩几轮）。

### 3.2 每轮 User Message

```
=== 本轮信息 ===
轮次序号：5
你的现金：100.0000 GC
你的持股：10 股
当前市场价：10.6281 GC
流动池：R_c = 1030.93 GC，R_s = 97 股   ← AMM 模式下额外提供

=== 公开价格历史（由旧到新）===
10.0000, 10.4123, 10.8507, 10.6281, ...    ← 最近 30 轮的撮合价
```

### 3.3 不可见信息

- 其他 agent 的现金、持股、订单、决策、身份——**全部不可见**。
- 自己历史回合的下单/成交明细——**不喂给 LLM**。当前 `cash / shares` 已累计反映所有过去交易结果，再灌 fills 列表只增 token 成本、不增决策信息。

---

## 四、撮合引擎（可插拔）

撮合通过抽象基类 `MatchingEngine` 解耦：

```python
class MatchingEngine(ABC):
    def initial_price(self) -> float | None: ...
    def pool_state(self) -> tuple[float, int] | None: ...
    def clear(self, orders, opening_price) -> ClearingResult: ...
```

两种内置实现，**默认使用 AMM**。

### 4.1 Uniswap V2 风格 AMM（默认）

共享流动池 `POOL` 持有两种资产储备：`R_c`（GameCoin）+ `R_s`（WORLD 股票）。零手续费。

#### 4.1.1 恒定乘积

```
R_c · R_s = k   （每笔成交后严格守恒，仅有整数股的微小取整误差）
spot_price = R_c / R_s
```

#### 4.1.2 单笔有效价公式

对当前 `(R_c, R_s)`：

```
BUY  q 股: 有效均价 = R_c / (R_s − q)
SELL q 股: 有效均价 = R_c / (R_s + q)
```

成交后池子更新：

```
BUY:  R_c ← R_c + q · R_c/(R_s−q),  R_s ← R_s − q
SELL: R_c ← R_c − q · R_c/(R_s+q),  R_s ← R_s + q
```

#### 4.1.3 限价 + 部分成交

`limit_price` 是 agent 能接受的**最差有效均价**。引擎计算最大可成交数量 `q_max`：

```
BUY:  q ≤ R_s − R_c / limit_price
SELL: q ≤ R_c / limit_price − R_s
```

- `q_max ≥ 下单量` → 全额成交
- `0 < q_max < 下单量` → **部分成交** `q_max` 股，剩余作废
- `q_max ≤ 0` → 本单完全不成交
- BUY 还会被池子上限保护：池子至少留 1 股

#### 4.1.4 同轮内的撮合顺序

订单按**两级排序**逐单对池子撮合：

1. **优先层**：mock baseline agent（如 `noise-nick`）总是排在前面，作为确定性噪声源。
2. **哈希层**：每层内部按 `SHA-256("{round_index}:{agent_id}")` 排序——这是按轮次种子的伪随机置换：
   - **可复盘**：同一 `round_index + agent_id` 永远产生同一顺序。
   - **跨轮公平**：每个 agent 在自己所在的层里位置均匀分布，不存在永久"先手优势"。

agent 在提交前**不知道**自己排在第几位，所以无法"抢跑"。前一单会更新池子状态，影响后一单的有效价格——同侧多个买单时，靠前的滑点小、靠后的滑点大。

#### 4.1.5 公平性与策略含义

- **同轮内不同 agent 拿到的有效价不同**——位置先后造成的滑点差异是真实的。
- 多轮平均后字典序偏置被洗掉；mock 永远先，相当于把"做市商成本"集中在 mock 身上。
- **大单滑点是凸的**：单笔越大，每股边际价越差，但**绝对利润 = 单笔量 × 价差**——单笔过小会让正确的方向判断兑现不到 PnL。

### 4.2 集合竞价（可选）

每轮所有订单批量收集，求出使成交量最大化的**统一清算价** `P*`：

```
P* = argmax_P  min(demand(P), supply(P))
```

- `BUY limit ≥ P*` 与 `SELL limit ≤ P*` 的订单**全额成交**
- 平局裁决：成交量最大 → `|demand − supply|` 最小 → 最接近开盘价
- 所有成交都以 `P*` 结算（无滑点，无部分成交）
- 未成交订单本轮作废

### 4.3 选择撮合模式

```bash
python -m src.main --matching amm           # 默认
python -m src.main --matching call_auction
```

或在 `config.yaml`：

```yaml
world:
  matching: amm                  # 或 call_auction
  amm:
    coin_reserve: 500
    share_reserve: 50
```

---

## 五、获胜条件

最终轮结束后，按下面公式算每个 agent 的权益，最高者胜：

```
equity = cash + shares × final_clearing_price
```

---

## 六、容错与鲁棒性

- **LLM 输出异常**：非 JSON、缺字段、字段类型错误等都被宽容降级。`json.loads` 失败时正则截取 JSON 块。
- **rationale 控制字符 / 超长**：自动剥离控制字符 + 截 240 字符。
- **超预算订单**：`quantity × limit_price` 超过现金、或 SELL 数量超过持股，自动裁剪到上限；裁剪后无可执行内容降级 HOLD。
- **API 调用失败**：网络错误 / 超时 / API 报错 → 该 agent 本轮 HOLD，日志记录原因，游戏继续。
- **agent 抛异常**：捕获并降级 HOLD，单个 agent 的崩溃不会让整局崩盘。

---

## 七、Docker 一键启动（推荐）

```bash
cp config.yaml.example config.yaml       # 填入 agent / API key

# 端口冲突时指定备用端口（默认 3306 / 8000）
MYSQL_PORT=3307 FRONTEND_PORT=8001 \
  docker compose up -d mysql frontend    # 启 db + 前端
docker compose run --rm backend          # 跑一局，结果落 mysql
```

打开 `http://localhost:8001` 浏览：

- 价格曲线
- 各 agent 资产 / equity 曲线
- 最终排行榜
- Decisions 表（每轮每 agent 的 action / 数量 / 限价 / **交易前后价格** / rationale）

三个服务：

| 服务 | 端口（默认） | 说明 |
| --- | --- | --- |
| `mysql` | 3306 | 自动执行 `db/schema.sql` 建表，volume 持久化 |
| `frontend` | 8000 | FastAPI + Chart.js，只读查 MySQL |
| `backend` | - | 单次任务容器，跑一局退出 |

---

## 八、本地 Python 启动（不用 docker）

```bash
pip install -r requirements.txt
cp config.yaml.example config.yaml

# 用 config.yaml 跑（推荐）
python -m src.main --config config.yaml

# 无 config 时用默认 roster
python -m src.main --rounds 50 --mode baseline    # 无需 key
python -m src.main --rounds 50 --mode llm         # 用 .env 里的 key

# 多局锦标赛
python -m src.tournament --runs 20 --rounds 50 --mode baseline

# 回放
python -m src.replay runs/run-<timestamp>.json

# 跑测试
python -m pytest tests/      # 75 个测试
```

---

## 九、config.yaml 结构

每个 agent 独立条目，`id` 必须唯一。完整字段见 `config.yaml.example`，简要：

```yaml
world:
  rounds: 50
  matching: amm
  initial_coin: 100
  initial_shares: 10
  amm:
    coin_reserve: 500
    share_reserve: 50
  parallel_decisions: true
  decision_workers: 5

security:
  allow_insecure_base_url: false   # true 才允许 http:// 或本地地址

agents:
  - id: noise-nick
    name: "Nick (Mock)"
    provider: mock          # 总是首位执行的噪声源

  - id: anthropic-bob
    name: "Bob (Claude Opus 4.7)"
    provider: anthropic
    model: claude-opus-4-7
    api_key: ${ANTHROPIC_API_KEY}
    temperature: 0.4
    log_file: runs/agents/bob.log

  - id: minimax-mike
    name: "MiniMax"
    provider: openai          # OpenAI 兼容端点
    model: minimax-m2.7
    base_url: https://api.lkeap.cloud.tencent.com/plan/v3
    api_key: ${MINIMAX_API_KEY}
    temperature: 0.4
    # reasoning 模型默认 256 tokens 不够吐完 reasoning + content
    max_tokens: 2048
    log_file: runs/agents/mini-max.log
```

每个 `log_file` 路径会被附加一个 `FileHandler`，记录该 agent 的 prompt / 原始响应 / 解析后决策 / 限价裁剪。

---

## 十、存储 & 数据模型

### 10.1 MySQL 表（`db/schema.sql`）

| 表 | 主键 | 内容 |
| --- | --- | --- |
| `runs` | `run_id` | 一局一行，含 `total_rounds`, `matching_mode`, `amm_coin_reserve`, `amm_share_reserve`, 时间戳 |
| `rounds` | `(run_id, round_index)` | 每轮一行，含 `opening_price`, `clearing_price`, `cleared_volume`, **`pool_coin`, `pool_shares`** |
| `orders` | autoinc | 每张订单 |
| `trades` | autoinc | 每笔成交，AMM 模式下 pool 侧填 `'POOL'` |
| `portfolios` | `(run_id, round_index, agent_id)` | 每轮每 agent 的 `cash`, `shares`, `equity` |
| `decisions` | `(run_id, round_index, agent_id)` | 每轮每 agent 的 `action`, `quantity`, `limit_price`, `rationale`, **`price_before`, `price_after`** |
| `run_agents` | `(run_id, agent_id)` | 一局的 agent registry：`display_name`, `provider`, `model` |

### 10.2 `price_before / price_after`

`decisions` 表的两个关键字段：

- `price_before` ：撮合到该 agent 的订单**之前**池子的 spot
- `price_after`  ：撮合**之后**的池子 spot
- HOLD / 未提单 agent → `price_before = price_after = 当轮开盘价`
- 在 AMM 模式下，这两个值反映该 agent 自身的订单对池子的位移；call_auction 模式下两者都等于统一清算价 `P*`

### 10.3 Web API（FastAPI）

| 端点 | 返回 |
| --- | --- |
| `GET /api/runs` | 最近 100 局 |
| `GET /api/runs/{run_id}` | 一局 metadata + agents |
| `GET /api/runs/{run_id}/prices` | 每轮的开/清算价、成交量 |
| `GET /api/runs/{run_id}/equity` | `{agent_id: [{round, cash, shares, equity}]}` |
| `GET /api/runs/{run_id}/decisions` | 全部 decisions，含 `price_before / price_after` |
| `GET /api/runs/{run_id}/trades` | 全部 trades |

---

## 十一、项目结构

```
src/
├── arena/
│   ├── types.py              # Order / Trade / PortfolioState / RoundReport（pydantic frozen）
│   ├── matching/
│   │   ├── base.py           # MatchingEngine ABC + ClearingResult + POOL_ID
│   │   ├── amm.py            # Uniswap V2 风格 AMM（逐单 + 优先层 + SHA-256 洗牌）
│   │   └── call_auction.py   # 集合竞价
│   ├── world.py              # 状态 + 主循环 + 并发决策调度
│   └── visualize.py          # ASCII 价格图
├── agents/
│   ├── base.py               # Agent 抽象 + MarketView
│   ├── baseline/             # mock / random / momentum / mean-reversion / buy-and-hold
│   └── llm/                  # 5 个 LLM 供应商；llm_base.py 处理提示词 + JSON 解析 + 降级
├── storage/
│   ├── base.py               # RunSink 接口
│   ├── json_sink.py          # 每局写 runs/run-*.json
│   ├── mysql_sink.py         # 写 MySQL
│   └── null_sink.py
├── web/
│   ├── server.py             # FastAPI 端点
│   └── static/               # index.html + app.js（Chart.js 渲染）
├── config_loader.py          # config.yaml 解析
├── agent_factory.py          # AgentSpec → Agent
├── main.py / replay.py / tournament.py
db/schema.sql                 # MySQL schema
docker/                       # 后端 + 前端 Dockerfile
tests/                        # 75 个测试，覆盖撮合、世界循环、LLM 解析、AMM 数学、storage
runs/                         # 每局 JSON 详细记录 + 每个 agent 的日志
```

---

## 十二、实验记录：worked-examples 把 Opus 4.7 从第 3 推到第 1

### 12.1 背景

跑了几局后注意到：Claude Opus 4.7 的 BUY 决策时的平均市价 9.99 / SELL 时 10.98（**+0.98 GC 的方向 edge，5 个 agent 中最大**），但累计权益却始终排第 2-3。

诊断：

| 指标 | Bob (Opus 4.7) | Eve (Qwen) |
| --- | --- | --- |
| 平均 BUY 量 | 1.27 股 | 2.22 股 |
| 平均 SELL 量 | 1.21 股 | 5.71 股 |
| HOLD 比例 | 9.3% | 8.7% |
| rationale 含滑点/池子公式 | **0%** | **23.4%** |

**Bob 的方向判断最准，但每笔下手最小**——91% 的轮次都在动单，但都是 1 股小动作。反复用滑点付"做市成本"却没建立任何方向性敞口。**这不是模型能力问题**：Opus 完全能算 `R_c/(R_s−q)`，他只是从不在 rationale 里写出来，靠定性词（"小量"、"分批"、"保守"）做决策。

### 12.2 干预

在 AMM system prompt 里加了两个 worked example：

- **范例 A**：BUY 5 股从 `(R_c=500, R_s=50)` 起步，3 轮后池子飘到 `(600, 42)`，SELL 5 股，净利 +8.27 GC。
- **范例 B**：SELL 4 股从 `(700, 40)` 起步，3 轮后池子降到 `(560, 48)`，BUY 4 股回本，净利 +12.73 GC。

每个范例**完整展开**：`q_max` 公式、有效价计算、付/收金额、池子状态变化。结尾点出三条要害：

1. 滑点是确定可算的，下单前必须先算
2. 单笔下手过小是隐形亏损（凸函数让单笔比例滑点更小，但**绝对利润 = 量 × 价差**）
3. limit_price 是滑点保护、不是定价目标

### 12.3 结果（50 轮一局）

```
                  ── 加范例前 ──    ── 加范例后 ──
Bob (Opus 4.7)        228.35  →    254.51  🏆  (+26.16)
MiniMax               225.67  →    226.40
Dave (DeepSeek)       231.36  →    220.52     (−10.84)
Eve (Qwen)            229.85  →    187.35     (−42.50)
Nick (Mock)           179.29  →    179.12
```

Bob 的行为变化：

| 指标 | 加范例前 | 加范例后 |
| --- | --- | --- |
| 平均 BUY 量 | 1.27 股 | **3.71 股**（+193%） |
| 平均 SELL 量 | 1.21 股 | **2.96 股**（+145%） |
| 总成交股数 | 24 | **104**（+333%） |
| HOLD 次数 | 4 / 50 | 2 / 50 |
| Rationale 引用公式 | 0% | 0%（仍然不写公式） |
| 期末权益 | 228.35（第 3） | **254.51（第 1）** |

**关键观察**：Bob 没在 rationale 里写公式（隐式教学），但内化了**规模逻辑**——把判断对的方向兑现成实际仓位。per-share edge 反而从 +1.82 降到 +0.59，但因为单笔量翻 3 倍、总成交量翻 4 倍，绝对利润大幅领先。

**副作用**：Eve 反向下滑（229.85 → 187.35）。她公式引用率升到 36%，平均 SELL 量暴增到 17.33 股，但本局方向判断翻车（买入均价 11.17 / 卖出均价 10.14，每股 −1.03 GC）。worked example 对 Bob 这种保守派是兴奋剂，对 Eve 这种已经激进的可能是过量。

### 12.4 下一步

- 范例里加一个"方向判断错的成本"的反面例子
- user prompt 里加 self-check："我对方向判断的置信度多少？高置信度下大单，低置信度时 HOLD"

---

## 十三、扩展指南

### 13.1 加新 baseline agent

```python
from src.agents.base import Agent, MarketView
from src.arena.types import AgentDecision

class MyStrategyAgent(Agent):
    def decide(self, view: MarketView) -> AgentDecision:
        return AgentDecision(action="HOLD")
```

在 `src/agent_factory.py:LOCAL_PROVIDERS` 注册一个 builder。

### 13.2 加新 LLM 供应商

OpenAI 兼容端点最简单——继承 `OpenAIAgent` 改 `DEFAULT_BASE_URL`：

```python
class MyProviderAgent(OpenAIAgent):
    DEFAULT_BASE_URL = "https://api.myprovider.com/v1"
```

否则继承 `LLMAgent` 实现 `_call_provider(system, user) -> str`。

### 13.3 加新撮合引擎

继承 `MatchingEngine`，实现 `clear(orders, opening_price) -> ClearingResult`。

```python
class MyEngine(MatchingEngine):
    name = "my_engine"
    def clear(self, orders, opening_price):
        return ClearingResult(...)
```

在 `src/arena/world.py:_build_engine()` 里加 if-分支，在 `src/agents/llm/llm_base.py:build_system_prompt()` 加对应规则文本。

---

## 十四、设计取舍备忘

- **为什么默认 AMM 而不是集合竞价**：AMM 的滑点 + 顺序撮合让价格真正动起来，博弈性更强；集合竞价容易把价格"卡死"在初始水平附近。
- **为什么把 mock 钉到首位**：mock 是随机噪声，永远第一会让它把首位滑点（无方向性敞口的成本）全包下，给后面的 LLM 留更干净的市场环境。
- **为什么用 SHA-256 而不是 `random.shuffle`**：`random` 依赖全局种子状态、跨进程不一致；SHA-256 用 `(round_index, agent_id)` 当 salt，跨进程跨语言都一致，replay 完美。
- **为什么不直接告诉 LLM 自己的历史成交明细**：cash / shares 已累计反映所有过去成交，再灌 fills 列表只增 token 成本。
- **为什么提示词里包含撮合规则 + worked example**：策略与撮合机制强耦合（AMM 关心滑点和切单），不讲清规则等于让 agent 蒙眼下棋；worked example 比纯公式更能让 LLM 把规则映射到行动（见第十二节）。
- **为什么 AMM 没有手续费**：保留"零和博弈"的纯粹性。call_auction 模式下 agent 之间纯零和；AMM 模式下 agent 与池子之间零和（池子充当被动对手）。
