# Xiamen Guangchen Intelligent - Joey Chan

1. **Vibe Coding —— 从零实现一个最小可用 Agent**（`mini_agent/`，不依赖 agent 框架，由接入`Claude Code`的`Opus 4.8`实现，经过 OpenAI-compatible LLM 验证）。
2. **架构设计题解答**（5 个模块各选 1 题，见下方 [Part 2](#part-2--架构设计题解答)）。

> 交付物索引在文末 [提交物清单](#提交物清单)。演示录屏见 [录屏](#录屏)。AI 协作与问题解决记录见 [`docs/ai-collaboration-log.md`](docs/ai-collaboration-log.md)。

---

## Part 1 · Vibe Coding：从零实现最小可用 Agent

**合规**：Agent Runtime 未使用 langgraph / openhands / openclaw 等框架；OpenAI SDK 仅作为访问 OpenAI-compatible 端点的 HTTP 客户端（非 agent 框架）。工具走**原生 function calling**（`tools=[...]` → `tool_calls`）。

### 快速开始

```bash
pip install -r requirements.txt
cp mini_agent/.env.example .env      # 填入 OpenAI-compatible 的 key（见下方“模型”说明）

python -m mini_agent.cli             # 交互式 REPL
python -m mini_agent.examples.demo   # 双 session 隔离演示（录屏用）
python -m pytest tests/ -q           # 33 个离线测试，无需网络、时间无关
```

REPL 命令：`/session <id>`、`/sessions`、`/todos`、`/trace`、`/help`、`/quit`；加 `--memory` 开启长期记忆。

> **模型说明**：循环依赖 OpenAI 风格的 `tool_calls`，弱模型可能会失败。实测 `Qwen/Qwen2.5-7B-Instruct` 会输出退化文本并在工具调用上超时
> **已验证可用模型**：`zai-org/GLM-5.2`、`deepseek-ai/DeepSeek-V3`、`Qwen/Qwen2.5-72B-Instruct`。配置 `CHAT_MODEL` 为其一即可。

### 设计要点

| 模块 | 文件 | 职责 |
|---|---|---|
| 核心循环 | `mini_agent/runtime.py` | 接收输入 → 决策 → 调工具 → 循环 → 收尾；含 `max_steps` 强制收尾、连续失败熔断、并行工具调用 |
| 工具注册 | `mini_agent/tools/registry.py` | 每个工具 = 名称 + 描述 + JSON Schema + 函数；生成给 LLM 的 `tools` 载荷 |
| 工具 | `mini_agent/tools/*.py` | `calculator`（安全 AST）、`search`（mock、确定性）、`weather`（mock）、`todo`（**用户级**，跨窗口共享） |
| 输出解析 | `mini_agent/parser.py` | 提取 思考 / 工具调用 / 最终答；对坏/截断 JSON 做 salvage |
| Session | `mini_agent/session.py` | 对话历史按 `session_id` 隔离（窗口互不影响）；**待办按用户共享**；线程安全 |
| Context | `mini_agent/context_manager.py` | 组装每步消息 + 基础压缩（只在 user 轮边界切，绝不切断工具对） |
| Trace | `mini_agent/trace.py` | 每次工具调用的执行日志（工具/参数/成功/延迟） |
| 长期记忆 | `mini_agent/memory.py` | 可选的 `MemoryBackend`，适配 AutoMemory（默认 `NullMemory`） |

**循环的安全边界**：无工具调用的回复=最终答返回用户；达 `max_steps` 强制无工具收尾；连续 3 步全失败则停止。**异常处理**：工具异常/未知工具/坏参数都被捕获并以 `{"error": …}` 回灌给模型自恢复。

**Context 与 Memory**：每步组装的 context 顺序为：system 提示 → 滚动摘要（旧轮压缩后才有）→ 长期记忆召回块（若启用）→ 近 N 轮原文。压缩在每轮开始触发，超阈值时把最旧的若干轮 LLM 摘要（保留实体/数字/未决事项）成滚动摘要，**只在 user 边界切**以免产生孤儿 tool 消息。详见 `mini_agent/README.md` 的 “Context management & memory” 一节。

### 长期记忆（AutoMemory 集成）

本仓库 **vendored** 了我自己 vibe-code 的长期记忆系统 **AutoMemory**（独立仓库：<https://github.com/Airmongsity/AutoMemory>），作为可插拔的记忆后端接入 mini_agent：

- **召回时机/放置**：每轮开头 `recall(user_input, user_id)` 一次 → 作为 `system` 块放在“提示/摘要之后、当前轮之前”，答完 `record()` 抽取入库，`report()` 留作效用反馈闭环。
- **作用域**：长期记忆按 `user_id` 命名空间（与 todo 一致）——用户在任一窗口说的个人事实，其他窗口也能召回（“Agent 逐渐认识你”）；对话上下文仍按 session 隔离。
- **已验证**：一个全新的 agent（空工作记忆）连到同一记忆库，能跨“重启”召回上一会话记住的事实。
- 开启方式：`python -m mini_agent.cli --memory`（AutoMemory 需要 `DEEPSEEK_API_KEY` 做抽取 + SiliconFlow key 做 embedding/rerank）。

### 录屏

> 📹 演示录屏：[**▶ record.mp4**](record.mp4)（点击在 GitHub 播放）。演示 Agent 核心流程：多步 loop、双 session 对话隔离、带工具的追问召回，以及用户级 todo / 长期记忆跨窗口共享。

---

## Part 2 · 架构设计题解答

### 模块一 · Context/Performance —— Q2：200 轮 context 快爆，如何压缩且保持流畅

**分层摘要**：近N轮保留原文，溢出旧轮由LLM摘要成滚动Summary，稳定事实抽取进长期记忆，需要用到时再召回。摘要保留数字、日期、人员等信息，只在user轮边界切割，避免切断assistant的`tool_calls`。

**取舍与实测**：纯摘要省 token 但丢细节；纯 RAG 保原文但挤占 context。我在 AutoMemory 上做过对照——LongMemEval-S 上压缩版 **67.3%** vs raw RAG **70.7%**，而两者 session Recall@k 都是 **0.993**。差距不是“检索不到”，而是“压缩丢了逐字细节”。

**我的选择**：滑窗 + 递归摘要（保 open threads/实体）为主，逐字敏感项挂原文指针，长期事实外置到 memory；复杂分层压缩非必要不做。

### 模块二 · Memory —— Q1：熟悉半个月后用户重复问旧问题，如何召回更合理

**方案（即 AutoMemory 的召回链路）**：混合检索（向量 ∪ BM25）→ 打分（`priority=(0.2+0.8·retention)·(1+…)`，`retention=exp(-Δt/S_eff)`）→ 单跳联想扩散 → rerank → top-k。

**“重复问”的关键闭环**：第一次问过 = 该记忆被**检索即复习**（刷新衰减锚点、`access_count+1`），且被 `report_outcome` 正反馈过 utility → **第二次召回排更高**。这就是“越用越熟”的机制。我的 AutoMemory **添加了显式 outcome-feedback / RL reranker**（RMM, ACL 2025，>10% 提升）。**主流开源记忆框架（例如 mem0, MemGPT(Letta), PowerMem）普遍缺显式效用反馈闭环——我在 AutoMemory 补了这一环**。我 held-out 上总体仍略低于 raw RAG，瓶颈是压缩而非检索（Recall@k 持平）。

**我的选择**：混合检索 + 遗忘曲线 + 检索即复习 + 效用反馈；重复问场景靠“复习 + 正反馈”自然收敛。

### 模块三 · Task —— Q2：每天早上 9 点根据昨天聊天做复盘总结

**方案（调度层 + 复用记忆构件）**：① cron/scheduler 按用户时区每天 09:00 触发；② 拉取该用户昨日会话；③ 走 AutoMemory `consolidate()` 把昨日内容抽成 **summary 记忆**，再 LLM 生成结构化复盘（做了什么/进展/未决/今日建议）；④ 复盘**写回长期记忆**（供后续召回）并推送。

**工程实现**：**幂等**（同一天只跑一次，dedup key=user+date）；**失败重试/补偿**；**时区**要对；**昨日无对话**兼容。**边界意识**：AutoMemory 已有 `consolidate()`/summary 构件，但“定时调度”属于 runtime/任务层、需新加。

**我的选择**：独立 scheduler 触发，归纳复用 AutoMemory，结果回灌记忆闭环；解决幂等/时区/空数据等鲁棒性问题。

### 模块四 · Tool / Session Runtime —— Q2：session busy 时又来新消息 / 异步事件

**方案（单会话串行化 + 事件队列 + 状态机）**：每个 session 一台状态机 `idle → busy → waiting_async`，**同 session 内串行**避免历史竞态；一个 per-session 事件队列消费 `user_msg` / `async_tool_done` / `cancel`；busy 时新事件**入队**按序处理；异步完成事件到达时把结果**回灌 context** 唤醒对应轮次继续 loop（对应异步工具的 “提交即返 task_id + 完成通知”）。**默认排队**（保证一致性）；**显式打断才抢占**（用户说“停/改”→ cancel 当前轮、清理半成品）；也可**合并**（把补充信息并入下一次 LLM 输入）。

**取舍**：抢占（响应快、易半成品/竞态）vs 排队（一致、可能让用户等）vs 合并（省一轮、语义可能串）。`mini_agent` 现为**单线程同步**，session 已按 id 隔离（`SessionManager` 线程安全 + 每 session 独立历史）；要支持异步，我会在此之上加“每 session 事件队列 + 状态机”，而不引入跨 session 全局锁——这是从 sync 到 async 的最小增量。

**我的选择**：per-session 串行 + 队列 + 状态机；默认排队、显式抢占、可选合并；异步结果经 task_id 回灌唤醒。

### 模块五 · Agent Runtime 架构对比 —— Q1：Claude Code 的工具输出 vs 国内 OpenAI-compatible function calling

**机制本质区别**
- **OpenAI-compatible（GLM/豆包/DeepSeek 等）**：工具结果是一条 `tool` 角色消息、`content` 是**纯字符串**；`tool_calls.function.arguments` 是**字符串化 JSON**。见 `mini_agent/runtime.py`（回灌 `{"role":"tool","tool_call_id":…,"content":json.dumps(payload)}`）与 `parser.py::salvage_json`。
- **Claude Code（Anthropic Messages）**：根据渠道获取的Claude Code源代码，工具结果是一个 **`tool_result` content block**，放进一条 `user` 消息的 content **数组**里，可与图片等 block 并列；每个工具自实现 `mapToolResultToToolResultBlockParam(content, toolUseID)` 把**强类型 `Output`** 序列化成模型可见 block（`claude-code-main/src/Tool.ts:557`）；带 **`is_error`** 标志把错误**带内**回传（`src/services/tools/toolExecution.ts:1032`）；图片作为顶层 block 与 tool_result 并列实现**同轮多模态**（`toolExecution.ts:1029`）；**并行工具**结果多 block 批在同一条 user 消息（`src/services/compact/sessionMemoryCompact.ts:211`）。

**Claude Code 的一个关键设计：模型视图 / UI 视图 / 搜索视图三分离**——同一工具输出维护 `mapToolResultToToolResultBlockParam`（喂模型）、`renderToolResultMessage`（终端 UI，`Tool.ts:566`）、`extractSearchText`（搜索索引，`Tool.ts:599`）三种表征；OpenAI-compatible 侧只有一份 `content` 字符串，模型视图=日志视图。

**优缺**
- *Anthropic 风格*：结构化 + 多模态 + `is_error` 语义化 + 视图分离 + 天然并行，表达力强、适合复杂/编码 agent；但协议复杂、实现成本高、生态绑定、token 更重。
- *OpenAI-compatible*：通用、协议极简、易调试；但结果被压成字符串→多模态/错误语义受限、易出坏 JSON 需要自行兜底、模型/日志视图混一。

**结论**：**Claude Code 把“工具结果”当成一等的结构化多模态对象并区分受众；OpenAI-compatible 把它当成一条字符串换取最大兼容性。** 做深度/多模态/长程 agent 选前者值那份复杂度；要跨多家国产模型快速落地选后者，代价是自己补充多模态，解决错误语义问题，损坏JSON兜底问题。

---

## 提交物清单

| 交付物 | 位置 |
|---|---|
| 代码（真实 LLM API、自研 runtime） | 本仓库 `mini_agent/` + `tests/` |
| 运行方式 / 系统设计 / memory 召回时机与放置 | 本 README + [`mini_agent/README.md`](mini_agent/README.md) |
| 终端操作录屏 | 见上方 [录屏](#录屏) |
| 架构设计题解答（5 模块） | 本 README [Part 2](#part-2--架构设计题解答) |
| AI Prompt 与问题解决记录 | [`docs/ai-collaboration-log.md`](docs/ai-collaboration-log.md) |
| 长期记忆系统（自研，vendored） | `AutoMemory/`（独立仓库 <https://github.com/Airmongsity/AutoMemory>） |

## 仓库结构

```
.
├── README.md                     # 本文件：提交总入口 + 架构设计题解答
├── requirements.txt
├── mini_agent/                   # Part 1：从零实现的 Agent（自研 runtime）
│   ├── runtime.py  llm_client.py  parser.py  session.py
│   ├── context_manager.py  trace.py  memory.py  cli.py
│   ├── tools/                     # registry + calculator/search/weather/todo
│   ├── examples/demo.py
│   └── README.md                  # 详细系统设计
├── tests/                        # 33 个离线测试（无网络、时间无关）
├── docs/
│   └── ai-collaboration-log.md    # AI Prompt 与问题解决记录
└── AutoMemory/                    # vendored 长期记忆系统（自研）
```
