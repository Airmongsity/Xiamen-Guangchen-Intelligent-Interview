# AI Prompt 与问题解决记录

本文件记录本项目开发中我如何**用 AI 辅助思考与排障**，以及每个关键决策的依据。核心原则是：让 AI 帮我**诊断和权衡**，决策与验收由我把关。下面按“问题 → 如何定位 → 决策/结论”组织。

## 0. 技术选型

- **Prompt 意图**：从零实现最小 Agent，语言 Python，走 OpenAI-compatible API，核心 runtime 自研（不用 langgraph/openhands/openclaw）。
- **决策**：工具调用用**原生 function calling**（`tools=[…]` → `tool_calls`）而非纯文本 ReAct 解析——因为选了 OpenAI-compatible 栈，结构化 `tool_calls` 最稳；同时 `parser.py` 仍对坏 JSON 兜底（现实中模型会把思考混进 content、或吐截断 JSON）。

## 1. “Agent 不能 function call” —— 其实是模型太弱，不是缺 MCP

- **现象**：接入后工具调用请求卡住/超时，一度怀疑是不是要建 MCP。
- **定位（分层排除）**：
  1. 先做**最小直连**调用（不带 tools）确认连通性 → 3.2s 返回，但 `Qwen/Qwen2.5-7B-Instruct` 输出退化（对“say hi”吐出编造的多轮对话）。
  2. 再单独测**带 tools** 的一次调用 → 25s 超时。
  3. 换 `deepseek-ai/DeepSeek-V3` / `Qwen/Qwen2.5-72B-Instruct` / `zai-org/GLM-5.2` 同一 key 测试 → 均 ~2s 返回**合法 tool_call**。
- **结论**：runtime、retry、trace 全部正常；瓶颈是 **7B 模型 function-calling 能力不足**。**无需 MCP**——mini_agent 用原生 function calling 即满足要求。README 明确标注了可用模型。

## 2. `.env` schema 不匹配 + BOM

- **现象**：`DEEPSEEK_API_KEY` 读不到，启动即报缺 key。
- **定位**：直接读 `.env` 字节，发现实际是 SiliconFlow 的通用 schema（`CHAT_API_URL`/`CHAT_API_KEY`/`CHAT_MODEL` + `EMBEDDING_*`/`RERANKER_*`），并无 `DEEPSEEK_API_KEY`；且 `CHAT_API_URL` 是含 `/chat/completions` 的完整端点。
- **决策**：`config.py` 兼容两套命名（`CHAT_*` 优先，回退 `DEEPSEEK_*`），并从 `CHAT_API_URL` 剥掉 `/chat/completions` 得到 SDK `base_url`；`.env` 读取用 `utf-8-sig` 以防 Windows 编辑器写入的 BOM。

## 3. Windows 控制台 emoji 崩溃

- **现象**：GLM 的回答带 emoji（🌤️），打印时 `UnicodeEncodeError: 'gbk' codec…`。
- **定位**：是 Windows 控制台默认 GBK，不是 agent 逻辑（agent 本身已正确产出答案）。
- **决策**：`cli.py` 启动时把 `stdout/stderr` reconfigure 为 utf-8，避免录屏时 REPL 崩。

## 4. Context 压缩的正确性陷阱：不能切断工具对

- **思考**：压缩旧轮时若在 assistant 的 `tool_calls` 与其 `tool` 结果之间切断，OpenAI 兼容 API 会直接报错（孤儿 tool 消息）。
- **决策**：`context_manager._safe_split` 只在 **user 轮边界**切；摘要作为 system 文本注入，不破坏工具对。并写了对应测试固化这条不变量。

## 5. 测试先行抓到真实行为差异

- **现象**：`weather` 工具测试因 `Xiamen` vs `xiamen` 断言失败。
- **定位**：工具**故意回显调用者原始拼写**（`city` 字段），而查表是大小写无关的——是测试断言写错，不是工具错。
- **决策**：改断言为比较天气数据字段而非回显拼写；工具行为保持不变。体现“测完再回复、回归先失败”的流程。

## 6. AutoMemory 接入

- **意图**：把我自研的长期记忆系统 AutoMemory 作为可插拔后端接入，满足“记住状态/带工具的追问/memory 召回时机与放置”。
- **关键设计**：
  - **作用域**用 `user_id` 命名空间（读源码确认 facade 每个方法都带 `user_id`）——见第 9 条，最终与 todo 一样按用户，而非按会话。
  - **召回时机/放置**：每轮开头 recall 一次（不在每步重复，省 LLM/embed 调用），作为 system 块放在“提示/摘要之后、当前轮之前”。
  - **打通 key 命名差异**：AutoMemory 读 `SILICONFLOW_API_KEY`，本仓库 `.env` 叫 `EMBEDDING_API_KEY`，适配器透传，无需改 `.env`。
  - **失败不破坏当轮**：recall/record 异常被捕获降级为 no-op。
- **验证**：新建一个空工作记忆的 agent，连同一记忆库，成功**跨“重启”**召回上一会话记住的事实（Rust/Globex）。

## 7. Embedding/Reranker 是否升级 —— 决定“求稳不换”

- **权衡**：AutoMemory 默认 `BAAI/bge-m3`（1024 维）+ `bge-reranker-v2-m3`。
- **结论（不换）**：① 换了会**作废 AutoMemory 已有的评测数字**；② **检索不是瓶颈**（session Recall@k ≈ 0.993 已近满），瓶颈是压缩丢逐字细节

## 8. 模块五 Q1 用真实源码坐实

- **做法**：直接读本机 `claude-code-main/` 源码，用 `Tool.ts:557`（`mapToolResultToToolResultBlockParam`）、`toolExecution.ts:1029/1032`（图片并列 + `is_error`）、`sessionMemoryCompact.ts:211`（并行 tool_result）等**行号级证据**支撑“Claude Code 结构化 block vs OpenAI 字符串”的对比，避免空谈。

## 9. todo 的作用域：从“会话级”改为“用户级”

- **现象**：录屏时在会话 A 加待办、切到新会话问“我的待办”返回空。
- **定位**：不是 bug——原实现把 todo 放在 `session.todos`（会话工作内存），换会话/重启即空。而我最初把它做成会话隔离，其实是为了演示 session 隔离。
- **重新审题**：题面只要求“两个窗口是独立 session、彼此不影响”——指的是**对话上下文隔离**，且是**同一个用户 A**，并未要求待办按会话隔离。一个提醒（todo）本就属于**用户**，应跨窗口可见。
- **决策**：`session.py` 把 todo 改为**用户级共享**（`SessionManager` 维护 `_todos_by_user`，同一 user 的各 session 的 `todos` 指向同一份列表）；对话历史仍按 session 隔离。更新了测试（`test_todo_is_user_scoped_not_session_scoped`、`test_todos_shared_across_sessions_of_same_user`）与 demo，真实模型验证：会话 A/B 加的待办能在会话 C 一并列出。
- **连带一致性**：既然 todo 是用户级，长期记忆（AutoMemory）也从会话级命名空间统一改为**用户级**（`memory.py`/`runtime.py` 用 `session.user_id` 而非 `session_id`），符合模块二“Agent 逐渐认识你”的定位；对话上下文仍按 session 隔离。三条作用域由此自洽：对话=会话级，todo/长期记忆=用户级。

---

**贯穿始终的方法**：先用最小实验把“连通性 / 我的代码 / 模型能力 / 环境（编码、.env）”分层隔离，再定位根因；每个决策都落到可运行的验证（33 个离线测试 + 真实 API 端到端）。
