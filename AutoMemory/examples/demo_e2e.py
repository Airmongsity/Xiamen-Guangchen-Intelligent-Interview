"""Live end-to-end demo: extraction -> recall -> feedback, with real providers."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from automem import AutoMemConfig, AutoMemory

db = os.path.join(tempfile.mkdtemp(), "demo.db")
am = AutoMemory(AutoMemConfig.from_env(db_path=db))

print("=== 1. add(): extract from a conversation ===")
events = am.add(
    [
        {"role": "user", "content": "我叫小明，在杭州做后端开发，主力语言是 Go，但最近在学 Rust。"},
        {"role": "assistant", "content": "Rust 的所有权模型一开始会比较难，建议从官方 book 入手。"},
        {"role": "user", "content": "好，对了我每周三晚上要去打羽毛球，那天别给我排太多任务。"},
    ]
)
for e in events:
    print(f"  [{e.event}] ({e.memory.memory_kind}, imp={e.memory.importance}) {e.memory.content}")

print("\n=== 2. remember(): self-authored memory ===")
m = am.remember("经验：小明偏好直接给结论的简洁回答，不要长篇大论")
print(f"  [{m.memory_kind}, imp={m.importance}, source={m.source}] {m.content}")

print("\n=== 3. observe + recall ===")
am.observe("user", "下周的排期帮我看一下")
result = am.recall("小明周三有什么安排？")
print(result.to_prompt())

print("\n=== 4. report_outcome ===")
out = am.report_outcome(retrieval_id=result.retrieval_id, quality=1.0)
print(f"  {out}")
top = result.long_term[0].memory
print(f"  top memory utility: {am.get(top.id).utility:.3f} (was 0.0)")

print("\n=== 5. update path: contradicting fact ===")
events = am.add([{"role": "user", "content": "更正一下，我现在主力语言换成 Rust 了，Go 只是维护老项目。"}])
for e in events:
    prev = f" (was: {e.previous_content})" if e.previous_content else ""
    content = e.memory.content if e.memory else "-"
    print(f"  [{e.event}] {content}{prev}")

print("\n=== stats ===")
print(f"  {am.stats()}")
am.close()
