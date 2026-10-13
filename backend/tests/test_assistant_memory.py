"""会话记忆测试：turn 分组、配对修复、截断安全、摘要触发。

重点是「截断绝不能拆散 tool_calls 与它的 tool 结果」—— 拆散即 400。
"""
from __future__ import annotations

from typing import Any

from app.storage.redis_store import MemorySessionStore

from assistant.memory import ConversationMemory, group_turns, repair


def _assistant_with_calls(*call_ids: str, content: Any = None) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": "想一下",
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "calculator", "arguments": "{}"}}
            for cid in call_ids
        ],
    }


def _tool_result(call_id: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": "60"}


# ---------- group_turns ----------

def test_group_turns_bounds_at_user_messages():
    """一条 user 开启一组，其后的 assistant / tool 全归入该组。

    以 user 为界（而非以 assistant 为界）是截断安全的前提：只保留最后一组时，
    历史必须以 user 开头，否则 repair 会把开头裁光、整个会话被清空。
    """
    messages = [
        {"role": "user", "content": "q1"},
        _assistant_with_calls("c1", "c2"),
        _tool_result("c1"),
        _tool_result("c2"),
        {"role": "assistant", "content": "答案"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "答案2"},
    ]
    groups = group_turns(messages)
    assert [len(g) for g in groups] == [5, 2]
    # 每组都以 user 开头
    assert all(g[0]["role"] == "user" for g in groups)


def test_group_turns_does_not_split_tool_pair():
    groups = group_turns([{"role": "user", "content": "q"}, _assistant_with_calls("c1"), _tool_result("c1")])
    assert [len(g) for g in groups] == [3]


def test_group_turns_drops_orphan_tool_messages():
    groups = group_turns([_tool_result("gone"), {"role": "user", "content": "q"}])
    assert [len(g) for g in groups] == [1]


# ---------- repair：配对损坏的两个方向 ----------

def test_repair_drops_orphan_tool_message():
    messages = [_tool_result("gone"), {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert [m["role"] for m in repair(messages)] == ["user", "assistant"]


def test_repair_strips_unanswered_tool_calls():
    """有声明没结果同样 400，必须剥离 tool_calls 降级为普通助手消息。"""
    messages = [{"role": "user", "content": "q"}, _assistant_with_calls("c9", content="有话要说")]
    repaired = repair(messages)
    assert repaired[-1]["role"] == "assistant"
    assert "tool_calls" not in repaired[-1]
    assert repaired[-1]["content"] == "有话要说"


def test_repair_drops_empty_dangling_assistant():
    """剥离 tool_calls 后内容为空，整条丢弃。"""
    messages = [{"role": "user", "content": "q"}, _assistant_with_calls("c9")]
    assert [m["role"] for m in repair(messages)] == ["user"]


def test_repair_trims_non_user_head():
    messages = [{"role": "assistant", "content": "hi"}, _tool_result("x"), {"role": "user", "content": "q"}]
    assert [m["role"] for m in repair(messages)] == ["user"]


def test_repair_keeps_intact_pairing_untouched():
    messages = [{"role": "user", "content": "q"}, _assistant_with_calls("c1"), _tool_result("c1")]
    assert repair(messages) == messages


# ---------- ConversationMemory ----------

async def test_roundtrip_preserves_tool_calls_and_reasoning():
    """MemorySessionStore 走 json 深拷贝，tool_calls / reasoning_content 必须存活。"""
    mem = ConversationMemory(MemorySessionStore())
    assistant = _assistant_with_calls("c1")
    await mem.append_many("s1", [{"role": "user", "content": "q"}, assistant, _tool_result("c1")])

    state = await mem.load("s1")
    stored = state["messages"][1]
    assert stored["reasoning_content"] == "想一下"
    assert stored["tool_calls"][0]["id"] == "c1"
    assert stored["tool_calls"][0]["function"]["name"] == "calculator"


async def test_build_prompt_prepends_system_and_appends_user():
    mem = ConversationMemory(MemorySessionStore())
    await mem.append_many("s1", [{"role": "user", "content": "早先的问题"}])
    prompt = await mem.build_prompt("s1", "系统提示", "新的问题")

    assert prompt[0] == {"role": "system", "content": "系统提示"}
    assert prompt[1] == {"role": "user", "content": "早先的问题"}
    assert prompt[-1] == {"role": "user", "content": "新的问题"}


async def test_build_prompt_includes_summary_when_present():
    mem = ConversationMemory(MemorySessionStore())
    await mem.save("s1", {"messages": [], "summary": "聊过退课", "revision": 0})
    prompt = await mem.build_prompt("s1", "系统", "q")
    assert "早前会话摘要" in prompt[1]["content"]
    assert "聊过退课" in prompt[1]["content"]


async def test_truncation_never_splits_a_tool_pair():
    """核心不变量：把历史压到只剩 1 组时，被保留的组必须是完整的。"""
    mem = ConversationMemory(MemorySessionStore(), max_groups=10, keep_groups=1)
    for i in range(12):
        await mem.append_many("s1", [
            {"role": "user", "content": f"q{i}"},
            _assistant_with_calls(f"c{i}"),
            _tool_result(f"c{i}"),
        ])

    state = await mem.load("s1")
    messages = state["messages"]
    declared = {
        tc["id"]
        for m in messages if m["role"] == "assistant" and m.get("tool_calls")
        for tc in m["tool_calls"]
    }
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    assert declared == answered, "截断后出现了未配对或孤儿的 tool 消息"
    assert messages[0]["role"] == "user"


async def test_shrink_triggers_on_char_budget_not_just_count():
    """reasoning trace 很长，按条数封顶不够，必须按字符预算触发。"""
    mem = ConversationMemory(MemorySessionStore(), max_groups=1000, keep_groups=1, max_chars=2000)
    for i in range(6):
        await mem.append_many("s1", [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": "很长" * 400, "reasoning_content": "思考" * 400},
        ])
    assert len((await mem.load("s1"))["messages"]) < 12


async def test_summarizer_is_called_and_result_stored():
    seen: list[list[dict]] = []

    async def fake_summary(messages: list[dict]) -> str:
        seen.append(messages)
        return "用户反复问退课"

    mem = ConversationMemory(MemorySessionStore(), max_groups=3, keep_groups=1, summarizer=fake_summary)
    for i in range(5):
        await mem.append_many("s1", [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}])

    assert seen, "摘要器没有被调用"
    assert "退课" in (await mem.load("s1"))["summary"]


async def test_summarizer_receives_stripped_reasoning():
    captured: list[list[dict]] = []

    async def fake_summary(messages: list[dict]) -> str:
        captured.extend(messages)
        return "摘要"

    mem = ConversationMemory(MemorySessionStore(), max_groups=2, keep_groups=1, summarizer=fake_summary)
    for i in range(4):
        await mem.append_many("s1", [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}", "reasoning_content": "很长的思考" * 50},
        ])
    assert captured and all("reasoning_content" not in m for m in captured)


async def test_summarizer_failure_degrades_to_plain_truncation():
    async def boom(messages: list[dict]) -> str:
        raise RuntimeError("模型挂了")

    mem = ConversationMemory(MemorySessionStore(), max_groups=2, keep_groups=1, summarizer=boom)
    for i in range(5):
        await mem.append_many("s1", [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}])
    # 摘要失败不能中断对话，仍应完成截断
    assert len((await mem.load("s1"))["messages"]) <= 4


async def test_clear_removes_session():
    mem = ConversationMemory(MemorySessionStore())
    await mem.append_many("s1", [{"role": "user", "content": "q"}])
    await mem.clear("s1")
    assert (await mem.load("s1"))["messages"] == []
