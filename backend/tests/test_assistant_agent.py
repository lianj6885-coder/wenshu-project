"""Agent 循环测试：用脚本化的假 LLM 驱动，全程离线。

覆盖三条不变量：每个 tool_call 必有且只有一条 tool 应答、assistant 与其工具
结果原子落盘、max_steps 耗尽后历史仍是合法请求。
"""
from __future__ import annotations

import copy
from typing import Annotated, Any

import pytest

from app.storage.redis_store import MemorySessionStore

from assistant.agent import AgentLoop
from assistant.llm import AssistantTurn
from assistant.memory import ConversationMemory
from assistant.tools import ToolRegistry, tool


@tool(description="回显文本")
def echo(text: Annotated[str, "要回显的文本"]) -> str:
    return f"echo:{text}"


@tool(description="会写文件（非只读）", readonly=False)
def write_thing(target: Annotated[str, "目标"]) -> str:
    return f"wrote:{target}"


class _FakeToolLLM:
    """按脚本依次返回 AssistantTurn，并记录每次收到的请求。"""

    def __init__(self, *turns: AssistantTurn) -> None:
        self.turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    async def chat(self, messages, *, tools=None, max_tokens=None) -> AssistantTurn:
        self.requests.append({"messages": copy.deepcopy(messages), "tools": tools, "max_tokens": max_tokens})
        if not self.turns:
            raise AssertionError("假 LLM 的脚本已用尽，说明循环多跑了一轮")
        return self.turns.pop(0)


def _call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _make_loop(llm, tmp_path, **kwargs) -> AgentLoop:
    registry = ToolRegistry([echo, write_thing])
    memory = ConversationMemory(MemorySessionStore())
    return AgentLoop(llm, registry, memory, workspace=tmp_path, max_tokens=4096, **kwargs)


async def test_direct_answer_without_tools(tmp_path):
    llm = _FakeToolLLM(AssistantTurn(content="你好，我是助手", finish_reason="stop"))
    loop = _make_loop(llm, tmp_path)

    assert await loop.run("s1", "你好") == "你好，我是助手"
    assert len(llm.requests) == 1


async def test_single_tool_call_roundtrip(tmp_path):
    llm = _FakeToolLLM(
        AssistantTurn(content=None, reasoning_content="需要回显", tool_calls=[_call("c1", "echo", '{"text":"hi"}')],
                      finish_reason="tool_calls"),
        AssistantTurn(content="结果是 echo:hi", finish_reason="stop"),
    )
    loop = _make_loop(llm, tmp_path)

    assert await loop.run("s1", "回显 hi") == "结果是 echo:hi"

    # 第二次请求必须带上 assistant 声明与其配对的 tool 结果
    second = llm.requests[1]["messages"]
    assistant = next(m for m in second if m.get("tool_calls"))
    tool_msg = next(m for m in second if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert tool_msg["content"] == "echo:hi"


async def test_parallel_tool_calls_all_get_answers(tmp_path):
    llm = _FakeToolLLM(
        AssistantTurn(content=None, tool_calls=[
            _call("c1", "echo", '{"text":"a"}'),
            _call("c2", "echo", '{"text":"b"}'),
            _call("c3", "echo", '{"text":"c"}'),
        ], finish_reason="tool_calls"),
        AssistantTurn(content="三个都做完了", finish_reason="stop"),
    )
    loop = _make_loop(llm, tmp_path)
    await loop.run("s1", "回显三个")

    second = llm.requests[1]["messages"]
    declared = {tc["id"] for m in second if m.get("tool_calls") for tc in m["tool_calls"]}
    answered = {m["tool_call_id"] for m in second if m["role"] == "tool"}
    assert declared == {"c1", "c2", "c3"}
    assert answered == declared


async def test_unknown_tool_error_is_fed_back_for_self_healing(tmp_path):
    llm = _FakeToolLLM(
        AssistantTurn(content=None, tool_calls=[_call("c1", "nonexistent", "{}")], finish_reason="tool_calls"),
        AssistantTurn(content="我换个方式回答", finish_reason="stop"),
    )
    loop = _make_loop(llm, tmp_path)
    assert await loop.run("s1", "调个不存在的工具") == "我换个方式回答"

    second = llm.requests[1]["messages"]
    tool_msg = next(m for m in second if m["role"] == "tool")
    assert "未知工具" in tool_msg["content"]  # 错误必须回灌，且非空


async def test_malformed_arguments_still_yield_one_answer(tmp_path):
    llm = _FakeToolLLM(
        AssistantTurn(content=None, tool_calls=[_call("c1", "echo", "坏的 JSON")], finish_reason="tool_calls"),
        AssistantTurn(content="已修正", finish_reason="stop"),
    )
    loop = _make_loop(llm, tmp_path)
    await loop.run("s1", "q")

    second = llm.requests[1]["messages"]
    assert [m for m in second if m["role"] == "tool"][0]["tool_call_id"] == "c1"


async def test_persisted_history_is_always_pairable(tmp_path):
    """落盘的历史必须能直接作为下一轮请求使用：配对完整、首条是 user。"""
    llm = _FakeToolLLM(
        AssistantTurn(content=None, tool_calls=[_call("c1", "echo", '{"text":"x"}'), _call("c2", "echo", '{"text":"y"}')],
                      finish_reason="tool_calls"),
        AssistantTurn(content="完成", finish_reason="stop"),
    )
    loop = _make_loop(llm, tmp_path)
    await loop.run("s1", "第一问")

    state = await loop.memory.load("s1")
    roles = [m["role"] for m in state["messages"]]
    assert roles[0] == "user"
    declared = {tc["id"] for m in state["messages"] if m.get("tool_calls") for tc in m["tool_calls"]}
    answered = {m["tool_call_id"] for m in state["messages"] if m["role"] == "tool"}
    assert declared == answered


async def test_reasoning_content_is_echoed_when_present(tmp_path):
    llm = _FakeToolLLM(AssistantTurn(content="答", reasoning_content="思考过程", finish_reason="stop"))
    loop = _make_loop(llm, tmp_path, echo_reasoning=True)
    await loop.run("s1", "q")

    state = await loop.memory.load("s1")
    assistant = next(m for m in state["messages"] if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "思考过程"


async def test_reasoning_echo_can_be_disabled(tmp_path):
    llm = _FakeToolLLM(AssistantTurn(content="答", reasoning_content="思考过程", finish_reason="stop"))
    loop = _make_loop(llm, tmp_path, echo_reasoning=False)
    await loop.run("s1", "q")

    state = await loop.memory.load("s1")
    assistant = next(m for m in state["messages"] if m["role"] == "assistant")
    assert "reasoning_content" not in assistant


async def test_max_steps_exhaustion_finishes_cleanly(tmp_path):
    turns = [
        AssistantTurn(content=None, tool_calls=[_call(f"c{i}", "echo", '{"text":"x"}')], finish_reason="tool_calls")
        for i in range(3)
    ]
    llm = _FakeToolLLM(*turns, AssistantTurn(content="最终回答", finish_reason="stop"))
    loop = _make_loop(llm, tmp_path, max_steps=3)

    assert await loop.run("s1", "一直调工具") == "最终回答"

    # 收尾那次调用必须不带 tools，否则模型可能继续调工具
    assert llm.requests[-1]["tools"] is None
    # 且历史仍然合法：没有未应答的 tool_calls
    final_messages = llm.requests[-1]["messages"]
    declared = {tc["id"] for m in final_messages if m.get("tool_calls") for tc in m["tool_calls"]}
    answered = {m["tool_call_id"] for m in final_messages if m["role"] == "tool"}
    assert declared == answered


async def test_truncated_output_reports_clearly(tmp_path):
    llm = _FakeToolLLM(AssistantTurn(content=None, finish_reason="length"))
    loop = _make_loop(llm, tmp_path)
    answer = await loop.run("s1", "q")
    assert "截断" in answer


async def test_non_readonly_tool_requires_confirmation(tmp_path):
    asked: list[str] = []

    async def confirmer(spec, args) -> bool:
        asked.append(spec.name)
        return False

    llm = _FakeToolLLM(
        AssistantTurn(content=None, tool_calls=[_call("c1", "write_thing", '{"target":"a"}')],
                      finish_reason="tool_calls"),
        AssistantTurn(content="好的，不写了", finish_reason="stop"),
    )
    loop = _make_loop(llm, tmp_path, confirmer=confirmer)
    await loop.run("s1", "写点东西")

    assert asked == ["write_thing"]
    second = llm.requests[1]["messages"]
    tool_msg = next(m for m in second if m["role"] == "tool")
    assert "拒绝" in tool_msg["content"]
    assert tool_msg["tool_call_id"] == "c1"


async def test_readonly_tool_skips_confirmation(tmp_path):
    async def confirmer(spec, args) -> bool:
        raise AssertionError("只读工具不该走确认")

    llm = _FakeToolLLM(
        AssistantTurn(content=None, tool_calls=[_call("c1", "echo", '{"text":"x"}')], finish_reason="tool_calls"),
        AssistantTurn(content="ok", finish_reason="stop"),
    )
    loop = _make_loop(llm, tmp_path, confirmer=confirmer)
    assert await loop.run("s1", "q") == "ok"


async def test_multi_turn_context_is_replayed(tmp_path):
    """第二轮请求必须包含第一轮的全部历史。"""
    llm = _FakeToolLLM(
        AssistantTurn(content="第一条回答", finish_reason="stop"),
        AssistantTurn(content="第二条回答", finish_reason="stop"),
    )
    loop = _make_loop(llm, tmp_path)
    await loop.run("s1", "第一问")
    await loop.run("s1", "第二问")

    second = llm.requests[1]["messages"]
    contents = [m.get("content") for m in second]
    assert "第一问" in contents and "第一条回答" in contents
    assert contents[-1] == "第二问"


async def test_system_prompt_lists_tools_and_workspace(tmp_path):
    llm = _FakeToolLLM(AssistantTurn(content="hi", finish_reason="stop"))
    loop = _make_loop(llm, tmp_path)
    await loop.run("s1", "q")

    system = llm.requests[0]["messages"][0]
    assert system["role"] == "system"
    assert "echo" in system["content"]
    assert str(tmp_path) in system["content"]


async def test_tool_schemas_are_passed_to_model(tmp_path):
    llm = _FakeToolLLM(AssistantTurn(content="hi", finish_reason="stop"))
    loop = _make_loop(llm, tmp_path)
    await loop.run("s1", "q")

    names = {s["function"]["name"] for s in llm.requests[0]["tools"]}
    assert names == {"echo", "write_thing"}
