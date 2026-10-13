"""Agent 循环：模型 ⇄ 工具往返，直到给出最终回答。

三条不可违反的不变量：
  1. 每个 tool_call 恰好产出一条 tool 消息（靠 ToolRegistry.invoke 永不抛异常
     保证）—— 缺一条配对，下一次请求就 400。
  2. append_many([assistant, *tools]) 原子写入，绝不拆成两次 —— 否则 _shrink
     可能切在声明与应答之间。
  3. 确认串行、执行并行 —— 用 asyncio.gather 批量弹交互式确认会同时问两个问题。
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from app.utils.logging import get_logger

from assistant.llm import ToolCallingLLM
from assistant.memory import ConversationMemory
from assistant.tools.registry import ToolRegistry, ToolResult, ToolSpec, parse_arguments

logger = get_logger(__name__)

DEFAULT_MAX_STEPS = 6

SYSTEM_PROMPT_TEMPLATE = """你是「文枢助手」，一个可以调用工具来完成任务的中文助手。

【工作方式】
- 需要事实、数字、时间或文件内容时调用工具，不要凭记忆猜测。
- 可以在一轮里并行调用多个工具；信息够了就直接回答，不要重复调用同一个工具。
- 工具执行失败时，错误信息会回传给你，请据此修正参数重试，或换一种做法。

【可用工具】
{tools}

【文件访问范围】
文件类工具被限制在工作根目录内：{workspace}"""


def build_system_prompt(registry: ToolRegistry, workspace: Path) -> str:
    lines = [
        f"- {s['function']['name']}：{s['function']['description']}"
        for s in registry.schemas()
    ]
    return SYSTEM_PROMPT_TEMPLATE.format(
        tools="\n".join(lines) or "- （当前没有可用工具）",
        workspace=workspace,
    )


class AgentLoop:
    def __init__(
        self,
        llm: ToolCallingLLM,
        registry: ToolRegistry,
        memory: ConversationMemory,
        *,
        workspace: Optional[Path] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_tokens: int = 4096,
        echo_reasoning: bool = True,
        on_tool_result: Optional[Callable[[ToolResult], None]] = None,
        on_thinking: Optional[Callable[[], None]] = None,
        confirmer: Optional[Callable[[ToolSpec, dict[str, Any]], Awaitable[bool]]] = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.memory = memory
        self.workspace = workspace or Path.cwd()
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.echo_reasoning = echo_reasoning
        self.on_tool_result = on_tool_result
        self.on_thinking = on_thinking
        self.confirmer = confirmer
        self.system_prompt = build_system_prompt(registry, self.workspace)

    async def run(self, session_id: str, user_input: str) -> str:
        messages = await self.memory.build_prompt(session_id, self.system_prompt, user_input)
        await self.memory.append_many(session_id, [{"role": "user", "content": user_input}])
        # 空工具集时传 None，让模型退化为纯对话
        schemas = self.registry.schemas() or None

        for _ in range(self.max_steps):
            if self.on_thinking:
                self.on_thinking()
            turn = await self.llm.chat(messages, tools=schemas, max_tokens=self.max_tokens)
            assistant_msg = turn.as_message(echo_reasoning=self.echo_reasoning)
            messages.append(assistant_msg)

            if not turn.tool_calls:
                await self.memory.append_many(session_id, [assistant_msg])
                if turn.truncated and not turn.content:
                    return "模型输出被截断（达到 max_tokens 上限），请重试或简化问题。"
                return turn.content or "（模型没有返回内容）"

            results = await self._execute(turn.tool_calls)
            tool_msgs = [r.as_message() for r in results]
            # 原子：assistant 声明与其全部 tool 应答必须一次落盘
            await self.memory.append_many(session_id, [assistant_msg, *tool_msgs])
            messages.extend(tool_msgs)

        # max_steps 耗尽。因为每次工具调用都必有应答，此刻历史仍是合法请求，
        # 直接追加一句收尾指令即可。
        messages.append({
            "role": "user",
            "content": "已达到工具调用上限，请基于以上信息直接给出最终回答。",
        })
        if self.on_thinking:
            self.on_thinking()
        final = await self.llm.chat(messages, tools=None, max_tokens=self.max_tokens)
        final_msg = final.as_message(echo_reasoning=self.echo_reasoning)
        await self.memory.append_many(session_id, [final_msg])
        return final.content or "已达到工具调用上限，未能给出最终回答。"

    # ---------- 内部 ----------

    async def _execute(self, tool_calls: list[dict[str, Any]]) -> list[ToolResult]:
        """执行一轮内的全部工具调用，保证每个都有且只有一条结果。"""
        needs_confirm = any(self._is_write(tc) for tc in tool_calls)
        if needs_confirm:
            # 确认是交互式的，串行执行以免同时弹出多个提问
            return [await self._one(tc) for tc in tool_calls]
        return list(await asyncio.gather(*(self._one(tc) for tc in tool_calls)))

    def _is_write(self, tool_call: dict[str, Any]) -> bool:
        spec = self.registry.get(self._name_of(tool_call))
        return spec is not None and not spec.readonly

    @staticmethod
    def _name_of(tool_call: dict[str, Any]) -> str:
        return (tool_call.get("function") or {}).get("name") or ""

    async def _one(self, tool_call: dict[str, Any]) -> ToolResult:
        name = self._name_of(tool_call)
        call_id = tool_call.get("id") or ""
        raw_args = (tool_call.get("function") or {}).get("arguments")

        spec = self.registry.get(name)
        if spec is not None and not spec.readonly:
            approved = False
            if self.confirmer is not None:
                try:
                    args = parse_arguments(raw_args)
                except ValueError:
                    args = {}
                approved = await self.confirmer(spec, args)
            if not approved:
                return ToolResult(
                    call_id, name, False,
                    "用户拒绝执行该操作（该工具会修改文件或系统状态）。",
                )

        # 参数解析与异常兜底全部交给 invoke —— 它保证不抛异常
        result = await self.registry.invoke(name, raw_args, tool_call_id=call_id)
        if self.on_tool_result is not None:
            self.on_tool_result(result)
        return result
