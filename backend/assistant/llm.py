"""工具调用 LLM 客户端。

复用文枢的 DeepSeekClient，但父类的 complete() 无法表达工具调用：它在
client.py:100 把响应塌缩成 `content or ""`，tool_calls 被静默丢弃，且返回类型
是 str。给它加 tools= 参数会变成「模型发了 tool_calls、调用方收到空字符串」
的静默错误，并把工具链路的 bug 引到生产问答系统的热路径上。

所以这里用子类纯增量地加一个 chat()，复用父类的 _headers() / _chat_url() /
self.timeout / LLMError。生产代码零改动。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import httpx

from app.llm.client import LLMError
from app.llm.deepseek import DeepSeekClient
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _normalize_tool_calls(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """归一化 tool_calls，并在网关漏发 id 时补齐。

    缺 id 会导致后续 tool 消息无法配对，直接 400 —— 而 id 是模型调用与结果
    绑定的唯一凭据，必须保证存在。
    """
    out: list[dict[str, Any]] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        out.append({
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "{}",
            },
        })
    return out


@dataclass
class AssistantTurn:
    """一次模型回复：可能带工具调用，也可能只有文本。"""

    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""

    @property
    def truncated(self) -> bool:
        """输出被 max_tokens 截断。"""
        return self.finish_reason == "length"

    def as_message(self, *, echo_reasoning: bool = True) -> dict[str, Any]:
        """转成可写回历史的 assistant 消息。

        有 tool_calls 时 content 保持 None（与 API 返回一致）；无 tool_calls 时
        归一为 ""，避免历史里出现 content=null 的普通助手消息。

        reasoning_content 的取舍见 settings.echo_reasoning() 的说明：带 tools 的
        请求必须回传，否则可能 400。
        """
        content: Optional[str]
        if self.content is not None:
            content = self.content
        else:
            content = None if self.tool_calls else ""

        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if echo_reasoning and self.reasoning_content:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class ToolCallingLLM(Protocol):
    """最小契约，便于测试注入假实现（对齐 tests/test_agents.py 的鸭子类型约定）。"""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> AssistantTurn: ...


class DeepSeekToolClient(DeepSeekClient):
    """在 DeepSeekClient 之上增加支持 function calling 的 chat()。"""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> AssistantTurn:
        if not self.api_key:
            raise LLMError("未配置 API Key，无法调用 LLM")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            # 刻意不发 temperature，也不发 tool_choice：deepseek-v4-flash 的
            # thinking 模式会拒绝 tool_choice 并禁用温度类采样参数，让 tool_choice
            # 默认 auto 是最安全的构造。非流式下显式 max_tokens 是必需的（与
            # client.py:118-121 记录的流式坑不同，那个坑只存在于流式）。
        else:
            payload["temperature"] = self.temperature

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self._chat_url(), headers=self._headers(), json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            logger.error("LLM(tools) HTTP %s: %s", exc.response.status_code, body)
            raise LLMError(f"LLM 请求失败 {exc.response.status_code}: {body}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM 网络错误: {exc}") from exc

        try:
            choice = data["choices"][0]
            raw = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM 响应格式异常: {data}") from exc

        return AssistantTurn(
            content=raw.get("content"),
            reasoning_content=raw.get("reasoning_content"),
            tool_calls=_normalize_tool_calls(raw.get("tool_calls") or []),
            finish_reason=choice.get("finish_reason") or "",
        )
