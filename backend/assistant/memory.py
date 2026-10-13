"""多轮会话记忆。

为什么不用文枢现成的 WorkingMemory（app/memory/working.py）—— 三个硬伤：
  1. :31 写死 {"role", "content"} 两键，存不下 tool_calls / reasoning_content。
     回放这种历史时，tool 消息会找不到配对的 tool_calls → 400。
  2. :32 的 messages[-max_history:] 是位置切片。工具轮之后列表形如
     [..., user, assistant+tc, tool, tool, assistant, ...]，切在 -10 完全可能
     落在 assistant+tc 与它的 tool 结果中间 → 孤儿 tool 消息 → 400。
     这在用工具的会话里是必然发生，不是边缘情况。
  3. 没有字符预算。它只按条数封顶 10 条，而单条 reasoning trace 就有 1-2 KB。

因此这里重写，但仍复用底层的 SessionStore ABC（其 MemorySessionStore 实现
JSON 安全、无需 connect()/close()），将来换 Redis 是 drop-in。
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Optional

from app.storage.redis_store import SessionStore
from app.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_GROUPS = 20
DEFAULT_KEEP_GROUPS = 8
DEFAULT_MAX_CHARS = 24000
DEFAULT_TTL = 1800
MAX_SUMMARY_CHARS = 2000

Summarizer = Callable[[list[dict[str, Any]]], Awaitable[str]]


def group_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """按对话轮分组：每条 user 消息开启一个新组，其后直到下一条 user 之前的
    assistant / tool 消息都归入该组。

    这是截断安全的核心 —— 按组切分永远不会拆散 tool_calls 与它的 tool 结果。

    以 user 为界的另一个理由同样关键：截断后历史必须以 user（或 system）开头。
    若按「assistant+其后工具」分组，只保留最后一组时历史会以 assistant 开头，
    repair 会正确地把它裁掉 —— 于是整个会话被清空。按 user 分组则不会发生。
    """
    groups: list[list[dict[str, Any]]] = []
    # 当前组内已声明、尚未应答的 tool_call id。
    # 不能只看「前一条是不是 assistant」—— 并行工具调用的序列是
    # assistant[c1,c2] → tool(c1) → tool(c2)，处理 c2 时前一条已是 tool(c1)，
    # 会被误判成孤儿丢弃。
    pending: set[str] = set()

    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            call_id = msg.get("tool_call_id")
            if call_id in pending:
                pending.discard(call_id)
                groups[-1].append(msg)
            else:
                logger.warning("丢弃孤儿 tool 消息: %s", call_id)
            continue

        pending = (
            {tc.get("id") for tc in msg.get("tool_calls") or []}
            if role == "assistant"
            else set()
        )
        if role == "user" or not groups:
            groups.append([msg])
        else:
            groups[-1].append(msg)
    return groups


def repair(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """兜底修复，保证历史是一段合法请求。

    配对损坏有两个方向，都必须查：
      - 孤儿 tool 消息（有结果没声明）—— 由 group_turns 直接丢弃；
      - 未被应答的 tool_calls 声明（有声明没结果）—— 在此剥离 tool_calls，
        降级成普通助手消息；若剥离后内容为空则整条丢弃。

    最后裁掉开头非 user/system 的消息：首条不能是 tool 或 assistant。
    """
    flat = [m for g in group_turns(messages) for m in g]
    answered = {m.get("tool_call_id") for m in flat if m.get("role") == "tool"}

    out: list[dict[str, Any]] = []
    for msg in flat:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            ids = {tc.get("id") for tc in msg["tool_calls"]}
            if not ids <= answered:
                msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                if not msg.get("content"):
                    continue
        out.append(msg)

    while out and out[0].get("role") not in ("user", "system"):
        out.pop(0)
    return out


def _strip_reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """去掉 reasoning_content 的副本。摘要没必要看思考过程，纯浪费 token。"""
    return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]


def _total_chars(groups: list[list[dict[str, Any]]]) -> int:
    return sum(len(json.dumps(m, ensure_ascii=False, default=str)) for g in groups for m in g)


class ConversationMemory:
    """会话记忆：持久化完整的 API 线格式消息，并在组边界做摘要/截断。"""

    def __init__(
        self,
        store: SessionStore,
        *,
        max_groups: int = DEFAULT_MAX_GROUPS,
        keep_groups: int = DEFAULT_KEEP_GROUPS,
        max_chars: int = DEFAULT_MAX_CHARS,
        ttl: int = DEFAULT_TTL,
        summarizer: Optional[Summarizer] = None,
    ) -> None:
        self.store = store
        self.max_groups = max_groups
        self.keep_groups = keep_groups
        self.max_chars = max_chars
        self.ttl = ttl
        self.summarizer = summarizer

    # ---------- 读写 ----------

    async def load(self, session_id: str) -> dict[str, Any]:
        """读取会话状态。返回的是副本（MemorySessionStore 深拷贝 / Redis 反序列化），
        所以修改它不会自动回写 —— 必须显式 save。"""
        state = await self.store.get_session(session_id) or {}
        state.setdefault("messages", [])
        state.setdefault("summary", "")
        state.setdefault("revision", 0)
        return state

    async def save(self, session_id: str, state: dict[str, Any]) -> None:
        # 永远显式回写：MemorySessionStore.get_session 返回的是活引用，而
        # RedisSessionStore 返回副本，依赖别名会在换后端时静默失效。
        state["revision"] = int(state.get("revision", 0)) + 1
        await self.store.set_session(session_id, state, ttl=self.ttl)

    async def append_many(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """把一组消息**原子地**追加进历史。

        调用方必须一次传入完整的 turn 组（assistant + 它全部 tool 结果），
        绝不能拆成两次调用 —— 否则 _shrink 可能切在声明与应答之间。
        """
        if not messages:
            return
        state = await self.load(session_id)
        state["messages"].extend(messages)
        await self._shrink(state)
        await self.save(session_id, state)

    async def build_prompt(self, session_id: str, system: str, user_input: str) -> list[dict[str, Any]]:
        """拼出本轮请求的消息列表。只读，不持久化。"""
        state = await self.load(session_id)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if state["summary"]:
            messages.append({
                "role": "system",
                "content": "早前会话摘要（仅供理解上下文，细节以最近对话为准）：\n" + state["summary"],
            })
        messages.extend(state["messages"])
        messages.append({"role": "user", "content": user_input})
        return messages

    async def clear(self, session_id: str) -> None:
        # MemorySessionStore 的 ttl 被完全忽略且无清理，/clear 必须真的删。
        await self.store.delete_session(session_id)

    # ---------- 内部 ----------

    async def _shrink(self, state: dict[str, Any]) -> None:
        groups = group_turns(state["messages"])
        if len(groups) <= self.max_groups and _total_chars(groups) <= self.max_chars:
            return

        keep = groups[-self.keep_groups :]
        old = groups[: -self.keep_groups]

        if not old:
            # 组数没超上限，只是字符超了，而单个 turn 组自身就超预算。
            # 这时无可摘要的内容，只能原样保留 —— 单轮 24k 字符属病态输入。
            logger.warning(
                "单轮会话内容已达 %d 字符，超过预算 %d，无法再截断",
                _total_chars(groups), self.max_chars,
            )
            state["messages"] = repair([m for g in groups for m in g])
            return

        if self.summarizer is not None:
            try:
                summary = await self.summarizer(_strip_reasoning([m for g in old for m in g]))
                if summary:
                    merged = f"{state['summary']}\n{summary}".strip() if state["summary"] else summary
                    state["summary"] = merged[-MAX_SUMMARY_CHARS:]
            except Exception as exc:  # noqa: BLE001 —— 摘要失败绝不能中断对话
                logger.warning("会话摘要失败，退化为直接截断: %s", exc)

        state["messages"] = repair([m for g in keep for m in g])
