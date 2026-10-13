"""命令行入口：交互式 REPL 与一次性问答。

用法：
    cd backend
    python -m assistant.cli                          # 进入 REPL
    python -m assistant.cli --once "现在几点了"        # 只问一句
    python -m assistant.cli --workspace ../docs      # 指定文件工具沙箱根目录
    python -m assistant.cli --no-tools               # 禁用工具，退化为纯对话
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from app.llm.client import ChatMessage, LLMError
from app.storage.redis_store import MemorySessionStore
from app.utils.logging import setup_logging

from assistant.agent import DEFAULT_MAX_STEPS, AgentLoop
from assistant.llm import DeepSeekToolClient
from assistant.memory import ConversationMemory
from assistant.settings import (
    echo_reasoning,
    get_assistant_settings,
    max_output_tokens,
    workspace_root,
)
from assistant.tools import ToolRegistry, ToolResult, build_builtin_tools

HELP = """可用命令：
  /help      显示本帮助
  /new       开始新会话（清空当前上下文）
  /history   查看当前会话的消息条数与摘要
  /tools     列出已注册的工具
  /quit      退出（同 /exit、Ctrl-D）"""


def make_summarizer(llm: DeepSeekToolClient):
    """会话摘要器。刻意走不带 tools 的 complete() —— 那条路径不受 reasoning /
    工具配对约束，是生产已验证过的。"""
    async def summarize(messages: list[dict]) -> str:
        conversation = "\n".join(
            f"{m.get('role')}: {str(m.get('content') or '')[:800]}"
            for m in messages
            if m.get("content")
        )
        if not conversation.strip():
            return ""
        return await llm.complete(
            [
                ChatMessage.system(
                    "你是会话摘要助手。把下面的对话压缩成不超过 200 字的要点，"
                    "保留关键事实、结论与未决问题。只输出摘要正文。"
                ),
                ChatMessage.user(conversation),
            ],
            temperature=0.0,
            max_tokens=512,
        )

    return summarize


def print_tool_result(result: ToolResult) -> None:
    mark = "✓" if result.ok else "✗"
    preview = result.content.replace("\n", " ")[:120]
    print(f"  [{mark} {result.name}] {preview}", flush=True)


async def repl(loop: AgentLoop, memory: ConversationMemory, session_id: str) -> None:
    print("文枢助手已就绪。（/help 查看命令，Ctrl-D 退出）")
    while True:
        try:
            line = await asyncio.to_thread(input, "\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/exit", "/q"):
            print("再见。")
            return
        if text == "/help":
            print(HELP)
            continue
        if text == "/new":
            await memory.clear(session_id)
            print("已开始新会话。")
            continue
        if text == "/tools":
            names = loop.registry.names()
            print("已注册工具：" + ("、".join(names) if names else "（无）"))
            continue
        if text == "/history":
            state = await memory.load(session_id)
            print(f"消息 {len(state['messages'])} 条，revision={state['revision']}")
            if state["summary"]:
                print("摘要：" + state["summary"])
            continue

        try:
            answer = await loop.run(session_id, text)
        except LLMError as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print("\n[已中断本轮]")
            continue
        print(f"\n{answer}")


async def main() -> int:
    # argparse 在 main 内部构造，对齐 scripts/doctor.py 的约定
    parser = argparse.ArgumentParser(description="文枢轻量 Agent 助手")
    parser.add_argument("--session", default="", help="会话 id（默认随机生成）")
    parser.add_argument("--once", default="", help="只问一句就退出，不进入 REPL")
    parser.add_argument("--workspace", default="", help="文件工具的沙箱根目录（默认 cwd）")
    parser.add_argument("--no-tools", action="store_true", help="禁用工具，退化为纯对话")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="工具往返轮数上限")
    args = parser.parse_args()

    # Windows 控制台默认 GBK：模型一旦返回 emoji 或罕见字符，print 会抛
    # UnicodeEncodeError 让整个 REPL 崩掉。这里保留原编码（改 UTF-8 反而会让
    # 中文变乱码），只把不可编码字符替换掉。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover —— 非 TextIOWrapper 时跳过
            pass

    settings = get_assistant_settings()
    # 只能在入口调用一次：setup_logging 会替换 root handler，库代码与测试里
    # 调用会劫持对方的日志配置。
    setup_logging(settings.log_level)

    if not settings.deepseek_api_key:
        print("未配置 DEEPSEEK_API_KEY。请在 program/.env 中填写后重试。", file=sys.stderr)
        return 1

    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else workspace_root()
    if not workspace.is_dir():
        print(f"工作根目录不存在：{workspace}", file=sys.stderr)
        return 1

    registry = ToolRegistry([] if args.no_tools else build_builtin_tools(workspace))
    llm = DeepSeekToolClient(settings)
    # 助手不依赖 Redis：直接用具内存实现。需要跨进程持久化时，可换成
    # build_session_store(settings)（但那个会按 STORAGE_MODE 决定是否连 Redis）。
    store = MemorySessionStore()
    memory = ConversationMemory(store, summarizer=make_summarizer(llm))
    loop = AgentLoop(
        llm,
        registry,
        memory,
        workspace=workspace,
        max_steps=args.max_steps,
        max_tokens=max_output_tokens(settings),
        echo_reasoning=echo_reasoning(),
        on_tool_result=print_tool_result,
        on_thinking=lambda: print("思考中…", flush=True),
    )

    print(f"模型 {settings.deepseek_model} · 工具 {len(registry.names())} 个 · 沙箱 {workspace}")

    session_id = args.session or f"cli-{uuid.uuid4().hex[:8]}"

    if args.once:
        try:
            print(await loop.run(session_id, args.once))
        except LLMError as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            return 1
        return 0

    await repl(loop, memory, session_id)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
