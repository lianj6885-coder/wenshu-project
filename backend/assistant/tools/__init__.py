"""工具层：注册表、路径沙箱、内置工具。"""
from __future__ import annotations

from assistant.tools.builtin import build_builtin_tools, safe_eval
from assistant.tools.registry import (
    MAX_TOOL_RESULT_CHARS,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    parse_arguments,
    stringify,
    tool,
)
from assistant.tools.sandbox import is_binary, resolve_in_root

__all__ = [
    "MAX_TOOL_RESULT_CHARS",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "build_builtin_tools",
    "is_binary",
    "parse_arguments",
    "resolve_in_root",
    "safe_eval",
    "stringify",
    "tool",
]
