"""工具注册表：@tool 装饰器、JSON Schema 生成、永不抛异常的 invoke。

Schema 刻意手写而非用 pydantic：create_model + model_json_schema() 会产出
$defs / $ref / 每个属性一个 title，而 DeepSeek 的 function calling 建议扁平
schema。用 inspect.signature + get_type_hints 手写约 40 行映射，支持
Annotated[str, "描述"]、list[X]、Optional[X]，并留 schema= 逃生口。
"""
from __future__ import annotations

import inspect
import json
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union, get_args, get_origin, get_type_hints

from app.utils.logging import get_logger

logger = get_logger(__name__)

# 工具结果超过该长度即截断。工具输出会原样进上下文，不设限会让一次
# read_file 就吃掉整个预算。
MAX_TOOL_RESULT_CHARS = 8000

_UNION_TYPES = (Union, types.UnionType)
_PY_TO_JSON: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


# ---------- Schema 生成 ----------

def _unwrap_annotated(tp: Any) -> tuple[Any, str]:
    """拆出 Annotated[X, "描述"] 里的 X 与描述。"""
    if get_origin(tp) is not None and hasattr(tp, "__metadata__"):
        args = get_args(tp)
        desc = next((m for m in tp.__metadata__ if isinstance(m, str)), "")
        return args[0], desc
    return tp, ""


def _is_optional(tp: Any) -> bool:
    origin = get_origin(tp)
    return origin in _UNION_TYPES and type(None) in get_args(tp)


def _json_schema_for(tp: Any) -> dict[str, Any]:
    origin = get_origin(tp)
    if origin in _UNION_TYPES:
        # Optional[X] / X | None：取非 None 的那一支
        args = [a for a in get_args(tp) if a is not type(None)]
        return _json_schema_for(args[0]) if len(args) == 1 else {"type": "object"}
    if origin in (list, tuple, set, frozenset):
        args = get_args(tp)
        return {"type": "array", "items": _json_schema_for(args[0]) if args else {"type": "string"}}
    if origin is dict:
        return {"type": "object"}
    return {"type": _PY_TO_JSON.get(tp, "string")}


def build_parameters_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """从函数签名生成 JSON Schema。

    刻意不发 additionalProperties: false —— 部分 OpenAI 兼容网关会拒绝它。
    等价的效果由 invoke() 的未知参数检查达成，且兼容性更好。
    """
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:  # noqa: BLE001 —— 类型注解无法解析时退化为全部 string
        hints = {}

    sig = inspect.signature(fn)
    props: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        ann, desc = _unwrap_annotated(hints.get(pname, str))
        entry = _json_schema_for(ann)
        if desc:
            entry["description"] = desc
        if param.default is inspect.Parameter.empty:
            if not _is_optional(ann):
                required.append(pname)
        else:
            entry["default"] = param.default
        props[pname] = entry

    out: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


def _first_doc_line(fn: Callable[..., Any]) -> str:
    doc = (fn.__doc__ or "").strip()
    return doc.splitlines()[0].strip() if doc else fn.__name__


# ---------- 数据结构 ----------

@dataclass(frozen=True)
class ToolSpec:
    """一个可调用工具的完整描述。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    readonly: bool = True

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """一次工具执行的结果，永远可转成合法的 tool 消息。"""

    tool_call_id: str
    name: str
    ok: bool
    content: str

    def as_message(self) -> dict[str, Any]:
        # content 必须是字符串，且不带 name 键 —— 部分网关会拒绝未知字段。
        return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}


def tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    readonly: bool = True,
    schema: Optional[dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], ToolSpec]:
    """把一个函数注册为可调用工具。

    readonly=False 的工具会在 AgentLoop 中先经过确认回调再执行。
    """

    def deco(fn: Callable[..., Any]) -> ToolSpec:
        return ToolSpec(
            name=name or fn.__name__,
            description=description or _first_doc_line(fn),
            parameters=schema or build_parameters_schema(fn),
            func=fn,
            readonly=readonly,
        )

    return deco


# ---------- 参数解析与结果序列化 ----------

def parse_arguments(raw: Any) -> dict[str, Any]:
    """解析模型给出的 arguments。容忍夹带说明文字与 markdown 围栏。

    失败一律抛 ValueError，由调用方转成回灌给模型的错误消息。
    """
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {}
    if not isinstance(raw, str):
        raise ValueError(f"arguments 类型错误：{type(raw).__name__}")

    text = raw.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"不是合法 JSON：{text[:120]}") from None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise ValueError(f"不是合法 JSON：{text[:120]}") from None

    if not isinstance(parsed, dict):
        raise ValueError(f"arguments 必须是 JSON 对象，实际为 {type(parsed).__name__}")
    return parsed


def stringify(value: Any) -> str:
    """把工具返回值转成字符串，并施加长度上限。

    MemorySessionStore.set_session 的 json.dumps(default=str) 会把非 JSON 值
    静默字符串化（Path 会变成 "PosixPath('x')"），所以在进入消息列表前就
    统一转成字符串，避免模型收到对象 repr。
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + f"\n...(结果已截断，原长 {len(text)} 字符)"
    return text


# ---------- 注册表 ----------

class ToolRegistry:
    """工具集合。schemas() 给模型看，invoke() 执行。

    invoke() 的核心契约：任何情况下都返回 ToolResult，绝不 raise。未知工具、
    坏 JSON、缺参、工具内部异常 —— 全部变成 ok=False 的结果回灌给模型让它
    自愈。这是「每个 tool_call 必有配对 tool 消息」得以成立的前提，缺一条
    配对就会让下一次请求 400。
    """

    def __init__(self, tools: Optional[list[ToolSpec]] = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for spec in tools or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            logger.warning("工具 %s 被重复注册，后者覆盖前者", spec.name)
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema() for spec in self._tools.values()]

    async def invoke(self, name: str, raw_arguments: Any, tool_call_id: str = "") -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(
                tool_call_id, name, False,
                f"未知工具：{name}。可用工具：{', '.join(self.names())}",
            )

        try:
            args = parse_arguments(raw_arguments)
        except ValueError as exc:
            return ToolResult(
                tool_call_id, name, False,
                f"参数解析失败：{exc}。arguments 必须是 JSON 对象字符串，请修正后重新调用。",
            )

        props = set(spec.parameters.get("properties") or {})
        if unknown := set(args) - props:
            return ToolResult(
                tool_call_id, name, False,
                f"不支持的参数：{sorted(unknown)}。接受的参数：{sorted(props)}",
            )
        if missing := set(spec.parameters.get("required") or []) - set(args):
            return ToolResult(tool_call_id, name, False, f"缺少必填参数：{sorted(missing)}")

        try:
            result = spec.func(**args)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001 —— 必须回灌给模型，而不是让 CLI 崩溃
            logger.warning("工具 %s 执行失败: %s: %s", name, type(exc).__name__, exc)
            return ToolResult(tool_call_id, name, False, f"工具执行失败：{type(exc).__name__}: {exc}")

        return ToolResult(tool_call_id, name, True, stringify(result))
