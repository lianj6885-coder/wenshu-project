"""工具层测试：Schema 生成、参数兜底、calculator 安全、路径沙箱。

全部离线，不需要真实 LLM。
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import pytest

from assistant.tools import (
    ToolRegistry,
    build_builtin_tools,
    resolve_in_root,
    safe_eval,
    tool,
)


# ---------- Schema 生成 ----------

@tool(description="测试用工具")
def _sample(
    name: Annotated[str, "名称"],
    count: Annotated[int, "数量"] = 3,
    tags: Optional[list[str]] = None,
    flag: bool = False,
) -> str:
    return name


def test_schema_derives_types_and_required():
    params = _sample.parameters
    assert params["type"] == "object"
    assert params["properties"]["name"] == {"type": "string", "description": "名称"}
    assert params["properties"]["count"] == {"type": "integer", "description": "数量", "default": 3}
    assert params["properties"]["tags"] == {"type": "array", "items": {"type": "string"}, "default": None}
    assert params["properties"]["flag"] == {"type": "boolean", "default": False}
    # 只有无默认值的参数才是必填
    assert params["required"] == ["name"]


def test_schema_snake_case_required_excludes_optional_union():
    @tool()
    def _opt_only(value: Optional[str] = None) -> str:
        """可选参数。"""
        return value or ""

    assert "required" not in _opt_only.parameters


def test_schema_is_flat_no_refs():
    """DeepSeek 建议扁平 schema：不能出现 $defs / $ref。"""
    dumped = str(_sample.parameters)
    assert "$ref" not in dumped and "$defs" not in dumped


# ---------- calculator 安全 ----------

@pytest.mark.parametrize("expr,expected", [
    ("(12+8)*3", 60),
    ("2**10", 1024),
    ("7 // 2", 3),
    ("7 % 3", 1),
    ("-5 + 3", -2),
    ("10 / 4", 2.5),
])
def test_safe_eval_allows_arithmetic(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo pwned')",
    "open('/etc/passwd').read()",
    "().__class__.__bases__",
    "os.system('ls')",
    "x = 1",
    "[i for i in range(3)]",
    "lambda: 1",
    "1 if True else 2",
])
def test_safe_eval_rejects_non_arithmetic(expr):
    """Name / Call / Attribute / Subscript 等节点没有分支，直接拒绝。"""
    with pytest.raises(ValueError):
        safe_eval(expr)


def test_safe_eval_rejects_huge_power():
    """9**9**9 是白名单挡不住的内存耗尽，必须在结算前卡住指数。"""
    with pytest.raises(ValueError) as exc:
        safe_eval("9**9**9")
    assert "幂指数过大" in str(exc.value)


def test_safe_eval_rejects_overlong():
    with pytest.raises(ValueError, match="过长"):
        safe_eval("1+" * 200 + "1")


def test_safe_eval_rejects_bool_constant():
    with pytest.raises(ValueError):
        safe_eval("True")


# ---------- invoke 永不抛异常 ----------

@pytest.fixture()
def registry(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(build_builtin_tools(tmp_path))


async def test_unknown_tool_returns_error(registry: ToolRegistry):
    result = await registry.invoke("nope", "{}")
    assert result.ok is False
    assert "未知工具" in result.content
    assert result.content  # 必须非空：空字符串会让模型无法自愈


async def test_malformed_json_returns_error(registry: ToolRegistry):
    result = await registry.invoke("calculator", "这不是 JSON")
    assert result.ok is False
    assert "参数解析失败" in result.content


async def test_json_with_surrounding_prose_is_recovered(registry: ToolRegistry):
    """模型偶尔会在 arguments 里夹带说明文字，应能救回来。"""
    result = await registry.invoke("calculator", '当然，参数是 {"expression": "1+1"}')
    assert result.ok is True
    assert result.content == "2"


async def test_unknown_argument_rejected(registry: ToolRegistry):
    result = await registry.invoke("calculator", '{"expression": "1", "bogus": 1}')
    assert result.ok is False
    assert "不支持的参数" in result.content


async def test_missing_required_argument_rejected(registry: ToolRegistry):
    result = await registry.invoke("calculator", "{}")
    assert result.ok is False
    assert "缺少必填参数" in result.content


async def test_tool_exception_is_captured_not_raised(registry: ToolRegistry):
    """工具内部异常必须变成 ok=False 的结果 —— 这是「每个 tool_call 都有配对」的前提。"""
    result = await registry.invoke("calculator", '{"expression": "1/0"}')
    assert result.ok is False
    assert "ZeroDivisionError" in result.content


async def test_result_message_shape(registry: ToolRegistry):
    result = await registry.invoke("calculator", '{"expression": "1+1"}', tool_call_id="call_x")
    msg = result.as_message()
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_x"
    assert isinstance(msg["content"], str)  # content 必须是字符串
    assert "name" not in msg                # 部分网关会拒绝未知字段


async def test_non_json_result_is_stringified():
    """Path / 对象等非 JSON 值必须在此转成字符串，否则会经
    MemorySessionStore 的 json.dumps(default=str) 变成 "PosixPath('x')"。"""
    import json

    @tool(description="返回 Path")
    def _returns_path() -> Path:
        return Path("sub/file.txt")

    reg = ToolRegistry([_returns_path])
    result = await reg.invoke("_returns_path", "{}")
    assert result.ok is True
    assert isinstance(result.content, str)
    # 反序列化后应还原成路径字符串；不写死分隔符，Windows 上是反斜杠
    assert json.loads(result.content) == str(Path("sub/file.txt"))


# ---------- 路径沙箱 ----------

def test_resolve_in_root_allows_inside(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    assert resolve_in_root(tmp_path, "a.txt") == (tmp_path / "a.txt").resolve()


@pytest.mark.parametrize("escape", ["../secret", "../../etc/passwd", "a/../../out", "./../up"])
def test_resolve_in_root_blocks_escape(tmp_path: Path, escape: str):
    with pytest.raises(ValueError, match="路径越界"):
        resolve_in_root(tmp_path, escape)


def test_resolve_in_root_blocks_symlink_escape(tmp_path: Path):
    """符号链接逃逸：resolve() 先跟随链接再判断，因此挡得住。"""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("当前环境无权限创建符号链接（Windows 需开发者模式）")
    with pytest.raises(ValueError, match="路径越界"):
        resolve_in_root(root, "link")


async def test_file_tools_respect_sandbox(tmp_path: Path):
    reg = ToolRegistry(build_builtin_tools(tmp_path))
    result = await reg.invoke("read_file", '{"path": "../../../etc/passwd"}')
    assert result.ok is False
    assert "路径越界" in result.content


async def test_read_file_rejects_binary(tmp_path: Path):
    (tmp_path / "a.png").write_bytes(b"\x89PNG\r\n")
    reg = ToolRegistry(build_builtin_tools(tmp_path))
    result = await reg.invoke("read_file", '{"path": "a.png"}')
    assert result.ok is False
    assert "二进制" in result.content


async def test_list_files_output_is_console_safe(tmp_path: Path):
    """回归：目录名曾用 emoji 标记，在 Windows GBK 控制台会让 print 抛
    UnicodeEncodeError 直接崩掉。输出必须能被 GBK 编码。"""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    reg = ToolRegistry(build_builtin_tools(tmp_path))
    result = await reg.invoke("list_files", "{}")
    assert result.ok is True
    result.content.encode("gbk")  # 不可编码即失败
    assert "sub/" in result.content


async def test_find_text_returns_matches(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    reg = ToolRegistry(build_builtin_tools(tmp_path))
    result = await reg.invoke("find_text", '{"pattern": "world"}')
    assert result.ok is True
    assert "world" in result.content and ":2:" in result.content
