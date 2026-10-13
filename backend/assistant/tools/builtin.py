"""五个内置工具：calculator / get_current_time / list_files / read_file / find_text。

用工厂函数而非模块级常量，是为了不在 import 时读取 ASSISTANT_WORKSPACE ——
根目录在调用时才解析，测试也可以直接注入。
"""
from __future__ import annotations

import ast
import operator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

from assistant.settings import workspace_root
from assistant.tools.registry import ToolSpec, tool
from assistant.tools.sandbox import is_binary, resolve_in_root

# ---------- calculator 的安全求值 ----------

MAX_EXPRESSION_CHARS = 200
MAX_POW_EXPONENT = 1000
MAX_LIST_ENTRIES = 200
MAX_READ_BYTES = 100_000
MAX_SCAN_FILES = 500

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> int | float:
    """白名单 AST 遍历。绝不调用 eval()。

    Name / Call / Attribute / Subscript 等节点没有对应分支，会直接落到最后的
    raise，因此 __import__、open、os.system 之类都无法表达。
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        # 显式排除 bool —— True/False 是 int 的子类，会混进算术里
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"不支持的常量：{node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        # 白名单挡不住 9**9**9 这类内存耗尽：必须求值前先卡指数。
        # 注意 9**9**9 解析为 9**(9**9)，right 是 BinOp，此处求出的 9**9 代价很小，
        # 检查发生在大幂结算之前。
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POW_EXPONENT:
            raise ValueError(f"幂指数过大：{right}，上限 {MAX_POW_EXPONENT}")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARY.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符：{type(node.op).__name__}")
        return op(_eval_node(node.operand))

    raise ValueError(f"不支持的表达式节点：{type(node).__name__}")


def safe_eval(expression: str) -> int | float:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("表达式为空")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"表达式过长（{len(expression)} 字符），上限 {MAX_EXPRESSION_CHARS}")
    try:
        # mode="eval" 天然拒绝语句（赋值 / import / lambda 体），但会以
        # SyntaxError 抛出。统一转成 ValueError，让回灌给模型的错误消息一致且可读。
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误：{exc.msg}") from None
    return _eval_node(tree)


# ---------- 工厂 ----------

def build_builtin_tools(root: Optional[Path] = None) -> list[ToolSpec]:
    """构造内置工具集，所有文件类工具被限制在 root 之内。"""
    workspace = (root or workspace_root()).resolve()

    @tool(description="计算纯算术表达式，支持 + - * / // % ** 与括号。")
    def calculator(expression: Annotated[str, "算术表达式，如 (12+8)*3"]) -> str:
        value = safe_eval(expression)
        # 浮点结果去掉无意义的尾零
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    @tool(description="获取当前日期时间。")
    def get_current_time(
        timezone: Annotated[str, "IANA 时区名，如 Asia/Shanghai"] = "Asia/Shanghai",
    ) -> str:
        try:
            from zoneinfo import ZoneInfo
        except ImportError:  # pragma: no cover —— Python 3.9+ 恒可用
            raise ValueError("当前 Python 缺少 zoneinfo 模块") from None
        try:
            tz = ZoneInfo(timezone)
        except Exception as exc:  # noqa: BLE001 —— 含 ZoneInfoNotFoundError
            # Windows 没有系统时区库，需 pip install tzdata
            raise ValueError(f"无法解析时区 {timezone!r}（Windows 下可能需要 pip install tzdata）：{exc}") from None
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z (%A)")

    @tool(description="列出工作根目录下某个目录的文件与子目录。")
    def list_files(path: Annotated[str, "相对工作根目录的路径"] = ".") -> list[str]:
        target = resolve_in_root(workspace, path)
        if not target.exists():
            raise ValueError(f"路径不存在：{path}")
        if target.is_file():
            return [target.name]
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        # 不用 emoji 标记目录：Windows 控制台默认 GBK 编码，emoji 会让 print 直接
        # 抛 UnicodeEncodeError（不是显示乱码，是崩掉）。目录名后缀 "/" 已足够表意。
        out = [f"{p.name}/" if p.is_dir() else p.name for p in entries[:MAX_LIST_ENTRIES]]
        if len(entries) > MAX_LIST_ENTRIES:
            out.append(f"...（共 {len(entries)} 项，仅显示前 {MAX_LIST_ENTRIES} 项）")
        return out

    @tool(description="读取工作根目录下的一个文本文件。")
    def read_file(
        path: Annotated[str, "相对工作根目录的文件路径"],
        max_bytes: Annotated[int, "最多读取的字节数"] = 8000,
    ) -> str:
        target = resolve_in_root(workspace, path)
        if not target.exists():
            raise ValueError(f"文件不存在：{path}")
        if not target.is_file():
            raise ValueError(f"不是文件：{path}")
        if is_binary(target):
            raise ValueError(f"拒绝读取二进制文件：{target.name}")
        limit = max(1, min(int(max_bytes), MAX_READ_BYTES))
        raw = target.read_bytes()[:limit]
        text = raw.decode("utf-8", errors="replace")
        if len(raw) == limit and target.stat().st_size > limit:
            text += f"\n...(仅读取前 {limit} 字节，文件共 {target.stat().st_size} 字节)"
        return text

    @tool(description="在工作根目录下按子串搜索文本文件，返回 文件:行号: 内容。")
    def find_text(
        pattern: Annotated[str, "要搜索的子串（非正则）"],
        root_path: Annotated[str, "相对工作根目录的搜索起点"] = ".",
        max_matches: Annotated[int, "最多返回的匹配数"] = 50,
    ) -> list[str]:
        if not pattern:
            raise ValueError("搜索关键词为空")
        start = resolve_in_root(workspace, root_path)
        if not start.exists():
            raise ValueError(f"路径不存在：{root_path}")
        limit = max(1, min(int(max_matches), 500))
        files = [start] if start.is_file() else sorted(p for p in start.rglob("*") if p.is_file())

        matches: list[str] = []
        scanned = 0
        for path in files:
            if scanned >= MAX_SCAN_FILES:
                matches.append(f"...（已扫描 {MAX_SCAN_FILES} 个文件后停止）")
                break
            if is_binary(path) or not path.is_file():
                continue
            scanned += 1
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
                if pattern in line:
                    rel = path.relative_to(workspace)
                    matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(matches) >= limit:
                        return matches
        return matches or [f"未找到匹配 {pattern!r} 的内容"]

    return [calculator, get_current_time, list_files, read_file, find_text]
