"""配置加载：.env 搜寻 + Settings 获取。

背景：根目录 .env 才是真实配置，backend/.env 不存在。而 app/config.py 的
Settings(env_file=".env") 按 cwd 解析，从 backend/ 运行时找不到任何东西，
会静默退化成「密钥为空」。

因此助手内部任何模块都不得直接调用 get_settings()，一律走
get_assistant_settings()，以保证 .env 先于 lru_cache 生效。
"""
from __future__ import annotations

import os
from pathlib import Path

from app.config import Settings, get_settings

# 助手默认的模型输出预算。文枢默认的 2048 是 reasoning_content + content +
# tool_calls 共享的预算，思考 1500 token 后只剩 500 给工具调用，会出现
# finish_reason=length 或空轮。这里刻意放宽。
DEFAULT_MAX_TOKENS = 4096


def _strip_inline_comment(value: str) -> str:
    """剥掉 ` # 说明` 形式的行内注释。

    根 .env 里有 18 行带行内注释（如 `RERANKER_ENABLED=true  # 是否启用重排序`）。
    不剥离的话，值会带上注释尾巴进 os.environ，而 pydantic 解析 bool/int 时直接
    抛 ValidationError，**整个 Settings 构造失败、程序起不来**。

    scripts/doctor.py 的同名函数不剥离注释，是因为它压根不构造 Settings、只直接
    读 os.environ —— 这里刻意与它分道扬镳。

    只把「前面是空白」的 # 当注释起点，这样值内部的 #（如密码）不会被误切。
    """
    indices = [value.find(m) for m in (" #", "\t#") if m in value]
    return value[: min(indices)].strip() if indices else value.strip()


def load_env() -> None:
    """加载 program/.env 或 backend/.env（不覆盖已存在的环境变量）。

    逻辑源自 scripts/doctor.py:_load_env，但刻意复制而非 import：正式包反向
    依赖诊断脚本会造成层次颠倒，且 pyproject.toml 的 include 一旦不含 scripts*
    就会 import 失败。行内注释的处理与之不同，见 _strip_inline_comment。
    """
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",          # 从 backend/ 运行时读 program/.env
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            key = key.strip()
            raw = raw.strip()
            if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
                value = raw[1:-1]            # 引号包裹：内容原样，其中的 # 不是注释
            else:
                value = _strip_inline_comment(raw)
            os.environ.setdefault(key, value)
        return


def get_assistant_settings() -> Settings:
    """必须先加载 .env，再清缓存取 settings。

    get_settings 带 @lru_cache：一旦被调用过，就把当时的环境固化了。若进程里
    别处已经先调用过（例如测试的 conftest），单纯「记得先调」是不够的 ——
    cache_clear() 才是让这件事可恢复、而非靠调用顺序碰运气的那一步。
    """
    load_env()
    get_settings.cache_clear()
    return get_settings()


def max_output_tokens(settings: Settings) -> int:
    """模型输出预算：ASSISTANT_MAX_TOKENS 优先，否则用助手默认值。

    刻意不复用 settings.deepseek_max_tokens（文枢的 2048 对工具调用太小）。
    """
    raw = os.environ.get("ASSISTANT_MAX_TOKENS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_MAX_TOKENS


def echo_reasoning() -> bool:
    """是否在后续请求中回传 reasoning_content。

    DeepSeek 文档要求：请求带 tools 时必须回传每轮 assistant 消息的
    reasoning_content，否则可能 400。但早期 deepseek-reasoner 的指引相反。
    选错的代价是立即 400 且无 workaround，故留一个环境变量逃生口，
    以便一行翻转而不用改代码。
    """
    raw = os.environ.get("ASSISTANT_ECHO_REASONING", "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def workspace_root() -> Path:
    """文件工具的沙箱根目录。"""
    raw = os.environ.get("ASSISTANT_WORKSPACE", "").strip()
    return Path(raw).expanduser().resolve() if raw else Path.cwd().resolve()
