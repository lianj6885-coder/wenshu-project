"""文枢轻量 Agent 助手：通用对话 + 工具调用 + 多轮记忆。

从文枢后端抽出的最小核心，不依赖 Mongo / Redis / Chroma / pi-agent 服务。

用法：
    cd backend
    python -m assistant.cli

注意：本模块刻意不产生任何导入副作用 —— 尤其不得在此调用 load_env()，
否则 import assistant 就会改写进程环境变量。
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
