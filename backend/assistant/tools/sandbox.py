"""路径沙箱：把用户/模型给出的路径限制在工作根目录之内。"""
from __future__ import annotations

from pathlib import Path

# 明显的二进制后缀，读取/搜索时直接拒绝，避免把乱码喂给模型。
BINARY_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico",
    ".pdf", ".zip", ".gz", ".tar", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".pyo",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".woff", ".woff2", ".ttf",
})


def resolve_in_root(root: Path, user_path: str) -> Path:
    """把用户路径解析到 root 之内，越界抛 ValueError。

    先 resolve() 再判断是关键：resolve() 会跟随符号链接，因此指向根目录之外
    的符号链接会被 is_relative_to 拦下。若先判断再 resolve，就会有一个
    符号链接逃逸的窗口。
    """
    root_resolved = root.resolve()
    candidate = (root_resolved / user_path).resolve()
    if candidate != root_resolved and not candidate.is_relative_to(root_resolved):
        raise ValueError(f"路径越界：{user_path} 不在工作根目录 {root_resolved} 之内")
    return candidate


def is_binary(path: Path) -> bool:
    return path.suffix.lower() in BINARY_SUFFIXES
