"""支持 `python -m assistant` 直接进入 CLI。"""
from __future__ import annotations

import asyncio
import sys

from assistant.cli import main

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
