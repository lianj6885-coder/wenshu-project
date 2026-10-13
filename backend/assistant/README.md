# 文枢轻量 Agent 助手

从文枢后端**抽出**的最小核心，做成一个独立的通用对话助手：**工具调用 + 多轮记忆**。

不依赖 MongoDB / Redis / Chroma / pi-agent 服务，不需要 Docker，不需要 `build_container()`。

## 快速开始

```bash
cd backend
python -m assistant.cli                          # 交互式 REPL
python -m assistant.cli --once "现在几点了"        # 只问一句
python -m assistant.cli --workspace ../docs      # 指定文件工具沙箱根目录
python -m assistant.cli --no-tools               # 禁用工具，退化为纯对话
python -m assistant.cli --max-steps 10           # 放宽工具往返轮数（默认 6）
```

REPL 命令：`/help` `/new` `/history` `/tools` `/quit`。

密钥复用文枢根目录的 `.env`（`DEEPSEEK_API_KEY`），无需额外配置。

## 五个内置工具

| 工具 | 说明 |
|---|---|
| `calculator` | 算术表达式。**白名单 AST 求值，绝不 `eval()`** |
| `get_current_time` | 指定时区的当前时间 |
| `list_files` | 列出目录 |
| `read_file` | 读文本文件 |
| `find_text` | 按子串搜索，返回 `文件:行号: 内容` |

后三个被限制在**工作根目录**（`ASSISTANT_WORKSPACE`，默认 cwd）之内，路径越界与符号链接逃逸都会被拒。

## 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `ASSISTANT_MAX_TOKENS` | `4096` | 模型输出预算。**刻意不复用 `DEEPSEEK_MAX_TOKENS`（2048）** —— 那个值是 `reasoning_content` + `content` + `tool_calls` 共享的，思考 1500 token 后只剩 500 给工具调用 |
| `ASSISTANT_ECHO_REASONING` | `true` | 是否回传 `reasoning_content`。见下方「坑」 |
| `ASSISTANT_WORKSPACE` | cwd | 文件工具的沙箱根目录 |

## 复用了什么

| 来源 | 复用的部分 |
|---|---|
| `app/llm/deepseek.py` | `DeepSeekClient`（被 `DeepSeekToolClient` 继承） |
| `app/llm/client.py` | `_headers()` / `_chat_url()` / `LLMError` / `ChatMessage` / `_extract_json` |
| `app/storage/redis_store.py` | `SessionStore` ABC + `MemorySessionStore` |
| `app/config.py` / `app/utils/logging.py` | `Settings` / `setup_logging` |
| `scripts/doctor.py` | `_load_env()` 的思路（**复制并改进**，见下） |

**`app/` 下没有任何文件被修改。**

## 设计要点

### 工具调用没有走 `LLMClient.complete()`

`complete()` 在 `client.py:100` 把响应塌缩成 `content or ""`，`tool_calls` 会被静默丢弃，且返回类型是 `str`。给它加 `tools=` 参数会变成「模型发了工具调用、调用方收到空字符串」的静默错误，还会把工具链路的 bug 引到生产问答的热路径上。

因此新增 `DeepSeekToolClient(DeepSeekClient)`，用子类纯增量地加 `chat()`，复用父类的 `_headers()` / `_chat_url()` / `self.timeout` / `LLMError`。

### 没有用 `WorkingMemory`

`app/memory/working.py` 有三个硬伤：

1. `:31` 写死 `{"role", "content"}` 两键，存不下 `tool_calls` / `reasoning_content`。
2. `:32` 的 `messages[-max_history:]` 是**位置切片**。工具轮之后列表形如 `[..., user, assistant+tc, tool, tool, assistant, ...]`，切在 `-10` 完全可能落在声明与结果**中间** → 孤儿 tool 消息 → 400。这在用工具的会话里是必然发生。
3. 没有字符预算 —— 单条 reasoning trace 就有 1–2 KB，按条数封顶 10 条远远不够。

`ConversationMemory` 因此重写，但仍复用底层的 `SessionStore` ABC。

### 截断以 `user` 消息为界

`group_turns()` 让每条用户消息开启一个分组，其后的 assistant / tool 全归入该组。两个理由：

- 保证 `tool_calls` 与它的 `tool` 结果永远同进同出；
- 保证截断后历史仍以 `user` 开头。若按「assistant + 其后工具」分组，只保留最后一组时历史会以 `assistant` 开头，`repair()` 会正确地裁掉它 —— 于是**整个会话被清空**。

`repair()` 双向查配对：孤儿 `tool` 消息（有结果没声明）和未被应答的 `tool_calls` 声明（有声明没结果）**都会 400**，两个方向都要修。

### `invoke()` 永不抛异常

未知工具、arguments 是坏 JSON、缺参、工具内部异常 —— 全部变成 `ok=False` 的结果回灌给模型让它自愈。这是「每个 `tool_call` 必有配对」得以成立的前提。

### v1 不做流式

工具循环结构上就消费不了流（要攒齐带 id 的完整 `tool_calls` 才能执行），而且 `client.py:118-121` 已记录：`deepseek-v4-flash` 流式带 `max_tokens` 会陷入无限思考。非流式 + 显式 `max_tokens` 是生产验证过的路径。

## 坑

### `reasoning_content` 的回传规则

DeepSeek 文档要求：**请求带 `tools` 时，必须回传每轮 assistant 消息的 `reasoning_content`**，否则可能 400。但早期 `deepseek-reasoner` 的指引相反。选错的代价是立即 400 且无 workaround，所以做成环境变量逃生口：

```bash
ASSISTANT_ECHO_REASONING=false python -m assistant.cli
```

**实测结论**（2026-09-23，`deepseek-v4-flash`）：模型在「决定调工具」那轮会返回 `reasoning_content`，在「拿到工具结果后作答」那轮**不返回**。两种情况 DeepSeek 都接受（三轮对话均 200）。当前实现按「有则带、无则不带」处理。

### `.env` 的行内注释必须剥离

根目录 `.env` 有 18 行形如 `RERANKER_ENABLED=true  # 是否启用重排序` 的带注释行。

`scripts/doctor.py:_load_env` **不剥离**行内注释，那样写没问题 —— 因为它压根不构造 `Settings`，只直接读 `os.environ`。但助手复用了 pydantic 的 `Settings`，于是那些带注释尾巴的值会让 `bool` / `int` 字段解析失败，**整个 `Settings` 构造抛 ValidationError、程序起不来**。

所以 `assistant/settings.py` 的 `load_env()` 刻意与 doctor 分道扬镳：剥掉「前面是空白」的 ` #`（值内部的 `#` 不误切），引号包裹的值原样保留。

### `get_settings()` 的 `lru_cache`

`get_settings` 带 `@lru_cache`，一旦被调用就把当时的环境固化了。若进程里别处先调用过（例如测试的 `conftest.py`），单纯「记得先调 `load_env`」是不够的。

助手内部任何模块都**不得直接调用 `get_settings()`**，一律走 `get_assistant_settings()`（先 `load_env` → `cache_clear` → 再取）。

### `MemorySessionStore` 的行为

- **`ttl` 被完全忽略**，无过期无清理，会话在进程生命周期内累积。`/clear` 会真的调 `delete_session`。
- `set_session` 走 `json.loads(json.dumps(data, default=str))`：非 JSON 值会被**静默字符串化**。所以 `ToolRegistry` 用 `stringify()` 在进消息列表前就统一转字符串。
- `get_session` 返回的是**活引用**（而 `RedisSessionStore` 返回副本），所以永远显式回写、绝不依赖别名。

## 测试

```bash
# 需要 pytest；用独立环境避免污染其它项目
conda create -n wenshu-assistant python=3.12 -y
conda activate wenshu-assistant
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  pytest pytest-asyncio httpx pydantic pydantic-settings \
  motor pymongo redis numpy scikit-learn jieba fastapi "uvicorn[standard]" \
  prometheus-client python-multipart pypdf python-docx beautifulsoup4

cd backend
python -m pytest tests/test_assistant_*.py -q     # 仅助手：69 通过
python -m pytest -q                               # 全量：128 通过
```

> 依赖之所以要装这么多：`backend/tests/conftest.py:7` 在模块级 `import app.deps`，会把 motor / pymongo / redis 整条链拉起来。助手自身**不依赖**它们。

测试文件：

| 文件 | 覆盖 |
|---|---|
| `tests/test_assistant_tools.py` | Schema 生成、参数兜底、calculator 安全（含 `9**9**9`）、路径沙箱、GBK 控制台编码回归 |
| `tests/test_assistant_memory.py` | 分组、双向配对修复、字符预算触发摘要、截断不拆散工具配对 |
| `tests/test_assistant_agent.py` | 直答、单/并行工具、未知工具自愈、`max_steps` 收尾、只读工具免确认、多轮上下文回放 |

## 不做的事

- **不接 `services/pi-agent`** —— 它有现成的 9 工具循环（`tools.ts`），但工具全是文枢内部接口（`retrieve_documents` / `lookup_calendar` / `save_skill` …）且需 Node 服务常驻，与"独立轻量"相反。
- **不接 RAG / 向量检索 / BM25** —— 本次只需通用对话 + 工具调用。
- **不改 `app/`** —— 唯一可选改动是 `pyproject.toml:33` 的 `include` 加 `"assistant*"`（仅 `pip install -e .` 需要；`python -m` 不需要）。

## 加自己的工具

```python
from typing import Annotated
from assistant.tools import tool, ToolRegistry

@tool(description="查询某人的电话号码。")
def lookup_phone(
    name: Annotated[str, "姓名"],
    dept: Annotated[str, "部门（可选）"] = "",
) -> dict:
    return {"name": name, "phone": "1234"}

registry = ToolRegistry([lookup_phone])
```

Schema 由类型注解与 `Annotated[str, "描述"]` 自动生成。需要写操作就加 `readonly=False`，`AgentLoop` 会在执行前走 `confirmer` 回调。
