"""
engine/tool_backends/ — 工具执行后端包（Phase 0 地基）。

- ``base.py``：``ToolBackend`` 抽象基类 + ``BackendContext``/``BackendResult``
  + record/replay（``RecordReplayStore``）+ diff 哈希（``diff_hash``）+ 注册表。
- ``dsl.py`` ：默认声明式后端（导入即注册为 ``"dsl"``，与历史 sandbox 行为字节级一致）。
- ``filesystem.py`` / ``sqlite.py`` / ``browser.py`` / ``http_mock.py``：
  Phase 1 真实后端，导入即注册。

sandbox 通过 ``create_backend`` / ``get_backend_class`` 按 ``ToolSpec.backend`` 选择后端。
"""
from __future__ import annotations

from tool_backends.base import (
    BackendContext,
    BackendResult,
    InMemoryRecordReplayStore,
    RecordReplayStore,
    ToolBackend,
    available_backends,
    canonical_json,
    create_backend,
    diff_hash,
    get_backend_class,
    is_registered,
    register_backend,
)
# 导入 dsl 子模块即触发 "dsl" 后端自注册（务必保留，否则默认后端不可用）
from tool_backends.dsl import DslBackend
from tool_backends.filesystem import FilesystemBackend
from tool_backends.sqlite import SQLiteBackend
from tool_backends.browser import BrowserBackend
from tool_backends.http_mock import HttpMockBackend

__all__ = [
    "BackendContext",
    "BackendResult",
    "RecordReplayStore",
    "InMemoryRecordReplayStore",
    "ToolBackend",
    "DslBackend",
    "FilesystemBackend",
    "SQLiteBackend",
    "BrowserBackend",
    "HttpMockBackend",
    "register_backend",
    "get_backend_class",
    "is_registered",
    "available_backends",
    "create_backend",
    "diff_hash",
    "canonical_json",
]
