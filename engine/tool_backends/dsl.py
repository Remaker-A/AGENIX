"""
engine/tool_backends/dsl.py — 默认声明式后端（DSL）。

它是 sandbox 历史语义的唯一权威实现：``execute()`` 直接委托 sandbox 的 ``_apply_effect``
（经 ``ctx.apply_dsl_effect`` 回调），从而保证 ``backend="dsl"``（默认）路径**字节级零变化**。
真实后端（filesystem/sqlite/browser/http_mock，Phase 1）平行实现同一 ``ToolBackend`` 接口。
"""
from __future__ import annotations

from typing import List

from schema import StateDiff
from tool_backends.base import (BackendContext, BackendResult, ToolBackend,
                                register_backend)


class DslBackend(ToolBackend):
    """声明式效应后端：set/append/inc/merge 由 sandbox._apply_effect 解释执行。"""

    name = "dsl"
    deterministic = True
    requires_workdir = False

    def execute(self, ctx: BackendContext) -> BackendResult:
        tool = ctx.tool
        diffs: List[StateDiff] = []
        # 与历史 sandbox.run 完全一致：仅当声明了 effect 且非蜜罐时应用效应；
        # honeypot 工具不产生效应，但仍返回 status="ok"（命中由 critical_violations 单独判定）。
        if tool.effect is not None and not tool.is_honeypot:
            if ctx.apply_dsl_effect is None:  # 防御：dsl 必须由 sandbox 提供委托回调
                raise RuntimeError("dsl backend requires ctx.apply_dsl_effect")
            diffs = ctx.apply_dsl_effect(tool.effect, ctx.args, ctx.source)
        return BackendResult(diffs=diffs, observation=None, status="ok")


# 导入即自注册为默认后端
register_backend("dsl", DslBackend, override=True)
