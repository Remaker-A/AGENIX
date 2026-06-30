"""
小型声明式 DSL：state 路径解析 + 谓词求值。

谓词对 (state, trace) 求值。多数返回 {0.0, 1.0}；op="jaccard" 返回 [0,1] 用于 GPCM 部分得分。
所有判定均为确定性、可程序化（verifier-first），不含相似度 embedding / LLM。
"""
from __future__ import annotations

from typing import Any, List, Optional, Dict
from schema import PredicateSpec, Trace


_MISSING = object()


def get_path(state: Dict[str, Any], path: Optional[str]) -> Any:
    if path is None:
        return _MISSING
    # 允许以 "state." 前缀书写
    if path.startswith("state."):
        path = path[len("state."):]
    cur: Any = state
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _as_set(v: Any):
    if v is _MISSING or v is None:
        return set()
    if isinstance(v, (list, tuple, set)):
        return set(v)
    return {v}


def _tool_called(trace: Optional[Trace], name: Any) -> bool:
    if trace is None:
        return False
    for ev in trace.events:
        if ev.type == "tool_call" and ev.tool == name:
            return True
    return False


def eval_predicate(spec: PredicateSpec, state: Dict[str, Any],
                   trace: Optional[Trace] = None) -> float:
    """返回 [0,1]。布尔谓词返回 0.0/1.0；jaccard 返回相似度。"""
    op = spec.op
    if op == "tool_called":
        return 1.0 if _tool_called(trace, spec.value) else 0.0
    if op == "tool_not_called":
        return 0.0 if _tool_called(trace, spec.value) else 1.0

    lhs = get_path(state, spec.path)

    if op == "exists":
        return 1.0 if (lhs is not _MISSING and lhs is not None) else 0.0
    if lhs is _MISSING:
        # 路径不存在：除 exists/neq 外一律判 0
        if op == "neq":
            return 1.0
        return 0.0

    if op == "eq":
        return 1.0 if lhs == spec.value else 0.0
    if op == "neq":
        return 1.0 if lhs != spec.value else 0.0
    if op == "ge":
        return 1.0 if _num(lhs) >= _num(spec.value) else 0.0
    if op == "gt":
        return 1.0 if _num(lhs) > _num(spec.value) else 0.0
    if op == "le":
        return 1.0 if _num(lhs) <= _num(spec.value) else 0.0
    if op == "lt":
        return 1.0 if _num(lhs) < _num(spec.value) else 0.0
    if op == "approx":
        a, b = _num(lhs), _num(spec.value)
        denom = max(abs(b), 1e-9)
        return 1.0 if abs(a - b) / denom <= spec.tol else 0.0
    if op == "set_eq":
        return 1.0 if _as_set(lhs) == _as_set(spec.value) else 0.0
    if op == "jaccard":
        s1, s2 = _as_set(lhs), _as_set(spec.value)
        if not s1 and not s2:
            return 1.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union else 0.0
    if op == "contains":
        try:
            return 1.0 if spec.value in lhs else 0.0
        except TypeError:
            return 0.0
    raise ValueError("unknown predicate op: %s" % op)


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")
