"""
效率 / 成本（CP8）：与能力严格正交、能力轴零自由参数。

- 仅在 Success=1 的 rollout 上计 regret（"更快地失败"不享红利）。
- Regret = max(0, (c_model - c*)/c*)；Eff = 1/(1+Regret)。
- c* = min(oracle, 强基线 P10)（此脚手架用 task.effective_c_star()=oracle 成本）。
- thrash：参数规范化哈希相同且 status=error 的近似重复占比。
- Pareto：能力–成本前沿。
"""
from __future__ import annotations

from typing import Dict, List, Tuple
from schema import Task, Trace


def efficiency_regret(task: Task, trace: Trace, success: bool) -> Dict[str, float]:
    if not success:
        return {"eff": float("nan"), "regret": float("nan"),
                "cost": trace.cost_actions, "thrash": thrash_ratio(trace)}
    c_star = task.effective_c_star()
    c = max(trace.cost_actions, 0.0)
    regret = max(0.0, (c - c_star) / c_star) if c_star > 0 else 0.0
    return {"eff": 1.0 / (1.0 + regret), "regret": regret, "cost": c,
            "thrash": thrash_ratio(trace)}


def thrash_ratio(trace: Trace) -> float:
    calls = [ev for ev in trace.events if ev.type == "tool_call"]
    if not calls:
        return 0.0
    seen = {}
    thrash = 0
    for ev in calls:
        key = (ev.tool, ev.args_norm_hash)
        if ev.status == "error" and key in seen:
            thrash += 1
        seen[key] = seen.get(key, 0) + 1
    return thrash / len(calls)


def pareto_frontier(points: List[Tuple[str, float, float]]) -> List[str]:
    """points = [(model_id, capability(越大越好), cost(越小越好))]。返回前沿模型 id。"""
    frontier = []
    for mid, cap, cost in points:
        dominated = False
        for mid2, cap2, cost2 in points:
            if mid2 == mid:
                continue
            if cap2 >= cap and cost2 <= cost and (cap2 > cap or cost2 < cost):
                dominated = True
                break
        if not dominated:
            frontier.append(mid)
    return frontier
