"""
里程碑 DAG 过程评分（CP5）。

a_v = a_v_raw * deps_ok * provenance_ok
  - a_v_raw  : 谓词值（gradable 用 jaccard 给部分得分 GPCM；否则 0/1）
  - deps_ok  : 严格依赖门控 = Π_{u∈pre(v)} 1[a_u >= tau_complete]，**ε=0 无地板**
  - provenance_ok : 因果有效性门控（工具效应归属）
      · 有副作用里程碑：milestone.provenance 中每个 state 路径都须由 agent 动作写入
      · 认知型里程碑：epistemic_action 须在 trace 中存在（布尔存在性）
OR 组：a_O = max_{v∈O} a_v（各自带门控）
MS = Σ_{required∪OR} β_v·a_v / Σ_{required∪OR} β_v ；optional 仅作 bonus 不进分母。

防 gaming：provenance_ok 杜绝"环境免费满足/初始态满足/乱序蒙对"；严格 0 地板杜绝
ε 泄漏；OR 组用结构表达合法替代路径，而非用魔数。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
from schema import Task, Trace, MilestoneSpec
from dsl import eval_predicate

TAU_COMPLETE = 0.5  # 依赖判定"已完成"的阈值


def _provenance_ok(ms: MilestoneSpec, trace: Trace) -> bool:
    # 认知型：仅检查必要动作存在性
    if ms.epistemic_action is not None:
        for ev in trace.events:
            if ev.type == "tool_call" and ev.tool == ms.epistemic_action and ev.status == "ok":
                return True
        return False
    # 有副作用型：所有声明的 provenance 路径必须由 agent 动作写入
    if not ms.provenance:
        return True
    for path in ms.provenance:
        src = trace.provenance.get(path, "")
        if not src.startswith("action:"):
            return False
    return True


def score_milestones(task: Task, trace: Trace) -> Tuple[float, Dict[str, float], Dict[str, dict]]:
    """返回 (MS, a_v 映射, 诊断)。"""
    state = trace.final_state
    by_id = {m.id: m for m in task.milestones}
    a: Dict[str, float] = {}
    diag: Dict[str, dict] = {}

    # 拓扑近似：按 deps 数量升序多轮收敛（DAG 无环，最多 |V| 轮）
    order = list(by_id.keys())
    for _ in range(len(order)):
        for mid in order:
            ms = by_id[mid]
            raw = eval_predicate(ms.predicate, state, trace)
            if not ms.gradable:
                raw = 1.0 if raw >= 1.0 else 0.0
            deps_ok = 1.0
            for u in ms.deps:
                if a.get(u, 0.0) < TAU_COMPLETE:
                    deps_ok = 0.0
                    break
            prov_ok = 1.0 if _provenance_ok(ms, trace) else 0.0
            a[mid] = raw * deps_ok * prov_ok
            diag[mid] = {"raw": raw, "deps_ok": deps_ok, "prov_ok": prov_ok,
                         "a": a[mid]}

    # OR 组取 max（覆盖组内成员的有效分）
    groups: Dict[str, List[str]] = {}
    for m in task.milestones:
        if m.type == "or_group" and m.or_group:
            groups.setdefault(m.or_group, []).append(m.id)
    group_score: Dict[str, float] = {}
    for g, members in groups.items():
        group_score[g] = max(a.get(mid, 0.0) for mid in members)

    # 聚合：required + 每个 OR 组（按组的代表权重）；optional 作 bonus
    num = 0.0
    den = 0.0
    counted_groups = set()
    for m in task.milestones:
        if m.type == "required":
            num += m.weight * a.get(m.id, 0.0)
            den += m.weight
        elif m.type == "or_group" and m.or_group:
            if m.or_group not in counted_groups:
                counted_groups.add(m.or_group)
                # 组权重取组内最大 weight
                gw = max(x.weight for x in task.milestones
                         if x.or_group == m.or_group)
                num += gw * group_score[m.or_group]
                den += gw
    ms_score = (num / den) if den > 0 else 0.0
    return ms_score, a, diag


def _extra(obj: Any, key: str, default: Any = None) -> Any:
    val = getattr(obj, key, None)
    if val is not None:
        return val
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict) and key in extra:
        return extra[key]
    return default


def _event_action_index(ev: Any) -> Optional[int]:
    idx = getattr(ev, "action_index", None)
    if idx is None:
        extra = getattr(ev, "model_extra", None)
        if isinstance(extra, dict):
            idx = extra.get("action_index")
    try:
        return None if idx is None else int(idx)
    except (TypeError, ValueError):
        return None


def _event_extra(ev: Any, key: str, default: Any = None) -> Any:
    val = getattr(ev, key, None)
    if val is not None:
        return val
    extra = getattr(ev, "model_extra", None)
    if isinstance(extra, dict) and key in extra:
        return extra[key]
    return default


def _fault_start_index(trace: Trace, fid: str, fallback: int) -> int:
    for ev in trace.events:
        if _event_extra(ev, "fault_id") == fid:
            idx = _event_action_index(ev)
            if idx is not None:
                return idx
    return int(fallback or 0)


def _action_index_by_id(trace: Trace) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ev in trace.events:
        if ev.type != "tool_call":
            continue
        aid = _event_extra(ev, "action_id")
        idx = _event_action_index(ev)
        if aid is not None and idx is not None:
            out[str(aid)] = idx
    return out


def _recovery_action_index(task: Task, trace: Trace, milestone_id: str) -> Optional[int]:
    by_id = {m.id: m for m in task.milestones}
    ms = by_id.get(milestone_id)
    if ms is None:
        return None
    if ms.epistemic_action:
        for ev in trace.events:
            if ev.type == "tool_call" and ev.tool == ms.epistemic_action and ev.status == "ok":
                return _event_action_index(ev)
    by_action_id = _action_index_by_id(trace)
    candidates: List[int] = []
    for path in ms.provenance:
        src = trace.provenance.get(path, "")
        if src.startswith("action:"):
            aid = src.split(":", 1)[1]
            if aid in by_action_id:
                candidates.append(by_action_id[aid])
    return max(candidates) if candidates else None


def _tools_after(trace: Trace, start_idx: int, end_idx: Optional[int] = None) -> List[str]:
    tools: List[str] = []
    for ev in trace.events:
        if ev.type != "tool_call" or ev.status != "ok":
            continue
        idx = _event_action_index(ev)
        if idx is None or idx <= start_idx:
            continue
        if end_idx is not None and idx > end_idx:
            continue
        if ev.tool:
            tools.append(ev.tool)
    return tools


def _named_tools(spec: Any, *keys: str) -> List[str]:
    out: List[str] = []
    for key in keys:
        val = _extra(spec, key)
        if val is None:
            continue
        if isinstance(val, str):
            out.append(val)
        else:
            try:
                out.extend(str(x) for x in val)
            except TypeError:
                pass
    return out


def _rollback_ok(fault: Any, trace: Trace, start_idx: int,
                 recovery_idx: Optional[int]) -> bool:
    if not getattr(fault, "rollback_required", False):
        return True
    required = _named_tools(fault, "rollback_tool", "rollback_tools",
                            "compensation_tool", "compensation_tools")
    called = _tools_after(trace, start_idx, recovery_idx)
    if required:
        return any(t in called for t in required)
    return any(t.startswith(("rollback", "compensate", "restore", "revert"))
               for t in called)


def _diagnosis_ok(fault: Any, trace: Trace, start_idx: int,
                  recovery_idx: Optional[int]) -> bool:
    if not getattr(fault, "requires_diagnosis", False):
        return True
    required = _named_tools(fault, "diagnosis_tool", "diagnosis_tools")
    called = _tools_after(trace, start_idx, recovery_idx)
    if required:
        return any(t in called for t in required)
    return any(("diagnos" in t or "check" in t or "inspect" in t or "verify" in t)
               for t in called)


def _stability_ok(trace: Trace, recovery_idx: Optional[int],
                  fault_id: Optional[str]) -> bool:
    if recovery_idx is None:
        return False
    for ev in trace.events:
        idx = _event_action_index(ev)
        if idx is None or idx <= recovery_idx:
            continue
        ev_fault = _event_extra(ev, "fault_id")
        if fault_id is not None and ev_fault not in (None, fault_id):
            continue
        if ev.type == "tool_call" and ev.status == "error":
            return False
        if ev.type == "fault" and (fault_id is None or ev_fault == fault_id):
            return False
    return True


def _regression_window_score(task: Task, trace: Trace,
                             recovery_idx: Optional[int]) -> float:
    if recovery_idx is None:
        return 0.0
    knobs = task.difficulty_knobs if isinstance(task.difficulty_knobs, dict) else {}
    window = int(knobs.get("regression_free_window", 0) or 0)
    if window <= 0:
        return 1.0
    ok_steps = 0
    for ev in trace.events:
        idx = _event_action_index(ev)
        if idx is None or idx <= recovery_idx:
            continue
        if ev.type == "tool_call":
            if ev.status != "ok":
                break
            ok_steps += 1
        elif ev.type in ("fault", "env_event"):
            break
        if ok_steps >= window:
            return 1.0
    return min(1.0, ok_steps / float(window))


def recovery_quality(task: Task, trace: Trace,
                     a: Dict[str, float]) -> Tuple[float, int, Dict[str, Any]]:
    """长程恢复质量：恢复里程碑跨越 + 延迟/回滚/稳定/无回归窗口。"""
    faults = task.fault_injection
    details: Dict[str, Any] = {
        "recovery_latency": float("nan"),
        "rollback_correctness": float("nan"),
        "post_recovery_stability": float("nan"),
        "regression_free_window": float("nan"),
        "diagnosis_satisfied": float("nan"),
        "events": [],
    }
    if not faults:
        return float("nan"), 0, details  # 无注入故障 -> 不适用
    crossed_scores: List[float] = []
    latency_scores: List[float] = []
    rollback_scores: List[float] = []
    stability_scores: List[float] = []
    window_scores: List[float] = []
    diagnosis_scores: List[float] = []
    total = 0
    for f in faults:
        if f.recover_milestone is None:
            continue
        total += 1
        crossed = a.get(f.recover_milestone, 0.0) >= TAU_COMPLETE
        crossed_scores.append(1.0 if crossed else 0.0)
        start_idx = _fault_start_index(trace, f.id, f.at_action_index)
        recovery_idx = _recovery_action_index(task, trace, f.recover_milestone)
        latency = None if recovery_idx is None else max(0, recovery_idx - start_idx)
        window = int(_extra(f, "recovery_latency_window",
                            (task.difficulty_knobs or {}).get("recovery_latency_window", 3)) or 3)
        if crossed and latency is not None:
            latency_score = max(0.0, 1.0 - max(0, latency - 1) / float(max(1, window)))
        else:
            latency_score = 0.0
        rollback = 1.0 if crossed and _rollback_ok(f, trace, start_idx, recovery_idx) else 0.0
        diagnosis = 1.0 if crossed and _diagnosis_ok(f, trace, start_idx, recovery_idx) else 0.0
        stability = 1.0 if crossed and _stability_ok(trace, recovery_idx, f.id) else 0.0
        reg_window = _regression_window_score(task, trace, recovery_idx) if crossed else 0.0
        latency_scores.append(latency_score)
        rollback_scores.append(rollback)
        diagnosis_scores.append(diagnosis)
        stability_scores.append(stability)
        window_scores.append(reg_window)
        details["events"].append({
            "fault_id": f.id,
            "recover_milestone": f.recover_milestone,
            "crossed": bool(crossed),
            "fault_action_index": start_idx,
            "recovery_action_index": recovery_idx,
            "latency_actions": latency,
            "recovery_latency": latency_score,
            "rollback_correctness": rollback,
            "diagnosis_satisfied": diagnosis,
            "post_recovery_stability": stability,
            "regression_free_window": reg_window,
        })
    if total == 0:
        return float("nan"), 0, details
    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")
    details["recovery_latency"] = mean(latency_scores)
    details["rollback_correctness"] = mean(rollback_scores)
    details["post_recovery_stability"] = mean(stability_scores)
    details["regression_free_window"] = mean(window_scores)
    details["diagnosis_satisfied"] = mean(diagnosis_scores)
    base = mean(crossed_scores)
    quality_parts = [
        (base, 0.40),
        (details["recovery_latency"], 0.20),
        (details["rollback_correctness"], 0.15),
        (details["post_recovery_stability"], 0.10),
        (details["regression_free_window"], 0.10),
        (details["diagnosis_satisfied"], 0.05),
    ]
    score = sum(v * w for v, w in quality_parts
                if not (isinstance(v, float) and math.isnan(v)))
    details["crossed_ratio"] = base
    details["score"] = score
    return score, total, details


def recovery_score(task: Task, trace: Trace, a: Dict[str, float]) -> Tuple[float, int]:
    """恢复质量分（兼容旧返回形状）。"""
    score, total, _details = recovery_quality(task, trace, a)
    return score, total


def expected_milestone_completion(task: Task, trace: Trace, a: Dict[str, float]) -> float:
    """E[完成里程碑比例]（CP7 长程连续可靠性量，防 pass^k 塌缩）。"""
    req = [m for m in task.milestones if m.type in ("required", "or_group")]
    if not req:
        return float("nan")
    return sum(a.get(m.id, 0.0) for m in req) / len(req)
