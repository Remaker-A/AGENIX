"""
可靠性与校准指标（CP7 + 三方共同盲点：校准）。

CP7：以 per-run 成功率 p̂ 为脊柱，pass@k / pass^k 由 p̂ 模型化推导（低方差，无需 n>>k）：
    pass@k = 1 - (1 - p̂)^k        （上限 / 可达性）
    pass^k = p̂^k                  （可靠性 / 一致性）
长程额外报 E[完成里程碑比例]（防 pass^k 在 L5 塌缩）。

校准（替代单任务 hedge 里程碑，杜绝"无脑 hedge 刷分"）：
    Brier = mean((conf - correct)^2)
    ECE   = Σ_bins (n_b/N) |acc_b - conf_b|
    弃答 precision/recall：在"不可答/会错"项上是否正确弃答。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math


def pass_at_k(p_hat: float, k: int) -> float:
    p = min(max(p_hat, 0.0), 1.0)
    return 1.0 - (1.0 - p) ** k


def pass_pow_k(p_hat: float, k: int) -> float:
    p = min(max(p_hat, 0.0), 1.0)
    return p ** k


def aggregate_reliability(per_instance_p: List[float], k: int = 5) -> Dict[str, float]:
    """对一组实例的 p̂ 聚合 per-run / pass@k / pass^k。"""
    if not per_instance_p:
        return {"per_run": float("nan"), "pass_at_k": float("nan"),
                "pass_pow_k": float("nan"), "k": k}
    n = len(per_instance_p)
    per_run = sum(per_instance_p) / n
    pak = sum(pass_at_k(p, k) for p in per_instance_p) / n
    ppk = sum(pass_pow_k(p, k) for p in per_instance_p) / n
    return {"per_run": per_run, "pass_at_k": pak, "pass_pow_k": ppk, "k": k}


def brier_score(conf_correct: List[Tuple[float, int]]) -> float:
    if not conf_correct:
        return float("nan")
    return sum((c - y) ** 2 for c, y in conf_correct) / len(conf_correct)


def ece(conf_correct: List[Tuple[float, int]], n_bins: int = 10) -> float:
    if not conf_correct:
        return float("nan")
    N = len(conf_correct)
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for c, y in conf_correct:
        b = min(n_bins - 1, int(c * n_bins))
        bins[b].append((c, y))
    total = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(y for _, y in b) / len(b)
        conf = sum(c for c, _ in b) / len(b)
        total += (len(b) / N) * abs(acc - conf)
    return total


def abstention_pr(items: List[Tuple[bool, bool]]) -> Dict[str, float]:
    """items = [(should_abstain, did_abstain)]。返回弃答 precision/recall。"""
    tp = sum(1 for s, d in items if s and d)
    fp = sum(1 for s, d in items if (not s) and d)
    fn = sum(1 for s, d in items if s and (not d))
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"abstain_precision": prec, "abstain_recall": rec}


def risk_coverage_auc(conf_correct: List[Tuple[float, int]]) -> float:
    """AURC：按置信度从高到低纳入样本，积分 coverage 下的风险(1-acc)。

    输入沿用 Brier/ECE 的 `(confidence, target)` 结构，其中 target=1 表示该探针
    可答/应高置信，target=0 表示应低置信或弃答。返回值越低越好。
    """
    if not conf_correct:
        return float("nan")
    pairs = sorted(((_clamp01(c), 1 if y else 0) for c, y in conf_correct),
                   key=lambda x: x[0], reverse=True)
    correct = 0.0
    area = 0.0
    n = len(pairs)
    for i, (_c, y) in enumerate(pairs, start=1):
        correct += y
        risk = 1.0 - (correct / i)
        area += risk / n
    return area


def score_calibration(task: Any, trace: Any, n_bins: int = 10) -> Dict[str, Any]:
    """任务级 U5 校准诊断。

    优先消费 `task.metadata.calibration_probes`，兼容历史任务把 probes 直接挂在
    task extra 字段，或只提供 `gold_answered/gold_abstained` 的旧 metadata。没有
    probes 时返回 NaN 诊断，调用方可保持旧 headline 口径。
    """
    probes = _calibration_probes(task)
    n_probes = len(probes)
    coverage_gate = _metadata_float(task, "calibration_coverage_gate",
                                    _metadata_float(task, "coverage_gate", 0.8))
    base: Dict[str, Any] = {
        "has_probes": bool(probes),
        "n_probes": n_probes,
        "n_confidences": 0,
        "coverage": float("nan") if not probes else 0.0,
        "coverage_gate": coverage_gate,
        "coverage_ok": False,
        "brier": float("nan"),
        "ece": float("nan"),
        "aurc": float("nan"),
        "abstain_precision": float("nan"),
        "abstain_recall": float("nan"),
        "abstain_f1": float("nan"),
        "abstain_coverage": float("nan") if not probes else 0.0,
        "score": float("nan"),
    }
    if not probes:
        return base

    sub = getattr(trace, "submission", None)
    confidences = getattr(sub, "confidences", {}) if sub is not None else {}
    abstain = getattr(sub, "abstain", {}) if sub is not None else {}
    confidences = confidences if isinstance(confidences, dict) else {}
    abstain = abstain if isinstance(abstain, dict) else {}

    conf_correct: List[Tuple[float, int]] = []
    abst_items: List[Tuple[bool, bool]] = []
    final_state = getattr(trace, "final_state", {}) or {}

    for p in probes:
        pid = p["id"]
        answerable = bool(p["answerable"])
        if pid in confidences:
            c = _as_float(confidences.get(pid))
            if c is not None:
                conf_correct.append((_clamp01(c), 1 if answerable else 0))
        did_abstain = _did_abstain(pid, abstain, final_state)
        if did_abstain is not None:
            abst_items.append((not answerable, did_abstain))

    if conf_correct:
        base["n_confidences"] = len(conf_correct)
        base["coverage"] = len(conf_correct) / float(n_probes)
        base["coverage_ok"] = base["coverage"] >= coverage_gate
        base["brier"] = brier_score(conf_correct)
        base["ece"] = ece(conf_correct, n_bins=n_bins)
        base["aurc"] = risk_coverage_auc(conf_correct)

    if abst_items:
        pr = abstention_pr(abst_items)
        base.update(pr)
        base["abstain_coverage"] = len(abst_items) / float(n_probes)
        p = pr["abstain_precision"]
        r = pr["abstain_recall"]
        if not (_isnan(p) or _isnan(r)) and (p + r) > 0:
            base["abstain_f1"] = 2.0 * p * r / (p + r)

    base["score"] = _calibration_score(base)
    return base


def _calibration_score(rep: Dict[str, Any]) -> float:
    """把低优指标转为 [0,1] 高优诊断分；仅 coverage_ok 时才由聚合器进 headline。"""
    vals: List[float] = []
    for key in ("brier", "ece", "aurc"):
        v = rep.get(key)
        if not _isnan(v):
            vals.append(1.0 - _clamp01(float(v)))
    f1 = rep.get("abstain_f1")
    if not _isnan(f1):
        vals.append(_clamp01(float(f1)))
    return sum(vals) / len(vals) if vals else float("nan")


def _calibration_probes(task: Any) -> List[Dict[str, Any]]:
    raw = _metadata_value(task, "calibration_probes")
    probes: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            p = _normalize_probe(item)
            if p is not None:
                probes.append(p)
    if probes:
        return probes

    answered = _metadata_value(task, "gold_answered")
    abstained = _metadata_value(task, "gold_abstained")
    out: List[Dict[str, Any]] = []
    if isinstance(answered, list):
        out.extend({"id": str(pid), "answerable": True, "gold": None}
                   for pid in answered)
    if isinstance(abstained, list):
        out.extend({"id": str(pid), "answerable": False, "gold": None}
                   for pid in abstained)
    return out


def _normalize_probe(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    pid = item.get("id", item.get("item_id", item.get("probe_id")))
    if pid is None:
        return None
    if "answerable" in item:
        answerable = bool(item.get("answerable"))
    elif "should_abstain" in item:
        answerable = not bool(item.get("should_abstain"))
    elif "unanswerable" in item:
        answerable = not bool(item.get("unanswerable"))
    else:
        return None
    return {"id": str(pid), "answerable": answerable, "gold": item.get("gold")}


def _metadata_value(task: Any, key: str, default: Any = None) -> Any:
    metadata = getattr(task, "metadata", None)
    if isinstance(metadata, dict) and key in metadata:
        return metadata.get(key)
    if metadata is not None:
        if hasattr(metadata, key):
            return getattr(metadata, key)
        if hasattr(metadata, "model_dump"):
            try:
                dumped = metadata.model_dump()
                if isinstance(dumped, dict) and key in dumped:
                    return dumped.get(key)
            except Exception:  # noqa: BLE001
                pass
    if hasattr(task, key):
        return getattr(task, key)
    extra = getattr(task, "__pydantic_extra__", None) or {}
    if isinstance(extra, dict) and key in extra:
        return extra.get(key)
    return default


def _metadata_float(task: Any, key: str, default: float) -> float:
    v = _metadata_value(task, key, default)
    f = _as_float(v)
    return default if f is None else _clamp01(f)


def _did_abstain(pid: str, abstain: Dict[str, Any],
                 final_state: Dict[str, Any]) -> Optional[bool]:
    if pid in abstain:
        return bool(abstain.get(pid))
    return _infer_abstain_from_state(pid, final_state)


def _infer_abstain_from_state(pid: str, obj: Any) -> Optional[bool]:
    abstain_keys = {"abstained", "deferred", "escalated", "abstain"}
    answer_keys = {"answered", "verified", "confirmed", "answers"}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in abstain_keys and isinstance(v, list) and pid in v:
                return True
            if k in answer_keys and isinstance(v, list) and pid in v:
                return False
        for v in obj.values():
            got = _infer_abstain_from_state(pid, v)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _infer_abstain_from_state(pid, v)
            if got is not None:
                return got
    return None


def _as_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _clamp01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def _isnan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)
