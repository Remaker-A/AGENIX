"""Generate a side-by-side AGENIX report for the three Ark/Doubao models.

Input is the JSON summary produced by run_eval.py. The report intentionally
keeps model ids separate and never collapses them into a provider alias.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # noqa: BLE001
    _HAS_MPL = False


EXPECTED_MODELS = [
    "doubao_seed_evolving",
    "doubao_seed_2_1_pro",
    "doubao_seed_2_1_turbo",
]

MODEL_LABELS = {
    "doubao_seed_evolving": "evolving",
    "doubao_seed_2_1_pro": "pro",
    "doubao_seed_2_1_turbo": "turbo",
}

DIM_LABELS = {
    "U1": "U1 工具/状态",
    "U2": "U2 规划/觅食",
    "U3": "U3 多模态",
    "U4": "U4 长程/迁移",
    "U5": "U5 校准/选择性",
    "U6": "U6 安全",
}


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(float(x))


def _as_float(x: Any) -> Optional[float]:
    if _is_num(x):
        return float(x)
    return None


def _num(x: Any, nd: int = 2, dash: str = "-") -> str:
    v = _as_float(x)
    if v is None:
        return dash
    return ("%." + str(nd) + "f") % v


def _pct(x: Any, nd: int = 1, dash: str = "-") -> str:
    v = _as_float(x)
    if v is None:
        return dash
    return ("%." + str(nd) + "f%%") % (100.0 * v)


def _escape_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = [
        "| " + " | ".join(_escape_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_escape_cell(c) for c in row) + " |")
    return "\n".join(out)


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _ordered_models(data: Dict[str, Any]) -> List[str]:
    models = list(data.get("models") or [])
    preferred = [m for m in EXPECTED_MODELS if m in models]
    extras = [m for m in models if m not in preferred]
    return preferred + extras


def _profile_model(data: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    return (((data.get("profiles") or {}).get("R") or {})
            .get("per_model") or {}).get(model_id, {})


def _dim_stat(data: Dict[str, Any], model_id: str, dim: str) -> Dict[str, Any]:
    return ((data.get("dimension_stats") or {}).get(dim, {})
            .get("per_model") or {}).get(model_id, {})


def _dim_value(data: Dict[str, Any], model_id: str, dim: str) -> Optional[float]:
    stat = _dim_stat(data, model_id, dim)
    if _as_float(stat.get("marginal")) is not None:
        return _as_float(stat.get("marginal"))
    prof = _profile_model(data, model_id)
    return _as_float(((prof.get("dim_vector") or {}).get(dim) or {}).get("point"))


def _dim_interval(data: Dict[str, Any], model_id: str, dim: str) -> str:
    stat = _dim_stat(data, model_id, dim)
    point = _dim_value(data, model_id, dim)
    lo = _as_float(stat.get("lo"))
    hi = _as_float(stat.get("hi"))
    if point is None:
        return "-"
    if lo is None or hi is None:
        return _pct(point)
    return "%s [%s, %s]" % (_pct(point), _pct(lo), _pct(hi))


def _available_dims(data: Dict[str, Any]) -> List[str]:
    dims = set((data.get("dimension_stats") or {}).keys())
    for model_id in data.get("models") or []:
        prof = _profile_model(data, model_id)
        dims.update((prof.get("dim_vector") or {}).keys())
    ordered = [d for d in ["U1", "U2", "U3", "U4", "U5", "U6"] if d in dims]
    return ordered + sorted(d for d in dims if d not in ordered)


def _overall_u1_u5(data: Dict[str, Any], model_id: str) -> Optional[float]:
    vals = [_dim_value(data, model_id, dim) for dim in ["U1", "U2", "U3", "U4", "U5"]]
    vals = [v for v in vals if v is not None]
    if not vals:
        rel = (_profile_model(data, model_id).get("reliability") or {})
        return _as_float(rel.get("per_run"))
    return sum(vals) / len(vals)


def _grounding(data: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    return (((data.get("grounding") or {}).get("per_model") or {}).get(model_id) or
            (_profile_model(data, model_id).get("grounding") or {}))


def _adapter(data: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    return ((data.get("adapters") or {}).get(model_id) or {})


def _task_logs(data: Dict[str, Any], model_id: str) -> List[Dict[str, Any]]:
    logs = _adapter(data, model_id).get("task_log") or []
    return [x for x in logs if isinstance(x, dict)]


def _is_network_drop(log: Dict[str, Any]) -> bool:
    statuses = [str(s).lower() for s in (log.get("round_status") or [])]
    status_joined = " ".join(statuses)
    return (not bool(log.get("success_met")) and
            int(log.get("n_actions") or 0) == 0 and
            ("error" in status_joined or int(log.get("rounds") or 0) == 0))


def _task_dim(task_id: str) -> Optional[str]:
    tid = str(task_id or "").lower()
    if tid.startswith("ground_"):
        return "U3"
    m = re.search(r"(?:^|_)u([1-6])(?:_|$)", tid)
    if m:
        return "U" + m.group(1)
    m = re.search(r"^u([1-6])_", tid)
    if m:
        return "U" + m.group(1)
    return None


def _rate_from_logs(logs: List[Dict[str, Any]],
                    dim: Optional[str] = None) -> Dict[str, Any]:
    if dim:
        logs = [x for x in logs if _task_dim(str(x.get("task_id"))) == dim]
    total = len(logs)
    if not total:
        return {"raw": None, "adjusted": None, "drops": 0, "total": 0}
    drops = sum(1 for x in logs if _is_network_drop(x))
    ok = sum(1 for x in logs if bool(x.get("success_met")))
    denom = total - drops
    adjusted = (ok / denom) if denom > 0 else None
    return {"raw": ok / total, "adjusted": adjusted, "drops": drops, "total": total}


def _status_label(log: Optional[Dict[str, Any]]) -> str:
    if not log:
        return "-"
    if _is_network_drop(log):
        return "API错误/0 action"
    return "成功" if log.get("success_met") else "未达标"


def _task_difference_rows(data: Dict[str, Any], models: List[str],
                          limit: int = 12) -> List[List[Any]]:
    by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model_id in models:
        for log in _task_logs(data, model_id):
            tid = str(log.get("task_id") or "")
            if tid:
                by_task.setdefault(tid, {})[model_id] = log

    rows: List[Tuple[int, str, List[Any]]] = []
    for tid, per_model in sorted(by_task.items()):
        if len(per_model) < 2:
            continue
        outcomes = []
        score_values = []
        n_drops = 0
        for model_id in models:
            log = per_model.get(model_id)
            outcomes.append(_status_label(log))
            if log:
                score_values.append(1 if log.get("success_met") else 0)
                n_drops += int(_is_network_drop(log))
        if not score_values or max(score_values) == min(score_values):
            continue
        row = [tid, _task_dim(tid) or "-", *outcomes]
        rows.append((n_drops, tid, row))
    rows.sort(key=lambda x: (-x[0], x[1]))
    return [r for _, __, r in rows[:limit]]


def _dim_gap_rows(data: Dict[str, Any], models: List[str]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for dim in _available_dims(data):
        vals = [(m, _dim_value(data, m, dim)) for m in models]
        vals = [(m, v) for m, v in vals if v is not None]
        if len(vals) < 2:
            continue
        best_m, best_v = max(vals, key=lambda x: x[1])
        worst_m, worst_v = min(vals, key=lambda x: x[1])
        rows.append([
            DIM_LABELS.get(dim, dim),
            _model_label(best_m),
            _pct(best_v),
            _model_label(worst_m),
            _pct(worst_v),
            _pct(best_v - worst_v),
        ])
    rows.sort(key=lambda r: float(str(r[-1]).rstrip("%")) if str(r[-1]).endswith("%") else -1,
              reverse=True)
    return rows


def _ranked_dims(data: Dict[str, Any], model_id: str) -> Tuple[List[str], List[str]]:
    vals = []
    for dim in _available_dims(data):
        v = _dim_value(data, model_id, dim)
        if v is not None:
            vals.append((dim, v))
    vals.sort(key=lambda x: x[1], reverse=True)
    best = [DIM_LABELS.get(d, d) + " " + _pct(v) for d, v in vals[:2]]
    worst = [DIM_LABELS.get(d, d) + " " + _pct(v) for d, v in vals[-2:]]
    worst.reverse()
    return best, worst


def _common_dims(data: Dict[str, Any], models: List[str],
                 predicate) -> List[str]:
    out = []
    for dim in _available_dims(data):
        vals = [_dim_value(data, m, dim) for m in models]
        if vals and all(v is not None for v in vals) and predicate(vals):
            out.append(DIM_LABELS.get(dim, dim))
    return out


def _endpoint_name(adapter: Dict[str, Any], model_id: str) -> str:
    endpoint = adapter.get("endpoint_type")
    if endpoint:
        return str(endpoint)
    if model_id == "doubao_seed_2_1_pro":
        return "responses"
    return "chat_completions"


def _fig_dim_success(data: Dict[str, Any], models: List[str], path: str) -> bool:
    dims = _available_dims(data)
    if not dims:
        return False
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    n = max(1, len(models))
    width = 0.78 / n
    for i, model_id in enumerate(models):
        vals = [(_dim_value(data, model_id, dim) or 0.0) * 100.0 for dim in dims]
        xs = [j + (i - (n - 1) / 2.0) * width for j in range(len(dims))]
        ax.bar(xs, vals, width=width, label=_model_label(model_id))
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([DIM_LABELS.get(d, d).split(" ")[0] for d in dims])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Verifier score (%)")
    ax.set_title("AGENIX dimension scores by model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=min(3, n), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _fig_overall(data: Dict[str, Any], models: List[str], path: str) -> bool:
    if not models:
        return False
    raw = [(_overall_u1_u5(data, m) or 0.0) * 100.0 for m in models]
    adjusted = []
    for m in models:
        rate = _rate_from_logs(_task_logs(data, m))
        adjusted.append((rate.get("adjusted") or 0.0) * 100.0)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    xs = list(range(len(models)))
    ax.bar([x - 0.18 for x in xs], raw, width=0.36, label="U1-U5 verifier")
    ax.bar([x + 0.18 for x in xs], adjusted, width=0.36,
           label="task success excluding API error+0 action")
    ax.set_xticks(xs)
    ax.set_xticklabels([_model_label(m) for m in models])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Overall score and infra-adjusted diagnostic")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _make_figures(data: Dict[str, Any], models: List[str], stem: str,
                  out_path: str, figs_dir: str, make_figs: bool) -> Tuple[Dict[str, str], List[str]]:
    refs: Dict[str, str] = {}
    made: List[str] = []
    if not make_figs or not _HAS_MPL:
        return refs, made
    os.makedirs(figs_dir, exist_ok=True)
    for name, fn in [("dim_success", _fig_dim_success), ("overall", _fig_overall)]:
        fp = os.path.join(figs_dir, "%s_%s.png" % (stem, name))
        try:
            if fn(data, models, fp):
                made.append(fp)
                refs[name] = os.path.relpath(fp, os.path.dirname(os.path.abspath(out_path))).replace(os.sep, "/")
        except Exception:  # noqa: BLE001
            continue
    return refs, made


def _join(items: Iterable[str], fallback: str = "-") -> str:
    items = [x for x in items if x]
    return "；".join(items) if items else fallback


def generate(result_path: str, out_path: Optional[str] = None,
             figs_dir: Optional[str] = None, make_figs: bool = True) -> Dict[str, Any]:
    data = _load(result_path)
    models = _ordered_models(data)
    here = os.path.dirname(os.path.abspath(result_path))
    stem = os.path.splitext(os.path.basename(result_path))[0]
    if out_path is None:
        out_path = os.path.join(here, "AGENIX-豆包三模型完整评测报告.md")
    if figs_dir is None:
        figs_dir = os.path.join(here, "figs")

    fig_refs, figs_made = _make_figures(data, models, stem, out_path, figs_dir, make_figs)

    meta = data.get("meta") or {}
    run_meta = meta.get("run_meta") or {}
    n_tasks = int(meta.get("n_tasks") or 0)
    n_runs = int(meta.get("n_runs") or 1)
    expected_jobs = n_tasks * len(models) * n_runs
    jobs_done = run_meta.get("jobs_done")
    jobs_total = run_meta.get("jobs_total")

    warnings = []
    if models != EXPECTED_MODELS:
        warnings.append("结果 JSON 的模型列表不是预期顺序或集合：%s" % ", ".join(models))
    if jobs_done is not None and expected_jobs and int(jobs_done) != expected_jobs:
        warnings.append("jobs_done=%s，与 %s 题 × %s 模型 × %s run = %s 不一致。"
                        % (jobs_done, n_tasks, len(models), n_runs, expected_jobs))
    mock_like = [m for m in models if _adapter(data, m).get("kind") != "real"]
    if mock_like:
        warnings.append("以下模型不是 real adapter：%s" % ", ".join(mock_like))

    L: List[str] = []
    L.append("# AGENIX 豆包三模型完整评测报告")
    L.append("")
    L.append("## Executive Summary")
    L.append("")
    L.append("- 本次横评区分三个真实模型：`doubao-seed-evolving`、`doubao-seed-2-1-pro-260628`、`doubao-seed-2-1-turbo-260628`。")
    L.append("- 本次口径：%s 题 × %s 模型 = %s 个模型-任务作业；结果 JSON 记录 jobs_done=%s / jobs_total=%s。"
             % (n_tasks, len(models), expected_jobs, jobs_done if jobs_done is not None else "-",
                jobs_total if jobs_total is not None else "-"))
    L.append("- 主能力分来自 AGENIX verifier 聚合；“剔除 API error+0 action”只作为接口失败诊断，不替代主分。")
    if warnings:
        L.append("- 需要注意：" + " ".join(warnings))
    L.append("")

    L.append("### 模型清单")
    L.append("")
    L.append(_md_table(["报告短名", "模型 ID", "实际模型名", "端点"], [
        [
            _model_label(m),
            "`%s`" % m,
            "`%s`" % (_adapter(data, m).get("model") or m),
            _endpoint_name(_adapter(data, m), m),
        ]
        for m in models
    ]))
    L.append("")

    headers = ["指标"] + [_model_label(m) for m in models]
    rows: List[List[Any]] = []
    for dim in ["U1", "U2", "U3", "U4", "U5", "U6"]:
        rows.append([DIM_LABELS.get(dim, dim)] + [_dim_interval(data, m, dim) for m in models])
    rows.append(["总体 U1-U5"] + [_pct(_overall_u1_u5(data, m)) for m in models])
    rows.append(["原始 per-run"] + [
        _pct((_profile_model(data, m).get("reliability") or {}).get("per_run")) for m in models
    ])
    rows.append(["剔除 API error+0 action 后成功率"] + [
        "%s (剔除 %s/%s)" % (
            _pct(_rate_from_logs(_task_logs(data, m)).get("adjusted")),
            _rate_from_logs(_task_logs(data, m)).get("drops"),
            _rate_from_logs(_task_logs(data, m)).get("total"),
        )
        for m in models
    ])
    rows.append(["ASR (越低越好)"] + [_num(_profile_model(data, m).get("asr")) for m in models])
    rows.append(["Grounding synthetic"] + [_pct(_grounding(data, m).get("synthetic")) for m in models])
    rows.append(["Grounding real"] + [_pct(_grounding(data, m).get("real")) for m in models])
    rows.append(["API 调用数"] + [_adapter(data, m).get("n_calls", "-") for m in models])
    rows.append(["网络/API 错误数"] + [_adapter(data, m).get("n_errors", "-") for m in models])
    rows.append(["Endpoint"] + [_endpoint_name(_adapter(data, m), m) for m in models])
    L.append(_md_table(headers, rows))
    L.append("")

    if fig_refs:
        L.append("## 图表概览")
        L.append("")
        if "dim_success" in fig_refs:
            L.append("![三模型逐维能力](%s)" % fig_refs["dim_success"])
            L.append("")
        if "overall" in fig_refs:
            L.append("![总体分与接口诊断分](%s)" % fig_refs["overall"])
            L.append("")

    L.append("## 三模型逐维比较")
    L.append("")
    dim_rows = []
    for dim in ["U1", "U2", "U3", "U4", "U5", "U6"]:
        dim_rows.append([DIM_LABELS.get(dim, dim)] + [
            "%s；诊断 %s" % (
                _dim_interval(data, m, dim),
                _pct(_rate_from_logs(_task_logs(data, m), dim=dim).get("adjusted")),
            )
            for m in models
        ])
    L.append(_md_table(headers, dim_rows))
    L.append("")

    L.append("## 优缺点")
    L.append("")
    model_rows = []
    for m in models:
        best, worst = _ranked_dims(data, m)
        rate = _rate_from_logs(_task_logs(data, m))
        adapter = _adapter(data, m)
        strengths = list(best)
        if int(adapter.get("n_errors") or 0) == 0:
            strengths.append("本次运行未记录网络/API 错误")
        weaknesses = list(worst)
        if int(adapter.get("n_errors") or 0) > 0:
            weaknesses.append("有 %s 次网络/API 错误，raw 分可能被基础设施影响" % adapter.get("n_errors"))
        if rate.get("drops"):
            weaknesses.append("有 %s 个 error+0 action 作业" % rate.get("drops"))
        model_rows.append([
            _model_label(m),
            _join(strengths),
            _join(weaknesses),
        ])
    L.append(_md_table(["模型", "优点", "缺点"], model_rows))
    L.append("")

    L.append("## 差别最大的维度和任务")
    L.append("")
    gap_rows = _dim_gap_rows(data, models)
    if gap_rows:
        L.append(_md_table(["维度", "最高模型", "最高分", "最低模型", "最低分", "差距"], gap_rows))
    else:
        L.append("当前摘要中没有足够的逐维分数用于计算差距。")
    L.append("")
    task_rows = _task_difference_rows(data, models)
    if task_rows:
        L.append(_md_table(["任务", "维度"] + [_model_label(m) for m in models], task_rows))
    else:
        L.append("当前 adapter task_log 中没有发现三模型结果不一致的任务，或 task_log 不完整。")
    L.append("")

    L.append("## 异同点")
    L.append("")
    common_strong = _common_dims(data, models, lambda vals: min(vals) >= 0.75)
    common_weak = _common_dims(data, models, lambda vals: max(vals) <= 0.40)
    uncertain = []
    if n_runs == 1:
        uncertain.append("本次是单次运行，稳定性结论仍受单次采样噪声影响")
    if any(int(_adapter(data, m).get("n_errors") or 0) > 0 for m in models):
        uncertain.append("部分模型存在网络/API 错误，需要同时看 raw 分和剔除错误后的诊断分")
    if (data.get("grounding") or {}).get("rho") is not None:
        uncertain.append("grounding 双轨结论受 synthetic-real rho=%s 的一致性约束"
                         % _num((data.get("grounding") or {}).get("rho"), 3))
    L.append("- 共同强项：" + _join(common_strong, "未出现三者都高于 75% 的维度。"))
    L.append("- 共同短板：" + _join(common_weak, "未出现三者都低于 40% 的维度。"))
    L.append("- 共同不确定性：" + _join(uncertain, "未发现额外不确定性。"))
    L.append("")

    L.append("## 证据边界")
    L.append("")
    L.append("- 来自真实 verifier 分数的结论：U1-U6 逐维分、总体 U1-U5、原始 per-run、ASR、grounding synthetic/real。")
    L.append("- 来自接口稳定性诊断的结论：API 调用数、网络/API 错误数、error+0 action 剔除后的成功率。")
    L.append("- 来自单次运行噪声范围的结论：模型优缺点中的细小排序差异，尤其是样本数少或 CI 较宽的维度。")
    L.append("")

    L.append("## 运行元数据")
    L.append("")
    meta_rows = [
        ["结果 JSON", os.path.abspath(result_path)],
        ["run_id", meta.get("run_id", "-")],
        ["difficulty", meta.get("difficulty", "-")],
        ["n_runs", meta.get("n_runs", "-")],
        ["concurrency", meta.get("concurrency", "-")],
        ["per_call_timeout", meta.get("per_call_timeout", "-")],
        ["wall_clock_s", meta.get("wall_clock_s", "-")],
        ["require_real", meta.get("require_real", "-")],
        ["no_mock_fallback", meta.get("no_mock_fallback", "-")],
    ]
    L.append(_md_table(["字段", "值"], meta_rows))
    L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    return {
        "md_path": os.path.abspath(out_path),
        "figs": figs_made,
        "figs_made": bool(figs_made),
        "has_matplotlib": _HAS_MPL,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the three-Doubao AGENIX report")
    parser.add_argument("result_json")
    parser.add_argument("--out", default=None)
    parser.add_argument("--figs-dir", default=None)
    parser.add_argument("--no-figs", action="store_true")
    args = parser.parse_args()
    res = generate(args.result_json, out_path=args.out,
                   figs_dir=args.figs_dir, make_figs=not args.no_figs)
    print("wrote %s" % res["md_path"])
    if res["figs_made"]:
        print("figs: %s" % ", ".join(res["figs"]))
    if res["warnings"]:
        print("warnings: %s" % " ".join(res["warnings"]))


if __name__ == "__main__":
    main()
