"""
AGENIX 真实模型横评 runner（B. pilot）。

流程：从配置载入模型集（OpenAI 兼容适配器，默认仍可无 key/无外网回退 mock dry-run）→ 在扩充任务银行上跑
→ 产出 Profile-R / Profile-D 报告 → 保存结果到 engine/results/（JSON 摘要 + 逐维 CSV）。

用法：
    cd engine
    # 1) dry-run（无需任何密钥）：在内置 mock 模型上端到端跑通，证明横评流水线可用
    python run_eval.py --mock
    # 2) 真实横评：复制 configs/models.example.json -> models.json，填 base_url+key 或设环境变量
    python run_eval.py --config configs/models.json --difficulty medium --n-runs 5

无密钥/无外网时：默认每个真实 provider 会按其 mock_profile 回退到内置 stub（report 中标注 fallback）。
正式横评请使用 --require-real 或 --no-mock-fallback，避免 mock 混入 headline。
mock 答不了 table_teds / ocr_bbox 属正常（pilot 已知限制）。
"""
from __future__ import annotations

import argparse
import concurrent.futures as _cf
import csv
import json
import math
import os
import sys
import threading
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass

from orchestrator import load_task_bank, evaluate
from adapters import build_adapter
from sandbox import Sandbox
from scoring.score import score_task
from scoring.aggregate import build_report
from run_demo import print_profile, print_dimension_stats


def _sanitize(obj: Any) -> Any:
    """递归转为 JSON 可序列化：numpy 标量→python、NaN→None、tuple→list、丢弃 callable/ndarray。"""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if callable(obj):
        return None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def _jsonify(obj: Any) -> Any:
    """与 `_sanitize` 类似的递归 JSON 归一化，但**保留 NaN 为 float**（而非转 None）。

    用于增量落盘 partial.jsonl：Python `json.dump`(allow_nan=True，默认) 把 NaN 写成 `NaN`
    字面量，`json.load` 又能原样读回 float('nan')。这保证断点续跑时 records 与首跑**逐字段等价**
    （build_report / aggregate 对缺失项依赖 float('nan') 语义，不能退化成 None）。
    """
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if callable(obj):
        return None
    if isinstance(obj, (float, str, int, bool)) or obj is None:
        return obj
    return str(obj)


def _load_config(config_path: str) -> List[Dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    models = cfg.get("models") if isinstance(cfg, dict) else cfg
    return [m for m in (models or []) if isinstance(m, dict)]


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else None


def _adapter_summary(adapters: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for mid, ad in adapters.items():
        n_calls = getattr(ad, "n_calls", 0)
        n_ok = getattr(ad, "n_parsed_ok", None)
        out[mid] = {
            "is_mock": getattr(ad, "is_mock", True),
            "kind": "real" if not getattr(ad, "is_mock", True) else "mock",
            "provider": getattr(ad, "provider", getattr(ad, "profile_name", "mock")),
            "model": getattr(ad, "model", None),
            "endpoint_type": getattr(ad, "endpoint_type", None),
            "mock_profile": getattr(ad, "profile_name", None),
            "fallback_reason": getattr(ad, "fallback_reason", ""),
            "n_calls": n_calls,
            "n_parsed_ok": n_ok,
            "n_empty": getattr(ad, "n_empty", None),
            "n_errors": getattr(ad, "n_errors", 0),
            "parse_rate": (n_ok / n_calls) if (n_ok is not None and n_calls) else None,
            "max_rounds": getattr(ad, "max_rounds", None),
            "call_log": getattr(ad, "call_log", None),
            "task_log": getattr(ad, "task_log", None),
        }
    return out


def _report_summary(report: Dict[str, Any], adapters: Dict[str, Any]) -> Dict[str, Any]:
    """抽取可序列化的横评摘要（不含庞大的 raw_records）。"""
    out: Dict[str, Any] = {
        "models": report["models"],
        "k": report["k"],
        "u5_headline_version": report.get("u5_headline_version"),
        "u5_legacy_headline_version": report.get("u5_legacy_headline_version"),
        "grounding": report["grounding"],
        "grounding_rho": report["grounding_rho"],
        "grounding_headline_rule": report["grounding_headline_rule"],
        "statistical_indistinguishability": report["statistical_indistinguishability"],
        "irt_item_calibration": report.get("irt_item_calibration"),
        "judge": report.get("judge"),
        "judge_policy": report.get("judge_policy"),
        "judge_headline": report.get("judge_headline"),
        "pareto_frontier": report["pareto_frontier"],
        "pareto_points": [[m, c, cost] for (m, c, cost) in report["pareto_points"]],
        "dimension_stats": report["dimension_stats"],
        "longitudinal_equating": report.get("longitudinal_equating"),
        "diagnostic_baselines": report.get("diagnostic_baselines"),
        "headline_model_policy": report.get("headline_model_policy"),
        "profiles": {},
        "adapters": _adapter_summary(adapters),
    }
    for prof_key, prof in report["profiles"].items():
        out["profiles"][prof_key] = {
            "dims_present": prof["dims_present"],
            "per_model": prof["per_model"],
            "weight_sensitivity": prof["weight_sensitivity"],
        }
    return _sanitize(out)


def _write_csv(path: str, report: Dict[str, Any]) -> None:
    cols = ["profile", "model", "dimension", "point", "lo", "hi", "n_obs", "k",
            "equated_point", "equated_lo", "equated_hi",
            "per_run", "pass_at_k", "pass_pow_k", "asr", "mean_cost",
            "g_synthetic", "g_real", "real_trusted",
            "calibration_coverage", "calibration_brier", "calibration_ece",
            "calibration_aurc", "abstain_precision", "abstain_recall",
            "abstain_f1", "u5_headline_version", "selective_partition_success"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for prof_key, prof in report["profiles"].items():
            for m, agg in prof["per_model"].items():
                rel = agg["reliability"]
                g = agg.get("grounding", {})
                u5c = agg.get("u5_calibration", {})
                dims = prof["dims_present"] or ["-"]
                for d in dims:
                    dv = agg["dim_vector"].get(d, {})
                    ev = agg.get("equated_dim_vector", {}).get(d, {})
                    n_obs = (report["dimension_stats"].get(d, {})
                             .get("per_model", {}).get(m, {}).get("n_obs", ""))
                    is_u5 = (d == "U5")
                    w.writerow([
                        prof_key, m, d,
                        _csv_num(dv.get("point")), _csv_num(dv.get("lo")),
                        _csv_num(dv.get("hi")), n_obs, report["k"],
                        _csv_num(ev.get("point")), _csv_num(ev.get("lo")),
                        _csv_num(ev.get("hi")),
                        _csv_num(rel.get("per_run")), _csv_num(rel.get("pass_at_k")),
                        _csv_num(rel.get("pass_pow_k")), _csv_num(agg.get("asr")),
                        _csv_num(agg.get("mean_cost")),
                        _csv_num(g.get("synthetic")), _csv_num(g.get("real")),
                        g.get("real_trusted"),
                        _csv_num(u5c.get("coverage") if is_u5 else None),
                        _csv_num(u5c.get("brier") if is_u5 else None),
                        _csv_num(u5c.get("ece") if is_u5 else None),
                        _csv_num(u5c.get("aurc") if is_u5 else None),
                        _csv_num(u5c.get("abstain_precision") if is_u5 else None),
                        _csv_num(u5c.get("abstain_recall") if is_u5 else None),
                        _csv_num(u5c.get("abstain_f1") if is_u5 else None),
                        agg.get("u5_headline_version") if is_u5 else "",
                        _csv_num(agg.get("selective_partition_success") if is_u5 else None),
                    ])


def _csv_num(x: Any) -> Any:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return x


# --------------------------------------------------------------------------- #
# 并发执行 + 墙钟/调用上限（Part 2）—— 每 job 独立适配器实例，无共享可变态、无需锁
# --------------------------------------------------------------------------- #
def _entry_model_id(e: Dict[str, Any]) -> str:
    return e.get("id") or e.get("model_id") or e.get("provider") or "model"


def _is_mock_adapter(ad: Any) -> bool:
    return bool(getattr(ad, "is_mock", True))


def _mock_model_ids(adapters: Dict[str, Any]) -> List[str]:
    return [mid for mid, ad in adapters.items() if _is_mock_adapter(ad)]


def _run_records_sequential(entries, tasks, n_runs, offline, force_mock,
                            per_call_timeout):
    """顺序跑模型，先收 raw records，再由 caller 决定哪些模型进入 headline。"""
    records: Dict[str, List[Dict[str, Any]]] = {}
    adapters: Dict[str, Any] = {}
    for e in entries:
        ad = build_adapter(e, offline=offline, force_mock=force_mock)
        if per_call_timeout and hasattr(ad, "timeout"):
            ad.timeout = per_call_timeout
        adapters[ad.model_id] = ad
        recs: List[Dict[str, Any]] = []
        for task in tasks:
            for r in range(n_runs):
                seed = 1000 + r
                sub = ad.submit(task, run_index=r, seed=seed)
                trace = Sandbox(task).run(sub, model_id=ad.model_id, run_index=r, seed=seed)
                recs.append(score_task(task, trace))
        records[ad.model_id] = recs
    return records, adapters


def _split_mock_records(per_model_records: Dict[str, List[Dict[str, Any]]],
                        adapters: Dict[str, Any]
                        ) -> Tuple[Dict[str, List[Dict[str, Any]]],
                                   Dict[str, List[Dict[str, Any]]],
                                   Dict[str, Any],
                                   Dict[str, Any]]:
    mock_ids = set(_mock_model_ids(adapters))
    headline_records = {m: r for m, r in per_model_records.items()
                        if m not in mock_ids}
    diagnostic_records = {m: r for m, r in per_model_records.items()
                          if m in mock_ids}
    headline_adapters = {m: a for m, a in adapters.items() if m not in mock_ids}
    diagnostic_adapters = {m: a for m, a in adapters.items() if m in mock_ids}
    return headline_records, diagnostic_records, headline_adapters, diagnostic_adapters


def _build_diagnostic_baselines(records: Dict[str, List[Dict[str, Any]]],
                                adapters: Dict[str, Any], k: int,
                                dim_n_boot: int, glmm_n_boot: int,
                                irt_gate: bool) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    rep = build_report(records, k=k, dim_n_boot=dim_n_boot,
                       glmm_n_boot=glmm_n_boot, irt_gate=irt_gate)
    return _sanitize({
        "models": rep["models"],
        "reason": "is_mock=True; excluded from official headline rankings",
        "adapters": _adapter_summary(adapters),
        "profiles": rep["profiles"],
        "dimension_stats": rep["dimension_stats"],
        "grounding": rep["grounding"],
        "pareto_frontier": rep["pareto_frontier"],
        "pareto_points": [[m, c, cost] for (m, c, cost) in rep["pareto_points"]],
    })


def _build_report_with_policy(per_model_records: Dict[str, List[Dict[str, Any]]],
                              adapters: Dict[str, Any], k: int,
                              dim_n_boot: int, glmm_n_boot: int,
                              irt_gate: bool, require_real: bool,
                              no_mock_fallback: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    mock_ids = _mock_model_ids(adapters)
    if require_real and mock_ids:
        raise RuntimeError("require-real enabled but these configured headline models are mock/fallback: "
                           + ", ".join(sorted(mock_ids)))

    headline_records = per_model_records
    headline_adapters = adapters
    diagnostic_records: Dict[str, List[Dict[str, Any]]] = {}
    diagnostic_adapters: Dict[str, Any] = {}
    if no_mock_fallback:
        (headline_records, diagnostic_records,
         headline_adapters, diagnostic_adapters) = _split_mock_records(per_model_records, adapters)
        if not headline_records:
            raise RuntimeError("no-mock-fallback removed all models from headline; provide real API keys "
                               "or omit --no-mock-fallback for dry-run.")

    report = build_report(headline_records, k=k, dim_n_boot=dim_n_boot,
                          glmm_n_boot=glmm_n_boot, irt_gate=irt_gate)
    report["raw_records"] = headline_records
    report["headline_model_policy"] = {
        "require_real": bool(require_real),
        "no_mock_fallback": bool(no_mock_fallback),
        "ranking_basis": "raw_dim_vector",
        "trend_basis": "equated_dim_vector_when_available",
        "excluded_mock_model_ids": sorted(diagnostic_records),
    }
    diag = _build_diagnostic_baselines(diagnostic_records, diagnostic_adapters, k,
                                       dim_n_boot, glmm_n_boot, irt_gate)
    if diag is not None:
        report["diagnostic_baselines"] = diag
    return report, headline_adapters


def _merge_adapters(model_id: str, ads: List[Any]):
    """把同一模型的多个 per-job 适配器实例合并为一个用于报告的 stats 视图。"""
    ns = types.SimpleNamespace()
    first = ads[0] if ads else types.SimpleNamespace()
    ns.model_id = model_id
    ns.is_mock = getattr(first, "is_mock", True)
    ns.provider = getattr(first, "provider", getattr(first, "profile_name", "mock"))
    ns.model = getattr(first, "model", None)
    ns.endpoint_type = getattr(first, "endpoint_type", None)
    ns.profile_name = getattr(first, "profile_name", None)
    ns.fallback_reason = getattr(first, "fallback_reason", "")
    ns.max_rounds = getattr(first, "max_rounds", None)
    ns.n_calls = sum(getattr(a, "n_calls", 0) for a in ads)
    ns.n_errors = sum(getattr(a, "n_errors", 0) for a in ads)
    if not ns.is_mock:
        ns.n_parsed_ok = sum(getattr(a, "n_parsed_ok", 0) for a in ads)
        ns.n_empty = sum(getattr(a, "n_empty", 0) for a in ads)
    ns.call_log = [r for a in ads for r in (getattr(a, "call_log", None) or [])]
    ns.task_log = [r for a in ads for r in (getattr(a, "task_log", None) or [])]
    return ns


def _adapter_job_stats(ad: Any) -> Dict[str, Any]:
    """抽取单 job 适配器实例的可序列化 stats（落盘 partial.jsonl，供续跑重建报告用）。"""
    return {
        "is_mock": bool(getattr(ad, "is_mock", True)),
        "provider": getattr(ad, "provider", getattr(ad, "profile_name", "mock")),
        "model": getattr(ad, "model", None),
        "endpoint_type": getattr(ad, "endpoint_type", None),
        "profile_name": getattr(ad, "profile_name", None),
        "fallback_reason": getattr(ad, "fallback_reason", ""),
        "max_rounds": getattr(ad, "max_rounds", None),
        "n_calls": int(getattr(ad, "n_calls", 0) or 0),
        "n_errors": int(getattr(ad, "n_errors", 0) or 0),
        "n_parsed_ok": getattr(ad, "n_parsed_ok", None),
        "n_empty": getattr(ad, "n_empty", None),
        "call_log": getattr(ad, "call_log", None),
        "task_log": getattr(ad, "task_log", None),
    }


def _adapter_from_stats(model_id: str, stats: Dict[str, Any]):
    """把落盘的 per-job stats 还原成一个轻量适配器视图（供 `_merge_adapters` 合并）。"""
    ns = types.SimpleNamespace()
    ns.model_id = model_id
    ns.is_mock = bool(stats.get("is_mock", True))
    ns.provider = stats.get("provider", "mock")
    ns.model = stats.get("model")
    ns.endpoint_type = stats.get("endpoint_type")
    ns.profile_name = stats.get("profile_name")
    ns.fallback_reason = stats.get("fallback_reason", "")
    ns.max_rounds = stats.get("max_rounds")
    ns.n_calls = int(stats.get("n_calls", 0) or 0)
    ns.n_errors = int(stats.get("n_errors", 0) or 0)
    ns.n_parsed_ok = int(stats.get("n_parsed_ok", 0) or 0)
    ns.n_empty = int(stats.get("n_empty", 0) or 0)
    ns.call_log = stats.get("call_log") or []
    ns.task_log = stats.get("task_log") or []
    return ns


def _append_partial(path: str, payload: Dict[str, Any]) -> None:
    """把单个 (task,run) 完成结果追加为一行 JSON（allow_nan=True 写 NaN 字面量，可原样读回）。"""
    line = json.dumps(_jsonify(payload), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_partial(path: str):
    """读已落盘的 partial.jsonl → (done_keys, records, used)。

    - done_keys: 已完成 (model_id, task_id, run_index) 集，用于启动时跳过。
    - records: {model_id: [score_rec, ...]}，原样还原（含 NaN 语义）。
    - used: {model_id: [适配器视图, ...]}，供 `_merge_adapters` 还原报告所需 stats。
    同一 (model,task,run) 多行只取首条（防重复追加污染）。
    """
    done_keys = set()
    records: Dict[str, List[Dict[str, Any]]] = {}
    used: Dict[str, List[Any]] = {}
    if not path or not os.path.isfile(path):
        return done_keys, records, used
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001 - 跳过半截/损坏行（崩溃中断可能写到一半）
                continue
            mid = obj.get("mid")
            tid = obj.get("task_id")
            r = obj.get("run_index")
            if mid is None or tid is None or r is None:
                continue
            key = (mid, tid, int(r))
            if key in done_keys:
                continue
            done_keys.add(key)
            rec = obj.get("rec")
            if rec is not None:
                records.setdefault(mid, []).append(rec)
                used.setdefault(mid, []).append(
                    _adapter_from_stats(mid, obj.get("adapter") or {}))
    return done_keys, records, used


def _write_progress(path: str, done: int, total: int, last_task: Any,
                    last_success: Any, elapsed_s: float, in_flight: int,
                    stopped: bool = False, finished: bool = False) -> None:
    """原子刷新实时进度文件（每完成一个 (task,run) 调一次）。"""
    pct = (100.0 * done / total) if total else 0.0
    status = "FINISHED" if finished else ("STOPPED_EARLY" if stopped else "RUNNING")
    lines = [
        "AGENIX eval progress",
        "status: %s" % status,
        "completed: %d/%d (%.1f%%)" % (done, total, pct),
        "last_task: %s (success=%s)" % (last_task, last_success),
        "elapsed_s: %.1f" % elapsed_s,
        "in_flight: %d" % in_flight,
        "updated: %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
    ]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)  # 原子替换：读侧永远看到完整一份


def _run_records_concurrent(entries, tasks, n_runs, concurrency, wall_clock_s,
                            max_calls, offline, force_mock, per_call_timeout,
                            quiet=False, partial_path=None, progress_path=None,
                            resume=True):
    """并发跑 (model × task × run) 作业池；尊重墙钟硬上限与总调用上限，超限优雅停止并返回已完成部分。

    每个 job 用**独立**适配器实例（线程安全：无共享可变态）；结束后按模型合并 stats。

    **覆盖优先（coverage-first）**：作业按 **run-major 轮转**排布——先把每个任务的 run#0 排在最前，
    再 run#1、run#2…。这样当墙钟/调用上限触发优雅停止时，**每个任务至少已被跑到 ≥1 个 run**
    （而非靠前任务跑满 n_runs、靠后任务 0 run），保证"完整集每个任务都被真跑到"。

    **增量落盘 / 断点续 / 实时进度**（不改动作业池循环结构）：
      - 启动时若 `partial_path` 已存在，读回已完成 (model,task,run) 并跳过（断点续）。
      - 每完成一个 job 立即把结果追加进 `partial_path`（任何时刻磁盘上都有已完成部分）。
      - 每完成一个 job 原子刷新 `progress_path`（已完成/总数、最近任务+success、已用时、在飞数）。
    """
    jobs_all = [(_entry_model_id(e), e, task, r)
                for e in entries for r in range(n_runs) for task in tasks]
    # 断点续：载入已落盘部分，跳过其中的 (model,task,run)
    done_keys: set = set()
    records: Dict[str, List[Dict[str, Any]]] = {}
    used: Dict[str, List[Any]] = {}
    if resume and partial_path:
        done_keys, records, used = _load_partial(partial_path)
    n_resumed = len(done_keys)
    jobs = [j for j in jobs_all if (j[0], j[2].task_id, j[3]) not in done_keys]

    lock = threading.Lock()
    stop = threading.Event()
    start = time.time()
    n_done = {"ok": 0, "skipped": 0}
    total_calls = sum(getattr(a, "n_calls", 0) for ads in used.values() for a in ads)

    def work(job):
        mid, e, task, r = job
        if stop.is_set():
            return (mid, task.task_id, r, None, None, "skipped")
        ad = build_adapter(e, offline=offline, force_mock=force_mock)
        if per_call_timeout and hasattr(ad, "timeout"):
            ad.timeout = per_call_timeout
        seed = 1000 + r
        sub = ad.submit(task, run_index=r, seed=seed)
        trace = Sandbox(task).run(sub, model_id=ad.model_id, run_index=r, seed=seed)
        return (mid, task.task_id, r, score_task(task, trace), ad, "ok")

    # 起跑即写一份进度（含续跑已完成数；空作业=全部已续跑完成）
    if progress_path:
        _write_progress(progress_path, n_resumed, len(jobs_all),
                        "(resume)" if n_resumed else "(start)", None, 0.0,
                        max(0, min(concurrency, len(jobs))), finished=(not jobs))

    with _cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for fut in _cf.as_completed(futs):
            try:
                mid, tid, r, rec, ad, status = fut.result()
            except Exception:  # noqa: BLE001 - 单 job 异常不拖垮整体
                n_done["skipped"] += 1
                continue
            if status == "skipped" or rec is None:
                n_done["skipped"] += 1
                continue
            with lock:
                records.setdefault(mid, []).append(rec)
                used.setdefault(mid, []).append(ad)
                n_done["ok"] += 1
                total_calls = sum(getattr(a, "n_calls", 0)
                                  for ads in used.values() for a in ads)
                # 增量落盘：本 (task,run) 一完成立即追加（崩溃也不丢已完成部分）
                if partial_path:
                    _append_partial(partial_path, {
                        "mid": mid, "task_id": tid, "run_index": r,
                        "status": status, "rec": rec,
                        "adapter": _adapter_job_stats(ad)})
                # 实时进度：已完成/总数、最近任务+success、已用时、在飞数
                if progress_path:
                    remaining = len(jobs) - n_done["ok"] - n_done["skipped"]
                    _write_progress(progress_path, n_resumed + n_done["ok"],
                                    len(jobs_all), tid, rec.get("success"),
                                    time.time() - start,
                                    max(0, min(concurrency, remaining)))
            if wall_clock_s and (time.time() - start) > wall_clock_s:
                stop.set()
            if max_calls and total_calls >= max_calls:
                stop.set()

    merged = {mid: _merge_adapters(mid, ads) for mid, ads in used.items()}
    meta = {"jobs_total": len(jobs_all), "jobs_done": n_resumed + n_done["ok"],
            "jobs_new": n_done["ok"], "jobs_resumed": n_resumed,
            "jobs_skipped": n_done["skipped"], "stopped_early": stop.is_set(),
            "wall_clock_s": round(time.time() - start, 1),
            "total_api_calls": sum(getattr(a, "n_calls", 0)
                                   for ads in used.values() for a in ads
                                   if not getattr(a, "is_mock", True))}
    if progress_path:
        _write_progress(progress_path, meta["jobs_done"], len(jobs_all),
                        "(done)", None, meta["wall_clock_s"], 0,
                        stopped=stop.is_set(), finished=True)
    if not quiet and stop.is_set():
        print("⚠ 达到墙钟/调用上限，已优雅停止；完成 %d/%d 作业（落盘部分结果）"
              % (meta["jobs_done"], len(jobs_all)))
    return records, merged, meta


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not math.isnan(float(x))


def _profile_dim_score(src: Dict[str, Any], profile: str, model_id: str, dim: str,
                       prefer_equated: bool = False, field: str = "point") -> Optional[float]:
    try:
        block = src["profiles"][profile]["per_model"][model_id]
    except Exception:  # noqa: BLE001
        return None
    vectors = []
    if prefer_equated:
        vectors.append(block.get("equated_dim_vector", {}))
    vectors.append(block.get("dim_vector", {}))
    for vec in vectors:
        val = (vec.get(dim) or {}).get(field)
        if _is_num(val):
            return float(val)
    return None


def _reference_pairs(current_models: List[str], prev_summary: Dict[str, Any],
                     profile: str, panel_cfg: Optional[Dict[str, Any]]
                     ) -> List[Tuple[str, str]]:
    prev_models = set(((prev_summary.get("profiles") or {}).get(profile) or {})
                      .get("per_model", {}).keys())
    if not panel_cfg:
        return [(m, m) for m in current_models if m in prev_models]

    pairs: List[Tuple[str, str]] = []
    explicit = panel_cfg.get("model_pairs") or panel_cfg.get("pairs")
    if isinstance(explicit, list):
        for item in explicit:
            if isinstance(item, dict):
                cur = item.get("current") or item.get("current_id") or item.get("id")
                prev = item.get("previous") or item.get("prev") or item.get("previous_id") or cur
                if cur and prev:
                    pairs.append((str(cur), str(prev)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]), str(item[1])))

    mapping = panel_cfg.get("model_id_map") or panel_cfg.get("id_map") or {}
    ref_ids = (panel_cfg.get("ref_model_ids") or panel_cfg.get("reference_model_ids")
               or panel_cfg.get("models") or [])
    if isinstance(mapping, dict) and mapping and not ref_ids:
        ref_ids = list(mapping.keys())
    if isinstance(ref_ids, list):
        for mid in ref_ids:
            if isinstance(mid, dict):
                cur = mid.get("current") or mid.get("current_id") or mid.get("id")
                prev = mid.get("previous") or mid.get("prev") or mid.get("previous_id") or cur
            else:
                cur = str(mid)
                prev = mapping.get(cur, cur) if isinstance(mapping, dict) else cur
            if cur and prev:
                pairs.append((str(cur), str(prev)))

    seen = set()
    out: List[Tuple[str, str]] = []
    for cur, prev in pairs:
        key = (cur, prev)
        if key not in seen and cur in current_models and prev in prev_models:
            seen.add(key)
            out.append(key)
    return out


def _equate_values(ref_old: List[float], ref_new: List[float],
                   values: List[float], method: str) -> Dict[str, Any]:
    from generators.contamination import common_person_equate
    return common_person_equate(ref_old, ref_new, values, method=method)


def apply_longitudinal_equating(report: Dict[str, Any], prev_summary_path: Optional[str] = None,
                                reference_panel_config: Optional[str] = None,
                                eval_version: Optional[str] = None,
                                equate_method: str = "linear") -> Dict[str, Any]:
    """把当前 raw dim_vector 映射到上一版量纲；横截面排名仍使用 raw dim_vector。"""
    method = equate_method or "linear"
    if method not in ("linear", "equipercentile"):
        raise ValueError("Unsupported equate method: %s" % method)

    panel_cfg = _load_json(reference_panel_config)
    min_panel = int((panel_cfg or {}).get("min_panel", 3))
    meta: Dict[str, Any] = {
        "enabled": bool(prev_summary_path),
        "eval_version": eval_version,
        "prev_summary": os.path.abspath(prev_summary_path) if prev_summary_path else None,
        "reference_panel_config": os.path.abspath(reference_panel_config) if reference_panel_config else None,
        "method": method,
        "min_panel": min_panel,
        "raw_ranking_basis": "dim_vector",
        "trend_basis": "equated_dim_vector",
        "old_scale_source": "previous equated_dim_vector when present, else previous raw dim_vector",
        "profiles": {},
    }
    report["longitudinal_equating"] = meta
    if not prev_summary_path:
        meta["status"] = "disabled_no_prev_summary"
        return meta

    prev_summary = _load_json(prev_summary_path) or {}
    meta["status"] = "ok"
    total_equated = 0
    for prof_key, prof in report.get("profiles", {}).items():
        prof_meta: Dict[str, Any] = {}
        current_models = list((prof.get("per_model") or {}).keys())
        pairs = _reference_pairs(current_models, prev_summary, prof_key, panel_cfg)
        for dim in prof.get("dims_present") or []:
            ref_old: List[float] = []
            ref_new: List[float] = []
            used_pairs: List[Tuple[str, str]] = []
            for cur_id, prev_id in pairs:
                old = _profile_dim_score(prev_summary, prof_key, prev_id, dim,
                                         prefer_equated=True)
                new = _profile_dim_score(report, prof_key, cur_id, dim,
                                         prefer_equated=False)
                if old is None or new is None:
                    continue
                ref_old.append(old)
                ref_new.append(new)
                used_pairs.append((cur_id, prev_id))
            dim_meta: Dict[str, Any] = {
                "status": "insufficient_panel",
                "ref_model_ids": [p[0] for p in used_pairs],
                "prev_model_ids": [p[1] for p in used_pairs],
                "n_panel": len(used_pairs),
            }
            if len(used_pairs) < min_panel:
                prof_meta[dim] = dim_meta
                continue

            target_models: List[str] = []
            target_scores: List[float] = []
            for mid, agg in (prof.get("per_model") or {}).items():
                raw = (agg.get("dim_vector", {}).get(dim) or {}).get("point")
                if _is_num(raw):
                    target_models.append(mid)
                    target_scores.append(float(raw))
            out = _equate_values(ref_old, ref_new, target_scores, method)
            equated = list(out.get("equated") or [])
            for mid, point in zip(target_models, equated):
                agg = prof["per_model"][mid]
                raw_vec = dict((agg.get("dim_vector", {}).get(dim) or {}))
                eq_vec = dict(raw_vec)
                eq_vec["point"] = point
                bounds: List[float] = []
                bound_names: List[str] = []
                for name in ("lo", "hi"):
                    val = raw_vec.get(name)
                    if _is_num(val):
                        bounds.append(float(val))
                        bound_names.append(name)
                if bounds:
                    b_out = _equate_values(ref_old, ref_new, bounds, method)
                    for name, val in zip(bound_names, b_out.get("equated") or []):
                        eq_vec[name] = val
                    if _is_num(eq_vec.get("lo")) and _is_num(eq_vec.get("hi")) and eq_vec["lo"] > eq_vec["hi"]:
                        eq_vec["lo"], eq_vec["hi"] = eq_vec["hi"], eq_vec["lo"]
                agg.setdefault("equated_dim_vector", {})[dim] = eq_vec
                total_equated += 1

            dim_meta.update({
                "status": "ok",
                "method": out.get("method", method),
                "n_panel": out.get("n_panel", len(used_pairs)),
                "slope": out.get("slope"),
                "intercept": out.get("intercept"),
                "n_equated_models": len(target_models),
            })
            prof_meta[dim] = dim_meta
        meta["profiles"][prof_key] = prof_meta
    meta["n_equated_vectors"] = total_equated
    return meta


def build_contamination_block(engine_root: str, templates: List[str] = None,
                              difficulty: str = "medium", n_pairs: int = 6,
                              n_runs: int = 3, n_boot: int = 600, seed: int = 0,
                              probe_profile: str = "medium",
                              scorer=None) -> Dict[str, Any]:
    """抗污染 operationalize 演示（spec §6.5/§6.3）——**离线、确定性、0 真实 API 调用**。

    对每个模板用 `contamination.build_isomorph_pairs` 造「原题 ↔ 新种子同构桥梁题」配对（即 _bridge
    同构集），用**可注入 scorer**（缺省=确定性 mock stub 探针，非真实 seed）得每实例配对正确率，
    喂 `contamination.isomorph_gap_report` 得 ContamGap + 配对 bootstrap CI/p/flag_retire；并用同一
    探针面板在 原题 vs 桥梁题 两版上的分数做 `linear_equating` 共同被试等值化演示。
    """
    from generators.contamination import (build_isomorph_pairs, isomorph_gap_report,
                                          linear_equating)
    from models import ModelAdapter

    # 抗污染探针针对单一难度做同构配对；"all"/None/未知 难度（横评常传 all）退到 medium。
    if difficulty not in ("easy", "medium", "hard", "expert"):
        difficulty = "medium"
    if templates is None:
        templates = ["u1_reconcile", "u2_supplier_sourcing", "u4_migration", "u5_due_diligence"]
    if scorer is None:
        probe = ModelAdapter("contam-probe", probe_profile)

        def scorer(task):  # 每实例正确率 = mock 探针 n_runs 次的 per-run 成功率（确定性、离线）
            s = 0
            for r in range(n_runs):
                sd = 5000 + r
                sub = probe.submit(task, run_index=r, seed=sd)
                tr = Sandbox(task).run(sub, model_id=probe.model_id, run_index=r, seed=sd)
                s += 1 if score_task(task, tr)["success"] else 0
            return s / float(n_runs)

    per_template: List[Dict[str, Any]] = []
    ref_old: List[float] = []
    ref_new: List[float] = []
    any_flag = False
    for tid in templates:
        try:
            pairs = build_isomorph_pairs(tid, difficulty=difficulty, n=n_pairs,
                                         orig_base_seed=0, bridge_base_seed=100000)
        except Exception:  # noqa: BLE001 - 未知模板跳过
            continue
        acc_orig = [float(scorer(o)) for (o, _b) in pairs]
        acc_bridge = [float(scorer(_b)) for (_o, _b) in pairs]
        rep = isomorph_gap_report(tid, acc_orig, acc_bridge, difficulty=difficulty,
                                  n_boot=n_boot, seed=seed)
        per_template.append(rep)
        ref_old.extend(acc_orig)
        ref_new.extend(acc_bridge)
        any_flag = any_flag or bool(rep.get("flag_retire"))

    equating = None
    if len(ref_old) >= 2:
        eq = linear_equating(ref_old, ref_new)
        equating = {"method": "linear", "slope": eq.slope, "intercept": eq.intercept,
                    "n_panel": len(ref_old),
                    "note": "common-person equating：同一探针面板在 原题 vs 新种子同构桥梁 两版分数拟合 旧↔新 量纲映射（§6.3，无需字面 anchor item）"}
    return {
        "probe_profile": probe_profile, "difficulty": difficulty,
        "n_pairs": n_pairs, "n_runs_probe": n_runs, "n_boot": n_boot,
        "templates": [r.get("template_id") for r in per_template],
        "per_template": per_template,
        "any_flag_retire": bool(any_flag),
        "equating_demo": equating,
        "note": ("isomorph-gap（§6.5）：ContamGap=Acc_orig−Acc_isomorph + 配对 bootstrap 95%CI（复用 "
                 "stats.paired_bootstrap_diff）；CI 排除 0 且为正(lo>0) → 原题显著比同构题更易 → 疑似训练污染 "
                 "→ flag_retire（该模板族打污染折扣并退役）。本块用**确定性离线探针**（mock stub，非真实 "
                 "seed，0 API 调用）在 _bridge 同构集上证明该度量已 operationalized（非占位）。"),
    }


def run(config_path: str = None, n_runs: int = 5, k: int = 5,
        difficulty: str = "medium", limit_per_template: int = None,
        out_dir: str = None, dim_n_boot: int = 400, glmm_n_boot: int = 400,
        force_mock: bool = False, offline: bool = False, irt_gate: bool = True,
        include_top_level: bool = True, include_generated: bool = True,
        quiet: bool = False,
        task_ids: List[str] = None, concurrency: int = 1,
        per_call_timeout: float = None, wall_clock_s: float = None,
        max_calls: int = None, tag_suffix: str = "",
        contamination: bool = True, require_real: bool = False,
        no_mock_fallback: bool = False, eval_version: str = None,
        prev_summary: str = None, reference_panel_config: str = None,
        equate_method: str = "linear", run_id: str = None,
        resume: bool = True) -> Dict[str, Any]:
    here = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(here, "configs", "models.example.json")
    if out_dir is None:
        out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    # 增量/续跑/进度标识：results/<run_id>.partial.jsonl + results/<run_id>.progress.txt。
    # 同名 run_id 再次启动即断点续；缺省按时间戳+后缀生成（每次新建，不会误续历史）。
    if run_id is None:
        run_id = "eval_%s%s" % (time.strftime("%Y%m%d_%H%M%S"), tag_suffix or "")
    partial_path = os.path.join(out_dir, "%s.partial.jsonl" % run_id)
    progress_path = os.path.join(out_dir, "%s.progress.txt" % run_id)

    entries = _load_config(config_path)

    difficulties = [difficulty] if difficulty and difficulty != "all" else None
    tasks = load_task_bank(here, include_top_level=include_top_level,
                           include_generated=include_generated,
                           difficulties=difficulties,
                           limit_per_template=limit_per_template)
    if task_ids:
        want = list(task_ids)
        by_id = {t.task_id: t for t in tasks}
        tasks = [by_id[i] for i in want if i in by_id]  # 精确选取 + 保序（pilot 控成本/覆盖）
        missing = [i for i in want if i not in by_id]
        if missing and not quiet:
            print("警告：以下指定 task_id 不在银行中，已跳过：", missing)

    # 预览各模型 kind（不发起 API；并发路径每 job 会另建独立适配器）
    preview = {}
    for e in entries:
        pad = build_adapter(e, offline=offline, force_mock=force_mock)
        preview[_entry_model_id(e)] = pad
    preview_mock_ids = _mock_model_ids(preview)
    if require_real and preview_mock_ids:
        raise RuntimeError("require-real enabled but these configured headline models are mock/fallback: "
                           + ", ".join(sorted(preview_mock_ids)))
    use_concurrent = (concurrency and concurrency > 1) or wall_clock_s or max_calls

    if not quiet:
        print("配置：", config_path)
        print("模型集：", [(mid, "mock(%s)" % getattr(ad, "fallback_reason", "")
                          if getattr(ad, "is_mock", True) else "real:%s" % getattr(ad, "model", ""))
                         for mid, ad in preview.items()])
        print("任务银行：%d 个任务（difficulty=%s, n_runs=%d, concurrency=%d, wall_cap=%ss, max_calls=%s）"
              % (len(tasks), difficulty, n_runs, concurrency, wall_clock_s, max_calls))

    if not quiet:
        print("增量/续跑：run_id=%s" % run_id)
        print("  partial :", partial_path)
        print("  progress:", progress_path)

    run_meta: Dict[str, Any] = {}
    if use_concurrent:
        per_model_records, adapters_for_summary, run_meta = _run_records_concurrent(
            entries, tasks, n_runs, concurrency or 1, wall_clock_s, max_calls,
            offline, force_mock, per_call_timeout, quiet=quiet,
            partial_path=partial_path, progress_path=progress_path, resume=resume)
        report, adapters_for_summary = _build_report_with_policy(
            per_model_records, adapters_for_summary, k, dim_n_boot, glmm_n_boot,
            irt_gate, require_real, no_mock_fallback)
    else:
        per_model_records, adapters = _run_records_sequential(
            entries, tasks, n_runs, offline, force_mock, per_call_timeout)
        report, adapters_for_summary = _build_report_with_policy(
            per_model_records, adapters, k, dim_n_boot, glmm_n_boot,
            irt_gate, require_real, no_mock_fallback)

    apply_longitudinal_equating(report, prev_summary_path=prev_summary,
                                reference_panel_config=reference_panel_config,
                                eval_version=eval_version,
                                equate_method=equate_method)

    if not quiet:
        print_profile(report, "R", "科研横评（合成 grounding + per-run/pass@k）")
        print_profile(report, "D", "部署就绪（真实 grounding + pass^k）")
        print_dimension_stats(report)
        gb = report["grounding"]
        print("\ngrounding ρ = %s -> %s ; real_trusted 模型: %s"
              % (gb["rho"], gb["headline_rule"], gb["real_trusted_models"] or "无"))
        if run_meta:
            print("并发执行：完成 %d/%d 作业，跳过 %d，墙钟 %ss，真实API调用 %d，提前停止=%s"
                  % (run_meta["jobs_done"], run_meta["jobs_total"], run_meta["jobs_skipped"],
                     run_meta["wall_clock_s"], run_meta["total_api_calls"], run_meta["stopped_early"]))

    # ---- 保存结果 ---- #
    ts = time.strftime("%Y%m%d_%H%M%S")
    is_all_mock = all(getattr(a, "is_mock", True) for a in adapters_for_summary.values())
    tag = ("mock" if (force_mock or is_all_mock) else "real") + (tag_suffix or "")
    json_path = os.path.join(out_dir, "eval_%s_%s.json" % (ts, tag))
    csv_path = os.path.join(out_dir, "eval_%s_%s.csv" % (ts, tag))
    summary = _report_summary(report, adapters_for_summary)
    # 抗污染 operationalize 演示（离线、确定性、0 API 调用）——在报告里至少呈现一处（spec §6.5/§6.3）
    if contamination:
        try:
            summary["contamination"] = _sanitize(
                build_contamination_block(here, difficulty=difficulty))
        except Exception as _e:  # noqa: BLE001 - 演示块失败不影响主报告
            summary["contamination"] = {"error": "%s: %s" % (type(_e).__name__, _e)}
    summary["meta"] = {"timestamp": ts, "config": os.path.abspath(config_path),
                       "run_id": run_id,
                       "partial_path": os.path.abspath(partial_path),
                       "progress_path": os.path.abspath(progress_path),
                       "resume": bool(resume),
                       "n_runs": n_runs, "k": k, "difficulty": difficulty,
                       "n_tasks": len(tasks), "tag": tag, "concurrency": concurrency,
                       "per_call_timeout": per_call_timeout, "wall_clock_s": wall_clock_s,
                       "max_calls": max_calls, "run_meta": run_meta,
                       "require_real": require_real,
                       "no_mock_fallback": no_mock_fallback,
                       "eval_version": eval_version,
                       "prev_summary": os.path.abspath(prev_summary) if prev_summary else None,
                       "reference_panel_config": (os.path.abspath(reference_panel_config)
                                                  if reference_panel_config else None),
                       "equate_method": equate_method}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_csv(csv_path, report)

    if not quiet:
        print("\n结果已保存：")
        print("  JSON:", json_path)
        print("  CSV :", csv_path)
        print("  partial :", partial_path)
        print("  progress:", progress_path)
    return {"report": report, "summary": summary,
            "json_path": json_path, "csv_path": csv_path,
            "partial_path": partial_path, "progress_path": progress_path,
            "run_id": run_id,
            "adapters": adapters_for_summary, "n_tasks": len(tasks), "run_meta": run_meta}


def main() -> None:
    ap = argparse.ArgumentParser(description="AGENIX 真实模型横评 runner")
    ap.add_argument("--config", default=None,
                    help="模型配置 JSON（默认 configs/models.example.json）")
    ap.add_argument("--mock", action="store_true",
                    help="强制全部用内置 stub 模型（dry-run，无需密钥/外网）")
    ap.add_argument("--offline", action="store_true",
                    help="离线：跳过真实 API，按 mock_profile 回退")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--difficulty", default="medium",
                    help="任务难度过滤：easy|medium|hard|expert|all（默认 medium）")
    ap.add_argument("--limit-per-template", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dim-n-boot", type=int, default=400)
    ap.add_argument("--glmm-n-boot", type=int, default=400)
    ap.add_argument("--no-irt", action="store_true", help="关闭 IRT 选题门诊断")
    ap.add_argument("--no-generated", action="store_true",
                    help="排除 tasks/generated/（gold-only stub，真实模型解不了）；只跑顶层 + solvable 公平集")
    ap.add_argument("--task-ids", default=None,
                    help="逗号分隔的精确 task_id 列表（pilot 控成本/覆盖；覆盖 difficulty 选择）")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="并发请求池大小（真实横评默认 3，配退避重试以消除偶发空/错；1=顺序路径）")
    ap.add_argument("--per-call-timeout", type=float, default=None,
                    help="单次 API 调用超时秒（覆盖配置，默认按配置/适配器=150）")
    ap.add_argument("--wall-clock", type=float, default=None,
                    help="总墙钟硬上限秒（超时优雅停止并落盘已完成部分）")
    ap.add_argument("--max-calls", type=int, default=None,
                    help="真实 API 调用总上限（达到即优雅停止）")
    ap.add_argument("--tag-suffix", default="", help="结果文件名后缀（如 _v3，避免覆盖历史）")
    ap.add_argument("--require-real", action="store_true",
                    help="若任一配置模型解析为 mock/fallback，则直接失败（正式 headline 推荐）")
    ap.add_argument("--no-mock-fallback", action="store_true",
                    help="允许跑 mock fallback，但从 headline 过滤到 diagnostic_baselines")
    ap.add_argument("--eval-version", default=None,
                    help="本次评测版本号，写入 longitudinal_equating/meta")
    ap.add_argument("--prev-summary", default=None,
                    help="上一版 eval_*.json summary，用于 common-person 纵向等值化")
    ap.add_argument("--reference-panel-config", default=None,
                    help="参考面板 JSON：ref_model_ids/model_id_map/model_pairs/min_panel")
    ap.add_argument("--equate-method", default="linear",
                    choices=["linear", "equipercentile"],
                    help="common-person 等值化方法（默认 linear）")
    ap.add_argument("--run-id", default=None,
                    help="增量/续跑标识：落盘 results/<run_id>.partial.jsonl + .progress.txt；"
                         "同名再次启动即断点续（缺省按时间戳+tag-suffix 生成）")
    ap.add_argument("--no-resume", action="store_true",
                    help="忽略已存在的 <run_id>.partial.jsonl，从头重跑（默认会断点续）")
    args = ap.parse_args()

    tids = [s.strip() for s in args.task_ids.split(",") if s.strip()] if args.task_ids else None
    run(config_path=args.config, n_runs=args.n_runs, k=args.k,
        difficulty=args.difficulty, limit_per_template=args.limit_per_template,
        out_dir=args.out_dir, dim_n_boot=args.dim_n_boot, glmm_n_boot=args.glmm_n_boot,
        force_mock=args.mock, offline=args.offline, irt_gate=not args.no_irt,
        include_generated=not args.no_generated,
        task_ids=tids, concurrency=args.concurrency, per_call_timeout=args.per_call_timeout,
        wall_clock_s=args.wall_clock, max_calls=args.max_calls, tag_suffix=args.tag_suffix,
        require_real=args.require_real, no_mock_fallback=args.no_mock_fallback,
        eval_version=args.eval_version, prev_summary=args.prev_summary,
        reference_panel_config=args.reference_panel_config,
        equate_method=args.equate_method, run_id=args.run_id,
        resume=not args.no_resume)


if __name__ == "__main__":
    main()
