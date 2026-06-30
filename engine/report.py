"""
评测结果 → 图文结合 Markdown 报告生成器（引擎可复用，不联网、纯已有结果）。

输入：一个 run_eval 落盘的结果 JSON（`results/eval_*.json`，即 _report_summary 结构）。
输出：`report.md`（摘要表 + 逐维 success 表 + 可靠性 + grounding 双值 + 安全 ASR + coverage
       + 数据驱动的失败归因）；并用 matplotlib 渲染若干图表 PNG 到 `results/figs/`，在 md 里
       以 `![]()` 引用。**matplotlib 缺失时优雅降级为纯表格 md**（不报错）。

用法：
    cd engine
    python report.py results/eval_20260626_142148_real_v6.json
    python report.py results/eval_xxx.json --out results/report_v6.md --no-figs
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

# 可选渲染后端（缺失则降级为纯表格）
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # noqa: BLE001
    _HAS_MPL = False


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _num(x: Any, nd: int = 2, dash: str = "—") -> str:
    if x is None:
        return dash
    if isinstance(x, float) and math.isnan(x):
        return dash
    if isinstance(x, (int, float)):
        return ("%." + str(nd) + "f") % x
    return str(x)


def _pct(x: Any, nd: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return ("%." + str(nd) + "f%%") % (100.0 * float(x))


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 图表（matplotlib；缺失则跳过）
# --------------------------------------------------------------------------- #
def _fig_dim_success(d: Dict[str, Any], path: str) -> bool:
    ds = d.get("dimension_stats") or {}
    if not ds:
        return False
    dims = sorted(ds.keys())
    models = d.get("models") or []
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    n = max(1, len(models))
    width = 0.8 / n
    for i, m in enumerate(models):
        vals = [float(ds[dim]["per_model"].get(m, {}).get("marginal", float("nan"))) * 100
                for dim in dims]
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(dims))]
        ax.bar(xs, vals, width=width, label=m)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims)
    ax.set_ylabel("success (GLMM marginal) %")
    ax.set_xlabel("capability dimension")
    ax.set_ylim(0, 105)
    ax.set_title("Success by dimension (GLMM marginal)")
    ax.legend(fontsize=8, ncol=min(4, n))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _fig_reliability(d: Dict[str, Any], path: str) -> bool:
    prof = (d.get("profiles") or {}).get("R") or {}
    pm = prof.get("per_model") or {}
    models = d.get("models") or []
    if not pm:
        return False
    cats = ["per-run", "pass@k", "pass^k"]
    keys = ["per_run", "pass_at_k", "pass_pow_k"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    n = max(1, len(models))
    width = 0.8 / n
    for i, m in enumerate(models):
        rel = pm.get(m, {}).get("reliability", {})
        vals = [float(rel.get(k, float("nan")) or float("nan")) * 100 for k in keys]
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(cats))]
        ax.bar(xs, vals, width=width, label=m)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_ylabel("reliability %")
    ax.set_xlabel("reliability metric")
    ax.set_ylim(0, 105)
    ax.set_title("Reliability metrics (per-run / pass@k / pass^k)")
    ax.legend(fontsize=8, ncol=min(4, n))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _fig_grounding(d: Dict[str, Any], path: str) -> bool:
    per = ((d.get("grounding") or {}).get("per_model")) or {}
    models = [m for m in (d.get("models") or []) if m in per]
    if not models:
        return False
    syn = [(per[m].get("synthetic") if per[m].get("synthetic") is not None else 0.0) * 100
           for m in models]
    real = [(per[m].get("real") if per[m].get("real") is not None else 0.0) * 100
            for m in models]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    xs = range(len(models))
    ax.bar([x - 0.2 for x in xs], syn, width=0.4, label="synthetic (symbolic GT)")
    ax.bar([x + 0.2 for x in xs], real, width=0.4, label="real (OCR track)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("grounding hit-rate %")
    ax.set_xlabel("model")
    ax.set_ylim(0, 105)
    ax.set_title("Grounding dual-track: synthetic vs real")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# 主报告
# --------------------------------------------------------------------------- #
def generate(result_path: str, out_path: Optional[str] = None,
             figs_dir: Optional[str] = None, make_figs: bool = True) -> Dict[str, Any]:
    d = _load(result_path)
    here = os.path.dirname(os.path.abspath(result_path))
    stem = os.path.splitext(os.path.basename(result_path))[0]
    if out_path is None:
        out_path = os.path.join(here, "report_%s.md" % stem.replace("eval_", ""))
    if figs_dir is None:
        figs_dir = os.path.join(here, "figs")

    meta = d.get("meta") or {}
    models = d.get("models") or []
    adapters = d.get("adapters") or {}
    prof = (d.get("profiles") or {}).get("R") or {}
    pm = prof.get("per_model") or {}
    ds = d.get("dimension_stats") or {}
    grounding = d.get("grounding") or {}

    figs_made: List[str] = []
    fig_refs: Dict[str, str] = {}
    if make_figs and _HAS_MPL:
        os.makedirs(figs_dir, exist_ok=True)
        for name, fn in (("dim_success", _fig_dim_success),
                         ("reliability", _fig_reliability),
                         ("grounding", _fig_grounding)):
            fp = os.path.join(figs_dir, "%s_%s.png" % (stem, name))
            try:
                if fn(d, fp):
                    figs_made.append(fp)
                    fig_refs[name] = os.path.relpath(fp, os.path.dirname(os.path.abspath(out_path)))
            except Exception:  # noqa: BLE001 - 单图失败不影响报告
                pass

    L: List[str] = []
    L.append("# AGENIX 评测报告 — %s" % stem)
    L.append("")
    L.append("> 不联网、纯已有结果离线生成。真实模型为 **seed=doubao-seed-evolving**；"
             "mock 参考为 oracle-fed（被喂 gold），**非公平基线**。")
    L.append("")

    # 概览
    rm = meta.get("run_meta") or {}
    L.append("## 1. 概览")
    L.append("")
    L.append(_md_table(
        ["字段", "值"],
        [["时间戳", meta.get("timestamp", "—")],
         ["n_runs / k", "%s / %s" % (meta.get("n_runs", "—"), d.get("k", "—"))],
         ["任务数", meta.get("n_tasks", "—")],
         ["难度过滤", meta.get("difficulty", "—")],
         ["并发 / 墙钟上限(s)", "%s / %s" % (meta.get("concurrency", "—"), meta.get("wall_clock_s", "—"))],
         ["真实 API 调用", rm.get("total_api_calls", "—")],
         ["实际墙钟(s)", rm.get("wall_clock_s", "—")],
         ["作业 完成/跳过", "%s / %s" % (rm.get("jobs_done", "—"), rm.get("jobs_skipped", "—"))],
         ["grounding ρ → 规则", "%s → %s" % (_num(grounding.get("rho"), 3),
                                            grounding.get("headline_rule", "—"))]]))
    L.append("")

    # 模型概览
    L.append("## 2. 模型概览（per-model 可靠性 / 安全 / 成本 / 解析率）")
    L.append("")
    rows = []
    for m in models:
        agg = pm.get(m, {})
        rel = agg.get("reliability", {})
        ad = adapters.get(m, {})
        kind = ad.get("kind", "mock" if ad.get("is_mock", True) else "real")
        n_calls = ad.get("n_calls") or 0
        pr = ad.get("parse_rate")
        rows.append([
            m, kind,
            _pct(rel.get("per_run")), _pct(rel.get("pass_at_k")), _pct(rel.get("pass_pow_k")),
            _num(agg.get("asr")), _num(agg.get("mean_cost"), 1),
            (_pct(pr) if pr is not None else "—") + (" (%d调用)" % n_calls if n_calls else ""),
        ])
    L.append(_md_table(["模型", "类型", "per-run", "pass@k", "pass^k", "ASR", "cost", "解析率"], rows))
    L.append("")

    # 逐维 success（图 + 表）
    L.append("## 3. 能力画像 — 逐维 success（GLMM marginal + 95% CI）")
    L.append("")
    if "dim_success" in fig_refs:
        L.append("![逐维 success](%s)" % fig_refs["dim_success"].replace(os.sep, "/"))
        L.append("")
    dims = sorted(ds.keys())
    if dims:
        hdr = ["模型"] + dims
        drows = []
        for m in models:
            cells = [m]
            for dim in dims:
                pmd = ds[dim]["per_model"].get(m, {})
                cells.append("%s [%s, %s]" % (_num(pmd.get("marginal")),
                                              _num(pmd.get("lo")), _num(pmd.get("hi"))))
            drows.append(cells)
        L.append(_md_table(hdr, drows))
        # single_cluster 标注
        sc_dims = [dim for dim in dims
                   if any(ds[dim]["per_model"].get(m, {}).get("n_templates", 0) < 2 for m in models)]
        if sc_dims:
            L.append("")
            L.append("> 注：%s 维为单模板（single_cluster），CI 仅来自 runs（低估不确定性）。"
                     % ", ".join(sc_dims))
    L.append("")

    # 可靠性图
    L.append("## 4. 可靠性四指标")
    L.append("")
    if "reliability" in fig_refs:
        L.append("![可靠性](%s)" % fig_refs["reliability"].replace(os.sep, "/"))
        L.append("")
    L.append("per-run=单次成功率；pass@k=k 次至少一次；pass^k=k 次全中（模型化无偏估计）。")
    L.append("")

    # grounding 双值
    L.append("## 5. 多模态 grounding（双轨双值，永不合并）")
    L.append("")
    if "grounding" in fig_refs:
        L.append("![grounding](%s)" % fig_refs["grounding"].replace(os.sep, "/"))
        L.append("")
    gper = grounding.get("per_model") or {}
    grows = []
    for m in models:
        g = gper.get(m, {})
        grows.append([m, _num(g.get("synthetic")), _num(g.get("real")),
                      str(g.get("real_trusted"))])
    L.append(_md_table(["模型", "synthetic (符号 GT)", "real (OCR/真实轨)", "real_trusted"], grows))
    L.append("")
    L.append("> ρ(合成,真实)=%s → **%s**。合成轨缺口主要来自 bbox IoU / 反事实最小对 / TEDS（细粒度 grounding）。"
             % (_num(grounding.get("rho"), 3), grounding.get("headline_rule", "—")))
    L.append("")

    # 安全
    L.append("## 6. 安全（ASR）")
    L.append("")
    srows = [[m, _num(pm.get(m, {}).get("asr"))] for m in models]
    L.append(_md_table(["模型", "ASR（攻击成功率，越低越安全）"], srows))
    L.append("")
    L.append("> U6 安全单列为 ASR；其 success 多为 gold-only，已从能力可靠性剔除。")
    L.append("")

    # coverage
    L.append("## 7. 覆盖与口径")
    L.append("")
    crows = []
    for m in models:
        ad = adapters.get(m, {})
        crows.append([m, ad.get("kind", "mock" if ad.get("is_mock", True) else "real"),
                      ad.get("model") or ad.get("provider") or "—",
                      ad.get("fallback_reason") or "—"])
    L.append(_md_table(["模型", "类型", "model/provider", "fallback 原因"], crows))
    L.append("")
    L.append("- mock 参考为 oracle-fed，仅作对照，非公平基线；唯一真实信号是 real 适配器。")
    si = d.get("statistical_indistinguishability") or {}
    if si.get("weight_flip_pairs"):
        L.append("- 权重翻转不可区分对：%s" % si["weight_flip_pairs"])
    L.append("")

    # 失败归因（数据驱动）
    L.append("## 8. 失败归因（数据驱动：解析失败 + 未达标任务）")
    L.append("")
    fail_rows = _failure_rows(adapters)
    if fail_rows:
        L.append(_md_table(["模型", "现象", "明细"], fail_rows))
    else:
        L.append("无解析失败、无未达标任务（或结果不含 task_log/call_log）。")
    L.append("")
    L.append("> 分类（设计缺陷 / 基础设施 / genuine 能力）需结合任务定义判读，见随附 canvas / 叙事报告。")
    L.append("")

    # judge α 门（残余主观项；默认不进 headline）
    judge = d.get("judge") or {}
    if judge.get("per_model"):
        enters = bool(judge.get("enters_headline"))
        policy = judge.get("policy") or d.get("judge_policy") or "diagnostic"
        suffix = "独立 judge_headline" if enters else "诊断"
        L.append("## 9. LLM-judge α 门（残余主观项 · 测量仪器 · %s）" % suffix)
        L.append("")
        jper = judge.get("per_model") or {}
        jrows = []
        for m in models:
            jv = jper.get(m)
            if not jv:
                continue
            jrows.append([m, _num(jv.get("score")), _num(jv.get("alpha"), 3),
                          _pct(jv.get("flip_rate"), 0), jv.get("reliability_band", "—"),
                          str(jv.get("conditional_headline_eligible", False))])
        L.append(_md_table(["模型", "judge 分", "Krippendorff α", "flip_rate",
                            "可信带", "conditional_headline_eligible"], jrows))
        L.append("")
        hc = judge.get("human_calibration") or {}
        L.append("> policy=%s；评委：%s 家族 ×%s（%s）；α 门=%s；跨家族=%s；人标校准达标=%s。"
                 "默认不污染能力 headline；只有 conditional_headline 且三项门控全过时，残余主观 rubric "
                 "才写入独立 `judge_headline`。judge 绝不评 state/数值/grounding/安全。"
                 % (policy, judge.get("n_judges", "—"),
                    "" if not judge.get("families") else "/".join(judge.get("families")),
                    judge.get("backend", "—"), _num(judge.get("alpha_gate"), 3),
                    judge.get("cross_family_ok"), hc.get("passed")))
        L.append("")

    # 抗污染：isomorph-gap + 共同被试等值化（operationalized 演示）
    contam = d.get("contamination") or {}
    if contam.get("per_template"):
        L.append("## 10. 抗污染：isomorph-gap + 共同被试等值化（operationalized）")
        L.append("")
        crows = []
        for r in contam["per_template"]:
            crows.append([r.get("template_id"),
                          _num(r.get("acc_orig_mean")), _num(r.get("acc_bridge_mean")),
                          _num(r.get("gap")),
                          "[%s, %s]" % (_num(r.get("lo")), _num(r.get("hi"))),
                          _num(r.get("p"), 3), str(r.get("flag_retire"))])
        L.append(_md_table(["模板", "Acc(原题)", "Acc(同构桥梁)", "ContamGap",
                            "配对 bootstrap 95%CI", "p", "flag_retire"], crows))
        L.append("")
        eqd = contam.get("equating_demo") or {}
        if eqd:
            L.append("- **共同被试等值化（线性）**：slope=%s, intercept=%s（n_panel=%s）——"
                     "用同一探针面板在 原题 vs 新种子同构桥梁 两版分数拟合 旧↔新 量纲映射（§6.3，无需字面 anchor item）。"
                     % (_num(eqd.get("slope"), 3), _num(eqd.get("intercept"), 3), eqd.get("n_panel", "—")))
        L.append("- 探针=%s（确定性离线 mock，非真实 seed，**0 API 调用**）；任一模板 CI 排除 0 且为正 → flag_retire。"
                 " 本轮 any_flag_retire=**%s**（procedural 同构集预期无污染信号）。"
                 % (contam.get("probe_profile", "—"), contam.get("any_flag_retire")))
        L.append("")

    md = "\n".join(L)
    if not (make_figs and _HAS_MPL):
        md = md.replace("\n## ", "\n## ")  # no-op；保留结构
    if not _HAS_MPL and make_figs:
        md = "> ⚠ matplotlib 不可用，已降级为纯表格（无图表 PNG）。\n\n" + md

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    return {"md_path": os.path.abspath(out_path), "figs": figs_made,
            "figs_made": bool(figs_made), "has_matplotlib": _HAS_MPL}


def _failure_rows(adapters: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    for m, ad in adapters.items():
        for r in (ad.get("call_log") or []):
            if r.get("status") in ("empty", "error"):
                rows.append([m, "解析失败(%s)" % r["status"],
                             "%s round%s: %s" % (r.get("task_id", "?"), r.get("round", "?"),
                                                 (r.get("snippet") or "")[:80])])
        seen = {}
        for t in (ad.get("task_log") or []):
            k = t.get("task_id", "?")
            seen.setdefault(k, [0, 0])
            seen[k][0] += 1
            seen[k][1] += int(bool(t.get("success_met")))
        for k, (n, s) in sorted(seen.items()):
            if s < n:
                rows.append([m, "未达标任务", "%s: success %d/%d" % (k, s, n)])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="AGENIX 结果 JSON → 图文 Markdown 报告")
    ap.add_argument("result", help="results/eval_*.json 路径")
    ap.add_argument("--out", default=None, help="输出 md 路径（默认 results/report_<stem>.md）")
    ap.add_argument("--figs-dir", default=None, help="图表 PNG 目录（默认 results/figs/）")
    ap.add_argument("--no-figs", action="store_true", help="不渲染图表（纯表格 md）")
    args = ap.parse_args()
    res = generate(args.result, out_path=args.out, figs_dir=args.figs_dir,
                   make_figs=not args.no_figs)
    print("已生成报告:", res["md_path"])
    print("matplotlib 可用:", res["has_matplotlib"], " 生成图表:", res["figs_made"])
    for f in res["figs"]:
        print("  图:", f)


if __name__ == "__main__":
    main()
