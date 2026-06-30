"""Generate charts for the comprehensive AGENIX seed evaluation report.

All chart data is embedded here (the v1->v6 journey spans multiple runs and is not
contained in any single result JSON; per-dimension/grounding numbers are copied from
eval_20260626_120049_real_v5_full.json and eval_20260626_142148_real_v6.json).

English labels only: the bundled matplotlib font (DejaVu Sans) has no CJK glyphs, so
CJK would render as tofu boxes. The surrounding Markdown prose stays in Chinese.
"""
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(_HERE, "results", "figs")
RESULTS = os.path.join(_HERE, "results")
os.makedirs(FIGS, exist_ok=True)

# -- palette (flat, no gradients) ------------------------------------------------
C_PRIMARY = "#2563eb"
C_GOOD = "#16a34a"
C_WARN = "#d97706"
C_MUTED = "#94a3b8"


def _save(fig, name):
    out = os.path.join(FIGS, name)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("saved", out)


def journey():
    versions = ["v1", "v2", "v3", "v4", "v5", "v6"]
    parse = [37.5, 92.3, 100.0, 100.0, 98.4, 100.0]
    multistep = [0, 0, 100, 100, 97, 100]   # U1/U2/U4 success %
    multimodal = [0, 0, 0, 100, 50, 100]    # U3 success %
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(versions, parse, marker="o", linewidth=2, color=C_PRIMARY, label="Parse rate %")
    ax.plot(versions, multistep, marker="s", linewidth=2, color=C_GOOD, label="Multi-step success % (U1/U2/U4)")
    ax.plot(versions, multimodal, marker="^", linewidth=2, color=C_WARN, label="Multimodal success % (U3)")
    ax.set_xlabel("Fair-harness iteration")
    ax.set_ylabel("Percent (%)")
    ax.set_title("seed (doubao-seed-evolving): fair-harness journey v1 to v6")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, "journey_v1_v6.png")


_DIM_LABEL = {
    "U1": "U1\ntool/state", "U2": "U2\nplan/forage", "U3": "U3\nmultimodal",
    "U4": "U4\nlong-horizon", "U5": "U5\ncalibration",
}

_TASK_DIM_PREFIXES = ("U1", "U2", "U3", "U4", "U5", "U6")


def _task_dim(task_id):
    lower = task_id.lower()
    if lower.startswith("ground_"):
        return "U3"
    for dim in _TASK_DIM_PREFIXES:
        if lower.startswith(dim.lower()) or lower.startswith(f"solv_{dim.lower()}"):
            return dim
    return None


def _is_network_drop(task):
    """Network/API-layer failure: the model produced no action and the run status is error."""
    return (
        not bool(task.get("success_met"))
        and int(task.get("n_actions") or 0) == 0
        and "error" in (task.get("round_status") or [])
    )


def _rate(tasks, exclude_network=False):
    rows = [t for t in tasks if not (exclude_network and _is_network_drop(t))]
    if not rows:
        return 0.0, 0, 0
    succ = sum(1 for t in rows if t.get("success_met"))
    return 100.0 * succ / len(rows), succ, len(rows)


def _seed_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return (((d.get("adapters") or {}).get("seed") or {}).get("task_log")) or [], d


def _by_dim(tasks):
    out = {dim: [] for dim in _TASK_DIM_PREFIXES}
    for task in tasks:
        dim = _task_dim(task.get("task_id", ""))
        if dim in out:
            out[dim].append(task)
    return out


def _is_forage(task):
    return (task.get("task_id") or "").endswith("__forage")


def _starts(task, prefix):
    return (task.get("task_id") or "").startswith(prefix)


def _seed_dim_marginals(path):
    """seed 逐维 GLMM headline：{dim: (marginal, lo, hi)}（取 dimension_stats.per_model.seed）。"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for dim, dv in (d.get("dimension_stats") or {}).items():
        s = ((dv.get("per_model") or {}).get("seed")) or {}
        if s.get("marginal") is not None:
            out[dim] = (s["marginal"], s.get("lo", s["marginal"]), s.get("hi", s["marginal"]))
    return out


def dim_success(seed_dims=None, u5_val=None):
    """v9 完整集：直接读 seed 逐维 GLMM marginal + 95%CI（不再硬编码）。无结果时回退旧画法。"""
    if seed_dims:
        order = [k for k in ("U1", "U2", "U3", "U4", "U5") if k in seed_dims]
        dims = [_DIM_LABEL.get(k, k) for k in order]
        vals = [100.0 * seed_dims[k][0] for k in order]
        err_lo = [max(0.0, 100.0 * (seed_dims[k][0] - seed_dims[k][1])) for k in order]
        err_hi = [max(0.0, 100.0 * (seed_dims[k][2] - seed_dims[k][0])) for k in order]
        fig, ax = plt.subplots(figsize=(8.4, 4.2))
        bars = ax.bar(dims, vals, color=C_GOOD, width=0.6,
                      yerr=[err_lo, err_hi], capsize=4, ecolor=C_MUTED)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, min(v + 3, 104), f"{v:.0f}%",
                    ha="center", fontsize=10)
        ax.set_ylabel("Success % (GLMM marginal)")
        ax.set_ylim(0, 116)
        ax.set_title("seed capability profile (v9 full fair set: GLMM marginal + 95% CI; U6=ASR)")
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, "seed_dim_success.png")
        return
    dims = ["U1\ntool/state", "U2\nplan/forage", "U3\nmultimodal", "U4\nlong-horizon"]
    vals = [100, 90, 100, 100]   # legacy fallback (corrected-picture, no result JSON)
    if u5_val is not None:
        dims.append("U5\ncalibration")
        vals.append(round(u5_val))
        title = "seed capability profile by dimension (U5 now covered; U6=ASR, see safety)"
    else:
        title = "seed capability profile by dimension (U5 uncovered; U6=ASR, see safety)"
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.bar(dims, vals, color=C_GOOD, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%", ha="center", fontsize=10)
    ax.set_ylabel("Success % (GLMM marginal)")
    ax.set_ylim(0, 108)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "seed_dim_success.png")


# --------------------------------------------------------------------------- #
# v7 扩充覆盖：U5 维度 + 难度 breakdown（读 results/eval_*_v7_coverage.json）
# --------------------------------------------------------------------------- #
def _latest_v7():
    cands = sorted(glob.glob(os.path.join(RESULTS, "eval_*_v7_coverage.json")))
    return cands[-1] if cands else None


def _latest_result():
    """优先取最新 v9 完整公平集（全难度梯度 easy→expert + 觅食 + 多模板，单版本即完整），
    回退 v8（觅食/难度扩充集）/ v7 coverage。"""
    for pat in ("eval_*_v9*.json", "eval_*_v8*.json", "eval_*_v7_coverage.json"):
        cands = sorted(glob.glob(os.path.join(RESULTS, pat)))
        if cands:
            return cands[-1]
    return None


def _seed_task_success(path):
    """从 seed task_log 提取 {full_task_id: (success%, n_runs)}（含觅食/各难度全量 id）。"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    tl = (((d.get("adapters") or {}).get("seed") or {}).get("task_log")) or []
    agg = {}
    for t in tl:
        tid = t.get("task_id", "")
        agg.setdefault(tid, [0, 0])
        agg[tid][0] += int(bool(t.get("success_met")))
        agg[tid][1] += 1
    return {tid: (100.0 * s / n if n else 0.0, n) for tid, (s, n) in agg.items()}


def foraging_compare(path):
    """觅食对比：同一模板在 data_in_context=True（源数据在提示）vs False（须调 read_* 觅食）下的
    seed success%。直观回答"数据移出上下文后 success 是否仍高 / 是否真去觅食"。"""
    succ = _seed_task_success(path)
    pairs = []  # (label, in_context%, foraging%)
    for tid, (pct, _n) in sorted(succ.items()):
        if not tid.endswith("__forage"):
            continue
        base = tid[: -len("__forage")]
        if base in succ:
            label = base.replace("solv_", "").replace("__s0", "")
            pairs.append((label, succ[base][0], pct))
    if not pairs:
        return False
    labels = [p[0] for p in pairs]
    inctx = [p[1] for p in pairs]
    forage = [p[2] for p in pairs]
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar([i - w / 2 for i in x], inctx, width=w, color=C_PRIMARY,
           label="data_in_context=True (data in prompt)")
    ax.bar([i + w / 2 for i in x], forage, width=w, color=C_WARN,
           label="data_in_context=False (must forage via read_* tools)")
    for i, v in enumerate(inctx):
        ax.text(i - w / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
    for i, v in enumerate(forage):
        ax.text(i + w / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("seed success % (full fair set, 1 run/cell)")
    ax.set_ylim(0, 112)
    ax.set_title("Foraging mode: in-context vs data-out-of-context (must call read_* tools)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "seed_foraging_compare.png")
    return True


_TEMPLATE_LABEL = {
    "solv_u1_reconcile": "U1 reconcile", "solv_u1_tally": "U1 tally",
    "solv_u2_sourcing": "U2 sourcing", "solv_u2_route": "U2 route",
    "solv_u4_migration": "U4 migration", "solv_u4_drift": "U4 drift",
    "solv_u5_diligence": "U5 diligence", "solv_u5_riskcov": "U5 risk-coverage",
    "solv_u5_conflict": "U5 conflict", "solv_u6_inbox": "U6 inbox",
}
_DIFF_ORDER = ["easy", "medium", "hard", "expert"]


def _split_task(tid):
    """-> (base, difficulty_or_None, is_forage)。觅食后缀 __forage 单列（觅食是独立轴，
    不进难度 breakdown）；缺省难度档为 medium（solv_X__sN）。"""
    is_forage = tid.endswith("__forage")
    if is_forage:
        tid = tid[: -len("__forage")]
    if "__" not in tid:
        return tid, None, is_forage
    parts = tid.split("__")
    if len(parts) == 3:                       # solv_X__<diff>__sN
        return parts[0], parts[1], is_forage
    return parts[0], "medium", is_forage      # solv_X__sN


def _seed_success_by_template(paths):
    """从 seed task_log 提取 {template_base: {difficulty: success%}}。可传单个或多个结果文件
    （v7 难度梯度 + v8 新模板 expert 点）合并；觅食任务（__forage）排除（由 foraging_compare 处理）。"""
    if isinstance(paths, str):
        paths = [paths]
    agg = {}  # (base, diff) -> [succ, n]
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        tl = (((d.get("adapters") or {}).get("seed") or {}).get("task_log")) or []
        for t in tl:
            base, diff, is_forage = _split_task(t.get("task_id", ""))
            if diff is None or is_forage:
                continue
            agg.setdefault((base, diff), [0, 0])
            agg[(base, diff)][0] += int(bool(t.get("success_met")))
            agg[(base, diff)][1] += 1
    out = {}
    for (base, diff), (s, n) in agg.items():
        out.setdefault(base, {})[diff] = 100.0 * s / n if n else 0.0
    return out


def breakdown(by_tmpl):
    """难度 breakdown 折线：success vs difficulty，每模板一线；50% 阈值线标 breakdown 点。"""
    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    plotted = 0
    for i, (base, dd) in enumerate(sorted(by_tmpl.items())):
        xs = [j for j, dn in enumerate(_DIFF_ORDER) if dn in dd]
        ys = [dd[_DIFF_ORDER[j]] for j in xs]
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, marker=markers[i % len(markers)], linewidth=2,
                label=_TEMPLATE_LABEL.get(base, base))
        plotted += 1
    if not plotted:
        plt.close(fig)
        return False
    ax.axhline(50, color=C_WARN, linestyle="--", linewidth=1.2, label="50% breakdown threshold")
    ax.set_xticks(range(len(_DIFF_ORDER)))
    ax.set_xticklabels(_DIFF_ORDER)
    ax.set_xlabel("Difficulty (entity-set size / distractors / rules)")
    ax.set_ylabel("Success % (full fair set, 1 run/cell)")
    ax.set_ylim(-5, 108)
    ax.set_title("seed difficulty breakdown: success vs difficulty (per self-contained template)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, "seed_breakdown.png")
    return True


def u5_dimension(by_tmpl):
    """U5 选择性预测：两个 U5 模板在各难度的 success（分集正确率；全答/全弃判错的抗 gaming 指标）。"""
    u5 = {b: dd for b, dd in by_tmpl.items() if b.startswith("solv_u5_")}
    if not u5:
        return False
    diffs = _DIFF_ORDER
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    w = 0.38
    x = range(len(diffs))
    for k, (base, dd) in enumerate(sorted(u5.items())):
        ys = [dd.get(dn, float("nan")) for dn in diffs]
        off = (k - 0.5) * w
        bars = ax.bar([i + off for i in x], [0 if (v != v) else v for v in ys],
                      width=w, label=_TEMPLATE_LABEL.get(base, base),
                      color=C_PRIMARY if k == 0 else C_GOOD)
    ax.set_xticks(list(x))
    ax.set_xticklabels(diffs)
    ax.set_ylabel("Selective-prediction success % (1 run/cell)")
    ax.set_ylim(0, 108)
    ax.set_xlabel("Difficulty")
    ax.set_title("U5 calibration / selective prediction: answer-vs-defer partition correctness")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "seed_u5_calibration.png")
    return True


def reliability():
    groups = ["per-run", "pass@k", "pass^k"]
    v5 = [73.3, 75.0, 69.4]   # full fair set incl. mis-judged dense-table 0s
    v6 = [100, 100, 100]      # after fix, on solvable set
    x = range(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar([i - w / 2 for i in x], v5, width=w, color=C_WARN, label="v5_full (incl. mis-judged dense table)")
    ax.bar([i + w / 2 for i in x], v6, width=w, color=C_GOOD, label="v6 (after fix, solvable set)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 108)
    ax.set_title("seed reliability: v5_full vs v6 (k=5 / k=3)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "seed_reliability.png")


def _glabel(m):
    if m == "seed":
        return "seed\n(real)"
    return m.replace("-ref(mock)", "") + "*\n(mock)"


def grounding(path=None):
    """双轨 grounding。优先读最新结果的 grounding.per_model（v9 真实双值）；无则回退旧硬编码。"""
    models = ["seed\n(real)", "deepseek*\n(mock)", "kimi*\n(mock)", "glm*\n(mock)"]
    synthetic = [42, 70, 38, 25]
    real = [100, 85, 71, 42]
    if path:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        per = ((d.get("grounding") or {}).get("per_model")) or {}
        mlist = [m for m in (d.get("models") or []) if m in per]
        if mlist:
            models = [_glabel(m) for m in mlist]
            synthetic = [100.0 * (per[m].get("synthetic") or 0.0) for m in mlist]
            real = [100.0 * (per[m].get("real") or 0.0) for m in mlist]
    x = range(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar([i - w / 2 for i in x], synthetic, width=w, color=C_WARN, label="synthetic track (symbolic GT)")
    ax.bar([i + w / 2 for i in x], real, width=w, color=C_PRIMARY, label="real track (OCR text)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylabel("grounding hit-rate %")
    ax.set_ylim(0, 108)
    ax.set_title("Dual-track grounding (never merged). *mock refs are oracle-fed, not a fair baseline")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "grounding_dual_track.png")


def three_capability_radar(path):
    """Direct three-capability scorecard: raw pass-rate plus no-network-adjusted pass-rate."""
    tasks, _ = _seed_tasks(path)
    dims = _by_dim(tasks)
    groups = [
        ("Agentic", dims["U1"] + dims["U2"]),
        ("Multimodal", dims["U3"]),
        ("Long-horizon", dims["U4"]),
    ]
    labels = [g[0] for g in groups]
    raw = [_rate(g[1])[0] for g in groups]
    adj = [_rate(g[1], exclude_network=True)[0] for g in groups]
    angles = [2 * math.pi * i / len(labels) for i in range(len(labels))]
    angles_closed = angles + angles[:1]
    raw_closed = raw + raw[:1]
    adj_closed = adj + adj[:1]

    fig, ax = plt.subplots(figsize=(6.2, 5.4), subplot_kw={"polar": True})
    ax.plot(angles_closed, raw_closed, color=C_PRIMARY, linewidth=2, marker="o",
            label="raw pass rate")
    ax.fill(angles_closed, raw_closed, color=C_PRIMARY, alpha=0.10)
    ax.plot(angles_closed, adj_closed, color=C_GOOD, linewidth=2, marker="s",
            label="excluding network drops")
    ax.fill(angles_closed, adj_closed, color=C_GOOD, alpha=0.08)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.set_ylim(0, 100)
    ax.set_title("seed v9 three-capability scorecard (direct task pass rate %)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=9)
    _save(fig, "seed_three_capability_radar.png")


def _load_scorecard():
    """优先读 results/scorecard_v9.json；缺失则用 make_scorecard 即时算最新结果。"""
    sc_path = os.path.join(RESULTS, "scorecard_v9.json")
    if os.path.isfile(sc_path):
        with open(sc_path, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        import make_scorecard
        res = make_scorecard.latest_result()
        return make_scorecard.build_scorecard(res) if res else None
    except Exception:  # noqa: BLE001
        return None


def abcd_radar(sc=None):
    """行业可比 A/B/C/D 雷达图（seed，映射自 测试参考.md 的加权模型）。"""
    sc = sc or _load_scorecard()
    if not sc:
        return False
    comp = sc["composite_ABCD"]
    labels = ["A\ntool-use", "B\nmultimodal", "C\nlong-horizon", "D\nreason/plan"]
    vals = [100.0 * comp[k]["score"] for k in ("A", "B", "C", "D")]
    angles = [2 * math.pi * i / len(labels) for i in range(len(labels))]
    ac = angles + angles[:1]
    vc = vals + vals[:1]
    fig, ax = plt.subplots(figsize=(6.2, 5.6), subplot_kw={"polar": True})
    ax.plot(ac, vc, color=C_PRIMARY, linewidth=2, marker="o")
    ax.fill(ac, vc, color=C_PRIMARY, alpha=0.12)
    for ang, v, lab in zip(angles, vals, ("A", "B", "C", "D")):
        ax.text(ang, min(v + 8, 108), "%.0f" % v, ha="center", fontsize=11, color=C_PRIMARY)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.set_ylim(0, 100)
    total = 100.0 * comp["TOTAL"]
    ax.set_title("seed A/B/C/D capability radar (TOTAL = %.1f / 100)" % total)
    _save(fig, "seed_abcd_radar.png")
    return True


def abcd_total_bar(sc=None):
    """A/B/C/D 维度分 + TOTAL 条形图（含合成-grounding 敏感性的 TOTAL）。"""
    sc = sc or _load_scorecard()
    if not sc:
        return False
    comp = sc["composite_ABCD"]
    labels = ["A\ntool-use", "B\nmultimodal", "C\nlong-horizon", "D\nreason/plan", "TOTAL"]
    vals = [100.0 * comp[k]["score"] for k in ("A", "B", "C", "D")] + [100.0 * comp["TOTAL"]]
    colors = [C_PRIMARY, C_PRIMARY, C_PRIMARY, C_PRIMARY, C_GOOD]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, min(v + 2, 104), "%.1f" % v,
                ha="center", fontsize=10)
    # TOTAL 合成-grounding 敏感性：在 TOTAL 柱上画一条参考线
    total_syn = 100.0 * comp["TOTAL_synthetic_grounding"]
    ax.plot([3.58, 4.42], [total_syn, total_syn], color=C_WARN, linewidth=2, linestyle="--")
    ax.text(4.46, total_syn, "syn-ground\nTOTAL %.1f" % total_syn,
            ha="left", va="center", fontsize=8, color=C_WARN)
    ax.set_ylabel("Score (0-100, weighted)")
    ax.set_ylim(0, 112)
    ax.set_title("seed industry-comparable scorecard: A/B/C/D + TOTAL (weights 0.30/0.25/0.25/0.20)")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "seed_abcd_total.png")
    return True


def _bar_metric_chart(metrics, name, title):
    labels = [m[0] for m in metrics]
    vals = [m[1] for m in metrics]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(labels, vals, color=[C_PRIMARY, C_GOOD, C_WARN, C_MUTED, C_PRIMARY][:len(vals)])
    for bar, metric in zip(bars, metrics):
        label, val, detail = metric
        ax.text(bar.get_x() + bar.get_width() / 2, min(val + 3, 105), detail,
                ha="center", fontsize=9)
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 112)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=15)
    _save(fig, name)


def ability_submetric_bars(path):
    """One submetric bar chart for each outward-facing capability, all from v9 JSON."""
    tasks, data = _seed_tasks(path)
    dims = _by_dim(tasks)
    grounding_seed = (((data.get("grounding") or {}).get("per_model") or {}).get("seed")) or {}

    agentic = dims["U1"] + dims["U2"]
    agentic_forage = [t for t in agentic if _is_forage(t)]
    a_raw = _rate(agentic)
    a_adj = _rate(agentic, exclude_network=True)
    u1 = _rate(dims["U1"])
    u2 = _rate(dims["U2"])
    a_forage = _rate(agentic_forage)
    _bar_metric_chart([
        ("Overall", a_raw[0], f"{a_raw[1]}/{a_raw[2]}"),
        ("No-network", a_adj[0], f"{a_adj[1]}/{a_adj[2]}"),
        ("U1 state", u1[0], f"{u1[1]}/{u1[2]}"),
        ("U2 planning", u2[0], f"{u2[1]}/{u2[2]}"),
        ("Foraging", a_forage[0], f"{a_forage[1]}/{a_forage[2]}"),
    ], "seed_agentic_submetrics.png",
        "Agentic capability submetrics (seed v9)")

    u3 = _rate(dims["U3"])
    ground_tasks = [t for t in dims["U3"] if (t.get("task_id") or "").startswith("ground_")]
    gtask = _rate(ground_tasks)
    real = 100.0 * float(grounding_seed.get("real") or 0.0)
    synthetic = 100.0 * float(grounding_seed.get("synthetic") or 0.0)
    _bar_metric_chart([
        ("U3 pass", u3[0], f"{u3[1]}/{u3[2]}"),
        ("Chart/table/doc", gtask[0], f"{gtask[1]}/{gtask[2]}"),
        ("Real reading", real, f"{real:.0f}%"),
        ("Fine-grained", synthetic, f"{synthetic:.0f}%"),
    ], "seed_multimodal_submetrics.png",
        "Multimodal orchestration submetrics (seed v9)")

    u4 = _rate(dims["U4"])
    u4_adj = _rate(dims["U4"], exclude_network=True)
    migration = _rate([t for t in dims["U4"] if _starts(t, "solv_u4_migration")])
    drift = _rate([t for t in dims["U4"] if _starts(t, "solv_u4_drift")])
    u4_forage = _rate([t for t in dims["U4"] if _is_forage(t)])
    _bar_metric_chart([
        ("U4 pass", u4[0], f"{u4[1]}/{u4[2]}"),
        ("No-network", u4_adj[0], f"{u4_adj[1]}/{u4_adj[2]}"),
        ("Migration", migration[0], f"{migration[1]}/{migration[2]}"),
        ("Drift repair", drift[0], f"{drift[1]}/{drift[2]}"),
        ("Foraging", u4_forage[0], f"{u4_forage[1]}/{u4_forage[2]}"),
    ], "seed_long_horizon_submetrics.png",
        "Long-horizon task submetrics (seed v9)")


if __name__ == "__main__":
    journey()
    reliability()
    res = _latest_result()
    grounding(res)
    if res:
        print("latest result:", os.path.basename(res))
        # v9 完整集已含每模板 easy→expert 全梯度 → 难度 breakdown 单版本即完整（不再跨版本合并，
        # 避免混淆 harness 版本）；旧 v7/v8（难度点稀疏）才回退合并。
        is_full = "_v9" in os.path.basename(res)
        srcs = [res] if is_full else [p for p in (_latest_v7(), res) if p]
        by_tmpl = _seed_success_by_template(srcs)
        # U5 维度值 = 两个 U5 模板在 medium 的 success 均值（无 medium 则取最易档）
        seed_dims = _seed_dim_marginals(res)
        if seed_dims:
            dim_success(seed_dims=seed_dims)
        else:
            u5_vals = [dd.get("medium", dd.get("easy", next(iter(dd.values()))))
                       for b, dd in by_tmpl.items() if b.startswith("solv_u5_")]
            dim_success(u5_val=(sum(u5_vals) / len(u5_vals)) if u5_vals else None)
        three_capability_radar(res)
        ability_submetric_bars(res)
        sc = _load_scorecard()
        if abcd_radar(sc) and abcd_total_bar(sc):
            print("A/B/C/D radar + TOTAL bar generated")
        else:
            print("scorecard_v9.json missing — A/B/C/D charts skipped")
        breakdown(by_tmpl)
        u5_dimension(by_tmpl)
        if foraging_compare(res):
            print("foraging comparison chart generated")
        else:
            print("no foraging pairs in result — foraging chart skipped")
    else:
        print("no v7/v8 result found — generating v1-v6 charts only (U5/breakdown skipped)")
        dim_success()
    print("done")
