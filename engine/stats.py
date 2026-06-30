"""
统计主干（CP1 / CP3 / CP7）—— 真实可用实现。

设计目标（spec §5.1 / §5.2 / §5.3）：
  - **统计主干 = GLMM / 混合效应**：模型为固定效应，模板(template)/实例(instance)/运行(run)
    为嵌套随机效应，logit 链。
        logit P(success_{m,t,i,r}) = θ_m + u_template(t) + u_instance(t,i)
  - 报告 θ_m 模型对比 + **95% 两级聚类 bootstrap CI**（先重采样模板，再模板内重采样实例/运行）
    + 多重比较校正（Holm / BH）+ 效应量（Cliff's δ）+ 方差分解（模板/实例/运行）
    + 设计效应 Deff = 1 + (m−1)ρ 的有效样本量 / 样本量估计。
  - **权重敏感性**（CP1）：Dirichlet(α) 采样 ≥10^4 组权重 → 成对 Kendall's τ / 翻转概率；
    翻转概率 > 0.30 的模型对判"统计不可区分"。

后端策略（务必产出非退化 CI）：
  - **主估计器 = statsmodels 真实 GLMM**（`BinomialBayesMixedGLM`，变分贝叶斯，确定性）：
    环境装有 statsmodels（见 requirements_stats.txt）时，`fit_glmm` 用它估固定效应 θ_m
    （logit）+ 随机效应 log-SD（模板/实例）。**θ_m 点估计 = headline 主估计器**。
  - **CI 恒由两级聚类 bootstrap 给出**（先重采样模板、再模板内重采样实例/run；仅依赖
    numpy）：`glmm_marginal_success` / `_paired_cluster_contrast` 产出每模型边际 CI 与配对
    对比 CI —— 无论后端是否为 statsmodels，CI 都走 bootstrap（spec §5.1："CI 仍用两级聚类
    bootstrap"）。
  - **statsmodels 缺省时的回退**：可辩护的两级聚类 bootstrap 混合效应估计——模板等权边际 θ_m
    （logit）+ 嵌套方差分量（矩估计）。两后端产物结构一致，`backend` 字段标注实际后端
    （"statsmodels-BinomialBayesMixedGLM" 或 "bootstrap-mixed-effects"）。**多模板时 CI 有
    正常宽度**。

向后兼容：保留 `cluster_bootstrap_ci / glmm_marginal_success / paired_bootstrap_diff /
spearman_rho / kendall_tau / weight_sensitivity` 的签名与返回键，新增能力以新函数 / 可选参数
实现，不破坏 `scoring/aggregate.py` 等调用方。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 可选真实 GLMM 后端（缺省环境未安装；安装后自动启用）。
try:  # pragma: no cover - 取决于运行环境是否装了 statsmodels
    import pandas as _pd  # noqa: F401
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM  # noqa: F401
    _HAS_STATSMODELS = True
except Exception:  # noqa: BLE001
    _HAS_STATSMODELS = False


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _expit(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


# --------------------------------------------------------------------------- #
# Bootstrap（保留既有签名）
# --------------------------------------------------------------------------- #
def cluster_bootstrap_ci(values_by_cluster: Dict[str, List[float]],
                         n_boot: int = 2000, alpha: float = 0.05,
                         seed: int = 0) -> Tuple[float, float, float]:
    """两级聚类 bootstrap：先有放回抽模板(cluster)，再在被抽模板内有放回抽实例。
    返回 (point, lo, hi)。"""
    rng = np.random.default_rng(seed)
    clusters = list(values_by_cluster.keys())
    flat = [v for vs in values_by_cluster.values() for v in vs]
    if not flat:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(flat))
    if len(clusters) == 0:
        return point, point, point
    boot_means = []
    for _ in range(n_boot):
        chosen = rng.choice(len(clusters), size=len(clusters), replace=True)
        vals = []
        for ci in chosen:
            arr = values_by_cluster[clusters[ci]]
            if not arr:
                continue
            idx = rng.choice(len(arr), size=len(arr), replace=True)
            vals.extend(arr[j] for j in idx)
        if vals:
            boot_means.append(np.mean(vals))
    if not boot_means:
        return point, point, point
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_bootstrap_diff(a: List[float], b: List[float], n_boot: int = 2000,
                          seed: int = 0) -> Dict[str, float]:
    """配对 bootstrap：返回 Δ=mean(a-b) 的点估计、95% CI、双侧 p。"""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    d = a[:n] - b[:n]
    point = float(np.mean(d))
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots.append(np.mean(d[idx]))
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {"delta": point, "lo": float(lo), "hi": float(hi), "p": float(min(1.0, p))}


# --------------------------------------------------------------------------- #
# 相关 / 排名（保留既有签名）
# --------------------------------------------------------------------------- #
def _rank(x: List[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = x.argsort()
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def spearman_rho(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def kendall_tau(x: List[float], y: List[float]) -> float:
    """Kendall's τ-b（处理并列）。"""
    n = len(x)
    if n < 2:
        return float("nan")
    conc = disc = 0
    tx = ty = 0  # 仅在 x / 仅在 y 的并列对
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            s = dx * dy
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
            else:
                if dx == 0 and dy != 0:
                    tx += 1
                elif dy == 0 and dx != 0:
                    ty += 1
                # dx==0 且 dy==0 不计入任何一侧
    n0 = conc + disc + tx
    n1 = conc + disc + ty
    denom = math.sqrt(n0 * n1)
    return (conc - disc) / denom if denom > 0 else float("nan")


# --------------------------------------------------------------------------- #
# 效应量：Cliff's δ（非参，不依赖正态）
# --------------------------------------------------------------------------- #
def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """δ = [#(a>b) − #(a<b)] / (na·nb) ∈ [−1,1]。>0 表示 a 随机性优于 b。"""
    a = [x for x in a if not _is_nan(x)]
    b = [x for x in b if not _is_nan(x)]
    if not a or not b:
        return float("nan")
    av = np.asarray(a, dtype=float)[:, None]
    bv = np.asarray(b, dtype=float)[None, :]
    gt = np.sum(av > bv)
    lt = np.sum(av < bv)
    return float((gt - lt) / (len(a) * len(b)))


# --------------------------------------------------------------------------- #
# 多重比较校正
# --------------------------------------------------------------------------- #
def holm_correction(pvals: Sequence[float]) -> List[float]:
    """Holm–Bonferroni 步降校正，返回与输入同序的校正后 p（已做单调化与截断到 1）。"""
    p = list(pvals)
    m = len(p)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    """Benjamini–Hochberg (BH) FDR 校正，返回与输入同序的校正后 q 值。"""
    p = list(pvals)
    m = len(p)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * m / (rank + 1)
        prev = min(prev, val)
        adj[idx] = min(1.0, prev)
    return adj


def _apply_correction(pvals: Sequence[float], method: str) -> List[float]:
    method = (method or "holm").lower()
    if method in ("bh", "fdr", "benjamini-hochberg"):
        return benjamini_hochberg(pvals)
    if method in ("none", "raw"):
        return [min(1.0, float(x)) for x in pvals]
    return holm_correction(pvals)


# --------------------------------------------------------------------------- #
# 嵌套结构归一化：template -> instance -> [run 结果]
# --------------------------------------------------------------------------- #
def _normalize_nested(data: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    """把若干输入形态归一化为 {template: {instance: [run_values]}}。

    接受：
      - {template: [v, v, ...]}              -> 每个 v 视作"单 run 的实例"
      - {template: [[r,r], [r,r], ...]}      -> 外层=实例，内层=run
      - {template: {instance: [r, r, ...]}}  -> 已是嵌套
    """
    out: Dict[str, Dict[str, List[float]]] = {}
    for tmpl, val in data.items():
        inst_map: Dict[str, List[float]] = {}
        if isinstance(val, dict):
            for inst, runs in val.items():
                runs = [float(x) for x in runs if not _is_nan(x)]
                if runs:
                    inst_map[str(inst)] = runs
        elif isinstance(val, (list, tuple)):
            for k, item in enumerate(val):
                if isinstance(item, (list, tuple)):
                    runs = [float(x) for x in item if not _is_nan(x)]
                    if runs:
                        inst_map["i%d" % k] = runs
                elif not _is_nan(item):
                    inst_map["i%d" % k] = [float(item)]
        if inst_map:
            out[str(tmpl)] = inst_map
    return out


def _template_balanced_marginal(nested: Dict[str, Dict[str, List[float]]]) -> float:
    """模板等权边际估计：实例内取均值 -> 模板内取均值 -> 模板间取均值。

    这是随机效应"对模板总体平均"(population-average) 的边际成功率估计，避免大模板/多实例
    主导（与 GLMM 边际目标一致）。"""
    tmpl_means = []
    for inst_map in nested.values():
        inst_means = [float(np.mean(runs)) for runs in inst_map.values() if runs]
        if inst_means:
            tmpl_means.append(float(np.mean(inst_means)))
    return float(np.mean(tmpl_means)) if tmpl_means else float("nan")


def _resample_nested(nested: Dict[str, Dict[str, List[float]]],
                     rng: np.random.Generator,
                     templates: Optional[List[str]] = None
                     ) -> Dict[str, Dict[str, List[float]]]:
    """两级聚类重采样：模板有放回 -> 实例有放回 -> run 有放回。
    可传入固定的 `templates`（用于配对对比时多个模型共用同一套重采样模板）。"""
    keys = list(nested.keys())
    if not keys:
        return {}
    if templates is None:
        chosen_t = rng.choice(len(keys), size=len(keys), replace=True)
        templates_iter = [keys[i] for i in chosen_t]
    else:
        templates_iter = [t for t in templates if t in nested]
    out: Dict[str, Dict[str, List[float]]] = {}
    for bi, t in enumerate(templates_iter):
        inst_map = nested[t]
        inst_keys = list(inst_map.keys())
        chosen_i = rng.choice(len(inst_keys), size=len(inst_keys), replace=True)
        new_inst: Dict[str, List[float]] = {}
        for bj, ii in enumerate(chosen_i):
            runs = inst_map[inst_keys[ii]]
            idx = rng.choice(len(runs), size=len(runs), replace=True)
            new_inst["b%d_%d" % (bi, bj)] = [runs[j] for j in idx]
        out["bt%d" % bi] = new_inst
    return out


# --------------------------------------------------------------------------- #
# 方差分解（嵌套随机效应，响应尺度矩估计）+ 设计效应
# --------------------------------------------------------------------------- #
def variance_decomposition(data: Dict[str, Any]) -> Dict[str, float]:
    """嵌套随机效应方差分解（ANOVA 矩估计，响应/概率尺度）。

    y_{tir} = μ + a_t + b_{ti} + e_{tir}，a_t~(0,σ²_T)，b_{ti}~(0,σ²_I)，e~(0,σ²_E)。
    返回方差分量、ICC（模板/实例）、各层规模，截断负方差为 0（标准做法）。
    """
    nested = _normalize_nested(data)
    # 展平
    all_runs: List[float] = []
    inst_means: List[Tuple[float, int]] = []   # (实例均值, 实例 run 数)
    tmpl_stats: List[Tuple[float, int, int]] = []  # (模板均值, 模板 run 数, 模板实例数)
    n_inst_total = 0
    for inst_map in nested.values():
        t_runs: List[float] = []
        t_inst_means: List[Tuple[float, int]] = []
        for runs in inst_map.values():
            m = float(np.mean(runs))
            inst_means.append((m, len(runs)))
            t_inst_means.append((m, len(runs)))
            t_runs.extend(runs)
            all_runs.extend(runs)
        if t_runs:
            tmpl_stats.append((float(np.mean(t_runs)), len(t_runs), len(t_inst_means)))
            n_inst_total += len(t_inst_means)

    N = len(all_runs)
    T = len(tmpl_stats)
    out = {"sigma2_template": 0.0, "sigma2_instance": 0.0, "sigma2_residual": 0.0,
           "icc_template": 0.0, "icc_instance": 0.0,
           "n_runs": float(N), "n_templates": float(T),
           "n_instances": float(n_inst_total),
           "avg_runs_per_instance": float("nan"),
           "avg_runs_per_template": float("nan")}
    if N == 0 or T == 0:
        return out
    grand = float(np.mean(all_runs))

    # 残差 SS（run 在实例内）
    sse = 0.0
    for inst_map in nested.values():
        for runs in inst_map.values():
            mu_i = float(np.mean(runs))
            sse += sum((x - mu_i) ** 2 for x in runs)
    df_e = max(1, N - n_inst_total)
    mse = sse / df_e

    # 实例 SS（实例均值 vs 模板均值，按 run 数加权）
    ssb = 0.0
    for inst_map, (mu_t, _, _) in zip(nested.values(), tmpl_stats):
        for runs in inst_map.values():
            ssb += len(runs) * (float(np.mean(runs)) - mu_t) ** 2
    df_b = max(1, n_inst_total - T)
    msb = ssb / df_b

    # 模板 SS（模板均值 vs 总均值，按 run 数加权）
    ssa = sum(nr * (mu_t - grand) ** 2 for (mu_t, nr, _) in tmpl_stats)
    df_a = max(1, T - 1)
    msa = ssa / df_a

    n_bar = N / n_inst_total if n_inst_total else 1.0       # 平均每实例 run 数
    m_bar = N / T if T else 1.0                              # 平均每模板 run 数

    sigma2_e = max(0.0, mse)
    sigma2_i = max(0.0, (msb - mse) / n_bar) if n_bar > 0 else 0.0
    sigma2_t = max(0.0, (msa - msb) / m_bar) if m_bar > 0 else 0.0
    total = sigma2_e + sigma2_i + sigma2_t
    out.update({
        "sigma2_template": sigma2_t,
        "sigma2_instance": sigma2_i,
        "sigma2_residual": sigma2_e,
        "icc_template": (sigma2_t / total) if total > 0 else 0.0,
        "icc_instance": (sigma2_i / total) if total > 0 else 0.0,
        "avg_runs_per_instance": float(n_bar),
        "avg_runs_per_template": float(m_bar),
    })
    return out


def _pooled_within_model_variance(
        nested_by_model: Dict[str, Dict[str, Dict[str, List[float]]]]) -> Dict[str, float]:
    """逐模型做嵌套方差分解，再按 run 数加权合并方差分量（模型内 ICC）。

    相比"把所有模型模板池化"，此法不把模型间均值差算进 σ²_template，ICC/Deff 更可辩护。
    """
    comps = {"sigma2_template": 0.0, "sigma2_instance": 0.0, "sigma2_residual": 0.0}
    runs_acc = inst_acc = tmpl_acc = 0.0
    wsum = 0.0
    n_bar_acc = m_bar_acc = 0.0
    for nested in nested_by_model.values():
        vd = variance_decomposition(nested)
        w = vd["n_runs"]
        if w <= 0:
            continue
        wsum += w
        for k in comps:
            comps[k] += w * vd[k]
        runs_acc += vd["n_runs"]
        inst_acc += vd["n_instances"]
        tmpl_acc += vd["n_templates"]
        if not _is_nan(vd["avg_runs_per_instance"]):
            n_bar_acc += w * vd["avg_runs_per_instance"]
        if not _is_nan(vd["avg_runs_per_template"]):
            m_bar_acc += w * vd["avg_runs_per_template"]
    if wsum <= 0:
        return variance_decomposition({})
    for k in comps:
        comps[k] /= wsum
    total = sum(comps.values())
    return {**comps,
            "icc_template": (comps["sigma2_template"] / total) if total > 0 else 0.0,
            "icc_instance": (comps["sigma2_instance"] / total) if total > 0 else 0.0,
            "n_runs": runs_acc, "n_templates": tmpl_acc, "n_instances": inst_acc,
            "avg_runs_per_instance": (n_bar_acc / wsum),
            "avg_runs_per_template": (m_bar_acc / wsum)}


def design_effect(icc: float, avg_cluster_size: float) -> float:
    """Deff = 1 + (m − 1)·ρ。"""
    if _is_nan(icc) or _is_nan(avg_cluster_size):
        return float("nan")
    return 1.0 + max(0.0, avg_cluster_size - 1.0) * max(0.0, icc)


def effective_n(n_obs: float, deff: float) -> float:
    if _is_nan(deff) or deff <= 0:
        return float("nan")
    return n_obs / deff


def mcnemar_sample_size(p_disc: float, effect: float, alpha: float = 0.05,
                        power: float = 0.8) -> float:
    """配对二元（McNemar）所需"判别对"数近似。

    p_disc：判别对（两模型结论不同）的总比例；effect：判别对中偏向某模型的不平衡度
    |p_b − p_c| / p_disc ∈ (0,1]。返回所需配对实例数（已按判别比例反推总样本）。
    """
    from math import sqrt
    # 标准正态分位（避免引 scipy 依赖于此处）
    z_a = _norm_ppf(1 - alpha / 2)
    z_b = _norm_ppf(power)
    if p_disc <= 0 or effect <= 0:
        return float("nan")
    # 判别对内部偏向比例
    psi = effect
    n_disc = (z_a + z_b) ** 2 / (psi ** 2)
    return float(n_disc / p_disc)


def glmm_sample_size(base_rate: float, target_lift: float, icc: float,
                     avg_cluster_size: float, alpha: float = 0.05,
                     power: float = 0.8) -> Dict[str, float]:
    """聚类设计下检测成功率提升 target_lift 所需样本量（含设计效应放大）。

    先按两比例检验算独立样本量 n_iid，再乘 Deff 得聚类样本量 N_cluster，并给出所需模板数。
    """
    p1 = min(max(base_rate, 1e-4), 1 - 1e-4)
    p2 = min(max(base_rate + target_lift, 1e-4), 1 - 1e-4)
    z_a = _norm_ppf(1 - alpha / 2)
    z_b = _norm_ppf(power)
    pbar = (p1 + p2) / 2.0
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar))
           + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n_iid = num / ((p2 - p1) ** 2) if p2 != p1 else float("inf")
    deff = design_effect(icc, avg_cluster_size)
    n_cluster = n_iid * deff
    n_templates = n_cluster / avg_cluster_size if avg_cluster_size > 0 else float("nan")
    return {"n_iid_per_arm": float(n_iid), "deff": float(deff),
            "n_cluster_per_arm": float(n_cluster),
            "n_templates_per_arm": float(n_templates)}


def _norm_ppf(p: float) -> float:
    """标准正态分位（Acklam 近似，避免在样本量函数里引 scipy）。"""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --------------------------------------------------------------------------- #
# GLMM 边际成功率（向后兼容入口；升级为两级聚类 bootstrap）
# --------------------------------------------------------------------------- #
def glmm_marginal_success(values_by_cluster: Dict[str, Any],
                          n_boot: int = 2000, seed: int = 0,
                          alpha: float = 0.05) -> Dict[str, float]:
    """GLMM 边际成功率（CP3 脊柱）。模板等权边际 + 两级聚类 bootstrap 95% CI。

    - 输入兼容 `{template: [values]}`（既有调用方），亦支持嵌套
      `{template: {instance: [runs]}}` / `{template: [[runs],...]}`。
    - **多模板** → 模板间 + 模板内变异共同进入 CI，CI 有正常宽度。
    - **单模板** → 退化为模板内（实例/run）重采样，CI 非零宽但 `single_cluster=True`
      标注（模板间方差不可识别，会低估不确定性）。

    返回至少含 {point, lo, hi}，并附 logit / n_templates / single_cluster 等诊断键。
    """
    nested = _normalize_nested(values_by_cluster)
    if not nested:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "logit": float("nan"), "n_templates": 0, "single_cluster": True}
    point = _template_balanced_marginal(nested)
    rng = np.random.default_rng(seed)
    boots: List[float] = []
    for _ in range(n_boot):
        rs = _resample_nested(nested, rng)
        est = _template_balanced_marginal(rs)
        if not _is_nan(est):
            boots.append(est)
    if not boots:
        return {"point": point, "lo": point, "hi": point,
                "logit": _logit(point), "n_templates": len(nested),
                "single_cluster": len(nested) < 2}
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"point": float(point), "lo": lo, "hi": hi,
            "logit": _logit(point), "se_boot": float(np.std(boots, ddof=1)),
            "n_templates": len(nested), "single_cluster": len(nested) < 2}


# --------------------------------------------------------------------------- #
# 真实 GLMM 拟合（statsmodels 优先；缺省走 bootstrap 混合效应）
# --------------------------------------------------------------------------- #
def fit_glmm(data_by_model: Dict[str, Dict[str, Any]],
             link: str = "logit") -> Dict[str, Any]:
    """拟合 logit P(success) = θ_m + u_template + u_instance。

    输入：data_by_model[model][template] = [runs] 或 {instance:[runs]} 或 [[runs],...]。
    输出：{backend, fixed_effects{model:{theta_logit, marginal}}, variance_components, ...}。
    statsmodels 可用时用 BinomialBayesMixedGLM 估 θ_m（logit）与随机效应方差；
    否则用模板等权边际 logit 作 θ_m + 嵌套矩估计方差分量（可辩护近似）。
    """
    models = list(data_by_model.keys())
    nested_by_model = {m: _normalize_nested(data_by_model[m]) for m in models}

    if _HAS_STATSMODELS:  # pragma: no cover - 取决于环境
        try:
            return _fit_glmm_statsmodels(nested_by_model)
        except Exception:  # noqa: BLE001 - 回退到 bootstrap 近似
            pass

    fixed: Dict[str, Dict[str, float]] = {}
    for m in models:
        marg = _template_balanced_marginal(nested_by_model[m])
        fixed[m] = {"theta_logit": _logit(marg) if not _is_nan(marg) else float("nan"),
                    "marginal": marg}
    # 池化所有模型数据做整体方差分解（模板/实例/残差）
    pooled: Dict[str, Dict[str, List[float]]] = {}
    for m in models:
        for t, inst_map in nested_by_model[m].items():
            key = "%s::%s" % (m, t)
            pooled[key] = inst_map
    vc = variance_decomposition(pooled)
    return {"backend": "bootstrap-mixed-effects", "link": link,
            "fixed_effects": fixed, "variance_components": vc,
            "models": models}


def _fit_glmm_statsmodels(nested_by_model: Dict[str, Dict[str, Dict[str, List[float]]]]
                          ) -> Dict[str, Any]:  # pragma: no cover - 环境相关
    """statsmodels BinomialBayesMixedGLM 真实 GLMM（变分贝叶斯，确定性）。"""
    import pandas as pd
    rows = []
    for m, nested in nested_by_model.items():
        for t, inst_map in nested.items():
            for i, runs in inst_map.items():
                for r, y in enumerate(runs):
                    rows.append({"model": m, "template": "%s" % t,
                                 "instance": "%s::%s" % (t, i),
                                 "y": int(round(float(y)))})
    df = pd.DataFrame(rows)
    md = BinomialBayesMixedGLM.from_formula(
        "y ~ C(model)",
        {"template": "0 + C(template)", "instance": "0 + C(instance)"}, df)
    res = md.fit_vb()
    # 固定效应：截距 + C(model) 对比；换算每模型 logit
    fe = dict(zip(md.exog_names, np.asarray(res.fe_mean)))
    intercept = fe.get("Intercept", 0.0)
    models = list(nested_by_model.keys())
    fixed: Dict[str, Dict[str, float]] = {}
    ref = None
    for m in models:
        col = "C(model)[T.%s]" % m
        if col in fe:
            logit = intercept + fe[col]
        else:
            ref = m
            logit = intercept
        fixed[m] = {"theta_logit": float(logit), "marginal": float(_expit(logit))}
    vcp = {name: float(v) for name, v in zip(md.vcp_names, np.asarray(res.vcp_mean))}
    return {"backend": "statsmodels-BinomialBayesMixedGLM", "link": "logit",
            "fixed_effects": fixed, "variance_components_logit": vcp,
            "reference_model": ref, "models": models}


# --------------------------------------------------------------------------- #
# 模型对比主估计器（headline）
# --------------------------------------------------------------------------- #
def glmm_model_comparison(data_by_model: Dict[str, Dict[str, Any]],
                          baseline: Optional[str] = None,
                          n_boot: int = 2000, alpha: float = 0.05,
                          correction: str = "holm", seed: int = 0,
                          all_pairs: bool = True) -> Dict[str, Any]:
    """GLMM/混合效应模型对比（CP3 主输出）。

    返回：
      - backend：实际 GLMM 后端
      - per_model：{model: {marginal, logit, lo, hi, ci_width, n_obs, n_templates}}
      - contrasts：成对/对基线对比 {a,b, delta, delta_logit, lo, hi, p, p_adj,
        significant, cliffs_delta}（CI 为配对两级聚类 bootstrap，p 已做 Holm/BH 校正）
      - variance_components / icc：嵌套方差分解
      - design_effect：{rho(=icc_template), avg_cluster_size, deff, n_obs, n_eff}
      - sample_size：检测中位提升所需样本量（含 Deff 放大）
    """
    models = list(data_by_model.keys())
    nested_by_model = {m: _normalize_nested(data_by_model[m]) for m in models}
    glmm = fit_glmm(data_by_model)

    # 每模型边际 + 两级聚类 bootstrap CI（各自重采样）
    per_model: Dict[str, Dict[str, Any]] = {}
    for m in models:
        est = glmm_marginal_success(nested_by_model[m], n_boot=n_boot, seed=seed, alpha=alpha)
        n_obs = sum(len(r) for im in nested_by_model[m].values() for r in im.values())
        per_model[m] = {"marginal": est["point"], "logit": est.get("logit"),
                        "lo": est["lo"], "hi": est["hi"],
                        "ci_width": (est["hi"] - est["lo"]) if not _is_nan(est["lo"]) else float("nan"),
                        "n_obs": int(n_obs), "n_templates": est.get("n_templates", 0),
                        "single_cluster": est.get("single_cluster", True)}

    # 成对对比（配对两级聚类 bootstrap：共用同一套重采样模板）
    if baseline is not None and not all_pairs:
        pairs = [(m, baseline) for m in models if m != baseline]
    else:
        pairs = [(models[i], models[j]) for i in range(len(models))
                 for j in range(i + 1, len(models))]

    contrasts: List[Dict[str, Any]] = []
    raw_p: List[float] = []
    for (a, b) in pairs:
        c = _paired_cluster_contrast(nested_by_model[a], nested_by_model[b],
                                     n_boot=n_boot, alpha=alpha, seed=seed)
        # Cliff's δ 用每实例均值序列
        a_inst = [float(np.mean(r)) for im in nested_by_model[a].values() for r in im.values()]
        b_inst = [float(np.mean(r)) for im in nested_by_model[b].values() for r in im.values()]
        c["cliffs_delta"] = cliffs_delta(a_inst, b_inst)
        c["a"], c["b"] = a, b
        contrasts.append(c)
        raw_p.append(c["p"])

    p_adj = _apply_correction(raw_p, correction)
    for c, q in zip(contrasts, p_adj):
        c["p_adj"] = q
        c["significant"] = (not _is_nan(q)) and q < alpha and (c["lo"] > 0 or c["hi"] < 0)

    # 方差分解：逐模型分解后按 run 数加权合并（消除模型间均值差污染 σ²_template），
    # 得到"模型内"嵌套方差分量与 ICC —— 这是设计效应所需的聚类内相关。
    vc = _pooled_within_model_variance(nested_by_model)
    rho = vc["icc_template"]
    m_bar = vc["avg_runs_per_template"]
    deff = design_effect(rho, m_bar)
    n_total = sum(pm["n_obs"] for pm in per_model.values())
    marg_vals = [pm["marginal"] for pm in per_model.values() if not _is_nan(pm["marginal"])]
    base_rate = float(np.mean(marg_vals)) if marg_vals else float("nan")
    lift = (max(marg_vals) - min(marg_vals)) / 2.0 if len(marg_vals) >= 2 else 0.1
    ss = glmm_sample_size(base_rate, max(lift, 1e-3), rho, max(m_bar, 1.0))

    return {"backend": glmm["backend"], "models": models,
            "per_model": per_model, "contrasts": contrasts,
            "correction": correction,
            "variance_components": vc,
            "design_effect": {"rho": rho, "avg_cluster_size": m_bar,
                              "deff": deff, "n_obs": n_total,
                              "n_eff": effective_n(n_total, deff)},
            "sample_size": ss,
            "glmm_fit": glmm}


def _paired_cluster_contrast(nested_a: Dict[str, Dict[str, List[float]]],
                             nested_b: Dict[str, Dict[str, List[float]]],
                             n_boot: int, alpha: float, seed: int) -> Dict[str, float]:
    """两模型的边际差 Δ = marginal_a − marginal_b 的配对两级聚类 bootstrap。

    两模型在共同模板上评测（common items）：每次 bootstrap 先对共同模板有放回抽样，
    两模型在**同一套**重采样模板上各自重算边际，取差 → 配对、相关感知的 CI 与 p。
    """
    pa = _template_balanced_marginal(nested_a)
    pb = _template_balanced_marginal(nested_b)
    delta = pa - pb
    common = [t for t in nested_a.keys() if t in nested_b]
    rng = np.random.default_rng(seed + 7919)
    boots: List[float] = []
    pool = common if common else None
    keys_a = list(nested_a.keys())
    keys_b = list(nested_b.keys())
    for _ in range(n_boot):
        if pool:
            chosen = [pool[i] for i in rng.choice(len(pool), size=len(pool), replace=True)]
            ra = _resample_nested(nested_a, rng, templates=chosen)
            rb = _resample_nested(nested_b, rng, templates=chosen)
        else:
            # 无共同模板：各自独立重采样（非配对）
            ca = [keys_a[i] for i in rng.choice(len(keys_a), size=len(keys_a), replace=True)]
            cb = [keys_b[i] for i in rng.choice(len(keys_b), size=len(keys_b), replace=True)]
            ra = _resample_nested(nested_a, rng, templates=ca)
            rb = _resample_nested(nested_b, rng, templates=cb)
        ea = _template_balanced_marginal(ra)
        eb = _template_balanced_marginal(rb)
        if not _is_nan(ea) and not _is_nan(eb):
            boots.append(ea - eb)
    if not boots:
        return {"delta": delta, "delta_logit": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    boots_arr = np.asarray(boots)
    lo = float(np.percentile(boots_arr, 100 * alpha / 2))
    hi = float(np.percentile(boots_arr, 100 * (1 - alpha / 2)))
    p = 2.0 * min(float(np.mean(boots_arr <= 0)), float(np.mean(boots_arr >= 0)))
    dlogit = (_logit(pa) - _logit(pb)) if (not _is_nan(pa) and not _is_nan(pb)) else float("nan")
    return {"delta": float(delta), "delta_logit": float(dlogit),
            "lo": lo, "hi": hi, "p": float(min(1.0, p))}


# --------------------------------------------------------------------------- #
# 权重敏感性（CP1）—— Dirichlet 采样 + Kendall τ + 翻转概率
# --------------------------------------------------------------------------- #
def weight_sensitivity(dim_scores: Dict[str, Dict[str, float]],
                       dims: List[str], n_samples: int = 10000,
                       seed: int = 0, alpha: Optional[Sequence[float]] = None,
                       flip_threshold: float = 0.30) -> Dict[str, object]:
    """dim_scores[model][dim] = 标准化分。W ~ Dir(α) 采样（默认 α=1 对称）权重，统计：
    - top1_prob：各模型成为 rank1 的概率
    - flip_prob：每对模型相对基线(等权)排名的翻转概率（>flip_threshold → 不可区分）
    - pairwise_kendall_tau / mean_kendall_tau：采样排名 vs 等权基线排名的 Kendall τ
    - indistinguishable_pairs：翻转概率超阈的模型对
    保持既有返回键（top1_prob / flip_prob / indistinguishable_pairs）向后兼容。
    """
    rng = np.random.default_rng(seed)
    models = list(dim_scores.keys())
    M = np.array([[float(dim_scores[m].get(d, 0.0)) for d in dims] for m in models])
    K = len(dims)
    if K == 0 or len(models) == 0:
        return {"top1_prob": {}, "flip_prob": {}, "indistinguishable_pairs": [],
                "mean_kendall_tau": float("nan"), "pairwise_kendall_tau": {}}
    a_vec = np.ones(K) if alpha is None else np.asarray(alpha, dtype=float)

    base = M.mean(axis=1)
    base_order = np.argsort(-base)
    base_rankpos = {models[base_order[r]]: r for r in range(len(models))}
    base_scores = list(base)

    top1 = {m: 0 for m in models}
    flips = {(models[i], models[j]): 0
             for i in range(len(models)) for j in range(i + 1, len(models))}
    tau_sum = 0.0
    for _ in range(n_samples):
        w = rng.dirichlet(a_vec)
        comp = M @ w
        order = np.argsort(-comp)
        top1[models[order[0]]] += 1
        rankpos = {models[order[r]]: r for r in range(len(models))}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                mi, mj = models[i], models[j]
                if (base_rankpos[mi] < base_rankpos[mj]) != (rankpos[mi] < rankpos[mj]):
                    flips[(mi, mj)] += 1
        tau_sum += kendall_tau(base_scores, list(comp))

    top1_prob = {m: top1[m] / n_samples for m in models}
    flip_prob = {k: v / n_samples for k, v in flips.items()}
    indistinguishable = [k for k, v in flip_prob.items() if v > flip_threshold]
    return {"top1_prob": top1_prob, "flip_prob": flip_prob,
            "indistinguishable_pairs": indistinguishable,
            "mean_kendall_tau": tau_sum / n_samples,
            "flip_threshold": flip_threshold}
