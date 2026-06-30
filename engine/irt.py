"""
IRT 题目校准（spec §5.1 / §8）—— **仅做 item 难度/区分度校准**，绝不估 per-published-model
潜在 θ、绝不进 headline（CP3）。

为什么 IRT 在这套引擎里被限死在"题目侧"：约 5–15 个被测模型下 IRT 对 per-model θ 不可识别
（后验被先验吞没=循环论证）。因此 IRT 只用于：
  - 用**合成被试梯队**（≥~20–30 个能力跨度足够的消融/降级 agent）校准题目 a(区分度)/b(难度)；
  - 配合 Fisher 信息 I_i(θ)=a_i² P(1−P) 做信息驱动选题 / 抗饱和 / 退役饱和题；
  - **参数恢复自检**：在已知参数的模拟数据上估计，若 r(â,a_true)≥0.8 且 r(b̂,b_true)≥0.8
    则标"可信(trusted)"，否则标 untrusted —— 仅作诊断、不得用于选题决策。

估计方法：**MML-EM + Gauss–Hermite 求积**（θ~N(0,1) 先验固定量纲），M 步用分组数据的
Newton-Raphson 拟合每题 logistic（2PL: a=斜率, b=−截距/斜率；1PL/Rasch: a≡1）。
GPCM（多级计分）为可选能力，用 scipy 数值优化 M 步。仅依赖 numpy（GPCM 用 scipy）。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((xm ** 2).sum()) * float((ym ** 2).sum()))
    return float((xm * ym).sum() / denom) if denom > 0 else float("nan")


def _gauss_hermite(n_quad: int) -> Tuple[np.ndarray, np.ndarray]:
    """N(0,1) 求积节点与权重（probabilist's Hermite）。Σ w = 1。"""
    nodes, w = np.polynomial.hermite_e.hermegauss(n_quad)
    w = w / math.sqrt(2.0 * math.pi)
    w = w / w.sum()
    return nodes, w


# --------------------------------------------------------------------------- #
# 合成被试梯队 / 模拟作答（二级计分）
# --------------------------------------------------------------------------- #
def synthetic_subject_ladder(n_subjects: int = 600, low: float = -3.0,
                             high: float = 3.0, kind: str = "normal",
                             seed: int = 0) -> np.ndarray:
    """生成"能力跨度足够"的合成被试梯队 θ（消融/降级 agent 的真 competence 跨度）。

    kind='normal' → θ~N(0,1)（与 MML 先验匹配，恢复最稳）；
    kind='uniform' → θ~U(low,high)（显式阶梯）；
    kind='ladder' → 在 [low,high] 上等距阶梯（确定性消融档）。
    """
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(low, high, size=n_subjects)
    if kind == "ladder":
        return np.linspace(low, high, n_subjects)
    return rng.normal(0.0, 1.0, size=n_subjects)


def simulate_responses(thetas: np.ndarray, a: np.ndarray, b: np.ndarray,
                       model: str = "2pl", seed: int = 0) -> np.ndarray:
    """按 2PL/1PL 生成 0/1 作答矩阵 [n_subjects, n_items]。1PL 强制 a≡1。"""
    rng = np.random.default_rng(seed)
    thetas = np.asarray(thetas, dtype=float)
    a = np.ones_like(b) if model == "1pl" else np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    z = a[None, :] * (thetas[:, None] - b[None, :])
    p = _expit(z)
    return (rng.random(p.shape) < p).astype(float)


# --------------------------------------------------------------------------- #
# 分组 logistic 的 Newton M 步（2PL/1PL）
# --------------------------------------------------------------------------- #
def _fit_item_grouped(X: np.ndarray, Nq: np.ndarray, rq: np.ndarray,
                      a0: float, b0: float, fix_slope: bool,
                      n_iter: int = 50) -> Tuple[float, float]:
    """对分组二项数据 (节点 X_q, 试验数 N_q, 成功数 r_q) 拟合 logit p = α + βX。
    返回 (a, b)：2PL 中 a=β, b=−α/β；1PL 固定 β=1。"""
    beta = 1.0 if fix_slope else max(a0, 1e-2)
    alpha = -a0 * b0 if not fix_slope else -b0
    for _ in range(n_iter):
        eta = alpha + beta * X
        p = _expit(eta)
        w = Nq * p * (1.0 - p)
        resid = rq - Nq * p
        g_a = float(resid.sum())
        H_aa = -float(w.sum()) - 1e-6
        if fix_slope:
            step = g_a / H_aa
            alpha_new = alpha - step
            if abs(alpha_new - alpha) < 1e-7:
                alpha = alpha_new
                break
            alpha = alpha_new
            continue
        g_b = float((X * resid).sum())
        H_ab = -float((X * w).sum())
        H_bb = -float((X * X * w).sum()) - 1e-6
        det = H_aa * H_bb - H_ab * H_ab
        if abs(det) < 1e-12:
            break
        da = (H_bb * g_a - H_ab * g_b) / det
        db = (H_aa * g_b - H_ab * g_a) / det
        alpha_new = alpha - da
        beta_new = beta - db
        beta_new = float(min(max(beta_new, 0.05), 8.0))
        if abs(alpha_new - alpha) < 1e-7 and abs(beta_new - beta) < 1e-7:
            alpha, beta = alpha_new, beta_new
            break
        alpha, beta = alpha_new, beta_new
    if fix_slope:
        return 1.0, float(min(max(-alpha, -6.0), 6.0))
    a = float(min(max(beta, 0.05), 8.0))
    b = float(min(max(-alpha / a, -6.0), 6.0))
    return a, b


# --------------------------------------------------------------------------- #
# MML-EM 校准（2PL / 1PL）
# --------------------------------------------------------------------------- #
def calibrate_items(responses: np.ndarray, model: str = "2pl",
                    n_quad: int = 31, max_iter: int = 200, tol: float = 1e-4,
                    seed: int = 0) -> Dict[str, Any]:
    """MML-EM 校准题目参数。

    responses: [n_subjects, n_items] 的 0/1（可含 np.nan 表示缺答）。
    返回 {a, b, model, n_iter, converged, loglik, fisher_info_fn}。θ 量纲由 N(0,1) 先验固定。
    """
    R = np.asarray(responses, dtype=float)
    n_subj, n_items = R.shape
    fix_slope = (model == "1pl")
    X, gq = _gauss_hermite(n_quad)
    Q = len(X)

    # 初值：b=按题正确率反推的难度，a=1
    with np.errstate(invalid="ignore"):
        pbar = np.nanmean(R, axis=0)
    pbar = np.clip(pbar, 0.02, 0.98)
    a = np.ones(n_items)
    b = -np.log(pbar / (1.0 - pbar))  # 高正确率 → 低难度
    b = np.clip(b, -4.0, 4.0)

    mask = ~np.isnan(R)
    Rfill = np.where(mask, R, 0.0)

    prev_ll = -np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # ----- E 步 -----
        # P[q, item]
        P = _expit(a[None, :] * (X[:, None] - b[None, :]))  # [Q, items]
        P = np.clip(P, 1e-6, 1 - 1e-6)
        logP = np.log(P)
        log1mP = np.log(1.0 - P)
        # 每被试在每节点的对数似然：Σ_item [y logP + (1-y) log(1-P)]（缺答跳过）
        # L[subj, q]
        # 用矩阵乘：贡献 = R·logP^T + (1-R)·log1mP^T，但要按 mask 处理缺答
        ll_correct = (Rfill * mask) @ logP.T            # [subj, Q]
        ll_wrong = ((1.0 - Rfill) * mask) @ log1mP.T    # [subj, Q]
        log_lik_sq = ll_correct + ll_wrong              # [subj, Q]
        log_post = log_lik_sq + np.log(gq)[None, :]
        m = log_post.max(axis=1, keepdims=True)
        post = np.exp(log_post - m)
        denom = post.sum(axis=1, keepdims=True)
        post = post / denom                              # f_sq [subj, Q]
        # 边际对数似然
        marg_ll = float((m.ravel() + np.log(denom.ravel())).sum())

        # 期望计数：N_q[item], r_q[item]（按题分别，因缺答按题不同）
        # N_q[item] = Σ_s post[s,q] * mask[s,item]
        Nq = post.T @ mask                                # [Q, items]
        rq = post.T @ (Rfill * mask)                      # [Q, items]

        # ----- M 步 -----
        for j in range(n_items):
            a[j], b[j] = _fit_item_grouped(X, Nq[:, j], rq[:, j],
                                           a0=a[j], b0=b[j], fix_slope=fix_slope)

        if abs(marg_ll - prev_ll) < tol * (1.0 + abs(prev_ll)):
            converged = True
            prev_ll = marg_ll
            break
        prev_ll = marg_ll

    def fisher_info_fn(theta: float) -> np.ndarray:
        p = _expit(a * (theta - b))
        return (a ** 2) * p * (1.0 - p)

    return {"a": a, "b": b, "model": model, "n_iter": it,
            "converged": converged, "loglik": prev_ll,
            "n_subjects": n_subj, "n_items": n_items,
            "fisher_info_fn": fisher_info_fn}


def item_information(a: float, b: float, theta: float) -> float:
    """Fisher 信息 I_i(θ)=a² P(1−P)（用于信息驱动选题）。"""
    p = float(_expit(np.array([a * (theta - b)]))[0])
    return float(a * a * p * (1.0 - p))


# --------------------------------------------------------------------------- #
# 参数恢复自检（r ≥ 0.8 门）
# --------------------------------------------------------------------------- #
def recovery_correlations(a_true: Sequence[float], a_est: Sequence[float],
                          b_true: Sequence[float], b_est: Sequence[float],
                          model: str = "2pl", threshold: float = 0.8) -> Dict[str, Any]:
    """计算恢复相关并判定 trusted。

    2PL：需 r(â,a)≥thr 且 r(b̂,b)≥thr；1PL：仅判 r(b̂,b)≥thr（a≡1 无需恢复）。
    用于"故意打乱估计值应判 untrusted"的核验。
    """
    r_b = pearson_r(b_true, b_est)
    if model == "1pl":
        trusted = (not math.isnan(r_b)) and r_b >= threshold
        return {"r_a": float("nan"), "r_b": r_b, "trusted": bool(trusted),
                "threshold": threshold, "model": model}
    r_a = pearson_r(a_true, a_est)
    trusted = ((not math.isnan(r_a)) and r_a >= threshold
               and (not math.isnan(r_b)) and r_b >= threshold)
    return {"r_a": r_a, "r_b": r_b, "trusted": bool(trusted),
            "threshold": threshold, "model": model}


def parameter_recovery(model: str = "2pl", n_subjects: int = 600,
                       n_items: int = 40, threshold: float = 0.8,
                       ladder_kind: str = "normal", seed: int = 0,
                       n_quad: int = 31, a_range: Tuple[float, float] = (0.7, 2.2),
                       b_range: Tuple[float, float] = (-2.0, 2.0)) -> Dict[str, Any]:
    """在合成被试梯队上做参数恢复自检：模拟 → 校准 → 相关 → trusted 判定。

    返回含 r_a/r_b/trusted 及真值与估计数组（供"打乱 est → untrusted"核验）。
    """
    rng = np.random.default_rng(seed)
    if model == "1pl":
        a_true = np.ones(n_items)
    else:
        a_true = rng.uniform(a_range[0], a_range[1], size=n_items)
    b_true = rng.uniform(b_range[0], b_range[1], size=n_items)
    thetas = synthetic_subject_ladder(n_subjects, kind=ladder_kind, seed=seed + 1)
    resp = simulate_responses(thetas, a_true, b_true, model=model, seed=seed + 2)
    fit = calibrate_items(resp, model=model, n_quad=n_quad, seed=seed + 3)
    rep = recovery_correlations(a_true, fit["a"], b_true, fit["b"],
                                model=model, threshold=threshold)
    rep.update({"a_true": a_true, "a_est": fit["a"],
                "b_true": b_true, "b_est": fit["b"],
                "converged": fit["converged"], "n_iter": fit["n_iter"],
                "n_subjects": n_subjects, "n_items": n_items,
                "status": "trusted" if rep["trusted"] else "untrusted"})
    return rep


# --------------------------------------------------------------------------- #
# GPCM（多级计分，可选）—— MML-EM + scipy 数值 M 步
# --------------------------------------------------------------------------- #
def _gpcm_probs(theta: np.ndarray, a: float, thresholds: np.ndarray) -> np.ndarray:
    """GPCM 类别概率。thresholds 为 K−1 个步骤难度（类别 1..K-1 的步参）。
    返回 [len(theta), K]。"""
    K = len(thresholds) + 1
    # 累积：s_k = Σ_{c=1}^{k} a(θ − δ_c)，s_0 = 0
    steps = np.zeros((len(theta), K))
    for k in range(1, K):
        steps[:, k] = steps[:, k - 1] + a * (theta - thresholds[k - 1])
    steps = steps - steps.max(axis=1, keepdims=True)
    ex = np.exp(steps)
    return ex / ex.sum(axis=1, keepdims=True)


def calibrate_gpcm(responses: np.ndarray, n_cats: Sequence[int],
                   n_quad: int = 31, max_iter: int = 100, tol: float = 1e-3,
                   seed: int = 0) -> Dict[str, Any]:
    """GPCM 校准（可选能力）。responses[s,i] ∈ {0..n_cats[i]-1}，可含 np.nan。

    返回 {a:[items], thresholds:[item -> array(K_i-1)], converged}。
    依赖 scipy.optimize（M 步逐题数值优化期望完整数据对数似然）。
    """
    from scipy.optimize import minimize

    R = np.asarray(responses, dtype=float)
    n_subj, n_items = R.shape
    X, gq = _gauss_hermite(n_quad)
    Q = len(X)
    a = np.ones(n_items)
    thresholds: List[np.ndarray] = []
    for i in range(n_items):
        K = int(n_cats[i])
        thresholds.append(np.linspace(-1.0, 1.0, K - 1) if K > 1 else np.zeros(0))

    mask = ~np.isnan(R)
    R_int = np.where(mask, R, 0.0).astype(int)
    prev_ll = -np.inf
    converged = False
    for _ in range(max_iter):
        # E 步（向量化）：每被试节点后验
        log_lik_sq = np.zeros((n_subj, Q))
        for i in range(n_items):
            P = np.clip(_gpcm_probs(X, a[i], thresholds[i]), 1e-9, 1.0)  # [Q, K]
            logPT = np.log(P).T                       # [K, Q]
            contrib = logPT[R_int[:, i]]              # [n_subj, Q]
            log_lik_sq += contrib * mask[:, i][:, None]
        log_post = log_lik_sq + np.log(gq)[None, :]
        mmax = log_post.max(axis=1, keepdims=True)
        ex = np.exp(log_post - mmax)
        post = ex / ex.sum(axis=1, keepdims=True)
        marg_ll = float((mmax.ravel() + np.log(ex.sum(axis=1))).sum())

        # M 步：逐题最大化期望完整数据对数似然
        for i in range(n_items):
            K = int(n_cats[i])
            if K < 2:
                continue
            # 期望类别计数 n_qk[q,k]（按类别向量化）
            yi = R_int[:, i]
            mi = mask[:, i]
            n_qk = np.zeros((Q, K))
            for k in range(K):
                sel = mi & (yi == k)
                if sel.any():
                    n_qk[:, k] = post[sel].sum(axis=0)

            def neg_ll(params):
                aa = max(params[0], 0.05)
                thr = params[1:]
                P = np.clip(_gpcm_probs(X, aa, thr), 1e-9, 1.0)
                return -float((n_qk * np.log(P)).sum())

            x0 = np.concatenate([[a[i]], thresholds[i]])
            res = minimize(neg_ll, x0, method="Nelder-Mead",
                           options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-3})
            a[i] = max(float(res.x[0]), 0.05)
            thresholds[i] = res.x[1:]

        if abs(marg_ll - prev_ll) < tol * (1.0 + abs(prev_ll)):
            converged = True
            prev_ll = marg_ll
            break
        prev_ll = marg_ll

    return {"a": a, "thresholds": thresholds, "model": "gpcm",
            "converged": converged, "loglik": prev_ll}
