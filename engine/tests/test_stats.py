"""
统计主干测试（spec §5 / §7 / §8）。覆盖四组：

  ① GLMM/混合效应：在合成**多模板**数据上拟合，模型对比的 CI **非零宽**且方向覆盖真值
     （强模型 > 弱模型，差异 CI 排除 0）；单模板时退化并被标注。
  ② IRT 参数恢复自检：已知 a/b 模拟 → 校准 → r(â,a)、r(b̂,b) ≥ 0.8 判 trusted；
     故意打乱估计值 → 相关崩塌 → 判 untrusted。
  ③ 权重敏感性（Dirichlet）：能区分"稳定排名"（无翻转对）与"翻转对"（翻转概率>0.30
     → 统计不可区分）。
  ④ Krippendorff's α：一致/对立/部分一致玩具数据行为正确，并精确复现经典数据集 α。

运行：  cd engine && python -m pytest tests/test_stats.py -q
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import stats  # noqa: E402
import irt    # noqa: E402
from judge.panel import (JudgePanel, krippendorff_alpha,  # noqa: E402
                         fit_isotonic_calibrator)


# --------------------------------------------------------------------------- #
# 合成多模板数据生成（template -> instance -> [run 0/1]）
# --------------------------------------------------------------------------- #
def _synth_model(rng, p_base, tmpl_sd, n_t=7, n_i=6, n_r=5):
    data = {}
    for t in range(n_t):
        pt = min(max(p_base + rng.normal(0.0, tmpl_sd), 0.03), 0.97)
        data["t%d" % t] = {
            "i%d" % i: [1.0 if rng.random() < pt else 0.0 for _ in range(n_r)]
            for i in range(n_i)
        }
    return data


# --------------------------------------------------------------------------- #
# ① GLMM / 混合效应
# --------------------------------------------------------------------------- #
def test_glmm_multitemplate_ci_nonzero_and_direction():
    rng = np.random.default_rng(42)
    data = {"strong": _synth_model(rng, 0.85, 0.05),
            "weak": _synth_model(rng, 0.50, 0.07)}
    res = stats.glmm_model_comparison(data, n_boot=800, seed=7, correction="holm")

    # 后端标注（本环境无 statsmodels → bootstrap 混合效应）
    assert "backend" in res and res["backend"]

    # 每模型：多模板、CI 非零宽、点估计落在 CI 内
    for m, pm in res["per_model"].items():
        assert pm["n_templates"] >= 2, m
        assert pm["single_cluster"] is False, m
        assert pm["hi"] > pm["lo"], (m, pm)            # 非退化 CI
        assert pm["ci_width"] > 0.0, (m, pm)
        assert pm["lo"] - 1e-9 <= pm["marginal"] <= pm["hi"] + 1e-9, (m, pm)

    # 方向：strong 边际 > weak 边际
    assert res["per_model"]["strong"]["marginal"] > res["per_model"]["weak"]["marginal"]

    # 模型对比：strong − weak 的差 CI 排除 0、显著、方向正确
    sw = next(c for c in res["contrasts"]
              if {c["a"], c["b"]} == {"strong", "weak"})
    sign = 1.0 if sw["a"] == "strong" else -1.0
    assert sign * sw["delta"] > 0.0, sw                # 方向覆盖真值
    assert sw["lo"] > 0.0 or sw["hi"] < 0.0, sw        # 差异 CI 排除 0
    assert sw["significant"] is True, sw
    assert sw["p_adj"] <= sw["p"] + 1e-9               # 多重比较校正后不更小（保守）
    assert abs(sw["cliffs_delta"]) > 0.2               # 非平凡效应量

    # 方差分解 + 设计效应可用且合理
    vc = res["variance_components"]
    assert vc["sigma2_residual"] >= 0.0
    assert 0.0 <= vc["icc_template"] <= 1.0
    de = res["design_effect"]
    assert de["deff"] >= 1.0
    assert de["n_eff"] <= de["n_obs"]


@pytest.mark.skipif(not stats._HAS_STATSMODELS,
                    reason="statsmodels 未安装：回退 bootstrap 后端，真 GLMM 主估计器路径跳过")
def test_glmm_statsmodels_real_backend_fixed_effects():
    """真 GLMM（statsmodels BinomialBayesMixedGLM）作为 θ_m 主估计器（spec §5.1 / CP3）：
      - fit_glmm 后端标注为 statsmodels-BinomialBayesMixedGLM；
      - 固定效应 θ_m（logit）+ 边际成功率方向正确（strong > weak）；
      - 随机效应 log-SD（模板/实例）分量存在；
      - 经主入口 glmm_model_comparison：backend 透传，且 per_model/对比 CI **仍由两级聚类
        bootstrap** 给出（非退化、差异 CI 排除 0）。
    """
    rng = np.random.default_rng(2024)
    data = {"strong": _synth_model(rng, 0.85, 0.05, n_t=6, n_i=5, n_r=5),
            "weak": _synth_model(rng, 0.45, 0.07, n_t=6, n_i=5, n_r=5)}

    fit = stats.fit_glmm(data)
    assert fit["backend"] == "statsmodels-BinomialBayesMixedGLM"
    fe = fit["fixed_effects"]
    assert fe["strong"]["theta_logit"] > fe["weak"]["theta_logit"], fe   # θ_m 方向
    assert fe["strong"]["marginal"] > fe["weak"]["marginal"], fe
    assert "variance_components_logit" in fit                            # 随机效应分量
    assert {"template", "instance"}.issubset(fit["variance_components_logit"].keys())

    # 主入口：真 GLMM 后端透传，CI 仍走两级聚类 bootstrap（与后端无关）
    res = stats.glmm_model_comparison(data, n_boot=600, seed=3)
    assert res["backend"] == "statsmodels-BinomialBayesMixedGLM"
    for m, pm in res["per_model"].items():
        assert pm["hi"] > pm["lo"], (m, pm)                              # bootstrap CI 非退化
    sw = next(c for c in res["contrasts"] if {c["a"], c["b"]} == {"strong", "weak"})
    sign = 1.0 if sw["a"] == "strong" else -1.0
    assert sign * sw["delta"] > 0.0, sw                                  # 方向覆盖真值
    assert sw["lo"] > 0.0 or sw["hi"] < 0.0, sw                          # 差异 CI 排除 0


def test_glmm_marginal_success_backcompat_and_single_cluster():
    rng = np.random.default_rng(0)
    # 多模板 → CI 非零宽
    multi = _synth_model(rng, 0.7, 0.08, n_t=6)
    est = stats.glmm_marginal_success(multi, n_boot=600, seed=1)
    assert set(["point", "lo", "hi"]).issubset(est.keys())   # 向后兼容键
    assert est["hi"] > est["lo"]
    assert est["single_cluster"] is False
    assert est["n_templates"] == 6

    # 单模板（旧调用形态 {cluster:[values]}）→ 退化但被标注
    single = stats.glmm_marginal_success({"only": [1, 1, 0, 1, 0, 1, 1, 0]},
                                         n_boot=400, seed=1)
    assert single["single_cluster"] is True
    assert single["lo"] <= single["point"] <= single["hi"]


def test_multiple_comparison_monotone():
    pvals = [0.001, 0.02, 0.03, 0.5]
    holm = stats.holm_correction(pvals)
    bh = stats.benjamini_hochberg(pvals)
    assert all(0.0 <= q <= 1.0 for q in holm + bh)
    # 校正后 p 不小于原始 p
    assert all(h >= p - 1e-12 for h, p in zip(holm, pvals))
    assert all(b >= p - 1e-12 for b, p in zip(bh, pvals))


# --------------------------------------------------------------------------- #
# ② IRT 参数恢复自检
# --------------------------------------------------------------------------- #
def test_irt_recovery_2pl_trusted():
    rep = irt.parameter_recovery(model="2pl", n_subjects=600, n_items=40, seed=0)
    assert rep["r_a"] >= 0.8, rep["r_a"]
    assert rep["r_b"] >= 0.8, rep["r_b"]
    assert rep["trusted"] is True
    assert rep["status"] == "trusted"


def test_irt_recovery_1pl_trusted():
    rep = irt.parameter_recovery(model="1pl", n_subjects=500, n_items=30, seed=0)
    assert rep["r_b"] >= 0.8, rep["r_b"]
    assert rep["trusted"] is True


def test_irt_shuffled_params_untrusted():
    rep = irt.parameter_recovery(model="2pl", n_subjects=600, n_items=40, seed=0)
    rng = np.random.default_rng(123)
    a_sh = np.array(rep["a_est"]).copy()
    b_sh = np.array(rep["b_est"]).copy()
    rng.shuffle(a_sh)
    rng.shuffle(b_sh)
    chk = irt.recovery_correlations(rep["a_true"], a_sh, rep["b_true"], b_sh,
                                    model="2pl", threshold=0.8)
    # 打乱后相关崩塌 → 不可信
    assert chk["trusted"] is False
    assert not (chk["r_a"] >= 0.8 and chk["r_b"] >= 0.8)


# --------------------------------------------------------------------------- #
# ③ 权重敏感性（Dirichlet）：稳定排名 vs 翻转对
# --------------------------------------------------------------------------- #
def test_weight_sensitivity_stable_ranking():
    stable = {"A": {"d1": 2.0, "d2": 2.0},
              "B": {"d1": 0.0, "d2": 0.0},
              "C": {"d1": -2.0, "d2": -2.0}}
    res = stats.weight_sensitivity(stable, ["d1", "d2"], n_samples=12000, seed=1)
    assert res["indistinguishable_pairs"] == []        # 无翻转对
    assert res["top1_prob"]["A"] > 0.99                 # A 几乎恒为 rank1
    assert res["mean_kendall_tau"] > 0.99               # 排名与基线高度一致
    assert all(v <= 0.30 for v in res["flip_prob"].values())


def test_weight_sensitivity_flip_pair_indistinguishable():
    # A 在 d1 占优、B 在 d2 占优 → 排名随权重翻转
    flip = {"A": {"d1": 1.0, "d2": 0.0},
            "B": {"d1": 0.0, "d2": 0.9}}
    res = stats.weight_sensitivity(flip, ["d1", "d2"], n_samples=12000, seed=1)
    pair = ("A", "B")
    assert res["flip_prob"][pair] > 0.30                # 翻转概率超阈
    assert pair in res["indistinguishable_pairs"]       # 判"统计不可区分"
    assert res["mean_kendall_tau"] < 0.5                # 排名不稳定


# --------------------------------------------------------------------------- #
# ④ Krippendorff's α
# --------------------------------------------------------------------------- #
def test_krippendorff_consistent_and_inconsistent():
    # 完全一致 → 1.0
    assert abs(krippendorff_alpha([[1, 1, 1], [0, 0, 0], [1, 1, 1]]) - 1.0) < 1e-9
    # 系统对立 → ≤ 0（差于随机）
    assert krippendorff_alpha([[0, 1], [1, 0], [0, 1], [1, 0]]) <= 0.0
    # 部分一致 → (0,1) 之间
    partial = krippendorff_alpha([[1, 1, 0], [0, 0, 0], [1, 1, 1], [1, 0, 0]])
    assert 0.0 < partial < 1.0


def test_krippendorff_canonical_values():
    N = float("nan")
    units = [[1, 1, N, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3],
             [2, 2, 2, 2], [1, 2, 3, 4], [4, 4, 4, 4], [1, 1, 2, 1],
             [2, 2, 2, 2], [N, 5, 5, 5], [N, N, 1, 1], [N, 3, N, N]]
    assert abs(krippendorff_alpha(units, "nominal") - 0.743) < 0.01
    assert abs(krippendorff_alpha(units, "interval") - 0.849) < 0.01
    assert abs(krippendorff_alpha(units, "ordinal") - 0.815) < 0.01


def test_panel_headline_gate_and_flip():
    rubric = ["accurate", "clear", "complete"]

    # 评委高度一致 → α 高 → 可进 headline
    def agree(resp, ctx):
        return [1, 1, 0]
    panel_ok = JudgePanel({"a": agree, "b": agree, "c": agree})
    r_ok = panel_ok.score("some response", rubric)
    assert r_ok["alpha"] >= 0.667
    assert r_ok["headline_eligible"] is True
    assert r_ok["enters_headline"] is True              # 向后兼容键
    assert r_ok["flip_rate"] == 0.0                     # stub 评委位置无关 → 无翻转

    # 评委系统不一致 → α 低 → 剔出 headline
    def jA(resp, ctx):
        return [1, 1, 1]

    def jB(resp, ctx):
        return [0, 0, 0]

    def jC(resp, ctx):
        return [1, 0, 1]
    panel_bad = JudgePanel({"a": jA, "b": jB, "c": jC})
    r_bad = panel_bad.score("some response", rubric)
    assert r_bad["alpha"] < 0.667
    assert r_bad["headline_eligible"] is False
    assert r_bad["reliability_band"] == "drop_from_headline"


def test_panel_position_flip_detected():
    rubric = ["q1", "q2"]

    # 位置敏感评委：forward 全 1，reverse 全 0 → 两序不一致 → 全部记 flip 且保守判 0
    def positional(resp, ctx):
        return [1, 1] if ctx.get("order") == "forward" else [0, 0]
    panel = JudgePanel({"a": positional, "b": positional, "c": positional})
    r = panel.score("resp", rubric)
    assert r["flip_rate"] == 1.0                        # 全部 checkpoint 翻转
    assert r["score_raw"] == 0.0                        # 不一致 → 保守按未通过


def test_human_calibration_isotonic():
    # 单调关系 → Spearman=1，isotonic 后 MAE 小
    judge = [0.1, 0.3, 0.45, 0.6, 0.8, 0.95]
    human = [0.0, 0.25, 0.5, 0.55, 0.85, 1.0]
    cal, report = fit_isotonic_calibrator(judge, human)
    assert abs(report["spearman"] - 1.0) < 1e-9
    assert report["mae"] < 0.1
    # 单调不减
    assert cal(0.2) <= cal(0.7) + 1e-9
