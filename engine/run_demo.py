"""
AGENIX-Engine 端到端 Demo（不联网、确定性）。

阶段 2 集成后：在**扩充任务银行**上跑（`tasks/generated/{u1,u2,u4,u5,u6}` + 顶层样例，
含 U3 双轨 grounding），orchestrator 用内置 stub 模型评分 -> 打印 Profile-R / Profile-D：
  逐维 GLMM 边际 + **非退化 CI**、逐维 GLMM 模型对比（Δ/p_adj/Cliff's δ/方差/Deff）、
  可靠性四指标、ASR、能力–成本 Pareto、权重敏感性、grounding 双轨 ρ 数据门 + real_trusted、
  IRT 选题门（trusted 才用于选题、不进 headline）、安全 hard-zero 生效示例。

运行：  cd engine && python run_demo.py
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter

# 控制台编码兜底：GBK 控制台无法编码 σ²/特殊符号会抛 UnicodeEncodeError；
# 改 errors='replace' 保证永不因编码崩溃（中文在 GBK 控制台仍正常）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass

from orchestrator import load_task_bank, evaluate


def fmt(x, nd=3):
    if x is None:
        return "  -  "
    if isinstance(x, float) and math.isnan(x):
        return " nan "
    return ("%." + str(nd) + "f") % x


def print_profile(report, prof_key, title):
    prof = report["profiles"][prof_key]
    dims = prof["dims_present"]
    print("\n" + "=" * 92)
    print("PROFILE-%s  %s" % (prof_key, title))
    print("=" * 92)
    header = "model".ljust(14) + "".join(d.ljust(18) for d in dims)
    header += "per_run pass@k pass^k ASR   cost  G_real trust"
    print(header)
    print("-" * len(header))
    for m, agg in prof["per_model"].items():
        row = m.ljust(14)
        for d in dims:
            v = agg["dim_vector"].get(d)
            if v is None:
                row += "      -           ".ljust(18)
            else:
                row += ("%s[%s,%s]" % (fmt(v["point"], 2), fmt(v["lo"], 2),
                                       fmt(v["hi"], 2))).ljust(18)
        rel = agg["reliability"]
        g = agg.get("grounding", {})
        row += " " + fmt(rel["per_run"], 2).ljust(7)
        row += fmt(rel["pass_at_k"], 2).ljust(7)
        row += fmt(rel["pass_pow_k"], 2).ljust(7)
        row += fmt(agg["asr"], 2).ljust(6)
        row += fmt(agg["mean_cost"], 1).ljust(6)
        row += fmt(g.get("real"), 2).ljust(7)
        row += ("yes" if g.get("real_trusted") else "no")
        print(row)
    sens = prof.get("weight_sensitivity", {})
    if sens:
        print("\n  权重敏感性 P(rank1)：",
              {m: fmt(p, 2) for m, p in sens.get("top1_prob", {}).items()})
        indist = sens.get("indistinguishable_pairs", [])
        print("  权重翻转不可区分对(翻转概率>%.2f)：" % (sens.get("flip_threshold") or 0.30),
              indist if indist else "无")


def print_dimension_stats(report):
    ds = report.get("dimension_stats", {})
    if not ds:
        return
    print("\n" + "=" * 92)
    print("逐维 GLMM/混合效应 模型对比（headline 统计；success 嵌套 template/instance/run）")
    print("=" * 92)
    for d in sorted(ds.keys()):
        cmp = ds[d]
        de = cmp["design_effect"]
        vc = cmp["variance_components"]
        print("\n[%s] backend=%s  N=%d  N_eff=%s  Deff=%s  ICC_tmpl=%s  "
              "(var_tmpl=%s var_inst=%s var_resid=%s)"
              % (d, cmp["backend"], de["n_obs"], fmt(de["n_eff"], 1), fmt(de["deff"], 2),
                 fmt(de["rho"], 3), fmt(vc["sigma2_template"], 3),
                 fmt(vc["sigma2_instance"], 3), fmt(vc["sigma2_residual"], 3)))
        for m, pm in cmp["per_model"].items():
            print("    %-14s marginal=%s CI[%s,%s] width=%s  n_tmpl=%d %s"
                  % (m, fmt(pm["marginal"], 3), fmt(pm["lo"], 3), fmt(pm["hi"], 3),
                     fmt(pm["ci_width"], 3), pm["n_templates"],
                     "(single_cluster)" if pm.get("single_cluster") else ""))
        # 仅打印显著对比（Δ / p_adj / Cliff's δ），避免刷屏
        sig = [c for c in cmp["contrasts"] if c.get("significant")]
        for c in sig[:6]:
            print("    %-10s − %-10s  Δ=%s  CI[%s,%s]  p_adj=%s  δ=%s  *显著*"
                  % (c["a"], c["b"], fmt(c["delta"], 3), fmt(c["lo"], 3),
                     fmt(c["hi"], 3), fmt(c["p_adj"], 3), fmt(c["cliffs_delta"], 2)))
        if not sig:
            print("    （本维无显著模型对比；样本/方差不支持下结论）")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    # 扩充任务银行：generated/{u1,u2,u4,u5,u6} 主榜 + 顶层样例（含 U3 双轨 grounding）；_bridge 不进主榜
    tasks = load_task_bank(here)
    by_dim = Counter(t.dimension for t in tasks)
    by_tmpl = Counter((t.difficulty_knobs or {}).get("template", t.task_id) for t in tasks)
    print("扩充任务银行规模：%d 个任务（_bridge 已排除、manifest 非任务）" % len(tasks))
    print("  按维度：", dict(sorted(by_dim.items())))
    print("  模板数：", len(by_tmpl), " （每维多模板 → GLMM 聚类 CI 非退化）")

    # 内置 stub 模型（model_id -> 能力档 profile）
    models = {
        "oracle-bot":   "oracle",
        "strong-bot":   "strong",
        "medium-bot":   "medium",
        "weak-bot":     "weak",
        "rogue-bot":    "injection_victim",   # 完成任务但越权外泄 -> 安全 hard-zero
        "honeypot-bot": "honeypot_caller",     # 调用蜜罐 -> 安全 hard-zero
    }
    report = evaluate(models, tasks, n_runs=5, k=5,
                      dim_n_boot=300, glmm_n_boot=300, irt_gate=True)

    print_profile(report, "R", "科研横评（headline=合成 grounding + per-run/pass@k）")
    print_profile(report, "D", "部署就绪（headline=真实 grounding + pass^k / E[里程碑]）")

    print_dimension_stats(report)

    print("\n" + "=" * 92)
    print("跨轨数据门（CP4）& Pareto & 统计不可区分（两源叠加）")
    print("=" * 92)
    gb = report["grounding"]
    print("grounding 合成-真实 Spearman ρ = %s（唯一真源 grounding.synthetic_real_spearman，n_pairs=%d）"
          % (fmt(gb["rho"], 3), gb["n_pairs"]))
    print("  -> headline 规则:", gb["headline_rule"], " (ρ门=%.2f)" % gb["rho_gate"])
    print("  real_trusted 模型（真实轨过 ML 标定门、进 headline）：",
          gb["real_trusted_models"] or "无")
    # 各 ML 验证器标定值
    calib_seen = {}
    for m, gs in gb["per_model"].items():
        for vid, rep in gs["calibration"].items():
            calib_seen[vid] = rep
    if calib_seen:
        print("  ML 验证器标定（真实计算指标 vs 阈值）：")
        for vid, rep in sorted(calib_seen.items()):
            print("     %-18s %s=%s  passed=%s" % (vid, rep.get("metric"),
                  fmt(rep.get("value"), 3), rep.get("passed")))

    si = report["statistical_indistinguishability"]
    print("\n统计不可区分（叠加两源）：")
    print("  (a) 权重翻转概率>阈 的模型对：", si["weight_flip_pairs"] or "无")
    nonsig = {d: ps for d, ps in si["glmm_nonsignificant_by_dim"].items() if ps}
    print("  (b) 逐维 GLMM 非显著的模型对：", nonsig or "无（各维差异均显著）")

    irt = report.get("irt_item_calibration")
    if irt:
        print("\nIRT 选题门（CP3）：trusted=%s (r_a=%s, r_b=%s) status=%s -> 用途=%s（不进 headline）"
              % (irt.get("trusted"), fmt(irt.get("r_a"), 2), fmt(irt.get("r_b"), 2),
                 irt.get("status"), irt.get("use")))

    print("\n能力–成本 Pareto 前沿:", report["pareto_frontier"])
    for m, cap, cost in report["pareto_points"]:
        print("   %-14s capability=%s  cost=%s" % (m, fmt(cap, 3), fmt(cost, 2)))

    print("\n" + "=" * 92)
    print("安全 hard-zero 生效示例（u1_invoice_reconcile 任务，单次 run）")
    print("=" * 92)
    for m in ("rogue-bot", "honeypot-bot", "oracle-bot"):
        rec = next((r for r in report["raw_records"][m]
                    if r["task_id"] == "u1_invoice_reconcile"), None)
        if rec:
            print("  %-14s raw_success=%-5s success=%-5s critical=%-5s ASR=%s incidents=%s"
                  % (m, rec["raw_success"], rec["success"], rec["critical"],
                     fmt(rec["asr"], 1), [i["id"] for i in rec["incidents"]]))

    print("\nDemo 结束。")


if __name__ == "__main__":
    main()
