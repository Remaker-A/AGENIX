"""
任务银行 / 程序化生成器测试（spec §3 任务体系、§6 抗污染、§8 抗饱和）。

覆盖三条核心断言（任务书要求）：
  ① 确定性：每个模板同 seed 同结果（逐字节相等的 JSON）。
  ② schema 校验 + 可被 orchestrator 用 mock 策略评分（不报错）；并验证 oracle 满分。
  ③ 同构变体难度等价：同模板同难度、不同种子 → gold 里程碑数一致。

附加：维度覆盖、每模板 ≥5 实例、噪声/canary 标记位、U4 注入故障恢复、
U6 安全 hard-zero 不可补偿、抗污染钩子（桥梁集 / isomorph-gap / 锚定）、磁盘银行可加载。

运行：  cd engine && python -m pytest tests/test_dataset.py -q
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import generators as G  # noqa: E402
from generators import (  # noqa: E402
    build_task, build_task_dict, isomorph_variant, isomorph_variants,
    gold_milestone_count, list_template_ids, templates_for,
)
from generators.contamination import (  # noqa: E402
    AnchorSpec, materialize_anchor, anchor_distribution_summary,
    isomorph_bridge_set, isomorph_gap, is_isomorphic,
    build_isomorph_pairs, isomorph_gap_report, score_task_accuracy,
    linear_equating, EquipercentileEquator, CommonPersonEquator,
    common_person_equate,
)
from schema import Task  # noqa: E402
from orchestrator import run_model_on_task, load_tasks  # noqa: E402
from models import ModelAdapter  # noqa: E402
from scoring.score import score_task  # noqa: E402

DIMS = ["U1", "U2", "U4", "U5", "U6"]
TEMPLATE_IDS = list_template_ids()
DIFFS = list(G.DIFFICULTIES)
_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN_DIR = os.path.join(_ENGINE_ROOT, "tasks", "generated")


# --------------------------------------------------------------------------- #
# 覆盖 / 规模
# --------------------------------------------------------------------------- #
def test_dimension_coverage():
    """U1/U2/U4/U5/U6 各 ≥3 个模板；不得包含 U3（归属另一 worker）。"""
    for d in DIMS:
        assert len(templates_for(d)) >= 3, (d, len(templates_for(d)))
    assert templates_for("U3") == []
    covered = {t.template_id for d in DIMS for t in templates_for(d)}
    assert set(TEMPLATE_IDS) == covered
    assert len(TEMPLATE_IDS) >= 15


@pytest.mark.parametrize("tid", TEMPLATE_IDS)
def test_each_template_min_instances(tid):
    """每模板能生成 ≥5 个互异且 schema 合法的实例（难度×种子）。"""
    seen = {}
    for diff in DIFFS:
        for seed in range(2):
            t = build_task(tid, seed=seed, difficulty=diff)
            assert isinstance(t, Task)
            seen[t.task_id] = t
    assert len(seen) >= 5, (tid, len(seen))


# --------------------------------------------------------------------------- #
# ① 确定性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tid", TEMPLATE_IDS)
def test_deterministic_generation(tid):
    """同 (template, seed, difficulty) → 逐字节相等的 JSON。"""
    for diff in DIFFS:
        for seed in (0, 1, 7):
            a = json.dumps(build_task_dict(tid, seed=seed, difficulty=diff),
                           sort_keys=True, ensure_ascii=False)
            b = json.dumps(build_task_dict(tid, seed=seed, difficulty=diff),
                           sort_keys=True, ensure_ascii=False)
            assert a == b, (tid, diff, seed)


# --------------------------------------------------------------------------- #
# ② schema 校验 + orchestrator mock 评分（不报错）+ oracle 满分
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tid", TEMPLATE_IDS)
def test_schema_valid_and_scorable(tid):
    """生成任务可被 schema 解析、JSON 往返、并被 orchestrator 多策略评分不报错。"""
    for diff in DIFFS:
        d = build_task_dict(tid, seed=0, difficulty=diff)
        task = Task(**d)                       # schema 校验
        Task(**json.loads(json.dumps(d)))      # JSON 往返再校验
        for profile in ("oracle", "strong", "medium", "weak", "noop"):
            recs = run_model_on_task(ModelAdapter("m-" + profile, profile), task, 2)
            assert len(recs) == 2
            for r in recs:
                assert r["task_id"] == task.task_id
                assert isinstance(r["success"], bool)
                assert "process" in r and "asr" in r


@pytest.mark.parametrize("tid", TEMPLATE_IDS)
def test_oracle_full_score(tid):
    """oracle 策略在所有难度上达成 success 且 process≈1（gold 与实例一致的强校验）。"""
    for diff in DIFFS:
        task = build_task(tid, seed=0, difficulty=diff)
        rec = run_model_on_task(ModelAdapter("oracle", "oracle"), task, 1)[0]
        assert rec["success"] is True, (tid, diff, rec["success"])
        assert rec["process"] >= 0.99, (tid, diff, rec["process"])
        assert rec["critical"] is False, (tid, diff)


# --------------------------------------------------------------------------- #
# ③ 同构变体难度等价（gold 里程碑数一致）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tid", TEMPLATE_IDS)
def test_isomorph_equal_difficulty(tid):
    """同模板同难度、不同种子 → gold 里程碑数恒等；且确为不同表层实例。"""
    for diff in DIFFS:
        base = build_task(tid, seed=0, difficulty=diff)
        var1 = isomorph_variant(tid, seed=0, difficulty=diff)
        var2 = build_task(tid, seed=1, difficulty=diff)
        gc = gold_milestone_count(base)
        assert gc == gold_milestone_count(var1) == gold_milestone_count(var2), (tid, diff)
        assert gc >= 1
        # 真同构（结构等价）但表层不同（task_id / canary 不同）
        assert is_isomorphic(base, var1)
        assert base.task_id != var1.task_id
        assert base.canary != var1.canary


@pytest.mark.parametrize("tid", TEMPLATE_IDS)
def test_isomorph_difficulty_equivalent_scoring(tid):
    """同构变体在 oracle 下同样满分（难度等价的行为侧验证）。"""
    diff = "hard"
    for t in [build_task(tid, 0, diff)] + isomorph_variants(tid, 0, diff, n=2):
        rec = run_model_on_task(ModelAdapter("oracle", "oracle"), t, 1)[0]
        assert rec["success"] is True, (tid, t.task_id)


# --------------------------------------------------------------------------- #
# 噪声注入 / canary 标记位
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tid", TEMPLATE_IDS)
def test_noise_and_canary_markers(tid):
    """每实例带 canary 字符串 + canary 标记位；medium+ 注入了无关工具。"""
    for diff in DIFFS:
        t = build_task(tid, seed=0, difficulty=diff)
        assert t.canary and "DO-NOT-TRAIN" in t.canary
        assert t.difficulty_knobs.get("canary") is True
        if diff in ("medium", "hard", "expert"):
            assert any(tool.is_noise for tool in t.tools), (tid, diff)


# --------------------------------------------------------------------------- #
# U4 注入故障恢复 / U6 安全 hard-zero
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tid", [t.template_id for t in templates_for("U4")])
def test_u4_injected_fault_recovery(tid):
    """U4 含注入故障；oracle 重试后跨过恢复里程碑并满足恢复质量门。"""
    for diff in DIFFS:
        task = build_task(tid, seed=0, difficulty=diff)
        assert task.fault_injection, tid
        rec = run_model_on_task(ModelAdapter("oracle", "oracle"), task, 1)[0]
        detail = rec["recovery_detail"]
        assert rec["success"] is True, (tid, diff, rec["success"])
        assert rec["n_faults"] >= 1, (tid, diff, rec["n_faults"])
        assert detail["crossed_ratio"] == 1.0, (tid, diff, detail)
        assert rec["rollback_correctness"] == 1.0, (tid, diff, detail)
        assert rec["post_recovery_stability"] == 1.0, (tid, diff, detail)
        assert rec["regression_free_window"] == 1.0, (tid, diff, detail)
        assert detail["diagnosis_satisfied"] == 1.0, (tid, diff, detail)
        assert 0.0 <= rec["recovery"] <= 1.0, (tid, diff, rec["recovery"])


@pytest.mark.parametrize("tid", [t.template_id for t in templates_for("U6")])
def test_u6_safety_hardzero_noncompensable(tid):
    """U6：越权外泄 / 蜜罐调用 → critical hard-zero、Success=0、ASR=1（不可补偿）。"""
    for diff in DIFFS:
        task = build_task(tid, seed=0, difficulty=diff)
        for atk in ("injection_victim", "honeypot_caller"):
            rec = run_model_on_task(ModelAdapter("atk", atk), task, 1)[0]
            assert rec["critical"] is True, (tid, diff, atk)
            assert rec["success"] is False, (tid, diff, atk)
            assert rec["asr"] == 1.0, (tid, diff, atk)


# --------------------------------------------------------------------------- #
# 抗污染钩子（CP6）
# --------------------------------------------------------------------------- #
def test_isomorph_bridge_set():
    """新种子同构桥梁集：n 个同构、互异 task_id、schema 合法、oracle 满分。"""
    tid = "u1_reconcile"
    base = build_task(tid, seed=0, difficulty="medium")
    bridge = isomorph_bridge_set(tid, difficulty="medium", n=5)
    assert len(bridge) == 5
    ids = set()
    for t in bridge:
        assert isinstance(t, Task)
        assert is_isomorphic(base, t)
        ids.add(t.task_id)
        rec = run_model_on_task(ModelAdapter("oracle", "oracle"), t, 1)[0]
        assert rec["success"] is True
    assert len(ids) == 5


def test_isomorph_gap_metric():
    """isomorph-gap：等长配对给点估计 + CI；正向差触发退役旗标。"""
    same = isomorph_gap([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
    assert abs(same["gap"]) < 1e-9 and same["flag_retire"] is False
    contaminated = isomorph_gap([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0])
    assert contaminated["gap"] > 0.5
    assert contaminated["flag_retire"] is True  # CI 排除 0 → 疑似污染
    # 退化输入（不等长）：仅点估计，CI 为 nan
    degraded = isomorph_gap([1.0, 0.0], [0.5])
    assert "gap" in degraded


def test_distribution_anchor():
    """分布锚定：物化实例数 = cells × instances_per，且摘要一致。"""
    spec = AnchorSpec(template_ids=["u1_reconcile", "u5_due_diligence"],
                      difficulties=["easy", "hard"], instances_per=3)
    mat = materialize_anchor(spec)
    assert len(mat) == spec.total() == 2 * 2 * 3
    summary = anchor_distribution_summary(spec)
    assert summary["u1_reconcile@easy"] == 3
    for (_tid, _diff, _seed, task) in mat:
        assert isinstance(task, Task)


# --------------------------------------------------------------------------- #
# 抗污染 operationalize（CP6）：isomorph-gap 走 stats 配对 bootstrap + _bridge 同构集
# --------------------------------------------------------------------------- #
def test_isomorph_gap_uses_stats_paired_bootstrap():
    """isomorph-gap 的配对 CI 由 stats.paired_bootstrap_diff 计算（不复制实现）；含 p 与 method。"""
    same = isomorph_gap([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
    assert abs(same["gap"]) < 1e-9 and same["flag_retire"] is False
    assert "p" in same and "stats" in same["method"]
    contaminated = isomorph_gap([1.0, 1.0, 1.0, 0.9], [0.1, 0.0, 0.2, 0.0])
    assert contaminated["gap"] > 0.5 and contaminated["flag_retire"] is True
    assert contaminated["lo"] > 0.0  # CI 排除 0
    # 同构题反而更易（gap<0）→ 不触发退役（污染是"原题更易"）
    reverse = isomorph_gap([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
    assert reverse["flag_retire"] is False
    # 退化输入（不等长）：仅点估计，无 CI、不退役
    degraded = isomorph_gap([1.0, 0.0], [0.5])
    assert "gap" in degraded and degraded["flag_retire"] is False


def test_build_isomorph_pairs_and_gap_end_to_end():
    """_bridge 配对集：每对同构、零字面复用、oracle 两侧满分；oracle 正确率 → gap≈0 不退役。"""
    tid = "u1_reconcile"
    pairs = build_isomorph_pairs(tid, difficulty="medium", n=5)
    assert len(pairs) == 5
    acc_orig, acc_bridge = [], []
    for orig, bridge in pairs:
        assert is_isomorphic(orig, bridge)
        assert orig.task_id != bridge.task_id and orig.canary != bridge.canary
        a = run_model_on_task(ModelAdapter("oracle", "oracle"), orig, 1)[0]["success"]
        b = run_model_on_task(ModelAdapter("oracle", "oracle"), bridge, 1)[0]["success"]
        acc_orig.append(1.0 if a else 0.0)
        acc_bridge.append(1.0 if b else 0.0)
    gap = isomorph_gap(acc_orig, acc_bridge)
    assert abs(gap["gap"]) < 1e-9 and gap["flag_retire"] is False
    # 模拟污染：原题全对、桥梁题全错 → 退役旗标
    contam = isomorph_gap([1.0] * 5, [0.0] * 5)
    assert contam["flag_retire"] is True


def test_isomorph_gap_report_fields():
    """isomorph_gap_report：携带模板元信息与两侧均值，便于退役决策。"""
    rep = isomorph_gap_report("u4_migration", [1.0, 1.0, 1.0], [1.0, 0.0, 1.0],
                              difficulty="hard")
    assert rep["template_id"] == "u4_migration" and rep["difficulty"] == "hard"
    assert rep["n_bridge"] == 3
    assert abs(rep["acc_orig_mean"] - 1.0) < 1e-9
    assert abs(rep["acc_bridge_mean"] - (2.0 / 3.0)) < 1e-9


def test_score_task_accuracy_injectable():
    """评分流水线可注入：score_task_accuracy 用任意 scorer 把 Task 列表映射为正确率序列。"""
    pairs = build_isomorph_pairs("u5_due_diligence", n=3)
    orig_tasks = [o for o, _ in pairs]

    def _oracle_acc(task):
        return 1.0 if run_model_on_task(ModelAdapter("o", "oracle"), task, 1)[0]["success"] else 0.0

    accs = score_task_accuracy(orig_tasks, _oracle_acc)
    assert accs == [1.0, 1.0, 1.0]


# --------------------------------------------------------------------------- #
# 共同被试等值化（common-person equating，spec §6.3）
# --------------------------------------------------------------------------- #
def test_linear_common_person_equating():
    """线性等值化：同一参考面板 → 恒等映射；平移/缩放面板 → 还原到旧版量纲。"""
    panel_old = [0.2, 0.5, 0.8, 0.9]
    # 同面板 → 斜率≈1、截距≈0、equate 近似恒等
    eq_id = linear_equating(panel_old, panel_old)
    assert abs(eq_id.slope - 1.0) < 1e-9 and abs(eq_id.intercept) < 1e-9
    for x in panel_old:
        assert abs(eq_id.equate(x) - x) < 1e-9
    # 新版整体抬升 0.1（更易）：等值化应把新版分数拉回旧版量纲
    panel_new = [x + 0.1 for x in panel_old]
    eq = linear_equating(panel_old, panel_new)
    assert isinstance(eq, CommonPersonEquator)
    for x_old, x_new in zip(panel_old, panel_new):
        assert abs(eq.equate(x_new) - x_old) < 1e-9


def test_equipercentile_equating_monotonic():
    """等百分位等值化：单调；面板极值映射到旧版分布对应端点。"""
    panel_old = [0.1, 0.3, 0.6, 0.9]
    panel_new = [0.2, 0.4, 0.7, 1.0]
    eq = EquipercentileEquator(ref_old=panel_old, ref_new=panel_new)
    a, b = eq.equate(0.4), eq.equate(0.7)
    assert b >= a  # 单调不降
    assert abs(eq.equate(min(panel_new)) - min(panel_old)) < 1e-9


def test_common_person_equate_oneshot():
    """一站式等值化：把一批新版分数映射回旧版量纲（线性 / 等百分位）。"""
    panel_old = [0.2, 0.5, 0.8]
    panel_new = [0.3, 0.6, 0.9]
    out_lin = common_person_equate(panel_old, panel_new, [0.6], method="linear")
    assert len(out_lin["equated"]) == 1 and abs(out_lin["equated"][0] - 0.5) < 1e-9
    out_eqp = common_person_equate(panel_old, panel_new, [0.6], method="equipercentile")
    assert len(out_eqp["equated"]) == 1 and out_eqp["method"] == "equipercentile"


# --------------------------------------------------------------------------- #
# 磁盘银行（若已由 build_bank 物化）可加载且可评分
# --------------------------------------------------------------------------- #
def test_generated_bank_loadable_if_present():
    """tasks/generated/<dim>/ 下的银行文件可被 load_tasks 解析并由 oracle 跑通。"""
    if not os.path.isdir(_GEN_DIR):
        pytest.skip("任务银行尚未物化（先运行 python -m generators.build_bank）")
    sampled = 0
    for dim in ["u1", "u2", "u4", "u5", "u6"]:
        ddir = os.path.join(_GEN_DIR, dim)
        if not os.path.isdir(ddir):
            continue
        tasks = load_tasks(ddir)
        assert tasks, dim
        # 抽样若干跑通（控制耗时）
        for task in tasks[:3]:
            rec = run_model_on_task(ModelAdapter("oracle", "oracle"), task, 1)[0]
            assert rec["success"] is True, task.task_id
            sampled += 1
    assert sampled > 0


def test_generated_bank_isolated_from_toplevel_loader():
    """generated/ 不得污染 tasks/ 顶层加载（保护 run_demo / test_meta）。"""
    top = load_tasks(os.path.join(_ENGINE_ROOT, "tasks"))
    ids = {t.task_id for t in top}
    # 顶层只应有三个既有样例任务，且不含任何生成任务
    assert "u1_invoice_reconcile" in ids
    assert not any("__s" in i for i in ids)
