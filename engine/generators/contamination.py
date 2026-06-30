"""
抗污染钩子（spec §6 CP6）。

实现"最小但可被后续调用"的三件套：
1) 按模板/难度的**分布锚定**（distribution anchoring）：声明一个 (模板×难度) 配额规格，
   并据此确定性地物化一批实例 → 维持跨版本可比的题型/难度结构。
2) **新种子同构桥梁集**（isomorph bridge set）：对某 (模板,难度) 用全新种子再生成 N 个
   同构实例（零字面复用），充当 common-person equating 的"同构桥梁"（§6.3 兜底路径）。
3) **isomorph-gap 污染度量**：ContamGap = Acc_orig - Acc_isomorph，配对（cluster）bootstrap
   CI 占位；CI 排除 0 → 该模板族打污染折扣并标注退役（§6.5）。接口可被后续评分流水线调用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from generators.base import DIFFICULTIES, get_template, make_rng, Ctx
from schema import Task
# 只读复用统计主干（CP3）的配对 bootstrap；**不修改 stats.py**
from stats import paired_bootstrap_diff


# --------------------------------------------------------------------------- #
# 1) 分布锚定
# --------------------------------------------------------------------------- #
@dataclass
class AnchorSpec:
    """(模板×难度) 配额：每个 cell 生成 instances_per 个实例。"""
    template_ids: List[str]
    difficulties: List[str]
    instances_per: int = 5
    base_seed: int = 0

    def cells(self) -> List[Tuple[str, str]]:
        return [(t, d) for t in self.template_ids for d in self.difficulties]

    def total(self) -> int:
        return len(self.cells()) * self.instances_per


def materialize_anchor(spec: AnchorSpec):
    """按锚定规格确定性物化实例。返回 [(template_id, difficulty, seed, Task)]。

    种子布局：seed = base_seed + i（i 为该 cell 内序号）→ 同 cell 不同种子=同构变体。
    """
    from generators import build_task  # 延迟导入避免环

    out: List[Tuple[str, str, int, Task]] = []
    for (tid, diff) in spec.cells():
        for i in range(spec.instances_per):
            seed = spec.base_seed + i
            out.append((tid, diff, seed, build_task(tid, seed=seed, difficulty=diff)))
    return out


def anchor_distribution_summary(spec: AnchorSpec) -> Dict[str, int]:
    """锚定的目标分布摘要（cell -> 配额），供卫生体检对照。"""
    return {"%s@%s" % (t, d): spec.instances_per for (t, d) in spec.cells()}


# --------------------------------------------------------------------------- #
# 2) 新种子同构桥梁集
# --------------------------------------------------------------------------- #
def isomorph_bridge_set(template_id: str, difficulty: str = "medium",
                        n: int = 5, base_seed: int = 100000) -> List[Task]:
    """对 (template_id, difficulty) 用**全新种子**生成 n 个同构实例（桥梁集）。

    这些实例与历史实例零字面复用，但结构/难度等价（金标里程碑数恒定），
    可作为 common-person equating 的同构桥梁，绕开"字面 anchor 最易被污染"悖论。
    """
    from generators import build_task

    return [build_task(template_id, seed=base_seed + i, difficulty=difficulty)
            for i in range(n)]


def is_isomorphic(a: Task, b: Task) -> bool:
    """同构判据（最小版）：同模板、同难度、同金标里程碑数。"""
    ka = a.difficulty_knobs.get("template")
    kb = b.difficulty_knobs.get("template")
    da = a.difficulty_knobs.get("difficulty")
    db = b.difficulty_knobs.get("difficulty")
    return (ka == kb and da == db
            and gold_milestone_count(a) == gold_milestone_count(b))


def gold_milestone_count(task: Task) -> int:
    return len([m for m in task.milestones if m.type in ("required", "or_group")])


# --------------------------------------------------------------------------- #
# 3) isomorph-gap 污染度量（operationalized：配对 bootstrap 走 stats，绑定 _bridge 同构集）
# --------------------------------------------------------------------------- #
def isomorph_gap(acc_orig: List[float], acc_isomorph: List[float],
                 n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """ContamGap = mean(Acc_orig) - mean(Acc_isomorph) + 配对 bootstrap 95% CI。

    实装（spec §6.5）：CI 由统计主干 `stats.paired_bootstrap_diff` 计算（**不复制实现、
    不改 stats.py**），与 GLMM 主干用同一套配对 bootstrap，保证一致性。

    入参：
      - 每实例**配对**的正确率（等长列表，acc_orig[i] 与 acc_isomorph[i] 同一同构变体的原题 vs
        新种子桥梁题）→ 配对差 Δ=mean(orig-iso) 的点估计 + 95% CI + 双侧 p；
      - 退化为两组（长度不等）→ 仅点估计，CI 标 nan（占位）。
    判退役（spec §6.5）：CI **排除 0 且为正**（lo>0）→ 原题显著比同构题更易 → 疑似训练污染，
      `flag_retire=True`（该模板族打污染折扣并标注退役）。
    返回键向后兼容：{gap, lo, hi, flag_retire, n_pairs}，并补 {delta, p, method}。
    """
    import numpy as np

    if not acc_orig or not acc_isomorph:
        return {"gap": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "flag_retire": False, "n_pairs": 0, "p": float("nan"),
                "delta": float("nan"), "method": "empty"}

    gap = float(np.mean(acc_orig) - np.mean(acc_isomorph))
    # 配对 bootstrap（仅当等长且可视为同实例配对）→ 复用 stats.paired_bootstrap_diff
    if len(acc_orig) == len(acc_isomorph) and len(acc_orig) >= 2:
        res = paired_bootstrap_diff(list(acc_orig), list(acc_isomorph),
                                    n_boot=n_boot, seed=seed)
        lo, hi = res["lo"], res["hi"]
        flag = (not _is_nan(lo)) and lo > 0.0  # CI 排除 0 且为正 → 疑似污染
        return {"gap": gap, "delta": res["delta"], "lo": lo, "hi": hi,
                "p": res["p"], "flag_retire": bool(flag), "n_pairs": len(acc_orig),
                "method": "paired_bootstrap(stats)"}
    return {"gap": gap, "delta": gap, "lo": float("nan"), "hi": float("nan"),
            "p": float("nan"), "flag_retire": False, "n_pairs": 0,
            "method": "point_only(unequal_length)"}


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and x != x


def build_isomorph_pairs(template_id: str, difficulty: str = "medium", n: int = 5,
                         orig_base_seed: int = 0, bridge_base_seed: int = 100000
                         ) -> List[Tuple[Task, Task]]:
    """构造「原题 ↔ 新种子同构桥梁题」配对集（spec §6.3/§6.5 的 _bridge 同构集）。

    pair i = (build_task(tid, orig_base_seed+i), build_task(tid, bridge_base_seed+i))：
      两者**同构**（同模板/难度/金标里程碑数）但**零字面复用**（不同种子 → 命名/取值不同）。
    按 index i 配对即「共同结构变体」配对，喂给 isomorph_gap 做配对 bootstrap。
    """
    from generators import build_task
    pairs: List[Tuple[Task, Task]] = []
    for i in range(n):
        orig = build_task(template_id, seed=orig_base_seed + i, difficulty=difficulty)
        bridge = build_task(template_id, seed=bridge_base_seed + i, difficulty=difficulty)
        pairs.append((orig, bridge))
    return pairs


def isomorph_gap_report(template_id: str, acc_orig: List[float],
                        acc_bridge: List[float], difficulty: str = "medium",
                        n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """模板族的污染度量报告：把每实例配对正确率喂入 isomorph_gap，附模板元信息。

    acc_orig[i] / acc_bridge[i]：同一配对（build_isomorph_pairs 的 pair i）原题 / 桥梁题的正确率
    （由评分流水线对各 Task 跑模型后得到；本函数与具体模型无关，只做统计）。
    """
    res = dict(isomorph_gap(acc_orig, acc_bridge, n_boot=n_boot, seed=seed))
    res.update({"template_id": template_id, "difficulty": difficulty,
                "n_bridge": len(acc_bridge),
                "acc_orig_mean": (float(sum(acc_orig)) / len(acc_orig)) if acc_orig else float("nan"),
                "acc_bridge_mean": (float(sum(acc_bridge)) / len(acc_bridge)) if acc_bridge else float("nan")})
    return res


def score_task_accuracy(tasks: List[Task], scorer: Callable[[Task], float]) -> List[float]:
    """便捷器：对一组 Task 用 `scorer(task)->[0,1]` 得每实例正确率序列（顺序对齐）。

    `scorer` 可包裹「跑模型 n 次 → per-run 成功率 / 里程碑比例」；本模块不依赖具体模型，
    保持评分流水线可注入（CP6 接口占位转为可运行接口）。
    """
    return [float(scorer(t)) for t in tasks]


# --------------------------------------------------------------------------- #
# 4) 共同被试等值化（common-person equating，spec §6.3）—— 最小可运行实现
# --------------------------------------------------------------------------- #
@dataclass
class CommonPersonEquator:
    """把「新版分数」映射回「旧版量纲」的等值化器（冻结参考面板法，§6.3）。

    用 ≥3 模型参考面板在两版上的分数拟合映射 → 新版分数可与历史可比，**无需字面 anchor item**。
    """
    slope: float
    intercept: float
    method: str = "linear"

    def equate(self, x: float) -> float:
        return self.slope * float(x) + self.intercept

    def equate_all(self, xs: List[float]) -> List[float]:
        return [self.equate(x) for x in xs]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def linear_equating(ref_old: List[float], ref_new: List[float]) -> CommonPersonEquator:
    """线性（均值/标准差）等值化：X_eq = (x - mean_new)/sd_new * sd_old + mean_old。

    ref_old / ref_new：**同一冻结参考面板**（≥3 模型）在 旧版 / 新版 上的分数（顺序无关，
    用各自的均值与 SD）。sd_new=0（面板无散度）时退化为平移（slope=1）。
    """
    mo, so = _mean(ref_old), _std(ref_old)
    mn, sn = _mean(ref_new), _std(ref_new)
    slope = (so / sn) if sn > 0 else 1.0
    intercept = mo - slope * mn
    return CommonPersonEquator(slope=slope, intercept=intercept, method="linear")


@dataclass
class EquipercentileEquator:
    """等百分位等值化（§6.3 备选）：按百分位秩把新版分数映射到旧版分布对应分位值。"""
    ref_old: List[float]
    ref_new: List[float]
    method: str = "equipercentile"

    def equate(self, x: float) -> float:
        import numpy as np
        old = np.sort(np.asarray(self.ref_old, dtype=float))
        new = np.sort(np.asarray(self.ref_new, dtype=float))
        if len(old) == 0 or len(new) == 0:
            return float(x)
        # x 在新版分布中的百分位秩（线性插值）
        pct = float(np.interp(x, new, np.linspace(0.0, 100.0, len(new))))
        # 映射到旧版分布同百分位的分位值
        return float(np.percentile(old, pct))


def common_person_equate(ref_old: List[float], ref_new: List[float],
                         new_scores: List[float], method: str = "linear") -> Dict[str, Any]:
    """一站式：用参考面板把一组「新版分数」等值化到旧版量纲。返回映射后分数 + 等值化器参数。"""
    if method == "equipercentile":
        eq = EquipercentileEquator(ref_old=list(ref_old), ref_new=list(ref_new))
        equated = [eq.equate(x) for x in new_scores]
        return {"method": method, "equated": equated,
                "n_panel": min(len(ref_old), len(ref_new))}
    eq = linear_equating(ref_old, ref_new)
    return {"method": "linear", "equated": eq.equate_all(new_scores),
            "slope": eq.slope, "intercept": eq.intercept,
            "n_panel": min(len(ref_old), len(ref_new))}
