"""
AGENIX 程序化任务生成器（generators）—— 公共入口。

用法：
    from generators import build_task, build_task_dict, list_templates
    task = build_task("u1_reconcile", seed=0, difficulty="medium")   # -> schema.Task
    variant = isomorph_variant("u1_reconcile", seed=0, difficulty="medium")  # 同构变体

设计契约：
  - (template, seed) 确定性 → 一个符合 schema 的具体 Task（gold 里程碑 DAG + provenance 门控）。
  - 同 seed 同结果；新 seed = 同构变体（改名/改值/换顺序但难度等价 → gold 里程碑数恒定）。
  - 难度分级 easy/medium/hard/expert；噪声注入（无关工具/误导/部分可观测）；canary 标记位。
覆盖维度：U1 / U2 / U4 / U5 / U6（U3 由另一 worker 负责，本包不涉及）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Union

from schema import Task

from generators.base import (  # noqa: F401
    DIFFICULTIES, Ctx, Template, TaskBuilder, make_rng, canary_tag,
    get_template, all_templates, templates_for, REGISTRY,
)
# 导入模板模块以触发注册
from generators import templates as _templates  # noqa: F401

# 同构变体的确定性"新种子"偏移（与历史种子拉开距离，制造零字面复用的桥梁）
_ISOMORPH_SEED_OFFSET = 7919


def build_task_dict(template_id: str, seed: int = 0,
                    difficulty: str = "medium") -> Dict[str, Any]:
    """确定性生成一个任务的纯 dict（JSON 可序列化）。同 (id,seed,difficulty) 恒等。"""
    tmpl = get_template(template_id)
    if difficulty not in DIFFICULTIES:
        raise ValueError("unknown difficulty: %s" % difficulty)
    rng = make_rng(template_id, difficulty, seed)
    ctx = Ctx(template_id=template_id, dimension=tmpl.dimension,
              difficulty=difficulty, seed=seed, rng=rng)
    builder = tmpl.builder_fn(ctx)
    return builder.to_dict()


def build_task(template_id: str, seed: int = 0,
               difficulty: str = "medium") -> Task:
    """确定性生成并通过 schema 校验，返回 schema.Task。"""
    return Task(**build_task_dict(template_id, seed=seed, difficulty=difficulty))


def isomorph_variant(template_id: str, seed: int = 0,
                     difficulty: str = "medium") -> Task:
    """返回单个同构变体（新种子、同难度）：表层不同但难度等价。"""
    return build_task(template_id, seed=seed + _ISOMORPH_SEED_OFFSET,
                      difficulty=difficulty)


def isomorph_variants(template_id: str, seed: int = 0, difficulty: str = "medium",
                      n: int = 3) -> List[Task]:
    """返回 n 个同构变体（连续新种子、同难度）。"""
    base = seed + _ISOMORPH_SEED_OFFSET
    return [build_task(template_id, seed=base + i, difficulty=difficulty)
            for i in range(n)]


def gold_milestone_count(task_or_dict: Union[Task, Dict[str, Any]]) -> int:
    """金标里程碑数（required ∪ or_group）——同构等价的判据。"""
    if isinstance(task_or_dict, Task):
        ms = task_or_dict.milestones
        return len([m for m in ms if m.type in ("required", "or_group")])
    ms = task_or_dict.get("milestones", [])
    return len([m for m in ms if m.get("type") in ("required", "or_group")])


def list_templates() -> List[Dict[str, str]]:
    """列出所有已注册模板的 (id, dimension, description)。"""
    return [{"template_id": t.template_id, "dimension": t.dimension,
             "description": t.description} for t in all_templates()]


def list_template_ids() -> List[str]:
    return [t.template_id for t in all_templates()]


__all__ = [
    "DIFFICULTIES", "Template", "TaskBuilder",
    "build_task", "build_task_dict", "isomorph_variant", "isomorph_variants",
    "gold_milestone_count", "list_templates", "list_template_ids",
    "templates_for", "all_templates", "get_template",
]
