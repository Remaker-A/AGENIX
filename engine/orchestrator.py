"""
编排器：对一组模型跑一组任务（每任务 n 次 run）-> 打分 -> 聚合为双 Profile 报告。
确定性、不联网。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from schema import Task
from sandbox import Sandbox
from models import ModelAdapter
from scoring.score import score_task
from scoring.aggregate import build_report


def load_tasks(task_dir: str) -> List[Task]:
    """加载单一目录下的 *.json（**不递归**）为 Task 列表。

    注意：刻意不递归——`tasks/` 顶层只含三类既有样例任务，generated/ 银行不得污染顶层加载
    （test_dataset.test_generated_bank_isolated_from_toplevel_loader 依赖此行为）。
    扩充任务银行请用 `load_task_bank`。
    """
    tasks = []
    if not os.path.isdir(task_dir):
        return tasks
    for fn in sorted(os.listdir(task_dir)):
        if fn.endswith(".json"):
            fp = os.path.join(task_dir, fn)
            if not os.path.isfile(fp):
                continue
            with open(fp, "r", encoding="utf-8") as f:
                tasks.append(Task(**json.load(f)))
    return tasks


# 进主榜的生成银行子目录（_bridge/ 仅用于 isomorph_gap，绝不进主评测；manifest.json 非任务）。
# u3 目前承载 Phase 2 真实媒介 pilot 骨架；正式难度筛选不会误把 pilot 当 medium/hard。
BANK_SUBDIRS = ("u1", "u2", "u3", "u4", "u5", "u6")


def load_task_bank(engine_root: str = None, include_top_level: bool = True,
                   include_generated: bool = True, difficulties=None,
                   templates=None, dimensions=None,
                   limit_per_template: int = None,
                   include_solvable: bool = True) -> List[Task]:
    """扩充任务银行加载器（A. 集成）：把 `tasks/generated/{u1,u2,u3,u4,u5,u6}/*.json` 并入评测。

    契约：
      - **排除 `_bridge/`**（仅供 isomorph_gap 污染度量，不进主榜）与 `manifest.json`（非任务）。
      - 默认并入 `tasks/` 顶层既有样例任务（含 U3 双轨 grounding + u1/u4），形成 U1–U6 全覆盖。
      - 可选过滤：difficulties / templates / dimensions；limit_per_template 控制每模板实例数（提速）。
      - 按 task_id 去重，稳定排序（顶层在前、再按维度子目录）。
    """
    if engine_root is None:
        engine_root = os.path.dirname(os.path.abspath(__file__))
    diffs = set(difficulties) if difficulties else None
    tmpls = set(templates) if templates else None
    dims = set(dimensions) if dimensions else None

    seen = set()
    out: List[Task] = []
    per_tmpl_count: Dict[str, int] = {}

    def _accept(t: Task) -> bool:
        if t.task_id in seen:
            return False
        knobs = t.difficulty_knobs if isinstance(t.difficulty_knobs, dict) else {}
        decl_diff = knobs.get("difficulty")
        # 难度过滤只作用于"声明了难度"的任务（生成银行）；顶层样例任务无难度旋钮，恒保留
        if diffs is not None and decl_diff is not None and decl_diff not in diffs:
            return False
        tmpl = knobs.get("template") or t.task_id
        if tmpls is not None and tmpl not in tmpls:
            return False
        if dims is not None and t.dimension not in dims:
            return False
        if limit_per_template is not None:
            if per_tmpl_count.get(tmpl, 0) >= limit_per_template:
                return False
            per_tmpl_count[tmpl] = per_tmpl_count.get(tmpl, 0) + 1
        seen.add(t.task_id)
        return True

    if include_top_level:
        for t in load_tasks(os.path.join(engine_root, "tasks")):
            if _accept(t):
                out.append(t)
    if include_solvable:
        # 自包含可解任务（Part 1）：源数据在 initial_state、gold 可由数据推导（dual-solvable）
        for t in load_tasks(os.path.join(engine_root, "tasks", "solvable")):
            if _accept(t):
                out.append(t)
    if include_generated:
        gen_root = os.path.join(engine_root, "tasks", "generated")
        for sub in BANK_SUBDIRS:
            for t in load_tasks(os.path.join(gen_root, sub)):
                if _accept(t):
                    out.append(t)
    return out


def run_model_on_task(adapter: ModelAdapter, task: Task, n_runs: int) -> List[Dict[str, Any]]:
    records = []
    for run_idx in range(n_runs):
        seed = 1000 + run_idx
        sub = adapter.submit(task, run_index=run_idx, seed=seed)
        sb = Sandbox(task)
        trace = sb.run(sub, model_id=adapter.model_id, run_index=run_idx, seed=seed)
        records.append(score_task(task, trace))
    return records


def evaluate(models: Dict[str, Any], tasks: List[Task], n_runs: int = 5,
             k: int = 5, dim_n_boot: int = 2000, glmm_n_boot: int = 2000,
             irt_gate: bool = False, rho_gate: float = 0.8) -> Dict[str, Any]:
    """对一组模型跑一组任务并聚合为完整报告。

    models: {model_id: profile_name}（内置 stub 策略），或 {model_id: adapter}
            （任意带 `.model_id` 与 `.submit(task, run_index, seed)` 的适配器，见 adapters/）。
    新增 *_n_boot / irt_gate / rho_gate 透传到 build_report（提速/接线统计主干）。
    """
    per_model_records: Dict[str, List[Dict[str, Any]]] = {}
    for mid, spec in models.items():
        adapter = spec if hasattr(spec, "submit") else ModelAdapter(mid, spec)
        recs: List[Dict[str, Any]] = []
        for task in tasks:
            recs.extend(run_model_on_task(adapter, task, n_runs))
        per_model_records[mid] = recs
    report = build_report(per_model_records, k=k, dim_n_boot=dim_n_boot,
                          glmm_n_boot=glmm_n_boot, irt_gate=irt_gate, rho_gate=rho_gate)
    report["raw_records"] = per_model_records
    return report
