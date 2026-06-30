"""
生成任务银行：把程序化生成结果物化到 engine/tasks/generated/<dim>/ 下。

布局：
  tasks/generated/<u1|u2|u4|u5|u6>/<template>__<difficulty>__s<seed>.json   # 主银行（锚定分布）
  tasks/generated/_bridge/<template>__<difficulty>__s<seed>.json            # 新种子同构桥梁集
  tasks/generated/manifest.json                                              # 计数 + 锚定摘要 + canary 索引

不会写入 tasks/ 顶层（避免被 orchestrator.load_tasks / run_demo / test_meta 自动加载）。
运行：  cd engine && python -m generators.build_bank   （或 python generators/build_bank.py）
"""
from __future__ import annotations

import json
import os
import sys

# 允许以脚本方式运行：把 engine 根目录加入 sys.path
_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

import generators as G  # noqa: E402
from generators.contamination import (  # noqa: E402
    AnchorSpec, anchor_distribution_summary, isomorph_bridge_set,
)

GENERATED_DIR = os.path.join(_ENGINE_ROOT, "tasks", "generated")

# 锚定：每个 (模板 × 难度) cell 生成 N_PER_CELL 个实例（同 cell 不同种子 = 同构变体）
N_PER_CELL = 2
BRIDGE_PER_TEMPLATE = 2
BRIDGE_DIFFICULTY = "medium"


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build() -> dict:
    template_ids = G.list_template_ids()
    spec = AnchorSpec(template_ids=template_ids, difficulties=list(G.DIFFICULTIES),
                      instances_per=N_PER_CELL, base_seed=0)

    counts_by_dim: dict = {}
    counts_by_template: dict = {}
    canary_index: dict = {}
    files_written = []

    # ---- 主银行（按锚定分布） ---- #
    for tid in template_ids:
        tmpl = G.get_template(tid)
        dim = tmpl.dimension
        for diff in G.DIFFICULTIES:
            for i in range(N_PER_CELL):
                d = G.build_task_dict(tid, seed=i, difficulty=diff)
                # 双保险：通过 schema 校验后再落盘
                G.build_task(tid, seed=i, difficulty=diff)
                rel = os.path.join(dim.lower(), d["task_id"] + ".json")
                path = os.path.join(GENERATED_DIR, rel)
                _write_json(path, d)
                files_written.append(rel)
                counts_by_dim[dim] = counts_by_dim.get(dim, 0) + 1
                counts_by_template[tid] = counts_by_template.get(tid, 0) + 1
                canary_index[d["task_id"]] = d["canary"]

    # ---- 新种子同构桥梁集 ---- #
    bridge_count = 0
    for tid in template_ids:
        bridge = isomorph_bridge_set(tid, difficulty=BRIDGE_DIFFICULTY,
                                     n=BRIDGE_PER_TEMPLATE)
        for t in bridge:
            d = t.model_dump(mode="json")
            rel = os.path.join("_bridge", d["task_id"] + ".json")
            _write_json(os.path.join(GENERATED_DIR, rel), d)
            files_written.append(rel)
            canary_index[d["task_id"]] = d["canary"]
            bridge_count += 1

    manifest = {
        "generator": "AGENIX generators",
        "dimensions_covered": ["U1", "U2", "U4", "U5", "U6"],
        "n_templates": len(template_ids),
        "templates": G.list_templates(),
        "n_per_cell": N_PER_CELL,
        "difficulties": list(G.DIFFICULTIES),
        "total_main_tasks": sum(counts_by_dim.values()),
        "total_bridge_tasks": bridge_count,
        "counts_by_dimension": counts_by_dim,
        "counts_by_template": counts_by_template,
        "anchor_distribution": anchor_distribution_summary(spec),
        "canary_index": canary_index,
    }
    _write_json(os.path.join(GENERATED_DIR, "manifest.json"), manifest)
    return manifest


def main() -> None:
    m = build()
    print("已生成任务银行 ->", GENERATED_DIR)
    print("模板数:", m["n_templates"], " 主银行任务:", m["total_main_tasks"],
          " 桥梁集:", m["total_bridge_tasks"])
    print("按维度计数:", m["counts_by_dimension"])


if __name__ == "__main__":
    main()
