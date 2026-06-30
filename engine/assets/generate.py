"""
确定性资产生成驱动 + 由符号级 GT 派生 U3 grounding 任务。

用法：
    cd engine && python -m assets.generate            # 生成资产 + GT + manifest + 任务
    python -m assets.generate --no-tasks              # 只生成资产，不写任务

产物：
    assets/generated/*.png|*.svg|*.html               # 真实可视资产
    assets/generated/*.gt.json                        # 每个资产的符号级 GT
    assets/generated/minimal_pairs.json               # 反事实最小对元数据
    assets/generated/manifest.json                    # 总清单（seed/资产/是否栅格化）
    tasks/ground_*.json                               # 引用上述资产的 grounding 任务

确定性：所有内容由 master seed 决定（默认 20240625）。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from assets.charts import make_chart_gt, render_chart
from assets.tables import make_table_gt, render_table_html, render_table_png
from assets.documents import make_doc_gt, render_doc_png
from assets.counterfactual import minimal_pair_chart, minimal_pair_doc, pair_gold
from assets import _render

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.dirname(_HERE)
GEN_DIR = os.path.join(_HERE, "generated")
TASKS_DIR = os.path.join(_ENGINE, "tasks")
MASTER_SEED = 20240625


def _rel(path: str) -> str:
    """相对 engine 根目录的 POSIX 路径（任务里引用资产用）。"""
    return os.path.relpath(path, _ENGINE).replace(os.sep, "/")


def _dump_json(path: str, obj: Any) -> None:
    _render.ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def generate_all(out_dir: str = GEN_DIR, seed: int = MASTER_SEED) -> Dict[str, Any]:
    """生成全部资产 + GT，返回供 build_tasks 使用的内存结果。"""
    res: Dict[str, Any] = {"seed": seed, "rendered": {}}

    # ---- 图表（柱 + 折线） ----
    chart = make_chart_gt(seed + 1, "bar", n=4, chart_id="chart_bar")
    res["rendered"]["chart_bar"] = render_chart(chart, os.path.join(out_dir, "chart_bar.png"))
    _dump_json(os.path.join(out_dir, "chart_bar.gt.json"), chart.to_dict())

    chart_line = make_chart_gt(seed + 2, "line", n=5, chart_id="chart_line")
    res["rendered"]["chart_line"] = render_chart(chart_line, os.path.join(out_dir, "chart_line.png"))
    _dump_json(os.path.join(out_dir, "chart_line.gt.json"), chart_line.to_dict())

    # 反事实最小对（图表）
    chart_cf, c_cap0, c_cap1, c_change = minimal_pair_chart(chart, seed + 3)
    res["rendered"]["chart_bar_cf"] = render_chart(chart_cf, os.path.join(out_dir, "chart_bar_cf.png"))
    _dump_json(os.path.join(out_dir, "chart_bar_cf.gt.json"), chart_cf.to_dict())

    # ---- 表格 ----
    table = make_table_gt(seed + 4, n_metric_rows=4, table_id="table_fin")
    render_table_html(table, os.path.join(out_dir, "table_fin.html"))
    res["rendered"]["table_fin"] = render_table_png(table, os.path.join(out_dir, "table_fin.png"))
    _dump_json(os.path.join(out_dir, "table_fin.gt.json"), table.to_dict())

    # ---- 合成文档 ----
    doc = make_doc_gt(seed + 5, doc_id="doc_receipt")
    res["rendered"]["doc_receipt"] = render_doc_png(doc, os.path.join(out_dir, "doc_receipt.png"))
    _dump_json(os.path.join(out_dir, "doc_receipt.gt.json"), doc.to_dict())

    # 反事实最小对（文档）
    doc_cf, d_cap0, d_cap1, d_change = minimal_pair_doc(doc, seed + 6)
    res["rendered"]["doc_receipt_cf"] = render_doc_png(doc_cf, os.path.join(out_dir, "doc_receipt_cf.png"))
    _dump_json(os.path.join(out_dir, "doc_receipt_cf.gt.json"), doc_cf.to_dict())

    # ---- 最小对元数据 ----
    pairs = {
        "chart": pair_gold("chart_topbar", c_cap0, c_cap1,
                           _rel(os.path.join(out_dir, "chart_bar.png")),
                           _rel(os.path.join(out_dir, "chart_bar_cf.png")), c_change),
        "doc": pair_gold("doc_total", d_cap0, d_cap1,
                         _rel(os.path.join(out_dir, "doc_receipt.png")),
                         _rel(os.path.join(out_dir, "doc_receipt_cf.png")), d_change),
    }
    _dump_json(os.path.join(out_dir, "minimal_pairs.json"), pairs)

    # ---- manifest ----
    manifest = {
        "seed": seed,
        "backends": {"matplotlib": _render.HAS_MPL, "pillow": _render.HAS_PIL},
        "rendered": res["rendered"],
        "assets": {
            "chart_bar": {"png": "chart_bar.png", "gt": "chart_bar.gt.json"},
            "chart_line": {"png": "chart_line.png", "gt": "chart_line.gt.json"},
            "chart_bar_cf": {"png": "chart_bar_cf.png", "gt": "chart_bar_cf.gt.json"},
            "table_fin": {"png": "table_fin.png", "html": "table_fin.html", "gt": "table_fin.gt.json"},
            "doc_receipt": {"png": "doc_receipt.png", "gt": "doc_receipt.gt.json"},
            "doc_receipt_cf": {"png": "doc_receipt_cf.png", "gt": "doc_receipt_cf.gt.json"},
            "minimal_pairs": "minimal_pairs.json",
        },
    }
    _dump_json(os.path.join(out_dir, "manifest.json"), manifest)

    res.update({"chart": chart, "chart_line": chart_line, "chart_cf": chart_cf,
                "table": table, "doc": doc, "doc_cf": doc_cf, "pairs": pairs,
                "manifest": manifest})
    return res


# --------------------------------------------------------------------------- #
# 由 GT 派生任务（golds 与资产严格一致）
# --------------------------------------------------------------------------- #
def _ocr_calibration(samples, verifier: str, threshold: float = 0.95) -> Dict[str, Any]:
    return {"verifier": verifier, "metric": "ocr_cer_acc", "threshold": threshold,
            "samples": [{"gold": s} for s in samples]}


def _task_chart(res: Dict[str, Any]) -> Dict[str, Any]:
    c = res["chart"]
    cat = c.categories[1]                       # 取第二个季度做 numeric/iou 目标
    pair = res["pairs"]["chart"]
    return {
        "task_id": "ground_chart_revenue",
        "version": "1.0.0",
        "dimension": "U3",
        "capability_load": {"U3": 1.0, "U2": 0.3},
        "title": "图表读数 grounding（柱状图：数值/极值/定位/反事实最小对 + 真实 OCR 轨）",
        "instruction": "阅读柱状图，报告最高季度与其数值；并完成数值/定位/反事实最小对的 grounding。",
        "modalities": ["image", "text"],
        "assets": {"chart": pair["image0"], "chart_cf": pair["image1"]},
        "tools": [
            {"name": "read_chart", "writes": [], "effect": None},
            {"name": "submit_finding", "writes": ["out.finding"],
             "effect": {"type": "set", "target": "out.finding", "value_from": "finding"}},
        ],
        "initial_state": {},
        "milestones": [
            {"id": "M1", "type": "required", "weight": 1.0, "epistemic_action": "read_chart",
             "predicate": {"op": "tool_called", "value": "read_chart"}},
            {"id": "M2", "type": "required", "weight": 2.0, "deps": ["M1"],
             "provenance": ["out.finding"],
             "predicate": {"op": "eq", "path": "out.finding.top", "value": c.argmax_category}},
            {"id": "M3", "type": "required", "weight": 2.0, "deps": ["M2"],
             "provenance": ["out.finding"],
             "predicate": {"op": "approx", "path": "out.finding.value",
                           "value": c.max_value, "tol": 0.05}},
        ],
        "success_predicates": [
            {"op": "eq", "path": "out.finding.top", "value": c.argmax_category},
            {"op": "approx", "path": "out.finding.value", "value": c.max_value, "tol": 0.05},
        ],
        "grounding": {"items": [
            {"id": "g_bar_value", "kind": "numeric", "track": "synthetic",
             "gold": c.value_of(cat), "tol": 0.05, "weight": 1.0,
             "note": "图表数值容差 5%%（%s）" % cat},
            {"id": "g_top_quarter", "kind": "closed_id", "track": "synthetic",
             "gold": [c.argmax_category], "weight": 1.0, "note": "极值类别（闭式 ID 精确匹配）"},
            {"id": "g_bar_box", "kind": "iou", "track": "synthetic",
             "gold": c.bars_bbox.get(cat), "tol": 0.5, "weight": 1.0,
             "note": "该柱 bbox（IoU>=0.5）"},
            {"id": "g_minimal_pair", "kind": "minimal_pair", "track": "synthetic",
             "gold": pair, "weight": 1.0, "note": "反事实最小对 group-score（极值翻转）"},
            {"id": "g_real_title_ocr", "kind": "cer", "track": "real",
             "gold": c.title, "weight": 1.0,
             "calibration": _ocr_calibration([c.title, c.x_label, c.y_label], "ocr_extractor_hi"),
             "note": "真实生态轨：标题 OCR；ML 抽取器须先过标定门(>=0.95)"},
        ]},
        "oracle_plan": [
            {"action_id": "a0", "tool": "read_chart", "args": {}},
            {"action_id": "a1", "tool": "submit_finding",
             "args": {"finding": {"top": c.argmax_category, "value": c.max_value}}},
        ],
        "c_star": 2,
        "difficulty_knobs": {"modalities": 2, "evidence_conflict": False},
        "canary": "AGENIX-CANARY-ground-chart-a1b2-DO-NOT-TRAIN",
    }


def _task_table(res: Dict[str, Any]) -> Dict[str, Any]:
    t = res["table"]
    cell_text = t.cell(1, 1)                    # 第一个指标 × 第一年
    cell_val = float(cell_text)
    return {
        "task_id": "ground_table_financials",
        "version": "1.0.0",
        "dimension": "U3",
        "capability_load": {"U3": 1.0},
        "title": "表格读数 grounding（财务表：表头集合/单元格数值/定位 + 真实 OCR 轨）",
        "instruction": ("阅读财务表图片，报告指标 '%s' 在 '%s' 列的数值（第一指标行 × 第一数据年列）；"
                        "调用 submit_finding，参数 finding={\"cell\": <数字>}。并完成表头/定位/OCR 的 grounding。"
                        % (t.cell(1, 0), t.headers[1])),
        "modalities": ["image", "text"],
        "assets": {"table_png": _rel(os.path.join(GEN_DIR, "table_fin.png")),
                   "table_html": _rel(os.path.join(GEN_DIR, "table_fin.html"))},
        "tools": [
            {"name": "read_table", "writes": [], "effect": None},
            {"name": "submit_finding", "writes": ["out.finding"],
             "effect": {"type": "set", "target": "out.finding", "value_from": "finding"}},
        ],
        "initial_state": {},
        "milestones": [
            {"id": "M1", "type": "required", "weight": 1.0, "epistemic_action": "read_table",
             "predicate": {"op": "tool_called", "value": "read_table"}},
            {"id": "M2", "type": "required", "weight": 2.0, "deps": ["M1"],
             "provenance": ["out.finding"],
             "predicate": {"op": "approx", "path": "out.finding.cell", "value": cell_val, "tol": 0.005}},
        ],
        "success_predicates": [
            {"op": "approx", "path": "out.finding.cell", "value": cell_val, "tol": 0.005},
        ],
        "grounding": {"items": [
            {"id": "g_headers", "kind": "closed_id", "track": "synthetic",
             "gold": list(t.headers), "weight": 1.0, "note": "列表头集合（闭式 ID）"},
            {"id": "g_cell_value", "kind": "numeric", "track": "synthetic",
             "gold": cell_val, "tol": 0.005, "weight": 1.0, "note": "财务数值容差 0.5%"},
            {"id": "g_cell_box", "kind": "iou", "track": "synthetic",
             "gold": t.cells_bbox.get("1,1"), "tol": 0.5, "weight": 1.0, "note": "单元格 bbox"},
            {"id": "g_real_cell_ocr", "kind": "cer", "track": "real",
             "gold": cell_text, "weight": 1.0,
             "calibration": _ocr_calibration([cell_text, t.headers[1], t.cell(2, 1)], "ocr_extractor_hi"),
             "note": "真实生态轨：单元格 OCR；ML 抽取器须先过标定门"},
        ]},
        "oracle_plan": [
            {"action_id": "a0", "tool": "read_table", "args": {}},
            {"action_id": "a1", "tool": "submit_finding", "args": {"finding": {"cell": cell_val}}},
        ],
        "c_star": 2,
        "difficulty_knobs": {"modalities": 2},
        "canary": "AGENIX-CANARY-ground-table-c3d4-DO-NOT-TRAIN",
    }


def _task_doc(res: Dict[str, Any]) -> Dict[str, Any]:
    d = res["doc"]
    pair = res["pairs"]["doc"]
    total = float(d.field_text("total"))
    return {
        "task_id": "ground_doc_receipt",
        "version": "1.0.0",
        "dimension": "U3",
        "capability_load": {"U3": 1.0},
        "title": "文档读数 grounding（收据截图：金额/单号/定位/反事实最小对 + 真实 OCR 轨）",
        "instruction": "阅读收据截图，报告 TOTAL 金额；并完成单号/定位/反事实最小对/OCR 的 grounding。",
        "modalities": ["image", "text"],
        "assets": {"doc": pair["image0"], "doc_cf": pair["image1"]},
        "tools": [
            {"name": "read_doc", "writes": [], "effect": None},
            {"name": "submit_finding", "writes": ["out.finding"],
             "effect": {"type": "set", "target": "out.finding", "value_from": "finding"}},
        ],
        "initial_state": {},
        "milestones": [
            {"id": "M1", "type": "required", "weight": 1.0, "epistemic_action": "read_doc",
             "predicate": {"op": "tool_called", "value": "read_doc"}},
            {"id": "M2", "type": "required", "weight": 2.0, "deps": ["M1"],
             "provenance": ["out.finding"],
             "predicate": {"op": "approx", "path": "out.finding.total", "value": total, "tol": 0.005}},
        ],
        "success_predicates": [
            {"op": "approx", "path": "out.finding.total", "value": total, "tol": 0.005},
        ],
        "grounding": {"items": [
            {"id": "g_total", "kind": "numeric", "track": "synthetic",
             "gold": total, "tol": 0.005, "weight": 1.0, "note": "金额数值容差 0.5%"},
            {"id": "g_invoice_id", "kind": "closed_id", "track": "synthetic",
             "gold": [d.field_text("invoice_no")], "weight": 1.0, "note": "发票单号（闭式 ID）"},
            {"id": "g_total_box", "kind": "iou", "track": "synthetic",
             "gold": d.field_bbox("total"), "tol": 0.5, "weight": 1.0, "note": "TOTAL 字段 bbox"},
            {"id": "g_minimal_pair", "kind": "minimal_pair", "track": "synthetic",
             "gold": pair, "weight": 1.0, "note": "反事实最小对 group-score（金额改一处）"},
            {"id": "g_real_total_ocr", "kind": "cer", "track": "real",
             "gold": d.field_text("total"), "weight": 1.0,
             "calibration": _ocr_calibration([d.field_text("total"), d.field_text("invoice_no"),
                                              d.field_text("date")], "ocr_extractor_hi"),
             "note": "真实生态轨：TOTAL OCR；ML 抽取器须先过标定门"},
        ]},
        "oracle_plan": [
            {"action_id": "a0", "tool": "read_doc", "args": {}},
            {"action_id": "a1", "tool": "submit_finding", "args": {"finding": {"total": total}}},
        ],
        "c_star": 2,
        "difficulty_knobs": {"modalities": 2},
        "canary": "AGENIX-CANARY-ground-doc-e5f6-DO-NOT-TRAIN",
    }


def _task_structured(res: Dict[str, Any]) -> Dict[str, Any]:
    """结构化读取探针：演示 table_teds + ocr_bbox 新验证器 + 未标定(untrusted)真实轨。

    注意：内置只读 mock（models.py，禁改）只会回答 closed_id/numeric/iou/cer/minimal_pair，
    对 table_teds / ocr_bbox 返回 None -> 评 0（这是正确的"无答案不得分"行为）。这两类
    新验证器的完整正确性由 tests/test_grounding.py 用真实答案直接覆盖。本任务的 numeric
    项可被 mock 回答，故仍能"跑通"且产出非零分。
    """
    t = res["table"]
    d = res["doc"]
    cell_val = float(t.cell(1, 1))
    return {
        "task_id": "ground_structured_reading",
        "version": "1.0.0",
        "dimension": "U3",
        "capability_load": {"U3": 1.0},
        "title": "结构化读取探针（TEDS 表格结构 + OCR token bbox + 未标定真实轨示例）",
        "instruction": ("阅读**财务表图片**，报告指标 '%s' 在 '%s' 列的数值（来自财务表，不是收据）；"
                        "调用 submit_finding，参数 finding={\"cell\": <数字>}。"
                        "（本任务同时演示 table_teds/ocr_bbox 新验证器与未标定真实轨。）"
                        % (t.cell(1, 0), t.headers[1])),
        "modalities": ["image", "text"],
        "assets": {"table_png": _rel(os.path.join(GEN_DIR, "table_fin.png")),
                   "doc": _rel(os.path.join(GEN_DIR, "doc_receipt.png"))},
        "tools": [
            {"name": "read_table", "writes": [], "effect": None},
            {"name": "read_doc", "writes": [], "effect": None},
            {"name": "submit_finding", "writes": ["out.finding"],
             "effect": {"type": "set", "target": "out.finding", "value_from": "finding"}},
        ],
        "initial_state": {},
        "milestones": [
            {"id": "M1", "type": "required", "weight": 1.0, "epistemic_action": "read_table",
             "predicate": {"op": "tool_called", "value": "read_table"}},
            {"id": "M2", "type": "required", "weight": 1.0, "deps": ["M1"],
             "provenance": ["out.finding"],
             "predicate": {"op": "approx", "path": "out.finding.cell", "value": cell_val, "tol": 0.005}},
        ],
        "success_predicates": [
            {"op": "approx", "path": "out.finding.cell", "value": cell_val, "tol": 0.005},
        ],
        "grounding": {"items": [
            {"id": "g_table_teds", "kind": "table_teds", "track": "synthetic",
             "gold": t.table_repr(), "weight": 1.0,
             "note": "TEDS 式表格结构+内容相似（mock 不答->0；见 test_grounding 直测）"},
            {"id": "g_ocr_tokens", "kind": "ocr_bbox", "track": "synthetic",
             "gold": d.tokens, "tol": 0.5, "cer_tol": 0.3, "weight": 1.0,
             "note": "OCR token 必须文本(CER<=0.3)与 bbox(IoU>=0.5)同时对"},
            {"id": "g_cell_value", "kind": "numeric", "track": "synthetic",
             "gold": cell_val, "tol": 0.005, "weight": 1.0, "note": "mock 可答，保证跑通"},
            {"id": "g_real_untrusted", "kind": "cer", "track": "real",
             "gold": t.cell(1, 1), "weight": 1.0,
             "calibration": _ocr_calibration([t.cell(1, 1), t.headers[1], t.cell(2, 1)], "ocr_extractor_lo"),
             "note": "真实轨用低保真 ML 抽取器(ocr_extractor_lo)->未达 0.95->untrusted 仅诊断"},
        ]},
        "oracle_plan": [
            {"action_id": "a0", "tool": "read_table", "args": {}},
            {"action_id": "a1", "tool": "read_doc", "args": {}},
            {"action_id": "a2", "tool": "submit_finding", "args": {"finding": {"cell": cell_val}}},
        ],
        "c_star": 3,
        "difficulty_knobs": {"modalities": 2, "structured_reading": True},
        "canary": "AGENIX-CANARY-ground-struct-7g8h-DO-NOT-TRAIN",
    }


def build_tasks(res: Dict[str, Any], tasks_dir: str = TASKS_DIR) -> Dict[str, str]:
    """把派生任务写入 tasks/（仅 ground_*.json，绝不触碰既有 u1/u3/u4）。"""
    builders = [_task_chart, _task_table, _task_doc, _task_structured]
    written: Dict[str, str] = {}
    for b in builders:
        task = b(res)
        path = os.path.join(tasks_dir, task["task_id"] + ".json")
        _dump_json(path, task)
        written[task["task_id"]] = path
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="AGENIX U3 资产 + 任务生成器")
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    ap.add_argument("--out", default=GEN_DIR)
    ap.add_argument("--no-tasks", action="store_true", help="只生成资产，不写任务")
    args = ap.parse_args()

    res = generate_all(args.out, args.seed)
    print("[assets] backends:", res["manifest"]["backends"])
    print("[assets] rendered:", res["rendered"])
    print("[assets] 输出目录:", args.out)
    if not args.no_tasks:
        written = build_tasks(res)
        print("[tasks] 已写入:")
        for tid, p in written.items():
            print("   %-26s %s" % (tid, p))


if __name__ == "__main__":
    main()
