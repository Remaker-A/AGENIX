"""
U3 双轨多模态 grounding 真实实现的测试（spec §4.4 / CP4）。

覆盖任务要求：
  ① 反事实最小对 group-score：仅图+文都正确为 1，单侧对→0；
  ② IoU / CER / 数值容差 / 闭式 ID 匹配在正/负例上行为正确；
  ③ "蒙对/乱猜/语言先验"拿不到分（抗 gaming）；
  ④ ML 验证器可靠性标定门：达阈(≥0.95)→进 headline，未达→untrusted 仅诊断；
  ⑤ 合成-真实 Spearman ρ 数据门按相关性正确切 headline 规则；
并附：新结构化验证器（table_teds / ocr_bbox）正负例、向后兼容契约、资产 GT 确定性、
以及"新任务可被 orchestrator 用 mock 策略端到端跑通"。

运行：python -m pytest tests/test_grounding.py -q  或  python tests/test_grounding.py
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scoring.grounding as G                                  # noqa: E402
from scoring.grounding import (                                # noqa: E402
    score_grounding, score_item, table_teds, ocr_bbox_score,
    minimal_pair_group_score, calibrate_verifier, run_ml_verifier, numeric_relaxed,
    f1_set, precision_recall_f1_set, iou_xywh, cer,
    temporal_iou, video_event_id_score, doc_layout_f1,
    synthetic_real_spearman, grounding_headline_rule)
from schema import Task, GroundingItem, GroundingSpec          # noqa: E402
from orchestrator import load_tasks, load_task_bank, evaluate  # noqa: E402
from assets.charts import make_chart_gt                        # noqa: E402
from assets.documents import make_doc_gt, render_doc_png       # noqa: E402
from assets import _render                                     # noqa: E402
from assets import pixel_ocr as PX                             # noqa: E402

_REAL_BACKENDS = ("pil_template_match", "pytesseract")

_ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TASK_DIR = os.path.join(_ENGINE, "tasks")


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


def _mk_task(items):
    return Task(task_id="t_inline", dimension="U3",
                grounding=GroundingSpec(items=[GroundingItem(**it) for it in items]))


# 一组复用的 OCR 标定样本
_OCR_SAMPLES = [{"gold": "REVENUE"}, {"gold": "FY2024"}, {"gold": "Net Income 1234"}]
_DET_BOXES = [[10, 10, 40, 20], [80, 10, 40, 20], [150, 10, 40, 20], [10, 60, 40, 20],
              [80, 60, 40, 20], [150, 60, 40, 20], [10, 110, 40, 20], [80, 110, 40, 20]]
_DET_SAMPLES = [{"gold_boxes": _DET_BOXES},
                {"gold_boxes": [[x + 5, y + 5, w, h] for x, y, w, h in _DET_BOXES]},
                {"gold_boxes": [[x + 9, y + 9, w, h] for x, y, w, h in _DET_BOXES]}]


# --------------------------------------------------------------------------- #
# ① 反事实最小对 group-score
# --------------------------------------------------------------------------- #
def test_minimal_pair_group_only_when_both_sides_correct():
    both = {"i0c0": 1.0, "i0c1": 0.0, "i1c0": 0.0, "i1c1": 1.0}
    text_only = {"i0c0": 0.5, "i0c1": 0.1, "i1c0": 0.6, "i1c1": 0.7}   # 行对、列错
    image_only = {"i0c0": 0.6, "i0c1": 0.7, "i1c0": 0.1, "i1c1": 0.8}  # 列对、行错
    neither = {"i0c0": 0.0, "i0c1": 1.0, "i1c0": 1.0, "i1c1": 0.0}

    assert minimal_pair_group_score(both) == {"text": 1.0, "image": 1.0, "group": 1.0}
    r = minimal_pair_group_score(text_only)
    assert r["text"] == 1.0 and r["image"] == 0.0 and r["group"] == 0.0
    r = minimal_pair_group_score(image_only)
    assert r["image"] == 1.0 and r["text"] == 0.0 and r["group"] == 0.0
    assert minimal_pair_group_score(neither)["group"] == 0.0

    # 经 score_item（kind=minimal_pair）一致
    item = {"id": "mp", "kind": "minimal_pair", "track": "synthetic", "gold": {}}
    assert score_item(GroundingItem(**item), {"mp": {"scores": both}}) == 1.0
    assert score_item(GroundingItem(**item), {"mp": {"scores": text_only}}) == 0.0
    assert score_item(GroundingItem(**item), {"mp": {"scores": image_only}}) == 0.0


# --------------------------------------------------------------------------- #
# ② IoU / CER / 数值容差 / 闭式 ID
# --------------------------------------------------------------------------- #
def test_closed_id_exact_match():
    gold = ["INV-002", "INV-005"]
    assert _approx(f1_set(["INV-005", "INV-002"], gold), 1.0)         # 顺序无关
    assert f1_set([], gold) == 0.0                                    # 漏报
    partial = f1_set(["INV-002"], gold)                               # 半对
    assert 0.6 < partial < 0.7
    prec, rec, f1 = precision_recall_f1_set(["INV-002"], gold)
    assert _approx(prec, 1.0) and _approx(rec, 0.5)


def test_numeric_relaxed_tolerance():
    # 财务 0.5%
    assert numeric_relaxed(14.0, 14.0, 0.005) == 1.0
    assert numeric_relaxed(14.05, 14.0, 0.005) == 1.0                 # 0.357% 内
    assert numeric_relaxed(13.0, 14.0, 0.005) == 0.0                  # 7.1% 超
    # 图表 5%
    assert numeric_relaxed(12.0, 12.5, 0.05) == 1.0                   # 4% 内
    assert numeric_relaxed(11.0, 12.5, 0.05) == 0.0                   # 12% 超
    assert numeric_relaxed(None, 12.5, 0.05) == 0.0                   # 未作答


def test_iou_threshold():
    box = [100, 100, 50, 40]
    assert iou_xywh(box, box) == 1.0
    assert iou_xywh([105, 103, 50, 40], box) >= 0.5                   # 小偏移仍 IoU≥0.5
    assert iou_xywh([400, 400, 50, 40], box) == 0.0                   # 不相交
    item = GroundingItem(**{"id": "b", "kind": "iou", "track": "synthetic",
                            "gold": box, "tol": 0.5})
    assert score_item(item, {"b": list(box)}) == 1.0
    assert score_item(item, {"b": [400, 400, 50, 40]}) == 0.0
    assert score_item(item, {"b": None}) == 0.0


def test_cer_ocr():
    assert _approx(cer("Q3 Net 14.0M", "Q3 Net 14.0M"), 0.0)
    assert _approx(cer("Q3 Net 13.0M", "Q3 Net 14.0M"), 1.0 / 12.0)   # 1/12 字符错
    item = GroundingItem(**{"id": "o", "kind": "cer", "track": "real", "gold": "Q3 Net 14.0M"})
    assert score_item(item, {"o": "Q3 Net 14.0M"}) == 1.0
    assert score_item(item, {"o": "Q3 Net 13.0M"}) > 0.9
    assert score_item(item, {"o": "totally wrong"}) < 0.5
    assert score_item(item, {"o": None}) == 0.0


# --------------------------------------------------------------------------- #
# 新结构化验证器：table_teds / ocr_bbox
# --------------------------------------------------------------------------- #
def test_table_teds_structure_and_content():
    gold = {"grid": [["Metric", "FY22", "FY23"], ["Revenue", "100.0", "110.0"],
                     ["COGS", "40.0", "44.0"]]}
    assert _approx(table_teds(gold, gold), 1.0)                       # 完全一致
    one_cell = {"grid": [["Metric", "FY22", "FY23"], ["Revenue", "100.5", "110.0"],
                         ["COGS", "40.0", "44.0"]]}
    s1 = table_teds(one_cell, gold)
    assert 0.9 < s1 < 1.0                                             # 仅一格不同
    missing_row = {"grid": [["Metric", "FY22", "FY23"], ["Revenue", "100.0", "110.0"]]}
    s2 = table_teds(missing_row, gold)
    assert s2 < s1                                                    # 缺一行更低
    assert table_teds({"grid": []}, gold) < 0.3                       # 空表近 0
    # score_item：未作答(None) -> 0
    item = GroundingItem(**{"id": "tt", "kind": "table_teds", "track": "synthetic", "gold": gold})
    assert score_item(item, {"tt": gold}) == 1.0
    assert score_item(item, {"tt": None}) == 0.0


def test_ocr_bbox_requires_text_and_box():
    gold = [{"text": "TOTAL", "bbox": [10, 10, 50, 12]},
            {"text": "407.04", "bbox": [80, 10, 40, 12]}]
    assert ocr_bbox_score(gold, gold) == 1.0
    wrong_box = [{"text": "TOTAL", "bbox": [300, 300, 50, 12]},
                 {"text": "407.04", "bbox": [400, 300, 40, 12]}]
    assert ocr_bbox_score(wrong_box, gold) == 0.0                     # 文对框错
    wrong_text = [{"text": "XXXXX", "bbox": [10, 10, 50, 12]},
                  {"text": "999.99", "bbox": [80, 10, 40, 12]}]
    assert ocr_bbox_score(wrong_text, gold) == 0.0                    # 框对文错
    half = [{"text": "TOTAL", "bbox": [10, 10, 50, 12]},
            {"text": "999.99", "bbox": [80, 10, 40, 12]}]
    assert 0.0 < ocr_bbox_score(half, gold) < 1.0                     # 半对
    item = GroundingItem(**{"id": "ob", "kind": "ocr_bbox", "track": "synthetic",
                            "gold": gold, "tol": 0.5, "cer_tol": 0.3})
    assert score_item(item, {"ob": gold}) == 1.0
    assert score_item(item, {"ob": None}) == 0.0


def test_temporal_video_and_layout_verifiers():
    # temporal_iou：重叠段占并集，score_item 按 tol 阈值化
    assert _approx(temporal_iou([3.0, 7.0], [3.0, 7.0]), 1.0)
    assert 0.0 < temporal_iou({"start_s": 3.0, "end_s": 5.0}, {"span": [4.0, 7.0]}) < 1.0
    item_t = GroundingItem(**{"id": "span", "kind": "temporal_iou", "track": "real",
                              "gold": [3.2, 6.8], "tol": 0.5})
    assert score_item(item_t, {"span": [3.0, 6.9]}) == 1.0
    assert score_item(item_t, {"span": [8.0, 9.0]}) == 0.0

    # video_event_id：闭式事件 ID F1，支持 events dict/list
    gold_events = {"events": [{"event_id": "forklift_enters_zone"},
                              {"event_id": "operator_warns"}]}
    assert video_event_id_score(["forklift_enters_zone", "operator_warns"], gold_events) == 1.0
    partial = video_event_id_score({"event_id": "forklift_enters_zone"}, gold_events)
    assert 0.0 < partial < 1.0

    # doc_layout_f1：label + bbox + 可选文本同时命中
    gold_layout = [{"label": "title", "bbox": [10, 10, 80, 20], "text": "RECEIPT"},
                   {"label": "total", "bbox": [110, 80, 60, 18], "text": "407.04"}]
    assert doc_layout_f1(gold_layout, gold_layout) == 1.0
    wrong_label = [{"label": "body", "bbox": [10, 10, 80, 20], "text": "RECEIPT"}]
    assert doc_layout_f1(wrong_label, gold_layout) == 0.0
    item_l = GroundingItem(**{"id": "layout", "kind": "doc_layout_f1", "track": "real",
                              "gold": gold_layout, "tol": 0.5, "cer_tol": 0.3})
    assert score_item(item_l, {"layout": gold_layout}) == 1.0


# --------------------------------------------------------------------------- #
# ③ 蒙对 / 乱猜 / 语言先验拿不到分
# --------------------------------------------------------------------------- #
def test_guessing_and_priors_get_no_credit():
    # 最小对：任何"忽略图像、只按语言先验偏好某 caption"的策略都拿不到 group 分
    prefer_c0 = {"i0c0": 0.9, "i0c1": 0.1, "i1c0": 0.9, "i1c1": 0.1}
    prefer_c1 = {"i0c0": 0.1, "i0c1": 0.9, "i1c0": 0.1, "i1c1": 0.9}
    all_tie = {"i0c0": 0.5, "i0c1": 0.5, "i1c0": 0.5, "i1c1": 0.5}
    for s in (prefer_c0, prefer_c1, all_tie):
        assert minimal_pair_group_score(s)["group"] == 0.0

    # 闭式 ID：霰弹式全报（提高召回但牺牲精确）被 F1 惩罚
    shotgun = f1_set(["A", "B", "C", "D", "E"], ["A"])
    assert shotgun < 0.5
    # 数值/IoU：明显乱猜 -> 0
    assert numeric_relaxed(0.0, 14.0, 0.005) == 0.0
    assert iou_xywh([0, 0, 5, 5], [200, 200, 50, 50]) == 0.0
    # 空作答 -> 全 0（无答案不得分）
    item_id = GroundingItem(**{"id": "x", "kind": "closed_id", "track": "synthetic", "gold": ["A"]})
    assert score_item(item_id, {}) == 0.0


# --------------------------------------------------------------------------- #
# ④ ML 验证器可靠性标定门
# --------------------------------------------------------------------------- #
def test_calibrate_verifier_metric_and_threshold():
    hi = calibrate_verifier({"verifier": "ocr_extractor_hi", "metric": "ocr_cer_acc",
                             "threshold": 0.95, "samples": _OCR_SAMPLES})
    lo = calibrate_verifier({"verifier": "ocr_extractor_lo", "metric": "ocr_cer_acc",
                             "threshold": 0.95, "samples": _OCR_SAMPLES})
    assert hi["value"] >= 0.95 and hi["passed"] is True
    assert lo["value"] < 0.95 and lo["passed"] is False

    dhi = calibrate_verifier({"verifier": "detector_hi", "metric": "detection_f1",
                              "threshold": 0.95, "samples": _DET_SAMPLES})
    dlo = calibrate_verifier({"verifier": "detector_lo", "metric": "detection_f1",
                              "threshold": 0.95, "samples": _DET_SAMPLES})
    assert dhi["value"] >= 0.95 and dhi["passed"] is True
    assert dlo["value"] < 0.95 and dlo["passed"] is False


# --------------------------------------------------------------------------- #
# ④' 真实**像素级** ML 验证器（不再是注入 error_rate 的桩；真的从像素得结果）
# --------------------------------------------------------------------------- #
def test_ml_verifier_is_real_pixel_reader_not_stub():
    # 后端为真实读图器（PIL 模板匹配或 tesseract），而非确定性扰动桩
    assert PX.backend_name() in _REAL_BACKENDS
    assert calibrate_verifier({"verifier": "ocr_extractor_hi", "metric": "ocr_cer_acc",
                               "samples": _OCR_SAMPLES})["backend"] in _REAL_BACKENDS

    # 把字符串渲染成像素再读回 —— 结果**仅由像素决定**（识别器看不到任何 gold）
    for s in ["REVENUE", "FY2024", "407.04", "INV-72042", "Net Income 1234"]:
        img = PX.render_text_image(s, size=22)
        assert PX.ocr_gray(img, tmpl_size=22) == s, (s, PX.ocr_gray(img, tmpl_size=22))

    # 真·读像素的铁证：喂入 A 图必得 A、喂入 B 图必得 B（桩会无视像素按 gold 注扰动）
    assert PX.ocr_gray(PX.render_text_image("ABC 123", size=22), tmpl_size=22) == "ABC 123"
    assert PX.ocr_gray(PX.render_text_image("XYZ 789", size=22), tmpl_size=22) == "XYZ 789"


def test_real_ocr_calibration_passes_clean_fails_on_degraded_pixels():
    # 标定样本取自**真实生成资产**的符号 GT（图表标题/坐标轴 + 文档字段）
    chart = make_chart_gt(20240625 + 1, "bar", n=4, chart_id="chart_bar")
    doc = make_doc_gt(20240625 + 5, doc_id="doc_receipt")
    samples = [{"gold": chart.title}, {"gold": chart.y_label},
               {"gold": doc.field_text("total")}, {"gold": doc.field_text("invoice_no")}]

    hi = calibrate_verifier({"verifier": "ocr_extractor_hi", "metric": "ocr_cer_acc",
                             "threshold": 0.95, "samples": samples})
    assert hi["backend"] in _REAL_BACKENDS
    assert hi["value"] >= 0.95 and hi["passed"] is True            # 干净像素 -> 达阈

    # **同一 gold、同一识别管线**，仅把输入像素重度降质 -> 真的读不准 -> 未达阈
    lo = calibrate_verifier({"verifier": "ocr_extractor_lo", "metric": "ocr_cer_acc",
                             "threshold": 0.95, "samples": samples})
    assert lo["value"] < 0.95 and lo["passed"] is False
    assert lo["value"] < hi["value"]                               # 降质确实更差


def test_real_detection_recovers_boxes_from_pixels():
    boxes = _DET_BOXES
    # 干净渲染 -> 连通域真的还原出每个框，且 IoU 高 -> F1 达阈
    pred = run_ml_verifier("detector_hi", {"gold_boxes": boxes})[1]
    assert len(pred) == len(boxes)
    hi = calibrate_verifier({"verifier": "detector_hi", "metric": "detection_f1",
                             "threshold": 0.95, "samples": [{"gold_boxes": boxes}]})
    assert hi["value"] >= 0.95 and hi["passed"] is True
    # 重度腐蚀像素 -> 框缩水/缺失 -> IoU 跌破 0.5 -> 未达阈
    lo = calibrate_verifier({"verifier": "detector_lo", "metric": "detection_f1",
                             "threshold": 0.95, "samples": [{"gold_boxes": boxes}]})
    assert lo["value"] < 0.95 and lo["passed"] is False


def test_real_verifier_reads_actual_asset_png():
    """读**磁盘上真实渲染的 PNG**（非内存）：证明验证器真的在读 PNG 资产像素。"""
    if not _render.HAS_PIL:
        return  # 无 PIL 环境跳过（脚本模式直接返回，pytest 下视为通过）
    doc = make_doc_gt(20240630, doc_id="doc_png_test")
    tmpdir = tempfile.mkdtemp(prefix="agenix_ocr_")
    png = os.path.join(tmpdir, "doc.png")
    assert render_doc_png(doc, png) is True
    assert os.path.isfile(png)

    # 直接从 PNG 像素裁 TOTAL 字段区域做 OCR，应还原金额文本
    total_txt = doc.field_text("total")
    got_total = PX.ocr_image(png, bbox=doc.field_bbox("total"))
    assert cer(got_total, total_txt) <= 0.2, (total_txt, got_total)
    # 标题 token "RECEIPT" 从 PNG 像素精确还原
    assert PX.ocr_image(png, bbox=doc.tokens[0]["bbox"]) == "RECEIPT"


def test_calibration_gate_uses_real_pixels_end_to_end():
    # 真实轨 cer 项 + 高保真"读干净像素"验证器 -> 标定真算 -> 进 headline
    chart = make_chart_gt(20240625 + 1, "bar", n=4, chart_id="chart_bar")
    item = {"id": "r", "kind": "cer", "track": "real", "gold": chart.title,
            "calibration": {"verifier": "ocr_extractor_hi", "metric": "ocr_cer_acc",
                            "threshold": 0.95,
                            "samples": [{"gold": chart.title}, {"gold": chart.x_label},
                                        {"gold": chart.y_label}]}}
    res = score_grounding(_mk_task([item]), {"r": chart.title})
    rep = res["calibration"]["r"]
    assert rep["backend"] in _REAL_BACKENDS and rep["passed"] is True
    assert _approx(res["real"], 1.0) and res["real_trusted"] is True


def test_calibration_gate_admits_or_demotes_real_track():
    # 真实 OCR 项 + 高保真验证器 -> 标定通过 -> 进 headline（real 非 nan）
    hi_item = {"id": "r", "kind": "cer", "track": "real", "gold": "REVENUE",
               "calibration": {"verifier": "ocr_extractor_hi", "metric": "ocr_cer_acc",
                               "threshold": 0.95, "samples": _OCR_SAMPLES}}
    res_hi = score_grounding(_mk_task([hi_item]), {"r": "REVENUE"})
    assert res_hi["calibration"]["r"]["passed"] is True
    assert _approx(res_hi["real"], 1.0)                  # 进 headline
    assert res_hi["real_trusted"] is True
    assert res_hi["real_headline_eligible"] is True

    # 同样答对，但低保真验证器 -> 未达阈 -> untrusted 仅诊断（real=nan，diagnostic 保留）
    lo_item = dict(hi_item)
    lo_item["calibration"] = {"verifier": "ocr_extractor_lo", "metric": "ocr_cer_acc",
                              "threshold": 0.95, "samples": _OCR_SAMPLES}
    res_lo = score_grounding(_mk_task([lo_item]), {"r": "REVENUE"})
    assert res_lo["calibration"]["r"]["passed"] is False
    assert _isnan(res_lo["real"])                        # 不进 headline
    assert _approx(res_lo["real_diagnostic"], 1.0)       # 仍作诊断
    assert res_lo["real_trusted"] is False
    assert res_lo["real_headline_eligible"] is False

    # 标定门也适用于检测类指标（attach 到任意 real 项）
    det_item = {"id": "rb", "kind": "iou", "track": "real", "gold": [10, 10, 40, 20],
                "tol": 0.5,
                "calibration": {"verifier": "detector_lo", "metric": "detection_f1",
                                "threshold": 0.95, "samples": _DET_SAMPLES}}
    res_det = score_grounding(_mk_task([det_item]), {"rb": [10, 10, 40, 20]})
    assert res_det["calibration"]["rb"]["passed"] is False
    assert _isnan(res_det["real"]) and res_det["real_trusted"] is False


def test_two_tracks_never_merged_into_scalar():
    # 合成 + 真实两轨各自成值，返回向量；不存在任何"合并标量"键
    items = [
        {"id": "s", "kind": "numeric", "track": "synthetic", "gold": 10.0, "tol": 0.005},
        {"id": "r", "kind": "cer", "track": "real", "gold": "X",
         "calibration": {"verifier": "ocr_extractor_hi", "metric": "ocr_cer_acc",
                         "threshold": 0.95, "samples": _OCR_SAMPLES}},
    ]
    res = score_grounding(_mk_task(items), {"s": 10.0, "r": "X"})
    assert set(["synthetic", "real"]).issubset(res.keys())
    assert res["vector"] == {"synthetic": res["synthetic"], "real": res["real"]}
    assert "combined" not in res and "scalar" not in res and "overall" not in res


# --------------------------------------------------------------------------- #
# ⑤ 合成-真实 Spearman ρ 数据门
# --------------------------------------------------------------------------- #
def test_rho_data_gate_switches_headline_rule():
    concordant = [(0.1, 0.12), (0.4, 0.38), (0.6, 0.65), (0.9, 0.88)]
    rho_hi = synthetic_real_spearman(concordant)
    assert rho_hi >= 0.8
    assert grounding_headline_rule(rho_hi) == "synthetic_only_real_audit"

    discordant = [(0.1, 0.9), (0.4, 0.6), (0.6, 0.4), (0.9, 0.1)]
    rho_lo = synthetic_real_spearman(discordant)
    assert rho_lo < 0.8
    assert grounding_headline_rule(rho_lo) == "synthetic_and_real_coheadline"

    # 边界 + 不可估（单点/nan）保守并列
    assert grounding_headline_rule(0.85) == "synthetic_only_real_audit"
    assert grounding_headline_rule(0.79) == "synthetic_and_real_coheadline"
    assert grounding_headline_rule(float("nan")) == "synthetic_and_real_coheadline"
    assert _isnan(synthetic_real_spearman([(0.5, 0.5)]))   # <2 点不可估


# --------------------------------------------------------------------------- #
# 向后兼容契约 + 资产确定性
# --------------------------------------------------------------------------- #
def test_backward_compatible_contract():
    # 无 grounding 的任务 -> nan/nan，键齐全（score.py / aggregate.py 依赖）
    res = score_grounding(Task(task_id="n", dimension="U1"), {})
    assert _isnan(res["synthetic"]) and _isnan(res["real"]) and res["per_item"] == {}
    # 既有 u3 任务仍可评分，两轨在 [0,1]
    u3 = {t.task_id: t for t in load_tasks(_TASK_DIR)}["u3_chart_discrepancy"]
    answers = {"g_chart_q3": 12.5, "g_pdf_q3": 14.0, "g_chart_box": [10, 20, 40, 15],
               "g_minimal_pair": {"scores": {"i0c0": 1, "i0c1": 0, "i1c0": 0, "i1c1": 1}},
               "g_real_ocr": "Q3 Net 14.0M"}
    r = score_grounding(u3, answers)
    assert _approx(r["synthetic"], 1.0) and _approx(r["real"], 1.0)


def test_asset_gt_determinism():
    a, b = make_chart_gt(123, "bar"), make_chart_gt(123, "bar")
    assert a.values == b.values and a.argmax_category == b.argmax_category
    assert make_chart_gt(123, "bar").values != make_chart_gt(124, "bar").values
    d1, d2 = make_doc_gt(7), make_doc_gt(7)
    assert d1.field_text("total") == d2.field_text("total")


# --------------------------------------------------------------------------- #
# 新任务可被 orchestrator 用 mock 策略端到端跑通
# --------------------------------------------------------------------------- #
def test_new_tasks_run_through_orchestrator():
    tasks = [t for t in load_tasks(_TASK_DIR) if t.task_id.startswith("ground_")]
    assert len(tasks) >= 3
    report = evaluate({"oracle-bot": "oracle", "weak-bot": "weak"}, tasks, n_runs=3, k=3)
    assert "grounding_rho" in report and "grounding_headline_rule" in report

    recs = {r["task_id"]: r for r in report["raw_records"]["oracle-bot"]
            if r["task_id"].startswith("ground_")}
    # 三个主任务：oracle 合成轨满分，真实轨（hi 标定）进 headline 且满分
    for tid in ("ground_chart_revenue", "ground_table_financials", "ground_doc_receipt"):
        g = recs[tid]["grounding"]
        assert _approx(g["synthetic"], 1.0), (tid, g["synthetic"])
        assert _approx(g["real"], 1.0) and g["real_trusted"] is True, (tid, g)
    # 结构化探针：未标定真实轨(lo) -> real=nan 仅诊断；新 kind mock 不答 -> 合成<1 但仍跑通
    gs = recs["ground_structured_reading"]["grounding"]
    assert _isnan(gs["real"]) and gs["real_trusted"] is False
    assert not _isnan(gs["real_diagnostic"])
    assert 0.0 < gs["synthetic"] < 1.0


def test_u3_generated_pilot_templates_are_loadable():
    tasks = load_task_bank(_ENGINE, include_top_level=False, include_solvable=False,
                           dimensions=["U3"])
    templates = {(t.difficulty_knobs or {}).get("template") for t in tasks}
    assert {"u3_chart", "u3_document", "u3_table", "u3_webpage", "u3_video"}.issubset(templates)
    assert all((t.difficulty_knobs or {}).get("difficulty") == "pilot" for t in tasks)


# --------------------------------------------------------------------------- #
# 脚本模式运行（与 test_meta.py 一致的兜底 runner）
# --------------------------------------------------------------------------- #
_ALL = [test_minimal_pair_group_only_when_both_sides_correct,
        test_closed_id_exact_match, test_numeric_relaxed_tolerance, test_iou_threshold,
        test_cer_ocr, test_table_teds_structure_and_content,
        test_ocr_bbox_requires_text_and_box, test_temporal_video_and_layout_verifiers,
        test_guessing_and_priors_get_no_credit,
        test_calibrate_verifier_metric_and_threshold,
        test_ml_verifier_is_real_pixel_reader_not_stub,
        test_real_ocr_calibration_passes_clean_fails_on_degraded_pixels,
        test_real_detection_recovers_boxes_from_pixels,
        test_real_verifier_reads_actual_asset_png,
        test_calibration_gate_uses_real_pixels_end_to_end,
        test_calibration_gate_admits_or_demotes_real_track,
        test_two_tracks_never_merged_into_scalar, test_rho_data_gate_switches_headline_rule,
        test_backward_compatible_contract, test_asset_gt_determinism,
        test_new_tasks_run_through_orchestrator, test_u3_generated_pilot_templates_are_loadable]


if __name__ == "__main__":
    failed = 0
    for fn in _ALL:
        try:
            fn()
            print("PASS  %s" % fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL  %s -> %r" % (fn.__name__, e))
    print("\n%d/%d passed" % (len(_ALL) - failed, len(_ALL)))
    sys.exit(1 if failed else 0)
