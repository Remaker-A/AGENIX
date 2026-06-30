"""
多模态 grounding 验证器（CP4）：**双轨**（synthetic / real）+ typed verifier 菜单 +
ML 验证器可靠性标定门 + 合成-真实 Spearman ρ 数据门。

只含**可程序化判定**，绝无 CLIP/embedding（spec §4.4，弃 CLIP 理由：相似≠正确）：
  closed_id     : 闭式 ID 证据集合的精确 F1（节点对齐=精确比对，杜绝相似度/LLM 循环）
  numeric       : relaxed 相对误差 |ŷ-y|/max(|y|,ε) <= tol（财务 0.5% / 图表 5%）
  iou           : 边界框 IoU >= tol（默认 0.5）
  cer           : 1 - 归一化字符错误率（OCR）
  minimal_pair  : 反事实最小对 group-score = image-score ∧ text-score（击穿语言先验）
  table_teds    : TEDS 式表格结构+内容相似（树编辑距离归一化，简化实现）
  ocr_bbox      : OCR token —— 文本(CER<=cer_tol) 与 bbox(IoU>=tol) **同时**对才计数，取 F1
  temporal_iou : 视频/音频事件时间段 IoU（默认阈值 0.5）
  video_event_id : 视频事件闭式 ID 精确 F1
  doc_layout_f1 : 文档/网页 layout block 标签+bbox（可选文本）联合 F1

**双轨双值**（spec §4.4 / §5.4）：返回 {synthetic: G_syn, real: G_real}，**永不合成单标量**。
  - synthetic 轨：合成-符号 GT，可每 seed 重采样（抗污染、grounded-reasoning 横比）。
  - real 轨：真实生态层；其中**任何 ML 判定器（检测/分割/抽取）须先过可靠性标定门**——
    在人工校准集上报 precision/recall 或 CER，达阈（默认 ≥0.95）才"进 headline"，否则
    仅作诊断（real_diagnostic），不计入 real headline 聚合（spec §4.4 ML 准入门）。
  - 真实 ML 验证器为**真实像素读取**实现（`assets/pixel_ocr.py`）：OCR=渲染/读取真实 PNG 字形
    → 连通域切分 → 基线对齐的原生像素模板匹配；检测=连通域还原 bbox。**不再注入 error_rate**。
    高/低保真之分由**真实图像降质**驱动（hi 读干净图→达阈；lo 读重度模糊/腐蚀图→识别真崩→未达阈）。
    标定门逻辑为真（真实计算 CER 一致率 / 检测 F1 并按阈值门控）。若运行环境装有 tesseract 二进制，
    OCR 自动优先 pytesseract（否则用上述确定性 PIL 像素读取，因其可复现且我们自有资产字体）。

另提供（供集成阶段接入 build_report / Profile 选择，本模块不自行合并两轨）：
  synthetic_real_spearman + grounding_headline_rule（ρ≥0.8→合成 headline+真实审计；ρ<0.8→并列）。

向后兼容：score_grounding(task, answers) 签名与返回键 {"synthetic","real","per_item"} 不变。
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from schema import Task, GroundingItem

NAN = float("nan")
DEFAULT_IOU_THR = 0.5
DEFAULT_TEMPORAL_IOU_THR = 0.5
DEFAULT_CER_TOL = 0.3
DEFAULT_CALIB_THRESHOLD = 0.95  # spec §4.4：与人工 GT 一致率 ≥0.95 才进 headline


def _isnan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


# --------------------------------------------------------------------------- #
# 闭式 typed verifier —— 低层度量
# --------------------------------------------------------------------------- #
def precision_recall_f1_set(pred, gold) -> Tuple[float, float, float]:
    """集合精确比对的 (precision, recall, F1)。"""
    p, g = set(pred or []), set(gold or [])
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p or not g:
        return 0.0, 0.0, 0.0
    tp = len(p & g)
    prec = tp / len(p)
    rec = tp / len(g)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def f1_set(pred, gold) -> float:
    return precision_recall_f1_set(pred, gold)[2]


def iou_xywh(a, b) -> float:
    """两个 [x,y,w,h] 框的 IoU。"""
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix = max(0.0, min(ax2, bx2) - max(ax, bx))
    iy = max(0.0, min(ay2, by2) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


_iou = iou_xywh  # 内部别名（与历史命名兼容）


def edit_distance(a: str, b: str) -> int:
    """Levenshtein 编辑距离（确定性 DP）。"""
    a, b = str(a), str(b)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def cer(pred: str, gold: str) -> float:
    """字符错误率 = edit(pred,gold)/|gold|（gold 空时：pred 也空→0，否则→1）。"""
    pred, gold = str(pred), str(gold)
    if len(gold) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, gold) / len(gold)


def numeric_relaxed(pred: Any, gold: Any, tol: float) -> float:
    """relaxed-acc：|ŷ-y|/max(|y|,ε) <= tol → 1，否则 0。"""
    if pred is None or gold is None:
        return 0.0
    try:
        denom = max(abs(float(gold)), 1e-9)
        return 1.0 if abs(float(pred) - float(gold)) / denom <= float(tol) else 0.0
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# 结构化 typed verifier —— TEDS（表格） / ocr_bbox（文档）
# --------------------------------------------------------------------------- #
def _as_grid(t: Any) -> List[List[str]]:
    if t is None:
        return []
    if isinstance(t, dict):
        t = t.get("grid", [])
    out: List[List[str]] = []
    for row in t or []:
        out.append([str(c) for c in row])
    return out


def table_teds(pred: Any, gold: Any) -> float:
    """TEDS 式表格相似（简化树编辑距离，归一化到 [0,1]）。

    树 = table 节点 + 每行节点 + 每个单元格（叶）节点；按 (行索引, 列索引) 对齐：
      - 缺/多行：该行节点 + 其单元格 计入插删代价；
      - 缺/多单元格：单元格插删代价 1；
      - 单元格不一致：用归一化字符串编辑（min(1, edit/|gold_cell|)）作替换代价。
    teds = 1 - cost / max(|nodes_pred|, |nodes_gold|)。
    """
    P, G = _as_grid(pred), _as_grid(gold)
    if not P and not G:
        return 1.0

    def nodes(grid: List[List[str]]) -> int:
        return 1 + len(grid) + sum(len(r) for r in grid)

    denom = max(nodes(P), nodes(G), 1)
    cost = 0.0
    for r in range(max(len(P), len(G))):
        pr = P[r] if r < len(P) else None
        gr = G[r] if r < len(G) else None
        if pr is None or gr is None:
            present = gr if gr is not None else pr
            cost += 1 + len(present or [])      # 行节点 + 其单元格的插删
            continue
        for c in range(max(len(pr), len(gr))):
            pc = pr[c] if c < len(pr) else None
            gc = gr[c] if c < len(gr) else None
            if pc is None or gc is None:
                cost += 1.0
            elif pc == gc:
                cost += 0.0
            else:
                cost += min(1.0, edit_distance(pc, gc) / max(len(gc), 1))
    return max(0.0, min(1.0, 1.0 - cost / denom))


def ocr_bbox_score(pred_tokens: Any, gold_tokens: Any,
                   iou_thr: float = DEFAULT_IOU_THR,
                   cer_tol: float = DEFAULT_CER_TOL) -> float:
    """OCR token+bbox 联合 F1：一个 token 命中需 **文本(CER<=cer_tol) 与 bbox(IoU>=iou_thr) 同时**成立。"""
    P = pred_tokens or []
    G = gold_tokens or []
    if not P and not G:
        return 1.0
    if not P or not G:
        return 0.0
    matched_g = set()
    tp = 0
    for pt in P:
        if not isinstance(pt, dict):
            continue
        ptext = str(pt.get("text", ""))
        pbox = pt.get("bbox")
        best_gi = None
        best_iou = iou_thr
        for gi, gt in enumerate(G):
            if gi in matched_g or not isinstance(gt, dict):
                continue
            iv = iou_xywh(pbox, gt.get("bbox"))
            if iv >= best_iou and cer(ptext, str(gt.get("text", ""))) <= cer_tol:
                best_gi, best_iou = gi, iv
        if best_gi is not None:
            matched_g.add(best_gi)
            tp += 1
    prec = tp / len(P)
    rec = tp / len(G)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def _as_span(x: Any) -> Optional[Tuple[float, float]]:
    """把 [start,end] 或 {"start_s","end_s"} 等常见标注形态归一化为时间段。"""
    if isinstance(x, dict):
        for key in ("span", "temporal_span", "time_span", "segment"):
            if key in x:
                return _as_span(x.get(key))
        start = x.get("start_s", x.get("start", x.get("begin")))
        end = x.get("end_s", x.get("end", x.get("stop")))
        x = [start, end]
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        try:
            start, end = float(x[0]), float(x[1])
        except (TypeError, ValueError):
            return None
        if math.isfinite(start) and math.isfinite(end) and end > start:
            return start, end
    return None


def temporal_iou(pred_span: Any, gold_span: Any) -> float:
    """两个时间段的 IoU，输入支持 [start,end] 或含 span/start/end 的 dict。"""
    p = _as_span(pred_span)
    g = _as_span(gold_span)
    if p is None or g is None:
        return 0.0
    ps, pe = p
    gs, ge = g
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def _event_ids(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, dict):
        if isinstance(x.get("events"), list):
            return _event_ids(x.get("events"))
        vals = []
        for key in ("event_id", "id", "label", "event"):
            if x.get(key) is not None:
                vals.append(str(x.get(key)))
        if isinstance(x.get("ids"), list):
            vals.extend(str(v) for v in x.get("ids"))
        return vals
    if isinstance(x, (list, tuple, set)):
        vals: List[str] = []
        for v in x:
            vals.extend(_event_ids(v) if isinstance(v, dict) else [str(v)])
        return vals
    return [str(x)]


def video_event_id_score(pred: Any, gold: Any) -> float:
    """视频事件闭式 ID 精确 F1；支持单 ID、ID 列表或 events dict。"""
    return f1_set(_event_ids(pred), _event_ids(gold))


def _layout_blocks(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, dict):
        for key in ("blocks", "layout_blocks", "items", "elements"):
            if isinstance(x.get(key), list):
                x = x.get(key)
                break
        else:
            x = [x]
    if not isinstance(x, (list, tuple)):
        return []
    blocks: List[Dict[str, Any]] = []
    for b in x:
        if isinstance(b, dict):
            blocks.append(b)
    return blocks


def _layout_label(block: Dict[str, Any]) -> str:
    return str(block.get("label", block.get("type", block.get("role", ""))))


def doc_layout_f1(pred_blocks: Any, gold_blocks: Any,
                  iou_thr: float = DEFAULT_IOU_THR,
                  cer_tol: float = DEFAULT_CER_TOL) -> float:
    """文档/网页 layout block F1：label 匹配 + bbox IoU；若 gold 有 text，则文本也须过 CER 门。"""
    P = _layout_blocks(pred_blocks)
    G = _layout_blocks(gold_blocks)
    if not P and not G:
        return 1.0
    if not P or not G:
        return 0.0
    matched_g = set()
    tp = 0
    for pb in P:
        pbox = pb.get("bbox")
        plabel = _layout_label(pb)
        ptext = str(pb.get("text", ""))
        best_gi = None
        best_iou = iou_thr
        for gi, gb in enumerate(G):
            if gi in matched_g:
                continue
            glabel = _layout_label(gb)
            if glabel and plabel != glabel:
                continue
            gtext = gb.get("text")
            if gtext is not None and cer(ptext, str(gtext)) > cer_tol:
                continue
            iv = iou_xywh(pbox, gb.get("bbox"))
            if iv >= best_iou:
                best_gi, best_iou = gi, iv
        if best_gi is not None:
            matched_g.add(best_gi)
            tp += 1
    prec = tp / len(P)
    rec = tp / len(G)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- #
# 反事实最小对 group-score（spec §4.4）
# --------------------------------------------------------------------------- #
def minimal_pair_group_score(scores: Dict[str, float]) -> Dict[str, float]:
    """给定 4 组匹配分 s(i,c)（键 i0c0/i0c1/i1c0/i1c1），返回 text/image/group 三分。

    text  = 1[s(i0,c0)>s(i0,c1) ∧ s(i1,c1)>s(i1,c0)]   （每张图内，正确 caption 更高）
    image = 1[s(i0,c0)>s(i1,c0) ∧ s(i1,c1)>s(i0,c1)]   （每个 caption 下，正确图更高）
    group = text ∧ image                                （两侧都对才得分）
    """
    try:
        a = float(scores["i0c0"]); b = float(scores["i0c1"])
        c = float(scores["i1c0"]); d = float(scores["i1c1"])
    except (KeyError, TypeError, ValueError):
        return {"text": 0.0, "image": 0.0, "group": 0.0}
    text = 1.0 if (a > b and d > c) else 0.0
    image = 1.0 if (a > c and d > b) else 0.0
    group = 1.0 if (text >= 1.0 and image >= 1.0) else 0.0
    return {"text": text, "image": image, "group": group}


def _minimal_pair(ans: Dict[str, Any], gold: Dict[str, Any]) -> float:
    s = (ans or {}).get("scores", {})
    return minimal_pair_group_score(s)["group"]


# --------------------------------------------------------------------------- #
# ML 验证器（**真实像素读取**）+ 可靠性标定门（逻辑为真）
# --------------------------------------------------------------------------- #
# degrade: 真实图像降质级别（替代旧的 error_rate）。hi 读干净图，lo 读重度降质图。
_VERIFIERS: Dict[str, Dict[str, Any]] = {
    # 高保真：读干净渲染像素 -> 标定指标 ≥0.95 -> 进 headline
    "ocr_extractor_hi": {"type": "ocr", "degrade": "clean"},
    "detector_hi":      {"type": "detect", "degrade": "clean"},
    # 低保真：读重度降质像素（模糊致粘连 / 腐蚀）-> 识别真崩 <0.95 -> untrusted 仅诊断
    "ocr_extractor_lo": {"type": "ocr", "degrade": "heavy"},
    "detector_lo":      {"type": "detect", "degrade": "heavy"},
}


def _pixel_backend():
    """惰性载入真实像素验证器后端（assets.pixel_ocr）；不可用时返回 None（永不在 import 期崩）。"""
    try:
        from assets import pixel_ocr  # noqa: WPS433
        return pixel_ocr
    except Exception:  # noqa: BLE001 - PIL/assets 缺失 -> 保守降级（标定不可得 -> untrusted）
        return None


def run_ml_verifier(verifier_id: str, sample: Dict[str, Any]) -> Tuple[str, Any]:
    """跑一个真实 ML 验证器：**读像素**得预测。

    OCR：把 sample['gold'] 渲染成像素（按 degrade 施降质）→ 连通域切分 → 原生像素模板匹配
         还原字符串。返回 ('ocr', pred_text)。
    检测：把 sample['gold_boxes'] 渲染成实心块（按 degrade 施降质）→ 连通域还原 bbox。
         返回 ('detect', pred_boxes)。
    后端不可用时抛 RuntimeError（由 calibrate_verifier 捕获并判 value=NaN→untrusted）。
    """
    vcfg = _VERIFIERS.get(verifier_id, {"type": "ocr", "degrade": "heavy"})
    px = _pixel_backend()
    if px is None:
        raise RuntimeError("pixel verifier backend unavailable (need numpy+Pillow)")
    degrade = str(vcfg.get("degrade", "clean"))
    if vcfg.get("type") == "detect":
        return "detect", px.run_detect_verifier(sample.get("gold_boxes", []) or [], degrade=degrade)
    return "ocr", px.run_ocr_verifier(str(sample.get("gold", "")), degrade=degrade)


def _detection_f1(pred_boxes, gold_boxes, iou_thr: float = 0.5) -> float:
    P = pred_boxes or []
    G = gold_boxes or []
    if not P and not G:
        return 1.0
    if not P or not G:
        return 0.0
    matched = set()
    tp = 0
    for pb in P:
        best_gi, best_iou = None, iou_thr
        for gi, gb in enumerate(G):
            if gi in matched:
                continue
            iv = iou_xywh(pb, gb)
            if iv >= best_iou:
                best_gi, best_iou = gi, iv
        if best_gi is not None:
            matched.add(best_gi)
            tp += 1
    prec = tp / len(P)
    rec = tp / len(G)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


_CALIB_CACHE: Dict[str, Dict[str, Any]] = {}   # 标定确定性 -> 按 spec 记忆化（避免逐 trace 重算）


def _calib_key(spec: Dict[str, Any]) -> Optional[str]:
    try:
        return json.dumps({"v": spec.get("verifier"),
                           "m": spec.get("metric", "ocr_cer_acc"),
                           "t": float(spec.get("threshold", DEFAULT_CALIB_THRESHOLD)),
                           "s": spec.get("samples", []) or []},
                          sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return None


def calibrate_verifier(spec: Dict[str, Any]) -> Dict[str, Any]:
    """对 ML 验证器跑人工校准集，**真实读像素得预测**并计算一致性指标，按阈值门控。

    spec = {"verifier": id, "metric": "ocr_cer_acc"|"detection_f1", "threshold": 0.95,
            "samples": [{"gold": "..."} | {"gold_boxes": [...]}]}
    返回 {"verifier","metric","value","threshold","passed","n","backend"}。
    （结果仅由 spec 决定，故按 spec 记忆化：同一标定集跨 trace/模型只算一次。）
    """
    key = _calib_key(spec)
    if key is not None and key in _CALIB_CACHE:
        return dict(_CALIB_CACHE[key])

    vid = spec.get("verifier")
    metric = spec.get("metric", "ocr_cer_acc")
    threshold = float(spec.get("threshold", DEFAULT_CALIB_THRESHOLD))
    samples = spec.get("samples", []) or []
    px = _pixel_backend()
    backend = px.backend_name() if px is not None else "unavailable"

    value = NAN
    try:
        if metric == "ocr_cer_acc":
            # 字符加权一致率 = 1 - Σ edit / Σ |gold|（比逐样本平均更稳）
            tot_edit, tot_chars = 0, 0
            for s in samples:
                gold = str(s.get("gold", ""))
                _, pred = run_ml_verifier(vid, s)        # 真实读像素得预测
                tot_edit += edit_distance(str(pred), gold)
                tot_chars += max(len(gold), 0)
            value = 1.0 - (tot_edit / tot_chars) if tot_chars else NAN
        elif metric == "detection_f1":
            f1s = []
            for s in samples:
                gb = s.get("gold_boxes", []) or []
                _, pb = run_ml_verifier(vid, s)          # 真实读像素得检测框
                f1s.append(_detection_f1(pb, gb, iou_thr=0.5))
            value = sum(f1s) / len(f1s) if f1s else NAN
    except Exception:  # noqa: BLE001 - 后端不可用/读图异常 -> value=NaN -> untrusted（保守，绝不误升 headline）
        value = NAN

    passed = (not _isnan(value)) and value >= threshold
    rep = {"verifier": vid, "metric": metric, "value": value, "threshold": threshold,
           "passed": bool(passed), "n": len(samples), "backend": backend}
    if key is not None:
        _CALIB_CACHE[key] = dict(rep)
    return rep


def _item_calibration(item: GroundingItem) -> Optional[Dict[str, Any]]:
    """读取 grounding 项上声明的 ML 验证器标定规格（GroundingItem extra=allow）。"""
    spec = getattr(item, "calibration", None)
    if isinstance(spec, dict) and spec.get("verifier"):
        return spec
    return None


# --------------------------------------------------------------------------- #
# 单项评分
# --------------------------------------------------------------------------- #
def score_item(item: GroundingItem, answers: Dict[str, Any]) -> float:
    """对单个 grounding 项打分。**绝不因答案类型异常而崩溃**：真实模型可能给出任意形态的
    答案（如把 minimal_pair 答成字符串、把 iou 答成数字），无法授信即判 0（与"未知 kind→0"
    同philosophy）——保证真实横评不被单个畸形答案中断。"""
    try:
        return _score_item_dispatch(item, answers)
    except Exception:  # noqa: BLE001 - 任意畸形答案 -> 无法授信 -> 0（不抛错）
        return 0.0


def _score_item_dispatch(item: GroundingItem, answers: Dict[str, Any]) -> float:
    ans = answers.get(item.id) if isinstance(answers, dict) else None
    kind = item.kind
    gold = item.gold
    if kind == "closed_id":
        return f1_set(ans, gold)
    if kind == "numeric":
        return numeric_relaxed(ans, gold, getattr(item, "tol", 0.0))
    if kind == "iou":
        if not isinstance(ans, (list, tuple)) or not gold:
            return 0.0
        return 1.0 if iou_xywh(ans, gold) >= (getattr(item, "tol", 0.0) or DEFAULT_IOU_THR) else 0.0
    if kind == "cer":
        if ans is None:
            return 0.0
        return max(0.0, 1.0 - cer(ans, gold))
    if kind == "minimal_pair":
        return _minimal_pair(ans if isinstance(ans, dict) else {}, gold or {})
    if kind == "table_teds":
        if not isinstance(ans, (dict, list)):
            return 0.0
        return table_teds(ans, gold)
    if kind == "ocr_bbox":
        if not isinstance(ans, (list, tuple)):
            return 0.0
        cer_tol = float(getattr(item, "cer_tol", DEFAULT_CER_TOL))
        return ocr_bbox_score(ans, gold, iou_thr=(getattr(item, "tol", 0.0) or DEFAULT_IOU_THR),
                              cer_tol=cer_tol)
    if kind == "temporal_iou":
        gold_span = gold if gold is not None else getattr(item, "temporal_span", None)
        thr = getattr(item, "tol", 0.0) or DEFAULT_TEMPORAL_IOU_THR
        return 1.0 if temporal_iou(ans, gold_span) >= thr else 0.0
    if kind == "video_event_id":
        return video_event_id_score(ans, gold)
    if kind == "doc_layout_f1":
        cer_tol = float(getattr(item, "cer_tol", DEFAULT_CER_TOL))
        return doc_layout_f1(ans, gold, iou_thr=(getattr(item, "tol", 0.0) or DEFAULT_IOU_THR),
                             cer_tol=cer_tol)
    # 未知 kind：无法授信 -> 0（保证 orchestrator 跑通，不抛错）
    return 0.0


def _agg(pairs: List[Tuple[float, float]]) -> float:
    if not pairs:
        return NAN
    wsum = sum(w for w, _ in pairs)
    return sum(w * s for w, s in pairs) / wsum if wsum else NAN


# --------------------------------------------------------------------------- #
# 主入口：双轨评分 + 标定门（签名/返回键向后兼容）
# --------------------------------------------------------------------------- #
def score_grounding(task: Task, answers: Dict[str, Any]) -> Dict[str, Any]:
    """对单个 trace 的 grounding 答案打分，返回**双轨双值**（绝不合并为单标量）。

    返回键：
      synthetic          : 合成轨聚合 G_syn（[0,1] 或 nan）
      real               : 真实轨 **headline-eligible** 聚合 G_real（仅含标定通过/无需标定项；nan 表示无可进 headline 的真实项）
      real_diagnostic    : 真实轨**全部**项聚合（含 untrusted ML 项，仅诊断）
      real_trusted       : 所有声明的 ML 验证器是否都过标定门（无 ML 项则按是否有可进 headline 的真实项）
      real_headline_eligible : 是否存在可进 headline 的真实项
      calibration        : {item_id: 标定报告}
      per_item           : {item_id: 分数}
      vector             : {"synthetic":..., "real":...}（双值向量，显式不合并）
    """
    answers = answers or {}
    empty = {"synthetic": NAN, "real": NAN, "real_diagnostic": NAN,
             "real_trusted": False, "real_headline_eligible": False,
             "calibration": {}, "per_item": {},
             "vector": {"synthetic": NAN, "real": NAN}}
    if task.grounding is None or not task.grounding.items:
        return empty

    per_item: Dict[str, float] = {}
    syn_pairs: List[Tuple[float, float]] = []
    real_headline_pairs: List[Tuple[float, float]] = []
    real_all_pairs: List[Tuple[float, float]] = []
    calibration: Dict[str, Any] = {}
    ml_gate_flags: List[bool] = []

    for item in task.grounding.items:
        sc = score_item(item, answers)
        per_item[item.id] = sc
        track = getattr(item, "track", "synthetic")
        w = float(getattr(item, "weight", 1.0))
        if track == "real":
            real_all_pairs.append((w, sc))
            calib_spec = _item_calibration(item)
            if calib_spec is not None:
                rep = calibrate_verifier(calib_spec)
                calibration[item.id] = rep
                ml_gate_flags.append(rep["passed"])
                eligible = rep["passed"]            # 未达阈 -> untrusted -> 仅诊断
            else:
                eligible = True                     # 直接闭式真实验证器（无 ML 抽取器）
            if eligible:
                real_headline_pairs.append((w, sc))
        else:
            syn_pairs.append((w, sc))

    g_syn = _agg(syn_pairs)
    g_real_headline = _agg(real_headline_pairs)
    g_real_diag = _agg(real_all_pairs)
    headline_eligible = len(real_headline_pairs) > 0
    real_trusted = headline_eligible and (all(ml_gate_flags) if ml_gate_flags else True)

    return {
        "synthetic": g_syn,
        "real": g_real_headline,
        "real_diagnostic": g_real_diag,
        "real_trusted": bool(real_trusted),
        "real_headline_eligible": bool(headline_eligible),
        "calibration": calibration,
        "per_item": per_item,
        "vector": {"synthetic": g_syn, "real": g_real_headline},
    }


# --------------------------------------------------------------------------- #
# 合成-真实 Spearman ρ 数据门（grounding 层实现；供集成阶段接入报告/Profile 选择）
# --------------------------------------------------------------------------- #
def _rank(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based 平均秩（处理并列）
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def synthetic_real_spearman(pairs: List[Tuple[float, float]]) -> float:
    """pairs = [(G_syn_m, G_real_m) per model]，忽略含 nan 的对。返回 Spearman ρ。"""
    xs = [float(s) for s, r in pairs if not _isnan(s) and not _isnan(r)]
    ys = [float(r) for s, r in pairs if not _isnan(s) and not _isnan(r)]
    if len(xs) < 2:
        return NAN
    rx, ry = _rank(xs), _rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    dx = [v - mx for v in rx]
    dy = [v - my for v in ry]
    num = sum(a * b for a, b in zip(dx, dy))
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    return num / den if den > 0 else NAN


def grounding_headline_rule(rho: float, gate: float = 0.8) -> str:
    """ρ 数据门 -> headline 规则（与 aggregate.build_report 字符串一致，便于集成）。

    ρ>=gate → 'synthetic_only_real_audit'（合成轨独立 headline + 真实轨审计）
    ρ<gate 或 不可估 → 'synthetic_and_real_coheadline'（两轨并列，测不同构念不可互替）
    """
    if _isnan(rho) or rho < gate:
        return "synthetic_and_real_coheadline"
    return "synthetic_only_real_audit"
