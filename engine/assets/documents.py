"""
合成文档 / 截图资产生成器（收据风格）。

- make_doc_gt(seed)：**纯函数**，确定性产出可控文本内容 + 命名字段（符号级 GT）。
- render_doc_png(gt, path)：PIL 渲染，用 textbbox **实测**每个 token 的像素 bbox 回填 GT，
  使 OCR token+bbox 真值与图像严格对齐。

缺 PIL 时退化：用字符宽度估算 bbox，跳过 PNG。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from assets.gt import DocGT, xywh_from_ltrb
from assets import _render


_VENDORS = ["ACME CLOUD INC", "NORTHWIND LABS", "GLOBEX SYSTEMS", "INITECH LLC"]


def make_doc_gt(seed: int, doc_id: Optional[str] = None) -> DocGT:
    """生成一张收据/发票截图的文本字段 GT（金额、日期、单号、供应商）。"""
    rng = np.random.default_rng(seed)
    vendor = _VENDORS[int(rng.integers(0, len(_VENDORS)))]
    invoice_no = "INV-%05d" % int(rng.integers(10000, 99999))
    date = "2024-%02d-%02d" % (int(rng.integers(1, 13)), int(rng.integers(1, 28)))
    subtotal = round(float(rng.uniform(100.0, 900.0)), 2)
    tax = round(subtotal * 0.08, 2)
    total = round(subtotal + tax, 2)
    did = doc_id or ("doc_%d" % seed)
    gt = DocGT(doc_id=did, title="RECEIPT")
    # 字段内容（bbox 在渲染期回填）
    gt.fields = {
        "vendor": {"text": vendor, "bbox": []},
        "invoice_no": {"text": invoice_no, "bbox": []},
        "date": {"text": date, "bbox": []},
        "subtotal": {"text": "%.2f" % subtotal, "bbox": []},
        "tax": {"text": "%.2f" % tax, "bbox": []},
        "total": {"text": "%.2f" % total, "bbox": []},
    }
    return gt


# 文档布局：每行 (字段名 or None, 标签文本, 值字段名 or None, 直接值文本)
def _layout(gt: DocGT) -> List[Tuple[Optional[str], str, Optional[str], str]]:
    f = gt.fields
    return [
        (None, gt.title, None, ""),
        ("vendor", "Vendor:", "vendor", f["vendor"]["text"]),
        (None, "Invoice:", "invoice_no", f["invoice_no"]["text"]),
        (None, "Date:", "date", f["date"]["text"]),
        (None, "Subtotal:", "subtotal", f["subtotal"]["text"]),
        (None, "Tax (8%):", "tax", f["tax"]["text"]),
        (None, "TOTAL:", "total", f["total"]["text"]),
    ]


def render_doc_png(gt: DocGT, png_path: str) -> bool:
    W, H = 460, 300
    gt.image_size = [W, H]
    margin_x, top, line_h = 24, 18, 38
    label_x, value_x = margin_x, 230

    if not _render.HAS_PIL:
        _estimate_doc_bbox(gt, label_x, value_x, top, line_h)
        return False

    from PIL import Image, ImageDraw
    font = _render.pil_font(18)
    title_font = _render.pil_font(24)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    tokens: List[Dict[str, Any]] = []
    y = top
    for i, (label_field, label, value_field, value) in enumerate(_layout(gt)):
        is_title = (i == 0)
        f = title_font if is_title else font
        # 标签
        l, t, r, b = _render.measure_text(d, (label_x, y), label, f)
        d.text((label_x, y), label, fill="black", font=f)
        tokens.append({"text": label, "bbox": xywh_from_ltrb(l, t, r, b)})
        # 值（右列）
        if value:
            vl, vt, vr, vb = _render.measure_text(d, (value_x, y), value, font)
            d.text((value_x, y), value, fill=(20, 60, 140), font=font)
            vbb = xywh_from_ltrb(vl, vt, vr, vb)
            tokens.append({"text": value, "bbox": vbb})
            if value_field and value_field in gt.fields:
                gt.fields[value_field]["bbox"] = vbb
        y += line_h if not is_title else line_h + 6

    gt.tokens = tokens
    _render.ensure_dir(png_path)
    img.save(png_path)
    return True


def _estimate_doc_bbox(gt: DocGT, label_x: int, value_x: int, top: int,
                       line_h: int) -> None:
    tokens: List[Dict[str, Any]] = []
    y = top
    for i, (label_field, label, value_field, value) in enumerate(_layout(gt)):
        ch = 11 if i == 0 else 9
        h = 22 if i == 0 else 16
        tokens.append({"text": label,
                       "bbox": [float(label_x), float(y), ch * len(label), float(h)]})
        if value:
            vbb = [float(value_x), float(y), 9.0 * len(value), 16.0]
            tokens.append({"text": value, "bbox": vbb})
            if value_field and value_field in gt.fields:
                gt.fields[value_field]["bbox"] = vbb
        y += line_h + (6 if i == 0 else 0)
    gt.tokens = tokens
