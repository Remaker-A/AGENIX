"""
真实**像素级** OCR / 框检测验证器（替换 grounding.py 里按 error_rate 注入扰动的确定性
stub）。本模块**真的从图像像素得出结果**：

  - OCR：渲染（或读取磁盘 PNG 的）真实字形像素 → 连通域切分 → **基线归一化的字形模板匹配**
    → 还原字符串。识别过程**只看像素**，不偷看 gold（与任何 OCR 一致：标定需要"图像↔真值"配对，
    但预测仅由像素决定）。
  - 检测：把框渲染成实心块 → 连通域标注 → 还原每个块的 bbox。

为什么用纯 PIL 像素读取而非 pytesseract/easyocr：本机 `pytesseract` 模块在位但**tesseract 二进制
缺失**（Windows 上不可 pip 安装），`easyocr` 需 torch（未装）。我们**自己渲染资产**、掌握字体，
故确定性的模板匹配是"真实读图"且可复现；若运行环境存在 tesseract 二进制，OCR 会**自动优先**用它
（见 `ocr_gray` 的 `_tesseract_gray`）。

高/低保真之分由**真实图像降质**驱动（不再注入预测错误）：
  - hi：读干净渲染 → 识别近乎无误 → CER 一致率 / 检测 F1 ≥ 阈。
  - lo：读重度降质（高斯模糊致字形粘连 / 框腐蚀）→ 识别真的崩 → 指标 < 阈 → 标定门拦下。

依赖：numpy（必需）、Pillow（必需，渲染+读图）；scipy 可选（连通域/形态学，缺失走纯实现）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from assets import _render

# 域内字符集（闭词表 OCR：限定候选可显著降低混淆，真实 OCR 亦常用受限字符集）
CHARSET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " .,:%()-/$"
)

_REF_SIZE = 22          # 默认模板字号（标定渲染同字号 → 原生像素匹配，无重采样伪影）
_CAP_RATIO = 0.72       # cap_height / font_size（DejaVuSans 实测 ≈0.71–0.73）
_INK_THRESH = 128       # 二值化阈值（<阈为墨）

_TEMPLATES: Dict[int, List[Tuple[str, np.ndarray]]] = {}   # size -> [(char, placed_mask)]
_TESS_OK: Optional[bool] = None


def _frame_dims(size: int) -> Tuple[int, int, int]:
    """某字号下的匹配帧 (高, 基线行, 宽)，随字号线性缩放（模板/观测同字号→同帧）。"""
    h = max(12, int(round(size * 1.7)))
    base_row = max(8, int(round(size * 1.15)))   # 基线之上容升部/括号，之下容降部
    w = max(8, int(round(size * 1.6)))
    return h, base_row, w


# --------------------------------------------------------------------------- #
# 字体 / 渲染
# --------------------------------------------------------------------------- #
def _font(size: int):
    f = _render.pil_font(size)
    if f is None:
        raise RuntimeError("Pillow 字体不可用，无法进行像素级 OCR")
    return f


def render_text_image(text: str, size: int = 22, pad: int = 6) -> np.ndarray:
    """把一行文本渲染为灰度 numpy 数组（白底黑字）。供标定"渲染→读像素"用。"""
    from PIL import Image, ImageDraw
    font = _font(size)
    probe = Image.new("L", (8, 8), 255)
    pd = ImageDraw.Draw(probe)
    try:
        l, t, r, b = pd.textbbox((0, 0), text or " ", font=font)
    except Exception:  # noqa: BLE001
        l, t, r, b = 0, 0, int(8 * len(text or " ")), size
    w = max(1, int(r - l)) + pad * 2
    h = max(1, int(b - t)) + pad * 2
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    d.text((pad - l, pad - t), text or "", fill=0, font=font)
    return np.asarray(img, dtype=np.uint8)


def render_boxes_image(boxes: Sequence[Sequence[float]], pad: int = 8) -> Tuple[np.ndarray, int, int]:
    """把若干 [x,y,w,h] 框渲染成实心黑块（白底）。返回 (灰度数组, ox, oy) 平移量。"""
    from PIL import Image, ImageDraw
    if not boxes:
        return np.full((16, 16), 255, dtype=np.uint8), 0, 0
    xs = [float(b[0]) for b in boxes]
    ys = [float(b[1]) for b in boxes]
    xe = [float(b[0]) + float(b[2]) for b in boxes]
    ye = [float(b[1]) + float(b[3]) for b in boxes]
    ox, oy = int(min(xs)) - pad, int(min(ys)) - pad
    W = int(max(xe)) - ox + pad
    H = int(max(ye)) - oy + pad
    img = Image.new("L", (max(1, W), max(1, H)), 255)
    d = ImageDraw.Draw(img)
    for b in boxes:
        x, y, w, h = float(b[0]) - ox, float(b[1]) - oy, float(b[2]), float(b[3])
        # 画到 [x, x+w-1]：使连通域 bbox 宽≈w（避免 +1 偏移）
        d.rectangle([x, y, x + max(0.0, w - 1), y + max(0.0, h - 1)], fill=0)
    return np.asarray(img, dtype=np.uint8), ox, oy


# --------------------------------------------------------------------------- #
# 图像降质（真实地破坏像素，使低保真验证器"读不准"）
# --------------------------------------------------------------------------- #
def degrade_gray(gray: np.ndarray, level: str, kind: str) -> np.ndarray:
    """对灰度图施加真实降质。level: 'clean'|'heavy'；kind: 'ocr'|'detect'。"""
    if not level or level == "clean":
        return gray
    from PIL import Image, ImageFilter
    img = Image.fromarray(gray)
    if kind == "ocr":
        # 重度降采样(50%)+模糊 -> 字形混叠/粘连/钝化 -> 仍有墨可读，但**读得满是错**（非空乱码）
        w, h = img.size
        img = img.resize((max(1, int(w * 0.5)), max(1, int(h * 0.5))), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=1.2))
        arr = np.asarray(img, dtype=np.uint8)
        return np.where(arr < _INK_THRESH, 0, 255).astype(np.uint8)
    # detect：腐蚀（缩小框）-> IoU 跌破 0.5
    mask = np.asarray(img, dtype=np.uint8) < _INK_THRESH
    mask = _binary_erosion(mask, iterations=6)
    return np.where(mask, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 连通域 / 形态学（scipy 可选，纯实现兜底）
# --------------------------------------------------------------------------- #
def _label(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-连通连通域标注，返回 (label_array, n)。"""
    try:
        from scipy import ndimage
        structure = np.ones((3, 3), dtype=bool)
        lab, n = ndimage.label(mask, structure=structure)
        return lab, int(n)
    except Exception:  # noqa: BLE001
        return _label_pure(mask)


def _label_pure(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    n = 0
    from collections import deque
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                n += 1
                dq = deque([(i, j)])
                lab[i, j] = n
                while dq:
                    y, x = dq.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] == 0:
                                lab[ny, nx] = n
                                dq.append((ny, nx))
    return lab, n


def _binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    try:
        from scipy import ndimage
        return ndimage.binary_erosion(mask, iterations=iterations)
    except Exception:  # noqa: BLE001
        m = mask
        for _ in range(max(1, iterations)):
            m2 = m.copy()
            m2[:-1, :] &= m[1:, :]
            m2[1:, :] &= m[:-1, :]
            m2[:, :-1] &= m[:, 1:]
            m2[:, 1:] &= m[:, :-1]
            m = m2
        return m


def _components_bboxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """返回各连通域的紧致 bbox: (x0, y0, x1, y1)（含端点）。"""
    lab, n = _label(mask)
    out: List[Tuple[int, int, int, int]] = []
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        if ys.size == 0:
            continue
        out.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return out


# --------------------------------------------------------------------------- #
# 字形切分（连通域 + x 重叠合并 + 空格判定）
# --------------------------------------------------------------------------- #
def _merge_into_glyphs(comps: List[Tuple[int, int, int, int]]
                       ) -> List[Tuple[int, int, int, int]]:
    """把连通域按 x 重叠合并成字形簇（处理 '%' ':' 'i' 等多部件字形）。"""
    if not comps:
        return []
    comps = sorted(comps, key=lambda b: b[0])
    glyphs: List[List[int]] = []
    for (x0, y0, x1, y1) in comps:
        if glyphs and x0 <= glyphs[-1][2]:        # 与上一簇 x 区间相交/相接 -> 合并
            g = glyphs[-1]
            g[0] = min(g[0], x0); g[1] = min(g[1], y0)
            g[2] = max(g[2], x1); g[3] = max(g[3], y1)
        else:
            glyphs.append([x0, y0, x1, y1])
    return [tuple(g) for g in glyphs]


def _place_glyph(mask: np.ndarray, gx0: int, gy0: int, gx1: int, gy1: int,
                 baseline: float, size: int) -> np.ndarray:
    """把字形墨迹**原生像素**放入固定帧：基线对齐 base_row、左缘对齐 col 0（不缩放）。

    模板与观测同字号 → 同帧 → 正确字符像素几近逐格吻合（含降部/升部/中划高度信息，
    可辨大小写、'.'↔',' 、'-'↔'_' 、'o'↔'0' 等）。"""
    fh, base_row, fw = _frame_dims(size)
    canvas = np.zeros((fh, fw), dtype=bool)
    shift_r = int(round(base_row - baseline))   # 使该行基线落到 base_row
    for r in range(gy0, gy1 + 1):
        rr = r + shift_r
        if rr < 0 or rr >= fh:
            continue
        for c in range(gx0, gx1 + 1):
            cc = c - gx0
            if cc < 0 or cc >= fw:
                continue
            if mask[r, c]:
                canvas[rr, cc] = True
    return canvas


def _build_templates(size: int = _REF_SIZE) -> List[Tuple[str, np.ndarray]]:
    """构建（缓存）某字号下、与观测同一放置流程产出的字形模板（原生像素）。

    `render_text_image` 以 (pad-l, pad-t) 绘制，使字形墨顶恒在 row=pad，据此算其基线行
    = pad + (H底 - 字符墨顶) = pad + (Hbb[3] - bb[1])，再经 `_place_glyph` 基线对齐入帧。
    """
    if size in _TEMPLATES:
        return _TEMPLATES[size]
    from PIL import Image, ImageDraw
    font = _font(size)
    probe = Image.new("L", (8, 8), 255)
    pd = ImageDraw.Draw(probe)
    hb = pd.textbbox((0, 0), "H", font=font)
    pad = 6
    tmpls: List[Tuple[str, np.ndarray]] = []
    for ch in CHARSET:
        if ch == " ":
            continue
        bb = pd.textbbox((0, 0), ch, font=font)
        gray = render_text_image(ch, size=size, pad=pad)
        mask = gray < _INK_THRESH
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        gy0 = int(ys.min())
        baseline = float(gy0 + (hb[3] - bb[1]))   # 墨顶在 row=gy0，基线 = 墨顶 + (H底-字符顶)
        placed = _place_glyph(mask, int(xs.min()), gy0, int(xs.max()),
                              int(ys.max()), baseline=baseline, size=size)
        tmpls.append((ch, placed))
    _TEMPLATES[size] = tmpls
    return tmpls


# --------------------------------------------------------------------------- #
# 识别核心
# --------------------------------------------------------------------------- #
def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """模板相似度：全帧像素一致率与墨迹 IoU 的均衡（前者辨"洞/笔画缺失"如 0↔U，
    后者辨整体形状）。正确字符同时最大化二者。"""
    eq = float(np.count_nonzero(a == b)) / a.size
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    iou = inter / union if union else (1.0 if not a.any() and not b.any() else 0.0)
    return 0.5 * eq + 0.5 * iou


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = mask[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def _match_glyph(placed: np.ndarray, tmpl_size: int) -> str:
    """最近邻模板匹配，带 ±2 行 / ±1 列抖动搜索（吸收亚像素对齐误差）。"""
    best_ch, best = "?", -1.0
    variants = [placed] + [_shift(placed, dy, dx)
                           for dy in (-2, -1, 1, 2) for dx in (-1, 0, 1)]
    for ch, tmpl in _build_templates(tmpl_size):
        v = max(_similarity(var, tmpl) for var in variants)
        if v > best:
            best, best_ch = v, ch
    return best_ch


def ocr_gray(gray: np.ndarray, prefer_tesseract: bool = True,
             tmpl_size: Optional[int] = None) -> str:
    """对灰度行图做 OCR，返回识别串。优先 tesseract（若二进制在），否则原生像素模板匹配。

    tmpl_size=None 时由观测 cap 高**自动估字号**（size≈cap_h/_CAP_RATIO），使模板与观测同字号
    → 原生像素对齐匹配（无重采样伪影）。标定路径显式传入渲染字号。"""
    if prefer_tesseract and _tesseract_available():
        t = _tesseract_gray(gray)
        if t is not None:
            return t
    mask = gray < _INK_THRESH
    if not mask.any():
        return ""
    comps = _components_bboxes(mask)
    glyphs = _merge_into_glyphs(comps)
    if not glyphs:
        return ""
    y1s = sorted(g[3] for g in glyphs)
    baseline = float(y1s[len(y1s) // 2])               # 基线 = 各字形底的中位数（抗 Q 尾/升部）
    cap_candidates = [baseline - g[1] for g in glyphs if abs(g[3] - baseline) <= 2]
    cap_h = max(cap_candidates) if cap_candidates else max(g[3] - g[1] for g in glyphs)
    cap_h = max(1.0, float(cap_h))
    size = tmpl_size if tmpl_size else max(8, int(round(cap_h / _CAP_RATIO)))
    widths = [g[2] - g[0] + 1 for g in glyphs]
    med_w = float(sorted(widths)[len(widths) // 2])
    out: List[str] = []
    prev_x1: Optional[int] = None
    for (x0, y0, x1, y1) in glyphs:
        if prev_x1 is not None and (x0 - prev_x1) > 0.55 * med_w:
            out.append(" ")
        placed = _place_glyph(mask, x0, y0, x1, y1, baseline, size)
        out.append(_match_glyph(placed, size))
        prev_x1 = x1
    return "".join(out)


def ocr_image(img: Any, bbox: Optional[Sequence[float]] = None,
              prefer_tesseract: bool = True, tmpl_size: Optional[int] = None) -> str:
    """对 PNG 路径 / PIL.Image / numpy 数组做 OCR；bbox=[x,y,w,h] 时只读该区域裁剪。"""
    gray = _to_gray(img)
    if bbox is not None and len(bbox) >= 4:
        x, y, w, h = (int(round(v)) for v in bbox[:4])
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
        gray = gray[y0:y1, x0:x1]
    return ocr_gray(gray, prefer_tesseract=prefer_tesseract, tmpl_size=tmpl_size)


def _to_gray(img: Any) -> np.ndarray:
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            return (0.299 * img[..., 0] + 0.587 * img[..., 1]
                    + 0.114 * img[..., 2]).astype(np.uint8)
        return img.astype(np.uint8)
    from PIL import Image
    if isinstance(img, str):
        im = Image.open(img).convert("L")
        return np.asarray(im, dtype=np.uint8)
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("L"), dtype=np.uint8)
    raise TypeError("不支持的图像类型: %r" % type(img))


# --------------------------------------------------------------------------- #
# 框检测（真实连通域 → bbox）
# --------------------------------------------------------------------------- #
def detect_boxes_gray(gray: np.ndarray, ox: int = 0, oy: int = 0,
                      min_area: int = 9) -> List[List[float]]:
    """对灰度图做连通域检测，返回各块 [x,y,w,h]（加回平移量 ox,oy）。"""
    mask = gray < _INK_THRESH
    if not mask.any():
        return []
    out: List[List[float]] = []
    for (x0, y0, x1, y1) in _components_bboxes(mask):
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w * h < min_area:
            continue
        out.append([float(x0 + ox), float(y0 + oy), float(w), float(h)])
    return out


# --------------------------------------------------------------------------- #
# 高层：跑一个 ML 验证器（读像素得预测）
# --------------------------------------------------------------------------- #
def run_ocr_verifier(gold_text: str, degrade: str = "clean",
                     render_size: int = 22) -> str:
    """真实 OCR 验证器：把 gold 渲染成像素 → 施降质 → 读像素还原字符串。"""
    gray = render_text_image(str(gold_text), size=render_size)
    gray = degrade_gray(gray, degrade, kind="ocr")
    return ocr_gray(gray, tmpl_size=render_size)


def run_detect_verifier(gold_boxes: Sequence[Sequence[float]],
                        degrade: str = "clean") -> List[List[float]]:
    """真实检测验证器：把框渲染成实心块 → 施降质 → 连通域还原 bbox。"""
    gray, ox, oy = render_boxes_image(gold_boxes)
    gray = degrade_gray(gray, degrade, kind="detect")
    return detect_boxes_gray(gray, ox=ox, oy=oy)


# --------------------------------------------------------------------------- #
# 可选 tesseract 后端（仅当二进制可用）
# --------------------------------------------------------------------------- #
def _tesseract_available() -> bool:
    global _TESS_OK
    if _TESS_OK is not None:
        return _TESS_OK
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _TESS_OK = True
    except Exception:  # noqa: BLE001
        _TESS_OK = False
    return _TESS_OK


def _tesseract_gray(gray: np.ndarray) -> Optional[str]:
    try:
        import pytesseract
        from PIL import Image
        cfg = "--psm 7"  # 单行
        txt = pytesseract.image_to_string(Image.fromarray(gray), config=cfg)
        return txt.strip()
    except Exception:  # noqa: BLE001
        return None


def backend_name() -> str:
    """返回当前 OCR 后端名（供报告/诊断）。"""
    return "pytesseract" if _tesseract_available() else "pil_template_match"
