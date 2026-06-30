"""
可选渲染后端探测与共享工具。

matplotlib / PIL 均为**可选**：缺失时，上层生成器退化为"仅产出符号级 GT（带估算
bbox）+ HTML"，渲染步骤被跳过并在 manifest 标注 rendered=False。评分与测试不依赖此处。
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

HAS_MPL = False
HAS_PIL = False

try:  # matplotlib（图表栅格渲染 + 真实 bbox 提取）
    import matplotlib
    matplotlib.use("Agg")  # 无界面、确定性
    import matplotlib.pyplot as plt  # noqa: F401
    HAS_MPL = True
except Exception:  # noqa: BLE001
    HAS_MPL = False

try:  # Pillow（合成文档/表格栅格渲染）
    from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    HAS_PIL = True
except Exception:  # noqa: BLE001
    HAS_PIL = False


def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def pil_font(size: int = 16):
    """取一个可度量 bbox 的字体：优先 matplotlib 自带 DejaVuSans.ttf，回退默认位图字体。"""
    if not HAS_PIL:
        return None
    from PIL import ImageFont
    try:
        if HAS_MPL:
            import matplotlib
            ttf = os.path.join(os.path.dirname(matplotlib.__file__),
                               "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
            if os.path.isfile(ttf):
                return ImageFont.truetype(ttf, size)
    except Exception:  # noqa: BLE001
        pass
    try:
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001
        return None


def measure_text(draw, xy: Tuple[float, float], text: str, font) -> Tuple[float, float, float, float]:
    """返回文本的 (l, t, r, b)。优先用 textbbox，回退到 textlength/字体高度估算。"""
    x, y = xy
    try:
        l, t, r, b = draw.textbbox((x, y), text, font=font)
        return float(l), float(t), float(r), float(b)
    except Exception:  # noqa: BLE001
        try:
            w = float(draw.textlength(text, font=font))
        except Exception:  # noqa: BLE001
            w = 7.0 * len(text)
        h = 14.0
        return x, y, x + w, y + h
