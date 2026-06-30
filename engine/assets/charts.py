"""
图表资产生成器：柱状图 / 折线图。

- make_chart_gt(seed, kind)：**纯函数**，确定性产出底层数值/类别/极值（符号级 GT）。
- render_chart(gt, png_path)：matplotlib(Agg) 栅格渲染，并从真实图元提取**像素级 bbox**
  （每根柱子 / 极值点 / 标题 / 坐标轴标签的 OCR token）回填进 GT —— bbox 与 PNG 严格对齐。

缺 matplotlib 时退化：用线性布局**估算** bbox，跳过 PNG（manifest 标 rendered=False）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from assets.gt import ChartGT, xywh_from_ltrb
from assets import _render


_CATEGORY_POOL = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
_TITLE_POOL = [
    "Quarterly Revenue (M USD)",
    "Segment Operating Margin",
    "Regional Net Profit (M USD)",
    "Cloud ARR by Quarter (M USD)",
]


def make_chart_gt(seed: int, kind: str = "bar", n: int = 4,
                  chart_id: Optional[str] = None) -> ChartGT:
    """确定性生成图表符号级 GT（数值/类别/极值）。"""
    rng = np.random.default_rng(seed)
    n = max(3, min(n, len(_CATEGORY_POOL)))
    categories = _CATEGORY_POOL[:n]
    base = float(rng.uniform(8.0, 16.0))
    values = []
    for _ in range(n):
        base = base + float(rng.uniform(-3.0, 4.5))
        values.append(round(max(1.0, base), 1))
    title = _TITLE_POOL[int(rng.integers(0, len(_TITLE_POOL)))]
    cid = chart_id or ("chart_%s_%d" % (kind, seed))
    gt = ChartGT(chart_id=cid, kind=kind, title=title,
                 x_label="Quarter", y_label="M USD",
                 categories=categories, values=values, unit="M USD")
    return gt.derive()


def render_chart(gt: ChartGT, png_path: str) -> bool:
    """渲染 PNG 并回填像素级 bbox + OCR token。返回是否真正栅格化。"""
    if not _render.HAS_MPL:
        _estimate_bbox(gt)
        return False

    import matplotlib.pyplot as plt

    dpi = 100
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=dpi)
    x = np.arange(len(gt.categories))

    bars = None
    if gt.kind == "line":
        ax.plot(x, gt.values, marker="o", color="#1f77b4", linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(gt.categories)
    else:
        bars = ax.bar(x, gt.values, color="#1f77b4", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(gt.categories)

    title_artist = ax.set_title(gt.title)
    xlabel_artist = ax.set_xlabel(gt.x_label)
    ylabel_artist = ax.set_ylabel(gt.y_label)
    ax.set_ylim(0, max(gt.values) * 1.25)

    _render.ensure_dir(png_path)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    gt.image_size = [int(W), int(H)]

    def to_img(bb) -> List[float]:
        # display 坐标（原点左下）-> 图像坐标（原点左上）
        return xywh_from_ltrb(bb.x0, H - bb.y1, bb.x1, H - bb.y0)

    # 绘图区
    gt.plot_bbox = to_img(ax.get_window_extent(renderer))

    # 每根柱子的 bbox（折线图则取极值点的小框）
    gt.bars_bbox = {}
    if bars is not None:
        for cat, rect in zip(gt.categories, bars):
            gt.bars_bbox[cat] = to_img(rect.get_window_extent(renderer))
    else:
        # 折线：用数据坐标->显示坐标，给每个点一个 12px 方框
        for i, cat in enumerate(gt.categories):
            dx, dy = ax.transData.transform((x[i], gt.values[i]))
            l, t, r, b = dx - 6, (H - dy) - 6, dx + 6, (H - dy) + 6
            gt.bars_bbox[cat] = [round(l, 1), round(t, 1), 12.0, 12.0]

    # OCR token（标题/坐标轴标签）
    gt.tokens = []
    for artist, text in ((title_artist, gt.title),
                         (xlabel_artist, gt.x_label),
                         (ylabel_artist, gt.y_label)):
        try:
            bb = artist.get_window_extent(renderer)
            gt.tokens.append({"text": text, "bbox": to_img(bb)})
        except Exception:  # noqa: BLE001
            pass

    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return True


def _estimate_bbox(gt: ChartGT) -> None:
    """无 matplotlib 时的线性布局估算（保证 GT 自洽、可被 IoU 验证器使用）。"""
    W, H = 640, 400
    gt.image_size = [W, H]
    left, right, top, bottom = 70, 610, 50, 350
    gt.plot_bbox = [left, top, right - left, bottom - top]
    n = len(gt.categories)
    span = (right - left) / max(1, n)
    vmax = max(gt.values) * 1.25
    gt.bars_bbox = {}
    for i, cat in enumerate(gt.categories):
        bx = left + span * i + span * 0.2
        bw = span * 0.6
        bh = (gt.values[i] / vmax) * (bottom - top)
        by = bottom - bh
        gt.bars_bbox[cat] = [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)]
    gt.tokens = [{"text": gt.title, "bbox": [W / 2 - 100, 20, 200, 18]},
                 {"text": gt.x_label, "bbox": [W / 2 - 30, 375, 60, 14]},
                 {"text": gt.y_label, "bbox": [10, H / 2 - 20, 14, 40]}]
