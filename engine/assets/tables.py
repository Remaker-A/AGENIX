"""
表格资产生成器：财务式表格。

- make_table_gt(seed)：**纯函数**，确定性产出表头 + 单元格内容（符号级 GT），
  其 table_repr() 供 TEDS 式结构比对验证器使用。
- render_table_html(gt, path)：始终可用（无依赖），输出 <table> HTML。
- render_table_png(gt, path)：PIL 网格渲染，按固定布局回填**每个单元格 bbox**。

缺 PIL 时退化：仍产出 HTML 与按布局估算的 cell bbox，跳过 PNG。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from assets.gt import TableGT
from assets import _render


_METRIC_POOL = ["Revenue", "COGS", "Gross Profit", "OpEx", "Net Income"]


def make_table_gt(seed: int, n_metric_rows: int = 4,
                  table_id: Optional[str] = None) -> TableGT:
    """生成财务表：列=[Metric, 年度...]，行=指标。"""
    rng = np.random.default_rng(seed)
    years = ["FY%d" % y for y in (2022, 2023, 2024)]
    headers = ["Metric"] + years
    metrics = _METRIC_POOL[:max(2, min(n_metric_rows, len(_METRIC_POOL)))]
    grid: List[List[str]] = [headers]
    for m in metrics:
        base = float(rng.uniform(50.0, 200.0))
        row = [m]
        for _ in years:
            base = base * float(rng.uniform(0.95, 1.18))
            row.append("%.1f" % round(base, 1))
        grid.append(row)
    tid = table_id or ("table_%d" % seed)
    return TableGT(table_id=tid, title="Financial Summary (M USD)",
                   headers=headers, grid=grid)


def render_table_html(gt: TableGT, html_path: str) -> bool:
    rows_html = []
    for r, row in enumerate(gt.grid):
        tag = "th" if r == 0 else "td"
        cells = "".join("<%s>%s</%s>" % (tag, _esc(c), tag) for c in row)
        rows_html.append("    <tr>%s</tr>" % cells)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>%s</title>"
        "<style>table{border-collapse:collapse;font-family:DejaVu Sans,Arial}"
        "th,td{border:1px solid #444;padding:6px 12px;text-align:right}"
        "th{background:#1f77b4;color:#fff}td:first-child,th:first-child"
        "{text-align:left}</style></head><body>"
        "<h3>%s</h3><table>\n%s\n</table></body></html>"
    ) % (_esc(gt.title), _esc(gt.title), "\n".join(rows_html))
    _render.ensure_dir(html_path)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return True


def render_table_png(gt: TableGT, png_path: str) -> bool:
    """PIL 网格渲染 + 回填 cell bbox（固定行高/列宽，bbox 与像素严格对齐）。"""
    n_rows, n_cols = gt.n_rows, gt.n_cols
    pad_x, pad_y = 14, 8
    row_h = 30
    col_w = 130
    x0, y0 = 10, 40
    W = x0 * 2 + col_w * n_cols
    H = y0 + row_h * n_rows + 10
    gt.image_size = [W, H]

    # 先把 cell bbox 写好（无论是否有 PIL，布局是确定的）
    gt.cells_bbox = {}
    for r in range(n_rows):
        for c in range(n_cols):
            cx = x0 + c * col_w
            cy = y0 + r * row_h
            gt.cells_bbox["%d,%d" % (r, c)] = [float(cx), float(cy),
                                               float(col_w), float(row_h)]

    if not _render.HAS_PIL:
        return False

    from PIL import Image, ImageDraw
    font = _render.pil_font(15)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((x0, 10), gt.title, fill="black", font=font)
    for r in range(n_rows):
        for c in range(n_cols):
            cx = x0 + c * col_w
            cy = y0 + r * row_h
            fill = (31, 119, 180) if r == 0 else (255, 255, 255)
            d.rectangle([cx, cy, cx + col_w, cy + row_h], outline=(60, 60, 60),
                        fill=fill)
            txt = str(gt.grid[r][c])
            color = "white" if r == 0 else "black"
            d.text((cx + pad_x, cy + pad_y), txt, fill=color, font=font)
    _render.ensure_dir(png_path)
    img.save(png_path)
    return True


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))
