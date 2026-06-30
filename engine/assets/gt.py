"""
符号级 Ground Truth（GT）数据结构与序列化 —— **纯 Python + numpy，无重型依赖**。

本模块只描述"资产的真值是什么"（数值、极值、单元格、OCR token、bbox、表格结构），
不负责把资产渲染成图片。这样做的目的：

  1. 确定性：GT 完全由 seed 决定，可复现；
  2. 解耦：scoring/grounding.py 与 tests 只依赖 GT（纯数据），不依赖 matplotlib/PIL；
     渲染库缺失时评分与测试仍可运行（见 assets/_render.py 的可选降级）。

bbox 统一用 [x, y, w, h]（左上原点，像素坐标，与渲染产物一致）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# 基础几何
# --------------------------------------------------------------------------- #
def xywh_from_ltrb(l: float, t: float, r: float, b: float) -> List[float]:
    """左上右下 -> [x, y, w, h]，保留 1 位小数（渲染像素足够）。"""
    return [round(float(l), 1), round(float(t), 1),
            round(float(r) - float(l), 1), round(float(b) - float(t), 1)]


def iou_xywh(a: List[float], b: List[float]) -> float:
    """两个 [x,y,w,h] 框的 IoU（与 scoring 中实现一致，供生成期自检）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix = max(0.0, min(ax2, bx2) - max(ax, bx))
    iy = max(0.0, min(ay2, by2) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# OCR token（文本 + bbox）
# --------------------------------------------------------------------------- #
@dataclass
class Token:
    text: str
    bbox: List[float]  # [x, y, w, h]

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "bbox": [round(float(v), 1) for v in self.bbox]}


# --------------------------------------------------------------------------- #
# 图表 GT（柱/折线）
# --------------------------------------------------------------------------- #
@dataclass
class ChartGT:
    chart_id: str
    kind: str                       # "bar" | "line"
    title: str
    x_label: str
    y_label: str
    categories: List[str]
    values: List[float]
    unit: str = ""
    # 极值（argmax/argmin 的类别名 + 数值）
    argmax_category: str = ""
    argmin_category: str = ""
    max_value: float = 0.0
    min_value: float = 0.0
    # 像素级符号 bbox（渲染后由 _render 填充；纯 GT 阶段可为空）
    bars_bbox: Dict[str, List[float]] = field(default_factory=dict)   # category -> bbox
    plot_bbox: Optional[List[float]] = None                          # 绘图区 bbox
    # 渲染期回填的 OCR token（标题/坐标轴标签等）
    tokens: List[Dict[str, Any]] = field(default_factory=list)
    image_size: Optional[List[int]] = None                           # [W, H]

    def derive(self) -> "ChartGT":
        """从 values 推导极值（确定性）。"""
        if self.values:
            mx = max(range(len(self.values)), key=lambda i: self.values[i])
            mn = min(range(len(self.values)), key=lambda i: self.values[i])
            self.argmax_category = self.categories[mx]
            self.argmin_category = self.categories[mn]
            self.max_value = float(self.values[mx])
            self.min_value = float(self.values[mn])
        return self

    def value_of(self, category: str) -> Optional[float]:
        if category in self.categories:
            return float(self.values[self.categories.index(category)])
        return None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 表格 GT（结构 + 单元格内容；TEDS 式比对的真值）
# --------------------------------------------------------------------------- #
@dataclass
class TableGT:
    table_id: str
    title: str
    headers: List[str]
    grid: List[List[str]]           # 含表头的二维网格（grid[0] = headers）
    cells_bbox: Dict[str, List[float]] = field(default_factory=dict)  # "r,c" -> bbox
    image_size: Optional[List[int]] = None

    @property
    def n_rows(self) -> int:
        return len(self.grid)

    @property
    def n_cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def cell(self, r: int, c: int) -> Optional[str]:
        if 0 <= r < len(self.grid) and 0 <= c < len(self.grid[r]):
            return self.grid[r][c]
        return None

    def table_repr(self) -> Dict[str, Any]:
        """供 table_teds 验证器使用的结构化表示（grid of strings）。"""
        return {"grid": [[str(c) for c in row] for row in self.grid]}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["n_rows"] = self.n_rows
        d["n_cols"] = self.n_cols
        return d


# --------------------------------------------------------------------------- #
# 合成文档 / 截图 GT（OCR token + bbox + 命名字段）
# --------------------------------------------------------------------------- #
@dataclass
class DocGT:
    doc_id: str
    title: str
    tokens: List[Dict[str, Any]] = field(default_factory=list)        # [{text,bbox}]
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # name -> {text,bbox}
    image_size: Optional[List[int]] = None

    def field_text(self, name: str) -> Optional[str]:
        f = self.fields.get(name)
        return f["text"] if f else None

    def field_bbox(self, name: str) -> Optional[List[float]]:
        f = self.fields.get(name)
        return f["bbox"] if f else None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
