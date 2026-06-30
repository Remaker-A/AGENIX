"""
AGENIX-Engine 多模态资产生成器（U3 双轨 grounding 用）。

程序化、确定性地生成带**符号级 GT** 的真实资产：
  - charts.py     图表（柱/折线）+ 像素级 bbox + 极值/数值 GT
  - tables.py     财务表（PNG/HTML）+ 单元格内容/结构（TEDS 真值）
  - documents.py  合成收据截图 + OCR token+bbox + 命名字段
  - counterfactual.py  反事实最小对（group-score 用）
  - generate.py   驱动：生成资产/GT/manifest 并派生 ground_*.json 任务

设计：符号级 GT（gt.py）为纯 Python+numpy，不依赖 matplotlib/PIL；渲染库可选。
"""
from __future__ import annotations

from assets.gt import ChartGT, TableGT, DocGT, Token, iou_xywh, xywh_from_ltrb
from assets.charts import make_chart_gt
from assets.tables import make_table_gt
from assets.documents import make_doc_gt
from assets.counterfactual import minimal_pair_chart, minimal_pair_doc, pair_gold

__all__ = [
    "ChartGT", "TableGT", "DocGT", "Token", "iou_xywh", "xywh_from_ltrb",
    "make_chart_gt", "make_table_gt", "make_doc_gt",
    "minimal_pair_chart", "minimal_pair_doc", "pair_gold",
]
