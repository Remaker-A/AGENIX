"""模板库聚合：导入各维度模板模块即完成注册（register 在模块顶层执行）。

只覆盖 U1 / U2 / U4 / U5 / U6（U3 多模态 grounding 由另一 worker 负责，本包不触碰）。
"""
from __future__ import annotations

from generators.templates import u1_templates  # noqa: F401
from generators.templates import u2_templates  # noqa: F401
from generators.templates import u4_templates  # noqa: F401
from generators.templates import u5_templates  # noqa: F401
from generators.templates import u6_templates  # noqa: F401

DIMENSIONS_COVERED = ["U1", "U2", "U4", "U5", "U6"]
