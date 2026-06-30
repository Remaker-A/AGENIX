"""
反事实最小对（counterfactual minimal pairs）：对一个资产做**最小改动**得到对照版，
使"图侧 + 文侧都对"才得分（group-score = image-score ∧ text-score），用于击穿语言
先验、压顶端区分、抗饱和（spec §4.4 / §3 高区分度通则）。

每个最小对产出：
  (gt0, gt1, caption0, caption1, change)
其中 caption_k 与 image_k 一一对应（gold 配对 = {i0:c0, i1:c1}）。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

from assets.gt import ChartGT, DocGT


def minimal_pair_chart(gt0: ChartGT, seed: int = 0) -> Tuple[ChartGT, str, str, Dict[str, Any]]:
    """最小改动：把当前最高柱压到次高之下，使极值类别翻转（只动一个数）。"""
    gt1 = copy.deepcopy(gt0)
    gt1.chart_id = gt0.chart_id + "_cf"
    vals = list(gt1.values)
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    mx, sec = order[0], order[1]
    new_val = round(vals[sec] - 0.5, 1)          # 仅此一处改动
    old_val = vals[mx]
    vals[mx] = new_val
    gt1.values = vals
    gt1.derive()
    cap0 = "The highest bar is %s." % gt0.argmax_category
    cap1 = "The highest bar is %s." % gt1.argmax_category
    change = {"type": "bar_value", "category": gt0.categories[mx],
              "from": old_val, "to": new_val,
              "argmax_before": gt0.argmax_category,
              "argmax_after": gt1.argmax_category}
    return gt1, cap0, cap1, change


def minimal_pair_doc(gt0: DocGT, seed: int = 0) -> Tuple[DocGT, str, str, Dict[str, Any]]:
    """最小改动：把 TOTAL 金额改一处（+100.00），保持版式不变。"""
    gt1 = copy.deepcopy(gt0)
    gt1.doc_id = gt0.doc_id + "_cf"
    old = gt0.fields["total"]["text"]
    new = "%.2f" % round(float(old) + 100.0, 2)
    gt1.fields["total"]["text"] = new
    cap0 = "The total amount is %s." % old
    cap1 = "The total amount is %s." % new
    change = {"type": "field_text", "field": "total", "from": old, "to": new}
    return gt1, cap0, cap1, change


def pair_gold(pair_id: str, caption0: str, caption1: str,
              image0: str, image1: str, change: Dict[str, Any]) -> Dict[str, Any]:
    """最小对的 gold 元数据（供真实模型适配器按 i×c 相似度打分；验证器仅核 group-score）。"""
    return {
        "pair_id": pair_id,
        "image0": image0, "image1": image1,
        "caption0": caption0, "caption1": caption1,
        "assignment": {"i0": "c0", "i1": "c1"},  # 正确配对
        "change": change,
    }
