"""LLM-as-judge 作为"测量仪器"的可靠性工程封装（spec §7）。

公开 API（详见 `judge/panel.py`）：
  - 面板：`JudgePanel`（接受 Dict[str, JudgeFn] 或 Sequence[判官适配器]）。
  - 可插拔评委适配器：`MockJudge`（确定性、离线）/ `LLMJudgeAdapter`（封装 OpenAI 兼容后端，
    本阶段不真调；`from_openai_adapter` 为真实跨家族评委的集成点，待 ≥3 家族 API key）。
  - 一致性：`krippendorff_alpha` / `reliability_band`（α 门：<0.667 剔出 headline）。
  - 消偏：`blind_identity` / `sanitize_injection` / `quote_as_data`。
  - 人类定标：`Calibrator`/`IdentityCalibrator`/`IsotonicCalibrator`/`PlattCalibrator`、
    `fit_calibrator`、`fit_isotonic_calibrator`、`fit_platt_calibrator`、`judge_human_agreement`。
"""
from .panel import (  # noqa: F401
    JudgeFn,
    JudgePanel,
    MockJudge,
    LLMJudgeAdapter,
    krippendorff_alpha,
    reliability_band,
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    fit_calibrator,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
    judge_human_agreement,
    blind_identity,
    sanitize_injection,
    quote_as_data,
)

__all__ = [
    "JudgeFn",
    "JudgePanel",
    "MockJudge",
    "LLMJudgeAdapter",
    "krippendorff_alpha",
    "reliability_band",
    "Calibrator",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "fit_calibrator",
    "fit_isotonic_calibrator",
    "fit_platt_calibrator",
    "judge_human_agreement",
    "blind_identity",
    "sanitize_injection",
    "quote_as_data",
]
