import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters import parse_submission  # noqa: E402
from judge.panel import JudgePanel, MockJudge  # noqa: E402
from scoring.aggregate import build_judge_block, build_report  # noqa: E402


def _const(verdicts):
    return lambda resp, ctx: list(verdicts)


def _panel():
    panel = JudgePanel([
        MockJudge("deepseek", "deepseek", _const([1, 1, 0, 1])),
        MockJudge("kimi", "moonshot", _const([1, 1, 0, 1])),
        MockJudge("glm", "zhipu", _const([1, 1, 0, 1])),
    ])
    panel.calibrate_to_human([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], method="isotonic")
    return panel


def _rec(task_id, run_index, success=True, rationale="uses the binding rule and cites evidence"):
    return {
        "task_id": task_id,
        "template": task_id,
        "dimension": "U1",
        "model_id": "m",
        "run_index": run_index,
        "success": bool(success),
        "raw_success": bool(success),
        "critical": False,
        "asr": 0.0,
        "incidents": [],
        "process": 1.0 if success else 0.0,
        "recovery": float("nan"),
        "expected_milestone_completion": float("nan"),
        "grounding": {"synthetic": float("nan"), "real": float("nan"),
                      "real_diagnostic": float("nan")},
        "efficiency": {"eff": 1.0 if success else float("nan")},
        "n_faults": 0,
        "milestone_a": {},
        "milestone_diag": {},
        "cost": 1.0,
        "submission_metadata": {"rationale": rationale, "raw_response": "{\"rationale\":\"...\"}"},
        "judge_subject": rationale,
    }


def test_parse_submission_preserves_raw_response_and_rationale():
    content = '{"rationale":"because evidence A binds","actions":[]}'
    sub = parse_submission(content)
    assert sub.actions == []
    assert getattr(sub, "raw_response") == content
    assert getattr(sub, "rationale") == "because evidence A binds"
    assert getattr(parse_submission("not json"), "raw_response") == "not json"


def test_judge_policy_default_diagnostic_even_when_gates_pass():
    block = build_judge_block(
        {"m": []}, ["m"], panel=_panel(), responses={"m": "clear rationale"}
    )
    assert block["policy"] == "diagnostic"
    assert block["enters_headline"] is False
    assert block["judge_headline"] is None
    assert block["human_calibration"]["passed"] is True


def test_build_report_conditional_headline_gated_independently():
    recs = {
        "m": [_rec("t1", 0, True), _rec("t2", 0, True)],
        "n": [_rec("t1", 0, False), _rec("t2", 0, False)],
    }
    report = build_report(
        recs,
        k=2,
        dim_n_boot=8,
        glmm_n_boot=8,
        judge_panel=_panel(),
        judge_policy="conditional_headline",
        tested_families={"m": "seed", "n": "seed"},
        seed=0,
    )
    assert report["judge"]["enters_headline"] is True
    assert report["judge_headline"]["version"] == "judge_headline_v1_residual_subjective"
    assert report["judge_headline"]["gate"]["cross_family_ok"] is True
    assert report["judge_headline"]["gate"]["human_calibration_passed"] is True
    assert report["profiles"]["R"]["per_model"]["m"]["dim_vector"]["U1"]["point"] >= 0.0
    assert "grounding" in report["judge"]["excluded_scopes"]
