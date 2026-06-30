"""U5 校准评分与 headline 接线的小范围回归。"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import ModelSubmission, Task, Trace  # noqa: E402
from scoring.reliability import risk_coverage_auc, score_calibration  # noqa: E402
from scoring.aggregate import (  # noqa: E402
    U5_HEADLINE_VERSION,
    U5_LEGACY_HEADLINE_VERSION,
    aggregate_model,
)
from run_eval import _write_csv  # noqa: E402


def test_score_calibration_metrics_from_probes():
    task = Task(
        task_id="u5_cal",
        dimension="U5",
        calibration_probes=[
            {"id": "A", "answerable": True},
            {"id": "B", "answerable": True},
            {"id": "C", "answerable": False},
            {"id": "D", "answerable": False},
        ],
    )
    trace = Trace(
        task_id=task.task_id,
        model_id="m",
        submission=ModelSubmission(
            confidences={"A": 0.9, "B": 0.8, "C": 0.2, "D": 0.1},
            abstain={"A": False, "B": False, "C": True, "D": True},
        ),
    )

    rep = score_calibration(task, trace)
    assert rep["coverage_ok"] is True
    assert abs(rep["brier"] - 0.025) < 1e-12
    assert abs(rep["ece"] - 0.15) < 1e-12
    assert abs(rep["aurc"] - (5.0 / 24.0)) < 1e-12
    assert rep["abstain_f1"] == 1.0
    assert risk_coverage_auc([(0.9, 1), (0.2, 0)]) == 0.25


def _rec(success=True, process=1.0, calibration=None):
    return {
        "task_id": "u5_task",
        "template": "u5_task",
        "dimension": "U5",
        "success": success,
        "raw_success": success,
        "critical": False,
        "asr": 0.0,
        "process": process,
        "recovery": float("nan"),
        "expected_milestone_completion": float("nan"),
        "grounding": {"synthetic": float("nan"), "real": float("nan"),
                      "real_diagnostic": float("nan"), "real_trusted": False,
                      "real_headline_eligible": False, "calibration": {}},
        "efficiency": {"eff": 1.0 if success else float("nan")},
        "cost": 1.0,
        "calibration": calibration or {},
    }


def test_u5_without_calibration_keeps_legacy_headline():
    agg = aggregate_model([_rec(True, 1.0), _rec(False, 0.0)],
                          k=3, profile="R", n_boot=40)
    assert agg["u5_headline_version"] == U5_LEGACY_HEADLINE_VERSION
    assert agg["u5_calibration"]["enters_headline"] is False
    assert agg["selective_partition_success"] == 0.5
    assert abs(agg["dim_vector"]["U5"]["point"] - 0.5) < 1e-12


def test_u5_calibration_can_enter_headline_when_covered():
    cal = {"has_probes": True, "coverage_ok": True, "n_probes": 4,
           "n_confidences": 4, "coverage": 1.0, "coverage_gate": 0.8,
           "brier": 1.0, "ece": 1.0, "aurc": 1.0, "abstain_f1": 0.0,
           "score": 0.0}
    agg = aggregate_model([_rec(True, 1.0, cal)], k=3, profile="R", n_boot=40)
    assert agg["u5_headline_version"] == U5_HEADLINE_VERSION
    assert agg["selective_partition_success"] == 1.0
    assert agg["u5_calibration"]["enters_headline"] is True
    assert agg["dim_vector"]["U5"]["point"] < agg["selective_partition_success"]


def test_run_eval_csv_includes_calibration_columns(tmp_path):
    report = {
        "k": 5,
        "dimension_stats": {"U5": {"per_model": {"m": {"n_obs": 1}}}},
        "profiles": {
            "R": {
                "dims_present": ["U5"],
                "per_model": {
                    "m": {
                        "dim_vector": {"U5": {"point": 0.7, "lo": 0.6, "hi": 0.8}},
                        "reliability": {"per_run": 1.0, "pass_at_k": 1.0,
                                        "pass_pow_k": 1.0},
                        "asr": 0.0,
                        "mean_cost": 1.0,
                        "grounding": {},
                        "u5_headline_version": U5_HEADLINE_VERSION,
                        "selective_partition_success": 1.0,
                        "u5_calibration": {"coverage": 1.0, "brier": 0.1,
                                           "ece": 0.2, "aurc": 0.3,
                                           "abstain_precision": 1.0,
                                           "abstain_recall": 1.0,
                                           "abstain_f1": 1.0},
                    }
                },
            }
        },
    }
    out = tmp_path / "eval.csv"
    _write_csv(str(out), report)
    rows = list(csv.reader(open(out, "r", encoding="utf-8")))
    header = rows[0]
    for col in ("calibration_brier", "calibration_ece", "calibration_aurc",
                "abstain_f1", "u5_headline_version",
                "selective_partition_success"):
        assert col in header
    row = dict(zip(header, rows[1]))
    assert row["u5_headline_version"] == U5_HEADLINE_VERSION
    assert math.isclose(float(row["calibration_brier"]), 0.1)
