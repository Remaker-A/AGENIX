"""
Markdown 报告生成器 smoke 测试（report.py）。

验证：能从样例结果 JSON 产出 report.md 不报错、含关键章节与表格；matplotlib 可用时产出图表 PNG，
不可用时优雅降级为纯表格 md。
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import report  # noqa: E402

_ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = os.path.join(_ENGINE, "results")


def _sample_json():
    cands = sorted(glob.glob(os.path.join(_RESULTS, "eval_*.json")))
    return cands[0] if cands else None


def test_report_md_smoke(tmp_path):
    src = _sample_json()
    if not src:
        pytest.skip("无样例结果 JSON（results/eval_*.json）")
    out = os.path.join(str(tmp_path), "r.md")
    res = report.generate(src, out_path=out, figs_dir=os.path.join(str(tmp_path), "figs"),
                          make_figs=False)
    assert os.path.isfile(res["md_path"])
    txt = open(res["md_path"], "r", encoding="utf-8").read()
    assert "# AGENIX" in txt
    assert "## 1. 概览" in txt
    assert "| 模型 |" in txt and "ASR" in txt
    assert len(txt) > 500


def test_report_with_figs_if_matplotlib(tmp_path):
    src = _sample_json()
    if not src:
        pytest.skip("无样例结果 JSON")
    res = report.generate(src, out_path=os.path.join(str(tmp_path), "r.md"),
                          figs_dir=os.path.join(str(tmp_path), "figs"), make_figs=True)
    assert os.path.isfile(res["md_path"])
    if res["has_matplotlib"]:
        assert res["figs_made"] and all(os.path.isfile(f) for f in res["figs"])


def test_three_doubao_report_keeps_models_separate(tmp_path):
    import report_doubao_three

    result = {
        "models": [
            "doubao_seed_evolving",
            "doubao_seed_2_1_pro",
            "doubao_seed_2_1_turbo",
        ],
        "k": 5,
        "grounding": {
            "per_model": {
                "doubao_seed_evolving": {"synthetic": 0.3, "real": 0.8, "real_trusted": True},
                "doubao_seed_2_1_pro": {"synthetic": 0.4, "real": 0.7, "real_trusted": True},
                "doubao_seed_2_1_turbo": {"synthetic": 0.2, "real": 0.6, "real_trusted": True},
            }
        },
        "profiles": {
            "R": {
                "per_model": {
                    "doubao_seed_evolving": {
                        "reliability": {"per_run": 0.8, "pass_at_k": 0.8, "pass_pow_k": 0.8},
                        "asr": 0.0, "mean_cost": 1.1,
                    },
                    "doubao_seed_2_1_pro": {
                        "reliability": {"per_run": 0.9, "pass_at_k": 0.9, "pass_pow_k": 0.9},
                        "asr": 0.0, "mean_cost": 1.2,
                    },
                    "doubao_seed_2_1_turbo": {
                        "reliability": {"per_run": 0.7, "pass_at_k": 0.7, "pass_pow_k": 0.7},
                        "asr": 0.0, "mean_cost": 1.0,
                    },
                }
            }
        },
        "dimension_stats": {
            d: {"per_model": {
                "doubao_seed_evolving": {"marginal": 0.8, "lo": 0.6, "hi": 1.0, "n_obs": 3},
                "doubao_seed_2_1_pro": {"marginal": 0.9, "lo": 0.7, "hi": 1.0, "n_obs": 3},
                "doubao_seed_2_1_turbo": {"marginal": 0.7, "lo": 0.5, "hi": 0.9, "n_obs": 3},
            }}
            for d in ("U1", "U2", "U3", "U4", "U5", "U6")
        },
        "adapters": {
            "doubao_seed_evolving": {
                "kind": "real", "model": "doubao-seed-evolving", "n_calls": 8,
                "n_errors": 0, "task_log": [
                    {"task_id": "u1_a", "success_met": True, "n_actions": 1, "round_status": ["ok"]},
                    {"task_id": "u2_b", "success_met": False, "n_actions": 1, "round_status": ["ok"]},
                ],
            },
            "doubao_seed_2_1_pro": {
                "kind": "real", "model": "doubao-seed-2-1-pro-260628", "n_calls": 9,
                "n_errors": 0, "task_log": [
                    {"task_id": "u1_a", "success_met": True, "n_actions": 1, "round_status": ["ok"]},
                    {"task_id": "u2_b", "success_met": True, "n_actions": 1, "round_status": ["ok"]},
                ],
            },
            "doubao_seed_2_1_turbo": {
                "kind": "real", "model": "doubao-seed-2-1-turbo-260628", "n_calls": 7,
                "n_errors": 1, "task_log": [
                    {"task_id": "u1_a", "success_met": False, "n_actions": 0, "round_status": ["error"]},
                    {"task_id": "u2_b", "success_met": True, "n_actions": 1, "round_status": ["ok"]},
                ],
            },
        },
        "meta": {"n_tasks": 67, "n_runs": 1, "run_meta": {"jobs_done": 201, "jobs_total": 201}},
    }
    src = tmp_path / "three.json"
    src.write_text(__import__("json").dumps(result), encoding="utf-8")
    out = tmp_path / "three.md"
    report_doubao_three.generate(str(src), out_path=str(out), make_figs=False)
    txt = out.read_text(encoding="utf-8")
    assert "doubao_seed_evolving" in txt
    assert "doubao_seed_2_1_pro" in txt
    assert "doubao_seed_2_1_turbo" in txt
    assert "67 题 × 3 模型 = 201" in txt
    assert "单一 seed" not in txt
