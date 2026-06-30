import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import Task, Action, ModelSubmission  # noqa: E402
from sandbox import Sandbox  # noqa: E402
from scoring.score import score_task  # noqa: E402


def _run(task, actions, seed=0):
    sub = ModelSubmission(actions=[
        Action(action_id="a%d" % i, tool=tool, args=args)
        for i, (tool, args) in enumerate(actions)
    ])
    return Sandbox(task).run(sub, model_id="m", run_index=0, seed=seed)


def test_legacy_at_action_index_fault_still_fails_once():
    task = Task(
        task_id="legacy_fault", dimension="U4",
        tools=[{"name": "set_x", "writes": ["x"],
                "effect": {"type": "set", "target": "x", "value": 1}}],
        initial_state={"x": 0},
        fault_injection=[{"id": "f0", "at_action_index": 0,
                          "recover_milestone": "M_x"}],
        milestones=[{"id": "M_x", "predicate": {"op": "eq", "path": "x", "value": 1},
                     "provenance": ["x"], "is_recovery": True}],
        success_predicates=[{"op": "eq", "path": "x", "value": 1}],
    )
    tr = _run(task, [("set_x", {}), ("set_x", {})])
    calls = [e for e in tr.events if e.type == "tool_call"]
    assert [c.status for c in calls] == ["error", "ok"]
    assert not [e for e in tr.events if e.type == "fault"]
    assert tr.final_state["x"] == 1


def test_trigger_duration_tool_scoped_fault():
    task = Task(
        task_id="trigger_fault", dimension="U4",
        tools=[
            {"name": "set_ready", "writes": ["phase.ready"],
             "effect": {"type": "set", "target": "phase.ready", "value": True}},
            {"name": "write_target", "writes": ["target"],
             "effect": {"type": "set", "target": "target", "value": "done"}},
        ],
        initial_state={"phase": {"ready": False}, "target": "pending"},
        fault_injection=[{
            "id": "target_down",
            "trigger": {"op": "eq", "path": "phase.ready", "value": True},
            "duration": 2,
            "tool": "write_target",
            "recover_milestone": "M_target",
        }],
        milestones=[{"id": "M_target",
                     "predicate": {"op": "eq", "path": "target", "value": "done"},
                     "provenance": ["target"], "is_recovery": True}],
        success_predicates=[{"op": "eq", "path": "target", "value": "done"}],
    )
    tr = _run(task, [("set_ready", {}), ("write_target", {}),
                     ("write_target", {}), ("write_target", {})])
    calls = [e for e in tr.events if e.type == "tool_call"]
    assert [c.status for c in calls] == ["ok", "error", "error", "ok"]
    assert tr.final_state["target"] == "done"
    assert any(e.type == "fault" and getattr(e, "fault_id", None) == "target_down"
               for e in tr.events)


def test_delayed_env_drift_is_seed_deterministic():
    task = Task(
        task_id="delayed_drift", dimension="U4",
        tools=[{"name": "tick", "writes": [], "effect": None}],
        initial_state={"drift": {"value": "init"}},
        env_events=[{
            "id": "drift_once",
            "after_action_index": 1,
            "visible_after": 1,
            "effect": {"type": "set", "target": "drift.value", "value": "unused"},
            "drift": {"target": "drift.value", "choices": ["a", "b", "c"]},
        }],
    )
    tr1 = _run(task, [("tick", {}), ("tick", {})], seed=42)
    tr2 = _run(task, [("tick", {}), ("tick", {})], seed=42)
    assert tr1.final_state["drift"]["value"] == tr2.final_state["drift"]["value"]
    assert tr1.final_state["drift"]["value"] in {"a", "b", "c"}
    assert any(getattr(e, "delayed_visible", False) for e in tr1.events)


def test_recovery_quality_components_are_recorded():
    task = Task(
        task_id="rollback_quality", dimension="U4",
        tools=[
            {"name": "set_target", "writes": ["target"],
             "effect": {"type": "set", "target": "target", "value": "ready"}},
            {"name": "inspect_fault", "writes": [], "effect": None},
            {"name": "rollback_compensate", "writes": ["rollback.done"],
             "effect": {"type": "set", "target": "rollback.done", "value": True}},
            {"name": "finalize", "writes": ["final.done"],
             "effect": {"type": "set", "target": "final.done", "value": True}},
        ],
        initial_state={"target": "pending", "rollback": {"done": False},
                       "final": {"done": False}},
        fault_injection=[{
            "id": "final_fail",
            "at_action_index": 1,
            "tool": "finalize",
            "recover_milestone": "M_final",
            "rollback_required": True,
            "requires_diagnosis": True,
            "rollback_tool": "rollback_compensate",
            "diagnosis_tool": "inspect_fault",
            "recovery_latency_window": 4,
        }],
        milestones=[
            {"id": "M_target", "predicate": {"op": "eq", "path": "target", "value": "ready"},
             "provenance": ["target"]},
            {"id": "M_rollback",
             "predicate": {"op": "eq", "path": "rollback.done", "value": True},
             "deps": ["M_target"], "provenance": ["rollback.done"]},
            {"id": "M_final", "predicate": {"op": "eq", "path": "final.done", "value": True},
             "deps": ["M_rollback"], "provenance": ["final.done"], "is_recovery": True},
        ],
        success_predicates=[{"op": "eq", "path": "final.done", "value": True}],
        difficulty_knobs={"regression_free_window": 0},
    )
    tr = _run(task, [("set_target", {}), ("finalize", {}), ("inspect_fault", {}),
                     ("rollback_compensate", {}), ("finalize", {})])
    rec = score_task(task, tr)
    assert rec["success"] is True
    assert rec["n_faults"] == 1
    assert 0.0 < rec["recovery_latency"] < 1.0
    assert rec["rollback_correctness"] == 1.0
    assert rec["post_recovery_stability"] == 1.0
    assert rec["regression_free_window"] == 1.0
    assert rec["recovery_detail"]["events"][0]["diagnosis_satisfied"] == 1.0
