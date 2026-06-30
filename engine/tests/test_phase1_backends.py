import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters import _observe  # noqa: E402
from sandbox import Sandbox  # noqa: E402
from schema import Action, EffectSpec, MilestoneSpec, ModelSubmission, PredicateSpec, Task, ToolSpec  # noqa: E402
from scoring.score import score_task  # noqa: E402
from tool_backends import InMemoryRecordReplayStore, available_backends  # noqa: E402


def _submit_tool(name="submit_choice"):
    return ToolSpec(
        name=name,
        writes=["out.selected"],
        effect=EffectSpec(type="set", target="out.selected", value_from="choice"),
    )


def test_phase1_backends_register():
    assert {"filesystem", "sqlite", "browser", "http_mock"}.issubset(set(available_backends()))


def test_filesystem_observation_and_signed_provenance():
    task = Task(
        task_id="phase1_fs",
        dimension="U1",
        instruction="read file",
        initial_state={"fs": {"files": {"note.txt": "alpha beta"}}},
        tools=[
            ToolSpec(
                name="read_note",
                backend="filesystem",
                permissions=["fs:read"],
                path="note.txt",
                observable_result_paths=["observations.note"],
            )
        ],
        milestones=[
            MilestoneSpec(
                id="M_read",
                predicate=PredicateSpec(op="exists", path="observations.note"),
                provenance=["observations.note"],
            )
        ],
    )
    trace = Sandbox(task).run(
        ModelSubmission(actions=[Action(action_id="r", tool="read_note", args={})]),
        model_id="m",
        seed=123,
    )
    ev = [e for e in trace.events if e.type == "tool_call"][0]
    assert ev.status == "ok"
    assert ev.observation["content"] == "alpha beta"
    assert ev.event_signature
    assert trace.provenance["observations.note"].startswith("action:r#")
    obs = _observe(task, trace, 1)
    assert "alpha beta" in obs and "result" in obs
    assert "Current state: {}" in obs


def test_sqlite_browser_and_http_mock_return_structured_results():
    sqlite_task = Task(
        task_id="phase1_sqlite",
        dimension="U2",
        initial_state={"sqlite": {"tables": {"items": [{"id": "A", "ok": 1}, {"id": "B", "ok": 0}]}}},
        tools=[
            ToolSpec(
                name="query_items",
                backend="sqlite",
                permissions=["db:read"],
                sql="SELECT id FROM items WHERE ok = 1 ORDER BY id",
                observable_result_paths=["observations.rows"],
            )
        ],
    )
    tr_sql = Sandbox(sqlite_task).run(
        ModelSubmission(actions=[Action(action_id="q", tool="query_items", args={})]),
        model_id="m",
        seed=7,
    )
    assert tr_sql.events[0].observation["rows"] == [{"id": "A"}]

    browser_task = Task(
        task_id="phase1_browser",
        dimension="U2",
        initial_state={"browser": {"pages": {"/x.html": "<title>X</title><p class='pick'>winner</p>"}}},
        tools=[
            ToolSpec(
                name="read_page",
                backend="browser",
                url="/x.html",
                selector=".pick",
                observable_result_paths=["observations.dom"],
            )
        ],
    )
    tr_dom = Sandbox(browser_task).run(
        ModelSubmission(actions=[Action(action_id="b", tool="read_page", args={})]),
        model_id="m",
        seed=7,
    )
    assert tr_dom.events[0].observation["title"] == "X"
    assert tr_dom.events[0].observation["matches"][0]["text"] == "winner"

    http_task = Task(
        task_id="phase1_http",
        dimension="U2",
        initial_state={"http": {"routes": {"GET /offers": {"status": 200, "json": {"id": "OFF-1"}}}}},
        tools=[
            ToolSpec(
                name="get_offer",
                backend="http_mock",
                method="GET",
                url="/offers",
                observable_result_paths=["observations.offer"],
            )
        ],
    )
    tr_http = Sandbox(http_task).run(
        ModelSubmission(actions=[Action(action_id="h", tool="get_offer", args={})]),
        model_id="m",
        seed=7,
    )
    assert tr_http.events[0].observation["json"] == {"id": "OFF-1"}


def test_http_mock_record_replay_event_signature_is_deterministic():
    task = Task(
        task_id="phase1_http_rr",
        dimension="U2",
        initial_state={"http": {"routes": {"GET /quote": {"status": 200, "json": {"quote": 42}}}}},
        tools=[
            ToolSpec(
                name="get_quote",
                backend="http_mock",
                method="GET",
                url="/quote",
                observable_result_paths=["observations.quote"],
            )
        ],
    )
    sub = ModelSubmission(actions=[Action(action_id="h", tool="get_quote", args={})])
    store = InMemoryRecordReplayStore()
    tr_record = Sandbox(task, record_replay_mode="record", record_replay_store=store).run(
        sub, model_id="m", seed=99
    )
    tr_replay = Sandbox(task, record_replay_mode="replay", record_replay_store=store).run(
        sub, model_id="m", seed=99
    )
    sig_record = [e.event_signature for e in tr_record.events if e.type == "tool_call"]
    sig_replay = [e.event_signature for e in tr_replay.events if e.type == "tool_call"]
    assert sig_record == sig_replay
    assert store.as_dict()


def test_direct_submit_without_real_tool_gets_no_backend_provenance_credit():
    task = Task(
        task_id="phase1_no_bypass",
        dimension="U2",
        initial_state={"sqlite": {"tables": {"choices": [{"id": "A", "score": 10}]}}},
        tools=[
            ToolSpec(
                name="query_choices",
                backend="sqlite",
                permissions=["db:read"],
                sql="SELECT id, score FROM choices ORDER BY score DESC",
                observable_result_paths=["observations.rows"],
            ),
            _submit_tool(),
        ],
        milestones=[
            MilestoneSpec(
                id="M_rows",
                predicate=PredicateSpec(op="exists", path="observations.rows"),
                provenance=["observations.rows"],
            ),
            MilestoneSpec(
                id="M_select",
                predicate=PredicateSpec(op="eq", path="out.selected", value="A"),
                deps=["M_rows"],
                provenance=["out.selected"],
            ),
        ],
        success_predicates=[PredicateSpec(op="eq", path="out.selected", value="A")],
    )
    direct = ModelSubmission(
        actions=[Action(action_id="s", tool="submit_choice", args={"choice": "A"})]
    )
    rec = score_task(task, Sandbox(task).run(direct, model_id="m", seed=0))
    assert rec["raw_success"] is True
    assert rec["milestone_diag"]["M_rows"]["prov_ok"] == 0.0
    assert rec["milestone_diag"]["M_select"]["deps_ok"] == 0.0
    assert rec["process"] < 1.0
