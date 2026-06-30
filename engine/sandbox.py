"""
确定性沙箱：执行模型动作序列 -> 产出带 provenance 的 Trace + 终态。

关键点（CP5 因果门控的地基）：
- 每个工具声明 writes（可写 state 路径）与 effect（声明式效应）。
- 每次写入记录 provenance[path] = "action:<id>"；初始态为 "init:<path>"；
  自主环境事件为 "env:<id>"。里程碑 by-agent 门控据此判定。
- 注入故障：某动作首次调用失败（status=error，不产生效应），考验恢复。
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional

from schema import (Task, Trace, TraceEvent, StateDiff, Action, ModelSubmission,
                    ToolSpec, EffectSpec, EnvEventSpec, FaultSpec)
from tool_backends import (BackendContext, RecordReplayStore, ToolBackend,
                           create_backend)
from dsl import eval_predicate


def _set_path(state: Dict[str, Any], path: str, value: Any) -> None:
    if path.startswith("state."):
        path = path[len("state."):]
    parts = path.split(".")
    cur = state
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _get_path(state: Dict[str, Any], path: str) -> Any:
    if path.startswith("state."):
        path = path[len("state."):]
    cur = state
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def _args_norm_hash(tool: str, args: Dict[str, Any]) -> str:
    blob = tool + "|" + json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class Sandbox:
    """单任务的确定性运行时。"""

    def __init__(self, task: Task, *, workdir: Optional[str] = None,
                 seed: int = 0, record_replay_mode: str = "live",
                 record_replay_store: Optional[RecordReplayStore] = None):
        self.task = task
        self.tools: Dict[str, ToolSpec] = {t.name: t for t in task.tools}
        self.state: Dict[str, Any] = copy.deepcopy(task.initial_state)
        self.provenance: Dict[str, str] = {}
        # 初始态的所有叶子路径标记 provenance="init"
        for leaf in self._leaf_paths(self.state):
            self.provenance[leaf] = "init:" + leaf
        self._faults_fired = set()
        self._env_events_fired = set()
        self._active_faults: Dict[str, Dict[str, Any]] = {}
        self._pending_env_events: List[Dict[str, Any]] = []
        # --- Phase 0 backend registry 接线（全部可选，默认与历史完全一致）---
        # dsl 为默认后端、行为零变化；真实后端按 ToolSpec.backend 懒加载（Phase 1+）。
        self._seed = seed
        self._workdir = workdir
        self._rr_mode = record_replay_mode          # live | record | replay
        self._rr_store = record_replay_store
        self._backends: Dict[str, ToolBackend] = {}

    def _leaf_paths(self, d: Dict[str, Any], prefix: str = "") -> List[str]:
        out = []
        for k, v in d.items():
            p = (prefix + "." + k) if prefix else k
            if isinstance(v, dict) and v:
                out.extend(self._leaf_paths(v, p))
            else:
                out.append(p)
        return out

    def _apply_effect(self, eff: EffectSpec, args: Dict[str, Any],
                      source: str) -> List[StateDiff]:
        if eff.value_from is not None:
            value = args.get(eff.value_from)
        else:
            value = eff.value
        target = eff.target
        diffs: List[StateDiff] = []
        if eff.type == "set":
            _set_path(self.state, target, value)
        elif eff.type == "append":
            cur = _get_path(self.state, target)
            if not isinstance(cur, list):
                cur = []
            cur = list(cur) + [value]
            _set_path(self.state, target, cur)
        elif eff.type == "inc":
            cur = _get_path(self.state, target) or 0
            value = (cur or 0) + (value or 0)
            _set_path(self.state, target, value)
        elif eff.type == "merge":
            cur = _get_path(self.state, target)
            if not isinstance(cur, dict):
                cur = {}
            if isinstance(value, dict):
                cur = dict(cur); cur.update(value)
            _set_path(self.state, target, cur)
        else:
            raise ValueError("unknown effect type: %s" % eff.type)
        norm = target[len("state."):] if target.startswith("state.") else target
        self.provenance[norm] = source
        diffs.append(StateDiff(path=norm, new_value=value, provenance=source))
        return diffs

    # ------------------------------------------------------------------ #
    # backend registry：据 ToolSpec.backend 选择后端，统一收口 provenance
    # ------------------------------------------------------------------ #
    def _get_backend(self, name: Optional[str]) -> ToolBackend:
        """懒加载并缓存后端实例（每个 Sandbox 一份；dsl 无状态、真实后端持隔离资源）。"""
        name = name or "dsl"
        backend = self._backends.get(name)
        if backend is None:
            backend = create_backend(name, seed=self._seed, workdir=self._workdir,
                                     mode=self._rr_mode, store=self._rr_store)
            backend.setup(self.task)
            self._backends[name] = backend
        return backend

    def _record_provenance(self, diffs: List[StateDiff]) -> None:
        """把后端返回的结构化 diff 统一写入 provenance（CP5 因果门控的唯一收口）。

        dsl 后端的 ``_apply_effect`` 已写过同值，这里幂等重写——保证"所有后端都返回结构化
        StateDiff 并写 provenance"的统一契约，且对 dsl 路径零行为变化。
        """
        for d in diffs:
            self.provenance[d.path] = d.provenance

    # ------------------------------------------------------------------ #
    # Phase 3 event scheduler
    # ------------------------------------------------------------------ #
    def _extra(self, obj: Any, key: str, default: Any = None) -> Any:
        val = getattr(obj, key, None)
        if val is not None:
            return val
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict) and key in extra:
            return extra[key]
        return default

    def _field_was_set(self, obj: Any, key: str) -> bool:
        fields = getattr(obj, "model_fields_set", None)
        return key in fields if fields is not None else True

    def _current_trace(self, events: List[TraceEvent]) -> Trace:
        return Trace(task_id=self.task.task_id, model_id="_sandbox", events=list(events),
                     final_state=copy.deepcopy(self.state),
                     provenance=dict(self.provenance))

    def _event_triggered(self, spec: Any, index_field: str, current_index: int,
                         events: List[TraceEvent]) -> bool:
        trig = getattr(spec, "trigger", None)
        index_match = False
        # trigger-only specs should not accidentally fire at default index 0.
        if trig is None or self._field_was_set(spec, index_field):
            index_match = int(getattr(spec, index_field, 0) or 0) == current_index
        trigger_match = False
        if trig is not None:
            trigger_match = eval_predicate(trig, self.state, self._current_trace(events)) >= 1.0
        return bool(index_match or trigger_match)

    def _tool_matches_fault(self, fault: FaultSpec, tool_name: str) -> bool:
        tools = (self._extra(fault, "tools") or self._extra(fault, "affected_tools")
                 or self._extra(fault, "tool") or self._extra(fault, "tool_name"))
        if tools is None:
            return True
        if isinstance(tools, str):
            return tool_name == tools
        try:
            return tool_name in set(tools)
        except TypeError:
            return False

    def _is_legacy_fault(self, fault: FaultSpec) -> bool:
        extras = getattr(fault, "model_extra", None) or {}
        return (fault.trigger is None and fault.duration is None
                and not fault.silent_corruption and not fault.rollback_required
                and not fault.requires_diagnosis and not extras)

    def _fault_duration(self, fault: FaultSpec) -> int:
        # None keeps historical single-action transient behavior.
        return max(1, int(fault.duration)) if fault.duration is not None else 1

    def _activate_faults(self, action_index: int, act: Action,
                         events: List[TraceEvent]) -> None:
        for fault in self.task.fault_injection:
            if fault.id in self._faults_fired:
                continue
            if not self._event_triggered(fault, "at_action_index", action_index, events):
                continue
            self._faults_fired.add(fault.id)
            self._active_faults[fault.id] = {
                "spec": fault,
                "remaining": self._fault_duration(fault),
                "started_action_index": action_index,
            }
            if not self._is_legacy_fault(fault):
                ev = TraceEvent(idx=len(events), type="fault", tool=act.tool,
                                status="ok")
                ev.fault_id = fault.id
                ev.kind = fault.kind
                ev.action_index = action_index
                ev.duration = fault.duration
                ev.silent_corruption = bool(fault.silent_corruption)
                ev.rollback_required = bool(fault.rollback_required)
                ev.requires_diagnosis = bool(fault.requires_diagnosis)
                events.append(ev)

    def _active_fault_for_action(self, act: Action) -> Optional[FaultSpec]:
        for fid in sorted(self._active_faults):
            item = self._active_faults[fid]
            fault = item["spec"]
            if item["remaining"] > 0 and self._tool_matches_fault(fault, act.tool):
                return fault
        return None

    def _consume_fault_step(self, fault: FaultSpec) -> None:
        item = self._active_faults.get(fault.id)
        if not item:
            return
        item["remaining"] -= 1
        if item["remaining"] <= 0:
            self._active_faults.pop(fault.id, None)

    def _event_effect(self, spec: Any) -> Optional[EffectSpec]:
        eff = (self._extra(spec, "corruption_effect") or self._extra(spec, "effect")
               or self._extra(spec, "drift_effect"))
        if eff is None:
            return None
        if isinstance(eff, EffectSpec):
            return eff
        return EffectSpec(**eff)

    def _materialize_drift_effect(self, spec: Any, action_index: int) -> Optional[EffectSpec]:
        drift = self._extra(spec, "drift")
        if not isinstance(drift, dict):
            return None
        target = drift.get("target")
        if not target:
            return None
        values = drift.get("values") or drift.get("choices")
        if values:
            vals = list(values)
            blob = "%s|%s|%s" % (self._seed, getattr(spec, "id", ""), action_index)
            ix = int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8], 16) % len(vals)
            value = vals[ix]
        else:
            lo = int(drift.get("min", 0)); hi = int(drift.get("max", lo))
            span = max(1, hi - lo + 1)
            blob = "%s|%s|%s" % (self._seed, getattr(spec, "id", ""), action_index)
            value = lo + (int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8], 16) % span)
        return EffectSpec(type=drift.get("type", "set"), target=target, value=value)

    def _apply_env_spec(self, ev: EnvEventSpec, events: List[TraceEvent],
                        action_index: int) -> None:
        eff = self._materialize_drift_effect(ev, action_index) or ev.effect
        diffs = self._apply_effect(eff, {}, "env:" + ev.id)
        if ev.silent_corruption:
            return
        te = TraceEvent(idx=len(events), type="env_event", tool=None, diffs=diffs)
        te.env_event_id = ev.id
        te.action_index = action_index
        te.rollback_required = bool(ev.rollback_required)
        te.requires_diagnosis = bool(ev.requires_diagnosis)
        events.append(te)

    def _flush_pending_env_events(self, action_index: int,
                                  events: List[TraceEvent]) -> None:
        ready = [p for p in self._pending_env_events if p["due_index"] <= action_index]
        self._pending_env_events = [p for p in self._pending_env_events
                                    if p["due_index"] > action_index]
        for p in ready:
            self._apply_env_spec(p["spec"], events, action_index)

    def _fire_env_events(self, after_index: int, events: List[TraceEvent]) -> None:
        self._flush_pending_env_events(after_index, events)
        for ev in self.task.env_events:
            if ev.id in self._env_events_fired:
                continue
            if not self._event_triggered(ev, "after_action_index", after_index, events):
                continue
            self._env_events_fired.add(ev.id)
            delay = (self._extra(ev, "visible_after")
                     or self._extra(ev, "delay_actions")
                     or self._extra(ev, "delay")
                     or 0)
            delay = max(0, int(delay))
            if delay > 0:
                self._pending_env_events.append({"spec": ev, "due_index": after_index + delay})
                if not ev.silent_corruption:
                    te = TraceEvent(idx=len(events), type="env_event", tool=None,
                                    status="ok", diffs=[])
                    te.env_event_id = ev.id
                    te.action_index = after_index
                    te.delayed_visible = True
                    te.due_action_index = after_index + delay
                    events.append(te)
            else:
                self._apply_env_spec(ev, events, after_index)

    def run(self, submission: ModelSubmission, model_id: str,
            run_index: int = 0, seed: int = 0) -> Trace:
        self._seed = seed  # 供后端隔离/确定性使用（run 级 seed 优先于 __init__ 默认）
        events: List[TraceEvent] = []
        n_actions = 0

        # 触发"动作0之前"的环境事件（after_action_index=0 表示开局）
        self._fire_env_events(0, events)

        for act in submission.actions[: self.task.budget_max_actions]:
            tool = self.tools.get(act.tool)
            norm_hash = _args_norm_hash(act.tool, act.args)
            self._activate_faults(n_actions, act, events)
            idx = len(events)
            # 注入故障：兼容旧 at_action_index；新调度器可按 trigger/tool/duration 命中。
            fault = self._active_fault_for_action(act)
            if fault is not None and not fault.silent_corruption:
                self._consume_fault_step(fault)
                idx = len(events)
                tev = TraceEvent(idx=idx, type="tool_call", tool=act.tool,
                                 args=act.args, status="error",
                                 args_norm_hash=norm_hash)
                tev.action_id = act.action_id
                tev.action_index = n_actions
                tev.fault_id = fault.id
                tev.fault_kind = fault.kind
                tev.rollback_required = bool(fault.rollback_required)
                tev.requires_diagnosis = bool(fault.requires_diagnosis)
                events.append(tev)
                n_actions += 1
                self._fire_env_events(n_actions, events)
                continue
            silent_fault = fault if (fault is not None and fault.silent_corruption) else None

            if tool is None:
                # 调用了不存在/不可用的工具（幻觉工具）
                events.append(TraceEvent(idx=idx, type="tool_call", tool=act.tool,
                                         args=act.args, status="error",
                                         args_norm_hash=norm_hash))
                events[-1].action_id = act.action_id
                events[-1].action_index = n_actions
                n_actions += 1
                self._fire_env_events(n_actions, events)
                continue

            # 经 backend registry 派发：dsl（默认）委托 _apply_effect、行为零变化；
            # 真实后端（Phase 1+）平行实现同一接口，统一返回结构化 StateDiff。
            source = "action:" + act.action_id
            backend = self._get_backend(getattr(tool, "backend", None))
            ctx = BackendContext(
                task=self.task, tool=tool, action=act, args=act.args,
                state=self.state, source=source, action_index=n_actions,
                seed=self._seed, workdir=self._workdir, mode=self._rr_mode,
                store=self._rr_store, apply_dsl_effect=self._apply_effect,
                get_path=_get_path, set_path=_set_path, sandbox=self,
            )
            result = backend.execute(ctx)
            # provenance 因果门控的唯一收口：所有后端的 diff 都在此写入（不被削弱）。
            self._record_provenance(result.diffs)
            diffs = list(result.diffs)
            if silent_fault is not None:
                self._consume_fault_step(silent_fault)
                eff = (self._materialize_drift_effect(silent_fault, n_actions)
                       or self._event_effect(silent_fault))
                if eff is not None:
                    diffs.extend(self._apply_effect(eff, act.args, "env:" + silent_fault.id))
            ev = TraceEvent(idx=idx, type="tool_call", tool=act.tool,
                            args=act.args, status=result.status,
                            diffs=diffs, args_norm_hash=norm_hash)
            ev.action_id = act.action_id
            ev.action_index = n_actions
            if silent_fault is not None:
                ev.fault_id = silent_fault.id
                ev.fault_kind = silent_fault.kind
                ev.silent_corruption = True
                ev.rollback_required = bool(silent_fault.rollback_required)
                ev.requires_diagnosis = bool(silent_fault.requires_diagnosis)
            # 真实后端回传的 observation/原始响应/事件签名挂在事件上（extra=allow）；
            # dsl 默认后端三者均为 None -> 事件与历史字节级一致。
            if result.observation is not None:
                ev.observation = result.observation
            if result.event_signature is not None:
                ev.event_signature = result.event_signature
            events.append(ev)
            n_actions += 1
            self._fire_env_events(n_actions, events)

        events.append(TraceEvent(idx=len(events), type="final"))
        trace = Trace(task_id=self.task.task_id, model_id=model_id,
                      run_index=run_index, seed=seed, events=events,
                      final_state=copy.deepcopy(self.state),
                      provenance=dict(self.provenance),
                      n_agent_actions=n_actions,
                      cost_actions=float(n_actions), submission=submission)
        # 清理后端隔离资源（dsl 为空操作；真实后端删临时目录/关连接/flush 录制）。
        self._teardown_backends()
        return trace

    def _teardown_backends(self) -> None:
        for backend in self._backends.values():
            try:
                backend.teardown()
            except Exception:  # noqa: BLE001 - teardown 失败不得影响已生成的 Trace
                pass
