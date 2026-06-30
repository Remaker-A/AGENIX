"""
engine/tool_backends/base.py — 真实工具后端的统一抽象（Phase 0 接口骨架）。

地基目标（与 verifier-first / 可复现护栏对齐）：
- **统一接口**：所有后端（dsl 默认 + 真实 filesystem/sqlite/browser/http_mock）都实现
  ``ToolBackend.execute``，返回结构化 ``BackendResult``（含 ``List[StateDiff]``），由 sandbox
  统一写入 ``trace.provenance``。
- **执行隔离**：``setup``/``teardown`` 提供临时目录/连接/容器生命周期钩子；``workdir`` 隔离。
- **确定性**：强制 ``seed``；``RecordReplayStore`` 把外部交互录制/回放，杜绝"假设外部 API 可复现"。
- **provenance 不被削弱**：``diff_hash()`` 产出确定性"后端事件签名"，真实后端把它并入 provenance
  （形如 ``action:<id>#<hash>``），仍以 ``action:`` 前缀通过 milestone 因果门控，同时把"路径写入"
  升级为"后端事件签名 + diff 哈希"，meta-test 可据此识别"绕过工具直接提交"。

注意：Phase 0 只落"接口 + 注册表 + dsl 默认后端"；真实后端在 Phase 1 落地（平行实现本接口并自注册）。
"""
from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from schema import Action, StateDiff, Task, ToolSpec


# --------------------------------------------------------------------------- #
# 确定性辅助：规范化 JSON + diff 哈希（record/replay 键 + 后端事件签名）
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    """确定性 JSON 序列化（排序键 / 紧凑分隔 / 兜底 str），用于 record-replay 键与 diff 哈希。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


def diff_hash(diffs: List[StateDiff], *, tool: Optional[str] = None,
              args: Optional[Dict[str, Any]] = None, extra: Any = None) -> str:
    """对一次后端调用产生的结构化 diff（+工具名/入参/附加证据）求确定性短哈希。

    作为"后端事件签名"并入 provenance，使因果门控从"路径写入"升级为"事件签名 + diff 哈希"。
    """
    payload = {
        "tool": tool,
        "args": args or {},
        "diffs": [
            {"path": d.path, "new_value": d.new_value, "provenance": d.provenance}
            for d in diffs
        ],
        "extra": extra,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 执行上下文 / 结果
# --------------------------------------------------------------------------- #
@dataclass
class BackendContext:
    """单次工具调用传给后端的执行上下文（由 sandbox 构造）。

    后端可读/改 ``state``，但任何状态变更都必须同时反映为返回的 ``StateDiff``——否则会击穿
    provenance 因果门控（这是 verifier-first 的立身之本）。
    """
    task: Task
    tool: ToolSpec
    action: Action
    args: Dict[str, Any]
    state: Dict[str, Any]              # 当前可变 state（最终落为 trace.final_state）
    source: str                        # provenance 来源，形如 "action:<action_id>"
    action_index: int = 0              # 第几个 agent 动作（0-based）
    seed: int = 0
    workdir: Optional[str] = None      # 隔离工作目录（真实后端落临时文件用）
    mode: str = "live"                 # live | record | replay
    store: Optional["RecordReplayStore"] = None
    # dsl 委托回调：sandbox._apply_effect（保证 dsl 语义唯一来源、字节级一致）
    apply_dsl_effect: Optional[Callable[..., List[StateDiff]]] = None
    # 通用 state 读写工具（sandbox._get_path / _set_path），供真实后端按路径回写 observable 结果
    get_path: Optional[Callable[[Dict[str, Any], str], Any]] = None
    set_path: Optional[Callable[[Dict[str, Any], str, Any], None]] = None
    sandbox: Any = None

    def replay_key(self) -> str:
        """该调用的确定性 record/replay 键：工具名 + 规范化入参 + 动作序号 + seed。"""
        return canonical_json({
            "tool": self.tool.name,
            "args": self.args,
            "action_index": self.action_index,
            "seed": self.seed,
        })


@dataclass
class BackendResult:
    """后端调用的结构化产出。sandbox 据此写 provenance、拼 TraceEvent、回传 _observe。"""
    diffs: List[StateDiff] = field(default_factory=list)  # 结构化状态变更（统一契约）
    observation: Any = None          # 回传给 _observe 的真实结果（stdout/SQL row/DOM 摘要/文件 diff）
    status: str = "ok"               # ok | error
    event_signature: Optional[str] = None  # 后端事件签名（diff_hash），并入 provenance
    raw_response: Any = None         # 原始响应（落盘 + 哈希；真实横评保留可评对象）
    replay_key: Optional[str] = None
    error: Optional[str] = None      # 错误信息（status=error 时）


# --------------------------------------------------------------------------- #
# record / replay 抽象（真实后端外部交互的确定性地基）
# --------------------------------------------------------------------------- #
class RecordReplayStore(abc.ABC):
    """record/replay 抽象：真实后端的外部交互必须可录制/回放以保证确定性。"""

    @abc.abstractmethod
    def has(self, key: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def record(self, key: str, response: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def replay(self, key: str) -> Any:
        raise NotImplementedError


class InMemoryRecordReplayStore(RecordReplayStore):
    """最简内存实现（默认）。真实持久化（JSONL 落盘 + 哈希）由 Phase 1 后端按需扩展。"""

    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = dict(initial or {})

    def has(self, key: str) -> bool:
        return key in self._data

    def record(self, key: str, response: Any) -> None:
        self._data[key] = response

    def replay(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError("no recorded response for replay key: %s" % key)
        return self._data[key]

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)


# --------------------------------------------------------------------------- #
# 后端抽象基类
# --------------------------------------------------------------------------- #
class ToolBackend(abc.ABC):
    """所有工具后端的统一基类。

    生命周期：sandbox 在需要时 ``create_backend(name, ...)`` -> ``setup(task)``
    -> ``execute(ctx)``（每次工具调用）-> ``teardown()``（run 结束）。
    子类必须实现 ``execute()``；``setup``/``teardown`` 默认空操作（dsl 无需隔离）。
    """

    #: 注册名（与 ToolSpec.backend 对应）
    name: str = "abstract"
    #: 该后端是否本身确定性（真实联网后端应置 False，并依赖 record/replay 达成确定性）
    deterministic: bool = True
    #: 是否需要隔离工作目录
    requires_workdir: bool = False

    def __init__(self, *, seed: int = 0, workdir: Optional[str] = None,
                 mode: str = "live", store: Optional[RecordReplayStore] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.seed = seed
        self.workdir = workdir
        self.mode = mode  # live | record | replay
        self.store = store if store is not None else InMemoryRecordReplayStore()
        self.config = dict(config or {})

    # -- 生命周期钩子（执行隔离）-- #
    def setup(self, task: Task) -> None:
        """准备隔离环境（临时目录 / seed DB / 连接）。默认空操作。"""
        return None

    def teardown(self) -> None:
        """清理隔离环境（删临时目录 / 关连接 / flush record store）。默认空操作。"""
        return None

    # -- 核心执行 -- #
    @abc.abstractmethod
    def execute(self, ctx: BackendContext) -> BackendResult:
        """执行一次工具调用，返回结构化 ``BackendResult``（含 ``List[StateDiff]``）。"""
        raise NotImplementedError

    # -- record/replay 工具方法 -- #
    def resolve(self, key: str, live_fn: Callable[[], Any]) -> Any:
        """按 ``mode`` 解析一次外部交互结果（真实后端调用外部资源时用）：

        - ``replay``：必须命中录制（否则 KeyError，强制确定性）。
        - ``record``：实跑并录制。
        - ``live`` ：直接实跑（不录制）。
        """
        if self.mode == "replay":
            return self.store.replay(key)
        result = live_fn()
        if self.mode == "record":
            self.store.record(key, result)
        return result

    # -- provenance 签名（不削弱因果门控）-- #
    def sign(self, ctx: BackendContext, diffs: List[StateDiff],
             extra: Any = None) -> str:
        """后端事件签名（diff 哈希）。"""
        return diff_hash(diffs, tool=ctx.tool.name, args=ctx.args, extra=extra)

    def signed_source(self, ctx: BackendContext, diffs: List[StateDiff],
                      extra: Any = None) -> str:
        """把后端事件签名并入 provenance 来源串：``action:<id>#<hash>``。

        仍以 ``action:`` 前缀通过 ``scoring/milestone._provenance_ok`` 的因果门控（**不削弱**），
        同时把"路径写入"升级为"后端事件签名 + diff 哈希"。真实后端（Phase 1）在产出 StateDiff
        时用本方法作为 ``provenance``；dsl 默认后端不使用（保持历史 ``action:<id>`` 字节级一致）。
        """
        return "%s#%s" % (ctx.source, self.sign(ctx, diffs, extra=extra))


# --------------------------------------------------------------------------- #
# 注册表（sandbox 据 ToolSpec.backend 选择后端）
# --------------------------------------------------------------------------- #
_REGISTRY: Dict[str, Type[ToolBackend]] = {}


def register_backend(name: str, cls: Type[ToolBackend], *,
                     override: bool = False) -> None:
    """注册一个工具后端。真实后端模块导入时自注册（见 dsl.py）。"""
    if not name:
        raise ValueError("backend name must be non-empty")
    if not isinstance(cls, type) or not issubclass(cls, ToolBackend):
        raise TypeError("backend cls must subclass ToolBackend: %r" % (cls,))
    if name in _REGISTRY and not override and _REGISTRY[name] is not cls:
        raise ValueError("backend already registered: %s" % name)
    _REGISTRY[name] = cls


def get_backend_class(name: str) -> Type[ToolBackend]:
    if name not in _REGISTRY:
        raise KeyError("unknown tool backend %r; registered=%s"
                       % (name, available_backends()))
    return _REGISTRY[name]


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def available_backends() -> List[str]:
    return sorted(_REGISTRY)


def create_backend(name: str, **kwargs: Any) -> ToolBackend:
    """实例化已注册后端；未知后端抛出含可用清单的 KeyError。"""
    return get_backend_class(name)(**kwargs)
