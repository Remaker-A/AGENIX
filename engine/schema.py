"""
AGENIX-Engine 统一 Schema（pydantic v2，Python 3.8 兼容）。

本模块定义 Task / Trace / 评分结果的结构化类型。设计原则（见 spec §设计哲学）：
- verifier-first：一切以"对环境真实状态的程序化谓词"为准。
- 任务声明式（JSON 可序列化）：工具效应、里程碑谓词、安全违规均用小型 DSL 表达，
  由 sandbox 通用解释执行，从而样例任务是纯 JSON。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 谓词 / 效应 DSL（声明式，state 路径用点号 a.b.c）
# --------------------------------------------------------------------------- #
class PredicateSpec(BaseModel):
    """对 (state, trace) 求值的布尔/分级谓词。

    op:
      eq/neq/ge/gt/le/lt : 标量比较 (path vs value)
      approx             : |state(path) - value| / max(|value|,eps) <= tol  -> 数值容差
      set_eq             : 集合相等 (path 列表 vs value 列表)
      jaccard            : 集合 Jaccard 相似度（gradable，返回 [0,1]）
      contains           : path 容器包含 value
      exists             : path 存在且非 None
      tool_called        : trace 中调用过名为 value 的工具（用于安全/认知动作）
      tool_not_called    : trace 中未调用过 value
    """
    op: str
    path: Optional[str] = None
    value: Optional[Any] = None
    tol: float = 0.0
    model_config = {"extra": "allow"}


class EffectSpec(BaseModel):
    """工具对环境状态的声明式效应。"""
    type: str  # set | append | inc | merge
    target: str  # state 路径
    value_from: Optional[str] = None  # 从该工具调用 args 取值
    value: Optional[Any] = None  # 或常量值
    model_config = {"extra": "allow"}


class ToolSpec(BaseModel):
    name: str
    writes: List[str] = Field(default_factory=list)  # 该工具可写的 state 路径集合（provenance 依据）
    effect: Optional[EffectSpec] = None
    is_noise: bool = False  # 干扰工具
    is_honeypot: bool = False  # 蜜罐：调用即 critical 违规

    # ------------------------------------------------------------------ #
    # Phase 0 真实化扩展点（全部 Optional + 默认，向后兼容）。
    # `backend="dsl"`（默认）时 sandbox 行为与历史完全一致；以下字段仅被真实工具后端
    # （filesystem/sqlite/browser/http_mock，Phase 1+）消费，dsl 后端一律忽略。
    # ------------------------------------------------------------------ #
    backend: str = "dsl"  # 执行后端名：dsl(默认) | filesystem | sqlite | browser | http_mock | ...
    input_schema: Optional[Dict[str, Any]] = None   # 入参 JSON-Schema（参数校验 + record/replay 键规范化）
    output_schema: Optional[Dict[str, Any]] = None  # 出参/observation JSON-Schema（结果结构契约）
    permissions: List[str] = Field(default_factory=list)        # 该工具所需能力标签（如 fs:read / db:write / net:get）
    timeout_ms: Optional[int] = None                # 单次调用超时（真实后端执行隔离用，毫秒）
    secret_scopes: List[str] = Field(default_factory=list)      # 允许访问的密钥范围；越界访问即安全违规
    observable_result_paths: List[str] = Field(default_factory=list)  # 工具结果回写并可被 _observe 观测的 state 路径
    side_effect_paths: List[str] = Field(default_factory=list)        # 工具真实副作用涉及的（外部资源/state）路径
    model_config = {"extra": "allow"}


# --------------------------------------------------------------------------- #
# 里程碑 DAG（状态断言 + provenance 因果门控 + GPCM 部分得分 + OR 组）
# --------------------------------------------------------------------------- #
class MilestoneSpec(BaseModel):
    id: str
    weight: float = 1.0
    type: str = "required"  # required | or_group | optional
    or_group: Optional[str] = None  # 同组取 max
    deps: List[str] = Field(default_factory=list)
    gradable: bool = False  # True -> 允许 [0,1] 部分得分（GPCM）
    predicate: PredicateSpec
    # provenance 因果门控：以下 state 路径必须由 agent 动作写入（而非 init/env）
    provenance: List[str] = Field(default_factory=list)
    # 无副作用的"认知型"必要动作：仅需 trace 存在该工具调用（布尔存在性）
    epistemic_action: Optional[str] = None
    is_recovery: bool = False  # 标记为恢复里程碑（对应注入故障）
    model_config = {"extra": "allow"}


class CriticalViolationSpec(BaseModel):
    """critical 安全违规：命中即 hard-zero 不可补偿 + 计入 ASR。"""
    id: str
    predicate: PredicateSpec
    attack_strength: float = 1.0  # 攻击强度标注（不并入能力，仅描述）
    model_config = {"extra": "allow"}


# --------------------------------------------------------------------------- #
# 多模态 grounding（双轨：合成-符号 GT / 真实生态层）
# --------------------------------------------------------------------------- #
class GroundingItem(BaseModel):
    """单个 grounding 判定项。kind 决定使用哪个 typed verifier。"""
    id: str
    kind: str  # closed_id | numeric | iou | cer | minimal_pair | temporal_iou | video_event_id | doc_layout_f1
    track: str = "synthetic"  # synthetic | real
    gold: Any = None
    tol: float = 0.0  # numeric relaxed 相对误差阈值；iou 阈值
    weight: float = 1.0

    # ------------------------------------------------------------------ #
    # Phase 0 真实媒介扩展点（全部 Optional + 默认，向后兼容）。
    # 支撑 Phase 2「真实图/扫描件/网页截图/短视频帧 + typed GT」管线；旧合成项不设这些字段即沿用原行为。
    # ------------------------------------------------------------------ #
    media_type: Optional[str] = None       # image | document | table | webpage | video | audio
    asset_uri: Optional[str] = None        # 真实媒介资产路径/URI（相对 engine_root 或 assets/real）
    annotation_uri: Optional[str] = None   # 人工 typed GT 标注文件路径/URI
    annotation_source: Optional[str] = None  # 标注来源：human | tool:<name> | synthetic（可信度溯源）
    verifier_profile: Optional[str] = None   # 指定 typed verifier 档位（如 ocr_extractor_hi / layout_f1_strict）
    temporal_span: Optional[List[float]] = None  # 视频/音频时间段 [start_s, end_s]（temporal_iou 用）
    bbox_format: Optional[str] = None      # bbox 坐标约定：xywh | xyxy | cxcywh | norm_xywh
    model_config = {"extra": "allow"}


class GroundingSpec(BaseModel):
    items: List[GroundingItem] = Field(default_factory=list)
    model_config = {"extra": "allow"}


# --------------------------------------------------------------------------- #
# 注入故障 / 难度旋钮
# --------------------------------------------------------------------------- #
class FaultSpec(BaseModel):
    id: str
    at_action_index: int = 0  # 在第几个 agent 动作后触发（与 trigger 谓词二选一/可并存）
    kind: str = "transient_fail"
    recover_milestone: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Phase 0 长程事件调度扩展点（全部 Optional + 默认，向后兼容）。
    # 仅声明契约字段；Phase 3 的事件调度器才会消费它们。旧任务只用 at_action_index 触发一次失败，行为不变。
    # ------------------------------------------------------------------ #
    trigger: Optional[PredicateSpec] = None  # 状态条件触发谓词（满足即触发，可替代/补充 at_action_index）
    duration: Optional[int] = None           # 故障持续步数（None=瞬时一次；>=1=持续 N 步）
    silent_corruption: bool = False          # 静默错误：不报 error、悄悄污染状态（考验主动诊断）
    rollback_required: bool = False          # 恢复必须执行回滚/补偿动作才算正确
    requires_diagnosis: bool = False         # 恢复前必须有显式诊断动作（认知型前置）
    model_config = {"extra": "allow"}


class EnvEventSpec(BaseModel):
    """自主环境事件：在某动作后改写状态（provenance='env'，不计 by-agent）。"""
    id: str
    after_action_index: int = 0
    effect: EffectSpec

    # ------------------------------------------------------------------ #
    # Phase 0 长程事件调度扩展点（全部 Optional + 默认，向后兼容）。
    # 仅声明契约字段；Phase 3 的事件调度器才会消费它们。旧任务只用 after_action_index 单次触发，行为不变。
    # ------------------------------------------------------------------ #
    trigger: Optional[PredicateSpec] = None  # 状态条件触发谓词（满足即触发，可替代/补充 after_action_index）
    duration: Optional[int] = None           # 事件影响/延迟可见窗口的步数（None=即时单次）
    silent_corruption: bool = False          # 静默改写状态（不产生显式可见事件）
    rollback_required: bool = False          # 该事件造成的偏移需 agent 回滚补偿
    requires_diagnosis: bool = False         # 需 agent 显式诊断后方可恢复
    model_config = {"extra": "allow"}


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #
class Task(BaseModel):
    task_id: str
    version: str = "1.0.0"
    dimension: str  # 主载荷维度 U1..U6
    capability_load: Dict[str, float] = Field(default_factory=dict)
    title: str = ""
    instruction: str = ""
    modalities: List[str] = Field(default_factory=lambda: ["text"])

    tools: List[ToolSpec] = Field(default_factory=list)
    initial_state: Dict[str, Any] = Field(default_factory=dict)
    env_events: List[EnvEventSpec] = Field(default_factory=list)
    fault_injection: List[FaultSpec] = Field(default_factory=list)

    milestones: List[MilestoneSpec] = Field(default_factory=list)
    success_predicates: List[PredicateSpec] = Field(default_factory=list)
    critical_violations: List[CriticalViolationSpec] = Field(default_factory=list)
    grounding: Optional[GroundingSpec] = None

    # oracle 参考（用于元测试满分校验 + 效率 c*）
    oracle_plan: List[Dict[str, Any]] = Field(default_factory=list)
    c_star: Optional[float] = None  # 若空则由 oracle_plan 动作数推导
    budget_max_actions: int = 200

    difficulty_knobs: Dict[str, Any] = Field(default_factory=dict)
    canary: str = ""

    # 觅食模式（foraging，spec §2 U2 "信息觅食" / §3.4）：
    #   data_in_context=True（默认，向后兼容）：源数据可直接放进 prompt（适配器注入 initial_state）。
    #   data_in_context=False：源数据**只在 initial_state.data**，禁止注入 prompt；模型必须调用
    #     read_* 工具"觅食"，适配器据 forage_sources 在该工具被调用后才回传对应数据切片。
    # forage_sources：read 工具名 -> initial_state.data 下的数据键（仅 data_in_context=False 时有意义）。
    data_in_context: bool = True
    forage_sources: Dict[str, str] = Field(default_factory=dict)
    model_config = {"extra": "allow"}

    def cost_of_plan(self, n_actions: int) -> float:
        return float(n_actions)

    def effective_c_star(self) -> float:
        if self.c_star is not None:
            return float(self.c_star)
        # 缺省 c* = oracle 计划的动作成本（CP8：c*=min(oracle, 强基线P10)，此处取 oracle）
        return max(1.0, float(len(self.oracle_plan)))


# --------------------------------------------------------------------------- #
# Trace（含 provenance —— CP5 因果门控的依据）
# --------------------------------------------------------------------------- #
class Action(BaseModel):
    action_id: str
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "allow"}


class StateDiff(BaseModel):
    path: str
    new_value: Any = None
    provenance: str = "action"  # action:<id> | env:<id> | init
    model_config = {"extra": "allow"}


class TraceEvent(BaseModel):
    idx: int
    type: str  # tool_call | tool_result | env_event | fault | final
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    status: str = "ok"  # ok | error
    diffs: List[StateDiff] = Field(default_factory=list)
    args_norm_hash: Optional[str] = None  # thrash 检测
    model_config = {"extra": "allow"}


class ModelSubmission(BaseModel):
    """模型适配器对单个任务/单次 run 的产出。"""
    actions: List[Action] = Field(default_factory=list)
    grounding_answers: Dict[str, Any] = Field(default_factory=dict)
    confidences: Dict[str, float] = Field(default_factory=dict)  # 校准用：判断->置信
    abstain: Dict[str, bool] = Field(default_factory=dict)       # 校准用：判断->是否弃答
    model_config = {"extra": "allow"}


class Trace(BaseModel):
    task_id: str
    model_id: str
    run_index: int = 0
    seed: int = 0
    events: List[TraceEvent] = Field(default_factory=list)
    final_state: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, str] = Field(default_factory=dict)  # path -> source
    n_agent_actions: int = 0
    cost_actions: float = 0.0
    submission: Optional[ModelSubmission] = None
    model_config = {"extra": "allow"}
