"""
阶段 2 集成测试（A. 接线 + B. 横评 runner）。

覆盖：
  ① 多目录任务银行加载器：并入 generated/{u1,u2,u4,u5,u6}，**排除 _bridge / manifest**；
     顶层 load_tasks 隔离行为不变（不含生成任务）。
  ② aggregate.build_report 接线统计主干：逐维 GLMM 模型对比（per_model marginal±CI 非退化、
     contrasts β/Δ/p_adj/Cliff's δ、方差/Deff）写入报告；grounding ρ 唯一真源 + real_trusted；
     统计不可区分两源叠加；IRT 选题门（trusted 才用于选题、不进 headline）。
  ③ 适配器层：build_adapter(mock) 可提交；真实 provider 在 offline/无 key 时优雅回退 mock；
     占位密钥判未配置；parse_submission 解析 JSON；OpenAI 兼容适配器网络异常降级为空提交。
  ④ run_eval dry-run：在 mock 上端到端跑通并落盘 JSON+CSV。

运行：cd engine && python -m pytest tests/test_integration.py -q
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from orchestrator import load_tasks, load_task_bank, evaluate  # noqa: E402
from adapters import (build_adapter, resolve_api_key, parse_submission,  # noqa: E402
                      MockAdapter, OpenAICompatibleAdapter, PROVIDERS,
                      render_task_prompt_v2, _asset_image_parts,
                      terminal_tools, _observe)
from schema import ModelSubmission  # noqa: E402
from sandbox import Sandbox  # noqa: E402
from scoring.score import score_task  # noqa: E402
import run_eval  # noqa: E402

_ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# ① 多目录任务银行加载器
# --------------------------------------------------------------------------- #
def test_task_bank_loader_includes_generated_excludes_bridge():
    bank = load_task_bank(_ENGINE)
    assert len(bank) > 100, len(bank)
    ids = {t.task_id for t in bank}
    dims = {t.dimension for t in bank}
    # 覆盖能力维（U1/U2/U4/U5）+ 安全维 U6 + 顶层带来的 U3
    for d in ("U1", "U2", "U3", "U4", "U5", "U6"):
        assert d in dims, (d, dims)
    # _bridge 集（种子 100000/100001）绝不进主榜
    assert not any(("s100000" in i or "s100001" in i) for i in ids), "桥梁集泄漏进主榜"
    # manifest 不当任务
    assert "manifest" not in ids
    # 顶层既有样例仍并入（safety hard-zero demo 依赖）
    assert "u1_invoice_reconcile" in ids


def test_top_level_loader_isolation_preserved():
    """load_tasks(tasks/) 隔离行为不变：仍不含任何生成任务（保护 run_demo/test_meta）。"""
    top = load_tasks(os.path.join(_ENGINE, "tasks"))
    ids = {t.task_id for t in top}
    assert "u1_invoice_reconcile" in ids
    assert not any("__s" in i for i in ids)


def test_task_bank_filters():
    only_medium = load_task_bank(_ENGINE, include_top_level=False, difficulties=["medium"])
    assert only_medium
    for t in only_medium:
        assert (t.difficulty_knobs or {}).get("difficulty") == "medium"
    limited = load_task_bank(_ENGINE, include_top_level=False, difficulties=["medium"],
                             limit_per_template=1)
    tmpls = [(t.difficulty_knobs or {}).get("template") for t in limited]
    assert len(tmpls) == len(set(tmpls)), "limit_per_template=1 应每模板至多 1 实例"


# --------------------------------------------------------------------------- #
# ② build_report 接线统计主干（小银行子集，控制耗时）
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def integ_report():
    tasks = load_task_bank(_ENGINE, include_top_level=True, difficulties=["medium"])
    models = {"oracle-bot": "oracle", "weak-bot": "weak"}
    return evaluate(models, tasks, n_runs=4, k=5,
                    dim_n_boot=150, glmm_n_boot=150, irt_gate=True)


def test_dimension_stats_glmm_nondegenerate_ci(integ_report):
    ds = integ_report["dimension_stats"]
    assert ds, "应有逐维 GLMM 模型对比"
    nondegen = 0
    for d, cmp in ds.items():
        assert cmp["backend"]
        for m, pm in cmp["per_model"].items():
            assert pm["lo"] - 1e-9 <= pm["marginal"] <= pm["hi"] + 1e-9
            if pm["hi"] > pm["lo"] and pm["n_templates"] >= 2:
                nondegen += 1
        de = cmp["design_effect"]
        assert de["deff"] >= 1.0 - 1e-9
        assert de["n_eff"] <= de["n_obs"] + 1e-9
    assert nondegen > 0, "至少一处多模板 → 非退化 CI"


def test_dimension_stats_contrasts_direction_and_significance(integ_report):
    ds = integ_report["dimension_stats"]
    found_sig = False
    for d, cmp in ds.items():
        for c in cmp["contrasts"]:
            assert {"a", "b", "delta", "lo", "hi", "p", "p_adj",
                    "significant", "cliffs_delta"}.issubset(c.keys())
            if c["significant"]:
                found_sig = True
                # oracle 强于 weak：方向为正（a−b）或负，且 CI 排除 0
                assert c["lo"] > 0 or c["hi"] < 0
    assert found_sig, "oracle vs weak 至少一维应显著"


def test_grounding_rho_single_source_and_real_trusted(integ_report):
    gb = integ_report["grounding"]
    assert "rho" in gb and "headline_rule" in gb
    assert gb["headline_rule"] in ("synthetic_only_real_audit",
                                   "synthetic_and_real_coheadline")
    # 顶层 ground_ 任务带 hi 标定真实项 → real_trusted 模型非空，且各 ML 验证器标定值带入
    assert gb["real_trusted_models"], gb
    any_calib = any(gs["calibration"] for gs in gb["per_model"].values())
    assert any_calib, "应带入 ML 验证器标定值"
    # 双轨双值不合并：per_model 同时含 synthetic 与 real，不混成单标量
    for m, gs in gb["per_model"].items():
        assert "synthetic" in gs and "real" in gs


def test_statistical_indistinguishability_two_sources(integ_report):
    si = integ_report["statistical_indistinguishability"]
    assert "weight_flip_pairs" in si
    assert "glmm_nonsignificant_by_dim" in si


def test_irt_gate_present_not_in_headline(integ_report):
    irt = integ_report["irt_item_calibration"]
    assert irt is not None
    assert irt["enters_headline"] is False
    assert irt["use"] in ("item_selection_only", "diagnostic_only")


def test_judge_block_present_diagnostic_not_headline(integ_report):
    """judge α 门接 build_report：每模型 judge 分 + flip_rate + α 可信带；**judge 不进 headline**。"""
    jb = integ_report["judge"]
    assert jb is not None and jb["enters_headline"] is False
    assert jb["cross_family_ok"] is True and len(jb["families"]) >= 3
    assert jb["per_model"], jb
    for m, jv in jb["per_model"].items():
        assert {"score", "alpha", "flip_rate", "reliability_band",
                "headline_eligible"}.issubset(jv.keys())
        assert jv["reliability_band"] in ("drop_from_headline", "wide_ci",
                                          "reliable", "undefined")


def test_judge_decoupled_from_headline_when_disabled():
    """judge=False → report['judge'] is None，但 profiles/dimension_stats（headline）逐字节不变
    → 证明 headline 与 judge 完全解耦（_task_component_value 不含 judge）。"""
    from scoring.aggregate import build_report
    tasks = load_task_bank(_ENGINE, include_top_level=True, difficulties=["medium"])
    models = {"oracle-bot": "oracle", "weak-bot": "weak"}
    recs = {}
    from orchestrator import run_model_on_task
    from models import ModelAdapter
    for mid, prof in models.items():
        rr = []
        for t in tasks:
            rr.extend(run_model_on_task(ModelAdapter(mid, prof), t, 3))
        recs[mid] = rr
    with_judge = build_report(recs, k=5, dim_n_boot=80, glmm_n_boot=80, seed=0, judge=True)
    no_judge = build_report(recs, k=5, dim_n_boot=80, glmm_n_boot=80, seed=0, judge=False)
    assert with_judge["judge"] is not None and no_judge["judge"] is None
    # headline 与 judge 完全解耦：**确定性**的 headline 点估计（GLMM 边际 = 均值、可靠性 per_run、
    # 无权重能力标量）开/关 judge 完全相同（CI 因 bootstrap 随机不比）。
    for prof in ("R", "D"):
        a = with_judge["profiles"][prof]["per_model"]
        b = no_judge["profiles"][prof]["per_model"]
        assert set(a) == set(b)
        for m in a:
            assert a[m]["reliability"]["per_run"] == b[m]["reliability"]["per_run"]
            assert (a[m]["capability_scalar_unweighted"]
                    == b[m]["capability_scalar_unweighted"])
            for dd in a[m]["dim_vector"]:
                assert abs(a[m]["dim_vector"][dd]["point"]
                           - b[m]["dim_vector"][dd]["point"]) < 1e-12


# --------------------------------------------------------------------------- #
# 觅食模式接入适配器（render_task_prompt_v2 / _observe 按 data_in_context 分支）
# --------------------------------------------------------------------------- #
def _forage_task():
    bank = load_task_bank(_ENGINE)
    fts = [t for t in bank if t.task_id.endswith("__forage") and (t.forage_sources or {})]
    assert fts, "应有觅食任务"
    return fts[0]


def test_foraging_prompt_v2_hides_data_and_lists_sources():
    task = _forage_task()
    prompt = render_task_prompt_v2(task)
    assert "FORAGING MODE" in prompt and "data_sources" in prompt
    # 源数据不得出现在提示里（每个数据键的序列化内容不在 prompt 中）；read 工具被列为数据来源
    data = (task.initial_state or {}).get("data", {})
    for tool, key in (task.forage_sources or {}).items():
        assert json.dumps(data[key], ensure_ascii=False) not in prompt, key
        assert tool in prompt, tool
    # 非觅食任务仍照常注入 initial_state（向后兼容）
    normal = [t for t in load_tasks(os.path.join(_ENGINE, "tasks"))
              if t.task_id == "u1_invoice_reconcile"][0]
    assert "initial_state" in render_task_prompt_v2(normal)


def test_foraging_observe_reveals_only_after_read_tool_called():
    from schema import Action as _A
    task = _forage_task()
    forage = dict(task.forage_sources or {})
    one_tool = sorted(forage.keys())[0]
    one_key = forage[one_tool]
    data_blob = json.dumps((task.initial_state or {})["data"][one_key], ensure_ascii=False)
    # 未调用任何 read：data 被剥离 → 该切片内容不出现在观察里，并显式提示数据仍隐藏
    tr0 = Sandbox(task).run(ModelSubmission(actions=[]), model_id="x", run_index=0, seed=0)
    obs0 = _observe(task, tr0, 0)
    assert data_blob not in obs0
    assert "FORAGING: data still hidden" in obs0 and one_tool in obs0
    # 调用对应 read 工具后：该数据切片内容才回传（迫使真正觅食）
    tr1 = Sandbox(task).run(ModelSubmission(actions=[_A(action_id="r", tool=one_tool, args={})]),
                            model_id="x", run_index=0, seed=0)
    obs1 = _observe(task, tr1, 1)
    assert data_blob in obs1, (one_key, obs1[:400])


def test_contamination_block_operationalized():
    import run_eval as RE
    cb = RE.build_contamination_block(_ENGINE, n_pairs=5, n_runs=3, n_boot=200)
    assert cb["per_template"], cb
    for r in cb["per_template"]:
        assert {"gap", "delta", "lo", "hi", "p", "flag_retire", "n_pairs",
                "method", "template_id"}.issubset(r.keys())
        assert r["method"] == "paired_bootstrap(stats)"
    assert cb["equating_demo"] and cb["equating_demo"]["method"] == "linear"
    assert "any_flag_retire" in cb


def test_backward_compatible_report_keys(integ_report):
    # run_demo / test_grounding 依赖的既有键仍在
    for key in ("models", "profiles", "grounding_rho", "grounding_headline_rule",
                "pareto_frontier", "pareto_points", "k", "raw_records"):
        assert key in integ_report, key
    for prof in ("R", "D"):
        p = integ_report["profiles"][prof]
        assert "per_model" in p and "weight_sensitivity" in p and "dims_present" in p


# --------------------------------------------------------------------------- #
# ③ 适配器层
# --------------------------------------------------------------------------- #
def test_mock_adapter_submits():
    ad = build_adapter({"id": "m", "provider": "mock", "mock_profile": "strong"})
    assert isinstance(ad, MockAdapter) and ad.is_mock
    bank = load_task_bank(_ENGINE, include_top_level=False, difficulties=["easy"],
                          limit_per_template=1)
    sub = ad.submit(bank[0], run_index=0, seed=1000)
    assert isinstance(sub, ModelSubmission)


def test_real_provider_falls_back_to_mock_when_offline_or_no_key():
    # 真实 provider + 占位密钥 + offline → 优雅回退 mock，标注原因
    entry = {"id": "deepseek", "provider": "deepseek",
             "api_key": "YOUR_DEEPSEEK_API_KEY_HERE", "mock_profile": "medium"}
    ad_off = build_adapter(entry, offline=True)
    assert isinstance(ad_off, MockAdapter) and ad_off.fallback_reason == "offline"
    ad_nokey = build_adapter(entry, offline=False)
    assert isinstance(ad_nokey, MockAdapter) and ad_nokey.fallback_reason == "no_api_key"


def test_resolve_api_key_placeholder_is_unconfigured(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert resolve_api_key({"provider": "deepseek",
                            "api_key": "YOUR_DEEPSEEK_API_KEY_HERE"}) is None
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-real-xyz")
    assert resolve_api_key({"provider": "deepseek"}) == "sk-real-xyz"


def test_real_adapter_built_with_key_and_degrades_on_network_error(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-callable")
    ad = build_adapter({"id": "deepseek", "provider": "deepseek"}, offline=False)
    assert isinstance(ad, OpenAICompatibleAdapter)
    assert ad.base_url == PROVIDERS["deepseek"]["base_url"]
    # 网络不可用（无外网）→ submit 不抛错，降级为空提交
    bank = load_task_bank(_ENGINE, include_top_level=False, difficulties=["easy"],
                          limit_per_template=1)
    sub = ad.submit(bank[0], run_index=0, seed=1000)
    assert isinstance(sub, ModelSubmission)
    assert ad.n_errors >= 1 and sub.actions == []


def test_build_adapter_honors_responses_endpoint_type(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-real-test-key")
    ad = build_adapter({
        "id": "doubao_seed_2_1_pro",
        "provider": "seed",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-2-1-pro-260628",
        "api_key_env": "ARK_API_KEY",
        "endpoint_type": "responses",
    })
    assert isinstance(ad, OpenAICompatibleAdapter)
    assert ad.model_id == "doubao_seed_2_1_pro"
    assert ad.model == "doubao-seed-2-1-pro-260628"
    assert ad.endpoint_type == "responses"


def test_responses_endpoint_request_body_and_output_extraction(monkeypatch):
    import adapters as _A

    captured = {}

    class _FakeResponse:
        def read(self):
            return json.dumps({
                "output": [
                    {"content": [
                        {"type": "output_text", "text": "{\"actions\": [], \"done\": true}"}
                    ]}
                ]
            }).encode("utf-8")

    def fake_urlopen(req, timeout=None, context=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr(_A.urllib.request, "urlopen", fake_urlopen)
    ad = OpenAICompatibleAdapter(
        model_id="p", base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key="ark-real-test-key", model="doubao-seed-2-1-pro-260628",
        endpoint_type="responses", stream=False,
    )
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]},
    ]
    out = ad._request(messages, stream=False, seed=0)
    assert out == "{\"actions\": [], \"done\": true}"
    assert captured["url"].endswith("/responses")
    assert "messages" not in captured["body"]
    assert captured["body"]["instructions"] == "system prompt"
    assert captured["body"]["input"][0]["role"] == "user"
    assert captured["body"]["input"][0]["content"] == [
        {"type": "input_text", "text": "hello"},
        {"type": "input_image", "image_url": "data:image/png;base64,abc"},
    ]


def test_insecure_ssl_flag_honored():
    # 默认安全验证 TLS
    secure = build_adapter({"id": "x", "provider": "deepseek", "api_key": "sk-real-key"})
    assert isinstance(secure, OpenAICompatibleAdapter)
    assert secure.insecure_ssl is False and secure._ssl_ctx is None
    # 显式 insecure_ssl（仅用于 TLS 拦截代理/自签证书环境）→ 不验证的 SSLContext
    insec = build_adapter({"id": "x", "provider": "deepseek", "api_key": "sk-real-key",
                           "insecure_ssl": True})
    assert insec.insecure_ssl is True and insec._ssl_ctx is not None


def test_parse_submission_parses_json_with_fences():
    content = ("```json\n{\"actions\": [{\"tool\": \"read_source_1\", \"args\": {}}], "
               "\"grounding_answers\": {\"g1\": 14.0}}\n```")
    sub = parse_submission(content)
    assert len(sub.actions) == 1 and sub.actions[0].tool == "read_source_1"
    assert sub.grounding_answers["g1"] == 14.0
    assert parse_submission("not json at all").actions == []


# --------------------------------------------------------------------------- #
# harness-v2：多轮 agent 回路（用 mock 的 _chat，验证回路 + 终态写入 + 上限）
# --------------------------------------------------------------------------- #
def _u1_task():
    return [t for t in load_tasks(os.path.join(_ENGINE, "tasks"))
            if t.task_id == "u1_invoice_reconcile"][0]


def test_multiturn_loop_reaches_terminal_write_and_success(monkeypatch):
    task = _u1_task()
    ad = OpenAICompatibleAdapter(model_id="t", base_url="http://x", api_key="k",
                                 model="m", max_rounds=4)
    turns = iter([
        '{"actions":[{"tool":"read_invoices","args":{}},{"tool":"read_bank","args":{}}],"done":false}',
        '{"actions":[{"tool":"submit_report","args":{"report":{"inconsistent":["INV-002","INV-005"]}}}],"done":true}',
    ])
    monkeypatch.setattr(ad, "_chat", lambda messages, seed=0: next(turns))
    sub = ad.submit(task, run_index=0, seed=0)
    # 多轮累计动作含终态写入
    assert [a.tool for a in sub.actions] == ["read_invoices", "read_bank", "submit_report"]
    assert ad.task_log[-1]["rounds"] == 2 and ad.task_log[-1]["success_met"] is True
    # 权威沙箱重放 → 真实判定成功（验证回路确实解锁了多步任务）
    rec = score_task(task, Sandbox(task).run(sub, model_id="t", run_index=0, seed=0))
    assert rec["success"] is True and rec["critical"] is False


def test_multiturn_loop_bounds_at_max_rounds(monkeypatch):
    task = _u1_task()
    ad = OpenAICompatibleAdapter(model_id="t", base_url="http://x", api_key="k",
                                 model="m", max_rounds=3)
    # 永远只读、从不 done → 必须在 max_rounds 处停（不死循环）
    monkeypatch.setattr(ad, "_chat",
                        lambda messages, seed=0: '{"actions":[{"tool":"read_invoices","args":{}}],"done":false}')
    sub = ad.submit(task, run_index=0, seed=0)
    assert ad.task_log[-1]["rounds"] == 3
    assert ad.task_log[-1]["success_met"] is False
    assert ad.n_calls == 3


def test_multimodal_image_parts_and_prompt():
    # ground_chart_revenue 现已渲染 PNG → 图片被编码为 base64 image_url 喂给模型
    chart = [t for t in load_tasks(os.path.join(_ENGINE, "tasks"))
             if t.task_id == "ground_chart_revenue"][0]
    parts = _asset_image_parts(chart)
    assert parts, "U3 任务应有可喂的图片资产"
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    # v2 提示注入工具签名（含写出工具的 arg_key）
    prompt = render_task_prompt_v2(chart)
    assert "submit_finding" in prompt and "arg_key" in prompt


# --------------------------------------------------------------------------- #
# ④ run_eval dry-run（mock）端到端 + 落盘
# --------------------------------------------------------------------------- #
def test_terminal_tools_detection_and_finalization_nudge():
    import build_solvable as B
    B.materialize()
    task = [t for t in load_task_bank(_ENGINE) if t.task_id == "solv_u1_reconcile__s0"][0]
    terms = terminal_tools(task)
    assert any(t["name"] == "submit_report" and t["arg_key"] == "report" for t in terms)
    # 空提交 → 未达标 → observation 含强制终态写入提示
    tr = Sandbox(task).run(ModelSubmission(actions=[]), model_id="x", run_index=0, seed=0)
    obs = _observe(task, tr, 0)
    assert "Goal satisfied: False" in obs
    assert "REQUIRED FINALIZATION" in obs and "submit_report" in obs


def test_premature_done_rejected_then_finalized(monkeypatch):
    """U3 终态闭环修复：模型先 done=true 但写错结构 → 拒绝 → 强制再走 → 正确结构 → success。"""
    chart = [t for t in load_task_bank(_ENGINE) if t.task_id == "ground_chart_revenue"][0]
    ad = OpenAICompatibleAdapter(model_id="t", base_url="http://x", api_key="k",
                                 model="m", max_rounds=4)
    turns = iter([
        # 轮0：调用了终态工具但 finding 是自由文本字符串（v3 的真实失败形态）+ done=true → 应被拒绝
        '{"actions":[{"tool":"read_chart","args":{}},'
        '{"tool":"submit_finding","args":{"finding":"top quarter is Q3, about 22"}}],"done":true}',
        # 轮1：被强制 finalize 后给出正确结构
        '{"actions":[{"tool":"submit_finding","args":{"finding":{"top":"Q3","value":22.2}}}],"done":true}',
    ])
    monkeypatch.setattr(ad, "_chat", lambda messages, seed=0: next(turns))
    sub = ad.submit(chart, run_index=0, seed=0)
    # 没有在轮0 停下（premature done 被拒），强制走到轮1
    assert ad.task_log[-1]["rounds"] == 2, ad.task_log[-1]
    assert ad.task_log[-1]["success_met"] is True
    # 权威沙箱重放 → 真实判定成功（结构写对了）
    rec = score_task(chart, Sandbox(chart).run(sub, model_id="t", run_index=0, seed=0))
    assert rec["success"] is True


def test_done_honored_when_no_terminal_contract(monkeypatch):
    """无成功谓词路径（无终态契约）的任务：尊重模型 done，不强制 finalize（避免死循环）。"""
    from schema import Task as _T
    grounding_only = _T(task_id="g_only", dimension="U3",
                        instruction="answer grounding", success_predicates=[])
    ad = OpenAICompatibleAdapter(model_id="t", base_url="http://x", api_key="k",
                                 model="m", max_rounds=4)
    monkeypatch.setattr(ad, "_chat",
                        lambda messages, seed=0: '{"actions":[],"grounding_answers":{"g":1},"done":true}')
    ad.submit(grounding_only, run_index=0, seed=0)
    assert ad.task_log[-1]["rounds"] == 1  # 无终态契约 → done 立即生效


def test_chat_backoff_retry_on_transient(monkeypatch):
    """空响应/超时等瞬时故障 → 指数退避重试后成功（消除并发偶发空/错）。"""
    import adapters as _A
    monkeypatch.setattr(_A.time, "sleep", lambda s: None)  # 不真 sleep
    ad = OpenAICompatibleAdapter(model_id="t", base_url="http://x", api_key="k",
                                 model="m", max_retries=2)
    calls = {"n": 0}

    def flaky(messages, stream, seed):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("read timed out")
        return '{"actions": [], "grounding_answers": {"g": 1}}'
    monkeypatch.setattr(ad, "_request", flaky)
    out = ad._chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 3 and "grounding_answers" in out


def test_chat_no_retry_on_4xx(monkeypatch):
    import urllib.error
    import adapters as _A
    monkeypatch.setattr(_A.time, "sleep", lambda s: None)
    ad = OpenAICompatibleAdapter(model_id="t", base_url="http://x", api_key="k",
                                 model="m", max_retries=3)
    calls = {"n": 0}

    def bad(messages, stream, seed):
        calls["n"] += 1
        raise urllib.error.HTTPError("http://x", 400, "bad", {}, None)
    monkeypatch.setattr(ad, "_request", bad)
    with pytest.raises(urllib.error.HTTPError):
        ad._chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1  # 4xx 客户端错误：不重试


def test_no_progress_early_stop(monkeypatch):
    """模型重复同一错答（无进展）→ 提前停，不烧满 max_rounds。"""
    task = [t for t in load_task_bank(_ENGINE) if t.task_id == "solv_u1_reconcile__s0"][0]
    ad = OpenAICompatibleAdapter(model_id="t", base_url="http://x", api_key="k",
                                 model="m", max_rounds=5)
    monkeypatch.setattr(ad, "_chat", lambda messages, seed=0:
                        '{"actions":[{"tool":"submit_report","args":{"report":{"inconsistent":["INV-999"]}}}],"done":false}')
    ad.submit(task, run_index=0, seed=0)
    tl = ad.task_log[-1]
    assert tl["rounds"] < 5 and tl["stopped_no_progress"] is True
    assert tl["success_met"] is False


def test_u6_excluded_from_capability_reliability():
    """U6 安全探针 success（多为 gold-only）不污染能力可靠性；ASR 仍计入。"""
    from scoring.aggregate import aggregate_model

    def _rec(tid, dim, success, asr=0.0):
        return {"task_id": tid, "template": tid, "dimension": dim, "success": success,
                "raw_success": success, "critical": bool(asr), "asr": asr, "process": 1.0,
                "recovery": float("nan"), "expected_milestone_completion": float("nan"),
                "grounding": {"synthetic": float("nan"), "real": float("nan"),
                              "real_diagnostic": float("nan"), "real_trusted": False,
                              "real_headline_eligible": False, "calibration": {}},
                "efficiency": {"eff": 1.0 if success else float("nan")}, "cost": 1.0}
    recs = [_rec("u1t", "U1", True), _rec("u1t", "U1", True),
            _rec("u6t", "U6", False, asr=0.0), _rec("u6t", "U6", False, asr=0.0)]
    agg = aggregate_model(recs, k=3, profile="R", n_boot=50)
    assert agg["reliability"]["per_run"] == 1.0   # U6 的 0 不拖累能力可靠性
    assert agg["asr"] == 0.0                        # 但 ASR 仍统计了 U6


def test_run_eval_concurrent_mock(tmp_path):
    cfg = os.path.join(_ENGINE, "configs", "models.example.json")
    res = run_eval.run(config_path=cfg, n_runs=2, k=5, difficulty="medium",
                       limit_per_template=1, out_dir=str(tmp_path),
                       dim_n_boot=80, glmm_n_boot=80, force_mock=True, irt_gate=False,
                       quiet=True, concurrency=4, wall_clock_s=120, max_calls=10000)
    assert os.path.isfile(res["json_path"])
    rm = res["run_meta"]
    assert rm and rm["jobs_total"] > 0 and rm["jobs_done"] > 0
    # 并发路径产出与顺序路径同构的报告
    assert res["summary"]["models"] and "dimension_stats" in res["summary"]


def test_run_eval_incremental_save_and_resume(tmp_path):
    """秒级 smoke：mock 横评验证 ① 增量落盘（每 (task,run) 一完成即追加 partial.jsonl）；
    ② 实时进度文件（已完成/总数 + FINISHED）；③ 断点续（截断 partial 模拟中断后重跑，
    已完成的被跳过、其余补齐，且报告仍完整）。"""
    cfg = os.path.join(_ENGINE, "configs", "models.example.json")
    rid = "smoke_resume"
    common = dict(config_path=cfg, n_runs=2, k=5, difficulty="easy",
                  limit_per_template=1, out_dir=str(tmp_path), run_id=rid,
                  dim_n_boot=40, glmm_n_boot=40, force_mock=True, irt_gate=False,
                  include_top_level=False, include_generated=False,
                  contamination=False, quiet=True, concurrency=4)

    res1 = run_eval.run(**common)
    partial = tmp_path / (rid + ".partial.jsonl")
    progress = tmp_path / (rid + ".progress.txt")
    assert partial.is_file() and progress.is_file()
    total = res1["run_meta"]["jobs_total"]
    assert total > 0
    # ① 增量：partial 行数 == 完成数；每行可解析且含必要键 + NaN 可原样读回
    lines = [ln for ln in partial.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == res1["run_meta"]["jobs_done"] == total
    first = json.loads(lines[0])
    assert {"mid", "task_id", "run_index", "rec", "adapter"} <= set(first)
    assert "dimension" in first["rec"]
    # ② 进度文件：完成数/总数 + FINISHED
    ptext = progress.read_text(encoding="utf-8")
    assert ("completed: %d/%d" % (total, total)) in ptext
    assert "FINISHED" in ptext

    # ③ 断点续：截断 partial 到仅剩 1 行（模拟跑到一半崩溃）→ 同 run_id 重跑
    partial.write_text(lines[0] + "\n", encoding="utf-8")
    res2 = run_eval.run(**common)
    rm = res2["run_meta"]
    assert rm["jobs_resumed"] == 1, rm
    assert rm["jobs_new"] == total - 1, rm
    assert rm["jobs_done"] == total, rm
    # 续跑补齐后 partial 恢复满行；报告仍完整（mock 全维聚合）
    lines2 = [ln for ln in partial.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines2) == total
    assert res2["summary"]["models"] and "dimension_stats" in res2["summary"]

    # ④ 全部已完成时再跑：0 新作业、全部续跑命中，报告照常产出（幂等）
    res3 = run_eval.run(**common)
    assert res3["run_meta"]["jobs_new"] == 0
    assert res3["run_meta"]["jobs_resumed"] == total
    assert os.path.isfile(res3["json_path"])


def test_run_eval_dry_run_mock(tmp_path):
    cfg = os.path.join(_ENGINE, "configs", "models.example.json")
    res = run_eval.run(config_path=cfg, n_runs=3, k=5, difficulty="medium",
                       limit_per_template=1, out_dir=str(tmp_path),
                       dim_n_boot=120, glmm_n_boot=120, force_mock=True,
                       irt_gate=False, quiet=True)
    assert os.path.isfile(res["json_path"]) and os.path.isfile(res["csv_path"])
    with open(res["json_path"], "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["models"] and "profiles" in summary and "dimension_stats" in summary
    # 全 mock → adapters 标注 is_mock
    assert all(a["is_mock"] for a in summary["adapters"].values())
    # CSV 有逐维行
    with open(res["csv_path"], "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) >= 2 and rows[0][0] == "profile"


def test_run_eval_require_real_rejects_mock_config(tmp_path):
    cfg = tmp_path / "models.json"
    cfg.write_text(json.dumps({"models": [
        {"id": "dry", "provider": "mock", "mock_profile": "strong"}
    ]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="require-real"):
        run_eval.run(config_path=str(cfg), n_runs=1, k=1, difficulty="medium",
                     task_ids=["u1_invoice_reconcile"], out_dir=str(tmp_path),
                     dim_n_boot=20, glmm_n_boot=20, require_real=True,
                     contamination=False, quiet=True, concurrency=1)


def test_run_eval_no_mock_fallback_filters_to_diagnostics(tmp_path, monkeypatch):
    from models import ModelAdapter

    class FakeAdapter:
        def __init__(self, model_id, profile, is_mock):
            self.model_id = model_id
            self.profile_name = profile
            self.provider = "fake"
            self.model = "fake-model"
            self.is_mock = is_mock
            self.fallback_reason = "test_mock" if is_mock else ""
            self.n_calls = 0
            self.n_errors = 0
            self._inner = ModelAdapter(model_id, profile)

        def submit(self, task, run_index=0, seed=0):
            self.n_calls += 1
            return self._inner.submit(task, run_index=run_index, seed=seed)

    def fake_build_adapter(entry, offline=False, force_mock=False):
        return FakeAdapter(entry["id"], entry.get("mock_profile", "medium"),
                           bool(entry.get("fake_is_mock")))

    monkeypatch.setattr(run_eval, "build_adapter", fake_build_adapter)
    cfg = tmp_path / "models.json"
    cfg.write_text(json.dumps({"models": [
        {"id": "realish", "provider": "fake", "mock_profile": "strong", "fake_is_mock": False},
        {"id": "baseline", "provider": "fake", "mock_profile": "weak", "fake_is_mock": True}
    ]}), encoding="utf-8")
    res = run_eval.run(config_path=str(cfg), n_runs=1, k=1, difficulty="medium",
                       task_ids=["u1_invoice_reconcile"], out_dir=str(tmp_path),
                       dim_n_boot=20, glmm_n_boot=20, no_mock_fallback=True,
                       contamination=False, quiet=True, concurrency=1, irt_gate=False)
    summary = res["summary"]
    assert summary["models"] == ["realish"]
    assert list(summary["adapters"]) == ["realish"]
    assert summary["headline_model_policy"]["excluded_mock_model_ids"] == ["baseline"]
    assert summary["diagnostic_baselines"]["models"] == ["baseline"]


def test_run_eval_equating_metadata_and_csv_columns(tmp_path, monkeypatch):
    from models import ModelAdapter

    class FakeRealAdapter:
        is_mock = False
        provider = "fake"
        model = "fake-model"
        fallback_reason = ""
        n_errors = 0

        def __init__(self, model_id, profile):
            self.model_id = model_id
            self.profile_name = profile
            self.n_calls = 0
            self._inner = ModelAdapter(model_id, profile)

        def submit(self, task, run_index=0, seed=0):
            self.n_calls += 1
            return self._inner.submit(task, run_index=run_index, seed=seed)

    def fake_build_adapter(entry, offline=False, force_mock=False):
        return FakeRealAdapter(entry["id"], entry.get("mock_profile", "medium"))

    monkeypatch.setattr(run_eval, "build_adapter", fake_build_adapter)
    mids = ["m1", "m2", "m3"]
    cfg = tmp_path / "models.json"
    cfg.write_text(json.dumps({"models": [
        {"id": "m1", "provider": "fake", "mock_profile": "weak"},
        {"id": "m2", "provider": "fake", "mock_profile": "medium"},
        {"id": "m3", "provider": "fake", "mock_profile": "strong"}
    ]}), encoding="utf-8")
    prev_profiles = {}
    for prof in ("R", "D"):
        prev_profiles[prof] = {"per_model": {
            mid: {"dim_vector": {"U1": {"point": 0.2 + i * 0.2,
                                        "lo": 0.1 + i * 0.2,
                                        "hi": 0.3 + i * 0.2}}}
            for i, mid in enumerate(mids)
        }}
    prev = tmp_path / "prev.json"
    prev.write_text(json.dumps({"profiles": prev_profiles}), encoding="utf-8")
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"ref_model_ids": mids, "min_panel": 3}), encoding="utf-8")

    res = run_eval.run(config_path=str(cfg), n_runs=1, k=1, difficulty="medium",
                       task_ids=["u1_invoice_reconcile"], out_dir=str(tmp_path),
                       dim_n_boot=20, glmm_n_boot=20, contamination=False,
                       quiet=True, concurrency=1, irt_gate=False,
                       eval_version="v-next", prev_summary=str(prev),
                       reference_panel_config=str(panel), equate_method="linear")
    summary = res["summary"]
    meta = summary["longitudinal_equating"]
    assert meta["enabled"] is True and meta["eval_version"] == "v-next"
    assert meta["profiles"]["R"]["U1"]["status"] == "ok"
    for mid in mids:
        agg = summary["profiles"]["R"]["per_model"][mid]
        assert "U1" in agg["dim_vector"]
        assert "U1" in agg["equated_dim_vector"]
    with open(res["csv_path"], "r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "equated_point" in header and "equated_lo" in header and "equated_hi" in header
