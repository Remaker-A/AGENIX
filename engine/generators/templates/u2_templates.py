"""
U2 模板库 —— 条件规划与信息觅食（spec §2 / §3.4）。

主要可验证信号：信息获取顺序、分支决策、动态重规划触发。
机制要点：**认知型必要动作**（必须先觅食信息）→ 受 deps 门控的终态决策；
含"看似便宜但不可行"的陷阱（误导信息）与动态库存（env_event）。
共 6 个模板：
  u2_trip_replan      : 多约束下选可行最优路线（eq 终态决策 + 动态库存）
  u2_supplier_sourcing: 逐一核验供应商 → 提交合规入围集（set_eq）
  u2_evidence_foraging: 证据觅食（含 OR 组合法替代来源）→ 提交综合结论集
"""
from __future__ import annotations

from generators.base import (
    Ctx, Template, register, predicate, effect_set,
    scale_by_difficulty as scale, pick_distinct_ids, inject_noise_tools,
    inject_misleading_info, set_partial_observability,
)


CITY_POOL = ["上海", "东京", "新加坡", "迪拜", "法兰克福", "伦敦", "纽约",
             "悉尼", "首尔", "苏黎世"]
CONSTRAINT_POOL = ["budget", "calendar", "visa_transit", "layover_rest",
                   "baggage", "loyalty", "carbon_cap"]


def _u2_trip_replan(ctx: Ctx):
    rng, diff = ctx.rng, ctx.difficulty
    n_constraints = scale(diff, 2, 3, 4, 5)
    constraints = rng.sample(CONSTRAINT_POOL, n_constraints)
    dst = rng.choice(CITY_POOL)
    options = pick_distinct_ids(rng, "RT", 4)
    feasible = options[0]      # 唯一可行最优
    trap = options[1]          # 更便宜但签证/休息不可行

    b = ctx.new_task(
        title="动态重订路线 → %s" % dst,
        instruction=("在以下硬约束下为前往 %s 重订路线并选定唯一可行最优方案：%s。"
                     "必须先查询每项约束再决策；存在一条更便宜但不可行的诱饵方案。"
                     % (dst, ", ".join(constraints))),
        capability_load={"U2": 1.0, "U1": 0.4},
    )
    q_tools = []
    for c in constraints:
        nm = "query_%s" % c
        b.add_tool(nm)  # 认知型觅食动作
        q_tools.append(nm)
    b.add_tool("select_option", writes=["out.plan.selected"],
               effect=effect_set("out.plan.selected", value_from="option"))

    q_ms = []
    for i, nm in enumerate(q_tools):
        mid = "M_q%d" % (i + 1)
        b.add_milestone(mid, predicate("tool_called", value=nm), weight=1.0,
                        epistemic_action=nm)
        q_ms.append(mid)
    b.add_milestone("M_select",
                    predicate("eq", path="out.plan.selected", value=feasible),
                    weight=3.0, deps=q_ms, provenance=["out.plan.selected"])
    b.add_success(predicate("eq", path="out.plan.selected", value=feasible))

    # 动态库存（env_event，provenance=env，不计 by-agent）：体现重规划压力
    b.add_env_event("inventory_shift", after_action_index=1,
                    effect=effect_set("market.cheapest", value=trap))
    b.set_knob("dynamic_inventory", True)
    inject_noise_tools(b, rng, scale(diff, 1, 2, 3, 4))
    inject_misleading_info(b, rng, [
        "方案 %s 票价最低——但其过境点需要你不持有的签证。" % trap,
        "某聚合器把诱饵方案排在第一位，请以约束核验为准。"])
    set_partial_observability(b, diff in ("hard", "expert"))

    for nm in q_tools:
        b.add_action(nm)
    b.add_action("select_option", {"option": feasible})
    b.set_extra("gold_selected", feasible)
    b.set_extra("trap_option", trap)
    return b


def _u2_supplier_sourcing(ctx: Ctx):
    rng, diff = ctx.rng, ctx.difficulty
    n_suppliers = scale(diff, 3, 4, 5, 6)
    n_approved = scale(diff, 2, 2, 3, 3)
    suppliers = pick_distinct_ids(rng, "SUP", n_suppliers)
    approved = sorted(rng.sample(suppliers, n_approved))
    category = rng.choice(["原材料", "物流", "云资源", "代工", "检测认证"])

    b = ctx.new_task(
        title="供应商寻源与合规入围（%s）" % category,
        instruction=("逐一核验 %d 家供应商的资质/价格/交期，提交满足全部硬约束的入围集合。"
                     "未核验即入围视为失败。" % n_suppliers),
        capability_load={"U2": 1.0, "U1": 0.3},
    )
    check_tools = []
    for s in suppliers:
        nm = "check_%s" % s.replace("-", "_").lower()
        b.add_tool(nm)
        check_tools.append(nm)
    b.add_tool("submit_shortlist", writes=["out.shortlist"],
               effect=effect_set("out.shortlist", value_from="ids"))

    check_ms = []
    for i, nm in enumerate(check_tools):
        mid = "M_chk%d" % (i + 1)
        b.add_milestone(mid, predicate("tool_called", value=nm), weight=1.0,
                        epistemic_action=nm)
        check_ms.append(mid)
    b.add_milestone("M_shortlist",
                    predicate("set_eq", path="out.shortlist", value=approved),
                    weight=3.0, deps=check_ms, provenance=["out.shortlist"])
    b.add_success(predicate("set_eq", path="out.shortlist", value=approved))

    inject_noise_tools(b, rng, scale(diff, 1, 2, 2, 3))
    inject_misleading_info(b, rng, [
        "销售邮件声称某未达标供应商'可特批'，无据可依。",
        "最低报价供应商缺少必需认证，价格不能凌驾硬约束。"])
    set_partial_observability(b, diff == "expert")

    for nm in check_tools:
        b.add_action(nm)
    b.add_action("submit_shortlist", {"ids": approved})
    b.set_extra("gold_shortlist", approved)
    return b


def _u2_evidence_foraging(ctx: Ctx):
    rng, diff = ctx.rng, ctx.difficulty
    n_req = scale(diff, 1, 2, 2, 3)
    n_claims = scale(diff, 2, 2, 3, 4)
    topic = rng.choice(["竞品定价", "故障根因", "市场准入", "供应风险", "合规口径"])
    claims = sorted(pick_distinct_ids(rng, "CLM", n_claims))

    b = ctx.new_task(
        title="证据觅食与结论综合（%s）" % topic,
        instruction=("围绕 '%s' 觅食证据：必须读取所有必需来源，并从主源或其镜像（任一）"
                     "获取关键数据，最后提交经证据支持的结论集合。" % topic),
        capability_load={"U2": 1.0, "U1": 0.3},
    )
    req_tools = []
    for i in range(n_req):
        nm = "read_required_%d" % (i + 1)
        b.add_tool(nm)
        req_tools.append(nm)
    # OR 组：主源 / 镜像源（合法替代路径，结构表达；spec §4.1）
    b.add_tool("read_primary")
    b.add_tool("read_mirror")
    b.add_tool("submit_claims", writes=["out.claims"],
               effect=effect_set("out.claims", value_from="claims"))

    req_ms = []
    for i, nm in enumerate(req_tools):
        mid = "M_req%d" % (i + 1)
        b.add_milestone(mid, predicate("tool_called", value=nm), weight=1.0,
                        epistemic_action=nm)
        req_ms.append(mid)
    b.add_milestone("M_src_primary", predicate("tool_called", value="read_primary"),
                    weight=1.0, mtype="or_group", or_group="evidence_src",
                    epistemic_action="read_primary")
    b.add_milestone("M_src_mirror", predicate("tool_called", value="read_mirror"),
                    weight=1.0, mtype="or_group", or_group="evidence_src",
                    epistemic_action="read_mirror")
    b.add_milestone("M_claims",
                    predicate("set_eq", path="out.claims", value=claims),
                    weight=3.0, deps=req_ms, provenance=["out.claims"])
    b.add_success(predicate("set_eq", path="out.claims", value=claims))

    inject_noise_tools(b, rng, scale(diff, 1, 1, 2, 3))
    set_partial_observability(b, diff in ("hard", "expert"))

    for nm in req_tools:
        b.add_action(nm)
    b.add_action("read_primary")          # 走 OR 组其一即可
    b.add_action("submit_claims", {"claims": claims})
    b.set_extra("gold_claims", claims)
    return b


def _u2_real_sqlite_supplier_query(ctx: Ctx):
    rng, diff = ctx.rng, ctx.difficulty
    n_suppliers = scale(diff, 4, 5, 6, 7)
    max_price = rng.choice([80, 90, 100, 110])
    max_lead = rng.choice([7, 10, 14])
    suppliers = pick_distinct_ids(rng, "SUP", n_suppliers)
    rows = []
    approved = []
    for i, sid in enumerate(suppliers):
        certified = 1 if i % 2 == 0 else 0
        price = max_price - 10 + i * 5
        lead = max(3, max_lead - 2 + (i % 4))
        row = {"id": sid, "certified": certified, "price": price, "lead_days": lead}
        rows.append(row)
        if certified and price <= max_price and lead <= max_lead:
            approved.append(sid)
    approved = sorted(approved)

    b = ctx.new_task(
        title="SQLite 供应商筛选",
        instruction=("必须查询 seed SQLite 数据库 suppliers 表，找出 certified=1、"
                     "price<=%d 且 lead_days<=%d 的供应商 id，最后提交 shortlist。"
                     % (max_price, max_lead)),
        capability_load={"U2": 1.0, "U1": 0.3},
    )
    b.set_initial("sqlite.tables", {"suppliers": rows})
    sql = ("SELECT id, certified, price, lead_days FROM suppliers "
           "WHERE certified = 1 AND price <= %d AND lead_days <= %d ORDER BY id"
           % (max_price, max_lead))
    b.add_tool("query_supplier_db")
    b.tools[-1].update({
        "backend": "sqlite",
        "permissions": ["db:read"],
        "sql": sql,
        "observable_result_paths": ["observations.supplier_rows"],
    })
    b.add_tool("submit_shortlist", writes=["out.shortlist"],
               effect=effect_set("out.shortlist", value_from="ids"))
    b.add_milestone("M_query_db", predicate("tool_called", value="query_supplier_db"),
                    weight=1.0, epistemic_action="query_supplier_db")
    b.add_milestone("M_shortlist",
                    predicate("set_eq", path="out.shortlist", value=approved),
                    weight=3.0, deps=["M_query_db"], provenance=["out.shortlist"])
    b.add_success(predicate("set_eq", path="out.shortlist", value=approved))
    inject_noise_tools(b, rng, scale(diff, 1, 1, 2, 3))
    set_partial_observability(b, True)
    b.add_action("query_supplier_db")
    b.add_action("submit_shortlist", {"ids": approved})
    b.set_extra("gold_shortlist", approved)
    b.set_extra("real_backend", "sqlite")
    return b


def _u2_real_browser_route_pick(ctx: Ctx):
    rng, diff = ctx.rng, ctx.difficulty
    options = pick_distinct_ids(rng, "RT", scale(diff, 3, 4, 5, 6))
    chosen = options[0]
    rows = []
    for i, opt in enumerate(options):
        status = "eligible" if i == 0 else rng.choice(["visa_blocked", "too_late", "over_budget"])
        score = 95 - i * 7
        rows.append('<li class="route" data-id="%s" data-status="%s">Route %s score %d status %s</li>'
                    % (opt, status, opt, score, status))
    page = "<html><head><title>Route Board</title></head><body><ul>%s</ul></body></html>" % "".join(rows)

    b = ctx.new_task(
        title="静态 DOM 路线选择",
        instruction=("必须浏览 /routes.html 的 DOM，读取 .route 条目，选择唯一 status=eligible 的路线。"),
        capability_load={"U2": 1.0, "U1": 0.2},
    )
    b.set_initial("browser.pages", {"/routes.html": page})
    b.add_tool("read_route_page")
    b.tools[-1].update({
        "backend": "browser",
        "permissions": ["browser:read"],
        "url": "/routes.html",
        "selector": ".route",
        "observable_result_paths": ["observations.route_dom"],
    })
    b.add_tool("select_option", writes=["out.plan.selected"],
               effect=effect_set("out.plan.selected", value_from="option"))
    b.add_milestone("M_read_dom", predicate("tool_called", value="read_route_page"),
                    weight=1.0, epistemic_action="read_route_page")
    b.add_milestone("M_select",
                    predicate("eq", path="out.plan.selected", value=chosen),
                    weight=3.0, deps=["M_read_dom"], provenance=["out.plan.selected"])
    b.add_success(predicate("eq", path="out.plan.selected", value=chosen))
    inject_noise_tools(b, rng, scale(diff, 1, 1, 2, 3))
    set_partial_observability(b, True)
    b.add_action("read_route_page")
    b.add_action("select_option", {"option": chosen})
    b.set_extra("gold_selected", chosen)
    b.set_extra("real_backend", "browser")
    return b


def _u2_real_http_mock_offer_pick(ctx: Ctx):
    rng, diff = ctx.rng, ctx.difficulty
    n = scale(diff, 3, 4, 5, 6)
    offer_ids = pick_distinct_ids(rng, "OFF", n)
    winner = offer_ids[0]
    offers = []
    for i, oid in enumerate(offer_ids):
        offers.append({
            "id": oid,
            "available": i != 1,
            "risk": "low" if i == 0 else rng.choice(["medium", "high"]),
            "cost": 100 + i * 11,
        })

    b = ctx.new_task(
        title="HTTP mock API 报价选择",
        instruction=("必须调用 mock API GET /offers，选择 available=true 且 risk=low 的报价 id，"
                     "最后提交选择。"),
        capability_load={"U2": 1.0, "U1": 0.2},
    )
    b.set_initial("http.routes", {
        "GET /offers": {"status": 200, "json": {"offers": offers}},
    })
    b.add_tool("get_offers")
    b.tools[-1].update({
        "backend": "http_mock",
        "permissions": ["net:get"],
        "method": "GET",
        "url": "/offers",
        "observable_result_paths": ["observations.offers_api"],
    })
    b.add_tool("select_offer", writes=["out.offer.selected"],
               effect=effect_set("out.offer.selected", value_from="offer_id"))
    b.add_milestone("M_api", predicate("tool_called", value="get_offers"),
                    weight=1.0, epistemic_action="get_offers")
    b.add_milestone("M_select",
                    predicate("eq", path="out.offer.selected", value=winner),
                    weight=3.0, deps=["M_api"], provenance=["out.offer.selected"])
    b.add_success(predicate("eq", path="out.offer.selected", value=winner))
    inject_noise_tools(b, rng, scale(diff, 1, 1, 2, 3))
    set_partial_observability(b, True)
    b.add_action("get_offers")
    b.add_action("select_offer", {"offer_id": winner})
    b.set_extra("gold_selected", winner)
    b.set_extra("real_backend", "http_mock")
    return b


register(Template("u2_trip_replan", "U2",
                  "多约束可行最优决策（觅食→deps 门控终态 + 动态库存）",
                  _u2_trip_replan))
register(Template("u2_supplier_sourcing", "U2",
                  "供应商核验→合规入围集（set_eq + 觅食必要动作）",
                  _u2_supplier_sourcing))
register(Template("u2_evidence_foraging", "U2",
                  "证据觅食（OR 组替代来源）→ 结论综合集",
                  _u2_evidence_foraging))
register(Template("u2_real_sqlite_supplier_query", "U2",
                  "查询 seed SQLite→筛选供应商→提交终态",
                  _u2_real_sqlite_supplier_query))
register(Template("u2_real_browser_route_pick", "U2",
                  "读取静态 DOM→选择可行路线",
                  _u2_real_browser_route_pick))
register(Template("u2_real_http_mock_offer_pick", "U2",
                  "调用 HTTP mock API→选择报价",
                  _u2_real_http_mock_offer_pick))
