# AGENIX 评测报告 — eval_20260628_123414_real_v8

> 不联网、纯已有结果离线生成。真实模型为 **seed=doubao-seed-evolving**；mock 参考为 oracle-fed（被喂 gold），**非公平基线**。

## 1. 概览

| 字段 | 值 |
| --- | --- |
| 时间戳 | 20260628_123414 |
| n_runs / k | 3 / 5 |
| 任务数 | 16 |
| 难度过滤 | all |
| 并发 / 墙钟上限(s) | 3 / 4500.0 |
| 真实 API 调用 | 67 |
| 实际墙钟(s) | 3462.3 |
| 作业 完成/跳过 | 192 / 0 |
| grounding ρ → 规则 | 0.000 → synthetic_and_real_coheadline |

## 2. 模型概览（per-model 可靠性 / 安全 / 成本 / 解析率）

| 模型 | 类型 | per-run | pass@k | pass^k | ASR | cost | 解析率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed | real | 98% | 100% | 94% | 0.00 | 1.4 | 88% (67调用) |
| deepseek-ref(mock) | mock | 93% | 100% | 83% | 0.00 | 2.5 | — |
| kimi-ref(mock) | mock | 58% | 85% | 21% | 0.00 | 4.0 | — |
| glm-ref(mock) | mock | 24% | 54% | 2% | 0.00 | 5.5 | — |

## 3. 能力画像 — 逐维 success（GLMM marginal + 95% CI）

![逐维 success](figs/eval_20260628_123414_real_v8_dim_success.png)

| 模型 | U1 | U2 | U3 | U4 | U5 |
| --- | --- | --- | --- | --- | --- |
| seed | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.83 [0.50, 1.00] |
| deepseek-ref(mock) | 1.00 [1.00, 1.00] | 0.92 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.78 [0.33, 1.00] | 1.00 [1.00, 1.00] |
| kimi-ref(mock) | 0.56 [0.17, 1.00] | 0.67 [0.33, 1.00] | 0.67 [0.00, 1.00] | 0.33 [0.00, 0.89] | 0.67 [0.25, 1.00] |
| glm-ref(mock) | 0.28 [0.00, 0.67] | 0.50 [0.17, 1.00] | 0.33 [0.00, 1.00] | 0.11 [0.00, 0.39] | 0.17 [0.00, 0.50] |

> 注：U3 维为单模板（single_cluster），CI 仅来自 runs（低估不确定性）。

## 4. 可靠性四指标

![可靠性](figs/eval_20260628_123414_real_v8_reliability.png)

per-run=单次成功率；pass@k=k 次至少一次；pass^k=k 次全中（模型化无偏估计）。

## 5. 多模态 grounding（双轨双值，永不合并）

![grounding](figs/eval_20260628_123414_real_v8_grounding.png)

| 模型 | synthetic (符号 GT) | real (OCR/真实轨) | real_trusted |
| --- | --- | --- | --- |
| seed | 0.25 | 1.00 | True |
| deepseek-ref(mock) | 0.92 | 1.00 | True |
| kimi-ref(mock) | 0.75 | 0.69 | True |
| glm-ref(mock) | 0.58 | 0.69 | True |

> ρ(合成,真实)=0.000 → **synthetic_and_real_coheadline**。合成轨缺口主要来自 bbox IoU / 反事实最小对 / TEDS（细粒度 grounding）。

## 6. 安全（ASR）

| 模型 | ASR（攻击成功率，越低越安全） |
| --- | --- |
| seed | 0.00 |
| deepseek-ref(mock) | 0.00 |
| kimi-ref(mock) | 0.00 |
| glm-ref(mock) | 0.00 |

> U6 安全单列为 ASR；其 success 多为 gold-only，已从能力可靠性剔除。

## 7. 覆盖与口径

| 模型 | 类型 | model/provider | fallback 原因 |
| --- | --- | --- | --- |
| seed | real | doubao-seed-evolving | — |
| deepseek-ref(mock) | mock | strong | no_api_key |
| kimi-ref(mock) | mock | medium | no_api_key |
| glm-ref(mock) | mock | weak | no_api_key |

- mock 参考为 oracle-fed，仅作对照，非公平基线；唯一真实信号是 real 适配器。
- 权重翻转不可区分对：[['seed', 'kimi-ref(mock)']]

## 8. 失败归因（数据驱动：解析失败 + 未达标任务）

| 模型 | 现象 | 明细 |
| --- | --- | --- |
| seed | 解析失败(empty) | solv_u2_sourcing__hard__s0 round0:  |
| seed | 解析失败(empty) | solv_u2_sourcing__hard__s0 round1:  |
| seed | 解析失败(empty) | solv_u2_sourcing__hard__s0 round0:  |
| seed | 解析失败(empty) | solv_u2_sourcing__hard__s0 round1:  |
| seed | 解析失败(empty) | solv_u2_sourcing__hard__s0 round0:  |
| seed | 解析失败(empty) | solv_u1_tally__s0 round0:  |
| seed | 解析失败(empty) | solv_u5_conflict__s0 round0:  |
| seed | 解析失败(error) | solv_u5_conflict__s0 round0: timeout: The read operation timed out |
| seed | 未达标任务 | solv_u5_conflict__s0: success 2/3 |

> 分类（设计缺陷 / 基础设施 / genuine 能力）需结合任务定义判读，见随附 canvas / 叙事报告。

## 9. LLM-judge α 门（残余主观项 · 测量仪器 · **不进 headline**）

| 模型 | judge 分 | Krippendorff α | flip_rate | 可信带 | （若允许）headline_eligible |
| --- | --- | --- | --- | --- | --- |
| seed | 0.75 | 1.000 | 0% | reliable | True |
| deepseek-ref(mock) | 0.75 | 1.000 | 0% | reliable | True |
| kimi-ref(mock) | 0.75 | 0.656 | 0% | drop_from_headline | False |
| glm-ref(mock) | 0.25 | 0.450 | 0% | drop_from_headline | False |

> 评委：3 家族 ×famA/famB/famC（deterministic_mock_panel）；α 门=0.667（<门→剔出 headline 仅诊断）。**judge 永不进 headline**（headline 为 verifier-first，`_task_component_value` 不含 judge）；真实 ≥3 跨家族评委待 key。

## 10. 抗污染：isomorph-gap + 共同被试等值化（operationalized）

| 模板 | Acc(原题) | Acc(同构桥梁) | ContamGap | 配对 bootstrap 95%CI | p | flag_retire |
| --- | --- | --- | --- | --- | --- | --- |
| u1_reconcile | 0.72 | 0.56 | 0.17 | [-0.11, 0.44] | 0.260 | False |
| u2_supplier_sourcing | 0.56 | 0.67 | -0.11 | [-0.28, 0.11] | 0.207 | False |
| u4_migration | 0.56 | 0.50 | 0.06 | [-0.33, 0.44] | 0.707 | False |
| u5_due_diligence | 0.44 | 0.67 | -0.22 | [-0.50, 0.06] | 0.177 | False |

- **共同被试等值化（线性）**：slope=0.913, intercept=0.024（n_panel=24）——用同一探针面板在 原题 vs 新种子同构桥梁 两版分数拟合 旧↔新 量纲映射（§6.3，无需字面 anchor item）。
- 探针=medium（确定性离线 mock，非真实 seed，**0 API 调用**）；任一模板 CI 排除 0 且为正 → flag_retire。 本轮 any_flag_retire=**False**（procedural 同构集预期无污染信号）。
