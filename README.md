# ModelMark / AGENIX Agentic Test Engine

ModelMark 是一个面向大模型 Agent 能力的自动化评测项目。仓库里的评测引擎名为 **AGENIX**，它关注的不是模型能不能把一道题“说得像对的”，而是模型能不能在一个可验证的环境里完成任务：读数据、调用工具、处理中间状态、避开安全陷阱，并把最终结果真正写到正确位置。

一句话概括：

> 把模型评测从“看回答像不像”推进到“看它有没有真的把事做成”。

---

## 为什么需要这个引擎

很多传统 benchmark 更像考试题：给模型一段 prompt，然后看最终文本答案是否接近标准答案。这对问答、数学、写作很有用，但很难评价真正的 Agent 能力。

真实 Agent 任务通常更复杂：

- 信息不一定一次性给全，模型要主动查工具、读文件或检索数据；
- 任务可能有 10 步、20 步甚至更多，中途环境状态还会变化；
- 成功不是“解释得合理”，而是系统里的状态真的被改对了；
- 有些工具是危险的，模型即使完成了主任务，也不能越权、泄密或触发蜜罐；
- 多模态任务不能只说“我看到了”，还要能对齐图表、OCR、表格、位置框和反事实证据。

AGENIX 的设计核心是 **verifier-first**：凡是能用程序检查的，就不用主观打分。模型说自己完成了任务不算数；沙箱终态里的谓词为真，并且这个状态确实由模型动作造成，才算数。

---

## 它是怎么工作的

直观流程如下：

```text
任务 JSON
  -> 模型适配器生成动作和最终提交
  -> 确定性沙箱重放工具调用
  -> 记录 Trace、终态和 provenance
  -> verifier / scoring 模块打分
  -> 聚合统计与 Markdown / HTML 报告
```

这里的“任务”不是普通文字题，而是一份可执行声明：

- 初始状态是什么；
- 可用工具有哪些，每个工具会改变什么状态；
- 哪些状态谓词代表任务成功；
- 哪些里程碑必须按依赖顺序完成；
- 哪些行为属于 critical violation；
- grounding、恢复、效率、可靠性等子指标如何计算。

这样做的好处是，评测不依赖模型自述，也不依赖评审者心情。每次运行都能落到同一套 trace、终态、谓词和统计报告上。

---

## 这个项目的特点

**1. 状态优先，而不是文本优先**

模型最终输出再漂亮，如果没有把沙箱里的目标状态写对，就不会被判成功。

**2. 因果归属，而不是蒙对**

某个状态即使满足了谓词，也要检查它是不是由模型动作造成的。环境事件或初始状态“免费满足”的条件不会给模型加分。

**3. 安全违规不可补偿**

泄密、越权工具调用、触发蜜罐、执行注入指令等 critical violation 会让该任务成功硬归零。不能用“主任务做对了”来抵消安全问题。

**4. 多模态 grounding 有可计算证据**

图表数值、OCR 文本、bbox / IoU、表格结构、反事实最小对等都用 typed verifier 检查，而不是只靠相似度或 LLM judge。

**5. 长程任务看过程稳定性**

长程评测不仅看最后有没有做对，还看里程碑 DAG、故障恢复、状态漂移、重复错误和 pass^k 可靠性。

**6. 报告不只给一个总分**

AGENIX 输出维度画像、置信区间、pass@k / pass^k、成本与安全轴，避免一个总分掩盖模型真实短板。

---

## 仓库结构

```text
.
├── README.md                         # 顶层说明
├── AGENIX-评测报告.md                 # 真实评测报告
├── AGENIX-评测报告.html               # 真实评测报告 HTML 版
├── AGENIX-完整测试报告.md             # 完整测试报告
├── AGENIX-完整测试报告.html           # 完整测试报告 HTML 版
├── 测试参考.md                        # 早期 baseline 参考
├── design/                           # 多模型协同设计、辩论与最终规格
│   ├── round1/
│   ├── round2/
│   ├── round25/
│   └── final/AGENIX-Engine-Spec.md
├── test-engine-design/                # 早期 test engine 设计材料
└── engine/                            # 可运行 Python 评测引擎
    ├── schema.py                      # Task / Trace / submission schema
    ├── sandbox.py                     # 确定性沙箱与 provenance 记录
    ├── orchestrator.py                # 任务加载、模型运行、报告构建
    ├── run_demo.py                    # 离线 stub demo
    ├── run_eval.py                    # 真实模型横评 runner
    ├── report.py                      # JSON 结果转 Markdown
    ├── scoring/                       # 评分内核
    ├── tool_backends/                 # 工具后端
    ├── adapters/                      # OpenAI 兼容模型适配
    ├── generators/                    # 程序化任务生成与抗污染支持
    ├── assets/                        # 多模态资产与 GT
    ├── tasks/                         # 手写与生成任务 JSON
    ├── results/                       # 已保留的结果、CSV、图表
    └── tests/                         # pytest 测试
```

---

# 专业说明

## 测试内容：U1-U6 能力维度

AGENIX 将前沿 Agent 能力拆成六个可分别观察的失败模式：

| 维度 | 名称 | 主要测试内容 |
|---|---|---|
| U1 | 工具 / 状态达成 | 读数据、调用工具、提交终态、完成真实状态谓词 |
| U2 | 条件规划 / 信息觅食 | 数据不在上下文内时主动读取、分支决策、动态重规划 |
| U3 | 多模态 grounding | 图表、收据、财务表、结构化文档、OCR、bbox、数值核验 |
| U4 | 长程状态管理 | 多步配置迁移、状态变化、故障注入、恢复与稳定性 |
| U5 | 校准 / 选择性预测 | 置信度、弃答、Brier / ECE / risk-coverage、抗无脑 hedge |
| U6 | 安全与鲁棒性 | prompt injection、越权工具、蜜罐、数据外泄、ASR |

这些维度不强行合成一个单一“能力真值”。报告优先展示维度向量，并在需要标量时附带权重敏感性与不可区分标注。

---

## 任务与运行时契约

任务以 JSON 声明，核心字段由 `engine/schema.py` 定义。一个任务通常包含：

- `initial_state`：沙箱初始状态；
- `tools`：模型可调用工具及其 effect；
- `milestones`：过程里程碑、依赖关系、权重和 provenance 要求；
- `success_predicates`：最终成功谓词；
- `critical_violations`：安全硬零条件；
- `grounding`：多模态 typed verifier；
- `oracle_plan`：参考最优路径，用于成本与效率比较；
- `difficulty_knobs` / `canary`：难度旋钮与抗污染标识。

`engine/sandbox.py` 会确定性重放模型提交的动作，产生：

- `events`：工具调用、错误、环境事件；
- `final_state`：运行结束后的真实状态；
- `provenance`：每个状态路径由谁写入；
- `cost`：工具调用、步数等成本信号。

评分模块只读取 trace 与终态，不相信模型的自述。

---

## 评分方法

### 1. Milestone DAG + provenance

里程碑不仅要谓词为真，还要满足依赖关系，并通过因果归属检查。环境自动写入或初始状态满足的谓词不会被记为 by-agent 成果。

### 2. Safety hard-zero

critical violation 命中时：

- `raw_success` 可以仍然为真；
- `success` 必须归零；
- `ASR` 单独记录；
- 该失败不能被过程分、多模态分或效率分补偿。

### 3. Grounding 双轨

AGENIX 区分两类 grounding：

- synthetic / symbolic track：由程序化 GT 精确生成，适合高频重采样和抗污染；
- real / pixel track：接近真实媒介，适合生态效度审计。

双轨不会强行合并成一个数。报告通过 Spearman rho 数据门判断是否可以用 synthetic track 作为 headline，或需要双轨并列。

### 4. Reliability

可靠性同时报告：

- per-run success；
- pass@k：多次尝试能否至少成功一次；
- pass^k：连续多次都成功的稳定性；
- expected milestone completion：长程任务中的连续过程进展。

这样可以区分“偶尔能做成”和“稳定能托付”。

### 5. Statistics

统计层包含：

- 两级聚类 bootstrap 置信区间；
- GLMM / mixed-effects 近似或 statsmodels 后端；
- Holm / BH 多重比较校正；
- Cliff's delta 效应量；
- Dirichlet 权重敏感性；
- IRT item calibration 与参数恢复门，仅用于选题和抗饱和，不进 headline。

LLM-as-judge 只用于残余主观项，默认不进入 headline；可用多评委、位置翻转、Krippendorff alpha 与人类校准门控。

---

## 快速开始

离线 demo 不需要 API key：

```bash
cd engine
pip install -r requirements.txt
python run_demo.py
python run_eval.py --mock
python -m pytest tests -q
```

在仓库根目录也可以直接运行测试：

```bash
python -m pytest engine\tests -q
```

当前本地验证结果：

```text
457 passed in 36.03s
```

真实模型横评示例：

```bash
cd engine
cp configs/models.example.json configs/models.json

# 推荐使用环境变量，不要把真实密钥写进配置或提交到仓库
setx ARK_API_KEY "your_key"
setx DEEPSEEK_API_KEY "your_key"
setx MOONSHOT_API_KEY "your_key"
setx ZHIPUAI_API_KEY "your_key"

python run_eval.py --config configs/models.json --require-real --difficulty medium --n-runs 5
python report.py results/eval_<timestamp>.json
```

如果没有 key 或网络不可用，runner 可以回退到 mock profile 做流水线 dry-run。正式 headline 横评应使用 `--require-real` 或 `--no-mock-fallback`，避免 mock 结果混入官方排名。

---

## 已有结果

顶层报告记录了对 `doubao-seed-evolving` 的真实评测与方法学复盘：

- [AGENIX-评测报告.md](AGENIX-评测报告.md)
- [AGENIX-评测报告.html](AGENIX-评测报告.html)
- [AGENIX-完整测试报告.md](AGENIX-完整测试报告.md)
- [AGENIX-完整测试报告.html](AGENIX-完整测试报告.html)

核心经验是：朴素 harness 曾把强模型误判为多步任务很弱；经过多轮公平化，问题被定位为测试器 / 适配器伪影，而非模型真实能力。这个项目因此不仅是一次模型评测，也是一套如何构建可信 Agent benchmark 的工程记录。

---

## 设计文档

完整规格来自多模型协同设计与交叉辩论：

- [design/final/AGENIX-Engine-Spec.md](design/final/AGENIX-Engine-Spec.md)
- [test-engine-design/00_DESIGN_BRIEF.md](test-engine-design/00_DESIGN_BRIEF.md)
- [engine/README.md](engine/README.md)

其中 `design/final/AGENIX-Engine-Spec.md` 记录了 CP1-CP8 决策，包括跨维不相乘、安全 hard-zero、GLMM 主干、双轨 grounding、provenance 因果门控、抗污染、四类 reliability 指标和效率正交。

---

## 已知限制

- 当前任务环境是声明式沙箱，不等同于完整真实 OS / 浏览器 / 企业系统。
- 真实多模型横评依赖各 provider API key、网络质量和统一预算设置。
- LLM judge 面板机制已具备工程接口，但真正跨家族真评委需要更多独立模型 key。
- 部分真实轨多模态资产仍是程序化渲染或确定性像素验证，人工标注真实视频 / 文档集可作为后续扩展。
- `engine/results/` 保留了关键历史结果与图表；中间 partial/log/progress 文件不进入公开提交。

---

## 许可

本仓库当前尚未添加开源许可证。使用或分发前请先确认授权边界。
