# Stage 2：分层证据状态与决策价值验证计划

**状态：计划已冻结，实验尚未开始**  
**前置结论：Stage 1 = PIVOT**  
**主要数据：SciFact 固定语料真实检索；PopQA 仅保留为受控对照**

## 1. 研究目标

Stage 2 检验一个比旧四态命题更精确的问题：真实 RAG 失败是否可以分解为检索可获得性、证据充分性、模型条件利用和闭卷能力的不稳定性，并且这种分解是否在下游动作选择上提供超过直接风险预测的增量价值。

旧命题写作 `Z=(C,M_pi)`，其中 `C` 同时承担了“文档被找回”“证据足够”“模型能使用”的含义。SciFact 结果否定了这种合并：文档进入 top-k 后，模型仍可能无法纠正错误记忆；增加更多文档还可能降低答案准确率。

Stage 2 的候选结构是：

$$
R_k \rightarrow S_{k,t} \rightarrow U_{\pi} \rightarrow Y_a,
$$

并允许：

$$
M_{\Pi} \rightarrow U_{\pi}, \qquad M_{\Pi} \rightarrow Y_a.
$$

这里 $$k$$ 是检索深度，$$t$$ 是截断/上下文构建协议，$$\pi$$ 是单个推理协议，$$\Pi$$ 是协议分布，$$a$$ 是动作。

研究目标不是证明这张图在哲学意义上“正确”，而是检验其中哪些节点可观测、可预测、可干预，并且能否降低真实决策成本。

## 2. 变量的操作定义

### 2.1 检索可获得性 $$R_k$$

定义为匹配 claim gold verdict 的标注证据文档是否进入 top-k。它由 SciFact provenance 直接构造，是当前 `c_star` 的真实含义。

$$
R_k = \mathbb{1}[\exists d \in top_k(q): d \text{ has matching annotated evidence}].
$$

当 $$R_k=0$$ 时，只能称为 unresolved retrieval。由于标注可能不完备，不能推断 top-k 中不存在任何实际有用证据。

### 2.2 证据充分性代理 $$S_{k,t}$$

Stage 2 首先审计两个代理，而非立即宣布 gold label：

- `rationale_retained`：实际截断后的 evidence prompt 是否完整保留至少一个标注 rationale sentence；
- `gold_document_retained`：匹配文档是否存在，且截断文本覆盖其第一个标注 rationale 的全部规范化 token。

如果 rationale 标注为空或句子解析无法匹配，该样本标为 `sufficiency_unresolved`，不能当负例。审计后再决定 $$S$$ 是否适合作为监督目标。

### 2.3 模型条件利用 $$U_\pi$$

不将“上下文回答正确”直接等同于利用。第一版使用答案转移类型：

| 闭卷结果 | 上下文结果 | 类型 | 可解释含义 |
|---:|---:|---|---|
| 0 | 1 | `correction` | 证据加入后纠正，正利用候选 |
| 1 | 0 | `interference` | 证据加入后退化，负利用/干扰候选 |
| 1 | 1 | `stable_correct` | 来源归因不明确 |
| 0 | 0 | `stable_wrong` | 证据不可用、未利用或任务失败 |

这只是 observational proxy。S2-E5 必须通过 rationale removal/replacement 干预，才能把 utilization 主张提升为因果证据。

### 2.4 协议分布下的闭卷能力 $$M_\Pi$$

给定一组预注册协议 $$\Pi$$：

$$
M_\Pi(q) = \frac{1}{|\Pi|}\sum_{\pi \in \Pi}
\mathbb{1}[Y_{memory,\pi}(q) \text{ correct}].
$$

同时保存协议熵、全一致率和多数决标签。原来的 `m_star` 是其中一个单协议 observation，不再视为稳定 latent knowledge。

### 2.5 动作结果 $$Y_a$$

当前动作集合保持：`use_context`、`use_memory`、`fuse`、`abstain`、`retrieve_again`。`retrieve_again` 使用 top-10 实际 verdict。若未来增加不同 retriever，必须将其视为新的具体动作，而不是抽象“更好检索”。

## 3. 核心假设

H1（断裂可识别性）：在 $$R_k=1$$ 的样本中，$$S$$ 与 $$U$$ 能解释显著的 correction/interference 差异；若不能，层级表示缺少可识别性。

H2（协议稳定性）：$$M_\Pi$$ 比单次 `m_star` 更好校准，并能改善动作风险；若多数 claim 在微小协议扰动下高度不稳定，论文必须将其描述为 inference reliability，而非 memory knowledge。

H3（结构增量价值）：分层或联合模型相对 direct action-risk baseline 在至少一个保守、预注册成本区域取得 bootstrap CI 排除零的 realized-cost 优势。

H4（检索深度条件价值）：模型能识别 top-10 对哪些样本有帮助、对哪些样本造成干扰，从而使条件式 retrieve-again 优于 always-top-5、always-top-10 和现有 top-5 policy。

H5（跨设置稳定性）：结构增量不能只存在于单一 prompt 或单一 top-k；合理标签代理、协议分布和至少一个外部 domain/model 设置下不得发生未解释的反转。

## 4. 工作包与执行顺序

### WP1：层级标签可观测性审计

实现 `scripts/audit_scifact_hierarchy.py`。它读取 top-5/top-10 retrieval 和 complete verdict JSONL，按 ID 对齐，重新构建实际截断 evidence，并输出文档召回、rationale retention、闭卷→上下文转移、top-5→top-10 转移和分组 outcome。

这一步不训练模型。其目的在于决定 $$S$$ 和 $$U$$ 的代理是否有足够样本、是否可重复构造，以及缺失标注比例是否允许监督学习。

通过条件：每个拟建模的二元目标在 train/validation/test 中都有可报告支持度；构造过程对缓存顺序不敏感；无法判断的样本显式进入 unresolved；人工抽查或程序一致性检查未发现系统性截断解析错误。

若失败：保留 $$R$$ 为 provenance，放弃把 rationale retention 当 gold sufficiency；将研究重心转向 potential outcome prediction 和干预式 utilization。

### WP2：Gate-B 协议稳定性

实现 `scripts/evaluate_protocol_stability.py`。在同一模型 revision 上运行固定协议族，并保存每个 claim、每个协议的原始输出和标准化 verdict。

建议的最小协议族包括三类 prompt 语义改写、两种答案格式约束，以及确定性 greedy 与固定低温采样。采样协议必须使用固定 seed，并至少重复三次；greedy 协议只需单次，但运行环境元数据必须持久化。协议族的确切文本在首次运行前写入 JSON 配置，不得根据 test 表现修改。

输出 claim-level mean correctness、variance、entropy、pairwise agreement、Fleiss-style agreement、transition matrix 和 protocol-conditioned calibration target。

通过条件：能构造稳定的 $$M_\Pi$$ 连续目标，且验证集上的预测与动作选择不依赖某一个异常 prompt。若协议噪声占主导，停止使用“memory state”术语，改用 closed-book protocol robustness。

### WP3：结构化模型与直接风险的公平比较

在冻结 S2-E1/S2-E2 标签后，实现统一训练与评估脚本。至少比较：

1. direct per-action risk：每个动作独立预测错误/不支持成本；
2. shared multi-task direct risk：共享 trunk、动作特定 heads；
3. factorized hierarchy：分别预测 $$R,S,U,M$$，再映射到动作 outcome；
4. joint dependency model：允许变量相关性的联合 head；
5. multinomial logistic 与 gradient-boosted tree 强基线；
6. 最佳验证集标量、检索-only、固定动作和 oracle。

模型容量要匹配样本规模。SciFact 第一轮不应训练大型深网；优先 logistic、浅层 MLP、树模型与正则化 multi-task heads。所有方法使用相同 split、相同 deployment-available features 和相同决策成本。

主要终点是 test realized mean cost。次要终点包括 regret-to-oracle、action counts、answer/abstention/retrieval rate、predicted-vs-realized calibration gap、NLL/Brier 和各失败类型 AUROC/AUPRC。

通过条件：结构方法相对最强 validation-selected direct-risk baseline 的 claim-bootstrap 优势在至少一个预注册保守成本区域中 CI 排除零，并且没有在相邻成本区域发生更大的反向损失。

### WP4：条件式 retrieve-again

将 top-5→top-10 看作具有实际 potential outcome 的动作。定义：

$$
\Delta_{10-5}(q) = L(Y_{top10}) - L(Y_{top5}) + c_{retrieve}.
$$

训练只能使用 top-5 时可得的 observation 与 retrieval provenance 预测 $$\Delta$$。不能使用 top-10 verdict、top-10 gold recall 或运行后才可见的特征作为部署时输入。

比较 always top-5、always top-10、基于 top-5 score margin 的阈值策略、direct uplift/risk、层级策略和 oracle retrieve policy。报告帮助、无变化、伤害三类的 precision/recall 以及净成本。

若任何合理 retrieval cost 下都无法优于 always top-5，应保留“更多检索通常无帮助”的负结果，并停止把 retrieve-again 作为核心贡献。

### WP5：证据利用干预

对一组按 $$R/S/M$$ 分层抽取的 claim 构造至少三种 evidence intervention：保留 rationale、移除 rationale、用同主题非证据句替换 rationale。若存在支持/反驳冲突，再增加冲突文档注入。

同一 claim 的所有干预共享 split 和 generation protocol。主要比较 verdict 转移、logit/hidden/attention observation 变化以及 action-policy 变化。该工作包用于区分“证据存在但未使用”和“证据本身不充分”。

不要先依赖 attention 权重宣称因果利用；只有答案对证据干预呈系统响应，内部信号才可作为 utilization predictor 分析。

### WP6：多文档冲突

在 WP5 之后构造多文档集合，因素包括支持/反驳比例、gold rank、重复证据、主题相似无关文档和总上下文长度。设计采用可控 factorial grid，但样本量根据 pilot 方差做 power/precision 规划，不能凭整齐数字拍脑袋。

核心结果是不同冲突强度下的 correction、interference、abstention 和 retrieve-again 决策曲线，而不是只报告总体准确率。

### WP7：跨域与跨模型验证

只有 WP3 在 SciFact 上通过后才执行。选择至少一个不同任务结构的数据集和一个不同模型家族。冻结标签定义、feature groups、成本区域和主要指标，允许的唯一适配是任务答案 evaluator 与证据 schema。

若只换模型不换任务，只能主张 cross-model；若只换数据不换模型，只能主张 cross-domain。不得合并措辞。

### WP8：论文与代码重构

在证据充分后重写 `docs/research-contract.md`、README、命名和 pipeline。旧 `c_star/m_star/state` 字段可在 schema v1 产物保留，但新 schema 使用 `retrieval_available`、`evidence_sufficiency_proxy`、`protocol_answerability` 和 `utilization_transition`。

论文贡献必须根据结果三选一：决策方法、失败归因框架，或经过严格验证的负结果。不要在方法失败时退回“17 个信号的新颖组合”。

## 5. 数据切分与防泄漏

SciFact 沿用 `artifacts/scifact/split.npz`。协议稳定性产生的同 claim 多条 generation 仍属于同一 claim，不允许跨 split。证据干预产生的同 claim 变体也必须保持在原 split。

所有类别权重、超参数、特征选择、成本阈值和模型选择使用 train/validation。Test 只在方法冻结后运行一次主要分析；调试输出不得据此更换标签或模型。若不可避免地进行了探索性 test 查看，必须将结果标为 exploratory，并另建外部验证集作为 confirmatory endpoint。

bootstrap 单位是 claim，而不是 generation 或 intervention row。多变体实验应 cluster bootstrap claim。

## 6. 成本评估

继续使用 answer error、unsupported、abstain 和 retrieval cost，但从三个离散 regime 扩展为预注册二维/三维成本网格。Safety-first 仍作为主要保守点。报告方法优胜区域、反转边界和 worst-case regret。

不得根据 test 结果挑选“最能体现优势”的成本。正文中必须同时展示固定 anchor regimes 和连续 sensitivity surface。若成本只在极窄区域支持结构模型，应明确写作 limited operational value。

## 7. 统计与不确定性

主要比较采用 paired claim bootstrap，至少 5,000 次；报告均值差、95% percentile CI 和结构方法胜出的 bootstrap 比例。对于稀有类型同时报告原始 support，不用 AUROC 掩盖极小样本。

多次模型 seed 应将训练随机性和 test sampling uncertainty 分开报告。可以先对 seed 平均后的 claim loss 做 bootstrap，也应补充 per-seed range。若结论依赖一个 seed，不通过。

校准至少报告 NLL、Brier、ECE 与 predicted-realized action-cost gap。ECE 只能作为描述性指标，不能单独用来证明校准改善。

## 8. 工程交付标准

每个工作包必须产生：一份可运行脚本、一份机器可读 JSON、一份简洁 Markdown 审计、一条 README 状态更新。模型 checkpoint 必须包含 feature names、normalization、temperature、label schema version 和 protocol hash。

新增 JSONL 使用稳定 claim ID。每次 join 都要检查计数、ID 集合、gold label 与 source split。每个生成产物持久化完整 prompt、raw answer、normalized outcome、模型名称、解码参数和数据协议。

所有 heavyweight generation 应可缓存、可恢复，并在 cache reuse 时验证必需字段。不得用输出文件存在作为唯一完整性判断。

## 9. 阶段决策

Stage 2 结束时必须给出以下之一：

- **GO — STRUCTURED DECISION MODEL**：分层/联合表示在真实检索与外部验证中提供稳定、显著、具操作意义的动作成本优势；
- **PIVOT TO FAILURE DIAGNOSTICS**：层级变量能可靠解释检索—利用断裂，但不能改善动作决策；
- **STOP STRUCTURED-STATE THESIS**：直接风险或简单模型解释所有收益，结构状态既不改善决策也不提供稳定诊断。

在 WP3 前不得宣布 GO；在 WP7 前不得主张跨域或跨模型普适性。

## 10. 近期里程碑

近期只按以下顺序推进：

1. 完成 S2-E1 层级标签可观测性审计；
2. 根据审计冻结 schema v2；
3. 完成 S2-E2 协议稳定性并定义 $$M_\Pi$$；
4. 注册成本网格和模型超参数范围；
5. 完成 S2-E3 公平决策比较；
6. 根据结果决定是否进入 retrieve-again、干预、多文档和跨域扩展。

这个顺序是科学约束，不只是项目管理偏好。
