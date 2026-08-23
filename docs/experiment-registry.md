# Credence 实验注册表

本文把 Stage 2 计划转换为可执行实验。状态只能使用 `planned`、`running`、`complete`、`failed`、`superseded`。任何实验开始前先冻结其配置；完成后填写实际命令、产物 hash、结果与判定。

## S2-E1：SciFact 层级标签可观测性审计

**状态：planned**  
**优先级：P0；下一项必须执行的实验**

**目的。** 确定 retrieval availability、rationale retention、observational utilization 和 top-k interference 是否能从现有产物无歧义构造，并测量其支持度与分歧。

**输入。**

```text
artifacts/scifact/retrieval_pilot.jsonl
artifacts/scifact/retrieval_pilot_top10.jsonl
artifacts/scifact/complete_verdicts.jsonl
artifacts/scifact/complete_verdicts_top10.jsonl
artifacts/scifact/split.npz
```

**实现。** 新建 `scripts/audit_scifact_hierarchy.py`。按 `id` 建索引并验证 ID 集合一致。用 `evaluate_scifact_answers.py` 的相同 `truncate_words` 逻辑重建 top-5 160 words/passage 与 top-10 120 words/passage 文本。对匹配 verdict 的 gold passage 判断：文档进入 top-k、gold rationale index 是否存在、规范化 rationale 是否完整出现在截断 passage 中。

**必须输出的行级字段。**

```text
id, split, gold_verdict
r5, r10
rationale_available_5, rationale_available_10
rationale_retained_5, rationale_retained_10
closedbook_correct, context5_correct, fusion5_correct
context10_correct, fusion10_correct
memory_to_context_transition
context5_to_context10_transition
retrieval_gain_10, rationale_gain_10
```

**必须输出的聚合。** 总体及 train/validation/test 支持度；按 `R/S/M` 分组的上下文和融合准确率；`00/01/10/11` 转移计数；top-10 help/harm/no-change；rationale annotation missing 与 parse failure；所有分母。

**产物。**

```text
artifacts/scifact/hierarchy_audit.json
artifacts/scifact/hierarchy_audit_rows.jsonl
docs/stage-2-hierarchy-audit.md
```

**验证。** 693 个唯一 ID；top-5/top-10 gold verdict 一致；旧 `c_star` 与重构 `R_k` 完全一致，否则实验 failed；每个聚合可由 row artifact 重算。

**决策。** 若 rationale retention 可稳定构造且各 split 有足够正负支持，则作为 $$S$$ proxy；否则仅保留为审计字段，不做监督目标。无论结果如何都不得把 `R` 恢复称为 usability。

---

## S2-E2：闭卷与上下文协议稳定性

**状态：planned**  
**前置：S2-E1 complete**

**目的。** 将一次 `m_star` 升级为协议分布下的 answerability，并测量 context/fusion 是否同样对轻微提示变化敏感。

**输入。** `retrieval_pilot.jsonl`、固定 split、S2-E1 schema decision。

**预注册配置。** 新建 `configs/scifact_protocols_v1.json`，至少包含三种语义等价 prompt 模板、两种 label-format instruction。Greedy 组合不重复；低温采样组合固定温度、top-p 与三 seed。配置文件生成后先记录 SHA-256，再运行 generation；不得根据 test 输出编辑同一 v1 文件。

**实现。** 新建 `scripts/evaluate_protocol_stability.py`。支持断点缓存，但逐行验证 `id/protocol_id/raw_answer/verdict/correct/prompt`。模型 revision、dtype、max tokens 和 tokenizer 配置写入 manifest。

**主要指标。** Claim-level mean correctness、variance、entropy、all-protocol agreement、majority correctness、pairwise transition；按 source split 与旧 state 分组。比较 single-run `m_star` 与 $$M_\Pi$$ 对后续动作风险的预测。

**产物。**

```text
configs/scifact_protocols_v1.json
artifacts/scifact/protocol_generations_v1.jsonl
artifacts/scifact/protocol_stability_v1.json
docs/gate-b-protocol-audit.md
```

**通过条件。** 协议族可复现；所有 claim 覆盖一致；连续 $$M_\Pi$$ 明确定义；结论不依赖删除某一个 prompt。若单次与多数决标签分歧率较高，schema v2 禁止 binary memory-state 表述。

---

## S2-E3：结构模型与直接风险比较

**状态：planned**  
**前置：S2-E1、S2-E2 complete；schema v2 frozen**

**目的。** 检验分层结构是否在动作成本上提供超过 direct risk 的增量，而不是只改善中间标签。

**输入。** `features.npz`、`split.npz`、hierarchy rows、protocol stability、top-5/top-10 outcomes。

**方法。** Direct per-action logistic；shared shallow MLP direct risk；multinomial logistic；gradient-boosted tree；factorized `R/S/U/M`；joint shallow model；best scalar；retrieval-only；fixed actions；oracle。SciFact 小样本阶段禁止未经论证的大模型 head。

**特征公平性。** 主比较仅使用动作前可得特征。任何使用 gold provenance 的模型必须标为 diagnostic oracle，不得与部署方法混在一起。Top-10 后验信息不得用于决定是否 retrieve again。

**主要终点。** Fixed test realized mean cost；相对 strongest validation-selected direct baseline 的 paired claim bootstrap delta。主要 anchor 为 safety-first；同时报告完整成本敏感性网格。

**产物。**

```text
artifacts/scifact/stage2_model_comparison.json
artifacts/scifact/stage2_predictions.jsonl
docs/stage-2-model-comparison.md
```

**通过条件。** 至少一个保守成本区域内，结构方法相对 direct baseline 的 95% CI 排除零；相邻成本区域无更大反向 regret；至少两 training seeds 结论同向。否则不能进入“结构化决策模型”GO。

---

## S2-E4：条件式 retrieve-again

**状态：planned**  
**前置：S2-E3 complete，且继续研究被判定有价值**

**目的。** 预测 top-10 相对 top-5 的条件收益与伤害。

**标签。** 使用 top-5/top-10 实际 context verdict 和 unresolved-evidence charge 计算每 claim loss delta。分别报告无 retrieval charge 的 outcome delta 与多个固定 charge 下的 action delta。

**基线。** Always top-5；always top-10；top-score/margin threshold；logistic uplift；direct risk；hierarchical policy；oracle。

**禁止泄漏。** 策略输入只能使用 top-5 后可得信息。Top-10 score、gold rank 6–10、top-10 verdict 只用于训练标签或 oracle，不进入部署预测特征。

**产物。** `artifacts/scifact/retrieve_again_evaluation.json` 与 `docs/retrieve-again-audit.md`。

**通过条件。** 在一段非零、合理 retrieval-cost 区域，条件策略显著优于 always top-5 和 always top-10；否则保留负结果并将该动作降级为非核心。

---

## S2-E5：证据利用干预

**状态：planned**  
**前置：S2-E1 schema frozen；S2-E3 完成可优先于或晚于本实验**

**目的。** 通过证据内容干预区分 evidence sufficiency 与 model utilization。

**抽样。** 在 train/validation 中开发，在 test 中冻结 confirmatory subset。按 `R`、rationale retention、闭卷多数决和 correction/interference 分层。样本量依据 pilot 转移率决定，并记录选择规则。

**干预。** 原始截断证据；保留 rationale 的最小证据；删除 rationale；同主题非证据句替换；条件允许时加入相反标签文档。所有变体保持 claim、答案协议和总长度尽可能匹配。

**指标。** Verdict transition、correctness、context reliance proxy、observation shift、policy shift。Attention 只能是关联特征，不能独立证明利用。

**产物。** `artifacts/scifact/evidence_interventions.jsonl`、`evidence_intervention_evaluation.json`、`docs/evidence-utilization-audit.md`。

---

## S2-E6：多文档冲突实验

**状态：planned**  
**前置：S2-E5 complete**

**目的。** 系统改变支持/反驳/无关/重复文档组成、rank 和上下文长度，测量模型利用与决策变化。

**设计。** 先 pilot 估计方差，再冻结 factorial cells。每个 claim 的所有变体 cluster 在一起。Gold claim、文档 provenance、句级标签和构建 seed 全部保存。

**主要输出。** Conflict strength 曲线、correction/interference rate、选择动作、calibration 与 worst-case regret。

**产物。** `artifacts/scifact/multidoc_conflict_*` 和 `docs/multidoc-conflict-audit.md`。

---

## S2-E7：跨模型与跨域验证

**状态：planned**  
**前置：S2-E3 pass**

**目的。** 检验主结论是否跨模型或跨任务，而不是只拟合 SciFact/Qwen 设置。

**冻结项。** Schema v2、主要成本、模型族、特征组和 primary endpoint。允许适配项只有任务 loader、答案 evaluator 和 evidence schema。

**最低要求。** 至少一个不同模型家族和一个不同任务结构；可分两步运行并分别表述 cross-model/cross-domain。

**失败处理。** 若方向反转，先分析 label observability、protocol distribution 和 retrieval shift，不得只报告平均值掩盖 domain failure。

---

## 注册修改规则

实验运行前允许修改 `planned` 条目，但必须在对应结果中记录最终注册文本 hash。实验开始后若发现实现错误，新建版本号（例如 S2-E2-v2），保留旧产物并标为 failed/superseded。不得原地覆盖不利结果后继续使用同一实验 ID。
