# Gate C real-retrieval pilot protocol

## Aim

Gate C asks whether the joint source-state abstraction retains decision value when evidence quality is produced by a real retriever rather than assigned by synthetic construction. The first pilot uses SciFact because it provides claims, a fixed scientific-abstract corpus, document-level support or contradiction labels, and sentence-level rationales. This removes the perfect-context assumption while keeping provenance auditable.

The pilot is not a direct replacement for PopQA. PopQA tests entity-valued short answers and synthetic context conflict; SciFact tests claim verification against retrieved scientific abstracts. Cross-task transfer is intentional: if the proposed state abstraction only works in the original construction, it should fail this gate.

## Retrieval baseline

The fixed corpus contains 5,183 scientific abstracts. The supervised pilot contains 693 claims with annotated evidence: 505 from the original training split and 188 from development. Retrieval is deterministic TF-IDF cosine ranking over unigrams and bigrams, with the top five abstracts persisted for each claim. Every passage record stores corpus ID, rank, score, title, full abstract, sentence boundaries, annotated evidence labels, and rationale sentence indices.

This initial sparse retriever obtains recall@5 of 0.8225 and evidence-document MRR of 0.6999. Annotated supporting evidence appears in the retrieved top five for 0.5483 of claims; annotated contradictory evidence appears for 0.2742. The remaining cases include retrieval failure and claims whose annotated documents fall outside the top five. Unannotated retrieved abstracts are not automatically labeled irrelevant or neutral.

## Experimental variables

For a claim $$q$$ and retrieved evidence set $$E$$, the real context state must be derived from evidence outcomes rather than retrieval rank alone. At minimum, evidence is separated into:

| Evidence condition | Operational meaning |
|---|---|
| supporting and sufficient | Annotated support document retrieved and rationale available |
| contradicting | Annotated contradiction document retrieved |
| mixed | Retrieved set contains support and contradiction evidence |
| unresolved retrieval | No annotated evidence appears in top $$k$$ |

The model-side variable remains protocol-specific. For SciFact it is not “knows the entity” but whether the closed-book verifier produces the correct claim verdict under a persisted prompt and decoding protocol. The answer outcome records the complete verdict, any explanation, correctness, and sentence-level evidence support.

## Required comparisons

The pilot will compare a joint evidence–memory posterior against independent marginals, joint linear modeling, scalar confidence, retrieval-only prediction, and direct answer-risk prediction. Policies choose among using retrieved evidence, using closed-book judgment, fusing, abstaining, and retrieving again. Unlike the controlled PopQA experiment, retrieve-again success and cost must be measured by extending the ranked list or changing the retrieval method; it may no longer be simulated as perfect.

A Gate-C pass requires a bootstrapped realized-cost advantage over the strongest direct baseline in at least one preregistered conservative regime, with no material reversal under reasonable evidence-label policies. Classification gains without decision gains do not pass.

## Result

The complete pilot contains 693 claims and a fixed claim-level split of 415 training, 139 validation, and 139 test claims. The 17-dimensional joint-state MLP obtains four-state accuracy 0.6906, macro-F1 0.4602, context AUROC 0.7625, and memory AUROC 0.8544. It predicts none of the eight `neither_available` test examples correctly, so classification alone is not evidence for the thesis.

Complete top-five verdict accuracy is 0.6898 closed-book, 0.8023 context-conditioned, and 0.8009 fused. The operational context label is weaker than its original name suggests: among state-2 claims—matching evidence retrieved but closed-book verdict wrong—context accuracy is only 0.5000. A retrieved gold document is therefore evidence availability, not model-conditioned evidence usability.

Empirical retrieve-again extends the same deterministic ranking from five to ten documents and generates a new context-conditioned verdict. Recall rises from 0.8225 to 0.8831, while context accuracy falls from 0.8023 to 0.7937 and fusion accuracy falls from 0.8009 to 0.7792. Deeper retrieval is measured rather than assumed perfect, and in this protocol its additional evidence and noise are harmful on average.

Under safety-first costs, the joint MLP achieves mean realized cost 0.1709, compared with 0.1924 for joint linear, 0.1835 for the validation-selected scalar baseline, 0.1878 for independent marginals, and 0.1881 for direct answer-risk prediction. The apparent advantage over direct risk is 0.0173, but its 95% claim-bootstrap interval is [-0.0112, 0.0471]. Under balanced costs, joint linear is slightly better than the MLP (0.1971 versus 0.1986); under coverage-first costs it is again better (0.1906 versus 0.1978). No learned policy selects retrieve-again at the preregistered costs because its measured answer degradation plus retrieval charge makes it dominated.

## Gate decision

**Gate C fails its registered pass criterion.** The joint posterior does not show a statistically resolved realized-cost advantage over the strongest direct baseline in a conservative regime, and simpler joint linear modeling reverses the ordering in two regimes. This is not evidence that joint dependence is useless; it is evidence that the current binary context variable and sample scale cannot support the stronger thesis.

The Stage-1 outcome is **PIVOT**. Subsequent work should distinguish at least four levels: document retrieval availability, retained rationale or evidence sufficiency, model-conditioned evidence utilization, and final answer risk. `C` should be renamed to retrieval availability in existing artifacts rather than described as context usability. Protocol-level stability of the memory variable also remains an unresolved Gate-B requirement.
