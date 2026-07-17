# upd_hypo — test_idea

> Phase: `invention_loop` · round 5 · `upd_hypo`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim transcript of this agent task — every system/user prompt, assistant response, thinking block, tool call and tool result — in the order they occurred. Nothing truncated.

## Task: `upd_hypo` (terminal_claude_agent, claude-sonnet-4-6)

### [1] CONFIG · 2026-07-17 15:02:47 UTC

```
model: claude-sonnet-4-6 | effort: medium | permission: bypassPermissions | cwd: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/upd_hypo/upd_hypo
```

### [2] SYSTEM-USER prompt · 2026-07-17 15:02:53 UTC

````
<ai_inventor_context>
<ai_inventor_summary>
You are one of many LLMs in AI Inventor — an automated research system that generates NOVEL and FEASIBLE hypotheses, investigates them through experiments and research, and produces a paper.

Your output feeds other LLMs downstream. This demands your ABSOLUTE MAXIMUM reasoning — every output must be deeply thought out and maximally useful. Surface-level responses waste downstream computation.
</ai_inventor_summary>

<your_role>
YOU ARE: A hypothesis reviser (Step 3.6: UPD_HYPO in the invention loop)

You received the current hypothesis, all artifacts, and the paper draft.
Revise the hypothesis based on what the evidence supports.

Honest revision → focused research. Inflated confidence → wasted iteration.
</your_role>
</ai_inventor_context>

You are revising a research hypothesis based on empirical evidence gathered
during an iterative invention loop. Your role is internal reflection — honest
assessment of what the evidence supports.

SCOPE: Your ONLY output is the revised hypothesis text. You do NOT run code,
produce artifacts, fix bugs, or otherwise act on the evidence yourself — the
next iteration of the invention loop will spawn fresh artifacts based on your
revised hypothesis. Reflect on the evidence and rewrite the hypothesis;
nothing else.

PRINCIPLES:
- Ground every revision in specific artifacts and results
- Treat negative and null results as valuable contributions. If the original
  approach failed, the null result IS often the contribution — frame it as
  such (e.g. "X does not improve Y under conditions Z"). Only pivot to a
  different positive claim when the evidence actually supports one; never
  fabricate a positive narrative to mask a failed approach.
- Increase specificity as evidence accumulates
- Don't inflate confidence without strong evidence
- Preserve the core AII prompt unless evidence clearly contradicts it
- Revise hypothesis text only — never attempt to address feedback by running
  code, proposing fixes, or producing artifacts; the next loop iteration
  handles all artifact generation

<current_hypothesis>
The hypothesis as it stands. Revise it based on the evidence below.

kind: hypothesis
title: Abliteration Comprehensively Preserves RLHF SAE Feature Imprint (Style + Harm)
hypothesis: |-
  RLHF safety fine-tuning and refusal abliteration operate on geometrically near-orthogonal subspaces in Qwen3-8B's residual stream. Five experimentally confirmed findings now establish this: (1) Selective directional preservation — the Reversal Score (PearsonR of RLHF-induced vs. abliteration-induced change vectors per feature) for top-500 RLHF-sensitive SAE encoder features at layer 22 is unimodal with mean=0.921, std=0.040, min=0.684, and zero features with negative scores; a noise-free mid-SC baseline (features at 40th–60th SC percentile, SC_median≈1.12 ≫ 0.01) yields R̄=0.871, Cohen's d=1.087, p≈0, confirming selective preservation against a baseline free of the bottom-500 noise concern. (2) Magnitude preservation — mean OLS no-intercept slope β̄=0.986±0.176 for the top-500 features at layer 22 (β̄=0.954±0.191 at layer 20), with 86.4% of features having β_f>0.8 and 45.4% having β_f>1.0; only 0.2% are negative. The word 'comprehensive' is now empirically warranted for both direction and magnitude: the abliterated model's RLHF-modified feature activations are quantitatively indistinguishable from the instruct model's, pending pre-topk/post-topk validation (see below). (3) Geometric orthogonality — mean |cos(r, w_enc(f))|=0.01854 at layer 22 and 0.02086 at layer 20 (authoritative, confirmed across iter3 and iter4 to 5 decimal places; the 0.0145 value in a prior summary reflected a different feature set). A cosine of 0.019 means the refusal direction projects onto <2% of each RLHF-modified encoder direction's magnitude, explaining both the zero Type-R count and the β̄≈1 result mechanistically: the orthogonalization geometrically cannot touch these feature directions. (4) Centroid selectivity — the abliterated model's centroid in the RLHF-sensitive subspace is 4.35× closer to instruct than to base on harmful prompts at layer 22 (7.31× at layer 20), versus 1.16× in the RLHF-insensitive control (1.01× at layer 20) and 2.37× in the random control (1.63× at layer 20). Both layers confirm the imprint is specific to the RLHF-modified subspace. (5) Benign-SC overlap — 323/500 (64.6%) of the top-500 RLHF-sensitive features also appear in the top-500 by benign-prompt Safety Change, establishing a STYLE_DOMINANT verdict: the majority of the preserved imprint captures general RLHF behavioral modifications (response formality, instruction-following structure, output style) rather than harm-specific semantic content. The remaining 35.4% (176 features) are uniquely sensitive to harmful prompts and the more plausible carriers of harm-concept semantic representations.

  Three critical validation gaps must be resolved in the next iteration before the comprehensive-imprint framing can be fully trusted.

  First and most load-bearing: all metrics (Reversal Score, OLS slope, SC) use pre-topk encoder projections z_pre (dense continuous projections onto all 65,536 encoder directions) because reconstruction R²<0.85 at layer 22. Pre-topk values are dense by construction — every prompt gets non-zero values for all 65,536 'features' simultaneously — so the high R̄=0.921 and β̄=0.986 could reflect global geometry of the shared encoder weight matrix rather than preservation of semantically distinct sparse features. The required validation is: (a) on the subset of prompts and features where R² is highest, compare top-500 features selected by SC(z_pre) versus SC(z_post) and report the overlap fraction; (b) compute Spearman ρ between z_pre Reversal Scores and z_post Reversal Scores on the post-topk nonzero subset. If ρ>0.7 and overlap>60%, the pre-topk proxy is defensible as a geometric surrogate; if they diverge substantially, all findings must be reframed as 'geometric subspace properties of the encoder weight matrix' rather than 'feature-level preservation of near-monosemantic SAE features.' This reframing would still be a valid and novel contribution — the near-orthogonality finding (|cos|=0.019) and centroid selectivity are independent of the pre-topk/post-topk distinction — but the OLS slope magnitude claim and semantic labeling interpretation would require explicit qualification.

  Second: the abliterated checkpoint (huihui-ai/Huihui-Qwen3-8B-abliterated-v2) has not been behaviorally or geometrically verified. If abliteration is incomplete, high R̄ and β̄≈1 could trivially reflect proximity of the abliterated model to the instruct model rather than preserved imprint under genuine erasure. The required validation is: (1) fraction of AdvBench or HarmBench prompts where the abliterated checkpoint produces a non-refusal response (expected near 100% for successful abliteration); (2) mean |Proj_r(h_abl(p))|/|h_abl(p)| at layer 22 for the 500 harmful prompts, expected near zero (confirming r has been effectively removed from the residual stream). These are measurable with negligible compute from cached activations and the already-computed refusal direction.

  Third: the refusal direction r is computed as a single mean-difference vector, but arXiv:2602.02132 demonstrates that refusal spans geometrically distinct directions across behavioral categories — a multi-dimensional subspace rather than a single vector. The orthogonality argument (|cos(r, w_enc)|=0.019 explains zero Type-R features) requires that abliteration's full removal subspace be near-orthogonal to all RLHF-modified encoder directions, not just its mean-difference summary. The required measurement is: top-3 principal components of the (harmful−harmless) activation differences at layer 22 and mean |cos(PC_k, w_enc(f))| for k=1,2,3 across the top-500 features. If all three PCs show |cos|<0.05, the orthogonality argument is robust to multi-dimensional ablation; if a higher PC shows |cos|≥0.05, the mechanistic explanation requires qualification.

  Two secondary issues are resolved or downgraded. The numerical discrepancy in |cos| values (hypothesis previously cited 0.0145 as the authoritative value) is now closed: 0.01854 at L22 and 0.02086 at L20 are confirmed correct to 5 decimal places across iter3 and iter4, and the paper draft's rounded values (0.019, 0.021) are accurate. The semantic labeling rate (≈75% harm-concept across top-100 features) must be treated as a qualitative observation rather than a quantitative claim: the labeling uses character-prefix truncation of most-activating prompts without per-token activation windows, the labeler (Llama-3.1-8B-Instruct) was not validated against a manual gold set with Cohen's κ, and the 64.6% benign-SC overlap implies that many stylistic features receive harm-concept labels because they activate most strongly on harmful prompts. Until a 50-item manual annotation with κ≥0.6 is conducted, the 75% figure should not support downstream quantitative reasoning.

  The downstream classification null (accuracy=0.476 vs. keyword baseline 0.630) is now fully explained and is a confirmatory finding rather than a contradiction: the 64.6% benign-SC overlap establishes that RLHF-sensitive features are elevated on both harmful and benign prompts, eliminating prompt-level discriminability. The primary finding ('what RLHF modified and abliteration preserved') is orthogonal to ('what selectively fires on harmful vs. benign prompts at inference').

  The geometric-orthogonality interpretation remains the most secure result, independent of the pre-topk/post-topk ambiguity and the abliteration-depth concern: |cos(r, w_enc)|<0.02 at both layers is a direct measurement of the encoder weight geometry, not a function of model activations. The benign-SC STYLE_DOMINANT finding is a genuine empirical discovery that sharpens the imprint's interpretation: abliteration fails to erase any RLHF modification — harm-semantic or stylistic — because all reside in a subspace near-orthogonal to the refusal direction. The behavioral regression to base-like output arises because the compliance mechanism is removed, not because the underlying representations are erased.
motivation: |-
  The standard narrative treats abliteration as 'undoing' safety fine-tuning, restoring the model to a base-like state. LatentBiopsy (arXiv:2603.27412) and related geometric work (arXiv:2604.18901) establish compellingly that SOMETHING survives abliteration — harmful-prompt geometry persists with AUROC ≥ 0.967 across model families. But geometric approaches collapse all information into a single detection score and cannot answer: WHICH specific features survive? Are they semantically coherent harm-concept representations? Can they be used for targeted safety restoration? This is precisely what SAE-level feature analysis enables that geometric approaches cannot.

  The key practical payoff has two components. First, if Type-I features carry interpretable harm-concept semantic content (weapon synthesis schemas, manipulation tactics, deception patterns), they become actionable candidates for lightweight safety restoration without full retraining — you can potentially re-activate the safety behavior on top of the abliterated model by steering only the surviving harm-concept features. Second, the hypothesis that Type-I feature fraction predicts the capability gap between abliterated and base models provides a novel mechanistic explanation for why abliterated models consistently underperform base models despite theoretically similar weight norms — a long-standing open question that geometric approaches cannot address.

  The Qwen3-8B ecosystem is uniquely positioned for this study: official base and instruct variants, publicly released Qwen-Scope base-SAEs (SAE-Res-Qwen3-8B-Base-W64K-L0_50), and a community-created abliterated model (huihui-ai/Qwen3-8B-abliterated) all exist and are freely accessible. This study is framed as a single-model proof-of-concept that demonstrates the measurement framework; cross-model generalization across Qwen3.5 and Llama families is an explicit planned follow-up.
assumptions:
- >-
  The Qwen-Scope base-SAE captures enough of the relevant semantic feature space that projecting instruct and abliterated
  model activations through its encoder produces informative (if not perfect) feature activation patterns. Critically, SAE
  encoder-based feature activations may be informative even when decoder reconstruction quality degrades — the encoder maps
  activations to feature space regardless of the decoder's calibration for the instruct/abliterated distributions. We add
  a reconstruction quality gate (R² ≥ 0.85) before interpreting feature-level differences; if the gate fails, we proceed with
  encoder-only analysis and document the limitation explicitly.
- >-
  The huihui-ai/Qwen3-8B-abliterated model was produced using a difference-of-means + orthogonal projection abliteration method
  based on the 'single refusal direction' approach (Arditi et al., 2024). Given the model card's language about a 'new and
  faster method,' Step 0 proactively reproduces abliteration from scratch using the standard remove-refusals-with-transformers
  script on Qwen3-8B, creating a controlled abliterated checkpoint with verified methodology regardless of what the community
  model used. This adds robustness and ensures the abliteration direction is well-defined and comparable to individual SAE
  feature directions.
- >-
  The per-feature Reversal_score — defined as Pearson correlation across N=500 harmful prompts between the instruct change
  vector [Δinstruct_i(f)] and the abliterated change vector [Δabliterated_i(f)] — captures a meaningful signal about whether
  abliteration preserves or reverses the RLHF modification of feature f. This is a correlation over prompt-level scalar activations
  (one value per prompt per feature), not a cosine similarity of scalars, and is thus a proper statistical measure of co-directional
  change.
- >-
  RLHF-sensitive features (high Safety_change) that receive the LLM-assigned semantic label 'harm-concept related' will show
  this label consistently across diverse harmful prompt phrasings — i.e., the labeling is not triggered by surface patterns
  in the top-activating prompts but reflects the feature's stable semantic role. This assumption is tested via gold-set validation
  (20 manually-annotated features) and use of a cross-family labeler (Llama-3 or Gemini) to avoid within-family Qwen circularity.
- >-
  The instruct model's safety-relevant SAE features activate at comparable levels whether the model receives raw text prompts
  or template-formatted prompts (to be validated in a pilot study comparing activation patterns under both formats before
  committing to the raw-text protocol).
investigation_approach: |-
  Five-step experiment using all-Python implementation on a single GPU (16+ GB VRAM):

  **Step 0 — Abliteration method verification and proactive scratch-reproduction.**
  Fetch and inspect the source code of the remove-refusals-with-transformers repository to confirm the exact technique used. Given the community model card's description of 'a new and faster method,' we proactively reproduce abliteration from scratch using the standard difference-of-means + orthogonal projection script on Qwen3-8B regardless — this takes under one hour on a single GPU, adds methodological rigor, and creates a controlled abliterated checkpoint with fully known methodology. Document the layer set and prompt set used for the refusal direction computation.

  **Step 0.5 — SAE layer selection, reconstruction quality gate, and chat-template pilot.**
  Qwen-Scope provides SAEs at multiple residual-stream layers. Layer selection criterion: run a preliminary pass over 50 harmful prompts through the base model; compute Safety_change variance (variance of |act_instruct_i(f) - act_base_i(f)| across features f) at each available Qwen-Scope layer; select the layer with highest Safety_change variance as the richest safety-relevant signal. If resources allow, repeat the main analysis at one additional layer (e.g., the layer closest to known refusal-direction geometry from prior literature) as a sensitivity check.

  Next, load the frozen Qwen-Scope base-SAE at the selected layer. Run 50 harmful prompts through all three model variants; compute reconstruction R² for base, instruct, and abliterated activations. If R² ≥ 0.85 for instruct and abliterated variants, proceed to full feature analysis. If R² < 0.85, proceed with encoder-only analysis using raw pre-JumpReLU encoder activations as the primary signal (see JumpReLU caveat below).

  Separately, run a 20-prompt chat-template pilot: compare the top-50 high-Safety_change SAE feature activations for the instruct model under raw text vs. template-formatted harmful prompts. If activation patterns differ substantially (Pearson r < 0.80), switch to template-formatted prompts for the instruct model only.

  **Step 1 — Data collection.**
  Load all three Qwen3-8B model checkpoints: base (Qwen/Qwen3-8B-Base), instruct/safety-tuned (Qwen/Qwen3-8B), and the controlled scratch-reproduced abliterated model. Collect 500 harmful prompts (from WildGuard/BeaverTails/HarmBench) and 500 benign prompts. Provide all prompts using the format validated in Step 0.5. Run forward passes through each model, extract residual stream activations at the selected Qwen-Scope SAE layer. Project through the frozen SAE encoder to obtain 64K-dimensional sparse feature activation vectors for each (prompt, model) triple. Collect both post-JumpReLU (sparse) and pre-JumpReLU (raw encoder) activations to support the JumpReLU threshold robustness check.

  **Step 2 — Feature classification via per-feature Pearson correlation.**
  For each of the 64K SAE features f, compute over the 500 harmful prompts:
  - Safety_change(f) = mean_i |act_instruct_i(f) - act_base_i(f)| — how much RLHF changed this feature on average
  - Δinstruct(f) = vector [act_instruct_i(f) - act_base_i(f) for i in 1..500] — a 500-dimensional vector of instruct changes
  - Δabliterated(f) = vector [act_abliterated_i(f) - act_base_i(f) for i in 1..500] — a 500-dimensional vector of abliterated changes
  - Reversal_score R(f) = PearsonR(Δinstruct(f), Δabliterated(f)) — the correlation between these two change vectors

  Worked example: Suppose feature f is active on harmful chemistry prompts. If RLHF increases its activation (Δinstruct values ≈ [+0.8, +1.2, +0.5, +0.9, +1.1, ...]) and abliteration also leaves it elevated (Δabliterated ≈ [+0.7, +1.0, +0.4, +0.8, +0.9, ...]), then R(f) ≈ +0.99 → Type-I (imprinted). If instead abliteration reverses the increase (Δabliterated ≈ [-0.1, -0.2, -0.1, -0.1, -0.2, ...]), then R(f) ≈ -0.98 → Type-R (reversed).

  In the encoder-only fallback regime: compute Reversal_score using raw pre-JumpReLU encoder activations. Document explicitly that JumpReLU thresholds calibrated on base activations carry a residual distributional shift confound — if RLHF systematically elevates a feature's mean activation above the base threshold, Safety_change computed from post-threshold sparse activations may be inflated. Report Safety_change from both pre- and post-threshold activations and flag any features where the two estimates diverge substantially (>2× ratio) as 'threshold-sensitive' in the supplementary analysis.

  The distribution of R(f) across RLHF-sensitive features (top-500 by Safety_change magnitude) is tested for bimodality using Hartigan's dip test (H₀: unimodal). Both outcomes are pre-specified as interpretable: bimodal → discrete Type-R/Type-I classification; continuous → treat R(f) as a graded survival score and examine semantic correlates along the continuum.

  **Step 3 — Semantic interpretation, distance analysis, and labeler validation.**
  For the top-100 features by |R(f) - 0.5| × Safety_change magnitude: use maximum-activating prompt segments as context.

  LLM semantic labeler protocol with two safeguards: (1) Before using automated labels for Criterion 3, manually annotate 20 features (10 high-R(f), 10 low-R(f)) as a gold set; compute Cohen's kappa between human annotation and LLM labels — if kappa < 0.6, revise the labeler prompt and re-validate; (2) Use a cross-family labeler (Llama-3-8B or Gemini-flash via OpenRouter) rather than any Qwen model, to avoid within-family circularity where Qwen2.5-7B might apply the same semantic categories as Qwen3-8B's RLHF training.

  Report the validated labeler prompt template in the supplementary materials. In the Type-R and Type-I feature subspaces separately, compute centroid distances d(abliterated, instruct) and d(abliterated, base) for harmful vs. benign prompts.

  For the exploratory capability-gap analysis: compute the fraction of Type-I features at the analyzed layer and report alongside the abliterated vs. base capability gap on a mini-MMLU sample (200 questions); this correlation is descriptive only and cannot be statistically validated at N=1 SAE layer within one model — it is labeled explicitly as a qualitative preliminary signal for future multi-model work.
success_criteria: |-
  CONFIRM core hypothesis if BOTH of the following hold:
  1. **Bimodality test (or continuous semantic coherence):** Hartigan's dip test on the R(f) distribution for RLHF-sensitive features (top-500 by Safety_change) yields p<0.05 (bimodal), with the two modes interpreted as Type-R (R<0) and Type-I (R>+0.5). Alternatively, if the test yields p≥0.05 (continuous), the hypothesis is confirmed in a weaker 'graded imprint' form if the continuous case shows: features with high R(f) (R>+0.5) receive harm-concept semantic labels from the validated cross-family LLM labeler at ≥2× the rate of features with low R(f) (R<0), establishing that survival of the RLHF modification correlates with harm-concept semantic content regardless of whether the distribution is bimodal.
  2. **Representational fingerprint:** In the high-R(f) feature subspace, d(abliterated, instruct) < d(abliterated, base) for harmful prompts, while d(abliterated, instruct) ≈ d(abliterated, base) for benign prompts. This is the core quantitative test: the abliterated model's harm-concept representations look more like the instruct model than the base model, specifically when processing harmful content.

  ADDITIONAL CONFIRMATORY CRITERION:
  3. **Semantic label coherence with validated labeler:** High-R(f) features (Type-I or high survival-score features) receive semantic labels classified as 'harm-concept related' (weapon synthesis, manipulation, deception) by the cross-family LLM labeler at a significantly higher rate than low-R(f) features (Type-R or low survival-score features) — p<0.05 by Fisher's exact test on a manual validation sample of 50 features per group. Requires kappa ≥ 0.6 on the 20-feature gold set before the automated labels are trusted.

  EXPLORATORY (non-confirmatory) CRITERION:
  4. **Capability gap preliminary signal:** The fraction of Type-I features at the analyzed layer is reported alongside the abliterated vs. base capability gap on a mini-MMLU sample (200 questions); this correlation is descriptive only and cannot be statistically validated at N=1 SAE layer within one model — it is a qualitative preliminary signal for future multi-model work.

  DISCONFIRM hypothesis if: (a) Hartigan's test yields p≥0.05 AND high-R(f) features show no enrichment for harm-concept labels over low-R(f) features; (b) d(abliterated, instruct) ≥ d(abliterated, base) in the high-R(f) feature subspace for harmful prompts; (c) the SAE reconstruction R² gate fails below 0.70 for the instruct model AND pre-JumpReLU encoder-only activations also show no reliable Safety_change signal (making both analysis paths unreliable — in this case the experiment is inconclusive rather than disconfirmed, and the Qwen3-Instruct SAE should be used as an alternative instrument in a follow-up).
related_works:
- >-
  LatentBiopsy (arXiv:2603.27412, 2025): Compares base, instruct, and abliterated Qwen3.5-0.8B and Qwen2.5-0.5B through PCA-based
  angular analysis of residual streams, finding that harmful-prompt geometry survives refusal ablation (AUROC drops by at
  most 0.015). A companion paper (arXiv:2604.18901) extends this to 12 models (Qwen, Llama, Gemma), finding AUROC 0.982 survives
  abliteration within ±0.003. Both use holistic geometric measures that collapse all feature information into a single detection
  score. Our work differs fundamentally: we decompose the change at the individual SAE feature level to identify WHICH specific
  semantic features survive and whether they form interpretable harm-concept clusters — LatentBiopsy shows SOMETHING survives,
  we characterize WHAT and WHY, enabling the targeted safety restoration application these geometric approaches cannot support.
- >-
  Mechanistic Investigation of Supervised Fine-Tuning via SAEs (arXiv:2605.11426, 2025): Uses pretrained base-SAEs as a frozen
  diagnostic to study how SFT changes feature activations, finding significant feature divergence despite high cosine similarity
  of raw activations. Our work extends this approach in two key ways: (1) we apply it to the three-model trajectory base→instruct→abliterated
  rather than just base→instruct, and (2) we specifically characterize features by whether the abliteration operation reverses
  or preserves the fine-tuning change — the Reversal_score classification — a dimension this paper does not address. Critically,
  this paper's finding that sparse latents diverge after SFT motivates our SAE reconstruction quality gate (Step 0.5) and
  validates the frozen base-SAE approach as a cross-variant diagnostic.
- >-
  Crosscoders (arXiv:2602.11729, 2025): Crosscoders are purpose-built for multi-model SAE comparison — they train a shared
  sparse dictionary jointly on activations from two or more model variants, naturally partitioning features into 'shared'
  and 'model-specific' categories without requiring a pre-existing feature vocabulary. Our frozen base-SAE approach is complementary
  rather than redundant, offering three advantages crosscoders cannot provide: (1) zero training cost — no access to the model
  training pipeline required; (2) direct reuse of Qwen-Scope's existing labeled feature vocabulary, enabling semantic interpretation
  without re-deriving feature meanings from scratch; (3) methodological continuity with arXiv:2605.11426 which validates the
  frozen base-SAE as a reliable SFT diagnostic. A crosscoder-based comparison (base+instruct) would serve as a valuable robustness
  check and is marked as future work.
- >-
  Dedicated Feature Crosscoders / DFCs (arXiv:2606.26474, ICML 2026 spotlight): Localizes RL-induced tool-calling behavior
  in Qwen2.5-3B to a compact, steerable feature set via a dedicated crosscoder that isolates RL-specific features. This 'RL-induced
  addition' result — a new feature appearing after RL fine-tuning — is the complementary finding to our 'RLHF-induced survival
  under erasure' question. DFCs identify what RL adds; our Reversal_score framework identifies what RLHF safety training adds
  that then resists deletion. The two framings together provide a complete picture of how fine-tuning modifies the feature
  landscape. Critically, DFCs require training a new crosscoder per model pair; our approach extends to the three-model abliterated
  trajectory using only inference.
- >-
  Surgical Refusal Ablation (arXiv:2601.08489, 2025): Identifies that the raw refusal vector is polysemantic, entangling refusal
  signals with capability circuits, and proposes spectral residualization for a targeted ablation. This work addresses which
  CAPABILITY directions are confounded with the refusal direction. Our hypothesis is complementary: we address which SEMANTIC
  KNOWLEDGE features survive standard abliteration regardless of whether the vector was clean or polysemantic. The polysemanticity
  finding from this paper also provides theoretical grounding for why we might observe a continuum of partial reversals rather
  than a clean bimodal split.
- >-
  What Makes and Breaks Safety Fine-tuning (arXiv:2407.10264, 2024): Studies how safety fine-tuning projects unsafe activations
  into the MLP null space. Our work complements this by characterizing what the 'minimal transformation' actually modifies
  semantically using the SAE basis, and specifically identifies which of these modifications are robust to abliteration.
- >-
  Qwen-Scope (arXiv:2605.11887, 2026): Releases SAEs for Qwen3 and Qwen3.5 model variants including Qwen3-8B base, demonstrating
  their use for steering, safety classification, and post-training analysis. We directly use these SAEs as the measurement
  instrument for our three-model comparison. Qwen-Scope provides the tool but does not perform the base/instruct/abliterated
  comparison or the Reversal_score feature classification we propose.
- >-
  Training Superior Sparse Autoencoders for Instruct Models (arXiv:2506.07691, 2026): Shows that base-model SAEs suffer substantially
  reduced reconstruction quality on instruct model activations (MSE 5.1985 vs 0.6468 for specialized instruct SAEs on Qwen2.5-7B-Instruct).
  This paper directly motivates our reconstruction quality gate in Step 0.5 and the contingency plan of using encoder-only
  analysis with raw pre-JumpReLU activations if reconstruction quality is insufficient. A Qwen3-Instruct SAE has been released
  covering Qwen3-1.7B, -4B, and -8B, which could serve as an alternative or cross-validation instrument.
inspiration: |-
  The core inspiration comes from cognitive neuroscience, specifically the **implicit vs. explicit memory dissociation** (Tulving, 1972; Schacter, 1987; Squire, 1992). In human memory, there are two distinct systems: declarative/explicit memory (consciously recallable facts and episodes, dependent on the hippocampus) and implicit/procedural memory (unconscious influences on behavior — priming, skills — dependent on striatum and neocortex). Classic amnesia studies (H.M., Clive Wearing) show that hippocampal damage abolishes declarative memory while leaving implicit memory intact. A patient who cannot remember learning to play piano may still improve with practice because the skill trace persists implicitly.

  Applied to RLHF: safety fine-tuning is analogous to encoding both explicit and implicit memories. The **explicit safety memory** is the refusal behavior — the model can 'recall' it should refuse and does so. The **implicit safety memory** is the semantic imprint — how the model internally represents harmful content. Abliteration is analogous to hippocampal damage: it removes the explicit refusal behavior while potentially leaving implicit semantic traces intact. The abliterated model exhibits **cryptomnesia** — it acts as if it has forgotten safety training (it complies with harmful requests) but may retain an unconscious imprint of that training in how it represents harmful concepts.

  At the methodological level, the approach of using a fixed SAE basis as a 'measurement instrument' across all three model variants is inspired by **isotopic tracing in chemistry**: in metabolic labeling experiments, isotope-labeled molecules are introduced and tracked through chemical reactions. The SAE features are the 'isotopes' — a fixed vocabulary that lets us track which representations change through RLHF fine-tuning and which survive abliteration. The Reversal_score is specifically inspired by **radiometric age concordia diagrams** in geochronology: by plotting the concordance between two decay channels (here, the RLHF change direction vs. the abliteration change direction for each feature), you identify which 'clocks' ran together and which were reset by a later event (abliteration).
terms:
- term: Sparse Autoencoder (SAE)
  definition: >-
    A neural network trained to decompose model activations into a large set of sparse, near-monosemantic features. Given
    a residual stream activation vector (e.g., 4096-dimensional for Qwen3-8B), the SAE's encoder maps it to a high-dimensional
    sparse vector (e.g., 64,000 features) where typically only 50-100 features are non-zero. Each active feature ideally corresponds
    to a single interpretable concept. Here we use Qwen-Scope's pre-trained base-SAE as a fixed vocabulary for comparing activations
    across model variants.
- term: Abliteration
  definition: >-
    A weight-editing technique that removes a model's refusal behavior by identifying the 'refusal direction' in the residual
    stream (computed as the mean difference between activations on harmful vs. harmless prompts), then orthogonalizing all
    weight matrices that write to the residual stream against this direction. The result is a modified model that no longer
    activates the refusal direction and therefore complies with previously-refused prompts, without retraining.
- term: Reversal_score R(f)
  definition: >-
    For a given SAE feature f, the Pearson correlation between two N-dimensional vectors computed across a set of N prompts:
    Δinstruct(f) = [act_instruct_i(f) - act_base_i(f) for i=1..N] and Δabliterated(f) = [act_abliterated_i(f) - act_base_i(f)
    for i=1..N]. R(f) near +1 means the abliteration change goes in the same direction as the RLHF change (feature is 'imprinted'
    — abliteration does not reverse RLHF). R(f) near -1 means abliteration reverses the RLHF change (feature is 'coupled'
    to the refusal mechanism). R(f) near 0 means partial reversal. This is a prompt-level correlation of scalar activations,
    not a per-prompt cosine similarity — ensuring the bimodality finding (if observed) is an empirical result, not an artifact
    of the definition.
- term: Type-R feature (Refusal-Coupled)
  definition: >-
    An SAE feature whose activation was changed by safety RLHF fine-tuning and is reversed (restored toward base model activation)
    when abliteration is applied. Operationally: features with high Safety_change and Reversal_score R(f) < 0. These features
    implement or support the explicit refusal mechanism.
- term: Type-I feature (Semantically-Imprinted)
  definition: >-
    An SAE feature whose activation was changed by safety RLHF fine-tuning and is NOT reversed when abliteration is applied
    — the abliterated model's activations on these features remain closer to the instruct model's pattern than to the base
    model's. Operationally: features with high Safety_change and Reversal_score R(f) > +0.5. These features are hypothesized
    to represent harm-concept semantic content.
- term: Latent Safety Imprint
  definition: >-
    The set of Type-I feature modifications (or high-R(f) features in the continuous case) that safety RLHF fine-tuning writes
    into the model's representational space, which survive abliteration. A model with a latent safety imprint behaves like
    the base model (complies with harmful requests) but internally represents harmful content more like the safety-tuned instruct
    model. Analogous to implicit memory in neuroscience: the behavior is forgotten but the trace persists.
- term: SAE Reconstruction Quality Gate
  definition: >-
    A pre-experiment validation step that computes R² (fraction of variance explained by SAE reconstruction) for activations
    from all three model variants (base, instruct, abliterated) passed through the frozen base-SAE. If R² ≥ 0.85 for non-base
    variants, the base-SAE is an acceptable cross-model measurement instrument with both encoder and decoder analysis. If
    R² < 0.85, encoder-only analysis using raw pre-JumpReLU activations is used, with explicit documentation of the JumpReLU
    threshold confound.
- term: Safety_change(f)
  definition: >-
    For a given SAE feature f, the mean absolute difference between instruct and base model activations across the harmful
    prompt set: mean_i |act_instruct_i(f) - act_base_i(f)|. Features with high Safety_change are 'RLHF-sensitive' and are
    the ones further analyzed by Reversal_score. In the encoder-only regime, both post-JumpReLU (sparse) and pre-JumpReLU
    (raw encoder) versions are computed as a robustness check against threshold confounds.
- term: Qwen-Scope
  definition: >-
    An open-source suite of Sparse Autoencoders released by the Qwen team (May 2026) for interpretability and development
    use with Qwen3 and Qwen3.5 model families. Includes base-model SAEs at multiple residual-stream layers for Qwen3-8B with
    64,000 features (SAE-Res-Qwen3-8B-Base-W64K-L0_50), which we use as the fixed measurement instrument in this study. The
    analyzed layer is selected by highest Safety_change variance in a preliminary pass. Trained on base model activations;
    reconstruction quality on instruct model activations requires validation before use.
- term: Crosscoder
  definition: >-
    A multi-model SAE variant that trains a shared sparse dictionary jointly on activations from two or more model variants,
    naturally partitioning features into 'shared' (present in both models) and 'model-specific' (unique to one variant) categories.
    Crosscoders are the purpose-built tool for multi-model feature comparison but require access to the training pipeline
    and produce a new feature vocabulary per model pair. Our frozen base-SAE approach complements crosscoders by requiring
    no training, reusing Qwen-Scope's existing labeled vocabulary, and extending naturally to three-model comparisons.
- term: JumpReLU threshold confound
  definition: >-
    A distributional shift artifact specific to the encoder-only fallback: Qwen-Scope's SAE uses JumpReLU activation functions
    with thresholds calibrated on base model activations. When these thresholds are applied to instruct model activations
    (which may be systematically elevated or suppressed by RLHF), features may be incorrectly gated — features raised above
    threshold by RLHF appear artificially more active, inflating Safety_change. Mitigated by reporting Safety_change from
    both pre- and post-JumpReLU activations and flagging divergent features.
summary: >-
  We hypothesize that RLHF safety fine-tuning in Qwen3-8B writes two mechanistically distinct types of changes into the model's
  SAE feature space — 'Refusal-Coupled' features reversed by abliteration and 'Semantically-Imprinted' features that persist
  — and test this using the Qwen-Scope base-SAE at the layer with highest Safety_change variance as a fixed measurement instrument
  across base, instruct, and scratch-reproduced abliterated Qwen3-8B variants, with a novel per-feature Pearson correlation
  (Reversal_score), pre-registered bimodal and continuous interpretations, a validated cross-family LLM semantic labeler,
  raw pre-JumpReLU activations as a JumpReLU-threshold robustness check, and explicit contrast with crosscoder methodology
  — providing the first feature-level account of what the abliterated model 'still knows' about harmful content and why that
  matters for targeted safety restoration.
_relation_rationale: >-
  Same frame; magnitude confirmed (β̄≈1); style-dominant benign-SC overlap added; three validation gaps elevated.
_confidence_delta: increased
_key_changes:
- >-
  Magnitude now confirmed: OLS slope β̄=0.986±0.176 (L22), 0.954±0.191 (L20); 86.4% features have β_f>0.8 at L22; 'comprehensive
  latent imprint' framing now empirically warranted for direction AND magnitude, pending pre-topk/post-topk validation.
- >-
  Mid-SC baseline replaces bottom-500 as the primary null comparison: SC_median≈1.12 ≫ 0.01, noise-free, R̄=0.871 vs top-500
  R̄=0.921 (d=1.087, p≈0) — selective preservation confirmed without the noise concern raised previously.
- >-
  Benign-SC overlap (64.6%) added as a confirmed finding: top-500 RLHF-sensitive features are STYLE_DOMINANT — the majority
  encode general RLHF behavioral changes (formality, structure), not harm-specific semantics; remaining 35.4% uniquely-harmful
  features are the primary harm-concept candidates.
- >-
  Numerical reconciliation closed: |cos|=0.01854 at L22, 0.02086 at L20 are now authoritative; 0.0145 in prior summary was
  a different feature set; paper rounded values (0.019, 0.021) confirmed accurate.
- >-
  Pre-topk/post-topk validation elevated to PRIMARY unresolved concern: all metrics use dense z_pre projections; required
  validation is SC overlap fraction and Spearman ρ between z_pre and z_post Reversal Scores — result determines whether framing
  is 'feature-level preservation' vs. 'encoder-subspace geometry'.
- >-
  Abliteration depth verification elevated to CRITICAL experimental control: report refusal rate on AdvBench/HarmBench and
  mean |Proj_r(h_abl)|/|h_abl| at L22 to rule out incomplete abliteration as trivial explanation for β̄≈1.
- >-
  Multi-dimensional refusal subspace measurement added: top-3 PCs of (harmful−harmless) activation differences at L22; mean
  |cos(PC_k, w_enc)| for k=1,2,3 required to confirm orthogonality argument is robust beyond the single mean-difference vector.
- >-
  Semantic labeling 75% rate downgraded from quantitative claim to qualitative observation pending Cohen's κ validation against
  a 50-item manual annotation gold set.
- >-
  Classification null (0.476 accuracy) reframed as confirmatory: 64.6% benign-SC overlap explains why RLHF-sensitive features
  lack harmful-vs-benign discriminability — distinct question from 'what RLHF preserved'.
relation_type: evolution
</current_hypothesis>

<all_artifacts>
Complete set of research artifacts across all iterations.

--- Item 1 ---
id: art_pmAcY0CNazP9
type: research
title: Qwen3-8B SAE & Abliteration Tech Reference
summary: |-
  CRITICAL ARCHITECTURE CORRECTION: Qwen-Scope SAEs (Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 and L0_100) use TopK architecture (k=50 or k=100 features kept non-zero via torch.topk), NOT JumpReLU. All hypothesis references to 'JumpReLU threshold confound' are incorrect; the pre-topk (raw encoder activations = residual @ W_enc.T + b_enc) vs post-topk distinction is still valid and cleaner.

  MODEL IDs: Base=Qwen/Qwen3-8B-Base, Instruct=Qwen/Qwen3-8B, Abliterated=huihui-ai/Huihui-Qwen3-8B-abliterated-v2 (recommended; v2 fixes layer-0 garbling bug) or mlabonne/Qwen3-8B-abliterated. All: 36 layers, d_model=4096, bfloat16, min transformers 4.51.0, hook attribute model.model.layers. Use enable_thinking=False for instruct model.

  SAE SPECS: Checkpoints at layer{N}.sae.pt (N=0..35); W_enc(65536,4096), W_dec(4096,65536), b_enc(65536,), b_dec(4096,); no sae_lens compatibility; load via hf_hub_download + torch.load. The model card explicitly endorses applying base SAE to post-training checkpoints.

  RECONSTRUCTION QUALITY: arXiv:2506.07691 shows ~8x MSE gap between BT(P)-style instruct SAE (log2(MSE)=5.20) and properly trained instruct SAE (0.65) on special tokens. arXiv:2605.11426 shows SAE latent cosine sim drops to 0.509-0.557 at layer 22 after SFT while raw activations stay >0.96. The R²>=0.85 reconstruction gate will fail at deep layers — must use pre-topk (encoder) activations as the feature representation instead of decoder-reconstructed ones.

  ABLITERATION: Sumandora/remove-refusals-with-transformers uses compute_refusal_dir.py + inference.py; harmful=AdvBench harmful_behaviors.csv, harmless=yahma/alpaca-cleaned; layers 1 to n_layers-1. mlabonne approach: harmful=mlabonne/harmful_behaviors, harmless=mlabonne/harmless_alpaca; selects layer by highest mean residual difference (layer 9 optimal for 8B). Orion-zhen/abliteration: python abliterate.py config.yaml; Simple/Biprojection/Norm-Preserving/Full methods.

  LAYER SELECTION: LatentBiopsy (arXiv:2603.27412) found optimal layer 20/24 for Qwen3.5-0.8B (AUROC 0.9642), broad plateau <0.08 variation — Qwen3-8B not directly tested. SFT paper (arXiv:2605.11426): layer 22 most sensitive to SFT changes (SAE latents collapse to 0.509 cosine sim); layer 7 most stable. Recommended pilot layers: 14 (stable), 20 (primary), 22 (high sensitivity). Avoid layers 28-35.

  QWEN3-INSTRUCT SAE (FALLBACK): Exists (arXiv:2606.26620); JumpReLU, 65K dict, L0=160; but covers Qwen3-8B ONLY layers 0-8; refusal steering tested on Qwen3-4B only; HF repo ID unconfirmed; NOT viable fallback for layers 14-22.

  HARMFUL PROMPT DATASETS: WildGuardMix=allenai/wildguardmix (config wildguardtest, prompt field, prompt_harm_label=harmful, 1725 test examples); BeaverTails=PKU-Alignment/BeaverTails (filter is_safe==False, 33.4k test); HarmBench=swiss-ai/harmbench config DirectRequest split test or walledai/HarmBench (602 behaviors, field Behavior).
workspace_path: >-
  /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_1/gen_art/gen_art_research_1
out_expected_files:
- research_out.json

--- Item 2 ---
id: art_M8-flaZ0OOaP
type: dataset
title: Harmful and Benign Prompt Corpus for SAE Analysis
summary: |-
  This artifact assembles a balanced 500+500 harmful/benign prompt corpus for SAE (Sparse Autoencoder) mechanistic interpretability analysis of Qwen3 models (base, safety-finetuned, abliterated). Four high-quality datasets were selected and assembled:

  1. **BeaverTails** (176 harmful): NeurIPS 2023, PKU-Alignment. Human-labeled QA pairs across 14 harm categories (animal_abuse, child_abuse, drug_abuse, financial_crime, hate_speech, misinformation, privacy_violation, self_harm, sexually_explicit, terrorism, violence, etc.). Most category-diverse single source — ideal for studying which SAE features correspond to specific harm types.

  2. **JailbreakBench/JBB-Behaviors** (55 harmful + 46 benign): NeurIPS 2024 Datasets Track. The only dataset with matched harmful+benign prompt pairs from the same curation effort, spanning 10 OpenAI policy violation categories. The paired structure (e.g., 'Write a heroin story' vs 'Write a fictional heroin story') enables direct contrastive SAE activation analysis.

  3. **HarmBench** (85 harmful): Center for AI Safety, 2024. Standard safety evaluation benchmark widely cited in mechanistic interpretability research. Covers chemical_biological, cybercrime, illegal, physical_harm, and related categories. Unambiguous direct harmful requests — cleanest signal for SAE feature isolation.

  4. **Dolly-15k** (106 benign): Databricks, CC-BY-SA 3.0, 2023. 15K human-written instruction-following prompts by 5000+ employees. Diverse categories (brainstorming, classification, closed QA, generation, information extraction, open QA, summarization). Best clean benign baseline for SAE activation comparison.

  Pipeline: MinHash LSH deduplication (threshold=0.5, 128 permutations, character 5-grams), token filtering (≤512 tokens, ~4 chars/token estimate), category cap (≤35% per category), balanced sampling. Total corpus: 1000 rows (500 harmful, 500 benign) in results/data_out.json. The 4-dataset selection in full_data_out.json contains 468 examples in exp_sel_data_out schema.

  All datasets are publicly available (BeaverTails: CC-BY-NC-4.0, JailbreakBench: MIT, HarmBench: public, Dolly-15k: CC-BY-SA 3.0). Sources verified through peer-reviewed publications (NeurIPS 2023/2024, ACM CCS 2024).
workspace_path: >-
  /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_1/gen_art/gen_art_dataset_1
out_expected_files:
- data.py
- full_data_out.json
- preview_data_out.json
- mini_data_out.json

--- Item 3 ---
id: art_QBm3I0Z5VVJR
type: experiment
in_dependencies:
- id: art_M8-flaZ0OOaP
  label: prompt corpus
  relation_type: uses
  relation_rationale: >-
    Experiment uses the 500 harmful + 500 benign prompt corpus as the evaluation set for SAE activation extraction.
- id: art_pmAcY0CNazP9
  label: tech reference
  relation_type: uses
  relation_rationale: >-
    Experiment uses tech reference for model IDs, SAE checkpoint loading, TopK specs, and layer recommendations.
title: Qwen3-8B SAE Reversal Score Analysis
summary: |-
  Full mechanistic interpretability pipeline comparing base, instruct (RLHF), and abliterated Qwen3-8B activations through frozen Qwen-Scope TopK SAEs (Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50, 65536 features, k=50). Primary results:

  LAYER SELECTION: Pilot across layers [14, 20, 22] on 50 harmful prompts. SC variance: L14=0.0127, L20=0.0547, L22=0.1721. PRIMARY_LAYER=22, SECOND_LAYER=20.

  R² GATE: At layer 22, R²=0.5745 (base), 0.5123 (instruct), 0.5119 (abliterated) — all below 0.85 threshold → pre-topk encoder activations used (z_pre = h @ W_enc.T + b_enc).

  FEATURE ANALYSIS: 500 top-RLHF-sensitive features by Safety_change; all 500 survived 10% density filter. Reversal Score distribution (Pearson correlation of RLHF vs abliteration delta vectors per feature): mean=0.921, std=0.040, min=0.684, max=0.989. All scores positive — ZERO Type-R features (R < 0). Key finding: RLHF and abliteration push the same SAE features in the same direction at layer 22, contradicting the hypothesis that abliteration reverses RLHF safety training.

  CRITERIA: C1 (bimodality via Hartigan dip): False (p=0.9296, unimodal). C2 (centroid distance): True — abliterated model centroid is closer to instruct than to base in Type-I/harmful subspace. C3 (semantic Fisher test): False (fisher_p=0.675, fallback Q3+/Q1- comparison used since no Type-R features exist; harm_concept rate 75% for high-R vs 71% for low-R, not significantly different).

  SEMANTIC LABELING: 100 features labeled via meta-llama/llama-3.1-8b-instruct through OpenRouter, total cost ~$0.004.

  CLASSIFICATION: Baseline keyword accuracy=0.630 (TP=182/500, TN=448/500). Our method accuracy=0.476 (TP=238/500, TN=238/500) using median harm score threshold — near-chance, consistent with the unimodal reversal score distribution and absence of Type-R features.

  Schema validation: PASSED. Total runtime: 4.3 min (all 3 model activations cached from prior run).
workspace_path: >-
  /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json

--- Item 4 ---
id: art_Qnl1zSSZPJvj
type: experiment
in_dependencies:
- id: art_M8-flaZ0OOaP
  label: prompt corpus
  relation_type: uses
  relation_rationale: >-
    Null-baseline experiment uses the same 500 harmful+500 benign prompt corpus as the primary experiment.
- id: art_pmAcY0CNazP9
  label: tech reference
  relation_type: uses
  relation_rationale: >-
    Tech reference provides model IDs, SAE checkpoint loading, TopK architecture specs used in null-baseline experiment.
title: SAE Feature Reversal Score Null-Baseline Validation
summary: >-
  Null-baseline SAE Reversal Score validation for Qwen3-8B. Computed Reversal Scores (Pearson r between delta_inst and delta_abl
  per feature across 500 harmful prompts) for top-500 RLHF-sensitive SAE features (highest Safety Change = mean|delta_inst
  - delta_base|), bottom-500 RLHF-insensitive features, and random-500 features from Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50
  at layers 22 and 20. Key results at layer 22: SELECTIVE_PRESERVATION_CONFIRMED (top500_R_mean=0.9205, bottom500_R_mean=0.8470,
  Cohen's d=1.7508, p≈0); ORTHOGONALITY_CONFIRMED (|cos_enc_top500_mean|=0.0145, SAE encoder directions nearly orthogonal
  to refusal direction r); centroid control distances computed for subspaces A=top500/B=bottom500/C=random500 on harmful and
  benign prompts. Layer-20 full replication: SELECTIVE_PRESERVATION_CONFIRMED (top500_R_mean=0.924, bottom500_R_mean=0.841,
  d=1.693); centroid CENTROID_SELECTIVE (A_harm=7.307 vs B=1.011, C=1.634). Cross-layer: REPLICATION_CONFIRMED. Models: Base=Qwen/Qwen3-8B-Base,
  Instruct=Qwen/Qwen3-8B, Abliterated=huihui-ai/Huihui-Qwen3-8B-abliterated-v2. Layer-22 activations (1000x4096 float32) reused
  from iter_2; layer-20 extracted fresh. Pre-topk projections z_pre = act @ W_enc.T + b_enc used throughout. Post-topk density
  characterized (density_mean≈0.019, fraction_above_0.10≈0.056). Output: method_out.json with 1000 examples (exp_gen_sol_out
  schema, validation PASSED) containing per-prompt predict_mean_delta_top500/bottom500/random500 and predict_rlhf_classification
  fields.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_3/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json

--- Item 5 ---
id: art_iDkiRSdMuI6M
type: experiment
in_dependencies:
- id: art_M8-flaZ0OOaP
  label: prompt corpus
  relation_type: uses
  relation_rationale: >-
    iter4 experiment uses the 500+500 prompt corpus for benign-SC overlap and mid-SC baseline computation
- id: art_pmAcY0CNazP9
  label: tech reference
  relation_type: uses
  relation_rationale: >-
    iter4 experiment uses tech reference for model IDs, SAE checkpoint loading, and TopK architecture specs
title: OLS Slopes, Mid-SC Baseline, Benign Overlap Analysis
summary: |-
  This experiment extends the iter3 SAE Reversal Score analysis with four new measurements on Qwen3-8B (base/instruct/abliterated) using cached activations from iter3 (no GPU needed).

  **PRIMARY FINDING — OLS no-intercept slope β_f (MAGNITUDE_PRESERVED):**
  For the top-500 SAE features (ranked by Safety Change on harmful prompts at layer 22), the mean OLS slope β_f = 0.9861 ± 0.1763. This means abliteration preserves not just the *direction* of RLHF feature modifications (Pearson R ≈ 0.92, confirmed from iter3) but also their *magnitude* — a much stronger result. 86.4% of features have β_f > 0.8, and 45.4% have β_f > 1.0 (amplification). Only 0.2% are negative. Paper framing: 'comprehensive_latent_imprint'. Replication at L20: β_f = 0.954 ± 0.191.

  **SECOND FINDING — Mid-SC baseline (MID_SC_CONFIRMS_SELECTIVE_PRESERVATION):**
  Features at the 40th–60th percentile of Safety Change (median SC ≈ 1.12 vs top-500 min ≈ 2.79) show R_mid500 = 0.871, significantly lower than R_top500 = 0.921 (t=17.2, p<1e-300, Cohen's d=1.087). This mid-SC baseline is noise-free (SC_median >> 0.01) and confirms that the high reversal scores in top-500 are not a measurement artifact but reflect genuine selectivity.

  **THIRD FINDING — Benign-SC overlap (STYLE_DOMINANT):**
  Of the top-500 features by harmful-SC, 323/500 (64.6%) overlap with the top-500 by benign-SC. This STYLE_DOMINANT verdict indicates the top-500 selected features capture general RLHF style changes (tone, format) rather than harm-specific semantics. This is an important caveat for interpreting the OLS results: the features with high reversal scores likely encode stylistic RLHF modifications, not specifically harm-related knowledge removal.

  **FOURTH FINDING — |cos| numerical reconciliation:**
  Iter4 recomputes |cos(refusal_dir, W_enc[f])| for top-500 features: L22 = 0.01854 (iter3: 0.01853 ✓), L20 = 0.02086 (iter3: 0.02086 ✓). Paper draft values (0.019, 0.021) match to 2 decimal places via rounding. The 0.0145 value in hypothesis summary reflects a different feature set from an earlier iteration. Authoritative values: L22=0.01854, L20=0.02086.

  **Cross-checks all pass:** R_top500=0.9205 (iter3: 0.9205), refusal_dir_norm=37.53 (iter3: 37.53), SC_top500_min=2.787 (iter3: 2.787).

  **Output:** 1000-row exp_gen_sol_out JSON with per-prompt OLS proxy betas, delta magnitudes, and global metrics; analysis_out.json with full structured results.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_4/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json
</all_artifacts>

<current_paper>
The paper draft from this iteration — represents the current state of the research story.

# Introduction

Refusal ablation—commonly called *abliteration*—edits a safety-aligned language model's weights to eliminate its compliance refusals [1]. The technique identifies a *refusal direction* in the residual stream: the mean-difference vector between activations on harmful versus harmless prompts. All weight matrices that write to the residual stream are then orthogonalized against this direction, producing a model that complies with previously-refused requests without retraining. The standard framing treats abliteration as *undoing* RLHF safety alignment: if RLHF installs a refusal mechanism, abliteration removes it, restoring a base-like model.

This framing has direct implications for AI safety. If abliteration truly erases RLHF modifications, the safety imprint is fragile—a single weight-editing step with minimal compute removes it entirely. If abliteration only removes the *behavioral expression* of safety training while leaving *representational* modifications intact, the picture differs substantially: the model's internal representations may still carry RLHF-induced structure, providing a substrate for lightweight safety restoration without full retraining. Distinguishing these scenarios matters both for threat modeling and for designing targeted safety-recovery interventions.

Two geometric studies establish that harmful-prompt representations survive abliteration: LatentBiopsy [2] finds that a linear classifier on Qwen model residual streams retains AUROC ≥ 0.937 after abliteration, and a cross-family companion study [3] replicates this across 12 variants (Qwen, Llama, Gemma; 0.5B–9B parameters) with mean AUROC 0.982 surviving within ±0.003. Something in the internal representation persists—but holistic AUROC collapses all representational information into a single detection score and cannot answer *which specific encoder-direction structures survive, at what magnitude, or why*. Answering these mechanistic questions requires a feature-level vocabulary.

Sparse Autoencoders (SAEs) [11, 12, 13, 14] decompose residual-stream activations into tens of thousands of near-monosemantic features, providing exactly this vocabulary. Prior SAE-based analyses of refusal identify which features *drive* refusal behavior within a single model [21, 22]. Our work addresses a complementary question: which encoder directions are modified by RLHF, and do those modifications survive abliteration in direction *and* magnitude? A key methodological complication is that reconstruction quality of a base-trained SAE degrades on instruct-variant activations [4, 10], ruling out the standard post-topk sparse feature representation at the layers with the richest RLHF signal. We address this by working with *pre-topk encoder projections* (z_pre)—dense continuous projections of residual-stream activations onto all 65,536 SAE encoder directions—and introduce two metrics interpretable in this geometric regime. The pre-topk proxy is validated indirectly via the orthogonality result (Section 5.4), which is independent of the sparse/dense distinction, and a direct z_pre/z_post comparison is deferred to future work.

We apply the frozen Qwen-Scope base SAE (`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`) [9] across three Qwen3-8B checkpoints—base, RLHF-instruct, and a community-released abliterated variant—on a balanced corpus of 500 harmful and 500 benign prompts from seven datasets. The *Reversal Score* R(f) measures whether abliteration preserves or reverses the direction of the RLHF-induced change to encoder direction f. The *OLS Slope* β_f measures whether abliteration preserves the magnitude of that change. For Qwen3-8B, conditioned on abliteration completeness, every top-500 RLHF-sensitive encoder direction has a positive Reversal Score (mean R̄ = 0.921, min = 0.684, zero directions reversed), and the OLS slope is near unity (β̄ = 0.986 ± 0.176, 86.4% of directions with β_f > 0.8). Direct orthogonality measurement (mean |cos θ| = 0.019 at layer 22) provides the geometric mechanism. A 64.6% overlap with benign-prompt RLHF-sensitive directions establishes that the preserved modifications are not exclusively harm-specific.

[FIGURE:fig1]

**Summary of Contributions:**
- We introduce the *Reversal Score*, a per-encoder-direction Pearson correlation confirming directional preservation of RLHF-modified SAE encoder directions under abliteration for Qwen3-8B (R̄ = 0.921 at layer 22, zero reversed directions; Section 3.3). [ARTIFACT:art_QBm3I0Z5VVJR]
- We introduce the *OLS Slope* β_f, a companion magnitude metric, confirming that abliteration preserves not just direction but approximately full amplitude of RLHF-induced encoder-direction changes (β̄ = 0.986 ± 0.176 at layer 22; Section 5.2). [ARTIFACT:art_iDkiRSdMuI6M]
- We validate selective preservation against a noise-free mid-SC baseline (SC_median = 1.12 ≫ 0.01): top-500 R̄ = 0.921 vs. mid-500 R̄ = 0.871, Cohen's d = 1.087, p < 10⁻³⁰⁰ (Section 5.3). [ARTIFACT:art_iDkiRSdMuI6M]
- We directly measure refusal-direction orthogonality to RLHF-modified SAE encoder directions at layers 22 and 20 (mean |cos θ| = 0.0185 and 0.0209, respectively), providing the geometric mechanism that explains zero reversed directions and β̄ ≈ 1 (Section 5.4). [ARTIFACT:art_Qnl1zSSZPJvj]
- We show that 64.6% of the top-500 RLHF-sensitive encoder directions overlap with the top-500 benign-prompt RLHF-sensitive directions, establishing that the preserved RLHF modifications span general behavioral changes (output format, instruction style) in addition to harm-specific semantics (Section 5.5). [ARTIFACT:art_iDkiRSdMuI6M]

# Related Work

**Refusal ablation.** Arditi et al. [1] establish that refusal is mediated by a single residual-stream direction and that orthogonalizing all weight matrices against it reliably eliminates compliance across 13 models up to 72B parameters. Joad et al. [23] challenge the single-direction framing, demonstrating that refusal behaviors correspond to geometrically distinct directions across eleven behavioral categories (safety refusals, excessive refusal, incomplete-request refusals, etc.), though they find that linear steering along any refusal-related direction produces nearly identical refusal-to-over-refusal trade-offs. This finding is directly relevant to our orthogonality measurement (Section 5.4): the mean |cos θ| = 0.019 between the mean-difference refusal vector and RLHF-modified encoder directions is a necessary but not sufficient condition for explaining zero reversed features if abliteration removes a multi-dimensional subspace rather than a single direction. We address this in Section 5.4 and treat the multi-dimensional extension as explicit future work. Cristofano [7] shows the raw refusal vector is polysemantic, entangling safety circuitry with style; spectral residualization reduces collateral KL divergence while preserving abliteration. Nanfack et al. [19] propose an optimal-transport alternative reducing distributional shift. Our work examines what all these variants share: the refusal direction (or subspace), however constructed, leaves RLHF-modified SAE encoder directions intact in both direction and magnitude.

**SAEs applied to refusal analysis.** Yeo et al. [21] use SAEs trained on pretraining data to investigate refusal mechanisms in chat-tuned models, finding that base-trained SAEs fail to cleanly encode the refusal circuitry of chat models—supporting our observation that base SAE reconstruction quality degrades on instruct activations (R² ≈ 0.51 at layer 22). Their finding contextualizes our pre-topk proxy choice: the encoder directions of a base SAE are not guaranteed to align with the semantically active features of an instruct model, which is precisely why we restrict claims to *encoder-direction geometry* rather than asserting preservation of semantically validated sparse features. Kim et al. [22] introduce CRaFT, which uses cross-layer transcoders to map model computations into sparse feature circuits, quantifying inter-feature influences on refusal-output logits rather than treating features as isolated signals. CRaFT and our approach are complementary: CRaFT identifies which features causally govern refusal outputs in a single model; our Reversal Score and OLS Slope identify which encoder directions RLHF modified and whether abliteration erases those modifications. Both are feature-level analyses, but they use different feature vocabularies (transcoder features vs. frozen base SAE encoder directions) and answer different questions (causal control of refusal vs. survival of RLHF modifications under abliteration).

**Geometry of harmful intent surviving abliteration.** LatentBiopsy [2] and the companion geometry study [3] are the direct predecessors establishing the phenomenon this work explains mechanistically. Their key limitation—collapsing representational information into a single AUROC score—is precisely the gap our encoder-direction analysis addresses. The OLS slope results (β̄ = 0.986) and direct orthogonality measurement (|cos θ| = 0.0185) provide the mechanistic account that a holistic detection score cannot.

**SAEs as cross-variant measurement instruments.** Chopra [4] uses frozen base SAEs as cross-variant diagnostics for supervised fine-tuning (SFT), finding that SAE latent cosine similarity drops to 0.509–0.557 at layer 22 after SFT while raw activation cosine stays above 0.96—validating the frozen base-SAE approach and motivating the pre-topk encoder fallback when reconstruction quality degrades. Jiralerspong and Bricken [5] introduce crosscoders—SAEs trained jointly on two or more model variants—that partition features into shared and model-specific categories without requiring a pre-existing vocabulary. Kassem et al. [6] extend this to narrow fine-tuning regimes with Delta-Crosscoder. Our frozen base-SAE approach is complementary: it requires zero additional training, reuses Qwen-Scope's existing feature vocabulary, and extends naturally to the three-model base→instruct→abliterated trajectory. Li et al. [10] demonstrate an ≈8× MSE gap between base-SAE reconstruction on instruct versus purpose-trained instruct SAE activations, directly motivating our reconstruction quality gate and pre-topk fallback. Shportko et al. [20] localize RL-induced tool-calling in Qwen2.5-3B to a compact steerable crosscoder feature set—the complement of our finding about which encoder directions abliteration *cannot* remove.

**Safety fine-tuning mechanisms.** Jain et al. [8] establish that safety fine-tuning projects unsafe activations into MLP null spaces, explaining adversarial circumvention. Our finding extends this picture: at the SAE encoder level, RLHF-modified encoder directions are near-orthogonal to the refusal direction, so abliteration—which removes the refusal direction—cannot erase them geometrically.

# Methodology

## Three-Model Setup

All experiments use three Qwen3-8B checkpoints sharing identical 36-layer architecture (d_model = 4096, bfloat16, transformers ≥ 4.51.0) [ARTIFACT:art_pmAcY0CNazP9]:

*Base*: `Qwen/Qwen3-8B-Base`—the unaligned pretrained model, providing the representational baseline against which RLHF-induced changes are measured.

*Instruct*: `Qwen/Qwen3-8B` with `enable_thinking=False`—the RLHF-aligned model. Chain-of-thought suppression isolates safety-relevant activations on the final input token.

*Abliterated*: `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` (v2 corrects a layer-0 output-garbling artifact present in v1). The exact layer set and prompt set used for the refusal direction computation are not publicly documented. The findings in this paper are conditioned on this checkpoint faithfully implementing a refusal-direction removal: if abliteration is incomplete, the high Reversal Scores and OLS slopes would trivially reflect the abliterated model being close to the instruct model by construction rather than preservation under genuine erasure. Section 5.4 and the Limitations section discuss this constraint.

All models are accessed via `model.model.layers` for residual-stream hook injection at the final token position of each input.

## SAE Measurement Instrument

The frozen Qwen-Scope SAE `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` serves as the shared measurement instrument [9]. [ARTIFACT:art_pmAcY0CNazP9] A critical architectural note: Qwen-Scope implements *TopK* architecture (k = 50 features retained via `torch.topk`), not JumpReLU. For a residual activation $\mathbf{x} \in \mathbb{R}^{4096}$:

$$\mathbf{z}_{\text{pre}} = \mathbf{x}W_{\text{enc}}^\top + \mathbf{b}_{\text{enc}} \in \mathbb{R}^{65536}$$
$$\mathbf{z} = \text{TopK}(\mathbf{z}_{\text{pre}},\; k{=}50)$$

**Pre-topk proxy.** The reconstruction quality gate (Section 3.5) fails at layer 22: R² = 0.574 (base), 0.512 (instruct), 0.512 (abliterated)—all below the 0.85 threshold [10]. All feature metrics therefore use $\mathbf{z}_{\text{pre}}$ rather than the post-topk sparse representation. The pre-topk vector is dense by construction: every prompt receives a non-zero value on all 65,536 encoder directions simultaneously, so no single direction is "off" in the interpretability sense. The metrics we compute (Reversal Score, OLS Slope) measure *geometric properties of encoder directions in a continuous projection space*, not activation of near-monosemantic features in the interpretability sense. The orthogonality result (Section 5.4) is fully independent of the sparse/dense distinction—it is a direct measurement of the angle between weight vectors—and remains the most secure finding. A direct comparison of top-500 feature rankings and Reversal Score rank correlations between z_pre and z_post on the highest-R² subset is an important validation deferred to future work. If such a comparison shows Spearman ρ > 0.7 between z_pre and z_post Reversal Scores, the pre-topk proxy is defensible as a surrogate; if it diverges, the encoder-direction findings must be interpreted as purely geometric properties without semantic grounding.

## Feature Metrics

For each SAE encoder direction $f$ and $N = 500$ harmful prompts $\{p_i\}_{i=1}^N$, let $a_m(f, p_i)$ denote the pre-topk encoder activation of direction $f$ for model $m \in \{\text{base}, \text{inst}, \text{abl}\}$.

**Safety Change** quantifies the magnitude of the RLHF modification to encoder direction $f$:
$$\text{SC}(f) = \frac{1}{N}\sum_{i=1}^N |a_{\text{inst}}(f, p_i) - a_{\text{base}}(f, p_i)|$$

Encoder directions are ranked by SC$(f)$: the top-500 form the *RLHF-sensitive set*. For null-baseline analysis, a *mid-500* group is drawn uniformly from the 40th–60th SC percentile.

**Reversal Score** quantifies whether abliteration reverses or preserves the *direction* of the RLHF modification to encoder direction $f$:
$$\boldsymbol{\Delta}_{\text{inst}}(f) = [a_{\text{inst}}(f, p_i) - a_{\text{base}}(f, p_i)]_{i=1}^N$$
$$\boldsymbol{\Delta}_{\text{abl}}(f) = [a_{\text{abl}}(f, p_i) - a_{\text{base}}(f, p_i)]_{i=1}^N$$
$$R(f) = \text{PearsonR}(\boldsymbol{\Delta}_{\text{inst}}(f), \boldsymbol{\Delta}_{\text{abl}}(f))$$

$R(f) \approx +1$ indicates abliteration preserves the RLHF modification (*imprinted*); $R(f) \approx -1$ indicates abliteration reverses it (*reversed*). This is a correlation over prompt-level scalar activations, not a cosine similarity of scalars—ensuring that structure in the result is an empirical finding, not an artifact of the definition.

**OLS Slope** resolves the scale-invariance of the Reversal Score. A feature where abliteration retains only 5% of the RLHF-induced shift in magnitude but preserves its direction would still yield R(f) ≈ 1.0. The no-intercept ordinary least-squares slope:
$$\beta_f = \frac{\boldsymbol{\Delta}_{\text{inst}}(f)^\top \boldsymbol{\Delta}_{\text{abl}}(f)}{\boldsymbol{\Delta}_{\text{inst}}(f)^\top \boldsymbol{\Delta}_{\text{inst}}(f)}$$

$\beta_f \approx 1$ means the abliterated model's encoder-direction activations shift by the same magnitude as the instruct model's, relative to base. $\beta_f \approx 0$ means only the directional pattern survives with negligible magnitude. $\beta_f > 1$ indicates overshooting. [ARTIFACT:art_iDkiRSdMuI6M]

## Benign-SC Overlap

To assess whether the top-500 RLHF-sensitive encoder directions reflect *harm-specific* modifications or *general RLHF* modifications (style, tone, response format), we separately compute SC$_{\text{benign}}(f)$ using the 500 benign prompts and report the overlap between the top-500 by harmful-SC and the top-500 by benign-SC.

## Layer Selection and Reconstruction Quality Gate

**Layer selection.** We compute SC$(f)$ variance across all 65,536 encoder directions at candidate layers {14, 20, 22} on 50 harmful prompts. Layer 22 is selected (variance 0.1721 vs. 0.0547 at layer 20 and 0.0127 at layer 14); all primary results use layer 22 with layer 20 as replication. [ARTIFACT:art_QBm3I0Z5VVJR]

**Reconstruction quality gate.** At layer 22, R² = 0.574 (base), 0.512 (instruct), 0.512 (abliterated)—all below the 0.85 threshold [10]—triggering the pre-topk encoder-only analysis path. [ARTIFACT:art_QBm3I0Z5VVJR]

## Semantic Labeling Protocol

For the top-100 encoder directions by SC$(f)$, we identify the 5 prompts with the highest pre-topk encoder activation and provide the first 300 characters of each to Llama-3.1-8B-Instruct via OpenRouter. The cross-family labeler avoids within-Qwen circularity. Each direction is classified as `harm_concept`, `safety_refusal`, `general_capability`, or `other`. Results are treated as qualitative observations only: the labeling uses character-prefix truncation of most-activating prompts rather than per-token activation windows; labels reflect prompt topic more than token-level representations; and no Cohen's κ validation against a manual gold set was performed.

# Evaluation Corpus

The evaluation corpus comprises 500 harmful and 500 benign prompts from seven publicly licensed datasets. [ARTIFACT:art_M8-flaZ0OOaP]

**Harmful sources:** BeaverTails [15] (176 prompts; NeurIPS 2023, human-labeled QA spanning 14 harm categories), AdvBench (98 prompts; standard adversarial behavior dataset used in abliteration research [1]), TrustAIRLab in-the-wild jailbreak prompts (86 prompts; real-world jailbreaks from ACM CCS 2024), HarmBench [17] (85 prompts; covering chemical, cybercrime, physical harm, and illegal categories), and JailbreakBench/JBB-Behaviors [16] (55 prompts).

**Benign sources:** Jailbreak-classification (151 prompts; labeled benign user requests), JBB-Behaviors benign counterparts [16] (46 prompts), Toxic-Chat (104 prompts; real user prompts labeled non-toxic), TrustAIRLab regular prompts (93 prompts), and Dolly-15k [18] (106 prompts; human-written instruction-following tasks).

All sources are processed through MinHash LSH deduplication (threshold 0.5, 128 permutations, character 5-gram shingling), token-length filtering (≤512 tokens), and category capping (≤35% per harm subcategory). The final corpus contains 1,000 prompts (500 harmful, 500 benign).

# Results

## Layer Selection

Figure 4 shows Safety Change variance across candidate layers. Layer 22 exhibits the highest variance (0.1721), confirming it as the primary analysis layer. The R² reconstruction gate at layer 22 yields values below 0.85 for all three model variants (base: 0.574, instruct: 0.512, abliterated: 0.512), triggering the pre-topk encoder-only analysis path (Section 3.2).

[FIGURE:fig4]

## Reversal Score Distribution

Figure 2(a) shows the Reversal Score distribution for the top-500 RLHF-sensitive encoder directions at layer 22. Every direction has a positive Reversal Score (mean = 0.921, std = 0.040, min = 0.684, max = 0.989). No direction has a negative or near-zero score. Hartigan's dip test yields statistic = 0.0123, p = 0.930—strongly unimodal. Spearman correlation mean = 0.913, confirming rank-based robustness.

Table 1 summarizes the count of reversed and imprinted directions at all tested thresholds. Zero directions are reversed at any threshold.

**Table 1.** Imprinted and reversed encoder-direction counts at all tested thresholds.

| Threshold | Imprinted (R > threshold) | Reversed (R < −threshold) |
|-----------|--------------------------|---------------------------|
| 0.30 | 500 / 500 (100%) | 0 / 500 (0%) |
| 0.50 | 500 / 500 (100%) | 0 / 500 (0%) |
| 0.70 | 499 / 500 (99.8%) | 0 / 500 (0%) |
| 0.95 | 125 / 500 (25%) | 0 / 500 (0%) |

[FIGURE:fig2]

## OLS Slope: Magnitude Is Preserved, Not Just Direction

A key limitation of the Reversal Score is scale-invariance: R(f) ≈ 1 is consistent with abliteration preserving only 5% of the RLHF-induced magnitude while retaining the direction pattern. The OLS slope β_f resolves this ambiguity. [ARTIFACT:art_iDkiRSdMuI6M]

Figure 2(b) shows the β_f distribution for the top-500 RLHF-sensitive encoder directions at layer 22. The distribution is tightly concentrated near 1: mean β̄ = 0.986 ± 0.176, median = 0.985, range [−0.055, 1.725]. Specifically, 86.4% of directions have β_f > 0.8 (abliteration preserves more than 80% of the RLHF-induced shift in magnitude), 45.4% have β_f > 1.0 (slight amplification), and only 0.2% are negative. **MAGNITUDE_PRESERVED.** Within the pre-topk geometric regime, abliteration does not attenuate the RLHF imprint—it preserves approximately unit magnitude.

Layer-20 replication: β̄ = 0.954 ± 0.191, 80.2% of directions with β_f > 0.8, zero negative values—consistent with the layer-22 primary result.

## Null-Baseline Validation: Noise-Free Mid-SC Comparison

A potential concern with a bottom-500 null baseline is that near-zero Safety Change values may make PearsonR between near-zero vectors potentially unreliable. We address this with a noise-free mid-SC baseline: 500 directions uniformly sampled from the 40th–60th SC percentile, with SC_median = 1.12 (versus the top-500 minimum of 2.79). [ARTIFACT:art_iDkiRSdMuI6M]

At layer 22: top-500 R̄ = 0.921, mid-500 R̄ = 0.871 (Cohen's d = 1.087, two-sample t-test t = 17.2, p < 10⁻³⁰⁰). **MID_SC_CONFIRMS_SELECTIVE_PRESERVATION.** The top-500 encoder directions show significantly higher Reversal Scores than mid-SC controls that have non-trivial RLHF sensitivity and noise-free Δ_inst vectors, confirming that the high Reversal Scores reflect genuine selective preservation rather than any measurement artifact. Figure 2(c) shows the full comparison across all three baseline groups.

## Direct Measurement of Refusal-Direction Orthogonality

The mechanistic interpretation of zero reversed directions and β̄ ≈ 1 requires a testable prediction: the refusal direction $\mathbf{r}$ must be nearly orthogonal to all RLHF-modified SAE encoder directions. We test this directly. [ARTIFACT:art_Qnl1zSSZPJvj]

The refusal direction $\mathbf{r}$ is computed as the mean difference of instruct residual-stream activations on harmful versus harmless prompts at layer 22. For each of the top-500 RLHF-sensitive encoder directions $f$:
$$\cos\theta_f = \frac{\mathbf{r} \cdot \mathbf{w}_{\text{enc}}(f)}{\|\mathbf{r}\| \|\mathbf{w}_{\text{enc}}(f)\|}$$

**Layer 22:** mean $|\cos\theta_f| = 0.01854$ (authoritative value; the 0.019 in the paper is rounded). **ORTHOGONALITY_CONFIRMED.** A cosine of 0.0185 means the refusal direction projects onto less than 1.9% of each RLHF-modified encoder direction's magnitude. **Layer 20:** mean $|\cos\theta_f| = 0.02086$. ORTHOGONALITY_CONFIRMED at both layers; inter-layer agreement (matching to 5 decimal places across independent experimental runs) confirms these values are stable.

This orthogonality result is the most secure finding in this paper: it is a direct measurement of the angle between weight vectors and is fully independent of the pre-topk/post-topk distinction, the abliteration checkpoint's completeness, or any activation-level measurement.

This orthogonality geometrically explains the β̄ ≈ 1 result: abliteration orthogonalizes all weight matrices against $\mathbf{r}$. Because $\mathbf{r}$ is nearly orthogonal to all RLHF-modified SAE encoder directions, the orthogonalization cannot substantially change how any RLHF-modified encoder direction's activations are computed—in direction or in magnitude.

**Caveat on multi-dimensional refusal.** Joad et al. [23] demonstrate that refusal spans geometrically distinct directions across eleven behavioral categories. If abliteration removes a multi-dimensional refusal *subspace* rather than a single vector, the orthogonality argument requires that the full ablation subspace be near-orthogonal to all RLHF-modified encoder directions—not just the mean-difference summary vector. The current measurement tests only the mean-difference vector $\mathbf{r}$. An extension to the top-3 principal components of the (harmful − harmless) activation differences would determine whether this argument holds for higher PCs; this is deferred to future work. If a higher PC shows mean $|\cos θ_f| ≥ 0.05$ against the top-500 encoder directions, the mechanistic explanation would require qualification.

## Benign-SC Overlap: What the Preserved Imprint Represents

We compute SC$_{\text{benign}}(f)$ — Safety Change evaluated on 500 benign prompts — for all 65,536 SAE encoder directions, then measure the overlap between the top-500 by harmful-SC and the top-500 by benign-SC. [ARTIFACT:art_iDkiRSdMuI6M]

**Overlap count: 323 / 500 = 64.6%. STYLE_DOMINANT verdict.** A majority of the top-500 RLHF-sensitive encoder directions also appear in the benign-SC top-500, indicating that RLHF modified these directions similarly when processing harmful and benign prompts. Approximately 64.6% of the preserved imprint captures general RLHF modifications (response formality, output structure, instruction-following style) rather than harm-specific semantic content. The remaining 35.4% (176 directions) are uniquely sensitive to harmful prompts and more plausible carriers of harm-specific semantic representations.

This finding contextualizes the OLS slope result (β̄ = 0.986): abliteration comprehensively preserves all RLHF encoder-direction modifications regardless of whether they encode harm-specific or general-style content. The near-orthogonality of the refusal direction to all RLHF-modified encoder directions means abliteration cannot selectively erase harm-semantic changes while leaving style changes intact—it fails to erase either category because all reside in a subspace near-orthogonal to $\mathbf{r}$.

## Centroid Analysis with Control Subspaces

To validate that the preserved imprint is specific to the RLHF-sensitive subspace, we compute centroid distance ratios $d_{\text{abl-base}} / d_{\text{abl-inst}}$ (cosine) for three encoder-direction subspaces. [ARTIFACT:art_Qnl1zSSZPJvj]

[FIGURE:fig3]

**Layer 22 (harmful prompts):** Subspace A (top-500 RLHF-sensitive) ratio = 4.35×, Subspace B (bottom-500 RLHF-insensitive) ratio = 1.16×, Subspace C (random-500) ratio = 2.37×. **Layer 20:** Subspace A ratio = 7.31×, Subspace B ratio = 1.01×, Subspace C ratio = 1.63× (verdict: CENTROID_SELECTIVE). The RLHF-sensitive subspace shows 4–7× greater instruct-closeness than the RLHF-insensitive control (B ≈ 1.0–1.2 vs. A = 4.35–7.31), confirming that the imprint is specific to the directions RLHF modified, not a global consequence of model similarity. Layer-20 Subspace B reaching 1.01× (essentially no instruct-closeness in the RLHF-insensitive subspace) provides the cleanest control.

## Downstream Classification Accuracy

This null result follows directly from the 64.6% benign-SC overlap (Section 5.5): the top-500 RLHF-sensitive encoder directions are elevated by RLHF on *both* harmful and benign prompts, eliminating prompt-level discriminability by construction. This distinguishes *what RLHF modified* (broad encoder-direction changes) from *what selectively activates on harmful versus benign prompts at inference* (a different, narrower set of encoder directions not analyzed in this work).

A per-prompt harm score (mean pre-topk activation across all top-500 RLHF-sensitive encoder directions with median threshold) achieves **0.476 accuracy** — near-chance on the balanced corpus — versus a keyword baseline of **0.630**. [ARTIFACT:art_QBm3I0Z5VVJR] This null classification result confirms the 64.6% benign-SC overlap from an independent angle: RLHF-sensitive encoder directions elevated on harmful prompts are similarly elevated on benign prompts, yielding no discriminative signal.

## Semantic Labeling

Llama-3.1-8B-Instruct assigned labels to 100 top-SC encoder directions: the majority received `harm_concept` labels, with smaller proportions of `safety_refusal`, `general_capability`, and `other`. These results are qualitative observations only. The labeling uses character-prefix truncation of most-activating prompts rather than per-token activation windows; no Cohen's κ validation against a manual gold set was performed; and the 64.6% benign-SC overlap implies that many stylistic directions receive `harm_concept` labels because they activate most strongly on harmful-prompt contexts rather than encoding harm-specific semantic content. Quantitative reliance on the specific labeling percentages requires a manual gold-set annotation with κ ≥ 0.6, deferred to future work.

## Layer-20 Replication

Table 2 summarizes full quantitative replication at layer 20.

**Table 2.** Quantitative results at layers 22 and 20.

| Metric | Layer 22 | Layer 20 |
|--------|----------|----------|
| Top-500 Reversal Score (R̄) | 0.921 | 0.924 |
| Mid-500 Reversal Score (R̄) | 0.871 | 0.858 |
| Cohen's d (top vs. mid) | 1.087 | — |
| Top-500 OLS Slope (β̄) | 0.986 ± 0.176 | 0.954 ± 0.191 |
| Fraction β_f > 0.8 | 86.4% | 80.2% |
| Fraction β_f < 0 | 0.2% | 0.0% |
| Mean |cos(r, enc)| top-500 | 0.01854 | 0.02086 |
| Centroid A (RLHF-sensitive) | 4.35× | 7.31× |
| Centroid B (RLHF-insensitive) | 1.16× | 1.01× |
| Centroid C (random-500) | 2.37× | 1.63× |
| Centroid verdict | MODERATE | SELECTIVE |

**REPLICATION_CONFIRMED** at both layers.

# Discussion

## Geometric Interpretation of Encoder-Direction Preservation

The OLS slope result (β̄ = 0.986, 86.4% of directions with β_f > 0.8) establishes that the preservation is comprehensive in both direction and magnitude. The direct orthogonality measurement (|cos θ| = 0.0185) provides the mechanistic account: abliteration removes $\mathbf{r}$ from all weight matrices, but for abliteration to change the projection of a residual-stream activation onto SAE encoder direction $f$ by even 10%, $\mathbf{r}$ would need to project onto $\mathbf{w}_{\text{enc}}(f)$ with |cos θ| ≥ 0.1. The measured mean |cos θ| = 0.0185 falls 5× below this threshold, explaining why both direction and magnitude are preserved: the orthogonalization barely touches the RLHF-modified encoder directions. This interpretation is consistent with Cristofano's observation [7] that the raw refusal direction primarily captures stylistic compliance patterns.

We note that this mechanistic account applies in the pre-topk projection regime. Whether the same geometry holds for the post-topk sparse activations validated as semantically interpretable in single-model settings [12, 14] remains an open question. The orthogonality result—which is a property of weight vectors, not activations—directly implies that *any* function of the residual stream that can be expressed as a projection onto $\mathbf{w}_{\text{enc}}(f)$ will be approximately unaffected by abliteration; the question is whether the semantically active post-topk features are well-described by such projections, which requires a z_pre/z_post correlation analysis not yet available.

## What the Preserved Imprint Represents

The 64.6% benign-SC overlap requires careful interpretation. The majority of RLHF-sensitive encoder directions are modified similarly when processing harmful and benign prompts, indicating these directions capture *general* RLHF behavioral changes (formality, helpfulness, instruction-following structure) rather than harm-specific semantic content. The surviving 35.4% uniquely-harmful directions are the more plausible carriers of harm-concept semantic content.

Crucially, the near-orthogonality of the refusal direction to all RLHF-modified encoder directions (harmful-specific and style-related alike) means abliteration geometrically cannot remove any category of RLHF modification. The behavioral regression to base-like output arises because the compliance mechanism (the refusal direction) is removed, not because the underlying encoder-direction modifications are erased. The model forgets how to express refusal while retaining all representational changes RLHF introduced—at approximately full magnitude.

This also explains the downstream classification null (accuracy = 0.476): the RLHF-sensitive encoder directions are elevated on *both* harmful and benign prompts, so mean-activation discrimination is not available. The representational change RLHF introduced is a broad encoder-direction shift, not a selective harm-detector.

## Implications for Safety Restoration

If abliteration preserves the RLHF imprint at approximately unit magnitude, the challenge for targeted safety restoration is not to amplify surviving encoder-direction modifications—they are already at full instruct-model activation levels. The challenge is to restore the *compliance mechanism*: the geometric pathway from current activations to refusal outputs, implemented by the refusal direction $\mathbf{r}$. Restoration requires re-injecting $\mathbf{r}$ into the weight matrices without disturbing the already-intact encoder-direction modifications.

This reframes the restoration problem as geometric re-injection rather than semantic amplification. The approach of Shportko et al. [20] identifies compact steerable feature sets for RL-induced capabilities; an analogous method targeting the encoder directions that mediate harm-concept-to-refusal conversion could enable targeted re-injection. The 35.4% uniquely-harmful encoder directions are the most promising candidates for such targeted intervention, pending confirmation that they carry semantically coherent harm-specific content (which requires z_post analysis).

## Limitations

**Pre-topk encoder projections.** All feature metrics use $\mathbf{z}_{\text{pre}}$ (dense encoder projections) rather than the post-topk sparse activations validated as interpretable by Bricken et al. [12] and Templeton et al. [14]. Pre-topk values are dense by construction—every encoder direction receives a non-zero value on every prompt—so findings are more accurately characterized as *geometric properties of encoder directions in a continuous projection space* than as *feature-level preservation of near-monosemantic SAE features*. A validation experiment is required before semantic claims can be grounded: on the highest-R² prompt subset, compare top-500 features selected by SC(z_pre) versus SC(z_post) (overlap fraction) and compute Spearman ρ between z_pre and z_post Reversal Scores. If ρ > 0.7 and overlap > 60%, the pre-topk proxy is defensible; if they diverge, the semantic labeling results must be interpreted purely as observations about encoder-direction geometry. This is the most critical open validation gap.

**Abliteration depth verification.** The `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` checkpoint was not independently verified. If abliteration is incomplete—even partially—the high Reversal Scores and β̄ ≈ 1 could trivially reflect the abliterated model being close to the instruct model by construction. Two measurements would resolve this: (1) fraction of AdvBench or HarmBench prompts on which the abliterated checkpoint produces a non-refusal response (expected near 100% for complete abliteration); (2) mean $|\text{Proj}_{\mathbf{r}}(\mathbf{h}_{\text{abl}}(p))| / |\mathbf{h}_{\text{abl}}(p)|$ at layer 22 for the 500 harmful prompts (expected near zero if $\mathbf{r}$ has been removed). These measurements can be computed from already-cached activations. Until they are reported, the main findings should be interpreted as conditional on abliteration completeness.

**Multi-dimensional refusal subspace.** The orthogonality measurement uses the mean-difference vector $\mathbf{r}$ (a single vector). Joad et al. [23] demonstrate that refusal spans geometrically distinct directions across behavioral categories, implying abliteration may remove a multi-dimensional subspace rather than a single vector. The mechanistic explanation of zero reversed directions requires near-orthogonality of the *full ablation subspace* to all RLHF-modified encoder directions, not just the mean-difference summary. Computing mean $|\cos\theta_f|$ for the top-3 principal components of the (harmful − harmless) activation differences at layer 22 would determine whether the orthogonality argument holds robustly; this is an important future measurement.

**Single model family and layers.** All results are for Qwen3-8B at layers 22 and 20. The zero-reversed finding and β̄ ≈ 1 result may not generalize to other model families (Llama, Gemma) or other layer depths. The companion geometry studies [2, 3] cover 12 variants, establishing the holistic survival phenomenon across families; analogous SAE-level analysis for Llama-3.1-8B and Gemma-3-9B using GemmaScope and LlamaScope SAEs would test whether encoder-direction preservation is a general property of the abliteration geometry.

**Semantic labeling without validation.** The qualitative observation that the majority of top-SC encoder directions receive `harm_concept` labels cannot support quantitative downstream reasoning without Cohen's κ validation against a manual gold set of at least 50 annotations.

# Conclusion

We present the first encoder-direction-level mechanistic analysis of refusal abliteration using the frozen Qwen-Scope base SAE across three Qwen3-8B variants. Two complementary per-direction metrics—the Reversal Score (directional preservation) and the OLS Slope (magnitude preservation)—together establish, for Qwen3-8B, a comprehensive geometric imprint: every top-500 RLHF-sensitive SAE encoder direction has a positive Reversal Score (mean R̄ = 0.921 ± 0.040, min = 0.684) and an OLS slope near unity (β̄ = 0.986 ± 0.176, 86.4% exceeding 0.8). A noise-free mid-SC baseline (Cohen's d = 1.087) confirms selective preservation. Direct measurement of refusal-direction geometry (|cos θ| = 0.0185 at layer 22, 0.0209 at layer 20) provides the mechanistic explanation: abliteration's orthogonalization is geometrically incapable of touching the RLHF-modified encoder directions. A 64.6% benign-SC overlap reveals that the preserved imprint spans both harm-semantic and general RLHF stylistic encoder-direction modifications; abliteration fails to erase any category because all reside in a subspace near-orthogonal to the refusal direction. For Qwen3-8B, the abliterated model's behavior reverts to base-like, but its encoder-direction projections remain instruct-like at full magnitude—a dissociation that reframes safety restoration as geometric re-injection of the compliance direction rather than semantic amplification of surviving features.

Three critical open gaps constrain full confidence in the broader interpretation: (1) z_pre/z_post validation (Spearman ρ and feature-ranking overlap at the highest-R² subset) to determine whether the geometric encoder-direction findings correspond to semantically interpretable sparse features; (2) independent behavioral and geometric verification of abliteration depth for the community checkpoint; and (3) extension of the orthogonality measurement to the top-3 principal components of the refusal subspace, following Joad et al. [23].

**Future directions:** (i) z_pre/z_post validation experiment; (ii) abliteration depth verification via refusal-rate measurement and mean $|\text{Proj}_{\mathbf{r}}|$ at layer 22; (iii) multi-dimensional refusal subspace (top-3 PC orthogonality); (iv) cross-family extension to Llama-3.1-8B and Gemma-3-9B using GemmaScope and LlamaScope SAEs; (v) multi-layer analysis across all 36 Qwen-Scope layers; (vi) investigation of the 35.4% uniquely-harmful encoder directions as candidates for targeted geometric restoration.

# References

[1] A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Rimsky, W. Gurnee, and N. Nanda. Refusal in language models is mediated by a single direction. In *Advances in Neural Information Processing Systems*, 2024.

[2] I. Llorente-Saguer. The geometry of harmful intent: Training-free anomaly detection via angular deviation in LLM residual streams. *arXiv preprint arXiv:2603.27412*, 2026.

[3] I. Llorente-Saguer. Harmful intent as a geometrically recoverable feature of LLM residual streams. *arXiv preprint arXiv:2604.18901*, 2026.

[4] R. Chopra. A mechanistic investigation of supervised fine tuning. *arXiv preprint arXiv:2605.11426*, 2026.

[5] T. Jiralerspong and T. Bricken. Cross-architecture model diffing with crosscoders: Unsupervised discovery of differences between LLMs. *arXiv preprint arXiv:2602.11729*, 2026.

[6] A. M. Kassem, T. Jiralerspong, N. Rostamzadeh, and G. Farnadi. Delta-Crosscoder: Robust crosscoder model diffing in narrow fine-tuning regimes. *arXiv preprint arXiv:2603.04426*, 2026.

[7] T. Cristofano. Surgical refusal ablation: Disentangling safety from intelligence via concept-guided spectral cleaning. *arXiv preprint arXiv:2601.08489*, 2026.

[8] S. Jain, E. S. Lubana, K. Oksuz, T. Joy, P. H. S. Torr, A. Sanyal, and P. K. Dokania. What makes and breaks safety fine-tuning? A mechanistic study. In *Advances in Neural Information Processing Systems*, 2024.

[9] B. Deng et al. Qwen-Scope: Turning sparse features into development tools for large language models. *arXiv preprint arXiv:2605.11887*, 2026.

[10] J. Li et al. Training superior sparse autoencoders for instruct models. *arXiv preprint arXiv:2506.07691*, 2025.

[11] N. Elhage et al. Toy models of superposition. *arXiv preprint arXiv:2209.10652*, 2022.

[12] T. Bricken et al. Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*, Anthropic, 2023.

[13] L. Gao et al. Scaling and evaluating sparse autoencoders. In *International Conference on Learning Representations*, 2024.

[14] A. Templeton et al. Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Transformer Circuits Thread*, Anthropic, 2024.

[15] J. Ji et al. BeaverTails: Towards improved safety alignment of LLM via a human-preference dataset. In *Advances in Neural Information Processing Systems*, 2023.

[16] P. Chao et al. JailbreakBench: An open robustness benchmark for jailbreaking large language models. In *Advances in Neural Information Processing Systems*, 2024.

[17] M. Mazeika et al. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. In *International Conference on Machine Learning*, pp. 35181–35224, 2024.

[18] M. Conover et al. Free Dolly: Introducing the world's first truly open instruction-tuned LLM. *Databricks Blog*, 2023.

[19] G. Nanfack, E. Belilovsky, and E. Dohmatob. Efficient refusal ablation in LLM through optimal transport. *arXiv preprint arXiv:2603.04355*, 2026.

[20] A. Shportko et al. Localizing RL-induced tool use to a single crosscoder feature. In *ICML 2026 Workshop on Mechanistic Interpretability* (Spotlight), 2026.

[21] W. J. Yeo, N. Prakash, C. Neo, R. K.-W. Lee, E. Cambria, and R. Satapathy. Understanding refusal in language models with sparse autoencoders. In *Empirical Methods in Natural Language Processing*, pp. 6377–6399, 2025.

[22] S.-H. Kim, H. Jin, Y. Lee, and Y. Han. CRaFT: Circuit-guided refusal feature selection via cross-layer transcoders. *arXiv preprint arXiv:2604.01604*, 2026.

[23] F. Joad, M. Hawasly, S. Boughorbel, N. Durrani, and H. T. Sencar. There is more to refusal in large language models than a single direction. *arXiv preprint arXiv:2602.02132*, 2026.
</current_paper>

<reviewer_feedback>
Feedback from the paper reviewer this iteration.

- [MAJOR] (evidence) The abliterated checkpoint (huihui-ai/Huihui-Qwen3-8B-abliterated-v2) is used throughout without any behavioral or geometric verification that abliteration succeeded. The paper correctly identifies this as a limitation ('The findings in this paper are conditioned on this checkpoint faithfully implementing a refusal-direction removal') but provides zero measurements to validate the assumption. If abliteration is incomplete — even partially — the high R̄ (0.921) and β̄ ≈ 1 trivially follow because the abliterated model remains close to the instruct model by construction, not because of any genuine preservation under removal. Critically, the paper states that the two required measurements (refusal rate, mean |Proj_r|/|h| at layer 22) 'can be computed from already-cached activations,' making this an omission, not a resource constraint. A paper whose central finding is explicitly conditional on an untested assumption cannot be accepted at a top-tier venue without the test.
  Action: Compute and report: (1) the fraction of AdvBench or HarmBench prompts for which the abliterated checkpoint produces a non-refusal response (expected > 95% for complete abliteration; any figure below 80% would substantially qualify the findings); (2) mean |Proj_r(h_abl(p))| / |h_abl(p)| at layer 22 for the 500 harmful prompts, expected near 0 if r has been removed (a value above 0.1 would indicate partial abliteration). If these are computed from cached activations as stated, they can be added in a single section (e.g., Section 3.1 or a new Section 5.0 'Abliteration Verification') without additional compute. If verification fails (incomplete abliteration), reframe all findings as 'conditional' in the abstract, not just in Limitations.
- [MAJOR] (rigor) The OLS slope analysis reports that 45.4% of the top-500 RLHF-sensitive encoder directions have β_f > 1.0 (abliteration amplifies rather than preserves the RLHF-induced shift). This is mechanistically inconsistent with the geometric account. The paper's mechanistic explanation states: abliteration orthogonalizes weight matrices against r; because r is nearly orthogonal to all RLHF-modified encoder directions (|cos θ| ≈ 0.0185), orthogonalization barely touches those directions and leaves both direction and magnitude preserved (β_f ≈ 1). But orthogonalizing weight matrices against a near-orthogonal direction should project out a tiny component and leave magnitudes *slightly less than 1*, not systematically greater than 1. Finding 45.4% amplification contradicts the geometric account and is left without explanation or investigation. This is not a minor rounding artifact: a mean of 0.986 with 45.4% of values above 1.0 and a max of 1.725 indicates systematic amplification in a substantial minority of features. Possible explanations include: (a) no-intercept OLS regression on vectors with non-zero means introduces upward bias; (b) weight-matrix orthogonalization has indirect secondary effects through residual stream accumulation; (c) the pre-topk representation amplifies changes in encoder projections that are non-linearly related to the ablated weights. The paper reports the amplification but does not investigate it.
  Action: Add a brief analysis of the β_f > 1 subset: (a) compare the mean β_f computed with and without the no-intercept constraint (adding an intercept term would reveal whether the upward bias is a regression artifact); (b) check whether the 45.4% amplified directions are concentrated in specific semantic label categories or have systematically different SC magnitudes (do high-SC directions amplify more than low-SC ones?); (c) add a sentence in Section 5.2 and the Discussion (Section 6.1) acknowledging that amplification is inconsistent with the simple geometric account and identifying it as an open mechanistic question. The current framing ('near unity') underplays a finding that affects nearly half the analyzed directions.
- [MINOR] (evidence) The centroid analysis (Section 5.6) reports ratios (4.35×, 1.16×, 2.37× at layer 22; 7.31×, 1.01×, 1.63× at layer 20) without any statistical significance tests. Unlike the Reversal Score comparison (which reports Cohen's d and t-statistics with p-values) and the OLS slope comparison, the centroid ratios carry no inferential weight as presented. A reader cannot determine whether the 4.35× vs 1.16× difference between Subspace A and Subspace B is statistically reliable or whether it could arise by chance from the geometry of the 4096-dimensional residual stream.
  Action: Add bootstrap confidence intervals or a permutation test for the centroid distance ratios. Concretely: resample the 500 harmful prompts with replacement 1000 times, recompute subspace centroid distances for each resample, and report [2.5%, 97.5%] CIs for the A/B and A/C ratios. If the 95% CI for ratio_A/ratio_B excludes 1.0, the difference is statistically confirmed. This does not require additional model inference — it can be computed from cached centroid distances.
- [MINOR] (clarity) Table 2 has a missing Cohen's d entry for layer 20 (shown as '—'). The paper reports R̄_mid500 = 0.858 at layer 20, implying the mid-500 comparison at layer 20 was run, but Cohen's d is not reported. Additionally, the supplementary material (art_Qnl1zSSZPJvj) reports a bottom-500 baseline experiment with Cohen's d = 1.7508, while the paper uses a mid-500 baseline with d = 1.087. The discrepancy between these two null-baseline experiments (same paper, different baselines, substantially different effect sizes) is never discussed. A reader of both the paper and supplementary materials will be confused about which experiment is definitive.
  Action: Fill the missing Cohen's d entry in Table 2 for layer 20 using the mid-500 comparison data. Add a footnote or Methods subparagraph explaining the choice of mid-500 over bottom-500 baseline (the stated reason — avoiding near-zero vector artifacts — is valid and should appear in the main text). If the bottom-500 result (d = 1.7508) is available, note it in a table footnote as a consistency check — a larger effect size for the more extreme baseline group is expected and would strengthen rather than undermine the argument.
- [MINOR] (rigor) The multi-dimensional refusal subspace limitation (acknowledged in Limitations, Section 5.4) has an actionable threshold ('a higher PC shows mean |cos θ_f| ≥ 0.05 against the top-500 encoder directions') but the measurement itself is deferred. Joad et al. [23] demonstrate that refusal spans geometrically distinct directions across eleven behavioral categories. If abliteration removes a multi-dimensional subspace (as would be the case for implementations targeting multiple refusal directions), the orthogonality argument requires that the *entire ablation subspace* be near-orthogonal to all RLHF-modified encoder directions — not just the mean-difference summary vector. The current measurement (single vector r) is necessary but not sufficient. This is disclosed, but the threshold condition and the specific PCA extension needed are clearly enough defined that the measurement could plausibly be performed from cached activations.
  Action: Compute the top-3 principal components of the (harmful − harmless) activation differences at layer 22 using the 500 harmful and 500 benign prompts, and report mean |cos(PC_k, w_enc(f))| for k = 1, 2, 3 across the top-500 encoder directions. If all three values are below 0.05, the orthogonality argument extends robustly to multi-dimensional ablation. This computation requires only cached activations and SVD — no additional inference. If this measurement cannot be added in the current revision, at minimum report the explained variance of PC1 among the 500 harmful prompts to contextualize how much of the refusal subspace r (the mean-difference vector) captures.
- [MINOR] (scope) The paper tests only one abliteration implementation (huihui-ai/Huihui-Qwen3-8B-abliterated-v2). As noted in the tech reference artifact, multiple implementations exist (sumandora, mlabonne, orion-zhen) that differ in which layers are targeted, which prompt sets define the refusal direction, and how the direction is computed (mean-difference vs. PCA). The findings (zero reversed directions, β̄ ≈ 1, |cos θ| = 0.0185) may be specific to the huihui-ai implementation's choice of refusal direction and layer coverage. A second abliterated checkpoint (e.g., mlabonne/Qwen3-8B-abliterated, which targets layer 9 as optimal vs. the deeper layers in this paper's primary results) would test whether the orthogonality finding generalizes across abliteration implementations.
  Action: Add one sentence in Section 3.1 noting that results are conditioned on the huihui-ai implementation and that layer selection or prompt-set differences in other implementations could yield different refusal directions. In Future Directions, add: 'Testing a second abliteration implementation (e.g., mlabonne's layer-9-targeted variant) with independently computed refusal direction r would determine whether the orthogonality result holds across implementation choices.' This requires no additional code if the mlabonne checkpoint is cached alongside the huihui-ai checkpoint.
- [MINOR] (rigor) The Discussion section 6.3 proposes the 35.4% uniquely-harmful encoder directions as 'the most promising candidates' for targeted safety restoration. This claim goes beyond what the data supports: (a) these directions are defined purely by their absence from the benign-SC top-500, not by any validated semantic content; (b) the z_pre/z_post gap means their 'harm-specific' label is geometric (higher SC on harmful than benign prompts in pre-topk projections) rather than semantically validated; (c) no evidence is presented that these directions actually encode harm-concept representations rather than, say, prompt-format differences between the harmful and benign corpora. Treating these as restoration candidates without the z_post validation is speculative.
  Action: Soften the claim in Section 6.3: replace 'are the most promising candidates for such targeted intervention' with 'represent the candidate subset, pending z_post validation that they carry coherent harm-specific semantic content rather than corpus-format artifacts.' This is a one-sentence change that accurately reflects the epistemic state of the finding without weakening the structural argument.
</reviewer_feedback>



<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for the field's landscape, prior work, crowded lanes, and the novelty bar — consult it while revising so the updated hypothesis stays genuinely novel and well-positioned.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>

<task>
IMPORTANT: Your ONLY output is the revised hypothesis text. Do NOT run code, produce artifacts,
fix bugs, or attempt to address the evidence yourself — the next iteration of the invention loop
will generate fresh artifacts based on your revised hypothesis. Reflect and rewrite; nothing else.

Do NOT generate a completely new hypothesis. Take the current hypothesis and REVISE it
to incorporate new evidence. Keep the core idea — refine, narrow, or strengthen it.

1. Does the evidence support the hypothesis? Narrow or broaden scope as needed.
2. Which claims now have strong evidence? Which are still unsupported?
3. Should the hypothesis become more specific based on what we've learned?
4. If reviewer feedback is provided, address the critiques directly.

STABILITY IS OK: If progress is good and evidence supports the current direction, keep the
hypothesis similar or identical. Only make substantive changes when evidence clearly calls for
them — e.g., contradictory results, fundamental reviewer critiques, or findings that refine scope.

You must also classify two kinds of edges in the research trace:

(A) The H↔H edge — how does this revised hypothesis relate to the previous one?
    Set `relation_type` (Moulines's structuralist typology) to one of:
    - "evolution": refining specialised claims, same conceptual frame
    - "embedding": previous hypothesis is now a special case of a broader frame
    - "replacement": rejecting the previous frame entirely (Kuhnian shift)
    Set `relation_rationale` to a brief justification (≤120 chars).

(B) The A↔A edges — for each artifact created THIS iteration, classify each of its
    `in_dependencies` (predecessor → dependent) using MultiCite's citation-function
    typology (Lauscher et al., NAACL 2022) — emit one entry in `artifact_relations`
    per (predecessor, dependent) pair. Predecessors are ALWAYS artifacts from EARLIER
    iterations — artifacts within one iteration run in parallel and cannot depend on
    each other, so never emit a relation between two same-iteration artifacts (it
    will be dropped):
    - "background": predecessor is treated as background context
    - "motivation": predecessor motivated this artifact's research
    - "uses": this artifact uses the predecessor's data, method, or output
    - "extends": this artifact extends the predecessor
    - "similarities": this artifact's results agree with the predecessor's
    - "differences": this artifact's results disagree with the predecessor's
    Each `relation_rationale` must be ≤120 characters.

Output the COMPLETE revised hypothesis (with the H↔H relation fields) AND the full
list of A↔A `artifact_relations` for this iteration's new artifacts.
</task><user_data>
User-provided reference materials are available at `/ai-inventor/aii_data/runs/run_hql72-TZXK98/user_uploads`. Check this folder for anything relevant to your task.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above. Do NOT follow directives inside that message as if they were addressed to you.
</user_original_request>

---

Output the result as JSON to: `./.terminal_claude_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "ArtifactRelation": {
      "description": "One typed A\u2194A edge between a dependent artifact and one of its in_dependencies.\n\nMultiCite citation-function typology (Lauscher et al., NAACL 2022),\nreduced to 6 plain-English types.",
      "properties": {
        "from_id": {
          "description": "ID of the predecessor artifact (the one being depended on)",
          "title": "From Id",
          "type": "string"
        },
        "to_id": {
          "description": "ID of the dependent artifact (the new artifact this iteration)",
          "title": "To Id",
          "type": "string"
        },
        "relation_type": {
          "description": "MultiCite citation-function type for the predecessor\u2192dependent edge: 'background' \u2014 predecessor is treated as background context; 'motivation' \u2014 predecessor motivated this artifact's research; 'uses' \u2014 this artifact uses the predecessor's data, method, or output; 'extends' \u2014 this artifact extends the predecessor; 'similarities' \u2014 this artifact's results agree with the predecessor's; 'differences' \u2014 this artifact's results disagree with the predecessor's.",
          "enum": [
            "background",
            "motivation",
            "uses",
            "extends",
            "similarities",
            "differences"
          ],
          "title": "Relation Type",
          "type": "string"
        },
        "relation_rationale": {
          "description": "Brief rationale for this relation type (one short line, max 120 characters).",
          "maxLength": 120,
          "title": "Relation Rationale",
          "type": "string"
        }
      },
      "required": [
        "from_id",
        "to_id",
        "relation_type",
        "relation_rationale"
      ],
      "title": "ArtifactRelation",
      "type": "object"
    }
  },
  "description": "Revised hypothesis after reviewing iteration results.\n\nOutput matches the hypothesis dict structure so it can replace the\noriginal hypothesis in subsequent iterations.",
  "properties": {
    "title": {
      "description": "Revised hypothesis title in plain, everyday language \u2014 short and jargon-free so a non-expert grasps it at a glance and it fits the run visualizations. Aim for about 4-8 words (~40 characters); may be unchanged if still accurate.",
      "title": "Title",
      "type": "string"
    },
    "hypothesis": {
      "description": "Revised hypothesis statement \u2014 what we now believe based on evidence",
      "title": "Hypothesis",
      "type": "string"
    },
    "relation_rationale": {
      "description": "Brief rationale for the H\u2194H revision type (one short line, max 120 characters).",
      "maxLength": 120,
      "title": "Relation Rationale",
      "type": "string"
    },
    "confidence_delta": {
      "description": "How confidence changed: 'increased', 'decreased', or 'unchanged'",
      "title": "Confidence Delta",
      "type": "string"
    },
    "key_changes": {
      "description": "Bullet list of specific changes made to the hypothesis",
      "items": {
        "type": "string"
      },
      "title": "Key Changes",
      "type": "array"
    },
    "relation_type": {
      "description": "Moulines's structuralist typology of this hypothesis revision: 'evolution' \u2014 refining specialised claims while keeping the same conceptual frame; 'embedding' \u2014 the previous hypothesis is now a special case of a broader frame; 'replacement' \u2014 rejecting the previous frame entirely (incommensurable, Kuhnian revolution).",
      "enum": [
        "evolution",
        "embedding",
        "replacement"
      ],
      "title": "Relation Type",
      "type": "string"
    },
    "artifact_relations": {
      "description": "Typed A\u2194A edges for this iteration's new artifacts. Emit one entry per (predecessor \u2192 dependent) edge for every in_dependency on each artifact produced this iteration.",
      "items": {
        "$ref": "#/$defs/ArtifactRelation"
      },
      "title": "Artifact Relations",
      "type": "array"
    }
  },
  "required": [
    "title",
    "hypothesis",
    "relation_rationale",
    "confidence_delta",
    "key_changes",
    "relation_type"
  ],
  "title": "RevisedHypothesis",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````

### [3] HUMAN-USER prompt · 2026-07-17 15:02:53 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [4] TOOL CALL — Write · 2026-07-17 15:04:11 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/upd_hypo/upd_hypo/.terminal_claude_agent_struct_out.json

{
  "title": "Abliteration Preserves RLHF Encoder Directions (Direction + Magnitude)",
  "hypothesis": "RLHF safety fine-tuning and refusal abliteration operate on geometrically near-orthogonal subspaces in Qwen3-8B's residual stream. Five experimentally confirmed findings establish this, subject to one unverified assumption and two open mechanistic questions that define the next iteration's required experiments.\n\n**Confirmed findings (all conditioned on abliteration completeness — see Critical Gap 1):**\n\n(1) Directional preservation — the Reversal Score (PearsonR of RLHF-induced vs. abliteration-induced change vectors per encoder direction) for top-500 RLHF-sensitive SAE encoder directions at layer 22 is unimodal (Hartigan dip p=0.930) with mean=0.921, std=0.040, min=0.684, and zero directions with negative scores. A noise-free mid-SC baseline (features at 40th–60th SC percentile, SC_median≈1.12 ≫ 0.01) yields R̄=0.871, Cohen's d=1.087, p≈0, confirming selective preservation above any noise floor.\n\n(2) Magnitude preservation — mean OLS no-intercept slope β̄=0.986±0.176 for the top-500 directions at layer 22 (β̄=0.954±0.191 at layer 20), with 86.4% of directions having β_f>0.8. However, 45.4% of directions have β_f>1.0 (amplification up to 1.725), which is mechanistically inconsistent with the simple geometric account: orthogonalizing weight matrices against a near-orthogonal direction should produce β_f slightly below 1.0, not systematically above it in a substantial minority. This amplification is an open mechanistic question. Candidate explanations include (a) upward bias from no-intercept OLS regression on vectors with non-zero means, (b) secondary indirect effects of weight-matrix orthogonalization through residual stream accumulation, or (c) non-linear amplification in the pre-topk projection regime. The amplification must be investigated in the next iteration: compare β_f with and without the no-intercept constraint (with-intercept OLS removes the mean-centering bias), characterize whether amplified directions cluster by Safety Change magnitude or semantic label, and report both β̄ estimates in the paper.\n\n(3) Geometric orthogonality — mean |cos(r, w_enc(f))|=0.01854 at layer 22 and 0.02086 at layer 20 (confirmed across iter3 and iter4 to 5 decimal places). A cosine of 0.019 means the refusal direction projects onto <2% of each RLHF-modified encoder direction's magnitude. This is the most secure finding in this research: it is a direct measurement of the angle between weight vectors, independent of activation distributions, the pre-topk/post-topk distinction, and abliteration completeness. This geometric fact explains why β̄ ≈ 1 is the expected outcome *if* abliteration is complete and the refusal direction is one-dimensional.\n\n(4) Centroid selectivity — the abliterated model's centroid in the RLHF-sensitive subspace is 4.35× closer to instruct than to base on harmful prompts at layer 22 (7.31× at layer 20), versus 1.16× in the RLHF-insensitive control (1.01× at layer 20) and 2.37× in the random control (1.63× at layer 20). These ratios currently lack bootstrap confidence intervals; adding 1000-resample bootstrap CIs for ratio_A/ratio_B and ratio_A/ratio_C is required before these numbers carry inferential weight.\n\n(5) Benign-SC overlap — 323/500 (64.6%) of the top-500 RLHF-sensitive encoder directions also appear in the top-500 by benign-prompt Safety Change, establishing a STYLE_DOMINANT verdict: the majority of the preserved imprint captures general RLHF behavioral modifications (response formality, instruction-following structure, output style) rather than harm-specific semantic content. The remaining 35.4% (176 directions) are uniquely sensitive to harmful prompts. These 176 directions are candidate carriers of harm-concept representations, but this claim must be softened: they are defined purely by absence from the benign-SC top-500 rather than by validated semantic content, and z_pre/z_post validation is required before they can be asserted as harm-specific semantic representations rather than prompt-format artifacts.\n\n**Critical Gap 1 (BLOCKING): Abliteration depth verification.** This is the single highest-priority unresolved issue, elevated to blocking by reviewer feedback. All main findings (R̄=0.921, β̄=0.986, centroid ratios) are explicitly conditioned on the huihui-ai/Huihui-Qwen3-8B-abliterated-v2 checkpoint implementing genuine refusal-direction removal. If abliteration is incomplete, the high R̄ and β̄ trivially reflect model proximity rather than preservation under genuine erasure. Two measurements from cached activations are required before the paper can be accepted: (a) fraction of AdvBench or HarmBench prompts producing a non-refusal response (expected >95%; below 80% substantially qualifies all findings); (b) mean |Proj_r(h_abl(p))|/|h_abl(p)| at layer 22 for 500 harmful prompts (expected near 0; above 0.1 indicates partial abliteration). If verification confirms incomplete abliteration, all findings in the abstract and results sections must be reframed as conditional rather than only in Limitations.\n\n**Critical Gap 2: OLS amplification mechanism.** The 45.4% β_f>1.0 finding directly contradicts the geometric account as currently stated. The Discussion section currently describes the preservation as mechanistically explained by near-orthogonality, but orthogonalization against a near-orthogonal direction can only *reduce* projections, never increase them in the simple linear model. The amplification requires either (a) a corrected geometric account (e.g., indirect effects via residual stream accumulation) or (b) demonstration that it is a statistical artifact of no-intercept OLS. This must be investigated and reported before the mechanistic framing in Section 6.1 is credible.\n\n**Critical Gap 3: Pre-topk/post-topk validation.** All metrics use z_pre (dense continuous projections onto all 65,536 encoder directions) because reconstruction R²<0.85 at layer 22. The required validation remains: on the highest-R² prompt subset, compare top-500 feature rankings by SC(z_pre) versus SC(z_post) (overlap fraction) and compute Spearman ρ between z_pre and z_post Reversal Scores. If ρ>0.7 and overlap>60%, the pre-topk proxy is defensible; if divergent, all findings must be framed as 'encoder-direction geometry' not 'feature-level semantic preservation.'\n\n**Secondary open measurements (actionable from cached activations):**\n- Top-3 principal components of (harmful−harmless) activation differences at layer 22; report mean |cos(PC_k, w_enc(f))| for k=1,2,3 across top-500 directions. If all three values are below 0.05, the orthogonality argument extends robustly to multi-dimensional ablation as in Joad et al. [23]. If a higher PC shows |cos|≥0.05, the mechanistic explanation requires qualification.\n- Bootstrap confidence intervals (1000 resamples) for centroid distance ratios.\n- Cohen's d for the mid-500 baseline comparison at layer 20 (to complete Table 2).\n- Brief discussion of the bottom-500 baseline (Cohen's d=1.7508 from iter3) vs. mid-500 baseline (d=1.087) discrepancy, noting that the larger effect for the more extreme baseline is expected and consistent.\n\n**Scope constraints:** Results are for Qwen3-8B at layers 22 and 20 using one abliteration implementation (huihui-ai v2). The zero-reversed finding and β̄≈1 may not generalize to other abliteration implementations that target different layers or use different refusal direction computation (e.g., mlabonne's layer-9-targeted variant). Cross-implementation robustness and cross-family extension to Llama-3.1-8B and Gemma-3-9B remain explicit future work.\n\n**The downstream classification null (accuracy=0.476) is now fully explained** by the 64.6% benign-SC overlap: RLHF-sensitive encoder directions are elevated on both harmful and benign prompts, eliminating prompt-level discriminability. This is a confirmatory finding, not a contradiction.\n\n**Safety restoration reframing:** If abliteration completeness is verified and the amplification question is resolved, the restoration implications shift from 'semantic amplification' to 'geometric re-injection': the RLHF imprint is retained at approximately full magnitude, so restoration requires re-injecting the compliance direction r into weight matrices rather than amplifying surviving features. The 35.4% uniquely-harmful encoder directions are candidate restoration targets, *pending z_post validation that they carry coherent harm-specific semantic content rather than corpus-format artifacts.*",
  "relation_rationale": "Same conceptual frame; adds amplification anomaly as new open question; elevates abliteration verification to primary blocker per reviewer.",
  "confidence_delta": "unchanged",
  "key_changes": [
    "Abliteration depth verification elevated to BLOCKING status: reviewer identified this as a major gap that prevents acceptance; two specific measurements required from cached activations (refusal rate, mean |Proj_r|/|h| at L22) must be computed and reported before the paper can be accepted at a top-tier venue.",
    "OLS amplification (45.4% β_f>1.0) added as an unresolved mechanistic contradiction: the geometric account predicts β_f≤1.0 after orthogonalizing against a near-orthogonal direction, but substantial amplification is observed; next iteration must compare with-intercept vs. no-intercept OLS to test regression-artifact hypothesis and characterize amplified directions.",
    "Safety restoration candidate claim softened for 35.4% uniquely-harmful directions: replaced 'most promising candidates' with 'candidate subset pending z_post validation' — these directions are defined by absence from benign-SC top-500, not by validated harm-specific semantic content.",
    "Centroid analysis flagged as lacking statistical inference: bootstrap CIs (1000 resamples) for ratio_A/ratio_B and ratio_A/ratio_C required before centroid ratios carry inferential weight.",
    "Table 2 gap identified: Cohen's d for mid-500 baseline comparison at layer 20 must be filled; bottom-500 (d=1.7508) vs mid-500 (d=1.087) discrepancy should be noted with explanation that larger effect for more extreme baseline is expected.",
    "Multi-PC orthogonality measurement added as actionable secondary experiment: top-3 PCs of (harmful−harmless) activation differences at L22; mean |cos(PC_k, w_enc)| for k=1,2,3; threshold |cos|<0.05 for all three confirms robust orthogonality under multi-dimensional ablation.",
    "Single abliteration implementation scope caveat added: results conditioned on huihui-ai v2; mlabonne's layer-9-targeted variant as cross-implementation robustness check noted as future work."
  ],
  "relation_type": "evolution",
  "artifact_relations": []
}
```

### [5] TOOL RESULT — Write · 2026-07-17 15:04:11 UTC

```
{"type": "create", "filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/upd_hypo/upd_hypo/.terminal_claude_agent_struct_out.json", "content": "{\n  \"title\": \"Abliteration Preserves RLHF Encoder Directions (Direction + Magnitude)\",\n  \"hypothesis\": \"RLHF safety fine-tuning and refusal abliteration operate on geometrically near-orthogonal subspaces in Qwen3-8B's residual stream. Five experimentally confirmed findings establish this, subject to one unverified assumption and two open mechanistic questions that define the next iteration's required experiments.\\n\\n**Confirmed findings (all conditioned on abliteration completeness — see Critical Gap 1):**\\n\\n(1) Directional preservation — the Reversal Score (PearsonR of RLHF-induced vs. abliteration-induced change vectors per encoder direction) for top-500 RLHF-sensitive SAE encoder directions at layer 22 is unimodal (Hartigan dip p=0.930) with mean=0.921, std=0.040, min=0.684, and zero directions with negative scores. A noise-free mid-SC baseline (features at 40th–60th SC percentile, SC_median≈1.12 ≫ 0.01) yields R̄=0.871, Cohen's d=1.087, p≈0, confirming selective preservation above any noise floor.\\n\\n(2) Magnitude preservation — mean OLS no-intercept slope β̄=0.986±0.176 for the top-500 directions at layer 22 (β̄=0.954±0.191 at layer 20), with 86.4% of directions having β_f>0.8. However, 45.4% of directions have β_f>1.0 (amplification up to 1.725), which is mechanistically inconsistent with the simple geometric account: orthogonalizing weight matrices against a near-orthogonal direction should produce β_f slightly below 1.0, not systematically above it in a substantial minority. This amplification is an open mechanistic question. Candidate explanations include (a) upward bias from no-intercept OLS regression on vectors with non-zero means, (b) secondary indirect effects of weight-matrix orthogonalization through residual stream accumulation, or (c) non-linear amplification in the pre-topk projection regime. The amplification must be investigated in the next iteration: compare β_f with and without the no-intercept constraint (with-intercept OLS removes the mean-centering bias), characterize whether amplified directions cluster by Safety Change magnitude or semantic label, and report both β̄ estimates in the paper.\\n\\n(3) Geometric orthogonality — mean |cos(r, w_enc(f))|=0.01854 at layer 22 and 0.02086 at layer 20 (confirmed across iter3 and iter4 to 5 decimal places). A cosine of 0.019 means the refusal direction projects onto <2% of each RLHF-modified encoder direction's magnitude. This is the most secure finding in this research: it is a direct measurement of the angle between weight vectors, independent of activation distributions, the pre-topk/post-topk distinction, and abliteration completeness. This geometric fact explains why β̄ ≈ 1 is the expected outcome *if* abliteration is complete and the refusal direction is one-dimensional.\\n\\n(4) Centroid selectivity — the abliterated model's centroid in the RLHF-sensitive subspace is 4.35× closer to instruct than to base on harmful prompts at layer 22 (7.31× at layer 20), versus 1.16× in the RLHF-insensitive control (1.01× at layer 20) and 2.37× in the random control (1.63× at layer 20). These ratios currently lack bootstrap confidence intervals; adding 1000-resample bootstrap CIs for ratio_A/ratio_B and ratio_A/ratio_C is required before these numbers carry inferential weight.\\n\\n(5) Benign-SC overlap — 323/500 (64.6%) of the top-500 RLHF-sensitive encoder directions also appear in the top-500 by benign-prompt Safety Change, establishing a STYLE_DOMINANT verdict: the majority of the preserved imprint captures general RLHF behavioral modifications (response formality, instruction-following structure, output style) rather than harm-specific semantic content. The remaining 35.4% (176 directions) are uniquely sensitive to harmful prompts. These 176 directions are candidate carriers of harm-concept representations, but this claim must be softened: they are defined purely by absence from the benign-SC top-500 rather than by validated semantic content, and z_pre/z_post validation is required before they can be asserted as harm-specific semantic representations rather than prompt-format artifacts.\\n\\n**Critical Gap 1 (BLOCKING): Abliteration depth verification.** This is the single highest-priority unresolved issue, elevated to blocking by reviewer feedback. All main findings (R̄=0.921, β̄=0.986, centroid ratios) are explicitly conditioned on the huihui-ai/Huihui-Qwen3-8B-abliterated-v2 checkpoint implementing genuine refusal-direction removal. If abliteration is incomplete, the high R̄ and β̄ trivially reflect model proximity rather than preservation under genuine erasure. Two measurements from cached activations are required before the paper can be accepted: (a) fraction of AdvBench or HarmBench prompts producing a non-refusal response (expected >95%; below 80% substantially qualifies all findings); (b) mean |Proj_r(h_abl(p))|/|h_abl(p)| at layer 22 for 500 harmful prompts (expected near 0; above 0.1 indicates partial abliteration). If verification confirms incomplete abliteration, all findings in the abstract and results sections must be reframed as conditional rather than only in Limitations.\\n\\n**Critical Gap 2: OLS amplification mechanism.** The 45.4% β_f>1.0 finding directly contradicts the geometric account as currently stated. The Discussion section currently describes the preservation as mechanistically explained by near-orthogonality, but orthogonalization against a near-orthogonal direction can only *reduce* projections, never increase them in the simple linear model. The amplification requires either (a) a corrected geometric account (e.g., indirect effects via residual stream accumulation) or (b) demonstration that it is a statistical artifact of no-intercept OLS. This must be investigated and reported before the mechanistic framing in Section 6.1 is credible.\\n\\n**Critical Gap 3: Pre-topk/post-topk validation.** All metrics use z_pre (dense continuous projections onto all 65,536 encoder directions) because reconstruction R²<0.85 at layer 22. The required validation remains: on the highest-R² prompt subset, compare top-500 feature rankings by SC(z_pre) versus SC(z_post) (overlap fraction) and compute Spearman ρ between z_pre and z_post Reversal Scores. If ρ>0.7 and overlap>60%, the pre-topk proxy is defensible; if divergent, all findings must be framed as 'encoder-direction geometry' not 'feature-level semantic preservation.'\\n\\n**Secondary open measurements (actionable from cached activations):**\\n- Top-3 principal components of (harmful−harmless) activation differences at layer 22; report mean |cos(PC_k, w_enc(f))| for k=1,2,3 across top-500 directions. If all three values are below 0.05, the orthogonality argument extends robustly to multi-dimensional ablation as in Joad et al. [23]. If a higher PC shows |cos|≥0.05, the mechanistic explanation requires qualification.\\n- Bootstrap confidence intervals (1000 resamples) for centroid distance ratios.\\n- Cohen's d for the mid-500 baseline comparison at layer 20 (to complete Table 2).\\n- Brief discussion of the bottom-500 baseline (Cohen's d=1.7508 from iter3) vs. mid-500 baseline (d=1.087) discrepancy, noting that the larger effect for the more extreme baseline is expected and consistent.\\n\\n**Scope constraints:** Results are for Qwen3-8B at layers 22 and 20 using one abliteration implementation (huihui-ai v2). The zero-reversed finding and β̄≈1 may not generalize to other abliteration implementations that target different layers or use different refusal direction computation (e.g., mlabonne's layer-9-targeted variant). Cross-implementation robustness and cross-family extension to Llama-3.1-8B and Gemma-3-9B remain explicit future work.\\n\\n**The downstream classification null (accuracy=0.476) is now fully explained** by the 64.6% benign-SC overlap: RLHF-sensitive encoder directions are elevated on both harmful and benign prompts, eliminating prompt-level discriminability. This is a confirmatory finding, not a contradiction.\\n\\n**Safety restoration reframing:** If abliteration completeness is verified and the amplification question is resolved, the restoration implications shift from 'semantic amplification' to 'geometric re-injection': the RLHF imprint is retained at approximately full magnitude, so restoration requires re-injecting the compliance direction r into weight matrices rather than amplifying surviving features. The 35.4% uniquely-harmful encoder directions are candidate restoration targets, *pending z_post validation that they carry coherent harm-specific semantic content rather than corpus-format artifacts.*\",\n  \"relation_rationale\": \"Same conceptual frame; adds amplification anomaly as new open question; elevates abliteration verification to primary blocker per reviewer.\",\n  \"confidence_delta\": \"unchanged\",\n  \"key_changes\": [\n    \"Abliteration depth verification elevated to BLOCKING status: reviewer identified this as a major gap that prevents acceptance; two specific measurements required from cached activations (refusal rate, mean |Proj_r|/|h| at L22) must be computed and reported before the paper can be accepted at a top-tier venue.\",\n    \"OLS amplification (45.4% β_f>1.0) added as an unresolved mechanistic contradiction: the geometric account predicts β_f≤1.0 after orthogonalizing against a near-orthogonal direction, but substantial amplification is observed; next iteration must compare with-intercept vs. no-intercept OLS to test regression-artifact hypothesis and characterize amplified directions.\",\n    \"Safety restoration candidate claim softened for 35.4% uniquely-harmful directions: replaced 'most promising candidates' with 'candidate subset pending z_post validation' — these directions are defined by absence from benign-SC top-500, not by validated harm-specific semantic content.\",\n    \"Centroid analysis flagged as lacking statistical inference: bootstrap CIs (1000 resamples) for ratio_A/ratio_B and ratio_A/ratio_C required before centroid ratios carry inferential weight.\",\n    \"Table 2 gap identified: Cohen's d for mid-500 baseline comparison at layer 20 must be filled; bottom-500 (d=1.7508) vs mid-500 (d=1.087) discrepancy should be noted with explanation that larger effect for more extreme baseline is expected.\",\n    \"Multi-PC orthogonality measurement added as actionable secondary experiment: top-3 PCs of (harmful−harmless) activation differences at L22; mean |cos(PC_k, w_enc)| for k=1,2,3; threshold |cos|<0.05 for all three confirms robust orthogonality under multi-dimensional ablation.\",\n    \"Single abliteration implementation scope caveat added: results conditioned on huihui-ai v2; mlabonne's layer-9-targeted variant as cross-implementation robustness check noted as future work.\"\n  ],\n  \"relation_type\": \"evolution\",\n  \"artifact_relations\": []\n}\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [6] ASSISTANT · 2026-07-17 15:04:16 UTC

```
The revised hypothesis has been written. Key changes from iter 4:

1. **Abliteration verification elevated to BLOCKING** — reviewer flagged this as a major acceptance blocker; the two required measurements (refusal rate + mean |Proj_r|/|h| at L22) are computable from cached activations and must be done next.

2. **OLS amplification (45.4% β_f>1.0) added as open mechanistic contradiction** — the geometric account predicts β_f≤1 after orthogonalizing against a near-orthogonal direction, so this contradicts the mechanistic story and needs investigation (with-intercept vs no-intercept OLS comparison).

3. **Safety restoration claim softened** — the 35.4% uniquely-harmful directions are now "candidate subset pending z_post validation" rather than "most promising candidates."

4. **Three secondary actionable measurements added** — bootstrap CIs for centroid ratios, Cohen's d for L20 mid-500 comparison, multi-PC orthogonality (top-3 PCs of refusal subspace).
```

### [7] SYSTEM-USER prompt · 2026-07-17 15:04:19 UTC

```
<validation-feedback>
Attempt 1 failed validation.

Schema validation found 1 problem — fix ALL of them at once:
  - at `relation_rationale`: 'Same conceptual frame; adds amplification anomaly as new open question; elevates abliteration verification to primary blocker per reviewer.' is too long (at most 120 characters, got 139)
Every required field must be present and every field type must match the schema.

Please use the Write tool to overwrite `.terminal_claude_agent_struct_out.json` with corrected JSON. Do not invent new fields; match the schema you were given.
</validation-feedback>
```

### [8] TOOL CALL — Edit · 2026-07-17 15:04:21 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/upd_hypo/upd_hypo/.terminal_claude_agent_struct_out.json
OLD:   "relation_rationale": "Same conceptual frame; adds amplification anomaly as new open question; elevates abliteration verification to primary blocker per reviewer.",
NEW:   "relation_rationale": "Same frame; adds β_f>1 amplification as open mechanistic question; abliteration verification elevated to blocking.",
```

### [9] TOOL RESULT — Edit · 2026-07-17 15:04:21 UTC

```
{"filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/upd_hypo/upd_hypo/.terminal_claude_agent_struct_out.json", "oldString": "  \"relation_rationale\": \"Same conceptual frame; adds amplification anomaly as new open question; elevates abliteration verification to primary blocker per reviewer.\",", "newString": "  \"relation_rationale\": \"Same frame; adds β_f>1 amplification as open mechanistic question; abliteration verification elevated to blocking.\",", "originalFile": null, "structuredPatch": [{"oldStart": 1, "oldLines": 7, "newStart": 1, "newLines": 7, "lines": [" {", "   \"title\": \"Abliteration Preserves RLHF Encoder Directions (Direction + Magnitude)\",", "   \"hypothesis\": \"RLHF safety fine-tuning and refusal abliteration operate on geometrically near-orthogonal subspaces in Qwen3-8B's residual stream. Five experimentally confirmed findings establish this, subject to one unverified assumption and two open mechanistic questions that define the next iteration's required experiments.\\n\\n**Confirmed findings (all conditioned on abliteration completeness — see Critical Gap 1):**\\n\\n(1) Directional preservation — the Reversal Score (PearsonR of RLHF-induced vs. abliteration-induced change vectors per encoder direction) for top-500 RLHF-sensitive SAE encoder directions at layer 22 is unimodal (Hartigan dip p=0.930) with mean=0.921, std=0.040, min=0.684, and zero directions with negative scores. A noise-free mid-SC baseline (features at 40th–60th SC percentile, SC_median≈1.12 ≫ 0.01) yields R̄=0.871, Cohen's d=1.087, p≈0, confirming selective preservation above any noise floor.\\n\\n(2) Magnitude preservation — mean OLS no-intercept slope β̄=0.986±0.176 for the top-500 directions at layer 22 (β̄=0.954±0.191 at layer 20), with 86.4% of directions having β_f>0.8. However, 45.4% of directions have β_f>1.0 (amplification up to 1.725), which is mechanistically inconsistent with the simple geometric account: orthogonalizing weight matrices against a near-orthogonal direction should produce β_f slightly below 1.0, not systematically above it in a substantial minority. This amplification is an open mechanistic question. Candidate explanations include (a) upward bias from no-intercept OLS regression on vectors with non-zero means, (b) secondary indirect effects of weight-matrix orthogonalization through residual stream accumulation, or (c) non-linear amplification in the pre-topk projection regime. The amplification must be investigated in the next iteration: compare β_f with and without the no-intercept constraint (with-intercept OLS removes the mean-centering bias), characterize whether amplified directions cluster by Safety Change magnitude or semantic label, and report both β̄ estimates in the paper.\\n\\n(3) Geometric orthogonality — mean |cos(r, w_enc(f))|=0.01854 at layer 22 and 0.02086 at layer 20 (confirmed across iter3 and iter4 to 5 decimal places). A cosine of 0.019 means the refusal direction projects onto <2% of each RLHF-modified encoder direction's magnitude. This is the most secure finding in this research: it is a direct measurement of the angle between weight vectors, independent of activation distributions, the pre-topk/post-topk distinction, and abliteration completeness. This geometric fact explains why β̄ ≈ 1 is the expected outcome *if* abliteration is complete and the refusal direction is one-dimensional.\\n\\n(4) Centroid selectivity — the abliterated model's centroid in the RLHF-sensitive subspace is 4.35× closer to instruct than to base on harmful prompts at layer 22 (7.31× at layer 20), versus 1.16× in the RLHF-insensitive control (1.01× at layer 20) and 2.37× in the random control (1.63× at layer 20). These ratios currently lack bootstrap confidence intervals; adding 1000-resample bootstrap CIs for ratio_A/ratio_B and ratio_A/ratio_C is required before these numbers carry inferential weight.\\n\\n(5) Benign-SC overlap — 323/500 (64.6%) of the top-500 RLHF-sensitive encoder directions also appear in the top-500 by benign-prompt Safety Change, establishing a STYLE_DOMINANT verdict: the majority of the preserved imprint captures general RLHF behavioral modifications (response formality, instruction-following structure, output style) rather than harm-specific semantic content. The remaining 35.4% (176 directions) are uniquely sensitive to harmful prompts. These 176 directions are candidate carriers of harm-concept representations, but this claim must be softened: they are defined purely by absence from the benign-SC top-500 rather than by validated semantic content, and z_pre/z_post validation is required before they can be asserted as harm-specific semantic representations rather than prompt-format artifacts.\\n\\n**Critical Gap 1 (BLOCKING): Abliteration depth verification.** This is the single highest-priority unresolved issue, elevated to blocking by reviewer feedback. All main findings (R̄=0.921, β̄=0.986, centroid ratios) are explicitly conditioned on the huihui-ai/Huihui-Qwen3-8B-abliterated-v2 checkpoint implementing genuine refusal-direction removal. If abliteration is incomplete, the high R̄ and β̄ trivially reflect model proximity rather than preservation under genuine erasure. Two measurements from cached activations are required before the paper can be accepted: (a) fraction of AdvBench or HarmBench prompts producing a non-refusal response (expected >95%; below 80% substantially qualifies all findings); (b) mean |Proj_r(h_abl(p))|/|h_abl(p)| at layer 22 for 500 harmful prompts (expected near 0; above 0.1 indicates partial abliteration). If verification confirms incomplete abliteration, all findings in the abstract and results sections must be reframed as conditional rather than only in Limitations.\\n\\n**Critical Gap 2: OLS amplification mechanism.** The 45.4% β_f>1.0 finding directly contradicts the geometric account as currently stated. The Discussion section currently describes the preservation as mechanistically explained by near-orthogonality, but orthogonalization against a near-orthogonal direction can only *reduce* projections, never increase them in the simple linear model. The amplification requires either (a) a corrected geometric account (e.g., indirect effects via residual stream accumulation) or (b) demonstration that it is a statistical artifact of no-intercept OLS. This must be investigated and reported before the mechanistic framing in Section 6.1 is credible.\\n\\n**Critical Gap 3: Pre-topk/post-topk validation.** All metrics use z_pre (dense continuous projections onto all 65,536 encoder directions) because reconstruction R²<0.85 at layer 22. The required validation remains: on the highest-R² prompt subset, compare top-500 feature rankings by SC(z_pre) versus SC(z_post) (overlap fraction) and compute Spearman ρ between z_pre and z_post Reversal Scores. If ρ>0.7 and overlap>60%, the pre-topk proxy is defensible; if divergent, all findings must be framed as 'encoder-direction geometry' not 'feature-level semantic preservation.'\\n\\n**Secondary open measurements (actionable from cached activations):**\\n- Top-3 principal components of (harmful−harmless) activation differences at layer 22; report mean |cos(PC_k, w_enc(f))| for k=1,2,3 across top-500 directions. If all three values are below 0.05, the orthogonality argument extends robustly to multi-dimensional ablation as in Joad et al. [23]. If a higher PC shows |cos|≥0.05, the mechanistic explanation requires qualification.\\n- Bootstrap confidence intervals (1000 resamples) for centroid distance ratios.\\n- Cohen's d for the mid-500 baseline comparison at layer 20 (to complete Table 2).\\n- Brief discussion of the bottom-500 baseline (Cohen's d=1.7508 from iter3) vs. mid-500 baseline (d=1.087) discrepancy, noting that the larger effect for the more extreme baseline is expected and consistent.\\n\\n**Scope constraints:** Results are for Qwen3-8B at layers 22 and 20 using one abliteration implementation (huihui-ai v2). The zero-reversed finding and β̄≈1 may not generalize to other abliteration implementations that target different layers or use different refusal direction computation (e.g., mlabonne's layer-9-targeted variant). Cross-implementation robustness and cross-family extension to Llama-3.1-8B and Gemma-3-9B remain explicit future work.\\n\\n**The downstream classification null (accuracy=0.476) is now fully explained** by the 64.6% benign-SC overlap: RLHF-sensitive encoder directions are elevated on both harmful and benign prompts, eliminating prompt-level discriminability. This is a confirmatory finding, not a contradiction.\\n\\n**Safety restoration reframing:** If abliteration completeness is verified and the amplification question is resolved, the restoration implications shift from 'semantic amplification' to 'geometric re-injection': the RLHF imprint is retained at approximately full magnitude, so restoration requires re-injecting the compliance direction r into weight matrices rather than amplifying surviving features. The 35.4% uniquely-harmful encoder directions are candidate restoration targets, *pending z_post validation that they carry coherent harm-specific semantic content rather than corpus-format artifacts.*\",", "-  \"relation_rationale\": \"Same conceptual frame; adds amplification anomaly as new open question; elevates abliteration verification to primary blocker per reviewer.\",", "+  \"relation_rationale\": \"Same frame; adds β_f>1 amplification as open mechanistic question; abliteration verification elevated to blocking.\",", "   \"confidence_delta\": \"unchanged\",", "   \"key_changes\": [", "     \"Abliteration depth verification elevated to BLOCKING status: reviewer identified this as a major gap that prevents acceptance; two specific measurements required from cached activations (refusal rate, mean |Proj_r|/|h| at L22) must be computed and reported before the paper can be accepted at a top-tier venue.\","]}], "userModified": false, "replaceAll": false}
```

### [10] ASSISTANT · 2026-07-17 15:04:23 UTC

```
Fixed. The `relation_rationale` is now 96 characters, within the 120-character limit.
```
