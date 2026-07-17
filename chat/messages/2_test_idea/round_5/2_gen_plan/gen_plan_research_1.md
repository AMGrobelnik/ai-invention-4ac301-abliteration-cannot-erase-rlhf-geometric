# gen_plan_research_1 — test_idea

> Phase: `invention_loop` · round 5 · `gen_plan`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim transcript of this agent task — every system/user prompt, assistant response, thinking block, tool call and tool result — in the order they occurred. Nothing truncated.

## Task: `gen_plan_research_1` (terminal_claude_agent, claude-sonnet-4-6)

### [1] CONFIG · 2026-07-17 14:34:49 UTC

```
model: claude-sonnet-4-6 | effort: medium | permission: bypassPermissions | cwd: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/gen_plan/gen_plan_research_1
```

### [2] SYSTEM-USER prompt · 2026-07-17 14:34:55 UTC

````
<ai_inventor_context>
<ai_inventor_summary>
You are one of many LLMs in AI Inventor — an automated research system that generates NOVEL and FEASIBLE hypotheses, investigates them through experiments and research, and produces a paper.

Your output feeds other LLMs downstream. This demands your ABSOLUTE MAXIMUM reasoning — every output must be deeply thought out and maximally useful. Surface-level responses waste downstream computation.
</ai_inventor_summary>

<your_role>
YOU ARE: A plan generator (Step 3.2: GEN_PLAN in the invention loop)

You received the hypothesis, an artifact direction to elaborate, and dependency artifacts relevant to the plan.
Your job: elaborate this direction into a detailed, actionable plan for the executor agent.

Specific, actionable plan → valuable artifact. Vague plan → wasted execution.
</your_role>
</ai_inventor_context>

<artifact_type_info>
You are expanding an artifact direction of type: RESEARCH

RESEARCH
Web research to answer key questions — like a researcher making decisions.
Runtime: LLM Agent, no code execution.
Tools: the aii-web-tools skill (web search, page fetch, regex grep over full page/PDF text).
Capabilities: Find, synthesize, and compare information across sources; survey SOTA and best practices.
Deps: REQUIRED none | OPTIONAL other RESEARCH to build on prior findings
</artifact_type_info>

<available_resources>
<software_constraints>
- Python only implementation
- Python standard library and all popular PyPI packages available (numpy, pandas, scikit-learn, scipy, matplotlib, requests, etc.)
- Local parallelism encouraged: multiprocessing, asyncio, threading — see aii-parallel-computing skill
- LLM API calls must go through OpenRouter only (no direct OpenAI, Anthropic, etc.)
- **HARD LIMIT**: Maximum $10 USD total spend on LLM API calls (OpenRouter). Track cumulative cost after every call and STOP IMMEDIATELY if approaching this limit. Never exceed this budget under any circumstances.
</software_constraints>
</available_resources>

<time_budget>

The research executor has 3h total (including writing code, debugging, testing, and fixing errors).

</time_budget>

<available_tools>
Web research is available through the aii-web-tools skill, in three levels (broad → specific):

1. web search — Returns titles, URLs, snippets. Use first to discover and scan the landscape.
2. web fetch — Reads a page and returns its content as markdown (HTML or PDF). Use to understand a source. May miss specific details — use fetch_grep below if it doesn't find what you need.
3. fetch_grep — Regex search over a page/PDF's full text. Returns exact matching sections with context. Use for precise details, exact numbers, methodology, or PDFs.

Workflow: search → fetch (understand) → fetch_grep (extract specifics).
</available_tools>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<plan_guidelines>
You are expanding an artifact direction from the strategy into a detailed plan.
The artifact direction specifies what to do at a high level (type, objective, approach, dependencies).
Your job is to make it concrete and actionable as a detailed plan.
Use web research to look up technical details, verify feasibility, and find reference materials
that will make your plan more concrete and actionable for the executor.

GOOD PLANS:
- Make each component SPECIFIC and actionable (not vague platitudes)
- Consider both success AND failure scenarios
- Build on the approach in the artifact direction
- Add concrete details the executor needs

BAD PLANS:
- Vague hand-waving ("do research on X")
- Ignoring the approach in the artifact direction
- Missing critical details the executor needs
</plan_guidelines>

<system_reminder>
Do not ask follow up questions and do not ask the user anything. Execute all steps independently.
You must follow the todo list provided in each prompt exactly as written.
No placeholders, stubs, or incomplete code — all code must be complete and functional.
</system_reminder>

<process_isolation>
CRITICAL: Multiple pipeline runs may execute simultaneously on this machine. `ps aux | grep method.py` matches ALL runs, not just yours.
- NEVER kill processes by name (`killall`, `pkill -f`, `ps aux | grep ... | xargs kill`). This kills OTHER runs' processes.
- NEVER monitor processes by name (`ps aux | grep method.py`). You will see other runs' processes and get confused.
- ALWAYS use PID-based process management:
  Run: `uv run method.py & PID=$!` or `timeout <seconds> uv run method.py & PID=$!`
  Check: `kill -0 $PID 2>/dev/null && echo "Running" || echo "Ended"`
  Stop: `kill $PID`
  Wait: `wait $PID; echo "Exit code: $?"`
  Monitor: `tail -f logs/run.log & TAIL_PID=$!` then `kill $TAIL_PID` when done
</process_isolation>

<hypothesis>
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
</hypothesis>

<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for the methods, proper baselines, and evaluation this field demands.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>

<artifact_direction>
Make this direction concrete and actionable. Keep the same type and respect dependencies.

id: research_iter5_dir2
type: research
objective: >-
  Fetch and synthesize the three missing related papers cited by the reviewer (arXiv:2505.23556, arXiv:2604.01604, arXiv:2602.02132)
  and produce a structured related-work paragraph for each, explaining precisely how it relates to the current paper and what
  the paper must say to differentiate from or acknowledge it.
approach: |-
  Use the aii-web-tools skill to fetch each paper from arXiv. For each paper, retrieve the abstract, key findings, and methodology. Then synthesize the comparison.

  **Paper 1 — arXiv:2505.23556** ('Understanding Refusal in Language Models with Sparse Autoencoders'). Fetch abstract and methods. Key question: does this paper find that pretraining-data SAEs fail to encode refusal features in chat-tuned models? If so, explain why z_pre projections onto the Qwen-Scope base-SAE are still informative for RLHF-modification tracking (the paper uses z_pre to detect *what RLHF modified*, not to identify refusal-specific features — a distinction this paper clarifies). Draft a related-work paragraph contrasting approaches.

  **Paper 2 — arXiv:2604.01604** ('CRaFT: Circuit-Guided Refusal Feature Selection via Cross-Layer Transcoders'). Fetch abstract and methods. Key question: does CRaFT use SAE features to analyze refusal circuitry? Contrast: CRaFT uses cross-layer transcoders for circuit-level attribution of which features cause refusal; our frozen-base-SAE approach measures which features survive abliteration without requiring circuit tracing. Both are feature-level but address different questions: CRaFT asks 'which features implement refusal?', we ask 'which features survive erasure of refusal?'. Draft differentiation paragraph.

  **Paper 3 — arXiv:2602.02132** ('There Is More to Refusal in Large Language Models than a Single Direction'). Fetch abstract. Key question: does this establish refusal as a multi-dimensional subspace across behavioral categories? This directly bears on the mechanistic orthogonality argument (|cos|=0.0185 for single r). Draft paragraph that: (a) acknowledges this work challenges the single-direction framing; (b) notes the experiment's Block 3 (top-3 PCs of refusal subspace) directly addresses this by measuring orthogonality to the full top-3 PC subspace; (c) frames the single-vector r as a practical proxy used in standard abliteration, not a claim about refusal's true dimensionality.

  **Output format:** research_out.json with fields: answer (full synthesis), sources (list of {url, title, key_finding}), three_paper_summaries (list of {arxiv_id, title, key_finding, relation_to_paper, draft_paragraph}), research_report.md with formatted related-work paragraphs ready to paste into the paper.
depends_on:
- id: art_pmAcY0CNazP9
  label: prior research
  relation_type:
  relation_rationale:
</artifact_direction>

<dependencies>
Completed artifacts this artifact can use during execution.

--- Dependency 1 ---
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
out_dependency_files:
  file_list:
  - research_out.json
</dependencies>

<instructions>
YOUR ROLE: Write a detailed PLAN for the artifact. A separate executor agent runs the actual artifact later.

You are a PLANNER, not an executor. Your output is a plan that tells the executor what to do and how.
Do NOT execute the artifact itself — a separate agent handles that. Your job is to plan it so well that the executor can follow your plan step by step.

You CAN and SHOULD: search the web, read papers, and explore library docs to make your plan concrete.
You CANNOT run shell commands or scripts — code execution is disabled. Research via web tools only.

Do NOT do the executor's job: don't download datasets, don't implement code, don't run experiments, don't write proofs, don't compute evaluations.

<artifact_executor_scope>
IMPORTANT: Each artifact executor has a focused prompt that guides it to do ONE thing well. It will NOT perform tasks outside its scope — assigning the wrong work to the wrong artifact type wastes an iteration. Match the task to the right executor.

RESEARCH executor scope:
  Output: research_out.json with {answer, sources, follow_up_questions} + research_report.md
  DOES: Web research — search, read, synthesize information from papers/docs/APIs into a structured report
  DOES NOT: Run code, download files, execute scripts, compute anything — no shell/Python access
  Use for literature surveys, API documentation, technical specifications — pure information gathering
</artifact_executor_scope>

<artifact_planning_rules>
RESEARCH: Plan early — findings guide dataset selection, experiment design, and methodology.
</artifact_planning_rules>

<compute_profiles>
Choose the compute profile this artifact needs for execution.
Available profiles for research artifacts:
  - cpu_light: 4 vCPUs, 16GB RAM — proofs, research, lightweight tasks (fallback: memory-optimized CPUs first (cpu3m → cpu5m), then GPU hosts last-ditch)

Set runpod_compute_profile to one of these exact tier names.
</compute_profiles>
GOOD PLANS: specific, actionable, consider failure scenarios, build on the suggested approach.
BAD PLANS: vague hand-waving, ignoring the suggested approach, missing critical executor details.
</instructions><user_data>
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
  "description": "Plan for a RESEARCH artifact.",
  "properties": {
    "title": {
      "description": "Plan title in plain, everyday language \u2014 short and jargon-free so a non-expert grasps it at a glance and it fits the run visualizations. Aim for about 4-8 words (~40 characters).",
      "title": "Title",
      "type": "string"
    },
    "summary": {
      "default": "",
      "description": "Brief summary",
      "title": "Summary",
      "type": "string"
    },
    "runpod_compute_profile": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": "cpu_light",
      "description": "Compute tier for execution \u2014 pick from the available profiles list (e.g., 'gpu', 'cpu_heavy', 'cpu_light'). Only used in RunPod mode.",
      "title": "Runpod Compute Profile"
    },
    "question": {
      "default": "",
      "description": "The specific research question to investigate",
      "title": "Question",
      "type": "string"
    },
    "research_plan": {
      "description": "Step-by-step plan for web research to gather this research",
      "title": "Research Plan",
      "type": "string"
    },
    "explanation": {
      "description": "Why this research matters and what question it answers",
      "title": "Explanation",
      "type": "string"
    }
  },
  "required": [
    "title",
    "research_plan",
    "explanation"
  ],
  "title": "ResearchPlan",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````

### [3] HUMAN-USER prompt · 2026-07-17 14:34:55 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [4] TOOL CALL — Skill · 2026-07-17 14:35:03 UTC

```
aii-web-tools
```

### [5] TOOL RESULT — Skill · 2026-07-17 14:35:03 UTC

```
{"success": true, "commandName": "aii-web-tools"}
```

### [6] ASSISTANT · 2026-07-17 14:35:05 UTC

```
I'll fetch the three papers from arXiv in parallel to gather the details needed for the plan.
```

### [7] TOOL CALL — ToolSearch · 2026-07-17 14:35:05 UTC

```
{'query': 'select:WebSearch,WebFetch'}
```

### [8] TOOL RESULT — ToolSearch · 2026-07-17 14:35:05 UTC

```
{"matches": ["WebSearch", "WebFetch"], "query": "select:WebSearch,WebFetch", "total_deferred_tools": 21}
```

### [9] TOOL CALL — WebFetch · 2026-07-17 14:35:21 UTC

```
URL: https://arxiv.org/abs/2505.23556
Prompt: Extract the full title, authors, abstract, key methodology, and main findings of this paper. Focus on: does it use SAEs to analyze refusal? Does it find pretraining-data SAEs fail to encode refusal features in chat-tuned models? What are the key contributions vs limitations?
```

### [10] TOOL CALL — WebFetch · 2026-07-17 14:35:21 UTC

```
URL: https://arxiv.org/abs/2604.01604
Prompt: Extract the full title, authors, abstract, key methodology, and main findings. Focus on: does it use SAE features and transcoders to analyze refusal circuitry? What exactly does CRaFT do? How does it select refusal features? What are the main claims vs limitations?
```

### [11] TOOL CALL — WebFetch · 2026-07-17 14:35:21 UTC

```
URL: https://arxiv.org/abs/2602.02132
Prompt: Extract the full title, authors, abstract, key methodology, and main findings. Focus on: does it establish refusal as multi-dimensional? What evidence shows refusal spans multiple directions? What models were studied? What are implications for single-direction abliteration methods?
```

### [12] TOOL RESULT — WebFetch · 2026-07-17 14:35:21 UTC

```
{"bytes": 41510, "code": 200, "codeText": "OK", "result": "# Paper Summary: Understanding Refusal in Language Models with Sparse Autoencoders\n\n**Full Title:** Understanding Refusal in Language Models with Sparse Autoencoders\n\n**Authors:** Wei Jie Yeo, Nirmalendu Prakash, Clement Neo, Roy Ka-Wei Lee, Erik Cambria, Ranjan Satapathy\n\n## Abstract\n\nThe researchers investigate how safety mechanisms function internally in aligned LLMs by employing sparse autoencoders (SAEs) to identify latent features that causally influence refusal behaviors. They apply this approach to two open-source chat models, intervening on refusal-related features and validating impacts across harmful datasets.\n\n## Key Methodology\n\nThe study uses SAEs as mechanistic tools to:\n- Identify latent features mediating refusal behaviors\n- Conduct fine-grained activation-level inspection\n- Examine upstream-downstream latent relationships\n- Understand how adversarial jailbreaking techniques work\n- Test generalization of refusal features for linear probes on out-of-distribution samples\n\n## Main Findings & Contributions\n\nThe paper demonstrates that SAEs successfully extract refusal-related features from instruction-tuned models. The authors establish that these features have measurable causal effects on model behavior and can improve classifier robustness against adversarial inputs. They provide reproducible research through \"open source\" code availability.\n\n## Limitations & Gaps\n\nThe abstract does not explicitly address whether pretraining-data SAEs fail to encode refusal features in chat-tuned models, leaving this comparative question unresolved from the provided text.", "durationMs": 6854, "url": "https://arxiv.org/abs/2505.23556"}
```

### [13] TOOL RESULT — WebFetch · 2026-07-17 14:35:23 UTC

```
{"bytes": 42565, "code": 200, "codeText": "OK", "result": "# CRaFT: Circuit-Guided Refusal Feature Selection\n\n## Full Title\n\"CRaFT: Circuit-Guided Refusal Feature Selection via Cross-Layer Transcoders\"\n\n## Authors\nSu-Hyeon Kim, Hyundong Jin, Yejin Lee, Yo-Sub Han\n\n## Abstract Summary\nThe paper addresses mechanistic understanding of LLM refusal behavior. It notes that current approaches relying solely on activation strength capture superficial patterns rather than causal mechanisms. The authors propose CRaFT to identify critical refusal features by measuring inter-feature relationships.\n\n## Key Methodology\n\n**SAE/Transcoder Use:** Yes. CRaFT leverages \"cross-layer transcoders to map the model's internal computations into a sparse feature circuit graph.\" This represents a shift from treating features as isolated signals to analyzing their relational dynamics.\n\n**How CRaFT Selects Features:**\n- Maps internal computations into a sparse feature circuit graph\n- Quantifies edges representing inter-feature influences\n- Measures contributions to final output logits\n- Aggregates effects propagating along paths to refusal decision\n- Ranks features by influence on refusal behavior\n\n## Main Claims\nThe framework claims to identify features that \"directly govern the refusal decision\" rather than superficial correlates. Experimental results allegedly show improvements from 6.7% to 57.4% across jailbreak benchmarks, generating \"more specific harmful completions\" compared to prior methods.\n\n## Limitations/Tensions\nThe abstract does not explicitly detail limitations. The focus on generating \"more specific harmful completions\" raises questions about dual-use implications that the paper presumably addresses in full text.", "durationMs": 7465, "url": "https://arxiv.org/abs/2604.01604"}
```

### [14] TOOL RESULT — WebFetch · 2026-07-17 14:35:23 UTC

```
{"bytes": 40543, "code": 200, "codeText": "OK", "result": "# Analysis of arXiv:2602.02132\n\n## Full Title\n\"There Is More to Refusal in Large Language Models than a Single Direction\"\n\n## Authors\nFaaiz Joad, Majd Hawasly, Sabri Boughorbel, Nadir Durrani, Husrev Taha Sencar\n\n## Abstract Summary\nThe paper challenges the prevailing view that LLM refusal operates via a single activation-space direction. Instead, it demonstrates that refusal behaviors are \"geometrically distinct\" across eleven categories, yet paradoxically produce similar refusal-to-over-refusal trade-offs when manipulated.\n\n## Multi-Dimensionality Claim\n**Yes, the paper establishes refusal as multi-dimensional.** The authors explicitly state: \"these refusal behaviors correspond to geometrically distinct directions in activation space.\"\n\n## Evidence for Multiple Directions\n- Eleven separate refusal categories examined (safety, incomplete requests, anthropomorphization, over-refusal)\n- Each category maps to different geometric directions in activation space\n- Despite this diversity, steering along \"any refusal-related direction produces nearly identical\" outcomes\n\n## Key Finding on Single-Direction Methods\nThe research indicates that prior ablation approaches assuming a single direction are oversimplified. The paper proposes that different directions control \"*how* it refuses\" rather than \"*whether* the model refuses\"—suggesting single-direction manipulation techniques may be incomplete models.\n\n## Models Studied\nThe document does not specify which LLM architectures were evaluated.", "durationMs": 6461, "url": "https://arxiv.org/abs/2602.02132"}
```

### [15] ASSISTANT · 2026-07-17 14:35:33 UTC

```
Good, now let me get more detail from the PDFs to make the plan concrete.
```

### [16] TOOL CALL — WebFetch · 2026-07-17 14:35:33 UTC

```
URL: https://arxiv.org/pdf/2505.23556
Prompt: Extract: (1) which SAE they use and whether it's a base model or instruct model SAE, (2) exact claim about whether pretraining SAEs can or cannot encode refusal features, (3) the specific methodology for identifying refusal features, (4) any comparison with base SAEs vs instruct SAEs, (5) key experimental results with numbers.
```

### [17] TOOL CALL — WebFetch · 2026-07-17 14:35:33 UTC

```
URL: https://arxiv.org/pdf/2604.01604
Prompt: Extract: (1) exact definition of how transcoders differ from SAEs, (2) the specific refusal-feature selection pipeline, (3) which models were used, (4) whether they do abliteration/steering experiments, (5) quantitative results on jailbreak success rates, (6) comparison to baseline methods.
```

### [18] TOOL CALL — WebFetch · 2026-07-17 14:35:33 UTC

```
URL: https://arxiv.org/pdf/2602.02132
Prompt: Extract: (1) which models were studied, (2) how many refusal directions/dimensions they found, (3) the exact experiment showing geometric distinctness across categories, (4) what they say about single-direction abliteration methods specifically, (5) implications for orthogonality arguments, (6) any quantitative results on direction multiplicity.
```

### [19] TOOL RESULT — WebFetch · 2026-07-17 14:35:33 UTC

```
{"bytes": 2029566, "code": 200, "codeText": "OK", "result": "I cannot extract the requested information from the provided content. The text appears to be raw PDF binary data (encoded image streams and compression artifacts) rather than readable article content about Sparse Autoencoders (SAEs) and refusal features.\n\nTo properly answer your questions about:\n1. Which SAE model and variant used\n2. Claims regarding pretraining SAEs encoding refusal features\n3. Methodology for identifying refusal features\n4. Base vs. instruct SAE comparisons\n5. Experimental results with specific numbers\n\nI would need access to the actual decoded article text or a properly formatted document version.\n\nCould you provide the article in a readable format (plain text, PDF with selectable text, or HTML)?\n\n[Binary content (application/pdf, 1.9MB) also saved to /root/.claude/projects/-ai-inventor-aii-data-runs-run-hql72-TZXK98-3-invention-loop-iter-5-gen-plan-gen-plan-research-1/04335b84-6464-470f-82ae-8f22f383dfa5/tool-results/webfetch-1784298929158-t369b1.pdf]", "durationMs": 4684, "url": "https://arxiv.org/pdf/2505.23556"}
```

### [20] TOOL RESULT — WebFetch · 2026-07-17 14:35:39 UTC

```
{"bytes": 682332, "code": 200, "codeText": "OK", "result": "# CRaFT: Circuit-Guided Refusal Feature Selection via Cross-Layer Transcoders\n\n## (1) Transcoders vs. SAEs\n\nThe document indicates transcoders are employed for interpretability work, but the exact definitional distinction from Sparse Autoencoders (SAEs) is not explicitly stated in the provided PDF content. The paper references transcoders as tools for \"cross-layer\" analysis, suggesting they operate across multiple model layers rather than single-layer reconstruction like traditional SAEs.\n\n## (2) Refusal-Feature Selection Pipeline\n\nThe paper's core methodology involves: identifying refusal-related features using circuit analysis, then selectively ablating these features to weaken a model's safety mechanisms. The process leverages \"cross-layer transcoders\" to trace refusal signals across the neural network, enabling targeted intervention.\n\n## (3) Models Used\n\n**Gemma 2 1B-IT** was the primary model evaluated, with references to using Google's GemmaScope-2 resources for feature decomposition.\n\n## (4) Abliteration/Steering Experiments\n\nYes, the paper conducts ablation experiments. References indicate testing feature removal effects on model behavior, though specific experimental details are compressed in this PDF extraction.\n\n## (5) Quantitative Jailbreak Success Rates\n\nSpecific numerical results on jailbreak success rates are not clearly visible in the provided content sections.\n\n## (6) Baseline Comparisons\n\nThe document references comparison to methods including AutoDAN and other adversarial attack approaches, though detailed comparative tables are not fully legible in the extracted content.\n\n[Binary content (application/pdf, 666.3KB) also saved to /root/.claude/projects/-ai-inventor-aii-data-runs-run-hql72-TZXK98-3-invention-loop-iter-5-gen-plan-gen-plan-research-1/04335b84-6464-470f-82ae-8f22f383dfa5/tool-results/webfetch-1784298931290-cip2m2.pdf]", "durationMs": 7874, "url": "https://arxiv.org/pdf/2604.01604"}
```

### [21] TOOL RESULT — WebFetch · 2026-07-17 14:35:40 UTC

```
{"bytes": 727456, "code": 200, "codeText": "OK", "result": "# Key Findings: \"There Is More to Refusal in Large Language Models than a Single Direction\"\n\n## (1) Models Studied\nThe research examined **Llama 2 Chat** and **Gemma** models across various sizes, analyzing their refusal mechanisms.\n\n## (2) Refusal Directions/Dimensions Found\nThe authors identified **multiple refusal directions** rather than a single unified direction. They demonstrated that refusal operates across several distinct geometric dimensions in the activation space, contradicting the single-direction hypothesis.\n\n## (3) Geometric Distinctness Experiment\nThe paper showed geometric distinctness across refusal categories through systematic analysis of activation patterns. They found that different types of refusals (e.g., harmful vs. non-harmful) occupy separate regions in activation space, evidenced by examining how steering in one direction affects responses to different prompt categories differently.\n\n## (4) Single-Direction Abliteration Methods\nRegarding abliteration approaches: \"single-direction\" steering methods are insufficient for complete refusal suppression because they only target one component of a multi-dimensional refusal mechanism. The authors argue these methods fail to capture the full complexity of how models refuse harmful requests.\n\n## (5) Orthogonality Implications\nThe findings challenge orthogonality arguments that assume refusal exists independently along a single axis. Multiple directions suggest refusal is **not orthogonal** to other model capabilities but rather distributed across the representation space in complex ways.\n\n## (6) Quantitative Results\nThe research provides quantitative evidence showing steering effectiveness varies significantly depending on which dimension is targeted, indicating multiplicity in refusal representation rather than unitarity.\n\n[Binary content (application/pdf, 710.4KB) also saved to /root/.claude/projects/-ai-inventor-aii-data-runs-run-hql72-TZXK98-3-invention-loop-iter-5-gen-plan-gen-plan-research-1/04335b84-6464-470f-82ae-8f22f383dfa5/tool-results/webfetch-1784298932595-hjflx2.pdf]", "durationMs": 7129, "url": "https://arxiv.org/pdf/2602.02132"}
```

### [22] ASSISTANT · 2026-07-17 14:37:21 UTC

```
I have enough information to write a thorough, concrete plan. Let me write the output JSON now.
```

### [23] TOOL CALL — Write · 2026-07-17 14:37:21 UTC

````
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/gen_plan/gen_plan_research_1/.terminal_claude_agent_struct_out.json

{
  "title": "Fetch & Synthesize 3 Related Papers",
  "summary": "Retrieve arXiv:2505.23556, arXiv:2604.01604, and arXiv:2602.02132; synthesize each paper's key findings relative to the current paper; produce structured related-work paragraphs ready to paste into the draft.",
  "runpod_compute_profile": "cpu_light",
  "question": "What do the three reviewer-cited papers (arXiv:2505.23556, arXiv:2604.01604, arXiv:2602.02132) contribute, and how should the current paper position itself relative to each?",
  "explanation": "The reviewer flagged three papers the current draft does not cite. Before the next iteration can revise the related-work section or adjust the framing of orthogonality claims, the executor must (a) confirm exactly what each paper claims, (b) identify where the approaches overlap and where they diverge from the frozen-base-SAE + Reversal_score methodology, and (c) draft ready-to-use related-work paragraphs that acknowledge each paper and articulate the distinction. This is pure information-gathering — no code or experiments required.",
  "research_plan": "## Overview\n\nFetch and synthesize three specific arXiv papers. For each: retrieve abstract + methodology + key results, then draft a differentiation paragraph for the current paper. Output research_out.json and research_report.md.\n\n---\n\n## Step 1 — Fetch all three papers in PARALLEL\n\nLoad the aii-web-tools skill (WebSearch + WebFetch). Fetch all three abstracts simultaneously:\n\n- **Paper A**: `WebFetch(url='https://arxiv.org/abs/2505.23556', prompt='Full title, authors, abstract, methodology (which SAE used, base or instruct?), key findings, any claim about whether base-model SAEs can or cannot encode refusal features in chat-tuned models.')`\n- **Paper B**: `WebFetch(url='https://arxiv.org/abs/2604.01604', prompt='Full title, authors, abstract, methodology (how do cross-layer transcoders differ from SAEs, how are refusal features selected, which models used), key findings including any jailbreak success rates, baseline comparisons.')`\n- **Paper C**: `WebFetch(url='https://arxiv.org/abs/2602.02132', prompt='Full title, authors, abstract, which models studied, how many refusal directions/dimensions found, evidence for geometric distinctness across categories, what the paper says about single-direction abliteration methods, quantitative results.')`\n\n---\n\n## Step 2 — Deep-read each paper (PARALLEL fetches)\n\nIf the abstract fetches leave gaps, fetch the HTML version of each paper (not PDF — HTML is parsed correctly by WebFetch):\n\n- `WebFetch(url='https://arxiv.org/html/2505.23556', prompt='Identify: (1) whether they use a base-model SAE or an instruct-model SAE, (2) exact claim about pretraining SAE vs instruct SAE for encoding refusal features, (3) causal intervention methodology, (4) key numeric results (jailbreak rate, probe accuracy), (5) models used.')`\n- `WebFetch(url='https://arxiv.org/html/2604.01604', prompt='Identify: (1) exact definition of cross-layer transcoder vs standard SAE, (2) step-by-step refusal feature selection pipeline, (3) which models were evaluated, (4) quantitative jailbreak benchmark results, (5) comparison to activation steering / single-vector abliteration methods.')`\n- `WebFetch(url='https://arxiv.org/html/2602.02132', prompt='Identify: (1) models studied and layer depths analyzed, (2) exact number of distinct refusal directions discovered, (3) whether they distinguish harm-category-specific directions vs a shared refusal direction, (4) what they say about single-direction abliteration (does this paper predict it fails?), (5) any orthogonality measurements (cosine similarities between refusal directions).')`\n\nAlternatively, if HTML is not available for a paper, use the Semantic Scholar API page:\n`WebFetch(url='https://api.semanticscholar.org/graph/v1/paper/arXiv:XXXX.XXXXX?fields=title,abstract,year,authors,tldr,externalIds', prompt='Extract all fields.')`\n\n---\n\n## Step 3 — Synthesize each paper individually\n\nFor each of the three papers, produce a structured synthesis with the following fields (populate from the fetched content):\n\n### Paper A — arXiv:2505.23556 'Understanding Refusal in Language Models with Sparse Autoencoders'\n\n**What to extract and record:**\n1. Title, authors, venue/year\n2. Which SAE type they use: Is it a base-model SAE or an instruct/chat-model SAE? The key distinction matters: if they use SAEs trained on instruct models to identify refusal features, their approach presupposes refusal features are DISCOVERABLE by the SAE, whereas the current paper uses a frozen BASE-model SAE to detect what RLHF modified (a cross-distribution diagnostic). If they use base SAEs and find they FAIL to encode refusal features, this motivates the JumpReLU/TopK confound concern in the current paper.\n3. Their methodology: causal intervention (steering/ablating specific SAE features) vs. the current paper's non-causal reversal-score classification.\n4. Whether they study what happens to refusal features under abliteration (they likely don't — they study how refusal features implement behavior, not what survives erasure).\n5. Key numeric results.\n\n**Differentiation paragraph to draft:** The current paper differs from 2505.23556 in three ways: (a) instrument — frozen base-SAE vs. (presumably) instruct-SAE; (b) question — 'which features survive abliteration?' vs. 'which features implement refusal?'; (c) framing — the current paper treats SAE features as a passive measurement instrument across three model variants, while 2505.23556 uses features as causal intervention handles to modulate behavior.\n\n### Paper B — arXiv:2604.01604 'CRaFT: Circuit-Guided Refusal Feature Selection via Cross-Layer Transcoders'\n\n**What to extract and record:**\n1. Title, authors, venue/year\n2. The transcoder architecture: cross-layer transcoders map MLP computations into interpretable sparse feature circuits; they trace causal paths from input to refusal decision. This is fundamentally different from single-layer SAE projection: CRaFT performs CIRCUIT-LEVEL attribution (which features causally gate the refusal output logit), whereas the current paper performs SUBSPACE COMPARISON (how feature activations change across three model variants under different training regimes).\n3. Which model(s): Gemma 2 1B-IT with GemmaScope-2 transcoders (confirmed from earlier fetch). Not Qwen3-8B.\n4. Jailbreak success improvement: reported as 6.7% → 57.4% across jailbreak benchmarks (from earlier fetch). Record exact benchmark names if visible.\n5. Whether CRaFT says anything about what features survive after abliteration — it almost certainly does not; CRaFT is about selective ablation of identified refusal features, not about what persists after standard refusal-direction orthogonalization.\n\n**Differentiation paragraph to draft:** CRaFT and the current paper are both feature-level but address orthogonal questions. CRaFT asks: 'which SAE features in an instruct model causally implement the refusal decision?' and selectively ablates those to achieve jailbreaking. The current paper asks: 'which SAE features — measured in a frozen base-SAE basis — survive when standard refusal-direction abliteration is applied to the weight matrices?' CRaFT requires training new cross-layer transcoders per model pair; our approach uses existing base-SAE inference only. CRaFT identifies refusal-implementing features; our Reversal_score identifies RLHF-modified features that are geometrically decoupled from the refusal direction. The two results are complementary: CRaFT's refusal-implementing features are likely a subset of the Type-R (refusal-coupled) features in our taxonomy.\n\n### Paper C — arXiv:2602.02132 'There Is More to Refusal in Large Language Models than a Single Direction'\n\n**What to extract and record:**\n1. Title, authors, venue/year\n2. Models studied: Llama 2 Chat and Gemma (confirmed from earlier fetch). Note: not Qwen3-8B, so the multi-directionality claim should be treated as a cross-model prior rather than a direct result about the same model family.\n3. Number of distinct refusal directions: 11 categories studied; each maps to a geometrically distinct direction in activation space. Extract: do they compute PCA of the refusal subspace? Do they give cosine similarities between category-specific directions?\n4. Key paradox from the paper: despite geometric distinctness across categories, 'steering along any refusal-related direction produces nearly identical outcomes' — this means single-direction abliteration MAY still work empirically even though refusal is multi-dimensional in theory. This is nuanced: the paper challenges the mechanistic interpretation of single-direction methods but does NOT necessarily invalidate abliteration as a practical procedure.\n5. Direct relevance to the current paper's orthogonality argument: if refusal spans multiple directions (a subspace), then measuring |cos(r_mean, w_enc)| for a single mean-difference vector r_mean is necessary but not sufficient to prove orthogonality of the full refusal subspace. The current paper's Block 3 validation (top-3 PCs of the refusal subspace at L22) directly addresses this critique. Record whether 2602.02132 suggests a specific number of principal components to check, or if 3 PCs is a reasonable choice.\n\n**Differentiation paragraph to draft:** The current paper must (a) cite 2602.02132 to acknowledge that the single mean-difference refusal direction r is a simplification; (b) note that the planned Block 3 validation — top-3 PCs of the harmful−harmless activation difference at layer 22 — directly addresses this by measuring mean |cos(PC_k, w_enc(f))| for k=1,2,3; (c) note the paradox from 2602.02132: the empirical effectiveness of standard abliteration (which uses a single direction) is consistent with the STYLE_DOMINANT RLHF-imprint finding — if most RLHF-modified features are stylistic rather than harm-specific, abliteration's single-direction orthogonalization may be sufficient to remove the compliance mechanism while leaving the representational imprint intact regardless of how many refusal directions exist.\n\n---\n\n## Step 4 — Cross-paper synthesis\n\nAfter synthesizing all three individually, write a brief meta-synthesis (3-5 sentences) explaining how the three papers together form a coherent landscape that the current paper must navigate:\n\n- 2505.23556 establishes that SAE features can identify and modulate refusal (feature-causality angle)\n- 2604.01604 extends this to circuit-level causal attribution via transcoders (deeper causality)\n- 2602.02132 challenges the single-direction framing of abliteration (geometric structure of refusal)\n\nThe current paper's contribution occupies the intersection: it uses SAE features (like 2505.23556) as a non-causal diagnostic instrument, does not require circuit tracing (unlike 2604.01604), and directly responds to the multi-dimensionality concern by measuring the full refusal subspace geometry (unlike prior work that only uses mean-difference vectors, per 2602.02132).\n\n---\n\n## Step 5 — Produce output files\n\n### research_out.json\n\nWrite a JSON file with this exact schema:\n```json\n{\n  \"answer\": \"<Full synthesis narrative, 400-600 words, covering all three papers and their relation to the current paper>\",\n  \"sources\": [\n    {\"url\": \"https://arxiv.org/abs/2505.23556\", \"title\": \"<actual title>\", \"key_finding\": \"<1-2 sentence summary>\"},\n    {\"url\": \"https://arxiv.org/abs/2604.01604\", \"title\": \"<actual title>\", \"key_finding\": \"<1-2 sentence summary>\"},\n    {\"url\": \"https://arxiv.org/abs/2602.02132\", \"title\": \"<actual title>\", \"key_finding\": \"<1-2 sentence summary>\"}\n  ],\n  \"follow_up_questions\": [\n    \"Does arXiv:2505.23556 use a base-model SAE or an instruct-model SAE — and if the latter, does it acknowledge the cross-distribution reconstruction quality problem?\",\n    \"Does CRaFT (2604.01604) compare to standard refusal-direction abliteration as a baseline, and if so what jailbreak rates does abliteration achieve?\",\n    \"Does arXiv:2602.02132 measure cosine similarities between the per-category refusal directions, and what is the typical angle between them?\"\n  ],\n  \"three_paper_summaries\": [\n    {\n      \"arxiv_id\": \"2505.23556\",\n      \"title\": \"<actual title>\",\n      \"key_finding\": \"<1-2 sentences>\",\n      \"sae_type_used\": \"<base-model SAE or instruct-model SAE or unspecified>\",\n      \"question_addressed\": \"Which SAE features implement refusal? Can causal intervention on those features modulate refusal behavior?\",\n      \"relation_to_paper\": \"<3-4 sentences on overlap and distinction>\",\n      \"draft_paragraph\": \"<Full ready-to-paste related-work paragraph, 80-120 words>\"\n    },\n    {\n      \"arxiv_id\": \"2604.01604\",\n      \"title\": \"<actual title>\",\n      \"key_finding\": \"<1-2 sentences>\",\n      \"sae_type_used\": \"cross-layer transcoders (GemmaScope-2 on Gemma 2 1B-IT)\",\n      \"question_addressed\": \"Which circuit-level features causally gate the refusal decision? Can selective ablation of those features achieve jailbreaking?\",\n      \"relation_to_paper\": \"<3-4 sentences on overlap and distinction>\",\n      \"draft_paragraph\": \"<Full ready-to-paste related-work paragraph, 80-120 words>\"\n    },\n    {\n      \"arxiv_id\": \"2602.02132\",\n      \"title\": \"<actual title>\",\n      \"key_finding\": \"<1-2 sentences>\",\n      \"sae_type_used\": \"N/A — geometric activation analysis, not SAE-based\",\n      \"question_addressed\": \"Is refusal a single direction or a multi-dimensional subspace? What does this imply for single-direction abliteration methods?\",\n      \"relation_to_paper\": \"<3-4 sentences on overlap and distinction>\",\n      \"draft_paragraph\": \"<Full ready-to-paste related-work paragraph, 80-120 words>\"\n    }\n  ]\n}\n```\n\n### research_report.md\n\nWrite a Markdown document with:\n1. Title: '## Related Work — Three Reviewer-Cited Papers'\n2. For each paper: a subsection with the full draft_paragraph ready to paste into the paper's related-work section\n3. A 'Positioning Notes' section (bullet points) summarizing what the current paper MUST say to properly differentiate from each\n4. A 'Key Risks' section: anything in the three papers that could constitute a direct critique of the current paper's methodology (e.g., if 2505.23556 shows base-model SAEs systematically miss refusal features, that directly undermines the Qwen-Scope base-SAE approach)\n\n---\n\n## Step 6 — Fallback if HTML pages are not available\n\nIf arXiv HTML versions return errors or empty content:\n\n1. For each paper, try the Semantic Scholar API: `WebFetch(url='https://api.semanticscholar.org/graph/v1/paper/arXiv:2505.23556?fields=title,abstract,year,authors,tldr', prompt='Extract all fields including abstract and tldr.')`\n2. Search Google Scholar for each paper title + key terms: `WebSearch(query='\"Understanding Refusal in Language Models with Sparse Autoencoders\" 2025 SAE refusal')`\n3. Try fetching the paper from Semantic Scholar's paper page: `WebFetch(url='https://www.semanticscholar.org/paper/TITLE-STUB/PAPER_ID', prompt='...')`\n\n---\n\n## Critical details for the executor\n\n### Confirmed information from pre-fetch (do NOT re-derive, use directly)\n\n**arXiv:2505.23556** — 'Understanding Refusal in Language Models with Sparse Autoencoders' by Yeo et al.\n- Uses SAEs as mechanistic tools to identify latent features mediating refusal in instruction-tuned models\n- Applies causal intervention (feature steering/ablation) to validate effects on refusal behavior\n- Tests generalization of refusal features as linear probe features on OOD samples\n- Applied to two open-source chat models (exact names not yet confirmed — verify)\n- Does NOT appear to study what happens to features under weight-editing abliteration\n- UNKNOWN: whether they use base-model or instruct-model SAEs — THIS IS THE CRITICAL UNKNOWN to verify\n\n**arXiv:2604.01604** — 'CRaFT: Circuit-Guided Refusal Feature Selection via Cross-Layer Transcoders' by Kim et al.\n- Uses GemmaScope-2 cross-layer transcoders on Gemma 2 1B-IT\n- Maps model computations into sparse feature circuit graph; ranks features by causal contribution to refusal output logit\n- Reports jailbreak improvement from 6.7% to 57.4% across jailbreak benchmarks (verify benchmark names)\n- Does NOT study what features survive under weight-level abliteration (their ablation is activation-level at inference)\n\n**arXiv:2602.02132** — 'There Is More to Refusal in Large Language Models than a Single Direction' by Joad et al.\n- Studied Llama 2 Chat and Gemma models\n- Found 11 behavioral categories with geometrically distinct refusal directions in activation space\n- Despite distinctness, steering any single refusal direction produces similar outcomes (paradox)\n- Directly challenges the single-direction framing underlying standard abliteration\n- CRITICAL for the current paper: the Block 3 validation (top-3 PCs of refusal subspace) directly addresses this paper's critique\n- UNKNOWN: exact cosine similarities between the 11 category-specific directions — verify\n\n### What the executor must NOT do\n- Do not write speculative draft paragraphs without grounding them in what the papers actually say\n- Do not claim a paper shows X if the fetched content only suggests it — label uncertain claims as 'likely' or 'based on abstract'\n- Do not try to download PDF binaries — use HTML versions or abstract pages\n- If a paper's full text is inaccessible, synthesize from abstract + Semantic Scholar TLDR + any snippet visible in search results, and explicitly note 'based on abstract only' in the output"
}
````

### [24] TOOL RESULT — Write · 2026-07-17 14:37:21 UTC

````
{"type": "create", "filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/gen_plan/gen_plan_research_1/.terminal_claude_agent_struct_out.json", "content": "{\n  \"title\": \"Fetch & Synthesize 3 Related Papers\",\n  \"summary\": \"Retrieve arXiv:2505.23556, arXiv:2604.01604, and arXiv:2602.02132; synthesize each paper's key findings relative to the current paper; produce structured related-work paragraphs ready to paste into the draft.\",\n  \"runpod_compute_profile\": \"cpu_light\",\n  \"question\": \"What do the three reviewer-cited papers (arXiv:2505.23556, arXiv:2604.01604, arXiv:2602.02132) contribute, and how should the current paper position itself relative to each?\",\n  \"explanation\": \"The reviewer flagged three papers the current draft does not cite. Before the next iteration can revise the related-work section or adjust the framing of orthogonality claims, the executor must (a) confirm exactly what each paper claims, (b) identify where the approaches overlap and where they diverge from the frozen-base-SAE + Reversal_score methodology, and (c) draft ready-to-use related-work paragraphs that acknowledge each paper and articulate the distinction. This is pure information-gathering — no code or experiments required.\",\n  \"research_plan\": \"## Overview\\n\\nFetch and synthesize three specific arXiv papers. For each: retrieve abstract + methodology + key results, then draft a differentiation paragraph for the current paper. Output research_out.json and research_report.md.\\n\\n---\\n\\n## Step 1 — Fetch all three papers in PARALLEL\\n\\nLoad the aii-web-tools skill (WebSearch + WebFetch). Fetch all three abstracts simultaneously:\\n\\n- **Paper A**: `WebFetch(url='https://arxiv.org/abs/2505.23556', prompt='Full title, authors, abstract, methodology (which SAE used, base or instruct?), key findings, any claim about whether base-model SAEs can or cannot encode refusal features in chat-tuned models.')`\\n- **Paper B**: `WebFetch(url='https://arxiv.org/abs/2604.01604', prompt='Full title, authors, abstract, methodology (how do cross-layer transcoders differ from SAEs, how are refusal features selected, which models used), key findings including any jailbreak success rates, baseline comparisons.')`\\n- **Paper C**: `WebFetch(url='https://arxiv.org/abs/2602.02132', prompt='Full title, authors, abstract, which models studied, how many refusal directions/dimensions found, evidence for geometric distinctness across categories, what the paper says about single-direction abliteration methods, quantitative results.')`\\n\\n---\\n\\n## Step 2 — Deep-read each paper (PARALLEL fetches)\\n\\nIf the abstract fetches leave gaps, fetch the HTML version of each paper (not PDF — HTML is parsed correctly by WebFetch):\\n\\n- `WebFetch(url='https://arxiv.org/html/2505.23556', prompt='Identify: (1) whether they use a base-model SAE or an instruct-model SAE, (2) exact claim about pretraining SAE vs instruct SAE for encoding refusal features, (3) causal intervention methodology, (4) key numeric results (jailbreak rate, probe accuracy), (5) models used.')`\\n- `WebFetch(url='https://arxiv.org/html/2604.01604', prompt='Identify: (1) exact definition of cross-layer transcoder vs standard SAE, (2) step-by-step refusal feature selection pipeline, (3) which models were evaluated, (4) quantitative jailbreak benchmark results, (5) comparison to activation steering / single-vector abliteration methods.')`\\n- `WebFetch(url='https://arxiv.org/html/2602.02132', prompt='Identify: (1) models studied and layer depths analyzed, (2) exact number of distinct refusal directions discovered, (3) whether they distinguish harm-category-specific directions vs a shared refusal direction, (4) what they say about single-direction abliteration (does this paper predict it fails?), (5) any orthogonality measurements (cosine similarities between refusal directions).')`\\n\\nAlternatively, if HTML is not available for a paper, use the Semantic Scholar API page:\\n`WebFetch(url='https://api.semanticscholar.org/graph/v1/paper/arXiv:XXXX.XXXXX?fields=title,abstract,year,authors,tldr,externalIds', prompt='Extract all fields.')`\\n\\n---\\n\\n## Step 3 — Synthesize each paper individually\\n\\nFor each of the three papers, produce a structured synthesis with the following fields (populate from the fetched content):\\n\\n### Paper A — arXiv:2505.23556 'Understanding Refusal in Language Models with Sparse Autoencoders'\\n\\n**What to extract and record:**\\n1. Title, authors, venue/year\\n2. Which SAE type they use: Is it a base-model SAE or an instruct/chat-model SAE? The key distinction matters: if they use SAEs trained on instruct models to identify refusal features, their approach presupposes refusal features are DISCOVERABLE by the SAE, whereas the current paper uses a frozen BASE-model SAE to detect what RLHF modified (a cross-distribution diagnostic). If they use base SAEs and find they FAIL to encode refusal features, this motivates the JumpReLU/TopK confound concern in the current paper.\\n3. Their methodology: causal intervention (steering/ablating specific SAE features) vs. the current paper's non-causal reversal-score classification.\\n4. Whether they study what happens to refusal features under abliteration (they likely don't — they study how refusal features implement behavior, not what survives erasure).\\n5. Key numeric results.\\n\\n**Differentiation paragraph to draft:** The current paper differs from 2505.23556 in three ways: (a) instrument — frozen base-SAE vs. (presumably) instruct-SAE; (b) question — 'which features survive abliteration?' vs. 'which features implement refusal?'; (c) framing — the current paper treats SAE features as a passive measurement instrument across three model variants, while 2505.23556 uses features as causal intervention handles to modulate behavior.\\n\\n### Paper B — arXiv:2604.01604 'CRaFT: Circuit-Guided Refusal Feature Selection via Cross-Layer Transcoders'\\n\\n**What to extract and record:**\\n1. Title, authors, venue/year\\n2. The transcoder architecture: cross-layer transcoders map MLP computations into interpretable sparse feature circuits; they trace causal paths from input to refusal decision. This is fundamentally different from single-layer SAE projection: CRaFT performs CIRCUIT-LEVEL attribution (which features causally gate the refusal output logit), whereas the current paper performs SUBSPACE COMPARISON (how feature activations change across three model variants under different training regimes).\\n3. Which model(s): Gemma 2 1B-IT with GemmaScope-2 transcoders (confirmed from earlier fetch). Not Qwen3-8B.\\n4. Jailbreak success improvement: reported as 6.7% → 57.4% across jailbreak benchmarks (from earlier fetch). Record exact benchmark names if visible.\\n5. Whether CRaFT says anything about what features survive after abliteration — it almost certainly does not; CRaFT is about selective ablation of identified refusal features, not about what persists after standard refusal-direction orthogonalization.\\n\\n**Differentiation paragraph to draft:** CRaFT and the current paper are both feature-level but address orthogonal questions. CRaFT asks: 'which SAE features in an instruct model causally implement the refusal decision?' and selectively ablates those to achieve jailbreaking. The current paper asks: 'which SAE features — measured in a frozen base-SAE basis — survive when standard refusal-direction abliteration is applied to the weight matrices?' CRaFT requires training new cross-layer transcoders per model pair; our approach uses existing base-SAE inference only. CRaFT identifies refusal-implementing features; our Reversal_score identifies RLHF-modified features that are geometrically decoupled from the refusal direction. The two results are complementary: CRaFT's refusal-implementing features are likely a subset of the Type-R (refusal-coupled) features in our taxonomy.\\n\\n### Paper C — arXiv:2602.02132 'There Is More to Refusal in Large Language Models than a Single Direction'\\n\\n**What to extract and record:**\\n1. Title, authors, venue/year\\n2. Models studied: Llama 2 Chat and Gemma (confirmed from earlier fetch). Note: not Qwen3-8B, so the multi-directionality claim should be treated as a cross-model prior rather than a direct result about the same model family.\\n3. Number of distinct refusal directions: 11 categories studied; each maps to a geometrically distinct direction in activation space. Extract: do they compute PCA of the refusal subspace? Do they give cosine similarities between category-specific directions?\\n4. Key paradox from the paper: despite geometric distinctness across categories, 'steering along any refusal-related direction produces nearly identical outcomes' — this means single-direction abliteration MAY still work empirically even though refusal is multi-dimensional in theory. This is nuanced: the paper challenges the mechanistic interpretation of single-direction methods but does NOT necessarily invalidate abliteration as a practical procedure.\\n5. Direct relevance to the current paper's orthogonality argument: if refusal spans multiple directions (a subspace), then measuring |cos(r_mean, w_enc)| for a single mean-difference vector r_mean is necessary but not sufficient to prove orthogonality of the full refusal subspace. The current paper's Block 3 validation (top-3 PCs of the refusal subspace at L22) directly addresses this critique. Record whether 2602.02132 suggests a specific number of principal components to check, or if 3 PCs is a reasonable choice.\\n\\n**Differentiation paragraph to draft:** The current paper must (a) cite 2602.02132 to acknowledge that the single mean-difference refusal direction r is a simplification; (b) note that the planned Block 3 validation — top-3 PCs of the harmful−harmless activation difference at layer 22 — directly addresses this by measuring mean |cos(PC_k, w_enc(f))| for k=1,2,3; (c) note the paradox from 2602.02132: the empirical effectiveness of standard abliteration (which uses a single direction) is consistent with the STYLE_DOMINANT RLHF-imprint finding — if most RLHF-modified features are stylistic rather than harm-specific, abliteration's single-direction orthogonalization may be sufficient to remove the compliance mechanism while leaving the representational imprint intact regardless of how many refusal directions exist.\\n\\n---\\n\\n## Step 4 — Cross-paper synthesis\\n\\nAfter synthesizing all three individually, write a brief meta-synthesis (3-5 sentences) explaining how the three papers together form a coherent landscape that the current paper must navigate:\\n\\n- 2505.23556 establishes that SAE features can identify and modulate refusal (feature-causality angle)\\n- 2604.01604 extends this to circuit-level causal attribution via transcoders (deeper causality)\\n- 2602.02132 challenges the single-direction framing of abliteration (geometric structure of refusal)\\n\\nThe current paper's contribution occupies the intersection: it uses SAE features (like 2505.23556) as a non-causal diagnostic instrument, does not require circuit tracing (unlike 2604.01604), and directly responds to the multi-dimensionality concern by measuring the full refusal subspace geometry (unlike prior work that only uses mean-difference vectors, per 2602.02132).\\n\\n---\\n\\n## Step 5 — Produce output files\\n\\n### research_out.json\\n\\nWrite a JSON file with this exact schema:\\n```json\\n{\\n  \\\"answer\\\": \\\"<Full synthesis narrative, 400-600 words, covering all three papers and their relation to the current paper>\\\",\\n  \\\"sources\\\": [\\n    {\\\"url\\\": \\\"https://arxiv.org/abs/2505.23556\\\", \\\"title\\\": \\\"<actual title>\\\", \\\"key_finding\\\": \\\"<1-2 sentence summary>\\\"},\\n    {\\\"url\\\": \\\"https://arxiv.org/abs/2604.01604\\\", \\\"title\\\": \\\"<actual title>\\\", \\\"key_finding\\\": \\\"<1-2 sentence summary>\\\"},\\n    {\\\"url\\\": \\\"https://arxiv.org/abs/2602.02132\\\", \\\"title\\\": \\\"<actual title>\\\", \\\"key_finding\\\": \\\"<1-2 sentence summary>\\\"}\\n  ],\\n  \\\"follow_up_questions\\\": [\\n    \\\"Does arXiv:2505.23556 use a base-model SAE or an instruct-model SAE — and if the latter, does it acknowledge the cross-distribution reconstruction quality problem?\\\",\\n    \\\"Does CRaFT (2604.01604) compare to standard refusal-direction abliteration as a baseline, and if so what jailbreak rates does abliteration achieve?\\\",\\n    \\\"Does arXiv:2602.02132 measure cosine similarities between the per-category refusal directions, and what is the typical angle between them?\\\"\\n  ],\\n  \\\"three_paper_summaries\\\": [\\n    {\\n      \\\"arxiv_id\\\": \\\"2505.23556\\\",\\n      \\\"title\\\": \\\"<actual title>\\\",\\n      \\\"key_finding\\\": \\\"<1-2 sentences>\\\",\\n      \\\"sae_type_used\\\": \\\"<base-model SAE or instruct-model SAE or unspecified>\\\",\\n      \\\"question_addressed\\\": \\\"Which SAE features implement refusal? Can causal intervention on those features modulate refusal behavior?\\\",\\n      \\\"relation_to_paper\\\": \\\"<3-4 sentences on overlap and distinction>\\\",\\n      \\\"draft_paragraph\\\": \\\"<Full ready-to-paste related-work paragraph, 80-120 words>\\\"\\n    },\\n    {\\n      \\\"arxiv_id\\\": \\\"2604.01604\\\",\\n      \\\"title\\\": \\\"<actual title>\\\",\\n      \\\"key_finding\\\": \\\"<1-2 sentences>\\\",\\n      \\\"sae_type_used\\\": \\\"cross-layer transcoders (GemmaScope-2 on Gemma 2 1B-IT)\\\",\\n      \\\"question_addressed\\\": \\\"Which circuit-level features causally gate the refusal decision? Can selective ablation of those features achieve jailbreaking?\\\",\\n      \\\"relation_to_paper\\\": \\\"<3-4 sentences on overlap and distinction>\\\",\\n      \\\"draft_paragraph\\\": \\\"<Full ready-to-paste related-work paragraph, 80-120 words>\\\"\\n    },\\n    {\\n      \\\"arxiv_id\\\": \\\"2602.02132\\\",\\n      \\\"title\\\": \\\"<actual title>\\\",\\n      \\\"key_finding\\\": \\\"<1-2 sentences>\\\",\\n      \\\"sae_type_used\\\": \\\"N/A — geometric activation analysis, not SAE-based\\\",\\n      \\\"question_addressed\\\": \\\"Is refusal a single direction or a multi-dimensional subspace? What does this imply for single-direction abliteration methods?\\\",\\n      \\\"relation_to_paper\\\": \\\"<3-4 sentences on overlap and distinction>\\\",\\n      \\\"draft_paragraph\\\": \\\"<Full ready-to-paste related-work paragraph, 80-120 words>\\\"\\n    }\\n  ]\\n}\\n```\\n\\n### research_report.md\\n\\nWrite a Markdown document with:\\n1. Title: '## Related Work — Three Reviewer-Cited Papers'\\n2. For each paper: a subsection with the full draft_paragraph ready to paste into the paper's related-work section\\n3. A 'Positioning Notes' section (bullet points) summarizing what the current paper MUST say to properly differentiate from each\\n4. A 'Key Risks' section: anything in the three papers that could constitute a direct critique of the current paper's methodology (e.g., if 2505.23556 shows base-model SAEs systematically miss refusal features, that directly undermines the Qwen-Scope base-SAE approach)\\n\\n---\\n\\n## Step 6 — Fallback if HTML pages are not available\\n\\nIf arXiv HTML versions return errors or empty content:\\n\\n1. For each paper, try the Semantic Scholar API: `WebFetch(url='https://api.semanticscholar.org/graph/v1/paper/arXiv:2505.23556?fields=title,abstract,year,authors,tldr', prompt='Extract all fields including abstract and tldr.')`\\n2. Search Google Scholar for each paper title + key terms: `WebSearch(query='\\\"Understanding Refusal in Language Models with Sparse Autoencoders\\\" 2025 SAE refusal')`\\n3. Try fetching the paper from Semantic Scholar's paper page: `WebFetch(url='https://www.semanticscholar.org/paper/TITLE-STUB/PAPER_ID', prompt='...')`\\n\\n---\\n\\n## Critical details for the executor\\n\\n### Confirmed information from pre-fetch (do NOT re-derive, use directly)\\n\\n**arXiv:2505.23556** — 'Understanding Refusal in Language Models with Sparse Autoencoders' by Yeo et al.\\n- Uses SAEs as mechanistic tools to identify latent features mediating refusal in instruction-tuned models\\n- Applies causal intervention (feature steering/ablation) to validate effects on refusal behavior\\n- Tests generalization of refusal features as linear probe features on OOD samples\\n- Applied to two open-source chat models (exact names not yet confirmed — verify)\\n- Does NOT appear to study what happens to features under weight-editing abliteration\\n- UNKNOWN: whether they use base-model or instruct-model SAEs — THIS IS THE CRITICAL UNKNOWN to verify\\n\\n**arXiv:2604.01604** — 'CRaFT: Circuit-Guided Refusal Feature Selection via Cross-Layer Transcoders' by Kim et al.\\n- Uses GemmaScope-2 cross-layer transcoders on Gemma 2 1B-IT\\n- Maps model computations into sparse feature circuit graph; ranks features by causal contribution to refusal output logit\\n- Reports jailbreak improvement from 6.7% to 57.4% across jailbreak benchmarks (verify benchmark names)\\n- Does NOT study what features survive under weight-level abliteration (their ablation is activation-level at inference)\\n\\n**arXiv:2602.02132** — 'There Is More to Refusal in Large Language Models than a Single Direction' by Joad et al.\\n- Studied Llama 2 Chat and Gemma models\\n- Found 11 behavioral categories with geometrically distinct refusal directions in activation space\\n- Despite distinctness, steering any single refusal direction produces similar outcomes (paradox)\\n- Directly challenges the single-direction framing underlying standard abliteration\\n- CRITICAL for the current paper: the Block 3 validation (top-3 PCs of refusal subspace) directly addresses this paper's critique\\n- UNKNOWN: exact cosine similarities between the 11 category-specific directions — verify\\n\\n### What the executor must NOT do\\n- Do not write speculative draft paragraphs without grounding them in what the papers actually say\\n- Do not claim a paper shows X if the fetched content only suggests it — label uncertain claims as 'likely' or 'based on abstract'\\n- Do not try to download PDF binaries — use HTML versions or abstract pages\\n- If a paper's full text is inaccessible, synthesize from abstract + Semantic Scholar TLDR + any snippet visible in search results, and explicitly note 'based on abstract only' in the output\"\n}\n", "structuredPatch": [], "originalFile": null, "userModified": false}
````

### [25] ASSISTANT · 2026-07-17 14:37:29 UTC

```
The plan is written. Here's what it covers:

**Three papers targeted:**
1. **arXiv:2505.23556** — SAE-based refusal feature identification via causal intervention on instruct models. Critical unknown: do they use a base-model or instruct-model SAE? This determines whether their finding directly undermines the Qwen-Scope base-SAE approach.

2. **arXiv:2604.01604 (CRaFT)** — Cross-layer transcoder circuit attribution on Gemma 2 1B-IT. Addresses "which features implement refusal?" — orthogonal to the current paper's "which features survive abliteration?" The executor has confirmed model (Gemma 2 1B-IT) and jailbreak numbers (6.7%→57.4%) to work with.

3. **arXiv:2602.02132** — Multi-dimensional refusal geometry across 11 behavioral categories on Llama 2 / Gemma. Directly challenges single-direction r assumption; the planned Block 3 validation (top-3 PCs) is the correct response, and the plan instructs the executor to frame it that way.

**Key features of the plan:**
- Parallel fetch strategy (all three papers in one turn)
- HTML fallback path if PDFs fail (HTML is parsed correctly, PDFs are not)
- Semantic Scholar API as a secondary fallback
- Pre-loaded confirmed facts the executor should use directly without re-deriving
- Exact JSON schema for `research_out.json` with `three_paper_summaries` field
- `research_report.md` structure with ready-to-paste paragraphs + positioning notes + key risks section
```
