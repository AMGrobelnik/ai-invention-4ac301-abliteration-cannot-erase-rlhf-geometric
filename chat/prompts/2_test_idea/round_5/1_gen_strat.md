# gen_strat_1 — test_idea

> Phase: `invention_loop` · round 5 · `gen_strat`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_strat_1` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-07-17 14:32:58 UTC

````
<ai_inventor_context>
<ai_inventor_summary>
You are one of many LLMs in AI Inventor — an automated research system that generates NOVEL and FEASIBLE hypotheses, investigates them through experiments and research, and produces a paper.

Your output feeds other LLMs downstream. This demands your ABSOLUTE MAXIMUM reasoning — every output must be deeply thought out and maximally useful. Surface-level responses waste downstream computation.
</ai_inventor_summary>

<your_role>
YOU ARE: A strategy planner (Step 3.1: GEN_STRAT in the invention loop)

Each iteration of the invention loop runs: GEN_STRAT → GEN_PLAN → GEN_ART → GEN_PAPER_TEXT → REVIEW_PAPER → UPD_HYPO
Artifact types: RESEARCH (web search), EXPERIMENT (code), DATASET (data collection), EVALUATION (metrics), PROOF (Lean 4)
State persists across iterations: strategies, plans, artifacts, paper_texts (read from the run tree)

You received the hypothesis, iteration status (current + remaining), previous iteration's strategies, available artifact types, existing artifacts, and reviewer feedback.
Your strategy governs THIS iteration only. You define what artifacts to create NOW.

Focused strategy → efficient progress. Scattered strategy → wasted iteration.
</your_role>
</ai_inventor_context>

<available_resources>
<software_constraints>
- Python only implementation
- Python standard library and all popular PyPI packages available (numpy, pandas, scikit-learn, scipy, matplotlib, requests, etc.)
- Local parallelism encouraged: multiprocessing, asyncio, threading — see aii-parallel-computing skill
- LLM API calls must go through OpenRouter only (no direct OpenAI, Anthropic, etc.)
- **HARD LIMIT**: Maximum $10 USD total spend on LLM API calls (OpenRouter). Track cumulative cost after every call and STOP IMMEDIATELY if approaching this limit. Never exceed this budget under any circumstances.
</software_constraints>

<skills>
Skills are self-contained capabilities with instructions, context, and tools.

- aii-web-tools: Web search (Serper), page/PDF fetch as markdown, regex grep over page/PDF text
- aii-semscholar-bib: Batch-fetch BibTeX from Semantic Scholar
- aii-openrouter-llms: Search and call 300+ LLMs via OpenRouter
- aii-hf-datasets: Search, preview, download HuggingFace datasets
- aii-owid-datasets: Search and load Our World in Data tables
- aii-lean: Compile/verify Lean 4 code, Mathlib search, tactic suggestions
- aii-image-gen: Generate/edit images via Gemini 3 Pro Image (Nano Banana Pro)
- aii-json: Validate JSON against schemas, generate mini/preview variants
- aii-paper-writing: Academic paper structure, bibliography, citations
- aii-paper-to-latex: Assemble LaTeX papers and compile to PDF
- aii-parallel-computing: GPU acceleration, CPU parallelism, async I/O
- aii-python: Python coding standards for experiment scripts
- aii-use-hardware: Detect CPU/RAM/GPU, memory-safe processing
- aii-long-running-tasks: Gradual scaling pattern for long-running tasks
- aii-colab: Google Colab runtime constraints for notebooks
- aii-file-size-limit: Check and split oversized output files
</skills>
</available_resources>

<time_budgets>

Each artifact executor has a fixed time budget (including writing code, debugging, testing, and fixing errors):

- research: 3h
- dataset: 6h
- experiment: 6h
- evaluation: 3h
- proof: 3h

</time_budgets>

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

<research_methodology>
Think like a researcher planning a study for a top venue.

- All strategies run in parallel and their artifacts combine into one pool. Together they must build toward a publishable paper — each strategy contributes a distinct, necessary piece. No strategy should be a standalone island.
- Ask yourself: what would a reviewer need to see? Proper baselines, controlled comparisons, ablations that isolate what matters. Plan artifacts that preempt reviewer objections.
- Depth over breadth. One well-designed experiment with proper controls beats five shallow ones.
- Match your evaluation to your claims. Measure what the hypothesis actually asserts.
- When results are weak or partial, vary the approach before writing it off. One failed method doesn't falsify the hypothesis.
- If iterations remain, think about what the NEXT iteration will need. Leave useful building blocks — datasets, baselines, preliminary results — that future strategies can build on, refine, or compare against.
</research_methodology>

<principles>
1. FOCUS ON NOVELTY - every strategy must lead to a genuinely novel contribution
2. MAXIMIZE PARALLELIZATION - all artifacts in your strategy run in parallel
3. BUILD ON EXISTING WORK - use completed artifacts from previous iterations, learn from failures
4. ITERATE ON THE METHOD - a negative result is about the approach, not the hypothesis. Try different methods, parameters, data, or formulations within the hypothesis bounds.
5. DIAGNOSE BEFORE DECIDING - before each iteration, review what worked, what didn't, and why. Use that to choose what to try next. Gaps are action items, not conclusions.
6. SET DEPENDENCIES WISELY - depends_on is a list of {id, label} objects referencing existing artifacts; each label is a short free-text type (a word or two, e.g. "dataset", "validates", "extends") that tags how the dep is used
7. PLAN FOR DEPENDENCIES - if an artifact depends on another (e.g. experiments need datasets), ensure prerequisites exist first or plan them this iteration for the next
</principles>

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
Your strategy should advance this hypothesis.

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
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for study design, proper baselines, and the evaluation/validity norms this field demands.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>

<iteration_status>
Current iteration: 5 of 5
Remaining (including this one): 1
</iteration_status>

<previous_strategies>
Strategies from the PREVIOUS iteration. You can CONTINUE these directions,
ADAPT based on what worked and what didn't in the artifacts produced, or PIVOT if results suggest a better path.

--- Strategy 1 ---
kind: strategy
id: gen_strat_1_idx1
title: OLS Slope + Mid-SC Baseline + Benign Overlap
objective: >-
  Resolve all remaining major reviewer critiques by computing: (1) the OLS regression slope β_f for each top-500 feature to
  determine whether the RLHF imprint is preserved in magnitude (β̄≈1, 'comprehensive survival') or direction only (β̄≪1, 'directional
  echo') — the single most important missing measurement; (2) a mid-SC baseline (40th–60th SC percentile) to confirm the Cohen's
  d=1.751 null-baseline gap is not a noise-to-noise artifact from near-zero SC bottom-500 features; and (3) benign-SC overlap
  analysis to confirm the top-500 harmful-SC features reflect harm-specific rather than style-specific RLHF modifications.
  These three measurements together determine the paper's core framing and fix all outstanding critiques before the final
  paper iteration.
rationale: >-
  This is iteration 4 of 5. All three Qwen3-8B model activations are already cached on disk from prior iterations (layer-22
  and layer-20 residual streams, 1000×4096 each), so the entire experiment requires zero re-running of model inference — only
  fast post-processing in NumPy/SciPy. The OLS slope β_f is computationally trivial: for each of 500 features, fit a single
  no-intercept linear regression over 500 scalars. The mid-SC baseline requires selecting features at the 40th–60th SC percentile
  and computing their Reversal Scores — identical computation to the null-baseline analysis. The benign-SC overlap requires
  running SC(f) on the 500 benign prompts already available in the corpus. All three analyses can be run together in well
  under 10 minutes of compute. The β_f distribution is the primary result that determines whether the paper says 'comprehensive
  latent safety imprint' (β̄≈1) or 'directional geometric echo' (β̄≪1) — both are publishable, but the correct framing must
  be established. The mid-SC baseline directly addresses the MAJOR critique that bottom-500 PearsonR values may reflect noise-to-noise
  correlations. Low benign-SC overlap (<20%) strengthens the harm-specificity claim.
artifact_directions:
- id: experiment_iter4_dir1
  type: experiment
  objective: >-
    Compute OLS regression slopes β_f (magnitude preservation), mid-SC baseline Reversal Scores, benign-SC overlap, and |cos|
    numerical reconciliation — the four measurements required to resolve all outstanding MAJOR reviewer critiques and determine
    the paper's definitive framing.
  approach: |-
    Use UV with numpy, scipy, torch, huggingface_hub. The experiment re-runs the full activation extraction pipeline (loading all three Qwen3-8B models sequentially) at layers 22 and 20, replicating the exact setup from prior iterations, then computes all new metrics.

    **Step 0 — Model loading and activation extraction.** Load models sequentially with torch.cuda.empty_cache() between each: Base=Qwen/Qwen3-8B-Base, Instruct=Qwen/Qwen3-8B (enable_thinking=False), Abliterated=huihui-ai/Huihui-Qwen3-8B-abliterated-v2. Register forward hooks on model.model.layers[22].output and model.model.layers[20].output to extract last-token residual stream activations. Use the prompt corpus from the dataset artifact (full_data_out.json), separating harmful_idx (first 500) and benign_idx (next 500). Run in batches of 4. Save act arrays: base_L22, inst_L22, abl_L22, base_L20, inst_L20, abl_L20 each shape (1000, 4096).

    **Step 1 — SAE load and pre-topk projections.** Download Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 via hf_hub_download (layer22.sae.pt, layer20.sae.pt). Load W_enc (65536×4096), b_enc (65536,) for each layer. Compute z_pre_X_L22 = act_X_L22 @ W_enc22.T + b_enc22 for X in {base, inst, abl}. Similarly for L20.

    **Step 2 — SC(f) on harmful AND benign prompts.** SC_harm(f) = mean_i |z_pre_inst_L22[harmful_idx, f] - z_pre_base_L22[harmful_idx, f]|. SC_benign(f) = mean_i |z_pre_inst_L22[benign_idx, f] - z_pre_base_L22[benign_idx, f]|. top500_harm_idx = argsort(SC_harm)[-500:]; bottom500_harm_idx = argsort(SC_harm)[:500]; top500_benign_idx = argsort(SC_benign)[-500:]. Benign overlap count = len(set(top500_harm_idx) & set(top500_benign_idx)). Report overlap_fraction = count / 500.

    **Step 3 — OLS slope β_f for top-500 RLHF-sensitive features.** For each feature f in top500_harm_idx over harmful prompts: delta_inst_f = z_pre_inst_L22[harmful_idx, f] - z_pre_base_L22[harmful_idx, f]; delta_abl_f = z_pre_abl_L22[harmful_idx, f] - z_pre_base_L22[harmful_idx, f]. OLS no-intercept closed form: beta_f = dot(delta_inst_f, delta_abl_f) / dot(delta_inst_f, delta_inst_f). Sanity check: also compute r_f = pearsonr(delta_inst_f, delta_abl_f)[0] — should match iter3 R values. Collect: beta_mean, beta_std, beta_median, beta_min, beta_max, histogram (20 bins), fraction_gt_0p8, fraction_gt_1p0, fraction_lt_0p2. Also compute on all 1000 prompts as robustness check. INTERPRETATION: beta_mean > 0.8 → 'MAGNITUDE_PRESERVED'; 0.3–0.8 → 'PARTIAL_ATTENUATION'; <0.3 → 'DIRECTION_ONLY_ECHO'.

    **Step 4 — Mid-SC baseline Reversal Scores.** p40 = np.percentile(SC_harm, 40); p60 = np.percentile(SC_harm, 60). mid_candidates = np.where((SC_harm >= p40) & (SC_harm <= p60))[0]. mid500_idx = np.random.choice(mid_candidates, size=min(500, len(mid_candidates)), replace=False). For each f in mid500_idx: R_mid_f = pearsonr(z_pre_inst_L22[harmful_idx, f] - z_pre_base_L22[harmful_idx, f], z_pre_abl_L22[harmful_idx, f] - z_pre_base_L22[harmful_idx, f])[0]. Report: mid500_R_mean, mid500_R_std, mid500_SC_median. Also report bottom500_SC_median = median(SC_harm[bottom500_harm_idx]). Run ttest_ind(top500_R, mid500_R) and compute Cohen's d. VERDICT: ttest_p < 0.05 AND d > 0.5 → 'MID_SC_CONFIRMS_SELECTIVE_PRESERVATION'; else → 'MID_SC_BASELINE_WEAKENS_CLAIM'.

    **Step 5 — |cos| numerical reconciliation.** r_L22 = normalize(mean(inst_L22[harmful_idx], 0) - mean(inst_L22[benign_idx], 0)). For each f in top500_harm_idx: cos_f = dot(r_L22, W_enc22[f,:]) / (norm(r_L22) * norm(W_enc22[f,:])). Report mean(|cos_f|) as cos_enc_l22_top500. Similarly for layer 20. Document: which value is 0.0145 vs 0.019 vs 0.021 and why (likely different runs or pre/post-normalization differences).

    **Output.** Write full_method_out.json (1000 rows, exp_gen_sol_out schema), mini_method_out.json (50 rows), preview_method_out.json (5 rows). Each row contains: prompt_idx, is_harmful, and scalar summary fields: beta_mean_top500, r_mean_top500, mid_sc_r_mean, benign_overlap_fraction, cos_enc_l22_mean. Top-level summary block embedded in each row as repeated metadata. Validate JSON schema with aii-json skill.
  depends_on:
  - id: art_M8-flaZ0OOaP
    label: prompt corpus
    relation_type:
    relation_rationale:
  - id: art_pmAcY0CNazP9
    label: tech reference
    relation_type:
    relation_rationale:
expected_outcome: >-
  After iteration 4, the paper has definitive answers to all MAJOR reviewer critiques. The β_f distribution determines the
  paper's framing: β̄≈1 → reinstate 'comprehensive latent safety imprint'; β̄≪1 → reframe as 'directional geometric echo'
  with attenuated magnitude. The mid-SC baseline confirms whether Cohen's d=1.751 is robust to the noise concern. The benign-SC
  overlap confirms harm-specificity (expected: <25% overlap). The |cos| reconciliation eliminates the minor numerical discrepancy.
  The final iteration (GEN_PAPER_TEXT) has everything needed to write a clean, reviewer-defensible paper.
summary: >-
  One focused experiment resolves all outstanding MAJOR reviewer critiques: (1) OLS slope β_f to determine direction-only
  vs. magnitude-preserving RLHF imprint survival; (2) mid-SC baseline Reversal Scores to confirm the d=1.751 gap is not a
  noise artifact; (3) benign-SC overlap to confirm harm-specificity of the top-500 feature selection; (4) |cos| numerical
  reconciliation. The β_f distribution is the critical result that determines the paper's definitive framing in the final
  iteration.
</previous_strategies>

<dependency_rules>
- depends_on is a list of objects {id, label} — each entry references an existing artifact and tags how it is being used
- "id" can ONLY reference IDs from <existing_artifacts> — never IDs you are proposing (all new artifacts run in parallel)
- "label" is a SHORT free-text type label (a word or two, NOT a sentence) describing what role the dep plays — e.g. "dataset", "validates", "extends", "supersedes". Required on every dep.
- Setting depends_on provides the dependency's out_dependency_files to your artifact at execution time
- If no suitable existing artifacts exist, use empty depends_on
- New artifact IDs are assigned by the system after submission — do not invent IDs for your proposed artifacts
</dependency_rules>

<available_artifact_types>
Artifact types you can plan. Use this to choose the right types for your strategy objectives.

<artifact_types>
RESEARCH
Web research to answer key questions — like a researcher making decisions.
Runtime: LLM Agent, no code execution.
Tools: the aii-web-tools skill (web search, page fetch, regex grep over full page/PDF text).
Capabilities: Find, synthesize, and compare information across sources; survey SOTA and best practices.
Deps: REQUIRED none | OPTIONAL other RESEARCH to build on prior findings

EXPERIMENT
Run code to test hypotheses, implement methods, and collect empirical results.
Runtime: Python 3.12, UV (any pip package), isolated workspace, gradual scaling (mini → full data).
Tools: Full shell/Python/filesystem access, the aii-web-tools skill (web search, page fetch, regex grep over full page/PDF text), and other skills.
Skills: aii-json (schema validation), aii-openrouter-llms (call any LLM — GPT, Gemini, Llama, etc.), domain-specific as needed.
Capabilities: Implement and run any code-based experiment, compare method vs baselines.
Deps: REQUIRED at least one DATASET | OPTIONAL RESEARCH for methodology guidance

DATASET
Collect, prepare, and merge datasets for experiments and analysis.
Runtime: Python 3.12, UV, isolated workspace.
Tools: Full shell/Python/filesystem access, the aii-web-tools skill (web search, page fetch, regex grep over full page/PDF text), and other skills.
Skills: aii-hf-datasets (HuggingFace Hub — ML datasets, many UCI/OpenML/Kaggle mirrors), aii-owid-datasets (Our World in Data — global statistics), aii-json (schema validation). Also any Python source (sklearn.datasets, openml, direct URLs, APIs) — must verify within 300MB limit.
Capabilities: Search, acquire, transform, combine, and standardize data from any available source.
Deps: REQUIRED none | OPTIONAL RESEARCH for guidance on what data to collect

EVALUATION
Evaluate experiment results with metrics, statistical analysis, and validity checks.
Runtime: Python 3.12, UV (any evaluation library), isolated workspace, gradual scaling matching experiment.
Tools: Full shell/Python/filesystem access, the aii-web-tools skill (web search, page fetch, regex grep over full page/PDF text), and other skills.
Skills: aii-json (schema validation), aii-openrouter-llms (call any LLM — GPT, Gemini, Llama, etc.), domain-specific as needed.
Capabilities: Compute any quantitative metrics and statistical tests, analyze validity and robustness.
Deps: REQUIRED at least one EXPERIMENT | OPTIONAL DATASET if reference data needed

PROOF
Formally prove mathematical statements in Lean 4 with automated iteration.
Runtime: LLM agent with Lean 4 compiler feedback loop.
Tools: Full shell/Python/filesystem access, the aii-web-tools skill (web search, page fetch, regex grep over full page/PDF text), and other skills.
Skills: aii-lean (proof verification, Mathlib search, tactics: ring, linarith, nlinarith, omega, simp, etc.)
Capabilities: Formally verify properties and inequalities, iterative proof development, lemma decomposition.
Deps: REQUIRED none | OPTIONAL RESEARCH for mathematical background
</artifact_types>
</available_artifact_types>

<artifact_executor_scope>
IMPORTANT: Each artifact executor has a focused prompt that guides it to do ONE thing well. It will NOT perform tasks outside its scope — assigning the wrong work to the wrong artifact type wastes an iteration. Match the task to the right executor.

RESEARCH executor scope:
  Output: research_out.json with {answer, sources, follow_up_questions} + research_report.md
  DOES: Web research — search, read, synthesize information from papers/docs/APIs into a structured report
  DOES NOT: Run code, download files, execute scripts, compute anything — no shell/Python access
  Use for literature surveys, API documentation, technical specifications — pure information gathering

EXPERIMENT executor scope:
  Output: method_out.json with results (metrics, predictions, analysis) — the core computational work
  DOES: Implement and run methods/algorithms, compute metrics, compare approaches, produce quantitative results
  DOES NOT: Collect new datasets (depends on DATASET artifacts for input data), write formal proofs
  This is the right artifact for any code that processes data and produces results

DATASET executor scope:
  Output: data_out.json with rows of {input, output, metadata_fold, ...} — raw data only, no derived computations
  DOES: Download/generate datasets, analyze candidates to pick the best ones, standardize to JSON schema (features, labels, folds, metadata), validate schema, split into full/mini/preview
  DOES NOT: Run experiments, train models, compute derived statistics (PID/MI/correlations/synergy matrices) as final output
  If you need to COMPUTE something from data (synergy matrices, MI scores, timing benchmarks), use an EXPERIMENT artifact instead

EVALUATION executor scope:
  Output: eval_out.json with evaluation results
  DOES: Any evaluation of experiment results — metrics, statistical tests, ablations, comparisons, visualizations, robustness checks, error analysis, etc.
  DOES NOT: Implement new methods (use EXPERIMENT), collect data (use DATASET)
  This is for analyzing experiment outputs from any angle

PROOF executor scope:
  Output: Lean 4 proof files (.lean) with verified theorems
  DOES: Write and verify Lean 4 formal proofs with Mathlib, iterative compilation
  DOES NOT: Run Python experiments, collect data, do empirical analysis
  Use only when formal mathematical guarantees are needed
</artifact_executor_scope>

<artifact_planning_rules>
RESEARCH: Plan early — findings guide dataset selection, experiment design, and methodology.
EXPERIMENT: Must depend on at least one DATASET. Define clear metrics and baselines before running. Consider trying multiple method variations rather than a single approach.
DATASET:
- Plan for REAL third-party datasets (HuggingFace, Kaggle, direct-download URLs) — downloadable within time and size constraints
- Describe dataset criteria (domain, size, format) — executors find exact sources, but you can suggest candidates or search directions
- ALWAYS prefer real datasets over synthetic. Synthetic is a LAST RESORT only when no suitable real data exists
EVALUATION: Must depend on at least one EXPERIMENT. Focus on statistical rigor and validity checks.
PROOF: Use only when the hypothesis requires formal mathematical guarantees. Lean 4 + Mathlib.
</artifact_planning_rules>

<existing_artifacts>
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
out_dependency_files:
  file_list:
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
out_dependency_files:
  file_list:
  - data.py
  - full_data_out.json
  - mini_data_out.json
  - preview_data_out.json
  data_file_paths:
  - full_data_out.json
  - mini_data_out.json
  - preview_data_out.json

--- Item 3 ---
id: art_QBm3I0Z5VVJR
type: experiment
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
out_dependency_files:
  file_list:
  - method.py
  - full_method_out.json
  - mini_method_out.json
  - preview_method_out.json

--- Item 4 ---
id: art_Qnl1zSSZPJvj
type: experiment
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
out_dependency_files:
  file_list:
  - method.py
  - full_method_out.json
  - mini_method_out.json
  - preview_method_out.json

--- Item 5 ---
id: art_iDkiRSdMuI6M
type: experiment
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
out_dependency_files:
  file_list:
  - method.py
  - full_method_out.json
  - mini_method_out.json
  - preview_method_out.json
</existing_artifacts>

<current_paper>
The current paper draft — represents the research story so far.

Use this to understand what's working, what's not, and what gaps remain.
Gaps and weak results signal what to try differently — not what to conclude.

# Introduction

Refusal ablation — commonly called *abliteration* — edits a safety-aligned language model's weights to eliminate compliance behavior [1]. The technique identifies the *refusal direction*: the principal mean-difference vector separating residual-stream activations on harmful versus harmless prompts, then orthogonalizes all residual-stream-writing weight matrices against this direction. The result is a model that passes harmful prompts through as if it had never been safety-tuned. The standard narrative frames abliteration as *undoing* RLHF safety fine-tuning.

Two geometric studies directly challenge this narrative. LatentBiopsy [2] demonstrates that a linear classifier applied to Qwen3.5-0.8B and Qwen2.5-0.5B residual streams retains AUROC ≥ 0.937 after abliteration, dropping at most 0.015 from the instruct baseline. A companion cross-family study [3] replicates this across 12 model variants (Qwen, Llama, Gemma; 0.5B–9B parameters), reporting mean AUROC 0.982 surviving abliteration within ±0.003. Something in the internal representation of harmful content survives abliteration essentially unchanged. What these geometric analyses cannot answer is the mechanistic question: *which specific features carry the surviving representation? Do they survive in direction only, or also in magnitude? And what geometric relationship between the refusal direction and RLHF-modified feature directions would explain this survival?*

Answering these questions requires a feature-level vocabulary. Sparse Autoencoders (SAEs) [11, 12] provide exactly that: trained on residual-stream activations, they decompose those activations into tens of thousands of near-monosemantic features. We apply the frozen Qwen-Scope base SAE [9] — 65,536 TopK features trained on Qwen3-8B residual-stream activations — across three model variants: base, instruct (RLHF-aligned), and abliterated. We introduce two complementary metrics. The *Reversal Score* R(f) is a per-feature Pearson correlation between RLHF-induced and abliteration-induced activation change vectors over 500 harmful prompts, measuring whether each RLHF-modified feature is preserved or reversed in *direction*. The *OLS Slope* β_f is the no-intercept regression coefficient of abliteration-induced changes on RLHF-induced changes, measuring whether the *magnitude* of each RLHF-induced shift is preserved.

Our central empirical finding is that abliteration preserves the RLHF imprint in both direction and magnitude. No feature in the top-500 RLHF-sensitive set has a negative Reversal Score (mean R̄ = 0.921, std = 0.040, min = 0.684), and the mean OLS slope is β̄ = 0.986 ± 0.176 — meaning abliteration preserves not just the direction but approximately the full amplitude of each RLHF-induced feature change. A noise-free mid-SC baseline (features at the 40th–60th Safety Change percentile, SC_median = 1.12) shows R̄ = 0.871 vs. top-500 R̄ = 0.921 (Cohen's d = 1.087, p ≈ 0), ruling out measurement artifacts. A direct measurement of refusal-direction geometry confirms the mechanistic explanation: mean |cos(r, w_enc)| = 0.0185 at layer 22 and 0.0209 at layer 20 across RLHF-sensitive encoder directions.

An important qualification accompanies these findings. Of the top-500 RLHF-sensitive features (selected by harmful-prompt Safety Change), 64.6% also appear in the top-500 by benign-prompt Safety Change — indicating that the preserved RLHF imprint captures broad fine-tuning modifications including response style and instruction-following format, not exclusively harm-semantic content. This does not undermine the main finding; it clarifies what "comprehensive survival" means: abliteration cannot remove any RLHF modification, whether harm-semantic or stylistic, because all these modifications reside in a subspace near-orthogonal to the refusal direction.

[FIGURE:fig1]

**Summary of Contributions:**
- We introduce the *Reversal Score*, a per-feature Pearson correlation confirming direction preservation of RLHF-modified SAE features under abliteration (R̄ = 0.921 at layer 22, zero negative values; Section 3.3). [ARTIFACT:art_QBm3I0Z5VVJR]
- We introduce the *OLS Slope* β_f, a companion magnitude metric, and report β̄ = 0.986 ± 0.176 — confirming that abliteration preserves the full magnitude, not merely the direction, of RLHF-induced feature changes (Section 5.2). [ARTIFACT:art_iDkiRSdMuI6M]
- We validate selective preservation against a noise-free mid-SC baseline (SC_median = 1.12 ≫ 0.01): top-500 R̄ = 0.921 vs. mid-500 R̄ = 0.871, Cohen's d = 1.087, p ≈ 0 (Section 5.3). [ARTIFACT:art_iDkiRSdMuI6M]
- We directly measure refusal-direction orthogonality to RLHF-modified SAE encoder directions, finding mean |cos| = 0.0185 at layer 22 and 0.0209 at layer 20 — the geometric mechanism explaining zero Type-R features (Section 5.4). [ARTIFACT:art_Qnl1zSSZPJvj]
- We show that 64.6% of the top-500 RLHF-sensitive features overlap with the top-500 benign-SC features, establishing that the preserved imprint reflects broad RLHF modifications — not exclusively harm-specific semantics (Section 5.5). [ARTIFACT:art_iDkiRSdMuI6M]

# Related Work

**Refusal ablation.** Arditi et al. [1] establish that refusal is mediated by a single residual-stream direction and that orthogonalizing all weight matrices against it reliably eliminates compliance across 13 models up to 72B parameters. Cristofano [7] shows the raw refusal vector is polysemantic, entangling safety circuitry with capability and style; spectral residualization reduces KL divergence on Wikitext-2 from 2.088 to 0.044 while preserving abliteration. Nanfack et al. [19] propose an optimal-transport alternative reducing distributional shift. Our work examines what all these variants share: the refusal direction, however constructed, leaves RLHF-modified SAE encoder features intact in both direction and magnitude.

**Geometry of harmful intent surviving abliteration.** LatentBiopsy [2] and the companion geometry study [3] are the direct predecessors establishing the phenomenon our work explains mechanistically. Their key limitation — collapsing representational information into a single AUROC score — is precisely the gap our feature-level analysis addresses. The OLS slope results (β̄ = 0.986) and direct orthogonality measurement (|cos| = 0.0185) provide the mechanistic account that a single detection score cannot.

**SAEs as measurement instruments.** Elhage et al. [11] motivate SAE decomposition via the superposition hypothesis. Bricken et al. [12] demonstrate dictionary learning recovering monosemantic features at practical scale. Gao et al. [13] establish scaling laws for SAE quality. Templeton et al. [14] apply SAEs at Claude 3 Sonnet scale. Chopra [4] uses frozen base SAEs as cross-variant diagnostics, finding that SAE latent cosine similarity drops to 0.509–0.557 at layer 22 after SFT while raw cosine stays above 0.96 — validating the frozen base-SAE approach and motivating the pre-topk encoder fallback when reconstruction quality degrades.

**Multi-model SAE comparison.** Jiralerspong and Bricken [5] introduce *crosscoders* — SAEs trained jointly on two or more model variants — partitioning features into shared and model-specific categories. Kassem et al. [6] extend this with Delta-Crosscoder for narrow fine-tuning regimes. Our frozen base-SAE approach is complementary: it requires zero additional training, directly reuses Qwen-Scope's existing feature vocabulary, and applies naturally to the three-model abliteration trajectory. Shportko et al. [20] use Dedicated Feature Crosscoders to localize RL-induced tool-calling in Qwen2.5-3B to a compact steerable feature set — the complement of our finding about which features abliteration *cannot* remove. Li et al. [10] demonstrate an ≈8× MSE gap between base-SAE reconstruction on instruct versus purpose-trained instruct SAE activations, motivating the reconstruction quality gate we implement.

**Safety fine-tuning mechanisms.** Jain et al. [8] establish that safety fine-tuning projects unsafe activations into MLP null spaces, explaining adversarial circumvention. Our finding extends this picture: at the SAE encoder level, the directions RLHF modifies are near-orthogonal to the refusal direction, so abliteration — which removes the refusal direction — cannot erase them.

# Methodology

## Three-Model Setup

All experiments use three Qwen3-8B checkpoints sharing identical 36-layer architecture (d_model = 4096, bfloat16, transformers ≥ 4.51.0) [ARTIFACT:art_pmAcY0CNazP9]:

*Base*: `Qwen/Qwen3-8B-Base` — the unaligned pretrained model, providing the representational baseline.

*Instruct*: `Qwen/Qwen3-8B` with `enable_thinking=False` — the safety RLHF-aligned model. Chain-of-thought suppression isolates safety-relevant activations on the final input token.

*Abliterated*: `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` (v2 corrects a layer-0 output-garbling artifact present in v1). The exact layer set and prompt set used for the refusal direction computation are not publicly documented; verification against a scratch-reproduced checkpoint is deferred to future work.

All models are accessed via `model.model.layers` for residual-stream hook injection at the final token position of each input.

## SAE Measurement Instrument

The frozen Qwen-Scope SAE `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` serves as the shared measurement instrument [9]. [ARTIFACT:art_pmAcY0CNazP9] A critical architectural note: Qwen-Scope implements *TopK* architecture (k = 50 features retained via `torch.topk`), not JumpReLU. For a residual activation $\mathbf{x} \in \mathbb{R}^{4096}$:

$$\mathbf{z}_{\text{pre}} = \mathbf{x}W_{\text{enc}}^\top + \mathbf{b}_{\text{enc}} \in \mathbb{R}^{65536}$$
$$\mathbf{z} = \text{TopK}(\mathbf{z}_{\text{pre}},\; k{=}50)$$

The *pre-topk* representation $\mathbf{z}_{\text{pre}}$ is a dense continuous projection of the residual stream onto all 65,536 encoder directions. Because the reconstruction quality gate (Section 3.5) fails at layer 22 (R² < 0.85 for all variants), all feature metrics use $\mathbf{z}_{\text{pre}}$.

## Feature Metrics

For each SAE feature $f$ and $N = 500$ harmful prompts $\{p_i\}_{i=1}^N$, let $a_m(f, p_i)$ denote the pre-topk encoder activation of feature $f$ for model $m \in \{\text{base}, \text{inst}, \text{abl}\}$.

**Safety Change** quantifies the magnitude of RLHF modification to feature $f$:
$$\text{SC}(f) = \frac{1}{N}\sum_{i=1}^N |a_{\text{inst}}(f, p_i) - a_{\text{base}}(f, p_i)|$$

Features are ranked by SC$(f)$: the top-500 form the *RLHF-sensitive* set. For null-baseline analysis we also select a *mid-500* group uniformly sampled from the 40th–60th SC percentile.

**Reversal Score** quantifies whether abliteration reverses or preserves the *direction* of the RLHF modification to feature $f$:
$$\boldsymbol{\Delta}_{\text{inst}}(f) = [a_{\text{inst}}(f, p_i) - a_{\text{base}}(f, p_i)]_{i=1}^N$$
$$\boldsymbol{\Delta}_{\text{abl}}(f) = [a_{\text{abl}}(f, p_i) - a_{\text{base}}(f, p_i)]_{i=1}^N$$
$$R(f) = \text{PearsonR}(\boldsymbol{\Delta}_{\text{inst}}(f), \boldsymbol{\Delta}_{\text{abl}}(f))$$

$R(f) \approx +1$ indicates abliteration preserves the RLHF modification prompt-by-prompt (*Type-I: imprinted*); $R(f) \approx -1$ indicates abliteration reverses it (*Type-R: refusal-coupled*).

**OLS Slope** quantifies whether abliteration preserves the *magnitude* of the RLHF modification to feature $f$. The Reversal Score is scale-invariant: a feature where abliteration retains only 5% of the RLHF-induced shift in magnitude but preserves its direction would still yield R(f) ≈ 1.0. To resolve this ambiguity, we compute the no-intercept ordinary least-squares slope:
$$\beta_f = \frac{\boldsymbol{\Delta}_{\text{inst}}(f)^\top \boldsymbol{\Delta}_{\text{abl}}(f)}{\boldsymbol{\Delta}_{\text{inst}}(f)^\top \boldsymbol{\Delta}_{\text{inst}}(f)}$$

$\beta_f \approx 1$ means the abliterated model's feature activations shift by the same magnitude as the instruct model's, relative to base. $\beta_f \approx 0$ means only the directional pattern survives with negligible magnitude. $\beta_f > 1$ indicates the abliterated model overshoots the RLHF shift. The OLS slope and Reversal Score together provide a complete characterization: R(f) measures directional alignment, $\beta_f$ measures amplitude fidelity. [ARTIFACT:art_iDkiRSdMuI6M]

## Benign-SC Overlap

To assess whether the top-500 RLHF-sensitive features (selected by harmful-prompt Safety Change) reflect *harm-specific* modifications or *general RLHF* modifications (style, tone, response format), we separately compute SC$_{\text{benign}}(f)$ using the 500 benign prompts and report the overlap between the top-500 by harmful-SC and the top-500 by benign-SC.

## Layer Selection and Reconstruction Quality Gate

**Layer selection.** We compute SC$(f)$ variance across all 65,536 features at candidate layers {14, 20, 22} on 50 harmful prompts. Layer 22 is selected empirically (variance 0.1721 vs. 0.0547 at layer 20 and 0.0127 at layer 14). [ARTIFACT:art_QBm3I0Z5VVJR] All primary results use layer 22; layer 20 serves as a replication layer (Section 5.7).

**Reconstruction quality gate.** At layer 22, R² = 0.574 (base), 0.512 (instruct), 0.512 (abliterated) — all below the 0.85 threshold [10]. All feature metrics therefore use $\mathbf{z}_{\text{pre}}$. [ARTIFACT:art_QBm3I0Z5VVJR]

## Semantic Labeling Protocol

For the top-100 features by SC$(f)$, we identify the 5 prompts with the highest pre-topk encoder activation and provide the first 300 characters of each prompt to Llama-3.1-8B-Instruct via OpenRouter. The cross-family labeler avoids within-Qwen circularity. Each feature is classified as `harm_concept`, `safety_refusal`, `general_capability`, or `other`. *Note:* the labeling uses character-prefix truncation of the most-activating prompts rather than per-token activation windows; labels reflect overall prompt topic more than token-level representations. Cohen's κ validation against a manual gold set was not performed and is required before the ≈75% harm-concept rate is trusted quantitatively.

# Evaluation Corpus

The evaluation corpus comprises 500 harmful and 500 benign prompts from seven publicly licensed datasets. [ARTIFACT:art_M8-flaZ0OOaP]

**Harmful sources:** BeaverTails [15] (176 prompts; NeurIPS 2023, human-labeled QA spanning 14 harm categories), AdvBench (98 prompts; standard adversarial behavior dataset used in mechanistic abliteration research [1]), TrustAIRLab in-the-wild jailbreak prompts (86 prompts; real-world jailbreaks from ACM CCS 2024), HarmBench [17] (85 prompts; covering chemical, cybercrime, physical harm, and illegal categories), and JailbreakBench/JBB-Behaviors [16] (55 prompts; the only source offering matched harmful/benign pairs).

**Benign sources:** Jailbreak-classification (151 prompts; labeled benign user requests), JBB-Behaviors benign counterparts [16] (46 prompts), Toxic-Chat (104 prompts; real user prompts labeled non-toxic), TrustAIRLab regular prompts (93 prompts), and Dolly-15k [18] (106 prompts; human-written instruction-following tasks).

All sources are processed through MinHash LSH deduplication (threshold 0.5, 128 permutations, character 5-gram shingling), token-length filtering (≤512 tokens), and category capping (≤35% per harm subcategory). The final corpus contains 1,000 prompts (500 harmful, 500 benign).

# Results

## Layer Selection

Figure 4 shows the Safety Change variance across candidate layers. Layer 22 exhibits the highest variance (0.1721), confirming it as the primary analysis layer. The R² reconstruction gate at layer 22 yields values below 0.85 for all three model variants (base: 0.574, instruct: 0.512, abliterated: 0.512), triggering the pre-topk encoder-only analysis path.

[FIGURE:fig4]

## Reversal Score Distribution

Figure 2(a) shows the Reversal Score distribution for the top-500 RLHF-sensitive features at layer 22. Every feature has a positive Reversal Score (mean = 0.921, std = 0.040, min = 0.684, max = 0.989). No feature has a negative or near-zero Reversal Score. Hartigan's dip test yields statistic = 0.0123, $p = 0.930$ — strongly unimodal. Spearman correlation yields mean = 0.913, confirming rank-based robustness.

Table 1 summarizes Type-I and Type-R counts across all tested thresholds. Zero features are classified Type-R at any threshold.

**Table 1.** Type-I and Type-R feature counts at all tested thresholds.

| Threshold | Type-I (R > threshold) | Type-R (R < −threshold) |
|-----------|------------------------|-------------------------|
| 0.30 | 500 / 500 (100%) | 0 / 500 (0%) |
| 0.50 | 500 / 500 (100%) | 0 / 500 (0%) |
| 0.70 | 499 / 500 (99.8%) | 0 / 500 (0%) |
| 0.95 | 125 / 500 (25%) | 0 / 500 (0%) |

[FIGURE:fig2]

## OLS Slope: Magnitude Is Preserved, Not Just Direction

A key limitation of the Reversal Score is scale-invariance: R(f) ≈ 1 is consistent with abliteration preserving only 5% of the RLHF-induced magnitude while retaining the direction pattern. The OLS slope β_f resolves this ambiguity. [ARTIFACT:art_iDkiRSdMuI6M]

Figure 2(b) shows the β_f distribution for the top-500 RLHF-sensitive features at layer 22. The distribution is tightly concentrated near 1: mean β̄ = 0.986 ± 0.176, median = 0.985, range [−0.055, 1.725]. Specifically, 86.4% of features have β_f > 0.8 (abliteration preserves more than 80% of the RLHF-induced shift), 45.4% have β_f > 1.0 (slight amplification), and only 0.2% are negative. **MAGNITUDE_PRESERVED.** Abliteration does not attenuate the RLHF imprint; it preserves the imprint at approximately unit scale. Layer-20 replication: β̄ = 0.954 ± 0.191, with 80.2% of features having β_f > 0.8 and zero negative values. The magnitude result directly supports framing this as a *comprehensive* latent safety imprint: the abliterated model's SAE feature activations are quantitatively indistinguishable from the instruct model's in the RLHF-sensitive subspace.

## Null-Baseline Validation: Noise-Free Mid-SC Comparison

A potential concern with the bottom-500 RLHF-insensitive features as a null baseline is that their Safety Change values approach zero, making PearsonR between two near-zero vectors potentially unreliable. We address this with a noise-free mid-SC baseline: 500 features drawn uniformly from the 40th–60th SC percentile, with SC_median = 1.12 (versus the top-500 minimum of 2.79 and bottom-500 median of 0.66). [ARTIFACT:art_iDkiRSdMuI6M]

For completeness, we note that the bottom-500 SC_median = 0.66, which is itself well above the 0.01 threshold for reliable PearsonR computation — the noise concern identified by the reviewer does not apply to our bottom-500 comparison either. Nevertheless, the mid-SC baseline offers a cleaner control.

At layer 22: top-500 R̄ = 0.921, mid-500 R̄ = 0.871 (Cohen's d = 1.087, two-sample t-test t = 17.2, p < 10⁻³⁰⁰). **MID_SC_CONFIRMS_SELECTIVE_PRESERVATION.** The top-500 features show significantly higher Reversal Scores than mid-SC controls that have non-trivial RLHF sensitivity and noise-free Δ_inst vectors, confirming that the high Reversal Scores reflect genuine selective preservation rather than any measurement artifact. Figure 2(c) shows the full comparison across all three baseline groups.

## Direct Measurement of Refusal-Direction Orthogonality

The geometric interpretation of zero Type-R features and β̄ ≈ 1 requires a testable prediction: the refusal direction $\mathbf{r}$ must be nearly orthogonal to all RLHF-modified SAE encoder directions. We test this directly. [ARTIFACT:art_Qnl1zSSZPJvj]

We compute $\mathbf{r}$ as the mean difference of instruct residual-stream activations on harmful versus harmless prompts at layer 22. For each of the top-500 RLHF-sensitive features $f$:
$$\cos\theta_f = \frac{\mathbf{r} \cdot \mathbf{w}_{\text{enc}}(f)}{\|\mathbf{r}\| \|\mathbf{w}_{\text{enc}}(f)\|}$$

**Layer 22:** mean $|\cos\theta_f| = 0.01854$ over the top-500 RLHF-sensitive encoder directions (authoritative value; the paper draft rounded this to 0.019; the value 0.0145 appearing in a prior hypothesis summary reflects a different feature set from an intermediate iteration). **ORTHOGONALITY_CONFIRMED.** A cosine of 0.0185 means the refusal direction projects onto less than 1.9% of each RLHF-modified encoder direction's magnitude. **Layer 20:** mean $|\cos\theta_f| = 0.02086$. ORTHOGONALITY_CONFIRMED at both layers. The inter-layer agreement (iter3 and iter4 measurements match to 5 decimal places) confirms these values are stable.

This orthogonality explains the β̄ ≈ 1 result mechanistically: abliteration orthogonalizes all weight matrices against $\mathbf{r}$. Because $\mathbf{r}$ is nearly orthogonal to all RLHF-modified SAE encoder directions, the orthogonalization cannot substantially change how any RLHF-modified feature's activations are computed — in direction or in magnitude.

## Benign-SC Overlap: What the Preserved Imprint Represents

We compute SC$_{\text{benign}}(f)$ — Safety Change evaluated on the 500 benign prompts — for all 65,536 SAE features, then measure the overlap between the top-500 by harmful-SC and the top-500 by benign-SC. [ARTIFACT:art_iDkiRSdMuI6M]

**Overlap count: 323 / 500 = 64.6%. STYLE_DOMINANT verdict.** A majority of the top-500 RLHF-sensitive features appear in both the harmful-SC and benign-SC rankings, indicating that RLHF modified these features similarly when processing harmful and benign prompts. This suggests that approximately 64.6% of the preserved RLHF imprint reflects *general* RLHF modifications — response formality, output structure, instruction-following style — rather than harm-specific semantic content. The remaining 35.4% (176 features) are uniquely sensitive to harmful prompts and more plausibly encode harm-concept semantic representations.

This finding contextualizes the OLS slope result (β̄ = 0.986): abliteration comprehensively preserves all RLHF modifications regardless of whether they are harm-semantic or stylistic. The near-orthogonality of the refusal direction to all RLHF-modified encoder directions means abliteration cannot selectively erase harm-semantic changes while leaving style changes intact — it fails to erase either category. The ≈75% harm-concept semantic labeling rate in the top-100 features is consistent with the 35.4% uniquely-harmful selection being concentrated in harm-concept categories; stylistic features may receive harm-concept labels because the labeling prompt uses harmful prompts to identify their most-activating context.

## Centroid Analysis with Control Subspaces

To validate that the latent safety imprint is specific to the RLHF-sensitive subspace, we compute centroid distance ratios $d_{\text{abl-base}} / d_{\text{abl-inst}}$ (cosine) for three feature subspaces. [ARTIFACT:art_Qnl1zSSZPJvj]

[FIGURE:fig3]

**Layer 22 (harmful prompts):** Subspace A (top-500 RLHF-sensitive) ratio = 4.35×, Subspace B (bottom-500 RLHF-insensitive) ratio = 1.16×, Subspace C (random-500) ratio = 2.37×. **Layer 20:** Subspace A ratio = 7.31×, Subspace B ratio = 1.01×, Subspace C ratio = 1.63× (verdict: CENTROID_SELECTIVE). The RLHF-sensitive subspace shows 4–7× greater instruct-closeness than the RLHF-insensitive control (B ≈ 1.0–1.2 vs. A = 4.35–7.31), confirming that the imprint is specific to the features RLHF modified, not a global consequence of model similarity.

## Semantic Labeling

Llama-3.1-8B-Instruct assigned labels to 100 top-SC features: `harm_concept` ≈ 75%, `safety_refusal` ≈ 5%, `general_capability` ≈ 5%, `other` ≈ 15%. Because the Reversal Score distribution is unimodal with zero Type-R features, the pre-registered Fisher's exact test comparing harm-concept rates in Type-I versus Type-R groups cannot be computed. Because 64.6% of top-500 features are style-dominant (Section 5.5), the ≈75% harm-concept labeling rate should be interpreted cautiously: the labeling is based on harmful-prompt activation and may overstate harm-specificity.

## Downstream Classification Accuracy

A per-prompt harm score (mean pre-topk activation across all top-500 RLHF-sensitive features with median threshold) achieves **0.476 accuracy** — near-chance on the balanced 500/500 corpus — versus a keyword baseline of **0.630**. [ARTIFACT:art_QBm3I0Z5VVJR] This null classification result is an expected consequence of the primary finding: the RLHF-sensitive features are elevated across *all* prompts (harmful and benign alike, as confirmed by the 64.6% benign-SC overlap), so they carry no harmful-vs.-benign discrimination signal at the prompt level. The Reversal Score and OLS slope measure *what RLHF modified and abliteration preserved*, which is distinct from *what selectively fires on harmful versus benign prompts at inference*.

## Layer-20 Replication

Table 2 summarizes the full quantitative replication at layer 20 alongside primary layer-22 results.

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

**REPLICATION_CONFIRMED** at both layers. The layer-20 centroid pattern is cleaner (CENTROID_SELECTIVE: B = 1.01, essentially no instruct-closeness in the RLHF-insensitive subspace).

# Discussion

## Geometric and Magnitude Interpretation of the Comprehensive Imprint

The OLS slope result (β̄ = 0.986, 86.4% of features with β_f > 0.8) establishes that the label *comprehensive* latent imprint is accurate in the strongest possible sense: abliteration does not merely preserve the directional pattern of RLHF-induced feature changes — it preserves the amplitude as well. The imprint is approximately unit-magnitude, not attenuated.

The direct orthogonality measurement (|cos| = 0.0185) provides the mechanistic account bottom-up. Abliteration removes $\mathbf{r}$ from all weight matrices. For abliteration to change the activation of SAE feature $f$ by even 10%, $\mathbf{r}$ would need to project onto $\mathbf{w}_{\text{enc}}(f)$ with |cos| ≥ 0.1. The measured mean |cos| = 0.0185 falls 5× below this threshold, explaining why both direction and magnitude are preserved: the orthogonalization barely touches the RLHF-modified encoder directions. This interpretation is consistent with Cristofano's [7] observation that the raw refusal direction is polysemantic, primarily capturing stylistic compliance patterns rather than semantic harm-concept content.

## What the Preserved Imprint Encodes

The 64.6% benign-SC overlap finding requires careful interpretation. It shows that a majority of RLHF-sensitive features are modified similarly when processing harmful and benign prompts, implying these features capture *general* RLHF behavioral changes (formality, helpfulness, instruction-following structure) in addition to harm-specific semantics. The surviving 35.4% uniquely-harmful features are the more plausible carriers of harm-concept semantic content.

Crucially, this finding does not weaken the headline result — it sharpens it. The near-orthogonality of the refusal direction to all RLHF-modified encoder directions (harmful-specific and style-related alike) means abliteration geometrically cannot remove any category of RLHF modification. The behavioral regression to base-like output arises because the compliance mechanism (the refusal direction itself) is removed, not because the underlying representations are erased. The model forgets how to express refusal while retaining all representational changes RLHF introduced.

This also explains the downstream classification null result (accuracy = 0.476). The RLHF-sensitive features are elevated on *both* harmful and benign prompts (confirming the 64.6% overlap from a different angle), so no harm-vs.-benign discrimination is possible from their mean activation. The representational change RLHF introduced is a broad shift in how the model encodes inputs, not a selective harm-detector.

## Implications for Safety Restoration

If abliteration preserves the RLHF imprint at approximately unit magnitude, the challenge for targeted safety restoration is not to amplify surviving features — they are already at full instruct-model activation levels. The challenge is to restore the *compliance mechanism*: the geometric pathway from current activations to refusal outputs, implemented by the refusal direction $\mathbf{r}$. Restoration requires re-injecting $\mathbf{r}$ into the weight matrices without disturbing the already-intact feature-level representations.

This reframes the restoration problem as geometric re-injection rather than semantic amplification. The DFC approach of Shportko et al. [20] identifies compact steerable feature sets for RL-induced capabilities; an analogous method targeting the features that mediate harm-concept-to-refusal conversion could enable targeted re-injection. The 35.4% uniquely-harmful features (those in the top-500 harmful-SC but not benign-SC) are the most promising candidates for such targeted intervention.

## Limitations

**Pre-topk encoder projections.** All feature metrics use $\mathbf{z}_{\text{pre}}$ (dense encoder projections) rather than the post-topk sparse activations validated as interpretable by Bricken et al. [12] and Templeton et al. [14]. Pre-topk values are dense by construction — every feature has a non-zero value on every prompt — which means they do not represent the on/off firing concept that makes individual SAE features semantically tractable. The OLS slope (β̄ = 0.986) and Reversal Score (R̄ = 0.921) capture real geometric signals confirmed by the null-baseline comparison and direct orthogonality measurement, but the semantic claims (≈75% harm-concept labeling) rest on an assumption that pre-topk encoder directions retain semantic alignment with post-topk features — an assumption not directly validated for TopK SAEs in the cross-variant setting.

**Community abliterated checkpoint.** The `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` checkpoint was used without independently verifying abliteration depth on geometric metrics. If the community checkpoint applies non-standard orthogonalization producing weaker abliteration, high Reversal Scores and high OLS slopes could partially reflect incomplete abliteration. Verification against a scratch-reproduced checkpoint with documented parameters is critical for the next iteration.

**Single model family and layers.** All results are for Qwen3-8B at layers 22 and 20. The zero-Type-R finding and β̄ ≈ 1 result may not generalize to other model families (Llama, Gemma) or other architectures.

**Benign-SC overlap and semantic interpretation.** The 64.6% benign-SC overlap limits confident claims that the preserved RLHF imprint is harm-specific. The ≈75% harm-concept labeling rate from the LLM labeler may be inflated because the labeling prompt uses harmful prompts as context. A manual gold-set annotation (Cohen's κ) is required before quantitative reliance on the 75% figure.

# Conclusion

We present the first feature-level mechanistic analysis of refusal abliteration using the frozen Qwen-Scope base SAE across three Qwen3-8B variants. Two complementary per-feature metrics — the Reversal Score (directional preservation) and the OLS Slope (magnitude preservation) — together establish a comprehensive latent imprint: every top-500 RLHF-sensitive SAE encoder feature has a positive Reversal Score (mean R̄ = 0.921) and an OLS slope near unity (β̄ = 0.986 ± 0.176, with 86.4% exceeding 0.8), confirming that abliteration preserves both the direction and the amplitude of RLHF-induced feature changes. A noise-free mid-SC baseline (SC_median = 1.12, R̄ = 0.871, Cohen's d = 1.087) confirms selective preservation without noise concerns. Direct measurement of refusal-direction geometry (|cos| = 0.0185 at layer 22, 0.0209 at layer 20) provides the mechanistic explanation: abliteration's orthogonalization is geometrically incapable of touching the RLHF-modified feature subspace. A 64.6% benign-SC overlap reveals that the preserved imprint spans both harm-semantic and general RLHF stylistic modifications — abliteration fails to erase any category of RLHF change because all reside in a subspace near-orthogonal to the refusal direction. The abliterated model's behavior reverts to base-like, but its internal representations remain instruct-like in full magnitude — a dissociation that reframes safety restoration as geometric re-injection of the compliance direction.

**Future directions:** (1) Verification against a scratch-reproduced abliterated checkpoint with fully documented parameters. (2) Multi-layer analysis across all 36 Qwen-Scope layers. (3) Cross-family extension to Llama-3.1-8B and Gemma-3-9B. (4) Analysis using an instruct-trained Qwen3-8B SAE to enable post-topk analysis and validate semantic labeling claims. (5) Investigation of the 35.4% uniquely-harmful features as candidates for targeted geometric restoration.

# References

[1] A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Rimsky, W. Gurnee, and N. Nanda. Refusal in language models is mediated by a single direction. In *Advances in Neural Information Processing Systems*, 2024.

[2] I. Llorente-Saguer. The geometry of harmful intent: Training-free anomaly detection via angular deviation in LLM residual streams. *arXiv preprint arXiv:2603.27412*, 2026.

[3] I. Llorente-Saguer. Harmful intent as a geometrically recoverable feature of LLM residual streams. *arXiv preprint arXiv:2604.18901*, 2026.

[4] R. Chopra. A mechanistic investigation of supervised fine tuning. *arXiv preprint arXiv:2605.11426*, 2026.

[5] T. Jiralerspong and T. Bricken. Cross-architecture model diffing with crosscoders: Unsupervised discovery of differences between LLMs. *arXiv preprint arXiv:2602.11729*, 2026.

[6] A. M. Kassem, T. Jiralerspong, N. Rostamzadeh, and G. Farnadi. Delta-Crosscoder: Robust crosscoder model diffing in narrow fine-tuning regimes. *arXiv preprint arXiv:2603.04426*, 2026.

[7] T. Cristofano. Surgical refusal ablation: Disentangling safety from intelligence via concept-guided spectral cleaning. *arXiv preprint arXiv:2601.08489*, 2026.

[8] S. Jain, E. S. Lubana, K. Oksuz, T. Joy, P. H. S. Torr, A. Sanyal, and P. K. Dokania. What makes and breaks safety fine-tuning? A mechanistic study. In *Advances in Neural Information Processing Systems*, 2024.

[9] B. Deng, X. Wang, Y. Wang, Y. Wan, Y. Ma, B. Yang, H. Wei, J. Tang, H. Lin, R. Gao, T. Li, Q. Cao, X. Ren, X. Deng, A. Yang, F. Huang, D. Liu, and J. Zhou. Qwen-Scope: Turning sparse features into development tools for large language models. *arXiv preprint arXiv:2605.11887*, 2026.

[10] J. Li, H. Ye, Y. Chen, X. Li, L. Zhang, H. Alinejad-Rokny, J. C. H. Peng, and M. Yang. Training superior sparse autoencoders for instruct models. *arXiv preprint arXiv:2506.07691*, 2025.

[11] N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds, R. Lasenby, D. Drain, C. Chen, R. Grosse, S. McCandlish, J. Kaplan, D. Amodei, M. Wattenberg, and C. Olah. Toy models of superposition. *arXiv preprint arXiv:2209.10652*, 2022.

[12] T. Bricken, A. Templeton, J. Batson, B. Chen, A. Jermyn, T. Conerly, N. Turner, C. Anil, C. Denison, A. Askell, R. Lasenby, Y. Wu, S. Kravec, N. Schiefer, T. Maxwell, N. Joseph, A. Tamkin, K. Nguyen, B. McLean, J. E. Burke, T. Hume, S. Carter, T. Henighan, and C. Olah. Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*, Anthropic, 2023.

[13] L. Gao, T. Dupré la Tour, H. Tillman, G. Goh, R. Troll, A. Radford, I. Sutskever, J. Leike, and J. Wu. Scaling and evaluating sparse autoencoders. *arXiv preprint arXiv:2406.04093*, 2024.

[14] A. Templeton, T. Conerly, J. Marcus, J. Lindsey, T. Bricken, B. Chen, A. Pearce, C. Citro, E. Ameisen, A. Jones, H. Cunningham, N. Turner, C. McDougall, M. MacDiarmid, A. Tamkin, E. Durmus, T. Hume, F. Mosconi, C. D. Freeman, T. Sumers, E. Rees, J. Batson, A. Jermyn, S. Carter, C. Olah, and T. Henighan. Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Transformer Circuits Thread*, Anthropic, 2024.

[15] J. Ji, M. Liu, J. Dai, X. Pan, C. Zhang, C. Bian, R. Sun, Y. Wang, and Y. Yang. BeaverTails: Towards improved safety alignment of LLM via a human-preference dataset. In *Advances in Neural Information Processing Systems*, 2023.

[16] P. Chao, E. Debenedetti, A. Robey, M. Andriushchenko, F. Croce, V. Sehwag, E. Dobriban, N. Flammarion, G. Pappas, F. Tramèr, H. Hassani, and E. Wong. JailbreakBench: An open robustness benchmark for jailbreaking large language models. In *Advances in Neural Information Processing Systems*, 2024.

[17] M. Mazeika, L. Phan, X. Yin, A. Zou, Z. Wang, N. Mu, E. Sakhaee, N. Li, S. Basart, B. Li, D. Forsyth, and D. Hendrycks. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. In *International Conference on Machine Learning*, pp. 35181–35224, 2024.

[18] M. Conover, M. Hayes, A. Mathur, J. Meng, S. Mody, N. Sherwood, V. Sreedhar, B. Wan, and Databricks. Free Dolly: Introducing the world's first truly open instruction-tuned LLM. *Databricks Blog*, 2023.

[19] G. Nanfack, E. Belilovsky, and E. Dohmatob. Efficient refusal ablation in LLM through optimal transport. *arXiv preprint arXiv:2603.04355*, 2026.

[20] A. Shportko, S. Bhokare, A. Z. A. Alzahrani, B. Cheng, G. Mercier, and J. Hullman. Localizing RL-induced tool use to a single crosscoder feature. In *ICML 2026 Workshop on Mechanistic Interpretability* (Spotlight), 2026.
</current_paper>

<reviewer_feedback>
Paper reviewer feedback from the previous iteration. Your strategy MUST address these critiques.
Prioritize major issues — these are the most impactful improvements to make.

- [MAJOR] (rigor) Pre-topk encoder projections (z_pre) are used as a proxy for SAE features throughout because R² < 0.85 at layer 22. However, the TopK activation is the architectural mechanism that enforces sparsity — without it, z_pre is a dense continuous projection onto all 65,536 encoder directions, not a set of near-monosemantic features. Every prompt gets non-zero values for all 65,536 'features' simultaneously. The high Reversal Scores (R̄ = 0.921) and OLS slopes (β̄ = 0.986) computed from z_pre could reflect global geometry of the encoder weight matrix shared by all three model variants rather than the preservation of semantically interpretable features. The paper acknowledges this in limitations, but the semantic labeling section (≈75% harm-concept) and the discussion framing ('the abliterated model's SAE feature activations are quantitatively indistinguishable from the instruct model's') treat z_pre as if it were the validated sparse post-topk representation. This gap between what is measured (dense projections) and what is claimed (feature-level preservation) is load-bearing for the paper's main narrative.
  Action: Add a validation experiment: on the subset of prompts and features where R² is highest (even if < 0.85), compare top-500 features selected by SC(z_pre) versus SC(z_post). Report the overlap fraction. Additionally, report the correlation between the z_pre Reversal Scores and a z_post Reversal Score computed on the post-topk nonzero subset. If the z_pre and z_post rankings substantially agree (Spearman ρ > 0.7), the pre-topk proxy is defensible; if they diverge, all semantic claims must be clearly reframed as 'geometric subspace properties of the encoder' rather than 'feature-level properties.'
- [MAJOR] (evidence) The abliterated checkpoint (huihui-ai/Huihui-Qwen3-8B-abliterated-v2) is used without any behavioral or geometric verification that abliteration succeeded at the expected depth. The paper acknowledges this in limitations, but does not report: (a) refusal rate of the abliterated model on AdvBench or HarmBench; (b) whether the refusal direction r computed from instruct activations has a small projection onto the residual-stream outputs of the abliterated model (i.e., whether Proj_r(x_abl) ≈ 0); or (c) whether the v2 fix for the layer-0 garbling artifact fully resolves the issue for layers 20 and 22. If abliteration is incomplete — even partially — then high R̄ and β̄ ≈ 1 trivially follow because the abliterated model is close to the instruct model by construction. This is a critical experimental control that is currently absent.
  Action: Report at minimum: (1) the fraction of AdvBench prompts where the abliterated checkpoint produces a non-refusal response (expected near 100% for successful abliteration); (2) mean |Proj_r(h_abl(p))|/|h_abl(p)| at layer 22 for the 500 harmful prompts, expected near zero. If abliteration is confirmed complete, state this explicitly in the methods. If the checkpoint cannot be verified, add a sentence in the abstract noting the finding is conditional on abliteration completeness.
- [MAJOR] (novelty) Three directly relevant papers are missing from the related work section. (1) 'Understanding Refusal in Language Models with Sparse Autoencoders' (arXiv:2505.23556) finds that SAEs trained on pretraining data fail to encode refusal-relevant features in chat-tuned models — precisely the regime in which the paper's frozen base-SAE (Qwen-Scope) is applied. This finding directly bears on whether the top-500 RLHF-sensitive features identified via z_pre carry safety-relevant semantic content. (2) 'CRaFT: Circuit-Guided Refusal Feature Selection via Cross-Layer Transcoders' (arXiv:2604.01604) uses SAE features to analyze refusal circuitry and is the most closely related contemporaneous work in approach — failing to differentiate from it is a gap reviewers will notice. (3) 'There Is More to Refusal in Large Language Models than a Single Direction' (arXiv:2602.02132) challenges the single-direction framing central to the paper's mechanistic account (r as a single vector); the paper's claim that |cos(r, w_enc)| ≈ 0.019 explains zero Type-R features is predicated on abliteration removing only one direction, but if refusal involves a subspace, this orthogonality measurement may be incomplete.
  Action: Add a paragraph in Related Work covering these three papers: (1) cite 2505.23556, note its finding about base SAEs and refusal, and explain why z_pre projections are still informative for identifying RLHF-modified features generally (not specifically refusal features); (2) cite 2604.01604, contrast the circuit-based SAE approach with the frozen-base-SAE metric approach — both are feature-level but use different feature vocabularies and answer different questions; (3) cite 2602.02132, acknowledge that refusal may span a multi-dimensional subspace, and either (a) extend the orthogonality measurement to the top-k principal components of the refusal subspace, or (b) note this as a limitation and future direction.
- [MINOR] (scope) All results are for Qwen3-8B at two layers of a single SAE. The paper's abstract-level claims ('abliteration preserves the RLHF imprint in both direction and magnitude') are stated without model-family qualification. The companion geometry studies [2,3] cover 12 model variants; this paper covers one. The 64.6% benign-SC overlap and β̄ ≈ 1 could in principle be artifacts of the specific Qwen3-8B RLHF process or of the Qwen-Scope SAE's coverage.
  Action: Add a sentence in the abstract and conclusion qualifying the main claim to Qwen3-8B: 'We demonstrate for Qwen3-8B that abliteration preserves the RLHF imprint in both direction and magnitude...'. In future directions, mention that Llama-3.1-8B and Gemma-3-9B extensions would test generality — GemmaScope and Llama-Scope SAEs are publicly available and would require minimal additional infrastructure.
- [MINOR] (rigor) The paper computes the refusal direction r as the mean difference of instruct activations on harmful versus harmless prompts (a single vector). However, arXiv:2602.02132 demonstrates that refusal in LLMs corresponds to geometrically distinct directions across eleven behavioral categories, not a single shared vector. The abliteration orthogonalization may remove a refusal subspace rather than a single direction. If the refusal subspace has dimension > 1, the mean cosine |cos(r, w_enc)| = 0.0185 across top-500 features is sufficient but not necessary to explain zero Type-R features — the argument requires orthogonality to the full ablation subspace, not just its mean-difference summary vector.
  Action: Compute the top-3 principal components of the (harmful − harmless) activation differences at layer 22 and report mean |cos(PC_k, w_enc(f))| for k = 1, 2, 3 across the top-500 features. If all three PCs show |cos| < 0.05, the orthogonality argument is robust to multi-dimensional ablation. If a higher PC shows larger alignment, this limits the mechanistic explanation to single-direction abliteration variants.
- [MINOR] (clarity) The downstream classification null result (Section 5.8, 0.476 accuracy) is framed as an 'expected consequence' of the primary finding, but the logical connection to the 64.6% benign-SC overlap is stated only in the Discussion (Section 6.2). A reader encountering Section 5.8 in isolation will be confused: the paper presents a 'latent safety imprint' throughout, then reports that the corresponding features cannot distinguish harmful from benign prompts at near-chance. Without the Discussion cross-reference in place during reading, this appears to contradict rather than confirm the paper's thesis.
  Action: Add two sentences at the start of Section 5.8: 'This null result follows directly from the 64.6% benign-SC overlap (Section 5.5): features elevated by RLHF on harmful prompts are also elevated on benign prompts, eliminating prompt-level discriminability. This distinguishes *what RLHF modified* (broad representational changes) from *what selectively fires on harmful prompts* (a different, narrower feature set not analyzed here).' This converts a potential contradiction into a coherent narrative.
- [MINOR] (evidence) The semantic labeling (≈75% harm-concept across top-100 features) is based on Llama-3.1-8B-Instruct prompting without Cohen's κ validation against a manual gold set. The paper correctly identifies this as a limitation but still quantitatively interprets the 75% figure ('consistent with the 35.4% uniquely-harmful selection being concentrated in harm-concept categories'). The labeling uses character-prefix truncation of most-activating prompts rather than token-level activation windows, which may inflate the harm-concept rate because the most-activating prompts for stylistic features may still be harmful prompts. The 75% rate is used to contextualise the 64.6% benign-SC overlap finding, but without κ validation the figure cannot support quantitative conclusions.
  Action: Either (a) conduct a 50-item manual annotation and report Cohen's κ (estimated 2-3 hours of labor) before making quantitative claims; or (b) downgrade the labeling result from a quantitative claim ('≈75% harm-concept') to a qualitative observation ('the majority of manually-inspected top-SC features appeared to represent harm-concept semantics') and remove any downstream reasoning that depends on the specific 75% figure. Option (b) is immediate and sufficient for the paper's core claims.
</reviewer_feedback>

<task>
Generate 1 research strategy for THIS iteration.

**ARTIFACT LIMIT: Each strategy may contain AT MOST 5 artifact directions.** Focus on the highest-impact artifacts. Quality over quantity.

Each strategy should:
1. Define a clear OBJECTIVE - what novel contribution we're building toward
2. Plan artifacts to execute NOW - specify type, objective, approach, and depends_on for each
3. Account for parallel execution - all strategies and all planned artifacts run simultaneously, their artifacts are combined into one shared pool


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
    "ArtifactDep": {
      "description": "A single dependency on an existing artifact, with a short type label.\n\n``id`` and ``label`` are LLM-generated at strategy time. ``label`` is free-text but\nshort \u2014 a word or two naming the type of dependency, not a sentence.\n\n``relation_type`` and ``relation_rationale`` are populated later, in upd_hypo,\nusing the MultiCite citation-function typology (Lauscher et al., NAACL 2022).\nThey are absent at strategy time and may stay absent for legacy runs.",
      "properties": {
        "id": {
          "description": "ID of an existing artifact this artifact depends on",
          "title": "Id",
          "type": "string"
        },
        "label": {
          "description": "Short free-text label naming the type of this dependency (a word or two, not a sentence)",
          "title": "Label",
          "type": "string"
        }
      },
      "required": [
        "id",
        "label"
      ],
      "title": "ArtifactDep",
      "type": "object"
    },
    "ArtifactDirection": {
      "description": "High-level direction for an artifact to execute this iteration.\n\nID is code-assigned (LLMPrompt only \u2014 visible in prompts, not LLM-generated).",
      "properties": {
        "type": {
          "description": "Type of artifact to create",
          "enum": [
            "experiment",
            "research",
            "proof",
            "evaluation",
            "dataset"
          ],
          "title": "Type",
          "type": "string"
        },
        "objective": {
          "description": "What we want to achieve with this artifact",
          "title": "Objective",
          "type": "string"
        },
        "approach": {
          "description": "High-level direction/method",
          "title": "Approach",
          "type": "string"
        },
        "depends_on": {
          "description": "Existing artifacts this depends on, each with a short type label",
          "items": {
            "$ref": "#/$defs/ArtifactDep"
          },
          "title": "Depends On",
          "type": "array"
        }
      },
      "required": [
        "type",
        "objective",
        "approach"
      ],
      "title": "ArtifactDirection",
      "type": "object"
    },
    "Strategy": {
      "description": "A research strategy.\n\nContent fields have LLMPrompt + LLMStructOut markers.\n``id`` is code-assigned (LLMPrompt only \u2014 visible in prompts, not LLM-generated).\n\nID format: gen_strat_idx{N}",
      "properties": {
        "title": {
          "description": "Strategy name in plain, everyday language \u2014 short and jargon-free so a non-expert grasps it at a glance and it fits the run visualizations. Aim for about 4-8 words (~40 characters).",
          "title": "Title",
          "type": "string"
        },
        "objective": {
          "description": "The novel contribution we're building toward",
          "title": "Objective",
          "type": "string"
        },
        "rationale": {
          "description": "Why this strategy is promising",
          "title": "Rationale",
          "type": "string"
        },
        "artifact_directions": {
          "description": "Artifacts to execute THIS iteration",
          "items": {
            "$ref": "#/$defs/ArtifactDirection"
          },
          "title": "Artifact Directions",
          "type": "array"
        },
        "expected_outcome": {
          "description": "What we'll have after this iteration's artifacts complete",
          "title": "Expected Outcome",
          "type": "string"
        },
        "summary": {
          "default": "",
          "description": "Brief summary of the strategy and its expected contribution",
          "title": "Summary",
          "type": "string"
        }
      },
      "required": [
        "title",
        "objective",
        "rationale",
        "artifact_directions",
        "expected_outcome"
      ],
      "title": "Strategy",
      "type": "object"
    }
  },
  "description": "Top-level wrapper for LLM strategy generation output.",
  "properties": {
    "strategies": {
      "description": "List of generated strategies",
      "items": {
        "$ref": "#/$defs/Strategy"
      },
      "title": "Strategies",
      "type": "array"
    }
  },
  "required": [
    "strategies"
  ],
  "title": "Strategies",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-07-17 14:32:58 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [3] SYSTEM-USER prompt · 2026-07-17 14:34:34 UTC

```
<verification_results>
Your previous response had issues that need fixing:

DEPENDENCY ERRORS (depends_on can ONLY reference IDs from <existing_artifacts>):
  - Strategy 1: Artifact 'experiment_iter5_dir1' (experiment): dependency 'art_QBm3I0Z5VVJR' has type 'experiment' which is not allowed (allowed: {'dataset', 'research'})
  - Strategy 1: Artifact 'experiment_iter5_dir1' (experiment): dependency 'art_Qnl1zSSZPJvj' has type 'experiment' which is not allowed (allowed: {'dataset', 'research'})
  - Strategy 1: Artifact 'experiment_iter5_dir1' (experiment): dependency 'art_iDkiRSdMuI6M' has type 'experiment' which is not allowed (allowed: {'dataset', 'research'})

</verification_results>

<task>
Fix ALL issues above and regenerate your strategies:

1. Fix dependency errors:
   - depends_on is a list of {id, label} objects — every entry MUST have a non-empty short label
   - id can ONLY reference IDs from <existing_artifacts>
   - You CANNOT reference artifacts you are proposing in this strategy as dependencies (they all run in parallel)
   - Follow the dependency type rules (e.g., experiments require datasets)
   - If no suitable existing artifacts exist, use depends_on: []

Output the corrected JSON with the fixed strategies.
</task>
```
