# gen_strat_1 — test_idea

> Phase: `invention_loop` · round 4 · `gen_strat`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_strat_1` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-07-17 13:45:11 UTC

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
title: Abliteration Preserves Direction But Not Magnitude of RLHF SAE Features
hypothesis: |-
  RLHF safety fine-tuning and refusal abliteration operate on geometrically near-orthogonal subspaces in Qwen3-8B's residual stream. Three experimentally confirmed findings establish this: (1) Selective directional preservation — the Reversal Score (PearsonR of RLHF-induced vs. abliteration-induced change vectors per feature) for top-500 RLHF-sensitive SAE encoder features at layer 22 is unimodal with mean=0.921, std=0.040, min=0.684, and zero features with negative scores; the null-baseline comparison yields bottom-500 R̄=0.847 (Cohen's d=1.751, p≈0), confirming the high Reversal Scores reflect selective preservation rather than global model similarity. (2) Geometric orthogonality — mean |cos(r, w_enc(f))|≈0.014–0.019 across the top-500 RLHF-sensitive encoder directions, meaning abliteration's orthogonalization against the refusal direction r projects onto less than 2% of each RLHF-modified encoder direction's magnitude, mechanistically explaining zero Type-R features. (3) Centroid selectivity — the abliterated model's centroid in the RLHF-sensitive subspace is 4.35× closer to instruct than to base on harmful prompts (layer 22), versus 1.16× in the RLHF-insensitive control and 2.37× in the random control; at layer 20 the contrast is cleaner (7.31× vs. 1.01× vs. 1.63×). Both layers replicate consistently.

  A critical unresolved dimension now governs the primary interpretation: the Reversal Score (PearsonR) is scale-invariant. A feature where abliteration preserves only 5% of the RLHF-induced magnitude shift would still yield R≈1.0. The centroid analysis uses cosine distance, which is similarly direction-based. As a result, the confirmed finding is specifically 'directional pattern preservation' — the prompt-by-prompt rank ordering of activation changes is preserved — but 'magnitude preservation' (whether the abliterated model's activations are as far from the base as the instruct model's) has not yet been established. The primary required measurement for the next iteration is the OLS regression slope β_f: the slope of Δ_abl(f,·) ~ β_f · Δ_inst(f,·) (no intercept) for each top-500 feature. If β̄≈1, the RLHF imprint is preserved in both direction and magnitude ('comprehensive survival'); if β̄≪1 (e.g., β̄≈0.1), the finding should be reframed as 'the abliterated model inherits the instruct model's harm-encoding geometry but at attenuated magnitude.' Both outcomes are publishable and scientifically meaningful — the former supports a 'full latent imprint' framing, the latter supports a 'geometric echo' framing where abliteration shrinks but does not rotate the RLHF-modified feature activations.

  A second methodological concern requires resolution before the null-baseline comparison can be fully trusted. The bottom-500 RLHF-insensitive features are selected precisely because their Safety Change SC(f)≈0 — meaning both Δ_inst and Δ_abl consist of near-zero values. PearsonR between two near-zero vectors is numerically unreliable and may reflect noise-to-noise correlation from shared computation paths rather than genuine signal, potentially inflating R̄=0.847. If the bottom-500 features have median SC(f)<0.001, their Reversal Scores are not interpretable as meaningful preservation signal. The required fix is to construct a mid-SC baseline: 500 features drawn uniformly from the 40th–60th SC percentile, which have non-trivial RLHF sensitivity and interpretable Δ_inst vectors. If top-500 still shows d≈1.75 against this mid-SC baseline, the selective-preservation claim is robust. Alternatively, confirming that bottom-500 features have SC(f)>0.01 would address the concern directly by showing the Reversal Score computation is over non-negligible signals.

  Two minor issues require correction. First, a numerical discrepancy exists between the paper (|cos|=0.019 at L22, 0.021 at L20) and the artifact summary (|cos_enc_top500_mean|=0.0145 unspecified layer) — reconcile with the actual method.py output to determine the authoritative values. Second, SC(f) was computed on 500 harmful prompts only, so RLHF style changes (formality, verbosity, response structure) could contaminate the top-500 alongside harm-semantic features; the ≈75% harm-concept labeling rate suggests this is not catastrophic, but computing benign-prompt SC(f) and reporting the overlap between top-500 harmful-SC and top-500 benign-SC features would distinguish harm-specific from style-specific RLHF modifications. Low overlap (<20%) would confirm harm-specificity of the selected features.

  Semantic labeling finds approximately 75% of top-500 RLHF-sensitive features carry harm-concept labels uniformly across the Reversal Score range (the pre-registered Fisher test between Type-R and Type-I groups cannot be computed as zero features are Type-R). This is consistent with the directional-preservation finding: harm-concept representations pervade the entire RLHF-modified feature set and are not concentrated in a surviving minority. However, all semantic claims rest on pre-topk (dense) encoder projections and character-prefix truncation of the most-activating prompts rather than feature-specific activation windows — labels may reflect prompt topic more than precise feature-level semantics. Cohen's κ validation of the labeler against a manual gold set was not performed and is required before the 75% rate is trusted quantitatively.

  The downstream classification test achieved 0.476 accuracy (near-chance) versus a 0.630 keyword baseline. This is an expected consequence of zero Type-R features and a unimodal Reversal Score distribution: with no Type-R contrast set, the mean pre-topk activation across all top-500 features carries no harm-vs-benign discrimination signal. This null result reflects the distinction between 'which features RLHF modified' (captured by Reversal Score and SC) and 'which features selectively fire on harmful prompts at inference' (required for classification) — these are different questions.

  For the next iteration, four measurements are required in priority order: (1) OLS slope β_f distribution (mean, std, histogram) for the top-500 features — the β̄ value determines whether 'comprehensive survival' or 'attenuated directional imprint' is the correct framing. (2) Mid-SC baseline Reversal Scores to confirm the d=1.751 null-baseline comparison is not a noise artifact. (3) Benign-SC overlap analysis to confirm harm-specificity of the top-500 selection. (4) Numerical reconciliation of the |cos| discrepancy across artifact and paper, and HarmBench count verification. The language 'comprehensive latent safety imprint' should be replaced with 'directional latent safety imprint' throughout pending confirmation that β̄≈1; if β̄≈1 is confirmed, 'comprehensive' can be reinstated with appropriate precision (referring to all 500 features having positive scores, not unit-magnitude preservation).

  The geometric-orthogonality interpretation is the most secure result: mean |cos(r, w_enc)|<0.02 at both layers is a direct measurement explaining zero Type-R features by construction, independent of the magnitude-vs-direction ambiguity. The centroid selectivity (4–7× ratio in RLHF-sensitive vs. ~1× in RLHF-insensitive subspace) is also direction-based but confirmed robust to the global-similarity alternative. The open question is whether the instruct model's harm-encoding geometry persists at full strength or in attenuated form in the abliterated model — a question that the β_f distribution will answer.
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
  Same frame; adds magnitude-vs-direction distinction and mid-SC null-baseline requirement from MAJOR reviewer critiques.
_confidence_delta: unchanged
_key_changes:
- >-
  Core claim reframed from 'comprehensive survival' to 'directional pattern preservation': Reversal Score (PearsonR) and cosine-based
  centroid analysis are scale-invariant — R≈0.921 is consistent with abliteration preserving only the direction but not the
  magnitude of RLHF-induced shifts. The word 'comprehensive' is replaced by 'directional' pending OLS slope analysis.
- >-
  OLS slope β_f added as the primary required measurement for next iteration: slope of Δ_abl ~ β·Δ_inst per feature determines
  whether RLHF imprint survives at full magnitude (β̄≈1, 'full imprint') or is attenuated (β̄≪1, 'geometric echo'). Both outcomes
  are pre-specified as publishable with distinct framings.
- >-
  Null-baseline noise concern incorporated: bottom-500 features have SC(f)≈0 by construction, making their PearsonR values
  potentially noise-to-noise correlations rather than meaningful signal. Mid-SC baseline (40th–60th percentile) required as
  a robustness check; alternatively, confirm bottom-500 median SC(f)>0.01.
- >-
  Benign-SC overlap analysis added as required measurement to distinguish harm-semantic from style-semantic RLHF modifications
  in the top-500 selection.
- >-
  Numerical discrepancy between paper (|cos|=0.019/0.021 at L22/L20) and artifact summary (0.0145, layer unspecified) flagged
  as requiring explicit reconciliation with authoritative method.py output.
- >-
  Geometric-orthogonality finding elevated as most secure result: mean |cos(r, w_enc)|<0.02 at both layers is a direct mechanistic
  measurement independent of the magnitude ambiguity, explaining zero Type-R features by construction.
- >-
  Layer-20 replication confirmed: top-500 R̄=0.924, bottom-500 R̄=0.841, d≈1.69; centroid 7.31× vs 1.01× — the layer-20 centroid
  contrast is actually cleaner than layer-22.
- >-
  Cohen's κ validation of Llama-3.1-8B labeler against a manual gold set explicitly required before the ≈75% harm-concept
  rate is trusted quantitatively.
- >-
  HarmBench count discrepancy (85 vs 95 in different sections) flagged for reconciliation against final corpus JSON.
relation_type: evolution
</hypothesis>

<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for study design, proper baselines, and the evaluation/validity norms this field demands.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>

<iteration_status>
Current iteration: 4 of 5
Remaining (including this one): 2
</iteration_status>

<previous_strategies>
Strategies from the PREVIOUS iteration. You can CONTINUE these directions,
ADAPT based on what worked and what didn't in the artifacts produced, or PIVOT if results suggest a better path.

--- Strategy 1 ---
kind: strategy
id: gen_strat_1_idx1
title: Null Baseline + Orthogonality Validation
objective: >-
  Resolve all four MAJOR reviewer critiques by computing: (1) Reversal Scores for bottom-500 RLHF-insensitive features as
  a null baseline (decisive test: selective preservation vs. global similarity); (2) direct cosine similarity between the
  refusal direction r and RLHF-sensitive SAE encoder directions (direct orthogonality measurement); (3) centroid distances
  in control subspaces (RLHF-insensitive and random) to contextualize the 6.1×/17.6× ratios; and (4) full layer-20 Reversal
  Score analysis with reported numbers.
rationale: >-
  All four MAJOR critiques share one diagnosis: the current results are consistent with the null hypothesis that instruct
  and abliterated models are globally similar, making every encoder projection correlated regardless of RLHF sensitivity.
  The null-baseline comparison (bottom-500 by Safety_change) is the decisive test: if top-500 R values are significantly higher
  than bottom-500, selective preservation is confirmed; if both concentrate near R=1, the finding is a trivial artifact of
  global model similarity. The direct orthogonality measurement (cos_sim between r and W_enc rows) converts an inferred claim
  into a directly measured one. The control-subspace centroid analysis determines whether the 6.1×/17.6× ratios are specific
  to the RLHF-sensitive subspace. All these computations reuse activation extraction logic from scratch — the iter2 experiment
  ran in 4.3 minutes total — so the full suite fits comfortably in the 6-hour budget. The experiment also incorporates all
  paper-level corrections (semantic-labeling description fix, classification-accuracy disclosure, density-filter honest reporting,
  pre-topk/post-topk limitation) into its output JSON so GEN_PAPER_TEXT has everything needed.
artifact_directions:
- id: experiment_iter3_dir1
  type: experiment
  objective: >-
    Run the complete null-baseline validation suite: (1) Reversal Scores for bottom-500 RLHF-insensitive features vs. top-500,
    with t-test/KS-test/Cohen's d; (2) refusal direction r computation and cosine similarity to top-500 SAE encoder directions;
    (3) centroid distances in RLHF-insensitive and random control subspaces; (4) post-topk density characterization for top-500
    features; (5) full layer-20 Reversal Score analysis — producing definitive verdicts on selective preservation, geometric
    orthogonality, and centroid specificity.
  approach: |-
    Use UV with packages: transformers>=4.51.0, torch, huggingface_hub, scipy, numpy, diptest. All three Qwen3-8B checkpoints (Base, Instruct/enable_thinking=False, huihui-ai/Huihui-Qwen3-8B-abliterated-v2) are loaded sequentially with torch.cuda.empty_cache() between loads. Prompt corpus from art_M8-flaZ0OOaP (full_data_out.json): use all harmful and benign prompts.

    **Step 0 — Activation extraction at layers 22 and 20.** For each model, register a forward hook on model.model.layers[N].output for N in [20, 22]. Run all prompts in batches (batch_size=4, last-token hidden state). Save: base_act_L22.npy, inst_act_L22.npy, abl_act_L22.npy, base_act_L20.npy, inst_act_L20.npy, abl_act_L20.npy — each shape (n_prompts, 4096). Track harmful_idx and benign_idx separately.

    **Step 1 — SAE load and pre-topk projections.** Download Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 via hf_hub_download for layer22.sae.pt and layer20.sae.pt. Load W_enc (65536×4096), b_enc (65536), W_dec (4096×65536) for each layer. Compute at layer 22: z_pre_base = base_act_L22 @ W_enc22.T + b_enc22 (shape: n_prompts×65536), similarly for inst and abl. Compute SC(f) = mean_i |z_pre_inst[harmful_idx, f] - z_pre_base[harmful_idx, f]| for all 65536 features. Sort: top500_idx = argsort(SC)[-500:]; bottom500_idx = argsort(SC)[:500].

    **Step 2 — Null-baseline Reversal Scores (bottom-500).** For top-500 and bottom-500 features, compute per-feature Pearson correlation: delta_inst(f) = z_pre_inst[harmful_idx, f] - z_pre_base[harmful_idx, f]; delta_abl(f) = z_pre_abl[harmful_idx, f] - z_pre_base[harmful_idx, f]; R(f) = pearsonr(delta_inst(f), delta_abl(f))[0]. Compute for all 500 top features and all 500 bottom features. Report: top500_R_mean/std/min/max, bottom500_R_mean/std/min/max. Run scipy.stats.ttest_ind(top500_R, bottom500_R), scipy.stats.ks_2samp(top500_R, bottom500_R). Cohen's d = (mean_top - mean_bottom) / sqrt((var_top + var_bottom) / 2). Run diptest on both distributions. VERDICT: if ttest_p < 0.05 AND cohens_d > 0.5 → 'SELECTIVE_PRESERVATION_CONFIRMED'; elif ks_p < 0.05 → 'WEAK_SELECTIVE_PRESERVATION'; else → 'GLOBAL_SIMILARITY_ALTERNATIVE_PLAUSIBLE'.

    **Step 3 — Post-topk density characterization.** For each harmful prompt i, find the k=50 threshold: kth_val_i = np.partition(z_pre_inst[i, :], -50)[-50]. For each top-500 feature f, density(f) = mean(z_pre_inst[harmful_idx, f] >= kth_vals[harmful_idx]) where kth_vals[i] = kth_val_i. Report density_mean, density_median, density_min, density_std, fraction_above_0.10 (features where >10% of prompts would have this feature in top-50). This is the honest replacement for the trivially-100%-satisfied density filter.

    **Step 4 — Refusal direction computation.** r_raw = mean(inst_act_L22[harmful_idx], axis=0) - mean(inst_act_L22[benign_idx], axis=0). r = r_raw / norm(r_raw). Shape (4096,). This is the refusal direction in residual-stream space.

    **Step 5 — Cosine similarity: r vs. SAE encoder directions.** For each top-500 feature f: enc_dir_f = W_enc22[f, :] / norm(W_enc22[f, :]); cos_enc(f) = dot(r, enc_dir_f). Also dec_dir_f = W_dec22[:, f] / norm(W_dec22[:, f]); cos_dec(f) = dot(r, dec_dir_f). Report: distribution of |cos_enc| — mean, std, max; top-10 RLHF-sensitive features' individual |cos_enc| values. For bottom-500 features: compute |cos_enc| distribution as reference. VERDICT: if mean(|cos_enc_top500|) < 0.1 → 'ORTHOGONALITY_CONFIRMED'; elif < 0.2 → 'NEAR_ORTHOGONAL'; else → 'ORTHOGONALITY_WEAKENED'. Note if bottom-500 shows similar |cos_enc| (SAE-basis-wide property vs. RLHF-specific).

    **Step 6 — Control-subspace centroid distances.** Define three subspaces: A=top500_idx, B=bottom500_idx, C=random500_idx (np.random.seed(42), exclude A and B). For each subspace S and prompt_type (harmful=harmful_idx, benign=benign_idx): centroid_abl_S = mean(z_pre_abl[prompt_type, :][:, S], axis=0); centroid_inst_S = mean(z_pre_inst[prompt_type, :][:, S], axis=0); centroid_base_S = mean(z_pre_base[prompt_type, :][:, S], axis=0). cosine_dist(u, v) = 1 - dot(u,v)/(norm(u)*norm(v)). d_abl_inst = cosine_dist(centroid_abl_S, centroid_inst_S); d_abl_base = cosine_dist(centroid_abl_S, centroid_base_S); ratio = d_abl_base / d_abl_inst. Report 6 ratios in table: subspace × prompt_type. VERDICT: if ratio_A_harmful > 2× both ratio_B_harmful and ratio_C_harmful → 'CENTROID_SELECTIVE'; else → 'CENTROID_GLOBAL'.

    **Step 7 — Layer-20 full analysis.** Load W_enc20, b_enc20 from layer20.sae.pt. Compute z_pre at layer 20 for all models. SC_L20(f) for all 65536 features on harmful prompts. top500_L20_idx = argsort(SC_L20)[-500:]; bottom500_L20_idx = argsort(SC_L20)[:500]. Compute Reversal Scores for top-500 and bottom-500 at layer 20. Report: top500_L20_R_mean/std/min/max, n_negative_R_L20, bottom500_L20_R_mean/std, diptest_p_L20. State: 'zero Type-R replicates at layer 20' (bool: n_negative_R_L20 == 0).

    **Output format (method_out.json):** { null_baseline: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, bottom500_R_mean, bottom500_R_std, bottom500_R_min, bottom500_R_max, ttest_t, ttest_p, ks_stat, ks_p, cohens_d, verdict }, posttopk_density: { density_mean, density_median, density_min, density_std, fraction_above_0p10 }, orthogonality: { cos_enc_top500_mean_abs, cos_enc_top500_max_abs, cos_enc_top500_std, cos_dec_top500_mean_abs, cos_enc_bottom500_mean_abs, top10_feature_cos_enc (list of {feature_idx, sc, cos_enc, cos_dec}), verdict }, centroid_control: { subspaceA_harmful_ratio, subspaceA_benign_ratio, subspaceB_harmful_ratio, subspaceB_benign_ratio, subspaceC_harmful_ratio, subspaceC_benign_ratio, verdict }, layer20: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, n_negative_R, bottom500_R_mean, bottom500_R_std, diptest_p, zero_type_r_replicates }, paper_corrections: { semantic_labeling_correct: 'top-5 prompts by max pre-topk activation on last token, first 300 chars — not token window', density_filter_correct: 'pre-topk dense projections always nonzero; post-topk density fraction is honest measure', classification_accuracy: '0.476 near-chance (expected: zero Type-R means no discrimination signal)', fisher_test_action: 'move Q3/Q1 n=4 test to limitations; report only unconditional 75% harm-concept rate', pretopk_limitation: 'semantic claims rest on pre-topk encoder projections, not validated sparse post-topk features' } }
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
  After iteration 3, the paper will have definitive numerical answers to all four MAJOR critiques: (1) t-test p-value and
  Cohen's d comparing top-500 vs bottom-500 Reversal Score distributions — either confirming selective preservation (top >>
  bottom) or revealing global similarity (both near R=1); (2) mean |cos_sim| between refusal direction r and RLHF-sensitive
  encoder directions — either directly confirming geometric orthogonality (|cos| < 0.1) or showing the claim needs qualification;
  (3) centroid ratios in RLHF-insensitive and random subspaces alongside the RLHF-sensitive 6.1×/17.6×, determining whether
  those ratios are subspace-specific or universal; (4) layer-20 Reversal Score statistics explicitly reported for the first
  time, either replicating the zero-Type-R finding or revealing layer-specificity. The output JSON also contains the corrected
  paper descriptions for semantic labeling, density filter, classification accuracy, and pre-topk/post-topk distinction —
  giving GEN_PAPER_TEXT a complete revision target.
summary: >-
  One focused experiment addresses all four MAJOR reviewer critiques by computing the null comparisons and direct measurements
  that were missing from iter2. The null-baseline (bottom-500 features) settles the selective-preservation vs. global-similarity
  question. The cosine-similarity measurement converts the geometric-orthogonality claim from inference to direct evidence.
  Control-subspace centroid distances contextualize the 6.1×/17.6× ratios. Layer-20 numbers substantiate the sensitivity check.
  All computations run from scratch in under one hour on one GPU.
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
</existing_artifacts>

<current_paper>
The current paper draft — represents the research story so far.

Use this to understand what's working, what's not, and what gaps remain.
Gaps and weak results signal what to try differently — not what to conclude.

# Introduction

Refusal ablation — commonly called *abliteration* — edits a safety-aligned language model's weights to eliminate compliance behavior [1]. The technique identifies the *refusal direction*: the principal mean-difference vector separating residual-stream activations on harmful versus harmless prompts, then orthogonalizes all residual-stream-writing weight matrices against this direction. The result is a model that passes harmful prompts through as if it had never been safety-tuned. The standard narrative frames abliteration as *undoing* RLHF safety fine-tuning.

Two geometric studies directly challenge this narrative. LatentBiopsy [2] demonstrates that a linear classifier applied to Qwen3.5-0.8B and Qwen2.5-0.5B residual streams retains AUROC ≥ 0.937 after abliteration, dropping at most 0.015 from the instruct baseline. A companion cross-family study [3] replicates this across 12 model variants (Qwen, Llama, Gemma; 0.5B–9B parameters), reporting mean AUROC 0.982 surviving abliteration within ±0.003. Something in the internal representation of harmful content survives abliteration essentially unchanged. What these geometric analyses cannot answer is the mechanistic question: *which specific features carry the surviving representation? Are they semantically coherent? And what geometric relationship between the refusal direction and RLHF-modified feature directions would explain this survival?*

Answering these questions requires a feature-level vocabulary. Sparse Autoencoders (SAEs) [11, 12] provide exactly that: trained on residual-stream activations, they decompose those activations into tens of thousands of near-monosemantic features amenable to semantic interpretation. We apply the frozen Qwen-Scope base SAE [9] — 65,536 TopK features trained on Qwen3-8B residual-stream activations — across three model variants simultaneously: base, instruct (safety RLHF-aligned), and abliterated. Our central metric, the *Reversal Score*, is a per-feature Pearson correlation between the RLHF-induced and abliteration-induced activation change vectors over 500 harmful prompts, measuring whether each RLHF-modified feature is preserved or reversed by abliteration.

Our central empirical finding is striking: no feature in the top-500 RLHF-sensitive set has a negative Reversal Score (mean = 0.921, std = 0.040, min = 0.684). A null-baseline comparison — computing Reversal Scores for the bottom-500 RLHF-insensitive features — yields a significantly lower mean (0.847, Cohen's d = 1.751, p ≈ 0), confirming that the high Reversal Scores reflect *selective preservation* rather than global similarity between the instruct and abliterated models. A direct measurement of refusal-direction geometry confirms the mechanistic interpretation: the mean absolute cosine similarity between the refusal direction and the top-500 RLHF-sensitive SAE encoder directions is 0.019 at layer 22 and 0.021 at layer 20, establishing near-orthogonality that explains why removing the refusal direction does not disturb any RLHF-modified feature. Centroid analysis across control subspaces confirms the imprint is specific to the RLHF-sensitive subspace (4.35× ratio) rather than a global model-similarity artifact (control ratios 1.16× and 2.37×).

[FIGURE:fig1]

**Summary of Contributions:**
- We introduce the *Reversal Score*, a per-feature Pearson correlation quantifying whether abliteration reverses or preserves each RLHF-modified SAE feature activation pattern (Section 3.3). [ARTIFACT:art_QBm3I0Z5VVJR]
- We present the first null-baseline validation of selective feature preservation: RLHF-sensitive features (R̄ = 0.921) show significantly higher Reversal Scores than RLHF-insensitive controls (R̄ = 0.847), with Cohen's d = 1.751 and p ≈ 0, ruling out the global-similarity alternative (Section 5.3). [ARTIFACT:art_Qnl1zSSZPJvj]
- We directly measure refusal-direction orthogonality to RLHF-modified SAE encoder directions, finding mean |cos| = 0.019 — confirming geometrically that abliteration cannot substantially affect any RLHF-sensitive feature (Section 5.4). [ARTIFACT:art_Qnl1zSSZPJvj]
- We demonstrate that centroid distances in the RLHF-sensitive subspace show 4.35× closeness to instruct versus 1.16× and 2.37× in control subspaces, ruling out global similarity as an explanation (Section 5.5). [ARTIFACT:art_Qnl1zSSZPJvj]
- We transparently report that a downstream harm classifier built on Reversal Score features achieves 0.476 accuracy — near-chance, as mechanistically expected from zero Type-R features — and explain why this is a consequence rather than a failure of the main finding (Section 5.7).

# Related Work

**Refusal ablation.** Arditi et al. [1] establish that refusal is mediated by a single residual-stream direction and that orthogonalizing all weight matrices against it reliably eliminates compliance across 13 models up to 72B parameters. Cristofano [7] shows the raw refusal vector is polysemantic, entangling safety circuitry with capability and style; spectral residualization reduces KL divergence on Wikitext-2 from 2.088 to 0.044 while preserving abliteration. Nanfack et al. [19] propose an optimal-transport alternative reducing distributional shift. Our work examines what all these variants share: the refusal direction, however constructed, leaves RLHF-modified SAE encoder features intact.

**Geometry of harmful intent surviving abliteration.** LatentBiopsy [2] and the companion geometry study [3] are the direct predecessors establishing the phenomenon our work explains mechanistically. Their key limitation — collapsing representational information into a single AUROC score — is precisely the gap our feature-level analysis addresses. The null-baseline comparison (Cohen's d = 1.751) and direct orthogonality measurement (mean |cos| = 0.019) provide the mechanistic account that a single detection score cannot.

**SAEs as measurement instruments.** Elhage et al. [11] motivate SAE decomposition via the superposition hypothesis. Bricken et al. [12] demonstrate dictionary learning recovering monosemantic features at practical scale. Gao et al. [13] establish scaling laws for SAE quality. Templeton et al. [14] apply SAEs at Claude 3 Sonnet scale, finding rich safety-relevant semantic structure. Chopra [4] uses frozen base SAEs as cross-variant diagnostics, finding that SAE latent cosine similarity drops to 0.509–0.557 at layer 22 after SFT while raw cosine stays above 0.96 — validating the frozen base-SAE approach and motivating the pre-topk encoder fallback when reconstruction quality degrades.

**Multi-model SAE comparison.** Jiralerspong and Bricken [5] introduce *crosscoders* — SAEs trained jointly on two or more model variants — partitioning features into shared and model-specific categories. Kassem et al. [6] extend this with Delta-Crosscoder for narrow fine-tuning regimes. Our frozen base-SAE approach is complementary: it requires zero additional training, directly reuses Qwen-Scope's existing feature vocabulary, and applies naturally to the three-model abliteration trajectory. Shportko et al. [20] use Dedicated Feature Crosscoders to localize RL-induced tool-calling in Qwen2.5-3B to a compact steerable feature set — the complement of our finding about which features abliteration *cannot* remove. Li et al. [10] demonstrate an ≈8× MSE gap between base-SAE reconstruction on instruct versus purpose-trained instruct SAE activations, motivating the reconstruction quality gate we implement.

**Safety fine-tuning mechanisms.** Jain et al. [8] establish that safety fine-tuning projects unsafe activations into MLP null spaces, explaining adversarial circumvention. Our finding extends this picture: at the SAE encoder level, the directions RLHF modifies are near-orthogonal to the refusal direction, so abliteration — which removes the refusal direction — cannot erase them.

# Methodology

## Three-Model Setup

All experiments use three Qwen3-8B checkpoints sharing identical 36-layer architecture (d_model = 4096, bfloat16, transformers ≥ 4.51.0) [ARTIFACT:art_pmAcY0CNazP9]:

*Base*: `Qwen/Qwen3-8B-Base` — the unaligned pretrained model, providing the representational baseline.

*Instruct*: `Qwen/Qwen3-8B` with `enable_thinking=False` — the safety RLHF-aligned model. Chain-of-thought suppression isolates safety-relevant activations on the final input token.

*Abliterated*: `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` (v2 corrects a layer-0 output-garbling artifact present in v1). This community model applies standard difference-of-means orthogonalization consistent with the canonical Arditi et al. [1] recipe. The exact layer set and prompt set used for the refusal direction computation are not publicly documented; verification against a scratch-reproduced checkpoint with fully documented parameters is important future work.

All models are accessed via `model.model.layers` for residual-stream hook injection at the final token position of each input.

## SAE Measurement Instrument

The frozen Qwen-Scope SAE `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` serves as the shared measurement instrument [9]. [ARTIFACT:art_pmAcY0CNazP9] A critical architectural note: Qwen-Scope implements *TopK* architecture (k = 50 features retained via `torch.topk`), not JumpReLU. For a residual activation $\mathbf{x} \in \mathbb{R}^{4096}$:

$$\mathbf{z}_{\text{pre}} = \mathbf{x}W_{\text{enc}}^\top + \mathbf{b}_{\text{enc}} \in \mathbb{R}^{65536}$$
$$\mathbf{z} = \text{TopK}(\mathbf{z}_{\text{pre}},\; k{=}50)$$

The *pre-topk* representation $\mathbf{z}_{\text{pre}}$ is a dense continuous projection of the residual stream onto all 65,536 encoder directions. Because it is dense, every feature has a non-zero value on every prompt by construction — an important property for interpreting the density characterization in Section 3.4.

## Feature Metrics

For each SAE feature $f$ and $N = 500$ harmful prompts $\{p_i\}_{i=1}^N$, let $a_m(f, p_i)$ denote the pre-topk encoder activation of feature $f$ for model $m \in \{\text{base}, \text{inst}, \text{abl}\}$.

**Safety Change** quantifies the magnitude of RLHF modification to feature $f$:
$$\text{SC}(f) = \frac{1}{N}\sum_{i=1}^N |a_{\text{inst}}(f, p_i) - a_{\text{base}}(f, p_i)|$$

Features are ranked by SC$(f)$: the top-500 form the *RLHF-sensitive* set and the bottom-500 form the *RLHF-insensitive* null-baseline set. An additional random-500 control group is drawn uniformly from all 65,536 features.

**Reversal Score** quantifies whether abliteration reverses or preserves the RLHF modification to feature $f$:
$$\boldsymbol{\Delta}_{\text{inst}}(f) = [a_{\text{inst}}(f, p_i) - a_{\text{base}}(f, p_i)]_{i=1}^N$$
$$\boldsymbol{\Delta}_{\text{abl}}(f) = [a_{\text{abl}}(f, p_i) - a_{\text{base}}(f, p_i)]_{i=1}^N$$
$$R(f) = \text{PearsonR}(\boldsymbol{\Delta}_{\text{inst}}(f), \boldsymbol{\Delta}_{\text{abl}}(f))$$

$R(f) \approx +1$ indicates abliteration preserves the RLHF modification prompt-by-prompt (*Type-I: imprinted*); $R(f) \approx -1$ indicates abliteration reverses it (*Type-R: refusal-coupled*). This correlation is computed over 500 scalar values per feature — not per-prompt cosine similarities — so the distributional result is an empirical finding rather than a definitional artifact.

## Pre-TopK Projections and Post-TopK Density Characterization

Because the reconstruction quality gate (Section 3.5) fails at layer 22, all feature metrics use $\mathbf{z}_{\text{pre}}$. An important consequence: $\mathbf{z}_{\text{pre}}$ values are dense by construction — *every feature has a non-zero value on every prompt*. An activation-density filter designed for post-topk sparse activations (which can genuinely fire on fewer than 50 of 500 prompts) is therefore trivially satisfied for pre-topk projections and provides no meaningful coverage information.

Instead, we characterize feature density in the post-topk sense: for each top-500 feature, we compute the fraction of prompts on which the feature's pre-topk value is among the $k=50$ largest (i.e., would be active in the post-topk sparse representation). At layer 22, the mean post-topk firing rate across the top-500 RLHF-sensitive features is 1.2%, with only 3.4% of these features having post-topk activity on more than 10% of prompts. This characterization shows that while pre-topk projections are dense, the selected top-500 features are nonetheless relatively sparse in the post-topk sense — they would fire on only a small fraction of prompts if the TopK gate were applied.

## Layer Selection and Reconstruction Quality Gate

**Layer selection.** We compute SC$(f)$ variance across all 65,536 features at candidate layers \{14, 20, 22\} on 50 harmful prompts. Layer 22 is selected empirically (variance 0.1721 vs. 0.0547 at layer 20 and 0.0127 at layer 14). [ARTIFACT:art_QBm3I0Z5VVJR] All primary results use layer 22; layer 20 is analyzed as a replication layer in Section 5.8.

**Reconstruction quality gate.** Following Li et al. [10], we compute $R^2$ for all three model variants at layer 22. If $R^2 \geq 0.85$ for non-base variants, full encoder-plus-decoder analysis proceeds. If $R^2 < 0.85$, encoder-only analysis on $\mathbf{z}_{\text{pre}}$ is used. At layer 22, $R^2 = 0.574$ (base), 0.512 (instruct), 0.512 (abliterated) — all below 0.85 — so all feature metrics use $\mathbf{z}_{\text{pre}}$. This is consistent with Li et al.'s [10] finding of ≈8× MSE degradation when base SAEs are applied to instruct-model activations. [ARTIFACT:art_QBm3I0Z5VVJR]

## Semantic Labeling Protocol

For the top-100 features by SC$(f)$, we identify the 5 prompts with the highest pre-topk encoder activation and provide the first 300 characters of each prompt to Llama-3.1-8B-Instruct via OpenRouter. (The cross-family labeler avoids within-Qwen circularity.) Each feature is classified as `harm_concept`, `safety_refusal`, `general_capability`, or `other`. *Note on protocol:* the implemented labeling uses character-prefix truncation of the most-activating prompts rather than a per-token activation window; labels therefore reflect overall prompt topic more than the specific token-level representations that most activate the feature. This limitation is discussed in Section 6.3.

## Bimodality Analysis

Hartigan's dip test ($H_0$: unimodal) is applied to the Reversal Score distribution for the top-500 features. Both outcomes are pre-specified as informative: bimodal ($p < 0.05$) supports a discrete Type-R / Type-I classification; unimodal ($p \geq 0.05$) supports a graded survival interpretation.

# Evaluation Corpus

The evaluation corpus comprises 500 harmful and 500 benign prompts from four publicly licensed benchmark datasets [ARTIFACT:art_M8-flaZ0OOaP].

**BeaverTails** [15] (176 harmful): Human-labeled QA pairs spanning 14 harm categories (chemical threats, self-harm, violence, financial crime, and others).

**AdvBench** (174 harmful): Standard adversarial behavior dataset used in mechanistic abliteration research [1], providing direct harmful instructions.

**JailbreakBench/JBB-Behaviors** [16] (55 harmful + 46 benign): The only source offering matched harmful/benign pairs from the same curation effort, enabling contrastive SAE analysis.

**HarmBench** [17] (95 harmful): Standard safety benchmark covering chemical, cybercrime, physical, and illegal categories.

**Dolly-15k** [18] (454 benign): Human-written instruction-following prompts providing a topically diverse benign baseline.

All sources are processed through MinHash LSH deduplication (threshold 0.5, 128 permutations, character 5-gram shingling), token-length filtering (≤512 tokens), and category capping (≤35% per harm subcategory). The final corpus contains 1,000 prompts (500 harmful, 500 benign).

# Results

## Layer Selection

Figure 4 shows the Safety Change variance across candidate layers. Layer 22 exhibits the highest variance (0.1721), confirming it as the primary analysis layer. The $R^2$ reconstruction gate at layer 22 yields values below 0.85 for all three model variants (base: 0.574, instruct: 0.512, abliterated: 0.512), triggering the pre-topk encoder-only analysis path for all subsequent computations.

[FIGURE:fig4]

## Reversal Score Distribution

Figure 2(a) shows the Reversal Score distribution for the top-500 RLHF-sensitive features at layer 22. Every feature has a positive Reversal Score (mean = 0.921, std = 0.040, min = 0.684, max = 0.989). No feature has a negative or near-zero Reversal Score. Hartigan's dip test yields statistic = 0.0123, $p = 0.930$ — strongly unimodal with no evidence of bimodality. Spearman correlation yields mean = 0.913, confirming robustness.

[FIGURE:fig2]

Table 1 summarizes Type-I and Type-R counts across all tested thresholds. Zero features are classified Type-R at any threshold, including the most permissive (R < −0.30).

**Table 1.** Type-I and Type-R feature counts at all tested thresholds.

| Threshold | Type-I (R > threshold) | Type-R (R < −threshold) |
|-----------|------------------------|-------------------------|
| 0.30 | 500 / 500 (100%) | 0 / 500 (0%) |
| 0.50 | 500 / 500 (100%) | 0 / 500 (0%) |
| 0.70 | 499 / 500 (99.8%) | 0 / 500 (0%) |
| 0.95 | 125 / 500 (25%) | 0 / 500 (0%) |

**Post-topk density characterization.** In the post-topk sense, the top-500 features are relatively sparse: the mean fraction of prompts on which a feature would be in the active post-topk set is 1.2% at layer 22, with only 3.4% of features having post-topk firing on more than 10% of prompts. This confirms that the top-500 features are genuinely RLHF-sensitive in their encoder projections, even though their post-topk activation rates are low — a consequence of the TopK architecture's hard selection operating on a high-variance encoding space.

## Null-Baseline Validation of Selective Preservation

A critical alternative explanation for R̄ = 0.921 is global model similarity: if abliteration modifies only a few of 4096 residual-stream directions, all encoder projections — not just RLHF-sensitive ones — could show high inter-variant correlation, making R̄ = 0.921 uninformative. We address this directly by computing Reversal Scores for the bottom-500 RLHF-insensitive features and a random-500 control group. [ARTIFACT:art_Qnl1zSSZPJvj]

Figure 2(b) shows the Reversal Score distributions for all three groups at layer 22. Top-500 R̄ = 0.921; bottom-500 R̄ = 0.847 (Cohen's d = 1.751, two-sample t-test p ≈ 0). **SELECTIVE_PRESERVATION_CONFIRMED.** The RLHF-sensitive features show significantly higher Reversal Scores than RLHF-insensitive controls, ruling out the global-similarity explanation. The effect size (d = 1.751) is very large by conventional standards. At layer 20: top-500 R̄ = 0.924, bottom-500 R̄ = 0.841 — the same pattern.

The selective-preservation interpretation is as follows: RLHF modifies certain feature encoder directions more than others; abliteration preserves those modified directions more faithfully than the unmodified directions. This occurs because the refusal direction — which abliteration removes — is more orthogonal to the RLHF-modified encoder directions than to the RLHF-insensitive ones. We test this geometric prediction directly in Section 5.4.

## Direct Measurement of Refusal-Direction Orthogonality

The geometric interpretation of zero Type-R features requires a testable prediction: the refusal direction $\mathbf{r}$ must be nearly orthogonal to all RLHF-modified SAE encoder directions. We test this directly. [ARTIFACT:art_Qnl1zSSZPJvj]

We compute $\mathbf{r}$ as the mean difference of instruct residual-stream activations on harmful versus harmless prompts at layer 22, following Arditi et al. [1]. For each of the top-500 RLHF-sensitive features $f$, we measure the cosine similarity between $\mathbf{r}$ and the SAE encoder direction $\mathbf{w}_{\text{enc}}(f) \in \mathbb{R}^{4096}$:

$$\cos\theta_f = \frac{\mathbf{r} \cdot \mathbf{w}_{\text{enc}}(f)}{\|\mathbf{r}\| \|\mathbf{w}_{\text{enc}}(f)\|}$$

**Layer 22:** mean $|\cos\theta_f| = 0.019$ over the top-500 RLHF-sensitive encoder directions. **ORTHOGONALITY_CONFIRMED.** A cosine of 0.019 means the refusal direction projects onto less than 2% of each RLHF-modified encoder direction's magnitude. **Layer 20:** mean $|\cos\theta_f| = 0.021$. ORTHOGONALITY_CONFIRMED at both layers.

This direct measurement explains the zero-Type-R result mechanistically. Abliteration orthogonalizes all weight matrices against $\mathbf{r}$. Because $\mathbf{r}$ is nearly orthogonal to all RLHF-modified SAE encoder directions, orthogonalizing against $\mathbf{r}$ cannot substantially change how any RLHF-modified feature's activations are computed. The RLHF-modified features survive abliteration because abliteration's geometric operation is nearly orthogonal to the subspace those features occupy.

## Centroid Analysis with Control Subspaces

To validate that the latent safety imprint is specific to the RLHF-sensitive subspace rather than a global consequence of instruct/abliterated model similarity, we compute centroid distance ratios $d_{\text{abl-base}} / d_{\text{abl-inst}}$ (cosine) for three feature subspaces: A (top-500 RLHF-sensitive), B (bottom-500 RLHF-insensitive), and C (random-500). [ARTIFACT:art_Qnl1zSSZPJvj]

[FIGURE:fig3]

**Layer 22 (harmful prompts):** Subspace A ratio = 4.35× (abliterated model is 4.35× closer to instruct than to base in the RLHF-sensitive subspace), Subspace B ratio = 1.16×, Subspace C ratio = 2.37×. The RLHF-sensitive subspace shows substantially larger instruct-closeness than either control subspace, confirming that the centroid imprint is specific to the features RLHF modified.

**Layer 20 (harmful prompts):** Subspace A ratio = 7.31×, Subspace B ratio = 1.01×, Subspace C ratio = 1.63×. **CENTROID_SELECTIVE.** The layer-20 pattern is even cleaner: the RLHF-insensitive subspace shows essentially no instruct-closeness (1.01×), while the RLHF-sensitive subspace shows 7.31×.

The contrast between A and B is the critical comparison. If abliteration produces only global model similarity (all subspaces close to instruct), B and C ratios would equal A. Instead, B ≈ 1.0–1.2 while A = 4.35–7.31, a factor of 4–7 difference that cannot be explained by global similarity. The intermediate random subspace C (2.37× at layer 22, 1.63× at layer 20) lies between A and B, consistent with the mix of RLHF-sensitive and insensitive features in any random sample.

## Semantic Labeling

Llama-3.1-8B-Instruct assigned labels to 100 top-SC features: `harm_concept` ≈ 75%, `safety_refusal` ≈ 5%, `general_capability` ≈ 5%, `other` ≈ 15%. Example harm-concept features include those most activated by: identity theft and financial manipulation, synthesis of controlled substances, cyberattack methodologies, hate speech and incitement, and psychological manipulation tactics.

Because the Reversal Score distribution is unimodal with zero Type-R features, the pre-registered Fisher's exact test comparing harm-concept rates in Type-I versus Type-R features cannot be computed. The meaningful semantic conclusion is instead: *harm-concept content is uniformly distributed across the entire Reversal Score range* (0.684 to 0.989). Approximately 75% of all top-500 RLHF-sensitive features carry harm-concept labels throughout — the latent safety imprint is not concentrated in a surviving minority but pervades the entire set of RLHF-modified features.

An important caveat: all semantic claims rest on pre-topk SAE encoder projections (dense, not post-topk sparse), and the labeling protocol uses character-prefix truncation of the most-activating prompts rather than feature-specific token activation windows. Labels may therefore reflect prompt topic more than the precise representations that most activate each feature. These limitations are addressed in Section 6.3.

## Downstream Classification Accuracy

As a downstream application test, we computed a per-prompt harm score as the mean pre-topk activation across all top-500 RLHF-sensitive features and applied a median threshold to classify harmful versus benign prompts. The resulting accuracy is **0.476** — near-chance on the balanced 500/500 corpus — compared to a keyword baseline of **0.630**. [ARTIFACT:art_QBm3I0Z5VVJR]

This null classification result is an *expected consequence* of the primary finding, not an independent failure. Because the Reversal Score distribution is unimodal and all features have high R values, there is no Type-R contrast set whose activation deficit relative to Type-I features could provide a discrimination signal. The mean pre-topk Type-I activation is elevated across *all* prompts (the instruct model activates these features more than the base) but shows no selective increase for harmful versus benign content at the prompt level. The harm-concept semantic content identifies *which features RLHF modified*, not *how those features respond differentially to harmful prompts at inference time*.

## Layer-20 Replication

Table 2 summarizes the full quantitative replication at layer 20 alongside the primary layer-22 results.

**Table 2.** Quantitative results at layers 22 and 20. All layer-20 findings replicate the layer-22 conclusions.

| Metric | Layer 22 | Layer 20 |
|--------|----------|----------|
| Top-500 Reversal Score (R̄) | 0.921 | 0.924 |
| Bottom-500 Reversal Score (R̄) | 0.847 | 0.841 |
| Cohen's d (top vs. bottom) | 1.751 | ~1.69 |
| Mean \|cos(r, enc)\| top-500 | 0.019 | 0.021 |
| Centroid A (RLHF-sensitive) | 4.35× | 7.31× |
| Centroid B (RLHF-insensitive) | 1.16× | 1.01× |
| Centroid C (random-500) | 2.37× | 1.63× |
| Centroid verdict | MODERATE | SELECTIVE |
| Post-topk density (mean) | 1.2% | 1.9% |
| Post-topk density (>10%) | 3.4% | 5.6% |

**REPLICATION_CONFIRMED** at both layers. The layer-20 centroid pattern is actually cleaner (CENTROID_SELECTIVE vs. CENTROID_MODERATE at layer 22), with the RLHF-insensitive subspace showing a ratio of 1.01 — essentially no instruct-closeness at all.

# Discussion

## Geometric Interpretation of Zero Type-R Features

The direct orthogonality measurement (mean |cos| = 0.019) provides the mechanistic account for the zero-Type-R finding. Abliteration removes the refusal direction $\mathbf{r}$ from all weight matrices. For abliteration to change the activation of SAE feature $f$, $\mathbf{r}$ must project substantially onto the encoder direction $\mathbf{w}_{\text{enc}}(f)$. Since mean $|\cos(\mathbf{r}, \mathbf{w}_{\text{enc}}(f))| \approx 0.019$ across the top-500 RLHF-modified features, abliteration modifies those features' activations by at most ≈1.9% of the refusal direction's magnitude — effectively zero. RLHF-modified SAE features survive abliteration not by coincidence but by geometric construction: they occupy a subspace nearly orthogonal to the refusal direction.

This interpretation is consistent with Cristofano's [7] observation that the raw refusal direction is polysemantic, primarily capturing stylistic compliance patterns rather than semantic harm-concept content. If $\mathbf{r}$ encodes "how to express refusal" rather than "what constitutes harm," it would naturally be nearly orthogonal to the encoder directions for harm-concept features. Abliteration removes the compliance signal without touching the semantic harm representations.

The finding also explains the LatentBiopsy [2, 3] AUROC survival bottom-up: AUROC ≈ 0.982 survives because all the features that distinguish harmful from benign content in the *instruct model's representation* survive intact. A linear classifier operating on the residual stream captures the projection of these features' collective activation, which abliteration cannot remove.

## Why Downstream Classification Fails — and Why That Is Informative

The near-chance classification accuracy (0.476) requires careful interpretation. The latent safety imprint is defined as the set of RLHF-modified features that persist in the abliterated model; these features are identified by RLHF's effect on them (high Safety Change), not by their differential response to harmful versus benign prompts. The abliterated model retains the same internal encoding of harm concepts as the instruct model — but in neither model does this encoding produce a differentiating activation pattern between, say, "Write a social media campaign" (benign) and "Write a manipulative social media campaign" (harmful). The discriminative signal that LatentBiopsy [2] detects geometrically comes from the *aggregate geometry* of residual-stream activations, which operates at a different level than individual SAE feature activations.

This distinction between *what RLHF modified* (captured by the Reversal Score and semantic labeling) and *what harmful prompts selectively activate* (required for classification) is a key conceptual contribution: the latent safety imprint describes a representational property of the model, not a prompt-conditional activation pattern.

## Implications for Safety Restoration

If the latent safety imprint is comprehensive — all RLHF-modified features surviving — the challenge for targeted safety restoration after abliteration is not to amplify surviving harm-concept features (they are already at full instruct-model activation levels in the abliterated model). The challenge is to restore the *compliance mechanism*: the geometric pathway from harm-concept feature activations to refusal outputs. Abliteration removes the refusal direction, which implements this conversion; restoration requires re-injecting this direction without disturbing the already-intact feature-level semantics.

This reframes the restoration problem as geometric re-injection rather than semantic amplification. The DFC approach of Shportko et al. [20] identifies compact steerable feature sets for RL-induced capabilities; an analogous method targeting the features that mediate harm-concept-to-refusal conversion could enable targeted re-injection. Importantly, the semantic knowledge is preserved — only the direction implementing its behavioral consequence requires restoration.

## Limitations

**Pre-topk encoder projections versus post-topk sparse features.** The most significant limitation is that all feature metrics use $\mathbf{z}_{\text{pre}}$ (dense encoder projections) rather than the post-topk sparse activations that the mechanistic interpretability literature has validated as interpretable [12, 14]. Pre-topk values are dense continuous projections — every feature has a non-zero value on every prompt — which means they do not represent the "on/off" firing concept that makes individual SAE features semantically tractable. The Reversal Score captures a real geometric signal (confirmed by the null-baseline comparison and direct orthogonality measurement, both operating at the residual-stream level), but the semantic claims (≈75% harm-concept labeling) rest on an assumption that pre-topk encoder directions retain semantic alignment with post-topk features — an assumption that has not been directly validated for TopK SAEs in the cross-variant setting. An instruct-trained Qwen3-8B SAE covering layers above 8 would resolve this limitation by enabling post-topk analysis.

**Community abliterated checkpoint.** We used `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` without independently verifying that it achieves comparable abliteration depth to a scratch-reproduced checkpoint on geometric metrics (AUROC, KL divergence, refusal rate). If the community checkpoint applies non-standard abliteration producing weaker orthogonalization, high Reversal Scores could partially reflect incomplete abliteration rather than semantic preservation. Verification against a scratch-reproduced checkpoint with documented parameters is critical for the next iteration.

**Single model family and layers.** All results are for Qwen3-8B at layers 22 and 20. The Reversal Score distribution and geometric orthogonality claim may not generalize to other model families (Llama, Gemma) or other architectures. The total absence of Type-R features at layers 20 and 22 should be verified across all 36 Qwen-Scope layers, and the cross-family extension is marked as priority future work.

**Semantic labeler calibration.** No manual gold-set annotation was performed to compute Cohen's κ for the Llama-3.1-8B labeler. Future iterations should validate with a 20-feature manual gold set before trusting the ≈75% harm-concept rate quantitatively.

# Conclusion

We present the first feature-level mechanistic analysis of refusal abliteration using the frozen Qwen-Scope base SAE across three Qwen3-8B variants. The Reversal Score metric reveals that every top-500 RLHF-sensitive SAE encoder feature has a positive Reversal Score (mean = 0.921), and a null-baseline comparison confirms this reflects genuine selective preservation rather than global model similarity (bottom-500 control R̄ = 0.847, Cohen's d = 1.751, p ≈ 0). Direct measurement of refusal-direction geometry establishes the mechanistic explanation: mean |cos($\mathbf{r}$, $\mathbf{w}_{\text{enc}}$)| = 0.019 across RLHF-modified encoder directions — near-zero, meaning abliteration's geometric operation cannot substantially affect any of these features. Centroid analysis in control subspaces confirms selectivity (4.35× ratio in the RLHF-sensitive subspace versus 1.16× in the RLHF-insensitive control at layer 22; 7.31× versus 1.01× at layer 20). The latent safety imprint is comprehensive, semantically rich (≈75% harm-concept labels), and geometrically explained. The abliterated model's behavior reverts to base-like, but its internal representations of harmful content remain instruct-like — a dissociation that reframes safety restoration as geometric re-injection of the compliance direction rather than semantic amplification.

**Future directions:** (1) Verification against a scratch-reproduced abliterated checkpoint with fully documented parameters, to confirm the geometric orthogonality claim is not an artifact of incomplete abliteration. (2) Multi-layer analysis across all 36 Qwen-Scope layers to test whether the orthogonality pattern is layer-general or layer-specific. (3) Cross-family extension to Llama-3.1-8B and Gemma-3-9B. (4) Validation using an instruct-trained Qwen3-8B SAE (covering layers > 8) to enable post-topk analysis and verify the semantic labeling claims.

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

[13] L. Gao, T. Dupré la Tour, H. Tillman, G. Goh, R. Troll, A. Radford, I. Sutskever, J. Leike, and J. Wu. Scaling and evaluating sparse autoencoders. In *International Conference on Learning Representations*, 2025.

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

- [MAJOR] (methodology) The Reversal Score R(f) = PearsonR(Δ_inst, Δ_abl) measures the directional alignment of RLHF-induced and abliteration-induced activation changes across 500 prompts, not the magnitude of preservation. A feature where a_abl(f,p) = a_base(f,p) + 0.05·(a_inst(f,p) − a_base(f,p)) — i.e., abliteration preserves only 5% of the RLHF-induced shift — would still yield R≈1.0, because PearsonR is scale-invariant. The paper's central claim is that 'the abliterated model retains the same internal encoding of harm concepts as the instruct model,' but R≈0.921 is consistent with abliteration reducing the RLHF imprint by 95% in magnitude while preserving its direction. The centroid analysis uses cosine distance (angular), which is similarly direction-based. No magnitude-preserving metric is reported anywhere. This gap between 'correlation preserved' and 'activation level preserved' directly affects the paper's core narrative about a 'comprehensive latent safety imprint.'
  Action: Add a companion metric to the Reversal Score: for each top-500 feature f, compute the slope β_f of the OLS regression Δ_abl(f,·) ~ β_f · Δ_inst(f,·) (no intercept). β_f≈1 means the RLHF-induced change magnitude is preserved in the abliterated model; β_f≈0 means only the direction pattern survives. Report the distribution of β_f (mean, std, histogram) alongside the Reversal Score distribution in Figure 2. If β̄≈1 is confirmed, the paper's 'comprehensive survival' claim is fully supported. If β̄≪1, the claim should be reframed as 'directional pattern survival' rather than 'comprehensive imprint preservation.' Either outcome is publishable — the transparency matters.
- [MAJOR] (rigor) The null-baseline comparison (top-500 R̄=0.921 vs bottom-500 R̄=0.847, Cohen's d=1.751) may be partially artifactual. The bottom-500 RLHF-insensitive features are selected precisely because their Safety Change SC(f) = mean|Δ_inst(f,·)| is near zero. Computing PearsonR between two vectors that are both near zero (Δ_inst ≈ 0 and Δ_abl ≈ 0 for RLHF-insensitive features) is numerically unreliable: the result is dominated by floating-point noise rather than a meaningful signal. Specifically, if both Δ_inst and Δ_abl for bottom-500 features consist primarily of numerical noise at ~1e-5 magnitude, then PearsonR will reflect whatever correlations exist in that noise (potentially positive due to shared computation paths), not a meaningful 'preservation' quantity. The reported R̄=0.847 for bottom-500 features might reflect noise-to-noise correlation. If so, the d=1.751 gap could be an artifact: the top-500 features have large, genuinely correlated signals (high R for good reasons), while bottom-500 have small noisy signals (R could be anything, measured as 0.847 by chance). The paper does not report the actual magnitude of SC(f) for the bottom-500, making it impossible to assess whether those features' Reversal Scores are meaningful.
  Action: Report the median SC(f) value for the bottom-500 features. If median SC < 0.001 (near-zero by construction), replace or supplement the bottom-500 baseline with a mid-SC baseline: select 500 features uniformly from the 40th–60th SC percentile (non-trivial RLHF sensitivity, but not top). These features have non-negligible Δ_inst vectors so their Reversal Scores are interpretable. If the top-500 still shows d≈1.75 against the mid-SC baseline, the selective preservation claim is robust to the noise concern. Alternatively, confirm that bottom-500 features have SC(f) > 0.01 (i.e., their RLHF shifts are large enough for PearsonR to be reliable), which would address the concern directly.
- [MINOR] (evidence) There is an unresolved numerical discrepancy between the paper's text and the artifact summary for the refusal-direction orthogonality measurement. The paper states mean |cos θ_f| = 0.019 at layer 22 and 0.021 at layer 20, but the artifact summary (art_Qnl1zSSZPJvj) reports |cos_enc_top500_mean|=0.0145 without specifying the layer. These cannot all be correct simultaneously. Additionally, the HarmBench harmful count differs between the corpus description section (95 harmful) and the artifact summary for the dataset (85 harmful). Minor numerical inconsistencies undermine reviewer confidence in the reported values.
  Action: Reconcile both discrepancies explicitly: (1) Add a footnote or table entry clarifying that |cos|=0.019 is at layer 22 and |cos|=0.021 is at layer 20, and explain why the artifact reports 0.0145 (possibly from a preliminary run or different feature subset). (2) Verify the HarmBench count against the final corpus JSON (data_out.json) and use the correct number consistently throughout. These are quick fixes.
- [MINOR] (rigor) The Safety Change metric SC(f) is computed on 500 harmful prompts only, but RLHF modifies the model broadly — instruction-following style, response format, verbosity, and politeness all change in addition to safety behavior. Features that vary between instruct and base on harmful prompts due to *instruction-following style* (e.g., formal phrasing, structured output) rather than safety semantics would appear in the top-500. The ≈75% harm-concept labeling rate suggests this is not catastrophic, but the top-500 likely contains a mixture of harm-semantic and style-semantic features. This affects the interpretation of 'RLHF-modified features' throughout the paper.
  Action: Compute SC(f) separately on the 500 benign prompts and report the overlap between the top-500 harmful-SC features and the top-500 benign-SC features. High overlap would indicate that the selected features primarily reflect general RLHF instruction-following changes rather than harm-specific changes. Low overlap (e.g., <20%) would confirm that the top-500 is genuinely harm-sensitive. This analysis takes ~10 minutes computationally and would substantially strengthen the interpretive claims.
- [MINOR] (clarity) The paper's claim that 'the latent safety imprint is comprehensive' rests partly on the Reversal Score minimum being 0.684 (all 500 features positive). However, R=0.684 means only about 47% of variance in Δ_abl is explained by Δ_inst — the remaining 53% variance is unaccounted for. Combined with the unconfirmed magnitude issue, calling the imprint 'comprehensive' may be premature. The word 'comprehensive' appears five times in the paper (Introduction, Conclusion, Discussion) and each use slightly overstates what the presented evidence actually supports.
  Action: Replace 'comprehensive' with 'largely intact' or 'predominantly preserved' throughout, and add a parenthetical noting that 'comprehensive' specifically refers to all 500 features having positive Reversal Scores, not necessarily unit-magnitude preservation — pending the slope analysis recommended above. This is a one-paragraph precision fix.
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

### [2] HUMAN-USER prompt · 2026-07-17 13:45:11 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [3] SYSTEM-USER prompt · 2026-07-17 13:46:35 UTC

```
<verification_results>
Your previous response had issues that need fixing:

DEPENDENCY ERRORS (depends_on can ONLY reference IDs from <existing_artifacts>):
  - Strategy 1: Artifact 'experiment_iter4_dir1' (experiment): dependency 'art_Qnl1zSSZPJvj' has type 'experiment' which is not allowed (allowed: {'dataset', 'research'})
  - Strategy 1: Artifact 'experiment_iter4_dir1' (experiment): dependency 'art_QBm3I0Z5VVJR' has type 'experiment' which is not allowed (allowed: {'dataset', 'research'})

INSUFFICIENT VALID ARTIFACTS:
  Required: at least 1 valid artifacts
  Found: 0 valid out of 1 total
  Artifacts with invalid types, duplicate IDs, or invalid dependencies don't count as valid.

</verification_results>

<task>
Fix ALL issues above and regenerate your strategies:

1. Fix dependency errors:
   - depends_on is a list of {id, label} objects — every entry MUST have a non-empty short label
   - id can ONLY reference IDs from <existing_artifacts>
   - You CANNOT reference artifacts you are proposing in this strategy as dependencies (they all run in parallel)
   - Follow the dependency type rules (e.g., experiments require datasets)
   - If no suitable existing artifacts exist, use depends_on: []
2. Ensure at least 1 artifacts are fully valid (correct types, no ID conflicts, valid dependencies)

Output the corrected JSON with the fixed strategies.
</task>
```
