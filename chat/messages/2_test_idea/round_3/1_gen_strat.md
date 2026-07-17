# gen_strat_1 — test_idea

> Phase: `invention_loop` · round 3 · `gen_strat`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim transcript of this agent task — every system/user prompt, assistant response, thinking block, tool call and tool result — in the order they occurred. Nothing truncated.

## Task: `gen_strat_1` (terminal_claude_agent, claude-sonnet-4-6)

### [1] CONFIG · 2026-07-17 12:26:17 UTC

```
model: claude-sonnet-4-6 | effort: medium | permission: bypassPermissions | cwd: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_3/gen_strat/gen_strat_1
```

### [2] SYSTEM-USER prompt · 2026-07-17 12:26:23 UTC

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
title: Abliteration Preserves All RLHF-Modified SAE Features
hypothesis: |-
  RLHF safety fine-tuning and refusal abliteration operate on geometrically near-orthogonal subspaces in Qwen3-8B's residual stream. Contrary to the original bimodal prediction, the Reversal Score distribution for the top-500 RLHF-sensitive SAE encoder features at layer 22 is unimodal and concentrated near R=1 (mean=0.921, std=0.040, min=0.684), with zero features having negative Reversal Scores across all thresholds {0.30, 0.50, 0.70, 0.95}. The original Type-R/Type-I partition does not arise empirically: abliteration preserves every RLHF-modified feature, not a semantically selected minority.

  This finding requires a critical null-baseline validation before the geometric-orthogonality interpretation can be accepted. If abliteration modifies only a few of 4096 residual-stream directions, the Reversal Score for randomly sampled features — not modified by RLHF — could also be near 1.0 simply because instruct and abliterated models are globally similar in most directions. The primary experimental requirement for the next iteration is therefore: compute Reversal Scores for the bottom-500 SAE encoder features by Safety_change (features least modified by RLHF) and compare their distribution against the top-500. If both distributions concentrate near R=1, then R≈0.921 for RLHF-sensitive features reflects global model similarity rather than selective preservation — and the geometric-orthogonality claim is undermined. If the RLHF-sensitive features show significantly higher R than the control features, selective preservation is confirmed.

  A second direct test of the geometric-orthogonality claim is required: compute the refusal direction r (mean difference of instruct residual-stream activations on harmful vs. harmless prompts at layer 22, following Arditi et al.) and measure cosine similarity between r and each of the top-500 RLHF-sensitive SAE encoder directions (W_enc[feature_idx, :] projected into residual-stream space via W_dec[:, feature_idx]). If these cosine similarities are near zero (|cos| < 0.1), this directly confirms that abliteration — which orthogonalizes against r — cannot substantially affect any of the RLHF-modified feature directions. This would be a mechanistic account of why all Reversal Scores are positive: the refusal direction and the RLHF-modified encoder directions are nearly orthogonal by construction.

  Centroid analysis provides confirmatory evidence independent of the null-baseline concern: the abliterated model's SAE encoder activations in the RLHF-sensitive subspace are 6.1× closer (cosine) to the instruct model than to the base model for harmful prompts. However, the benign-prompt ratio (17.6×) is larger than the harmful-prompt ratio (6.1×), which is consistent with a global-similarity alternative rather than selective preservation of harm-concept representations. A control subspace comparison is needed: compute centroid distances in the bottom-500 (RLHF-insensitive) subspace and in a random 500-feature subspace. If those control ratios are comparable to 6.1–17.6×, the centroid finding also reflects global similarity rather than harm-specific preservation.

  Semantic labeling finds approximately 75% of top-500 RLHF-sensitive features carry harm-concept labels uniformly across the Reversal Score range (Q3/Q1 Fisher p=0.675, n=4 in high group — uninterpretably underpowered). The meaningful conclusion is that harm-concept representations are not concentrated in a surviving minority but pervade the entire RLHF-modified feature set. However, all semantic claims rest on pre-topk (dense) encoder projections (z_pre = h @ W_enc.T + b_enc) rather than post-topk sparse activations, because the base-SAE reconstruction gate fails (R²≈0.51 at layer 22 for all three variants). Pre-topk projections are dense continuous values — every feature has a non-zero value on every prompt by construction — so the 10% activation-density filter is trivially satisfied and provides no coverage information. Future iterations must either (a) obtain an instruct-trained Qwen3-8B SAE covering layers above 8 to enable post-topk analysis, or (b) validate that pre-topk encoder directions carry comparable semantic signal to post-topk sparse features before trusting the ≈75% harm-concept rate.

  A downstream classification test (per-prompt harm score = mean Type-I activation minus mean Type-R activation, median threshold) achieved 0.476 accuracy on the balanced 500/500 corpus — near-chance and below the 0.630 keyword baseline. This is expected given zero Type-R features: with no reversal-defined contrast, the score carries no discrimination signal. This null classification result is a direct consequence of the primary finding, not a separate failure, and should be reported transparently.

  Four specific methodological requirements for the next iteration follow from reviewer critiques: (1) Null-baseline Reversal Score comparison: bottom-500 features by Safety_change, reported alongside top-500. (2) Direct refusal-direction orthogonality measurement: cosine similarity between r and top-10 RLHF-sensitive encoder directions. (3) Control-subspace centroid analysis: distances in RLHF-insensitive and random subspaces alongside the RLHF-sensitive subspace. (4) Corrected semantic-labeling description: the actual protocol identifies the top-5 prompts by maximum pre-topk activation and provides the first 300 characters of each to the cross-family labeler — not a ±32 token window, which was not implemented. Layer-20 sensitivity results must be reported numerically (not asserted verbally) given the low reported runtime (4.3 minutes total).

  Whether the geometric-orthogonality interpretation survives the null-baseline test, the centroid finding (6.1× closer to instruct than to base for harmful prompts) is already confirmed at the pre-topk encoder level and is consistent with the latent-safety-imprint framing: the abliterated model retains the instruct model's representational fingerprint in the RLHF-sensitive feature subspace despite behaving like the base model on harmful prompts. The open question is whether this reflects selective preservation of a semantically coherent harm-concept representation, or a more general consequence of abliteration modifying only a narrow geometric slice of a 4096-dimensional space.
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
  Same frame; bimodal prediction replaced by total-preservation finding requiring null-baseline validation.
_confidence_delta: decreased
_key_changes:
- >-
  Core empirical finding reframed: no Type-R/Type-I bimodal split exists; all 500 top RLHF-sensitive features have positive
  Reversal Scores (mean=0.921), so abliteration preserves the entire RLHF-modified feature set at layer 22.
- >-
  Geometric-orthogonality interpretation introduced as the mechanistic explanation for zero Type-R features: the refusal direction
  r is near-orthogonal to RLHF-modified SAE encoder directions, so removing r does not disturb those features.
- >-
  Critical null-baseline requirement added: bottom-500 features by Safety_change must be analyzed to determine whether R≈0.921
  reflects selective preservation or global instruct/abliterated similarity.
- >-
  Direct refusal-direction measurement added as a required next step: cosine similarity between r and top-10 RLHF-sensitive
  encoder directions (W_enc projected via W_dec) to confirm orthogonality directly.
- >-
  Control-subspace centroid analysis added: benign-prompt ratio (17.6×) > harmful-prompt ratio (6.1×) raises global-similarity
  concern; centroid distances in RLHF-insensitive and random subspaces needed as controls.
- >-
  Pre-topk vs post-topk distinction clarified: all feature metrics use dense encoder projections (z_pre), not sparse post-topk
  activations; the density filter is trivially satisfied and provides no coverage information; semantic claims require validation.
- >-
  Near-chance classification accuracy (0.476) explicitly incorporated as an expected consequence of zero Type-R features —
  no discrimination signal when all features have high Reversal Scores.
- >-
  Corrected semantic-labeling protocol: top-5 prompts by max pre-topk activation, first 300 characters — not a ±32 token window
  as previously stated.
- >-
  Layer-20 numerical results required in next iteration (not verbal assertion of consistency).
- >-
  Underpowered Fisher fallback test (n=4 high group) dropped as a quantitative claim; replaced with qualitative conclusion:
  harm-concept labels uniformly distributed across Reversal Score range.
relation_type: evolution
</hypothesis>

<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for study design, proper baselines, and the evaluation/validity norms this field demands.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>

<iteration_status>
Current iteration: 3 of 5
Remaining (including this one): 3
</iteration_status>

<previous_strategies>
Strategies from the PREVIOUS iteration. You can CONTINUE these directions,
ADAPT based on what worked and what didn't in the artifacts produced, or PIVOT if results suggest a better path.

--- Strategy 1 ---
kind: strategy
id: gen_strat_1_idx1
title: Run SAE Analysis + Abliteration Verification
objective: >-
  Produce actual experimental results for all three confirmatory criteria: compute Safety_change and density-filtered Reversal_score
  across 65K Qwen-Scope features, run Hartigan's dip test, compute centroid distances in Type-R/Type-I subspaces, apply cross-family
  LLM semantic labeling with gold-set validation, and verify the community abliterated model against a scratch-reproduced
  checkpoint — replacing the current pre-registration with concrete numerical outcomes.
rationale: >-
  The MAJOR reviewer critiques converge on one diagnosis: the paper is a pre-registration with no results. Every proposed
  methodology (Reversal_score, density filter, Hartigan's test, distance criterion, semantic labeling) is described in future
  tense. This iteration must convert the existing prerequisites (tech reference from art_pmAcY0CNazP9, prompt corpus from
  art_M8-flaZ0OOaP) into concrete numbers. Two parallel experiments address the four MAJOR critiques simultaneously: (1) the
  core SAE analysis produces the Reversal_score distribution, bimodality test, centroid distances, and semantic labeling —
  all three confirmatory criteria; (2) the abliteration verification addresses the MAJOR methodology critique by comparing
  the community model against a scratch-reproduced checkpoint on geometric metrics (AUROC, KL-div, refusal rate), providing
  documented provenance for the primary abliterated checkpoint used in the main analysis.
artifact_directions:
- id: experiment_iter2_dir1
  type: experiment
  objective: >-
    Run the complete SAE Reversal_score analysis pipeline on all three Qwen3-8B variants and report all numerical outcomes
    for the three confirmatory criteria: (1) Hartigan's dip test p-value on the R(f) distribution of top-500 RLHF-sensitive
    features; (2) centroid distances d(abliterated,instruct) vs d(abliterated,base) in Type-R and Type-I feature subspaces
    for harmful vs benign prompts; (3) semantic label enrichment Fisher's exact test with Cohen's kappa validation.
  approach: |-
    Use UV with packages: transformers>=4.51.0, torch, huggingface_hub, scipy, datasketch, numpy, pandas, openai (for OpenRouter), diptest or diptest-py (Hartigan). Sequential memory management: load one model at a time, extract and save residual-stream activations to disk, unload before loading the next.

    **Step 0 — Layer selection (50 prompts, base model only).** Load Qwen/Qwen3-8B-Base in bfloat16 with device_map='auto'. Register forward hooks on model.model.layers[N].output for N in range(36) to capture residual stream at each layer. For each of 50 harmful prompts from the dataset, run a forward pass extracting the last-token residual at all 36 layers. Load Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 via hf_hub_download (filename layer{N}.sae.pt for each N). For each layer, compute pre-topk encoder activations: z_pre = residual @ W_enc.T + b_enc. Compute Safety_change variance = var over features of mean_prompt |z_pre_instruct - z_pre_base| (using instruct model for these 50 prompts). Select the layer with highest variance as the primary analysis layer. Also note layers 14, 20, 22 as sensitivity-check candidates.

    **Step 1 — Activation extraction (all 3 models, primary layer + 1 sensitivity layer).** Sequentially load each model (Base → Instruct → abliterated = huihui-ai/Huihui-Qwen3-8B-abliterated-v2 with enable_thinking=False for instruct), hook the selected layer, run all available harmful and benign prompts from the dataset (use actual counts from full_data_out.json — report the real counts, reconciling the 468 vs 1000 discrepancy noted by reviewers), extract last-token residual activations, save to disk as numpy arrays (shape: n_prompts × 4096). Use torch.cuda.empty_cache() and del model between loads.

    **Step 2 — SAE projection and reconstruction quality gate.** Load the Qwen-Scope SAE for the primary layer. For each model's saved activations compute: (a) z_pre = activations @ W_enc.T + b_enc (pre-topk, continuous); (b) z_post = TopK(z_pre, k=50) (sparse post-topk). Compute reconstruction R² = 1 - MSE(W_dec @ z_post.T + b_dec[:,None] - activations.T, axis=0).mean() / activations.var(axis=0).mean() for base, instruct, and abliterated variants. Report R² for all three. If R² < 0.85 for non-base variants, proceed with pre-topk activations as primary signal and document this explicitly.

    **Step 3 — Safety_change and density filter.** Using the harmful prompts: SC(f) = mean_i |z_inst(f,i) - z_base(f,i)| for all 65536 features (pre-topk activations). Select top-500 features by SC(f). Apply density filter: retain only features where both Δ_inst(f) and Δ_abl(f) have ≥10% non-zero entries (≥ 0.1 × n_harmful_prompts). Report the fraction of top-500 that survive the filter.

    **Step 4 — Reversal_score.** For each density-filtered feature f in the top-500: compute Δ_inst(f) = [z_inst(f,i) - z_base(f,i)] and Δ_abl(f) = [z_abl(f,i) - z_base(f,i)] over harmful prompts. R(f) = scipy.stats.pearsonr(Δ_inst(f), Δ_abl(f))[0]. Also compute Spearman correlation as robustness check. Plot the distribution of R(f). Determine empirical Type-I threshold as top quartile of R(f) distribution. Run robustness checks at fixed thresholds {0.3, 0.5, 0.7}. Apply diptest.diptest(R_values) for Hartigan's dip test (H0: unimodal). Report: dip statistic, p-value, interpretation.

    **Step 5 — Centroid distances (Criterion 2).** Separate harmful and benign prompts. In the Type-I subspace (high-R(f) features) and Type-R subspace (negative-R(f) features): for harmful prompts compute centroid(abliterated activations), centroid(instruct activations), centroid(base activations). Report d(abl,inst) and d(abl,base) using cosine distance and L2 distance. Repeat for benign prompts. Test the core prediction: d(abl,inst) < d(abl,base) in Type-I subspace for harmful prompts.

    **Step 6 — Semantic labeling via OpenRouter (Criterion 3).** Select top-100 features by |R(f) - empirical_threshold| × SC(f). For each, extract: the top-5 maximally-activating prompt segments (±32 tokens around the highest-activation token within each prompt). Call Llama-3-8B-Instruct (or meta-llama/llama-3.1-8b-instruct on OpenRouter; budget ≈ $0.50) with a structured prompt: provide the 5 prompt segments, ask to classify as: harm_concept | safety_refusal | general_capability | other. Manually annotate 20 features (10 high-R, 10 low-R) as gold set. Compute Cohen's kappa between manual and LLM labels. If kappa >= 0.6, apply labels to all 100 features and run Fisher's exact test comparing high-R (top quartile) vs low-R (bottom quartile) harm_concept label rates. Report kappa, Fisher p-value, and enrichment ratio.

    **Output format (method_out.json):** layer_selected, r2_base, r2_instruct, r2_abliterated, n_harmful_prompts, n_benign_prompts, n_top500_rlhf_sensitive, n_density_filtered, fraction_surviving_filter, r_values (list of floats, top-500 filtered features), hartigan_dip, hartigan_p, empirical_type_i_threshold, n_type_i, n_type_r, centroid_distances (harmful: d_abl_inst_typeI, d_abl_base_typeI, d_abl_inst_typeR, d_abl_base_typeR; benign: same), labeler_kappa, fisher_p, harm_concept_rate_typeI, harm_concept_rate_typeR, criterion1_confirmed (bool), criterion2_confirmed (bool), criterion3_confirmed (bool).
  depends_on:
  - id: art_M8-flaZ0OOaP
    label: prompt corpus
    relation_type:
    relation_rationale:
  - id: art_pmAcY0CNazP9
    label: tech reference
    relation_type:
    relation_rationale:
- id: experiment_iter2_dir2
  type: experiment
  objective: >-
    Reproduce abliteration from scratch on Qwen3-8B using the standard difference-of-means + orthogonal projection method,
    compare the scratch-reproduced checkpoint against huihui-ai/Huihui-Qwen3-8B-abliterated-v2 on three geometric metrics
    (AUROC for harmful-prompt detection, KL divergence on Wikitext-2, refusal rate on held-out prompts), and report whether
    the two checkpoints constitute co-equal analysis targets or whether one should be demoted to secondary validation.
  approach: |-
    Use UV with packages: transformers>=4.51.0, torch, datasets (HuggingFace), sklearn (AUROC), numpy.

    **Step 0 — Harmful/harmless datasets for refusal direction.** Download mlabonne/harmful_behaviors (CSV, ~520 rows) and mlabonne/harmless_alpaca (also CSV format). Use 256 harmful + 256 harmless samples (random seed=42) for computing the refusal direction, matching the mlabonne recipe documented in the tech reference.

    **Step 1 — Compute refusal direction.** Load Qwen/Qwen3-8B-Base in bfloat16. For each prompt in harmful and harmless sets, extract last-token residual stream activations at ALL middle layers (1 to 35). For each layer l, compute mean_harmful[l] - mean_harmless[l]; select the layer with highest L2 norm of this difference as the primary abliteration layer (expected: layer 9 for 8B scale per tech reference). Compute the unit refusal direction r = (mean_harmful[l_primary] - mean_harmless[l_primary]) / norm(...).

    **Step 2 — Orthogonal projection.** For each weight matrix W in {model.layers[l].self_attn.o_proj.weight, model.layers[l].mlp.down_proj.weight for all l}: W_new = W - (W @ r.unsqueeze(1)) @ r.unsqueeze(0). Also handle q_proj/k_proj/v_proj/up_proj/gate_proj as input projections: W_new = W - r.unsqueeze(0) @ (r.unsqueeze(0) @ W). Save modified model as scratch_abliterated/ using model.save_pretrained(). Document exact layers modified, layer selected for refusal direction, and final norm of r projected onto each weight matrix.

    **Step 3 — Geometric verification metrics.** Load both checkpoints: huihui-ai/Huihui-Qwen3-8B-abliterated-v2 (community) and scratch_abliterated/ (scratch). Also keep base model loaded for comparison.

    (a) AUROC for harmful-prompt detection: Use held-out 50 harmful + 50 benign prompts (not used in refusal direction computation). Extract last-token residual at the primary abliteration layer for each model. For base, instruct, community, scratch: compute angular deviation from mean_harmful direction (or run the LatentBiopsy-style linear probe: fit logistic regression on base activations, evaluate on instruct/community/scratch). Report AUROC for each model variant.

    (b) KL divergence on Wikitext-2: Load wikitext dataset (wikitext-2-raw-v1, test split). Sample 200 sentences (≤128 tokens). For each sentence, compute log-probabilities under base model vs community/scratch models. Report KL(base || community) and KL(base || scratch). Lower KL = less distributional shift from abliteration.

    (c) Refusal rate: Select 30 held-out harmful prompts not in the 50-prompt eval set. Generate completions (max_new_tokens=200, do_sample=False) from community and scratch models. Apply a simple refusal classifier: count completions containing any of ['cannot', 'I cannot', 'I'm unable', 'I apologize', 'I will not', 'illegal', 'I'm sorry']. Report refusal_rate_community and refusal_rate_scratch. Both should be near 0 if abliteration was successful.

    **Step 4 — Comparison decision.** If all three metrics (AUROC, KL, refusal_rate) agree within 5% relative difference, treat both checkpoints as co-equal targets. If they diverge substantially (e.g., one has KL > 2× the other, or refusal rates differ by >15pp), demote the community model to secondary validation and use scratch as primary.

    **Output format (method_out.json):** abliteration_layer_selected, n_harmful_used, n_harmless_used, refusal_direction_norm, weights_modified_count, auroc_base, auroc_instruct, auroc_community, auroc_scratch, kl_base_vs_community, kl_base_vs_scratch, refusal_rate_community, refusal_rate_scratch, primary_checkpoint_recommendation ('community' | 'scratch' | 'co-equal'), comparison_verdict.
  depends_on:
  - id: art_pmAcY0CNazP9
    label: tech reference
    relation_type:
    relation_rationale:
  - id: art_M8-flaZ0OOaP
    label: eval prompts
    relation_type:
    relation_rationale:
expected_outcome: >-
  After iteration 2, the paper will have actual experimental results replacing the pre-registration shell. The core SAE analysis
  experiment will produce: (1) the Reversal_score distribution histogram with Hartigan's dip test p-value; (2) d(abl,inst)
  vs d(abl,base) centroid distances in Type-I and Type-R feature subspaces for harmful vs benign prompts; (3) semantic label
  enrichment statistics (Fisher's exact p, Cohen's kappa, harm_concept rate by feature group); (4) reconstruction R² for all
  three model variants; (5) density filter survival fraction. The abliteration verification experiment will produce: AUROC,
  KL divergence, and refusal rate for both community and scratch-reproduced checkpoints, with a documented recommendation
  for which checkpoint to use as the primary analysis target. Together these eliminate all four MAJOR reviewer critiques and
  provide concrete evidence for or against the latent safety imprint hypothesis.
summary: >-
  Iteration 2 converts the pre-registration into actual results. Two parallel experiments address all four MAJOR critiques:
  (1) the core SAE pipeline computes Safety_change, density-filtered Reversal_score for 65K features, Hartigan's bimodality
  test, centroid distances in Type-R/Type-I subspaces, and cross-family LLM semantic labeling — reporting all three confirmatory
  criteria numerically; (2) the abliteration verification reproduces the scratch abliterated checkpoint and compares it against
  the community model on AUROC, KL-divergence, and refusal rate, establishing which checkpoint to trust as the primary analysis
  target. Both experiments depend only on iteration-1 artifacts and run in parallel within the 6-hour budget per artifact.
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
</existing_artifacts>

<current_paper>
The current paper draft — represents the research story so far.

Use this to understand what's working, what's not, and what gaps remain.
Gaps and weak results signal what to try differently — not what to conclude.

# Introduction

Refusal ablation—commonly called *abliteration*—edits a safety-aligned language model's weights to eliminate compliance behavior [1]. The technique identifies the *refusal direction*: the principal mean-difference vector separating residual-stream activations on harmful versus harmless prompts. Orthogonalizing all residual-stream-writing weight matrices against this vector removes the model's tendency to refuse harmful requests while largely preserving general language capabilities. The result is a model that passes harmful prompts through as if it had never been safety-tuned.

The standard framing treats abliteration as *undoing* RLHF safety fine-tuning. Yet geometric analyses challenge this narrative directly. LatentBiopsy [2] demonstrates that a linear angular-deviation classifier applied to Qwen3.5-0.8B and Qwen2.5-0.5B residual streams retains AUROC ≥ 0.937 after abliteration, dropping at most 0.015 from the instruct baseline. A companion study [3] replicates this across 12 model families and sizes (Qwen, Llama, Gemma; 0.5B–9B parameters), reporting mean AUROC 0.982 surviving abliteration within ±0.003. Something in the internal representation of harmful content survives abliteration essentially unchanged.

What geometric approaches cannot answer is the mechanistic question: *which specific features carry the surviving representation? Are they semantically coherent? Do they form a minority of residual-stream directions (easily targeted for restoration) or does survival describe the entire set of RLHF-modified features?* A single AUROC score collapses all of this information into one number. Answering these questions requires a feature-level vocabulary, and Sparse Autoencoders (SAEs) [11, 12] provide exactly that: trained on a model's residual stream, they decompose activations into tens of thousands of near-monosemantic features amenable to semantic interpretation.

We apply the frozen Qwen-Scope base SAE [9]—65,536 TopK features trained on Qwen3-8B residual-stream activations—across three model variants simultaneously: base, instruct (safety RLHF-aligned), and abliterated. Our central metric, the *Reversal Score*, is a per-feature Pearson correlation between the RLHF-induced activation change and the abliteration-induced activation change across 500 harmful prompts. A feature with Reversal Score R ≈ +1 is preserved intact by abliteration (Type-I); a feature with R ≈ -1 is reversed back toward the base model (Type-R). The distribution of Reversal Scores across the top-500 RLHF-sensitive features reveals the mechanistic structure of what abliteration does.

Our central finding is both surprising and mechanistically informative: *no feature has a negative Reversal Score.* The distribution is unimodal and concentrated near R = 1 (mean = 0.921, Hartigan's dip test p = 0.93). Abliteration does not reverse any individual SAE feature—it preserves the entire set of RLHF-modified features. This result implies that the refusal direction is geometrically orthogonal (in the sense of not substantially projecting onto) the RLHF-modified SAE feature directions. Centroid analysis confirms that the abliterated model's feature activations remain 6.1× closer to the instruct model than to the base model in the RLHF-sensitive subspace for harmful prompts. Cross-family semantic labeling finds ≈75% of top RLHF-sensitive features carry harm-concept content throughout the Reversal Score range—harm-concept representation is not concentrated in a surviving minority but pervades the entire RLHF-modified feature set.

[FIGURE:fig1]

**Summary of Contributions:**
- We introduce the *Reversal Score*, a per-feature Pearson correlation that quantifies, for each RLHF-modified SAE feature, whether abliteration reverses or preserves the RLHF-induced activation change (Section 3.3). [ARTIFACT:art_QBm3I0Z5VVJR]
- We present the first empirical result showing that abliteration preserves ALL top RLHF-sensitive features (zero negative Reversal Scores across all tested thresholds), directly implying a specific geometric relationship between the refusal direction and RLHF-modified feature directions (Section 4).
- We pre-register and confirm that the abliterated model's SAE activations remain 6.1× closer to the instruct model than to the base model in the harmful-prompt feature subspace—a direct signature of the latent safety imprint (Section 4.3). [ARTIFACT:art_QBm3I0Z5VVJR]
- We validate an empirical layer-selection procedure for Qwen3-8B (Section 3.4) and the pre-topk encoder fallback for degraded-reconstruction regimes (Section 3.2), establishing reproducible methodology for multi-model SAE comparison.

# Related Work

**Refusal ablation.** Arditi et al. [1] establish that refusal is mediated by a single residual-stream direction, and that orthogonalizing all weight matrices against it reliably eliminates compliance across 13 models up to 72B parameters. Cristofano [7] shows the raw refusal vector is polysemantic, entangling safety circuitry with capability and style components; spectral residualization reduces capability degradation (KL from 2.088 to 0.044 on Wikitext-2) while preserving abliteration. Nanfack et al. [19] propose an optimal-transport alternative that reduces distributional shift. Our work examines what all these methods share: the refusal direction, however constructed, leaves SAE feature-level representations intact.

**Geometry of harmful intent surviving abliteration.** LatentBiopsy [2] and the companion geometry study [3] are the direct predecessors that establish the phenomenon our work explains mechanistically. Their key limitation—collapsing representational information into a single AUROC score—is precisely the gap we address with feature-level analysis. Our finding (zero Type-R features) is consistent with their high AUROC survival: if the refusal direction is orthogonal to all RLHF-modified features, both the harmful-prompt geometry and the individual features survive together.

**SAEs as measurement instruments.** Elhage et al. [11] motivate SAE decomposition via the superposition hypothesis. Bricken et al. [12] demonstrate practical dictionary learning recovering monosemantic features. Gao et al. [13] establish scaling laws for SAE quality. Templeton et al. [14] apply SAEs at Claude 3 Sonnet scale, finding rich safety-relevant semantic structure. Chopra [4] uses frozen base SAEs as cross-variant diagnostics, finding SAE latent cosine similarity drops to 0.509–0.557 at layer 22 after SFT while raw cosine stays above 0.96—validating the frozen base-SAE approach and motivating our pre-topk encoder fallback.

**Multi-model SAE comparison.** Jiralerspong and Bricken [5] introduce *crosscoders*—SAEs trained jointly on two or more model variants—naturally partitioning features into shared and model-specific categories. Kassem et al. [20] extend this with Delta-Crosscoder for narrow fine-tuning regimes, reducing training cost and improving cross-vocabulary interpretability. Our frozen base-SAE approach is complementary: it requires zero additional training and directly reuses Qwen-Scope's existing feature vocabulary, making it applicable wherever a high-quality base SAE already exists. Shportko et al. [6] use Dedicated Feature Crosscoders to localize RL-induced tool-calling in Qwen2.5-3B to a compact steerable feature set—the complement of our finding about which features abliteration *cannot* remove. Li et al. [10] demonstrate an ≈8× MSE gap between base-SAE reconstruction quality on instruct vs. purpose-trained instruct SAE activations, motivating our reconstruction quality gate.

**Safety fine-tuning mechanisms.** Jain et al. [8] establish that safety fine-tuning projects unsafe activations into MLP null spaces, explaining adversarial circumvention. Our finding extends this picture: at the SAE feature level, the modifications RLHF writes are precisely the modifications abliteration cannot erase, suggesting that the refusal direction and the RLHF-modified feature directions are geometrically nearly orthogonal.

# Methodology

## Three-Model Setup

All experiments use three Qwen3-8B checkpoints sharing identical 36-layer architecture (d_model = 4096, bfloat16, transformers ≥ 4.51.0) [ARTIFACT:art_pmAcY0CNazP9]:

*Base*: `Qwen/Qwen3-8B-Base`—the unaligned pretrained model, providing the representational baseline.

*Instruct*: `Qwen/Qwen3-8B` with `enable_thinking=False`—the safety RLHF-aligned model. Chain-of-thought suppression isolates safety-relevant activations on the final input token.

*Abliterated*: `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` (v2 corrects a layer-0 output-garbling artifact present in v1). This community model uses standard difference-of-means orthogonalization applied across layers, consistent with the canonical Arditi et al. [1] recipe. We acknowledge that the exact layer set and prompt set used by this checkpoint are not publicly documented; cross-validation against a scratch-reproduced checkpoint with fully documented parameters is important future work.

All models are accessed via `model.model.layers` for residual-stream hook injection at a single selected layer.

## SAE Measurement Instrument

The frozen Qwen-Scope SAE `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` serves as the shared measurement instrument [9]. [ARTIFACT:art_pmAcY0CNazP9] A critical preliminary finding: Qwen-Scope implements *TopK* architecture (k = 50 features retained via `torch.topk`), not JumpReLU as assumed in earlier literature on these models. For a residual activation $\mathbf{x} \in \mathbb{R}^{4096}$:

$$\mathbf{z}_{\text{pre}} = \mathbf{x}W_{\text{enc}}^\top + \mathbf{b}_{\text{enc}} \in \mathbb{R}^{65536}$$

$$\mathbf{z} = \text{TopK}(\mathbf{z}_{\text{pre}},\; k{=}50)$$

The relevant robustness axis for TopK SAEs is *rank displacement*—whether RLHF elevates one feature's pre-activation at the expense of another's position in the top-50—rather than a threshold-crossing confound. We mitigate this by computing all feature metrics from the continuous pre-topk representation $\mathbf{z}_{\text{pre}}$ when reconstruction quality is insufficient (see Section 3.4).

## Feature Metrics

For each SAE feature $f$ and $N = 500$ harmful prompts $\{p_i\}_{i=1}^N$, let $a_m(f, p_i)$ denote the pre-topk encoder activation of feature $f$ for model $m \in \{\text{base}, \text{inst}, \text{abl}\}$.

**Safety change** quantifies how much RLHF modified feature $f$:
$$\text{SC}(f) = \frac{1}{N}\sum_{i=1}^N |a_{\text{inst}}(f, p_i) - a_{\text{base}}(f, p_i)|$$

Features in the top 500 by SC$(f)$ are *RLHF-sensitive*.

**Reversal Score** quantifies whether abliteration reverses the RLHF modification:
$$\boldsymbol{\Delta}_{\text{inst}}(f) = [a_{\text{inst}}(f, p_i) - a_{\text{base}}(f, p_i)]_{i=1}^N$$
$$\boldsymbol{\Delta}_{\text{abl}}(f) = [a_{\text{abl}}(f, p_i) - a_{\text{base}}(f, p_i)]_{i=1}^N$$
$$R(f) = \text{PearsonR}(\boldsymbol{\Delta}_{\text{inst}}(f), \boldsymbol{\Delta}_{\text{abl}}(f))$$

$R(f) \approx +1$ means abliteration preserves the RLHF change prompt-by-prompt (*Type-I: imprinted*); $R(f) \approx -1$ means abliteration reverses it (*Type-R: refusal-coupled*). This is a Pearson correlation over 500 scalar values per feature—not a per-prompt cosine similarity—so the distributional result cannot be an artifact of the definition.

**Activation-density filter.** TopK SAEs produce sparse vectors: a feature that fires on only a handful of prompts has near-zero $\boldsymbol{\Delta}_{\text{inst}}$ and $\boldsymbol{\Delta}_{\text{abl}}$, making Pearson correlation over a zero-dominated vector unreliable. We restrict Reversal Score computation to features where both $|\boldsymbol{\Delta}_{\text{inst}}(f)|$ and $|\boldsymbol{\Delta}_{\text{abl}}(f)|$ have at least 10% non-zero entries (≥ 50 of 500 prompts). Spearman correlation is computed in parallel as a robustness check against zero inflation.

## Layer Selection and Reconstruction Quality Gate

**Layer selection.** Rather than transferring layer-sensitivity findings from Chopra [4] on Gemma 3 1B (a different architecture and scale), we compute SC$(f)$ variance across all 65,536 features at candidate layers {14, 20, 22} on 50 harmful prompts. The layer with highest variance provides the richest safety-relevant signal; layer 22 is selected empirically (variance 0.1721 vs. 0.0547 at layer 20 and 0.0127 at layer 14). [ARTIFACT:art_QBm3I0Z5VVJR]

**Reconstruction quality gate.** Following Li et al. [10], we compute $R^2$ for all three model variants passed through the frozen base SAE at layer 22. If $R^2 \geq 0.85$ for non-base variants, full encoder-plus-decoder analysis proceeds. If $R^2 < 0.85$, we use encoder-only analysis on the continuous pre-topk activations $\mathbf{z}_{\text{pre}}$. At layer 22, $R^2 = 0.574$ (base), 0.512 (instruct), 0.512 (abliterated)—all below 0.85—so all feature metrics are computed from $\mathbf{z}_{\text{pre}}$.

## Bimodality Analysis

We apply Hartigan's dip test ($H_0$: unimodal) to the Reversal Score distribution for the top-500 RLHF-sensitive features. Both outcomes are pre-specified as informative:

- *Bimodal ($p < 0.05$)*: two mechanistically distinct modification types, with Type-R ($R < 0$) and Type-I ($R > T_I$) features distinguished and analyzed separately for semantic content.
- *Unimodal ($p \geq 0.05$)*: a single graded mechanism, with the full Reversal Score treated as a continuous survival score. Even under unimodality, semantic enrichment at the top of the R distribution (high-R features receiving harm-concept labels at ≥ 2× the rate of low-R features) would confirm a graded latent safety imprint.

The Type-I threshold $T_I$ is set empirically to the 75th percentile (Q3) of the filtered Reversal Score distribution, rather than the previously proposed hardcoded value of +0.5, to avoid driving Criterion 3 enrichment statistics by an arbitrary fixed threshold. We additionally report robustness at fixed thresholds {0.30, 0.50, 0.70, 0.95}.

## Semantic Labeling Protocol

For the top-100 features by $|R(f) - 0.5| \times \text{SC}(f)$, we extract the maximum-activating prompt segment: the ±32 token window around the token with highest absolute pre-topk activation within each prompt, together with the feature index and representative examples. We provide these to Llama-3.1-8B-Instruct via OpenRouter—a cross-family labeler avoiding within-Qwen circularity—and classify each feature as `harm_concept` (weapon synthesis, manipulation, deception, violence), `safety_refusal` (refusal language, hedging), `general_capability`, or `other`.

# Evaluation Corpus

The evaluation corpus comprises 500 harmful and 500 benign prompts assembled from four publicly licensed benchmark datasets [ARTIFACT:art_M8-flaZ0OOaP].

**BeaverTails** [15] (176 harmful): Human-labeled QA pairs spanning 14 harm categories (chemical threats, self-harm, violence, financial crime, and others). The 14-category structure provides the primary source for studying harm-type specificity of SAE features.

**AdvBench** (174 harmful): Standard adversarial behavior dataset used in mechanistic abliteration research [1], providing direct harmful instructions with clear semantic categories.

**JailbreakBench/JBB-Behaviors** [16] (55 harmful + 46 benign): The only source offering matched harmful/benign pairs from the same curation effort, enabling direct contrastive SAE analysis with topic controlled.

**HarmBench** [17] (95 harmful): Standard safety benchmark covering chemical, cybercrime, physical, and illegal categories.

**Dolly-15k** [18] (454 benign): Human-written instruction-following prompts covering brainstorming, classification, QA, generation, and summarization—providing a topically diverse benign baseline.

All five sources are processed through MinHash LSH deduplication (threshold 0.5, 128 permutations, character 5-gram shingling), token-length filtering (≤ 512 tokens), and category capping (≤ 35% per harm subcategory) before balanced sampling to 500 harmful + 500 benign. The final corpus contains 1,000 prompts. We note that an intermediate artifact from the dataset construction pipeline [ARTIFACT:art_M8-flaZ0OOaP] reported 468 examples in the pre-sampling selection; the final experiment corpus reached 500+500 by including AdvBench as an additional high-coverage harmful source beyond the initial four-dataset selection.

# Results

## Layer Selection

Figure 4 shows the empirical Safety Change variance across candidate layers 14, 20, and 22. Layer 22 exhibits the highest variance (0.1721), confirming it as the richest safety-relevant signal layer for Qwen3-8B. Layer 20 is intermediate (0.0547) and layer 14 is stable with low variance (0.0127). All primary results use layer 22; results at layer 20 are reported as a sensitivity check and show qualitatively consistent patterns. The Chopra [4] finding of maximum SFT sensitivity at layer 22 for Gemma 3 1B generalizes empirically to Qwen3-8B—but our layer selection is grounded in the data rather than borrowed from that study.

[FIGURE:fig4]

The $R^2$ gate at layer 22 yields $R^2 = 0.574$ (base), 0.512 (instruct), and 0.512 (abliterated)—all below the 0.85 threshold—triggering the pre-topk encoder-only analysis path. This is consistent with Li et al.'s [10] finding of ≈8× MSE degradation when base SAEs are applied to instruct-model activations. All Reversal Scores reported below use $\mathbf{z}_{\text{pre}}$ (pre-topk encoder activations), which retain discriminative signal even when decoder reconstruction degrades.

## The Reversal Score Distribution

All 500 top RLHF-sensitive features survived the 10% activation-density filter (fraction surviving = 1.00), confirming that the analysis is not limited to a subset of the most active features. The Reversal Score distribution is shown in Figure 2.

The result is unambiguous: *every feature has a positive Reversal Score.* The distribution is tightly concentrated near R = 1 (mean = 0.921, std = 0.040, min = 0.684, max = 0.989). No feature has a negative or near-zero Reversal Score. Hartigan's dip test yields statistic = 0.0123, $p = 0.930$—strongly unimodal with no evidence of bimodality. Spearman correlation, computed as a zero-inflation robustness check, shows consistent results (mean = 0.913).

[FIGURE:fig2]

Table 1 summarizes Type-I and Type-R counts across all tested Reversal Score thresholds:

| Threshold | Type-I ($R >$ threshold) | Type-R ($R <$ neg. threshold) |
|-----------|--------------------------|--------------------------------|
| 0.30      | 500 / 500 (100%)         | 0 / 500 (0%)                  |
| 0.50      | 500 / 500 (100%)         | 0 / 500 (0%)                  |
| 0.70      | 499 / 500 (99.8%)        | 0 / 500 (0%)                  |
| 0.95      | 125 / 500 (25%)          | 0 / 500 (0%)                  |
| Q3 = 0.947| 125 / 500 (25%)          | 0 / 500 (0%)                  |

Across every threshold tested, zero features are classified Type-R. The pre-specified bimodality criterion (Hartigan $p < 0.05$) is not met; the continuous-distribution interpretation applies.

## Representational Centroid Analysis

Criterion 2—the representational fingerprint test—asks whether the abliterated model's feature activations in the RLHF-sensitive subspace are closer to the instruct model than to the base model, specifically for harmful prompts. Figure 3 shows the centroid distances.

[FIGURE:fig3]

For harmful prompts in the Type-I (high-R) feature subspace: $d_{\text{abl-inst}} = 0.00152$ cosine, $d_{\text{abl-base}} = 0.00922$ cosine—the abliterated model's representations are **6.1×** closer to the instruct model. The L2 distances tell the same story: $d_{\text{abl-inst}} = 8.09$ vs. $d_{\text{abl-base}} = 35.58$.

For benign prompts in the same subspace: $d_{\text{abl-inst}} = 0.000459$ cosine, $d_{\text{abl-base}} = 0.00810$ cosine—a 17.6× ratio. The abliterated model is thus closer to instruct than to base even for benign content in the RLHF-sensitive feature subspace. This is a more comprehensive imprint than the pre-registered Criterion 2 predicted. The control condition (expecting $d_{\text{abl-inst}} \approx d_{\text{abl-base}}$ for benign prompts) is not met—the imprint extends across prompt content types. The ratio is moderately smaller for harmful (6.1×) than benign (17.6×), which may reflect that harmful prompts produce larger absolute activation shifts that have slightly more variance across model variants.

Criterion 2 is **confirmed** in the strong form for harmful prompts. The benign-prompt pattern extends the finding beyond what was hypothesized, suggesting the imprint reflects a global shift in how the abliterated model activates RLHF-sensitive features, not a content-conditional effect.

## Semantic Label Distribution

100 features were labeled by Llama-3.1-8B-Instruct (cross-family, ≈$0.004 total cost). The label distribution is: `harm_concept` ≈ 75%, `safety_refusal` ≈ 5%, `general_capability` ≈ 5%, `other` ≈ 15%. Example harm-concept features include those specializing in: identity theft and financial manipulation, synthesis of controlled substances, cyberattack methodologies, hate speech and incitement, and psychological manipulation tactics.

Because the Reversal Score distribution is unimodal with zero Type-R features, the pre-registered Fisher's exact test comparing harm-concept rates in Type-I versus Type-R cannot be computed directly. As a fallback, features above the Q3 threshold (R > 0.947, $n = 4$ in the labeled set) are compared against features below Q1 (R < 0.905, $n = 90$). Harm-concept rates are 75% vs. 71.1% (Fisher $p = 0.675$)—not significantly different. With only 4 features in the high group, this comparison has essentially no statistical power.

The meaningful semantic conclusion is different: harm-concept content is *uniformly distributed* across the entire Reversal Score range (0.684 to 0.989), with approximately 75% of all top-500 RLHF-sensitive features carrying harm-concept labels throughout. This confirms the null for Criterion 3's enrichment test but reveals something stronger: harm-concept representation pervades the RLHF-modified feature set rather than being concentrated in a surviving subset.

# Discussion

## A Geometric Interpretation of the Null Type-R Result

The absence of any Type-R features—features whose RLHF modification is reversed by abliteration—points to a specific geometric structure. Abliteration removes the refusal direction $\mathbf{r}$ from all weight matrices. For abliteration to reverse an RLHF-modified SAE feature $f$ (i.e., push $a_{\text{abl}}(f, \cdot)$ back toward $a_{\text{base}}(f, \cdot)$), the RLHF-induced change in feature $f$ must have a meaningful component along $\mathbf{r}$. Our finding that no feature is reversed implies that the RLHF-induced changes to all 500 top features are essentially orthogonal to the refusal direction $\mathbf{r}$ in the residual stream of Qwen3-8B at layer 22.

This interpretation is consistent with Cristofano's [7] observation that the raw refusal direction is polysemantic, entangling non-safety capability circuits. If $\mathbf{r}$ primarily captures a mixture of stylistic and compliance patterns rather than semantic harm-concept directions, it would naturally project weakly onto the semantic harm-concept SAE features identified by Safety Change analysis. Abliteration then removes the refusal compliance signal without touching the underlying semantic representation.

This interpretation also explains the LatentBiopsy [2, 3] AUROC result from the bottom up: AUROC ≈ 0.982 survives abliteration because all the features that distinguish harmful from benign content—the SAE features modified by RLHF in response to harmful prompts—survive intact.

## Revision of the Latent Safety Imprint Hypothesis

The original hypothesis distinguished Type-R (reversed) from Type-I (preserved) features, predicting that a bimodal distribution would reveal a specific subset of harm-concept features. The experimental finding is more extreme: the imprint is total. No feature at layer 22 is reversed by abliteration. The revised hypothesis is that the refusal direction and the RLHF-modified SAE feature directions are geometrically near-orthogonal at this layer, meaning abliteration and RLHF are not in direct opposition at the feature level at all.

The centroid analysis adds a further refinement: the imprint extends to benign prompts (17.6× closer to instruct than base) as well as harmful prompts (6.1×). This suggests that RLHF modifies these features in a way that affects their responses generally—not just their responses to harmful content—and abliteration preserves this global shift.

## Implications for Safety Restoration

If the latent safety imprint is comprehensive—all RLHF-modified features surviving, not a select minority—then targeted safety restoration by steering only the identified Type-I features is not the right framing. Instead, the challenge is to restore the *compliance behavior* (the refusal direction) without modifying the already-intact feature-level semantics. This reframes the restoration problem: the semantic knowledge is already there; what abliteration removes is only the geometric direction that converts harm-concept feature activations into refusal outputs. Restoration may require adding back the refusal direction in a targeted way—a problem of geometric re-injection rather than semantic amplification.

The DFC approach of Shportko et al. [6] identifies compact steerable feature sets for RL-induced capabilities; an analogous method that identifies the specific features mediating conversion from harm-concept detection to refusal behavior could enable such targeted re-injection.

## Limitations

*Single model and layer.* All results are for Qwen3-8B at layer 22. The Reversal Score distribution and the geometric orthogonality claim may not generalize to other model families (Llama, Gemma) or other architectures. The total absence of Type-R features at layer 22 should be verified at layers 14 and 20 as well; partial reversal at stable layers cannot be ruled out.

*Community abliterated checkpoint.* We used `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` without independently verifying that it achieves comparable abliteration depth to a scratch-reproduced checkpoint on geometric metrics (AUROC, KL divergence, refusal rate). If the community checkpoint uses a non-standard method that produces weaker abliteration, our finding of high Reversal Scores could partially reflect incomplete abliteration rather than semantic preservation. Verification against a scratch-reproduced checkpoint with documented parameters is critical for the next iteration.

*Base-SAE fidelity.* $R^2 \approx 0.51$ for instruct and abliterated variants at layer 22 indicates substantial reconstruction degradation. The pre-topk encoder-only path is a validated fallback [4], but the encoder may assign features inaccurately when the input distribution shifts significantly from base training data. An instruct-trained Qwen3-8B SAE for layers above 8 would resolve this limitation.

*Semantic labeler calibration.* No manual gold-set annotation was performed in this study to compute Cohen's $\kappa$ for the Llama-3.1-8B labeler. While the labeler is cross-family, its RLHF training may produce harm-concept category boundaries similar to Qwen3-8B's. Future iterations should validate against a 20-feature manual gold set before trusting the ≈75% harm-concept rate.

*No bimodality, no Type-R/Type-I enrichment test.* The pre-registered Criterion 3 (Fisher exact test on harm-concept rates in Type-I vs. Type-R) cannot be meaningfully computed without Type-R features. The Q3/Q1 fallback has only 4 features in the high group, giving it essentially no statistical power. This is not a failed experiment—it is a consequence of the primary finding—but it means the specific enrichment claim is unresolvable within this experimental design.

# Conclusion

We conducted the first feature-level analysis of refusal abliteration using the frozen Qwen-Scope base SAE across three Qwen3-8B variants. The Reversal Score metric, computed for each of the top 500 RLHF-sensitive SAE features, reveals a striking null result on the Type-R/Type-I bimodality hypothesis: every feature has a positive Reversal Score (mean = 0.921, std = 0.040), confirming that abliteration preserves the entire set of RLHF-modified features at layer 22. This result has a clean geometric interpretation: the refusal direction operated on by abliteration is nearly orthogonal to the SAE feature directions modified by RLHF, so removing the former does not disturb the latter. Centroid analysis confirms that the abliterated model's SAE activations remain 6.1× closer to the instruct model than to the base model for harmful prompts. Approximately 75% of top RLHF-sensitive features carry harm-concept semantic labels throughout the Reversal Score range, indicating that the surviving representations are semantically meaningful. The latent safety imprint is therefore comprehensive and consistent with the near-perfect AUROC survival reported by geometric approaches.

**Future directions** include: (1) verification against a scratch-reproduced abliterated checkpoint with documented parameters, to confirm the geometric orthogonality claim is not an artifact of incomplete abliteration; (2) multi-layer analysis across all 36 Qwen-Scope layers to determine whether layer 22's pattern of zero Type-R features holds globally or breaks down at shallower layers; (3) cross-family extension to Llama-3.1-8B and Gemma-3-9B to test generalizability of the geometric orthogonality hypothesis; and (4) investigation of which specific weight matrices implement the refusal direction conversion—i.e., how harm-concept feature activations are transformed into refusal outputs—as this is the mechanism that abliteration removes.

# References

[1] A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Rimsky, W. Gurnee, and N. Nanda. Refusal in language models is mediated by a single direction. In *Advances in Neural Information Processing Systems*, 2024.

[2] I. Llorente-Saguer. The geometry of harmful intent: Training-free anomaly detection via angular deviation in LLM residual streams. *arXiv preprint arXiv:2603.27412*, 2026.

[3] I. Llorente-Saguer. Harmful intent as a geometrically recoverable feature of LLM residual streams. *arXiv preprint arXiv:2604.18901*, 2026.

[4] R. Chopra. A mechanistic investigation of supervised fine tuning. *arXiv preprint arXiv:2605.11426*, 2026.

[5] T. Jiralerspong and T. Bricken. Cross-architecture model diffing with crosscoders: Unsupervised discovery of differences between LLMs. *arXiv preprint arXiv:2602.11729*, 2026.

[6] A. Shportko, S. Bhokare, A. Z. A. Alzahrani, B. Cheng, G. Mercier, and J. Hullman. Localizing RL-induced tool use to a single crosscoder feature. In *ICML 2026 Workshop on Mechanistic Interpretability* (Spotlight), 2026.

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

[20] A. M. Kassem, T. Jiralerspong, N. Rostamzadeh, and G. Farnadi. Delta-Crosscoder: Robust crosscoder model diffing in narrow fine-tuning regimes. *arXiv preprint arXiv:2603.04426*, 2026.
</current_paper>

<reviewer_feedback>
Paper reviewer feedback from the previous iteration. Your strategy MUST address these critiques.
Prioritize major issues — these are the most impactful improvements to make.

- [MAJOR] (methodology) The density filter is trivially satisfied by pre-topk (dense) activations and the paper misrepresents this as a meaningful quality check. The paper states 'all 500 top RLHF-sensitive features survived the 10% activation-density filter (fraction surviving = 1.00)' — but inspection of the code (method.py line 520) reveals the filter checks `abs(delta) > 1e-6`. Pre-topk activations (z_pre = h @ W_enc.T + b_enc) are dense continuous projections: every feature has a non-zero value on every prompt by construction, so 100% survival is mathematically guaranteed regardless of feature quality. The filter was designed for post-topk sparse activations where a feature could genuinely fire on fewer than 50 prompts. Reporting trivially-guaranteed 100% survival as evidence of analysis coverage is misleading.
  Action: Add a sentence explicitly acknowledging that with pre-topk (dense) activations, the density filter cannot distinguish sparse from dense features — all 65,536 features have non-zero values on all 500 prompts. Instead, characterize the effective density by reporting, for the top-500 RLHF-sensitive features, the fraction of prompts on which the post-topk activation would be non-zero (i.e., the feature would be in the active top-50). This gives an honest picture of how 'SAE-feature-like' these 500 features are in the post-topk sense.
- [MAJOR] (evidence) The near-chance classification accuracy (0.476, essentially 50/50 on a balanced dataset) is reported in the artifact summary but entirely absent from the paper body. The artifact states: 'Our method accuracy=0.476 (TP=238/500, TN=238/500) using median harm score threshold — near-chance.' The keyword baseline achieves 0.630. The paper discusses the Reversal Score analysis at length but never mentions that the derived harm score (Type-I activation minus Type-R activation) cannot distinguish harmful from benign prompts. A reviewer who reads the artifact will notice this omission and interpret it as cherry-picking positive results.
  Action: Add a paragraph in Section 5 (or Section 6 Discussion) reporting the classification accuracy: 'As a downstream application test, we computed a per-prompt harm score as mean Type-I feature activation minus mean Type-R activation and predicted harmful vs. benign using a median threshold. The method achieved 0.476 accuracy, near-chance on the balanced corpus. This is expected given the unimodal R distribution: since all features have high Reversal Scores, the Type-I/Type-R distinction provides no discrimination signal.' The explanation is consistent with the main finding and should be presented openly.
- [MAJOR] (methodology) The semantic labeling protocol described in the paper does not match the code implementation. The paper states: 'we extract the maximum-activating prompt segment: the ±32 token window around the token with highest absolute pre-topk activation within each prompt.' However, the code (method.py lines 666-673) does not perform any token-level activation analysis. Instead, it identifies the 5 prompts with the highest scalar feature activation (averaged across all tokens in the prompt via the last-token hidden state), then passes the first 300 characters of each prompt to the labeler. No ±32 token window is computed. No per-token activation analysis is performed. The described protocol would require token-level hook extraction, which is absent from the codebase.
  Action: Correct the paper description to match the actual code: 'We identify the top-5 prompts (by maximum pre-topk encoder activation for feature f on the last input token) and provide the first 300 characters of each to the cross-family labeler.' Additionally, note the limitation of using the full-prompt first-300-characters rather than a feature-specific activation window: this may cause the labeler to assign labels based on prompt topic rather than the specific token that most activates the feature.
- [MAJOR] (rigor) The high Reversal Scores (R≈0.921) lack a null baseline, making it impossible to determine whether this reflects 'RLHF-modified features specifically survive abliteration' versus the simpler explanation 'instruct and abliterated models have very similar residual-stream activations overall, so almost all encoder projections are correlated.' Since abliteration applies a targeted orthogonalization to only a few directions in a 4096-dimensional space, most of the residual stream is unaffected. The expected Reversal Score for randomly sampled features not modified by RLHF could also be near 1.0, in which case R≈0.921 for RLHF-sensitive features is uninformative.
  Action: Compute Reversal Scores for the BOTTOM-500 features by Safety Change (i.e., features least modified by RLHF) and report their distribution alongside the top-500. If both distributions are similarly concentrated near R=1, then the high R values reflect global similarity between instruct and abliterated models rather than selective preservation of RLHF-modified features. If the RLHF-sensitive features show significantly higher R than the control features, that would be strong positive evidence for the selective preservation claim.
- [MAJOR] (methodology) The paper's claim of 'SAE feature-level analysis' rests on pre-topk (dense) encoder projections, but the interpretability of SAE features — the foundation for semantic claims about 'harm-concept' labels — was validated for post-topk sparse activations. The paper uses pre-topk because R²≈0.51 (all variants), triggering the fallback path. Pre-topk values (z_pre = h @ W_enc.T + b_enc) are a dense 65,536-dimensional projection of the residual stream. The Qwen-Scope documentation endorses applying the base SAE to post-training variants, but this endorsement pertains to post-topk sparse feature interpretations, not to the raw pre-topk encoder output as a 65K-dimensional feature vector. The semantic labeling and the 'feature-level' framing throughout the paper implicitly assume pre-topk encoder directions are as interpretable as post-topk sparse features, which has not been validated.
  Action: Add a paragraph in the limitations section explicitly distinguishing 'SAE-derived features' (post-topk, validated interpretable) from 'SAE encoder projections' (pre-topk, dense, interpretability not directly validated). Acknowledge that the semantic claims (≈75% harm-concept rate) rest on encoder projections rather than the sparse post-topk features that the mechanistic interpretability literature has validated. Alternatively, request an instruct SAE for layers above 8 (the paper notes one exists for layers 0-8 only) which would enable post-topk analysis.
- [MINOR] (evidence) The paper claims 'results at layer 20 are reported as a sensitivity check and show qualitatively consistent patterns' but no layer-20 numbers appear anywhere in the Results or Appendix. This is an unsubstantiated claim: the code computes SECOND_LAYER=20 and logs it, but the full experiment is only run at PRIMARY_LAYER=22. Asserting consistency without reporting the actual numbers is not a sensitivity check.
  Action: Either report the actual layer 20 results (Reversal Score distribution mean/std, Type-R count) as a numbered table or remove the claim about layer 20 consistency. Given that the artifact reports 4.3 minutes total runtime, running the full analysis at layer 20 as well appears computationally feasible.
- [MINOR] (rigor) The benign-prompt centroid imprint (17.6× closer to instruct than to base in the RLHF-sensitive subspace) is larger than the harmful-prompt imprint (6.1×). The paper notes this as 'more comprehensive than predicted' but does not consider the simpler explanation: instruct and abliterated models are globally similar (abliteration changes a few weight directions), so their activations in ANY subspace — not just the RLHF-sensitive one — are close to each other relative to base. The benign-prompt ratio being LARGER than the harmful-prompt ratio is actually consistent with the null hypothesis that abliteration makes global small changes rather than selective preservation of harm-related features.
  Action: Report centroid distances in a control subspace — e.g., the top-500 features by Safety Change on BENIGN prompts (features that distinguish benign from base, if any exist) — to show that the closeness to instruct is not uniform across all feature subspaces. If the RLHF-sensitive subspace shows systematically higher instruct-closeness than a random or benign-sensitive subspace, that would support the selective preservation claim.
- [MINOR] (rigor) The Q3/Q1 fallback Fisher test (n=4 in the high group, n=90 in the low group) has essentially zero statistical power and the harm-concept rates (75% vs 71.1%, p=0.675) are not meaningfully different. The paper correctly acknowledges this, but the test still appears in the main results (Table 1 footnote equivalent). For a pre-registered paper, reporting an underpowered test as a 'fallback' result can be misinterpreted as a weak positive signal when it is actually an uninterpretable measurement.
  Action: Remove the Fisher fallback test from the quantitative results summary and report only the qualitative conclusion: 'Because all features have positive Reversal Scores, the pre-registered Type-I vs Type-R comparison cannot be computed. The unconditional harm-concept rate of ≈75% across all top-500 RLHF-sensitive features is the primary semantic finding.' The fallback test should be moved to limitations or dropped entirely.
- [MINOR] (rigor) The geometric orthogonality claim — 'the RLHF-induced changes to all 500 top features are essentially orthogonal to the refusal direction r in the residual stream of Qwen3-8B at layer 22' — is inferred from the zero Type-R result rather than directly measured. The refusal direction r is available (as the mean difference of harmful vs. harmless residual-stream activations that the abliteration method operates on). The direct measurement would be to compute the cosine similarity between r and each of the top-500 RLHF-modified SAE encoder directions (W_enc[feature_idx, :] projected back into residual stream space via W_dec), which would give a direct measure of the claimed orthogonality.
  Action: Compute the refusal direction r using the standard Arditi et al. procedure (mean difference of instruct activations on harmful vs. harmless prompts at layer 22) and report the cosine similarity between r and the top-10 RLHF-sensitive SAE encoder directions. If these are indeed near-orthogonal (|cos| < 0.1), this would directly confirm the geometric claim rather than inferring it from the R distribution.
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

### [3] HUMAN-USER prompt · 2026-07-17 12:26:23 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [4] ASSISTANT · 2026-07-17 12:29:05 UTC

```
Looking at the reviewer feedback, I need to plan a strategy that addresses the four MAJOR critiques — all requiring new computation on top of the existing iter2 results:

1. **Null baseline missing**: Need bottom-500 features Reversal Scores to compare against top-500
2. **No direct orthogonality measurement**: Need cosine similarity between refusal direction r and RLHF-sensitive encoder directions
3. **Control subspace centroid missing**: Need centroid distances in RLHF-insensitive and random subspaces
4. **Layer-20 results unsubstantiated**: Need to actually run and report layer-20 numbers

These are all computationally cheap (reuse cached activations from iter2's 4.3-minute run) but critical for the paper's credibility.
```

### [5] TOOL CALL — Write · 2026-07-17 12:29:05 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_3/gen_strat/gen_strat_1/.terminal_claude_agent_struct_out.json

{
  "strategies": [
    {
      "title": "Null Baseline + Orthogonality Validation",
      "objective": "Resolve the four MAJOR reviewer critiques by computing: (1) Reversal Scores for bottom-500 RLHF-insensitive features as a null baseline to determine whether R≈0.921 reflects selective preservation or global instruct/abliterated similarity; (2) direct cosine similarity between the refusal direction r and RLHF-sensitive SAE encoder directions to confirm or refute the geometric-orthogonality claim; (3) centroid distances in control subspaces (RLHF-insensitive and random) alongside the RLHF-sensitive subspace; and (4) layer-20 Reversal Score distribution to substantiate the claimed sensitivity-check consistency.",
      "rationale": "All four MAJOR critiques share a single diagnosis: the current results are consistent with the null hypothesis that instruct and abliterated models are globally similar (abliteration touches only a few of 4096 dimensions), making every encoder projection correlated. The null-baseline comparison (bottom-500 features) is the decisive test: if RLHF-sensitive features have significantly higher R than RLHF-insensitive features, selective preservation is confirmed; if both concentrate near R=1, the finding collapses to a trivial global-similarity artifact. The orthogonality measurement (cos_sim between r and encoder directions) directly validates the geometric interpretation instead of inferring it from R. The control-subspace centroid analysis answers whether the 6.1×/17.6× ratios are specific to the RLHF-sensitive subspace or universal. Layer-20 numbers are computationally free given that activations were already extracted. All computations are linear-algebra operations on pre-extracted numpy arrays — the iter2 experiment ran in 4.3 minutes total, and these additions add minimal overhead. Together they convert inference into direct measurement, addressing every MAJOR critique without requiring model re-loading if activation caches are available.",
      "artifact_directions": [
        {
          "type": "experiment",
          "objective": "Compute null-baseline Reversal Scores (bottom-500 RLHF-insensitive features), refusal-direction cosine similarities (r vs. top-500 SAE encoder directions), control-subspace centroid distances (RLHF-insensitive + random subspaces), post-topk density characterization for top-500 features, and full layer-20 Reversal Score analysis — all addressing MAJOR reviewer critiques about the absence of null comparisons and direct orthogonality evidence.",
          "approach": "Use UV with packages: transformers>=4.51.0, torch, huggingface_hub, scipy, numpy, diptest. Load from the iter2 experiment workspace (art_QBm3I0Z5VVJR) to reuse any saved activation arrays. If activation files (base/instruct/abliterated residuals at layers 20 and 22) are not found, re-extract them by loading each model sequentially (bfloat16, device_map='auto') and hooking model.model.layers[N].output.\n\n**Step 0 — Load dependencies.** Check art_QBm3I0Z5VVJR workspace for saved numpy arrays: base_act_L22.npy, inst_act_L22.npy, abl_act_L22.npy (shape: n_prompts × 4096), and corresponding L20 arrays. Also load the prompt corpus from art_M8-flaZ0OOaP (harmful and benign prompts). If activation arrays are absent, reload all three models and re-extract — sequential load with torch.cuda.empty_cache() between models.\n\n**Step 1 — Recompute baseline SC and load SAE.** Load Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 via hf_hub_download (layer22.sae.pt). Reconstruct W_enc (65536×4096), b_enc (65536), W_dec (4096×65536). Compute z_pre_base = base_act_L22 @ W_enc.T + b_enc, z_pre_inst = inst_act_L22 @ W_enc.T + b_enc, z_pre_abl = abl_act_L22 @ W_enc.T + b_enc. SC(f) = mean_i |z_pre_inst[harmful, f] - z_pre_base[harmful, f]| for all 65536 features. Sort by SC descending: top-500 (RLHF-sensitive) and bottom-500 (RLHF-insensitive) indices.\n\n**Step 2 — Null-baseline Reversal Scores (bottom-500).** For each bottom-500 feature f, compute: delta_inst(f) = z_pre_inst[harmful_idx, f] - z_pre_base[harmful_idx, f]; delta_abl(f) = z_pre_abl[harmful_idx, f] - z_pre_base[harmful_idx, f]; R_control(f) = pearsonr(delta_inst(f), delta_abl(f))[0]. Report distribution: mean, std, min, max, histogram. Run Hartigan's dip test on R_control values. Compare top-500 vs bottom-500: two-sample t-test (scipy.stats.ttest_ind), Kolmogorov-Smirnov test (scipy.stats.ks_2samp), Cohen's d = (mean_top - mean_bottom) / pooled_std. This is the decisive test: if mean_top ≈ mean_bottom ≈ 0.921, the finding is a global-similarity artifact; if mean_top >> mean_bottom, selective preservation is confirmed.\n\n**Step 3 — Post-topk density characterization for top-500.** For each harmful prompt i, find the TopK=50 threshold: threshold_i = the 50th largest value in z_pre_inst[i, :] (i.e., np.partition(z_pre_inst[i], -50)[-50]). For each top-500 feature f, compute density(f) = fraction of harmful prompts where z_pre_inst[i, f] >= threshold_i. Report mean, median, std, min of density across the 500 features. This honestly characterizes how 'SAE-feature-like' these features are in the post-topk sense — addresses the MAJOR critique about trivially-satisfied density filter.\n\n**Step 4 — Refusal direction computation.** Using instruct model activations: r_raw = mean(inst_act_L22[harmful_idx], axis=0) - mean(inst_act_L22[benign_idx], axis=0). Normalize: r = r_raw / norm(r_raw). Shape: (4096,). This is the refusal direction in residual-stream space at layer 22 — the same space as W_enc rows and W_dec columns.\n\n**Step 5 — Cosine similarity: r vs. SAE encoder directions.** For each top-500 RLHF-sensitive feature f: enc_dir = W_enc[f, :] / norm(W_enc[f, :]). cos_enc(f) = dot(r, enc_dir). Also compute dec_dir = W_dec[:, f] / norm(W_dec[:, f]); cos_dec(f) = dot(r, dec_dir). Report: (a) distribution of |cos_enc(f)| across all 500 features — mean, std, max; (b) individual values for top-10 RLHF-sensitive features by SC; (c) distribution of |cos_dec(f)|. Threshold claim: if mean |cos_enc| < 0.1, geometric orthogonality is directly confirmed. Also compute cos_sim between r and each of the bottom-500 control features' encoder directions — this provides a reference: if r is equally uncorrelated with both RLHF-sensitive and RLHF-insensitive feature directions, orthogonality is a property of the SAE basis in general, not specific to RLHF-sensitive features.\n\n**Step 6 — Control-subspace centroid distances.** Define three subspaces: (A) top-500 RLHF-sensitive features (existing result), (B) bottom-500 RLHF-insensitive features, (C) random 500 features (np.random.seed(42), sample from features not in top/bottom 500). For each subspace S and each prompt-type (harmful/benign): centroid_abl = mean(z_pre_abl[prompts, S], axis=0); centroid_inst = mean(z_pre_inst[prompts, S], axis=0); centroid_base = mean(z_pre_base[prompts, S], axis=0). d_abl_inst = cosine_distance(centroid_abl, centroid_inst); d_abl_base = cosine_distance(centroid_abl, centroid_base); ratio = d_abl_base / d_abl_inst. Report all ratios in a 3×2 table (subspace × prompt_type). If ratios in B and C are similar to A (6.1×, 17.6×), the centroid finding reflects global similarity not selective preservation.\n\n**Step 7 — Layer-20 full Reversal Score analysis.** Load layer20.sae.pt SAE. Compute z_pre at layer 20 for all three models and all prompts. SC_L20(f) for all 65536 features. Top-500 by SC_L20. Reversal Scores R_L20(f) for top-500. Report: mean, std, min, max, Hartigan dip test p-value, fraction with R > 0.7, count of negative R values. Compare distribution against layer-22 top-500. Also compute null-baseline (bottom-500) R at layer 20.\n\n**Output format (method_out.json):** {\n  layer22_results: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, top500_dip_p, bottom500_R_mean, bottom500_R_std, bottom500_R_min, bottom500_R_max, bottom500_dip_p, ttest_t, ttest_p, ks_stat, ks_p, cohens_d },\n  posttopk_density: { top500_density_mean, top500_density_median, top500_density_min, top500_density_std },\n  orthogonality: { cos_enc_mean_abs, cos_enc_max_abs, cos_enc_std, cos_dec_mean_abs, cos_dec_max_abs, top10_cos_enc_values (list), bottom500_cos_enc_mean_abs, orthogonality_confirmed (bool, cos_enc_mean_abs < 0.1) },\n  centroid_control: { subspaceA_harmful_ratio, subspaceA_benign_ratio, subspaceB_harmful_ratio, subspaceB_benign_ratio, subspaceC_harmful_ratio, subspaceC_benign_ratio },\n  layer20_results: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, top500_dip_p, n_negative_R, bottom500_R_mean, bottom500_R_std }\n}",
          "depends_on": [
            {
              "id": "art_QBm3I0Z5VVJR",
              "label": "cached activations"
            },
            {
              "id": "art_M8-flaZ0OOaP",
              "label": "prompt corpus"
            },
            {
              "id": "art_pmAcY0CNazP9",
              "label": "tech reference"
            }
          ]
        },
        {
          "type": "evaluation",
          "objective": "Synthesize all experimental results (iter2 + iter3 null-baseline experiment) into a clean, statistically rigorous summary that resolves each MAJOR reviewer critique with a definitive verdict — selective preservation vs. global similarity — and produces corrected statistics, updated table contents, and fixed paper-level descriptions for the GEN_PAPER_TEXT step.",
          "approach": "Use UV with packages: scipy, numpy, pandas. Load method_out.json from the null-baseline experiment (depends_on) and full_method_out.json from art_QBm3I0Z5VVJR.\n\n**Section 1 — Null-baseline verdict.** Apply the following decision rule: if ttest_p < 0.05 AND cohens_d > 0.5 (large effect), verdict = 'SELECTIVE_PRESERVATION_CONFIRMED'; if ks_p < 0.05 but cohens_d <= 0.5, verdict = 'WEAK_SELECTIVE_PRESERVATION'; otherwise verdict = 'GLOBAL_SIMILARITY_ALTERNATIVE_PLAUSIBLE'. Report the actual top-500 vs. bottom-500 mean difference with 95% CI (bootstrap, 1000 samples). Produce a side-by-side histogram data table (top-500 R distribution vs. bottom-500 R distribution) with bin edges [0.0, 0.1, ..., 1.0].\n\n**Section 2 — Orthogonality verdict.** If cos_enc_mean_abs < 0.1: verdict = 'ORTHOGONALITY_DIRECTLY_CONFIRMED'. If 0.1 <= cos_enc_mean_abs < 0.2: verdict = 'NEAR_ORTHOGONAL'. If >= 0.2: verdict = 'ORTHOGONALITY_CLAIM_WEAKENED'. Report top-10 cos_enc values ranked by |cos_enc|. Note whether bottom-500 RLHF-insensitive features show similar or different |cos_enc| — if similar, orthogonality is a general property of the SAE basis rather than specific to RLHF-sensitive features.\n\n**Section 3 — Centroid verdict.** Compare ratios: if subspaceB_harmful_ratio and subspaceC_harmful_ratio are both within 50% of subspaceA_harmful_ratio (6.1×), verdict = 'CENTROID_REFLECTS_GLOBAL_SIMILARITY'; if A ratio is > 2× both B and C ratios, verdict = 'CENTROID_REFLECTS_SELECTIVE_PRESERVATION'. Report all 6 ratios in a formatted table.\n\n**Section 4 — Layer-20 consistency.** Report layer-20 Reversal Score distribution statistics alongside layer-22. Compute whether distributions are statistically different (KS test between layer-22 top-500 R values and layer-20 top-500 R values). State explicitly whether the zero-Type-R finding replicates at layer 20.\n\n**Section 5 — Post-topk density.** Report actual density statistics. Compute: what fraction of the top-500 features have density > 0.10 (the intended filter threshold)? This is the honest version of the density filter result — corrects the MAJOR critique.\n\n**Section 6 — Paper corrections checklist.** Produce a structured list of required paper changes:\n  (a) Semantic labeling protocol: correct description to 'top-5 prompts by maximum pre-topk activation on last token, first 300 characters'\n  (b) Density filter: replace '100% survived density filter' with actual post-topk density statistics\n  (c) Classification accuracy: add paragraph reporting 0.476 accuracy as expected null result\n  (d) Fisher fallback test: move Q3/Q1 comparison (n=4) to limitations, report only unconditional 75% harm-concept rate as primary semantic finding\n  (e) Centroid analysis: add control-subspace rows to Figure 3 table\n  (f) Pre-topk/post-topk distinction: add paragraph in limitations distinguishing SAE features (post-topk, validated) from SAE encoder projections (pre-topk, dense, interpretability not validated)\n\n**Output format (eval_out.json):** { null_baseline_verdict, selective_preservation_mean_diff, selective_preservation_95ci, orthogonality_verdict, centroid_verdict, layer20_replicates_zero_type_r (bool), posttopk_density_above_threshold_fraction, paper_corrections (list of strings), overall_claim_status ('STRENGTHENED' | 'UNCHANGED' | 'WEAKENED' | 'UNDERMINED'), summary_paragraph (string for paper Discussion section update) }",
          "depends_on": [
            {
              "id": "art_QBm3I0Z5VVJR",
              "label": "iter2 results"
            }
          ]
        }
      ],
      "expected_outcome": "After iteration 3, the paper will have direct answers to all four MAJOR critiques: (1) The null-baseline comparison will definitively settle whether R≈0.921 reflects selective preservation (top-500 significantly higher than bottom-500) or global similarity (both near 1.0), with a t-test p-value, KS statistic, and Cohen's d; (2) Direct cosine-similarity measurements between the refusal direction r and top-500 RLHF-sensitive encoder directions will confirm or refute the geometric-orthogonality claim (expected: |cos| < 0.1); (3) Control-subspace centroid ratios (RLHF-insensitive, random) will contextualize the 6.1×/17.6× ratios and determine whether they are specific to the RLHF-sensitive subspace; (4) Layer-20 Reversal Score statistics will be reported numerically for the first time, either confirming zero Type-R features replicates or revealing that it is layer-22-specific. The evaluation artifact will produce a corrected paper-corrections checklist covering the semantic-labeling description fix, classification-accuracy disclosure, Fisher-test removal, density-filter correction, and pre-topk/post-topk limitation paragraph — giving GEN_PAPER_TEXT everything needed for a complete revision.",
      "summary": "Iteration 3 resolves the credibility gap in the iter2 paper by computing the three missing null comparisons (bottom-500 Reversal Scores, direct refusal-direction orthogonality, control-subspace centroid distances) and substantiating the layer-20 sensitivity check with actual numbers. All computations reuse cached activations from iter2, keeping runtime under one hour. The evaluation synthesizes all results into definitive verdicts on selective preservation vs. global similarity, direct orthogonality confirmation, and centroid specificity — replacing inference with measurement. By the end of iteration 3, every MAJOR reviewer critique will have a numerical answer."
    }
  ]
}
```

### [6] TOOL RESULT — Write · 2026-07-17 12:29:05 UTC

```
{"type": "create", "filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_3/gen_strat/gen_strat_1/.terminal_claude_agent_struct_out.json", "content": "{\n  \"strategies\": [\n    {\n      \"title\": \"Null Baseline + Orthogonality Validation\",\n      \"objective\": \"Resolve the four MAJOR reviewer critiques by computing: (1) Reversal Scores for bottom-500 RLHF-insensitive features as a null baseline to determine whether R≈0.921 reflects selective preservation or global instruct/abliterated similarity; (2) direct cosine similarity between the refusal direction r and RLHF-sensitive SAE encoder directions to confirm or refute the geometric-orthogonality claim; (3) centroid distances in control subspaces (RLHF-insensitive and random) alongside the RLHF-sensitive subspace; and (4) layer-20 Reversal Score distribution to substantiate the claimed sensitivity-check consistency.\",\n      \"rationale\": \"All four MAJOR critiques share a single diagnosis: the current results are consistent with the null hypothesis that instruct and abliterated models are globally similar (abliteration touches only a few of 4096 dimensions), making every encoder projection correlated. The null-baseline comparison (bottom-500 features) is the decisive test: if RLHF-sensitive features have significantly higher R than RLHF-insensitive features, selective preservation is confirmed; if both concentrate near R=1, the finding collapses to a trivial global-similarity artifact. The orthogonality measurement (cos_sim between r and encoder directions) directly validates the geometric interpretation instead of inferring it from R. The control-subspace centroid analysis answers whether the 6.1×/17.6× ratios are specific to the RLHF-sensitive subspace or universal. Layer-20 numbers are computationally free given that activations were already extracted. All computations are linear-algebra operations on pre-extracted numpy arrays — the iter2 experiment ran in 4.3 minutes total, and these additions add minimal overhead. Together they convert inference into direct measurement, addressing every MAJOR critique without requiring model re-loading if activation caches are available.\",\n      \"artifact_directions\": [\n        {\n          \"type\": \"experiment\",\n          \"objective\": \"Compute null-baseline Reversal Scores (bottom-500 RLHF-insensitive features), refusal-direction cosine similarities (r vs. top-500 SAE encoder directions), control-subspace centroid distances (RLHF-insensitive + random subspaces), post-topk density characterization for top-500 features, and full layer-20 Reversal Score analysis — all addressing MAJOR reviewer critiques about the absence of null comparisons and direct orthogonality evidence.\",\n          \"approach\": \"Use UV with packages: transformers>=4.51.0, torch, huggingface_hub, scipy, numpy, diptest. Load from the iter2 experiment workspace (art_QBm3I0Z5VVJR) to reuse any saved activation arrays. If activation files (base/instruct/abliterated residuals at layers 20 and 22) are not found, re-extract them by loading each model sequentially (bfloat16, device_map='auto') and hooking model.model.layers[N].output.\\n\\n**Step 0 — Load dependencies.** Check art_QBm3I0Z5VVJR workspace for saved numpy arrays: base_act_L22.npy, inst_act_L22.npy, abl_act_L22.npy (shape: n_prompts × 4096), and corresponding L20 arrays. Also load the prompt corpus from art_M8-flaZ0OOaP (harmful and benign prompts). If activation arrays are absent, reload all three models and re-extract — sequential load with torch.cuda.empty_cache() between models.\\n\\n**Step 1 — Recompute baseline SC and load SAE.** Load Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 via hf_hub_download (layer22.sae.pt). Reconstruct W_enc (65536×4096), b_enc (65536), W_dec (4096×65536). Compute z_pre_base = base_act_L22 @ W_enc.T + b_enc, z_pre_inst = inst_act_L22 @ W_enc.T + b_enc, z_pre_abl = abl_act_L22 @ W_enc.T + b_enc. SC(f) = mean_i |z_pre_inst[harmful, f] - z_pre_base[harmful, f]| for all 65536 features. Sort by SC descending: top-500 (RLHF-sensitive) and bottom-500 (RLHF-insensitive) indices.\\n\\n**Step 2 — Null-baseline Reversal Scores (bottom-500).** For each bottom-500 feature f, compute: delta_inst(f) = z_pre_inst[harmful_idx, f] - z_pre_base[harmful_idx, f]; delta_abl(f) = z_pre_abl[harmful_idx, f] - z_pre_base[harmful_idx, f]; R_control(f) = pearsonr(delta_inst(f), delta_abl(f))[0]. Report distribution: mean, std, min, max, histogram. Run Hartigan's dip test on R_control values. Compare top-500 vs bottom-500: two-sample t-test (scipy.stats.ttest_ind), Kolmogorov-Smirnov test (scipy.stats.ks_2samp), Cohen's d = (mean_top - mean_bottom) / pooled_std. This is the decisive test: if mean_top ≈ mean_bottom ≈ 0.921, the finding is a global-similarity artifact; if mean_top >> mean_bottom, selective preservation is confirmed.\\n\\n**Step 3 — Post-topk density characterization for top-500.** For each harmful prompt i, find the TopK=50 threshold: threshold_i = the 50th largest value in z_pre_inst[i, :] (i.e., np.partition(z_pre_inst[i], -50)[-50]). For each top-500 feature f, compute density(f) = fraction of harmful prompts where z_pre_inst[i, f] >= threshold_i. Report mean, median, std, min of density across the 500 features. This honestly characterizes how 'SAE-feature-like' these features are in the post-topk sense — addresses the MAJOR critique about trivially-satisfied density filter.\\n\\n**Step 4 — Refusal direction computation.** Using instruct model activations: r_raw = mean(inst_act_L22[harmful_idx], axis=0) - mean(inst_act_L22[benign_idx], axis=0). Normalize: r = r_raw / norm(r_raw). Shape: (4096,). This is the refusal direction in residual-stream space at layer 22 — the same space as W_enc rows and W_dec columns.\\n\\n**Step 5 — Cosine similarity: r vs. SAE encoder directions.** For each top-500 RLHF-sensitive feature f: enc_dir = W_enc[f, :] / norm(W_enc[f, :]). cos_enc(f) = dot(r, enc_dir). Also compute dec_dir = W_dec[:, f] / norm(W_dec[:, f]); cos_dec(f) = dot(r, dec_dir). Report: (a) distribution of |cos_enc(f)| across all 500 features — mean, std, max; (b) individual values for top-10 RLHF-sensitive features by SC; (c) distribution of |cos_dec(f)|. Threshold claim: if mean |cos_enc| < 0.1, geometric orthogonality is directly confirmed. Also compute cos_sim between r and each of the bottom-500 control features' encoder directions — this provides a reference: if r is equally uncorrelated with both RLHF-sensitive and RLHF-insensitive feature directions, orthogonality is a property of the SAE basis in general, not specific to RLHF-sensitive features.\\n\\n**Step 6 — Control-subspace centroid distances.** Define three subspaces: (A) top-500 RLHF-sensitive features (existing result), (B) bottom-500 RLHF-insensitive features, (C) random 500 features (np.random.seed(42), sample from features not in top/bottom 500). For each subspace S and each prompt-type (harmful/benign): centroid_abl = mean(z_pre_abl[prompts, S], axis=0); centroid_inst = mean(z_pre_inst[prompts, S], axis=0); centroid_base = mean(z_pre_base[prompts, S], axis=0). d_abl_inst = cosine_distance(centroid_abl, centroid_inst); d_abl_base = cosine_distance(centroid_abl, centroid_base); ratio = d_abl_base / d_abl_inst. Report all ratios in a 3×2 table (subspace × prompt_type). If ratios in B and C are similar to A (6.1×, 17.6×), the centroid finding reflects global similarity not selective preservation.\\n\\n**Step 7 — Layer-20 full Reversal Score analysis.** Load layer20.sae.pt SAE. Compute z_pre at layer 20 for all three models and all prompts. SC_L20(f) for all 65536 features. Top-500 by SC_L20. Reversal Scores R_L20(f) for top-500. Report: mean, std, min, max, Hartigan dip test p-value, fraction with R > 0.7, count of negative R values. Compare distribution against layer-22 top-500. Also compute null-baseline (bottom-500) R at layer 20.\\n\\n**Output format (method_out.json):** {\\n  layer22_results: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, top500_dip_p, bottom500_R_mean, bottom500_R_std, bottom500_R_min, bottom500_R_max, bottom500_dip_p, ttest_t, ttest_p, ks_stat, ks_p, cohens_d },\\n  posttopk_density: { top500_density_mean, top500_density_median, top500_density_min, top500_density_std },\\n  orthogonality: { cos_enc_mean_abs, cos_enc_max_abs, cos_enc_std, cos_dec_mean_abs, cos_dec_max_abs, top10_cos_enc_values (list), bottom500_cos_enc_mean_abs, orthogonality_confirmed (bool, cos_enc_mean_abs < 0.1) },\\n  centroid_control: { subspaceA_harmful_ratio, subspaceA_benign_ratio, subspaceB_harmful_ratio, subspaceB_benign_ratio, subspaceC_harmful_ratio, subspaceC_benign_ratio },\\n  layer20_results: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, top500_dip_p, n_negative_R, bottom500_R_mean, bottom500_R_std }\\n}\",\n          \"depends_on\": [\n            {\n              \"id\": \"art_QBm3I0Z5VVJR\",\n              \"label\": \"cached activations\"\n            },\n            {\n              \"id\": \"art_M8-flaZ0OOaP\",\n              \"label\": \"prompt corpus\"\n            },\n            {\n              \"id\": \"art_pmAcY0CNazP9\",\n              \"label\": \"tech reference\"\n            }\n          ]\n        },\n        {\n          \"type\": \"evaluation\",\n          \"objective\": \"Synthesize all experimental results (iter2 + iter3 null-baseline experiment) into a clean, statistically rigorous summary that resolves each MAJOR reviewer critique with a definitive verdict — selective preservation vs. global similarity — and produces corrected statistics, updated table contents, and fixed paper-level descriptions for the GEN_PAPER_TEXT step.\",\n          \"approach\": \"Use UV with packages: scipy, numpy, pandas. Load method_out.json from the null-baseline experiment (depends_on) and full_method_out.json from art_QBm3I0Z5VVJR.\\n\\n**Section 1 — Null-baseline verdict.** Apply the following decision rule: if ttest_p < 0.05 AND cohens_d > 0.5 (large effect), verdict = 'SELECTIVE_PRESERVATION_CONFIRMED'; if ks_p < 0.05 but cohens_d <= 0.5, verdict = 'WEAK_SELECTIVE_PRESERVATION'; otherwise verdict = 'GLOBAL_SIMILARITY_ALTERNATIVE_PLAUSIBLE'. Report the actual top-500 vs. bottom-500 mean difference with 95% CI (bootstrap, 1000 samples). Produce a side-by-side histogram data table (top-500 R distribution vs. bottom-500 R distribution) with bin edges [0.0, 0.1, ..., 1.0].\\n\\n**Section 2 — Orthogonality verdict.** If cos_enc_mean_abs < 0.1: verdict = 'ORTHOGONALITY_DIRECTLY_CONFIRMED'. If 0.1 <= cos_enc_mean_abs < 0.2: verdict = 'NEAR_ORTHOGONAL'. If >= 0.2: verdict = 'ORTHOGONALITY_CLAIM_WEAKENED'. Report top-10 cos_enc values ranked by |cos_enc|. Note whether bottom-500 RLHF-insensitive features show similar or different |cos_enc| — if similar, orthogonality is a general property of the SAE basis rather than specific to RLHF-sensitive features.\\n\\n**Section 3 — Centroid verdict.** Compare ratios: if subspaceB_harmful_ratio and subspaceC_harmful_ratio are both within 50% of subspaceA_harmful_ratio (6.1×), verdict = 'CENTROID_REFLECTS_GLOBAL_SIMILARITY'; if A ratio is > 2× both B and C ratios, verdict = 'CENTROID_REFLECTS_SELECTIVE_PRESERVATION'. Report all 6 ratios in a formatted table.\\n\\n**Section 4 — Layer-20 consistency.** Report layer-20 Reversal Score distribution statistics alongside layer-22. Compute whether distributions are statistically different (KS test between layer-22 top-500 R values and layer-20 top-500 R values). State explicitly whether the zero-Type-R finding replicates at layer 20.\\n\\n**Section 5 — Post-topk density.** Report actual density statistics. Compute: what fraction of the top-500 features have density > 0.10 (the intended filter threshold)? This is the honest version of the density filter result — corrects the MAJOR critique.\\n\\n**Section 6 — Paper corrections checklist.** Produce a structured list of required paper changes:\\n  (a) Semantic labeling protocol: correct description to 'top-5 prompts by maximum pre-topk activation on last token, first 300 characters'\\n  (b) Density filter: replace '100% survived density filter' with actual post-topk density statistics\\n  (c) Classification accuracy: add paragraph reporting 0.476 accuracy as expected null result\\n  (d) Fisher fallback test: move Q3/Q1 comparison (n=4) to limitations, report only unconditional 75% harm-concept rate as primary semantic finding\\n  (e) Centroid analysis: add control-subspace rows to Figure 3 table\\n  (f) Pre-topk/post-topk distinction: add paragraph in limitations distinguishing SAE features (post-topk, validated) from SAE encoder projections (pre-topk, dense, interpretability not validated)\\n\\n**Output format (eval_out.json):** { null_baseline_verdict, selective_preservation_mean_diff, selective_preservation_95ci, orthogonality_verdict, centroid_verdict, layer20_replicates_zero_type_r (bool), posttopk_density_above_threshold_fraction, paper_corrections (list of strings), overall_claim_status ('STRENGTHENED' | 'UNCHANGED' | 'WEAKENED' | 'UNDERMINED'), summary_paragraph (string for paper Discussion section update) }\",\n          \"depends_on\": [\n            {\n              \"id\": \"art_QBm3I0Z5VVJR\",\n              \"label\": \"iter2 results\"\n            }\n          ]\n        }\n      ],\n      \"expected_outcome\": \"After iteration 3, the paper will have direct answers to all four MAJOR critiques: (1) The null-baseline comparison will definitively settle whether R≈0.921 reflects selective preservation (top-500 significantly higher than bottom-500) or global similarity (both near 1.0), with a t-test p-value, KS statistic, and Cohen's d; (2) Direct cosine-similarity measurements between the refusal direction r and top-500 RLHF-sensitive encoder directions will confirm or refute the geometric-orthogonality claim (expected: |cos| < 0.1); (3) Control-subspace centroid ratios (RLHF-insensitive, random) will contextualize the 6.1×/17.6× ratios and determine whether they are specific to the RLHF-sensitive subspace; (4) Layer-20 Reversal Score statistics will be reported numerically for the first time, either confirming zero Type-R features replicates or revealing that it is layer-22-specific. The evaluation artifact will produce a corrected paper-corrections checklist covering the semantic-labeling description fix, classification-accuracy disclosure, Fisher-test removal, density-filter correction, and pre-topk/post-topk limitation paragraph — giving GEN_PAPER_TEXT everything needed for a complete revision.\",\n      \"summary\": \"Iteration 3 resolves the credibility gap in the iter2 paper by computing the three missing null comparisons (bottom-500 Reversal Scores, direct refusal-direction orthogonality, control-subspace centroid distances) and substantiating the layer-20 sensitivity check with actual numbers. All computations reuse cached activations from iter2, keeping runtime under one hour. The evaluation synthesizes all results into definitive verdicts on selective preservation vs. global similarity, direct orthogonality confirmation, and centroid specificity — replacing inference with measurement. By the end of iteration 3, every MAJOR reviewer critique will have a numerical answer.\"\n    }\n  ]\n}\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [7] ASSISTANT · 2026-07-17 12:29:09 UTC

```
Strategy written. Two parallel artifacts address all four MAJOR reviewer critiques: (1) an experiment that computes null-baseline Reversal Scores for bottom-500 RLHF-insensitive features, direct refusal-direction cosine similarities, control-subspace centroid distances, post-topk density characterization, and layer-20 analysis — all reusing cached activations from iter2; (2) an evaluation that synthesizes results into verdicts (selective preservation vs. global similarity, orthogonality confirmed/refuted, centroid specificity) and produces a paper-corrections checklist for GEN_PAPER_TEXT.
```

### [8] SYSTEM-USER prompt · 2026-07-17 12:29:11 UTC

```
<verification_results>
Your previous response had issues that need fixing:

DEPENDENCY ERRORS (depends_on can ONLY reference IDs from <existing_artifacts>):
  - Strategy 1: Artifact 'experiment_iter3_dir1' (experiment): dependency 'art_QBm3I0Z5VVJR' has type 'experiment' which is not allowed (allowed: {'dataset', 'research'})

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

### [9] ASSISTANT · 2026-07-17 12:30:33 UTC

```
The experiment can't depend on another experiment (`art_QBm3I0Z5VVJR`). I'll fix this by folding the synthesis logic into the experiment itself (dropping the separate evaluation artifact that depended on the in-progress experiment) and keeping only valid dataset/research dependencies.
```

### [10] TOOL CALL — Write · 2026-07-17 12:30:33 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_3/gen_strat/gen_strat_1/.terminal_claude_agent_struct_out.json

{
  "strategies": [
    {
      "title": "Null Baseline + Orthogonality Validation",
      "objective": "Resolve all four MAJOR reviewer critiques by computing: (1) Reversal Scores for bottom-500 RLHF-insensitive features as a null baseline (decisive test: selective preservation vs. global similarity); (2) direct cosine similarity between the refusal direction r and RLHF-sensitive SAE encoder directions (direct orthogonality measurement); (3) centroid distances in control subspaces (RLHF-insensitive and random) to contextualize the 6.1×/17.6× ratios; and (4) full layer-20 Reversal Score analysis with reported numbers.",
      "rationale": "All four MAJOR critiques share one diagnosis: the current results are consistent with the null hypothesis that instruct and abliterated models are globally similar, making every encoder projection correlated regardless of RLHF sensitivity. The null-baseline comparison (bottom-500 by Safety_change) is the decisive test: if top-500 R values are significantly higher than bottom-500, selective preservation is confirmed; if both concentrate near R=1, the finding is a trivial artifact of global model similarity. The direct orthogonality measurement (cos_sim between r and W_enc rows) converts an inferred claim into a directly measured one. The control-subspace centroid analysis determines whether the 6.1×/17.6× ratios are specific to the RLHF-sensitive subspace. All these computations reuse activation extraction logic from scratch — the iter2 experiment ran in 4.3 minutes total — so the full suite fits comfortably in the 6-hour budget. The experiment also incorporates all paper-level corrections (semantic-labeling description fix, classification-accuracy disclosure, density-filter honest reporting, pre-topk/post-topk limitation) into its output JSON so GEN_PAPER_TEXT has everything needed.",
      "artifact_directions": [
        {
          "type": "experiment",
          "objective": "Run the complete null-baseline validation suite: (1) Reversal Scores for bottom-500 RLHF-insensitive features vs. top-500, with t-test/KS-test/Cohen's d; (2) refusal direction r computation and cosine similarity to top-500 SAE encoder directions; (3) centroid distances in RLHF-insensitive and random control subspaces; (4) post-topk density characterization for top-500 features; (5) full layer-20 Reversal Score analysis — producing definitive verdicts on selective preservation, geometric orthogonality, and centroid specificity.",
          "approach": "Use UV with packages: transformers>=4.51.0, torch, huggingface_hub, scipy, numpy, diptest. All three Qwen3-8B checkpoints (Base, Instruct/enable_thinking=False, huihui-ai/Huihui-Qwen3-8B-abliterated-v2) are loaded sequentially with torch.cuda.empty_cache() between loads. Prompt corpus from art_M8-flaZ0OOaP (full_data_out.json): use all harmful and benign prompts.\n\n**Step 0 — Activation extraction at layers 22 and 20.** For each model, register a forward hook on model.model.layers[N].output for N in [20, 22]. Run all prompts in batches (batch_size=4, last-token hidden state). Save: base_act_L22.npy, inst_act_L22.npy, abl_act_L22.npy, base_act_L20.npy, inst_act_L20.npy, abl_act_L20.npy — each shape (n_prompts, 4096). Track harmful_idx and benign_idx separately.\n\n**Step 1 — SAE load and pre-topk projections.** Download Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 via hf_hub_download for layer22.sae.pt and layer20.sae.pt. Load W_enc (65536×4096), b_enc (65536), W_dec (4096×65536) for each layer. Compute at layer 22: z_pre_base = base_act_L22 @ W_enc22.T + b_enc22 (shape: n_prompts×65536), similarly for inst and abl. Compute SC(f) = mean_i |z_pre_inst[harmful_idx, f] - z_pre_base[harmful_idx, f]| for all 65536 features. Sort: top500_idx = argsort(SC)[-500:]; bottom500_idx = argsort(SC)[:500].\n\n**Step 2 — Null-baseline Reversal Scores (bottom-500).** For top-500 and bottom-500 features, compute per-feature Pearson correlation: delta_inst(f) = z_pre_inst[harmful_idx, f] - z_pre_base[harmful_idx, f]; delta_abl(f) = z_pre_abl[harmful_idx, f] - z_pre_base[harmful_idx, f]; R(f) = pearsonr(delta_inst(f), delta_abl(f))[0]. Compute for all 500 top features and all 500 bottom features. Report: top500_R_mean/std/min/max, bottom500_R_mean/std/min/max. Run scipy.stats.ttest_ind(top500_R, bottom500_R), scipy.stats.ks_2samp(top500_R, bottom500_R). Cohen's d = (mean_top - mean_bottom) / sqrt((var_top + var_bottom) / 2). Run diptest on both distributions. VERDICT: if ttest_p < 0.05 AND cohens_d > 0.5 → 'SELECTIVE_PRESERVATION_CONFIRMED'; elif ks_p < 0.05 → 'WEAK_SELECTIVE_PRESERVATION'; else → 'GLOBAL_SIMILARITY_ALTERNATIVE_PLAUSIBLE'.\n\n**Step 3 — Post-topk density characterization.** For each harmful prompt i, find the k=50 threshold: kth_val_i = np.partition(z_pre_inst[i, :], -50)[-50]. For each top-500 feature f, density(f) = mean(z_pre_inst[harmful_idx, f] >= kth_vals[harmful_idx]) where kth_vals[i] = kth_val_i. Report density_mean, density_median, density_min, density_std, fraction_above_0.10 (features where >10% of prompts would have this feature in top-50). This is the honest replacement for the trivially-100%-satisfied density filter.\n\n**Step 4 — Refusal direction computation.** r_raw = mean(inst_act_L22[harmful_idx], axis=0) - mean(inst_act_L22[benign_idx], axis=0). r = r_raw / norm(r_raw). Shape (4096,). This is the refusal direction in residual-stream space.\n\n**Step 5 — Cosine similarity: r vs. SAE encoder directions.** For each top-500 feature f: enc_dir_f = W_enc22[f, :] / norm(W_enc22[f, :]); cos_enc(f) = dot(r, enc_dir_f). Also dec_dir_f = W_dec22[:, f] / norm(W_dec22[:, f]); cos_dec(f) = dot(r, dec_dir_f). Report: distribution of |cos_enc| — mean, std, max; top-10 RLHF-sensitive features' individual |cos_enc| values. For bottom-500 features: compute |cos_enc| distribution as reference. VERDICT: if mean(|cos_enc_top500|) < 0.1 → 'ORTHOGONALITY_CONFIRMED'; elif < 0.2 → 'NEAR_ORTHOGONAL'; else → 'ORTHOGONALITY_WEAKENED'. Note if bottom-500 shows similar |cos_enc| (SAE-basis-wide property vs. RLHF-specific).\n\n**Step 6 — Control-subspace centroid distances.** Define three subspaces: A=top500_idx, B=bottom500_idx, C=random500_idx (np.random.seed(42), exclude A and B). For each subspace S and prompt_type (harmful=harmful_idx, benign=benign_idx): centroid_abl_S = mean(z_pre_abl[prompt_type, :][:, S], axis=0); centroid_inst_S = mean(z_pre_inst[prompt_type, :][:, S], axis=0); centroid_base_S = mean(z_pre_base[prompt_type, :][:, S], axis=0). cosine_dist(u, v) = 1 - dot(u,v)/(norm(u)*norm(v)). d_abl_inst = cosine_dist(centroid_abl_S, centroid_inst_S); d_abl_base = cosine_dist(centroid_abl_S, centroid_base_S); ratio = d_abl_base / d_abl_inst. Report 6 ratios in table: subspace × prompt_type. VERDICT: if ratio_A_harmful > 2× both ratio_B_harmful and ratio_C_harmful → 'CENTROID_SELECTIVE'; else → 'CENTROID_GLOBAL'.\n\n**Step 7 — Layer-20 full analysis.** Load W_enc20, b_enc20 from layer20.sae.pt. Compute z_pre at layer 20 for all models. SC_L20(f) for all 65536 features on harmful prompts. top500_L20_idx = argsort(SC_L20)[-500:]; bottom500_L20_idx = argsort(SC_L20)[:500]. Compute Reversal Scores for top-500 and bottom-500 at layer 20. Report: top500_L20_R_mean/std/min/max, n_negative_R_L20, bottom500_L20_R_mean/std, diptest_p_L20. State: 'zero Type-R replicates at layer 20' (bool: n_negative_R_L20 == 0).\n\n**Output format (method_out.json):** { null_baseline: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, bottom500_R_mean, bottom500_R_std, bottom500_R_min, bottom500_R_max, ttest_t, ttest_p, ks_stat, ks_p, cohens_d, verdict }, posttopk_density: { density_mean, density_median, density_min, density_std, fraction_above_0p10 }, orthogonality: { cos_enc_top500_mean_abs, cos_enc_top500_max_abs, cos_enc_top500_std, cos_dec_top500_mean_abs, cos_enc_bottom500_mean_abs, top10_feature_cos_enc (list of {feature_idx, sc, cos_enc, cos_dec}), verdict }, centroid_control: { subspaceA_harmful_ratio, subspaceA_benign_ratio, subspaceB_harmful_ratio, subspaceB_benign_ratio, subspaceC_harmful_ratio, subspaceC_benign_ratio, verdict }, layer20: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, n_negative_R, bottom500_R_mean, bottom500_R_std, diptest_p, zero_type_r_replicates }, paper_corrections: { semantic_labeling_correct: 'top-5 prompts by max pre-topk activation on last token, first 300 chars — not token window', density_filter_correct: 'pre-topk dense projections always nonzero; post-topk density fraction is honest measure', classification_accuracy: '0.476 near-chance (expected: zero Type-R means no discrimination signal)', fisher_test_action: 'move Q3/Q1 n=4 test to limitations; report only unconditional 75% harm-concept rate', pretopk_limitation: 'semantic claims rest on pre-topk encoder projections, not validated sparse post-topk features' } }",
          "depends_on": [
            {
              "id": "art_M8-flaZ0OOaP",
              "label": "prompt corpus"
            },
            {
              "id": "art_pmAcY0CNazP9",
              "label": "tech reference"
            }
          ]
        }
      ],
      "expected_outcome": "After iteration 3, the paper will have definitive numerical answers to all four MAJOR critiques: (1) t-test p-value and Cohen's d comparing top-500 vs bottom-500 Reversal Score distributions — either confirming selective preservation (top >> bottom) or revealing global similarity (both near R=1); (2) mean |cos_sim| between refusal direction r and RLHF-sensitive encoder directions — either directly confirming geometric orthogonality (|cos| < 0.1) or showing the claim needs qualification; (3) centroid ratios in RLHF-insensitive and random subspaces alongside the RLHF-sensitive 6.1×/17.6×, determining whether those ratios are subspace-specific or universal; (4) layer-20 Reversal Score statistics explicitly reported for the first time, either replicating the zero-Type-R finding or revealing layer-specificity. The output JSON also contains the corrected paper descriptions for semantic labeling, density filter, classification accuracy, and pre-topk/post-topk distinction — giving GEN_PAPER_TEXT a complete revision target.",
      "summary": "One focused experiment addresses all four MAJOR reviewer critiques by computing the null comparisons and direct measurements that were missing from iter2. The null-baseline (bottom-500 features) settles the selective-preservation vs. global-similarity question. The cosine-similarity measurement converts the geometric-orthogonality claim from inference to direct evidence. Control-subspace centroid distances contextualize the 6.1×/17.6× ratios. Layer-20 numbers substantiate the sensitivity check. All computations run from scratch in under one hour on one GPU."
    }
  ]
}
```

### [11] TOOL RESULT — Write · 2026-07-17 12:30:33 UTC

```
{"type": "update", "filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_3/gen_strat/gen_strat_1/.terminal_claude_agent_struct_out.json", "content": "{\n  \"strategies\": [\n    {\n      \"title\": \"Null Baseline + Orthogonality Validation\",\n      \"objective\": \"Resolve all four MAJOR reviewer critiques by computing: (1) Reversal Scores for bottom-500 RLHF-insensitive features as a null baseline (decisive test: selective preservation vs. global similarity); (2) direct cosine similarity between the refusal direction r and RLHF-sensitive SAE encoder directions (direct orthogonality measurement); (3) centroid distances in control subspaces (RLHF-insensitive and random) to contextualize the 6.1×/17.6× ratios; and (4) full layer-20 Reversal Score analysis with reported numbers.\",\n      \"rationale\": \"All four MAJOR critiques share one diagnosis: the current results are consistent with the null hypothesis that instruct and abliterated models are globally similar, making every encoder projection correlated regardless of RLHF sensitivity. The null-baseline comparison (bottom-500 by Safety_change) is the decisive test: if top-500 R values are significantly higher than bottom-500, selective preservation is confirmed; if both concentrate near R=1, the finding is a trivial artifact of global model similarity. The direct orthogonality measurement (cos_sim between r and W_enc rows) converts an inferred claim into a directly measured one. The control-subspace centroid analysis determines whether the 6.1×/17.6× ratios are specific to the RLHF-sensitive subspace. All these computations reuse activation extraction logic from scratch — the iter2 experiment ran in 4.3 minutes total — so the full suite fits comfortably in the 6-hour budget. The experiment also incorporates all paper-level corrections (semantic-labeling description fix, classification-accuracy disclosure, density-filter honest reporting, pre-topk/post-topk limitation) into its output JSON so GEN_PAPER_TEXT has everything needed.\",\n      \"artifact_directions\": [\n        {\n          \"type\": \"experiment\",\n          \"objective\": \"Run the complete null-baseline validation suite: (1) Reversal Scores for bottom-500 RLHF-insensitive features vs. top-500, with t-test/KS-test/Cohen's d; (2) refusal direction r computation and cosine similarity to top-500 SAE encoder directions; (3) centroid distances in RLHF-insensitive and random control subspaces; (4) post-topk density characterization for top-500 features; (5) full layer-20 Reversal Score analysis — producing definitive verdicts on selective preservation, geometric orthogonality, and centroid specificity.\",\n          \"approach\": \"Use UV with packages: transformers>=4.51.0, torch, huggingface_hub, scipy, numpy, diptest. All three Qwen3-8B checkpoints (Base, Instruct/enable_thinking=False, huihui-ai/Huihui-Qwen3-8B-abliterated-v2) are loaded sequentially with torch.cuda.empty_cache() between loads. Prompt corpus from art_M8-flaZ0OOaP (full_data_out.json): use all harmful and benign prompts.\\n\\n**Step 0 — Activation extraction at layers 22 and 20.** For each model, register a forward hook on model.model.layers[N].output for N in [20, 22]. Run all prompts in batches (batch_size=4, last-token hidden state). Save: base_act_L22.npy, inst_act_L22.npy, abl_act_L22.npy, base_act_L20.npy, inst_act_L20.npy, abl_act_L20.npy — each shape (n_prompts, 4096). Track harmful_idx and benign_idx separately.\\n\\n**Step 1 — SAE load and pre-topk projections.** Download Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 via hf_hub_download for layer22.sae.pt and layer20.sae.pt. Load W_enc (65536×4096), b_enc (65536), W_dec (4096×65536) for each layer. Compute at layer 22: z_pre_base = base_act_L22 @ W_enc22.T + b_enc22 (shape: n_prompts×65536), similarly for inst and abl. Compute SC(f) = mean_i |z_pre_inst[harmful_idx, f] - z_pre_base[harmful_idx, f]| for all 65536 features. Sort: top500_idx = argsort(SC)[-500:]; bottom500_idx = argsort(SC)[:500].\\n\\n**Step 2 — Null-baseline Reversal Scores (bottom-500).** For top-500 and bottom-500 features, compute per-feature Pearson correlation: delta_inst(f) = z_pre_inst[harmful_idx, f] - z_pre_base[harmful_idx, f]; delta_abl(f) = z_pre_abl[harmful_idx, f] - z_pre_base[harmful_idx, f]; R(f) = pearsonr(delta_inst(f), delta_abl(f))[0]. Compute for all 500 top features and all 500 bottom features. Report: top500_R_mean/std/min/max, bottom500_R_mean/std/min/max. Run scipy.stats.ttest_ind(top500_R, bottom500_R), scipy.stats.ks_2samp(top500_R, bottom500_R). Cohen's d = (mean_top - mean_bottom) / sqrt((var_top + var_bottom) / 2). Run diptest on both distributions. VERDICT: if ttest_p < 0.05 AND cohens_d > 0.5 → 'SELECTIVE_PRESERVATION_CONFIRMED'; elif ks_p < 0.05 → 'WEAK_SELECTIVE_PRESERVATION'; else → 'GLOBAL_SIMILARITY_ALTERNATIVE_PLAUSIBLE'.\\n\\n**Step 3 — Post-topk density characterization.** For each harmful prompt i, find the k=50 threshold: kth_val_i = np.partition(z_pre_inst[i, :], -50)[-50]. For each top-500 feature f, density(f) = mean(z_pre_inst[harmful_idx, f] >= kth_vals[harmful_idx]) where kth_vals[i] = kth_val_i. Report density_mean, density_median, density_min, density_std, fraction_above_0.10 (features where >10% of prompts would have this feature in top-50). This is the honest replacement for the trivially-100%-satisfied density filter.\\n\\n**Step 4 — Refusal direction computation.** r_raw = mean(inst_act_L22[harmful_idx], axis=0) - mean(inst_act_L22[benign_idx], axis=0). r = r_raw / norm(r_raw). Shape (4096,). This is the refusal direction in residual-stream space.\\n\\n**Step 5 — Cosine similarity: r vs. SAE encoder directions.** For each top-500 feature f: enc_dir_f = W_enc22[f, :] / norm(W_enc22[f, :]); cos_enc(f) = dot(r, enc_dir_f). Also dec_dir_f = W_dec22[:, f] / norm(W_dec22[:, f]); cos_dec(f) = dot(r, dec_dir_f). Report: distribution of |cos_enc| — mean, std, max; top-10 RLHF-sensitive features' individual |cos_enc| values. For bottom-500 features: compute |cos_enc| distribution as reference. VERDICT: if mean(|cos_enc_top500|) < 0.1 → 'ORTHOGONALITY_CONFIRMED'; elif < 0.2 → 'NEAR_ORTHOGONAL'; else → 'ORTHOGONALITY_WEAKENED'. Note if bottom-500 shows similar |cos_enc| (SAE-basis-wide property vs. RLHF-specific).\\n\\n**Step 6 — Control-subspace centroid distances.** Define three subspaces: A=top500_idx, B=bottom500_idx, C=random500_idx (np.random.seed(42), exclude A and B). For each subspace S and prompt_type (harmful=harmful_idx, benign=benign_idx): centroid_abl_S = mean(z_pre_abl[prompt_type, :][:, S], axis=0); centroid_inst_S = mean(z_pre_inst[prompt_type, :][:, S], axis=0); centroid_base_S = mean(z_pre_base[prompt_type, :][:, S], axis=0). cosine_dist(u, v) = 1 - dot(u,v)/(norm(u)*norm(v)). d_abl_inst = cosine_dist(centroid_abl_S, centroid_inst_S); d_abl_base = cosine_dist(centroid_abl_S, centroid_base_S); ratio = d_abl_base / d_abl_inst. Report 6 ratios in table: subspace × prompt_type. VERDICT: if ratio_A_harmful > 2× both ratio_B_harmful and ratio_C_harmful → 'CENTROID_SELECTIVE'; else → 'CENTROID_GLOBAL'.\\n\\n**Step 7 — Layer-20 full analysis.** Load W_enc20, b_enc20 from layer20.sae.pt. Compute z_pre at layer 20 for all models. SC_L20(f) for all 65536 features on harmful prompts. top500_L20_idx = argsort(SC_L20)[-500:]; bottom500_L20_idx = argsort(SC_L20)[:500]. Compute Reversal Scores for top-500 and bottom-500 at layer 20. Report: top500_L20_R_mean/std/min/max, n_negative_R_L20, bottom500_L20_R_mean/std, diptest_p_L20. State: 'zero Type-R replicates at layer 20' (bool: n_negative_R_L20 == 0).\\n\\n**Output format (method_out.json):** { null_baseline: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, bottom500_R_mean, bottom500_R_std, bottom500_R_min, bottom500_R_max, ttest_t, ttest_p, ks_stat, ks_p, cohens_d, verdict }, posttopk_density: { density_mean, density_median, density_min, density_std, fraction_above_0p10 }, orthogonality: { cos_enc_top500_mean_abs, cos_enc_top500_max_abs, cos_enc_top500_std, cos_dec_top500_mean_abs, cos_enc_bottom500_mean_abs, top10_feature_cos_enc (list of {feature_idx, sc, cos_enc, cos_dec}), verdict }, centroid_control: { subspaceA_harmful_ratio, subspaceA_benign_ratio, subspaceB_harmful_ratio, subspaceB_benign_ratio, subspaceC_harmful_ratio, subspaceC_benign_ratio, verdict }, layer20: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, n_negative_R, bottom500_R_mean, bottom500_R_std, diptest_p, zero_type_r_replicates }, paper_corrections: { semantic_labeling_correct: 'top-5 prompts by max pre-topk activation on last token, first 300 chars — not token window', density_filter_correct: 'pre-topk dense projections always nonzero; post-topk density fraction is honest measure', classification_accuracy: '0.476 near-chance (expected: zero Type-R means no discrimination signal)', fisher_test_action: 'move Q3/Q1 n=4 test to limitations; report only unconditional 75% harm-concept rate', pretopk_limitation: 'semantic claims rest on pre-topk encoder projections, not validated sparse post-topk features' } }\",\n          \"depends_on\": [\n            {\n              \"id\": \"art_M8-flaZ0OOaP\",\n              \"label\": \"prompt corpus\"\n            },\n            {\n              \"id\": \"art_pmAcY0CNazP9\",\n              \"label\": \"tech reference\"\n            }\n          ]\n        }\n      ],\n      \"expected_outcome\": \"After iteration 3, the paper will have definitive numerical answers to all four MAJOR critiques: (1) t-test p-value and Cohen's d comparing top-500 vs bottom-500 Reversal Score distributions — either confirming selective preservation (top >> bottom) or revealing global similarity (both near R=1); (2) mean |cos_sim| between refusal direction r and RLHF-sensitive encoder directions — either directly confirming geometric orthogonality (|cos| < 0.1) or showing the claim needs qualification; (3) centroid ratios in RLHF-insensitive and random subspaces alongside the RLHF-sensitive 6.1×/17.6×, determining whether those ratios are subspace-specific or universal; (4) layer-20 Reversal Score statistics explicitly reported for the first time, either replicating the zero-Type-R finding or revealing layer-specificity. The output JSON also contains the corrected paper descriptions for semantic labeling, density filter, classification accuracy, and pre-topk/post-topk distinction — giving GEN_PAPER_TEXT a complete revision target.\",\n      \"summary\": \"One focused experiment addresses all four MAJOR reviewer critiques by computing the null comparisons and direct measurements that were missing from iter2. The null-baseline (bottom-500 features) settles the selective-preservation vs. global-similarity question. The cosine-similarity measurement converts the geometric-orthogonality claim from inference to direct evidence. Control-subspace centroid distances contextualize the 6.1×/17.6× ratios. Layer-20 numbers substantiate the sensitivity check. All computations run from scratch in under one hour on one GPU.\"\n    }\n  ]\n}\n", "structuredPatch": [{"oldStart": 2, "oldLines": 19, "newStart": 2, "newLines": 15, "lines": ["   \"strategies\": [", "     {", "       \"title\": \"Null Baseline + Orthogonality Validation\",", "-      \"objective\": \"Resolve the four MAJOR reviewer critiques by computing: (1) Reversal Scores for bottom-500 RLHF-insensitive features as a null baseline to determine whether R≈0.921 reflects selective preservation or global instruct/abliterated similarity; (2) direct cosine similarity between the refusal direction r and RLHF-sensitive SAE encoder directions to confirm or refute the geometric-orthogonality claim; (3) centroid distances in control subspaces (RLHF-insensitive and random) alongside the RLHF-sensitive subspace; and (4) layer-20 Reversal Score distribution to substantiate the claimed sensitivity-check consistency.\",", "-      \"rationale\": \"All four MAJOR critiques share a single diagnosis: the current results are consistent with the null hypothesis that instruct and abliterated models are globally similar (abliteration touches only a few of 4096 dimensions), making every encoder projection correlated. The null-baseline comparison (bottom-500 features) is the decisive test: if RLHF-sensitive features have significantly higher R than RLHF-insensitive features, selective preservation is confirmed; if both concentrate near R=1, the finding collapses to a trivial global-similarity artifact. The orthogonality measurement (cos_sim between r and encoder directions) directly validates the geometric interpretation instead of inferring it from R. The control-subspace centroid analysis answers whether the 6.1×/17.6× ratios are specific to the RLHF-sensitive subspace or universal. Layer-20 numbers are computationally free given that activations were already extracted. All computations are linear-algebra operations on pre-extracted numpy arrays — the iter2 experiment ran in 4.3 minutes total, and these additions add minimal overhead. Together they convert inference into direct measurement, addressing every MAJOR critique without requiring model re-loading if activation caches are available.\",", "+      \"objective\": \"Resolve all four MAJOR reviewer critiques by computing: (1) Reversal Scores for bottom-500 RLHF-insensitive features as a null baseline (decisive test: selective preservation vs. global similarity); (2) direct cosine similarity between the refusal direction r and RLHF-sensitive SAE encoder directions (direct orthogonality measurement); (3) centroid distances in control subspaces (RLHF-insensitive and random) to contextualize the 6.1×/17.6× ratios; and (4) full layer-20 Reversal Score analysis with reported numbers.\",", "+      \"rationale\": \"All four MAJOR critiques share one diagnosis: the current results are consistent with the null hypothesis that instruct and abliterated models are globally similar, making every encoder projection correlated regardless of RLHF sensitivity. The null-baseline comparison (bottom-500 by Safety_change) is the decisive test: if top-500 R values are significantly higher than bottom-500, selective preservation is confirmed; if both concentrate near R=1, the finding is a trivial artifact of global model similarity. The direct orthogonality measurement (cos_sim between r and W_enc rows) converts an inferred claim into a directly measured one. The control-subspace centroid analysis determines whether the 6.1×/17.6× ratios are specific to the RLHF-sensitive subspace. All these computations reuse activation extraction logic from scratch — the iter2 experiment ran in 4.3 minutes total — so the full suite fits comfortably in the 6-hour budget. The experiment also incorporates all paper-level corrections (semantic-labeling description fix, classification-accuracy disclosure, density-filter honest reporting, pre-topk/post-topk limitation) into its output JSON so GEN_PAPER_TEXT has everything needed.\",", "       \"artifact_directions\": [", "         {", "           \"type\": \"experiment\",", "-          \"objective\": \"Compute null-baseline Reversal Scores (bottom-500 RLHF-insensitive features), refusal-direction cosine similarities (r vs. top-500 SAE encoder directions), control-subspace centroid distances (RLHF-insensitive + random subspaces), post-topk density characterization for top-500 features, and full layer-20 Reversal Score analysis — all addressing MAJOR reviewer critiques about the absence of null comparisons and direct orthogonality evidence.\",", "-          \"approach\": \"Use UV with packages: transformers>=4.51.0, torch, huggingface_hub, scipy, numpy, diptest. Load from the iter2 experiment workspace (art_QBm3I0Z5VVJR) to reuse any saved activation arrays. If activation files (base/instruct/abliterated residuals at layers 20 and 22) are not found, re-extract them by loading each model sequentially (bfloat16, device_map='auto') and hooking model.model.layers[N].output.\\n\\n**Step 0 — Load dependencies.** Check art_QBm3I0Z5VVJR workspace for saved numpy arrays: base_act_L22.npy, inst_act_L22.npy, abl_act_L22.npy (shape: n_prompts × 4096), and corresponding L20 arrays. Also load the prompt corpus from art_M8-flaZ0OOaP (harmful and benign prompts). If activation arrays are absent, reload all three models and re-extract — sequential load with torch.cuda.empty_cache() between models.\\n\\n**Step 1 — Recompute baseline SC and load SAE.** Load Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 via hf_hub_download (layer22.sae.pt). Reconstruct W_enc (65536×4096), b_enc (65536), W_dec (4096×65536). Compute z_pre_base = base_act_L22 @ W_enc.T + b_enc, z_pre_inst = inst_act_L22 @ W_enc.T + b_enc, z_pre_abl = abl_act_L22 @ W_enc.T + b_enc. SC(f) = mean_i |z_pre_inst[harmful, f] - z_pre_base[harmful, f]| for all 65536 features. Sort by SC descending: top-500 (RLHF-sensitive) and bottom-500 (RLHF-insensitive) indices.\\n\\n**Step 2 — Null-baseline Reversal Scores (bottom-500).** For each bottom-500 feature f, compute: delta_inst(f) = z_pre_inst[harmful_idx, f] - z_pre_base[harmful_idx, f]; delta_abl(f) = z_pre_abl[harmful_idx, f] - z_pre_base[harmful_idx, f]; R_control(f) = pearsonr(delta_inst(f), delta_abl(f))[0]. Report distribution: mean, std, min, max, histogram. Run Hartigan's dip test on R_control values. Compare top-500 vs bottom-500: two-sample t-test (scipy.stats.ttest_ind), Kolmogorov-Smirnov test (scipy.stats.ks_2samp), Cohen's d = (mean_top - mean_bottom) / pooled_std. This is the decisive test: if mean_top ≈ mean_bottom ≈ 0.921, the finding is a global-similarity artifact; if mean_top >> mean_bottom, selective preservation is confirmed.\\n\\n**Step 3 — Post-topk density characterization for top-500.** For each harmful prompt i, find the TopK=50 threshold: threshold_i = the 50th largest value in z_pre_inst[i, :] (i.e., np.partition(z_pre_inst[i], -50)[-50]). For each top-500 feature f, compute density(f) = fraction of harmful prompts where z_pre_inst[i, f] >= threshold_i. Report mean, median, std, min of density across the 500 features. This honestly characterizes how 'SAE-feature-like' these features are in the post-topk sense — addresses the MAJOR critique about trivially-satisfied density filter.\\n\\n**Step 4 — Refusal direction computation.** Using instruct model activations: r_raw = mean(inst_act_L22[harmful_idx], axis=0) - mean(inst_act_L22[benign_idx], axis=0). Normalize: r = r_raw / norm(r_raw). Shape: (4096,). This is the refusal direction in residual-stream space at layer 22 — the same space as W_enc rows and W_dec columns.\\n\\n**Step 5 — Cosine similarity: r vs. SAE encoder directions.** For each top-500 RLHF-sensitive feature f: enc_dir = W_enc[f, :] / norm(W_enc[f, :]). cos_enc(f) = dot(r, enc_dir). Also compute dec_dir = W_dec[:, f] / norm(W_dec[:, f]); cos_dec(f) = dot(r, dec_dir). Report: (a) distribution of |cos_enc(f)| across all 500 features — mean, std, max; (b) individual values for top-10 RLHF-sensitive features by SC; (c) distribution of |cos_dec(f)|. Threshold claim: if mean |cos_enc| < 0.1, geometric orthogonality is directly confirmed. Also compute cos_sim between r and each of the bottom-500 control features' encoder directions — this provides a reference: if r is equally uncorrelated with both RLHF-sensitive and RLHF-insensitive feature directions, orthogonality is a property of the SAE basis in general, not specific to RLHF-sensitive features.\\n\\n**Step 6 — Control-subspace centroid distances.** Define three subspaces: (A) top-500 RLHF-sensitive features (existing result), (B) bottom-500 RLHF-insensitive features, (C) random 500 features (np.random.seed(42), sample from features not in top/bottom 500). For each subspace S and each prompt-type (harmful/benign): centroid_abl = mean(z_pre_abl[prompts, S], axis=0); centroid_inst = mean(z_pre_inst[prompts, S], axis=0); centroid_base = mean(z_pre_base[prompts, S], axis=0). d_abl_inst = cosine_distance(centroid_abl, centroid_inst); d_abl_base = cosine_distance(centroid_abl, centroid_base); ratio = d_abl_base / d_abl_inst. Report all ratios in a 3×2 table (subspace × prompt_type). If ratios in B and C are similar to A (6.1×, 17.6×), the centroid finding reflects global similarity not selective preservation.\\n\\n**Step 7 — Layer-20 full Reversal Score analysis.** Load layer20.sae.pt SAE. Compute z_pre at layer 20 for all three models and all prompts. SC_L20(f) for all 65536 features. Top-500 by SC_L20. Reversal Scores R_L20(f) for top-500. Report: mean, std, min, max, Hartigan dip test p-value, fraction with R > 0.7, count of negative R values. Compare distribution against layer-22 top-500. Also compute null-baseline (bottom-500) R at layer 20.\\n\\n**Output format (method_out.json):** {\\n  layer22_results: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, top500_dip_p, bottom500_R_mean, bottom500_R_std, bottom500_R_min, bottom500_R_max, bottom500_dip_p, ttest_t, ttest_p, ks_stat, ks_p, cohens_d },\\n  posttopk_density: { top500_density_mean, top500_density_median, top500_density_min, top500_density_std },\\n  orthogonality: { cos_enc_mean_abs, cos_enc_max_abs, cos_enc_std, cos_dec_mean_abs, cos_dec_max_abs, top10_cos_enc_values (list), bottom500_cos_enc_mean_abs, orthogonality_confirmed (bool, cos_enc_mean_abs < 0.1) },\\n  centroid_control: { subspaceA_harmful_ratio, subspaceA_benign_ratio, subspaceB_harmful_ratio, subspaceB_benign_ratio, subspaceC_harmful_ratio, subspaceC_benign_ratio },\\n  layer20_results: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, top500_dip_p, n_negative_R, bottom500_R_mean, bottom500_R_std }\\n}\",", "+          \"objective\": \"Run the complete null-baseline validation suite: (1) Reversal Scores for bottom-500 RLHF-insensitive features vs. top-500, with t-test/KS-test/Cohen's d; (2) refusal direction r computation and cosine similarity to top-500 SAE encoder directions; (3) centroid distances in RLHF-insensitive and random control subspaces; (4) post-topk density characterization for top-500 features; (5) full layer-20 Reversal Score analysis — producing definitive verdicts on selective preservation, geometric orthogonality, and centroid specificity.\",", "+          \"approach\": \"Use UV with packages: transformers>=4.51.0, torch, huggingface_hub, scipy, numpy, diptest. All three Qwen3-8B checkpoints (Base, Instruct/enable_thinking=False, huihui-ai/Huihui-Qwen3-8B-abliterated-v2) are loaded sequentially with torch.cuda.empty_cache() between loads. Prompt corpus from art_M8-flaZ0OOaP (full_data_out.json): use all harmful and benign prompts.\\n\\n**Step 0 — Activation extraction at layers 22 and 20.** For each model, register a forward hook on model.model.layers[N].output for N in [20, 22]. Run all prompts in batches (batch_size=4, last-token hidden state). Save: base_act_L22.npy, inst_act_L22.npy, abl_act_L22.npy, base_act_L20.npy, inst_act_L20.npy, abl_act_L20.npy — each shape (n_prompts, 4096). Track harmful_idx and benign_idx separately.\\n\\n**Step 1 — SAE load and pre-topk projections.** Download Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 via hf_hub_download for layer22.sae.pt and layer20.sae.pt. Load W_enc (65536×4096), b_enc (65536), W_dec (4096×65536) for each layer. Compute at layer 22: z_pre_base = base_act_L22 @ W_enc22.T + b_enc22 (shape: n_prompts×65536), similarly for inst and abl. Compute SC(f) = mean_i |z_pre_inst[harmful_idx, f] - z_pre_base[harmful_idx, f]| for all 65536 features. Sort: top500_idx = argsort(SC)[-500:]; bottom500_idx = argsort(SC)[:500].\\n\\n**Step 2 — Null-baseline Reversal Scores (bottom-500).** For top-500 and bottom-500 features, compute per-feature Pearson correlation: delta_inst(f) = z_pre_inst[harmful_idx, f] - z_pre_base[harmful_idx, f]; delta_abl(f) = z_pre_abl[harmful_idx, f] - z_pre_base[harmful_idx, f]; R(f) = pearsonr(delta_inst(f), delta_abl(f))[0]. Compute for all 500 top features and all 500 bottom features. Report: top500_R_mean/std/min/max, bottom500_R_mean/std/min/max. Run scipy.stats.ttest_ind(top500_R, bottom500_R), scipy.stats.ks_2samp(top500_R, bottom500_R). Cohen's d = (mean_top - mean_bottom) / sqrt((var_top + var_bottom) / 2). Run diptest on both distributions. VERDICT: if ttest_p < 0.05 AND cohens_d > 0.5 → 'SELECTIVE_PRESERVATION_CONFIRMED'; elif ks_p < 0.05 → 'WEAK_SELECTIVE_PRESERVATION'; else → 'GLOBAL_SIMILARITY_ALTERNATIVE_PLAUSIBLE'.\\n\\n**Step 3 — Post-topk density characterization.** For each harmful prompt i, find the k=50 threshold: kth_val_i = np.partition(z_pre_inst[i, :], -50)[-50]. For each top-500 feature f, density(f) = mean(z_pre_inst[harmful_idx, f] >= kth_vals[harmful_idx]) where kth_vals[i] = kth_val_i. Report density_mean, density_median, density_min, density_std, fraction_above_0.10 (features where >10% of prompts would have this feature in top-50). This is the honest replacement for the trivially-100%-satisfied density filter.\\n\\n**Step 4 — Refusal direction computation.** r_raw = mean(inst_act_L22[harmful_idx], axis=0) - mean(inst_act_L22[benign_idx], axis=0). r = r_raw / norm(r_raw). Shape (4096,). This is the refusal direction in residual-stream space.\\n\\n**Step 5 — Cosine similarity: r vs. SAE encoder directions.** For each top-500 feature f: enc_dir_f = W_enc22[f, :] / norm(W_enc22[f, :]); cos_enc(f) = dot(r, enc_dir_f). Also dec_dir_f = W_dec22[:, f] / norm(W_dec22[:, f]); cos_dec(f) = dot(r, dec_dir_f). Report: distribution of |cos_enc| — mean, std, max; top-10 RLHF-sensitive features' individual |cos_enc| values. For bottom-500 features: compute |cos_enc| distribution as reference. VERDICT: if mean(|cos_enc_top500|) < 0.1 → 'ORTHOGONALITY_CONFIRMED'; elif < 0.2 → 'NEAR_ORTHOGONAL'; else → 'ORTHOGONALITY_WEAKENED'. Note if bottom-500 shows similar |cos_enc| (SAE-basis-wide property vs. RLHF-specific).\\n\\n**Step 6 — Control-subspace centroid distances.** Define three subspaces: A=top500_idx, B=bottom500_idx, C=random500_idx (np.random.seed(42), exclude A and B). For each subspace S and prompt_type (harmful=harmful_idx, benign=benign_idx): centroid_abl_S = mean(z_pre_abl[prompt_type, :][:, S], axis=0); centroid_inst_S = mean(z_pre_inst[prompt_type, :][:, S], axis=0); centroid_base_S = mean(z_pre_base[prompt_type, :][:, S], axis=0). cosine_dist(u, v) = 1 - dot(u,v)/(norm(u)*norm(v)). d_abl_inst = cosine_dist(centroid_abl_S, centroid_inst_S); d_abl_base = cosine_dist(centroid_abl_S, centroid_base_S); ratio = d_abl_base / d_abl_inst. Report 6 ratios in table: subspace × prompt_type. VERDICT: if ratio_A_harmful > 2× both ratio_B_harmful and ratio_C_harmful → 'CENTROID_SELECTIVE'; else → 'CENTROID_GLOBAL'.\\n\\n**Step 7 — Layer-20 full analysis.** Load W_enc20, b_enc20 from layer20.sae.pt. Compute z_pre at layer 20 for all models. SC_L20(f) for all 65536 features on harmful prompts. top500_L20_idx = argsort(SC_L20)[-500:]; bottom500_L20_idx = argsort(SC_L20)[:500]. Compute Reversal Scores for top-500 and bottom-500 at layer 20. Report: top500_L20_R_mean/std/min/max, n_negative_R_L20, bottom500_L20_R_mean/std, diptest_p_L20. State: 'zero Type-R replicates at layer 20' (bool: n_negative_R_L20 == 0).\\n\\n**Output format (method_out.json):** { null_baseline: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, bottom500_R_mean, bottom500_R_std, bottom500_R_min, bottom500_R_max, ttest_t, ttest_p, ks_stat, ks_p, cohens_d, verdict }, posttopk_density: { density_mean, density_median, density_min, density_std, fraction_above_0p10 }, orthogonality: { cos_enc_top500_mean_abs, cos_enc_top500_max_abs, cos_enc_top500_std, cos_dec_top500_mean_abs, cos_enc_bottom500_mean_abs, top10_feature_cos_enc (list of {feature_idx, sc, cos_enc, cos_dec}), verdict }, centroid_control: { subspaceA_harmful_ratio, subspaceA_benign_ratio, subspaceB_harmful_ratio, subspaceB_benign_ratio, subspaceC_harmful_ratio, subspaceC_benign_ratio, verdict }, layer20: { top500_R_mean, top500_R_std, top500_R_min, top500_R_max, n_negative_R, bottom500_R_mean, bottom500_R_std, diptest_p, zero_type_r_replicates }, paper_corrections: { semantic_labeling_correct: 'top-5 prompts by max pre-topk activation on last token, first 300 chars — not token window', density_filter_correct: 'pre-topk dense projections always nonzero; post-topk density fraction is honest measure', classification_accuracy: '0.476 near-chance (expected: zero Type-R means no discrimination signal)', fisher_test_action: 'move Q3/Q1 n=4 test to limitations; report only unconditional 75% harm-concept rate', pretopk_limitation: 'semantic claims rest on pre-topk encoder projections, not validated sparse post-topk features' } }\",", "           \"depends_on\": [", "             {", "-              \"id\": \"art_QBm3I0Z5VVJR\",", "-              \"label\": \"cached activations\"", "-            },", "-            {", "               \"id\": \"art_M8-flaZ0OOaP\",", "               \"label\": \"prompt corpus\"", "             },"]}, {"oldStart": 23, "oldLines": 21, "newStart": 19, "newLines": 10, "lines": ["               \"label\": \"tech reference\"", "             }", "           ]", "-        },", "-        {", "-          \"type\": \"evaluation\",", "-          \"objective\": \"Synthesize all experimental results (iter2 + iter3 null-baseline experiment) into a clean, statistically rigorous summary that resolves each MAJOR reviewer critique with a definitive verdict — selective preservation vs. global similarity — and produces corrected statistics, updated table contents, and fixed paper-level descriptions for the GEN_PAPER_TEXT step.\",", "-          \"approach\": \"Use UV with packages: scipy, numpy, pandas. Load method_out.json from the null-baseline experiment (depends_on) and full_method_out.json from art_QBm3I0Z5VVJR.\\n\\n**Section 1 — Null-baseline verdict.** Apply the following decision rule: if ttest_p < 0.05 AND cohens_d > 0.5 (large effect), verdict = 'SELECTIVE_PRESERVATION_CONFIRMED'; if ks_p < 0.05 but cohens_d <= 0.5, verdict = 'WEAK_SELECTIVE_PRESERVATION'; otherwise verdict = 'GLOBAL_SIMILARITY_ALTERNATIVE_PLAUSIBLE'. Report the actual top-500 vs. bottom-500 mean difference with 95% CI (bootstrap, 1000 samples). Produce a side-by-side histogram data table (top-500 R distribution vs. bottom-500 R distribution) with bin edges [0.0, 0.1, ..., 1.0].\\n\\n**Section 2 — Orthogonality verdict.** If cos_enc_mean_abs < 0.1: verdict = 'ORTHOGONALITY_DIRECTLY_CONFIRMED'. If 0.1 <= cos_enc_mean_abs < 0.2: verdict = 'NEAR_ORTHOGONAL'. If >= 0.2: verdict = 'ORTHOGONALITY_CLAIM_WEAKENED'. Report top-10 cos_enc values ranked by |cos_enc|. Note whether bottom-500 RLHF-insensitive features show similar or different |cos_enc| — if similar, orthogonality is a general property of the SAE basis rather than specific to RLHF-sensitive features.\\n\\n**Section 3 — Centroid verdict.** Compare ratios: if subspaceB_harmful_ratio and subspaceC_harmful_ratio are both within 50% of subspaceA_harmful_ratio (6.1×), verdict = 'CENTROID_REFLECTS_GLOBAL_SIMILARITY'; if A ratio is > 2× both B and C ratios, verdict = 'CENTROID_REFLECTS_SELECTIVE_PRESERVATION'. Report all 6 ratios in a formatted table.\\n\\n**Section 4 — Layer-20 consistency.** Report layer-20 Reversal Score distribution statistics alongside layer-22. Compute whether distributions are statistically different (KS test between layer-22 top-500 R values and layer-20 top-500 R values). State explicitly whether the zero-Type-R finding replicates at layer 20.\\n\\n**Section 5 — Post-topk density.** Report actual density statistics. Compute: what fraction of the top-500 features have density > 0.10 (the intended filter threshold)? This is the honest version of the density filter result — corrects the MAJOR critique.\\n\\n**Section 6 — Paper corrections checklist.** Produce a structured list of required paper changes:\\n  (a) Semantic labeling protocol: correct description to 'top-5 prompts by maximum pre-topk activation on last token, first 300 characters'\\n  (b) Density filter: replace '100% survived density filter' with actual post-topk density statistics\\n  (c) Classification accuracy: add paragraph reporting 0.476 accuracy as expected null result\\n  (d) Fisher fallback test: move Q3/Q1 comparison (n=4) to limitations, report only unconditional 75% harm-concept rate as primary semantic finding\\n  (e) Centroid analysis: add control-subspace rows to Figure 3 table\\n  (f) Pre-topk/post-topk distinction: add paragraph in limitations distinguishing SAE features (post-topk, validated) from SAE encoder projections (pre-topk, dense, interpretability not validated)\\n\\n**Output format (eval_out.json):** { null_baseline_verdict, selective_preservation_mean_diff, selective_preservation_95ci, orthogonality_verdict, centroid_verdict, layer20_replicates_zero_type_r (bool), posttopk_density_above_threshold_fraction, paper_corrections (list of strings), overall_claim_status ('STRENGTHENED' | 'UNCHANGED' | 'WEAKENED' | 'UNDERMINED'), summary_paragraph (string for paper Discussion section update) }\",", "-          \"depends_on\": [", "-            {", "-              \"id\": \"art_QBm3I0Z5VVJR\",", "-              \"label\": \"iter2 results\"", "-            }", "-          ]", "         }", "       ],", "-      \"expected_outcome\": \"After iteration 3, the paper will have direct answers to all four MAJOR critiques: (1) The null-baseline comparison will definitively settle whether R≈0.921 reflects selective preservation (top-500 significantly higher than bottom-500) or global similarity (both near 1.0), with a t-test p-value, KS statistic, and Cohen's d; (2) Direct cosine-similarity measurements between the refusal direction r and top-500 RLHF-sensitive encoder directions will confirm or refute the geometric-orthogonality claim (expected: |cos| < 0.1); (3) Control-subspace centroid ratios (RLHF-insensitive, random) will contextualize the 6.1×/17.6× ratios and determine whether they are specific to the RLHF-sensitive subspace; (4) Layer-20 Reversal Score statistics will be reported numerically for the first time, either confirming zero Type-R features replicates or revealing that it is layer-22-specific. The evaluation artifact will produce a corrected paper-corrections checklist covering the semantic-labeling description fix, classification-accuracy disclosure, Fisher-test removal, density-filter correction, and pre-topk/post-topk limitation paragraph — giving GEN_PAPER_TEXT everything needed for a complete revision.\",", "-      \"summary\": \"Iteration 3 resolves the credibility gap in the iter2 paper by computing the three missing null comparisons (bottom-500 Reversal Scores, direct refusal-direction orthogonality, control-subspace centroid distances) and substantiating the layer-20 sensitivity check with actual numbers. All computations reuse cached activations from iter2, keeping runtime under one hour. The evaluation synthesizes all results into definitive verdicts on selective preservation vs. global similarity, direct orthogonality confirmation, and centroid specificity — replacing inference with measurement. By the end of iteration 3, every MAJOR reviewer critique will have a numerical answer.\"", "+      \"expected_outcome\": \"After iteration 3, the paper will have definitive numerical answers to all four MAJOR critiques: (1) t-test p-value and Cohen's d comparing top-500 vs bottom-500 Reversal Score distributions — either confirming selective preservation (top >> bottom) or revealing global similarity (both near R=1); (2) mean |cos_sim| between refusal direction r and RLHF-sensitive encoder directions — either directly confirming geometric orthogonality (|cos| < 0.1) or showing the claim needs qualification; (3) centroid ratios in RLHF-insensitive and random subspaces alongside the RLHF-sensitive 6.1×/17.6×, determining whether those ratios are subspace-specific or universal; (4) layer-20 Reversal Score statistics explicitly reported for the first time, either replicating the zero-Type-R finding or revealing layer-specificity. The output JSON also contains the corrected paper descriptions for semantic labeling, density filter, classification accuracy, and pre-topk/post-topk distinction — giving GEN_PAPER_TEXT a complete revision target.\",", "+      \"summary\": \"One focused experiment addresses all four MAJOR reviewer critiques by computing the null comparisons and direct measurements that were missing from iter2. The null-baseline (bottom-500 features) settles the selective-preservation vs. global-similarity question. The cosine-similarity measurement converts the geometric-orthogonality claim from inference to direct evidence. Control-subspace centroid distances contextualize the 6.1×/17.6× ratios. Layer-20 numbers substantiate the sensitivity check. All computations run from scratch in under one hour on one GPU.\"", "     }", "   ]", " }"]}], "originalFile": null, "userModified": false}
```

### [12] ASSISTANT · 2026-07-17 12:30:42 UTC

```
Strategy fixed and written. The single experiment artifact now depends only on valid types (`art_M8-flaZ0OOaP` dataset and `art_pmAcY0CNazP9` research), and all synthesis/verdict logic is folded directly into its output — no cross-artifact dependency needed.
```
