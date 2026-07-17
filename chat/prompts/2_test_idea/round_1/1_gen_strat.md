# gen_strat_1 — test_idea

> Phase: `invention_loop` · round 1 · `gen_strat`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_strat_1` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-07-17 09:58:34 UTC

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
title: Latent Safety Imprinting in RLHF
hypothesis: >-
  Safety RLHF fine-tuning modifies Qwen3-8B's internal representations in two mechanistically distinct ways, distinguishable
  only when comparing all three model variants (base, instruct, abliterated) through a fixed Sparse Autoencoder (SAE) basis.
  The first type — 'Refusal-Coupled' (Type-R) features — are reversed when abliteration is applied; they implement the explicit
  refusal machinery and their activation pattern in the abliterated model reverts close to the base model's. The second type
  — 'Semantically-Imprinted' (Type-I) features — represent the model's understanding of harmful content and persist structurally
  unchanged after abliteration; the abliterated model's activations on these features remain closer to the safety-tuned instruct
  model than to the base model. This creates a 'latent safety imprint': despite identical compliance behavior to the base
  model on harmful prompts, the abliterated model retains the instruct model's representational fingerprint on harm-concept
  features in the SAE basis. Whether the Type-R/Type-I distinction is sharply bimodal or represents a continuum is itself
  an empirical question whose answer reveals the underlying mechanism: a clean bimodal split suggests two categorically distinct
  machinery sets, while a continuous distribution suggests a single graded mechanism where each RLHF-modified feature is partially
  reversed. In either case, the fraction of RLHF-induced feature change that survives abliteration is predicted to be semantically
  coherent and harm-concept-related, providing the first feature-level account of what the abliterated model 'still knows'
  about harmful content.
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
</hypothesis>

<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for study design, proper baselines, and the evaluation/validity norms this field demands.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>

<iteration_status>
Current iteration: 1 of 5
Remaining (including this one): 5
</iteration_status>

<previous_strategies>
No previous strategies exist. This is the FIRST iteration.
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
None yet (first iteration).
</existing_artifacts>





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

### [2] HUMAN-USER prompt · 2026-07-17 09:58:34 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```
