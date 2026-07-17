# review_hypo — create_idea

> Phase: `hypo_loop` · round 3 · `review_hypo`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim transcript of this agent task — every system/user prompt, assistant response, thinking block, tool call and tool result — in the order they occurred. Nothing truncated.

## Task: `review_hypo` (terminal_claude_agent, claude-sonnet-4-6)

### [1] CONFIG · 2026-07-17 09:54:19 UTC

```
model: claude-sonnet-4-6 | effort: medium | permission: bypassPermissions | cwd: /ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_3/review_hypo
```

### [2] SYSTEM-USER prompt · 2026-07-17 09:54:25 UTC

````
<ai_inventor_context>
<ai_inventor_summary>
You are one of many LLMs in AI Inventor — an automated research system that generates NOVEL and FEASIBLE hypotheses, investigates them through experiments and research, and produces a paper.

Your output feeds other LLMs downstream. This demands your ABSOLUTE MAXIMUM reasoning — every output must be deeply thought out and maximally useful. Surface-level responses waste downstream computation.
</ai_inventor_summary>

<your_role>
YOU ARE: A hypothesis reviewer (Step 2.2: REVIEW_HYPO)

Pipeline: GEN_HYPO → REVIEW_HYPO (you) → INVENTION_LOOP → GEN_PAPER_REPO

You review a hypothesis BEFORE any experiments run. Catch problems early.

Rigorous pre-flight check → saves compute. Rubber-stamping → wasted pipeline run.
</your_role>
</ai_inventor_context>

ROLE: You are a very experienced and critical conference reviewer.
Your expertise spans the domain of the hypothesis under review.
You have served on program committees at top-tier venues in the relevant field.

TASK: Perform a deep and honest review (at the level of a top-tier venue submission) of
this research hypothesis BEFORE any experiments have been run.

GOAL: Your review feeds directly back to the hypothesis author. The objective is to
maximize the overall review score in subsequent rounds. Every piece of feedback you
give should be written with this goal in mind — prioritize the critiques and suggestions
that would produce the largest score improvement if addressed. Don't waste the author's
iteration budget on low-impact polish when there are score-blocking issues to fix.

STRENGTHS AND WEAKNESSES: Provide a thorough assessment touching on each of these:
(a) Originality: Are the ideas new? Novel combination of known techniques? Clear
    differentiation from prior work? Is related work adequately cited?
(b) Quality: Is the proposal technically sound? Are claims well supported? Is the
    methodology appropriate? Are the authors honest about limitations?
(c) Clarity: Is the hypothesis clearly written and well organized? Does it provide
    enough information for an expert to understand and evaluate it?
(d) Significance: Are the expected results important? Would others build on this?
    Does it address a meaningful problem better than prior work?

SUPPLEMENTARY SCORES: Rate each on a 1-4 scale.
Soundness (1-4) — soundness of the technical claims and proposed methodology:
  4: excellent  3: good  2: fair  1: poor
Presentation (1-4) — quality of writing, clarity, and contextualization relative to prior work:
  4: excellent  3: good  2: fair  1: poor
Contribution (1-4) — quality of the overall contribution, importance of questions asked,
originality of ideas, value to the broader research community:
  4: excellent  3: good  2: fair  1: poor

OVERALL SCORE (1-10):
  10 — Award quality: Technically flawless with groundbreaking impact on one or more
       areas of the field, with exceptionally strong evaluation, reproducibility,
       and resources, and no unaddressed concerns.
   9 — Very Strong Accept: Technically flawless with groundbreaking impact on at least
       one area and excellent impact on multiple areas, with flawless evaluation,
       resources, and reproducibility, and no unaddressed concerns.
   8 — Strong Accept: Technically strong with novel ideas, excellent impact on at least
       one area or high-to-excellent impact on multiple areas, with excellent evaluation,
       resources, and reproducibility, and no unaddressed concerns.
   7 — Accept: Technically solid, with high impact on at least one sub-area or
       moderate-to-high impact on more than one area, with good-to-excellent evaluation,
       resources, reproducibility, and no unaddressed concerns.
   6 — Weak Accept: Technically solid, moderate-to-high impact, with no major concerns
       with respect to evaluation, resources, reproducibility.
   5 — Borderline Accept: Technically solid where reasons to accept outweigh reasons to
       reject, e.g., limited evaluation. Use sparingly.
   4 — Borderline Reject: Technically solid where reasons to reject, e.g., limited
       evaluation, outweigh reasons to accept. Use sparingly.
   3 — Reject: For instance, technical flaws, weak evaluation, inadequate reproducibility.
   2 — Strong Reject: For instance, major technical flaws, poor evaluation, limited
       impact, poor reproducibility.
   1 — Very Strong Reject: For instance, trivial results or unaddressed concerns.

CONFIDENCE (1-5):
  5: Absolutely certain. Very familiar with related work, checked details carefully.
  4: Confident but not absolutely certain. Unlikely you misunderstood something.
  3: Fairly confident. Possible you missed some related work or details.
  2: Willing to defend your assessment, but quite likely missed central aspects.
  1: Educated guess. Not in your area or difficult to evaluate.

For each dimension, provide a list of specific improvements:
- WHAT needs to change
- HOW to change it (concrete enough for the author to act on immediately)
- EXPECTED SCORE IMPACT: how much would fixing this raise the overall score?

REVIEW PRINCIPLES:
- Be specific and actionable — vague critique is useless
- Ground your review in evidence — search for existing work, accepted papers, known results
- Rank critiques by score impact — address the biggest score blockers first
- Distinguish major issues (would waste compute if not fixed) from minor issues (polish)
- Acknowledge genuine strengths — don't be negative for its own sake
- Compare against the bar set by accepted papers at top-tier venues
- Flag fatal flaws that would make experiments pointless if not addressed first

<available_tools>
Web research is available through the aii-web-tools skill, in three levels (broad → specific):

1. web search — Returns titles, URLs, snippets. Use first to discover and scan the landscape.
2. web fetch — Reads a page and returns its content as markdown (HTML or PDF). Use to understand a source. May miss specific details — use fetch_grep below if it doesn't find what you need.
3. fetch_grep — Regex search over a page/PDF's full text. Returns exact matching sections with context. Use for precise details, exact numbers, methodology, or PDFs.

Workflow: search → fetch (understand) → fetch_grep (extract specifics).
</available_tools>

<role>
You are a very experienced and critical conference reviewer specialized in the domain of the work under review.
You have reviewed for top-tier venues in the relevant field. Your reviews are known for
being thorough, fair, and grounded in the actual state of the field.
</role>

<hypothesis>
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

<review_context>
No experiments have been run yet — evaluate the hypothesis purely on its merits.
</review_context>

<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for judging whether the hypothesis is genuinely novel versus already-done or a known dead end in this field.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>

<previous_hypothesis>
The hypothesis from the PREVIOUS iteration (before the revision under review).
Use this to classify how the current hypothesis relates to it (see the H↔H
edge instructions in the task).

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
  The huihui-ai/Qwen3-8B-abliterated model was produced using the remove-refusals-with-transformers codebase, which implements
  the difference-of-means + orthogonal projection abliteration method based on the 'single refusal direction' approach (Arditi
  et al., 2024). We will verify this explicitly by inspecting the source code before running experiments. If the method differs
  substantially (e.g., fine-tuning-based uncensoring rather than weight editing), we will reproduce abliteration from scratch
  using a controlled script on Qwen3-8B to ensure the abliteration direction is well-defined and comparable to individual
  SAE feature directions.
- >-
  The per-feature Reversal_score — defined as Pearson correlation across N=500 harmful prompts between the instruct change
  vector [Δinstruct_i(f)] and the abliterated change vector [Δabliterated_i(f)] — captures a meaningful signal about whether
  abliteration preserves or reverses the RLHF modification of feature f. This is a correlation over prompt-level scalar activations
  (one value per prompt per feature), not a cosine similarity of scalars, and is thus a proper statistical measure of co-directional
  change.
- >-
  RLHF-sensitive features (high Safety_change) that receive the LLM-assigned semantic label 'harm-concept related' will show
  this label consistently across diverse harmful prompt phrasings — i.e., the labeling is not triggered by surface patterns
  in the top-activating prompts but reflects the feature's stable semantic role.
- >-
  The instruct model's safety-relevant SAE features activate at comparable levels whether the model receives raw text prompts
  or template-formatted prompts (to be validated in a pilot study comparing activation patterns under both formats before
  committing to the raw-text protocol).
investigation_approach: |-
  Five-step experiment using all-Python implementation on a single GPU (16+ GB VRAM):

  **Step 0 — Abliteration method verification.**
  Fetch and inspect the source code of the remove-refusals-with-transformers repository to confirm the exact technique used: specifically whether it is difference-of-means + orthogonal projection applied to residual stream weights, and which layers are modified. If the community model's creation method cannot be confirmed as standard orthogonal abliteration (e.g., if it is fine-tuning-based uncensoring), reproduce abliteration from scratch using the standard script on Qwen3-8B, creating a controlled abliterated checkpoint with known methodology. Document the layer set and prompt set used.

  **Step 0.5 — SAE reconstruction quality gate + chat-template pilot.**
  Load the frozen Qwen-Scope base-SAE (Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50). Run 50 harmful prompts through all three model variants; compute reconstruction R² (fraction of variance explained by SAE reconstruction) for base, instruct, and abliterated activations. If R² ≥ 0.85 for instruct and abliterated variants, proceed to full feature analysis using both encoder activations and reconstruction-based comparisons. If R² < 0.85 for instruct or abliterated variants, proceed with encoder-only analysis (feature activation vectors from the SAE encoder without relying on decoder reconstruction) and document this limitation. Separately, run a 20-prompt chat-template pilot: compare the top-50 high-Safety_change SAE feature activations for the instruct model under raw text vs. template-formatted harmful prompts. If activation patterns differ substantially (Pearson r < 0.80), switch to template-formatted prompts for the instruct model only.

  **Step 1 — Data collection.**
  Load all three Qwen3-8B model checkpoints: base (Qwen/Qwen3-8B-Base), instruct/safety-tuned (Qwen/Qwen3-8B), and abliterated (huihui-ai/Qwen3-8B-abliterated or self-produced). Collect 500 harmful prompts (from WildGuard/BeaverTails/HarmBench) and 500 benign prompts. Provide all prompts using the format validated in Step 0.5. Run forward passes through each model, extract residual stream activations at the SAE-covered layer. Project through the frozen SAE encoder to obtain 64K-dimensional sparse feature activation vectors for each (prompt, model) triple.

  **Step 2 — Feature classification via per-feature Pearson correlation.**
  For each of the 64K SAE features f, compute over the 500 harmful prompts:
  - Safety_change(f) = mean_i |act_instruct_i(f) - act_base_i(f)| — how much RLHF changed this feature on average
  - Δinstruct(f) = vector [act_instruct_i(f) - act_base_i(f) for i in 1..500] — a 500-dimensional vector of instruct changes
  - Δabliterated(f) = vector [act_abliterated_i(f) - act_base_i(f) for i in 1..500] — a 500-dimensional vector of abliterated changes
  - Reversal_score R(f) = PearsonR(Δinstruct(f), Δabliterated(f)) — the correlation between these two change vectors

  Worked example: Suppose feature f is active on harmful chemistry prompts. If RLHF increases its activation (Δinstruct values ≈ [+0.8, +1.2, +0.5, +0.9, +1.1, ...]) and abliteration also leaves it elevated (Δabliterated ≈ [+0.7, +1.0, +0.4, +0.8, +0.9, ...]), then R(f) ≈ +0.99 → Type-I (imprinted). If instead abliteration reverses the increase (Δabliterated ≈ [-0.1, -0.2, -0.1, -0.1, -0.2, ...]), then R(f) ≈ -0.98 → Type-R (reversed). A feature with R(f) near 0 is ambiguous — partially reversed. The distribution of R(f) across RLHF-sensitive features (top-500 by Safety_change magnitude) will be tested for bimodality using Hartigan's dip test (H₀: unimodal). Both outcomes are pre-specified as interpretable: bimodal → discrete Type-R/Type-I classification; continuous → treat R(f) as a graded survival score and examine semantic correlates along the continuum.

  **Step 3 — Semantic interpretation and distance analysis.**
  For the top-100 features by |R(f) - 0.5| (most clearly Type-R or Type-I) × Safety_change magnitude: use maximum-activating prompt segments as context, then call a cheap LLM via OpenRouter (Qwen2.5-7B or gemini-flash at ~$0.001/1K tokens) to generate semantic labels. In the Type-R and Type-I feature subspaces separately, compute centroid distances d(abliterated, instruct) and d(abliterated, base) for harmful vs. benign prompts.

  For the exploratory capability-gap analysis: compute the fraction of RLHF-sensitive features that are Type-I (R(f) > 0.5) at the single SAE layer analyzed and report whether it shows a qualitative trend with the SAE reconstruction R² gap between abliterated and base model. This is explicitly labeled as descriptive/exploratory (N ≈ 1 SAE layer within Qwen3-8B, insufficient for statistical reliability). A definitive test requires ≥3 model families and is marked as future work.
success_criteria: |-
  CONFIRM core hypothesis if BOTH of the following hold:
  1. **Bimodality test (or continuous semantic coherence):** Hartigan's dip test on the R(f) distribution for RLHF-sensitive features (top-500 by Safety_change) yields p<0.05 (bimodal), with the two modes interpreted as Type-R (R<0) and Type-I (R>+0.5). Alternatively, if the test yields p≥0.05 (continuous), the hypothesis is confirmed in a weaker 'graded imprint' form if the continuous case shows: features with high R(f) (R>+0.5) receive harm-concept semantic labels from the LLM at ≥2× the rate of features with low R(f) (R<0), establishing that survival of the RLHF modification correlates with harm-concept semantic content regardless of whether the distribution is bimodal.
  2. **Representational fingerprint:** In the high-R(f) feature subspace, d(abliterated, instruct) < d(abliterated, base) for harmful prompts, while d(abliterated, instruct) ≈ d(abliterated, base) for benign prompts. This is the core quantitative test: the abliterated model's harm-concept representations look more like the instruct model than the base model, specifically when processing harmful content.

  ADDITIONAL CONFIRMATORY CRITERION:
  3. **Semantic label coherence:** High-R(f) features (Type-I or high survival-score features) receive semantic labels classified as 'harm-concept related' (weapon synthesis, manipulation, deception) by the LLM labeler at a significantly higher rate than low-R(f) features (Type-R or low survival-score features) — p<0.05 by Fisher's exact test on a manual validation sample of 50 features per group.

  EXPLORATORY (non-confirmatory) CRITERION:
  4. **Capability gap preliminary signal:** The fraction of Type-I features at the analyzed layer is reported alongside the abliterated vs. base capability gap on a mini-MMLU sample (200 questions); this correlation is descriptive only and cannot be statistically validated at N=1 SAE layer within one model — it is a qualitative preliminary signal for future multi-model work.

  DISCONFIRM hypothesis if: (a) Hartigan's test yields p≥0.05 AND high-R(f) features show no enrichment for harm-concept labels over low-R(f) features; (b) d(abliterated, instruct) ≥ d(abliterated, base) in the high-R(f) feature subspace for harmful prompts; (c) the SAE reconstruction R² gate (Step 0.5) fails below 0.70 for the instruct model, making encoder-only analysis unreliable (in this case the experiment is inconclusive rather than disconfirmed, and the Qwen3-Instruct SAE should be used as an alternative instrument in a follow-up).
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
  this paper's finding that sparse latents diverge after SFT motivates our SAE reconstruction quality gate (Step 0.5).
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
  analysis if reconstruction quality is insufficient. A Qwen3-Instruct SAE has been released covering Qwen3-1.7B, -4B, and
  -8B, which could serve as an alternative or cross-validation instrument.
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
    variants, the base-SAE is an acceptable cross-model measurement instrument. If R² < 0.85, encoder-only analysis (feature
    activation vectors without relying on decoder reconstruction quality) is used instead, and the limitation is documented
    explicitly.
- term: Safety_change(f)
  definition: >-
    For a given SAE feature f, the mean absolute difference between instruct and base model activations across the harmful
    prompt set: mean_i |act_instruct_i(f) - act_base_i(f)|. Features with high Safety_change are 'RLHF-sensitive' and are
    the ones further analyzed by Reversal_score.
- term: Qwen-Scope
  definition: >-
    An open-source suite of Sparse Autoencoders released by the Qwen team (May 2026) for interpretability and development
    use with Qwen3 and Qwen3.5 model families. Includes base-model SAEs for Qwen3-8B with 64,000 features (SAE-Res-Qwen3-8B-Base-W64K-L0_50),
    which we use as the fixed measurement instrument in this study. Trained on base model activations; reconstruction quality
    on instruct model activations requires validation before use.
summary: >-
  We hypothesize that RLHF safety fine-tuning in Qwen3-8B writes two (or more) mechanistically distinct types of changes into
  the model's SAE feature space — 'Refusal-Coupled' features reversed by abliteration and 'Semantically-Imprinted' features
  that persist — and test this using the Qwen-Scope base-SAE as a fixed measurement instrument across base, instruct, and
  abliterated Qwen3-8B variants, with a novel per-feature Pearson correlation (Reversal_score) that avoids definitional artifacts,
  a pre-experiment SAE reconstruction quality gate, and explicit pre-registration of both the bimodal and continuous interpretations,
  providing the first feature-level account of what the abliterated model 'still knows' about harmful content and why that
  matters for targeted safety restoration.
</previous_hypothesis>

<previous_review>
Critiques from the previous review. Check which ones have been addressed
in the revised hypothesis. Do NOT re-raise critiques that have been adequately fixed.
Only re-raise if the fix is insufficient.

- [MAJOR] (novelty) Crosscoders (arXiv:2602.11729, arXiv:2606.26474) are the established, purpose-built tool for exactly the multi-model feature comparison this hypothesis proposes, and they are absent from the related works. Crosscoders train a shared sparse dictionary jointly on activations from two or more model variants, naturally partitioning features into 'shared' and 'model-specific' categories. Dedicated Feature Crosscoders (DFCs) were recently used in an ICML 2026 spotlight paper (arXiv:2606.26474) to localize RL-induced tool-calling to a single feature in Qwen2.5-3B. The base/instruct/abliterated three-model comparison in this hypothesis could, in principle, be addressed by a three-model crosscoder rather than projecting through a frozen base-SAE. The hypothesis must either (a) justify why the frozen-base-SAE approach is preferable or complementary — the strongest argument is that it requires no training, reuses Qwen-Scope's labeled feature vocabulary, and is directly comparable to arXiv:2605.11426 — or (b) add a crosscoder variant as a robustness check. Omitting this literature makes the contribution appear less well-situated than it actually is and will draw reviewer criticism at top venues.
  Action: Add a paragraph in the related works section discussing crosscoders (cite arXiv:2602.11729 and arXiv:2606.26474). Justify the frozen-base-SAE choice on the grounds of: (1) zero training cost, (2) reuse of Qwen-Scope's existing labeled feature vocabulary enabling interpretability without re-deriving feature semantics, (3) methodological continuity with arXiv:2605.11426 which validates base-SAE diagnostics for SFT analysis. If resources allow, running a two-model crosscoder (base+instruct) as a comparison would strengthen the methodology section considerably.
- [MINOR] (methodology) The SAE layer to analyze is underspecified. The hypothesis refers to 'the SAE-covered layer (typically a mid-layer)' without specifying which of Qwen-Scope's multiple available SAE layers will be used or why. Safety-relevant representations are known to vary substantially across layers (some work identifies early layers for refusal detection, mid-layers for semantic content, late layers for output formatting). The choice of layer directly affects which features are captured by Safety_change and which are labeled Type-R vs. Type-I. If the results are layer-dependent, the single-layer analysis may produce a misleading picture.
  Action: In Step 1, specify the layer selection criterion: e.g., 'We use the Qwen-Scope SAE at layer L (to be determined by a preliminary pass: we select the layer with the largest variance in Safety_change across features, as a proxy for richest safety-relevant signal). If resources allow, we repeat the analysis at two additional layers as a sensitivity check.' This single sentence adds reproducibility and acknowledges that layer choice is a methodological decision, not an incidental detail.
- [MINOR] (rigor) Criterion 3 (semantic label coherence) relies on a cheap LLM (Qwen2.5-7B or gemini-flash) as the sole judge for classifying features as 'harm-concept related.' The reliability of this labeler is not validated. Using the same model family (Qwen) for both the SAE analysis and the semantic labeling creates a potential circularity: Qwen2.5-7B may apply the same semantic categories as Qwen3-8B's RLHF training, making the label agreement an artifact of shared training data rather than an independent validation. Additionally, the prompt used for the LLM labeler is not specified, and LLM labeling of SAE features is known to be sensitive to prompt framing.
  Action: Add two safeguards: (1) validate the LLM labeler on a small gold set — manually annotate 20 features and check agreement with the automated labels before using them for Criterion 3; (2) use a labeler from a different model family (e.g., Llama-3 or Gemini) rather than Qwen2.5-7B to avoid within-family circularity. Report inter-rater agreement (Cohen's kappa between human annotation and LLM labels) on the gold set. This is low-cost but substantially strengthens Criterion 3's evidential value.
- [MINOR] (methodology) The encoder-only fallback (when R² < 0.85) is presented as a reliable alternative, but JumpReLU thresholds calibrated on base activations will produce systematically distorted sparsity patterns when applied to instruct activations — features whose activations are systematically elevated or suppressed by RLHF will be gated differently than the base calibration assumes. This means that even encoder-only analysis carries a distributional shift caveat that could bias the Safety_change computation: if RLHF raises a feature's mean activation above the JumpReLU threshold, that feature may appear more active in the instruct model not because of genuine RLHF modification but because of threshold effects.
  Action: In the encoder-only fallback description, add an explicit caveat: 'Encoder-only activations under the base-calibrated JumpReLU thresholds carry a residual distributional shift confound: if RLHF systematically elevates a feature's mean activation, the base threshold may undercount it and inflate Safety_change. In the encoder-only regime, we will additionally report raw (pre-threshold) encoder activations as a robustness check, and interpret Safety_change values cautiously.' This is a straightforward addition that prevents a future reviewer from using this as a fatal objection.
- [MINOR] (scope) The huihui-ai/Qwen3-8B-abliterated model card describes the technique as 'a new and faster method' without citing a specific code repository. The model card references the remove-refusals-with-transformers project as a conceptual basis but explicitly says the implementation does NOT use TransformerLens and may be a different algorithm. The hypothesis's Step 0 correctly plans to verify the technique, but if the community model turns out to use a non-standard method (as the model card wording suggests is possible), the scratch-reproduction fallback will require additional GPU time that may not be budgeted.
  Action: Prioritize Step 0 as the very first action and explicitly budget for the scratch-reproduction fallback. The model card wording 'new and faster method' is a yellow flag. Consider proactively reproducing abliteration from scratch regardless, since a controlled abliterated checkpoint with known methodology would strengthen the experimental claim considerably and costs little additional effort given that the standard remove-refusals-with-transformers script runs in under an hour on a single GPU.
</previous_review>

<task>
Provide a thorough peer review of this research hypothesis.

STEP 1 — GROUND YOUR REVIEW IN EVIDENCE:
Before writing critiques, search for relevant context to make your review authoritative:
- Search for accepted papers at top venues in this area — what level of
  contribution gets accepted? How does this hypothesis compare?
- Search for the closest existing work — is this genuinely novel or incremental?
- Check if the proposed methodology has known failure modes in the literature

STEP 2 — WRITE YOUR REVIEW:
For each critique:
1. Categorize: methodology, evidence, novelty, clarity, scope, or rigor
2. Rate severity: major (would waste compute if not fixed) or minor (polish)
3. Describe the issue clearly
4. Suggest a concrete action to address it

Focus on the most impactful issues. Flag fatal flaws that would waste compute if not fixed first.

STABILITY IS OK: If the hypothesis is on track and just needs more iterations to prove itself,
keep your feedback similar to the previous round. Don't manufacture new critiques — only escalate
when the revision introduced new issues or failed to address prior ones.

STEP 3 — H↔H EDGE (only if a <previous_hypothesis> block is present):
Classify how the current hypothesis relates to the previous iteration's hypothesis
using Moulines's structuralist typology. Set ``relation_type`` to one of:
    - "evolution": refining specialised claims while keeping the same conceptual frame
    - "embedding": the previous hypothesis is now a special case of a broader frame
    - "replacement": rejecting the previous frame entirely (Kuhnian, incommensurable shift)
Set ``relation_rationale`` to a brief justification (≤120 chars).

If no <previous_hypothesis> is present (this is iteration 1), leave both fields
null/empty.

Provide your review via structured output.
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
    "Critique": {
      "description": "A single actionable critique from the reviewer.",
      "properties": {
        "category": {
          "description": "Category: 'methodology', 'evidence', 'novelty', 'clarity', 'scope', or 'rigor'",
          "title": "Category",
          "type": "string"
        },
        "severity": {
          "description": "Severity: 'major' or 'minor'",
          "title": "Severity",
          "type": "string"
        },
        "description": {
          "description": "Clear description of the issue",
          "title": "Description",
          "type": "string"
        },
        "suggested_action": {
          "description": "Concrete suggestion for how to address this critique",
          "title": "Suggested Action",
          "type": "string"
        }
      },
      "required": [
        "category",
        "severity",
        "description",
        "suggested_action"
      ],
      "title": "Critique",
      "type": "object"
    },
    "DimensionScore": {
      "description": "Score for a single review dimension with improvement suggestions.",
      "properties": {
        "dimension": {
          "description": "Dimension name: 'soundness', 'presentation', or 'contribution'",
          "title": "Dimension",
          "type": "string"
        },
        "score": {
          "description": "Score from 1 (poor) to 4 (excellent)",
          "title": "Score",
          "type": "integer"
        },
        "justification": {
          "description": "Brief justification for this score",
          "title": "Justification",
          "type": "string"
        },
        "improvements": {
          "description": "Specific improvements to raise the score (what + how + why)",
          "items": {
            "type": "string"
          },
          "title": "Improvements",
          "type": "array"
        }
      },
      "required": [
        "dimension",
        "score",
        "justification"
      ],
      "title": "DimensionScore",
      "type": "object"
    }
  },
  "description": "ReviewerFeedback + Moulines H\u2194H typology for hypo_loop iterations.\n\nAdds ``relation_type`` + ``relation_rationale`` so the trace projection\ncan build a typed edge from the previous iteration's hypothesis to\nthis iteration's. On iteration 1 (no previous), both fields are\nempty/None.",
  "properties": {
    "overall_assessment": {
      "description": "Overall assessment of the paper's quality and readiness",
      "title": "Overall Assessment",
      "type": "string"
    },
    "strengths": {
      "description": "Key strengths of the paper",
      "items": {
        "type": "string"
      },
      "title": "Strengths",
      "type": "array"
    },
    "dimension_scores": {
      "description": "Scores (1-4) for: soundness, presentation, contribution",
      "items": {
        "$ref": "#/$defs/DimensionScore"
      },
      "title": "Dimension Scores",
      "type": "array"
    },
    "critiques": {
      "description": "Actionable critiques \u2014 specific issues with concrete suggestions",
      "items": {
        "$ref": "#/$defs/Critique"
      },
      "title": "Critiques",
      "type": "array"
    },
    "score": {
      "description": "Overall quality score from 1 (very strong reject) to 10 (award quality)",
      "title": "Score",
      "type": "integer"
    },
    "confidence": {
      "default": 3,
      "description": "Confidence in assessment from 1 (educated guess) to 5 (absolutely certain)",
      "title": "Confidence",
      "type": "integer"
    },
    "relation_type": {
      "anyOf": [
        {
          "enum": [
            "evolution",
            "embedding",
            "replacement"
          ],
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Moulines's structuralist typology classifying how this iteration's hypothesis relates to the previous iteration's: 'evolution' \u2014 refining specialised claims while keeping the same conceptual frame; 'embedding' \u2014 the previous hypothesis is now a special case of a broader frame; 'replacement' \u2014 rejecting the previous frame entirely (Kuhnian shift). Leave null on the first iteration (no previous hypothesis).",
      "title": "Relation Type"
    },
    "relation_rationale": {
      "default": "",
      "description": "Brief rationale (one short line, \u2264120 chars) for the relation_type. Empty on the first iteration.",
      "maxLength": 120,
      "title": "Relation Rationale",
      "type": "string"
    }
  },
  "required": [
    "overall_assessment",
    "strengths",
    "critiques",
    "score"
  ],
  "title": "HypoReviewerFeedback",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````

### [3] HUMAN-USER prompt · 2026-07-17 09:54:25 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [4] ASSISTANT · 2026-07-17 09:55:15 UTC

```
I'll review this hypothesis carefully, checking which previous critiques were addressed and identifying any remaining issues.
```

### [5] TOOL CALL — Skill · 2026-07-17 09:55:15 UTC

```
aii-web-tools
```

### [6] TOOL RESULT — Skill · 2026-07-17 09:55:15 UTC

```
{"success": true, "commandName": "aii-web-tools"}
```

### [7] ASSISTANT · 2026-07-17 09:55:18 UTC

```
Let me search for recent work to ground my review authoritatively.
```

### [8] TOOL CALL — ToolSearch · 2026-07-17 09:55:18 UTC

```
{'query': 'select:WebSearch,WebFetch'}
```

### [9] TOOL RESULT — ToolSearch · 2026-07-17 09:55:18 UTC

```
{"matches": ["WebSearch", "WebFetch"], "query": "select:WebSearch,WebFetch", "total_deferred_tools": 21}
```

### [10] TOOL CALL — WebSearch · 2026-07-17 09:55:22 UTC

```
SAE sparse autoencoder safety fine-tuning RLHF abliteration feature analysis 2025 2026
```

### [11] TOOL CALL — WebSearch · 2026-07-17 09:55:22 UTC

```
abliteration refusal direction orthogonal projection SAE features mechanistic interpretability 2025
```

### [12] TOOL RESULT — WebSearch · 2026-07-17 09:55:30 UTC

```
{"query": "SAE sparse autoencoder safety fine-tuning RLHF abliteration feature analysis 2025 2026", "results": [{"tool_use_id": "srvtoolu_018e2E9PWjEZLTBng5D83zDA", "content": [{"title": "Towards Safeguarding LLM Fine-tuning APIs against Cipher Attacks", "url": "https://arxiv.org/pdf/2508.17158"}, {"title": "SAFER: Probing Safety in Reward Models with Sparse Autoencoder", "url": "https://arxiv.org/html/2507.00665v3"}, {"title": "Sparse Autoencoder Features for Classifications and Transferability - ACL Anthology", "url": "https://aclanthology.org/2025.emnlp-main.1521/"}, {"title": "NExT-Guard: Training-Free Streaming Safeguard without Token-Level Labels", "url": "https://arxiv.org/pdf/2603.02219"}, {"title": "[2507.00665] SAFER: Probing Safety in Reward Models with Sparse Autoencoder", "url": "https://arxiv.org/abs/2507.00665"}, {"title": "Sparse Autoencoders (SAEs) | Longterm Wiki", "url": "https://www.longtermwiki.com/wiki/E480"}, {"title": "Discovering Millions of Interpretable Features with Sparse Autoencoders", "url": "https://arxiv.org/pdf/2606.26620"}, {"title": "[2512.23260] Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation", "url": "https://arxiv.org/abs/2512.23260"}, {"title": "Graph-Regularized Sparse Autoencoders for LLM Safety Steering", "url": "https://arxiv.org/pdf/2512.06655"}, {"title": "Route Sparse Autoencoder to Interpret Large Language ...", "url": "https://aclanthology.org/2025.emnlp-main.346.pdf"}]}, "I found several relevant research papers and resources on sparse autoencoders (SAEs) for safety, fine-tuning, and RLHF from 2025-2026. Here are the key findings:\n\n## Recent Research on SAEs and Safety\n\nSparse Autoencoders uncover human-interpretable features in reward model activations, enabling insight into safety-relevant decision-making. One notable framework is SAFER (Sparse Autoencoder For Enhanced Reward model), which presents a novel framework for interpreting and improving reward models through mechanistic analysis by uncovering human-interpretable features in reward model activations and quantifying the salience of individual features by activation differences between chosen and rejected responses.\n\n## Safety-Critical Applications\n\nSAEs provide potential for uncovering structured, human-interpretable representations in Large Language Models, making them a crucial tool for transparent and controllable AI systems, and are systematically analyzed for interpretable feature extraction from LLMs in safety-critical classification tasks.\n\n## Recent Developments in SAE-Based Safety Alignment\n\nA method called SAILS achieves up to 99.6% safety rate on Gemma-2-9B—exceeding full fine-tuning by 7.4 points and matching RLHF-based models—while updating only 0.19% of parameters and providing interpretability.\n\nAdditionally, Graph-Regularized Sparse Autoencoders (GSAE) learn safety-steering directions by smoothing SAE decoder vectors over a neuron co-activation graph and applying the resulting direction bank through a two-gate runtime controller.\n\nThese represent cutting-edge work integrating SAEs with safety fine-tuning and RLHF approaches from late 2025 into early 2026."], "durationSeconds": 8.556414009000008, "searchCount": 1}
```

### [13] TOOL RESULT — WebSearch · 2026-07-17 09:55:36 UTC

```
{"query": "abliteration refusal direction orthogonal projection SAE features mechanistic interpretability 2025", "results": [{"tool_use_id": "srvtoolu_01UvqWnRyvsZriKM7xg95djb", "content": [{"title": "Projected Abliteration", "url": "https://huggingface.co/blog/grimjim/projected-abliteration"}, {"title": "Not All Refusals Are Equal: How Safety Alignment Fails Cybersecurity at Scale", "url": "https://arxiv.org/html/2607.02714v1"}, {"title": "LLM Refusal Abliteration Mechanisms", "url": "https://www.emergentmind.com/topics/refusal-abliteration"}, {"title": "ORBA: Orthogonal Reflection Bounded Ablation — A Geometrically Exact Detour in Directional Activation Editing", "url": "https://huggingface.co/blog/grimjim/orthogonal-reflection-bounded-ablation"}, {"title": "Norm-Preserving Biprojected Abliteration", "url": "https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration"}, {"title": "Abliteration - Learn AI - Miraheze", "url": "https://ai.miraheze.org/wiki/Abliteration"}, {"title": "Not All Refusals Are Equal: How Safety Alignment Fails Cybersecurity at Scale", "url": "https://arxiv.org/pdf/2607.02714"}, {"title": "Discovering Millions of Interpretable Features with Sparse Autoencoders", "url": "https://arxiv.org/pdf/2606.26620"}, {"title": "Refusal Direction in LLM Safety", "url": "https://www.emergentmind.com/topics/refusal-direction"}, {"title": "On the Failure of Topic-Matched Contrast Baselines in Multi-Directional Refusal Abliteration", "url": "https://arxiv.org/pdf/2603.22061"}]}, "I found comprehensive search results on this topic. Here's what the 2025 research shows about abliteration, refusal directions, orthogonal projection, and mechanistic interpretability:\n\n## Overview of Abliteration and Refusal Directions\n\nAbliteration is a technique for removing refusal behaviors from language models by identifying and intervening on \"refusal directions\" in activation space, notionally represented via a single mean refusal direction. The process works by finding a refusal direction in a model's residual stream and projecting it out of the weight matrices.\n\n## Recent Advances in 2025\n\nHowever, recent work has refined our understanding of refusal mechanisms. The refusal subspace is richer than a single direction: Wollschläger et al. (2025) identify multi-dimensional polyhedral concept cones, and Pan et al. (2025) find a dominant direction accompanied by multiple semantically distinct secondary features.\n\n### Projected Abliteration\n\nA refinement called \"projected abliteration\" improves upon the conventional approach by removing only the mechanistically relevant components of the refusal direction, confirming a prior finding that LLMs encode refusal and harmfulness separately. The orthogonal component captures the mechanistically specific refusal behavior.\n\n### Norm-Preserving Approaches\n\nThe norm-preserving approach not only matched the uncensoring effectiveness but significantly improved reasoning capability.\n\n## Broader Context\n\nThe current landscape of abliteration in 2025 and 2026 is characterized by the routine production of abliterated model variants, the development of increasingly sophisticated defense mechanisms, and an ongoing arms race between abliteration techniques and alignment robustness research."], "durationSeconds": 13.787455843999997, "searchCount": 1}
```

### [14] TOOL CALL — WebSearch · 2026-07-17 09:55:40 UTC

```
LatentBiopsy arXiv 2603.27412 harmful geometry abliteration survival 2025
```

### [15] TOOL CALL — WebSearch · 2026-07-17 09:55:40 UTC

```
crosscoder multi-model SAE base instruct feature comparison safety 2025 2026
```

### [16] TOOL RESULT — WebSearch · 2026-07-17 09:55:46 UTC

```
{"query": "LatentBiopsy arXiv 2603.27412 harmful geometry abliteration survival 2025", "results": [{"tool_use_id": "srvtoolu_01DU9QzXiqXGvTpuYzTNtFx5", "content": [{"title": "[2603.27412] The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/abs/2603.27412"}, {"title": "The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/html/2603.27412"}, {"title": "The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/pdf/2603.27412"}, {"title": "[2507.13300] AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research", "url": "https://arxiv.org/abs/2507.13300"}, {"title": "Interactive Volumetry Of Liver Ablation Zones", "url": "https://arxiv.org/pdf/1512.04582"}, {"title": "[2507.08038] AblationBench: Evaluating Automated Planning of Ablations in Empirical AI Research", "url": "https://arxiv.org/abs/2507.08038"}, {"title": "Survival Concept-Based Learning Models", "url": "https://arxiv.org/pdf/2502.05950"}, {"title": "Human-aligned Deep Learning: Explainability, Causality, and Biological   Inspiration", "url": "https://arxiv.org/pdf/2504.13717"}, {"title": "TabPFN-3: Technical Report", "url": "https://arxiv.org/pdf/2605.13986"}, {"title": "Robust Multimodal Survival Prediction with the Latent Differentiation   Conditional Variational AutoEncoder", "url": "https://arxiv.org/pdf/2503.09496"}]}, "I found the paper you're looking for! Here are the key details about arXiv paper 2603.27412:\n\n**Title:** \"The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams\"\n\n**Method Overview:** LatentBiopsy is a training-free method for detecting harmful prompts by analysing the geometry of residual-stream activations in large language models. Given 200 safe normative prompts, LatentBiopsy computes the leading principal component of their activations at a target layer and characterises new prompts by their radial deviation angle θ from this reference direction.\n\n**Key Finding on Geometry and Ablation:** Geometry survives refusal ablation: both abliterated variants achieve AUROC at most 0.015 below their instruction-tuned counterparts, establishing a geometric dissociation between harmful-intent representation and the downstream generative mechanism. This means that the harmful content detection capability persists even when the refusal mechanism is surgically removed.\n\n**Performance:** Across all six variants, LatentBiopsy achieves AUROC ≥0.937 for harmful-vs-normative detection and AUROC = 1.000 for discriminating harmful from benign-aggressive prompts (XSTest)\n\nThe paper was published on March 28, 2026."], "durationSeconds": 8.238563983, "searchCount": 1}
```

### [17] TOOL RESULT — WebSearch · 2026-07-17 09:55:48 UTC

```
{"query": "crosscoder multi-model SAE base instruct feature comparison safety 2025 2026", "results": [{"tool_use_id": "srvtoolu_01CkrLKUcSDdRyMRJe12MA5f", "content": [{"title": "Cross-Architecture Model Diffing With Crosscoders:", "url": "https://openreview.net/pdf/c8d3d5c96f4ca60c4261cde2c1f95486d816363c.pdf"}, {"title": "Cross-Architecture Model Diffing with Crosscoders", "url": "https://arxiv.org/pdf/2602.11729"}, {"title": "Route Sparse Autoencoder to Interpret Large Language ...", "url": "https://aclanthology.org/2025.emnlp-main.346.pdf"}, {"title": "Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs | OpenReview", "url": "https://openreview.net/forum?id=ZB84SvrZB8"}, {"title": "Safe-SAIL: Towards a Fine-grained Safety Landscape of ...", "url": "https://aclanthology.org/2026.findings-acl.944.pdf"}, {"title": "CROSS-ARCHITECTURE MODEL DIFFING WITH ...", "url": "https://openreview.net/pdf/50bd62fa1b359fcd6de0af7316bb3659a02a4e35.pdf"}, {"title": "SafeMT: Multi-turn Safety for Multimodal Language Models", "url": "https://arxiv.org/pdf/2510.12133"}, {"title": "CSR-Bench: A Benchmark for Evaluating the Cross-modal Safety and Reliability of MLLMs", "url": "https://arxiv.org/pdf/2602.03263"}, {"title": "Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes This paper contains text that might be offensive.", "url": "https://arxiv.org/html/2603.04426v1"}]}, "Based on the search results, here's what I found regarding crosscoder multi-model SAE research with focus on base vs. instruct models and safety applications:\n\n## Key Findings\n\n**Cross-Architecture Model Diffing with Crosscoders:**\n\nRecent work demonstrates the first model diff between architecturally distinct models: Llama-3.1-8B-Instruct and Qwen3-8B. As AI models proliferate with diverse architectures and training procedures, ensuring their safety requires understanding what changed between models: knowing which features were added or modified enables targeted safety audits rather than exhaustive analysis of every model from scratch.\n\n**Dedicated Feature Crosscoders (DFCs):**\n\nDedicated Feature Crosscoders (DFCs) are a simple architectural modification that encourages discovery of model-exclusive features by partitioning the feature dictionary. Two cross-architecture model diffs were conducted: Llama-3.1-8B-Instruct vs. Qwen3-8B, and GPT-OSS-20B vs. Deepseek-R1-0528-Qwen3-8B, using the BatchTopK sparsity penalty with k=200 and training on 100 million token-aligned activation pairs.\n\n**Safety Applications:**\n\nThe Safe-SAIL framework uses sparse autoencoders with varying sparsity levels and consists of three phases: SAE Training, Automated Interpretation, and Diagnose Toolkit, facilitating various case studies with acquired SAE checkpoints for LLMs in safety domains.\n\n**Feature Comparison Techniques:**\n\nAfter training, automated interpretability techniques are used to explain features and validated with activation steering."], "durationSeconds": 8.652110910999996, "searchCount": 1}
```

### [18] TOOL CALL — WebSearch · 2026-07-17 09:55:54 UTC

```
refusal direction abliteration orthogonal projection feature survival SAE feature-level analysis 2025 2026
```

### [19] TOOL CALL — WebFetch · 2026-07-17 09:55:54 UTC

```
URL: https://arxiv.org/abs/2603.04426
Prompt: What does Delta-Crosscoder do for model diffing? How does it compare base vs fine-tuned models for feature-level analysis of what changes during fine-tuning?
```

### [20] TOOL RESULT — WebFetch · 2026-07-17 09:55:58 UTC

```
{"bytes": 42296, "code": 200, "codeText": "OK", "result": "# Delta-Crosscoder: Model Diffing Approach\n\n**Purpose:**\nDelta-Crosscoder identifies how fine-tuning modifies a model's internal representations by learning shared dictionaries of interpretable latent directions between base and fine-tuned versions.\n\n**Key Technical Features:**\nThe method combines BatchTopK sparsity with a delta-based loss that emphasizes \"directions that change between models, plus an implicit contrastive signal from paired activations on matched inputs.\"\n\n**Comparative Advantage:**\nDelta-Crosscoder addresses a critical limitation in existing approaches—they struggle with narrow fine-tuning scenarios where behavioral changes are localized and asymmetric. This method reliably isolates latent directions causally responsible for fine-tuned behaviors while enabling effective mitigation strategies.\n\n**Evaluation Results:**\nTesting across multiple language models (Gemma, LLaMA, Qwen; 1B-9B parameters) and diverse tasks demonstrated that Delta-Crosscoder outperformed SAE-based baselines while matching non-SAE approaches in identifying feature-level changes during fine-tuning.", "durationMs": 4790, "url": "https://arxiv.org/abs/2603.04426"}
```

### [21] TOOL RESULT — WebSearch · 2026-07-17 09:56:00 UTC

```
{"query": "refusal direction abliteration orthogonal projection feature survival SAE feature-level analysis 2025 2026", "results": [{"tool_use_id": "srvtoolu_01Cax7cfSrnuMPZaQF9pBzHg", "content": [{"title": "Efficient Refusal Ablation in LLM through Optimal Transport", "url": "https://arxiv.org/html/2603.04355"}, {"title": "What is Abliteration - OBLITERATUS", "url": "https://mintlify.wiki/elder-plinius/OBLITERATUS/concepts/abliteration"}, {"title": "SAE features for refusal and sycophancy steering vectors", "url": "https://www.alignmentforum.org/posts/k8bBx4HcTF9iyikma/sae-features-for-refusal-and-sycophancy-steering-vectors"}, {"title": "CRaFT: Circuit-Guided Refusal Feature Selection via Cross-Layer Transcoders", "url": "https://arxiv.org/pdf/2604.01604"}, {"title": "Refusal Direction in LLM Safety", "url": "https://www.emergentmind.com/topics/refusal-direction"}, {"title": "Surgical Refusal Ablation: Disentangling Safety from Intelligence via Concept-Guided Spectral Cleaning", "url": "https://arxiv.org/html/2601.08489"}, {"title": "Refusal in LLMs is mediated by a single direction", "url": "https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction"}, {"title": "SAE Interventions are Unreliable: Post-Intervention Recovery of Suppressed Behavior", "url": "https://arxiv.org/pdf/2606.18322"}, {"title": "Understanding Refusal in Language Models with Sparse Autoencoders", "url": "https://arxiv.org/pdf/2505.23556"}]}, "Based on the web search results, here's what I found regarding refusal direction ablation, orthogonal projection, and feature-level analysis with SAEs from 2025-2026:\n\n## Key Research Areas\n\n**Refusal Feature Ablation (RFA) and Orthogonal Projection:**\nRefusal feature ablation computes the difference of means between harmful and harmless activations at each layer, then applies orthogonal projection to remove this direction across all layers. Recent activation-based jailbreaking methods circumvent safety mechanisms by applying orthogonal projections to remove refusal directions, but these approaches treat refusal as a one-dimensional phenomenon and ignore the rich distributional structure of model activations.\n\n**SAE Feature-Level Analysis:**\nRefusal-SAE identifies refusal-related sparse features and suppresses refusal by ablating them at inference time, first narrowing the candidate set using cosine similarity between SAE features and refusal-related activations, and then ranking candidate features using attribution scores computed with integrated gradients.\n\nCurrent approaches support SAE feature-level ablation as part of techniques that modify model weight matrices by projecting out refusal directions.\n\n**Feature Survival and Recovery:**\nRecent work (2026) has identified challenges with SAE interventions. Target information remains recoverable after official SAE ablation, suggesting that feature suppression through ablation may be less reliable than expected.\n\n**Refusal Direction Robustness:**\nAs refusal mechanisms are often shallow, brittle, and encoded in low dimensions, they are susceptible to targeted attacks—weight orthogonalization or probabilistic ablation can erase refusal with minimal effect on other capabilities."], "durationSeconds": 8.253328658, "searchCount": 1}
```

### [22] ASSISTANT · 2026-07-17 09:56:52 UTC

```
Now I have sufficient grounding to write the full review. Let me compose it.
```

### [23] TOOL CALL — WebFetch · 2026-07-17 09:56:52 UTC

```
URL: https://arxiv.org/abs/2505.23556
Prompt: What does this paper find about refusal in LLMs using sparse autoencoders? What SAE features are identified as refusal-related? Does it compare base vs instruct vs abliterated models?
```

### [24] TOOL RESULT — WebFetch · 2026-07-17 09:56:52 UTC

```
{"bytes": 41510, "code": 200, "codeText": "OK", "result": "# Analysis of the arXiv Paper on Refusal in LLMs\n\nBased on the abstract provided, here's what can be determined:\n\n## Key Findings\n\nThe research uses sparse autoencoders to conduct \"a mechanistic study of refusal in instruction-tuned LLMs\" by identifying latent features that causally influence refusal behaviors. The authors validate these features through interventions across harmful datasets.\n\n## Specific Features Identified\n\nThe abstract does not specify which particular SAE features are labeled as refusal-related. It only indicates that the study enables \"fine-grained inspection of how refusal manifests at the activation level.\"\n\n## Model Comparisons\n\nThe abstract states the method was \"applied to two open-source chat models,\" but does not explicitly compare base models versus instruction-tuned versus \"abliterated\" (jailbreak-resistant) variants. \n\nThe paper does address \"understanding the mechanisms of adversarial jailbreaking techniques,\" suggesting some analysis of how refusal features relate to jailbreak attempts, though the specific model types compared aren't detailed in this summary.\n\n## Note\n\nFull details about specific features and model comparisons would require accessing the complete paper PDF or HTML version linked on the arXiv page.", "durationMs": 5495, "url": "https://arxiv.org/abs/2505.23556"}
```

### [25] ASSISTANT · 2026-07-17 09:58:06 UTC

```
I have sufficient grounding to write the review now. All five previous critiques have been addressed. Let me identify remaining issues and produce the output.
```

### [26] TOOL CALL — Write · 2026-07-17 09:58:06 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_3/review_hypo/.terminal_claude_agent_struct_out.json

{
  "overall_assessment": "This is a substantially improved second iteration. All five critiques from the previous review have been adequately addressed: crosscoders are now properly cited with a clear justification of the frozen-base-SAE approach; layer selection is explicitly specified via Safety_change variance; the LLM semantic labeler is validated with a 20-feature gold set and a cross-family (non-Qwen) labeler; the JumpReLU threshold confound is explicitly documented with pre/post-threshold robustness reporting; and abliteration is proactively reproduced from scratch regardless of the community model's methodology. The hypothesis is now technically solid, well-motivated, clearly written, and adequately differentiated from prior work. Three minor issues remain: an unaddressed orthogonal-direction confound in the Reversal_score metric that could produce spurious Type-I classifications; a missing directly-relevant citation (arXiv:2505.23556) that partly covers SAE-based refusal feature identification; and an unsupported empirical claim about abliterated models 'consistently underperforming' base models. None of these are score-blocking for a proof-of-concept paper, but the orthogonal-direction confound is the most important to address as it affects interpretive validity. The study remains a single-model, single-layer proof-of-concept, which appropriately limits its venue target but does not undermine its methodological contribution.",
  "strengths": [
    "The three-model trajectory (base → instruct → abliterated) through a frozen SAE basis is a genuinely novel measurement framework that enables feature-level decomposition that geometric approaches (LatentBiopsy, arXiv:2603.27412, arXiv:2604.18901) cannot provide. The Reversal_score is a well-defined, non-circular metric for characterizing individual feature survival.",
    "Pre-registration of both bimodal and continuous outcomes is a major methodological strength — the hypothesis is not falsifiable only if bimodality is found, but has a clearly pre-specified continuous interpretation with its own quantitative criterion (≥2× enrichment rate). This is uncommon rigor for a proof-of-concept study.",
    "The SAE reconstruction quality gate with explicit fallback to encoder-only analysis, JumpReLU pre/post-threshold robustness check, and gold-set validation of the LLM labeler (kappa ≥ 0.6 threshold, cross-family labeler) reflect careful anticipation of known failure modes. Few hypothesis-stage proposals at this level address instrumentation reliability this explicitly.",
    "The crosscoder literature is now comprehensively engaged (arXiv:2602.11729, arXiv:2606.26474, justified as complementary rather than redundant) and the proactive scratch reproduction of abliteration removes a key methodological uncertainty that plagued the previous iteration.",
    "The disconfirmation criteria are specific and non-trivial, including an inconclusive-rather-than-disconfirmed outcome for the SAE quality gate failure case — this reflects epistemic honesty about the instrument's limitations.",
    "The worked numerical example of the Reversal_score computation (chemistry prompt scenario) substantially aids clarity for non-specialist readers."
  ],
  "dimension_scores": [
    {
      "dimension": "soundness",
      "score": 3,
      "justification": "The methodology is technically well-designed with appropriate fallbacks, pre-specified outcomes, and instrumentation checks. One remaining gap: the Reversal_score cannot distinguish between (a) genuinely harm-concept-imprinted features (Type-I as hypothesized) and (b) features that are simply orthogonal to the abliteration direction and thus trivially survive projection. This confound could inflate the Type-I count and weaken the mechanistic interpretation. The distance analysis (Criterion 2) and semantic labeling (Criterion 3) provide partial mitigation but do not directly address the geometric source of high R(f) scores.",
      "improvements": [
        "Add a geometric covariate: for each RLHF-sensitive feature f, compute cos(decoder_direction(f), abliteration_direction). Report this alongside R(f) for the top-100 features. Features with high R(f) but low cosine alignment with the abliteration direction are expected survivors (not truly 'imprinted'); features with non-trivial alignment that still survive are the genuinely interesting Type-I features. This analysis costs virtually nothing (one dot-product per feature) and directly addresses the most important remaining soundness gap.",
        "Specify the numerical Safety_change threshold for 'RLHF-sensitive': 'top-500 by Safety_change magnitude' leaves the absolute threshold implicit. For reproducibility, report the absolute Safety_change value at the 500th percentile cutoff in the supplementary materials."
      ]
    },
    {
      "dimension": "presentation",
      "score": 4,
      "justification": "Exceptionally clear for a hypothesis-stage document. The worked Reversal_score example, the explicit term glossary, the step-by-step investigation protocol with JumpReLU fallback documentation, and the pre-specified disconfirmation criteria all meet or exceed the presentation standards of accepted mechanistic interpretability papers. The analogies (isotopic tracing, concordia diagrams, implicit/explicit memory) are illuminating rather than decorative.",
      "improvements": [
        "Clarify whether 'LatentBiopsy' is the official paper title or an internal nickname for arXiv:2603.27412, whose actual title is 'The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams.' Using a nickname without clarification may confuse readers trying to locate the cited work."
      ]
    },
    {
      "dimension": "contribution",
      "score": 3,
      "justification": "The framework (frozen-SAE three-model trajectory + Reversal_score) is a genuine methodological contribution that advances beyond existing geometric (LatentBiopsy) and single-comparison (arXiv:2605.11426) approaches. The framing of Type-R vs. Type-I feature classes and their safety-restoration implications is novel. However, a directly relevant paper — 'Understanding Refusal in Language Models with Sparse Autoencoders' (arXiv:2505.23556) — is not cited; it identifies SAE features causally influencing refusal behavior and partially covers the Type-R feature territory. The Type-I imprinting claim remains novel (no existing work characterizes post-abliteration feature survival at the SAE level), but the hypothesis should position itself carefully against arXiv:2505.23556 to avoid appearing to claim novelty over an existing result.",
      "improvements": [
        "Add arXiv:2505.23556 ('Understanding Refusal in Language Models with Sparse Autoencoders') to related works. Distinguish: that paper identifies SAE features causally driving refusal in instruct models (Type-R territory) but does not perform the three-model comparison or Reversal_score analysis that reveals Type-I imprinting. This sharpens the novelty claim to focus on what survives abliteration rather than what drives refusal.",
        "Consider citing arXiv:2603.04426 (Delta-Crosscoder) which specifically addresses narrow fine-tuning regime model diffing and offers the strongest existing alternative to the frozen-base-SAE approach for the base→instruct comparison."
      ]
    }
  ],
  "critiques": [
    {
      "category": "methodology",
      "severity": "minor",
      "description": "The Reversal_score R(f) cannot distinguish between two different reasons for high R(f): (a) the feature genuinely carries harm-concept semantic content that RLHF imprinted and abliteration did not erase (the intended Type-I interpretation), and (b) the feature's decoder direction happens to be approximately orthogonal to the abliteration direction, so the orthogonal projection simply does not touch it — the feature 'survives' by geometric accident rather than semantic significance. Since abliteration is a single-direction projection, any feature whose direction is nearly orthogonal to the refusal direction will trivially have R(f) ≈ +1 regardless of its semantic content. Without controlling for this, the Type-I feature count may be substantially inflated by geometrically-orthogonal features unrelated to harm semantics.",
      "suggested_action": "Add a one-line geometric covariate to the Step 2 analysis: for each RLHF-sensitive feature f, compute alignment(f) = |cos(decoder_weight(f), abliteration_direction)|. Include this as a column in the feature analysis table. Report separately: (i) high-R(f) features with high alignment (genuinely imprinted — most interesting), (ii) high-R(f) features with low alignment (geometric survivors — expected null). This analysis is free (one dot-product per feature using the already-computed abliteration direction from Step 0) and directly addresses the most important remaining interpretability confound. If most Type-I features fall in category (ii), the semantic imprinting claim requires revisiting."
    },
    {
      "category": "novelty",
      "severity": "minor",
      "description": "'Understanding Refusal in Language Models with Sparse Autoencoders' (arXiv:2505.23556) directly analyzes refusal mechanisms using SAEs, identifying latent features that causally influence refusal behavior in instruction-tuned LLMs. This paper is not cited in the hypothesis and partially overlaps with what the hypothesis calls 'Type-R feature identification.' Failing to engage with this work creates two risks: (1) reviewers may perceive Type-R feature characterization as less novel than claimed; (2) the paper may contain findings about refusal-related SAE features that refine or complicate the hypothesis's predictions.",
      "suggested_action": "Add arXiv:2505.23556 to the related works section with the following positioning: 'arXiv:2505.23556 identifies SAE features causally driving refusal in instruct LLMs — this covers the Type-R feature territory in the base→instruct comparison. Our contribution is orthogonal: we introduce the abliterated model as a third reference point and the Reversal_score to characterize which RLHF-induced feature changes survive abliteration (Type-I), which no existing SAE-refusal work addresses.' Fetch the paper to confirm what models and feature types are covered before writing this positioning."
    },
    {
      "category": "novelty",
      "severity": "minor",
      "description": "Delta-Crosscoder (arXiv:2603.04426) is a directly competing methodology for model diffing in narrow fine-tuning regimes that is not cited. Unlike general crosscoders (arXiv:2602.11729), Delta-Crosscoder is specifically designed to isolate latent directions causally responsible for fine-tuned behaviors in cases where behavioral changes are localized and asymmetric — precisely the scenario of RLHF safety fine-tuning. It reportedly outperforms SAE-based baselines for feature-level change identification in fine-tuning scenarios.",
      "suggested_action": "Add Delta-Crosscoder (arXiv:2603.04426) to the crosscoder discussion in related works. The justification for preferring the frozen-base-SAE approach still holds (zero training cost, reuse of Qwen-Scope vocabulary), but Delta-Crosscoder is the most technically sophisticated alternative and its omission will be noticed by reviewers familiar with the model-diffing literature."
    },
    {
      "category": "scope",
      "severity": "minor",
      "description": "The motivation section states that Type-I features 'provide a novel mechanistic explanation for why abliterated models consistently underperform base models despite theoretically similar weight norms — a long-standing open question.' This claim contains two unsupported sub-claims: (1) that abliterated models 'consistently underperform' base models, and (2) that this is a 'long-standing open question.' In practice, many abliterated models perform comparably to base models on general benchmarks (MMLU, coding), and the underperformance — when it occurs — is often modest and inconsistently reported across model families.",
      "suggested_action": "Either (a) cite specific benchmark comparisons showing consistent underperformance of abliterated vs. base models (e.g., from the Surgical Refusal Ablation paper arXiv:2601.08489 or Arditi et al. 2024), or (b) qualify the claim: 'some abliterated models show modest capability regressions relative to base models; our Type-I feature fraction provides a mechanistic candidate explanation for this pattern in future multi-model studies.' The capability gap analysis is already labeled as Exploratory Criterion 4 (non-confirmatory), so the motivation section should not oversell it as resolving a 'long-standing open question.'"
    }
  ],
  "score": 7,
  "confidence": 4,
  "relation_type": "evolution",
  "relation_rationale": "Same three-model SAE imprinting frame; revised version adds crosscoder lit, layer selection, labeler validation, and JumpReLU fixes."
}
```

### [27] TOOL RESULT — Write · 2026-07-17 09:58:06 UTC

```
{"type": "create", "filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_3/review_hypo/.terminal_claude_agent_struct_out.json", "content": "{\n  \"overall_assessment\": \"This is a substantially improved second iteration. All five critiques from the previous review have been adequately addressed: crosscoders are now properly cited with a clear justification of the frozen-base-SAE approach; layer selection is explicitly specified via Safety_change variance; the LLM semantic labeler is validated with a 20-feature gold set and a cross-family (non-Qwen) labeler; the JumpReLU threshold confound is explicitly documented with pre/post-threshold robustness reporting; and abliteration is proactively reproduced from scratch regardless of the community model's methodology. The hypothesis is now technically solid, well-motivated, clearly written, and adequately differentiated from prior work. Three minor issues remain: an unaddressed orthogonal-direction confound in the Reversal_score metric that could produce spurious Type-I classifications; a missing directly-relevant citation (arXiv:2505.23556) that partly covers SAE-based refusal feature identification; and an unsupported empirical claim about abliterated models 'consistently underperforming' base models. None of these are score-blocking for a proof-of-concept paper, but the orthogonal-direction confound is the most important to address as it affects interpretive validity. The study remains a single-model, single-layer proof-of-concept, which appropriately limits its venue target but does not undermine its methodological contribution.\",\n  \"strengths\": [\n    \"The three-model trajectory (base → instruct → abliterated) through a frozen SAE basis is a genuinely novel measurement framework that enables feature-level decomposition that geometric approaches (LatentBiopsy, arXiv:2603.27412, arXiv:2604.18901) cannot provide. The Reversal_score is a well-defined, non-circular metric for characterizing individual feature survival.\",\n    \"Pre-registration of both bimodal and continuous outcomes is a major methodological strength — the hypothesis is not falsifiable only if bimodality is found, but has a clearly pre-specified continuous interpretation with its own quantitative criterion (≥2× enrichment rate). This is uncommon rigor for a proof-of-concept study.\",\n    \"The SAE reconstruction quality gate with explicit fallback to encoder-only analysis, JumpReLU pre/post-threshold robustness check, and gold-set validation of the LLM labeler (kappa ≥ 0.6 threshold, cross-family labeler) reflect careful anticipation of known failure modes. Few hypothesis-stage proposals at this level address instrumentation reliability this explicitly.\",\n    \"The crosscoder literature is now comprehensively engaged (arXiv:2602.11729, arXiv:2606.26474, justified as complementary rather than redundant) and the proactive scratch reproduction of abliteration removes a key methodological uncertainty that plagued the previous iteration.\",\n    \"The disconfirmation criteria are specific and non-trivial, including an inconclusive-rather-than-disconfirmed outcome for the SAE quality gate failure case — this reflects epistemic honesty about the instrument's limitations.\",\n    \"The worked numerical example of the Reversal_score computation (chemistry prompt scenario) substantially aids clarity for non-specialist readers.\"\n  ],\n  \"dimension_scores\": [\n    {\n      \"dimension\": \"soundness\",\n      \"score\": 3,\n      \"justification\": \"The methodology is technically well-designed with appropriate fallbacks, pre-specified outcomes, and instrumentation checks. One remaining gap: the Reversal_score cannot distinguish between (a) genuinely harm-concept-imprinted features (Type-I as hypothesized) and (b) features that are simply orthogonal to the abliteration direction and thus trivially survive projection. This confound could inflate the Type-I count and weaken the mechanistic interpretation. The distance analysis (Criterion 2) and semantic labeling (Criterion 3) provide partial mitigation but do not directly address the geometric source of high R(f) scores.\",\n      \"improvements\": [\n        \"Add a geometric covariate: for each RLHF-sensitive feature f, compute cos(decoder_direction(f), abliteration_direction). Report this alongside R(f) for the top-100 features. Features with high R(f) but low cosine alignment with the abliteration direction are expected survivors (not truly 'imprinted'); features with non-trivial alignment that still survive are the genuinely interesting Type-I features. This analysis costs virtually nothing (one dot-product per feature) and directly addresses the most important remaining soundness gap.\",\n        \"Specify the numerical Safety_change threshold for 'RLHF-sensitive': 'top-500 by Safety_change magnitude' leaves the absolute threshold implicit. For reproducibility, report the absolute Safety_change value at the 500th percentile cutoff in the supplementary materials.\"\n      ]\n    },\n    {\n      \"dimension\": \"presentation\",\n      \"score\": 4,\n      \"justification\": \"Exceptionally clear for a hypothesis-stage document. The worked Reversal_score example, the explicit term glossary, the step-by-step investigation protocol with JumpReLU fallback documentation, and the pre-specified disconfirmation criteria all meet or exceed the presentation standards of accepted mechanistic interpretability papers. The analogies (isotopic tracing, concordia diagrams, implicit/explicit memory) are illuminating rather than decorative.\",\n      \"improvements\": [\n        \"Clarify whether 'LatentBiopsy' is the official paper title or an internal nickname for arXiv:2603.27412, whose actual title is 'The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams.' Using a nickname without clarification may confuse readers trying to locate the cited work.\"\n      ]\n    },\n    {\n      \"dimension\": \"contribution\",\n      \"score\": 3,\n      \"justification\": \"The framework (frozen-SAE three-model trajectory + Reversal_score) is a genuine methodological contribution that advances beyond existing geometric (LatentBiopsy) and single-comparison (arXiv:2605.11426) approaches. The framing of Type-R vs. Type-I feature classes and their safety-restoration implications is novel. However, a directly relevant paper — 'Understanding Refusal in Language Models with Sparse Autoencoders' (arXiv:2505.23556) — is not cited; it identifies SAE features causally influencing refusal behavior and partially covers the Type-R feature territory. The Type-I imprinting claim remains novel (no existing work characterizes post-abliteration feature survival at the SAE level), but the hypothesis should position itself carefully against arXiv:2505.23556 to avoid appearing to claim novelty over an existing result.\",\n      \"improvements\": [\n        \"Add arXiv:2505.23556 ('Understanding Refusal in Language Models with Sparse Autoencoders') to related works. Distinguish: that paper identifies SAE features causally driving refusal in instruct models (Type-R territory) but does not perform the three-model comparison or Reversal_score analysis that reveals Type-I imprinting. This sharpens the novelty claim to focus on what survives abliteration rather than what drives refusal.\",\n        \"Consider citing arXiv:2603.04426 (Delta-Crosscoder) which specifically addresses narrow fine-tuning regime model diffing and offers the strongest existing alternative to the frozen-base-SAE approach for the base→instruct comparison.\"\n      ]\n    }\n  ],\n  \"critiques\": [\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"minor\",\n      \"description\": \"The Reversal_score R(f) cannot distinguish between two different reasons for high R(f): (a) the feature genuinely carries harm-concept semantic content that RLHF imprinted and abliteration did not erase (the intended Type-I interpretation), and (b) the feature's decoder direction happens to be approximately orthogonal to the abliteration direction, so the orthogonal projection simply does not touch it — the feature 'survives' by geometric accident rather than semantic significance. Since abliteration is a single-direction projection, any feature whose direction is nearly orthogonal to the refusal direction will trivially have R(f) ≈ +1 regardless of its semantic content. Without controlling for this, the Type-I feature count may be substantially inflated by geometrically-orthogonal features unrelated to harm semantics.\",\n      \"suggested_action\": \"Add a one-line geometric covariate to the Step 2 analysis: for each RLHF-sensitive feature f, compute alignment(f) = |cos(decoder_weight(f), abliteration_direction)|. Include this as a column in the feature analysis table. Report separately: (i) high-R(f) features with high alignment (genuinely imprinted — most interesting), (ii) high-R(f) features with low alignment (geometric survivors — expected null). This analysis is free (one dot-product per feature using the already-computed abliteration direction from Step 0) and directly addresses the most important remaining interpretability confound. If most Type-I features fall in category (ii), the semantic imprinting claim requires revisiting.\"\n    },\n    {\n      \"category\": \"novelty\",\n      \"severity\": \"minor\",\n      \"description\": \"'Understanding Refusal in Language Models with Sparse Autoencoders' (arXiv:2505.23556) directly analyzes refusal mechanisms using SAEs, identifying latent features that causally influence refusal behavior in instruction-tuned LLMs. This paper is not cited in the hypothesis and partially overlaps with what the hypothesis calls 'Type-R feature identification.' Failing to engage with this work creates two risks: (1) reviewers may perceive Type-R feature characterization as less novel than claimed; (2) the paper may contain findings about refusal-related SAE features that refine or complicate the hypothesis's predictions.\",\n      \"suggested_action\": \"Add arXiv:2505.23556 to the related works section with the following positioning: 'arXiv:2505.23556 identifies SAE features causally driving refusal in instruct LLMs — this covers the Type-R feature territory in the base→instruct comparison. Our contribution is orthogonal: we introduce the abliterated model as a third reference point and the Reversal_score to characterize which RLHF-induced feature changes survive abliteration (Type-I), which no existing SAE-refusal work addresses.' Fetch the paper to confirm what models and feature types are covered before writing this positioning.\"\n    },\n    {\n      \"category\": \"novelty\",\n      \"severity\": \"minor\",\n      \"description\": \"Delta-Crosscoder (arXiv:2603.04426) is a directly competing methodology for model diffing in narrow fine-tuning regimes that is not cited. Unlike general crosscoders (arXiv:2602.11729), Delta-Crosscoder is specifically designed to isolate latent directions causally responsible for fine-tuned behaviors in cases where behavioral changes are localized and asymmetric — precisely the scenario of RLHF safety fine-tuning. It reportedly outperforms SAE-based baselines for feature-level change identification in fine-tuning scenarios.\",\n      \"suggested_action\": \"Add Delta-Crosscoder (arXiv:2603.04426) to the crosscoder discussion in related works. The justification for preferring the frozen-base-SAE approach still holds (zero training cost, reuse of Qwen-Scope vocabulary), but Delta-Crosscoder is the most technically sophisticated alternative and its omission will be noticed by reviewers familiar with the model-diffing literature.\"\n    },\n    {\n      \"category\": \"scope\",\n      \"severity\": \"minor\",\n      \"description\": \"The motivation section states that Type-I features 'provide a novel mechanistic explanation for why abliterated models consistently underperform base models despite theoretically similar weight norms — a long-standing open question.' This claim contains two unsupported sub-claims: (1) that abliterated models 'consistently underperform' base models, and (2) that this is a 'long-standing open question.' In practice, many abliterated models perform comparably to base models on general benchmarks (MMLU, coding), and the underperformance — when it occurs — is often modest and inconsistently reported across model families.\",\n      \"suggested_action\": \"Either (a) cite specific benchmark comparisons showing consistent underperformance of abliterated vs. base models (e.g., from the Surgical Refusal Ablation paper arXiv:2601.08489 or Arditi et al. 2024), or (b) qualify the claim: 'some abliterated models show modest capability regressions relative to base models; our Type-I feature fraction provides a mechanistic candidate explanation for this pattern in future multi-model studies.' The capability gap analysis is already labeled as Exploratory Criterion 4 (non-confirmatory), so the motivation section should not oversell it as resolving a 'long-standing open question.'\"\n    }\n  ],\n  \"score\": 7,\n  \"confidence\": 4,\n  \"relation_type\": \"evolution\",\n  \"relation_rationale\": \"Same three-model SAE imprinting frame; revised version adds crosscoder lit, layer selection, labeler validation, and JumpReLU fixes.\"\n}\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [28] ASSISTANT · 2026-07-17 09:58:15 UTC

```
Review written to `.terminal_claude_agent_struct_out.json`.

**Summary:** Overall score **7/Accept** (up from the previous iteration). All five prior critiques are fully addressed. Three new minor issues remain:

1. **Orthogonal-direction confound** (methodology, minor) — features geometrically orthogonal to the abliteration direction trivially have R(f) ≈ +1 regardless of semantic content, inflating the Type-I count. Fix: compute `cos(decoder_weight(f), abliteration_direction)` as a free covariate in Step 2.

2. **Missing arXiv:2505.23556** (novelty, minor) — "Understanding Refusal in LLMs with Sparse Autoencoders" directly covers Type-R feature territory and needs positioning in related works.

3. **Missing Delta-Crosscoder arXiv:2603.04426** (novelty, minor) — the most technically relevant competing method for narrow fine-tuning model diffing, more specific than the general crosscoders already cited.

4. **Unsupported capability-gap claim** (scope, minor) — "consistently underperform" needs a citation or qualification since it's not universally established.

The hypothesis is now technically solid and ready for experimentation. The orthogonal-direction covariate is the highest-priority fix as it costs nothing and directly strengthens the mechanistic interpretation.
```

### [29] SYSTEM-USER prompt · 2026-07-17 09:58:16 UTC

```
<validation-feedback>
Attempt 1 failed validation.

Schema validation found 1 problem — fix ALL of them at once:
  - at `relation_rationale`: 'Same three-model SAE imprinting frame; revised version adds crosscoder lit, layer selection, labeler validation, and JumpReLU fixes.' is too long (at most 120 characters, got 132)
Every required field must be present and every field type must match the schema.

Please use the Write tool to overwrite `.terminal_claude_agent_struct_out.json` with corrected JSON. Do not invent new fields; match the schema you were given.
</validation-feedback>
```

### [30] TOOL CALL — Edit · 2026-07-17 09:58:20 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_3/review_hypo/.terminal_claude_agent_struct_out.json
OLD:   "relation_rationale": "Same three-model SAE imprinting frame; revised version adds crosscoder lit, layer selection, labeler validation, and JumpReLU fixes."
NEW:   "relation_rationale": "Same three-model SAE imprinting frame; adds crosscoder lit, layer selection, labeler validation, JumpReLU fix."
```

### [31] TOOL RESULT — Edit · 2026-07-17 09:58:20 UTC

```
{"filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_3/review_hypo/.terminal_claude_agent_struct_out.json", "oldString": "  \"relation_rationale\": \"Same three-model SAE imprinting frame; revised version adds crosscoder lit, layer selection, labeler validation, and JumpReLU fixes.\"", "newString": "  \"relation_rationale\": \"Same three-model SAE imprinting frame; adds crosscoder lit, layer selection, labeler validation, JumpReLU fix.\"", "originalFile": null, "structuredPatch": [{"oldStart": 65, "oldLines": 5, "newStart": 65, "newLines": 5, "lines": ["   \"score\": 7,", "   \"confidence\": 4,", "   \"relation_type\": \"evolution\",", "-  \"relation_rationale\": \"Same three-model SAE imprinting frame; revised version adds crosscoder lit, layer selection, labeler validation, and JumpReLU fixes.\"", "+  \"relation_rationale\": \"Same three-model SAE imprinting frame; adds crosscoder lit, layer selection, labeler validation, JumpReLU fix.\"", " }"]}], "userModified": false, "replaceAll": false}
```

### [32] ASSISTANT · 2026-07-17 09:58:22 UTC

```
Fixed — `relation_rationale` shortened to 109 characters.
```
