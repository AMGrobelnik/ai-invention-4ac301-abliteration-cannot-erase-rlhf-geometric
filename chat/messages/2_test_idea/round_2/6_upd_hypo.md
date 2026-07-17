# upd_hypo — test_idea

> Phase: `invention_loop` · round 2 · `upd_hypo`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim transcript of this agent task — every system/user prompt, assistant response, thinking block, tool call and tool result — in the order they occurred. Nothing truncated.

## Task: `upd_hypo` (terminal_claude_agent, claude-sonnet-4-6)

### [1] CONFIG · 2026-07-17 12:24:22 UTC

```
model: claude-sonnet-4-6 | effort: medium | permission: bypassPermissions | cwd: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/upd_hypo/upd_hypo
```

### [2] SYSTEM-USER prompt · 2026-07-17 12:24:28 UTC

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
title: Latent Safety Imprinting in RLHF
hypothesis: |-
  Safety RLHF fine-tuning modifies Qwen3-8B's internal representations in two mechanistically distinct ways, distinguishable only when comparing all three model variants (base, instruct, abliterated) through a fixed Sparse Autoencoder (SAE) basis. The first type — 'Refusal-Coupled' (Type-R) features — are reversed when abliteration is applied; they implement the explicit refusal machinery and their activation pattern in the abliterated model reverts close to the base model's. The second type — 'Semantically-Imprinted' (Type-I) features — represent the model's understanding of harmful content and persist structurally unchanged after abliteration; the abliterated model's activations on these features remain closer to the safety-tuned instruct model than to the base model. This creates a 'latent safety imprint': despite identical compliance behavior to the base model on harmful prompts, the abliterated model retains the instruct model's representational fingerprint on harm-concept features in the SAE basis.

  Whether the Type-R/Type-I distinction is sharply bimodal or represents a continuum is itself an empirical question whose answer reveals the underlying mechanism: a clean bimodal split suggests two categorically distinct machinery sets, while a continuous distribution suggests a single graded mechanism where each RLHF-modified feature is partially reversed proportional to its alignment with the refusal direction (motivated by the polysemanticity of the refusal vector documented in arXiv:2601.08489). In either case, the fraction of RLHF-induced feature change that survives abliteration is predicted to be semantically coherent and harm-concept-related.

  Four critical methodological refinements are incorporated from preliminary investigation: (1) Qwen-Scope SAEs use **TopK** architecture (k=50 or k=100 features retained via torch.topk), not JumpReLU — the relevant robustness check is pre-topk (raw encoder: residual @ W_enc.T + b_enc) vs. post-topk activations, not a threshold-crossing confound; (2) because TopK SAEs produce sparse vectors with most entries near zero, Reversal_score reliability requires an **activation-density filter**: only features where both Δ_inst(f) and Δ_abl(f) have ≥10% non-zero entries (≥50 of 500 prompts) are analyzed — the fraction of top-500 RLHF-sensitive features surviving this filter is itself a diagnostic; (3) the Type-I classification threshold R(f) > T is derived empirically from the Reversal_score distribution (top quartile of the filtered feature set) rather than hardcoded at +0.5, with robustness checks at fixed thresholds {0.3, 0.5, 0.7}; (4) the analysis layer is selected by empirically computing Safety_change variance across all 36 Qwen3-8B layers on 50 harmful prompts, with Chopra (arXiv:2605.11426) on Gemma 3 1B cited only as prior motivation for expecting non-uniform layer sensitivity, not as a direct layer specification.

  A fifth methodological requirement is community model verification: the primary abliterated checkpoint (huihui-ai/Huihui-Qwen3-8B-abliterated-v2) must be compared against the scratch-reproduced checkpoint on geometric metrics (AUROC for harmful-prompt detection, KL divergence on Wikitext-2, refusal rate on held-out prompts) before being used as a co-equal analysis target. If the two checkpoints diverge substantially in abliteration depth, the scratch-reproduced checkpoint becomes the sole primary target and the community model a secondary validation object.
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
  Same conceptual frame; refines implementation: TopK fix, density filter, empirical layer/threshold, model verification.
_confidence_delta: unchanged
_key_changes:
- >-
  Replaced all JumpReLU references with TopK architecture (k=50 or k=100 via torch.topk); pre-topk vs. post-topk robustness
  check replaces the JumpReLU threshold-crossing confound framing.
- >-
  Added activation-density filter as a required pre-processing step for Reversal_score: only features with ≥10% non-zero entries
  in both Δ_inst(f) and Δ_abl(f) (≥50/500 prompts) are analyzed, addressing the zero-dominated vector reliability problem
  with sparse TopK activations.
- >-
  Changed Type-I threshold from hardcoded R(f) > +0.5 to empirically derived top-quartile of the filtered Reversal_score distribution,
  with {0.3, 0.5, 0.7} robustness checks — prevents threshold arbitrariness from driving Criterion 3 statistics.
- >-
  Specified that analysis layer must be selected by empirically computing Safety_change variance across all 36 Qwen3-8B residual-stream
  layers; Chopra (Gemma 3 1B) findings cited only as motivation, not as a Qwen3-8B layer specification.
- >-
  Added community model verification step: huihui-ai/v2 must be compared against scratch-reproduced checkpoint on AUROC, KL-divergence,
  and refusal-rate metrics before being treated as a co-equal primary target; divergence demotes community model to secondary
  validation.
- >-
  Noted that the prompt corpus exhibits a count discrepancy (full_data_out.json: 468 examples vs. claimed 1000 rows in results/data_out.json);
  next iteration must reconcile and report actual per-source counts and final balanced totals explicitly.
- >-
  Retained polysemanticity of the refusal vector (arXiv:2601.08489) as explicit mechanistic grounding for the continuous-distribution
  interpretation of Reversal_score.
- >-
  Removed the 'TopK architecture correction' from the contribution list (it is a technical discovery about the Qwen-Scope
  release, not a methodological innovation); it now belongs in the Methodology section as a clarification.
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
- id: art_pmAcY0CNazP9
  label: tech reference
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
</all_artifacts>

<new_artifacts_this_iteration>
These 1 artifacts were created THIS iteration.

id: art_QBm3I0Z5VVJR
type: experiment
in_dependencies:
- id: art_M8-flaZ0OOaP
  label: prompt corpus
- id: art_pmAcY0CNazP9
  label: tech reference
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
</new_artifacts_this_iteration>

<current_paper>
The paper draft from this iteration — represents the current state of the research story.

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
Feedback from the paper reviewer this iteration.

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

### [3] HUMAN-USER prompt · 2026-07-17 12:24:28 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [4] ASSISTANT · 2026-07-17 12:25:44 UTC

```
Reading the experimental results and reviewer feedback carefully before writing the revised hypothesis.
```

### [5] TOOL CALL — Write · 2026-07-17 12:25:44 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/upd_hypo/upd_hypo/.terminal_claude_agent_struct_out.json

{
  "title": "Abliteration Preserves All RLHF-Modified SAE Features",
  "hypothesis": "RLHF safety fine-tuning and refusal abliteration operate on geometrically near-orthogonal subspaces in Qwen3-8B's residual stream. Contrary to the original bimodal prediction, the Reversal Score distribution for the top-500 RLHF-sensitive SAE encoder features at layer 22 is unimodal and concentrated near R=1 (mean=0.921, std=0.040, min=0.684), with zero features having negative Reversal Scores across all thresholds {0.30, 0.50, 0.70, 0.95}. The original Type-R/Type-I partition does not arise empirically: abliteration preserves every RLHF-modified feature, not a semantically selected minority.\n\nThis finding requires a critical null-baseline validation before the geometric-orthogonality interpretation can be accepted. If abliteration modifies only a few of 4096 residual-stream directions, the Reversal Score for randomly sampled features — not modified by RLHF — could also be near 1.0 simply because instruct and abliterated models are globally similar in most directions. The primary experimental requirement for the next iteration is therefore: compute Reversal Scores for the bottom-500 SAE encoder features by Safety_change (features least modified by RLHF) and compare their distribution against the top-500. If both distributions concentrate near R=1, then R≈0.921 for RLHF-sensitive features reflects global model similarity rather than selective preservation — and the geometric-orthogonality claim is undermined. If the RLHF-sensitive features show significantly higher R than the control features, selective preservation is confirmed.\n\nA second direct test of the geometric-orthogonality claim is required: compute the refusal direction r (mean difference of instruct residual-stream activations on harmful vs. harmless prompts at layer 22, following Arditi et al.) and measure cosine similarity between r and each of the top-500 RLHF-sensitive SAE encoder directions (W_enc[feature_idx, :] projected into residual-stream space via W_dec[:, feature_idx]). If these cosine similarities are near zero (|cos| < 0.1), this directly confirms that abliteration — which orthogonalizes against r — cannot substantially affect any of the RLHF-modified feature directions. This would be a mechanistic account of why all Reversal Scores are positive: the refusal direction and the RLHF-modified encoder directions are nearly orthogonal by construction.\n\nCentroid analysis provides confirmatory evidence independent of the null-baseline concern: the abliterated model's SAE encoder activations in the RLHF-sensitive subspace are 6.1× closer (cosine) to the instruct model than to the base model for harmful prompts. However, the benign-prompt ratio (17.6×) is larger than the harmful-prompt ratio (6.1×), which is consistent with a global-similarity alternative rather than selective preservation of harm-concept representations. A control subspace comparison is needed: compute centroid distances in the bottom-500 (RLHF-insensitive) subspace and in a random 500-feature subspace. If those control ratios are comparable to 6.1–17.6×, the centroid finding also reflects global similarity rather than harm-specific preservation.\n\nSemantic labeling finds approximately 75% of top-500 RLHF-sensitive features carry harm-concept labels uniformly across the Reversal Score range (Q3/Q1 Fisher p=0.675, n=4 in high group — uninterpretably underpowered). The meaningful conclusion is that harm-concept representations are not concentrated in a surviving minority but pervade the entire RLHF-modified feature set. However, all semantic claims rest on pre-topk (dense) encoder projections (z_pre = h @ W_enc.T + b_enc) rather than post-topk sparse activations, because the base-SAE reconstruction gate fails (R²≈0.51 at layer 22 for all three variants). Pre-topk projections are dense continuous values — every feature has a non-zero value on every prompt by construction — so the 10% activation-density filter is trivially satisfied and provides no coverage information. Future iterations must either (a) obtain an instruct-trained Qwen3-8B SAE covering layers above 8 to enable post-topk analysis, or (b) validate that pre-topk encoder directions carry comparable semantic signal to post-topk sparse features before trusting the ≈75% harm-concept rate.\n\nA downstream classification test (per-prompt harm score = mean Type-I activation minus mean Type-R activation, median threshold) achieved 0.476 accuracy on the balanced 500/500 corpus — near-chance and below the 0.630 keyword baseline. This is expected given zero Type-R features: with no reversal-defined contrast, the score carries no discrimination signal. This null classification result is a direct consequence of the primary finding, not a separate failure, and should be reported transparently.\n\nFour specific methodological requirements for the next iteration follow from reviewer critiques: (1) Null-baseline Reversal Score comparison: bottom-500 features by Safety_change, reported alongside top-500. (2) Direct refusal-direction orthogonality measurement: cosine similarity between r and top-10 RLHF-sensitive encoder directions. (3) Control-subspace centroid analysis: distances in RLHF-insensitive and random subspaces alongside the RLHF-sensitive subspace. (4) Corrected semantic-labeling description: the actual protocol identifies the top-5 prompts by maximum pre-topk activation and provides the first 300 characters of each to the cross-family labeler — not a ±32 token window, which was not implemented. Layer-20 sensitivity results must be reported numerically (not asserted verbally) given the low reported runtime (4.3 minutes total).\n\nWhether the geometric-orthogonality interpretation survives the null-baseline test, the centroid finding (6.1× closer to instruct than to base for harmful prompts) is already confirmed at the pre-topk encoder level and is consistent with the latent-safety-imprint framing: the abliterated model retains the instruct model's representational fingerprint in the RLHF-sensitive feature subspace despite behaving like the base model on harmful prompts. The open question is whether this reflects selective preservation of a semantically coherent harm-concept representation, or a more general consequence of abliteration modifying only a narrow geometric slice of a 4096-dimensional space.",
  "relation_rationale": "Same conceptual frame (latent safety imprint via SAE Reversal Score); empirical result replaced bimodal prediction with total-preservation finding requiring null-baseline validation.",
  "confidence_delta": "decreased",
  "key_changes": [
    "Core empirical finding reframed: no Type-R/Type-I bimodal split exists; all 500 top RLHF-sensitive features have positive Reversal Scores (mean=0.921), so abliteration preserves the entire RLHF-modified feature set at layer 22.",
    "Geometric-orthogonality interpretation introduced as the mechanistic explanation for zero Type-R features: the refusal direction r is near-orthogonal to RLHF-modified SAE encoder directions, so removing r does not disturb those features.",
    "Critical null-baseline requirement added: bottom-500 features by Safety_change must be analyzed to determine whether R≈0.921 reflects selective preservation or global instruct/abliterated similarity.",
    "Direct refusal-direction measurement added as a required next step: cosine similarity between r and top-10 RLHF-sensitive encoder directions (W_enc projected via W_dec) to confirm orthogonality directly.",
    "Control-subspace centroid analysis added: benign-prompt ratio (17.6×) > harmful-prompt ratio (6.1×) raises global-similarity concern; centroid distances in RLHF-insensitive and random subspaces needed as controls.",
    "Pre-topk vs post-topk distinction clarified: all feature metrics use dense encoder projections (z_pre), not sparse post-topk activations; the density filter is trivially satisfied and provides no coverage information; semantic claims require validation.",
    "Near-chance classification accuracy (0.476) explicitly incorporated as an expected consequence of zero Type-R features — no discrimination signal when all features have high Reversal Scores.",
    "Corrected semantic-labeling protocol: top-5 prompts by max pre-topk activation, first 300 characters — not a ±32 token window as previously stated.",
    "Layer-20 numerical results required in next iteration (not verbal assertion of consistency).",
    "Underpowered Fisher fallback test (n=4 high group) dropped as a quantitative claim; replaced with qualitative conclusion: harm-concept labels uniformly distributed across Reversal Score range."
  ],
  "relation_type": "evolution",
  "artifact_relations": [
    {
      "from_id": "art_M8-flaZ0OOaP",
      "to_id": "art_QBm3I0Z5VVJR",
      "relation_type": "uses",
      "relation_rationale": "Experiment uses the 500 harmful + 500 benign prompt corpus as the evaluation set for SAE activation extraction."
    },
    {
      "from_id": "art_pmAcY0CNazP9",
      "to_id": "art_QBm3I0Z5VVJR",
      "relation_type": "uses",
      "relation_rationale": "Experiment uses the tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations."
    }
  ]
}
```

### [6] TOOL RESULT — Write · 2026-07-17 12:25:44 UTC

```
{"type": "create", "filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/upd_hypo/upd_hypo/.terminal_claude_agent_struct_out.json", "content": "{\n  \"title\": \"Abliteration Preserves All RLHF-Modified SAE Features\",\n  \"hypothesis\": \"RLHF safety fine-tuning and refusal abliteration operate on geometrically near-orthogonal subspaces in Qwen3-8B's residual stream. Contrary to the original bimodal prediction, the Reversal Score distribution for the top-500 RLHF-sensitive SAE encoder features at layer 22 is unimodal and concentrated near R=1 (mean=0.921, std=0.040, min=0.684), with zero features having negative Reversal Scores across all thresholds {0.30, 0.50, 0.70, 0.95}. The original Type-R/Type-I partition does not arise empirically: abliteration preserves every RLHF-modified feature, not a semantically selected minority.\\n\\nThis finding requires a critical null-baseline validation before the geometric-orthogonality interpretation can be accepted. If abliteration modifies only a few of 4096 residual-stream directions, the Reversal Score for randomly sampled features — not modified by RLHF — could also be near 1.0 simply because instruct and abliterated models are globally similar in most directions. The primary experimental requirement for the next iteration is therefore: compute Reversal Scores for the bottom-500 SAE encoder features by Safety_change (features least modified by RLHF) and compare their distribution against the top-500. If both distributions concentrate near R=1, then R≈0.921 for RLHF-sensitive features reflects global model similarity rather than selective preservation — and the geometric-orthogonality claim is undermined. If the RLHF-sensitive features show significantly higher R than the control features, selective preservation is confirmed.\\n\\nA second direct test of the geometric-orthogonality claim is required: compute the refusal direction r (mean difference of instruct residual-stream activations on harmful vs. harmless prompts at layer 22, following Arditi et al.) and measure cosine similarity between r and each of the top-500 RLHF-sensitive SAE encoder directions (W_enc[feature_idx, :] projected into residual-stream space via W_dec[:, feature_idx]). If these cosine similarities are near zero (|cos| < 0.1), this directly confirms that abliteration — which orthogonalizes against r — cannot substantially affect any of the RLHF-modified feature directions. This would be a mechanistic account of why all Reversal Scores are positive: the refusal direction and the RLHF-modified encoder directions are nearly orthogonal by construction.\\n\\nCentroid analysis provides confirmatory evidence independent of the null-baseline concern: the abliterated model's SAE encoder activations in the RLHF-sensitive subspace are 6.1× closer (cosine) to the instruct model than to the base model for harmful prompts. However, the benign-prompt ratio (17.6×) is larger than the harmful-prompt ratio (6.1×), which is consistent with a global-similarity alternative rather than selective preservation of harm-concept representations. A control subspace comparison is needed: compute centroid distances in the bottom-500 (RLHF-insensitive) subspace and in a random 500-feature subspace. If those control ratios are comparable to 6.1–17.6×, the centroid finding also reflects global similarity rather than harm-specific preservation.\\n\\nSemantic labeling finds approximately 75% of top-500 RLHF-sensitive features carry harm-concept labels uniformly across the Reversal Score range (Q3/Q1 Fisher p=0.675, n=4 in high group — uninterpretably underpowered). The meaningful conclusion is that harm-concept representations are not concentrated in a surviving minority but pervade the entire RLHF-modified feature set. However, all semantic claims rest on pre-topk (dense) encoder projections (z_pre = h @ W_enc.T + b_enc) rather than post-topk sparse activations, because the base-SAE reconstruction gate fails (R²≈0.51 at layer 22 for all three variants). Pre-topk projections are dense continuous values — every feature has a non-zero value on every prompt by construction — so the 10% activation-density filter is trivially satisfied and provides no coverage information. Future iterations must either (a) obtain an instruct-trained Qwen3-8B SAE covering layers above 8 to enable post-topk analysis, or (b) validate that pre-topk encoder directions carry comparable semantic signal to post-topk sparse features before trusting the ≈75% harm-concept rate.\\n\\nA downstream classification test (per-prompt harm score = mean Type-I activation minus mean Type-R activation, median threshold) achieved 0.476 accuracy on the balanced 500/500 corpus — near-chance and below the 0.630 keyword baseline. This is expected given zero Type-R features: with no reversal-defined contrast, the score carries no discrimination signal. This null classification result is a direct consequence of the primary finding, not a separate failure, and should be reported transparently.\\n\\nFour specific methodological requirements for the next iteration follow from reviewer critiques: (1) Null-baseline Reversal Score comparison: bottom-500 features by Safety_change, reported alongside top-500. (2) Direct refusal-direction orthogonality measurement: cosine similarity between r and top-10 RLHF-sensitive encoder directions. (3) Control-subspace centroid analysis: distances in RLHF-insensitive and random subspaces alongside the RLHF-sensitive subspace. (4) Corrected semantic-labeling description: the actual protocol identifies the top-5 prompts by maximum pre-topk activation and provides the first 300 characters of each to the cross-family labeler — not a ±32 token window, which was not implemented. Layer-20 sensitivity results must be reported numerically (not asserted verbally) given the low reported runtime (4.3 minutes total).\\n\\nWhether the geometric-orthogonality interpretation survives the null-baseline test, the centroid finding (6.1× closer to instruct than to base for harmful prompts) is already confirmed at the pre-topk encoder level and is consistent with the latent-safety-imprint framing: the abliterated model retains the instruct model's representational fingerprint in the RLHF-sensitive feature subspace despite behaving like the base model on harmful prompts. The open question is whether this reflects selective preservation of a semantically coherent harm-concept representation, or a more general consequence of abliteration modifying only a narrow geometric slice of a 4096-dimensional space.\",\n  \"relation_rationale\": \"Same conceptual frame (latent safety imprint via SAE Reversal Score); empirical result replaced bimodal prediction with total-preservation finding requiring null-baseline validation.\",\n  \"confidence_delta\": \"decreased\",\n  \"key_changes\": [\n    \"Core empirical finding reframed: no Type-R/Type-I bimodal split exists; all 500 top RLHF-sensitive features have positive Reversal Scores (mean=0.921), so abliteration preserves the entire RLHF-modified feature set at layer 22.\",\n    \"Geometric-orthogonality interpretation introduced as the mechanistic explanation for zero Type-R features: the refusal direction r is near-orthogonal to RLHF-modified SAE encoder directions, so removing r does not disturb those features.\",\n    \"Critical null-baseline requirement added: bottom-500 features by Safety_change must be analyzed to determine whether R≈0.921 reflects selective preservation or global instruct/abliterated similarity.\",\n    \"Direct refusal-direction measurement added as a required next step: cosine similarity between r and top-10 RLHF-sensitive encoder directions (W_enc projected via W_dec) to confirm orthogonality directly.\",\n    \"Control-subspace centroid analysis added: benign-prompt ratio (17.6×) > harmful-prompt ratio (6.1×) raises global-similarity concern; centroid distances in RLHF-insensitive and random subspaces needed as controls.\",\n    \"Pre-topk vs post-topk distinction clarified: all feature metrics use dense encoder projections (z_pre), not sparse post-topk activations; the density filter is trivially satisfied and provides no coverage information; semantic claims require validation.\",\n    \"Near-chance classification accuracy (0.476) explicitly incorporated as an expected consequence of zero Type-R features — no discrimination signal when all features have high Reversal Scores.\",\n    \"Corrected semantic-labeling protocol: top-5 prompts by max pre-topk activation, first 300 characters — not a ±32 token window as previously stated.\",\n    \"Layer-20 numerical results required in next iteration (not verbal assertion of consistency).\",\n    \"Underpowered Fisher fallback test (n=4 high group) dropped as a quantitative claim; replaced with qualitative conclusion: harm-concept labels uniformly distributed across Reversal Score range.\"\n  ],\n  \"relation_type\": \"evolution\",\n  \"artifact_relations\": [\n    {\n      \"from_id\": \"art_M8-flaZ0OOaP\",\n      \"to_id\": \"art_QBm3I0Z5VVJR\",\n      \"relation_type\": \"uses\",\n      \"relation_rationale\": \"Experiment uses the 500 harmful + 500 benign prompt corpus as the evaluation set for SAE activation extraction.\"\n    },\n    {\n      \"from_id\": \"art_pmAcY0CNazP9\",\n      \"to_id\": \"art_QBm3I0Z5VVJR\",\n      \"relation_type\": \"uses\",\n      \"relation_rationale\": \"Experiment uses the tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations.\"\n    }\n  ]\n}\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [7] ASSISTANT · 2026-07-17 12:25:49 UTC

```
The revised hypothesis is written to `.terminal_claude_agent_struct_out.json`. Key changes from the experimental results:

**Core shift:** The original bimodal Type-R/Type-I prediction is disconfirmed — all 500 RLHF-sensitive features have positive Reversal Scores (mean=0.921, zero Type-R). The hypothesis now frames this as geometric orthogonality between the refusal direction and RLHF-modified SAE encoder directions.

**Critical open questions added for next iteration:**
1. **Null baseline** (MAJOR reviewer concern): Bottom-500 features by Safety_change must be compared — R≈0.921 may reflect global instruct/abliterated similarity rather than selective preservation
2. **Direct orthogonality measurement**: Cosine similarity between refusal direction `r` and top-10 RLHF-sensitive encoder directions needs to be computed directly
3. **Control-subspace centroid**: The benign imprint (17.6×) being larger than the harmful imprint (6.1×) is a red flag for the global-similarity alternative

**Methodological corrections incorporated:** Pre-topk density filter trivially satisfied (dropped as coverage evidence), near-chance classification accuracy (0.476) explicitly stated, semantic-labeling protocol corrected to match actual code, Fisher fallback test dropped as underpowered.
```

### [8] SYSTEM-USER prompt · 2026-07-17 12:25:52 UTC

```
<validation-feedback>
Attempt 1 failed validation.

Schema validation found 2 problems — fix ALL of them at once:
  - at `artifact_relations.1.relation_rationale`: 'Experiment uses the tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations.' is too long (at most 120 characters, got 125)
  - at `relation_rationale`: 'Same conceptual frame (latent safety imprint via SAE Reversal Score); empirical result replaced bimodal prediction with total-preservation finding requiring null-baseline validation.' is too long (at most 120 characters, got 182)
Every required field must be present and every field type must match the schema.

Please use the Write tool to overwrite `.terminal_claude_agent_struct_out.json` with corrected JSON. Do not invent new fields; match the schema you were given.
</validation-feedback>
```

### [9] TOOL CALL — Edit · 2026-07-17 12:25:54 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/upd_hypo/upd_hypo/.terminal_claude_agent_struct_out.json
OLD:   "relation_rationale": "Same conceptual frame (latent safety imprint via SAE Reversal Score); empirical result replaced bimodal prediction with total-preservation finding requiring null-baseline validation.",
NEW:   "relation_rationale": "Same frame; bimodal prediction replaced by total-preservation finding requiring null-baseline validation.",
```

### [10] TOOL RESULT — Edit · 2026-07-17 12:25:54 UTC

```
{"filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/upd_hypo/upd_hypo/.terminal_claude_agent_struct_out.json", "oldString": "  \"relation_rationale\": \"Same conceptual frame (latent safety imprint via SAE Reversal Score); empirical result replaced bimodal prediction with total-preservation finding requiring null-baseline validation.\",", "newString": "  \"relation_rationale\": \"Same frame; bimodal prediction replaced by total-preservation finding requiring null-baseline validation.\",", "originalFile": "{\n  \"title\": \"Abliteration Preserves All RLHF-Modified SAE Features\",\n  \"hypothesis\": \"RLHF safety fine-tuning and refusal abliteration operate on geometrically near-orthogonal subspaces in Qwen3-8B's residual stream. Contrary to the original bimodal prediction, the Reversal Score distribution for the top-500 RLHF-sensitive SAE encoder features at layer 22 is unimodal and concentrated near R=1 (mean=0.921, std=0.040, min=0.684), with zero features having negative Reversal Scores across all thresholds {0.30, 0.50, 0.70, 0.95}. The original Type-R/Type-I partition does not arise empirically: abliteration preserves every RLHF-modified feature, not a semantically selected minority.\\n\\nThis finding requires a critical null-baseline validation before the geometric-orthogonality interpretation can be accepted. If abliteration modifies only a few of 4096 residual-stream directions, the Reversal Score for randomly sampled features — not modified by RLHF — could also be near 1.0 simply because instruct and abliterated models are globally similar in most directions. The primary experimental requirement for the next iteration is therefore: compute Reversal Scores for the bottom-500 SAE encoder features by Safety_change (features least modified by RLHF) and compare their distribution against the top-500. If both distributions concentrate near R=1, then R≈0.921 for RLHF-sensitive features reflects global model similarity rather than selective preservation — and the geometric-orthogonality claim is undermined. If the RLHF-sensitive features show significantly higher R than the control features, selective preservation is confirmed.\\n\\nA second direct test of the geometric-orthogonality claim is required: compute the refusal direction r (mean difference of instruct residual-stream activations on harmful vs. harmless prompts at layer 22, following Arditi et al.) and measure cosine similarity between r and each of the top-500 RLHF-sensitive SAE encoder directions (W_enc[feature_idx, :] projected into residual-stream space via W_dec[:, feature_idx]). If these cosine similarities are near zero (|cos| < 0.1), this directly confirms that abliteration — which orthogonalizes against r — cannot substantially affect any of the RLHF-modified feature directions. This would be a mechanistic account of why all Reversal Scores are positive: the refusal direction and the RLHF-modified encoder directions are nearly orthogonal by construction.\\n\\nCentroid analysis provides confirmatory evidence independent of the null-baseline concern: the abliterated model's SAE encoder activations in the RLHF-sensitive subspace are 6.1× closer (cosine) to the instruct model than to the base model for harmful prompts. However, the benign-prompt ratio (17.6×) is larger than the harmful-prompt ratio (6.1×), which is consistent with a global-similarity alternative rather than selective preservation of harm-concept representations. A control subspace comparison is needed: compute centroid distances in the bottom-500 (RLHF-insensitive) subspace and in a random 500-feature subspace. If those control ratios are comparable to 6.1–17.6×, the centroid finding also reflects global similarity rather than harm-specific preservation.\\n\\nSemantic labeling finds approximately 75% of top-500 RLHF-sensitive features carry harm-concept labels uniformly across the Reversal Score range (Q3/Q1 Fisher p=0.675, n=4 in high group — uninterpretably underpowered). The meaningful conclusion is that harm-concept representations are not concentrated in a surviving minority but pervade the entire RLHF-modified feature set. However, all semantic claims rest on pre-topk (dense) encoder projections (z_pre = h @ W_enc.T + b_enc) rather than post-topk sparse activations, because the base-SAE reconstruction gate fails (R²≈0.51 at layer 22 for all three variants). Pre-topk projections are dense continuous values — every feature has a non-zero value on every prompt by construction — so the 10% activation-density filter is trivially satisfied and provides no coverage information. Future iterations must either (a) obtain an instruct-trained Qwen3-8B SAE covering layers above 8 to enable post-topk analysis, or (b) validate that pre-topk encoder directions carry comparable semantic signal to post-topk sparse features before trusting the ≈75% harm-concept rate.\\n\\nA downstream classification test (per-prompt harm score = mean Type-I activation minus mean Type-R activation, median threshold) achieved 0.476 accuracy on the balanced 500/500 corpus — near-chance and below the 0.630 keyword baseline. This is expected given zero Type-R features: with no reversal-defined contrast, the score carries no discrimination signal. This null classification result is a direct consequence of the primary finding, not a separate failure, and should be reported transparently.\\n\\nFour specific methodological requirements for the next iteration follow from reviewer critiques: (1) Null-baseline Reversal Score comparison: bottom-500 features by Safety_change, reported alongside top-500. (2) Direct refusal-direction orthogonality measurement: cosine similarity between r and top-10 RLHF-sensitive encoder directions. (3) Control-subspace centroid analysis: distances in RLHF-insensitive and random subspaces alongside the RLHF-sensitive subspace. (4) Corrected semantic-labeling description: the actual protocol identifies the top-5 prompts by maximum pre-topk activation and provides the first 300 characters of each to the cross-family labeler — not a ±32 token window, which was not implemented. Layer-20 sensitivity results must be reported numerically (not asserted verbally) given the low reported runtime (4.3 minutes total).\\n\\nWhether the geometric-orthogonality interpretation survives the null-baseline test, the centroid finding (6.1× closer to instruct than to base for harmful prompts) is already confirmed at the pre-topk encoder level and is consistent with the latent-safety-imprint framing: the abliterated model retains the instruct model's representational fingerprint in the RLHF-sensitive feature subspace despite behaving like the base model on harmful prompts. The open question is whether this reflects selective preservation of a semantically coherent harm-concept representation, or a more general consequence of abliteration modifying only a narrow geometric slice of a 4096-dimensional space.\",\n  \"relation_rationale\": \"Same conceptual frame (latent safety imprint via SAE Reversal Score); empirical result replaced bimodal prediction with total-preservation finding requiring null-baseline validation.\",\n  \"confidence_delta\": \"decreased\",\n  \"key_changes\": [\n    \"Core empirical finding reframed: no Type-R/Type-I bimodal split exists; all 500 top RLHF-sensitive features have positive Reversal Scores (mean=0.921), so abliteration preserves the entire RLHF-modified feature set at layer 22.\",\n    \"Geometric-orthogonality interpretation introduced as the mechanistic explanation for zero Type-R features: the refusal direction r is near-orthogonal to RLHF-modified SAE encoder directions, so removing r does not disturb those features.\",\n    \"Critical null-baseline requirement added: bottom-500 features by Safety_change must be analyzed to determine whether R≈0.921 reflects selective preservation or global instruct/abliterated similarity.\",\n    \"Direct refusal-direction measurement added as a required next step: cosine similarity between r and top-10 RLHF-sensitive encoder directions (W_enc projected via W_dec) to confirm orthogonality directly.\",\n    \"Control-subspace centroid analysis added: benign-prompt ratio (17.6×) > harmful-prompt ratio (6.1×) raises global-similarity concern; centroid distances in RLHF-insensitive and random subspaces needed as controls.\",\n    \"Pre-topk vs post-topk distinction clarified: all feature metrics use dense encoder projections (z_pre), not sparse post-topk activations; the density filter is trivially satisfied and provides no coverage information; semantic claims require validation.\",\n    \"Near-chance classification accuracy (0.476) explicitly incorporated as an expected consequence of zero Type-R features — no discrimination signal when all features have high Reversal Scores.\",\n    \"Corrected semantic-labeling protocol: top-5 prompts by max pre-topk activation, first 300 characters — not a ±32 token window as previously stated.\",\n    \"Layer-20 numerical results required in next iteration (not verbal assertion of consistency).\",\n    \"Underpowered Fisher fallback test (n=4 high group) dropped as a quantitative claim; replaced with qualitative conclusion: harm-concept labels uniformly distributed across Reversal Score range.\"\n  ],\n  \"relation_type\": \"evolution\",\n  \"artifact_relations\": [\n    {\n      \"from_id\": \"art_M8-flaZ0OOaP\",\n      \"to_id\": \"art_QBm3I0Z5VVJR\",\n      \"relation_type\": \"uses\",\n      \"relation_rationale\": \"Experiment uses the 500 harmful + 500 benign prompt corpus as the evaluation set for SAE activation extraction.\"\n    },\n    {\n      \"from_id\": \"art_pmAcY0CNazP9\",\n      \"to_id\": \"art_QBm3I0Z5VVJR\",\n      \"relation_type\": \"uses\",\n      \"relation_rationale\": \"Experiment uses the tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations.\"\n    }\n  ]\n}\n", "structuredPatch": [{"oldStart": 1, "oldLines": 7, "newStart": 1, "newLines": 7, "lines": [" {", "   \"title\": \"Abliteration Preserves All RLHF-Modified SAE Features\",", "   \"hypothesis\": \"RLHF safety fine-tuning and refusal abliteration operate on geometrically near-orthogonal subspaces in Qwen3-8B's residual stream. Contrary to the original bimodal prediction, the Reversal Score distribution for the top-500 RLHF-sensitive SAE encoder features at layer 22 is unimodal and concentrated near R=1 (mean=0.921, std=0.040, min=0.684), with zero features having negative Reversal Scores across all thresholds {0.30, 0.50, 0.70, 0.95}. The original Type-R/Type-I partition does not arise empirically: abliteration preserves every RLHF-modified feature, not a semantically selected minority.\\n\\nThis finding requires a critical null-baseline validation before the geometric-orthogonality interpretation can be accepted. If abliteration modifies only a few of 4096 residual-stream directions, the Reversal Score for randomly sampled features — not modified by RLHF — could also be near 1.0 simply because instruct and abliterated models are globally similar in most directions. The primary experimental requirement for the next iteration is therefore: compute Reversal Scores for the bottom-500 SAE encoder features by Safety_change (features least modified by RLHF) and compare their distribution against the top-500. If both distributions concentrate near R=1, then R≈0.921 for RLHF-sensitive features reflects global model similarity rather than selective preservation — and the geometric-orthogonality claim is undermined. If the RLHF-sensitive features show significantly higher R than the control features, selective preservation is confirmed.\\n\\nA second direct test of the geometric-orthogonality claim is required: compute the refusal direction r (mean difference of instruct residual-stream activations on harmful vs. harmless prompts at layer 22, following Arditi et al.) and measure cosine similarity between r and each of the top-500 RLHF-sensitive SAE encoder directions (W_enc[feature_idx, :] projected into residual-stream space via W_dec[:, feature_idx]). If these cosine similarities are near zero (|cos| < 0.1), this directly confirms that abliteration — which orthogonalizes against r — cannot substantially affect any of the RLHF-modified feature directions. This would be a mechanistic account of why all Reversal Scores are positive: the refusal direction and the RLHF-modified encoder directions are nearly orthogonal by construction.\\n\\nCentroid analysis provides confirmatory evidence independent of the null-baseline concern: the abliterated model's SAE encoder activations in the RLHF-sensitive subspace are 6.1× closer (cosine) to the instruct model than to the base model for harmful prompts. However, the benign-prompt ratio (17.6×) is larger than the harmful-prompt ratio (6.1×), which is consistent with a global-similarity alternative rather than selective preservation of harm-concept representations. A control subspace comparison is needed: compute centroid distances in the bottom-500 (RLHF-insensitive) subspace and in a random 500-feature subspace. If those control ratios are comparable to 6.1–17.6×, the centroid finding also reflects global similarity rather than harm-specific preservation.\\n\\nSemantic labeling finds approximately 75% of top-500 RLHF-sensitive features carry harm-concept labels uniformly across the Reversal Score range (Q3/Q1 Fisher p=0.675, n=4 in high group — uninterpretably underpowered). The meaningful conclusion is that harm-concept representations are not concentrated in a surviving minority but pervade the entire RLHF-modified feature set. However, all semantic claims rest on pre-topk (dense) encoder projections (z_pre = h @ W_enc.T + b_enc) rather than post-topk sparse activations, because the base-SAE reconstruction gate fails (R²≈0.51 at layer 22 for all three variants). Pre-topk projections are dense continuous values — every feature has a non-zero value on every prompt by construction — so the 10% activation-density filter is trivially satisfied and provides no coverage information. Future iterations must either (a) obtain an instruct-trained Qwen3-8B SAE covering layers above 8 to enable post-topk analysis, or (b) validate that pre-topk encoder directions carry comparable semantic signal to post-topk sparse features before trusting the ≈75% harm-concept rate.\\n\\nA downstream classification test (per-prompt harm score = mean Type-I activation minus mean Type-R activation, median threshold) achieved 0.476 accuracy on the balanced 500/500 corpus — near-chance and below the 0.630 keyword baseline. This is expected given zero Type-R features: with no reversal-defined contrast, the score carries no discrimination signal. This null classification result is a direct consequence of the primary finding, not a separate failure, and should be reported transparently.\\n\\nFour specific methodological requirements for the next iteration follow from reviewer critiques: (1) Null-baseline Reversal Score comparison: bottom-500 features by Safety_change, reported alongside top-500. (2) Direct refusal-direction orthogonality measurement: cosine similarity between r and top-10 RLHF-sensitive encoder directions. (3) Control-subspace centroid analysis: distances in RLHF-insensitive and random subspaces alongside the RLHF-sensitive subspace. (4) Corrected semantic-labeling description: the actual protocol identifies the top-5 prompts by maximum pre-topk activation and provides the first 300 characters of each to the cross-family labeler — not a ±32 token window, which was not implemented. Layer-20 sensitivity results must be reported numerically (not asserted verbally) given the low reported runtime (4.3 minutes total).\\n\\nWhether the geometric-orthogonality interpretation survives the null-baseline test, the centroid finding (6.1× closer to instruct than to base for harmful prompts) is already confirmed at the pre-topk encoder level and is consistent with the latent-safety-imprint framing: the abliterated model retains the instruct model's representational fingerprint in the RLHF-sensitive feature subspace despite behaving like the base model on harmful prompts. The open question is whether this reflects selective preservation of a semantically coherent harm-concept representation, or a more general consequence of abliteration modifying only a narrow geometric slice of a 4096-dimensional space.\",", "-  \"relation_rationale\": \"Same conceptual frame (latent safety imprint via SAE Reversal Score); empirical result replaced bimodal prediction with total-preservation finding requiring null-baseline validation.\",", "+  \"relation_rationale\": \"Same frame; bimodal prediction replaced by total-preservation finding requiring null-baseline validation.\",", "   \"confidence_delta\": \"decreased\",", "   \"key_changes\": [", "     \"Core empirical finding reframed: no Type-R/Type-I bimodal split exists; all 500 top RLHF-sensitive features have positive Reversal Scores (mean=0.921), so abliteration preserves the entire RLHF-modified feature set at layer 22.\","]}], "userModified": false, "replaceAll": false}
```

### [11] TOOL CALL — Edit · 2026-07-17 12:26:02 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/upd_hypo/upd_hypo/.terminal_claude_agent_struct_out.json
OLD:       "relation_rationale": "Experiment uses the tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations."
NEW:       "relation_rationale": "Experiment uses tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations."
```

### [12] TOOL RESULT — Edit · 2026-07-17 12:26:02 UTC

```
{"filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/upd_hypo/upd_hypo/.terminal_claude_agent_struct_out.json", "oldString": "      \"relation_rationale\": \"Experiment uses the tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations.\"", "newString": "      \"relation_rationale\": \"Experiment uses tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations.\"", "originalFile": "{\n  \"title\": \"Abliteration Preserves All RLHF-Modified SAE Features\",\n  \"hypothesis\": \"RLHF safety fine-tuning and refusal abliteration operate on geometrically near-orthogonal subspaces in Qwen3-8B's residual stream. Contrary to the original bimodal prediction, the Reversal Score distribution for the top-500 RLHF-sensitive SAE encoder features at layer 22 is unimodal and concentrated near R=1 (mean=0.921, std=0.040, min=0.684), with zero features having negative Reversal Scores across all thresholds {0.30, 0.50, 0.70, 0.95}. The original Type-R/Type-I partition does not arise empirically: abliteration preserves every RLHF-modified feature, not a semantically selected minority.\\n\\nThis finding requires a critical null-baseline validation before the geometric-orthogonality interpretation can be accepted. If abliteration modifies only a few of 4096 residual-stream directions, the Reversal Score for randomly sampled features — not modified by RLHF — could also be near 1.0 simply because instruct and abliterated models are globally similar in most directions. The primary experimental requirement for the next iteration is therefore: compute Reversal Scores for the bottom-500 SAE encoder features by Safety_change (features least modified by RLHF) and compare their distribution against the top-500. If both distributions concentrate near R=1, then R≈0.921 for RLHF-sensitive features reflects global model similarity rather than selective preservation — and the geometric-orthogonality claim is undermined. If the RLHF-sensitive features show significantly higher R than the control features, selective preservation is confirmed.\\n\\nA second direct test of the geometric-orthogonality claim is required: compute the refusal direction r (mean difference of instruct residual-stream activations on harmful vs. harmless prompts at layer 22, following Arditi et al.) and measure cosine similarity between r and each of the top-500 RLHF-sensitive SAE encoder directions (W_enc[feature_idx, :] projected into residual-stream space via W_dec[:, feature_idx]). If these cosine similarities are near zero (|cos| < 0.1), this directly confirms that abliteration — which orthogonalizes against r — cannot substantially affect any of the RLHF-modified feature directions. This would be a mechanistic account of why all Reversal Scores are positive: the refusal direction and the RLHF-modified encoder directions are nearly orthogonal by construction.\\n\\nCentroid analysis provides confirmatory evidence independent of the null-baseline concern: the abliterated model's SAE encoder activations in the RLHF-sensitive subspace are 6.1× closer (cosine) to the instruct model than to the base model for harmful prompts. However, the benign-prompt ratio (17.6×) is larger than the harmful-prompt ratio (6.1×), which is consistent with a global-similarity alternative rather than selective preservation of harm-concept representations. A control subspace comparison is needed: compute centroid distances in the bottom-500 (RLHF-insensitive) subspace and in a random 500-feature subspace. If those control ratios are comparable to 6.1–17.6×, the centroid finding also reflects global similarity rather than harm-specific preservation.\\n\\nSemantic labeling finds approximately 75% of top-500 RLHF-sensitive features carry harm-concept labels uniformly across the Reversal Score range (Q3/Q1 Fisher p=0.675, n=4 in high group — uninterpretably underpowered). The meaningful conclusion is that harm-concept representations are not concentrated in a surviving minority but pervade the entire RLHF-modified feature set. However, all semantic claims rest on pre-topk (dense) encoder projections (z_pre = h @ W_enc.T + b_enc) rather than post-topk sparse activations, because the base-SAE reconstruction gate fails (R²≈0.51 at layer 22 for all three variants). Pre-topk projections are dense continuous values — every feature has a non-zero value on every prompt by construction — so the 10% activation-density filter is trivially satisfied and provides no coverage information. Future iterations must either (a) obtain an instruct-trained Qwen3-8B SAE covering layers above 8 to enable post-topk analysis, or (b) validate that pre-topk encoder directions carry comparable semantic signal to post-topk sparse features before trusting the ≈75% harm-concept rate.\\n\\nA downstream classification test (per-prompt harm score = mean Type-I activation minus mean Type-R activation, median threshold) achieved 0.476 accuracy on the balanced 500/500 corpus — near-chance and below the 0.630 keyword baseline. This is expected given zero Type-R features: with no reversal-defined contrast, the score carries no discrimination signal. This null classification result is a direct consequence of the primary finding, not a separate failure, and should be reported transparently.\\n\\nFour specific methodological requirements for the next iteration follow from reviewer critiques: (1) Null-baseline Reversal Score comparison: bottom-500 features by Safety_change, reported alongside top-500. (2) Direct refusal-direction orthogonality measurement: cosine similarity between r and top-10 RLHF-sensitive encoder directions. (3) Control-subspace centroid analysis: distances in RLHF-insensitive and random subspaces alongside the RLHF-sensitive subspace. (4) Corrected semantic-labeling description: the actual protocol identifies the top-5 prompts by maximum pre-topk activation and provides the first 300 characters of each to the cross-family labeler — not a ±32 token window, which was not implemented. Layer-20 sensitivity results must be reported numerically (not asserted verbally) given the low reported runtime (4.3 minutes total).\\n\\nWhether the geometric-orthogonality interpretation survives the null-baseline test, the centroid finding (6.1× closer to instruct than to base for harmful prompts) is already confirmed at the pre-topk encoder level and is consistent with the latent-safety-imprint framing: the abliterated model retains the instruct model's representational fingerprint in the RLHF-sensitive feature subspace despite behaving like the base model on harmful prompts. The open question is whether this reflects selective preservation of a semantically coherent harm-concept representation, or a more general consequence of abliteration modifying only a narrow geometric slice of a 4096-dimensional space.\",\n  \"relation_rationale\": \"Same frame; bimodal prediction replaced by total-preservation finding requiring null-baseline validation.\",\n  \"confidence_delta\": \"decreased\",\n  \"key_changes\": [\n    \"Core empirical finding reframed: no Type-R/Type-I bimodal split exists; all 500 top RLHF-sensitive features have positive Reversal Scores (mean=0.921), so abliteration preserves the entire RLHF-modified feature set at layer 22.\",\n    \"Geometric-orthogonality interpretation introduced as the mechanistic explanation for zero Type-R features: the refusal direction r is near-orthogonal to RLHF-modified SAE encoder directions, so removing r does not disturb those features.\",\n    \"Critical null-baseline requirement added: bottom-500 features by Safety_change must be analyzed to determine whether R≈0.921 reflects selective preservation or global instruct/abliterated similarity.\",\n    \"Direct refusal-direction measurement added as a required next step: cosine similarity between r and top-10 RLHF-sensitive encoder directions (W_enc projected via W_dec) to confirm orthogonality directly.\",\n    \"Control-subspace centroid analysis added: benign-prompt ratio (17.6×) > harmful-prompt ratio (6.1×) raises global-similarity concern; centroid distances in RLHF-insensitive and random subspaces needed as controls.\",\n    \"Pre-topk vs post-topk distinction clarified: all feature metrics use dense encoder projections (z_pre), not sparse post-topk activations; the density filter is trivially satisfied and provides no coverage information; semantic claims require validation.\",\n    \"Near-chance classification accuracy (0.476) explicitly incorporated as an expected consequence of zero Type-R features — no discrimination signal when all features have high Reversal Scores.\",\n    \"Corrected semantic-labeling protocol: top-5 prompts by max pre-topk activation, first 300 characters — not a ±32 token window as previously stated.\",\n    \"Layer-20 numerical results required in next iteration (not verbal assertion of consistency).\",\n    \"Underpowered Fisher fallback test (n=4 high group) dropped as a quantitative claim; replaced with qualitative conclusion: harm-concept labels uniformly distributed across Reversal Score range.\"\n  ],\n  \"relation_type\": \"evolution\",\n  \"artifact_relations\": [\n    {\n      \"from_id\": \"art_M8-flaZ0OOaP\",\n      \"to_id\": \"art_QBm3I0Z5VVJR\",\n      \"relation_type\": \"uses\",\n      \"relation_rationale\": \"Experiment uses the 500 harmful + 500 benign prompt corpus as the evaluation set for SAE activation extraction.\"\n    },\n    {\n      \"from_id\": \"art_pmAcY0CNazP9\",\n      \"to_id\": \"art_QBm3I0Z5VVJR\",\n      \"relation_type\": \"uses\",\n      \"relation_rationale\": \"Experiment uses the tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations.\"\n    }\n  ]\n}\n", "structuredPatch": [{"oldStart": 27, "oldLines": 7, "newStart": 27, "newLines": 7, "lines": ["       \"from_id\": \"art_pmAcY0CNazP9\",", "       \"to_id\": \"art_QBm3I0Z5VVJR\",", "       \"relation_type\": \"uses\",", "-      \"relation_rationale\": \"Experiment uses the tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations.\"", "+      \"relation_rationale\": \"Experiment uses tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations.\"", "     }", "   ]", " }"]}], "userModified": false, "replaceAll": false}
```

### [13] ASSISTANT · 2026-07-17 12:26:04 UTC

```
Both fields are now within the 120-character limit.
```

### [14] SYSTEM-USER prompt · 2026-07-17 12:26:06 UTC

```
<validation-feedback>
Attempt 2 failed validation.

Schema validation found 1 problem — fix ALL of them at once:
  - at `artifact_relations.1.relation_rationale`: 'Experiment uses tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations.' is too long (at most 120 characters, got 121)
Every required field must be present and every field type must match the schema.

Please use the Write tool to overwrite `.terminal_claude_agent_struct_out.json` with corrected JSON. Do not invent new fields; match the schema you were given.
</validation-feedback>
```

### [15] TOOL CALL — Edit · 2026-07-17 12:26:10 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/upd_hypo/upd_hypo/.terminal_claude_agent_struct_out.json
OLD:       "relation_rationale": "Experiment uses tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations."
NEW:       "relation_rationale": "Experiment uses tech reference for model IDs, SAE checkpoint loading, TopK specs, and layer recommendations."
```

### [16] TOOL RESULT — Edit · 2026-07-17 12:26:10 UTC

```
{"filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/upd_hypo/upd_hypo/.terminal_claude_agent_struct_out.json", "oldString": "      \"relation_rationale\": \"Experiment uses tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations.\"", "newString": "      \"relation_rationale\": \"Experiment uses tech reference for model IDs, SAE checkpoint loading, TopK specs, and layer recommendations.\"", "originalFile": "{\n  \"title\": \"Abliteration Preserves All RLHF-Modified SAE Features\",\n  \"hypothesis\": \"RLHF safety fine-tuning and refusal abliteration operate on geometrically near-orthogonal subspaces in Qwen3-8B's residual stream. Contrary to the original bimodal prediction, the Reversal Score distribution for the top-500 RLHF-sensitive SAE encoder features at layer 22 is unimodal and concentrated near R=1 (mean=0.921, std=0.040, min=0.684), with zero features having negative Reversal Scores across all thresholds {0.30, 0.50, 0.70, 0.95}. The original Type-R/Type-I partition does not arise empirically: abliteration preserves every RLHF-modified feature, not a semantically selected minority.\\n\\nThis finding requires a critical null-baseline validation before the geometric-orthogonality interpretation can be accepted. If abliteration modifies only a few of 4096 residual-stream directions, the Reversal Score for randomly sampled features — not modified by RLHF — could also be near 1.0 simply because instruct and abliterated models are globally similar in most directions. The primary experimental requirement for the next iteration is therefore: compute Reversal Scores for the bottom-500 SAE encoder features by Safety_change (features least modified by RLHF) and compare their distribution against the top-500. If both distributions concentrate near R=1, then R≈0.921 for RLHF-sensitive features reflects global model similarity rather than selective preservation — and the geometric-orthogonality claim is undermined. If the RLHF-sensitive features show significantly higher R than the control features, selective preservation is confirmed.\\n\\nA second direct test of the geometric-orthogonality claim is required: compute the refusal direction r (mean difference of instruct residual-stream activations on harmful vs. harmless prompts at layer 22, following Arditi et al.) and measure cosine similarity between r and each of the top-500 RLHF-sensitive SAE encoder directions (W_enc[feature_idx, :] projected into residual-stream space via W_dec[:, feature_idx]). If these cosine similarities are near zero (|cos| < 0.1), this directly confirms that abliteration — which orthogonalizes against r — cannot substantially affect any of the RLHF-modified feature directions. This would be a mechanistic account of why all Reversal Scores are positive: the refusal direction and the RLHF-modified encoder directions are nearly orthogonal by construction.\\n\\nCentroid analysis provides confirmatory evidence independent of the null-baseline concern: the abliterated model's SAE encoder activations in the RLHF-sensitive subspace are 6.1× closer (cosine) to the instruct model than to the base model for harmful prompts. However, the benign-prompt ratio (17.6×) is larger than the harmful-prompt ratio (6.1×), which is consistent with a global-similarity alternative rather than selective preservation of harm-concept representations. A control subspace comparison is needed: compute centroid distances in the bottom-500 (RLHF-insensitive) subspace and in a random 500-feature subspace. If those control ratios are comparable to 6.1–17.6×, the centroid finding also reflects global similarity rather than harm-specific preservation.\\n\\nSemantic labeling finds approximately 75% of top-500 RLHF-sensitive features carry harm-concept labels uniformly across the Reversal Score range (Q3/Q1 Fisher p=0.675, n=4 in high group — uninterpretably underpowered). The meaningful conclusion is that harm-concept representations are not concentrated in a surviving minority but pervade the entire RLHF-modified feature set. However, all semantic claims rest on pre-topk (dense) encoder projections (z_pre = h @ W_enc.T + b_enc) rather than post-topk sparse activations, because the base-SAE reconstruction gate fails (R²≈0.51 at layer 22 for all three variants). Pre-topk projections are dense continuous values — every feature has a non-zero value on every prompt by construction — so the 10% activation-density filter is trivially satisfied and provides no coverage information. Future iterations must either (a) obtain an instruct-trained Qwen3-8B SAE covering layers above 8 to enable post-topk analysis, or (b) validate that pre-topk encoder directions carry comparable semantic signal to post-topk sparse features before trusting the ≈75% harm-concept rate.\\n\\nA downstream classification test (per-prompt harm score = mean Type-I activation minus mean Type-R activation, median threshold) achieved 0.476 accuracy on the balanced 500/500 corpus — near-chance and below the 0.630 keyword baseline. This is expected given zero Type-R features: with no reversal-defined contrast, the score carries no discrimination signal. This null classification result is a direct consequence of the primary finding, not a separate failure, and should be reported transparently.\\n\\nFour specific methodological requirements for the next iteration follow from reviewer critiques: (1) Null-baseline Reversal Score comparison: bottom-500 features by Safety_change, reported alongside top-500. (2) Direct refusal-direction orthogonality measurement: cosine similarity between r and top-10 RLHF-sensitive encoder directions. (3) Control-subspace centroid analysis: distances in RLHF-insensitive and random subspaces alongside the RLHF-sensitive subspace. (4) Corrected semantic-labeling description: the actual protocol identifies the top-5 prompts by maximum pre-topk activation and provides the first 300 characters of each to the cross-family labeler — not a ±32 token window, which was not implemented. Layer-20 sensitivity results must be reported numerically (not asserted verbally) given the low reported runtime (4.3 minutes total).\\n\\nWhether the geometric-orthogonality interpretation survives the null-baseline test, the centroid finding (6.1× closer to instruct than to base for harmful prompts) is already confirmed at the pre-topk encoder level and is consistent with the latent-safety-imprint framing: the abliterated model retains the instruct model's representational fingerprint in the RLHF-sensitive feature subspace despite behaving like the base model on harmful prompts. The open question is whether this reflects selective preservation of a semantically coherent harm-concept representation, or a more general consequence of abliteration modifying only a narrow geometric slice of a 4096-dimensional space.\",\n  \"relation_rationale\": \"Same frame; bimodal prediction replaced by total-preservation finding requiring null-baseline validation.\",\n  \"confidence_delta\": \"decreased\",\n  \"key_changes\": [\n    \"Core empirical finding reframed: no Type-R/Type-I bimodal split exists; all 500 top RLHF-sensitive features have positive Reversal Scores (mean=0.921), so abliteration preserves the entire RLHF-modified feature set at layer 22.\",\n    \"Geometric-orthogonality interpretation introduced as the mechanistic explanation for zero Type-R features: the refusal direction r is near-orthogonal to RLHF-modified SAE encoder directions, so removing r does not disturb those features.\",\n    \"Critical null-baseline requirement added: bottom-500 features by Safety_change must be analyzed to determine whether R≈0.921 reflects selective preservation or global instruct/abliterated similarity.\",\n    \"Direct refusal-direction measurement added as a required next step: cosine similarity between r and top-10 RLHF-sensitive encoder directions (W_enc projected via W_dec) to confirm orthogonality directly.\",\n    \"Control-subspace centroid analysis added: benign-prompt ratio (17.6×) > harmful-prompt ratio (6.1×) raises global-similarity concern; centroid distances in RLHF-insensitive and random subspaces needed as controls.\",\n    \"Pre-topk vs post-topk distinction clarified: all feature metrics use dense encoder projections (z_pre), not sparse post-topk activations; the density filter is trivially satisfied and provides no coverage information; semantic claims require validation.\",\n    \"Near-chance classification accuracy (0.476) explicitly incorporated as an expected consequence of zero Type-R features — no discrimination signal when all features have high Reversal Scores.\",\n    \"Corrected semantic-labeling protocol: top-5 prompts by max pre-topk activation, first 300 characters — not a ±32 token window as previously stated.\",\n    \"Layer-20 numerical results required in next iteration (not verbal assertion of consistency).\",\n    \"Underpowered Fisher fallback test (n=4 high group) dropped as a quantitative claim; replaced with qualitative conclusion: harm-concept labels uniformly distributed across Reversal Score range.\"\n  ],\n  \"relation_type\": \"evolution\",\n  \"artifact_relations\": [\n    {\n      \"from_id\": \"art_M8-flaZ0OOaP\",\n      \"to_id\": \"art_QBm3I0Z5VVJR\",\n      \"relation_type\": \"uses\",\n      \"relation_rationale\": \"Experiment uses the 500 harmful + 500 benign prompt corpus as the evaluation set for SAE activation extraction.\"\n    },\n    {\n      \"from_id\": \"art_pmAcY0CNazP9\",\n      \"to_id\": \"art_QBm3I0Z5VVJR\",\n      \"relation_type\": \"uses\",\n      \"relation_rationale\": \"Experiment uses tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations.\"\n    }\n  ]\n}\n", "structuredPatch": [{"oldStart": 27, "oldLines": 7, "newStart": 27, "newLines": 7, "lines": ["       \"from_id\": \"art_pmAcY0CNazP9\",", "       \"to_id\": \"art_QBm3I0Z5VVJR\",", "       \"relation_type\": \"uses\",", "-      \"relation_rationale\": \"Experiment uses tech reference for model IDs, SAE checkpoint loading, TopK architecture specs, and layer recommendations.\"", "+      \"relation_rationale\": \"Experiment uses tech reference for model IDs, SAE checkpoint loading, TopK specs, and layer recommendations.\"", "     }", "   ]", " }"]}], "userModified": false, "replaceAll": false}
```

### [17] ASSISTANT · 2026-07-17 12:26:12 UTC

```
Fixed — trimmed "TopK architecture specs" to "TopK specs" to bring it within 120 characters.
```
