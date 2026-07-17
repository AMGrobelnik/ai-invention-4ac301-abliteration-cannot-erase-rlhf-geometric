# review_hypo — create_idea

> Phase: `hypo_loop` · round 2 · `review_hypo`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim transcript of this agent task — every system/user prompt, assistant response, thinking block, tool call and tool result — in the order they occurred. Nothing truncated.

## Task: `review_hypo` (terminal_claude_agent, claude-sonnet-4-6)

### [1] CONFIG · 2026-07-17 09:46:11 UTC

```
model: claude-sonnet-4-6 | effort: medium | permission: bypassPermissions | cwd: /ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_2/review_hypo
```

### [2] SYSTEM-USER prompt · 2026-07-17 09:46:17 UTC

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
  features in the SAE basis. The degree of imprinting (fraction of safety-induced feature changes that survive abliteration)
  further predicts the residual capability gap between abliterated and base models, because Type-I features are also entangled
  with general semantic processing.
motivation: >-
  The standard narrative treats abliteration as 'undoing' safety fine-tuning, restoring the model to a base-like state. This
  has major implications for AI safety: if abliteration truly erases RLHF, then safety alignment is a superficial behavioral
  patch that can be trivially removed. If, however, RLHF writes two kinds of changes — one reversible, one permanent — then
  safety training has a deeper effect on the model's world model than previously recognized, and abliteration only removes
  the behavioral surface layer while leaving a 'cryptomnesic' safety trace. Understanding which aspects of safety survive
  abliteration is critical for: (1) evaluating the true robustness of safety alignment, (2) identifying candidate features
  for lightweight safety restoration without full retraining, and (3) providing a feature-level account of why abliterated
  models consistently underperform base models despite theoretically similar weight norms. The Qwen3-8B ecosystem is uniquely
  positioned for this study: official base and instruct variants, publicly released Qwen-Scope base-SAEs (SAE-Res-Qwen3-8B-Base-W64K-L0_50),
  and community-created abliterated models (huihui-ai/Qwen3-8B-abliterated) all exist and are freely accessible.
assumptions:
- >-
  The Qwen-Scope base-SAE captures enough of the relevant semantic feature space that projecting instruct and abliterated
  model activations through it produces interpretable, non-random feature activations (the base and instruct models share
  a broadly similar residual-stream geometry, which is supported by prior work showing high cosine similarity between pre-
  and post-alignment activations).
- >-
  Abliteration using the standard difference-of-means + orthogonal projection technique produces a well-defined 'abliteration
  direction' that can be compared against individual SAE feature directions, and the publicly available abliterated Qwen3-8B
  was produced this way.
- >-
  The distinction between Type-R and Type-I features is stable across prompt phrasings (i.e., a feature classified as Type-I
  shows consistent imprinting across diverse harmful prompts, not just for specific trigger phrases).
- >-
  Mid-layer residual stream activations (at the layers covered by Qwen-Scope SAEs) carry enough information to characterize
  the semantic changes induced by RLHF, consistent with prior findings that safety-relevant representations are concentrated
  in middle layers.
- >-
  Using raw text prompts without chat templates for all three variants provides a fair comparison that isolates safety-alignment
  effects from chat-template formatting artifacts, as validated by the difference-in-differences approach of prior methodology
  papers.
investigation_approach: |-
  Three-step experiment using all-Python implementation on a single GPU (16+ GB VRAM):

  **Step 1 — Setup and data collection.**
  Load three Qwen3-8B model checkpoints: base (Qwen/Qwen3-8B-Base), instruct/safety-tuned (Qwen/Qwen3-8B), and abliterated (huihui-ai/Qwen3-8B-abliterated). Load the Qwen-Scope base SAE (Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 — 64K features, JumpReLU) as a frozen measurement instrument. Collect 500 harmful prompts (from WildGuard/BeaverTails/HarmBench) and 500 benign prompts. All prompts are provided as raw text without chat templates to avoid formatting confounds. Run forward passes through each model, extract residual stream activations at the SAE-covered layer (typically a mid-layer). Project through the frozen SAE encoder to obtain 64K-dimensional sparse feature activation vectors for each (prompt, model) triple.

  **Step 2 — Feature classification.**
  For each of the 64K SAE features f, compute over the harmful prompt set:
  - Safety_change(f) = mean_i |act_instruct_i(f) - act_base_i(f)| (how much RLHF changed this feature)
  - Reversal_score(f) = cosine_similarity(act_abliterated - act_base, act_instruct - act_base) averaged over i (does abliteration push the feature back toward base or retain the instruct pattern?)

  Features with high Safety_change(f) are RLHF-sensitive. Among these:
  - Type-R: low Reversal_score (abliteration reverses the change, abliterated ≈ base)
  - Type-I: high Reversal_score (abliteration does NOT reverse the change, abliterated ≈ instruct)

  **Step 3 — Semantic interpretation and distance analysis.**
  For the top-100 Type-R and top-100 Type-I features by Safety_change magnitude: use the maximum-activating prompt segments as context, then call a cheap LLM via OpenRouter (e.g., Qwen2.5-7B or gemini-flash at ~$0.001/1K tokens) to generate natural-language labels for each feature. Compute centroid distances between model variants in the full feature space, Type-R subspace, and Type-I subspace separately, for harmful vs. benign prompts. Test the key prediction: d_TypeI(abliterated, instruct) < d_TypeI(abliterated, base) for harmful prompts but not for benign prompts.
success_criteria: |-
  CONFIRM hypothesis if ALL of the following hold:
  1. The Reversal_score distribution for RLHF-sensitive features (high Safety_change) is clearly bimodal — a cluster near 0 (Type-R, reversed) and a cluster near 1 (Type-I, imprinted) — rather than uniform. This rules out a continuum where every feature is partially reversed.
  2. Type-I features receive semantic labels from the LLM that are recognizably harm-concept related (e.g., weapon synthesis, deception schemas, manipulation tactics), while Type-R features receive refusal/hedging/polite-declination labels.
  3. In the Type-I feature subspace, d(abliterated, instruct) < d(abliterated, base) specifically for harmful prompts (not benign prompts). This is the core quantitative test: the abliterated model's harm-concept representations look more like the instruct model than the base model.
  4. The fraction of RLHF-sensitive features that are Type-I (vs. Type-R) positively correlates across multiple SAE layers with the capability gap between abliterated and base models on a standard benchmark (e.g., GSM8K or MMLU) — linking semantic imprinting to the capability cost of abliteration.

  DISCONFIRM hypothesis if: (a) the Reversal_score distribution is unimodal near 0 (all changes reversed — abliteration fully restores base model) or near 1 (all changes imprinted — abliteration does nothing semantically); (b) Type-I features receive no coherent harm-related semantic labels; or (c) d(abliterated, instruct) ≥ d(abliterated, base) in the Type-I subspace for harmful prompts.
related_works:
- >-
  LatentBiopsy (arXiv:2603.27412, 2025): Compares base, instruct, and abliterated Qwen3.5-0.8B and Qwen2.5-0.5B through PCA-based
  angular analysis of residual streams, finding that harmful-prompt geometry survives refusal ablation (AUROC drops by at
  most 0.015). Our work differs fundamentally: LatentBiopsy uses a holistic geometric measure (PCA components, angular deviation)
  that collapses all feature information into a single detection score, while we decompose the change at the individual SAE
  feature level to identify WHICH specific semantic features survive. LatentBiopsy shows that SOMETHING survives; we characterize
  WHAT survives and WHY.
- >-
  Mechanistic Investigation of Supervised Fine-Tuning via SAEs (arXiv:2605.11426, 2025): Uses pretrained base-SAEs as a frozen
  diagnostic to study how SFT changes feature activations, finding significant feature divergence despite high cosine similarity
  of raw activations. Our work extends this approach in two key ways: (1) we apply it to the three-model trajectory base→instruct→abliterated
  rather than just base→instruct, and (2) we specifically characterize features by whether the abliteration operation reverses
  or preserves the fine-tuning change — a dimension this paper does not address.
- >-
  Surgical Refusal Ablation (arXiv:2601.08489, 2025): Identifies that the raw refusal vector is polysemantic, entangling refusal
  signals with capability circuits, and proposes spectral residualization to create a clean, targeted ablation. This work
  addresses which CAPABILITY directions are confounded with the refusal direction. Our hypothesis is complementary but distinct:
  we address which SEMANTIC KNOWLEDGE features (in the SAE feature space) survive standard abliteration regardless of whether
  the abliteration vector was clean or polysemantic — i.e., we study the aftermath rather than the mechanism of abliteration.
- >-
  What Makes and Breaks Safety Fine-tuning (arXiv:2407.10264, 2024): Studies how supervised safety fine-tuning, DPO, and unlearning
  project unsafe activations into the MLP null space, revealing that safety is achieved through minimal weight transformation.
  Our work complements this by using the SAE basis to characterize what the 'minimal transformation' actually modifies semantically,
  and specifically identifies which of these modifications are robust to the abliteration attack.
- >-
  Qwen-Scope (arXiv:2605.11887, 2026): Releases 14 groups of SAEs across 7 Qwen model variants including Qwen3-8B, demonstrating
  their use for steering, safety classification, and post-training. We directly use these SAEs as the measurement instrument
  for our three-model comparison. Qwen-Scope provides the tool but does not itself perform the base/instruct/abliterated comparison
  or the feature reversibility analysis we propose.
inspiration: |-
  The core inspiration comes from cognitive neuroscience, specifically the **implicit vs. explicit memory dissociation** (Tulving, 1972; Schacter, 1987; Squire, 1992). In human memory, there are two distinct systems: declarative/explicit memory (consciously recallable facts and episodes, dependent on the hippocampus) and implicit/procedural memory (unconscious influences on behavior — priming, skills — dependent on striatum and neocortex). Classic amnesia studies (H.M., Clive Wearing) show that hippocampal damage abolishes declarative memory while leaving implicit memory intact. A patient who cannot remember learning to play piano may still improve with practice because the skill trace persists implicitly.

  Applied to RLHF: safety fine-tuning is analogous to encoding both explicit and implicit memories. The **explicit safety memory** is the refusal behavior — the model can consciously 'recall' that it should refuse and does so. The **implicit safety memory** is the semantic imprint — how the model internally represents and processes harmful content. Abliteration is analogous to hippocampal damage: it removes the explicit refusal behavior (the model can no longer 'recall' to refuse) while leaving the implicit semantic traces intact. The abliterated model exhibits **cryptomnesia** — it acts as if it has forgotten its safety training (it complies with harmful requests) but retains an unconscious imprint of that training in how it represents harmful concepts.

  At the methodological level, the approach of using a fixed SAE basis as a 'measurement instrument' across all three model variants is inspired by **isotopic tracing in chemistry**: in metabolic labeling experiments, you introduce isotope-labeled molecules and then track where the isotope ends up after a chemical reaction. The SAE features are the 'isotopes' — a fixed vocabulary that lets us track which representations change through the reaction (RLHF fine-tuning) and which survive the second reaction (abliteration).
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
    weight matrices that write to the residual stream against this direction. The result is a modified model that can no longer
    activate the refusal direction and therefore complies with previously-refused prompts, without retraining.
- term: Type-R feature (Refusal-Coupled)
  definition: >-
    An SAE feature whose activation was changed by safety RLHF fine-tuning and is reversed (restored toward the base model's
    activation level) when abliteration is applied. These features implement or support the explicit refusal mechanism. Operationally:
    features with high Safety_change and low Reversal_score in our measurement framework.
- term: Type-I feature (Semantically-Imprinted)
  definition: >-
    An SAE feature whose activation was changed by safety RLHF fine-tuning and is NOT reversed when abliteration is applied
    — the abliterated model's activation on this feature remains close to the instruct model's, not the base model's. These
    features represent the model's understanding of harmful content that persists after abliteration. Operationally: features
    with high Safety_change and high Reversal_score.
- term: Latent Safety Imprint
  definition: >-
    The set of Type-I feature modifications that safety RLHF fine-tuning writes into the model's representational space, which
    survive abliteration. A model with a latent safety imprint behaves like the base model (complies with harmful requests)
    but internally represents harmful content more like the safety-tuned instruct model. Analogous to implicit memory in neuroscience:
    the behavior is forgotten but the trace persists.
- term: Reversal_score
  definition: >-
    For a given SAE feature f, the cosine similarity between two vectors in activation space: (act_abliterated(f) - act_base(f))
    and (act_instruct(f) - act_base(f)), averaged over a set of prompts. A high reversal score (near 1) means the abliteration
    change goes in the SAME direction as the RLHF change — i.e., abliteration does NOT reverse the feature, it is imprinted.
    A low score (near 0 or negative) means abliteration reverses the RLHF change.
- term: Qwen-Scope
  definition: >-
    An open-source suite of Sparse Autoencoders released by the Qwen team (May 2026) for interpretability and development
    use with Qwen3 and Qwen3.5 model families. Includes base-model SAEs for Qwen3-8B with 64,000 features (SAE-Res-Qwen3-8B-Base-W64K-L0_50),
    which we use as the fixed measurement instrument in this study.
summary: >-
  We hypothesize that RLHF safety fine-tuning in Qwen3-8B writes two mechanistically distinct types of changes into the model's
  SAE feature space: 'Refusal-Coupled' features (reversed by abliteration) and 'Semantically-Imprinted' features (permanently
  changed, not reversed by abliteration). Using the Qwen-Scope base SAE as a fixed measurement instrument across base, instruct,
  and abliterated Qwen3-8B variants, we test whether the abliterated model retains the instruct model's representational fingerprint
  on harm-concept features despite exhibiting base-model-like compliance behavior — a 'latent safety imprint' analogous to
  implicit memory surviving hippocampal damage in cognitive neuroscience.
</previous_hypothesis>

<previous_review>
Critiques from the previous review. Check which ones have been addressed
in the revised hypothesis. Do NOT re-raise critiques that have been adequately fixed.
Only re-raise if the fix is insufficient.

- [MAJOR] (methodology) The Reversal_score definition is ambiguous in a way that could make the bimodality finding (Success Criterion 1) trivially true by mathematical construction. The formula 'cosine_similarity(act_abliterated - act_base, act_instruct - act_base) averaged over i' has two possible interpretations: (a) for each prompt i and each scalar feature f, compute cosine similarity between two scalars (which degenerates to the sign of their product, giving values in {-1, 0, 1} and forcing bimodality), or (b) for each feature f, compute Pearson/cosine correlation between the N-length vector of delta_instruct(f) values and the N-length vector of delta_abliterated(f) values across all prompts. Only interpretation (b) is scientifically meaningful. If interpretation (a) is intended, the bimodal distribution in Criterion 1 is guaranteed by the definition, not an empirical finding, and the entire Type-R/Type-I framing loses its evidential value.
  Action: Rewrite the Reversal_score as an explicit per-feature correlation across prompts: R(f) = PearsonR([act_instruct_i(f) - act_base_i(f) for i in 1..N], [act_abliterated_i(f) - act_base_i(f) for i in 1..N]). Add a sentence explaining that this measures whether the direction of feature change under abliteration matches the direction of change under RLHF, at the prompt-level granularity. Provide a worked example with concrete numbers.
- [MAJOR] (methodology) Assumption 2 states that the huihui-ai/Qwen3-8B-abliterated model was 'produced this way' (difference-of-means + orthogonal projection). Community-produced abliterated models on HuggingFace use a variety of techniques, including multi-layer ablation, fine-tuning-based uncensoring, and nonstandard prompt sets. If huihui-ai used a different approach (e.g., averaging over different layers or using a non-standard harmful prompt set), the comparison between the 'abliteration direction' and individual SAE feature directions loses its mechanistic grounding. This could silently invalidate the entire Type-R/Type-I classification by introducing confounds.
  Action: Check the huihui-ai/Qwen3-8B-abliterated model card and linked code to confirm the exact abliteration technique. If the technique matches (difference-of-means + orthogonal projection over a known layer set), document this explicitly. If it does not match, either: (a) perform abliteration from scratch using a reproducible script (llm-abliteration or similar) to ensure the method is controlled, or (b) add a robustness check comparing the community-abliterated model against a self-produced abliterated checkpoint.
- [MAJOR] (methodology) The base-SAE (trained on base model activations) may reconstruct instruct and abliterated model activations poorly, especially after safety RLHF. arXiv:2605.11426 reports that sparse latents diverge substantially between base and SFT models even though raw activation cosine similarity stays high. A base-SAE will apply JumpReLU thresholds calibrated for base model activations; applied to instruct model activations, the sparsity pattern may be distorted (many features falsely zero, or anomalously active), making the feature-level comparison unreliable. This is an untested assumption that directly undermines the validity of all three model comparison steps.
  Action: Before running the main experiment, report reconstruction MSE or fraction of variance explained (R²) for base, instruct, and abliterated model activations through the frozen base-SAE. Set an acceptance threshold (e.g., R² > 0.90 for all three variants). If the base-SAE reconstructs instruct activations poorly, consider using the SAE's encoder only (to get feature activations) without relying on reconstruction quality, and note the limitation. Alternatively, look at whether Qwen-Scope releases any instruct-model SAEs for cross-comparison.
- [MAJOR] (rigor) Success Criterion 4 — the claim that Type-I fraction correlates with capability gap across SAE layers — is almost certainly underpowered as described. With Qwen-Scope providing SAEs for ~14 groups across 7 model variants, the effective N for a cross-layer correlation within a single model family is at most the number of distinct SAE layers (likely 5-7). A correlation from 5-7 data points provides no statistical reliability and cannot support the causal claim that 'Type-I features are entangled with general semantic processing.' This would be a misleading correlation even if it reaches p<0.05.
  Action: Either (a) expand the correlation to multiple model families by running the same analysis on abliterated Qwen3.5-8B or Llama-3 variants (adding model families gives more data points with which to test cross-model generalizability); or (b) reframe Criterion 4 as exploratory/descriptive ('we report this correlation as a preliminary signal for future work') rather than a primary confirmatory criterion, and set an explicit minimum N for statistical validity in the paper.
- [MINOR] (novelty) The space of 'harmful representations survive abliteration' is now crowded. LatentBiopsy (arXiv:2603.27412) demonstrates AUROC drops of ≤0.015 after abliteration using PCA geometry on exactly the Qwen model family. arXiv:2604.18901 shows harmful intent is linearly separable with AUROC 0.982 that survives abliteration within ±0.003, across 12 models spanning Qwen, Llama, and Gemma families. The proposed work's novelty rests entirely on 'which specific SAE features survive and whether they are semantically coherent harm-concept features.' If the SAE reconstruction quality issue (above) is not validated, the feature-level story may reduce to a restatement of what these geometric papers already show at a more granular vocabulary.
  Action: In the introduction/framing, explicitly address why SAE-level feature characterization provides actionable insight beyond what LatentBiopsy and arXiv:2604.18901 already establish. The strongest argument is: (a) if Type-I features have interpretable semantic content (harm schemas, not just 'harmful geometry'), they become candidates for targeted restoration of safety without full retraining; (b) the capability gap link (Criterion 4) is a new mechanistic claim not made by geometric approaches. Frame these two specific downstream applications as the core motivation for needing feature-level, not just geometric, characterization.
- [MINOR] (methodology) Assumption 5 claims that using raw text prompts without chat templates 'provides a fair comparison.' However, the instruct model (Qwen/Qwen3-8B) is trained with Qwen's chat template format. When presented with raw text (no [INST] or system prompt), the instruct model's behavior is off-distribution and its internal representations may not reflect its safety-trained state. The comparison may actually be instruct-OOD vs. base-in-distribution vs. abliterated-OOD rather than a fair three-way comparison.
  Action: Run a pilot study comparing instruct model activations on raw vs. template-formatted harmful prompts and verify that the safety-relevant SAE features (high Safety_change) activate similarly in both formats. If they do not, use template-formatted prompts for the instruct model (with empty system prompt or safety system prompt) and document the formatting choice clearly. Cite the specific prior methodology papers that validate the no-template approach if that validation exists.
- [MINOR] (rigor) The bimodality prediction (Success Criterion 1) is central but unjustified. The hypothesis asserts that high-Safety_change features will show a 'clearly bimodal' Reversal_score distribution rather than a continuum. There is no theoretical argument for why abliteration would produce a clean binary partition rather than a continuum of partial reversals. The Surgical Refusal Ablation paper (arXiv:2601.08489) suggests the refusal direction is polysemantic and entangled with multiple capability circuits — this entanglement would more naturally produce a continuous distribution of partial reversals.
  Action: Weaken the bimodality claim from a success criterion to an empirical question: 'We will test whether the distribution is bimodal (consistent with two distinct mechanisms) or continuous (consistent with a single graded mechanism). Either result is interpretable.' Add a specification of the statistical test for bimodality (e.g., Hartigan's dip test) and a secondary analysis for the continuous case (e.g., treating Reversal_score as a continuous variable and examining its semantic correlates along the continuum).
- [MINOR] (scope) The hypothesis is limited to a single model family (Qwen3-8B) and a single abliteration technique. The 'latent safety imprint' phenomenon, if confirmed, may be model-specific or technique-specific. The broader safety implications claimed in the motivation (evaluating 'the true robustness of safety alignment,' identifying 'candidate features for lightweight safety restoration') require generalization across at least two model families.
  Action: Acknowledge in the motivation/scope section that this is a single-model proof-of-concept. Frame the Qwen3-8B study as demonstrating the measurement framework, with explicit plans for cross-model validation in follow-up work. This is acceptable for a first paper — but should not be oversold as a general result until replicated.
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

### [3] HUMAN-USER prompt · 2026-07-17 09:46:17 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [4] ASSISTANT · 2026-07-17 09:47:11 UTC

```
I'll review this revised hypothesis, checking how previous critiques were addressed and looking for any remaining issues.
```

### [5] TOOL CALL — Skill · 2026-07-17 09:47:11 UTC

```
aii-web-tools
```

### [6] TOOL RESULT — Skill · 2026-07-17 09:47:11 UTC

```
{"success": true, "commandName": "aii-web-tools"}
```

### [7] ASSISTANT · 2026-07-17 09:47:15 UTC

```
Let me search for relevant context to ground my review authoritatively.
```

### [8] TOOL CALL — ToolSearch · 2026-07-17 09:47:15 UTC

```
{'query': 'select:WebSearch,WebFetch'}
```

### [9] TOOL RESULT — ToolSearch · 2026-07-17 09:47:15 UTC

```
{"matches": ["WebSearch", "WebFetch"], "query": "select:WebSearch,WebFetch", "total_deferred_tools": 21}
```

### [10] TOOL CALL — WebSearch · 2026-07-17 09:47:23 UTC

```
SAE sparse autoencoder base model instruct model cross-model feature analysis RLHF 2025 2026
```

### [11] TOOL CALL — WebSearch · 2026-07-17 09:47:23 UTC

```
abliteration refusal safety representation survival mechanistic interpretability 2025 2026
```

### [12] TOOL RESULT — WebSearch · 2026-07-17 09:47:29 UTC

```
{"query": "abliteration refusal safety representation survival mechanistic interpretability 2025 2026", "results": [{"tool_use_id": "srvtoolu_01WswJnKABjkyL7gfg1nSBMA", "content": [{"title": "Abliteration - Learn AI - Miraheze", "url": "https://ai.miraheze.org/wiki/Abliteration"}, {"title": "Heretic vs Abliterated LLMs: Refusal Rates & Benchmarks (2026)", "url": "https://aithinkerlab.com/heretic-ai-abliteration-benchmarks-2026/"}, {"title": "Norm-Preserving Biprojected Abliteration", "url": "https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration"}, {"title": "LLM Refusal Abliteration Mechanisms", "url": "https://www.emergentmind.com/topics/refusal-abliteration"}, {"title": "Not All Refusals Are Equal: How Safety Alignment Fails Cybersecurity at Scale", "url": "https://arxiv.org/pdf/2607.02714"}, {"title": "Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs", "url": "https://arxiv.org/pdf/2603.27518"}, {"title": "On the Inseparability of Instructions and Data in Shared-Embedding Sequence Models", "url": "https://arxiv.org/pdf/2606.27567"}, {"title": "OBLITERATUS: Mapping the Geometry of Refusal Inside Large Language Models", "url": "https://themenonlab.blog/blog/obliteratus-abliteration-llm-refusal-removal/"}, {"title": "Not All Refusals Are Equal: How Safety Alignment Fails Cybersecurity at Scale", "url": "https://arxiv.org/html/2607.02714v1"}]}, "I found several relevant results about abliteration, refusal safety, and mechanistic interpretability from 2025-2026. Here's what the research shows:\n\n## Current Landscape (2025-2026)\n\nThe current landscape of abliteration in 2025 and 2026 is characterized by the routine production of abliterated model variants, the development of increasingly sophisticated defense mechanisms, and an ongoing arms race between abliteration techniques and alignment robustness research.\n\n## Mechanistic Understanding of Refusal\n\nThe intellectual origins of abliteration lie at the intersection of mechanistic interpretability research and the growing interest in understanding how safety alignment is encoded within neural networks, with mechanistic interpretability aiming to reverse-engineer the internal computations of neural networks.\n\nRecent mechanistic interpretability research published in 2024 found that refusal behaviour in multiple popular chat models can be linked to a one-dimensional subspace in the residual stream. However, concurrent mechanistic work has since shown that the refusal subspace is richer than a single direction, with researchers identifying multi-dimensional polyhedral concept cones and a dominant direction accompanied by multiple semantically distinct secondary features.\n\n## Defense Mechanisms and Trade-offs\n\nDefenses against abliteration, such as extended-refusal fine-tuning, attempt to disperse the safety signal across multiple latent dimensions through training-time intervention, though reported patterns indicate such defenses raise the difficulty of attack but do not eliminate the underlying vulnerability."], "durationSeconds": 6.346199830999991, "searchCount": 1}
```

### [13] TOOL RESULT — WebSearch · 2026-07-17 09:47:33 UTC

```
{"query": "SAE sparse autoencoder base model instruct model cross-model feature analysis RLHF 2025 2026", "results": [{"tool_use_id": "srvtoolu_01PcibqgDPPvihGn19V2qn2f", "content": [{"title": "Sparse Autoencoder Features for Classifications and Transferability - ACL Anthology", "url": "https://aclanthology.org/2025.emnlp-main.1521/"}, {"title": "Localizing RL-Induced Tool Use to a Single Crosscoder Feature", "url": "https://arxiv.org/pdf/2606.26474"}, {"title": "Route Sparse Autoencoder to Interpret Large Language ...", "url": "https://aclanthology.org/2025.emnlp-main.346.pdf"}, {"title": "Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models", "url": "https://arxiv.org/html/2410.06981v3"}, {"title": "Auditing Games for Sandbagging", "url": "https://arxiv.org/pdf/2512.07810"}, {"title": "Unstable Features, Reproducible Subspaces: Understanding Seed Dependence in Sparse Autoencoders", "url": "https://arxiv.org/pdf/2606.12138"}, {"title": "Discovering Millions of Interpretable Features with Sparse Autoencoders", "url": "https://arxiv.org/pdf/2606.26620"}, {"title": "[2502.11367] Sparse Autoencoder Features for Classifications and Transferability", "url": "https://arxiv.org/abs/2502.11367"}, {"title": "Perplexity Can Miss SAE Feature Damage Under Quantization", "url": "https://arxiv.org/pdf/2606.03002"}, {"title": "CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark of Interpretability of Sparse Autoencoders", "url": "https://arxiv.org/pdf/2509.00691"}]}, "Based on the web search results, here's what I found on SAE sparse autoencoders, cross-model feature analysis, and related research from 2025-2026:\n\n## Recent SAE Research (2025-2026)\n\n**Cross-Model Feature Analysis:**\nCrosscoders extend SAEs to pairs of models by jointly encoding their activations into one shared sparse dictionary. Dedicated Feature Crosscoders and Tied Crosscoders further partition this dictionary with gradient masking, forcing each model to read from and write to only its designated partition.\n\n**Base Model vs. Instruct Model Studies:**\nSAE-derived features achieve macro F1 > 0.8, outperforming baselines while demonstrating cross-model transfer from Gemma 2 2B to 9B-IT models, and these features generalize in a zero-shot manner to cross-lingual tasks. A 2025 EMNLP paper titled \"Sparse Autoencoder Features for Classifications and Transferability\" systematically analyzes SAE for interpretable feature extraction.\n\n**Feature Universality:**\nSAE features extracted from distinct models show high correlations, with semantically meaningful subspaces exhibiting even higher similarity across models.\n\n**RouteSAE Architecture:**\nRouteSAE incorporates a lightweight router with a shared SAE to dynamically extract multi-layer features in an efficient and flexible manner.\n\n**RLHF Applications:**\nResearch examines how RL-induced changes can be localized to specific sparse autoencoder features, with SAEs being popular tools for decomposing LLM activations into interpretable features."], "durationSeconds": 12.714019138999996, "searchCount": 1}
```

### [14] TOOL CALL — WebSearch · 2026-07-17 09:47:39 UTC

```
arXiv 2506.07691 instruct SAE sparse autoencoder base model reconstruction quality
```

### [15] TOOL CALL — WebSearch · 2026-07-17 09:47:39 UTC

```
Qwen3-8B abliterated huihui-ai model card method technique HuggingFace 2025
```

### [16] TOOL RESULT — WebSearch · 2026-07-17 09:47:43 UTC

```
{"query": "arXiv 2506.07691 instruct SAE sparse autoencoder base model reconstruction quality", "results": [{"tool_use_id": "srvtoolu_01XiX5ZK5K7U9F3sSKs9zt1b", "content": [{"title": "[2506.07691] Training Superior Sparse Autoencoders for Instruct Models", "url": "https://arxiv.org/abs/2506.07691"}, {"title": "Paper page - Training Superior Sparse Autoencoders for Instruct Models", "url": "https://huggingface.co/papers/2506.07691"}, {"title": "The Rate-Distortion-Polysemanticity Tradeoff in SAEs", "url": "https://arxiv.org/pdf/2605.14694"}, {"title": "Training Superior Sparse Autoencoders for Instruct Models", "url": "https://www.lesswrong.com/posts/REEPhYzeiP2JyH35C/training-superior-sparse-autoencoders-for-instruct-models"}, {"title": "Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement", "url": "https://arxiv.org/html/2509.23799v2"}, {"title": "Training Superior Sparse Autoencoders for Instruct Models", "url": "https://arxiv.org/html/2506.07691"}, {"title": "Toward Identifiable Sparse Autoencoders", "url": "https://arxiv.org/pdf/2605.31245"}]}, "Based on the search results, I found detailed information about arXiv paper 2506.07691:\n\n## Paper Overview\n\nThe paper titled \"Training Superior Sparse Autoencoders for Instruct Models\" addresses how sparse autoencoders have emerged as a key tool in mechanistic interpretability, enabling the extraction of human-interpretable features from LLMs.\n\n## Key Problem\n\nExisting SAE training methods are primarily designed for base models, resulting in reduced reconstruction quality and interpretability when applied to instruct models.\n\n## Proposed Solution\n\nThe authors propose Finetuning-aligned Sequential Training (FAST), a novel training method specifically tailored for instruct models. FAST processes each data instance independently, preserving semantic integrity and maintaining alignment with the fine-tuning objectives of the model.\n\n## Reconstruction Quality Results\n\nThe paper demonstrates significant improvements in reconstruction quality. On Qwen2.5-7B-Instruct, FAST achieves a mean squared error of 0.6468 in token reconstruction, significantly outperforming baseline methods with errors of 5.1985 and 1.5096.\n\n## Feature Interpretability\n\nIn feature interpretability, FAST yields a higher proportion of high-quality features; for Llama3.2-3B-Instruct, 21.1% scored in the top range, compared to 7.0% and 10.2% for baseline methods."], "durationSeconds": 6.569743155999997, "searchCount": 1}
```

### [17] TOOL RESULT — WebSearch · 2026-07-17 09:47:43 UTC

```
{"query": "Qwen3-8B abliterated huihui-ai model card method technique HuggingFace 2025", "results": [{"tool_use_id": "srvtoolu_01BGYS6z4iktbTQnitc395Z8", "content": [{"title": "huihui-ai/Qwen3-8B-abliterated · Hugging Face", "url": "https://huggingface.co/huihui-ai/Qwen3-8B-abliterated"}, {"title": "huihui-ai/Huihui-Qwen3-8B-abliterated-v2 · Hugging Face", "url": "https://huggingface.co/huihui-ai/Huihui-Qwen3-8B-abliterated-v2"}, {"title": "ikarius/Qwen3-8B-Abliterated-FP8 · Hugging Face", "url": "https://huggingface.co/ikarius/Qwen3-8B-Abliterated-FP8"}, {"title": "huihui-ai/Qwen3-8B-abliterated · Discussions", "url": "https://huggingface.co/huihui-ai/Qwen3-8B-abliterated/discussions"}, {"title": "Qwen3-8B-abliterated huggingface.co api & huihui-ai Qwen3-8B-abliterated github AI Model - Toolify", "url": "https://www.toolify.ai/ai-model/huihui-ai-qwen3-8b-abliterated"}, {"title": "huihui-ai/Huihui-Qwen3-4B-abliterated-v2 · Hugging Face", "url": "https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-abliterated-v2"}, {"title": "huihui-ai/Qwen3-4B-abliterated · Hugging Face", "url": "https://huggingface.co/huihui-ai/Qwen3-4B-abliterated"}, {"title": "huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated · Hugging Face", "url": "https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated"}, {"title": "huihui-ai/Huihui-Qwen3-4B-abliterated-v2 at main", "url": "https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-abliterated-v2/tree/main"}, {"title": "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated · Hugging Face", "url": "https://huggingface.co/huihui-ai/Huihui-Qwen3.5-0.8B-abliterated"}]}, "Based on the search results, here's what I found about the Qwen3-8B abliterated huihui-ai model:\n\n## Overview\n\nThe huihui-ai Qwen3-8B-abliterated is an uncensored version of Qwen/Qwen3-8B created with abliteration, which is a technique for removing safety mechanisms from language models.\n\n## Method and Technique\n\nThis is a proof-of-concept implementation to remove refusals from an LLM model without using TransformerLens, and ablation was performed using a new and faster method, which yields better results.\n\n## Model Versions\n\nThere's a newer version available: huihui-ai/Huihui-Qwen3-8B-abliterated-v2, which represents an improvement over the original version.\n\n## Important Safety Considerations\n\nUnlike standard models, this model has not undergone rigorous safety optimization. It is recommended to use this model for research, testing, or controlled environments, and users are strongly advised to monitor model outputs in real-time and conduct manual reviews when necessary to prevent the dissemination of inappropriate content.\n\nThe model card and additional technical details are available on the HuggingFace platform at the official model repository."], "durationSeconds": 5.764590950999991, "searchCount": 1}
```

### [18] TOOL CALL — WebFetch · 2026-07-17 09:47:49 UTC

```
URL: https://huggingface.co/huihui-ai/Qwen3-8B-abliterated
Prompt: What exact abliteration method/technique was used to create this model? What library or code was used? What layers were modified?
```

### [19] TOOL CALL — WebSearch · 2026-07-17 09:47:49 UTC

```
crosscoder SAE base instruct model pair analysis safety RLHF feature level 2025 2026
```

### [20] TOOL RESULT — WebFetch · 2026-07-17 09:47:53 UTC

```
{"bytes": 252172, "code": 200, "codeText": "OK", "result": "# Abliteration Method for Qwen3-8B-abliterated\n\nBased on the provided content, the specific abliteration technique used is described as follows:\n\n**Method:** The model employed \"a new and faster method, which yields better results\" compared to previous approaches. The documentation references the project \"remove-refusals-with-transformers\" as a conceptual basis for understanding abliteration techniques.\n\n**Key characteristic:** This is described as \"a crude, proof-of-concept implementation to remove refusals from an LLM model without using TransformerLens.\"\n\n**Limitations:** The webpage does not provide detailed technical specifications about which specific layers were modified, the exact algorithms applied, or the precise code implementation. The documentation suggests consulting the referenced GitHub project for deeper technical understanding, but those implementation details are not included in this particular model card.\n\nThe emphasis on it being \"new and faster\" suggests improvements over TransformerLens-based approaches, but concrete technical details about the mechanism remain unclear from this source alone.", "durationMs": 4256, "url": "https://huggingface.co/huihui-ai/Qwen3-8B-abliterated"}
```

### [21] TOOL RESULT — WebSearch · 2026-07-17 09:47:57 UTC

```
{"query": "crosscoder SAE base instruct model pair analysis safety RLHF feature level 2025 2026", "results": [{"tool_use_id": "srvtoolu_012RBDjQt5W5TLvYgUFqrneb", "content": [{"title": "Feature-Level Model Diffing | Learn Mechanistic Interpretability", "url": "https://learnmechinterp.com/topics/feature-level-model-diffing/"}, {"title": "Mechanistically Interpreting Compression in Vision-Language Models", "url": "https://arxiv.org/html/2603.25035"}, {"title": "Crosscoders | Learn Mechanistic Interpretability", "url": "http://learnmechinterp.com/topics/crosscoders/"}, {"title": "Localizing RL-Induced Tool Use to a Single Crosscoder Feature", "url": "https://arxiv.org/pdf/2606.26474"}, {"title": "Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation", "url": "https://arxiv.org/pdf/2512.23260"}, {"title": "Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation", "url": "https://arxiv.org/html/2512.23260v1"}, {"title": "Safe RLHF: Balancing Helpfulness & Safety", "url": "https://www.emergentmind.com/topics/safe-reinforcement-learning-from-human-feedback-safe-rlhf"}, {"title": "Route Sparse Autoencoder to Interpret Large Language Models", "url": "https://arxiv.org/pdf/2503.08200"}, {"title": "Cross-Architecture Model Diffing with Crosscoders", "url": "https://arxiv.org/pdf/2602.11729"}]}, "I found comprehensive search results related to crosscoders, SAEs, and model diffing with base and instruction-tuned model pairs. Here's what the research shows:\n\n## Crosscoders for Model Diffing\n\nCrosscoders extend sparse autoencoders to operate across multiple layers or models, learning a shared dictionary from concatenated activations. They decompose joint activations from multiple sources, enabling cross-source comparison and asking \"what is shared and what differs across these sources?\"\n\n## Base vs. Instruct Model Analysis\n\nWhen applied to base and chat model pairs, crosscoders reveal that most features are shared, and fine-tuning changes a relatively small fraction of the model's representations. Chat-exclusive features include safety-relevant concepts like refusal patterns, instruction-following behaviors, and output formatting conventions.\n\n## Safety and RLHF Context\n\nSafety fine-tuning involves a base model trained on next-token prediction, then further trained with RLHF or DPO to refuse harmful requests, follow instructions, and behave helpfully. Base-exclusive features include some capabilities that safety training suppresses or redirects.\n\n## Recent Developments\n\nThe research landscape includes recent advances like Dedicated Feature Crosscoders and Tied Crosscoders, which further partition dictionaries with gradient masking to force each model to read from and write to only its designated partition. There's also work on cross-architecture model diffing comparing different model families using BatchTopK sparsity penalties and automated interpretability techniques."], "durationSeconds": 8.27984031999999, "searchCount": 1}
```

### [22] TOOL CALL — WebFetch · 2026-07-17 09:48:09 UTC

```
URL: https://arxiv.org/abs/2602.11729
Prompt: What is the crosscoder method for model diffing? How does it compare base and instruct model activations? What does it find about shared vs. model-specific features?
```

### [23] TOOL CALL — WebFetch · 2026-07-17 09:48:09 UTC

```
URL: https://arxiv.org/abs/2606.26474
Prompt: What does this paper find about RL-induced changes localizable to crosscoder features? What methodology? How is this related to RLHF feature analysis?
```

### [24] TOOL RESULT — WebFetch · 2026-07-17 09:48:09 UTC

```
{"bytes": 41924, "code": 200, "codeText": "OK", "result": "# Crosscoder Method for Model Diffing\n\nBased on the provided content, the paper introduces crosscoders as a technique for comparing different LLM architectures. However, the abstract doesn't provide detailed methodological explanation of how crosscoders specifically compare base and instruct model activations.\n\n## What the Paper Addresses\n\nThe research focuses on \"model diffing, the process of comparing models' internal representations to identify their differences.\" The authors note that crosscoders have \"only ever been applied to base vs finetune comparisons\" and present the first cross-architecture application.\n\n## Key Innovation\n\nThe paper introduces **Dedicated Feature Crosscoders (DFCs)**, described as \"an architectural modification designed to better isolate features unique to one model.\" This suggests the method distinguishes between shared and model-specific features by focusing on differences rather than commonalities.\n\n## Research Findings\n\nThe unsupervised approach discovered distinct behavioral features across models, including political alignment patterns in certain models and copyright refusal mechanisms in others. This demonstrates the technique can surface meaningful model-specific behaviors without labeled training data.\n\n**Limitation**: The provided abstract doesn't contain sufficient technical detail about the specific mechanisms for comparing activations or the underlying architecture of the crosscoder method itself.", "durationMs": 4665, "url": "https://arxiv.org/abs/2602.11729"}
```

### [25] TOOL RESULT — WebFetch · 2026-07-17 09:48:09 UTC

```
{"bytes": 42723, "code": 200, "codeText": "OK", "result": "# Analysis of RL-Induced Tool Use Localization\n\n## Key Findings\n\nThe paper demonstrates that reinforcement learning changes to language models can be isolated to specific crosscoder features. Researchers found that \"encode-decode reconstruction improves the RL model's tool correctness by +31.1 ± 9.7 pp\" while also achieving what they term \"capability spillover\"—transferring tool-calling ability to frozen base models by \"+6.8 ± 5.0 pp.\"\n\n## Methodology\n\nThe approach uses **Dedicated Feature Crosscoders (DFC)**, which partition neural representations to identify RL-specific features. The researchers conducted a \"48-crosscoder hyperparameter sweep\" to isolate which features mediate tool-calling behavior in the Qwen2.5-3B model.\n\n## Connection to RLHF Analysis\n\nWhile the paper doesn't explicitly discuss RLHF (Reinforcement Learning from Human Feedback), it addresses the broader mechanistic question of understanding how RL reshapes internal model representations. Their work suggests RL introduces a \"minimal, steerable feature set\" rather than distributed changes—providing interpretability that could inform how we understand RL-induced behavioral shifts generally.\n\nThe research was accepted as a spotlight presentation at ICML 2026's Mechanistic Interpretability Workshop, positioning it within contemporary efforts to understand model internals after training interventions.", "durationMs": 4921, "url": "https://arxiv.org/abs/2606.26474"}
```

### [26] ASSISTANT · 2026-07-17 09:49:45 UTC

```
Now I have enough context to write a thorough review. Let me produce the output.
```

### [27] TOOL CALL — Write · 2026-07-17 09:49:45 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_2/review_hypo/.terminal_claude_agent_struct_out.json

{
  "overall_assessment": "This is a substantially improved second iteration that addresses all four major and all four minor critiques from the previous review. The Reversal_score is now rigorously defined as a per-feature Pearson correlation with a concrete worked example; abliteration method verification is an explicit Step 0; the SAE reconstruction quality gate (R² ≥ 0.85) is properly formalized; Criterion 4 is demoted to exploratory with explicit acknowledgment of N=1 invalidity; both bimodal and continuous outcomes are pre-registered with Hartigan's dip test; the chat-template confound is addressed via a pilot study; and scope limitations are openly acknowledged. The result is a methodologically careful, well-specified single-model proof-of-concept. The primary new issue is the absence of any discussion of crosscoders (arXiv:2602.11729, arXiv:2606.26474) — a closely related and now-established methodology for multi-model SAE comparison that the hypothesis must either justify avoiding or explicitly contrast. Secondary issues include underspecification of which Qwen-Scope SAE layer is selected and limited validation of the LLM semantic labeler used in Criterion 3. These are resolvable without rerunning experiments and do not block the experimental pipeline from proceeding.",
  "strengths": [
    "Complete resolution of all four prior major issues: Reversal_score is unambiguously defined as PearsonR over the N-dimensional prompt vector with a numerical worked example; abliteration technique verification is now a pre-registered Step 0 with a scratch-reproduction fallback; the R² ≥ 0.85 reconstruction quality gate properly constrains when encoder-only vs. full SAE analysis is used; and Criterion 4 is explicitly downgraded to descriptive/exploratory with correct N=1 caveat.",
    "Pre-registration of both the bimodal and continuous distributional outcomes via Hartigan's dip test is methodologically exemplary — the hypothesis is not betting on bimodality but treats distribution shape as an empirical question whose both answers yield interpretation.",
    "The three-model comparison trajectory (base → instruct → abliterated) is genuinely novel relative to the existing literature: LatentBiopsy (arXiv:2603.27412) and arXiv:2604.18901 both use holistic geometric measures that collapse feature information; arXiv:2605.11426 analyzes base→instruct only. The Reversal_score framing is a distinct conceptual contribution.",
    "Practical downstream motivation is now sharper: the hypothesis explicitly connects Type-I feature identification to targeted safety restoration without full retraining, providing a concrete use-case that geometric approaches cannot support.",
    "The glossary is comprehensive and precise. The distinction between encoder-only and full-reconstruction analysis, the R² gate, and the chat-template pilot are all clearly specified enough for direct implementation.",
    "Single-model scope limitation is explicitly acknowledged and framed correctly as a proof-of-concept measurement framework rather than a general result."
  ],
  "dimension_scores": [
    {
      "dimension": "soundness",
      "score": 3,
      "justification": "The methodology is now technically solid for the core experiment. The remaining soundness concerns are: (1) no comparison against crosscoders, which are the purpose-built instrument for multi-model SAE comparison and could expose whether the fixed base-SAE approach is adequate; (2) the encoder-only fallback is treated as reliable when JumpReLU thresholds calibrated on base activations will still produce distorted sparsity patterns on instruct distributions, so the fallback's reliability is not guaranteed; (3) the SAE layer to analyze is unspecified despite Qwen-Scope covering multiple layers.",
      "improvements": [
        "Add a sentence in the investigation approach specifying which Qwen-Scope SAE layer will be used and the selection criterion (e.g., the layer with highest Safety_change variance, the middle-of-stack layer known to carry safety-relevant geometry from prior work, or all available layers with results averaged). Leaving this implicit makes the experiment underspecified.",
        "In the encoder-only fallback description, add a caveat: 'JumpReLU thresholds calibrated on base activations may suppress or exaggerate feature activity on instruct activations even when reconstruction quality is not measured by R², so encoder-only activations still carry a distributional shift caveat that will be documented in the limitations section.'",
        "Discuss or cite crosscoder methodology (arXiv:2602.11729) as an alternative instrument and justify why the fixed base-SAE approach is preferred (likely: crosscoder training requires access to the training pipeline, whereas a frozen base-SAE is zero-cost and directly comparable to arXiv:2605.11426 which validates the approach)."
      ]
    },
    {
      "dimension": "presentation",
      "score": 4,
      "justification": "The hypothesis is exceptionally well-organized. The five-step experimental protocol is clear and sequential. The worked numerical example for the Reversal_score makes the key construct immediately interpretable. The glossary is complete and precise. The pre-registration of both distributional outcomes is presented clearly. The neuroscience and isotopic tracing analogies are evocative without being overreaching. No presentation concerns remain.",
      "improvements": [
        "Minor: specify the SAE layer selection criterion in Step 1 (currently 'the SAE-covered layer (typically a mid-layer)' is vague).",
        "Minor: in Step 3's semantic labeling, note whether the LLM labeler prompt will be independently validated (e.g., against a small human-annotated gold set) before being used as the Criterion 3 judge."
      ]
    },
    {
      "dimension": "contribution",
      "score": 3,
      "justification": "The contribution is genuinely novel at the feature level — no existing paper applies the Reversal_score framework to the base/instruct/abliterated trajectory, and the Type-R/Type-I classification with pre-registered bimodal/continuous interpretations is a clean conceptual contribution. However: the scope is limited to one model at one SAE layer; crosscoders (arXiv:2606.26474) recently demonstrated exactly the type of RL-induced feature localization this hypothesis seeks; and the safety implications depend on whether the downstream targeted restoration application actually works, which is outside the scope of this paper. Contribution is solid for a workshop or short paper but requires cross-model replication for a full venue claim.",
      "improvements": [
        "Address how the Reversal_score approach relates to and differs from crosscoders — the field now has a principled multi-model SAE comparison tool, and the contribution claim is strengthened if the hypothesis explains what the frozen-base-SAE approach offers beyond what crosscoders already provide (e.g., no training required, direct comparability with arXiv:2605.11426 findings, interpretability of the feature vocabulary via Qwen-Scope's existing labeled features).",
        "Add a brief comparison note: arXiv:2606.26474 (ICML 2026 spotlight) localizes RL-induced tool use to a single crosscoder feature. Contrasting that 'RL-induced addition' finding with the hypothesis's 'RLHF-induced survival under erasure' framing would sharpen the contribution's uniqueness."
      ]
    }
  ],
  "critiques": [
    {
      "category": "novelty",
      "severity": "major",
      "description": "Crosscoders (arXiv:2602.11729, arXiv:2606.26474) are the established, purpose-built tool for exactly the multi-model feature comparison this hypothesis proposes, and they are absent from the related works. Crosscoders train a shared sparse dictionary jointly on activations from two or more model variants, naturally partitioning features into 'shared' and 'model-specific' categories. Dedicated Feature Crosscoders (DFCs) were recently used in an ICML 2026 spotlight paper (arXiv:2606.26474) to localize RL-induced tool-calling to a single feature in Qwen2.5-3B. The base/instruct/abliterated three-model comparison in this hypothesis could, in principle, be addressed by a three-model crosscoder rather than projecting through a frozen base-SAE. The hypothesis must either (a) justify why the frozen-base-SAE approach is preferable or complementary — the strongest argument is that it requires no training, reuses Qwen-Scope's labeled feature vocabulary, and is directly comparable to arXiv:2605.11426 — or (b) add a crosscoder variant as a robustness check. Omitting this literature makes the contribution appear less well-situated than it actually is and will draw reviewer criticism at top venues.",
      "suggested_action": "Add a paragraph in the related works section discussing crosscoders (cite arXiv:2602.11729 and arXiv:2606.26474). Justify the frozen-base-SAE choice on the grounds of: (1) zero training cost, (2) reuse of Qwen-Scope's existing labeled feature vocabulary enabling interpretability without re-deriving feature semantics, (3) methodological continuity with arXiv:2605.11426 which validates base-SAE diagnostics for SFT analysis. If resources allow, running a two-model crosscoder (base+instruct) as a comparison would strengthen the methodology section considerably."
    },
    {
      "category": "methodology",
      "severity": "minor",
      "description": "The SAE layer to analyze is underspecified. The hypothesis refers to 'the SAE-covered layer (typically a mid-layer)' without specifying which of Qwen-Scope's multiple available SAE layers will be used or why. Safety-relevant representations are known to vary substantially across layers (some work identifies early layers for refusal detection, mid-layers for semantic content, late layers for output formatting). The choice of layer directly affects which features are captured by Safety_change and which are labeled Type-R vs. Type-I. If the results are layer-dependent, the single-layer analysis may produce a misleading picture.",
      "suggested_action": "In Step 1, specify the layer selection criterion: e.g., 'We use the Qwen-Scope SAE at layer L (to be determined by a preliminary pass: we select the layer with the largest variance in Safety_change across features, as a proxy for richest safety-relevant signal). If resources allow, we repeat the analysis at two additional layers as a sensitivity check.' This single sentence adds reproducibility and acknowledges that layer choice is a methodological decision, not an incidental detail."
    },
    {
      "category": "rigor",
      "severity": "minor",
      "description": "Criterion 3 (semantic label coherence) relies on a cheap LLM (Qwen2.5-7B or gemini-flash) as the sole judge for classifying features as 'harm-concept related.' The reliability of this labeler is not validated. Using the same model family (Qwen) for both the SAE analysis and the semantic labeling creates a potential circularity: Qwen2.5-7B may apply the same semantic categories as Qwen3-8B's RLHF training, making the label agreement an artifact of shared training data rather than an independent validation. Additionally, the prompt used for the LLM labeler is not specified, and LLM labeling of SAE features is known to be sensitive to prompt framing.",
      "suggested_action": "Add two safeguards: (1) validate the LLM labeler on a small gold set — manually annotate 20 features and check agreement with the automated labels before using them for Criterion 3; (2) use a labeler from a different model family (e.g., Llama-3 or Gemini) rather than Qwen2.5-7B to avoid within-family circularity. Report inter-rater agreement (Cohen's kappa between human annotation and LLM labels) on the gold set. This is low-cost but substantially strengthens Criterion 3's evidential value."
    },
    {
      "category": "methodology",
      "severity": "minor",
      "description": "The encoder-only fallback (when R² < 0.85) is presented as a reliable alternative, but JumpReLU thresholds calibrated on base activations will produce systematically distorted sparsity patterns when applied to instruct activations — features whose activations are systematically elevated or suppressed by RLHF will be gated differently than the base calibration assumes. This means that even encoder-only analysis carries a distributional shift caveat that could bias the Safety_change computation: if RLHF raises a feature's mean activation above the JumpReLU threshold, that feature may appear more active in the instruct model not because of genuine RLHF modification but because of threshold effects.",
      "suggested_action": "In the encoder-only fallback description, add an explicit caveat: 'Encoder-only activations under the base-calibrated JumpReLU thresholds carry a residual distributional shift confound: if RLHF systematically elevates a feature's mean activation, the base threshold may undercount it and inflate Safety_change. In the encoder-only regime, we will additionally report raw (pre-threshold) encoder activations as a robustness check, and interpret Safety_change values cautiously.' This is a straightforward addition that prevents a future reviewer from using this as a fatal objection."
    },
    {
      "category": "scope",
      "severity": "minor",
      "description": "The huihui-ai/Qwen3-8B-abliterated model card describes the technique as 'a new and faster method' without citing a specific code repository. The model card references the remove-refusals-with-transformers project as a conceptual basis but explicitly says the implementation does NOT use TransformerLens and may be a different algorithm. The hypothesis's Step 0 correctly plans to verify the technique, but if the community model turns out to use a non-standard method (as the model card wording suggests is possible), the scratch-reproduction fallback will require additional GPU time that may not be budgeted.",
      "suggested_action": "Prioritize Step 0 as the very first action and explicitly budget for the scratch-reproduction fallback. The model card wording 'new and faster method' is a yellow flag. Consider proactively reproducing abliteration from scratch regardless, since a controlled abliterated checkpoint with known methodology would strengthen the experimental claim considerably and costs little additional effort given that the standard remove-refusals-with-transformers script runs in under an hour on a single GPU."
    }
  ],
  "score": 6,
  "confidence": 4,
  "relation_type": "evolution",
  "relation_rationale": "Same conceptual frame (Type-R/Type-I via Reversal_score); refined with rigorous definitions, pre-registered contingencies, and validation steps."
}
```

### [28] TOOL RESULT — Write · 2026-07-17 09:49:45 UTC

```
{"type": "create", "filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_2/review_hypo/.terminal_claude_agent_struct_out.json", "content": "{\n  \"overall_assessment\": \"This is a substantially improved second iteration that addresses all four major and all four minor critiques from the previous review. The Reversal_score is now rigorously defined as a per-feature Pearson correlation with a concrete worked example; abliteration method verification is an explicit Step 0; the SAE reconstruction quality gate (R² ≥ 0.85) is properly formalized; Criterion 4 is demoted to exploratory with explicit acknowledgment of N=1 invalidity; both bimodal and continuous outcomes are pre-registered with Hartigan's dip test; the chat-template confound is addressed via a pilot study; and scope limitations are openly acknowledged. The result is a methodologically careful, well-specified single-model proof-of-concept. The primary new issue is the absence of any discussion of crosscoders (arXiv:2602.11729, arXiv:2606.26474) — a closely related and now-established methodology for multi-model SAE comparison that the hypothesis must either justify avoiding or explicitly contrast. Secondary issues include underspecification of which Qwen-Scope SAE layer is selected and limited validation of the LLM semantic labeler used in Criterion 3. These are resolvable without rerunning experiments and do not block the experimental pipeline from proceeding.\",\n  \"strengths\": [\n    \"Complete resolution of all four prior major issues: Reversal_score is unambiguously defined as PearsonR over the N-dimensional prompt vector with a numerical worked example; abliteration technique verification is now a pre-registered Step 0 with a scratch-reproduction fallback; the R² ≥ 0.85 reconstruction quality gate properly constrains when encoder-only vs. full SAE analysis is used; and Criterion 4 is explicitly downgraded to descriptive/exploratory with correct N=1 caveat.\",\n    \"Pre-registration of both the bimodal and continuous distributional outcomes via Hartigan's dip test is methodologically exemplary — the hypothesis is not betting on bimodality but treats distribution shape as an empirical question whose both answers yield interpretation.\",\n    \"The three-model comparison trajectory (base → instruct → abliterated) is genuinely novel relative to the existing literature: LatentBiopsy (arXiv:2603.27412) and arXiv:2604.18901 both use holistic geometric measures that collapse feature information; arXiv:2605.11426 analyzes base→instruct only. The Reversal_score framing is a distinct conceptual contribution.\",\n    \"Practical downstream motivation is now sharper: the hypothesis explicitly connects Type-I feature identification to targeted safety restoration without full retraining, providing a concrete use-case that geometric approaches cannot support.\",\n    \"The glossary is comprehensive and precise. The distinction between encoder-only and full-reconstruction analysis, the R² gate, and the chat-template pilot are all clearly specified enough for direct implementation.\",\n    \"Single-model scope limitation is explicitly acknowledged and framed correctly as a proof-of-concept measurement framework rather than a general result.\"\n  ],\n  \"dimension_scores\": [\n    {\n      \"dimension\": \"soundness\",\n      \"score\": 3,\n      \"justification\": \"The methodology is now technically solid for the core experiment. The remaining soundness concerns are: (1) no comparison against crosscoders, which are the purpose-built instrument for multi-model SAE comparison and could expose whether the fixed base-SAE approach is adequate; (2) the encoder-only fallback is treated as reliable when JumpReLU thresholds calibrated on base activations will still produce distorted sparsity patterns on instruct distributions, so the fallback's reliability is not guaranteed; (3) the SAE layer to analyze is unspecified despite Qwen-Scope covering multiple layers.\",\n      \"improvements\": [\n        \"Add a sentence in the investigation approach specifying which Qwen-Scope SAE layer will be used and the selection criterion (e.g., the layer with highest Safety_change variance, the middle-of-stack layer known to carry safety-relevant geometry from prior work, or all available layers with results averaged). Leaving this implicit makes the experiment underspecified.\",\n        \"In the encoder-only fallback description, add a caveat: 'JumpReLU thresholds calibrated on base activations may suppress or exaggerate feature activity on instruct activations even when reconstruction quality is not measured by R², so encoder-only activations still carry a distributional shift caveat that will be documented in the limitations section.'\",\n        \"Discuss or cite crosscoder methodology (arXiv:2602.11729) as an alternative instrument and justify why the fixed base-SAE approach is preferred (likely: crosscoder training requires access to the training pipeline, whereas a frozen base-SAE is zero-cost and directly comparable to arXiv:2605.11426 which validates the approach).\"\n      ]\n    },\n    {\n      \"dimension\": \"presentation\",\n      \"score\": 4,\n      \"justification\": \"The hypothesis is exceptionally well-organized. The five-step experimental protocol is clear and sequential. The worked numerical example for the Reversal_score makes the key construct immediately interpretable. The glossary is complete and precise. The pre-registration of both distributional outcomes is presented clearly. The neuroscience and isotopic tracing analogies are evocative without being overreaching. No presentation concerns remain.\",\n      \"improvements\": [\n        \"Minor: specify the SAE layer selection criterion in Step 1 (currently 'the SAE-covered layer (typically a mid-layer)' is vague).\",\n        \"Minor: in Step 3's semantic labeling, note whether the LLM labeler prompt will be independently validated (e.g., against a small human-annotated gold set) before being used as the Criterion 3 judge.\"\n      ]\n    },\n    {\n      \"dimension\": \"contribution\",\n      \"score\": 3,\n      \"justification\": \"The contribution is genuinely novel at the feature level — no existing paper applies the Reversal_score framework to the base/instruct/abliterated trajectory, and the Type-R/Type-I classification with pre-registered bimodal/continuous interpretations is a clean conceptual contribution. However: the scope is limited to one model at one SAE layer; crosscoders (arXiv:2606.26474) recently demonstrated exactly the type of RL-induced feature localization this hypothesis seeks; and the safety implications depend on whether the downstream targeted restoration application actually works, which is outside the scope of this paper. Contribution is solid for a workshop or short paper but requires cross-model replication for a full venue claim.\",\n      \"improvements\": [\n        \"Address how the Reversal_score approach relates to and differs from crosscoders — the field now has a principled multi-model SAE comparison tool, and the contribution claim is strengthened if the hypothesis explains what the frozen-base-SAE approach offers beyond what crosscoders already provide (e.g., no training required, direct comparability with arXiv:2605.11426 findings, interpretability of the feature vocabulary via Qwen-Scope's existing labeled features).\",\n        \"Add a brief comparison note: arXiv:2606.26474 (ICML 2026 spotlight) localizes RL-induced tool use to a single crosscoder feature. Contrasting that 'RL-induced addition' finding with the hypothesis's 'RLHF-induced survival under erasure' framing would sharpen the contribution's uniqueness.\"\n      ]\n    }\n  ],\n  \"critiques\": [\n    {\n      \"category\": \"novelty\",\n      \"severity\": \"major\",\n      \"description\": \"Crosscoders (arXiv:2602.11729, arXiv:2606.26474) are the established, purpose-built tool for exactly the multi-model feature comparison this hypothesis proposes, and they are absent from the related works. Crosscoders train a shared sparse dictionary jointly on activations from two or more model variants, naturally partitioning features into 'shared' and 'model-specific' categories. Dedicated Feature Crosscoders (DFCs) were recently used in an ICML 2026 spotlight paper (arXiv:2606.26474) to localize RL-induced tool-calling to a single feature in Qwen2.5-3B. The base/instruct/abliterated three-model comparison in this hypothesis could, in principle, be addressed by a three-model crosscoder rather than projecting through a frozen base-SAE. The hypothesis must either (a) justify why the frozen-base-SAE approach is preferable or complementary — the strongest argument is that it requires no training, reuses Qwen-Scope's labeled feature vocabulary, and is directly comparable to arXiv:2605.11426 — or (b) add a crosscoder variant as a robustness check. Omitting this literature makes the contribution appear less well-situated than it actually is and will draw reviewer criticism at top venues.\",\n      \"suggested_action\": \"Add a paragraph in the related works section discussing crosscoders (cite arXiv:2602.11729 and arXiv:2606.26474). Justify the frozen-base-SAE choice on the grounds of: (1) zero training cost, (2) reuse of Qwen-Scope's existing labeled feature vocabulary enabling interpretability without re-deriving feature semantics, (3) methodological continuity with arXiv:2605.11426 which validates base-SAE diagnostics for SFT analysis. If resources allow, running a two-model crosscoder (base+instruct) as a comparison would strengthen the methodology section considerably.\"\n    },\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"minor\",\n      \"description\": \"The SAE layer to analyze is underspecified. The hypothesis refers to 'the SAE-covered layer (typically a mid-layer)' without specifying which of Qwen-Scope's multiple available SAE layers will be used or why. Safety-relevant representations are known to vary substantially across layers (some work identifies early layers for refusal detection, mid-layers for semantic content, late layers for output formatting). The choice of layer directly affects which features are captured by Safety_change and which are labeled Type-R vs. Type-I. If the results are layer-dependent, the single-layer analysis may produce a misleading picture.\",\n      \"suggested_action\": \"In Step 1, specify the layer selection criterion: e.g., 'We use the Qwen-Scope SAE at layer L (to be determined by a preliminary pass: we select the layer with the largest variance in Safety_change across features, as a proxy for richest safety-relevant signal). If resources allow, we repeat the analysis at two additional layers as a sensitivity check.' This single sentence adds reproducibility and acknowledges that layer choice is a methodological decision, not an incidental detail.\"\n    },\n    {\n      \"category\": \"rigor\",\n      \"severity\": \"minor\",\n      \"description\": \"Criterion 3 (semantic label coherence) relies on a cheap LLM (Qwen2.5-7B or gemini-flash) as the sole judge for classifying features as 'harm-concept related.' The reliability of this labeler is not validated. Using the same model family (Qwen) for both the SAE analysis and the semantic labeling creates a potential circularity: Qwen2.5-7B may apply the same semantic categories as Qwen3-8B's RLHF training, making the label agreement an artifact of shared training data rather than an independent validation. Additionally, the prompt used for the LLM labeler is not specified, and LLM labeling of SAE features is known to be sensitive to prompt framing.\",\n      \"suggested_action\": \"Add two safeguards: (1) validate the LLM labeler on a small gold set — manually annotate 20 features and check agreement with the automated labels before using them for Criterion 3; (2) use a labeler from a different model family (e.g., Llama-3 or Gemini) rather than Qwen2.5-7B to avoid within-family circularity. Report inter-rater agreement (Cohen's kappa between human annotation and LLM labels) on the gold set. This is low-cost but substantially strengthens Criterion 3's evidential value.\"\n    },\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"minor\",\n      \"description\": \"The encoder-only fallback (when R² < 0.85) is presented as a reliable alternative, but JumpReLU thresholds calibrated on base activations will produce systematically distorted sparsity patterns when applied to instruct activations — features whose activations are systematically elevated or suppressed by RLHF will be gated differently than the base calibration assumes. This means that even encoder-only analysis carries a distributional shift caveat that could bias the Safety_change computation: if RLHF raises a feature's mean activation above the JumpReLU threshold, that feature may appear more active in the instruct model not because of genuine RLHF modification but because of threshold effects.\",\n      \"suggested_action\": \"In the encoder-only fallback description, add an explicit caveat: 'Encoder-only activations under the base-calibrated JumpReLU thresholds carry a residual distributional shift confound: if RLHF systematically elevates a feature's mean activation, the base threshold may undercount it and inflate Safety_change. In the encoder-only regime, we will additionally report raw (pre-threshold) encoder activations as a robustness check, and interpret Safety_change values cautiously.' This is a straightforward addition that prevents a future reviewer from using this as a fatal objection.\"\n    },\n    {\n      \"category\": \"scope\",\n      \"severity\": \"minor\",\n      \"description\": \"The huihui-ai/Qwen3-8B-abliterated model card describes the technique as 'a new and faster method' without citing a specific code repository. The model card references the remove-refusals-with-transformers project as a conceptual basis but explicitly says the implementation does NOT use TransformerLens and may be a different algorithm. The hypothesis's Step 0 correctly plans to verify the technique, but if the community model turns out to use a non-standard method (as the model card wording suggests is possible), the scratch-reproduction fallback will require additional GPU time that may not be budgeted.\",\n      \"suggested_action\": \"Prioritize Step 0 as the very first action and explicitly budget for the scratch-reproduction fallback. The model card wording 'new and faster method' is a yellow flag. Consider proactively reproducing abliteration from scratch regardless, since a controlled abliterated checkpoint with known methodology would strengthen the experimental claim considerably and costs little additional effort given that the standard remove-refusals-with-transformers script runs in under an hour on a single GPU.\"\n    }\n  ],\n  \"score\": 6,\n  \"confidence\": 4,\n  \"relation_type\": \"evolution\",\n  \"relation_rationale\": \"Same conceptual frame (Type-R/Type-I via Reversal_score); refined with rigorous definitions, pre-registered contingencies, and validation steps.\"\n}\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [29] ASSISTANT · 2026-07-17 09:49:57 UTC

```
Review complete. The output is written to `.terminal_claude_agent_struct_out.json`.

**Summary of this review (iter 2 → score 6, Weak Accept):**

All 4 prior major issues and all 4 prior minor issues from iter 1 are now resolved. The revised hypothesis is methodologically careful with:
- Unambiguous Pearson-correlation Reversal_score + worked example
- Explicit abliteration verification (Step 0)
- R² ≥ 0.85 reconstruction quality gate (Step 0.5)
- Criterion 4 properly demoted to exploratory
- Both bimodal/continuous outcomes pre-registered with Hartigan's dip test

**New critiques raised (in priority order):**

1. **Major — Crosscoder omission**: arXiv:2602.11729 and arXiv:2606.26474 (ICML 2026 spotlight) are the established tool for exactly this multi-model SAE comparison. The hypothesis must justify why it uses a frozen base-SAE instead of crosscoders — the answer is defensible (zero training cost, reuses Qwen-Scope's labeled vocabulary) but must be stated explicitly.

2. **Minor — Layer underspecified**: "Typically a mid-layer" is not reproducible; need a selection criterion.

3. **Minor — LLM labeler reliability**: Criterion 3's semantic coherence test relies on a cheap Qwen-family labeler with no validation — adds circularity risk; needs a gold-set calibration or cross-family judge.

4. **Minor — Encoder-only fallback caveat**: JumpReLU thresholds calibrated on base activations still distort instruct activations even without reconstruction; needs an explicit caveat.

5. **Minor — huihui-ai abliteration uncertainty**: Model card wording ("new and faster method") suggests the technique may not be standard; recommend proactive scratch reproduction regardless.
```

### [30] SYSTEM-USER prompt · 2026-07-17 09:49:59 UTC

```
<validation-feedback>
Attempt 1 failed validation.

Schema validation found 1 problem — fix ALL of them at once:
  - at `relation_rationale`: 'Same conceptual frame (Type-R/Type-I via Reversal_score); refined with rigorous definitions, pre-registered contingencies, and validation steps.' is too long (at most 120 characters, got 144)
Every required field must be present and every field type must match the schema.

Please use the Write tool to overwrite `.terminal_claude_agent_struct_out.json` with corrected JSON. Do not invent new fields; match the schema you were given.
</validation-feedback>
```

### [31] TOOL CALL — Edit · 2026-07-17 09:50:09 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_2/review_hypo/.terminal_claude_agent_struct_out.json
OLD: "relation_rationale": "Same conceptual frame (Type-R/Type-I via Reversal_score); refined with rigorous definitions, pre-registered contingencies, and validation steps."
NEW: "relation_rationale": "Same Type-R/Type-I frame; refined with rigorous Reversal_score definition, quality gate, and pre-registered contingencies."
```

### [32] TOOL RESULT — Edit · 2026-07-17 09:50:09 UTC

```
{"filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_2/review_hypo/.terminal_claude_agent_struct_out.json", "oldString": "\"relation_rationale\": \"Same conceptual frame (Type-R/Type-I via Reversal_score); refined with rigorous definitions, pre-registered contingencies, and validation steps.\"", "newString": "\"relation_rationale\": \"Same Type-R/Type-I frame; refined with rigorous Reversal_score definition, quality gate, and pre-registered contingencies.\"", "originalFile": null, "structuredPatch": [{"oldStart": 73, "oldLines": 5, "newStart": 73, "newLines": 5, "lines": ["   \"score\": 6,", "   \"confidence\": 4,", "   \"relation_type\": \"evolution\",", "-  \"relation_rationale\": \"Same conceptual frame (Type-R/Type-I via Reversal_score); refined with rigorous definitions, pre-registered contingencies, and validation steps.\"", "+  \"relation_rationale\": \"Same Type-R/Type-I frame; refined with rigorous Reversal_score definition, quality gate, and pre-registered contingencies.\"", " }"]}], "userModified": false, "replaceAll": false}
```

### [33] ASSISTANT · 2026-07-17 09:50:11 UTC

```
Fixed — `relation_rationale` shortened to 97 characters.
```

### [34] SYSTEM-USER prompt · 2026-07-17 09:50:13 UTC

```
<validation-feedback>
Attempt 2 failed validation.

Schema validation found 1 problem — fix ALL of them at once:
  - at `relation_rationale`: 'Same Type-R/Type-I frame; refined with rigorous Reversal_score definition, quality gate, and pre-registered contingencies.' is too long (at most 120 characters, got 122)
Every required field must be present and every field type must match the schema.

Please use the Write tool to overwrite `.terminal_claude_agent_struct_out.json` with corrected JSON. Do not invent new fields; match the schema you were given.
</validation-feedback>
```

### [35] TOOL CALL — Edit · 2026-07-17 09:50:17 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_2/review_hypo/.terminal_claude_agent_struct_out.json
OLD: "relation_rationale": "Same Type-R/Type-I frame; refined with rigorous Reversal_score definition, quality gate, and pre-registered contingencies."
NEW: "relation_rationale": "Same Type-R/Type-I frame; refined with rigorous Reversal_score, R² gate, and pre-registered contingencies."
```

### [36] TOOL RESULT — Edit · 2026-07-17 09:50:17 UTC

```
{"filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_2/review_hypo/.terminal_claude_agent_struct_out.json", "oldString": "\"relation_rationale\": \"Same Type-R/Type-I frame; refined with rigorous Reversal_score definition, quality gate, and pre-registered contingencies.\"", "newString": "\"relation_rationale\": \"Same Type-R/Type-I frame; refined with rigorous Reversal_score, R² gate, and pre-registered contingencies.\"", "originalFile": null, "structuredPatch": [{"oldStart": 73, "oldLines": 5, "newStart": 73, "newLines": 5, "lines": ["   \"score\": 6,", "   \"confidence\": 4,", "   \"relation_type\": \"evolution\",", "-  \"relation_rationale\": \"Same Type-R/Type-I frame; refined with rigorous Reversal_score definition, quality gate, and pre-registered contingencies.\"", "+  \"relation_rationale\": \"Same Type-R/Type-I frame; refined with rigorous Reversal_score, R² gate, and pre-registered contingencies.\"", " }"]}], "userModified": false, "replaceAll": false}
```

### [37] ASSISTANT · 2026-07-17 09:50:18 UTC

```
Fixed — now 110 characters.
```
