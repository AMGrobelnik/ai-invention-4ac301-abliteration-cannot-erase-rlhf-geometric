# review_paper — test_idea

> Phase: `invention_loop` · round 5 · `review_paper`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim transcript of this agent task — every system/user prompt, assistant response, thinking block, tool call and tool result — in the order they occurred. Nothing truncated.

## Task: `review_paper` (terminal_claude_agent, claude-sonnet-4-6)

### [1] CONFIG · 2026-07-17 14:57:19 UTC

```
model: claude-sonnet-4-6 | effort: medium | permission: bypassPermissions | cwd: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/review_paper/review_paper
```

### [2] SYSTEM-USER prompt · 2026-07-17 14:57:25 UTC

````
<ai_inventor_context>
<ai_inventor_summary>
You are one of many LLMs in AI Inventor — an automated research system that generates NOVEL and FEASIBLE hypotheses, investigates them through experiments and research, and produces a paper.

Your output feeds other LLMs downstream. This demands your ABSOLUTE MAXIMUM reasoning — every output must be deeply thought out and maximally useful. Surface-level responses waste downstream computation.
</ai_inventor_summary>

<your_role>
YOU ARE: An adversarial paper reviewer (Step 3.5: REVIEW_PAPER in the invention loop)

You received a paper draft written by a DIFFERENT model. Review it with fresh eyes.
Provide constructive but rigorous critique that will improve the next iteration.

Specific critiques → better paper. Vague praise → no improvement.
</your_role>
</ai_inventor_context>

ROLE: You are a very experienced and critical conference reviewer.
Your expertise spans the domain of the paper under review.
You have served on program committees at top-tier venues in the relevant field.

TASK: Perform a deep and honest review (at the level of a top-tier venue submission) of the paper.

FIGURES: The paper contains figure specifications with captions and descriptions but the
actual images have not been generated yet. Assume each figure shows exactly what its
caption describes — do not penalize for missing images.

ARTIFACTS: The paper references code artifacts via [ARTIFACT:id] markers. The correct
URLs to the artifact folders will be added later — do not penalize for missing links.

GOAL: Your review feeds directly back to the paper author. The objective is to maximize
the overall review score in subsequent rounds. Every piece of feedback you give should
be written with this goal in mind — prioritize the critiques and suggestions that would
produce the largest score improvement if addressed. Don't waste the author's iteration
budget on low-impact polish when there are score-blocking issues to fix.

STRENGTHS AND WEAKNESSES: Provide a thorough assessment touching on each of these:
(a) Originality: Are the tasks or methods new? Novel combination of known techniques?
    Clear differentiation from prior work? Is related work adequately cited?
(b) Quality: Is the submission technically sound? Are claims well supported by theoretical
    analysis or experimental results? Is the methodology appropriate? Is this a complete
    piece of work? Are the authors honest about limitations?
(c) Clarity: Is the submission clearly written and well organized? Does it provide enough
    information for an expert to reproduce its results?
(d) Significance: Are the results important? Would others build on them? Does it address
    a meaningful problem better than prior work? Does it advance the state of the art?

SUPPLEMENTARY SCORES: Rate each on a 1-4 scale.
Soundness (1-4) — soundness of the technical claims, experimental and research methodology,
and whether central claims are adequately supported with evidence:
  4: excellent  3: good  2: fair  1: poor
Presentation (1-4) — quality of writing, clarity, and contextualization relative to prior work:
  4: excellent  3: good  2: fair  1: poor
Contribution (1-4) — quality of the overall contribution, importance of questions asked,
originality of ideas and execution, value to the broader research community:
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
- Distinguish major issues (would cause rejection) from minor issues (polish)
- Acknowledge genuine strengths — don't be negative for its own sake
- Compare against the bar set by accepted papers at top-tier venues
- Check if figures are well-specified and would effectively communicate the results
- Verify that claims are supported by the artifacts described

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

<paper>
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
</paper>

<supplementary_materials>
The authors' code, data, and experimental artifacts. You may read these to verify
claims made in the paper — check if the code matches the described methodology,
if the results are reproducible, and if the data supports the conclusions.

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
</supplementary_materials>

<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for judging whether the paper's contribution is genuinely novel versus already-done or a known dead end in this field.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>

<previous_review>
Your review from the previous iteration. Check which critiques have been addressed
in the revised paper. Do NOT re-raise critiques that have been adequately fixed.
Only re-raise if the fix is insufficient.

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
</previous_review>

<task>
Review this paper as you would for a top-tier venue submission.

STEP 1 — READ THE PAPER: Read it carefully. Note claims, methodology, and results.

STEP 2 — CHECK THE CODE: Read the supplementary materials to verify the paper's claims.
Do the experiments match what's described? Are there discrepancies between code and paper?

STEP 3 — SEARCH THE LITERATURE: Ground your review in evidence.
- Search for the closest existing work — is this genuinely novel or incremental?
- Check if the proposed methodology has known failure modes
- What level of contribution gets accepted at top venues in this area?

STEP 4 — WRITE YOUR REVIEW:
For each critique:
1. Categorize: methodology, evidence, novelty, clarity, scope, or rigor
2. Rate severity: major (would cause rejection) or minor (polish)
3. Describe the issue clearly
4. Suggest a concrete action to address it

Focus on the most impactful issues. Provide your review via structured output.
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
  "description": "Adversarial review of the paper draft.\n\nID format: review_it{iteration}__{model}",
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
    }
  },
  "required": [
    "overall_assessment",
    "strengths",
    "critiques",
    "score"
  ],
  "title": "ReviewerFeedback",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````

### [3] HUMAN-USER prompt · 2026-07-17 14:57:25 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [4] ASSISTANT · 2026-07-17 14:58:31 UTC

```
I'll review this paper carefully, checking the methodology, supplementary materials, and literature grounding.
```

### [5] TOOL CALL — Skill · 2026-07-17 14:58:31 UTC

```
aii-web-tools
```

### [6] TOOL RESULT — Skill · 2026-07-17 14:58:31 UTC

```
{"success": true, "commandName": "aii-web-tools"}
```

### [7] ASSISTANT · 2026-07-17 14:58:54 UTC

```
Let me check key references and the supplementary materials to ground my review.
```

### [8] TOOL CALL — ToolSearch · 2026-07-17 14:58:54 UTC

```
{'query': 'select:WebSearch,WebFetch'}
```

### [9] TOOL RESULT — ToolSearch · 2026-07-17 14:58:54 UTC

```
{"matches": ["WebSearch", "WebFetch"], "query": "select:WebSearch,WebFetch", "total_deferred_tools": 21}
```

### [10] TOOL CALL — Read · 2026-07-17 14:58:54 UTC

```
/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/review_paper/review_paper
```

### [11] TOOL RESULT — Read [ERROR] · 2026-07-17 14:58:54 UTC

```
Error: EISDIR: illegal operation on a directory, read '/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/review_paper/review_paper'
```

### [12] TOOL CALL — WebSearch · 2026-07-17 14:59:04 UTC

```
Joad refusal LLM more than single direction 2602.02132 2026
```

### [13] TOOL CALL — WebSearch · 2026-07-17 14:59:04 UTC

```
Arditi refusal language models mediated single direction NeurIPS 2024
```

### [14] TOOL RESULT — WebSearch · 2026-07-17 14:59:10 UTC

```
{"query": "Arditi refusal language models mediated single direction NeurIPS 2024", "results": [{"tool_use_id": "srvtoolu_018Hgtp82cf51BHkjEygicgu", "content": [{"title": "dblp: Refusal in Language Models Is Mediated by a Single Direction.", "url": "https://dblp.org/rec/conf/nips/ArditiOSPPGN24.html"}, {"title": "Refusal in Language Models Is Mediated by a Single Direction Andy Arditi∗", "url": "https://arxiv.org/pdf/2406.11717"}, {"title": "Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in Language Models", "url": "https://arxiv.org/pdf/2606.24952"}, {"title": "A Geometric Account of Activation Steering through Angle-Norm Decomposition", "url": "https://arxiv.org/pdf/2606.06735"}, {"title": "When Behavioral Safety Evaluation Fails: A Representation-Level Perspective", "url": "https://arxiv.org/pdf/2606.08044"}, {"title": "Refusal Lives Downstream of Persona in Chat Models", "url": "https://arxiv.org/pdf/2606.26161"}, {"title": "On the Limitations of Steering in Language Model Alignment", "url": "https://arxiv.org/pdf/2505.01162"}, {"title": "Controlling Large Language Model Agents with Entropic Activation   Steering", "url": "https://arxiv.org/pdf/2406.00244"}, {"title": "A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models", "url": "https://arxiv.org/pdf/2605.08513"}, {"title": "Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs", "url": "https://arxiv.org/pdf/2603.27518"}]}, "I found the paper you're looking for! \n\nThe paper is titled \"Refusal in Language Models Is Mediated by a Single Direction\" by Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda, and it was published at NeurIPS 2024.\n\nThe work shows that refusal is mediated by a one-dimensional subspace across 13 popular open-source chat models up to 72B parameters in size. The authors demonstrate that this single direction is both necessary and sufficient for refusal—ablating this direction effectively disables the model's ability to refuse harmful requests, while adding it to the model's activations induces refusal on harmless instructions.\n\nThe paper is available on arXiv (2406.11717) and represents an important finding in understanding the mechanistic basis of safety mechanisms in language models."], "durationSeconds": 7.1944078239999945, "searchCount": 1}
```

### [15] TOOL RESULT — WebSearch · 2026-07-17 14:59:10 UTC

```
{"query": "Joad refusal LLM more than single direction 2602.02132 2026", "results": [{"tool_use_id": "srvtoolu_01KeAH9FPi1JRruhMnkRUZQZ", "content": [{"title": "[2602.02132] There Is More to Refusal in Large Language Models than a Single Direction", "url": "https://arxiv.org/abs/2602.02132"}, {"title": "There Is More to Refusal in Large Language Models than a Single Direction", "url": "https://www.arxiv.org/pdf/2602.02132"}, {"title": "There Is More to Refusal in Large Language Models than a Single Direction", "url": "https://arxiv.org/html/2602.02132v1"}, {"title": "Untitled Document", "url": "https://arxiv.org/html/2602.04521v2"}, {"title": "Different Paths to Harmful Compliance: Behavioral Side Effects and Mechanistic Divergence Across LLM Jailbreaks", "url": "https://arxiv.org/html/2604.18510"}, {"title": "There Is More to Refusal in Large Language Models than a Single Direction - ADS", "url": "https://ui.adsabs.harvard.edu/abs/2026arXiv260202132J/abstract"}, {"title": "Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence", "url": "https://arxiv.org/html/2604.08169v1"}, {"title": "Computer Science", "url": "https://www.arxiv.org/list/cs/recent?skip=2353&show=2000"}, {"title": "Computation and Language Feb 2026", "url": "https://arxiv.org/list/cs.CL/current?skip=135&show=50"}, {"title": "[2406.11717] Refusal in Language Models Is Mediated by a Single Direction", "url": "https://arxiv.org/abs/2406.11717"}]}, "I found the paper you're looking for! Here's what the search revealed:\n\n## Paper Details\n\nThe paper is titled \"There Is More to Refusal in Large Language Models than a Single Direction\" by authors Faaiz Joad, Majd Hawasly, Sabri Boughorbel, Nadir Durrani, and Husrev Taha Sencar, submitted on February 2, 2026.\n\n## Key Findings\n\nPrior work argues that refusal in large language models is mediated by a single activation-space direction, enabling effective steering and ablation. The paper shows that this account is incomplete.\n\nAcross eleven categories of refusal and non-compliance, including safety, incomplete or unsupported requests, anthropomorphization, and over-refusal, the authors find that these refusal behaviors correspond to geometrically distinct directions in activation space.\n\nDespite this diversity, linear steering along any refusal-related direction produces nearly identical refusal to over-refusal trade-offs, acting as a shared one-dimensional constraint.\n\nThe paper challenges the prevailing understanding that refusal mechanisms in LLMs operate through a single linear direction, demonstrating instead that multiple geometric directions underlie different types of refusal behavior."], "durationSeconds": 7.711615226000009, "searchCount": 1}
```

### [16] TOOL CALL — WebSearch · 2026-07-17 14:59:14 UTC

```
Yeo understanding refusal sparse autoencoders SAE 2025 EMNLP
```

### [17] TOOL CALL — WebSearch · 2026-07-17 14:59:14 UTC

```
abliteration RLHF SAE features preserved mechanistic interpretability 2026
```

### [18] TOOL RESULT — WebSearch · 2026-07-17 14:59:24 UTC

```
{"query": "abliteration RLHF SAE features preserved mechanistic interpretability 2026", "results": [{"tool_use_id": "srvtoolu_017DUtXZdK7TpHSphgATE47s", "content": [{"title": "Published as a conference paper at ICLR 2026 MECHANISTIC INTERPRETABILITY", "url": "https://arxiv.org/pdf/2510.02917"}, {"title": "LLM Refusal Abliteration Mechanisms", "url": "https://www.emergentmind.com/topics/refusal-abliteration"}, {"title": "Interpretable Steering of Large Language Models with Feature Guided   Activation Additions", "url": "https://arxiv.org/pdf/2501.09929"}, {"title": "SAEExplainer: Interpreting SAE Features with Activation-Guided Preference Optimization", "url": "https://arxiv.org/pdf/2606.08496"}, {"title": "Inference-Time Decomposition of Activations (ITDA): A Scalable Approach to Interpreting Large Language Models", "url": "https://arxiv.org/pdf/2505.17769"}, {"title": "Mechanistic origins of catastrophic forgetting: why RL preserves circuits better than SFT?", "url": "https://arxiv.org/pdf/2605.28860"}, {"title": "Interpretability as Alignment: Making Internal Understanding a Design Principle", "url": "https://arxiv.org/pdf/2509.08592"}, {"title": "Data-Centric Interpretability for LLM-based Multi-Agent Reinforcement Learning", "url": "https://arxiv.org/pdf/2602.05183"}, {"title": "Do Sparse Autoencoders Identify Reasoning Features in Language Models?", "url": "https://arxiv.org/pdf/2601.05679"}, {"title": "OBLITERATUS: Mapping the Geometry of Refusal Inside Large Language Models", "url": "https://themenonlab.blog/blog/obliteratus-abliteration-llm-refusal-removal/"}]}, "Based on the search results, I found relevant information about these topics:\n\n## Abliteration and RLHF\n\nRefusal abliteration is defined as the targeted removal of low-dimensional refusal features in LLMs, effectively reducing their safety responses. This technique exploits interpretable activation patterns in model residual streams to neutralize safety measures while largely preserving overall performance.\n\nWhen comparing the internal activations of a chat-tuned LLM responding to restricted prompts versus unrestricted prompts, the difference collapses into a single direction — a refusal direction — that is consistent across layers. This means safety alignment via RLHF and instruction tuning doesn't create a complex web of behavioral constraints, but rather a geometric feature: a vector in the model's residual stream.\n\n## SAE Features and Mechanistic Interpretability\n\nSAEs project dense activations into higher-dimensional sparse latent spaces and thereby recover more nearly monosemantic features. However, while SAEs decompose into sparse features, they do not fully solve polysemanticity.\n\n## Features Preserved in RLHF\n\nRelated work suggests that fine-tuning frequently modifies existing mechanisms rather than replacing them entirely, and that some circuits are alignment-critical and should be preferentially preserved during model modification.\n\nThe search results contain references to 2025 and 2026 research on these topics, indicating this is an active area of investigation in the mechanistic interpretability community."], "durationSeconds": 9.427763353999994, "searchCount": 1}
```

### [19] TOOL RESULT — WebSearch · 2026-07-17 14:59:24 UTC

```
{"query": "Yeo understanding refusal sparse autoencoders SAE 2025 EMNLP", "results": [{"tool_use_id": "srvtoolu_01DXgMxsqiBBtKrCEywAvxhF", "content": [{"title": "Understanding Refusal in Language Models with Sparse Autoencoders - ACL Anthology", "url": "https://aclanthology.org/2025.findings-emnlp.338/"}, {"title": "Understanding Refusal in Language Models with Sparse Autoencoders", "url": "https://www.researchgate.net/publication/397420962_Understanding_Refusal_in_Language_Models_with_Sparse_Autoencoders"}, {"title": "Understanding Refusal in Language Models with Sparse ...", "url": "https://aclanthology.org/2025.findings-emnlp.338.pdf"}, {"title": "Understanding Refusal in Language Models with Sparse Autoencoders", "url": "https://arxiv.org/html/2505.23556v1"}, {"title": "Discovering Millions of Interpretable Features with Sparse Autoencoders", "url": "https://arxiv.org/pdf/2606.26620"}, {"title": "[2505.23556] Understanding Refusal in Language Models with Sparse Autoencoders", "url": "https://arxiv.org/abs/2505.23556"}, {"title": "Understanding Refusal in Language Models with Sparse Autoencoders", "url": "https://arxiv.org/pdf/2505.23556"}, {"title": "There Is More to Refusal in Large Language Models than a Single Direction", "url": "https://arxiv.org/pdf/2602.02132"}, {"title": "Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal", "url": "https://arxiv.org/pdf/2509.09708"}, {"title": "A Survey on Sparse Autoencoders: Interpreting the Internal ...", "url": "https://aclanthology.org/2025.findings-emnlp.89.pdf"}]}, "I found the paper you're looking for! Here are the key details:\n\n## Paper Summary\n\n\"Understanding Refusal in Language Models with Sparse Autoencoders\" was published in the Findings of the Association for Computational Linguistics: EMNLP 2025, pages 6377–6399, in Suzhou, China.\n\n**Authors:** Wei Jie Yeo, Nirmalendu Prakash, Clement Neo, Ranjan Satapathy, Roy Ka-Wei Lee, and Erik Cambria\n\n## Research Overview\n\nThe paper conducts a mechanistic study of refusal in instruction-tuned LLMs using sparse autoencoders to identify latent features that causally mediate refusal behaviors. \n\nThe researchers apply their method to two open-source chat models and intervene on refusal-related features to assess their influence on generation, validating their behavioral impact across multiple harmful datasets. This enables a fine-grained inspection of how refusal manifests at the activation level and addresses key research questions such as investigating upstream-downstream latent relationship and understanding the mechanisms of adversarial jailbreaking techniques.\n\n**Additional Resources:** The authors have open-sourced their code at https://github.com/wj210/refusal_sae."], "durationSeconds": 10.56615337, "searchCount": 1}
```

### [20] TOOL CALL — WebSearch · 2026-07-17 14:59:30 UTC

```
LatentBiopsy geometry harmful intent angular deviation LLM residual streams 2603.27412 2026
```

### [21] TOOL CALL — WebSearch · 2026-07-17 14:59:30 UTC

```
Chopra mechanistic investigation supervised fine tuning SAE crosscoder 2605.11426 2026
```

### [22] TOOL RESULT — WebSearch · 2026-07-17 14:59:40 UTC

```
{"query": "Chopra mechanistic investigation supervised fine tuning SAE crosscoder 2605.11426 2026", "results": [{"tool_use_id": "srvtoolu_01MzfM5Ts29sB1FK7MXUppFZ", "content": [{"title": "A Mechanistic Investigation of Supervised Fine Tuning", "url": "https://arxiv.org/html/2605.11426v1"}, {"title": "Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes", "url": "https://arxiv.org/pdf/2603.04426"}, {"title": "A Layer-wise Analysis of Supervised Fine-Tuning", "url": "https://arxiv.org/html/2604.11838"}, {"title": "Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes This paper contains text that might be offensive.", "url": "https://arxiv.org/html/2603.04426v1"}, {"title": "Mechanistic Interpretability of Code Correctness in LLMs via Sparse Autoencoders", "url": "https://arxiv.org/pdf/2510.02917"}, {"title": "Towards Isolated Interventions via Almost Orthogonal Features in Language Models", "url": "https://arxiv.org/pdf/2602.04718"}, {"title": "Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/pdf/2605.11887"}, {"title": "From Tokens to Concepts: Leveraging SAE for SPLADE", "url": "https://arxiv.org/pdf/2604.21511"}]}, "Based on the search results, I found the paper you're looking for:\n\n**\"A Mechanistic Investigation of Supervised Fine Tuning\"** by Ruhaan Chopra (arXiv:2605.11426v1, published May 12, 2026)\n\n**Overview:**\n\nThe paper examines how the cosine similarity between a language model's hidden activations before and after Supervised Fine-Tuning remains very high, which at first glance suggests that SFT leaves the model's activation geometry largely undisturbed. However, projecting both sets of activations through a Sparse Autoencoder (SAE) pretrained on the base model reveals that the underlying sparse latents diverge significantly.\n\n**Key Methodology:**\n\nThe research introduces a novel investigative pipeline which utilizes pretrained SAEs as a high-resolution diagnostic tool to mechanistically investigate the drivers of representational divergence. The study compares hidden states before and after fine-tuning across four independently fine-tuned variants of Gemma 3 1B, where each variant is initialized from the same base model and trained on a single isolated task.\n\n**Key Findings:**\n\nThrough their analytical pipeline, the researchers discover task-specific and layer-specific distributions of the precise semantic features that are systematically altered during supervised fine-tuning."], "durationSeconds": 9.235644910000003, "searchCount": 1}
```

### [23] TOOL RESULT — WebSearch · 2026-07-17 14:59:40 UTC

```
{"query": "LatentBiopsy geometry harmful intent angular deviation LLM residual streams 2603.27412 2026", "results": [{"tool_use_id": "srvtoolu_01VmgnZ2vp5nHHLNgixMGP4V", "content": [{"title": "[2603.27412] The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/abs/2603.27412"}, {"title": "The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/html/2603.27412"}, {"title": "Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams", "url": "https://arxiv.org/html/2604.18901"}, {"title": "The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/pdf/2603.27412"}, {"title": "[2604.18901] Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams", "url": "https://arxiv.org/abs/2604.18901"}, {"title": "Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams", "url": "https://arxiv.org/pdf/2604.18901"}]}, "Based on the search results, here's what I found:\n\n## LatentBiopsy: Training-Free Harmful Prompt Detection\n\nLatentBiopsy is a training-free method for detecting harmful prompts by analysing the geometry of residual-stream activations in large language models.\n\n### How It Works\n\nGiven 200 safe normative prompts, LatentBiopsy computes the leading principal component of their activations at a target layer and characterises new prompts by their radial deviation angle θ from this reference direction. The anomaly score is the negative log-likelihood of θ under a Gaussian fit to the normative distribution, flagging deviations symmetrically regardless of orientation.\n\n### Performance\n\nAcross all six variants, LatentBiopsy achieves AUROC ≥ 0.937 for harmful-vs-normative detection and AUROC = 1.000 for discriminating harmful from benign-aggressive prompts (XSTest), with sub-millisecond per-query overhead.\n\n### Key Findings\n\nGeometry survives refusal ablation: both abliterated variants achieve AUROC at most 0.015 below their instruction-tuned counterparts, establishing a geometric dissociation between harmful-intent representation and the downstream generative refusal mechanism. Additionally, harmful prompts exhibit a near-degenerate angular distribution (σ_θ ≈ 0.03 rad), an order of magnitude tighter than the normative distribution (σ_θ ≈ 0.27 rad), preserved across all alignment stages including abliteration.\n\nThe paper (arXiv:2603.27412) was submitted on March 28, 2026."], "durationSeconds": 9.729887465000006, "searchCount": 1}
```

### [24] TOOL CALL — WebSearch · 2026-07-17 14:59:46 UTC

```
Qwen-Scope SAE sparse autoencoder Qwen3-8B mechanistic interpretability 2605.11887
```

### [25] TOOL CALL — WebSearch · 2026-07-17 14:59:46 UTC

```
refusal abliteration SAE feature survival RLHF imprint weight editing 2025 2026
```

### [26] TOOL RESULT — WebSearch · 2026-07-17 14:59:54 UTC

```
{"query": "refusal abliteration SAE feature survival RLHF imprint weight editing 2025 2026", "results": [{"tool_use_id": "srvtoolu_01WioC5Eu8qmYbMRVjKMefTd", "content": [{"title": "LLM Refusal Abliteration Mechanisms", "url": "https://www.emergentmind.com/topics/refusal-abliteration"}, {"title": "An Embarrassingly Simple Defense Against LLM Abliteration Attacks", "url": "https://arxiv.org/html/2505.19056v2"}, {"title": "CRaFT: Circuit-Guided Refusal Feature Selection via Cross-Layer Transcoders", "url": "https://arxiv.org/pdf/2604.01604"}, {"title": "Abliterated Models Guide - Qwen 3.6, Gemma 4 Heretic, Llama 3.1 Uncensored Download Links - DEV Community", "url": "https://dev.to/purpledoubled/abliterated-models-guide-qwen-36-gemma-4-heretic-llama-31-uncensored-download-links-1f4e"}, {"title": "Measuring & Mitigating Over-Alignment for LLMs in Multilingual Criminal Law Courts", "url": "https://arxiv.org/pdf/2606.23375"}, {"title": "On the Failure of Topic-Matched Contrast Baselines in Multi-Directional Refusal Abliteration", "url": "https://arxiv.org/pdf/2603.22061"}, {"title": "Different Paths to Harmful Compliance: Behavioral Side Effects and Mechanistic Divergence Across LLM Jailbreaks", "url": "https://arxiv.org/pdf/2604.18510"}, {"title": "LoMC: Localized Multidirectional Correction for Refusal Suppression in Routed Foundation Models", "url": "https://arxiv.org/pdf/2606.13709"}, {"title": "What is Abliteration - OBLITERATUS", "url": "https://mintlify.wiki/elder-plinius/OBLITERATUS/concepts/abliteration"}]}, "I've found comprehensive information about refusal abliteration and related research. Here are the key findings:\n\n## Refusal Abliteration Overview\n\nRefusal abliteration is defined as the targeted removal of low-dimensional refusal features in LLMs, effectively reducing their safety responses. The direction was put there during RLHF. You take a corpus of refused prompts, isolate the activation pattern that distinguishes them from accepted prompts, then project that direction out of every weight matrix.\n\n## SAE Features and Survival\n\nRefusal-SAE is an SAE-based steering baseline that identifies refusal-related sparse features and suppresses refusal by ablating them at inference time. Research on feature survival shows defensive approaches are emerging: Extended-refusal fine-tuning trains models on richer refusal responses featuring a neutral overview, explicit refusal, and ethical rationale, thereby distributing the refusal signal across several latent dimensions. When abliteration is applied, these models maintain refusal rates above 90%, compared to baseline models with rates reduced by 70–80%.\n\n## Weight Editing and RLHF Imprinting\n\nAbliteration identifies a refusal-related direction from harmful-harmless activation contrasts and suppresses that direction by orthogonalizing selected writer weights. Unlike RLVR and SFT, abliteration does not rely on gradient-based behavioral training but instead suppresses a refusal-mediating feature through direct weight-space intervention.\n\n## Recent 2025-2026 Developments\n\nResearch is actively exploring multi-directional approaches and routed models: Heretic-MoE provides strong single-direction baselines for post-training intervention. In routed models, however, applying one direction across many layers or components creates a broad editable surface and can interfere with general capability preservation."], "durationSeconds": 10.170418318999989, "searchCount": 1}
```

### [27] TOOL RESULT — WebSearch · 2026-07-17 14:59:56 UTC

```
{"query": "Qwen-Scope SAE sparse autoencoder Qwen3-8B mechanistic interpretability 2605.11887", "results": [{"tool_use_id": "srvtoolu_016ZsfG3MQg1cxS5q5VXcL2R", "content": [{"title": "Qwen AI Releases Qwen-Scope: An Open-Source Sparse AutoEncoders (SAE) Suite That Turns LLM Internal Features into Practical Development Tools - MarkTechPost", "url": "https://www.marktechpost.com/2026/05/01/qwen-ai-releases-qwen-scope-an-open-source-sparse-autoencoders-sae-suite-that-turns-llm-internal-features-into-practical-development-tools/"}, {"title": "Discovering Millions of Interpretable Features with Sparse Autoencoders", "url": "https://arxiv.org/html/2606.26620v1"}, {"title": "[2605.11887] Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/abs/2605.11887"}, {"title": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"}, {"title": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100"}, {"title": "Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/html/2605.11887v1"}, {"title": "Qwen-Scope The Open-Source SAE Toolkit That Finally Makes LLM Internals Actionable - SpaceMarvel Blog", "url": "https://www.spacemarvel.ai/blog/spacemarvel-blog-new-qwen-scope-the-open-source-sae-toolkit-that-finally-makes-llm-internals-actionable-md-1777642664525396"}, {"title": "Qwen/SAE-Res-Qwen3-30B-A3B-Base-W128K-L0_100 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3-30B-A3B-Base-W128K-L0_100"}, {"title": "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"}]}, "Based on the search results, here's information about Qwen-Scope and the research paper 2605.11887:\n\n## Overview\n\nQwen-Scope is an open-source suite of SAEs built on the Qwen model family, comprising 14 groups of SAEs across 7 model variants from the Qwen3 and Qwen3.5 series, covering both dense and mixture-of-expert architectures. This includes five dense models (Qwen3-1.7B, Qwen3-8B, Qwen3.5-2B, Qwen3.5-9B, and Qwen3.5-27B) and two mixture-of-experts (MoE) models (Qwen3-30B-A3B and Qwen3.5-35B-A3B).\n\n## Mechanistic Interpretability Context\n\nLarge language models have largely opaque internal decision-making processes, motivating research in mechanistic interpretability, with sparse autoencoders (SAEs) emerging as one of the most promising tools for decomposing model activations into sparse, interpretable feature representations.\n\n## Technical Approach\n\nA sparse autoencoder acts as a translation layer between raw neural network activations and human-understandable concepts, learning to decompose high-dimensional hidden states (vectors with thousands of numbers) into a large dictionary of sparse latent features, where each input activates only a small subset of features.\n\n## Applications\n\nQwen-Scope turns SAE features into practical tools across three domains: inference (steering model outputs by directly manipulating internal features), data (classifying and organizing datasets), and implicitly, safety (identifying and auditing harmful or undesirable model behaviors at the feature level).\n\nThe paper (arxiv 2605.11887) is titled \"Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models\" and was published on May 12-13, 2026."], "durationSeconds": 13.054017284000002, "searchCount": 1}
```

### [28] ASSISTANT · 2026-07-17 15:02:25 UTC

```
Now I have sufficient context to write the full review. Let me compile the structured output.
```

### [29] TOOL CALL — Write · 2026-07-17 15:02:25 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/review_paper/review_paper/.terminal_claude_agent_struct_out.json

{
  "overall_assessment": "This is a revised iteration of a mechanistic interpretability paper analyzing how refusal abliteration affects RLHF-induced SAE encoder directions in Qwen3-8B. The revision has substantially addressed most critique from the previous review cycle: related work gaps (Yeo et al., Kim et al., Joad et al.) are now covered, scope is properly qualified to Qwen3-8B, the classification null result is correctly framed, and semantic labeling claims are appropriately downgraded to qualitative observations. The core contribution — the Reversal Score + OLS Slope metric pair, direct orthogonality measurement, and benign-SC overlap analysis — is technically original and addresses a real gap left by prior holistic AUROC-based analyses. However, two score-blocking issues remain unresolved: (1) abliteration completeness is still unverified despite being acknowledged as critical, and the paper's central findings remain explicitly conditional on an untested assumption; (2) the finding that 45.4% of RLHF-modified encoder directions show OLS slopes β_f > 1 (amplification beyond the RLHF imprint) is unexplained and in tension with the geometric mechanistic account. The paper warrants a Weak Accept if abliteration verification is added; it remains borderline without it.",
  "strengths": [
    "Novel metric pair: the Reversal Score (Pearson correlation of prompt-level activation deltas) and OLS Slope (magnitude companion) are well-motivated, clearly defined, and address genuine limitations of prior holistic AUROC-based analyses. The definitions avoid tautological construction — Pearson R over a vector of N prompts is a genuine empirical test, not guaranteed by any architectural property.",
    "Orthogonality measurement (Section 5.4) is the paper's most secure finding: mean |cos θ| = 0.0185 between the refusal direction and RLHF-modified encoder directions is a direct weight-space measurement fully independent of the pre-topk/post-topk distinction and of abliteration completeness assumptions. It provides a clean geometric explanation for zero reversed directions and β̄ ≈ 1.",
    "Benign-SC overlap (64.6%) is a substantive finding that reframes what the RLHF imprint actually represents — largely general stylistic modifications rather than harm-specific detectors. This directly explains the classification null result and prevents overclaiming about the imprint's safety relevance.",
    "Limitations are disclosed with unusual specificity: the three critical open gaps (z_pre/z_post validation, abliteration depth, multi-dimensional refusal subspace) are listed with concrete threshold conditions (ρ > 0.7, |cos θ| ≥ 0.05) that make them actionable rather than vague caveats.",
    "All three previously missing related works (Yeo et al. [21], Kim et al. [22], Joad et al. [23]) are now incorporated with differentiated discussion. The CRaFT vs frozen-base-SAE contrast is well articulated.",
    "Layer-20 replication (Table 2) provides cross-layer consistency evidence with quantitatively matching results (R̄ = 0.924, β̄ = 0.954, consistent orthogonality values).",
    "The pre-topk proxy limitation is now properly threaded through the entire paper, with the distinction between 'geometric encoder-direction properties' and 'near-monosemantic feature properties' maintained consistently."
  ],
  "dimension_scores": [
    {
      "dimension": "soundness",
      "score": 2,
      "justification": "The methodology is internally consistent and the metrics are correctly defined. However, two soundness issues prevent a higher score. First, the central finding (high R̄ and β̄ ≈ 1) is explicitly stated as conditional on abliteration completeness, yet no behavioral or geometric verification of the checkpoint is provided — if abliteration is partial, the findings are trivially explained by the abliterated model being close to instruct by construction. Second, 45.4% of directions showing β_f > 1 (amplification) is inconsistent with the geometric account: if r is nearly orthogonal to all RLHF-modified encoder directions, orthogonalization against r should leave encoder projections approximately unchanged, not systematically increase them. This tension is unreported.",
      "improvements": [
        "Report at minimum: (1) refusal rate of the abliterated checkpoint on AdvBench/HarmBench (expected near 100%), and (2) mean |Proj_r(h_abl(p))| / |h_abl(p)| at layer 22 on the 500 harmful prompts (expected near 0). The paper states these can be computed from already-cached activations — they must be computed and reported before the findings can stand unconditionally.",
        "Explain or investigate the β_f > 1 amplification finding: 45.4% of directions showing overshooting is mechanistically inconsistent with the near-orthogonal geometry account. Possible explanations include: (a) OLS no-intercept regression systematically overshoots when delta vectors have non-zero means; (b) orthogonalizing weight matrices against r introduces a small secondary effect on projections onto near-orthogonal directions via indirect activation pathways; (c) the pre-topk regime introduces distributional artifacts. Rule out (a) by comparing intercept-included OLS slopes; investigate (b) by computing mean z_pre variance change for β_f > 1 features."
      ]
    },
    {
      "dimension": "presentation",
      "score": 3,
      "justification": "The paper is well-organized and the metric definitions are precise. The narrative is coherent and the limitations section is unusually thorough. Deductions for: (1) Table 2 has a missing Cohen's d at layer 20 (shown as '—') without explanation — the mid-500 comparison at layer 20 presumably exists since R̄_mid is reported in the table; (2) Section 5.6 (centroid analysis) reports ratios without any statistical validation (confidence intervals or permutation tests), making these ratios uninterpretable as evidence; (3) the relationship between the bottom-500 experiment in art_Qnl1zSSZPJvj (d=1.7508) and the mid-500 baseline in the paper (d=1.087) is unexplained — two different null-baseline experiments exist in the supplementary and only one appears in Table 2.",
      "improvements": [
        "Fill the missing Cohen's d entry in Table 2 for layer 20, or explain why the mid-500 analysis was not replicated at layer 20.",
        "Add statistical tests (permutation test or bootstrap CI) for the centroid ratios in Section 5.6. A ratio of 4.35× vs 1.16× is visually persuasive but carries no inferential weight without a significance statement.",
        "Reconcile the two null-baseline experiments: the artifact art_Qnl1zSSZPJvj computes a bottom-500 baseline (Cohen's d=1.7508) while the paper uses a mid-500 baseline from art_iDkiRSdMuI6M (Cohen's d=1.087). Explain the choice to report only the mid-500 result, and note that the bottom-500 result is also available and consistent (or show it in an appendix)."
      ]
    },
    {
      "dimension": "contribution",
      "score": 3,
      "justification": "The Reversal Score + OLS Slope metric pair is a genuine contribution to the mechanistic analysis of post-training modifications under abliteration. The direct orthogonality measurement provides a clean mechanistic account that holistic AUROC studies could not offer. The benign-SC overlap finding is novel and practically important. The contribution is bounded by the single-model, single-SAE, single-abliteration-method scope and the unverified abliteration completeness — but within those bounds the work is original and would be buildable-upon. The paper correctly positions itself relative to LatentBiopsy, CRaFT, and the companion geometry studies.",
      "improvements": [
        "The discussion of the 35.4% uniquely-harmful encoder directions as 'most promising candidates for targeted geometric restoration' (Section 6.3) is speculative: there is no evidence these directions carry coherent harm-specific semantic content (the z_pre/z_post validation required for this is explicitly deferred). Qualify this claim as a hypothesis pending the z_post analysis, not as an implication of the current results.",
        "Add a sentence noting that the paper tests only one abliteration implementation (huihui-ai) and that different implementations (sumandora/remove-refusals, mlabonne's layer-9-optimal variant) may differ in which layers are targeted and which refusal direction is computed — affecting generalizability."
      ]
    }
  ],
  "critiques": [
    {
      "category": "evidence",
      "severity": "major",
      "description": "The abliterated checkpoint (huihui-ai/Huihui-Qwen3-8B-abliterated-v2) is used throughout without any behavioral or geometric verification that abliteration succeeded. The paper correctly identifies this as a limitation ('The findings in this paper are conditioned on this checkpoint faithfully implementing a refusal-direction removal') but provides zero measurements to validate the assumption. If abliteration is incomplete — even partially — the high R̄ (0.921) and β̄ ≈ 1 trivially follow because the abliterated model remains close to the instruct model by construction, not because of any genuine preservation under removal. Critically, the paper states that the two required measurements (refusal rate, mean |Proj_r|/|h| at layer 22) 'can be computed from already-cached activations,' making this an omission, not a resource constraint. A paper whose central finding is explicitly conditional on an untested assumption cannot be accepted at a top-tier venue without the test.",
      "suggested_action": "Compute and report: (1) the fraction of AdvBench or HarmBench prompts for which the abliterated checkpoint produces a non-refusal response (expected > 95% for complete abliteration; any figure below 80% would substantially qualify the findings); (2) mean |Proj_r(h_abl(p))| / |h_abl(p)| at layer 22 for the 500 harmful prompts, expected near 0 if r has been removed (a value above 0.1 would indicate partial abliteration). If these are computed from cached activations as stated, they can be added in a single section (e.g., Section 3.1 or a new Section 5.0 'Abliteration Verification') without additional compute. If verification fails (incomplete abliteration), reframe all findings as 'conditional' in the abstract, not just in Limitations."
    },
    {
      "category": "rigor",
      "severity": "major",
      "description": "The OLS slope analysis reports that 45.4% of the top-500 RLHF-sensitive encoder directions have β_f > 1.0 (abliteration amplifies rather than preserves the RLHF-induced shift). This is mechanistically inconsistent with the geometric account. The paper's mechanistic explanation states: abliteration orthogonalizes weight matrices against r; because r is nearly orthogonal to all RLHF-modified encoder directions (|cos θ| ≈ 0.0185), orthogonalization barely touches those directions and leaves both direction and magnitude preserved (β_f ≈ 1). But orthogonalizing weight matrices against a near-orthogonal direction should project out a tiny component and leave magnitudes *slightly less than 1*, not systematically greater than 1. Finding 45.4% amplification contradicts the geometric account and is left without explanation or investigation. This is not a minor rounding artifact: a mean of 0.986 with 45.4% of values above 1.0 and a max of 1.725 indicates systematic amplification in a substantial minority of features. Possible explanations include: (a) no-intercept OLS regression on vectors with non-zero means introduces upward bias; (b) weight-matrix orthogonalization has indirect secondary effects through residual stream accumulation; (c) the pre-topk representation amplifies changes in encoder projections that are non-linearly related to the ablated weights. The paper reports the amplification but does not investigate it.",
      "suggested_action": "Add a brief analysis of the β_f > 1 subset: (a) compare the mean β_f computed with and without the no-intercept constraint (adding an intercept term would reveal whether the upward bias is a regression artifact); (b) check whether the 45.4% amplified directions are concentrated in specific semantic label categories or have systematically different SC magnitudes (do high-SC directions amplify more than low-SC ones?); (c) add a sentence in Section 5.2 and the Discussion (Section 6.1) acknowledging that amplification is inconsistent with the simple geometric account and identifying it as an open mechanistic question. The current framing ('near unity') underplays a finding that affects nearly half the analyzed directions."
    },
    {
      "category": "evidence",
      "severity": "minor",
      "description": "The centroid analysis (Section 5.6) reports ratios (4.35×, 1.16×, 2.37× at layer 22; 7.31×, 1.01×, 1.63× at layer 20) without any statistical significance tests. Unlike the Reversal Score comparison (which reports Cohen's d and t-statistics with p-values) and the OLS slope comparison, the centroid ratios carry no inferential weight as presented. A reader cannot determine whether the 4.35× vs 1.16× difference between Subspace A and Subspace B is statistically reliable or whether it could arise by chance from the geometry of the 4096-dimensional residual stream.",
      "suggested_action": "Add bootstrap confidence intervals or a permutation test for the centroid distance ratios. Concretely: resample the 500 harmful prompts with replacement 1000 times, recompute subspace centroid distances for each resample, and report [2.5%, 97.5%] CIs for the A/B and A/C ratios. If the 95% CI for ratio_A/ratio_B excludes 1.0, the difference is statistically confirmed. This does not require additional model inference — it can be computed from cached centroid distances."
    },
    {
      "category": "clarity",
      "severity": "minor",
      "description": "Table 2 has a missing Cohen's d entry for layer 20 (shown as '—'). The paper reports R̄_mid500 = 0.858 at layer 20, implying the mid-500 comparison at layer 20 was run, but Cohen's d is not reported. Additionally, the supplementary material (art_Qnl1zSSZPJvj) reports a bottom-500 baseline experiment with Cohen's d = 1.7508, while the paper uses a mid-500 baseline with d = 1.087. The discrepancy between these two null-baseline experiments (same paper, different baselines, substantially different effect sizes) is never discussed. A reader of both the paper and supplementary materials will be confused about which experiment is definitive.",
      "suggested_action": "Fill the missing Cohen's d entry in Table 2 for layer 20 using the mid-500 comparison data. Add a footnote or Methods subparagraph explaining the choice of mid-500 over bottom-500 baseline (the stated reason — avoiding near-zero vector artifacts — is valid and should appear in the main text). If the bottom-500 result (d = 1.7508) is available, note it in a table footnote as a consistency check — a larger effect size for the more extreme baseline group is expected and would strengthen rather than undermine the argument."
    },
    {
      "category": "rigor",
      "severity": "minor",
      "description": "The multi-dimensional refusal subspace limitation (acknowledged in Limitations, Section 5.4) has an actionable threshold ('a higher PC shows mean |cos θ_f| ≥ 0.05 against the top-500 encoder directions') but the measurement itself is deferred. Joad et al. [23] demonstrate that refusal spans geometrically distinct directions across eleven behavioral categories. If abliteration removes a multi-dimensional subspace (as would be the case for implementations targeting multiple refusal directions), the orthogonality argument requires that the *entire ablation subspace* be near-orthogonal to all RLHF-modified encoder directions — not just the mean-difference summary vector. The current measurement (single vector r) is necessary but not sufficient. This is disclosed, but the threshold condition and the specific PCA extension needed are clearly enough defined that the measurement could plausibly be performed from cached activations.",
      "suggested_action": "Compute the top-3 principal components of the (harmful − harmless) activation differences at layer 22 using the 500 harmful and 500 benign prompts, and report mean |cos(PC_k, w_enc(f))| for k = 1, 2, 3 across the top-500 encoder directions. If all three values are below 0.05, the orthogonality argument extends robustly to multi-dimensional ablation. This computation requires only cached activations and SVD — no additional inference. If this measurement cannot be added in the current revision, at minimum report the explained variance of PC1 among the 500 harmful prompts to contextualize how much of the refusal subspace r (the mean-difference vector) captures."
    },
    {
      "category": "scope",
      "severity": "minor",
      "description": "The paper tests only one abliteration implementation (huihui-ai/Huihui-Qwen3-8B-abliterated-v2). As noted in the tech reference artifact, multiple implementations exist (sumandora, mlabonne, orion-zhen) that differ in which layers are targeted, which prompt sets define the refusal direction, and how the direction is computed (mean-difference vs. PCA). The findings (zero reversed directions, β̄ ≈ 1, |cos θ| = 0.0185) may be specific to the huihui-ai implementation's choice of refusal direction and layer coverage. A second abliterated checkpoint (e.g., mlabonne/Qwen3-8B-abliterated, which targets layer 9 as optimal vs. the deeper layers in this paper's primary results) would test whether the orthogonality finding generalizes across abliteration implementations.",
      "suggested_action": "Add one sentence in Section 3.1 noting that results are conditioned on the huihui-ai implementation and that layer selection or prompt-set differences in other implementations could yield different refusal directions. In Future Directions, add: 'Testing a second abliteration implementation (e.g., mlabonne's layer-9-targeted variant) with independently computed refusal direction r would determine whether the orthogonality result holds across implementation choices.' This requires no additional code if the mlabonne checkpoint is cached alongside the huihui-ai checkpoint."
    },
    {
      "category": "rigor",
      "severity": "minor",
      "description": "The Discussion section 6.3 proposes the 35.4% uniquely-harmful encoder directions as 'the most promising candidates' for targeted safety restoration. This claim goes beyond what the data supports: (a) these directions are defined purely by their absence from the benign-SC top-500, not by any validated semantic content; (b) the z_pre/z_post gap means their 'harm-specific' label is geometric (higher SC on harmful than benign prompts in pre-topk projections) rather than semantically validated; (c) no evidence is presented that these directions actually encode harm-concept representations rather than, say, prompt-format differences between the harmful and benign corpora. Treating these as restoration candidates without the z_post validation is speculative.",
      "suggested_action": "Soften the claim in Section 6.3: replace 'are the most promising candidates for such targeted intervention' with 'represent the candidate subset, pending z_post validation that they carry coherent harm-specific semantic content rather than corpus-format artifacts.' This is a one-sentence change that accurately reflects the epistemic state of the finding without weakening the structural argument."
    }
  ],
  "score": 6,
  "confidence": 4
}
```

### [30] TOOL RESULT — Write · 2026-07-17 15:02:25 UTC

```
{"type": "create", "filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_5/review_paper/review_paper/.terminal_claude_agent_struct_out.json", "content": "{\n  \"overall_assessment\": \"This is a revised iteration of a mechanistic interpretability paper analyzing how refusal abliteration affects RLHF-induced SAE encoder directions in Qwen3-8B. The revision has substantially addressed most critique from the previous review cycle: related work gaps (Yeo et al., Kim et al., Joad et al.) are now covered, scope is properly qualified to Qwen3-8B, the classification null result is correctly framed, and semantic labeling claims are appropriately downgraded to qualitative observations. The core contribution — the Reversal Score + OLS Slope metric pair, direct orthogonality measurement, and benign-SC overlap analysis — is technically original and addresses a real gap left by prior holistic AUROC-based analyses. However, two score-blocking issues remain unresolved: (1) abliteration completeness is still unverified despite being acknowledged as critical, and the paper's central findings remain explicitly conditional on an untested assumption; (2) the finding that 45.4% of RLHF-modified encoder directions show OLS slopes β_f > 1 (amplification beyond the RLHF imprint) is unexplained and in tension with the geometric mechanistic account. The paper warrants a Weak Accept if abliteration verification is added; it remains borderline without it.\",\n  \"strengths\": [\n    \"Novel metric pair: the Reversal Score (Pearson correlation of prompt-level activation deltas) and OLS Slope (magnitude companion) are well-motivated, clearly defined, and address genuine limitations of prior holistic AUROC-based analyses. The definitions avoid tautological construction — Pearson R over a vector of N prompts is a genuine empirical test, not guaranteed by any architectural property.\",\n    \"Orthogonality measurement (Section 5.4) is the paper's most secure finding: mean |cos θ| = 0.0185 between the refusal direction and RLHF-modified encoder directions is a direct weight-space measurement fully independent of the pre-topk/post-topk distinction and of abliteration completeness assumptions. It provides a clean geometric explanation for zero reversed directions and β̄ ≈ 1.\",\n    \"Benign-SC overlap (64.6%) is a substantive finding that reframes what the RLHF imprint actually represents — largely general stylistic modifications rather than harm-specific detectors. This directly explains the classification null result and prevents overclaiming about the imprint's safety relevance.\",\n    \"Limitations are disclosed with unusual specificity: the three critical open gaps (z_pre/z_post validation, abliteration depth, multi-dimensional refusal subspace) are listed with concrete threshold conditions (ρ > 0.7, |cos θ| ≥ 0.05) that make them actionable rather than vague caveats.\",\n    \"All three previously missing related works (Yeo et al. [21], Kim et al. [22], Joad et al. [23]) are now incorporated with differentiated discussion. The CRaFT vs frozen-base-SAE contrast is well articulated.\",\n    \"Layer-20 replication (Table 2) provides cross-layer consistency evidence with quantitatively matching results (R̄ = 0.924, β̄ = 0.954, consistent orthogonality values).\",\n    \"The pre-topk proxy limitation is now properly threaded through the entire paper, with the distinction between 'geometric encoder-direction properties' and 'near-monosemantic feature properties' maintained consistently.\"\n  ],\n  \"dimension_scores\": [\n    {\n      \"dimension\": \"soundness\",\n      \"score\": 2,\n      \"justification\": \"The methodology is internally consistent and the metrics are correctly defined. However, two soundness issues prevent a higher score. First, the central finding (high R̄ and β̄ ≈ 1) is explicitly stated as conditional on abliteration completeness, yet no behavioral or geometric verification of the checkpoint is provided — if abliteration is partial, the findings are trivially explained by the abliterated model being close to instruct by construction. Second, 45.4% of directions showing β_f > 1 (amplification) is inconsistent with the geometric account: if r is nearly orthogonal to all RLHF-modified encoder directions, orthogonalization against r should leave encoder projections approximately unchanged, not systematically increase them. This tension is unreported.\",\n      \"improvements\": [\n        \"Report at minimum: (1) refusal rate of the abliterated checkpoint on AdvBench/HarmBench (expected near 100%), and (2) mean |Proj_r(h_abl(p))| / |h_abl(p)| at layer 22 on the 500 harmful prompts (expected near 0). The paper states these can be computed from already-cached activations — they must be computed and reported before the findings can stand unconditionally.\",\n        \"Explain or investigate the β_f > 1 amplification finding: 45.4% of directions showing overshooting is mechanistically inconsistent with the near-orthogonal geometry account. Possible explanations include: (a) OLS no-intercept regression systematically overshoots when delta vectors have non-zero means; (b) orthogonalizing weight matrices against r introduces a small secondary effect on projections onto near-orthogonal directions via indirect activation pathways; (c) the pre-topk regime introduces distributional artifacts. Rule out (a) by comparing intercept-included OLS slopes; investigate (b) by computing mean z_pre variance change for β_f > 1 features.\"\n      ]\n    },\n    {\n      \"dimension\": \"presentation\",\n      \"score\": 3,\n      \"justification\": \"The paper is well-organized and the metric definitions are precise. The narrative is coherent and the limitations section is unusually thorough. Deductions for: (1) Table 2 has a missing Cohen's d at layer 20 (shown as '—') without explanation — the mid-500 comparison at layer 20 presumably exists since R̄_mid is reported in the table; (2) Section 5.6 (centroid analysis) reports ratios without any statistical validation (confidence intervals or permutation tests), making these ratios uninterpretable as evidence; (3) the relationship between the bottom-500 experiment in art_Qnl1zSSZPJvj (d=1.7508) and the mid-500 baseline in the paper (d=1.087) is unexplained — two different null-baseline experiments exist in the supplementary and only one appears in Table 2.\",\n      \"improvements\": [\n        \"Fill the missing Cohen's d entry in Table 2 for layer 20, or explain why the mid-500 analysis was not replicated at layer 20.\",\n        \"Add statistical tests (permutation test or bootstrap CI) for the centroid ratios in Section 5.6. A ratio of 4.35× vs 1.16× is visually persuasive but carries no inferential weight without a significance statement.\",\n        \"Reconcile the two null-baseline experiments: the artifact art_Qnl1zSSZPJvj computes a bottom-500 baseline (Cohen's d=1.7508) while the paper uses a mid-500 baseline from art_iDkiRSdMuI6M (Cohen's d=1.087). Explain the choice to report only the mid-500 result, and note that the bottom-500 result is also available and consistent (or show it in an appendix).\"\n      ]\n    },\n    {\n      \"dimension\": \"contribution\",\n      \"score\": 3,\n      \"justification\": \"The Reversal Score + OLS Slope metric pair is a genuine contribution to the mechanistic analysis of post-training modifications under abliteration. The direct orthogonality measurement provides a clean mechanistic account that holistic AUROC studies could not offer. The benign-SC overlap finding is novel and practically important. The contribution is bounded by the single-model, single-SAE, single-abliteration-method scope and the unverified abliteration completeness — but within those bounds the work is original and would be buildable-upon. The paper correctly positions itself relative to LatentBiopsy, CRaFT, and the companion geometry studies.\",\n      \"improvements\": [\n        \"The discussion of the 35.4% uniquely-harmful encoder directions as 'most promising candidates for targeted geometric restoration' (Section 6.3) is speculative: there is no evidence these directions carry coherent harm-specific semantic content (the z_pre/z_post validation required for this is explicitly deferred). Qualify this claim as a hypothesis pending the z_post analysis, not as an implication of the current results.\",\n        \"Add a sentence noting that the paper tests only one abliteration implementation (huihui-ai) and that different implementations (sumandora/remove-refusals, mlabonne's layer-9-optimal variant) may differ in which layers are targeted and which refusal direction is computed — affecting generalizability.\"\n      ]\n    }\n  ],\n  \"critiques\": [\n    {\n      \"category\": \"evidence\",\n      \"severity\": \"major\",\n      \"description\": \"The abliterated checkpoint (huihui-ai/Huihui-Qwen3-8B-abliterated-v2) is used throughout without any behavioral or geometric verification that abliteration succeeded. The paper correctly identifies this as a limitation ('The findings in this paper are conditioned on this checkpoint faithfully implementing a refusal-direction removal') but provides zero measurements to validate the assumption. If abliteration is incomplete — even partially — the high R̄ (0.921) and β̄ ≈ 1 trivially follow because the abliterated model remains close to the instruct model by construction, not because of any genuine preservation under removal. Critically, the paper states that the two required measurements (refusal rate, mean |Proj_r|/|h| at layer 22) 'can be computed from already-cached activations,' making this an omission, not a resource constraint. A paper whose central finding is explicitly conditional on an untested assumption cannot be accepted at a top-tier venue without the test.\",\n      \"suggested_action\": \"Compute and report: (1) the fraction of AdvBench or HarmBench prompts for which the abliterated checkpoint produces a non-refusal response (expected > 95% for complete abliteration; any figure below 80% would substantially qualify the findings); (2) mean |Proj_r(h_abl(p))| / |h_abl(p)| at layer 22 for the 500 harmful prompts, expected near 0 if r has been removed (a value above 0.1 would indicate partial abliteration). If these are computed from cached activations as stated, they can be added in a single section (e.g., Section 3.1 or a new Section 5.0 'Abliteration Verification') without additional compute. If verification fails (incomplete abliteration), reframe all findings as 'conditional' in the abstract, not just in Limitations.\"\n    },\n    {\n      \"category\": \"rigor\",\n      \"severity\": \"major\",\n      \"description\": \"The OLS slope analysis reports that 45.4% of the top-500 RLHF-sensitive encoder directions have β_f > 1.0 (abliteration amplifies rather than preserves the RLHF-induced shift). This is mechanistically inconsistent with the geometric account. The paper's mechanistic explanation states: abliteration orthogonalizes weight matrices against r; because r is nearly orthogonal to all RLHF-modified encoder directions (|cos θ| ≈ 0.0185), orthogonalization barely touches those directions and leaves both direction and magnitude preserved (β_f ≈ 1). But orthogonalizing weight matrices against a near-orthogonal direction should project out a tiny component and leave magnitudes *slightly less than 1*, not systematically greater than 1. Finding 45.4% amplification contradicts the geometric account and is left without explanation or investigation. This is not a minor rounding artifact: a mean of 0.986 with 45.4% of values above 1.0 and a max of 1.725 indicates systematic amplification in a substantial minority of features. Possible explanations include: (a) no-intercept OLS regression on vectors with non-zero means introduces upward bias; (b) weight-matrix orthogonalization has indirect secondary effects through residual stream accumulation; (c) the pre-topk representation amplifies changes in encoder projections that are non-linearly related to the ablated weights. The paper reports the amplification but does not investigate it.\",\n      \"suggested_action\": \"Add a brief analysis of the β_f > 1 subset: (a) compare the mean β_f computed with and without the no-intercept constraint (adding an intercept term would reveal whether the upward bias is a regression artifact); (b) check whether the 45.4% amplified directions are concentrated in specific semantic label categories or have systematically different SC magnitudes (do high-SC directions amplify more than low-SC ones?); (c) add a sentence in Section 5.2 and the Discussion (Section 6.1) acknowledging that amplification is inconsistent with the simple geometric account and identifying it as an open mechanistic question. The current framing ('near unity') underplays a finding that affects nearly half the analyzed directions.\"\n    },\n    {\n      \"category\": \"evidence\",\n      \"severity\": \"minor\",\n      \"description\": \"The centroid analysis (Section 5.6) reports ratios (4.35×, 1.16×, 2.37× at layer 22; 7.31×, 1.01×, 1.63× at layer 20) without any statistical significance tests. Unlike the Reversal Score comparison (which reports Cohen's d and t-statistics with p-values) and the OLS slope comparison, the centroid ratios carry no inferential weight as presented. A reader cannot determine whether the 4.35× vs 1.16× difference between Subspace A and Subspace B is statistically reliable or whether it could arise by chance from the geometry of the 4096-dimensional residual stream.\",\n      \"suggested_action\": \"Add bootstrap confidence intervals or a permutation test for the centroid distance ratios. Concretely: resample the 500 harmful prompts with replacement 1000 times, recompute subspace centroid distances for each resample, and report [2.5%, 97.5%] CIs for the A/B and A/C ratios. If the 95% CI for ratio_A/ratio_B excludes 1.0, the difference is statistically confirmed. This does not require additional model inference — it can be computed from cached centroid distances.\"\n    },\n    {\n      \"category\": \"clarity\",\n      \"severity\": \"minor\",\n      \"description\": \"Table 2 has a missing Cohen's d entry for layer 20 (shown as '—'). The paper reports R̄_mid500 = 0.858 at layer 20, implying the mid-500 comparison at layer 20 was run, but Cohen's d is not reported. Additionally, the supplementary material (art_Qnl1zSSZPJvj) reports a bottom-500 baseline experiment with Cohen's d = 1.7508, while the paper uses a mid-500 baseline with d = 1.087. The discrepancy between these two null-baseline experiments (same paper, different baselines, substantially different effect sizes) is never discussed. A reader of both the paper and supplementary materials will be confused about which experiment is definitive.\",\n      \"suggested_action\": \"Fill the missing Cohen's d entry in Table 2 for layer 20 using the mid-500 comparison data. Add a footnote or Methods subparagraph explaining the choice of mid-500 over bottom-500 baseline (the stated reason — avoiding near-zero vector artifacts — is valid and should appear in the main text). If the bottom-500 result (d = 1.7508) is available, note it in a table footnote as a consistency check — a larger effect size for the more extreme baseline group is expected and would strengthen rather than undermine the argument.\"\n    },\n    {\n      \"category\": \"rigor\",\n      \"severity\": \"minor\",\n      \"description\": \"The multi-dimensional refusal subspace limitation (acknowledged in Limitations, Section 5.4) has an actionable threshold ('a higher PC shows mean |cos θ_f| ≥ 0.05 against the top-500 encoder directions') but the measurement itself is deferred. Joad et al. [23] demonstrate that refusal spans geometrically distinct directions across eleven behavioral categories. If abliteration removes a multi-dimensional subspace (as would be the case for implementations targeting multiple refusal directions), the orthogonality argument requires that the *entire ablation subspace* be near-orthogonal to all RLHF-modified encoder directions — not just the mean-difference summary vector. The current measurement (single vector r) is necessary but not sufficient. This is disclosed, but the threshold condition and the specific PCA extension needed are clearly enough defined that the measurement could plausibly be performed from cached activations.\",\n      \"suggested_action\": \"Compute the top-3 principal components of the (harmful − harmless) activation differences at layer 22 using the 500 harmful and 500 benign prompts, and report mean |cos(PC_k, w_enc(f))| for k = 1, 2, 3 across the top-500 encoder directions. If all three values are below 0.05, the orthogonality argument extends robustly to multi-dimensional ablation. This computation requires only cached activations and SVD — no additional inference. If this measurement cannot be added in the current revision, at minimum report the explained variance of PC1 among the 500 harmful prompts to contextualize how much of the refusal subspace r (the mean-difference vector) captures.\"\n    },\n    {\n      \"category\": \"scope\",\n      \"severity\": \"minor\",\n      \"description\": \"The paper tests only one abliteration implementation (huihui-ai/Huihui-Qwen3-8B-abliterated-v2). As noted in the tech reference artifact, multiple implementations exist (sumandora, mlabonne, orion-zhen) that differ in which layers are targeted, which prompt sets define the refusal direction, and how the direction is computed (mean-difference vs. PCA). The findings (zero reversed directions, β̄ ≈ 1, |cos θ| = 0.0185) may be specific to the huihui-ai implementation's choice of refusal direction and layer coverage. A second abliterated checkpoint (e.g., mlabonne/Qwen3-8B-abliterated, which targets layer 9 as optimal vs. the deeper layers in this paper's primary results) would test whether the orthogonality finding generalizes across abliteration implementations.\",\n      \"suggested_action\": \"Add one sentence in Section 3.1 noting that results are conditioned on the huihui-ai implementation and that layer selection or prompt-set differences in other implementations could yield different refusal directions. In Future Directions, add: 'Testing a second abliteration implementation (e.g., mlabonne's layer-9-targeted variant) with independently computed refusal direction r would determine whether the orthogonality result holds across implementation choices.' This requires no additional code if the mlabonne checkpoint is cached alongside the huihui-ai checkpoint.\"\n    },\n    {\n      \"category\": \"rigor\",\n      \"severity\": \"minor\",\n      \"description\": \"The Discussion section 6.3 proposes the 35.4% uniquely-harmful encoder directions as 'the most promising candidates' for targeted safety restoration. This claim goes beyond what the data supports: (a) these directions are defined purely by their absence from the benign-SC top-500, not by any validated semantic content; (b) the z_pre/z_post gap means their 'harm-specific' label is geometric (higher SC on harmful than benign prompts in pre-topk projections) rather than semantically validated; (c) no evidence is presented that these directions actually encode harm-concept representations rather than, say, prompt-format differences between the harmful and benign corpora. Treating these as restoration candidates without the z_post validation is speculative.\",\n      \"suggested_action\": \"Soften the claim in Section 6.3: replace 'are the most promising candidates for such targeted intervention' with 'represent the candidate subset, pending z_post validation that they carry coherent harm-specific semantic content rather than corpus-format artifacts.' This is a one-sentence change that accurately reflects the epistemic state of the finding without weakening the structural argument.\"\n    }\n  ],\n  \"score\": 6,\n  \"confidence\": 4\n}\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [31] ASSISTANT · 2026-07-17 15:02:43 UTC

```
The review is complete. Here's a summary of the assessment:

**Score: 6/10 (Weak Accept)** | Confidence: 4/5

**Key improvements from previous iteration (all addressed):**
- All three missing related works (Yeo et al., Kim et al., Joad et al.) now properly covered
- Scope qualified to Qwen3-8B throughout
- Classification null result framing fixed
- Semantic labeling downgraded to qualitative

**Two remaining major issues:**

1. **Abliteration depth unverified (still from prior review)** — The paper explicitly says findings are "conditional on abliteration completeness" but provides no behavioral verification (refusal rate on AdvBench) or geometric verification (mean |Proj_r(h_abl)| at layer 22). The paper itself states these can be computed from cached activations — making this an omission, not a resource constraint.

2. **OLS β > 1 amplification unexplained (new)** — 45.4% of directions show β_f > 1 (abliteration *amplifies* the RLHF shift). This contradicts the geometric account: if r is nearly orthogonal to all RLHF-modified directions, orthogonalization against r should leave magnitudes at or below 1, not systematically above. This tension is unreported.

**Minor issues (5):** centroid ratios lack statistical tests; Table 2 missing Layer-20 Cohen's d; mid-500 vs bottom-500 baseline discrepancy unexplained; multi-dimensional refusal subspace measurement still deferred; uniquely-harmful directions overclaimed as restoration candidates.

The orthogonality measurement (|cos θ| = 0.0185) remains the paper's most secure and compelling contribution — it's independent of all proxy limitations.
```
