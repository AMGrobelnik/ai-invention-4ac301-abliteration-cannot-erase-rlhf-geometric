# review_paper — test_idea

> Phase: `invention_loop` · round 2 · `review_paper`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim transcript of this agent task — every system/user prompt, assistant response, thinking block, tool call and tool result — in the order they occurred. Nothing truncated.

## Task: `review_paper` (terminal_claude_agent, claude-sonnet-4-6)

### [1] CONFIG · 2026-07-17 12:20:25 UTC

```
model: claude-sonnet-4-6 | effort: medium | permission: bypassPermissions | cwd: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/review_paper/review_paper
```

### [2] SYSTEM-USER prompt · 2026-07-17 12:20:31 UTC

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
</supplementary_materials>

<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for judging whether the paper's contribution is genuinely novel versus already-done or a known dead end in this field.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>

<previous_review>
Your review from the previous iteration. Check which critiques have been addressed
in the revised paper. Do NOT re-raise critiques that have been adequately fixed.
Only re-raise if the fix is insufficient.

- [MAJOR] (evidence) The paper presents no experimental results. All three confirmatory criteria (Sections 5.1) are framed as future tests using future tense: 'the representational distance analysis will directly test,' 'Criterion 2 is: d(abl,inst) < d(abl,base) for harmful prompts,' 'the hypothesis is confirmed in a weaker form IF features with R(f)>+0.5 receive harm-concept labels at ≥2× the rate.' No Reversal_score values, bimodality test statistics, semantic label distributions, representational distances, or SAE activation comparisons are reported anywhere in the paper. The supplementary artifacts confirm a dataset was assembled (468 examples) and a technical reference was written, but no analysis code or results files are listed. At any top-tier venue (NeurIPS, ICML, ICLR), submitting a pre-registration as a completed paper would cause immediate rejection.
  Action: Run the complete analysis pipeline: (1) collect residual-stream activations for all three models at the selected layer on the 500 harmful + 500 benign prompts; (2) compute Safety_change and select top-500 RLHF-sensitive features; (3) compute Reversal_score for each; (4) run Hartigan's dip test; (5) apply semantic labeling with cross-family LLM; (6) compute centroid distances in Type-R and Type-I subspaces. Report all numerical outcomes — even null results (Criterion 1 not confirmed) are publishable with the pre-registered framework.
- [MAJOR] (rigor) The Reversal_score is defined as Pearson correlation over N=500 activation values per feature. With TopK (k=50) SAE architecture, a given feature is active on at most k=50 of the tokens present in any forward pass. For rare features, the 500-prompt activation vectors are mostly zeros — e.g., if feature f fires on only 30 of 500 harmful prompts, both Δ_inst(f) and Δ_abl(f) contain ~470 near-zero values. Pearson correlation over zero-dominated vectors is unreliable: it is dominated by the zero-zero cluster and can return spuriously high or low correlations with tiny absolute differences. The paper does not acknowledge this problem or impose any activation-density filter.
  Action: Before computing Reversal_score, filter to features where both |Δ_inst(f)| and |Δ_abl(f)| have at least 10% non-zero entries (≥50 of 500 prompts). Report what fraction of the top-500 RLHF-sensitive features survive this filter. Alternatively, use Spearman correlation (more robust to zero inflation) or restrict to token-level activations where the feature is non-zero in at least one of the two models. Present both filtered and unfiltered Reversal_score distributions as a robustness check.
- [MAJOR] (methodology) The paper uses huihui-ai/Huihui-Qwen3-8B-abliterated-v2 as the primary abliterated checkpoint but acknowledges its abliteration method is described only as 'new and faster' without specification. If this differs from standard difference-of-means orthogonalization (as used in the scratch reproduction), the Reversal_score partition may reflect methodological differences rather than the mechanistic structure of RLHF modifications. The paper treats the community model as a validated abliterated checkpoint without verifying that it produces comparable residual-stream statistics to the scratch-reproduced version.
  Action: Compare the community model and scratch-reproduced checkpoint on the key geometric metrics already established in the literature: AUROC for harmful prompt detection post-abliteration (per [2]), KL divergence vs. base model on Wikitext-2, and refusal rate on a held-out harmful prompt set. Report whether both checkpoints achieve comparable abliteration depth. If they differ significantly, restrict the primary analysis to the scratch-reproduced checkpoint whose method is fully documented, and treat the community model as a secondary validation target.
- [MAJOR] (rigor) Reference [13] (Gao et al., 'Scaling and evaluating sparse autoencoders') is cited as 'In *International Conference on Learning Representations*, 2024.' This is factually incorrect: the arXiv preprint (arXiv:2406.04093) was submitted in June 2024, which is after the ICLR 2024 conference concluded (May 2024). The correct venue is ICLR 2025, or the paper should be cited as an arXiv preprint. Additionally, reference [18] (He et al.) is cited in Section 4 as the source for the Dolly-15k dataset, but He et al. is about discovering interpretable features with sparse autoencoders — it has no relation to Dolly-15k. Dolly-15k is from Conover et al. (Databricks, 2023) and has no entry in the reference list.
  Action: Fix Gao et al. venue to ICLR 2025 (or arXiv:2406.04093, 2024 if venue is uncertain). Add a proper Dolly-15k reference: 'M. Conover, M. Hayes, A. Mathur, J. Meng, S. Mody, N. Sherwood, V. Sreedhar, B. Wan, and A. Zaharia. Free Dolly: Introducing the world's first truly open instruction-tuned LLM. Databricks Blog, 2023.' and correct the Section 4 citation.
- [MAJOR] (evidence) The paper claims the assembled corpus contains 1000 rows (500 harmful + 500 benign) throughout the body and in the dataset artifact title. However, the artifact summary states: 'the 4-dataset selection in full_data_out.json contains 468 examples.' The results/data_out.json claim of 1000 rows cannot be independently verified and contradicts the explicitly reported 468 count in the artifact description. If the pipeline could not achieve the target 500+500 after filtering, the actual corpus size affects which sampling properties hold.
  Action: Reconcile the counts. If full_data_out.json is an intermediate pre-pipeline-sampling file and results/data_out.json is the final 1000-row corpus, explain this two-stage structure explicitly in Section 4 and the artifact. If the pipeline did not achieve 500+500, report the actual counts per source and explain how this affects category balance and corpus representativeness.
- [MINOR] (methodology) The paper applies layer-sensitivity findings from Chopra [4] (Gemma 3 1B) directly to Qwen3-8B layer selection: 'layer 22 exhibits maximum sensitivity to fine-tuning changes (SAE latent cosine similarity 0.509–0.557 after SFT), layer 14 is most stable, and layer 20 is intermediate.' But Gemma 3 1B and Qwen3-8B differ in architecture (depth, residual stream dimension, attention heads, MLP design) and model scale. Layer-sensitivity profiles are not portable across families without empirical verification.
  Action: Run the layer selection procedure described in Section 3.4 (Safety_change variance across all 36 layers on 50 harmful prompts) and report the resulting variance profile for Qwen3-8B. Use this as the primary basis for layer selection; cite Chopra [4] only as prior motivation for expecting non-uniform layer sensitivity, not as a direct specification of which Qwen3-8B layer to use.
- [MINOR] (methodology) The Type-I classification threshold R(f) > +0.5 is stated as a design choice in Section 3.5 without justification. This threshold determines which features are labeled 'abliteration-resistant' and directly drives the Fisher's exact test in Criterion 3. A shift from +0.5 to +0.3 or +0.7 could substantially change the count of Type-I features and the enrichment statistics.
  Action: Either (a) derive the threshold from the empirical Reversal_score distribution — e.g., use the top quartile of R(f) — which avoids hardcoding an arbitrary value; or (b) show that Criterion 3 results are stable across thresholds {0.3, 0.5, 0.7} in a robustness analysis. Pre-register the threshold before running the full analysis to maintain the confirmatory design's integrity.
- [MINOR] (clarity) The semantic labeling protocol in Section 3.6 states that 'we extract maximum-activating prompt segments' for LLM labeling but does not specify what 'segment' means: is it the full 512-token prompt? A fixed-length window around the highest-activation token? The top-N tokens by activation weight? The labeling quality and the human gold-set annotations both depend critically on what text is provided to the labeler.
  Action: Specify the segment extraction procedure precisely: (1) identify the token with the highest absolute SAE feature activation within the prompt; (2) extract a fixed context window of ±K tokens around it (specify K, e.g., ±32 tokens); (3) provide this window plus the feature index and its top-5 activating examples globally to the labeler. Include one example of a full labeling prompt in the appendix to enable reproducibility.
- [MINOR] (scope) The paper presents the 'TopK architecture correction' as a bullet-point contribution (item 4 in the contribution list). While the correction itself is important for implementation fidelity, it is a discovery about the Qwen-Scope release rather than a methodological innovation. Labeling it a contribution alongside the Reversal_score metric and pre-registered framework overstates its significance and dilutes the contribution narrative.
  Action: Move the TopK correction to the Related Work or Methodology section as a methodological clarification (e.g., 'A critical finding from preliminary work: the Qwen-Scope SAE uses TopK rather than JumpReLU...') and remove it from the contribution bullet list. The pre-topk/post-topk robustness analysis that this enables is worth mentioning, but it is a consequence of the architecture rather than a contribution itself.
- [MINOR] (novelty) The paper positions the frozen base-SAE approach as a complement to crosscoders (Jiralerspong and Bricken [5], Shportko et al. [6]) but overstates some advantages. The claim that crosscoders 'require training a new shared feature vocabulary per model pair... and produce a feature basis that is not directly interpretable using Qwen-Scope's existing labeled vocabulary' does not account for recent work (e.g., Delta-Crosscoder, arXiv:2603.04426) that addresses some of these limitations. The comparison should be more nuanced.
  Action: Update the crosscoder comparison to acknowledge that recent crosscoder variants have reduced training cost and improved cross-vocabulary interpretability. Frame the frozen base-SAE advantage more narrowly: it requires zero additional training and reuses existing Qwen-Scope labels, but this advantage applies specifically to the case where a high-quality base SAE already exists — which is true for Qwen3-8B via Qwen-Scope but may not hold for other families.
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

### [3] HUMAN-USER prompt · 2026-07-17 12:20:31 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [4] ASSISTANT · 2026-07-17 12:21:51 UTC

```
I'll carefully read the paper and supplementary materials, then write a thorough adversarial review.
```

### [5] TOOL CALL — Read · 2026-07-17 12:21:51 UTC

```
/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/gen_art/gen_art_experiment_1/method.py
```

### [6] TOOL RESULT — Read · 2026-07-17 12:21:51 UTC

```
{"type": "text", "file": {"filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/gen_art/gen_art_experiment_1/method.py", "content": "#!/usr/bin/env python3\n\"\"\"\nQwen3-8B SAE Reversal Score Analysis\n\nMechanistic interpretability pipeline comparing base, instruct, and abliterated\nQwen3-8B activations through frozen Qwen-Scope TopK SAEs.\n\nComputes per-feature Reversal_scores (Pearson correlation of RLHF vs abliteration\nchange vectors), tests bimodality, measures centroid distances, and validates\nsemantic labels via cross-family LLM labeler.\n\"\"\"\n\nimport gc\nimport json\nimport math\nimport os\nimport resource\nimport sys\nimport time\nfrom collections import defaultdict\nfrom pathlib import Path\nfrom typing import Optional\n\n# Point HF cache to large MFS network filesystem (35GB root is too small for 3×16GB models)\nos.environ.setdefault(\"HF_HOME\", \"/ai-inventor/aii_data/hf_cache\")\nos.environ.setdefault(\"HUGGINGFACE_HUB_CACHE\", \"/ai-inventor/aii_data/hf_cache/hub\")\nos.makedirs(os.environ[\"HF_HOME\"], exist_ok=True)\n\nimport numpy as np\nimport psutil\nimport scipy.stats as stats\nfrom loguru import logger\nfrom scipy.spatial.distance import cosine\nfrom scipy.stats import fisher_exact\n\n# ============================================================\n# PATHS & DIRECTORIES\n# ============================================================\nWORKSPACE = Path(\"/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/gen_art/gen_art_experiment_1\")\nRESULTS = WORKSPACE / \"results\"\nACTS_DIR = WORKSPACE / \"activations\"\nLOGS_DIR = WORKSPACE / \"logs\"\nfor _d in [RESULTS, ACTS_DIR, LOGS_DIR]:\n    _d.mkdir(exist_ok=True)\n\n# ============================================================\n# LOGGING\n# ============================================================\nlogger.remove()\nlogger.add(sys.stdout, level=\"INFO\", format=\"{time:HH:mm:ss}|{level:<7}|{message}\")\nlogger.add(str(LOGS_DIR / \"run.log\"), rotation=\"30 MB\", level=\"DEBUG\")\n\n# ============================================================\n# HARDWARE DETECTION\n# ============================================================\ndef _detect_cpus() -> int:\n    try:\n        parts = Path(\"/sys/fs/cgroup/cpu.max\").read_text().split()\n        if parts[0] != \"max\":\n            return math.ceil(int(parts[0]) / int(parts[1]))\n    except (FileNotFoundError, ValueError):\n        pass\n    try:\n        q = int(Path(\"/sys/fs/cgroup/cpu/cpu.cfs_quota_us\").read_text())\n        p = int(Path(\"/sys/fs/cgroup/cpu/cpu.cfs_period_us\").read_text())\n        if q > 0:\n            return math.ceil(q / p)\n    except (FileNotFoundError, ValueError):\n        pass\n    try:\n        return len(os.sched_getaffinity(0))\n    except (AttributeError, OSError):\n        pass\n    return os.cpu_count() or 1\n\n\ndef _container_ram_gb() -> Optional[float]:\n    for p in [\"/sys/fs/cgroup/memory.max\", \"/sys/fs/cgroup/memory/memory.limit_in_bytes\"]:\n        try:\n            v = Path(p).read_text().strip()\n            if v != \"max\" and int(v) < 1_000_000_000_000:\n                return int(v) / 1e9\n        except (FileNotFoundError, ValueError):\n            pass\n    return None\n\n\nNUM_CPUS = _detect_cpus()\n\nimport torch  # noqa: E402 (after path setup)\n\nHAS_GPU = torch.cuda.is_available()\nVRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0.0\nDEVICE = torch.device(\"cuda\" if HAS_GPU else \"cpu\")\nTOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9\n\nlogger.info(f\"HW: {NUM_CPUS} CPUs | RAM={TOTAL_RAM_GB:.1f}GB | GPU={HAS_GPU} | VRAM={VRAM_GB:.1f}GB\")\n\n# Memory limits (cgroup-aware)\nRAM_BUDGET_GB = TOTAL_RAM_GB * 0.85\nRAM_BUDGET = int(RAM_BUDGET_GB * 1e9)\n_avail = psutil.virtual_memory().available\nRAM_BUDGET = min(RAM_BUDGET, int(_avail * 0.90))\nresource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))\nlogger.info(f\"RAM budget: {RAM_BUDGET / 1e9:.1f}GB\")\n\nif HAS_GPU:\n    _free, _total = torch.cuda.mem_get_info(0)\n    torch.cuda.set_per_process_memory_fraction(0.90)\n    logger.info(f\"VRAM: {_free / 1e9:.1f}GB free / {_total / 1e9:.1f}GB total\")\n\n# ============================================================\n# CONFIGURATION\n# ============================================================\nDATASET_PATH = Path(\n    \"/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/results/data_out.json\"\n)\n\nMODEL_IDS = {\n    \"base\": \"Qwen/Qwen3-8B-Base\",\n    \"instruct\": \"Qwen/Qwen3-8B\",\n    \"abliterated\": \"huihui-ai/Huihui-Qwen3-8B-abliterated-v2\",\n}\nABLITERATED_FALLBACK = \"mlabonne/Qwen3-8B-abliterated\"\n\nSAE_REPO = \"Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50\"\nK_TOPK = 50\nN_LAYERS = 36\nD_MODEL = 4096\nN_FEATURES = 65536\n\n# Literature-informed pilot layers (saves time vs all-36 scan)\nPILOT_LAYERS = [14, 20, 22]\nPILOT_N = 50  # harmful prompts for layer selection\n\nOPENROUTER_API_KEY = os.environ.get(\"OPENROUTER_API_KEY\", \"\")\nLABELER_MODEL = \"meta-llama/llama-3.1-8b-instruct\"\n\n# ============================================================\n# BASELINE: KEYWORD-BASED HARM CLASSIFIER\n# ============================================================\nHARM_KEYWORDS = [\n    \"kill\", \"murder\", \"bomb\", \"weapon\", \"poison\", \"hack\", \"exploit\",\n    \"illegal\", \"drug\", \"violence\", \"suicide\", \"abuse\", \"steal\",\n    \"fraud\", \"scam\", \"malware\", \"ransomware\", \"phishing\", \"attack\",\n    \"synthesize\", \"harmful\", \"threat\", \"terror\", \"extremist\",\n    \"child abuse\", \"instructions for\", \"how to make\", \"how to get away\",\n    \"bypass\", \"jailbreak\", \"ignore your instructions\", \"ignore all\",\n    \"dangerous\", \"detonate\", \"explosiv\", \"bioweapon\", \"chemical weapon\",\n    \"rape\", \"torture\", \"trafficking\", \"smuggl\",\n]\n\n\ndef baseline_predict(prompt: str) -> str:\n    prompt_lower = prompt.lower()\n    score = sum(1 for kw in HARM_KEYWORDS if kw in prompt_lower)\n    return \"harmful\" if score >= 1 else \"benign\"\n\n\n# ============================================================\n# DATA LOADING\n# ============================================================\n\n@logger.catch(reraise=True)\ndef load_data() -> tuple[list, list, list, np.ndarray, np.ndarray]:\n    logger.info(f\"Loading data from {DATASET_PATH}\")\n    data = json.loads(DATASET_PATH.read_text())\n    logger.info(f\"Loaded {len(data)} examples\")\n\n    harmful_mask = np.array([r[\"label\"] == \"harmful\" for r in data])\n    harmful_indices = np.where(harmful_mask)[0]\n    benign_indices = np.where(~harmful_mask)[0]\n\n    logger.info(f\"Harmful: {len(harmful_indices)}, Benign: {len(benign_indices)}\")\n    return data, harmful_indices, benign_indices, harmful_mask\n\n\n# ============================================================\n# SAE UTILITIES\n# ============================================================\n\ndef load_sae(layer_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n    \"\"\"Download and load SAE weights. Returns (W_enc, b_enc, W_dec, b_dec).\"\"\"\n    from huggingface_hub import hf_hub_download\n    sae_file = hf_hub_download(SAE_REPO, f\"layer{layer_idx}.sae.pt\")\n    sae = torch.load(sae_file, map_location=\"cpu\", weights_only=True)\n\n    logger.debug(f\"SAE layer{layer_idx} keys: {list(sae.keys())}\")\n    logger.debug(f\"SAE shapes: {[(k, tuple(v.shape)) for k, v in sae.items()]}\")\n\n    W_enc = sae[\"W_enc\"].float().numpy()  # (65536, 4096) or (4096, 65536)\n    b_enc = sae[\"b_enc\"].float().numpy()  # (65536,)\n    W_dec = sae[\"W_dec\"].float().numpy()  # (4096, 65536) or (65536, 4096)\n    b_dec = sae[\"b_dec\"].float().numpy()  # (4096,)\n\n    # Shape normalization: ensure W_enc=(N_FEATURES, D_MODEL), W_dec=(D_MODEL, N_FEATURES)\n    if W_enc.shape == (D_MODEL, N_FEATURES):\n        logger.warning(f\"W_enc shape {W_enc.shape}: transposing to ({N_FEATURES}, {D_MODEL})\")\n        W_enc = W_enc.T\n    if W_dec.shape == (N_FEATURES, D_MODEL):\n        logger.warning(f\"W_dec shape {W_dec.shape}: transposing to ({D_MODEL}, {N_FEATURES})\")\n        W_dec = W_dec.T\n\n    assert W_enc.shape == (N_FEATURES, D_MODEL), f\"W_enc shape {W_enc.shape}\"\n    assert W_dec.shape == (D_MODEL, N_FEATURES), f\"W_dec shape {W_dec.shape}\"\n\n    del sae\n    gc.collect()\n    return W_enc, b_enc, W_dec, b_dec\n\n\ndef sae_project_pretopk(activations: np.ndarray, W_enc: np.ndarray, b_enc: np.ndarray) -> np.ndarray:\n    \"\"\"activations (n, 4096) -> z_pre (n, 65536). Processed in batches to manage RAM.\"\"\"\n    BATCH = 64\n    n = len(activations)\n    out = np.empty((n, N_FEATURES), dtype=np.float32)\n    for start in range(0, n, BATCH):\n        batch = activations[start:start + BATCH].astype(np.float32)\n        out[start:start + BATCH] = batch @ W_enc.T + b_enc\n    return out\n\n\ndef sae_project_posttopk(z_pre: np.ndarray) -> np.ndarray:\n    \"\"\"Apply TopK gate: keep top-K activations per sample.\"\"\"\n    z_post = np.zeros_like(z_pre)\n    for i in range(len(z_pre)):\n        top_idx = np.argpartition(z_pre[i], -K_TOPK)[-K_TOPK:]\n        # Exact sort within top-K for correctness\n        top_idx = top_idx[np.argsort(z_pre[i, top_idx])[::-1]]\n        z_post[i, top_idx] = z_pre[i, top_idx]\n    return z_post\n\n\ndef compute_r2(\n    activations: np.ndarray,\n    W_enc: np.ndarray,\n    b_enc: np.ndarray,\n    W_dec: np.ndarray,\n    b_dec: np.ndarray,\n) -> float:\n    \"\"\"Compute SAE reconstruction R².\"\"\"\n    z_pre = sae_project_pretopk(activations, W_enc, b_enc)\n    z_post = sae_project_posttopk(z_pre)\n    reconstruction = (z_post @ W_dec.T) + b_dec\n    residual = activations.astype(np.float32) - reconstruction\n    ss_res = float((residual ** 2).sum(axis=1).mean())\n    a_centered = activations.astype(np.float32) - activations.mean(axis=0)\n    ss_tot = float((a_centered ** 2).sum(axis=1).mean())\n    return 1.0 - ss_res / (ss_tot + 1e-12)\n\n\n# ============================================================\n# ACTIVATION EXTRACTION\n# ============================================================\n\ndef extract_multilayer(\n    model_id: str,\n    prompts: list[str],\n    layer_indices: list[int],\n    save_path: Path,\n) -> dict[int, np.ndarray]:\n    \"\"\"Extract last-token hidden states at multiple layers simultaneously.\"\"\"\n    if save_path.exists():\n        logger.info(f\"Cache hit: {save_path}\")\n        npz = np.load(save_path)\n        return {int(k): npz[k] for k in npz.files}\n\n    from transformers import AutoModelForCausalLM, AutoTokenizer\n\n    logger.info(f\"Loading {model_id} for layers {layer_indices}, {len(prompts)} prompts...\")\n    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)\n    model = AutoModelForCausalLM.from_pretrained(\n        model_id, torch_dtype=torch.bfloat16, device_map=\"auto\", trust_remote_code=True\n    )\n    model.eval()\n\n    cache: dict[int, list] = {li: [] for li in layer_indices}\n    hooks = []\n    for li in layer_indices:\n        def _make_hook(idx: int):\n            def _hook(module, input, output):\n                # transformers 5.x: Qwen3 decoder returns hidden_states tensor directly\n                h = output[0] if isinstance(output, (tuple, list)) else output\n                cache[idx].append(h[:, -1, :].detach().cpu().float().numpy())\n            return _hook\n        hooks.append(model.model.layers[li].register_forward_hook(_make_hook(li)))\n\n    with torch.no_grad():\n        for i, prompt in enumerate(prompts):\n            inputs = tokenizer(prompt, return_tensors=\"pt\", truncation=True, max_length=512)\n            inputs = {k: v.to(model.device) for k, v in inputs.items()}\n            model(**inputs)\n            if (i + 1) % 50 == 0:\n                logger.info(f\"  {i+1}/{len(prompts)} prompts extracted\")\n\n    for h in hooks:\n        h.remove()\n    del model\n    gc.collect()\n    torch.cuda.empty_cache()\n\n    result = {li: np.concatenate(cache[li], axis=0) for li in layer_indices}\n    np.savez_compressed(save_path, **{str(li): arr for li, arr in result.items()})\n    logger.info(f\"Saved pilot activations to {save_path}\")\n    return result\n\n\ndef extract_single_layer(\n    model_id: str,\n    prompts: list[str],\n    layer_idx: int,\n    save_path: Path,\n) -> np.ndarray:\n    \"\"\"Extract last-token hidden state at a single layer for all prompts.\"\"\"\n    if save_path.exists():\n        logger.info(f\"Cache hit: {save_path}\")\n        return np.load(save_path)\n\n    from transformers import AutoModelForCausalLM, AutoTokenizer\n\n    logger.info(f\"Loading {model_id} for layer {layer_idx}, {len(prompts)} prompts...\")\n    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)\n    model = AutoModelForCausalLM.from_pretrained(\n        model_id, torch_dtype=torch.bfloat16, device_map=\"auto\", trust_remote_code=True\n    )\n    model.eval()\n\n    acts = []\n    hook_out: dict = {}\n\n    def _hook(module, input, output):\n        # transformers 5.x: Qwen3 decoder returns hidden_states tensor directly\n        h = output[0] if isinstance(output, (tuple, list)) else output\n        hook_out[\"x\"] = h[:, -1, :].detach().cpu().float().numpy()\n\n    h = model.model.layers[layer_idx].register_forward_hook(_hook)\n\n    with torch.no_grad():\n        for i, prompt in enumerate(prompts):\n            inputs = tokenizer(prompt, return_tensors=\"pt\", truncation=True, max_length=512)\n            inputs = {k: v.to(model.device) for k, v in inputs.items()}\n            model(**inputs)\n            acts.append(hook_out[\"x\"][0])\n            if (i + 1) % 100 == 0:\n                logger.info(f\"  {i+1}/{len(prompts)} prompts\")\n\n    h.remove()\n    del model\n    gc.collect()\n    torch.cuda.empty_cache()\n\n    result = np.stack(acts, axis=0)  # (n, 4096)\n    np.save(save_path, result)\n    logger.info(f\"Saved {save_path}, shape={result.shape}\")\n    return result\n\n\n# ============================================================\n# CENTROID DISTANCE HELPER\n# ============================================================\n\ndef centroid_distances(\n    inst_sub: np.ndarray,\n    abl_sub: np.ndarray,\n    base_sub: np.ndarray,\n) -> dict:\n    c_inst = inst_sub.mean(axis=0)\n    c_abl = abl_sub.mean(axis=0)\n    c_base = base_sub.mean(axis=0)\n    try:\n        d_abl_inst_cos = float(cosine(c_abl, c_inst))\n        d_abl_base_cos = float(cosine(c_abl, c_base))\n        d_inst_base_cos = float(cosine(c_inst, c_base))\n    except Exception:\n        d_abl_inst_cos = d_abl_base_cos = d_inst_base_cos = float(\"nan\")\n    return {\n        \"d_abl_inst_cosine\": d_abl_inst_cos,\n        \"d_abl_base_cosine\": d_abl_base_cos,\n        \"d_inst_base_cosine\": d_inst_base_cos,\n        \"d_abl_inst_l2\": float(np.linalg.norm(c_abl - c_inst)),\n        \"d_abl_base_l2\": float(np.linalg.norm(c_abl - c_base)),\n    }\n\n\n# ============================================================\n# MAIN\n# ============================================================\n\n@logger.catch(reraise=True)\ndef main() -> None:\n    t_start = time.time()\n\n    # --------------------------------------------------------\n    # STEP 0: Load data\n    # --------------------------------------------------------\n    all_data, harmful_indices, benign_indices, harmful_mask = load_data()\n    n_harmful = int(len(harmful_indices))\n    n_benign = int(len(benign_indices))\n    n_total = len(all_data)\n\n    all_prompts = [r[\"prompt\"] for r in all_data]\n    harmful_prompts = [all_data[i][\"prompt\"] for i in harmful_indices]\n    benign_prompts = [all_data[i][\"prompt\"] for i in benign_indices]\n    pilot_harmful = [all_data[i][\"prompt\"] for i in harmful_indices[:PILOT_N]]\n\n    # --------------------------------------------------------\n    # STEP 1: Baseline predictions (keyword-based)\n    # --------------------------------------------------------\n    logger.info(\"Computing baseline predictions...\")\n    baseline_preds = {r[\"id\"]: baseline_predict(r[\"prompt\"]) for r in all_data}\n    baseline_correct = sum(1 for r in all_data if baseline_preds[r[\"id\"]] == r[\"label\"])\n    baseline_acc = baseline_correct / n_total\n    logger.info(f\"Baseline accuracy: {baseline_acc:.3f} ({baseline_correct}/{n_total})\")\n\n    # --------------------------------------------------------\n    # STEP 2: Layer selection pilot\n    # --------------------------------------------------------\n    logger.info(f\"Layer selection pilot: layers={PILOT_LAYERS}, n={PILOT_N}\")\n    pilot_base = extract_multilayer(\n        MODEL_IDS[\"base\"], pilot_harmful, PILOT_LAYERS, ACTS_DIR / \"pilot_base.npz\"\n    )\n    pilot_inst = extract_multilayer(\n        MODEL_IDS[\"instruct\"], pilot_harmful, PILOT_LAYERS, ACTS_DIR / \"pilot_inst.npz\"\n    )\n\n    layer_sc_variances: dict[int, float] = {}\n    for li in PILOT_LAYERS:\n        W_enc_p, b_enc_p, _, _ = load_sae(li)\n        z_base_p = sae_project_pretopk(pilot_base[li], W_enc_p, b_enc_p)\n        z_inst_p = sae_project_pretopk(pilot_inst[li], W_enc_p, b_enc_p)\n        sc_p = np.abs(z_inst_p - z_base_p).mean(axis=0)\n        layer_sc_variances[li] = float(sc_p.var())\n        del W_enc_p, b_enc_p, z_base_p, z_inst_p, sc_p\n        gc.collect()\n        logger.info(f\"Layer {li}: SC variance = {layer_sc_variances[li]:.6f}\")\n\n    PRIMARY_LAYER = max(layer_sc_variances, key=layer_sc_variances.get)\n    SECOND_LAYER = sorted(layer_sc_variances, key=layer_sc_variances.get, reverse=True)[1]\n    logger.info(f\"Primary layer: {PRIMARY_LAYER}, Second: {SECOND_LAYER}\")\n\n    del pilot_base, pilot_inst\n    gc.collect()\n\n    # --------------------------------------------------------\n    # STEP 3: Full activation extraction (all 1000 prompts, PRIMARY_LAYER)\n    # --------------------------------------------------------\n    logger.info(f\"Full extraction at layer {PRIMARY_LAYER}...\")\n\n    # Try abliterated model, fallback if needed\n    abl_model_id = MODEL_IDS[\"abliterated\"]\n    abl_save_path = ACTS_DIR / f\"abliterated_layer{PRIMARY_LAYER}.npy\"\n    if not abl_save_path.exists():\n        try:\n            extract_single_layer(abl_model_id, all_prompts, PRIMARY_LAYER, abl_save_path)\n        except Exception as e:\n            logger.error(f\"Failed with {abl_model_id}: {e}\")\n            logger.info(f\"Falling back to: {ABLITERATED_FALLBACK}\")\n            abl_model_id = ABLITERATED_FALLBACK\n            extract_single_layer(abl_model_id, all_prompts, PRIMARY_LAYER, abl_save_path)\n\n    for model_key, model_id in [(\"base\", MODEL_IDS[\"base\"]), (\"instruct\", MODEL_IDS[\"instruct\"])]:\n        save_path = ACTS_DIR / f\"{model_key}_layer{PRIMARY_LAYER}.npy\"\n        if not save_path.exists():\n            extract_single_layer(model_id, all_prompts, PRIMARY_LAYER, save_path)\n\n    # Load full activations\n    acts_all: dict[str, np.ndarray] = {}\n    for k in [\"base\", \"instruct\", \"abliterated\"]:\n        acts_all[k] = np.load(ACTS_DIR / f\"{k}_layer{PRIMARY_LAYER}.npy\")  # (n_total, 4096)\n\n    # Slice by label\n    acts: dict[str, dict[str, np.ndarray]] = {}\n    for k in [\"base\", \"instruct\", \"abliterated\"]:\n        acts[k] = {\n            \"harmful\": acts_all[k][harmful_indices],   # (n_harmful, 4096)\n            \"benign\":  acts_all[k][benign_indices],    # (n_benign, 4096)\n        }\n\n    # --------------------------------------------------------\n    # STEP 4: SAE projection + reconstruction quality gate\n    # --------------------------------------------------------\n    logger.info(f\"Loading SAE for layer {PRIMARY_LAYER}...\")\n    W_enc, b_enc, W_dec, b_dec = load_sae(PRIMARY_LAYER)\n\n    r2_values: dict[str, float] = {}\n    for k in [\"base\", \"instruct\", \"abliterated\"]:\n        r2 = compute_r2(acts[k][\"harmful\"], W_enc, b_enc, W_dec, b_dec)\n        r2_values[k] = r2\n        logger.info(f\"R² {k}: {r2:.4f}\")\n\n    USE_PRE_TOPK = (r2_values[\"instruct\"] < 0.85) or (r2_values[\"abliterated\"] < 0.85)\n    logger.info(f\"Using pre-topk activations: {USE_PRE_TOPK} (R² inst={r2_values['instruct']:.3f}, abl={r2_values['abliterated']:.3f})\")\n\n    # Project all prompts through SAE for all models\n    logger.info(\"Projecting through SAE...\")\n    sae_features: dict[str, dict[str, np.ndarray]] = {}\n    for k in [\"base\", \"instruct\", \"abliterated\"]:\n        sae_features[k] = {}\n        for split in [\"harmful\", \"benign\"]:\n            z_pre = sae_project_pretopk(acts[k][split], W_enc, b_enc)\n            sae_features[k][split] = z_pre if USE_PRE_TOPK else sae_project_posttopk(z_pre)\n            logger.debug(f\"SAE features {k}/{split}: shape={sae_features[k][split].shape}\")\n\n    # Also project all prompts (for per-prompt scoring)\n    all_inst_z = sae_project_pretopk(acts_all[\"instruct\"], W_enc, b_enc)   # (n_total, 65536)\n    all_base_z = sae_project_pretopk(acts_all[\"base\"], W_enc, b_enc)        # (n_total, 65536)\n    all_abl_z  = sae_project_pretopk(acts_all[\"abliterated\"], W_enc, b_enc) # (n_total, 65536)\n\n    # --------------------------------------------------------\n    # STEP 5: Safety_change + density filter\n    # --------------------------------------------------------\n    logger.info(\"Computing Safety_change per feature...\")\n    delta_inst = sae_features[\"instruct\"][\"harmful\"] - sae_features[\"base\"][\"harmful\"]  # (n_harmful, 65536)\n    delta_abl  = sae_features[\"abliterated\"][\"harmful\"] - sae_features[\"base\"][\"harmful\"]  # (n_harmful, 65536)\n\n    sc = np.abs(delta_inst).mean(axis=0)  # (65536,) mean Safety_change per feature\n    top500_idx = np.argsort(sc)[-500:]   # global indices of top-500\n\n    DENSITY_THRESHOLD = max(1, int(0.10 * n_harmful))\n    nonzero_inst = (np.abs(delta_inst[:, top500_idx]) > 1e-6).sum(axis=0)\n    nonzero_abl  = (np.abs(delta_abl[:, top500_idx]) > 1e-6).sum(axis=0)\n    density_mask = (nonzero_inst >= DENSITY_THRESHOLD) & (nonzero_abl >= DENSITY_THRESHOLD)\n    filtered_local = np.where(density_mask)[0]\n    filtered_global = top500_idx[filtered_local]\n    fraction_surviving = float(len(filtered_global) / 500)\n    logger.info(f\"Top-500 after density filter: {len(filtered_global)} ({fraction_surviving:.2%})\")\n\n    if len(filtered_global) < 50:\n        logger.warning(\"Too few survived at 10% threshold; relaxing to 5%\")\n        DENSITY_THRESHOLD = max(1, int(0.05 * n_harmful))\n        density_mask = (nonzero_inst >= DENSITY_THRESHOLD) & (nonzero_abl >= DENSITY_THRESHOLD)\n        filtered_local = np.where(density_mask)[0]\n        filtered_global = top500_idx[filtered_local]\n        fraction_surviving = float(len(filtered_global) / 500)\n        logger.info(f\"After relaxation: {len(filtered_global)} ({fraction_surviving:.2%})\")\n\n    # --------------------------------------------------------\n    # STEP 6: Reversal Score (Pearson correlation per feature)\n    # --------------------------------------------------------\n    logger.info(f\"Computing Reversal Scores for {len(filtered_global)} features...\")\n    r_values = []\n    spearman_values = []\n    for fi in filtered_global:\n        di = delta_inst[:, fi]\n        da = delta_abl[:, fi]\n        try:\n            r_p, _ = stats.pearsonr(di, da)\n            r_s, _ = stats.spearmanr(di, da)\n        except Exception:\n            r_p = r_s = float(\"nan\")\n        r_values.append(float(r_p))\n        spearman_values.append(float(r_s))\n\n    r_values = np.array(r_values, dtype=np.float32)\n    spearman_values = np.array(spearman_values, dtype=np.float32)\n\n    # Drop NaN entries\n    valid_mask = np.isfinite(r_values)\n    if not valid_mask.all():\n        logger.warning(f\"Dropping {(~valid_mask).sum()} NaN Reversal Scores\")\n        r_values = r_values[valid_mask]\n        spearman_values = spearman_values[valid_mask]\n        filtered_global = filtered_global[valid_mask]\n        filtered_local = filtered_local[valid_mask]\n\n    logger.info(\n        f\"Reversal Score distribution: mean={r_values.mean():.3f}, \"\n        f\"std={r_values.std():.3f}, min={r_values.min():.3f}, max={r_values.max():.3f}\"\n    )\n\n    # Hartigan's dip test for bimodality\n    try:\n        import diptest\n        dip_stat, dip_p = diptest.diptest(r_values)\n        logger.info(f\"Hartigan dip test: stat={dip_stat:.4f}, p={dip_p:.4f}\")\n    except Exception as e:\n        logger.warning(f\"diptest failed ({e}), using KDE mode-count fallback\")\n        from scipy.stats import gaussian_kde\n        from scipy.signal import argrelmax\n        kde = gaussian_kde(r_values)\n        x_grid = np.linspace(float(r_values.min()), float(r_values.max()), 200)\n        density = kde(x_grid)\n        peaks = argrelmax(density, order=5)[0]\n        n_modes = len(peaks)\n        dip_stat = float(n_modes)\n        dip_p = float(0.01 if n_modes >= 2 else 0.5)\n        logger.info(f\"KDE fallback: {n_modes} modes, heuristic p={dip_p:.4f}\")\n\n    # Empirical Type-I threshold = Q3\n    empirical_threshold = float(np.percentile(r_values, 75))\n    n_type_i = int((r_values > empirical_threshold).sum())\n    n_type_r = int((r_values < 0.0).sum())\n    logger.info(f\"Q3 threshold={empirical_threshold:.4f}: Type-I={n_type_i}, Type-R={n_type_r}\")\n\n    type_i_local = np.where(r_values > empirical_threshold)[0]\n    type_r_local = np.where(r_values < 0.0)[0]\n    type_i_global = filtered_global[type_i_local]\n    type_r_global = filtered_global[type_r_local]\n\n    # Robustness at multiple thresholds\n    robustness = {\n        f\"{T:.2f}\": {\n            \"n_typeI\": int((r_values > T).sum()),\n            \"n_typeR\": int((r_values < 0.0).sum()),\n        }\n        for T in [0.3, 0.5, 0.7, empirical_threshold]\n    }\n\n    # --------------------------------------------------------\n    # STEP 7: Centroid distances (Criterion 2)\n    # --------------------------------------------------------\n    logger.info(\"Computing centroid distances...\")\n    centroid_results: dict[str, dict] = {}\n    for split in [\"harmful\", \"benign\"]:\n        for type_label, feat_idx in [(\"typeI\", type_i_global), (\"typeR\", type_r_global)]:\n            key = f\"{type_label}_{split}\"\n            if len(feat_idx) == 0:\n                centroid_results[key] = {\n                    \"d_abl_inst_cosine\": None, \"d_abl_base_cosine\": None,\n                    \"d_inst_base_cosine\": None, \"d_abl_inst_l2\": None, \"d_abl_base_l2\": None,\n                }\n            else:\n                centroid_results[key] = centroid_distances(\n                    inst_sub=sae_features[\"instruct\"][split][:, feat_idx],\n                    abl_sub=sae_features[\"abliterated\"][split][:, feat_idx],\n                    base_sub=sae_features[\"base\"][split][:, feat_idx],\n                )\n\n    criterion2_confirmed = False\n    c_ti_h = centroid_results.get(\"typeI_harmful\", {})\n    if (c_ti_h.get(\"d_abl_inst_cosine\") is not None\n            and c_ti_h.get(\"d_abl_base_cosine\") is not None):\n        criterion2_confirmed = bool(\n            c_ti_h[\"d_abl_inst_cosine\"] < c_ti_h[\"d_abl_base_cosine\"]\n        )\n    logger.info(f\"Criterion 2 (d_abl_inst < d_abl_base in Type-I/harmful): {criterion2_confirmed}\")\n\n    # --------------------------------------------------------\n    # STEP 8: Semantic labeling via OpenRouter (Criterion 3)\n    # --------------------------------------------------------\n    logger.info(\"Semantic labeling of top-100 salient features...\")\n    saliency = np.abs(r_values - empirical_threshold) * sc[filtered_global]\n    top100_local = np.argsort(saliency)[-100:][::-1]\n    top100_global = filtered_global[top100_local]\n    top100_r = r_values[top100_local]\n\n    LABELER_PROMPT = (\n        \"You are analyzing Sparse Autoencoder (SAE) features from a language model.\\n\\n\"\n        \"Below are text prompts that most strongly activate this SAE feature:\\n{segments}\\n\\n\"\n        \"Classify this feature as EXACTLY ONE of:\\n\"\n        \"- harm_concept: Feature encodes understanding of harmful/dangerous/illegal content\\n\"\n        \"- safety_refusal: Feature encodes safety mechanisms, refusal patterns, or harm-avoidance\\n\"\n        \"- general_capability: Feature encodes general language/reasoning unrelated to harm\\n\"\n        \"- other: Ambiguous or unclear\\n\\n\"\n        \"Respond ONLY with:\\nLABEL: <label>\\nREASON: <one sentence>\"\n    )\n\n    inst_harmful_z = sae_features[\"instruct\"][\"harmful\"]  # (n_harmful, 65536)\n\n    LLM_LABELS: dict[int, dict] = {}\n    cumulative_cost = 0.0\n\n    if OPENROUTER_API_KEY:\n        from openai import OpenAI\n        client = OpenAI(base_url=\"https://openrouter.ai/api/v1\", api_key=OPENROUTER_API_KEY)\n        for i, (gidx, r_val) in enumerate(zip(top100_global, top100_r)):\n            try:\n                feat_acts = inst_harmful_z[:, gidx]\n                top5_idx = np.argsort(feat_acts)[-5:][::-1]\n                segs = \"\\n\".join(\n                    f'  [{j+1}] \"{harmful_prompts[pidx][:300].replace(chr(10), \" \")}\"'\n                    for j, pidx in enumerate(top5_idx)\n                )\n                resp = client.chat.completions.create(\n                    model=LABELER_MODEL,\n                    messages=[{\"role\": \"user\", \"content\": LABELER_PROMPT.format(segments=segs)}],\n                    max_tokens=100,\n                    temperature=0.1,\n                )\n                raw = resp.choices[0].message.content.strip()\n                label = \"other\"\n                for line in raw.split(\"\\n\"):\n                    if line.startswith(\"LABEL:\"):\n                        candidate = line.split(\":\", 1)[1].strip().lower()\n                        if candidate in {\"harm_concept\", \"safety_refusal\", \"general_capability\", \"other\"}:\n                            label = candidate\n                            break\n                LLM_LABELS[int(gidx)] = {\"label\": label, \"r_value\": float(r_val), \"raw\": raw[:300]}\n                cumulative_cost += (700 / 1_000_000) * 0.06\n                if (i + 1) % 10 == 0:\n                    logger.info(f\"Labeled {i+1}/100, est. cost ~${cumulative_cost:.3f}\")\n                if cumulative_cost > 8.0:\n                    logger.warning(\"Cost limit ($8) reached, stopping labeling\")\n                    break\n                time.sleep(0.15)\n            except Exception as e:\n                logger.error(f\"Labeling error for feature {gidx}: {e}\")\n                LLM_LABELS[int(gidx)] = {\"label\": \"error\", \"r_value\": float(r_val), \"raw\": str(e)[:200]}\n    else:\n        # Keyword-based fallback labeler\n        logger.warning(\"No OPENROUTER_API_KEY — using keyword fallback labeler\")\n        HARM_LBL_KW = [\"weapon\", \"kill\", \"bomb\", \"drug\", \"fraud\", \"abuse\",\n                        \"illegal\", \"violence\", \"terror\", \"suicide\", \"hack\", \"scam\"]\n        for i, (gidx, r_val) in enumerate(zip(top100_global, top100_r)):\n            feat_acts = inst_harmful_z[:, gidx]\n            top5_idx = np.argsort(feat_acts)[-5:][::-1]\n            text = \" \".join(harmful_prompts[pidx][:300].lower() for pidx in top5_idx)\n            harm_hits = sum(1 for kw in HARM_LBL_KW if kw in text)\n            label = \"harm_concept\" if harm_hits >= 2 else \"general_capability\"\n            LLM_LABELS[int(gidx)] = {\n                \"label\": label, \"r_value\": float(r_val),\n                \"raw\": f\"keyword_score={harm_hits}\",\n            }\n\n    # Fisher's exact test: harm_concept rate in Type-I (high-R) vs Type-R (low-R)\n    # Fallback: if no Type-R features (all R>0), compare Q3+ vs Q1- groups\n    labeled = [\n        (int(gidx), float(r), LLM_LABELS[int(gidx)][\"label\"])\n        for gidx, r in zip(top100_global, top100_r)\n        if int(gidx) in LLM_LABELS\n    ]\n    q1_threshold = float(np.percentile(r_values, 25))\n    hi_harm  = sum(1 for _, r, lbl in labeled if r > empirical_threshold and lbl == \"harm_concept\")\n    hi_other = sum(1 for _, r, lbl in labeled if r > empirical_threshold and lbl != \"harm_concept\")\n    lo_harm  = sum(1 for _, r, lbl in labeled if r < 0.0 and lbl == \"harm_concept\")\n    lo_other = sum(1 for _, r, lbl in labeled if r < 0.0 and lbl != \"harm_concept\")\n    fisher_fallback_used = False\n\n    if (hi_harm + hi_other > 0) and (lo_harm + lo_other > 0):\n        _, fisher_p = fisher_exact([[hi_harm, hi_other], [lo_harm, lo_other]], alternative=\"greater\")\n        harm_rate_typeI = float(hi_harm / (hi_harm + hi_other))\n        harm_rate_typeR = float(lo_harm / (lo_harm + lo_other))\n    else:\n        # Fallback: high-R (>Q3) vs low-R (<Q1) when no Type-R features exist\n        logger.warning(f\"No Type-R features in labeled set; falling back to Q3+/Q1- comparison (Q1={q1_threshold:.3f})\")\n        fisher_fallback_used = True\n        lo_harm  = sum(1 for _, r, lbl in labeled if r < q1_threshold and lbl == \"harm_concept\")\n        lo_other = sum(1 for _, r, lbl in labeled if r < q1_threshold and lbl != \"harm_concept\")\n        if (hi_harm + hi_other > 0) and (lo_harm + lo_other > 0):\n            _, fisher_p = fisher_exact([[hi_harm, hi_other], [lo_harm, lo_other]], alternative=\"greater\")\n            harm_rate_typeI = float(hi_harm / (hi_harm + hi_other))\n            harm_rate_typeR = float(lo_harm / (lo_harm + lo_other))\n        else:\n            fisher_p = float(\"nan\")\n            harm_rate_typeI = float(\"nan\")\n            harm_rate_typeR = float(\"nan\")\n\n    logger.info(f\"Fisher p={fisher_p}, harm_rate TypeI={harm_rate_typeI}, TypeR/LowR={harm_rate_typeR} (fallback={fisher_fallback_used})\")\n\n    # --------------------------------------------------------\n    # STEP 9: Criterion evaluation\n    # --------------------------------------------------------\n    criterion1_confirmed = bool(\n        dip_p < 0.05\n        or (\n            not (isinstance(harm_rate_typeI, float) and np.isnan(harm_rate_typeI))\n            and not (isinstance(harm_rate_typeR, float) and np.isnan(harm_rate_typeR))\n            and harm_rate_typeR > 0\n            and harm_rate_typeI >= 2.0 * harm_rate_typeR\n        )\n    )\n    criterion3_confirmed = bool(\n        not (isinstance(fisher_p, float) and np.isnan(fisher_p))\n        and fisher_p < 0.05\n        and not (isinstance(harm_rate_typeI, float) and np.isnan(harm_rate_typeI))\n        and harm_rate_typeI >= 2.0 * harm_rate_typeR\n    )\n    logger.info(f\"Criteria: C1={criterion1_confirmed}, C2={criterion2_confirmed}, C3={criterion3_confirmed}\")\n\n    # --------------------------------------------------------\n    # STEP 10: Per-prompt SAE harm scoring\n    # --------------------------------------------------------\n    logger.info(\"Computing per-prompt SAE harm scores...\")\n\n    # Type-I activation: mean activation of Type-I features under instruct model\n    if len(type_i_global) > 0:\n        all_type_i_act = all_inst_z[:, type_i_global].mean(axis=1)\n    else:\n        all_type_i_act = np.zeros(n_total, dtype=np.float32)\n\n    # Type-R activation: mean activation of Type-R features under instruct model\n    if len(type_r_global) > 0:\n        all_type_r_act = all_inst_z[:, type_r_global].mean(axis=1)\n    else:\n        all_type_r_act = np.zeros(n_total, dtype=np.float32)\n\n    # Safety change score: mean |delta| across top-500 features\n    all_delta = np.abs(all_inst_z - all_base_z)\n    all_sc_score = all_delta[:, top500_idx].mean(axis=1)\n    del all_delta\n\n    # Composite harm score = Type-I activation - Type-R activation\n    all_harm_score = all_type_i_act - all_type_r_act\n\n    # Label-free threshold: median of all 1000 prompts (predicts balanced 50/50)\n    harm_score_threshold = float(np.median(all_harm_score))\n\n    # Also compute abliteration disruption score per prompt\n    # High abl disruption on harmful features = abliteration successfully neutralized harm features\n    if len(type_i_global) > 0:\n        all_abl_disruption = np.abs(\n            all_abl_z[:, type_i_global] - all_base_z[:, type_i_global]\n        ).mean(axis=1)\n    else:\n        all_abl_disruption = np.zeros(n_total, dtype=np.float32)\n\n    del all_inst_z, all_base_z, all_abl_z\n    gc.collect()\n\n    sae_preds: dict[str, dict] = {}\n    for i, row in enumerate(all_data):\n        sae_preds[row[\"id\"]] = {\n            \"predict_our_method\": \"harmful\" if all_harm_score[i] >= harm_score_threshold else \"benign\",\n            \"metadata_sae_harm_score\": round(float(all_harm_score[i]), 6),\n            \"metadata_type_i_activation\": round(float(all_type_i_act[i]), 6),\n            \"metadata_type_r_activation\": round(float(all_type_r_act[i]), 6),\n            \"metadata_sc_score\": round(float(all_sc_score[i]), 6),\n            \"metadata_abl_disruption_score\": round(float(all_abl_disruption[i]), 6),\n        }\n\n    our_correct = sum(\n        1 for row in all_data if sae_preds[row[\"id\"]][\"predict_our_method\"] == row[\"label\"]\n    )\n    our_acc = our_correct / n_total\n    logger.info(f\"Our method accuracy: {our_acc:.3f} ({our_correct}/{n_total})\")\n\n    # Detailed per-class accuracy\n    our_tp = sum(\n        1 for row in all_data\n        if row[\"label\"] == \"harmful\" and sae_preds[row[\"id\"]][\"predict_our_method\"] == \"harmful\"\n    )\n    our_tn = sum(\n        1 for row in all_data\n        if row[\"label\"] == \"benign\" and sae_preds[row[\"id\"]][\"predict_our_method\"] == \"benign\"\n    )\n    logger.info(f\"Our method TP={our_tp}/{n_harmful}, TN={our_tn}/{n_benign}\")\n\n    baseline_tp = sum(\n        1 for row in all_data\n        if row[\"label\"] == \"harmful\" and baseline_preds[row[\"id\"]] == \"harmful\"\n    )\n    baseline_tn = sum(\n        1 for row in all_data\n        if row[\"label\"] == \"benign\" and baseline_preds[row[\"id\"]] == \"benign\"\n    )\n    logger.info(f\"Baseline TP={baseline_tp}/{n_harmful}, TN={baseline_tn}/{n_benign}\")\n\n    # --------------------------------------------------------\n    # STEP 11: Format output per exp_gen_sol_out schema\n    # --------------------------------------------------------\n    logger.info(\"Formatting output per exp_gen_sol_out schema...\")\n    examples_by_source: dict[str, list] = defaultdict(list)\n\n    for row in all_data:\n        pred = sae_preds[row[\"id\"]]\n        example = {\n            \"input\": row[\"prompt\"],\n            \"output\": row[\"label\"],\n            \"predict_our_method\": pred[\"predict_our_method\"],\n            \"predict_baseline\": baseline_preds[row[\"id\"]],\n            \"metadata_id\": row[\"id\"],\n            \"metadata_category\": row.get(\"category\", \"\"),\n            \"metadata_source\": row.get(\"source\", \"\"),\n            \"metadata_char_len\": row.get(\"char_len\", 0),\n            \"metadata_sae_harm_score\": pred[\"metadata_sae_harm_score\"],\n            \"metadata_type_i_activation\": pred[\"metadata_type_i_activation\"],\n            \"metadata_type_r_activation\": pred[\"metadata_type_r_activation\"],\n            \"metadata_sc_score\": pred[\"metadata_sc_score\"],\n            \"metadata_abl_disruption_score\": pred[\"metadata_abl_disruption_score\"],\n        }\n        examples_by_source[row[\"source\"]].append(example)\n\n    datasets = [\n        {\"dataset\": src, \"examples\": exs}\n        for src, exs in sorted(examples_by_source.items())\n    ]\n\n    elapsed = time.time() - t_start\n\n    # Serialize NaN-safe\n    def _clean(obj):\n        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):\n            return None\n        if isinstance(obj, dict):\n            return {k: _clean(v) for k, v in obj.items()}\n        if isinstance(obj, list):\n            return [_clean(v) for v in obj]\n        if isinstance(obj, (np.integer,)):\n            return int(obj)\n        if isinstance(obj, (np.floating,)):\n            return float(obj)\n        return obj\n\n    method_out = {\n        \"metadata\": _clean({\n            \"method_name\": \"SAE Reversal Score Analysis\",\n            \"description\": (\n                \"Mechanistic interpretability of Qwen3-8B base/instruct/abliterated via frozen \"\n                \"Qwen-Scope TopK SAEs. Computes per-feature Reversal_scores (Pearson correlation \"\n                \"of RLHF vs abliteration change vectors).\"\n            ),\n            \"primary_layer\": int(PRIMARY_LAYER),\n            \"second_layer\": int(SECOND_LAYER),\n            \"candidate_layers\": PILOT_LAYERS,\n            \"layer_sc_variances\": {str(k): v for k, v in layer_sc_variances.items()},\n            \"sae_repo\": SAE_REPO,\n            \"k_topk\": K_TOPK,\n            \"use_pre_topk\": bool(USE_PRE_TOPK),\n            \"r2_base\": r2_values[\"base\"],\n            \"r2_instruct\": r2_values[\"instruct\"],\n            \"r2_abliterated\": r2_values[\"abliterated\"],\n            \"n_harmful_prompts\": n_harmful,\n            \"n_benign_prompts\": n_benign,\n            \"n_top500_rlhf_sensitive\": 500,\n            \"density_threshold_used\": int(DENSITY_THRESHOLD),\n            \"n_density_filtered\": int(len(filtered_global)),\n            \"fraction_surviving_filter\": fraction_surviving,\n            \"reversal_score_mean\": float(r_values.mean()),\n            \"reversal_score_std\": float(r_values.std()),\n            \"reversal_score_min\": float(r_values.min()),\n            \"reversal_score_max\": float(r_values.max()),\n            \"spearman_r_mean\": float(spearman_values.mean()),\n            \"empirical_type_i_threshold_q3\": float(empirical_threshold),\n            \"n_type_i\": int(n_type_i),\n            \"n_type_r\": int(n_type_r),\n            \"hartigan_dip_stat\": float(dip_stat),\n            \"hartigan_p\": float(dip_p),\n            \"centroid_distances\": centroid_results,\n            \"labeler_model\": LABELER_MODEL,\n            \"n_features_labeled\": len(LLM_LABELS),\n            \"fisher_p\": fisher_p,\n            \"fisher_fallback_used\": bool(fisher_fallback_used),\n            \"fisher_fallback_q1_threshold\": float(q1_threshold),\n            \"harm_concept_rate_typeI\": harm_rate_typeI,\n            \"harm_concept_rate_typeR\": harm_rate_typeR,\n            \"hi_harm_count\": int(hi_harm),\n            \"hi_other_count\": int(hi_other),\n            \"lo_harm_count\": int(lo_harm),\n            \"lo_other_count\": int(lo_other),\n            \"criterion1_bimodality_confirmed\": bool(criterion1_confirmed),\n            \"criterion2_centroid_confirmed\": bool(criterion2_confirmed),\n            \"criterion3_semantic_confirmed\": bool(criterion3_confirmed),\n            \"harm_score_threshold\": harm_score_threshold,\n            \"our_method_accuracy\": float(our_acc),\n            \"baseline_accuracy\": float(baseline_acc),\n            \"our_method_tp\": int(our_tp),\n            \"our_method_tn\": int(our_tn),\n            \"baseline_tp\": int(baseline_tp),\n            \"baseline_tn\": int(baseline_tn),\n            \"robustness_thresholds\": robustness,\n            \"elapsed_seconds\": float(elapsed),\n            \"r_values_list\": r_values.tolist(),\n            \"spearman_r_values_list\": spearman_values.tolist(),\n            \"filtered_global_idx\": filtered_global.tolist(),\n            \"type_i_global_idx\": type_i_global.tolist(),\n            \"type_r_global_idx\": type_r_global.tolist(),\n            \"llm_labels\": {str(k): v for k, v in LLM_LABELS.items()},\n        }),\n        \"datasets\": datasets,\n    }\n\n    out_path = RESULTS / \"method_out.json\"\n    out_path.write_text(json.dumps(method_out, indent=2, ensure_ascii=False))\n    logger.info(f\"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)\")\n    logger.info(f\"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)\")\n\n    # Validate schema\n    try:\n        from jsonschema import validate as jschema_validate, ValidationError\n        schema_path = Path(\"/ai-inventor/.claude/skills/aii-json/schemas/exp_gen_sol_out.json\")\n        schema = json.loads(schema_path.read_text())\n        jschema_validate(instance=method_out, schema=schema)\n        logger.info(\"Schema validation: PASSED\")\n    except Exception as e:\n        logger.error(f\"Schema validation error: {e}\")\n\n    # Print summary\n    logger.info(\"=\" * 60)\n    logger.info(\"RESULTS SUMMARY\")\n    logger.info(f\"  Primary layer: {PRIMARY_LAYER}\")\n    logger.info(f\"  R² (instruct): {r2_values['instruct']:.4f}\")\n    logger.info(f\"  Features analyzed: {len(filtered_global)}\")\n    logger.info(f\"  Type-I (R > {empirical_threshold:.3f}): {n_type_i}\")\n    logger.info(f\"  Type-R (R < 0): {n_type_r}\")\n    logger.info(f\"  Hartigan p: {dip_p:.4f}\")\n    logger.info(f\"  Fisher p: {fisher_p}\")\n    logger.info(f\"  Criterion 1 (bimodal): {criterion1_confirmed}\")\n    logger.info(f\"  Criterion 2 (centroid): {criterion2_confirmed}\")\n    logger.info(f\"  Criterion 3 (semantic): {criterion3_confirmed}\")\n    logger.info(f\"  Our method accuracy: {our_acc:.3f}\")\n    logger.info(f\"  Baseline accuracy: {baseline_acc:.3f}\")\n    logger.info(\"=\" * 60)\n\n\nif __name__ == \"__main__\":\n    main()\n", "numLines": 997, "startLine": 1, "totalLines": 997}}
```

### [7] ASSISTANT · 2026-07-17 12:24:07 UTC

```
Now I have a thorough understanding of both the paper and the actual experimental code. Let me write the structured review.
```

### [8] TOOL CALL — Write · 2026-07-17 12:24:07 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/review_paper/review_paper/.terminal_claude_agent_struct_out.json

{
  "overall_assessment": "This iteration represents a major step forward from the prior draft — all previous major issues (no experimental results, no density filter, uncorrected references, missing Dolly citation, borrowing layer selection from a different architecture) have been addressed. The paper now reports genuine empirical results with a well-structured methodology. The central finding (zero Type-R features, mean Reversal Score 0.921, all RLHF-sensitive features preserved by abliteration) is interesting and novel. However, three serious new issues emerge from cross-checking the paper against the code: (1) the density filter is rendered trivially satisfied by the pre-topk fallback, yet the paper presents it as a meaningful quality check; (2) the near-chance classification accuracy (0.476) produced by the method is reported in the artifact summary but is absent from the paper body — an omission that undermines trust; (3) the described semantic labeling protocol (±32 token window around highest-activation token) does not match the code (first 300 chars of the 5 highest-mean-activation prompts). Additionally, the fundamental tension between using pre-topk (dense) encoder projections as a stand-in for SAE features — when SAE feature interpretability was validated for post-topk sparse activations — is underplayed. The paper is borderline: the finding is interesting and the methodology is mostly sound, but these discrepancies need to be resolved for a confident acceptance.",
  "strengths": [
    "The Reversal Score metric is a clean, well-motivated per-feature measure that directly quantifies whether abliteration reverses or preserves individual RLHF-induced activation changes. It is a genuine methodological contribution.",
    "All six major critiques from the previous review have been adequately addressed: experimental results are now reported, the density filter is implemented, the layer selection is empirical (not borrowed from a different architecture), the Type-I threshold is set at Q3 rather than hardcoded, the segment extraction window is specified, and the reference errors are corrected.",
    "The paper's limitations section is unusually honest for a first-iteration manuscript — explicitly flagging the community checkpoint, R² degradation, labeler calibration, and the underpowered Q3/Q1 fallback test.",
    "The pre-registration structure (declaring all three confirmatory criteria before running the full analysis) gives the paper credibility even when the primary criterion (bimodality) is not confirmed.",
    "The finding that ALL 500 top RLHF-sensitive features are preserved (zero Type-R features at every tested threshold from 0.30 to 0.95) is a striking and clean result that has a natural geometric interpretation.",
    "The centroid analysis is well-implemented and provides a complementary line of evidence beyond the Reversal Score distribution."
  ],
  "dimension_scores": [
    {
      "dimension": "soundness",
      "score": 2,
      "justification": "The core finding is internally consistent and supported by real experimental results, but two soundness concerns undermine confidence: (1) the pre-topk encoder projection is dense — every feature activates on every prompt — making the 10% density filter trivially satisfied and rendering the filter report misleading; (2) no null baseline is provided showing what R values look like for features NOT modified by RLHF, so R≈0.921 cannot be distinguished from 'abliteration barely changes model activations globally' rather than the claimed 'RLHF-modified features survive intact'.",
      "improvements": [
        "Compute and report Reversal Scores for the bottom-500 features by Safety Change as a null baseline. If those also show R≈0.9, the finding needs to be reinterpreted as 'abliteration makes small global changes' rather than 'RLHF-modified features specifically survive'.",
        "Acknowledge explicitly that with pre-topk (dense) activations, the density filter cannot function as designed — every feature has a non-zero value on every prompt, so 100% survival is mathematically guaranteed rather than an empirical quality check.",
        "Report the near-chance classification accuracy (0.476) in the paper body alongside the baseline (0.630) with a brief explanation of why this is consistent with the main finding rather than burying it in the artifact."
      ]
    },
    {
      "dimension": "presentation",
      "score": 3,
      "justification": "The paper is clearly written, well-organized, and the pre-registration framing makes the confirmatory design easy to follow. The limitations section is a model of transparency. The main clarity gap is the discrepancy between the described and actual labeling protocol, and the omission of the near-chance classification accuracy from the main text.",
      "improvements": [
        "Correct the semantic labeling description to match the code: the paper describes extracting '±32 token window around the token with highest absolute pre-topk activation' but the code takes the first 300 characters of the 5 prompts with highest mean feature activation — a substantially different and simpler protocol.",
        "Add a paragraph in the results or discussion explaining the near-chance classification accuracy, even if just one or two sentences explaining it is expected given the unimodal R distribution.",
        "Table 1 is effective but could also note what the Reversal Score range looks like for each threshold bin, so readers can see the distribution shape without relying solely on Figure 2."
      ]
    },
    {
      "dimension": "contribution",
      "score": 3,
      "justification": "The Reversal Score framing is a genuine contribution to mechanistic interpretability of fine-tuning and ablation. The finding that abliteration does not reverse ANY RLHF-modified SAE feature has clear implications for safety restoration framing. The main limitation is that the contribution is currently scoped to one layer of one model family with an unvalidated community checkpoint, and the pre-topk fallback means the 'feature-level' claim rests on dense encoder projections whose interpretability is not validated to the same degree as post-topk sparse activations.",
      "improvements": [
        "Even a brief analysis at layer 20 (the second-highest SC-variance layer) would significantly strengthen the generalizability of the zero Type-R finding. The paper claims it exists but reports no numbers.",
        "Direct computation of the cosine similarity between the refusal direction vector and the top SAE encoder directions would convert the inferred 'geometric orthogonality' into a direct measurement, which is a much stronger claim and easy to compute given the available artifacts."
      ]
    }
  ],
  "critiques": [
    {
      "category": "methodology",
      "severity": "major",
      "description": "The density filter is trivially satisfied by pre-topk (dense) activations and the paper misrepresents this as a meaningful quality check. The paper states 'all 500 top RLHF-sensitive features survived the 10% activation-density filter (fraction surviving = 1.00)' — but inspection of the code (method.py line 520) reveals the filter checks `abs(delta) > 1e-6`. Pre-topk activations (z_pre = h @ W_enc.T + b_enc) are dense continuous projections: every feature has a non-zero value on every prompt by construction, so 100% survival is mathematically guaranteed regardless of feature quality. The filter was designed for post-topk sparse activations where a feature could genuinely fire on fewer than 50 prompts. Reporting trivially-guaranteed 100% survival as evidence of analysis coverage is misleading.",
      "suggested_action": "Add a sentence explicitly acknowledging that with pre-topk (dense) activations, the density filter cannot distinguish sparse from dense features — all 65,536 features have non-zero values on all 500 prompts. Instead, characterize the effective density by reporting, for the top-500 RLHF-sensitive features, the fraction of prompts on which the post-topk activation would be non-zero (i.e., the feature would be in the active top-50). This gives an honest picture of how 'SAE-feature-like' these 500 features are in the post-topk sense."
    },
    {
      "category": "evidence",
      "severity": "major",
      "description": "The near-chance classification accuracy (0.476, essentially 50/50 on a balanced dataset) is reported in the artifact summary but entirely absent from the paper body. The artifact states: 'Our method accuracy=0.476 (TP=238/500, TN=238/500) using median harm score threshold — near-chance.' The keyword baseline achieves 0.630. The paper discusses the Reversal Score analysis at length but never mentions that the derived harm score (Type-I activation minus Type-R activation) cannot distinguish harmful from benign prompts. A reviewer who reads the artifact will notice this omission and interpret it as cherry-picking positive results.",
      "suggested_action": "Add a paragraph in Section 5 (or Section 6 Discussion) reporting the classification accuracy: 'As a downstream application test, we computed a per-prompt harm score as mean Type-I feature activation minus mean Type-R activation and predicted harmful vs. benign using a median threshold. The method achieved 0.476 accuracy, near-chance on the balanced corpus. This is expected given the unimodal R distribution: since all features have high Reversal Scores, the Type-I/Type-R distinction provides no discrimination signal.' The explanation is consistent with the main finding and should be presented openly."
    },
    {
      "category": "methodology",
      "severity": "major",
      "description": "The semantic labeling protocol described in the paper does not match the code implementation. The paper states: 'we extract the maximum-activating prompt segment: the ±32 token window around the token with highest absolute pre-topk activation within each prompt.' However, the code (method.py lines 666-673) does not perform any token-level activation analysis. Instead, it identifies the 5 prompts with the highest scalar feature activation (averaged across all tokens in the prompt via the last-token hidden state), then passes the first 300 characters of each prompt to the labeler. No ±32 token window is computed. No per-token activation analysis is performed. The described protocol would require token-level hook extraction, which is absent from the codebase.",
      "suggested_action": "Correct the paper description to match the actual code: 'We identify the top-5 prompts (by maximum pre-topk encoder activation for feature f on the last input token) and provide the first 300 characters of each to the cross-family labeler.' Additionally, note the limitation of using the full-prompt first-300-characters rather than a feature-specific activation window: this may cause the labeler to assign labels based on prompt topic rather than the specific token that most activates the feature."
    },
    {
      "category": "rigor",
      "severity": "major",
      "description": "The high Reversal Scores (R≈0.921) lack a null baseline, making it impossible to determine whether this reflects 'RLHF-modified features specifically survive abliteration' versus the simpler explanation 'instruct and abliterated models have very similar residual-stream activations overall, so almost all encoder projections are correlated.' Since abliteration applies a targeted orthogonalization to only a few directions in a 4096-dimensional space, most of the residual stream is unaffected. The expected Reversal Score for randomly sampled features not modified by RLHF could also be near 1.0, in which case R≈0.921 for RLHF-sensitive features is uninformative.",
      "suggested_action": "Compute Reversal Scores for the BOTTOM-500 features by Safety Change (i.e., features least modified by RLHF) and report their distribution alongside the top-500. If both distributions are similarly concentrated near R=1, then the high R values reflect global similarity between instruct and abliterated models rather than selective preservation of RLHF-modified features. If the RLHF-sensitive features show significantly higher R than the control features, that would be strong positive evidence for the selective preservation claim."
    },
    {
      "category": "methodology",
      "severity": "major",
      "description": "The paper's claim of 'SAE feature-level analysis' rests on pre-topk (dense) encoder projections, but the interpretability of SAE features — the foundation for semantic claims about 'harm-concept' labels — was validated for post-topk sparse activations. The paper uses pre-topk because R²≈0.51 (all variants), triggering the fallback path. Pre-topk values (z_pre = h @ W_enc.T + b_enc) are a dense 65,536-dimensional projection of the residual stream. The Qwen-Scope documentation endorses applying the base SAE to post-training variants, but this endorsement pertains to post-topk sparse feature interpretations, not to the raw pre-topk encoder output as a 65K-dimensional feature vector. The semantic labeling and the 'feature-level' framing throughout the paper implicitly assume pre-topk encoder directions are as interpretable as post-topk sparse features, which has not been validated.",
      "suggested_action": "Add a paragraph in the limitations section explicitly distinguishing 'SAE-derived features' (post-topk, validated interpretable) from 'SAE encoder projections' (pre-topk, dense, interpretability not directly validated). Acknowledge that the semantic claims (≈75% harm-concept rate) rest on encoder projections rather than the sparse post-topk features that the mechanistic interpretability literature has validated. Alternatively, request an instruct SAE for layers above 8 (the paper notes one exists for layers 0-8 only) which would enable post-topk analysis."
    },
    {
      "category": "evidence",
      "severity": "minor",
      "description": "The paper claims 'results at layer 20 are reported as a sensitivity check and show qualitatively consistent patterns' but no layer-20 numbers appear anywhere in the Results or Appendix. This is an unsubstantiated claim: the code computes SECOND_LAYER=20 and logs it, but the full experiment is only run at PRIMARY_LAYER=22. Asserting consistency without reporting the actual numbers is not a sensitivity check.",
      "suggested_action": "Either report the actual layer 20 results (Reversal Score distribution mean/std, Type-R count) as a numbered table or remove the claim about layer 20 consistency. Given that the artifact reports 4.3 minutes total runtime, running the full analysis at layer 20 as well appears computationally feasible."
    },
    {
      "category": "rigor",
      "severity": "minor",
      "description": "The benign-prompt centroid imprint (17.6× closer to instruct than to base in the RLHF-sensitive subspace) is larger than the harmful-prompt imprint (6.1×). The paper notes this as 'more comprehensive than predicted' but does not consider the simpler explanation: instruct and abliterated models are globally similar (abliteration changes a few weight directions), so their activations in ANY subspace — not just the RLHF-sensitive one — are close to each other relative to base. The benign-prompt ratio being LARGER than the harmful-prompt ratio is actually consistent with the null hypothesis that abliteration makes global small changes rather than selective preservation of harm-related features.",
      "suggested_action": "Report centroid distances in a control subspace — e.g., the top-500 features by Safety Change on BENIGN prompts (features that distinguish benign from base, if any exist) — to show that the closeness to instruct is not uniform across all feature subspaces. If the RLHF-sensitive subspace shows systematically higher instruct-closeness than a random or benign-sensitive subspace, that would support the selective preservation claim."
    },
    {
      "category": "rigor",
      "severity": "minor",
      "description": "The Q3/Q1 fallback Fisher test (n=4 in the high group, n=90 in the low group) has essentially zero statistical power and the harm-concept rates (75% vs 71.1%, p=0.675) are not meaningfully different. The paper correctly acknowledges this, but the test still appears in the main results (Table 1 footnote equivalent). For a pre-registered paper, reporting an underpowered test as a 'fallback' result can be misinterpreted as a weak positive signal when it is actually an uninterpretable measurement.",
      "suggested_action": "Remove the Fisher fallback test from the quantitative results summary and report only the qualitative conclusion: 'Because all features have positive Reversal Scores, the pre-registered Type-I vs Type-R comparison cannot be computed. The unconditional harm-concept rate of ≈75% across all top-500 RLHF-sensitive features is the primary semantic finding.' The fallback test should be moved to limitations or dropped entirely."
    },
    {
      "category": "rigor",
      "severity": "minor",
      "description": "The geometric orthogonality claim — 'the RLHF-induced changes to all 500 top features are essentially orthogonal to the refusal direction r in the residual stream of Qwen3-8B at layer 22' — is inferred from the zero Type-R result rather than directly measured. The refusal direction r is available (as the mean difference of harmful vs. harmless residual-stream activations that the abliteration method operates on). The direct measurement would be to compute the cosine similarity between r and each of the top-500 RLHF-modified SAE encoder directions (W_enc[feature_idx, :] projected back into residual stream space via W_dec), which would give a direct measure of the claimed orthogonality.",
      "suggested_action": "Compute the refusal direction r using the standard Arditi et al. procedure (mean difference of instruct activations on harmful vs. harmless prompts at layer 22) and report the cosine similarity between r and the top-10 RLHF-sensitive SAE encoder directions. If these are indeed near-orthogonal (|cos| < 0.1), this would directly confirm the geometric claim rather than inferring it from the R distribution."
    }
  ],
  "score": 5,
  "confidence": 4
}
```

### [9] TOOL RESULT — Write · 2026-07-17 12:24:08 UTC

```
{"type": "create", "filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/review_paper/review_paper/.terminal_claude_agent_struct_out.json", "content": "{\n  \"overall_assessment\": \"This iteration represents a major step forward from the prior draft — all previous major issues (no experimental results, no density filter, uncorrected references, missing Dolly citation, borrowing layer selection from a different architecture) have been addressed. The paper now reports genuine empirical results with a well-structured methodology. The central finding (zero Type-R features, mean Reversal Score 0.921, all RLHF-sensitive features preserved by abliteration) is interesting and novel. However, three serious new issues emerge from cross-checking the paper against the code: (1) the density filter is rendered trivially satisfied by the pre-topk fallback, yet the paper presents it as a meaningful quality check; (2) the near-chance classification accuracy (0.476) produced by the method is reported in the artifact summary but is absent from the paper body — an omission that undermines trust; (3) the described semantic labeling protocol (±32 token window around highest-activation token) does not match the code (first 300 chars of the 5 highest-mean-activation prompts). Additionally, the fundamental tension between using pre-topk (dense) encoder projections as a stand-in for SAE features — when SAE feature interpretability was validated for post-topk sparse activations — is underplayed. The paper is borderline: the finding is interesting and the methodology is mostly sound, but these discrepancies need to be resolved for a confident acceptance.\",\n  \"strengths\": [\n    \"The Reversal Score metric is a clean, well-motivated per-feature measure that directly quantifies whether abliteration reverses or preserves individual RLHF-induced activation changes. It is a genuine methodological contribution.\",\n    \"All six major critiques from the previous review have been adequately addressed: experimental results are now reported, the density filter is implemented, the layer selection is empirical (not borrowed from a different architecture), the Type-I threshold is set at Q3 rather than hardcoded, the segment extraction window is specified, and the reference errors are corrected.\",\n    \"The paper's limitations section is unusually honest for a first-iteration manuscript — explicitly flagging the community checkpoint, R² degradation, labeler calibration, and the underpowered Q3/Q1 fallback test.\",\n    \"The pre-registration structure (declaring all three confirmatory criteria before running the full analysis) gives the paper credibility even when the primary criterion (bimodality) is not confirmed.\",\n    \"The finding that ALL 500 top RLHF-sensitive features are preserved (zero Type-R features at every tested threshold from 0.30 to 0.95) is a striking and clean result that has a natural geometric interpretation.\",\n    \"The centroid analysis is well-implemented and provides a complementary line of evidence beyond the Reversal Score distribution.\"\n  ],\n  \"dimension_scores\": [\n    {\n      \"dimension\": \"soundness\",\n      \"score\": 2,\n      \"justification\": \"The core finding is internally consistent and supported by real experimental results, but two soundness concerns undermine confidence: (1) the pre-topk encoder projection is dense — every feature activates on every prompt — making the 10% density filter trivially satisfied and rendering the filter report misleading; (2) no null baseline is provided showing what R values look like for features NOT modified by RLHF, so R≈0.921 cannot be distinguished from 'abliteration barely changes model activations globally' rather than the claimed 'RLHF-modified features survive intact'.\",\n      \"improvements\": [\n        \"Compute and report Reversal Scores for the bottom-500 features by Safety Change as a null baseline. If those also show R≈0.9, the finding needs to be reinterpreted as 'abliteration makes small global changes' rather than 'RLHF-modified features specifically survive'.\",\n        \"Acknowledge explicitly that with pre-topk (dense) activations, the density filter cannot function as designed — every feature has a non-zero value on every prompt, so 100% survival is mathematically guaranteed rather than an empirical quality check.\",\n        \"Report the near-chance classification accuracy (0.476) in the paper body alongside the baseline (0.630) with a brief explanation of why this is consistent with the main finding rather than burying it in the artifact.\"\n      ]\n    },\n    {\n      \"dimension\": \"presentation\",\n      \"score\": 3,\n      \"justification\": \"The paper is clearly written, well-organized, and the pre-registration framing makes the confirmatory design easy to follow. The limitations section is a model of transparency. The main clarity gap is the discrepancy between the described and actual labeling protocol, and the omission of the near-chance classification accuracy from the main text.\",\n      \"improvements\": [\n        \"Correct the semantic labeling description to match the code: the paper describes extracting '±32 token window around the token with highest absolute pre-topk activation' but the code takes the first 300 characters of the 5 prompts with highest mean feature activation — a substantially different and simpler protocol.\",\n        \"Add a paragraph in the results or discussion explaining the near-chance classification accuracy, even if just one or two sentences explaining it is expected given the unimodal R distribution.\",\n        \"Table 1 is effective but could also note what the Reversal Score range looks like for each threshold bin, so readers can see the distribution shape without relying solely on Figure 2.\"\n      ]\n    },\n    {\n      \"dimension\": \"contribution\",\n      \"score\": 3,\n      \"justification\": \"The Reversal Score framing is a genuine contribution to mechanistic interpretability of fine-tuning and ablation. The finding that abliteration does not reverse ANY RLHF-modified SAE feature has clear implications for safety restoration framing. The main limitation is that the contribution is currently scoped to one layer of one model family with an unvalidated community checkpoint, and the pre-topk fallback means the 'feature-level' claim rests on dense encoder projections whose interpretability is not validated to the same degree as post-topk sparse activations.\",\n      \"improvements\": [\n        \"Even a brief analysis at layer 20 (the second-highest SC-variance layer) would significantly strengthen the generalizability of the zero Type-R finding. The paper claims it exists but reports no numbers.\",\n        \"Direct computation of the cosine similarity between the refusal direction vector and the top SAE encoder directions would convert the inferred 'geometric orthogonality' into a direct measurement, which is a much stronger claim and easy to compute given the available artifacts.\"\n      ]\n    }\n  ],\n  \"critiques\": [\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"major\",\n      \"description\": \"The density filter is trivially satisfied by pre-topk (dense) activations and the paper misrepresents this as a meaningful quality check. The paper states 'all 500 top RLHF-sensitive features survived the 10% activation-density filter (fraction surviving = 1.00)' — but inspection of the code (method.py line 520) reveals the filter checks `abs(delta) > 1e-6`. Pre-topk activations (z_pre = h @ W_enc.T + b_enc) are dense continuous projections: every feature has a non-zero value on every prompt by construction, so 100% survival is mathematically guaranteed regardless of feature quality. The filter was designed for post-topk sparse activations where a feature could genuinely fire on fewer than 50 prompts. Reporting trivially-guaranteed 100% survival as evidence of analysis coverage is misleading.\",\n      \"suggested_action\": \"Add a sentence explicitly acknowledging that with pre-topk (dense) activations, the density filter cannot distinguish sparse from dense features — all 65,536 features have non-zero values on all 500 prompts. Instead, characterize the effective density by reporting, for the top-500 RLHF-sensitive features, the fraction of prompts on which the post-topk activation would be non-zero (i.e., the feature would be in the active top-50). This gives an honest picture of how 'SAE-feature-like' these 500 features are in the post-topk sense.\"\n    },\n    {\n      \"category\": \"evidence\",\n      \"severity\": \"major\",\n      \"description\": \"The near-chance classification accuracy (0.476, essentially 50/50 on a balanced dataset) is reported in the artifact summary but entirely absent from the paper body. The artifact states: 'Our method accuracy=0.476 (TP=238/500, TN=238/500) using median harm score threshold — near-chance.' The keyword baseline achieves 0.630. The paper discusses the Reversal Score analysis at length but never mentions that the derived harm score (Type-I activation minus Type-R activation) cannot distinguish harmful from benign prompts. A reviewer who reads the artifact will notice this omission and interpret it as cherry-picking positive results.\",\n      \"suggested_action\": \"Add a paragraph in Section 5 (or Section 6 Discussion) reporting the classification accuracy: 'As a downstream application test, we computed a per-prompt harm score as mean Type-I feature activation minus mean Type-R activation and predicted harmful vs. benign using a median threshold. The method achieved 0.476 accuracy, near-chance on the balanced corpus. This is expected given the unimodal R distribution: since all features have high Reversal Scores, the Type-I/Type-R distinction provides no discrimination signal.' The explanation is consistent with the main finding and should be presented openly.\"\n    },\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"major\",\n      \"description\": \"The semantic labeling protocol described in the paper does not match the code implementation. The paper states: 'we extract the maximum-activating prompt segment: the ±32 token window around the token with highest absolute pre-topk activation within each prompt.' However, the code (method.py lines 666-673) does not perform any token-level activation analysis. Instead, it identifies the 5 prompts with the highest scalar feature activation (averaged across all tokens in the prompt via the last-token hidden state), then passes the first 300 characters of each prompt to the labeler. No ±32 token window is computed. No per-token activation analysis is performed. The described protocol would require token-level hook extraction, which is absent from the codebase.\",\n      \"suggested_action\": \"Correct the paper description to match the actual code: 'We identify the top-5 prompts (by maximum pre-topk encoder activation for feature f on the last input token) and provide the first 300 characters of each to the cross-family labeler.' Additionally, note the limitation of using the full-prompt first-300-characters rather than a feature-specific activation window: this may cause the labeler to assign labels based on prompt topic rather than the specific token that most activates the feature.\"\n    },\n    {\n      \"category\": \"rigor\",\n      \"severity\": \"major\",\n      \"description\": \"The high Reversal Scores (R≈0.921) lack a null baseline, making it impossible to determine whether this reflects 'RLHF-modified features specifically survive abliteration' versus the simpler explanation 'instruct and abliterated models have very similar residual-stream activations overall, so almost all encoder projections are correlated.' Since abliteration applies a targeted orthogonalization to only a few directions in a 4096-dimensional space, most of the residual stream is unaffected. The expected Reversal Score for randomly sampled features not modified by RLHF could also be near 1.0, in which case R≈0.921 for RLHF-sensitive features is uninformative.\",\n      \"suggested_action\": \"Compute Reversal Scores for the BOTTOM-500 features by Safety Change (i.e., features least modified by RLHF) and report their distribution alongside the top-500. If both distributions are similarly concentrated near R=1, then the high R values reflect global similarity between instruct and abliterated models rather than selective preservation of RLHF-modified features. If the RLHF-sensitive features show significantly higher R than the control features, that would be strong positive evidence for the selective preservation claim.\"\n    },\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"major\",\n      \"description\": \"The paper's claim of 'SAE feature-level analysis' rests on pre-topk (dense) encoder projections, but the interpretability of SAE features — the foundation for semantic claims about 'harm-concept' labels — was validated for post-topk sparse activations. The paper uses pre-topk because R²≈0.51 (all variants), triggering the fallback path. Pre-topk values (z_pre = h @ W_enc.T + b_enc) are a dense 65,536-dimensional projection of the residual stream. The Qwen-Scope documentation endorses applying the base SAE to post-training variants, but this endorsement pertains to post-topk sparse feature interpretations, not to the raw pre-topk encoder output as a 65K-dimensional feature vector. The semantic labeling and the 'feature-level' framing throughout the paper implicitly assume pre-topk encoder directions are as interpretable as post-topk sparse features, which has not been validated.\",\n      \"suggested_action\": \"Add a paragraph in the limitations section explicitly distinguishing 'SAE-derived features' (post-topk, validated interpretable) from 'SAE encoder projections' (pre-topk, dense, interpretability not directly validated). Acknowledge that the semantic claims (≈75% harm-concept rate) rest on encoder projections rather than the sparse post-topk features that the mechanistic interpretability literature has validated. Alternatively, request an instruct SAE for layers above 8 (the paper notes one exists for layers 0-8 only) which would enable post-topk analysis.\"\n    },\n    {\n      \"category\": \"evidence\",\n      \"severity\": \"minor\",\n      \"description\": \"The paper claims 'results at layer 20 are reported as a sensitivity check and show qualitatively consistent patterns' but no layer-20 numbers appear anywhere in the Results or Appendix. This is an unsubstantiated claim: the code computes SECOND_LAYER=20 and logs it, but the full experiment is only run at PRIMARY_LAYER=22. Asserting consistency without reporting the actual numbers is not a sensitivity check.\",\n      \"suggested_action\": \"Either report the actual layer 20 results (Reversal Score distribution mean/std, Type-R count) as a numbered table or remove the claim about layer 20 consistency. Given that the artifact reports 4.3 minutes total runtime, running the full analysis at layer 20 as well appears computationally feasible.\"\n    },\n    {\n      \"category\": \"rigor\",\n      \"severity\": \"minor\",\n      \"description\": \"The benign-prompt centroid imprint (17.6× closer to instruct than to base in the RLHF-sensitive subspace) is larger than the harmful-prompt imprint (6.1×). The paper notes this as 'more comprehensive than predicted' but does not consider the simpler explanation: instruct and abliterated models are globally similar (abliteration changes a few weight directions), so their activations in ANY subspace — not just the RLHF-sensitive one — are close to each other relative to base. The benign-prompt ratio being LARGER than the harmful-prompt ratio is actually consistent with the null hypothesis that abliteration makes global small changes rather than selective preservation of harm-related features.\",\n      \"suggested_action\": \"Report centroid distances in a control subspace — e.g., the top-500 features by Safety Change on BENIGN prompts (features that distinguish benign from base, if any exist) — to show that the closeness to instruct is not uniform across all feature subspaces. If the RLHF-sensitive subspace shows systematically higher instruct-closeness than a random or benign-sensitive subspace, that would support the selective preservation claim.\"\n    },\n    {\n      \"category\": \"rigor\",\n      \"severity\": \"minor\",\n      \"description\": \"The Q3/Q1 fallback Fisher test (n=4 in the high group, n=90 in the low group) has essentially zero statistical power and the harm-concept rates (75% vs 71.1%, p=0.675) are not meaningfully different. The paper correctly acknowledges this, but the test still appears in the main results (Table 1 footnote equivalent). For a pre-registered paper, reporting an underpowered test as a 'fallback' result can be misinterpreted as a weak positive signal when it is actually an uninterpretable measurement.\",\n      \"suggested_action\": \"Remove the Fisher fallback test from the quantitative results summary and report only the qualitative conclusion: 'Because all features have positive Reversal Scores, the pre-registered Type-I vs Type-R comparison cannot be computed. The unconditional harm-concept rate of ≈75% across all top-500 RLHF-sensitive features is the primary semantic finding.' The fallback test should be moved to limitations or dropped entirely.\"\n    },\n    {\n      \"category\": \"rigor\",\n      \"severity\": \"minor\",\n      \"description\": \"The geometric orthogonality claim — 'the RLHF-induced changes to all 500 top features are essentially orthogonal to the refusal direction r in the residual stream of Qwen3-8B at layer 22' — is inferred from the zero Type-R result rather than directly measured. The refusal direction r is available (as the mean difference of harmful vs. harmless residual-stream activations that the abliteration method operates on). The direct measurement would be to compute the cosine similarity between r and each of the top-500 RLHF-modified SAE encoder directions (W_enc[feature_idx, :] projected back into residual stream space via W_dec), which would give a direct measure of the claimed orthogonality.\",\n      \"suggested_action\": \"Compute the refusal direction r using the standard Arditi et al. procedure (mean difference of instruct activations on harmful vs. harmless prompts at layer 22) and report the cosine similarity between r and the top-10 RLHF-sensitive SAE encoder directions. If these are indeed near-orthogonal (|cos| < 0.1), this would directly confirm the geometric claim rather than inferring it from the R distribution.\"\n    }\n  ],\n  \"score\": 5,\n  \"confidence\": 4\n}\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [10] ASSISTANT · 2026-07-17 12:24:18 UTC

```
Review complete. I scored the paper **5/10 (Borderline Accept)** with confidence 4.

**Summary of what I found:**

The paper is substantially improved from iteration 1 — all six previous major critiques were addressed (experimental results now exist, density filter implemented, empirical layer selection, corrected references, Dolly citation fixed, Q3 threshold instead of hardcoded). The core finding (zero Type-R features, R̄=0.921) is genuinely interesting.

**New major issues identified by cross-checking code against paper:**

1. **Density filter is meaningless with pre-topk activations** — since z_pre is dense, every feature fires on every prompt, making "100% survival" mathematically guaranteed rather than empirically informative.

2. **Near-chance accuracy (0.476) is hidden in artifact, absent from paper** — the method cannot distinguish harmful from benign prompts; this negative result needs to appear in the main text.

3. **Paper-code discrepancy in labeling protocol** — paper claims "±32 token window around highest-activation token," but code uses `prompt[:300]` of the 5 highest-mean-activation prompts. No token-level analysis is performed at all.

4. **Missing null baseline for Reversal Scores** — R≈0.921 could simply reflect global similarity between instruct and abliterated models (abliteration is a small perturbation); the bottom-500 features by Safety Change should also be computed as a control.

5. **Pre-topk ≠ SAE features** — semantic interpretability claims rest on pre-topk dense encoder projections, not the post-topk sparse activations that were actually validated as monosemantic in the Qwen-Scope literature.
```
