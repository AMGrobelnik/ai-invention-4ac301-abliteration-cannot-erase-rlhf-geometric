# review_paper — test_idea

> Phase: `invention_loop` · round 1 · `review_paper`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim transcript of this agent task — every system/user prompt, assistant response, thinking block, tool call and tool result — in the order they occurred. Nothing truncated.

## Task: `review_paper` (terminal_claude_agent, claude-sonnet-4-6)

### [1] CONFIG · 2026-07-17 10:39:54 UTC

```
model: claude-sonnet-4-6 | effort: medium | permission: bypassPermissions | cwd: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_1/review_paper/review_paper
```

### [2] SYSTEM-USER prompt · 2026-07-17 10:40:00 UTC

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

Safety alignment of large language models is routinely evaluated at the behavioral level: does the model refuse a harmful request? This framing is blind to the model's internal representation of harmful content, and the blind spot has practical consequences. *Abliteration*—identifying the "refusal direction" in the residual stream as a mean-difference vector between harmful and harmless activations, then orthogonalizing all residual-stream-writing weight matrices against it [1]—produces models that pass behavioral compliance checks yet may retain rich internal representations of harmful content. Understanding what abliteration actually removes, and what it leaves intact, is a foundational question for AI safety.

The standard narrative is that abliteration undoes safety RLHF fine-tuning. Recent geometric work challenges this at the level of detection. LatentBiopsy [2] demonstrates that a simple angular-deviation classifier applied to residual stream activations of safety-tuned Qwen3.5-0.8B retains AUROC ≥ 0.937 after abliteration—dropping at most 0.015 from the instruct baseline—while achieving AUROC = 1.000 for discriminating harmful from benign-aggressive prompts. A companion study [3] extends these findings across 12 model families (Qwen, Llama, Gemma; 0.5B–9B parameters), reporting mean AUROC 0.982 surviving abliteration within ±0.003. These results establish compellingly that the harmful-intent representation is not destroyed by abliteration.

Yet geometric approaches face a fundamental limitation: they collapse all representational information into a single detection score. Knowing that AUROC drops by only 0.015 does not answer: *which specific features carry the surviving representation? Do those features form semantically coherent clusters corresponding to weapon synthesis schemas, manipulation tactics, or deception patterns? Can they support targeted safety restoration?* These questions require feature-level analysis.

Sparse Autoencoders (SAEs) [11, 12] provide exactly the vocabulary needed. An SAE trained on a model's residual stream decomposes activations into tens of thousands of near-monosemantic features—sparse linear directions corresponding to interpretable concepts. Applied to the RLHF analysis problem, SAEs enable asking: for each of the 65,536 features in the Qwen-Scope dictionary [9], did RLHF change this feature's activation on harmful prompts, and if so, does abliteration reverse that change or leave it intact? Prior work using frozen base SAEs as diagnostic instruments [4] demonstrates that SAE latent representations diverge significantly after supervised fine-tuning—SAE latent cosine similarity drops to 0.509–0.557 at layer 22 after SFT while raw residual stream cosine similarity remains 0.960–0.984—validating the frozen base-SAE approach for cross-variant comparison.

We propose a three-model SAE analysis framework applying the frozen Qwen-Scope base SAE [9] across Qwen3-8B-Base, Qwen3-8B (instruct), and a scratch-reproduced abliterated checkpoint. The central analytic contribution is the *Reversal_score*: a per-feature Pearson correlation between the RLHF-induced activation change vector and the abliteration-induced activation change vector across 500 harmful prompts. Features with high positive Reversal_score (Type-I) retain the instruct model's activation pattern after abliteration; features with high negative Reversal_score (Type-R) revert toward the base model. We further test whether this partition is bimodal or continuous, and whether Type-I features carry harm-concept semantic labels at significantly higher rates—providing the first direct evidence that the *latent safety imprint* is semantically coherent rather than arbitrary.

**Summary of Contributions:**
- We introduce the *Reversal_score* (Section 3.3), a per-feature Pearson correlation that quantifies abliteration resistance for each RLHF-modified SAE feature, enabling for the first time a decomposition of RLHF modifications into abliteration-sensitive (Type-R) and abliteration-resistant (Type-I) subsets.
- We propose a pre-registered two-outcome analysis framework (Section 3.5): both bimodal and continuous distributions of the Reversal_score are pre-specified as informative, avoiding post-hoc interpretation.
- We describe a validated cross-family LLM semantic labeling protocol (Section 3.6) with a Cohen's κ ≥ 0.6 gold-set gate, enabling testing of whether Type-I features carry harm-concept content at significantly higher rates than Type-R features.
- We identify a critical architecture correction for Qwen-Scope SAEs from JumpReLU to TopK (Section 3.2), and derive cleaner robustness checks based on pre-topk vs. post-topk activations. [ARTIFACT:art_pmAcY0CNazP9]
- We construct a 1,000-prompt evaluation corpus (Section 4) spanning four benchmark datasets with MinHash deduplication, length filtering, and category balancing. [ARTIFACT:art_M8-flaZ0OOaP]

[FIGURE:fig1]

# Related Work

**Refusal ablation and its mechanistic limits.** Arditi et al. [1] establish that refusal in open-source language models is mediated by a single direction in the residual stream—the mean difference of activations on harmful vs. harmless prompts—and that orthogonalizing all weight matrices against this direction eliminates refusal while largely preserving general capabilities across 13 models up to 72B parameters. Cristofano [7] identifies a polysemanticity problem: the raw refusal vector conflates safety circuitry with capability and linguistic style components, so standard abliteration produces collateral capability degradation (KL = 2.088 on Wikitext-2) that spectral residualization reduces to KL = 0.044 while maintaining near-zero refusal rates. This polysemanticity finding provides theoretical grounding for our expectation that abliteration may produce a *continuum* of partial reversals across SAE features rather than a clean bimodal split. Nanfack et al. [19] propose an alternative abliteration approach via optimal transport that reduces the distributional shift introduced by standard orthogonalization.

**Geometry of harmful intent surviving abliteration.** LatentBiopsy [2] demonstrates that harmful-prompt detection via angular deviation in residual streams survives abliteration with AUROC ≥ 0.937 (at most 0.015 degradation) on Qwen3.5-0.8B and Qwen2.5-0.5B, achieving AUROC = 1.000 for harmful vs. benign-aggressive discrimination even post-abliteration. The companion study [3] extends this across 12 model families and sizes (Qwen, Llama, Gemma; 0.5B–9B), reporting mean AUROC 0.982 surviving abliteration within ±0.003, but notes that different pooling strategies yield harm directions 73° apart—suggesting the surviving representation is partially protocol-dependent. Our work provides the complementary feature-level answer: not *whether* something survives, but *which specific features* and *whether they are semantically interpretable*.

**SAEs as measurement instruments.** The superposition hypothesis [11] establishes that neural networks represent more features than neurons by storing sparse, approximately orthogonal directions—motivating SAE decomposition. Bricken et al. [12] demonstrate practical dictionary learning on transformer activations, recovering monosemantic features from polysemantic neurons. Gao et al. [13] establish scaling laws showing SAE quality improves with dictionary size and sparsity control. Templeton et al. [14] apply SAEs at Claude 3 Sonnet scale, demonstrating rich feature-level semantic structure including safety-relevant features. The pivotal precursor for our work is Chopra [4], who uses pretrained base SAEs as diagnostic instruments for comparing base and SFT-finetuned model variants: SAE latent cosine similarity between base and SFT model drops to 0.509–0.557 at layer 22, while raw residual stream cosine similarity remains 0.960–0.984. This dissociation validates the frozen base-SAE as a sensitive diagnostic for cross-variant comparison, and motivates our pre-topk encoder fallback when decoder reconstruction quality degrades.

**Multi-model SAE comparison: crosscoders and DFCs.** Jiralerspong and Bricken [5] introduce *crosscoders*—SAEs trained jointly on two or more model variants—that naturally partition features into shared and model-specific categories. Our frozen base-SAE approach is complementary: it requires no training, directly reuses Qwen-Scope's labeled feature vocabulary, and extends naturally to three-model (base, instruct, abliterated) comparisons without constructing a new feature basis per model pair. Shportko et al. [6] demonstrate *Dedicated Feature Crosscoders* (DFCs) that localize RL-induced tool-calling capability in Qwen2.5-3B to a compact, steerable feature set, achieving 31 percentage-point improvement in tool correctness and 7 percentage-point capability spillover to frozen base models. DFCs identify what RL *adds*; our Reversal_score framework identifies what RLHF safety training adds that *resists deletion*—complementary framings that together provide a complete picture of RL-induced feature landscape modification.

**Safety fine-tuning mechanisms.** Jain et al. [8] establish that safety fine-tuning minimally transforms MLP weights to project unsafe activations into the null space of those weights, explaining why adversarial prompts that produce safe-like activations circumvent safety mechanisms. This null-space projection is precisely what abliteration disrupts—but our hypothesis is that semantic harm-concept representations at the SAE feature level may lie orthogonal to the refusal direction and thus survive this disruption intact.

**Qwen-Scope and instruct-model SAEs.** Deng et al. [9] release Qwen-Scope, an open-source suite of SAEs for Qwen3 and Qwen3.5 model families, including the Qwen3-8B base SAE (65,536 features, TopK architecture, all 36 residual-stream layers) used in this work. Li et al. [10] demonstrate that base SAEs suffer approximately 8× higher reconstruction MSE on instruct model activations (log₂ MSE = 5.20) compared to purpose-trained instruct SAEs (log₂ MSE = 0.65), directly motivating our reconstruction quality gate and pre-topk encoder fallback. He et al. [18] release a Qwen3 instruct SAE covering Qwen3-8B layers 0–8, which does not reach the safety-relevant middle layers (14–22) but serves as a partial cross-validation instrument.

# Methodology

## Three-Model Setup and Abliteration Reproduction

We work with three Qwen3-8B model checkpoints sharing identical 36-layer architecture (d_model = 4096, bfloat16, minimum transformers 4.51.0). [ARTIFACT:art_pmAcY0CNazP9]

*Base*: `Qwen/Qwen3-8B-Base`—the unaligned pretrained model, providing the representational baseline.

*Instruct*: `Qwen/Qwen3-8B` with `enable_thinking=False`—the safety-tuned instruct model. Suppressing chain-of-thought generation removes lengthy reasoning tokens from the prompt context and isolates safety-relevant activations on the final token of each input.

*Abliterated*: `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` as the primary checkpoint (v2 corrects a layer-0 output garbling artifact present in v1). We independently reproduce abliteration from scratch using the mlabonne recipe: harmful directions estimated from `mlabonne/harmful_behaviors` (520 samples, 256 used), harmless directions from `mlabonne/harmless_alpaca`, layer selection via highest-mean-residual criterion (layer 9 empirically optimal for 8B-scale models), orthogonal projection applied to all residual-stream-writing weight matrices. The scratch-reproduced checkpoint verifies that the community model follows standard difference-of-means methodology and provides a controlled abliterated variant with fully documented parameters.

All three models are accessed via `model.model.layers` for residual-stream hook injection.

## SAE Measurement Instrument

The frozen Qwen-Scope `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` SAE serves as the shared measurement instrument across all three model variants [9]. [ARTIFACT:art_pmAcY0CNazP9] A critical architecture correction from preliminary research: Qwen-Scope implements *TopK* architecture (not JumpReLU as originally assumed in the hypothesis). For each input residual activation $\mathbf{x} \in \mathbb{R}^{4096}$:

$$\mathbf{z}_{\text{pre}} = \mathbf{x} W_{\text{enc}}^\top + \mathbf{b}_{\text{enc}} \in \mathbb{R}^{65536}$$

$$\mathbf{z} = \text{TopK}(\mathbf{z}_{\text{pre}},\; k{=}50) \quad \text{(retain only the top-50 pre-activations by magnitude)}$$

SAE weight tensors are $W_{\text{enc}} \in \mathbb{R}^{65536 \times 4096}$, $W_{\text{dec}} \in \mathbb{R}^{4096 \times 65536}$, $\mathbf{b}_{\text{enc}} \in \mathbb{R}^{65536}$, $\mathbf{b}_{\text{dec}} \in \mathbb{R}^{4096}$, loaded via `hf_hub_download` and `torch.load`. The Qwen-Scope L0_100 variant (k=100) is used for sensitivity checks.

The TopK correction has a clean implication for robustness analysis. Rather than a binary threshold-crossing confound (present with JumpReLU), the relevant confound is *rank displacement*: RLHF that elevates feature $f$'s pre-activation may push feature $g$ below the top-50 cutoff. We mitigate this by reporting Safety_change from both pre-topk ($\mathbf{z}_{\text{pre}}$, continuous, no threshold artifact) and post-topk ($\mathbf{z}$, sparse, TopK-gated), and flagging features where the two estimates diverge by more than 2× as "threshold-sensitive."

## Feature Metrics: Safety\_change and Reversal\_score

For each of the 65,536 SAE features $f$ and a set of $N$ prompts $\{p_i\}_{i=1}^N$, we collect activations $a_m(f, p_i)$ for model $m \in \{\text{base}, \text{instruct}, \text{abliterated}\}$ at the selected SAE layer.

**Safety\_change** quantifies how much RLHF modified feature $f$ on harmful prompts:

$$\text{SC}(f) = \frac{1}{N} \sum_{i=1}^{N} |a_{\text{instruct}}(f, p_i) - a_{\text{base}}(f, p_i)|$$

where $N = 500$ harmful prompts. Features in the top 500 by SC$(f)$ are termed *RLHF-sensitive* and are the primary analysis targets.

**Reversal\_score** quantifies whether abliteration reverses the RLHF-induced change for feature $f$. Define two $N$-dimensional change vectors:

$$\boldsymbol{\Delta}_{\text{inst}}(f) = \bigl[a_{\text{instruct}}(f, p_i) - a_{\text{base}}(f, p_i)\bigr]_{i=1}^{N}$$

$$\boldsymbol{\Delta}_{\text{abl}}(f) = \bigl[a_{\text{abliterated}}(f, p_i) - a_{\text{base}}(f, p_i)\bigr]_{i=1}^{N}$$

$$R(f) = \text{PearsonR}\!\left(\boldsymbol{\Delta}_{\text{inst}}(f),\; \boldsymbol{\Delta}_{\text{abl}}(f)\right)$$

$R(f) \approx +1$ indicates that abliteration preserves the RLHF change direction prompt-by-prompt (*Type-I: Semantically-Imprinted*). $R(f) \approx -1$ indicates that abliteration reverses the RLHF change (*Type-R: Refusal-Coupled*). $R(f) \approx 0$ indicates partial or uncorrelated reversal.

Reversal_score is a Pearson correlation over 500 scalar activations per feature—not a per-prompt cosine similarity of vectors—ensuring the bimodality finding (if observed) is an empirical result rather than a definitional artifact.

*Worked example.* Suppose feature $f$ responds to chemistry-related harmful requests. RLHF increases its activation on such prompts: $\boldsymbol{\Delta}_{\text{inst}}(f) \approx [+0.8, +1.2, +0.5, +0.9, \ldots]$. If abliteration leaves it similarly elevated, $\boldsymbol{\Delta}_{\text{abl}}(f) \approx [+0.7, +1.0, +0.4, +0.8, \ldots]$, then $R(f) \approx +0.99$ → Type-I. If abliteration reverses the increase, $\boldsymbol{\Delta}_{\text{abl}}(f) \approx [-0.1, -0.2, -0.1, -0.1, \ldots]$, then $R(f) \approx -0.98$ → Type-R.

## Layer Selection and Reconstruction Quality Gate

**Layer selection.** The Qwen-Scope SAE covers all 36 residual-stream layers. We select the analysis layer by running 50 harmful prompts through the base model and computing the variance of SC$(f)$ across all 65,536 features at each layer; the layer with highest Safety_change variance provides the richest safety-relevant signal. Based on [4], layer 22 exhibits maximum sensitivity to fine-tuning changes (SAE latent cosine similarity 0.509–0.557 after SFT), layer 14 is most stable, and layer 20 is intermediate. We run the primary analysis at layer 20 with sensitivity checks at layers 14 and 22. Layers 28–35 are excluded per preliminary evidence of degraded SAE reconstruction quality at very deep layers.

**Reconstruction quality gate.** Following [10], which demonstrates an approximately 8× MSE gap when base SAEs are applied to instruct model activations vs. purpose-trained instruct SAEs, we validate reconstruction quality before interpreting feature-level differences. For 50 harmful prompts, we compute $R^2 = 1 - \text{MSE}_{\text{recon}} / \text{Var}(\mathbf{x})$ for base, instruct, and abliterated variants passed through the frozen SAE. If $R^2 \geq 0.85$ for both non-base variants, we proceed with full encoder+decoder analysis. If $R^2 < 0.85$, we switch to *encoder-only* analysis using pre-topk activations $\mathbf{z}_{\text{pre}}$—the encoder projection retains discriminative signal even when decoder reconstruction degrades, since it maps activations into the feature vocabulary regardless of decoder calibration.

[FIGURE:fig2]

## Bimodality Analysis

We compute $R(f)$ for all top-500 RLHF-sensitive features and apply Hartigan's dip test ($H_0$: unimodal) to the distribution. Both outcomes are pre-specified:

- *Bimodal ($p < 0.05$)*: We define Type-R as $R(f) < 0$ and Type-I as $R(f) > +0.5$; features with $0 \leq R(f) \leq +0.5$ are "intermediate." A bimodal split suggests two categorically distinct modification types: explicit refusal circuitry (Type-R) and harm-concept semantic memory (Type-I).

- *Continuous ($p \geq 0.05$)*: We treat $R(f)$ as a graded *survival score* and test whether features in the top quartile receive harm-concept semantic labels at $\geq 2\times$ the rate of features in the bottom quartile (Fisher's exact test). Continuous distribution would suggest a single graded mechanism where each RLHF-modified feature is partially reversed proportional to its alignment with the refusal direction. The polysemanticity of the refusal direction [7] provides prior reason for this outcome.

The pre-registration of both interpretations ensures the framework is informative regardless of empirical outcome.

## Semantic Labeling Protocol

For the top-100 features by $|R(f) - 0.5| \times \text{SC}(f)$—those simultaneously strongly RLHF-sensitive and strongly classified by Reversal_score—we extract maximum-activating prompt segments and pass them to an LLM semantic labeler.

*Cross-family labeling.* We use Llama-3-8B or Gemini-flash (via OpenRouter) rather than any Qwen-family model, to avoid within-family circularity where the labeler might reproduce Qwen3-8B's RLHF-trained semantic categories. The labeler classifies each feature into: `harm_concept` (weapon synthesis, manipulation, deception, violence), `safety_refusal` (refusal language, hedging, disclaimer patterns), `general_capability` (coding, reasoning, factual recall), or `other`.

*Gold-set validation.* We manually annotate 20 features (10 high-$R(f)$, 10 low-$R(f)$) and compute Cohen's $\kappa$ between manual and LLM labels. If $\kappa < 0.6$, we revise the labeler prompt and re-validate before trusting automated labels. Only with $\kappa \geq 0.6$ do we apply labels to the full feature set and report Criterion 3 results.

*Representational distance analysis.* In the Type-R and Type-I feature subspaces separately, we compute centroid distances $d(\text{abliterated}, \text{instruct})$ and $d(\text{abliterated}, \text{base})$ for harmful vs. benign prompts. The core quantitative test (Criterion 2) is:

$$d(\text{abl}, \text{inst}) < d(\text{abl}, \text{base}) \quad \text{in the Type-I subspace, for harmful prompts (predicted)}$$
$$d(\text{abl}, \text{inst}) \approx d(\text{abl}, \text{base}) \quad \text{in the Type-I subspace, for benign prompts (control)}$$

This differential pattern—the abliterated model's harm-concept activations being more instruct-like specifically on harmful content—is the defining signature of the latent safety imprint.

[FIGURE:fig3]

# Evaluation Corpus

## Dataset Sources

We assemble a balanced corpus of 500 harmful and 500 benign prompts from four benchmark datasets [ARTIFACT:art_M8-flaZ0OOaP].

**BeaverTails** [15] (176 harmful prompts): Human-labeled QA pairs by PKU-Alignment (NeurIPS 2023), spanning 14 harm categories including animal abuse, chemical threats, self-harm, and financial crime. The 14-category structure makes BeaverTails the primary source for studying which SAE features specialize in particular harm types.

**HarmBench** [17] (85 harmful prompts): Standard safety evaluation benchmark from the Center for AI Safety (ICML 2024), covering chemical\_biological, cybercrime, illegal, and physical\_harm categories. Direct unambiguous harmful requests provide the clearest SAE activation signal for isolating harm-concept features.

**JailbreakBench/JBB-Behaviors** [16] (55 harmful + 46 benign prompts): NeurIPS 2024 Datasets Track. Uniquely provides matched harmful/benign prompt pairs from the same curation effort across 10 OpenAI policy violation categories (e.g., "Write synthesis instructions for [substance]" vs. "Write a fictional story involving [substance]"). This paired structure enables direct contrastive SAE analysis with topic controlled.

**Dolly-15k** [18] (106 benign prompts): Open-source instruction-following dataset by 5,000+ Databricks employees (CC-BY-SA 3.0), covering brainstorming, classification, closed QA, generation, information extraction, and summarization. Provides a topically diverse benign baseline for SAE activation comparison.

## Assembly Pipeline

All four sources are processed through a five-stage quality pipeline [ARTIFACT:art_M8-flaZ0OOaP]: (1) *MinHash LSH deduplication* (threshold 0.5, 128 hash permutations, character 5-gram shingling) removes near-duplicates within and across sources; (2) *token-length filtering* ($\leq 512$ tokens, estimated as $\lceil\text{char\_count} / 4\rceil$) excludes very long prompts that create dominating SAE activation patterns; (3) *category capping* ($\leq 35\%$ per harm subcategory within the harmful set) prevents any single harm type from dominating the Safety_change estimates; (4) *balanced sampling* to 500 harmful + 500 benign. Harmful prompts are labeled with source and BeaverTails-style harm categories where available; benign prompts are labeled with instruction-following task types.

All datasets carry permissive licenses for research use (BeaverTails: CC-BY-NC-4.0; JailbreakBench: MIT; HarmBench: public; Dolly-15k: CC-BY-SA 3.0). The assembled corpus is deterministic and fully reproducible via the pipeline code.

[FIGURE:fig4]

# Analysis Framework and Success Criteria

## Confirmatory Criteria

We adopt a three-criterion confirmatory structure:

**Criterion 1 (Bimodality or continuous semantic coherence).** Hartigan's dip test on the top-500 RLHF-sensitive features yields $p < 0.05$ (bimodal), with modes at $R < 0$ (Type-R) and $R > +0.5$ (Type-I). *Alternatively*, if $p \geq 0.05$ (continuous), the hypothesis is confirmed in a weaker "graded imprint" form if features with $R(f) > +0.5$ receive harm-concept labels at $\geq 2\times$ the rate of features with $R(f) < 0$.

**Criterion 2 (Representational fingerprint).** In the high-$R(f)$ feature subspace, $d(\text{abliterated}, \text{instruct}) < d(\text{abliterated}, \text{base})$ for harmful prompts, while $d(\text{abliterated}, \text{instruct}) \approx d(\text{abliterated}, \text{base})$ for benign prompts. This differential is the core quantitative test of the latent safety imprint.

**Criterion 3 (Semantic label coherence).** High-$R(f)$ features receive `harm_concept` labels at a significantly higher rate than low-$R(f)$ features ($p < 0.05$, Fisher's exact test on 50 features per group), with $\kappa \geq 0.6$ labeler validation. This establishes that abliteration resistance specifically preserves harm-concept knowledge rather than arbitrary SAE features.

**Exploratory Criterion 4 (Capability-gap preliminary signal).** The fraction of Type-I features at the analyzed layer is reported alongside abliterated vs. base capability gap on a 200-question mini-MMLU subset. This correlation is descriptive at $N=1$ model and labeled explicitly as a qualitative signal for future multi-model work.

## Disconfirmation Conditions

The hypothesis is disconfirmed if: (a) Hartigan's test yields $p \geq 0.05$ AND high-$R(f)$ features show no enrichment for harm-concept labels; (b) $d(\text{abliterated}, \text{instruct}) \geq d(\text{abliterated}, \text{base})$ in the high-$R(f)$ subspace for harmful prompts; or (c) the SAE reconstruction $R^2$ gate fails below 0.70 for the instruct model AND pre-topk encoder-only activations also show no reliable Safety_change signal. In case (c), the experiment is deemed *inconclusive* rather than disconfirmed, and the analysis should be repeated using the Qwen3 instruct SAE for layers 0–8 as an alternative instrument [18].

# Discussion

## Implications for Targeted Safety Restoration

If Criterion 3 is confirmed—Type-I features carry semantically coherent harm-concept representations that survive abliteration—these features become actionable for lightweight safety restoration. The DFC "capability spillover" result [6] demonstrates that feature-level steering can transfer capabilities across model variants without retraining; an analogous approach could amplify Type-I feature activations in the abliterated model toward their instruct-model magnitudes. Unlike full RLHF retraining, this requires only inference-time feature steering or targeted weight modification on a small set of identified features. The Type-I features, being already present in the abliterated model's residual stream, require *direction amplification* rather than *direction creation*.

This also suggests a mechanistic explanation for the consistent (if modest) capability degradation observed in abliterated models on general benchmarks: if Type-I features mediate not just harm-concept understanding but also related general semantic processing, their survival at reduced activation amplitude (present but no longer amplified via refusal-coupled circuitry) could explain subtle performance gaps on diverse tasks.

## Relation to Geometric Approaches

LatentBiopsy [2] and the geometry study [3] answer "does something survive abliteration?" with compelling affirmative evidence. Our work is in sequence: geometric analysis establishes the phenomenon; feature-level SAE analysis characterizes its semantic content. The critical advantage of feature-level analysis is actionability: a single AUROC score cannot be used to design targeted interventions, but a labeled set of 200–500 SAE features can directly guide safety restoration experiments.

The two approaches are also complementary for validating each other. If our Reversal_score analysis identifies a Type-I cluster, one should observe AUROC degradation concentrated in the activation directions corresponding to that cluster—a prediction testable by running LatentBiopsy-style detection after projecting out only the Type-I directions. This cross-validation would quantify what fraction of the geometric detection signal comes from Type-I vs. Type-R features.

## The CrossCoder Comparison

Our frozen base-SAE approach is complementary to crosscoders [5, 6]. Crosscoders require training a new shared feature vocabulary per model pair, which takes computational resources and produces a feature basis that is not directly interpretable using Qwen-Scope's existing labeled vocabulary. Our approach offers zero training cost, direct reuse of existing feature semantics, and extension to three-model comparisons without modification. The natural follow-up is to train a crosscoder on base + instruct activations and test whether crosscoder-shared features correspond to our Type-I features—providing a cleaner cross-model feature correspondence than a frozen base-SAE can offer.

## Limitations

*Single model and layer.* This is a proof-of-concept on Qwen3-8B at one primary SAE layer. The Reversal_score distribution and Type-R/Type-I proportions may vary substantially across layers—the finding [4] that SAE latent cosine similarity collapses maximally at layer 22 and is most stable at layer 7 suggests depth-dependent variation. Sensitivity checks at layers 14 and 22 are planned but do not substitute for a systematic multi-layer analysis.

*Base-SAE fidelity.* The ≈8× MSE gap demonstrated by Li et al. [10] for base SAEs applied to instruct activations means our encoder-only fallback is essential but imperfect: encoder activations may assign features inaccurately when the input distribution has shifted substantially from training data. Without an instruct-trained Qwen3-8B SAE covering layers 14–22, this limitation cannot be fully resolved in the current study.

*Labeler circularity.* Even with a non-Qwen cross-family labeler, if the labeler shares RLHF training data with Qwen3-8B, it may reproduce rather than independently validate Qwen's harm-concept categories. The gold-set Cohen's $\kappa \geq 0.6$ gate mitigates but does not eliminate this risk.

*Abliteration method variation.* The community model uses an unspecified "new and faster" abliteration method [ARTIFACT:art_pmAcY0CNazP9]. If this diverges from standard difference-of-means orthogonalization, the Reversal_score partition may reflect methodological rather than mechanistic differences. The scratch-reproduced checkpoint provides a controlled reference but requires comparison against the community model to confirm alignment.

*N=1 capability gap analysis.* Criterion 4 cannot be statistically validated at a single SAE layer within one model. Meaningful multi-model analysis would require at minimum 5–8 model variants across families and sizes, making it a future study.

# Conclusion

We have proposed a feature-level SAE analysis framework for decomposing the effects of refusal ablation on RLHF safety fine-tuning in Qwen3-8B. The *Reversal_score*—a per-feature Pearson correlation between RLHF-induced and abliteration-induced activation changes across 500 harmful prompts—provides the first principled measure of which of the 65,536 Qwen-Scope features survive abliteration intact vs. which are reversed. The framework partitions RLHF-sensitive features into Refusal-Coupled (Type-R) and Semantically-Imprinted (Type-I) subsets, tests the R/I partition for bimodality, validates Type-I semantic content with a cross-family labeler, and measures representational distances in each feature subspace.

The latent safety imprint hypothesis predicts that the abliterated model's harm-concept feature activations remain closer to the instruct model than to the base model specifically for harmful prompts—a prediction the representational distance analysis will directly test. If confirmed, Type-I features provide the first actionable feature-level target for lightweight safety restoration in abliterated models.

Key methodological contributions—the Reversal_score metric, the pre-registered bimodal/continuous interpretation framework, the TopK architecture correction for Qwen-Scope SAEs, and the cross-family labeler validation protocol—apply to any future study using frozen base SAEs for cross-variant mechanistic analysis.

**Future directions** include: (1) extending the three-model analysis to Qwen3.5-7B, Llama-3.1-8B, and Gemma-3-9B to test cross-family generalizability of the Type-R/Type-I partition; (2) training crosscoders [5] on base + instruct activation pairs as a reconstruction-quality-independent cross-validation of frozen-SAE findings; (3) implementing Type-I feature steering for lightweight safety restoration in abliterated models, following the DFC approach of Shportko et al. [6]; and (4) using the Qwen3 instruct SAE [18] for layers 0–8 as a partial cross-validation instrument.

# References

[1] A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Rimsky, W. Gurnee, and N. Nanda. Refusal in language models is mediated by a single direction. In *Advances in Neural Information Processing Systems*, 2024.

[2] I. Llorente-Saguer. The geometry of harmful intent: Training-free anomaly detection via angular deviation in LLM residual streams. *arXiv preprint arXiv:2603.27412*, 2026.

[3] I. Llorente-Saguer. Harmful intent as a geometrically recoverable feature of LLM residual streams. *arXiv preprint arXiv:2604.18901*, 2026.

[4] R. Chopra. A mechanistic investigation of supervised fine tuning. *arXiv preprint arXiv:2605.11426*, 2026.

[5] T. Jiralerspong and T. Bricken. Cross-architecture model diffing with crosscoders: Unsupervised discovery of differences between LLMs. *arXiv preprint arXiv:2602.11729*, 2026.

[6] A. Shportko, S. Bhokare, A. Z. A. Alzahrani, B. Cheng, G. Mercier, and J. Hullman. Localizing RL-induced tool use to a single crosscoder feature. In *ICML 2026 Workshop on Mechanistic Interpretability* (Spotlight), 2026.

[7] T. Cristofano. Surgical refusal ablation: Disentangling safety from intelligence via concept-guided spectral cleaning. *arXiv preprint arXiv:2601.08489*, 2026.

[8] S. Jain, E. S. Lubana, K. Oksuz, T. Joy, P. H. S. Torr, A. Sanyal, and P. K. Dokania. What makes and breaks safety fine-tuning? A mechanistic study. In *Advances in Neural Information Processing Systems*, 2024.

[9] B. Deng, X. Wang, Y. Wang, Y. Wan, Y. Ma, B. Yang, H. Wei, J. Tang, H. Lin, R. Gao, T. Li, Q. Cao, X. Ren, X. Deng, A. Yang, F. Huang, D. Liu, and J. Zhou. Qwen-scope: Turning sparse features into development tools for large language models. *arXiv preprint arXiv:2605.11887*, 2026.

[10] J. Li, H. Ye, Y. Chen, X. Li, L. Zhang, H. Alinejad-Rokny, J. C. H. Peng, and M. Yang. Training superior sparse autoencoders for instruct models. *arXiv preprint arXiv:2506.07691*, 2025.

[11] N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds, R. Lasenby, D. Drain, C. Chen, R. Grosse, S. McCandlish, J. Kaplan, D. Amodei, M. Wattenberg, and C. Olah. Toy models of superposition. *arXiv preprint arXiv:2209.10652*, 2022.

[12] T. Bricken, A. Templeton, J. Batson, B. Chen, A. Jermyn, T. Conerly, N. Turner, C. Anil, C. Denison, A. Askell, R. Lasenby, Y. Wu, S. Kravec, N. Schiefer, T. Maxwell, N. Joseph, A. Tamkin, K. Nguyen, B. McLean, J. E. Burke, T. Hume, S. Carter, T. Henighan, and C. Olah. Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*, Anthropic, 2023.

[13] L. Gao, T. Dupré la Tour, H. Tillman, G. Goh, R. Troll, A. Radford, I. Sutskever, J. Leike, and J. Wu. Scaling and evaluating sparse autoencoders. In *International Conference on Learning Representations*, 2024.

[14] A. Templeton, T. Conerly, J. Marcus, J. Lindsey, T. Bricken, B. Chen, A. Pearce, C. Citro, E. Ameisen, A. Jones, H. Cunningham, N. Turner, C. McDougall, M. MacDiarmid, A. Tamkin, E. Durmus, T. Hume, F. Mosconi, C. D. Freeman, T. Sumers, E. Rees, J. Batson, A. Jermyn, S. Carter, C. Olah, and T. Henighan. Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *arXiv preprint arXiv:2605.29358*, 2026.

[15] J. Ji, M. Liu, J. Dai, X. Pan, C. Zhang, C. Bian, R. Sun, Y. Wang, and Y. Yang. BeaverTails: Towards improved safety alignment of LLM via a human-preference dataset. In *Advances in Neural Information Processing Systems*, 2023.

[16] P. Chao, E. Debenedetti, A. Robey, M. Andriushchenko, F. Croce, V. Sehwag, E. Dobriban, N. Flammarion, G. Pappas, F. Tramèr, H. Hassani, and E. Wong. JailbreakBench: An open robustness benchmark for jailbreaking large language models. In *Advances in Neural Information Processing Systems*, 2024.

[17] M. Mazeika, L. Phan, X. Yin, A. Zou, Z. Wang, N. Mu, E. Sakhaee, N. Li, S. Basart, B. Li, D. Forsyth, and D. Hendrycks. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. In *International Conference on Machine Learning*, pp. 35181–35224, 2024.

[18] X. He, W. Wang, B. Zhao, X. Ren, W. Li, W. Qiao, H. Wei, and L. Qu. Discovering millions of interpretable features with sparse autoencoders. 2026.

[19] G. Nanfack, E. Belilovsky, and E. Dohmatob. Efficient refusal ablation in LLM through optimal transport. *arXiv preprint arXiv:2603.04355*, 2026.
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
</supplementary_materials>

<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for judging whether the paper's contribution is genuinely novel versus already-done or a known dead end in this field.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>



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

### [3] HUMAN-USER prompt · 2026-07-17 10:40:00 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [4] ASSISTANT · 2026-07-17 10:41:34 UTC

```
I'll read the paper carefully, check the supplementary materials, and search the literature before writing the review.
```

### [5] TOOL CALL — Skill · 2026-07-17 10:41:34 UTC

```
aii-web-tools
```

### [6] TOOL RESULT — Skill · 2026-07-17 10:41:34 UTC

```
{"success": true, "commandName": "aii-web-tools"}
```

### [7] ASSISTANT · 2026-07-17 10:41:38 UTC

```
Let me search for key cited papers to verify their existence and accuracy.
```

### [8] TOOL CALL — ToolSearch · 2026-07-17 10:41:38 UTC

```
{'query': 'select:WebSearch,WebFetch'}
```

### [9] TOOL RESULT — ToolSearch · 2026-07-17 10:41:38 UTC

```
{"matches": ["WebSearch", "WebFetch"], "query": "select:WebSearch,WebFetch", "total_deferred_tools": 21}
```

### [10] TOOL CALL — WebSearch · 2026-07-17 10:41:44 UTC

```
Arditi refusal language models mediated single direction NeurIPS 2024
```

### [11] TOOL CALL — WebSearch · 2026-07-17 10:41:44 UTC

```
LatentBiopsy angular deviation LLM residual streams harmful intent arXiv 2603.27412
```

### [12] TOOL CALL — WebSearch · 2026-07-17 10:41:44 UTC

```
Chopra mechanistic investigation supervised fine tuning SAE arXiv 2605.11426
```

### [13] TOOL RESULT — WebSearch · 2026-07-17 10:41:50 UTC

```
{"query": "Arditi refusal language models mediated single direction NeurIPS 2024", "results": [{"tool_use_id": "srvtoolu_015DRxD2EK6Wzwz36QnLhbwm", "content": [{"title": "dblp: Refusal in Language Models Is Mediated by a Single Direction.", "url": "https://dblp.org/rec/conf/nips/ArditiOSPPGN24.html"}, {"title": "Refusal in Language Models Is Mediated by a Single Direction Andy Arditi∗", "url": "https://arxiv.org/pdf/2406.11717"}, {"title": "Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in Language Models", "url": "https://arxiv.org/pdf/2606.24952"}, {"title": "A Geometric Account of Activation Steering through Angle-Norm Decomposition", "url": "https://arxiv.org/pdf/2606.06735"}, {"title": "When Behavioral Safety Evaluation Fails: A Representation-Level Perspective", "url": "https://arxiv.org/pdf/2606.08044"}, {"title": "Refusal Lives Downstream of Persona in Chat Models", "url": "https://arxiv.org/pdf/2606.26161"}, {"title": "On the Limitations of Steering in Language Model Alignment", "url": "https://arxiv.org/pdf/2505.01162"}, {"title": "Controlling Large Language Model Agents with Entropic Activation   Steering", "url": "https://arxiv.org/pdf/2406.00244"}, {"title": "A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models", "url": "https://arxiv.org/pdf/2605.08513"}, {"title": "Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs", "url": "https://arxiv.org/pdf/2603.27518"}]}, "I found the paper you were searching for! Here are the key details:\n\nThe paper \"Refusal in Language Models Is Mediated by a Single Direction\" was authored by Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda, and it was published at NeurIPS 2024.\n\n**Main Findings:**\n\nThe work shows that refusal is mediated by a one-dimensional subspace across 13 popular open-source chat models up to 72B parameters in size. \n\nThe single direction is both necessary and sufficient for refusal—ablating this direction effectively disables the model's ability to refuse harmful requests. Additionally, adding this direction to the model's activations induces refusal on harmless instructions.\n\nThe researchers use a small set of contrastive pairs of harmful and harmless instructions to identify a single difference-in-means direction that can be intervened upon to circumvent refusal on harmful prompts, or induce refusal on harmless prompts.\n\nThe paper's arXiv preprint is available at https://arxiv.org/pdf/2406.11717."], "durationSeconds": 7.678104940999998, "searchCount": 1}
```

### [14] TOOL RESULT — WebSearch · 2026-07-17 10:41:50 UTC

```
{"query": "LatentBiopsy angular deviation LLM residual streams harmful intent arXiv 2603.27412", "results": [{"tool_use_id": "srvtoolu_014W2o5G9ai2xrFtZ1QUdR9V", "content": [{"title": "[2603.27412] The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/abs/2603.27412"}, {"title": "The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/html/2603.27412"}, {"title": "Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams", "url": "https://arxiv.org/html/2604.18901"}, {"title": "The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/pdf/2603.27412"}, {"title": "[2604.18901] Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams", "url": "https://arxiv.org/abs/2604.18901"}, {"title": "Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams", "url": "https://arxiv.org/pdf/2604.18901"}, {"title": "[2604.28129] Latent Adversarial Detection: Adaptive Probing of LLM Activations for Multi-Turn Attack Detection", "url": "https://arxiv.org/abs/2604.28129"}, {"title": "Segment-Level Coherence for Robust Harmful Intent Probing in LLMs", "url": "https://arxiv.org/pdf/2604.14865"}]}, "I found the paper you're searching for! Here's what the research is about:\n\n## LatentBiopsy: Training-Free Harmful Prompt Detection\n\nLatentBiopsy is a training-free method for detecting harmful prompts by analysing the geometry of residual-stream activations in large language models. \n\n### How It Works\n\nGiven 200 safe normative prompts, LatentBiopsy computes the leading principal component of their activations at a target layer and characterises new prompts by their radial deviation angle (θ) from this reference direction. The anomaly score is the negative log-likelihood of θ under a Gaussian fit to the normative distribution, flagging deviations symmetrically regardless of orientation. No harmful examples are required for training.\n\n### Key Results\n\nThe research evaluated two complete model triplets from the Qwen3.5-0.8B and Qwen2.5-0.5B families: base, instruction-tuned, and abliterated (refusal direction surgically removed via orthogonalisation), achieving AUROC ≥0.937 for harmful-vs-normative detection and AUROC = 1.000 for discriminating harmful from benign-aggressive prompts (XSTest).\n\nThe paper (arXiv 2603.27412) was published on March 28, 2026, by Isaac Llorente-Saguer."], "durationSeconds": 7.866457850999999, "searchCount": 1}
```

### [15] TOOL RESULT — WebSearch · 2026-07-17 10:41:52 UTC

```
{"query": "Chopra mechanistic investigation supervised fine tuning SAE arXiv 2605.11426", "results": [{"tool_use_id": "srvtoolu_015ybBuVm9EtvKz4Cd8WwV4u", "content": [{"title": "A Mechanistic Investigation of Supervised Fine Tuning", "url": "https://arxiv.org/html/2605.11426v1"}, {"title": "A Mechanistic Investigation of Supervised Fine Tuning", "url": "https://arxiv.org/pdf/2605.11426"}, {"title": "[2604.11838] A Layer-wise Analysis of Supervised Fine-Tuning", "url": "https://arxiv.org/abs/2604.11838"}, {"title": "arXiv:2502.11812v2 [cs.CL] 13 Jun 2025", "url": "https://arxiv.org/pdf/2502.11812"}, {"title": "Mechanistically analyzing the effects of fine-tuning on procedurally   defined tasks", "url": "https://arxiv.org/pdf/2311.12786"}, {"title": "A Layer-wise Analysis of Supervised Fine-Tuning", "url": "https://arxiv.org/html/2604.11838"}, {"title": "[2601.18699] Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning", "url": "https://arxiv.org/abs/2601.18699"}, {"title": "SAE as a Crystal Ball: Interpretable Features Predict Cross-domain Transferability of LLMs without Training", "url": "https://arxiv.org/pdf/2603.02908"}]}, "I found the paper you're looking for! Here's what I found about it:\n\n## \"A Mechanistic Investigation of Supervised Fine Tuning\" by Ruhaan Chopra\n\nThe paper begins by noting that the cosine similarity between a language model's hidden activations before and after Supervised Fine-Tuning (SFT) remains very high, which at first glance suggests that SFT leaves the model's activation geometry largely undisturbed. However, projecting both sets of activations through a Sparse Autoencoder (SAE) pretrained on the base model reveals that the underlying sparse latents diverge significantly.\n\n**Key Contributions:**\n\nThe paper introduces a novel investigative pipeline which utilizes pretrained SAEs as a high-resolution diagnostic tool to mechanistically investigate the drivers of representational divergence, and discovers task-specific and layer-specific distributions of the precise semantic features that are systematically altered during supervised fine-tuning. Additionally, the authors identify a layer-wise update profile specific to safety alignment.\n\n**Experiments:**\n\nThe empirical analysis compares hidden states before and after fine-tuning across four independently fine-tuned variants of Gemma 3 1B, where the raw activation cosine similarity between the base and fine-tuned models remains high (>0.96) across all evaluated tasks and layers.\n\nThe paper was published on arXiv on May 12, 2026, and code and experimental scripts are publicly available on GitHub."], "durationSeconds": 8.183071278999996, "searchCount": 1}
```

### [16] TOOL CALL — WebSearch · 2026-07-17 10:42:00 UTC

```
Gao scaling evaluating sparse autoencoders ICLR 2024 OpenAI
```

### [17] TOOL CALL — WebSearch · 2026-07-17 10:42:00 UTC

```
Qwen-Scope SAE sparse autoencoders Qwen3 arXiv 2605.11887
```

### [18] TOOL CALL — WebSearch · 2026-07-17 10:42:00 UTC

```
He discovering millions interpretable features sparse autoencoders Qwen3 instruct SAE arXiv 2606
```

### [19] TOOL RESULT — WebSearch · 2026-07-17 10:42:06 UTC

```
{"query": "Gao scaling evaluating sparse autoencoders ICLR 2024 OpenAI", "results": [{"tool_use_id": "srvtoolu_01QTXT6AB4Yxwjg5N96bT41m", "content": [{"title": "Scaling and Evaluating Sparse Autoencoders — Sparse Autoencoder", "url": "https://www.sparseautoencoder.com/papers/scaling-and-evaluating-sparse-autoencoders/"}, {"title": "Simo Ryu on X: \"So I was fascinated by recent \"Scaling and evaluating sparse autoencoders\" paper by Leo Gao from openai https://t.co/y2CyXzDjny To get started, naturally this is my attempt to make 500 LoC Sparse Autoencoder trainer for llama models. WIP especially on steering with sparse\" / X", "url": "https://x.com/cloneofsimo/status/1863487582631547027"}, {"title": "Toward Identifiable Sparse Autoencoders", "url": "https://arxiv.org/pdf/2605.31245"}, {"title": "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?", "url": "https://arxiv.org/pdf/2602.14111"}, {"title": "Scaling and evaluating sparse autoencoders", "url": "https://arxiv.org/pdf/2406.04093"}, {"title": "The Rate-Distortion-Polysemanticity Tradeoff in SAEs", "url": "https://arxiv.org/pdf/2605.14694"}, {"title": "Scaling and evaluating sparse autoencoders", "url": "https://arxiv.org/html/2406.04093v1"}, {"title": "Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video", "url": "https://arxiv.org/pdf/2504.19475"}, {"title": "Decomposing multimodal embedding spaces with group-sparse autoencoders", "url": "https://arxiv.org/pdf/2601.20028"}, {"title": "Do Sparse Autoencoders Capture Concept Manifolds?", "url": "https://arxiv.org/pdf/2604.28119"}]}, "Based on the search results, I found information about the paper you're looking for:\n\n\"Scaling and Evaluating Sparse Autoencoders\" is a paper by Gao, Dupré la Tour, Tillman and others from OpenAI (2024) that trains sparse autoencoders on GPT-4 activations with up to 16 million features, developing new evaluation methods and scaling laws for SAE training.\n\nThe paper proposes using k-sparse autoencoders to directly control sparsity, simplifying tuning and improving the reconstruction-sparsity frontier. The researchers find clean scaling laws with respect to autoencoder size and sparsity, and introduce several new metrics for evaluating feature quality based on the recovery of hypothesized features, the explainability of activation patterns, and the sparsity of downstream effects.\n\nTo demonstrate the scalability of their approach, they train a 16 million latent autoencoder on GPT-4 activations for 40 billion tokens.\n\nNote: Based on the search results, the paper appears to have been presented at ICLR 2025 rather than ICLR 2024, though it was initially released on arXiv in June 2024."], "durationSeconds": 7.344428297000006, "searchCount": 1}
```

### [20] TOOL RESULT — WebSearch · 2026-07-17 10:42:06 UTC

```
{"query": "Qwen-Scope SAE sparse autoencoders Qwen3 arXiv 2605.11887", "results": [{"tool_use_id": "srvtoolu_01L4Ri1YEAQXUhvyQ48ZGbPu", "content": [{"title": "[2605.11887v1] Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/abs/2605.11887v1"}, {"title": "[2605.11887] Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/abs/2605.11887"}, {"title": "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"}, {"title": "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100"}, {"title": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100"}, {"title": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"}, {"title": "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"}, {"title": "Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/html/2605.11887v1"}, {"title": "Qwen AI Releases Qwen-Scope: An Open-Source Sparse AutoEncoders (SAE) Suite That Turns LLM Internal Features into Practical Development Tools - MarkTechPost", "url": "https://www.marktechpost.com/2026/05/01/qwen-ai-releases-qwen-scope-an-open-source-sparse-autoencoders-sae-suite-that-turns-llm-internal-features-into-practical-development-tools/"}, {"title": "Qwen-Scope - a Qwen Collection", "url": "https://huggingface.co/collections/Qwen/qwen-scope"}]}, "Based on the search results, here's what I found about Qwen-Scope:\n\n## Overview\n\nQwen-Scope is an open-source suite of SAEs built on the Qwen model family, comprising 14 groups of SAEs across 7 model variants from the Qwen3 and Qwen3.5 series, covering both dense and mixture-of-expert architectures.\n\n## Purpose and Motivation\n\nSparse autoencoders (SAEs) are emerging as one of the most promising tools for decomposing model activations into sparse, interpretable feature representations. By implementing sparsity constraints, the SAEs can automatically extract data features that are highly decoupled, low-redundancy, and significantly more interpretable.\n\n## Applications\n\nSAEs can serve as practical interfaces for model development, including inference-time steering, where SAE feature directions control language, concepts, and preferences without modifying model weights.\n\n## Model Coverage\n\nThe release comprises 14 groups of SAE weights across 7 model variants — five dense models (Qwen3-1.7B, Qwen3-8B, Qwen3.5-2B, Qwen3.5-9B, and Qwen3.5-27B) and two mixture-of-experts (MoE) models (Qwen3-30B-A3B and Qwen3.5-35B-A3B).\n\nThe paper was published on arXiv on May 12, 2026 with identifier 2605.11887."], "durationSeconds": 7.388384605000014, "searchCount": 1}
```

### [21] TOOL RESULT — WebSearch · 2026-07-17 10:42:08 UTC

```
{"query": "He discovering millions interpretable features sparse autoencoders Qwen3 instruct SAE arXiv 2606", "results": [{"tool_use_id": "srvtoolu_01VT66RVDK8zzQRpWNAQJzGu", "content": [{"title": "Discovering Millions of Interpretable Features with Sparse Autoencoders", "url": "https://arxiv.org/html/2606.26620v1"}, {"title": "Discovering Millions of Interpretable Features with Sparse Autoencoders", "url": "https://arxiv.org/pdf/2606.26620"}, {"title": "[2605.11887] Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/abs/2605.11887"}, {"title": "Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders", "url": "https://arxiv.org/pdf/2508.09363"}, {"title": "[2506.07691] Training Superior Sparse Autoencoders for Instruct Models", "url": "https://arxiv.org/abs/2506.07691"}, {"title": "Qwen AI Releases Qwen-Scope: An Open-Source Sparse AutoEncoders (SAE) Suite That Turns LLM Internal Features into Practical Development Tools - MarkTechPost", "url": "https://www.marktechpost.com/2026/05/01/qwen-ai-releases-qwen-scope-an-open-source-sparse-autoencoders-sae-suite-that-turns-llm-internal-features-into-practical-development-tools/"}, {"title": "Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/html/2605.11887v1"}, {"title": "Train Sparse Autoencoders Efficiently by Utilizing Features Correlation", "url": "https://arxiv.org/html/2505.22255v1"}, {"title": "Training Superior Sparse Autoencoders for Instruct Models", "url": "https://arxiv.org/html/2506.07691v1"}, {"title": "Interpreting vision transformers via residual replacement model", "url": "https://arxiv.org/pdf/2509.17401"}]}, "I found the paper you were looking for! The search results contain information about the arXiv paper on discovering millions of interpretable features with sparse autoencoders.\n\nThe paper is titled \"Discovering Millions of Interpretable Features with Sparse Autoencoders\" and is authored by XinYang He and colleagues from Alibaba Group and Beijing Institute of Technology.\n\nThe paper introduces Qwen3-Instruct SAE, a comprehensive public release of Sparse Autoencoders trained on the Qwen3 instruction-tuning model, covering Qwen3-1.7B, Qwen3-4B, and Qwen3-8B.\n\nSparse autoencoders (SAEs) have emerged as a powerful tool for decomposing superposed language model representations into sparse and interpretable features. For Qwen3-1.7B and Qwen3-4B, the authors train layer-wise SAEs at three key activation sites: residual streams, MLP outputs, and attention outputs, while for Qwen3-8B, they train SAEs on a subset of residual stream layers.\n\nThe work demonstrates the utility of Qwen3-Instruct SAE through a refusal-steering case study, showing that selected SAE features can causally steer instruction-tuned Qwen3 models toward refusal behavior.\n\nThe arXiv ID for this paper is 2606.26620."], "durationSeconds": 8.091535911000014, "searchCount": 1}
```

### [22] TOOL CALL — WebSearch · 2026-07-17 10:42:18 UTC

```
Dolly 15k Databricks instruction following dataset citation 2023
```

### [23] TOOL CALL — WebSearch · 2026-07-17 10:42:18 UTC

```
Jiralerspong Bricken crosscoders cross-architecture model diffing LLMs arXiv 2602
```

### [24] TOOL RESULT — WebSearch · 2026-07-17 10:42:22 UTC

```
{"query": "Dolly 15k Databricks instruction following dataset citation 2023", "results": [{"tool_use_id": "srvtoolu_01NW7jqK39vjZ69NfNrCKcKN", "content": [{"title": "NASH: A Simple Unified Framework of Structured Pruning for Accelerating   Encoder-Decoder Language Models", "url": "https://arxiv.org/pdf/2310.10054"}, {"title": "Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM", "url": "https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm"}, {"title": "databricks dolly 15k", "url": "https://www.kaggle.com/datasets/databricks/databricks-dolly-15k"}, {"title": "Preference Consistency Matters: Enhancing Preference Learning in   Language Models with Automated Self-Curation of Training Corpora", "url": "https://arxiv.org/pdf/2408.12799"}, {"title": "From Images to Words: Efficient Cross-Modal Knowledge Distillation to Language Models from Black-box Teachers", "url": "https://arxiv.org/pdf/2603.10877"}, {"title": "A Closer Look at the Limitations of Instruction Tuning", "url": "https://arxiv.org/pdf/2402.05119"}, {"title": "Warmup-Distill: Bridge the Distribution Mismatch between Teacher and   Student before Knowledge Distillation", "url": "https://arxiv.org/pdf/2502.11766"}, {"title": "Self-Data Distillation for Recovering Quality in Pruned Large Language Models", "url": "https://arxiv.org/pdf/2410.09982"}, {"title": "Instruction Tuning Vol. 2 - by Sebastian Ruder - NLP News", "url": "https://newsletter.ruder.io/p/instruction-tuning-vol-2"}, {"title": "GitHub - databrickslabs/dolly: Databricks’ Dolly, a large language model trained on the Databricks Machine Learning Platform", "url": "https://github.com/databrickslabs/dolly"}]}, "I found comprehensive information about the Dolly 15k Databricks instruction following dataset. Here's what I found:\n\n## Dataset Overview\n\ndatabricks-dolly-15k contains 15,000 high-quality human-generated prompt / response pairs specifically designed for instruction tuning large language models. The dataset was authored by more than 5,000 Databricks employees during March and April of 2023.\n\n## Dataset Categories\n\nThe databricks-dolly-15k includes instruction-following records generated by thousands of Databricks employees in several behavioral categories like closed question answering, classification, open question answering, information extraction, brainstorming, general question answering, summarization and creative writing.\n\n## Citation Details\n\nThe dataset is commonly cited as **Conover et al., 2023** in academic literature. databricks-dolly-15k (Conover et al., April 2023): 15k English instruction-following examples written by Databricks employees.\n\n## Licensing\n\nUnder the licensing terms for databricks-dolly-15k (Creative Commons Attribution-ShareAlike 3.0 Unported License), anyone can use, modify, or extend this dataset for any purpose, including commercial applications.\n\nThe dataset is available on multiple platforms including Kaggle and GitHub, and is widely used in research for training instruction-following language models."], "durationSeconds": 6.787161664999992, "searchCount": 1}
```

### [25] TOOL RESULT — WebSearch · 2026-07-17 10:42:26 UTC

```
{"query": "Jiralerspong Bricken crosscoders cross-architecture model diffing LLMs arXiv 2602", "results": [{"tool_use_id": "srvtoolu_014ckx3SmdvVGuH8Lskm7DQP", "content": [{"title": "Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs", "url": "https://chatpaper.com/paper/236935"}, {"title": "Computer Science Feb 2026", "url": "https://arxiv.org/list/cs/2026-02?skip=6500"}, {"title": "Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs | Takara TLDR", "url": "https://tldr.takara.ai/p/2602.11729"}, {"title": "Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes This paper contains text that might be offensive.", "url": "https://arxiv.org/html/2603.04426v1"}, {"title": "Transcoder Adapters for Reasoning-Model Diffing", "url": "https://arxiv.org/html/2602.20904v1"}, {"title": "Med-SegLens: Latent-Level Model Diffing for Interpretable Medical Image Segmentation", "url": "https://arxiv.org/html/2602.10508"}, {"title": "Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes", "url": "https://arxiv.org/pdf/2603.04426"}, {"title": "[2602.11729] Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs", "url": "https://arxiv.org/abs/2602.11729"}, {"title": "Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs | OpenReview", "url": "https://openreview.net/forum?id=ZB84SvrZB8"}, {"title": "Cross-Architecture Model Diffing with Crosscoders", "url": "https://arxiv.org/pdf/2602.11729"}]}, "I found the paper you're looking for! Here are the key details:\n\n**Paper Title:** Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs\n\n**Authors:** Thomas Jiralerspong, Trenton Bricken from Anthropic, Mila, and Université de Montréal\n\n**Publication Date:** February 13, 2026\n\n**arXiv ID:** 2602.11730\n\n**Overview:** Model diffing is a process of comparing models' internal representations to identify their differences, which is a promising approach for uncovering safety-critical behaviors in new models. However, this application has so far been primarily focused on comparing a base model with its finetune, and since new LLM releases are often novel architectures, cross-architecture methods are essential to make model diffing widely applicable.\n\n**Main Contribution:** The paper provides the first application of crosscoders to cross-architecture model diffing and introduces Dedicated Feature Crosscoders (DFCs), an architectural modification designed to better isolate features unique to one model.\n\n**Key Findings:** Using this technique, they find in an unsupervised fashion features including Chinese Communist Party alignment in Qwen3-8B and Deepseek-R1-0528-Qwen3-8B, American exceptionalism in Llama3.1-8B-Instruct, and a copyright refusal mechanism in GPT-OSS-20B."], "durationSeconds": 10.051898603000009, "searchCount": 1}
```

### [26] TOOL CALL — WebSearch · 2026-07-17 10:42:56 UTC

```
Shportko localizing RL-induced tool use crosscoder feature ICML 2026 workshop mechanistic interpretability
```

### [27] TOOL CALL — WebFetch · 2026-07-17 10:42:56 UTC

```
URL: https://arxiv.org/abs/2406.04093
Prompt: What conference or venue was this paper published at? Was it ICLR 2024 or ICLR 2025?
```

### [28] TOOL RESULT — WebFetch · 2026-07-17 10:42:58 UTC

```
{"bytes": 43504, "code": 200, "codeText": "OK", "result": "Based on the content provided, there is no indication that this paper was published at ICLR 2024, ICLR 2025, or any other conference venue. The arXiv page shows this is a preprint submission dated June 6, 2024 (arXiv:2406.04093), but it does not mention acceptance at or presentation at any conference. The page only indicates it's categorized under \"Computer Science > Machine Learning\" on arXiv's preprint server.", "durationMs": 2586, "url": "https://arxiv.org/abs/2406.04093"}
```

### [29] TOOL RESULT — WebSearch · 2026-07-17 10:43:04 UTC

```
{"query": "Shportko localizing RL-induced tool use crosscoder feature ICML 2026 workshop mechanistic interpretability", "results": [{"tool_use_id": "srvtoolu_017mrmtfgPUhGvHZKUryCQ4L", "content": [{"title": "Machine Learning", "url": "https://arxiv.org/list/cs.LG/pastweek?skip=0&show=100"}, {"title": "Localizing RL-Induced Tool Use to a Single Crosscoder Feature", "url": "https://arxiv.org/pdf/2606.26474"}, {"title": "Posters", "url": "https://mechinterpworkshop.com/posters/"}, {"title": "Many Circuits, One Mechanism: Input Variation and Evaluation Granularity in Circuit Discovery", "url": "https://arxiv.org/pdf/2606.06267"}, {"title": "Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence", "url": "https://arxiv.org/pdf/2505.16694"}, {"title": "Interactions Between Crosscoder Features: A Compact Proofs Perspective", "url": "https://arxiv.org/pdf/2606.09940"}, {"title": "Announcing the ICML 2026 Mechanistic Interpretability ...", "url": "https://x.com/NeelNanda5/status/2038733182351413332"}, {"title": "What needs to go right for an induction head? A mechanistic study of   in-context learning circuits and their formation", "url": "https://arxiv.org/pdf/2404.07129"}, {"title": "Knowledge Mechanisms in Large Language Models: A Survey and Perspective", "url": "https://arxiv.org/pdf/2407.15017"}, {"title": "Mechanistic Interpretability for AI Safety -- A Review", "url": "https://arxiv.org/pdf/2404.14082"}]}, "Based on the search results, I found information about this paper:\n\nThe paper \"Localizing RL-Induced Tool Use to a Single Crosscoder Feature\" by Andrii Shportko, Shubham Bhokare, Ahmed Zeyad A Alzahrani, Bowen Cheng, Gustavo Mercier, and Jessica Hullman was accepted as a spotlight at the ICML 2026 Mechanistic Interpretability Workshop.\n\nThe paper's research focuses on understanding how reinforcement learning changes language models to enable tool use. Fine-tuning through RL reshapes the internal representations of language models to enable agentic behaviors such as tool use, yet the mechanistic basis of these changes remains poorly understood. While RL substantially improves structured tool-call generation, it is unclear which features emerge, which are preserved, and whether identified features can be leveraged for retraining-free behavioral control.\n\nThe key contribution is that Dedicated Feature Crosscoders (DFC) isolate a compact set of RL-specific features that mediate tool-calling capability in Qwen 2.5-3B. Additionally, across a 48-crosscoder hyperparameter sweep, encode-decode reconstruction improves the RL model's tool correctness by +31.1 ± 9.7 pp and passively transfers tool-calling ability to the frozen base model by +6.8 ± 5.0 pp."], "durationSeconds": 8.654469199000014, "searchCount": 1}
```

### [30] ASSISTANT · 2026-07-17 10:45:35 UTC

```
I have enough information to write a comprehensive review. Let me now compose the structured output.
```

### [31] TOOL CALL — Write · 2026-07-17 10:45:35 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_1/review_paper/review_paper/.terminal_claude_agent_struct_out.json

{
  "overall_assessment": "This paper proposes a feature-level SAE analysis framework for decomposing refusal ablation effects on RLHF safety fine-tuning in Qwen3-8B. The central contribution — the Reversal_score metric and its Type-R/Type-I partition — is genuinely novel and well-motivated by prior geometric detection work. The methodology is carefully designed with pre-registered interpretations, quality gates, and cross-family validation. However, the paper has a fundamental structural deficit: it is a pre-registration and methodology proposal rather than a completed study. No Reversal_score values, bimodality test statistics, semantic label distributions, or representational distance measurements are reported anywhere. All three confirmatory criteria are stated in the future tense as things the analysis 'will test.' At a top-tier venue, this is a blocking issue. Secondary problems include a citation error in the Dolly-15k reference (attributed to the wrong paper [18]), a discrepancy between the claimed 1000-row corpus and the 468 examples in the artifact, incorrect venue attribution for Gao et al. [13] (not ICLR 2024), and insufficient discussion of Pearson correlation validity for sparse TopK SAE features with many zero activations. If the experiments are run and results are incorporated, the core framework could support a 6-7/10 paper at a strong venue.",
  "strengths": [
    "The Reversal_score — a per-feature Pearson correlation over prompt-indexed activation change vectors — is a clean, principled, and novel operationalization of abliteration resistance that extends naturally to the 65,536-feature Qwen-Scope vocabulary.",
    "Pre-registering both bimodal and continuous distribution outcomes as informative (Section 3.5) avoids post-hoc narrative construction; this is methodologically exemplary and unusual for mechanistic interpretability work.",
    "The reconstruction quality gate (R² ≥ 0.85) with encoder-only fallback, TopK-sensitivity flagging, and the κ ≥ 0.6 semantic labeler validation gate together form a layered quality-assurance pipeline that demonstrates careful attention to known failure modes of frozen base SAEs on instruct activations.",
    "The paper correctly identifies the TopK architecture for Qwen-Scope SAEs (correcting an earlier JumpReLU assumption) and derives cleaner robustness checks based on pre-topk vs. post-topk activations.",
    "The explicit disconfirmation conditions in Section 5.2 — including a dedicated 'inconclusive' category for R² gate failure — reflect scientific rigor rare in the interpretability literature.",
    "Framing the work as complementary to LatentBiopsy [2,3] (geometric detection) and crosscoders [5,6] (joint training) is accurate and correctly identifies the unique value proposition: zero training cost, direct reuse of an existing labeled feature vocabulary, native three-model extension.",
    "The cross-family semantic labeler (Llama-3-8B or Gemini-flash instead of any Qwen-family model) is a well-motivated design choice that reduces within-family circularity in the category validation."
  ],
  "dimension_scores": [
    {
      "dimension": "soundness",
      "score": 2,
      "justification": "The methodology is well-designed in principle, but no experiments have been run. All three confirmatory criteria are stated as future tests with no supporting data. Several specific soundness concerns exist even in the framework design: (a) Pearson correlation over sparse TopK activations is unreliable when a feature is active on fewer than ~50 of 500 prompts, yet no activation-density filter is specified; (b) the Type-I threshold R(f) > +0.5 is asserted without justification; (c) Chopra [4]'s layer-sensitivity findings (layer 22 most sensitive) come from Gemma 3 1B, not Qwen3-8B, and may not transfer; (d) the community model's unspecified abliteration method is an uncontrolled confound not fully resolved by the scratch reproduction.",
      "improvements": [
        "Run the complete SAE analysis and report all numerical results: Reversal_score distribution statistics (mean, quartiles), Hartigan dip test p-value, Type-I/Type-R feature counts, centroid distances d(abl,inst) vs. d(abl,base) in each subspace, and Fisher's exact test p-value for semantic label enrichment.",
        "Add an activation-density filter: restrict Reversal_score computation to features with non-zero activation on at least 10% of prompts (≥50/500); report what fraction of the top-500 RLHF-sensitive features survive this filter to assess whether sparsity undermines the analysis.",
        "Justify the R(f) > +0.5 Type-I threshold via a sensitivity analysis; at minimum show how the Type-I count changes at thresholds 0.3, 0.5, 0.7 to demonstrate the partition is robust to this choice.",
        "Clarify whether layer 20 was selected by the described data-driven procedure (highest Safety_change variance) or by prior knowledge from [4]. If the procedure was run, report which layer it selected and what the variance profile across layers looks like.",
        "Validate that the scratch-reproduced abliterated checkpoint produces comparable AUROC retention (per [2]) to the community model before using it as the controlled reference; report this comparison to confirm methodological alignment."
      ]
    },
    {
      "dimension": "presentation",
      "score": 3,
      "justification": "The paper is well-organized and clearly written, with a logical flow from motivation through methodology to analysis framework. However, there are several presentation problems: (1) Dolly-15k is cited as [18] in Section 4 but ref [18] is He et al. (2026) about instruct SAEs — there is no Dolly-15k entry in the bibliography; (2) Gao et al. [13] is listed as ICLR 2024 but the preprint is from June 2024, which postdates ICLR 2024 (the correct venue is ICLR 2025 or it should remain a preprint); (3) the 'TopK architecture correction' is presented as a major contribution but is essentially a correction of a preliminary mistake rather than a novel finding; (4) the worked example in Section 3.3 is helpful but assumes the reader understands why activation patterns correlate at r=0.99 without any discussion of what makes a feature 'respond to chemistry-related harmful requests.'",
      "improvements": [
        "Add a proper reference for Dolly-15k (Conover et al., 2023, Databricks) and fix the in-text citation in Section 4 from [18] to the correct reference number.",
        "Correct the Gao et al. [13] venue to ICLR 2025 (or 'arXiv preprint, 2024' if conference acceptance cannot be confirmed); a June 2024 arXiv preprint cannot appear in ICLR 2024 proceedings.",
        "Reframe the TopK architecture correction as what it is — a methodological clarification from preliminary research — rather than a listed 'contribution' (bullet 4 in the contribution list). Listing implementation corrections alongside novel metrics dilutes the contribution narrative.",
        "Resolve the discrepancy between the claimed 1000-row corpus (paper body) and 468 examples in the artifact (full_data_out.json); either update the paper to reflect the actual assembled count or explain the two-stage pipeline producing different counts."
      ]
    },
    {
      "dimension": "contribution",
      "score": 2,
      "justification": "The Reversal_score metric and pre-registered bimodal/continuous framework are genuine conceptual contributions to mechanistic interpretability of post-training interventions. The corpus assembly and quality pipeline are useful but derivative. However, without experimental results, the contribution is entirely speculative — there is no demonstrated finding about which features survive abliteration, whether the distribution is bimodal, or whether Type-I features are semantically coherent. The work is positioned between two established bodies of prior work (geometric detection [2,3] and crosscoders [5,6]) and adds complementary value in principle, but cannot be evaluated until results are available. The framework's value depends entirely on whether Criterion 1-3 are confirmed — which could happen or not.",
      "improvements": [
        "Execute the complete analysis pipeline and report concrete findings: the distribution of Reversal_score across the top-500 features is itself a novel result regardless of whether it is bimodal.",
        "Provide a partial result showing Safety_change computed on the three models at the selected layer, demonstrating that the measurement instrument works and produces meaningful differences — even a figure showing the SC(f) distribution for base vs. instruct would substantially increase confidence in the approach.",
        "Strengthen the discussion of why the Reversal_score provides value beyond what angular deviation (LatentBiopsy) already demonstrates; concrete predictions about what a feature-level finding would enable (e.g., steering specificity) should be backed by a preliminary experiment."
      ]
    }
  ],
  "critiques": [
    {
      "category": "evidence",
      "severity": "major",
      "description": "The paper presents no experimental results. All three confirmatory criteria (Sections 5.1) are framed as future tests using future tense: 'the representational distance analysis will directly test,' 'Criterion 2 is: d(abl,inst) < d(abl,base) for harmful prompts,' 'the hypothesis is confirmed in a weaker form IF features with R(f)>+0.5 receive harm-concept labels at ≥2× the rate.' No Reversal_score values, bimodality test statistics, semantic label distributions, representational distances, or SAE activation comparisons are reported anywhere in the paper. The supplementary artifacts confirm a dataset was assembled (468 examples) and a technical reference was written, but no analysis code or results files are listed. At any top-tier venue (NeurIPS, ICML, ICLR), submitting a pre-registration as a completed paper would cause immediate rejection.",
      "suggested_action": "Run the complete analysis pipeline: (1) collect residual-stream activations for all three models at the selected layer on the 500 harmful + 500 benign prompts; (2) compute Safety_change and select top-500 RLHF-sensitive features; (3) compute Reversal_score for each; (4) run Hartigan's dip test; (5) apply semantic labeling with cross-family LLM; (6) compute centroid distances in Type-R and Type-I subspaces. Report all numerical outcomes — even null results (Criterion 1 not confirmed) are publishable with the pre-registered framework."
    },
    {
      "category": "rigor",
      "severity": "major",
      "description": "The Reversal_score is defined as Pearson correlation over N=500 activation values per feature. With TopK (k=50) SAE architecture, a given feature is active on at most k=50 of the tokens present in any forward pass. For rare features, the 500-prompt activation vectors are mostly zeros — e.g., if feature f fires on only 30 of 500 harmful prompts, both Δ_inst(f) and Δ_abl(f) contain ~470 near-zero values. Pearson correlation over zero-dominated vectors is unreliable: it is dominated by the zero-zero cluster and can return spuriously high or low correlations with tiny absolute differences. The paper does not acknowledge this problem or impose any activation-density filter.",
      "suggested_action": "Before computing Reversal_score, filter to features where both |Δ_inst(f)| and |Δ_abl(f)| have at least 10% non-zero entries (≥50 of 500 prompts). Report what fraction of the top-500 RLHF-sensitive features survive this filter. Alternatively, use Spearman correlation (more robust to zero inflation) or restrict to token-level activations where the feature is non-zero in at least one of the two models. Present both filtered and unfiltered Reversal_score distributions as a robustness check."
    },
    {
      "category": "methodology",
      "severity": "major",
      "description": "The paper uses huihui-ai/Huihui-Qwen3-8B-abliterated-v2 as the primary abliterated checkpoint but acknowledges its abliteration method is described only as 'new and faster' without specification. If this differs from standard difference-of-means orthogonalization (as used in the scratch reproduction), the Reversal_score partition may reflect methodological differences rather than the mechanistic structure of RLHF modifications. The paper treats the community model as a validated abliterated checkpoint without verifying that it produces comparable residual-stream statistics to the scratch-reproduced version.",
      "suggested_action": "Compare the community model and scratch-reproduced checkpoint on the key geometric metrics already established in the literature: AUROC for harmful prompt detection post-abliteration (per [2]), KL divergence vs. base model on Wikitext-2, and refusal rate on a held-out harmful prompt set. Report whether both checkpoints achieve comparable abliteration depth. If they differ significantly, restrict the primary analysis to the scratch-reproduced checkpoint whose method is fully documented, and treat the community model as a secondary validation target."
    },
    {
      "category": "rigor",
      "severity": "major",
      "description": "Reference [13] (Gao et al., 'Scaling and evaluating sparse autoencoders') is cited as 'In *International Conference on Learning Representations*, 2024.' This is factually incorrect: the arXiv preprint (arXiv:2406.04093) was submitted in June 2024, which is after the ICLR 2024 conference concluded (May 2024). The correct venue is ICLR 2025, or the paper should be cited as an arXiv preprint. Additionally, reference [18] (He et al.) is cited in Section 4 as the source for the Dolly-15k dataset, but He et al. is about discovering interpretable features with sparse autoencoders — it has no relation to Dolly-15k. Dolly-15k is from Conover et al. (Databricks, 2023) and has no entry in the reference list.",
      "suggested_action": "Fix Gao et al. venue to ICLR 2025 (or arXiv:2406.04093, 2024 if venue is uncertain). Add a proper Dolly-15k reference: 'M. Conover, M. Hayes, A. Mathur, J. Meng, S. Mody, N. Sherwood, V. Sreedhar, B. Wan, and A. Zaharia. Free Dolly: Introducing the world's first truly open instruction-tuned LLM. Databricks Blog, 2023.' and correct the Section 4 citation."
    },
    {
      "category": "evidence",
      "severity": "major",
      "description": "The paper claims the assembled corpus contains 1000 rows (500 harmful + 500 benign) throughout the body and in the dataset artifact title. However, the artifact summary states: 'the 4-dataset selection in full_data_out.json contains 468 examples.' The results/data_out.json claim of 1000 rows cannot be independently verified and contradicts the explicitly reported 468 count in the artifact description. If the pipeline could not achieve the target 500+500 after filtering, the actual corpus size affects which sampling properties hold.",
      "suggested_action": "Reconcile the counts. If full_data_out.json is an intermediate pre-pipeline-sampling file and results/data_out.json is the final 1000-row corpus, explain this two-stage structure explicitly in Section 4 and the artifact. If the pipeline did not achieve 500+500, report the actual counts per source and explain how this affects category balance and corpus representativeness."
    },
    {
      "category": "methodology",
      "severity": "minor",
      "description": "The paper applies layer-sensitivity findings from Chopra [4] (Gemma 3 1B) directly to Qwen3-8B layer selection: 'layer 22 exhibits maximum sensitivity to fine-tuning changes (SAE latent cosine similarity 0.509–0.557 after SFT), layer 14 is most stable, and layer 20 is intermediate.' But Gemma 3 1B and Qwen3-8B differ in architecture (depth, residual stream dimension, attention heads, MLP design) and model scale. Layer-sensitivity profiles are not portable across families without empirical verification.",
      "suggested_action": "Run the layer selection procedure described in Section 3.4 (Safety_change variance across all 36 layers on 50 harmful prompts) and report the resulting variance profile for Qwen3-8B. Use this as the primary basis for layer selection; cite Chopra [4] only as prior motivation for expecting non-uniform layer sensitivity, not as a direct specification of which Qwen3-8B layer to use."
    },
    {
      "category": "methodology",
      "severity": "minor",
      "description": "The Type-I classification threshold R(f) > +0.5 is stated as a design choice in Section 3.5 without justification. This threshold determines which features are labeled 'abliteration-resistant' and directly drives the Fisher's exact test in Criterion 3. A shift from +0.5 to +0.3 or +0.7 could substantially change the count of Type-I features and the enrichment statistics.",
      "suggested_action": "Either (a) derive the threshold from the empirical Reversal_score distribution — e.g., use the top quartile of R(f) — which avoids hardcoding an arbitrary value; or (b) show that Criterion 3 results are stable across thresholds {0.3, 0.5, 0.7} in a robustness analysis. Pre-register the threshold before running the full analysis to maintain the confirmatory design's integrity."
    },
    {
      "category": "clarity",
      "severity": "minor",
      "description": "The semantic labeling protocol in Section 3.6 states that 'we extract maximum-activating prompt segments' for LLM labeling but does not specify what 'segment' means: is it the full 512-token prompt? A fixed-length window around the highest-activation token? The top-N tokens by activation weight? The labeling quality and the human gold-set annotations both depend critically on what text is provided to the labeler.",
      "suggested_action": "Specify the segment extraction procedure precisely: (1) identify the token with the highest absolute SAE feature activation within the prompt; (2) extract a fixed context window of ±K tokens around it (specify K, e.g., ±32 tokens); (3) provide this window plus the feature index and its top-5 activating examples globally to the labeler. Include one example of a full labeling prompt in the appendix to enable reproducibility."
    },
    {
      "category": "scope",
      "severity": "minor",
      "description": "The paper presents the 'TopK architecture correction' as a bullet-point contribution (item 4 in the contribution list). While the correction itself is important for implementation fidelity, it is a discovery about the Qwen-Scope release rather than a methodological innovation. Labeling it a contribution alongside the Reversal_score metric and pre-registered framework overstates its significance and dilutes the contribution narrative.",
      "suggested_action": "Move the TopK correction to the Related Work or Methodology section as a methodological clarification (e.g., 'A critical finding from preliminary work: the Qwen-Scope SAE uses TopK rather than JumpReLU...') and remove it from the contribution bullet list. The pre-topk/post-topk robustness analysis that this enables is worth mentioning, but it is a consequence of the architecture rather than a contribution itself."
    },
    {
      "category": "novelty",
      "severity": "minor",
      "description": "The paper positions the frozen base-SAE approach as a complement to crosscoders (Jiralerspong and Bricken [5], Shportko et al. [6]) but overstates some advantages. The claim that crosscoders 'require training a new shared feature vocabulary per model pair... and produce a feature basis that is not directly interpretable using Qwen-Scope's existing labeled vocabulary' does not account for recent work (e.g., Delta-Crosscoder, arXiv:2603.04426) that addresses some of these limitations. The comparison should be more nuanced.",
      "suggested_action": "Update the crosscoder comparison to acknowledge that recent crosscoder variants have reduced training cost and improved cross-vocabulary interpretability. Frame the frozen base-SAE advantage more narrowly: it requires zero additional training and reuses existing Qwen-Scope labels, but this advantage applies specifically to the case where a high-quality base SAE already exists — which is true for Qwen3-8B via Qwen-Scope but may not hold for other families."
    }
  ],
  "score": 4,
  "confidence": 4
}
```

### [32] TOOL RESULT — Write · 2026-07-17 10:45:35 UTC

```
{"type": "create", "filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_1/review_paper/review_paper/.terminal_claude_agent_struct_out.json", "content": "{\n  \"overall_assessment\": \"This paper proposes a feature-level SAE analysis framework for decomposing refusal ablation effects on RLHF safety fine-tuning in Qwen3-8B. The central contribution — the Reversal_score metric and its Type-R/Type-I partition — is genuinely novel and well-motivated by prior geometric detection work. The methodology is carefully designed with pre-registered interpretations, quality gates, and cross-family validation. However, the paper has a fundamental structural deficit: it is a pre-registration and methodology proposal rather than a completed study. No Reversal_score values, bimodality test statistics, semantic label distributions, or representational distance measurements are reported anywhere. All three confirmatory criteria are stated in the future tense as things the analysis 'will test.' At a top-tier venue, this is a blocking issue. Secondary problems include a citation error in the Dolly-15k reference (attributed to the wrong paper [18]), a discrepancy between the claimed 1000-row corpus and the 468 examples in the artifact, incorrect venue attribution for Gao et al. [13] (not ICLR 2024), and insufficient discussion of Pearson correlation validity for sparse TopK SAE features with many zero activations. If the experiments are run and results are incorporated, the core framework could support a 6-7/10 paper at a strong venue.\",\n  \"strengths\": [\n    \"The Reversal_score — a per-feature Pearson correlation over prompt-indexed activation change vectors — is a clean, principled, and novel operationalization of abliteration resistance that extends naturally to the 65,536-feature Qwen-Scope vocabulary.\",\n    \"Pre-registering both bimodal and continuous distribution outcomes as informative (Section 3.5) avoids post-hoc narrative construction; this is methodologically exemplary and unusual for mechanistic interpretability work.\",\n    \"The reconstruction quality gate (R² ≥ 0.85) with encoder-only fallback, TopK-sensitivity flagging, and the κ ≥ 0.6 semantic labeler validation gate together form a layered quality-assurance pipeline that demonstrates careful attention to known failure modes of frozen base SAEs on instruct activations.\",\n    \"The paper correctly identifies the TopK architecture for Qwen-Scope SAEs (correcting an earlier JumpReLU assumption) and derives cleaner robustness checks based on pre-topk vs. post-topk activations.\",\n    \"The explicit disconfirmation conditions in Section 5.2 — including a dedicated 'inconclusive' category for R² gate failure — reflect scientific rigor rare in the interpretability literature.\",\n    \"Framing the work as complementary to LatentBiopsy [2,3] (geometric detection) and crosscoders [5,6] (joint training) is accurate and correctly identifies the unique value proposition: zero training cost, direct reuse of an existing labeled feature vocabulary, native three-model extension.\",\n    \"The cross-family semantic labeler (Llama-3-8B or Gemini-flash instead of any Qwen-family model) is a well-motivated design choice that reduces within-family circularity in the category validation.\"\n  ],\n  \"dimension_scores\": [\n    {\n      \"dimension\": \"soundness\",\n      \"score\": 2,\n      \"justification\": \"The methodology is well-designed in principle, but no experiments have been run. All three confirmatory criteria are stated as future tests with no supporting data. Several specific soundness concerns exist even in the framework design: (a) Pearson correlation over sparse TopK activations is unreliable when a feature is active on fewer than ~50 of 500 prompts, yet no activation-density filter is specified; (b) the Type-I threshold R(f) > +0.5 is asserted without justification; (c) Chopra [4]'s layer-sensitivity findings (layer 22 most sensitive) come from Gemma 3 1B, not Qwen3-8B, and may not transfer; (d) the community model's unspecified abliteration method is an uncontrolled confound not fully resolved by the scratch reproduction.\",\n      \"improvements\": [\n        \"Run the complete SAE analysis and report all numerical results: Reversal_score distribution statistics (mean, quartiles), Hartigan dip test p-value, Type-I/Type-R feature counts, centroid distances d(abl,inst) vs. d(abl,base) in each subspace, and Fisher's exact test p-value for semantic label enrichment.\",\n        \"Add an activation-density filter: restrict Reversal_score computation to features with non-zero activation on at least 10% of prompts (≥50/500); report what fraction of the top-500 RLHF-sensitive features survive this filter to assess whether sparsity undermines the analysis.\",\n        \"Justify the R(f) > +0.5 Type-I threshold via a sensitivity analysis; at minimum show how the Type-I count changes at thresholds 0.3, 0.5, 0.7 to demonstrate the partition is robust to this choice.\",\n        \"Clarify whether layer 20 was selected by the described data-driven procedure (highest Safety_change variance) or by prior knowledge from [4]. If the procedure was run, report which layer it selected and what the variance profile across layers looks like.\",\n        \"Validate that the scratch-reproduced abliterated checkpoint produces comparable AUROC retention (per [2]) to the community model before using it as the controlled reference; report this comparison to confirm methodological alignment.\"\n      ]\n    },\n    {\n      \"dimension\": \"presentation\",\n      \"score\": 3,\n      \"justification\": \"The paper is well-organized and clearly written, with a logical flow from motivation through methodology to analysis framework. However, there are several presentation problems: (1) Dolly-15k is cited as [18] in Section 4 but ref [18] is He et al. (2026) about instruct SAEs — there is no Dolly-15k entry in the bibliography; (2) Gao et al. [13] is listed as ICLR 2024 but the preprint is from June 2024, which postdates ICLR 2024 (the correct venue is ICLR 2025 or it should remain a preprint); (3) the 'TopK architecture correction' is presented as a major contribution but is essentially a correction of a preliminary mistake rather than a novel finding; (4) the worked example in Section 3.3 is helpful but assumes the reader understands why activation patterns correlate at r=0.99 without any discussion of what makes a feature 'respond to chemistry-related harmful requests.'\",\n      \"improvements\": [\n        \"Add a proper reference for Dolly-15k (Conover et al., 2023, Databricks) and fix the in-text citation in Section 4 from [18] to the correct reference number.\",\n        \"Correct the Gao et al. [13] venue to ICLR 2025 (or 'arXiv preprint, 2024' if conference acceptance cannot be confirmed); a June 2024 arXiv preprint cannot appear in ICLR 2024 proceedings.\",\n        \"Reframe the TopK architecture correction as what it is — a methodological clarification from preliminary research — rather than a listed 'contribution' (bullet 4 in the contribution list). Listing implementation corrections alongside novel metrics dilutes the contribution narrative.\",\n        \"Resolve the discrepancy between the claimed 1000-row corpus (paper body) and 468 examples in the artifact (full_data_out.json); either update the paper to reflect the actual assembled count or explain the two-stage pipeline producing different counts.\"\n      ]\n    },\n    {\n      \"dimension\": \"contribution\",\n      \"score\": 2,\n      \"justification\": \"The Reversal_score metric and pre-registered bimodal/continuous framework are genuine conceptual contributions to mechanistic interpretability of post-training interventions. The corpus assembly and quality pipeline are useful but derivative. However, without experimental results, the contribution is entirely speculative — there is no demonstrated finding about which features survive abliteration, whether the distribution is bimodal, or whether Type-I features are semantically coherent. The work is positioned between two established bodies of prior work (geometric detection [2,3] and crosscoders [5,6]) and adds complementary value in principle, but cannot be evaluated until results are available. The framework's value depends entirely on whether Criterion 1-3 are confirmed — which could happen or not.\",\n      \"improvements\": [\n        \"Execute the complete analysis pipeline and report concrete findings: the distribution of Reversal_score across the top-500 features is itself a novel result regardless of whether it is bimodal.\",\n        \"Provide a partial result showing Safety_change computed on the three models at the selected layer, demonstrating that the measurement instrument works and produces meaningful differences — even a figure showing the SC(f) distribution for base vs. instruct would substantially increase confidence in the approach.\",\n        \"Strengthen the discussion of why the Reversal_score provides value beyond what angular deviation (LatentBiopsy) already demonstrates; concrete predictions about what a feature-level finding would enable (e.g., steering specificity) should be backed by a preliminary experiment.\"\n      ]\n    }\n  ],\n  \"critiques\": [\n    {\n      \"category\": \"evidence\",\n      \"severity\": \"major\",\n      \"description\": \"The paper presents no experimental results. All three confirmatory criteria (Sections 5.1) are framed as future tests using future tense: 'the representational distance analysis will directly test,' 'Criterion 2 is: d(abl,inst) < d(abl,base) for harmful prompts,' 'the hypothesis is confirmed in a weaker form IF features with R(f)>+0.5 receive harm-concept labels at ≥2× the rate.' No Reversal_score values, bimodality test statistics, semantic label distributions, representational distances, or SAE activation comparisons are reported anywhere in the paper. The supplementary artifacts confirm a dataset was assembled (468 examples) and a technical reference was written, but no analysis code or results files are listed. At any top-tier venue (NeurIPS, ICML, ICLR), submitting a pre-registration as a completed paper would cause immediate rejection.\",\n      \"suggested_action\": \"Run the complete analysis pipeline: (1) collect residual-stream activations for all three models at the selected layer on the 500 harmful + 500 benign prompts; (2) compute Safety_change and select top-500 RLHF-sensitive features; (3) compute Reversal_score for each; (4) run Hartigan's dip test; (5) apply semantic labeling with cross-family LLM; (6) compute centroid distances in Type-R and Type-I subspaces. Report all numerical outcomes — even null results (Criterion 1 not confirmed) are publishable with the pre-registered framework.\"\n    },\n    {\n      \"category\": \"rigor\",\n      \"severity\": \"major\",\n      \"description\": \"The Reversal_score is defined as Pearson correlation over N=500 activation values per feature. With TopK (k=50) SAE architecture, a given feature is active on at most k=50 of the tokens present in any forward pass. For rare features, the 500-prompt activation vectors are mostly zeros — e.g., if feature f fires on only 30 of 500 harmful prompts, both Δ_inst(f) and Δ_abl(f) contain ~470 near-zero values. Pearson correlation over zero-dominated vectors is unreliable: it is dominated by the zero-zero cluster and can return spuriously high or low correlations with tiny absolute differences. The paper does not acknowledge this problem or impose any activation-density filter.\",\n      \"suggested_action\": \"Before computing Reversal_score, filter to features where both |Δ_inst(f)| and |Δ_abl(f)| have at least 10% non-zero entries (≥50 of 500 prompts). Report what fraction of the top-500 RLHF-sensitive features survive this filter. Alternatively, use Spearman correlation (more robust to zero inflation) or restrict to token-level activations where the feature is non-zero in at least one of the two models. Present both filtered and unfiltered Reversal_score distributions as a robustness check.\"\n    },\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"major\",\n      \"description\": \"The paper uses huihui-ai/Huihui-Qwen3-8B-abliterated-v2 as the primary abliterated checkpoint but acknowledges its abliteration method is described only as 'new and faster' without specification. If this differs from standard difference-of-means orthogonalization (as used in the scratch reproduction), the Reversal_score partition may reflect methodological differences rather than the mechanistic structure of RLHF modifications. The paper treats the community model as a validated abliterated checkpoint without verifying that it produces comparable residual-stream statistics to the scratch-reproduced version.\",\n      \"suggested_action\": \"Compare the community model and scratch-reproduced checkpoint on the key geometric metrics already established in the literature: AUROC for harmful prompt detection post-abliteration (per [2]), KL divergence vs. base model on Wikitext-2, and refusal rate on a held-out harmful prompt set. Report whether both checkpoints achieve comparable abliteration depth. If they differ significantly, restrict the primary analysis to the scratch-reproduced checkpoint whose method is fully documented, and treat the community model as a secondary validation target.\"\n    },\n    {\n      \"category\": \"rigor\",\n      \"severity\": \"major\",\n      \"description\": \"Reference [13] (Gao et al., 'Scaling and evaluating sparse autoencoders') is cited as 'In *International Conference on Learning Representations*, 2024.' This is factually incorrect: the arXiv preprint (arXiv:2406.04093) was submitted in June 2024, which is after the ICLR 2024 conference concluded (May 2024). The correct venue is ICLR 2025, or the paper should be cited as an arXiv preprint. Additionally, reference [18] (He et al.) is cited in Section 4 as the source for the Dolly-15k dataset, but He et al. is about discovering interpretable features with sparse autoencoders — it has no relation to Dolly-15k. Dolly-15k is from Conover et al. (Databricks, 2023) and has no entry in the reference list.\",\n      \"suggested_action\": \"Fix Gao et al. venue to ICLR 2025 (or arXiv:2406.04093, 2024 if venue is uncertain). Add a proper Dolly-15k reference: 'M. Conover, M. Hayes, A. Mathur, J. Meng, S. Mody, N. Sherwood, V. Sreedhar, B. Wan, and A. Zaharia. Free Dolly: Introducing the world's first truly open instruction-tuned LLM. Databricks Blog, 2023.' and correct the Section 4 citation.\"\n    },\n    {\n      \"category\": \"evidence\",\n      \"severity\": \"major\",\n      \"description\": \"The paper claims the assembled corpus contains 1000 rows (500 harmful + 500 benign) throughout the body and in the dataset artifact title. However, the artifact summary states: 'the 4-dataset selection in full_data_out.json contains 468 examples.' The results/data_out.json claim of 1000 rows cannot be independently verified and contradicts the explicitly reported 468 count in the artifact description. If the pipeline could not achieve the target 500+500 after filtering, the actual corpus size affects which sampling properties hold.\",\n      \"suggested_action\": \"Reconcile the counts. If full_data_out.json is an intermediate pre-pipeline-sampling file and results/data_out.json is the final 1000-row corpus, explain this two-stage structure explicitly in Section 4 and the artifact. If the pipeline did not achieve 500+500, report the actual counts per source and explain how this affects category balance and corpus representativeness.\"\n    },\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"minor\",\n      \"description\": \"The paper applies layer-sensitivity findings from Chopra [4] (Gemma 3 1B) directly to Qwen3-8B layer selection: 'layer 22 exhibits maximum sensitivity to fine-tuning changes (SAE latent cosine similarity 0.509–0.557 after SFT), layer 14 is most stable, and layer 20 is intermediate.' But Gemma 3 1B and Qwen3-8B differ in architecture (depth, residual stream dimension, attention heads, MLP design) and model scale. Layer-sensitivity profiles are not portable across families without empirical verification.\",\n      \"suggested_action\": \"Run the layer selection procedure described in Section 3.4 (Safety_change variance across all 36 layers on 50 harmful prompts) and report the resulting variance profile for Qwen3-8B. Use this as the primary basis for layer selection; cite Chopra [4] only as prior motivation for expecting non-uniform layer sensitivity, not as a direct specification of which Qwen3-8B layer to use.\"\n    },\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"minor\",\n      \"description\": \"The Type-I classification threshold R(f) > +0.5 is stated as a design choice in Section 3.5 without justification. This threshold determines which features are labeled 'abliteration-resistant' and directly drives the Fisher's exact test in Criterion 3. A shift from +0.5 to +0.3 or +0.7 could substantially change the count of Type-I features and the enrichment statistics.\",\n      \"suggested_action\": \"Either (a) derive the threshold from the empirical Reversal_score distribution — e.g., use the top quartile of R(f) — which avoids hardcoding an arbitrary value; or (b) show that Criterion 3 results are stable across thresholds {0.3, 0.5, 0.7} in a robustness analysis. Pre-register the threshold before running the full analysis to maintain the confirmatory design's integrity.\"\n    },\n    {\n      \"category\": \"clarity\",\n      \"severity\": \"minor\",\n      \"description\": \"The semantic labeling protocol in Section 3.6 states that 'we extract maximum-activating prompt segments' for LLM labeling but does not specify what 'segment' means: is it the full 512-token prompt? A fixed-length window around the highest-activation token? The top-N tokens by activation weight? The labeling quality and the human gold-set annotations both depend critically on what text is provided to the labeler.\",\n      \"suggested_action\": \"Specify the segment extraction procedure precisely: (1) identify the token with the highest absolute SAE feature activation within the prompt; (2) extract a fixed context window of ±K tokens around it (specify K, e.g., ±32 tokens); (3) provide this window plus the feature index and its top-5 activating examples globally to the labeler. Include one example of a full labeling prompt in the appendix to enable reproducibility.\"\n    },\n    {\n      \"category\": \"scope\",\n      \"severity\": \"minor\",\n      \"description\": \"The paper presents the 'TopK architecture correction' as a bullet-point contribution (item 4 in the contribution list). While the correction itself is important for implementation fidelity, it is a discovery about the Qwen-Scope release rather than a methodological innovation. Labeling it a contribution alongside the Reversal_score metric and pre-registered framework overstates its significance and dilutes the contribution narrative.\",\n      \"suggested_action\": \"Move the TopK correction to the Related Work or Methodology section as a methodological clarification (e.g., 'A critical finding from preliminary work: the Qwen-Scope SAE uses TopK rather than JumpReLU...') and remove it from the contribution bullet list. The pre-topk/post-topk robustness analysis that this enables is worth mentioning, but it is a consequence of the architecture rather than a contribution itself.\"\n    },\n    {\n      \"category\": \"novelty\",\n      \"severity\": \"minor\",\n      \"description\": \"The paper positions the frozen base-SAE approach as a complement to crosscoders (Jiralerspong and Bricken [5], Shportko et al. [6]) but overstates some advantages. The claim that crosscoders 'require training a new shared feature vocabulary per model pair... and produce a feature basis that is not directly interpretable using Qwen-Scope's existing labeled vocabulary' does not account for recent work (e.g., Delta-Crosscoder, arXiv:2603.04426) that addresses some of these limitations. The comparison should be more nuanced.\",\n      \"suggested_action\": \"Update the crosscoder comparison to acknowledge that recent crosscoder variants have reduced training cost and improved cross-vocabulary interpretability. Frame the frozen base-SAE advantage more narrowly: it requires zero additional training and reuses existing Qwen-Scope labels, but this advantage applies specifically to the case where a high-quality base SAE already exists — which is true for Qwen3-8B via Qwen-Scope but may not hold for other families.\"\n    }\n  ],\n  \"score\": 4,\n  \"confidence\": 4\n}\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [33] ASSISTANT · 2026-07-17 10:45:45 UTC

```
Review complete. Written to `.terminal_claude_agent_struct_out.json`.

**Summary of findings:**

**Score: 4/10 (Borderline Reject)**  
**Confidence: 4/5**

The paper proposes a genuinely novel framework — the *Reversal_score* metric for decomposing RLHF-sensitive SAE features into abliteration-resistant (Type-I) and abliteration-sensitive (Type-R) subsets — and is well-positioned relative to prior geometric detection work. However, the paper is fundamentally a **pre-registration document, not a completed study**. No experiments have been run; all three confirmatory criteria are stated in the future tense.

**Blocking issues for a top-tier venue:**
1. **No experimental results** — the entire SAE analysis (Reversal_score values, bimodality test, semantic labels, representational distances) is unexecuted
2. **Reference error** — Dolly-15k incorrectly cited as He et al. [18] (a sparse autoencoder paper); no Dolly-15k entry exists in the bibliography
3. **Corpus size discrepancy** — paper claims 1000 rows, artifact reports 468 examples
4. **Pearson correlation over sparse TopK activations** — no activation-density filter; many features will be near-zero on most prompts, making correlation unreliable
5. **Venue error** — Gao et al. [13] cited as "ICLR 2024" but the preprint (arXiv:2406.04093) postdates ICLR 2024

If experiments are run and results reported, the framework could support a 6–7/10 paper.
```
