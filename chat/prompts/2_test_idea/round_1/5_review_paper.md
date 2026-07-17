# review_paper — test_idea

> Phase: `invention_loop` · round 1 · `review_paper`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `review_paper` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-07-17 10:40:00 UTC

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

### [2] HUMAN-USER prompt · 2026-07-17 10:40:00 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [3] SKILL-INPUT — aii-web-tools · 2026-07-17 10:41:34 UTC

The agent loaded the **aii-web-tools** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-web-tools
description: "Web research toolkit: web search (Serper/Google), web page fetch as markdown (HTML and PDF), and regex grep over full page/PDF text. Use whenever a task needs to search the web, read a page, mine a paper/PDF, verify citations, or extract exact quotes, numbers, or methodology from a URL."
---

## Web tools

You have three web capabilities: **search**, **fetch**, and **grep** (exact
regex extraction over a full page or PDF).

**Pick where they come from, in this order:**

1. **If you have built-in `WebSearch` / `WebFetch` tools, PREFER those over the
   scripts below.** They may be **deferred tools** (listed by name but with
   schemas not yet loaded) — if so, call `ToolSearch("select:WebSearch,WebFetch")`
   ONCE to load them, then use them normally. Do not skip them just because they
   need that one extra load step; they are the preferred path. Pair them with the
   `aii_web_tools__fetch_grep` script below when you need exact text / numbers /
   methodology that a summary would miss, or when reading a PDF.
2. **Only if you have NO built-in `WebSearch` / `WebFetch`** (e.g. the OpenHands
   backend), use the scripts in this skill (below). They are our own
   implementations — Serper.dev for search, html2text + PyMuPDF for fetch, and
   regex grep over the full document text. They work without any built-in web
   tools.

Workflow either way: **search** (discover) → **fetch** (read for the gist) →
**grep** (pull exact details / read PDFs).

---

## Running the scripts

Run every script with the skill's pre-provisioned interpreter (it already has
`requests`, `html2text`, `pymupdf`, `python-dotenv`). Set `PY` once:

```bash
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-web-tools"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
```

### 1. Search the web (Serper.dev / Google)

```bash
$PY "$SKILL_DIR/scripts/aii_fast_web_search.py" --query "neuro-symbolic FOL translation LLM" --max-results 10
```

Returns ranked title / URL / snippet lines. Use it first to scan the
landscape; snippets are for discovery only — fetch a page before judging it.

### 2. Fetch a page as markdown (HTML or PDF)

```bash
$PY "$SKILL_DIR/scripts/aii_fast_web_fetch.py" fetch --url "https://arxiv.org/abs/2303.11366" --max-chars 10000
```

`--max-chars` caps output (default 10000); `--char-offset N` pages further in.
Handles PDFs transparently via PyMuPDF.

### 3. Grep a page or PDF (exact regex extraction)

```bash
$PY "$SKILL_DIR/scripts/aii_fast_web_fetch.py" grep --url "https://arxiv.org/pdf/2303.11366" --pattern "verbal reinforcement" --max-matches 20 --context-chars 200
```

Returns only the matching sections with surrounding context — the right tool
for exact numbers, table values, methodology, or long PDFs where a summary
would lose the detail. `-i` for case-insensitive.

**Parallelize** independent searches/fetches in one turn; only sequence a
fetch after the search that produced its URL.

---

## Notes

- The scripts call our ability server. If a script prints
  `Ability service not available`, the server is down — say so rather than
  silently improvising a different search method.
- Do **not** hand-roll your own `requests`/scraping for search when these
  tools are available: Serper returns clean Google results and the fetch/grep
  scripts already handle HTML, PDFs, and encoding.
````
