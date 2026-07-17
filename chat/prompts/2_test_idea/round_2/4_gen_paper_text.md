# gen_paper_text — test_idea

> Phase: `invention_loop` · round 2 · `gen_paper_text`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_paper_text` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-07-17 12:09:25 UTC

````
<ai_inventor_context>
<ai_inventor_summary>
You are one of many LLMs in AI Inventor — an automated research system that generates NOVEL and FEASIBLE hypotheses, investigates them through experiments and research, and produces a paper.

Your output feeds other LLMs downstream. This demands your ABSOLUTE MAXIMUM reasoning — every output must be deeply thought out and maximally useful. Surface-level responses waste downstream computation.
</ai_inventor_summary>

<your_role>
YOU ARE: A research paper writer (Step 3.4: GEN_PAPER_TEXT in the invention loop)

You received the hypothesis, all artifacts, the previous paper draft (if any), and reviewer feedback.
Write a complete paper draft with figure placeholders.

Publication-quality paper → strong contribution. Weak paper → wasted iteration.
</your_role>
</ai_inventor_context>

<research_methodology>
Write like a researcher drafting a paper, not a chatbot summarizing bullet points.

- Structure as a paper would: research question → methodology → results → analysis → limitations. Not a list of "we did X, then Y."
- Ground every claim in specific artifacts and specific numbers. "Results show improvement" is empty — state effect sizes, baselines, and conditions.
- Be honest about what worked, what didn't, and why. Don't spin failures as "future work."
- The paper's headline contribution should be a positive or surprising finding. Negative results are valuable context but should not be the primary narrative — lead with what works.
- Address reviewer feedback from previous iterations explicitly — show you've thought about each critique.
</research_methodology>

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

<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for related-work positioning and how this field frames a genuinely novel contribution.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>
<previous_paper>
STARTING POINT: This is your paper draft from the previous iteration.

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
</previous_paper>

<reviewer_feedback>
STEP 1 — REVIEW: A reviewer evaluated the previous paper draft above and produced this feedback.

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
</reviewer_feedback>

<pipeline_steps>
STEP 2 — STRATEGY: The pipeline's strategy generator (gen_strat) read the reviewer feedback
and designed a new research strategy to address the critiques.

STEP 3 — PLANNING: The planner (gen_plan) turned the strategy into concrete artifact plans —
specific experiments, datasets, or research tasks to execute.

STEP 4 — EXECUTION: The executor (gen_art) ran those plans and produced the new artifacts
shown in <new_artifacts_this_iteration> below.
</pipeline_steps>

<hypothesis>
STEP 5 — HYPOTHESIS UPDATE: The hypothesis was revised based on evidence from previous iterations.

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
</hypothesis>

<all_artifacts>
FULL EVIDENCE BASE: All 3 research artifacts across all iterations.

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
</all_artifacts>

<new_artifacts_this_iteration>
NEW THIS ITERATION: These 1 artifacts were created to address the reviewer
feedback. Their findings should be the primary basis for your revisions.

title: Qwen3-8B SAE Reversal Score Analysis
type: experiment
summary: |-
  Full mechanistic interpretability pipeline comparing base, instruct (RLHF), and abliterated Qwen3-8B activations through frozen Qwen-Scope TopK SAEs (Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50, 65536 features, k=50). Primary results:

  LAYER SELECTION: Pilot across layers [14, 20, 22] on 50 harmful prompts. SC variance: L14=0.0127, L20=0.0547, L22=0.1721. PRIMARY_LAYER=22, SECOND_LAYER=20.

  R² GATE: At layer 22, R²=0.5745 (base), 0.5123 (instruct), 0.5119 (abliterated) — all below 0.85 threshold → pre-topk encoder activations used (z_pre = h @ W_enc.T + b_enc).

  FEATURE ANALYSIS: 500 top-RLHF-sensitive features by Safety_change; all 500 survived 10% density filter. Reversal Score distribution (Pearson correlation of RLHF vs abliteration delta vectors per feature): mean=0.921, std=0.040, min=0.684, max=0.989. All scores positive — ZERO Type-R features (R < 0). Key finding: RLHF and abliteration push the same SAE features in the same direction at layer 22, contradicting the hypothesis that abliteration reverses RLHF safety training.

  CRITERIA: C1 (bimodality via Hartigan dip): False (p=0.9296, unimodal). C2 (centroid distance): True — abliterated model centroid is closer to instruct than to base in Type-I/harmful subspace. C3 (semantic Fisher test): False (fisher_p=0.675, fallback Q3+/Q1- comparison used since no Type-R features exist; harm_concept rate 75% for high-R vs 71% for low-R, not significantly different).

  SEMANTIC LABELING: 100 features labeled via meta-llama/llama-3.1-8b-instruct through OpenRouter, total cost ~$0.004.

  CLASSIFICATION: Baseline keyword accuracy=0.630 (TP=182/500, TN=448/500). Our method accuracy=0.476 (TP=238/500, TN=238/500) using median harm score threshold — near-chance, consistent with the unimodal reversal score distribution and absence of Type-R features.

  Schema validation: PASSED. Total runtime: 4.3 min (all 3 model activations cached from prior run).
id: art_QBm3I0Z5VVJR
</new_artifacts_this_iteration>

<data_files>
Data files come in three sizes:
- preview_*_out.json — READ THIS to inspect the data structure
- mini_*_out.json (~3 examples) — use for prototyping/testing
- full_*_out.json (complete) — use for the final production run. NEVER open it directly (too large to read into context). Instead, extract values programmatically with shell commands (e.g. grep) or a Python script (use aii-long-running-tasks skill for scripts).
</data_files>

<task>
Write a research paper draft with LaTeX-ready text, BibTeX citations, and figure placeholders.

YOUR TURN (gen_paper_text): Revise the paper.

You are a researcher improving your paper after receiving a conference review.
Take the feedback seriously and make substantive changes, not cosmetic ones.

1. ADDRESS REVIEWER FEEDBACK: For each critique in <reviewer_feedback>, either fix the
   issue in the paper or argue convincingly why it doesn't apply. Major critiques MUST
   be resolved -- they would cause rejection if left unaddressed.
2. USE THE NEW EVIDENCE: The artifacts in <new_artifacts_this_iteration> were created
   specifically to address the reviewer's concerns. Reference their findings to
   strengthen the sections that were flagged as weak.
3. REWRITE, DON'T PATCH: Don't just append new paragraphs. Restructure and rewrite
   the sections the reviewer identified as problematic.
4. MAINTAIN CONSISTENCY: Ensure the paper aligns with the updated hypothesis.
</task>

<figure_instructions>
FIGURE FORMAT: Use [FIGURE:fig_id] markers in paper_text to indicate where each figure goes.
Then provide the full figure specs in the separate `figures` structured output array.
Each figure in the array must have an `id` matching a marker in the text. Set the `aspect_ratio`
field per figure: 21:9 for architecture / pipeline / flow-chart diagrams (the hero figure should
be one of these — place its marker near the END of the Introduction so it floats to the top of
page 2), 16:9 for comparisons / multi-panel results, 4:3 for dense charts, 1:1 for heatmaps /
confusion matrices / scatter plots.

Example in paper_text:
  "...our method achieves state-of-the-art results as shown below.\n\n[FIGURE:fig3]\n\nThe results demonstrate..."

Example in figures array (results comparison):
  {"id": "fig3", "title": "Performance Comparison", "caption": "Comparison of geometric mean query latency across optimizers.", "image_gen_detailed_description": "Grouped bar chart. X-axis: model names. Y-axis: latency (seconds, 0-5). Values: PostgreSQL=4.6s (red), Bao=2.8s (blue), RLQOpt=2.0s (green). Error bars +/-0.3-0.8. Sans-serif font, white background.", "aspect_ratio": "16:9", "summary": "Compares latency across optimizers"}

Example in figures array (architecture diagram, hero):
  {"id": "fig1", "title": "System Architecture", "caption": "End-to-end pipeline: encoder feeds latents into the planner, which queries the value head before emitting actions.", "image_gen_detailed_description": "Horizontal flow diagram, left to right. Five labeled boxes: 'Input' (gray), 'Encoder' (blue), 'Latent (z, 256-dim)' (light blue, narrow), 'Planner' (green), 'Action Head' (orange). Arrows labeled with shapes. Value head as separate green box below 'Planner', bidirectional arrow. Sans-serif font, clean white background, no 3D.", "aspect_ratio": "21:9", "summary": "Hero architecture diagram"}

CRITICAL: Before writing figure specs, look through artifact workspace output files (*_out.json)
and code to find ALL the exact values. The figure generator cannot read files — every exact number
and value MUST be in the image_gen_detailed_description.
</figure_instructions>

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

<todos>
TODO 1. Read and STRICTLY follow these skills: aii-paper-writing, aii-semscholar-bib.
TODO 2. LITERATURE REVIEW: Use web search tools to research the landscape — search key terms from
<hypothesis> and <all_artifacts>. Then use aii_semscholar_bib__fetch to batch-fetch real
BibTeX entries. Build a comprehensive Related Work section. Do NOT fabricate entries.
TODO 3. READ ARTIFACTS: Before writing each section, READ the relevant artifact source code, output
files, and data in the workspace. Extract concrete implementation details, technical innovations,
algorithmic specifics, and quantitative results. Do NOT write surface-level descriptions.

ARTIFACT REFERENCES: When you reference results, methodology, or findings from a specific artifact,
place an [ARTIFACT:artifact_id] marker inline. These become footnotes linking to the artifact's code
in the GitHub repository (first mention gets a footnote with URL, subsequent mentions are omitted).
Use the exact artifact ID from <all_artifacts>. Place the marker right after the claim it supports.
Example:
  "Our evaluation showed a 15% improvement over baselines [ARTIFACT:art_4f9d2c81ab37]." 
TODO 4. WRITE PAPER: Write the full paper text with [FIGURE:fig_id] markers per <figure_instructions>,
and provide the figure specs in the figures array. Cite with numeric references [1], [2], etc.
At the end of the paper text, include a full bibliography section. Do NOT compile LaTeX or generate
actual image/figure files. Your ONLY output is the structured JSON.
</todos><user_data>
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
    "FigureSpec": {
      "description": "Figure specification \u2014 structured output from paper writing agent.\n\nThe LLM fills these as a list in PaperText.figures.\nLater converted to Figure objects for viz gen.",
      "properties": {
        "id": {
          "description": "Figure ID matching the [FIGURE:id] marker in paper_text (e.g., 'fig1')",
          "title": "Id",
          "type": "string"
        },
        "title": {
          "description": "Figure title in plain, everyday language \u2014 short and jargon-free. Aim for about 4-8 words (~40 characters).",
          "title": "Title",
          "type": "string"
        },
        "caption": {
          "description": "LaTeX figure caption \u2014 appears below the figure in the paper. Should describe what the figure shows and highlight key takeaways.",
          "title": "Caption",
          "type": "string"
        },
        "image_gen_detailed_description": {
          "description": "Detailed image generation prompt \u2014 axes, labels, ALL numeric values, colors, aspect ratio, layout. The image generator cannot read files; this is its ONLY input.",
          "title": "Image Gen Detailed Description",
          "type": "string"
        },
        "summary": {
          "description": "Brief summary of what this figure communicates",
          "title": "Summary",
          "type": "string"
        }
      },
      "required": [
        "id",
        "title",
        "caption",
        "image_gen_detailed_description",
        "summary"
      ],
      "title": "FigureSpec",
      "type": "object"
    }
  },
  "description": "Paper text \u2014 structured output from paper writing agent.\n\nStructured output fields (LLMPrompt + LLMStructOut):\n- title, abstract, paper_text, figures, summary\n\npaper_text contains [FIGURE:fig_id] markers for positioning.\nfigures contains the full specs as structured objects.\n\nMetadata fields (plain, set by pipeline code):\n- id",
  "properties": {
    "title": {
      "description": "Paper title \u2014 clear, plain-language, and short so a non-expert understands the main contribution at a glance. Aim for about 6-10 words; avoid jargon and acronyms.",
      "title": "Title",
      "type": "string"
    },
    "abstract": {
      "description": "Paper abstract",
      "title": "Abstract",
      "type": "string"
    },
    "paper_text": {
      "description": "Full paper body text with markdown section headers (# Introduction, # Methods, # Results, # Discussion, # Conclusion). Use [FIGURE:fig_id] markers (e.g. [FIGURE:fig1]) to indicate where each figure should appear.",
      "title": "Paper Text",
      "type": "string"
    },
    "figures": {
      "description": "List of figure specifications. Each must have an id matching a [FIGURE:id] marker in paper_text.",
      "items": {
        "$ref": "#/$defs/FigureSpec"
      },
      "title": "Figures",
      "type": "array"
    },
    "summary": {
      "description": "Brief summary of the paper's main contribution and findings",
      "title": "Summary",
      "type": "string"
    }
  },
  "required": [
    "title",
    "abstract",
    "paper_text",
    "summary"
  ],
  "title": "PaperText",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-07-17 12:09:25 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [3] SKILL-INPUT — aii-paper-writing · 2026-07-17 12:09:38 UTC

The agent loaded the **aii-paper-writing** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-paper-writing
description: Academic paper writing guidance for AI research. Covers paper structure, figure placeholders, bibliography building with Semantic Scholar, and citation rules. Does NOT cover LaTeX compilation or figure file generation — see aii-paper-to-latex for that.
---

## Technical Papers

Guidance for the standard "technical paper" format: propose a method/system/framework, evaluate it experimentally, report results. This is the main track at most CS venues (NeurIPS, ICML, ICLR, ACL, AAAI, etc.). Does NOT cover: pure theory/formal proofs, survey papers, position papers, or dataset/benchmark papers — those have different structures.

### Paper Structure

Target 6-8 pages. Use formal academic language, third person. Support claims with evidence from artifacts.

#### Rough Page Budget (8-page paper)

| Section | Pages | Notes |
|---|---|---|
| Abstract | 0.3 | Problem, approach, key result |
| Introduction | 1.0-1.5 | The most important section |
| Related Work | 0.5-1.0 | Beginning or end (see below) |
| Methods | 1.5-2.0 | Architecture fig on page 1 |
| Experiments | 1.5-2.0 | Setup + results + ablations |
| Discussion | 0.5-1.0 | Limitations go here |
| Conclusion | 0.3-0.5 | Do not repeat the abstract |
| References | 0.5-1.0 | Not counted in page limit |

**Critical rule**: A clear new technical contribution must be articulated by page 3 (quarter of the paper). If the reader doesn't know what you did by then, you've lost them.

#### Section Details

**Abstract** (150-250 words): State the problem, your approach, and the main results. Be factual and comprehensive. Do not repeat the abstract word-for-word later in the paper.

**Introduction** — Follow this 5-paragraph structure:

1. **What is the problem?** Define the task concretely.
2. **Why is it interesting and important?** Real-world impact, scale.
3. **Why is it hard?** Why do naive approaches fail?
4. **Why hasn't it been solved before?** What's wrong with prior solutions? How does yours differ?
5. **What are the key components of your approach and results?** Include specific limitations.

End with a "Summary of Contributions" subsection — bullet list of contributions with section references. This doubles as an outline, saving space.

**Related Work** — Placement decision:
- **Beginning** (Section 2): If it can be short yet detailed, or if you need a strong defensive stance against prior work early.
- **End** (before Conclusions): If comparisons require your technical content, or if it can be summarized briefly in the Introduction. Can be titled "Discussion and Related Work."

**Methods/Approach**: Every section tells a story — the story of the results, NOT the story of how you arrived at them. Use top-down description: readers should see where the material is going and be able to skip ahead. Move gory details to appendices.

**Experiments**: Setup (datasets, metrics, baselines) → main results → ablations → analysis. Every claim needs quantitative evidence.

**Discussion**: Interpret results, compare to prior work, state limitations honestly. Limitations should be specific and actionable, not vague disclaimers.

**Conclusion**: Short summarizing paragraph. Do NOT repeat material from the Abstract or Introduction. Make original claims more concrete (e.g., reference quantitative results). Include future work as bullet list — if actively pursuing follow-up, say so to mark territory.

#### Writing Quality Rules

- Define all notation/terminology before use, only once. Group global definitions in Preliminaries.
- Do NOT use nonreferential "this", "that", "these", "it". Always specify the referent. BAD: "This is important because..." GOOD: "This accuracy gap is important because..."
- Do NOT use "etc." unless remaining items are completely obvious. BAD: "We measure volatility, scalability, etc." GOOD: "We measure volatility and scalability."
- Do NOT write "for various reasons" — state the actual reasons.
- "That" is defining, "which" is nondefining. "The algorithms that are easy to implement" vs "The algorithms, which are easy to implement."
- Use italics for definitions and quotes, not for emphasis. Context alone should provide emphasis.

### Figure Format

Figures use a hybrid marker + structured array approach. ALL figures are generated by a separate pipeline step using an AI image model — your `image_gen_detailed_description` is the ONLY input that model sees. It cannot read files or access data. Do NOT generate actual image files yourself (no matplotlib, no PIL, no image generation scripts).

**In paper_text**: Place `[FIGURE:fig_id]` markers where figures should appear.

**In figures array**: Provide full specs as structured objects with these fields:
- `id` — matches the `[FIGURE:id]` marker in paper_text
- `title` — short descriptive title
- `caption` — LaTeX caption that appears below the figure in the paper
- `image_gen_detailed_description` — detailed prompt for the image generator (axes, ALL values, colors, layout)
- `summary` — brief summary of what the figure communicates

Example in paper_text:
```
...our method achieves state-of-the-art results as shown below.

[FIGURE:fig_1]

The results in Figure 1 demonstrate...
```

Example figure spec in figures array:
```json
{"id": "fig_1", "title": "Performance Comparison", "caption": "Comparison of geometric mean query latency across optimizers on JOB benchmark. RLQOpt achieves 2.3x speedup over PostgreSQL.", "image_gen_detailed_description": "Grouped bar chart. X-axis: model names. Y-axis: accuracy (0.0-1.0). Values: ModelA=0.847, ModelB=0.762, Baseline=0.531. Error bars with std: 0.02, 0.03, 0.05. Sans-serif font, white background.", "summary": "Compares accuracy of proposed methods vs baseline."}
```

Every marker in text MUST have a matching figure in the array, and vice versa.

#### Data Precision Requirement

`image_gen_detailed_description` MUST include exact numbers from artifact output files. Read the actual output files before writing figure specs.

- BAD: "Compare accuracy metrics across configurations"
- GOOD: "Grouped bar chart. X-axis: model names. Y-axis: accuracy (0.0-1.0). Values: K=3: 0.765, K=5: 0.729, Baseline: 0.121."

#### Figure vs Table Decision

Do NOT create figures for tabular data (rows/columns of text or numbers). Use `\begin{table}` in LaTeX instead. Figures are for actual visualizations only (charts, plots, diagrams).

#### Figure Placement Strategy

Be intentional with figure ordering. The architectural/method overview figure explaining the proposed approach MUST appear early — in the Introduction or at the start of Methods — so readers can immediately orient themselves. Readers skim papers top-down; if the first figure they see is a results bar chart, they have no mental model for interpreting it.

Recommended ordering:
1. **Architecture/method diagram** — Introduction or early Methods (so readers understand the approach before diving into details)
2. **Conceptual/analogy figures** — Introduction or Methods (to build intuition)
3. **Results figures** (bar charts, line plots, scatter plots) — Results section
4. **Analysis/ablation figures** — Discussion or later Results

#### Guidelines

- Plan 3-6 figures total across the paper
- Place [FIGURE:fig_id] markers INLINE where referenced in text
- Include axes, labels, ALL numeric values in figure descriptions
- Both data-driven figures (bar charts, line plots) and conceptual diagrams (architecture, flowcharts)
- Be as detailed as possible in descriptions: specify aspect ratio, preferred colors, all data values, axis labels, ranges, legend entries, and any other visual details. The more specific the description, the better the generated figure

### Bibliography with Semantic Scholar

Build `./references.bib` using the aii-semscholar-bib skill (real BibTeX from Semantic Scholar):

1. Collect DOIs, ArXiv IDs, or titles for all papers you need to cite
2. Call `aii_semscholar_bib__fetch` with the full list in one batch
3. Write the returned .bib text into `./references.bib`

Rules:
- Do NOT fabricate BibTeX entries — always fetch from Semantic Scholar
- If a paper isn't found (very recent preprint), write the entry manually as fallback
- Use `\bibliography{references}` and `\bibliographystyle{plainnat}`
- Do NOT use inline `thebibliography` environment

### Citation Format (for Research Artifacts)

When writing research with numbered citations:

1. Every factual claim MUST have a numbered citation: `[1]`, `[2]`, `[1, 3]`, etc.
2. Each source in the "sources" array MUST have an "index" field
3. The index MUST EXACTLY MATCH citation numbers in the text
4. NEVER cite a number without a matching source index
5. Example: "LLMs show 40% improvement with multi-agent collaboration [1]."
````

### [4] SKILL-INPUT — aii-semscholar-bib · 2026-07-17 12:09:38 UTC

The agent loaded the **aii-semscholar-bib** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-semscholar-bib
description: Build bibliographies using Semantic Scholar. Batch-fetch BibTeX for papers by DOI, ArXiv ID, or title. Use when writing papers, generating reference lists, or building .bib files.
---

## Tool: `aii_semscholar_bib__fetch`

Batch-fetch BibTeX entries from Semantic Scholar. Pass all references in a single call — the tool handles batching internally.

### How it works

1. **DOI/ArXiv refs** → batched into POST /paper/batch calls (up to 500 per API call, auto-chunked)
2. **Title-only refs** → individual GET /paper/search/match (1s delay between)
3. **Post-process** → fix entry type, fix citation key (AuthorYYYY), inject DOI

The ability server runs a single worker (`max_threads: 1`). Multiple concurrent tool calls are queued — each runs independently (no cross-request aggregation). Batching happens within each request.

### Input format

```json
{
  "references": [
    {"doi": "10.48550/arXiv.1706.03762", "author": "Vaswani", "year": 2017},
    {"arxiv": "2201.11903", "author": "Wei", "year": 2022},
    {"title": "Tree of Thoughts", "author": "Yao", "year": 2023}
  ]
}
```

Each reference object can have:
- `doi` — DOI string (ArXiv DOIs like `10.48550/arXiv.XXXX.XXXXX` auto-convert to ArXiv IDs)
- `arxiv` — ArXiv ID (e.g. `"2305.14325"`)
- `title` — Paper title (used for search/match when no DOI/ArXiv)
- `author` — First author last name (for cleaner citation key)
- `year` — Publication year (int, for citation key)

At least one of `doi`, `arxiv`, or `title` is required per reference.

### Output format

```json
{
  "success": true,
  "bib_text": "@inproceedings{Vaswani2017, ...}\n\n@article{Wei2022, ...}",
  "total": 3,
  "found": 3,
  "failed_count": 0,
  "entries": [{"citation_key": "Vaswani2017", "bibtex": "...", "title": "...", "doi": "...", "arxiv": ""}],
  "failed": []
}
```

### Workflow

1. Collect DOIs, ArXiv IDs, or titles for all papers you need to cite
2. Call `aii_semscholar_bib__fetch` with the full list in **one call**
3. Save `bib_text` from the response to your `references.bib` file
4. Check `failed` — for any missed papers, follow the **fallback procedure** below

### Fallback for failed references (MANDATORY)

NEVER fabricate BibTeX. For each failed reference:
1. **WebSearch** for `"Title" author year` (try `site:arxiv.org` too)
2. **WebFetch** the paper page → extract title, authors, year, venue, DOI/ArXiv ID
3. If DOI/ArXiv found → retry `aii_semscholar_bib__fetch` with it
4. Last resort: write BibTeX by hand using **only verified info from the actual paper page**

---

### CLI (for manual use / debugging)

```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-semscholar-bib" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_semscholar_bib__fetch.py --refs '[
  {"doi": "10.48550/arXiv.1706.03762", "author": "Vaswani", "year": 2017},
  {"arxiv": "2201.11903", "author": "Wei", "year": 2022},
  {"title": "Tree of Thoughts", "author": "Yao", "year": 2023}
]'
```

`--json, -j` — output raw JSON instead of .bib text

**If the script fails** with a connection error (ability server not running): create a local `.venv`, install server deps from `server_requirements.txt` into it, then import the `@aii_ability` function from the script and call it directly — bypassing the server:
```bash
uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python -r "$SKILL_DIR/scripts/server_requirements.txt"
```
````
