# gen_paper_text — test_idea

> Phase: `invention_loop` · round 3 · `gen_paper_text`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_paper_text` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-07-17 13:23:07 UTC

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
</previous_paper>

<reviewer_feedback>
STEP 1 — REVIEW: A reviewer evaluated the previous paper draft above and produced this feedback.

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

<all_artifacts>
FULL EVIDENCE BASE: All 4 research artifacts across all iterations.

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
</all_artifacts>

<new_artifacts_this_iteration>
NEW THIS ITERATION: These 1 artifacts were created to address the reviewer
feedback. Their findings should be the primary basis for your revisions.

title: SAE Feature Reversal Score Null-Baseline Validation
type: experiment
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
id: art_Qnl1zSSZPJvj
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

### [2] HUMAN-USER prompt · 2026-07-17 13:23:07 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [3] SKILL-INPUT — aii-paper-writing · 2026-07-17 13:23:13 UTC

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

### [4] SKILL-INPUT — aii-semscholar-bib · 2026-07-17 13:23:13 UTC

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

### [5] SKILL-INPUT — aii-web-tools · 2026-07-17 13:23:47 UTC

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
