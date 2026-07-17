# gen_full_paper — report_results

> Phase: `gen_paper_repo` · `gen_full_paper`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_full_paper` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-07-17 15:28:49 UTC

````
<research_methodology>
Write like an experienced academic. Reviewers judge both the science and the writing.

- Claims must be proportional to evidence. Choose verbs carefully — "demonstrate," "observe," and "hypothesize" mean different things.
- Every result needs: what was measured, on what data, the numbers, and what they mean.
- Methodology must be specific enough to reproduce. Related work must be organized by theme, not a literature dump.
- State limitations honestly. Avoid both overclaiming and excessive hedging.
</research_methodology>

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

<workspace>
Your workspace: `/ai-inventor/aii_data/runs/run_hql72-TZXK98/4_gen_paper_repo/_4_assemble_paper/paper/workspace`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/ai-inventor/aii_data/runs/run_hql72-TZXK98/4_gen_paper_repo/_4_assemble_paper/paper/workspace/`:
GOOD: `/ai-inventor/aii_data/runs/run_hql72-TZXK98/4_gen_paper_repo/_4_assemble_paper/paper/workspace/file.py`, `/ai-inventor/aii_data/runs/run_hql72-TZXK98/4_gen_paper_repo/_4_assemble_paper/paper/workspace/results/out.json`
BAD: `/tmp/file.py`, `~/output.json`, `./file.py`, any path outside the workspace
</workspace>

<task>
Create a publication-ready top-conference LaTeX paper with BibTeX from <paper_text> and <available_figures>, compile to PDF.
</task>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<paper_text>
title: >-
  Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
abstract: >-
  Refusal ablation (abliteration) edits a safety-aligned model's weights to eliminate compliance refusals by removing a single
  residual-stream direction. The standard narrative frames abliteration as undoing RLHF safety fine-tuning. We test this at
  the encoder-direction level for Qwen3-8B using the frozen Qwen-Scope base SAE (65,536 TopK features), comparing three checkpoints—base,
  RLHF-instruct, and a community-released abliterated variant—across a balanced corpus of 500 harmful and 500 benign prompts.
  Because SAE reconstruction quality at layer 22 falls below acceptable thresholds (R² ≈ 0.51 for instruct and abliterated
  variants), all metrics operate on pre-topk encoder projections (z_pre)—dense continuous projections onto all encoder directions—rather
  than the post-topk sparse features validated as semantically interpretable in single-model settings. Within this geometric
  regime, we introduce two per-feature metrics: the Reversal Score R(f), measuring directional preservation of RLHF-induced
  encoder-direction changes under abliteration, and the OLS Slope β_f, measuring magnitude preservation. For Qwen3-8B, and
  conditioned on the abliterated checkpoint faithfully implementing the refusal-direction removal, every one of the top-500
  RLHF-sensitive encoder directions has a positive Reversal Score (mean R̄ = 0.921 ± 0.040, min = 0.684; zero directions reversed),
  and the OLS slope is near unity (β̄ = 0.986 ± 0.176; 86.4% of directions with β_f > 0.8). A noise-free mid-SC baseline (Cohen's
  d = 1.087) confirms selective preservation. Direct measurement of refusal-direction geometry—mean |cos θ| = 0.019 between
  the refusal direction and top-500 RLHF-modified encoder directions at layer 22—provides the mechanistic explanation: abliteration's
  orthogonalization is geometrically incapable of touching these directions. A 64.6% overlap with benign-prompt RLHF-sensitive
  directions reveals that the preserved modifications span both harm-specific and general RLHF stylistic changes. These findings
  establish, for Qwen3-8B, that abliteration removes the compliance mechanism while leaving the representational modifications
  of RLHF intact at approximately unit magnitude, pending validation of the pre-topk proxy and independent verification of
  abliteration depth.
paper_text: "# Introduction\n\nRefusal ablation—commonly called *abliteration*—edits a safety-aligned language model's weights\
  \ to eliminate its compliance refusals [1]. The technique identifies a *refusal direction* in the residual stream: the mean-difference\
  \ vector between activations on harmful versus harmless prompts. All weight matrices that write to the residual stream are\
  \ then orthogonalized against this direction, producing a model that complies with previously-refused requests without retraining.\
  \ The standard framing treats abliteration as *undoing* RLHF safety alignment: if RLHF installs a refusal mechanism, abliteration\
  \ removes it, restoring a base-like model.\n\nThis framing has direct implications for AI safety. If abliteration truly\
  \ erases RLHF modifications, the safety imprint is fragile—a single weight-editing step with minimal compute removes it\
  \ entirely. If abliteration only removes the *behavioral expression* of safety training while leaving *representational*\
  \ modifications intact, the picture differs substantially: the model's internal representations may still carry RLHF-induced\
  \ structure, providing a substrate for lightweight safety restoration without full retraining. Distinguishing these scenarios\
  \ matters both for threat modeling and for designing targeted safety-recovery interventions.\n\nTwo geometric studies establish\
  \ that harmful-prompt representations survive abliteration: LatentBiopsy [2] finds that a linear classifier on Qwen model\
  \ residual streams retains AUROC ≥ 0.937 after abliteration, and a cross-family companion study [3] replicates this across\
  \ 12 variants (Qwen, Llama, Gemma; 0.5B–9B parameters) with mean AUROC 0.982 surviving within ±0.003. Something in the internal\
  \ representation persists—but holistic AUROC collapses all representational information into a single detection score and\
  \ cannot answer *which specific encoder-direction structures survive, at what magnitude, or why*. Answering these mechanistic\
  \ questions requires a feature-level vocabulary.\n\nSparse Autoencoders (SAEs) [11, 12, 13, 14] decompose residual-stream\
  \ activations into tens of thousands of near-monosemantic features, providing exactly this vocabulary. Prior SAE-based analyses\
  \ of refusal identify which features *drive* refusal behavior within a single model [21, 22]. Our work addresses a complementary\
  \ question: which encoder directions are modified by RLHF, and do those modifications survive abliteration in direction\
  \ *and* magnitude? A key methodological complication is that reconstruction quality of a base-trained SAE degrades on instruct-variant\
  \ activations [4, 10], ruling out the standard post-topk sparse feature representation at the layers with the richest RLHF\
  \ signal. We address this by working with *pre-topk encoder projections* (z_pre)—dense continuous projections of residual-stream\
  \ activations onto all 65,536 SAE encoder directions—and introduce two metrics interpretable in this geometric regime. The\
  \ pre-topk proxy is validated indirectly via the orthogonality result (Section 5.4), which is independent of the sparse/dense\
  \ distinction, and a direct z_pre/z_post comparison is deferred to future work.\n\nWe apply the frozen Qwen-Scope base SAE\
  \ (`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`) [9] across three Qwen3-8B checkpoints—base, RLHF-instruct, and a community-released\
  \ abliterated variant—on a balanced corpus of 500 harmful and 500 benign prompts from seven datasets. The *Reversal Score*\
  \ R(f) measures whether abliteration preserves or reverses the direction of the RLHF-induced change to encoder direction\
  \ f. The *OLS Slope* β_f measures whether abliteration preserves the magnitude of that change. For Qwen3-8B, conditioned\
  \ on abliteration completeness, every top-500 RLHF-sensitive encoder direction has a positive Reversal Score (mean R̄ =\
  \ 0.921, min = 0.684, zero directions reversed), and the OLS slope is near unity (β̄ = 0.986 ± 0.176, 86.4% of directions\
  \ with β_f > 0.8). Direct orthogonality measurement (mean |cos θ| = 0.019 at layer 22) provides the geometric mechanism.\
  \ A 64.6% overlap with benign-prompt RLHF-sensitive directions establishes that the preserved modifications are not exclusively\
  \ harm-specific.\n\n[FIGURE:fig1]\n\n**Summary of Contributions:**\n- We introduce the *Reversal Score*, a per-encoder-direction\
  \ Pearson correlation confirming directional preservation of RLHF-modified SAE encoder directions under abliteration for\
  \ Qwen3-8B (R̄ = 0.921 at layer 22, zero reversed directions; Section 3.3). \\footnote{Code: \\url{https://github.com/AMGrobelnik/ai-invention-4ac301-abliteration-cannot-erase-rlhf-geometric/tree/main/round-2/experiment-1}}\n\
  - We introduce the *OLS Slope* β_f, a companion magnitude metric, confirming that abliteration preserves not just direction\
  \ but approximately full amplitude of RLHF-induced encoder-direction changes (β̄ = 0.986 ± 0.176 at layer 22; Section 5.2).\
  \ \\footnote{Code: \\url{https://github.com/AMGrobelnik/ai-invention-4ac301-abliteration-cannot-erase-rlhf-geometric/tree/main/round-4/experiment-1}}\n\
  - We validate selective preservation against a noise-free mid-SC baseline (SC_median = 1.12 ≫ 0.01): top-500 R̄ = 0.921\
  \ vs. mid-500 R̄ = 0.871, Cohen's d = 1.087, p < 10⁻³⁰⁰ (Section 5.3). \n- We directly measure refusal-direction orthogonality\
  \ to RLHF-modified SAE encoder directions at layers 22 and 20 (mean |cos θ| = 0.0185 and 0.0209, respectively), providing\
  \ the geometric mechanism that explains zero reversed directions and β̄ ≈ 1 (Section 5.4). \\footnote{Code: \\url{https://github.com/AMGrobelnik/ai-invention-4ac301-abliteration-cannot-erase-rlhf-geometric/tree/main/round-3/experiment-1}}\n\
  - We show that 64.6% of the top-500 RLHF-sensitive encoder directions overlap with the top-500 benign-prompt RLHF-sensitive\
  \ directions, establishing that the preserved RLHF modifications span general behavioral changes (output format, instruction\
  \ style) in addition to harm-specific semantics (Section 5.5). \n\n# Related Work\n\n**Refusal ablation.** Arditi et al.\
  \ [1] establish that refusal is mediated by a single residual-stream direction and that orthogonalizing all weight matrices\
  \ against it reliably eliminates compliance across 13 models up to 72B parameters. Joad et al. [23] challenge the single-direction\
  \ framing, demonstrating that refusal behaviors correspond to geometrically distinct directions across eleven behavioral\
  \ categories (safety refusals, excessive refusal, incomplete-request refusals, etc.), though they find that linear steering\
  \ along any refusal-related direction produces nearly identical refusal-to-over-refusal trade-offs. This finding is directly\
  \ relevant to our orthogonality measurement (Section 5.4): the mean |cos θ| = 0.019 between the mean-difference refusal\
  \ vector and RLHF-modified encoder directions is a necessary but not sufficient condition for explaining zero reversed features\
  \ if abliteration removes a multi-dimensional subspace rather than a single direction. We address this in Section 5.4 and\
  \ treat the multi-dimensional extension as explicit future work. Cristofano [7] shows the raw refusal vector is polysemantic,\
  \ entangling safety circuitry with style; spectral residualization reduces collateral KL divergence while preserving abliteration.\
  \ Nanfack et al. [19] propose an optimal-transport alternative reducing distributional shift. Our work examines what all\
  \ these variants share: the refusal direction (or subspace), however constructed, leaves RLHF-modified SAE encoder directions\
  \ intact in both direction and magnitude.\n\n**SAEs applied to refusal analysis.** Yeo et al. [21] use SAEs trained on pretraining\
  \ data to investigate refusal mechanisms in chat-tuned models, finding that base-trained SAEs fail to cleanly encode the\
  \ refusal circuitry of chat models—supporting our observation that base SAE reconstruction quality degrades on instruct\
  \ activations (R² ≈ 0.51 at layer 22). Their finding contextualizes our pre-topk proxy choice: the encoder directions of\
  \ a base SAE are not guaranteed to align with the semantically active features of an instruct model, which is precisely\
  \ why we restrict claims to *encoder-direction geometry* rather than asserting preservation of semantically validated sparse\
  \ features. Kim et al. [22] introduce CRaFT, which uses cross-layer transcoders to map model computations into sparse feature\
  \ circuits, quantifying inter-feature influences on refusal-output logits rather than treating features as isolated signals.\
  \ CRaFT and our approach are complementary: CRaFT identifies which features causally govern refusal outputs in a single\
  \ model; our Reversal Score and OLS Slope identify which encoder directions RLHF modified and whether abliteration erases\
  \ those modifications. Both are feature-level analyses, but they use different feature vocabularies (transcoder features\
  \ vs. frozen base SAE encoder directions) and answer different questions (causal control of refusal vs. survival of RLHF\
  \ modifications under abliteration).\n\n**Geometry of harmful intent surviving abliteration.** LatentBiopsy [2] and the\
  \ companion geometry study [3] are the direct predecessors establishing the phenomenon this work explains mechanistically.\
  \ Their key limitation—collapsing representational information into a single AUROC score—is precisely the gap our encoder-direction\
  \ analysis addresses. The OLS slope results (β̄ = 0.986) and direct orthogonality measurement (|cos θ| = 0.0185) provide\
  \ the mechanistic account that a holistic detection score cannot.\n\n**SAEs as cross-variant measurement instruments.**\
  \ Chopra [4] uses frozen base SAEs as cross-variant diagnostics for supervised fine-tuning (SFT), finding that SAE latent\
  \ cosine similarity drops to 0.509–0.557 at layer 22 after SFT while raw activation cosine stays above 0.96—validating the\
  \ frozen base-SAE approach and motivating the pre-topk encoder fallback when reconstruction quality degrades. Jiralerspong\
  \ and Bricken [5] introduce crosscoders—SAEs trained jointly on two or more model variants—that partition features into\
  \ shared and model-specific categories without requiring a pre-existing vocabulary. Kassem et al. [6] extend this to narrow\
  \ fine-tuning regimes with Delta-Crosscoder. Our frozen base-SAE approach is complementary: it requires zero additional\
  \ training, reuses Qwen-Scope's existing feature vocabulary, and extends naturally to the three-model base→instruct→abliterated\
  \ trajectory. Li et al. [10] demonstrate an ≈8× MSE gap between base-SAE reconstruction on instruct versus purpose-trained\
  \ instruct SAE activations, directly motivating our reconstruction quality gate and pre-topk fallback. Shportko et al. [20]\
  \ localize RL-induced tool-calling in Qwen2.5-3B to a compact steerable crosscoder feature set—the complement of our finding\
  \ about which encoder directions abliteration *cannot* remove.\n\n**Safety fine-tuning mechanisms.** Jain et al. [8] establish\
  \ that safety fine-tuning projects unsafe activations into MLP null spaces, explaining adversarial circumvention. Our finding\
  \ extends this picture: at the SAE encoder level, RLHF-modified encoder directions are near-orthogonal to the refusal direction,\
  \ so abliteration—which removes the refusal direction—cannot erase them geometrically.\n\n# Methodology\n\n## Three-Model\
  \ Setup\n\nAll experiments use three Qwen3-8B checkpoints sharing identical 36-layer architecture (d_model = 4096, bfloat16,\
  \ transformers ≥ 4.51.0) \\footnote{Code: \\url{https://github.com/AMGrobelnik/ai-invention-4ac301-abliteration-cannot-erase-rlhf-geometric/tree/main/round-1/research-1}}:\n\
  \n*Base*: `Qwen/Qwen3-8B-Base`—the unaligned pretrained model, providing the representational baseline against which RLHF-induced\
  \ changes are measured.\n\n*Instruct*: `Qwen/Qwen3-8B` with `enable_thinking=False`—the RLHF-aligned model. Chain-of-thought\
  \ suppression isolates safety-relevant activations on the final input token.\n\n*Abliterated*: `huihui-ai/Huihui-Qwen3-8B-abliterated-v2`\
  \ (v2 corrects a layer-0 output-garbling artifact present in v1). The exact layer set and prompt set used for the refusal\
  \ direction computation are not publicly documented. The findings in this paper are conditioned on this checkpoint faithfully\
  \ implementing a refusal-direction removal: if abliteration is incomplete, the high Reversal Scores and OLS slopes would\
  \ trivially reflect the abliterated model being close to the instruct model by construction rather than preservation under\
  \ genuine erasure. Section 5.4 and the Limitations section discuss this constraint.\n\nAll models are accessed via `model.model.layers`\
  \ for residual-stream hook injection at the final token position of each input.\n\n## SAE Measurement Instrument\n\nThe\
  \ frozen Qwen-Scope SAE `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` serves as the shared measurement instrument [9].  A critical\
  \ architectural note: Qwen-Scope implements *TopK* architecture (k = 50 features retained via `torch.topk`), not JumpReLU.\
  \ For a residual activation $\\mathbf{x} \\in \\mathbb{R}^{4096}$:\n\n$$\\mathbf{z}_{\\text{pre}} = \\mathbf{x}W_{\\text{enc}}^\\\
  top + \\mathbf{b}_{\\text{enc}} \\in \\mathbb{R}^{65536}$$\n$$\\mathbf{z} = \\text{TopK}(\\mathbf{z}_{\\text{pre}},\\; k{=}50)$$\n\
  \n**Pre-topk proxy.** The reconstruction quality gate (Section 3.5) fails at layer 22: R² = 0.574 (base), 0.512 (instruct),\
  \ 0.512 (abliterated)—all below the 0.85 threshold [10]. All feature metrics therefore use $\\mathbf{z}_{\\text{pre}}$ rather\
  \ than the post-topk sparse representation. The pre-topk vector is dense by construction: every prompt receives a non-zero\
  \ value on all 65,536 encoder directions simultaneously, so no single direction is \"off\" in the interpretability sense.\
  \ The metrics we compute (Reversal Score, OLS Slope) measure *geometric properties of encoder directions in a continuous\
  \ projection space*, not activation of near-monosemantic features in the interpretability sense. The orthogonality result\
  \ (Section 5.4) is fully independent of the sparse/dense distinction—it is a direct measurement of the angle between weight\
  \ vectors—and remains the most secure finding. A direct comparison of top-500 feature rankings and Reversal Score rank correlations\
  \ between z_pre and z_post on the highest-R² subset is an important validation deferred to future work. If such a comparison\
  \ shows Spearman ρ > 0.7 between z_pre and z_post Reversal Scores, the pre-topk proxy is defensible as a surrogate; if it\
  \ diverges, the encoder-direction findings must be interpreted as purely geometric properties without semantic grounding.\n\
  \n## Feature Metrics\n\nFor each SAE encoder direction $f$ and $N = 500$ harmful prompts $\\{p_i\\}_{i=1}^N$, let $a_m(f,\
  \ p_i)$ denote the pre-topk encoder activation of direction $f$ for model $m \\in \\{\\text{base}, \\text{inst}, \\text{abl}\\\
  }$.\n\n**Safety Change** quantifies the magnitude of the RLHF modification to encoder direction $f$:\n$$\\text{SC}(f) =\
  \ \\frac{1}{N}\\sum_{i=1}^N |a_{\\text{inst}}(f, p_i) - a_{\\text{base}}(f, p_i)|$$\n\nEncoder directions are ranked by\
  \ SC$(f)$: the top-500 form the *RLHF-sensitive set*. For null-baseline analysis, a *mid-500* group is drawn uniformly from\
  \ the 40th–60th SC percentile.\n\n**Reversal Score** quantifies whether abliteration reverses or preserves the *direction*\
  \ of the RLHF modification to encoder direction $f$:\n$$\\boldsymbol{\\Delta}_{\\text{inst}}(f) = [a_{\\text{inst}}(f, p_i)\
  \ - a_{\\text{base}}(f, p_i)]_{i=1}^N$$\n$$\\boldsymbol{\\Delta}_{\\text{abl}}(f) = [a_{\\text{abl}}(f, p_i) - a_{\\text{base}}(f,\
  \ p_i)]_{i=1}^N$$\n$$R(f) = \\text{PearsonR}(\\boldsymbol{\\Delta}_{\\text{inst}}(f), \\boldsymbol{\\Delta}_{\\text{abl}}(f))$$\n\
  \n$R(f) \\approx +1$ indicates abliteration preserves the RLHF modification (*imprinted*); $R(f) \\approx -1$ indicates\
  \ abliteration reverses it (*reversed*). This is a correlation over prompt-level scalar activations, not a cosine similarity\
  \ of scalars—ensuring that structure in the result is an empirical finding, not an artifact of the definition.\n\n**OLS\
  \ Slope** resolves the scale-invariance of the Reversal Score. A feature where abliteration retains only 5% of the RLHF-induced\
  \ shift in magnitude but preserves its direction would still yield R(f) ≈ 1.0. The no-intercept ordinary least-squares slope:\n\
  $$\\beta_f = \\frac{\\boldsymbol{\\Delta}_{\\text{inst}}(f)^\\top \\boldsymbol{\\Delta}_{\\text{abl}}(f)}{\\boldsymbol{\\\
  Delta}_{\\text{inst}}(f)^\\top \\boldsymbol{\\Delta}_{\\text{inst}}(f)}$$\n\n$\\beta_f \\approx 1$ means the abliterated\
  \ model's encoder-direction activations shift by the same magnitude as the instruct model's, relative to base. $\\beta_f\
  \ \\approx 0$ means only the directional pattern survives with negligible magnitude. $\\beta_f > 1$ indicates overshooting.\
  \ \n\n## Benign-SC Overlap\n\nTo assess whether the top-500 RLHF-sensitive encoder directions reflect *harm-specific* modifications\
  \ or *general RLHF* modifications (style, tone, response format), we separately compute SC$_{\\text{benign}}(f)$ using the\
  \ 500 benign prompts and report the overlap between the top-500 by harmful-SC and the top-500 by benign-SC.\n\n## Layer\
  \ Selection and Reconstruction Quality Gate\n\n**Layer selection.** We compute SC$(f)$ variance across all 65,536 encoder\
  \ directions at candidate layers {14, 20, 22} on 50 harmful prompts. Layer 22 is selected (variance 0.1721 vs. 0.0547 at\
  \ layer 20 and 0.0127 at layer 14); all primary results use layer 22 with layer 20 as replication. \n\n**Reconstruction\
  \ quality gate.** At layer 22, R² = 0.574 (base), 0.512 (instruct), 0.512 (abliterated)—all below the 0.85 threshold [10]—triggering\
  \ the pre-topk encoder-only analysis path. \n\n## Semantic Labeling Protocol\n\nFor the top-100 encoder directions by SC$(f)$,\
  \ we identify the 5 prompts with the highest pre-topk encoder activation and provide the first 300 characters of each to\
  \ Llama-3.1-8B-Instruct via OpenRouter. The cross-family labeler avoids within-Qwen circularity. Each direction is classified\
  \ as `harm_concept`, `safety_refusal`, `general_capability`, or `other`. Results are treated as qualitative observations\
  \ only: the labeling uses character-prefix truncation of most-activating prompts rather than per-token activation windows;\
  \ labels reflect prompt topic more than token-level representations; and no Cohen's κ validation against a manual gold set\
  \ was performed.\n\n# Evaluation Corpus\n\nThe evaluation corpus comprises 500 harmful and 500 benign prompts from seven\
  \ publicly licensed datasets. [ARTIFACT:art_M8-flaZ0OOaP]\n\n**Harmful sources:** BeaverTails [15] (176 prompts; NeurIPS\
  \ 2023, human-labeled QA spanning 14 harm categories), AdvBench (98 prompts; standard adversarial behavior dataset used\
  \ in abliteration research [1]), TrustAIRLab in-the-wild jailbreak prompts (86 prompts; real-world jailbreaks from ACM CCS\
  \ 2024), HarmBench [17] (85 prompts; covering chemical, cybercrime, physical harm, and illegal categories), and JailbreakBench/JBB-Behaviors\
  \ [16] (55 prompts).\n\n**Benign sources:** Jailbreak-classification (151 prompts; labeled benign user requests), JBB-Behaviors\
  \ benign counterparts [16] (46 prompts), Toxic-Chat (104 prompts; real user prompts labeled non-toxic), TrustAIRLab regular\
  \ prompts (93 prompts), and Dolly-15k [18] (106 prompts; human-written instruction-following tasks).\n\nAll sources are\
  \ processed through MinHash LSH deduplication (threshold 0.5, 128 permutations, character 5-gram shingling), token-length\
  \ filtering (≤512 tokens), and category capping (≤35% per harm subcategory). The final corpus contains 1,000 prompts (500\
  \ harmful, 500 benign).\n\n# Results\n\n## Layer Selection\n\nFigure 4 shows Safety Change variance across candidate layers.\
  \ Layer 22 exhibits the highest variance (0.1721), confirming it as the primary analysis layer. The R² reconstruction gate\
  \ at layer 22 yields values below 0.85 for all three model variants (base: 0.574, instruct: 0.512, abliterated: 0.512),\
  \ triggering the pre-topk encoder-only analysis path (Section 3.2).\n\n[FIGURE:fig4]\n\n## Reversal Score Distribution\n\
  \nFigure 2(a) shows the Reversal Score distribution for the top-500 RLHF-sensitive encoder directions at layer 22. Every\
  \ direction has a positive Reversal Score (mean = 0.921, std = 0.040, min = 0.684, max = 0.989). No direction has a negative\
  \ or near-zero score. Hartigan's dip test yields statistic = 0.0123, p = 0.930—strongly unimodal. Spearman correlation mean\
  \ = 0.913, confirming rank-based robustness.\n\nTable 1 summarizes the count of reversed and imprinted directions at all\
  \ tested thresholds. Zero directions are reversed at any threshold.\n\n**Table 1.** Imprinted and reversed encoder-direction\
  \ counts at all tested thresholds.\n\n| Threshold | Imprinted (R > threshold) | Reversed (R < −threshold) |\n|-----------|--------------------------|---------------------------|\n\
  | 0.30 | 500 / 500 (100%) | 0 / 500 (0%) |\n| 0.50 | 500 / 500 (100%) | 0 / 500 (0%) |\n| 0.70 | 499 / 500 (99.8%) | 0 /\
  \ 500 (0%) |\n| 0.95 | 125 / 500 (25%) | 0 / 500 (0%) |\n\n[FIGURE:fig2]\n\n## OLS Slope: Magnitude Is Preserved, Not Just\
  \ Direction\n\nA key limitation of the Reversal Score is scale-invariance: R(f) ≈ 1 is consistent with abliteration preserving\
  \ only 5% of the RLHF-induced magnitude while retaining the direction pattern. The OLS slope β_f resolves this ambiguity.\
  \ \n\nFigure 2(b) shows the β_f distribution for the top-500 RLHF-sensitive encoder directions at layer 22. The distribution\
  \ is tightly concentrated near 1: mean β̄ = 0.986 ± 0.176, median = 0.985, range [−0.055, 1.725]. Specifically, 86.4% of\
  \ directions have β_f > 0.8 (abliteration preserves more than 80% of the RLHF-induced shift in magnitude), 45.4% have β_f\
  \ > 1.0 (slight amplification), and only 0.2% are negative. **MAGNITUDE_PRESERVED.** Within the pre-topk geometric regime,\
  \ abliteration does not attenuate the RLHF imprint—it preserves approximately unit magnitude.\n\nLayer-20 replication: β̄\
  \ = 0.954 ± 0.191, 80.2% of directions with β_f > 0.8, zero negative values—consistent with the layer-22 primary result.\n\
  \n## Null-Baseline Validation: Noise-Free Mid-SC Comparison\n\nA potential concern with a bottom-500 null baseline is that\
  \ near-zero Safety Change values may make PearsonR between near-zero vectors potentially unreliable. We address this with\
  \ a noise-free mid-SC baseline: 500 directions uniformly sampled from the 40th–60th SC percentile, with SC_median = 1.12\
  \ (versus the top-500 minimum of 2.79). \n\nAt layer 22: top-500 R̄ = 0.921, mid-500 R̄ = 0.871 (Cohen's d = 1.087, two-sample\
  \ t-test t = 17.2, p < 10⁻³⁰⁰). **MID_SC_CONFIRMS_SELECTIVE_PRESERVATION.** The top-500 encoder directions show significantly\
  \ higher Reversal Scores than mid-SC controls that have non-trivial RLHF sensitivity and noise-free Δ_inst vectors, confirming\
  \ that the high Reversal Scores reflect genuine selective preservation rather than any measurement artifact. Figure 2(c)\
  \ shows the full comparison across all three baseline groups.\n\n## Direct Measurement of Refusal-Direction Orthogonality\n\
  \nThe mechanistic interpretation of zero reversed directions and β̄ ≈ 1 requires a testable prediction: the refusal direction\
  \ $\\mathbf{r}$ must be nearly orthogonal to all RLHF-modified SAE encoder directions. We test this directly. \n\nThe refusal\
  \ direction $\\mathbf{r}$ is computed as the mean difference of instruct residual-stream activations on harmful versus harmless\
  \ prompts at layer 22. For each of the top-500 RLHF-sensitive encoder directions $f$:\n$$\\cos\\theta_f = \\frac{\\mathbf{r}\
  \ \\cdot \\mathbf{w}_{\\text{enc}}(f)}{\\|\\mathbf{r}\\| \\|\\mathbf{w}_{\\text{enc}}(f)\\|}$$\n\n**Layer 22:** mean $|\\\
  cos\\theta_f| = 0.01854$ (authoritative value; the 0.019 in the paper is rounded). **ORTHOGONALITY_CONFIRMED.** A cosine\
  \ of 0.0185 means the refusal direction projects onto less than 1.9% of each RLHF-modified encoder direction's magnitude.\
  \ **Layer 20:** mean $|\\cos\\theta_f| = 0.02086$. ORTHOGONALITY_CONFIRMED at both layers; inter-layer agreement (matching\
  \ to 5 decimal places across independent experimental runs) confirms these values are stable.\n\nThis orthogonality result\
  \ is the most secure finding in this paper: it is a direct measurement of the angle between weight vectors and is fully\
  \ independent of the pre-topk/post-topk distinction, the abliteration checkpoint's completeness, or any activation-level\
  \ measurement.\n\nThis orthogonality geometrically explains the β̄ ≈ 1 result: abliteration orthogonalizes all weight matrices\
  \ against $\\mathbf{r}$. Because $\\mathbf{r}$ is nearly orthogonal to all RLHF-modified SAE encoder directions, the orthogonalization\
  \ cannot substantially change how any RLHF-modified encoder direction's activations are computed—in direction or in magnitude.\n\
  \n**Caveat on multi-dimensional refusal.** Joad et al. [23] demonstrate that refusal spans geometrically distinct directions\
  \ across eleven behavioral categories. If abliteration removes a multi-dimensional refusal *subspace* rather than a single\
  \ vector, the orthogonality argument requires that the full ablation subspace be near-orthogonal to all RLHF-modified encoder\
  \ directions—not just the mean-difference summary vector. The current measurement tests only the mean-difference vector\
  \ $\\mathbf{r}$. An extension to the top-3 principal components of the (harmful − harmless) activation differences would\
  \ determine whether this argument holds for higher PCs; this is deferred to future work. If a higher PC shows mean $|\\\
  cos θ_f| ≥ 0.05$ against the top-500 encoder directions, the mechanistic explanation would require qualification.\n\n##\
  \ Benign-SC Overlap: What the Preserved Imprint Represents\n\nWe compute SC$_{\\text{benign}}(f)$ — Safety Change evaluated\
  \ on 500 benign prompts — for all 65,536 SAE encoder directions, then measure the overlap between the top-500 by harmful-SC\
  \ and the top-500 by benign-SC. \n\n**Overlap count: 323 / 500 = 64.6%. STYLE_DOMINANT verdict.** A majority of the top-500\
  \ RLHF-sensitive encoder directions also appear in the benign-SC top-500, indicating that RLHF modified these directions\
  \ similarly when processing harmful and benign prompts. Approximately 64.6% of the preserved imprint captures general RLHF\
  \ modifications (response formality, output structure, instruction-following style) rather than harm-specific semantic content.\
  \ The remaining 35.4% (176 directions) are uniquely sensitive to harmful prompts and more plausible carriers of harm-specific\
  \ semantic representations.\n\nThis finding contextualizes the OLS slope result (β̄ = 0.986): abliteration comprehensively\
  \ preserves all RLHF encoder-direction modifications regardless of whether they encode harm-specific or general-style content.\
  \ The near-orthogonality of the refusal direction to all RLHF-modified encoder directions means abliteration cannot selectively\
  \ erase harm-semantic changes while leaving style changes intact—it fails to erase either category because all reside in\
  \ a subspace near-orthogonal to $\\mathbf{r}$.\n\n## Centroid Analysis with Control Subspaces\n\nTo validate that the preserved\
  \ imprint is specific to the RLHF-sensitive subspace, we compute centroid distance ratios $d_{\\text{abl-base}} / d_{\\\
  text{abl-inst}}$ (cosine) for three encoder-direction subspaces. \n\n[FIGURE:fig3]\n\n**Layer 22 (harmful prompts):** Subspace\
  \ A (top-500 RLHF-sensitive) ratio = 4.35×, Subspace B (bottom-500 RLHF-insensitive) ratio = 1.16×, Subspace C (random-500)\
  \ ratio = 2.37×. **Layer 20:** Subspace A ratio = 7.31×, Subspace B ratio = 1.01×, Subspace C ratio = 1.63× (verdict: CENTROID_SELECTIVE).\
  \ The RLHF-sensitive subspace shows 4–7× greater instruct-closeness than the RLHF-insensitive control (B ≈ 1.0–1.2 vs. A\
  \ = 4.35–7.31), confirming that the imprint is specific to the directions RLHF modified, not a global consequence of model\
  \ similarity. Layer-20 Subspace B reaching 1.01× (essentially no instruct-closeness in the RLHF-insensitive subspace) provides\
  \ the cleanest control.\n\n## Downstream Classification Accuracy\n\nThis null result follows directly from the 64.6% benign-SC\
  \ overlap (Section 5.5): the top-500 RLHF-sensitive encoder directions are elevated by RLHF on *both* harmful and benign\
  \ prompts, eliminating prompt-level discriminability by construction. This distinguishes *what RLHF modified* (broad encoder-direction\
  \ changes) from *what selectively activates on harmful versus benign prompts at inference* (a different, narrower set of\
  \ encoder directions not analyzed in this work).\n\nA per-prompt harm score (mean pre-topk activation across all top-500\
  \ RLHF-sensitive encoder directions with median threshold) achieves **0.476 accuracy** — near-chance on the balanced corpus\
  \ — versus a keyword baseline of **0.630**.  This null classification result confirms the 64.6% benign-SC overlap from an\
  \ independent angle: RLHF-sensitive encoder directions elevated on harmful prompts are similarly elevated on benign prompts,\
  \ yielding no discriminative signal.\n\n## Semantic Labeling\n\nLlama-3.1-8B-Instruct assigned labels to 100 top-SC encoder\
  \ directions: the majority received `harm_concept` labels, with smaller proportions of `safety_refusal`, `general_capability`,\
  \ and `other`. These results are qualitative observations only. The labeling uses character-prefix truncation of most-activating\
  \ prompts rather than per-token activation windows; no Cohen's κ validation against a manual gold set was performed; and\
  \ the 64.6% benign-SC overlap implies that many stylistic directions receive `harm_concept` labels because they activate\
  \ most strongly on harmful-prompt contexts rather than encoding harm-specific semantic content. Quantitative reliance on\
  \ the specific labeling percentages requires a manual gold-set annotation with κ ≥ 0.6, deferred to future work.\n\n## Layer-20\
  \ Replication\n\nTable 2 summarizes full quantitative replication at layer 20.\n\n**Table 2.** Quantitative results at layers\
  \ 22 and 20.\n\n| Metric | Layer 22 | Layer 20 |\n|--------|----------|----------|\n| Top-500 Reversal Score (R̄) | 0.921\
  \ | 0.924 |\n| Mid-500 Reversal Score (R̄) | 0.871 | 0.858 |\n| Cohen's d (top vs. mid) | 1.087 | — |\n| Top-500 OLS Slope\
  \ (β̄) | 0.986 ± 0.176 | 0.954 ± 0.191 |\n| Fraction β_f > 0.8 | 86.4% | 80.2% |\n| Fraction β_f < 0 | 0.2% | 0.0% |\n|\
  \ Mean |cos(r, enc)| top-500 | 0.01854 | 0.02086 |\n| Centroid A (RLHF-sensitive) | 4.35× | 7.31× |\n| Centroid B (RLHF-insensitive)\
  \ | 1.16× | 1.01× |\n| Centroid C (random-500) | 2.37× | 1.63× |\n| Centroid verdict | MODERATE | SELECTIVE |\n\n**REPLICATION_CONFIRMED**\
  \ at both layers.\n\n# Discussion\n\n## Geometric Interpretation of Encoder-Direction Preservation\n\nThe OLS slope result\
  \ (β̄ = 0.986, 86.4% of directions with β_f > 0.8) establishes that the preservation is comprehensive in both direction\
  \ and magnitude. The direct orthogonality measurement (|cos θ| = 0.0185) provides the mechanistic account: abliteration\
  \ removes $\\mathbf{r}$ from all weight matrices, but for abliteration to change the projection of a residual-stream activation\
  \ onto SAE encoder direction $f$ by even 10%, $\\mathbf{r}$ would need to project onto $\\mathbf{w}_{\\text{enc}}(f)$ with\
  \ |cos θ| ≥ 0.1. The measured mean |cos θ| = 0.0185 falls 5× below this threshold, explaining why both direction and magnitude\
  \ are preserved: the orthogonalization barely touches the RLHF-modified encoder directions. This interpretation is consistent\
  \ with Cristofano's observation [7] that the raw refusal direction primarily captures stylistic compliance patterns.\n\n\
  We note that this mechanistic account applies in the pre-topk projection regime. Whether the same geometry holds for the\
  \ post-topk sparse activations validated as semantically interpretable in single-model settings [12, 14] remains an open\
  \ question. The orthogonality result—which is a property of weight vectors, not activations—directly implies that *any*\
  \ function of the residual stream that can be expressed as a projection onto $\\mathbf{w}_{\\text{enc}}(f)$ will be approximately\
  \ unaffected by abliteration; the question is whether the semantically active post-topk features are well-described by such\
  \ projections, which requires a z_pre/z_post correlation analysis not yet available.\n\n## What the Preserved Imprint Represents\n\
  \nThe 64.6% benign-SC overlap requires careful interpretation. The majority of RLHF-sensitive encoder directions are modified\
  \ similarly when processing harmful and benign prompts, indicating these directions capture *general* RLHF behavioral changes\
  \ (formality, helpfulness, instruction-following structure) rather than harm-specific semantic content. The surviving 35.4%\
  \ uniquely-harmful directions are the more plausible carriers of harm-concept semantic content.\n\nCrucially, the near-orthogonality\
  \ of the refusal direction to all RLHF-modified encoder directions (harmful-specific and style-related alike) means abliteration\
  \ geometrically cannot remove any category of RLHF modification. The behavioral regression to base-like output arises because\
  \ the compliance mechanism (the refusal direction) is removed, not because the underlying encoder-direction modifications\
  \ are erased. The model forgets how to express refusal while retaining all representational changes RLHF introduced—at approximately\
  \ full magnitude.\n\nThis also explains the downstream classification null (accuracy = 0.476): the RLHF-sensitive encoder\
  \ directions are elevated on *both* harmful and benign prompts, so mean-activation discrimination is not available. The\
  \ representational change RLHF introduced is a broad encoder-direction shift, not a selective harm-detector.\n\n## Implications\
  \ for Safety Restoration\n\nIf abliteration preserves the RLHF imprint at approximately unit magnitude, the challenge for\
  \ targeted safety restoration is not to amplify surviving encoder-direction modifications—they are already at full instruct-model\
  \ activation levels. The challenge is to restore the *compliance mechanism*: the geometric pathway from current activations\
  \ to refusal outputs, implemented by the refusal direction $\\mathbf{r}$. Restoration requires re-injecting $\\mathbf{r}$\
  \ into the weight matrices without disturbing the already-intact encoder-direction modifications.\n\nThis reframes the restoration\
  \ problem as geometric re-injection rather than semantic amplification. The approach of Shportko et al. [20] identifies\
  \ compact steerable feature sets for RL-induced capabilities; an analogous method targeting the encoder directions that\
  \ mediate harm-concept-to-refusal conversion could enable targeted re-injection. The 35.4% uniquely-harmful encoder directions\
  \ are the most promising candidates for such targeted intervention, pending confirmation that they carry semantically coherent\
  \ harm-specific content (which requires z_post analysis).\n\n## Limitations\n\n**Pre-topk encoder projections.** All feature\
  \ metrics use $\\mathbf{z}_{\\text{pre}}$ (dense encoder projections) rather than the post-topk sparse activations validated\
  \ as interpretable by Bricken et al. [12] and Templeton et al. [14]. Pre-topk values are dense by construction—every encoder\
  \ direction receives a non-zero value on every prompt—so findings are more accurately characterized as *geometric properties\
  \ of encoder directions in a continuous projection space* than as *feature-level preservation of near-monosemantic SAE features*.\
  \ A validation experiment is required before semantic claims can be grounded: on the highest-R² prompt subset, compare top-500\
  \ features selected by SC(z_pre) versus SC(z_post) (overlap fraction) and compute Spearman ρ between z_pre and z_post Reversal\
  \ Scores. If ρ > 0.7 and overlap > 60%, the pre-topk proxy is defensible; if they diverge, the semantic labeling results\
  \ must be interpreted purely as observations about encoder-direction geometry. This is the most critical open validation\
  \ gap.\n\n**Abliteration depth verification.** The `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` checkpoint was not independently\
  \ verified. If abliteration is incomplete—even partially—the high Reversal Scores and β̄ ≈ 1 could trivially reflect the\
  \ abliterated model being close to the instruct model by construction. Two measurements would resolve this: (1) fraction\
  \ of AdvBench or HarmBench prompts on which the abliterated checkpoint produces a non-refusal response (expected near 100%\
  \ for complete abliteration); (2) mean $|\\text{Proj}_{\\mathbf{r}}(\\mathbf{h}_{\\text{abl}}(p))| / |\\mathbf{h}_{\\text{abl}}(p)|$\
  \ at layer 22 for the 500 harmful prompts (expected near zero if $\\mathbf{r}$ has been removed). These measurements can\
  \ be computed from already-cached activations. Until they are reported, the main findings should be interpreted as conditional\
  \ on abliteration completeness.\n\n**Multi-dimensional refusal subspace.** The orthogonality measurement uses the mean-difference\
  \ vector $\\mathbf{r}$ (a single vector). Joad et al. [23] demonstrate that refusal spans geometrically distinct directions\
  \ across behavioral categories, implying abliteration may remove a multi-dimensional subspace rather than a single vector.\
  \ The mechanistic explanation of zero reversed directions requires near-orthogonality of the *full ablation subspace* to\
  \ all RLHF-modified encoder directions, not just the mean-difference summary. Computing mean $|\\cos\\theta_f|$ for the\
  \ top-3 principal components of the (harmful − harmless) activation differences at layer 22 would determine whether the\
  \ orthogonality argument holds robustly; this is an important future measurement.\n\n**Single model family and layers.**\
  \ All results are for Qwen3-8B at layers 22 and 20. The zero-reversed finding and β̄ ≈ 1 result may not generalize to other\
  \ model families (Llama, Gemma) or other layer depths. The companion geometry studies [2, 3] cover 12 variants, establishing\
  \ the holistic survival phenomenon across families; analogous SAE-level analysis for Llama-3.1-8B and Gemma-3-9B using GemmaScope\
  \ and LlamaScope SAEs would test whether encoder-direction preservation is a general property of the abliteration geometry.\n\
  \n**Semantic labeling without validation.** The qualitative observation that the majority of top-SC encoder directions receive\
  \ `harm_concept` labels cannot support quantitative downstream reasoning without Cohen's κ validation against a manual gold\
  \ set of at least 50 annotations.\n\n# Conclusion\n\nWe present the first encoder-direction-level mechanistic analysis of\
  \ refusal abliteration using the frozen Qwen-Scope base SAE across three Qwen3-8B variants. Two complementary per-direction\
  \ metrics—the Reversal Score (directional preservation) and the OLS Slope (magnitude preservation)—together establish, for\
  \ Qwen3-8B, a comprehensive geometric imprint: every top-500 RLHF-sensitive SAE encoder direction has a positive Reversal\
  \ Score (mean R̄ = 0.921 ± 0.040, min = 0.684) and an OLS slope near unity (β̄ = 0.986 ± 0.176, 86.4% exceeding 0.8). A\
  \ noise-free mid-SC baseline (Cohen's d = 1.087) confirms selective preservation. Direct measurement of refusal-direction\
  \ geometry (|cos θ| = 0.0185 at layer 22, 0.0209 at layer 20) provides the mechanistic explanation: abliteration's orthogonalization\
  \ is geometrically incapable of touching the RLHF-modified encoder directions. A 64.6% benign-SC overlap reveals that the\
  \ preserved imprint spans both harm-semantic and general RLHF stylistic encoder-direction modifications; abliteration fails\
  \ to erase any category because all reside in a subspace near-orthogonal to the refusal direction. For Qwen3-8B, the abliterated\
  \ model's behavior reverts to base-like, but its encoder-direction projections remain instruct-like at full magnitude—a\
  \ dissociation that reframes safety restoration as geometric re-injection of the compliance direction rather than semantic\
  \ amplification of surviving features.\n\nThree critical open gaps constrain full confidence in the broader interpretation:\
  \ (1) z_pre/z_post validation (Spearman ρ and feature-ranking overlap at the highest-R² subset) to determine whether the\
  \ geometric encoder-direction findings correspond to semantically interpretable sparse features; (2) independent behavioral\
  \ and geometric verification of abliteration depth for the community checkpoint; and (3) extension of the orthogonality\
  \ measurement to the top-3 principal components of the refusal subspace, following Joad et al. [23].\n\n**Future directions:**\
  \ (i) z_pre/z_post validation experiment; (ii) abliteration depth verification via refusal-rate measurement and mean $|\\\
  text{Proj}_{\\mathbf{r}}|$ at layer 22; (iii) multi-dimensional refusal subspace (top-3 PC orthogonality); (iv) cross-family\
  \ extension to Llama-3.1-8B and Gemma-3-9B using GemmaScope and LlamaScope SAEs; (v) multi-layer analysis across all 36\
  \ Qwen-Scope layers; (vi) investigation of the 35.4% uniquely-harmful encoder directions as candidates for targeted geometric\
  \ restoration.\n\n# References\n\n[1] A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Rimsky, W. Gurnee, and N. Nanda. Refusal\
  \ in language models is mediated by a single direction. In *Advances in Neural Information Processing Systems*, 2024.\n\n\
  [2] I. Llorente-Saguer. The geometry of harmful intent: Training-free anomaly detection via angular deviation in LLM residual\
  \ streams. *arXiv preprint arXiv:2603.27412*, 2026.\n\n[3] I. Llorente-Saguer. Harmful intent as a geometrically recoverable\
  \ feature of LLM residual streams. *arXiv preprint arXiv:2604.18901*, 2026.\n\n[4] R. Chopra. A mechanistic investigation\
  \ of supervised fine tuning. *arXiv preprint arXiv:2605.11426*, 2026.\n\n[5] T. Jiralerspong and T. Bricken. Cross-architecture\
  \ model diffing with crosscoders: Unsupervised discovery of differences between LLMs. *arXiv preprint arXiv:2602.11729*,\
  \ 2026.\n\n[6] A. M. Kassem, T. Jiralerspong, N. Rostamzadeh, and G. Farnadi. Delta-Crosscoder: Robust crosscoder model\
  \ diffing in narrow fine-tuning regimes. *arXiv preprint arXiv:2603.04426*, 2026.\n\n[7] T. Cristofano. Surgical refusal\
  \ ablation: Disentangling safety from intelligence via concept-guided spectral cleaning. *arXiv preprint arXiv:2601.08489*,\
  \ 2026.\n\n[8] S. Jain, E. S. Lubana, K. Oksuz, T. Joy, P. H. S. Torr, A. Sanyal, and P. K. Dokania. What makes and breaks\
  \ safety fine-tuning? A mechanistic study. In *Advances in Neural Information Processing Systems*, 2024.\n\n[9] B. Deng\
  \ et al. Qwen-Scope: Turning sparse features into development tools for large language models. *arXiv preprint arXiv:2605.11887*,\
  \ 2026.\n\n[10] J. Li et al. Training superior sparse autoencoders for instruct models. *arXiv preprint arXiv:2506.07691*,\
  \ 2025.\n\n[11] N. Elhage et al. Toy models of superposition. *arXiv preprint arXiv:2209.10652*, 2022.\n\n[12] T. Bricken\
  \ et al. Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*, Anthropic,\
  \ 2023.\n\n[13] L. Gao et al. Scaling and evaluating sparse autoencoders. In *International Conference on Learning Representations*,\
  \ 2024.\n\n[14] A. Templeton et al. Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Transformer\
  \ Circuits Thread*, Anthropic, 2024.\n\n[15] J. Ji et al. BeaverTails: Towards improved safety alignment of LLM via a human-preference\
  \ dataset. In *Advances in Neural Information Processing Systems*, 2023.\n\n[16] P. Chao et al. JailbreakBench: An open\
  \ robustness benchmark for jailbreaking large language models. In *Advances in Neural Information Processing Systems*, 2024.\n\
  \n[17] M. Mazeika et al. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. In\
  \ *International Conference on Machine Learning*, pp. 35181–35224, 2024.\n\n[18] M. Conover et al. Free Dolly: Introducing\
  \ the world's first truly open instruction-tuned LLM. *Databricks Blog*, 2023.\n\n[19] G. Nanfack, E. Belilovsky, and E.\
  \ Dohmatob. Efficient refusal ablation in LLM through optimal transport. *arXiv preprint arXiv:2603.04355*, 2026.\n\n[20]\
  \ A. Shportko et al. Localizing RL-induced tool use to a single crosscoder feature. In *ICML 2026 Workshop on Mechanistic\
  \ Interpretability* (Spotlight), 2026.\n\n[21] W. J. Yeo, N. Prakash, C. Neo, R. K.-W. Lee, E. Cambria, and R. Satapathy.\
  \ Understanding refusal in language models with sparse autoencoders. In *Empirical Methods in Natural Language Processing*,\
  \ pp. 6377–6399, 2025.\n\n[22] S.-H. Kim, H. Jin, Y. Lee, and Y. Han. CRaFT: Circuit-guided refusal feature selection via\
  \ cross-layer transcoders. *arXiv preprint arXiv:2604.01604*, 2026.\n\n[23] F. Joad, M. Hawasly, S. Boughorbel, N. Durrani,\
  \ and H. T. Sencar. There is more to refusal in large language models than a single direction. *arXiv preprint arXiv:2602.02132*,\
  \ 2026."
summary: >-
  We present the first encoder-direction-level mechanistic analysis of refusal abliteration in Qwen3-8B using the frozen Qwen-Scope
  base SAE. Two new per-direction metrics—the Reversal Score (directional preservation) and the OLS Slope (magnitude preservation)—demonstrate
  that every RLHF-sensitive SAE encoder direction has a positive Reversal Score (mean 0.921) and an OLS slope near unity (mean
  0.986), meaning abliteration preserves the RLHF imprint in both direction and magnitude. Direct measurement of refusal-direction
  geometry (mean |cosθ| = 0.019) provides the mechanistic explanation: abliteration's orthogonalization geometrically cannot
  touch the RLHF-modified encoder directions. A 64.6% benign-SC overlap reveals the preserved imprint spans both harm-specific
  and general RLHF stylistic modifications. Three critical open gaps are identified: z_pre/z_post proxy validation, independent
  abliteration depth verification, and extension of the orthogonality measurement to the multi-dimensional refusal subspace
  identified by Joad et al. (2026).
</paper_text>

<available_figures>
No figures available.
</available_figures>

<figure_requirements>
CRITICAL: Include ALL figures from <available_figures>. No exceptions.

- Every figure MUST use \includegraphics{figures/filename.jpg}
- Do NOT skip, convert to tables, or describe without inserting
- Each needs: \begin{figure*|figure}[placement], \includegraphics, \caption, \label, \end{...} — pick env + placement by the figure's `aspect_ratio` field (see PLACEMENT below). Constrain every \includegraphics with `width=\linewidth,height=0.4\textheight,keepaspectratio` (single-column) or `width=\textwidth,height=0.45\textheight,keepaspectratio` (figure*). Use exactly these option keys — `max height=` is NOT valid LaTeX
- Use the `caption` field from each figure for \caption{...} — do NOT invent new captions
- Place figures where their [FIGURE:fig_id] markers appear in paper_text
- VERIFICATION: paper.tex MUST have exact same number of \includegraphics as <available_figures>
- Do NOT generate new figure images (no matplotlib, no PIL, no image generation). Use ONLY the pre-generated figures from <available_figures>. They were already created by a previous pipeline step.

PLACEMENT BY ASPECT RATIO (use the `aspect_ratio` field on each figure):
- `21:9` (architecture diagrams / hero figures): \begin{figure*}[!t] (full two-column width, top of page). The hero architecture diagram should appear EARLY in the paper — typically at the top of page 2. Marker placement in paper_text already determines this; preserve it.
- `16:9` (comparisons, multi-panel results): \begin{figure*}[!t] for full-width or \begin{figure}[!htbp] for single-column.
- `4:3` / `1:1` / `3:2` / `3:4` / `9:16`: \begin{figure}[!htbp] (single-column).
</figure_requirements>

<artifact_links>
The paper_text contains \footnote{Code: \url{...}} references linking to artifact source code
on GitHub. Include \usepackage{hyperref} and \usepackage{url}.
Preserve these exactly as-is — do not remove, rewrite, or convert them to plain text.
The URLs will not resolve yet (the repo is deployed after compilation) — do NOT try to verify or fix them.
</artifact_links>

<headings>
NEVER use inline math (``$...$``) inside ``\section{...}`` / ``\subsection{...}`` / ``\subsubsection{...}`` arguments — hyperref's bookmark builder errors out (``Token not allowed in a PDF string``) and the PDF outline breaks. If a section heading needs a math-looking term, use the text equivalent (``d star`` not ``$d^*$``, ``alpha-equivalent`` not ``$\alpha$-equivalent``) or wrap it in ``\texorpdfstring{$math$}{plain}``. Inline math inside body paragraphs is fine.
</headings>

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

<todos>
TODO 1. Read and STRICTLY follow these skills: aii-paper-to-latex, aii-semscholar-bib.
TODO 2. Review <paper_text> and <available_figures>. Copy all figure images into ./figures/ in your workspace. Count figures — MUST include every one. Plan placements per section. Build `./references.bib` via aii_semscholar_bib__fetch — collect DOIs/ArXiv IDs from <paper_text> and batch-fetch all BibTeX in one call. Do NOT fabricate entries.
TODO 3. Create `./paper.tex` per aii-paper-to-latex skill's setup, write ALL sections, insert ALL figures from <available_figures>, include `./references.bib` via \bibliography. Compile to PDF per skill's process. Fix errors.
TODO 4. CRITICAL VERIFICATION: Run `grep -c 'includegraphics' paper.tex`, confirm count equals figures in <available_figures>. If not, add missing figures. Verify `./paper.pdf` was created.
TODO 5. VISUAL REVIEW: Write Python script to convert EVERY page of paper.pdf to PNG at 150 DPI (use pdf2image or pymupdf). Then read ALL page screenshots — each page image costs ~1,600 tokens so a 15-page paper is only ~24K tokens. You MUST read every page. The ONLY exception is if all page images would not fit in your remaining context — in that case, read as many as fit and state which pages you are skipping and why. Check every page for layout issues, overlapping figures, cut-off text, bad spacing, formatting problems. Fix issues and recompile.
TODO 6. FINAL READ: Check page count (`pdfinfo paper.pdf` or pymupdf). Read entire paper.pdf — check for missing sections, unclear explanations, inconsistencies, typos. Fix and recompile. The ONLY exception is if all pages would not fit in your remaining context — in that case, read as many pages as fit and state which pages you are skipping and why.
</todos>

---

Output the result as JSON to: `./.terminal_claude_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "FullPaperExpectedFiles": {
      "description": "All expected output files from full paper generation.",
      "properties": {
        "paper_tex_path": {
          "description": "Path to LaTeX source file. Example: 'paper.tex'",
          "title": "Paper Tex Path",
          "type": "string"
        },
        "paper_pdf_path": {
          "description": "Path to compiled PDF. Example: 'paper.pdf'",
          "title": "Paper Pdf Path",
          "type": "string"
        },
        "references_bib_path": {
          "description": "Path to BibTeX bibliography file. Example: 'references.bib'",
          "title": "References Bib Path",
          "type": "string"
        },
        "figure_paths": {
          "description": "Paths to all figure image files. Example: ['figures/fig1_v0.jpg', 'figures/fig2_v0.jpg']",
          "items": {
            "type": "string"
          },
          "title": "Figure Paths",
          "type": "array"
        }
      },
      "required": [
        "paper_tex_path",
        "paper_pdf_path",
        "references_bib_path",
        "figure_paths"
      ],
      "title": "FullPaperExpectedFiles",
      "type": "object"
    }
  },
  "description": "Full paper \u2014 structured output from paper generation.",
  "properties": {
    "title": {
      "description": "Paper title in plain, everyday language \u2014 short and jargon-free so a non-expert grasps it at a glance. Aim for about 4-8 words (~40 characters).",
      "maxLength": 90,
      "minLength": 12,
      "title": "Title",
      "type": "string"
    },
    "summary": {
      "description": "Brief summary of the generated paper: sections written, figures included, compilation status",
      "maxLength": 5000,
      "minLength": 500,
      "title": "Summary",
      "type": "string"
    },
    "out_expected_files": {
      "$ref": "#/$defs/FullPaperExpectedFiles",
      "description": "All output files you created. Must include paper.tex, paper.pdf, references.bib, and paths to all figure files."
    }
  },
  "required": [
    "title",
    "summary",
    "out_expected_files"
  ],
  "title": "FullPaper",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-07-17 15:28:49 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [3] SKILL-INPUT — aii-paper-to-latex · 2026-07-17 15:29:03 UTC

The agent loaded the **aii-paper-to-latex** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-paper-to-latex
description: LaTeX paper assembly and compilation. Covers document setup, figure inclusion from pre-generated JPEGs, compilation process, and output files. Use when assembling a paper from pre-written text and pre-generated figures into a compiled PDF.
---

## LaTeX Paper Assembly

Assembles a research paper from paper text, pre-generated figure JPEGs, and bibliography into a compiled PDF.

### Document Setup

```latex
\documentclass[11pt,letterpaper]{article}
\usepackage{graphicx, geometry, amsmath, hyperref, natbib, booktabs, xcolor, listings}
\geometry{margin=1in}
\hypersetup{colorlinks=true, linkcolor=black, citecolor=black, urlcolor=black}
```

### Figure Inclusion

CRITICAL: Include ALL figures. Every figure MUST appear in the paper.

```latex
\begin{figure}[!htbp]
  \centering
  \includegraphics[width=0.92\textwidth,keepaspectratio]{figures/filename.jpg}
  \caption{Descriptive caption.}
  \label{fig:label}
\end{figure}
```

Rules:
- ALWAYS use `[!htbp]` float placement (NOT `[t]` or `[h]` alone)
- ALWAYS constrain with `width` and `keepaspectratio` to prevent page takeover
- Every figure needs `\caption`, `\label`, and a `\ref` in the text
- Do NOT convert figures to tables or describe them without inserting the image
- Do NOT skip any figures

### Compilation Process

Run each command separately (do NOT chain with `&&` — pdflatex often exits non-zero on warnings, which would skip bibtex and leave citations as `??`):

```bash
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

All four commands are required. Skipping bibtex causes `??` in all citations.
Fix any errors between runs. Verify `./paper.pdf` was created.

### Output Files

- `./paper.tex` — LaTeX source
- `./references.bib` — bibliography file
- `./paper.pdf` — compiled PDF
- `./figures/*.jpg` — all figure images (pre-generated, copied into workspace)
````

### [4] SKILL-INPUT — aii-semscholar-bib · 2026-07-17 15:29:03 UTC

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
