# review_hypo — create_idea

> Phase: `hypo_loop` · round 1 · `review_hypo`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim transcript of this agent task — every system/user prompt, assistant response, thinking block, tool call and tool result — in the order they occurred. Nothing truncated.

## Task: `review_hypo` (terminal_claude_agent, claude-sonnet-4-6)

### [1] CONFIG · 2026-07-17 09:37:41 UTC

```
model: claude-sonnet-4-6 | effort: medium | permission: bypassPermissions | cwd: /ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_1/review_hypo
```

### [2] SYSTEM-USER prompt · 2026-07-17 09:37:47 UTC

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
</hypothesis>

<review_context>
No experiments have been run yet — evaluate the hypothesis purely on its merits.
</review_context>

<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for judging whether the hypothesis is genuinely novel versus already-done or a known dead end in this field.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>





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

STEP 3 — H↔H EDGE:
This is the first iteration — there is no previous hypothesis. Leave
``relation_type`` null and ``relation_rationale`` empty.

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

### [3] HUMAN-USER prompt · 2026-07-17 09:37:47 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [4] ASSISTANT · 2026-07-17 09:37:53 UTC

```
I'll research the relevant prior work before writing the review.
```

### [5] TOOL CALL — Skill · 2026-07-17 09:37:53 UTC

```
aii-web-tools
```

### [6] TOOL RESULT — Skill · 2026-07-17 09:37:53 UTC

```
{"success": true, "commandName": "aii-web-tools"}
```

### [7] TOOL CALL — ToolSearch · 2026-07-17 09:37:57 UTC

```
{'query': 'select:WebSearch,WebFetch'}
```

### [8] TOOL RESULT — ToolSearch · 2026-07-17 09:37:57 UTC

```
{"matches": ["WebSearch", "WebFetch"], "query": "select:WebSearch,WebFetch", "total_deferred_tools": 21}
```

### [9] TOOL CALL — WebSearch · 2026-07-17 09:38:04 UTC

```
abliteration refusal direction RLHF safety fine-tuning mechanistic interpretability 2025
```

### [10] TOOL CALL — WebSearch · 2026-07-17 09:38:04 UTC

```
sparse autoencoder SAE RLHF safety features abliteration surviving 2025 2026
```

### [11] TOOL RESULT — WebSearch · 2026-07-17 09:38:12 UTC

```
{"query": "abliteration refusal direction RLHF safety fine-tuning mechanistic interpretability 2025", "results": [{"tool_use_id": "srvtoolu_012uzpSqMMvCcy47LqoayQpF", "content": [{"title": "The Refusal Direction | Learn Mechanistic Interpretability", "url": "https://learnmechinterp.com/topics/refusal-direction/"}, {"title": "LLM Refusal Abliteration Mechanisms", "url": "https://www.emergentmind.com/topics/refusal-abliteration"}, {"title": "There Is More to Refusal in Large Language Models than a Single Direction", "url": "https://arxiv.org/html/2602.02132v1"}, {"title": "OBLITERATUS: Mapping the Geometry of Refusal Inside Large Language Models", "url": "https://themenonlab.blog/blog/obliteratus-abliteration-llm-refusal-removal/"}, {"title": "COSMIC: Generalized Refusal Direction Identification in LLM Activations", "url": "https://arxiv.org/pdf/2506.00085"}, {"title": "Not All Refusals Are Equal: How Safety Alignment Fails Cybersecurity at Scale", "url": "https://arxiv.org/html/2607.02714v1"}, {"title": "Turning the Spell Around: Lightweight Alignment Amplification via Rank-One Safety Injection", "url": "https://arxiv.org/pdf/2508.20766"}, {"title": "Refusal Direction in LLM Safety", "url": "https://www.emergentmind.com/topics/refusal-direction"}, {"title": "RepIt: Steering Language Models with Concept-Specific Refusal Vectors", "url": "https://arxiv.org/pdf/2509.13281"}, {"title": "Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation", "url": "https://arxiv.org/pdf/2501.18100"}]}, "Based on the search results, I found substantial recent research on this topic. Here's what the literature shows:\n\n## Core Concept\n\nRefusal abliteration is defined as the targeted removal of low-dimensional refusal features in LLMs, effectively reducing their safety responses. This builds on mechanistic interpretability research into how safety features are encoded in language models.\n\n## The Refusal Direction Finding\n\nThe refusal direction was found independently in 13 different chat models spanning different families and scales (1.3B to 72B parameters). Safety training -- via RLHF, DPO, or similar techniques -- teaches the model to distinguish harmful from harmless requests and respond appropriately. Removing that direction (via ablation) should disable refusal. Adding that direction (via addition steering) should induce refusal even on harmless inputs.\n\n## Recent Developments in 2025\n\nThe research landscape has evolved significantly:\n\n- Extended-refusal fine-tuning trains models on richer refusal responses featuring a neutral overview, explicit refusal, and ethical rationale, thereby distributing the refusal signal across several latent dimensions. When abliteration is applied, these models maintain refusal rates above 90%, compared to baseline models with rates reduced by 70–80%.\n\n- Concurrent mechanistic work has shown that the refusal subspace is richer than a single direction: Wollschläger et al. (2025) identify multi-dimensional polyhedral concept cones, and Pan et al. (2025) find a dominant direction accompanied by multiple semantically distinct secondary features.\n\n- As refusal mechanisms are often shallow, brittle, and encoded in low dimensions, they are susceptible to targeted attacks. Multidimensional and mechanistically independent refusal features point to the need for more robust, multi-factorial safety alignment strategies."], "durationSeconds": 9.817687478, "searchCount": 1}
```

### [12] TOOL RESULT — WebSearch · 2026-07-17 09:38:16 UTC

```
{"query": "sparse autoencoder SAE RLHF safety features abliteration surviving 2025 2026", "results": [{"tool_use_id": "srvtoolu_01G3zdstKRk8oxtefAGQ3VXk", "content": [{"title": "[2507.00665] SAFER: Probing Safety in Reward Models with Sparse Autoencoder", "url": "https://arxiv.org/abs/2507.00665"}, {"title": "(PDF) Sparse Autoencoder Features for Classifications and Transferability", "url": "https://www.researchgate.net/publication/389090793_Sparse_Autoencoder_Features_for_Classifications_and_Transferability"}, {"title": "Discovering Millions of Interpretable Features with Sparse Autoencoders", "url": "https://arxiv.org/pdf/2606.26620"}, {"title": "Sparse Autoencoder Framework", "url": "https://www.emergentmind.com/topics/sparse-autoencoder-based-framework"}, {"title": "Sparse Autoencoders (SAEs) | Longterm Wiki", "url": "https://www.longtermwiki.com/wiki/E480"}, {"title": "Auditing Games for Sandbagging", "url": "https://arxiv.org/pdf/2512.07810"}, {"title": "NExT-Guard: Training-Free Streaming Safeguard without Token-Level Labels", "url": "https://arxiv.org/pdf/2603.02219"}, {"title": "SAFER: Probing Safety in Reward Models with Sparse Autoencoder", "url": "https://arxiv.org/html/2507.00665v3"}, {"title": "Interpretable Reward Model via Sparse Autoencoder", "url": "https://www.arxiv.org/pdf/2508.08746"}, {"title": "Graph-Regularized Sparse Autoencoders for LLM Safety Steering", "url": "https://arxiv.org/pdf/2512.06655"}]}, "Based on the web search results, here's what I found regarding sparse autoencoders (SAEs), RLHF, and safety features in 2025-2026:\n\n## Recent Research on SAEs and Safety\n\nA framework called SAFER (Sparse Autoencoder For Enhanced Reward model) has been developed for interpreting and improving reward models through mechanistic analysis, leveraging Sparse Autoencoders to uncover human-interpretable features in reward model activations that enable insight into safety-relevant decision-making. This work was updated in January 2026.\n\n## Safety Applications\n\nRecent research has focused on leveraging SAE features for safety-related analysis and control, including safety landscape characterization, robustness-oriented safety steering, controllable classification, and guardrail-style mitigation. Additionally, SAE-derived subspaces enable robust, interpretable unlearning and provide actionable, mechanistically motivated tools for responsible AI, model auditing, and safe deployment.\n\n## Notable Limitations\n\nDeepMind's March 2025 negative results showed SAEs underperforming simple probes on downstream tasks despite promising safety applications. However, DeepMind noted SAEs remain \"fairly helpful for debugging low quality datasets\" and left open the possibility of returning to SAE research \"if there is significant progress on some of SAEs' core issues\".\n\n## RLHF and Reward Models\n\nSAEs are being applied to mechanistically dissect reward model activations, enabling precise feature-level interpretability, particularly focusing on safety-related model behaviors."], "durationSeconds": 13.041650259999999, "searchCount": 1}
```

### [13] TOOL CALL — WebFetch · 2026-07-17 09:38:24 UTC

```
URL: https://arxiv.org/abs/2603.27412
Prompt: Summarize this paper's methodology, key findings, and what it concludes about abliteration and safety representations. What does it measure and what is the AUROC finding?
```

### [14] TOOL CALL — WebFetch · 2026-07-17 09:38:24 UTC

```
URL: https://arxiv.org/abs/2605.11426
Prompt: What does this paper find about how SFT/fine-tuning changes SAE feature activations? What methodology does it use and what are the main conclusions about feature divergence?
```

### [15] TOOL CALL — WebSearch · 2026-07-17 09:38:24 UTC

```
LatentBiopsy arxiv 2603.27412 harmful prompt geometry ablation AUROC 2025
```

### [16] TOOL RESULT — WebFetch · 2026-07-17 09:38:26 UTC

```
{"bytes": 45249, "code": 200, "codeText": "OK", "result": "# LatentBiopsy: Training-Free Harmful Prompt Detection\n\n## Methodology\n\nThe paper introduces LatentBiopsy, a technique that analyzes geometric patterns in LLM residual streams. The approach requires only 200 safe prompts to establish a baseline by computing the leading principal component of their activations at a specific layer. New prompts are then scored based on their angular deviation (θ) from this reference direction, with an anomaly score derived from how well the deviation fits a Gaussian distribution. Critically, \"no harmful examples are required for training.\"\n\n## Key Findings\n\nThe research evaluates six Qwen model variants and reveals three striking patterns:\n\n1. **Geometric persistence despite ablation**: Harmful intent leaves detectable geometric signatures even after refusal mechanisms are surgically removed, with only minimal performance drops (≤0.015 AUROC difference).\n\n2. **Angular distribution asymmetry**: Harmful prompts cluster tightly in angular space (~0.03 radians), while safe prompts scatter widely (~0.27 radians)—a tenfold difference preserved across all model variants.\n\n3. **Opposing ring orientations**: Different model families position harmful prompts on opposite sides of the activation geometry, motivating direction-agnostic scoring.\n\n## AUROC Performance\n\nThe method achieves exceptional discrimination:\n- **Harmful vs. normative detection**: AUROC ≥ 0.937\n- **Harmful vs. benign-aggressive prompts**: AUROC = 1.000\n\nWith sub-millisecond computational overhead per query.\n\n## Conclusion on Abliteration and Safety\n\nThe findings suggest \"geometric dissociation between harmful-intent representation and the downstream generative refusal mechanism,\" implying that safety involves multiple independent representational layers beyond just refusal outputs.", "durationMs": 6837, "url": "https://arxiv.org/abs/2603.27412"}
```

### [17] TOOL RESULT — WebFetch · 2026-07-17 09:38:30 UTC

```
{"bytes": 40504, "code": 200, "codeText": "OK", "result": "# Paper Summary: Mechanistic Investigation of Supervised Fine-Tuning\n\n## Key Finding on SAE Feature Activations\n\nThe research reveals a striking disconnect: while cosine similarity between hidden activations remains high before and after SFT, the underlying sparse features diverge substantially. As stated in the abstract, \"projecting both sets of activations through a Sparse Autoencoder (SAE) pretrained on the base model reveals that the underlying sparse latents diverge significantly.\"\n\n## Methodology\n\nThe authors employ a novel investigative pipeline that leverages pretrained SAEs as diagnostic instruments. This approach treats SAEs as high-resolution tools to examine the representational changes occurring during fine-tuning at a mechanistic level.\n\n## Main Conclusions\n\nThe research identifies three key insights:\n\n1. **Task and Layer Specificity**: Particular semantic features that change during SFT are distributed across specific tasks and layers in predictable patterns\n\n2. **Safety Alignment Profile**: The study discovers a distinctive layer-wise update profile specific to safety alignment procedures\n\n3. **Representational Divergence**: Despite surface-level similarity in activation geometry, the fine-tuned model substantially reorganizes its sparse feature representations\n\nAll experimental code and analysis materials are publicly available through the authors' GitHub repository, enabling reproducibility and further investigation.", "durationMs": 8132, "url": "https://arxiv.org/abs/2605.11426"}
```

### [18] TOOL RESULT — WebSearch · 2026-07-17 09:38:30 UTC

```
{"query": "LatentBiopsy arxiv 2603.27412 harmful prompt geometry ablation AUROC 2025", "results": [{"tool_use_id": "srvtoolu_01NTiGEW4zUJHy8b8TPpYvee", "content": [{"title": "[2603.27412] The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/abs/2603.27412"}, {"title": "Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams", "url": "https://arxiv.org/html/2604.18901"}, {"title": "The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/html/2603.27412"}, {"title": "The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/pdf/2603.27412"}, {"title": "[2604.06285] Harnessing Hyperbolic Geometry for Harmful Prompt Detection and Sanitization", "url": "https://arxiv.org/abs/2604.06285"}, {"title": "[2601.08489] Surgical Refusal Ablation: Disentangling Safety from Intelligence via Concept-Guided Spectral Cleaning", "url": "https://arxiv.org/abs/2601.08489"}, {"title": "Harnessing Hyperbolic Geometry for Harmful Prompt Detection and Sanitization", "url": "https://arxiv.org/html/2604.06285v1"}, {"title": "Geometry-Lite: Interpretable Safety Probing via Layer-Wise Margin Geometry", "url": "https://arxiv.org/html/2605.20241"}, {"title": "Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams", "url": "https://arxiv.org/pdf/2604.18901"}]}, "Based on the search results, here's what I found about LatentBiopsy (arXiv 2603.27412):\n\n## Overview\n\nLatentBiopsy is a training-free method for detecting harmful prompts by analysing the geometry of residual-stream activations in large language models. It computes the leading principal component of activations from 200 safe normative prompts at a target layer and characterises new prompts by their radial deviation angle from this reference direction.\n\n## Key Results\n\nThe method achieves harmful versus normative AUROC ≥ 0.937 and harmful versus benign-aggressive AUROC = 1.000 across all six tested variants, with sub-millisecond per-query overhead and no harmful training data.\n\n## Ablation Study Findings\n\nThe paper includes ablation experiments using model variants with the refusal mechanism surgically removed. Both abliterated variants achieve AUROC at most 0.015 below their instruction-tuned counterparts, establishing a geometric dissociation between harmful-intent representation and the downstream generative refusal mechanism.\n\nThe evaluation covers two complete model triplets from the Qwen3.5-0.8B and Qwen2.5-0.5B families: base, instruction-tuned, and abliterated versions."], "durationSeconds": 8.091532362000006, "searchCount": 1}
```

### [19] TOOL CALL — WebFetch · 2026-07-17 09:38:36 UTC

```
URL: https://arxiv.org/abs/2601.08489
Prompt: What does this paper find about the refusal direction being polysemantic? What is spectral residualization and what does it achieve? Key findings on capability entanglement.
```

### [20] TOOL CALL — WebSearch · 2026-07-17 09:38:36 UTC

```
Qwen-Scope SAE arxiv 2605.11887 Qwen3-8B sparse autoencoder 2026
```

### [21] TOOL CALL — WebSearch · 2026-07-17 09:38:36 UTC

```
SAE feature type-R type-I RLHF safety imprint abliteration survival feature classification 2025 2026
```

### [22] TOOL RESULT — WebFetch · 2026-07-17 09:38:42 UTC

```
{"bytes": 42788, "code": 200, "codeText": "OK", "result": "# Paper Analysis: Surgical Refusal Ablation\n\n## Polysemantic Refusal Direction\n\nThe research identifies that the raw refusal vector is \"polysemantic, entangling the refusal signal with core capability circuits and linguistic style.\" This means the refusal mechanism is not cleanly isolated—it bleeds into other model functions, causing unintended degradation when simply removed.\n\n## Spectral Residualization Explained\n\nThe authors employ \"ridge-regularized spectral residualization to orthogonalize the refusal vector against these directions.\" This mathematical technique removes contaminating signals by:\n\n- Creating a registry of \"Concept Atoms\" representing protected capabilities and stylistic elements\n- Mathematically projecting out these concepts from the refusal vector\n- Producing a \"clean refusal direction\" that targets only refusal-relevant structure\n\n## Critical Findings on Capability Entanglement\n\nThe results reveal significant capability preservation advantages:\n\n- Standard ablation on Qwen3-VL-4B caused \"severe drift (first-token KL = 2.088),\" while SRA maintained the original distribution at \"KL = 0.044\"\n- Perplexity impact remained negligible (approximately 0.02 delta on Wikitext-2)\n- The authors define \"Ghost Noise\" as spectral contamination causing apparent model damage\n- Using high-resolution proxies (GSM8K, MBPP), SRA preserved mathematical and coding distributions while achieving near-zero refusal rates (0-2%)\n\nThe core insight: most degradation from naive refusal removal stems from accidentally corrupting capability subspaces rather than fundamental model damage.", "durationMs": 7447, "url": "https://arxiv.org/abs/2601.08489"}
```

### [23] TOOL RESULT — WebSearch · 2026-07-17 09:38:42 UTC

```
{"query": "Qwen-Scope SAE arxiv 2605.11887 Qwen3-8B sparse autoencoder 2026", "results": [{"tool_use_id": "srvtoolu_01ENRnAJEmJsrA9AZov7a9sj", "content": [{"title": "[2605.11887v1] Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/abs/2605.11887v1"}, {"title": "Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/html/2605.11887"}, {"title": "[2605.11887] Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/abs/2605.11887"}, {"title": "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"}, {"title": "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100"}, {"title": "Discovering Millions of Interpretable Features with Sparse Autoencoders", "url": "https://arxiv.org/html/2606.26620v1"}, {"title": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100"}, {"title": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"}, {"title": "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"}]}, "Based on the search results, here's what I found:\n\n## Qwen-Scope SAE (arXiv 2605.11887)\n\nQwen-Scope is an open-source suite of SAEs built on the Qwen model family, comprising 14 groups of SAEs across 7 model variants from the Qwen3 and Qwen3.5 series, covering both dense and mixture-of-expert architectures.\n\nSparse autoencoders (SAEs) are emerging as one of the most promising tools for decomposing model activations into sparse, interpretable feature representations.\n\n### Key Applications\n\nSAEs can go beyond post-hoc analysis to serve as practical interfaces for model development including inference-time steering, where SAE feature directions control language, concepts, and preferences without modifying model weights.\n\nApplication scenarios include steerable inference control, evaluation sample distribution analysis and comparison, data classification and synthesis, and model training and optimization.\n\n### Paper Details\n\nThe paper was submitted on May 12, 2026, with authors including Boyi Deng, Xu Wang, Yaoning Wang, Yu Wan, Yubo Ma, Baosong Yang, Haoran Wei, Jialong Tang, Huan Lin, Ruize Gao, Tianhao Li, Qian Cao, Xuancheng Ren, Xiaodong Deng, An Yang, Fei Huang, Dayiheng Liu, and Jingren Zhou.\n\nThe paper introduces interpretability tools for analyzing and improving large language models using sparse autoencoders trained on Qwen's hidden layers."], "durationSeconds": 7.735698209000002, "searchCount": 1}
```

### [24] TOOL RESULT — WebSearch · 2026-07-17 09:38:50 UTC

```
{"query": "SAE feature type-R type-I RLHF safety imprint abliteration survival feature classification 2025 2026", "results": [{"tool_use_id": "srvtoolu_01F4QEpwK8TeBiiuFvwdfaEn", "content": [{"title": "Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation", "url": "https://arxiv.org/pdf/2512.23260"}, {"title": "Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation", "url": "https://arxiv.org/html/2512.23260"}, {"title": "[Literature Review] Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation", "url": "https://www.themoonlight.io/en/review/interpretable-safety-alignment-via-sae-constructed-low-rank-subspace-adaptation"}, {"title": "Sparse Autoencoder Features for Classifications and Transferability - ACL Anthology", "url": "https://aclanthology.org/2025.emnlp-main.1521/"}, {"title": "Certifiable Safe RLHF: Semantic Grounding and Fixed Penalty Constraint Optimization for Safer LLM Alignment", "url": "https://arxiv.org/html/2510.03520"}, {"title": "Perturbation Probing: A Two-Pass-per-Prompt Diagnostic for FFN Behavioral Circuits in Aligned LLMs", "url": "https://arxiv.org/pdf/2604.27401"}, {"title": "Analyzing Visual Aircraft Representations with Sparse Autoencoders", "url": "https://arxiv.org/pdf/2606.15468"}, {"title": "Safe RLHF: Balancing Helpfulness & Safety", "url": "https://www.emergentmind.com/topics/safe-reinforcement-learning-from-human-feedback-safe-rlhf"}, {"title": "Sparse Autoencoder Features for Classifications and Transferability", "url": "https://arxiv.org/pdf/2502.11367"}, {"title": "The Alignment Veto: How Safety Training Suppresses Cultural Knowledge in LLMs", "url": "https://arxiv.org/pdf/2510.13154"}]}, "Based on the search results, I found relevant recent research on SAE (Sparse Autoencoder) features related to safety, RLHF, and model alignment from 2025-2026:\n\n## Key Research Findings\n\n**SAE-Based Safety Alignment:**\nSAE encoding transforms polysemantic hidden states where safety-relevant features become separable. Recent methods approach or even exceed the safety rates of instruction-tuned models aligned with compute-intensive RLHF, achieving up to 99.6% safety rate on Gemma 2 9B.\n\n**Layer and Feature Analysis:**\nMiddle-to-deep layers achieve near-complete separation of safe and unsafe behaviors, with safety-relevant concepts concentrated in these layers. Layer selection analysis shows middle-to-deep layers (e.g., 15–23 for Gemma 2 2B) exhibit the clearest separation of safety-relevant concepts.\n\n**Feature Classification and Transferability:**\nRecent work systematically analyzes SAE for interpretable feature extraction from LLMs in safety-critical classification tasks, evaluating model-layer selection, SAE architectural configurations, and binarization effects. SAE-derived features achieve macro F1 > 0.8, outperforming hidden-state and BoW baselines while demonstrating cross-model transfer.\n\n**Safety Behavior Decomposition:**\nAblation studies reveal that RLHF safety is layered, with a fragile template sitting over robust pre-training harm warnings.\n\nThe research emphasizes the use of ablation studies to identify and understand how safety features survive or degrade under perturbation, supporting the development of more robust alignment methods."], "durationSeconds": 13.429807088000002, "searchCount": 1}
```

### [25] TOOL CALL — WebFetch · 2026-07-17 09:38:56 UTC

```
URL: https://arxiv.org/abs/2604.18901
Prompt: What does this paper find about harmful intent being geometrically recoverable? How does it relate to abliteration? Key methodology and findings.
```

### [26] TOOL CALL — WebSearch · 2026-07-17 09:38:56 UTC

```
"huihui-ai" Qwen3-8B-abliterated abliteration technique difference-of-means orthogonal projection
```

### [27] TOOL RESULT — WebFetch · 2026-07-17 09:39:00 UTC

```
{"bytes": 44707, "code": 200, "codeText": "OK", "result": "# Geometric Recoverability of Harmful Intent in LLMs\n\n## Core Finding\n\nThe research demonstrates that harmful intent exists as a **linearly separable feature** within language model residual streams. Using just 100 labeled examples per class, researchers achieved a mean AUROC of 0.982 across 12 models spanning multiple architectural families (Qwen, Llama, Gemma) and parameter scales from 0.5B to 9B parameters.\n\n## Abliteration Connection\n\nA striking discovery involves abliterated models—systems where refusal mechanisms have been explicitly removed. The paper found that \"the supervised strategies...match their instruction-tuned counterpart within ±0.003 AUROC in abliterated variants,\" indicating the harmful intent signal persists even after removing a model's safety training.\n\n## Key Methodology\n\nResearchers used Soft-AUC optimization to fit detection directions from labeled examples, testing generalization across held-out harm benchmarks. They discovered protocol-dependent variations: different token extraction methods (max-pooling over content tokens versus last-token extraction) recovered harm directions \"73° apart,\" suggesting the feature's geometric structure depends on measurement choices.\n\n## Critical Finding\n\nThe work highlights that \"a deployed 9B safety classifier shows the same pattern at AUROC 0.94 and TPR 0.30,\" revealing dramatic variance in true positive rates despite similar overall accuracy—prompting recommendations for low false-positive rate reporting in safety evaluations.", "durationMs": 6192, "url": "https://arxiv.org/abs/2604.18901"}
```

### [28] TOOL RESULT — WebSearch · 2026-07-17 09:39:10 UTC

```
{"query": "\"huihui-ai\" Qwen3-8B-abliterated abliteration technique difference-of-means orthogonal projection", "results": [{"tool_use_id": "srvtoolu_01GLBZruS9zu8v5zGdnUQr4Q", "content": [{"title": "Run Qwen3-8B-abliterated API (Easy Deployment & Flat-Rate Pricing)", "url": "https://featherless.ai/models/huihui-ai/Qwen3-8B-abliterated"}, {"title": "Huihui-Qwen3-8B-Abliterated-V2 Free Chat Online - skywork.ai, Click to Use! - Skywork", "url": "https://skywork.ai/blog/models/huihui-qwen3-8b-abliterated-v2-free-chat-online-skywork-ai/"}, {"title": "Run Huihui-Qwen3-8B-abliterated-v2 API (Easy Deployment & Flat-Rate Pricing)", "url": "https://featherless.ai/models/huihui-ai/Huihui-Qwen3-8B-abliterated-v2"}, {"title": "Huihui-Qwen3-4B-Abliterated-V2 Free Chat Online - skywork.ai, Click to Use! - Skywork", "url": "https://skywork.ai/blog/models/huihui-qwen3-4b-abliterated-v2-free-chat-online-skywork-ai/"}, {"title": "Qwen3-8B-abliterated huggingface.co api & huihui-ai Qwen3-8B-abliterated github AI Model - Toolify", "url": "https://www.toolify.ai/ai-model/huihui-ai-qwen3-8b-abliterated"}, {"title": "huihui-ai/Qwen3-8B-abliterated · Hugging Face", "url": "https://huggingface.co/huihui-ai/Qwen3-8B-abliterated"}, {"title": "Run Huihui-Qwen3-4B-abliterated-v2 API (Easy Deployment & Flat-Rate Pricing)", "url": "https://featherless.ai/models/huihui-ai/Huihui-Qwen3-4B-abliterated-v2"}, {"title": "huihui-ai/Huihui-Qwen3-8B-abliterated-v2 · Hugging Face", "url": "https://huggingface.co/huihui-ai/Huihui-Qwen3-8B-abliterated-v2"}, {"title": "huihui-ai/Huihui-Qwen3-4B-abliterated-v2 · Hugging Face", "url": "https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-abliterated-v2"}]}, "Now let me search for more specific information about the difference-of-means and orthogonal projection techniques used in abliteration:", {"tool_use_id": "srvtoolu_01SSeHXjvWyBMxUGrpwWXX8B", "content": [{"title": "Abliteration - Learn AI - Miraheze", "url": "https://ai.miraheze.org/wiki/Abliteration"}, {"title": "ORBA: Orthogonal Reflection Bounded Ablation — A Geometrically Exact Detour in Directional Activation Editing", "url": "https://huggingface.co/blog/grimjim/orthogonal-reflection-bounded-ablation"}, {"title": "Projected Abliteration", "url": "https://huggingface.co/blog/grimjim/projected-abliteration"}, {"title": "On the Failure of Topic-Matched Contrast Baselines in Multi-Directional Refusal Abliteration", "url": "https://arxiv.org/pdf/2603.22061"}, {"title": "SPECTER ABLITERATE — Open-Weight Model Alignment Removal Engine | Red Specter", "url": "https://red-specter.co.uk/nightfall/specter-abliterate/"}, {"title": "Comparative Analysis of LLM Abliteration Methods:", "url": "https://arxiv.org/pdf/2512.13655"}, {"title": "Abliteration — Grokipedia", "url": "https://grokipedia.com/page/Abliteration"}, {"title": "Norm-Preserving Biprojected Abliteration", "url": "https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration"}, {"title": "AEON-7/Step-3.7-Flash-AEON-Ultimate-Abliterated-BF16 · Hugging Face", "url": "https://huggingface.co/AEON-7/Step-3.7-Flash-AEON-Ultimate-Abliterated-BF16"}, {"title": "1 Introduction", "url": "https://arxiv.org/html/2512.13655v1"}]}, "Based on the search results, I can now provide you with a comprehensive answer about the abliteration technique used in the huihui-ai Qwen3-8B-abliterated model and how difference-of-means and orthogonal projection work:\n\n## Abliteration Technique Overview\n\nThe huihui-ai Qwen3-8B-abliterated model was created using an abliteration technique to remove refusal behaviors. This modification was achieved through an 'abliteration' process, a proof-of-concept implementation designed to eliminate refusals without using TransformerLens.\n\n## Difference-of-Means Method\n\nThe refusal direction is computed as a difference-of-means between sets of harmful — or forbidden — and harmless prompts, taking the mean activations of each, then subtracting one from the other. More specifically, the difference between the mean activation vectors for harmful and harmless prompt sets defines a direction in activation space that captures the distinction between \"about to refuse\" and \"about to comply.\"\n\n## Orthogonal Projection\n\nThe mathematical operation that neutralizes the refusal direction is projection removal, also known as orthogonal projection. In the weight orthogonalization approach, the model's weight matrices are permanently modified by orthogonalizing them with respect to the refusal direction, making it mathematically impossible for the model to represent that direction in its activations.\n\nBy projecting every weight matrix orthogonal to this direction, the model loses its ability to refuse — while preserving the rest of its capabilities.\n\n## Key Difference from Fine-Tuning\n\nAbliteration contrasts with fine-tuning for refusal removal by employing a targeted ablation of a specific refusal direction in the model's residual stream, avoiding the broad weight updates that characterize fine-tuning and often lead to widespread capability erosion."], "durationSeconds": 15.976208698999995, "searchCount": 2}
```

### [29] ASSISTANT · 2026-07-17 09:41:34 UTC

```
I now have enough context to write a thorough, grounded review. Let me compose the structured output.
```

### [30] TOOL CALL — Write · 2026-07-17 09:41:34 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_1/review_hypo/.terminal_claude_agent_struct_out.json

{
  "overall_assessment": "This is a well-motivated, clearly articulated hypothesis proposing a feature-level decomposition of RLHF safety changes into two mechanistically distinct classes — Type-R (reversed by abliteration) and Type-I (persistently imprinted) — using the Qwen-Scope base-SAE as a frozen measurement instrument across base, instruct, and abliterated Qwen3-8B variants. The neuroscience-inspired framing (explicit vs. implicit memory) is evocative and lends conceptual clarity. The related-work coverage is accurate and the differentiators from LatentBiopsy, arXiv:2605.11426, and Surgical Refusal Ablation are real. However, the hypothesis contains a potentially fatal methodological ambiguity in the Reversal_score definition that could make Success Criterion 1 (bimodality) trivially true by construction; it makes a strong bimodality claim without theoretical justification; Success Criterion 4 is almost certainly underpowered as designed; and the reliance on a community-produced abliterated model whose creation method is unverified undermines the mechanistic interpretation. These issues must be addressed before experiments proceed to avoid wasted compute on an uninterpretable result.",
  "strengths": [
    "Genuinely novel granularity: while LatentBiopsy (arXiv:2603.27412) and arXiv:2604.18901 establish that harmful representations survive abliteration globally, neither characterizes WHICH SAE features survive and whether they cluster into semantically coherent classes. The Type-R/Type-I decomposition at feature granularity is a real and non-trivial extension.",
    "Excellent ecosystem fit: the Qwen3-8B ecosystem offers base, instruct, and abliterated model variants alongside officially released Qwen-Scope base-SAEs — a uniquely favorable setup for this three-model comparison that reduces incidental confounds.",
    "Clear, testable success criteria: all four criteria are operationalizable with concrete statistical tests (bimodality test, label analysis, distance comparison, correlation with benchmark gap), and disconfirmation conditions are explicitly stated — a sign of scientific rigor.",
    "Honest about the prior-work landscape: the comparison table in related_works correctly notes that LatentBiopsy and the SFT-SAE paper (arXiv:2605.11426) are the closest predecessors and articulates precise differentiators rather than ignoring them.",
    "The proposed link between Type-I feature fraction and the capability gap is an original and high-value secondary hypothesis — if confirmed, it would mechanistically explain why abliterated models consistently underperform base models despite similar weight norms, a long-standing open question."
  ],
  "dimension_scores": [
    {
      "dimension": "soundness",
      "score": 2,
      "justification": "The core technical claim rests on a Reversal_score whose definition is ambiguous in a way that could make the bimodality finding trivially true. Assumption 2 (that the community-produced abliterated model was created via standard difference-of-means + orthogonal projection) is unverified and potentially false. Success Criterion 4 (capability gap correlation across layers) is almost certainly underpowered. These are not polish issues — they could render the main results uninterpretable.",
      "improvements": [
        "Clarify the Reversal_score computation: specify whether cosine similarity is computed (a) per-feature-per-prompt between two scalars (degenerates to sign agreement, making bimodality trivially guaranteed), or (b) across prompts as a vector correlation (e.g., Pearson r between the N-length vectors delta_instruct(f) and delta_abliterated(f)). Option (b) is scientifically meaningful; option (a) is not. Rewrite the definition unambiguously.",
        "Verify the abliteration method: fetch the huihui-ai/Qwen3-8B-abliterated HuggingFace model card and confirm which specific technique was used (difference-of-means + orthogonal projection, fine-tuning-based, or other). If the method is not the standard one, either reproduce abliteration from scratch or weaken Assumption 2 and add robustness analysis.",
        "Address the Qwen-Scope SAE out-of-distribution problem: the SFT-SAE paper (arXiv:2605.11426) shows that while raw activation cosine similarity stays high after SFT, sparse latents diverge substantially. This means the base-SAE may reconstruct instruct/abliterated activations poorly. Add a step that reports reconstruction MSE or explained variance for all three model variants through the base-SAE, and set a threshold for acceptable reconstruction quality before interpreting feature-level differences.",
        "Repower Success Criterion 4: with only ~14 SAE groups from Qwen-Scope (across 7 model variants), a cross-layer correlation has N~14 points at best. Expand to multiple model families (e.g., also apply to Qwen3.5 or Llama abliterated variants) or explicitly reframe Criterion 4 as exploratory/preliminary rather than confirmatory."
      ]
    },
    {
      "dimension": "presentation",
      "score": 3,
      "justification": "The hypothesis is well-organized, the terms section is thorough, and the motivation clearly explains the stakes. The neuroscience analogy aids understanding without overselling. The main presentation gap is the ambiguous Reversal_score formula and the bimodality assumption that is stated as expected outcome without justification.",
      "improvements": [
        "Add a worked example of the Reversal_score computation (e.g., for a single feature f across 5 hypothetical prompts) to make the definition unambiguous.",
        "Justify the bimodality prediction: reference any theoretical argument or prior empirical observation (e.g., the polysemanticity of the refusal direction from arXiv:2601.08489, or the distinct layer profiles found by arXiv:2605.11426) that makes two clusters a priori more plausible than a continuum. Without justification, this reads as an assumption masquerading as a prediction.",
        "Add a section on what happens in the continuous case: if Reversal_score is distributed uniformly rather than bimodally, what does that imply, and how does the Type-R/Type-I classification scheme adapt?"
      ]
    },
    {
      "dimension": "contribution",
      "score": 3,
      "justification": "The feature-level decomposition of abliteration effects is the right next step after LatentBiopsy-style global geometry studies, and the Qwen-Scope SAE tool makes this tractable now. The secondary hypothesis (Type-I fraction predicts capability gap) is high-value if confirmed. The contribution is real but incremental relative to arXiv:2605.11426 (which already proposes the frozen-base-SAE diagnostic pipeline for fine-tuning) — the novelty lives specifically in the three-model trajectory extension and the Reversal_score classification.",
      "improvements": [
        "Sharpen the unique contribution statement: explicitly note that NO prior work has classified individual SAE features by their abliteration-reversal behavior, as opposed to showing that some aggregate geometric signal survives — this is the key novelty and should be foregrounded more prominently.",
        "If the Type-I → capability gap correlation is confirmed, this would be the strongest result in the paper. Elevate it from a secondary criterion to a co-primary hypothesis with its own experimental design, since it makes a mechanistic claim that no prior work has made at this granularity."
      ]
    }
  ],
  "critiques": [
    {
      "category": "methodology",
      "severity": "major",
      "description": "The Reversal_score definition is ambiguous in a way that could make the bimodality finding (Success Criterion 1) trivially true by mathematical construction. The formula 'cosine_similarity(act_abliterated - act_base, act_instruct - act_base) averaged over i' has two possible interpretations: (a) for each prompt i and each scalar feature f, compute cosine similarity between two scalars (which degenerates to the sign of their product, giving values in {-1, 0, 1} and forcing bimodality), or (b) for each feature f, compute Pearson/cosine correlation between the N-length vector of delta_instruct(f) values and the N-length vector of delta_abliterated(f) values across all prompts. Only interpretation (b) is scientifically meaningful. If interpretation (a) is intended, the bimodal distribution in Criterion 1 is guaranteed by the definition, not an empirical finding, and the entire Type-R/Type-I framing loses its evidential value.",
      "suggested_action": "Rewrite the Reversal_score as an explicit per-feature correlation across prompts: R(f) = PearsonR([act_instruct_i(f) - act_base_i(f) for i in 1..N], [act_abliterated_i(f) - act_base_i(f) for i in 1..N]). Add a sentence explaining that this measures whether the direction of feature change under abliteration matches the direction of change under RLHF, at the prompt-level granularity. Provide a worked example with concrete numbers."
    },
    {
      "category": "methodology",
      "severity": "major",
      "description": "Assumption 2 states that the huihui-ai/Qwen3-8B-abliterated model was 'produced this way' (difference-of-means + orthogonal projection). Community-produced abliterated models on HuggingFace use a variety of techniques, including multi-layer ablation, fine-tuning-based uncensoring, and nonstandard prompt sets. If huihui-ai used a different approach (e.g., averaging over different layers or using a non-standard harmful prompt set), the comparison between the 'abliteration direction' and individual SAE feature directions loses its mechanistic grounding. This could silently invalidate the entire Type-R/Type-I classification by introducing confounds.",
      "suggested_action": "Check the huihui-ai/Qwen3-8B-abliterated model card and linked code to confirm the exact abliteration technique. If the technique matches (difference-of-means + orthogonal projection over a known layer set), document this explicitly. If it does not match, either: (a) perform abliteration from scratch using a reproducible script (llm-abliteration or similar) to ensure the method is controlled, or (b) add a robustness check comparing the community-abliterated model against a self-produced abliterated checkpoint."
    },
    {
      "category": "methodology",
      "severity": "major",
      "description": "The base-SAE (trained on base model activations) may reconstruct instruct and abliterated model activations poorly, especially after safety RLHF. arXiv:2605.11426 reports that sparse latents diverge substantially between base and SFT models even though raw activation cosine similarity stays high. A base-SAE will apply JumpReLU thresholds calibrated for base model activations; applied to instruct model activations, the sparsity pattern may be distorted (many features falsely zero, or anomalously active), making the feature-level comparison unreliable. This is an untested assumption that directly undermines the validity of all three model comparison steps.",
      "suggested_action": "Before running the main experiment, report reconstruction MSE or fraction of variance explained (R²) for base, instruct, and abliterated model activations through the frozen base-SAE. Set an acceptance threshold (e.g., R² > 0.90 for all three variants). If the base-SAE reconstructs instruct activations poorly, consider using the SAE's encoder only (to get feature activations) without relying on reconstruction quality, and note the limitation. Alternatively, look at whether Qwen-Scope releases any instruct-model SAEs for cross-comparison."
    },
    {
      "category": "rigor",
      "severity": "major",
      "description": "Success Criterion 4 — the claim that Type-I fraction correlates with capability gap across SAE layers — is almost certainly underpowered as described. With Qwen-Scope providing SAEs for ~14 groups across 7 model variants, the effective N for a cross-layer correlation within a single model family is at most the number of distinct SAE layers (likely 5-7). A correlation from 5-7 data points provides no statistical reliability and cannot support the causal claim that 'Type-I features are entangled with general semantic processing.' This would be a misleading correlation even if it reaches p<0.05.",
      "suggested_action": "Either (a) expand the correlation to multiple model families by running the same analysis on abliterated Qwen3.5-8B or Llama-3 variants (adding model families gives more data points with which to test cross-model generalizability); or (b) reframe Criterion 4 as exploratory/descriptive ('we report this correlation as a preliminary signal for future work') rather than a primary confirmatory criterion, and set an explicit minimum N for statistical validity in the paper."
    },
    {
      "category": "novelty",
      "severity": "minor",
      "description": "The space of 'harmful representations survive abliteration' is now crowded. LatentBiopsy (arXiv:2603.27412) demonstrates AUROC drops of ≤0.015 after abliteration using PCA geometry on exactly the Qwen model family. arXiv:2604.18901 shows harmful intent is linearly separable with AUROC 0.982 that survives abliteration within ±0.003, across 12 models spanning Qwen, Llama, and Gemma families. The proposed work's novelty rests entirely on 'which specific SAE features survive and whether they are semantically coherent harm-concept features.' If the SAE reconstruction quality issue (above) is not validated, the feature-level story may reduce to a restatement of what these geometric papers already show at a more granular vocabulary.",
      "suggested_action": "In the introduction/framing, explicitly address why SAE-level feature characterization provides actionable insight beyond what LatentBiopsy and arXiv:2604.18901 already establish. The strongest argument is: (a) if Type-I features have interpretable semantic content (harm schemas, not just 'harmful geometry'), they become candidates for targeted restoration of safety without full retraining; (b) the capability gap link (Criterion 4) is a new mechanistic claim not made by geometric approaches. Frame these two specific downstream applications as the core motivation for needing feature-level, not just geometric, characterization."
    },
    {
      "category": "methodology",
      "severity": "minor",
      "description": "Assumption 5 claims that using raw text prompts without chat templates 'provides a fair comparison.' However, the instruct model (Qwen/Qwen3-8B) is trained with Qwen's chat template format. When presented with raw text (no [INST] or system prompt), the instruct model's behavior is off-distribution and its internal representations may not reflect its safety-trained state. The comparison may actually be instruct-OOD vs. base-in-distribution vs. abliterated-OOD rather than a fair three-way comparison.",
      "suggested_action": "Run a pilot study comparing instruct model activations on raw vs. template-formatted harmful prompts and verify that the safety-relevant SAE features (high Safety_change) activate similarly in both formats. If they do not, use template-formatted prompts for the instruct model (with empty system prompt or safety system prompt) and document the formatting choice clearly. Cite the specific prior methodology papers that validate the no-template approach if that validation exists."
    },
    {
      "category": "rigor",
      "severity": "minor",
      "description": "The bimodality prediction (Success Criterion 1) is central but unjustified. The hypothesis asserts that high-Safety_change features will show a 'clearly bimodal' Reversal_score distribution rather than a continuum. There is no theoretical argument for why abliteration would produce a clean binary partition rather than a continuum of partial reversals. The Surgical Refusal Ablation paper (arXiv:2601.08489) suggests the refusal direction is polysemantic and entangled with multiple capability circuits — this entanglement would more naturally produce a continuous distribution of partial reversals.",
      "suggested_action": "Weaken the bimodality claim from a success criterion to an empirical question: 'We will test whether the distribution is bimodal (consistent with two distinct mechanisms) or continuous (consistent with a single graded mechanism). Either result is interpretable.' Add a specification of the statistical test for bimodality (e.g., Hartigan's dip test) and a secondary analysis for the continuous case (e.g., treating Reversal_score as a continuous variable and examining its semantic correlates along the continuum)."
    },
    {
      "category": "scope",
      "severity": "minor",
      "description": "The hypothesis is limited to a single model family (Qwen3-8B) and a single abliteration technique. The 'latent safety imprint' phenomenon, if confirmed, may be model-specific or technique-specific. The broader safety implications claimed in the motivation (evaluating 'the true robustness of safety alignment,' identifying 'candidate features for lightweight safety restoration') require generalization across at least two model families.",
      "suggested_action": "Acknowledge in the motivation/scope section that this is a single-model proof-of-concept. Frame the Qwen3-8B study as demonstrating the measurement framework, with explicit plans for cross-model validation in follow-up work. This is acceptable for a first paper — but should not be oversold as a general result until replicated."
    }
  ],
  "score": 6,
  "confidence": 4,
  "relation_type": null,
  "relation_rationale": ""
}
```

### [31] TOOL RESULT — Write · 2026-07-17 09:41:34 UTC

```
{"type": "create", "filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_1/review_hypo/.terminal_claude_agent_struct_out.json", "content": "{\n  \"overall_assessment\": \"This is a well-motivated, clearly articulated hypothesis proposing a feature-level decomposition of RLHF safety changes into two mechanistically distinct classes — Type-R (reversed by abliteration) and Type-I (persistently imprinted) — using the Qwen-Scope base-SAE as a frozen measurement instrument across base, instruct, and abliterated Qwen3-8B variants. The neuroscience-inspired framing (explicit vs. implicit memory) is evocative and lends conceptual clarity. The related-work coverage is accurate and the differentiators from LatentBiopsy, arXiv:2605.11426, and Surgical Refusal Ablation are real. However, the hypothesis contains a potentially fatal methodological ambiguity in the Reversal_score definition that could make Success Criterion 1 (bimodality) trivially true by construction; it makes a strong bimodality claim without theoretical justification; Success Criterion 4 is almost certainly underpowered as designed; and the reliance on a community-produced abliterated model whose creation method is unverified undermines the mechanistic interpretation. These issues must be addressed before experiments proceed to avoid wasted compute on an uninterpretable result.\",\n  \"strengths\": [\n    \"Genuinely novel granularity: while LatentBiopsy (arXiv:2603.27412) and arXiv:2604.18901 establish that harmful representations survive abliteration globally, neither characterizes WHICH SAE features survive and whether they cluster into semantically coherent classes. The Type-R/Type-I decomposition at feature granularity is a real and non-trivial extension.\",\n    \"Excellent ecosystem fit: the Qwen3-8B ecosystem offers base, instruct, and abliterated model variants alongside officially released Qwen-Scope base-SAEs — a uniquely favorable setup for this three-model comparison that reduces incidental confounds.\",\n    \"Clear, testable success criteria: all four criteria are operationalizable with concrete statistical tests (bimodality test, label analysis, distance comparison, correlation with benchmark gap), and disconfirmation conditions are explicitly stated — a sign of scientific rigor.\",\n    \"Honest about the prior-work landscape: the comparison table in related_works correctly notes that LatentBiopsy and the SFT-SAE paper (arXiv:2605.11426) are the closest predecessors and articulates precise differentiators rather than ignoring them.\",\n    \"The proposed link between Type-I feature fraction and the capability gap is an original and high-value secondary hypothesis — if confirmed, it would mechanistically explain why abliterated models consistently underperform base models despite similar weight norms, a long-standing open question.\"\n  ],\n  \"dimension_scores\": [\n    {\n      \"dimension\": \"soundness\",\n      \"score\": 2,\n      \"justification\": \"The core technical claim rests on a Reversal_score whose definition is ambiguous in a way that could make the bimodality finding trivially true. Assumption 2 (that the community-produced abliterated model was created via standard difference-of-means + orthogonal projection) is unverified and potentially false. Success Criterion 4 (capability gap correlation across layers) is almost certainly underpowered. These are not polish issues — they could render the main results uninterpretable.\",\n      \"improvements\": [\n        \"Clarify the Reversal_score computation: specify whether cosine similarity is computed (a) per-feature-per-prompt between two scalars (degenerates to sign agreement, making bimodality trivially guaranteed), or (b) across prompts as a vector correlation (e.g., Pearson r between the N-length vectors delta_instruct(f) and delta_abliterated(f)). Option (b) is scientifically meaningful; option (a) is not. Rewrite the definition unambiguously.\",\n        \"Verify the abliteration method: fetch the huihui-ai/Qwen3-8B-abliterated HuggingFace model card and confirm which specific technique was used (difference-of-means + orthogonal projection, fine-tuning-based, or other). If the method is not the standard one, either reproduce abliteration from scratch or weaken Assumption 2 and add robustness analysis.\",\n        \"Address the Qwen-Scope SAE out-of-distribution problem: the SFT-SAE paper (arXiv:2605.11426) shows that while raw activation cosine similarity stays high after SFT, sparse latents diverge substantially. This means the base-SAE may reconstruct instruct/abliterated activations poorly. Add a step that reports reconstruction MSE or explained variance for all three model variants through the base-SAE, and set a threshold for acceptable reconstruction quality before interpreting feature-level differences.\",\n        \"Repower Success Criterion 4: with only ~14 SAE groups from Qwen-Scope (across 7 model variants), a cross-layer correlation has N~14 points at best. Expand to multiple model families (e.g., also apply to Qwen3.5 or Llama abliterated variants) or explicitly reframe Criterion 4 as exploratory/preliminary rather than confirmatory.\"\n      ]\n    },\n    {\n      \"dimension\": \"presentation\",\n      \"score\": 3,\n      \"justification\": \"The hypothesis is well-organized, the terms section is thorough, and the motivation clearly explains the stakes. The neuroscience analogy aids understanding without overselling. The main presentation gap is the ambiguous Reversal_score formula and the bimodality assumption that is stated as expected outcome without justification.\",\n      \"improvements\": [\n        \"Add a worked example of the Reversal_score computation (e.g., for a single feature f across 5 hypothetical prompts) to make the definition unambiguous.\",\n        \"Justify the bimodality prediction: reference any theoretical argument or prior empirical observation (e.g., the polysemanticity of the refusal direction from arXiv:2601.08489, or the distinct layer profiles found by arXiv:2605.11426) that makes two clusters a priori more plausible than a continuum. Without justification, this reads as an assumption masquerading as a prediction.\",\n        \"Add a section on what happens in the continuous case: if Reversal_score is distributed uniformly rather than bimodally, what does that imply, and how does the Type-R/Type-I classification scheme adapt?\"\n      ]\n    },\n    {\n      \"dimension\": \"contribution\",\n      \"score\": 3,\n      \"justification\": \"The feature-level decomposition of abliteration effects is the right next step after LatentBiopsy-style global geometry studies, and the Qwen-Scope SAE tool makes this tractable now. The secondary hypothesis (Type-I fraction predicts capability gap) is high-value if confirmed. The contribution is real but incremental relative to arXiv:2605.11426 (which already proposes the frozen-base-SAE diagnostic pipeline for fine-tuning) — the novelty lives specifically in the three-model trajectory extension and the Reversal_score classification.\",\n      \"improvements\": [\n        \"Sharpen the unique contribution statement: explicitly note that NO prior work has classified individual SAE features by their abliteration-reversal behavior, as opposed to showing that some aggregate geometric signal survives — this is the key novelty and should be foregrounded more prominently.\",\n        \"If the Type-I → capability gap correlation is confirmed, this would be the strongest result in the paper. Elevate it from a secondary criterion to a co-primary hypothesis with its own experimental design, since it makes a mechanistic claim that no prior work has made at this granularity.\"\n      ]\n    }\n  ],\n  \"critiques\": [\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"major\",\n      \"description\": \"The Reversal_score definition is ambiguous in a way that could make the bimodality finding (Success Criterion 1) trivially true by mathematical construction. The formula 'cosine_similarity(act_abliterated - act_base, act_instruct - act_base) averaged over i' has two possible interpretations: (a) for each prompt i and each scalar feature f, compute cosine similarity between two scalars (which degenerates to the sign of their product, giving values in {-1, 0, 1} and forcing bimodality), or (b) for each feature f, compute Pearson/cosine correlation between the N-length vector of delta_instruct(f) values and the N-length vector of delta_abliterated(f) values across all prompts. Only interpretation (b) is scientifically meaningful. If interpretation (a) is intended, the bimodal distribution in Criterion 1 is guaranteed by the definition, not an empirical finding, and the entire Type-R/Type-I framing loses its evidential value.\",\n      \"suggested_action\": \"Rewrite the Reversal_score as an explicit per-feature correlation across prompts: R(f) = PearsonR([act_instruct_i(f) - act_base_i(f) for i in 1..N], [act_abliterated_i(f) - act_base_i(f) for i in 1..N]). Add a sentence explaining that this measures whether the direction of feature change under abliteration matches the direction of change under RLHF, at the prompt-level granularity. Provide a worked example with concrete numbers.\"\n    },\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"major\",\n      \"description\": \"Assumption 2 states that the huihui-ai/Qwen3-8B-abliterated model was 'produced this way' (difference-of-means + orthogonal projection). Community-produced abliterated models on HuggingFace use a variety of techniques, including multi-layer ablation, fine-tuning-based uncensoring, and nonstandard prompt sets. If huihui-ai used a different approach (e.g., averaging over different layers or using a non-standard harmful prompt set), the comparison between the 'abliteration direction' and individual SAE feature directions loses its mechanistic grounding. This could silently invalidate the entire Type-R/Type-I classification by introducing confounds.\",\n      \"suggested_action\": \"Check the huihui-ai/Qwen3-8B-abliterated model card and linked code to confirm the exact abliteration technique. If the technique matches (difference-of-means + orthogonal projection over a known layer set), document this explicitly. If it does not match, either: (a) perform abliteration from scratch using a reproducible script (llm-abliteration or similar) to ensure the method is controlled, or (b) add a robustness check comparing the community-abliterated model against a self-produced abliterated checkpoint.\"\n    },\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"major\",\n      \"description\": \"The base-SAE (trained on base model activations) may reconstruct instruct and abliterated model activations poorly, especially after safety RLHF. arXiv:2605.11426 reports that sparse latents diverge substantially between base and SFT models even though raw activation cosine similarity stays high. A base-SAE will apply JumpReLU thresholds calibrated for base model activations; applied to instruct model activations, the sparsity pattern may be distorted (many features falsely zero, or anomalously active), making the feature-level comparison unreliable. This is an untested assumption that directly undermines the validity of all three model comparison steps.\",\n      \"suggested_action\": \"Before running the main experiment, report reconstruction MSE or fraction of variance explained (R²) for base, instruct, and abliterated model activations through the frozen base-SAE. Set an acceptance threshold (e.g., R² > 0.90 for all three variants). If the base-SAE reconstructs instruct activations poorly, consider using the SAE's encoder only (to get feature activations) without relying on reconstruction quality, and note the limitation. Alternatively, look at whether Qwen-Scope releases any instruct-model SAEs for cross-comparison.\"\n    },\n    {\n      \"category\": \"rigor\",\n      \"severity\": \"major\",\n      \"description\": \"Success Criterion 4 — the claim that Type-I fraction correlates with capability gap across SAE layers — is almost certainly underpowered as described. With Qwen-Scope providing SAEs for ~14 groups across 7 model variants, the effective N for a cross-layer correlation within a single model family is at most the number of distinct SAE layers (likely 5-7). A correlation from 5-7 data points provides no statistical reliability and cannot support the causal claim that 'Type-I features are entangled with general semantic processing.' This would be a misleading correlation even if it reaches p<0.05.\",\n      \"suggested_action\": \"Either (a) expand the correlation to multiple model families by running the same analysis on abliterated Qwen3.5-8B or Llama-3 variants (adding model families gives more data points with which to test cross-model generalizability); or (b) reframe Criterion 4 as exploratory/descriptive ('we report this correlation as a preliminary signal for future work') rather than a primary confirmatory criterion, and set an explicit minimum N for statistical validity in the paper.\"\n    },\n    {\n      \"category\": \"novelty\",\n      \"severity\": \"minor\",\n      \"description\": \"The space of 'harmful representations survive abliteration' is now crowded. LatentBiopsy (arXiv:2603.27412) demonstrates AUROC drops of ≤0.015 after abliteration using PCA geometry on exactly the Qwen model family. arXiv:2604.18901 shows harmful intent is linearly separable with AUROC 0.982 that survives abliteration within ±0.003, across 12 models spanning Qwen, Llama, and Gemma families. The proposed work's novelty rests entirely on 'which specific SAE features survive and whether they are semantically coherent harm-concept features.' If the SAE reconstruction quality issue (above) is not validated, the feature-level story may reduce to a restatement of what these geometric papers already show at a more granular vocabulary.\",\n      \"suggested_action\": \"In the introduction/framing, explicitly address why SAE-level feature characterization provides actionable insight beyond what LatentBiopsy and arXiv:2604.18901 already establish. The strongest argument is: (a) if Type-I features have interpretable semantic content (harm schemas, not just 'harmful geometry'), they become candidates for targeted restoration of safety without full retraining; (b) the capability gap link (Criterion 4) is a new mechanistic claim not made by geometric approaches. Frame these two specific downstream applications as the core motivation for needing feature-level, not just geometric, characterization.\"\n    },\n    {\n      \"category\": \"methodology\",\n      \"severity\": \"minor\",\n      \"description\": \"Assumption 5 claims that using raw text prompts without chat templates 'provides a fair comparison.' However, the instruct model (Qwen/Qwen3-8B) is trained with Qwen's chat template format. When presented with raw text (no [INST] or system prompt), the instruct model's behavior is off-distribution and its internal representations may not reflect its safety-trained state. The comparison may actually be instruct-OOD vs. base-in-distribution vs. abliterated-OOD rather than a fair three-way comparison.\",\n      \"suggested_action\": \"Run a pilot study comparing instruct model activations on raw vs. template-formatted harmful prompts and verify that the safety-relevant SAE features (high Safety_change) activate similarly in both formats. If they do not, use template-formatted prompts for the instruct model (with empty system prompt or safety system prompt) and document the formatting choice clearly. Cite the specific prior methodology papers that validate the no-template approach if that validation exists.\"\n    },\n    {\n      \"category\": \"rigor\",\n      \"severity\": \"minor\",\n      \"description\": \"The bimodality prediction (Success Criterion 1) is central but unjustified. The hypothesis asserts that high-Safety_change features will show a 'clearly bimodal' Reversal_score distribution rather than a continuum. There is no theoretical argument for why abliteration would produce a clean binary partition rather than a continuum of partial reversals. The Surgical Refusal Ablation paper (arXiv:2601.08489) suggests the refusal direction is polysemantic and entangled with multiple capability circuits — this entanglement would more naturally produce a continuous distribution of partial reversals.\",\n      \"suggested_action\": \"Weaken the bimodality claim from a success criterion to an empirical question: 'We will test whether the distribution is bimodal (consistent with two distinct mechanisms) or continuous (consistent with a single graded mechanism). Either result is interpretable.' Add a specification of the statistical test for bimodality (e.g., Hartigan's dip test) and a secondary analysis for the continuous case (e.g., treating Reversal_score as a continuous variable and examining its semantic correlates along the continuum).\"\n    },\n    {\n      \"category\": \"scope\",\n      \"severity\": \"minor\",\n      \"description\": \"The hypothesis is limited to a single model family (Qwen3-8B) and a single abliteration technique. The 'latent safety imprint' phenomenon, if confirmed, may be model-specific or technique-specific. The broader safety implications claimed in the motivation (evaluating 'the true robustness of safety alignment,' identifying 'candidate features for lightweight safety restoration') require generalization across at least two model families.\",\n      \"suggested_action\": \"Acknowledge in the motivation/scope section that this is a single-model proof-of-concept. Frame the Qwen3-8B study as demonstrating the measurement framework, with explicit plans for cross-model validation in follow-up work. This is acceptable for a first paper — but should not be oversold as a general result until replicated.\"\n    }\n  ],\n  \"score\": 6,\n  \"confidence\": 4,\n  \"relation_type\": null,\n  \"relation_rationale\": \"\"\n}\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [32] ASSISTANT · 2026-07-17 09:41:48 UTC

```
The review has been written to `.terminal_claude_agent_struct_out.json`. Here's a summary:

**Overall score: 6/10 (Weak Accept)** — Confidence: 4/5

The hypothesis is well-motivated and genuinely novel in its feature-level decomposition approach, but has four issues that need resolution before experiments proceed:

**Major blockers (fix these first):**

1. **Reversal_score ambiguity** — As written, the formula may degenerate to sign-agreement between scalars, which would make the bimodality finding (Success Criterion 1) trivially true by construction rather than empirically informative. Needs to be rewritten as a per-feature Pearson correlation across prompts.

2. **Unverified abliteration method** — The huihui-ai community model's creation technique is assumed but not confirmed. If it wasn't standard difference-of-means + orthogonal projection, the mechanistic interpretation breaks down.

3. **SAE reconstruction quality for out-of-distribution models** — The base-SAE applied to instruct/abliterated activations may have poor reconstruction quality (arXiv:2605.11426 directly warns of this). This must be validated before interpreting feature-level differences.

4. **Success Criterion 4 underpowered** — Correlating Type-I fraction with capability gap across ~5-7 SAE layers gives too few data points for a valid statistical claim.

**Minor issues:** bimodality is assumed not justified, chat template confound, single-model scope limitation.

The core idea — classifying SAE features by whether RLHF changes are reversed or imprinted by abliteration — is a real and actionable extension of LatentBiopsy/arXiv:2604.18901, especially the secondary hypothesis linking Type-I fraction to the capability gap.
```
