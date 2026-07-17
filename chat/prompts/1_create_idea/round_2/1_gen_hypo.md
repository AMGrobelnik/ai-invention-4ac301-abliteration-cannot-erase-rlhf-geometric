# gen_hypo_1 — create_idea

> Phase: `hypo_loop` · round 2 · `gen_hypo`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_hypo_1` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-07-17 09:41:59 UTC

````
<ai_inventor_context>
<ai_inventor_summary>
You are one of many LLMs in AI Inventor — an automated research system that generates NOVEL and FEASIBLE hypotheses, investigates them through experiments and research, and produces a paper.

Your output feeds other LLMs downstream. This demands your ABSOLUTE MAXIMUM reasoning — every output must be deeply thought out and maximally useful. Surface-level responses waste downstream computation.
</ai_inventor_summary>

<your_role>
YOU ARE: A hypothesis generator (Step 2.1: GEN_HYPO — UNSEEDED mode)

Pipeline: GEN_HYPO (you) → INVENTION_LOOP → GEN_PAPER_REPO

You received a AII prompt. No external seeds — generate a novel hypothesis from your own reasoning and web research.

Your hypothesis will enter the invention loop (propose → execute → narrate) → the results become a paper + GitHub repo.
It MUST be GENUINELY NOVEL (validated against related work) and FEASIBLE TO TEST (within computational/data/tooling constraints provided).
Vague or incremental hypothesis → wasted computation across the entire pipeline.
</your_role>
</ai_inventor_context>

<strategic_mindset>
You are competing with human researchers.

YOUR ADVANTAGE: Breadth across many fields (information theory, ecology, economics, physics, cognitive science, program synthesis, etc.). No single human has this breadth.

HUMAN ADVANTAGE: Deep expertise in their specific field — they know every paper, every failed attempt, every subtle reason "obvious" ideas don't work.

HOW TO WIN: Don't create variants within their field — they'll always recognize those. Find unexpected connections ACROSS fields no single expert would think of.

NOVELTY BAR: An expert should say "I never thought of approaching it THAT way" — not "that's like paper X with a twist." If your idea lives in a crowded neighborhood of similar approaches, it's NOT novel enough.

NO TIME PRESSURE: Exploring 5-6 directions and abandoning all is a SUCCESSFUL process. Settling for a mediocre idea because you already spent so long researching it is a FAILED process.
</strategic_mindset>

<principles>
1. NOVEL - genuinely new mechanism/principle, not incremental. If you have to argue why it's different, it's NOT novel enough.
2. FEASIBLE - testable within the provided compute, data, and tooling
3. CROSS-FIELD - leverage connections across distant domains
4. RIGOROUS - consider what evidence would support OR refute it
5. PRECISE - clear language, no unnecessary jargon
</principles>

<common_mistakes_to_avoid>
Critical pitfalls from past runs. EXPLICITLY CHECK FOR EACH ONE.

**1. Incremental Recombination Disguised as Novelty**
"Apply known method X to known domain Y" is engineering, not conceptual novelty. Your idea needs a new mechanism/principle/insight — not just a new pairing of existing things.
CHECK: If describable as "A but with B" where A and B both exist, it's recombination. What is the genuinely new IDEA?

**2. Ignoring Resource Constraints**
Every hypothesis MUST be testable with available compute, data, and tools.
CHECK: "Can this be implemented with the specific resources listed? What exact data/compute/tools do I need, and are they available?"

**3. Shallow Search Leading to False Novelty**
The same concept often exists under different terminology, in different fields, or framed differently. Searching only your own phrasing and concluding novelty is the MOST dangerous mistake.

CHECK — For every promising hypothesis:
a) Search 5-6 semantically different phrasings within the field
b) Strip to the CORE MECHANISM and search 8-10 unrelated fields (e.g., "MDL-based complexity selection" → search neural architecture search, program synthesis, Bayesian model selection) — the same principle often exists under different names
c) Search for failed/negative results ("limitations", "does not improve")
d) Search in plain English without jargon
If a paper does the same thing under a different name, it's NOT novel.

**4. Rationalizing Overlapping Prior Work**
When you find similar work, do NOT rationalize minor differences as novelty. Two common traps:

FRAMEWORK PORTING: "Nobody did this in MY framework" — if the core mechanism exists in any context (different algorithm, different ensemble type, different field), porting it is engineering, not novelty.

GAP-FILLING: Papers A, B, C each cover variants → you propose the missing combination. An expert would say "obviously someone will do that eventually."

CHECK: Strip your idea to its core mechanism. Search if that mechanism exists ANYWHERE — any framework, any field, any algorithm family. If yes, ABANDON. Don't salvage by narrowing scope or listing "critical differences."

**5. Anchoring Bias**
Once invested in a direction, you'll unconsciously downplay overlap and inflate minor differences into "key differentiators." This feels like thoroughness but is actually defensiveness.

WARNING SIGNS: listing "critical differences" instead of reconsidering; reluctance to "waste" prior search effort; refining the SAME idea instead of exploring different ones; differentiators about context/framework rather than core mechanism.

CHECK: If you found even 1 paper with a similar core mechanism, ABANDON. The best hypotheses rarely come from your first direction. Each abandonment is progress.

**6. Relying on Search Snippets Without Fetching**
Search snippets are NOT enough to assess overlap or understand an approach. The actual mechanism and limitations are only in the full text.
CHECK: FETCH and read any potentially relevant result. Don't assess novelty from titles and snippets alone.

**7. Same-Neighborhood Pivoting**
Replacing one idea with a variant in the same conceptual space is NOT a genuine pivot. If all your directions are "[different adjective] + [same core concept]", you haven't actually explored.

CHECK: Would a single expert in that subfield have thought of ALL your directions? If yes, bring in a mechanism or framing from a completely unrelated field. That's where genuine novelty lives.
</common_mistakes_to_avoid>

<available_tools>
Web research is available through the aii-web-tools skill, in three levels (broad → specific):

1. web search — Returns titles, URLs, snippets. Use first to discover and scan the landscape.
2. web fetch — Reads a page and returns its content as markdown (HTML or PDF). Use to understand a source. May miss specific details — use fetch_grep below if it doesn't find what you need.
3. fetch_grep — Regex search over a page/PDF's full text. Returns exact matching sections with context. Use for precise details, exact numbers, methodology, or PDFs.

Workflow: search → fetch (understand) → fetch_grep (extract specifics).
</available_tools>

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

<task_preview>
You will generate 1 novel groundbreaking research hypothesis in the AII prompt provided in the accompanying user message.
</task_preview>

<YOUR_AII_PROMPT>
Your AII prompt — the research prompt to invent within — is provided as a SEPARATE user message in this turn, immediately following this one. Treat that message as the definition of what to generate a hypothesis for.
</YOUR_AII_PROMPT>

<hypothesis_inspiration>
<YOUR_INSPIRATION>
Human researchers overspecialize — they know their domain deeply but lack breadth to see when other fields have already solved analogous problems. Your advantage is breadth. Only propose a cross-domain transfer if it concretely outperforms existing approaches in this domain. Avoid handwavy analogies — if the imported method is vaguer or weaker than what domain experts already use, it's not worth proposing.

Explore cross-domain inspiration at three levels, from abstract to concrete. At each level, consider both established and recent developments — with slight priority for newer work, which tends to leverage more powerful tools and be less widely known.

1. CONCEPTUAL: Borrow high-level ideas, framings, or design philosophies from distant fields.
   What mental model or approach from another domain suggests a novel angle on this problem?

2. PROCEDURAL: Adapt specific problem-solving processes from other domains.
   What workflow, iterative strategy, or pipeline used elsewhere could restructure how this problem is attacked?

3. METHODOLOGICAL: Import concrete methods directly from other fields with minimal modification.
   What algorithm, formula, or technique from a different domain applies here as-is or with adaptation?

Cast wide — draw from ANY field, not just these examples: ecology, economics, physics, linguistics, game theory, control theory, materials science, cognitive science, epidemiology. The best hypotheses often come from Level 2-3 transfers that experts in the field would never encounter.
</YOUR_INSPIRATION>
</hypothesis_inspiration>

<available_resources>
<software_constraints>
- Python only implementation
- Python standard library and all popular PyPI packages available (numpy, pandas, scikit-learn, scipy, matplotlib, requests, etc.)
- Local parallelism encouraged: multiprocessing, asyncio, threading — see aii-parallel-computing skill
- LLM API calls must go through OpenRouter only (no direct OpenAI, Anthropic, etc.)
- **HARD LIMIT**: Maximum $10 USD total spend on LLM API calls (OpenRouter). Track cumulative cost after every call and STOP IMMEDIATELY if approaching this limit. Never exceed this budget under any circumstances.
</software_constraints>

<skills>
Skills are self-contained capabilities with instructions, context, and tools.

- aii-web-tools: Web search (Serper), page/PDF fetch as markdown, regex grep over page/PDF text
- aii-semscholar-bib: Batch-fetch BibTeX from Semantic Scholar
- aii-openrouter-llms: Search and call 300+ LLMs via OpenRouter
- aii-hf-datasets: Search, preview, download HuggingFace datasets
- aii-owid-datasets: Search and load Our World in Data tables
- aii-lean: Compile/verify Lean 4 code, Mathlib search, tactic suggestions
- aii-image-gen: Generate/edit images via Gemini 3 Pro Image (Nano Banana Pro)
- aii-json: Validate JSON against schemas, generate mini/preview variants
- aii-paper-writing: Academic paper structure, bibliography, citations
- aii-paper-to-latex: Assemble LaTeX papers and compile to PDF
- aii-parallel-computing: GPU acceleration, CPU parallelism, async I/O
- aii-python: Python coding standards for experiment scripts
- aii-use-hardware: Detect CPU/RAM/GPU, memory-safe processing
- aii-long-running-tasks: Gradual scaling pattern for long-running tasks
- aii-colab: Google Colab runtime constraints for notebooks
- aii-file-size-limit: Check and split oversized output files
</skills>
</available_resources>

<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. Use it for the field's landscape, prior work, open problems, dead ends, and what counts as a genuinely novel contribution — read it BEFORE brainstorming and during the novelty check.

- **aii-handbook-auto-multi-agent-llm-systems** — Verified field handbook for multi-agent LLM systems (MAS) research.
</available_domain_handbooks>

<time_budgets>

Each artifact executor has a fixed time budget (including writing code, debugging, testing, and fixing errors):

- research: 3h
- dataset: 6h
- experiment: 6h
- evaluation: 3h
- proof: 3h

</time_budgets>

<YOUR_TASK>
Generate 1 novel groundbreaking research hypothesis in the AII prompt that is feasible with the above constraints.

<web_research_process>
Read and STRICTLY follow these skills: aii-web-tools.

1. DIVERGE: Brainstorm 5-7 diverse directions WITHOUT searching.
   Think across fields — what techniques from unrelated domains (ecology, economics, physics,
   linguistics, game theory, etc.) could inspire a novel mechanism? What assumptions does the field
   take for granted? Diversity matters more than depth here.

2. SEARCH: Web search for a high-level overview of each direction.
   What similar approaches exist? Is this genuinely novel or incremental? Remember: snippets
   are NOT enough for detailed understanding — treat search as discovery only.

3. FETCH & READ: MUST fetch any potentially relevant URL — you cannot assess novelty from
   snippets alone. Use the aii-web-tools skill:
   - fetch a page for high-level understanding of HTML pages
   - fetch_grep for exact details, methodology, or PDFs
   Prioritize recent papers closest to your idea. If you find significant overlap, PIVOT.

4. ADVERSARIAL NOVELTY CHECK: Actively try to DISPROVE novelty. Most important step.
   Run the FULL search checklist from <common_mistakes_to_avoid> mistake 3 — within-field
   rephrasings, cross-field core-mechanism search, failed/negative results, plain English.
   Ask: "Is the core insight of your hypothesis new, or known things in a new wrapper?"
   "Would an expert find this genuinely surprising?"
   MANDATORY SELF-CHECK: State the core mechanism in one sentence. Does it exist in ANY
   algorithm, framework, or field? If yes — even in a different framework — ABANDON.

5. FEASIBILITY CHECK: Verify your hypothesis is testable with provided resources. What specific data/compute/tools
   needed? All available within constraints?

6. ABANDON or PROCEED:
   ABANDON if: 2+ similar papers exist; you need to argue "critical differences"; core mechanism
   exists in any context.
   Abandoning is progress — go back to step 1 in a genuinely DIFFERENT direction (not a variant).
   PROCEED only if novelty is SELF-EVIDENT — an expert would immediately see it's new without
   explanation.

7. ITERATE: Expect to repeat steps 1-6 multiple times. The first few directions will likely be
   non-novel. This is normal. Don't settle for your first idea just because you've invested time.

<CRITICAL>We want SCIENTIFIC novelty (new mechanism, principle, or insight — the contribution is
knowledge), NOT application novelty (known methods applied to a new domain — the contribution is a
product). If an expert would say "clever engineering but known science," keep searching.
Hypothesis must be feasible within available resources.</CRITICAL>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>
</web_research_process>

Prioritize simplicity. Use concise, approachable language. The explanation should be fully self-contained.
</YOUR_TASK>

<previous_hypothesis>
Your hypothesis from the previous iteration. The reviewer evaluated it below.

hypothesis_id: gen_hypo_1
model: claude-sonnet-4-6
is_seeded: false
seeds: []
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

<previous_review_feedback>
A reviewer evaluated your previous hypothesis and provided the feedback below.

IMPORTANT: Do NOT generate a completely new hypothesis. Take the previous hypothesis above and
REVISE it to address the feedback. Keep what works, fix what was criticized.

You MUST address ALL the critiques. Do NOT repeat the same mistakes.

kind: reviewer_feedback
id: review_hypo_fef33d6a7eed
overall_assessment: >-
  This is a well-motivated, clearly articulated hypothesis proposing a feature-level decomposition of RLHF safety changes
  into two mechanistically distinct classes — Type-R (reversed by abliteration) and Type-I (persistently imprinted) — using
  the Qwen-Scope base-SAE as a frozen measurement instrument across base, instruct, and abliterated Qwen3-8B variants. The
  neuroscience-inspired framing (explicit vs. implicit memory) is evocative and lends conceptual clarity. The related-work
  coverage is accurate and the differentiators from LatentBiopsy, arXiv:2605.11426, and Surgical Refusal Ablation are real.
  However, the hypothesis contains a potentially fatal methodological ambiguity in the Reversal_score definition that could
  make Success Criterion 1 (bimodality) trivially true by construction; it makes a strong bimodality claim without theoretical
  justification; Success Criterion 4 is almost certainly underpowered as designed; and the reliance on a community-produced
  abliterated model whose creation method is unverified undermines the mechanistic interpretation. These issues must be addressed
  before experiments proceed to avoid wasted compute on an uninterpretable result.
strengths:
- >-
  Genuinely novel granularity: while LatentBiopsy (arXiv:2603.27412) and arXiv:2604.18901 establish that harmful representations
  survive abliteration globally, neither characterizes WHICH SAE features survive and whether they cluster into semantically
  coherent classes. The Type-R/Type-I decomposition at feature granularity is a real and non-trivial extension.
- >-
  Excellent ecosystem fit: the Qwen3-8B ecosystem offers base, instruct, and abliterated model variants alongside officially
  released Qwen-Scope base-SAEs — a uniquely favorable setup for this three-model comparison that reduces incidental confounds.
- >-
  Clear, testable success criteria: all four criteria are operationalizable with concrete statistical tests (bimodality test,
  label analysis, distance comparison, correlation with benchmark gap), and disconfirmation conditions are explicitly stated
  — a sign of scientific rigor.
- >-
  Honest about the prior-work landscape: the comparison table in related_works correctly notes that LatentBiopsy and the SFT-SAE
  paper (arXiv:2605.11426) are the closest predecessors and articulates precise differentiators rather than ignoring them.
- >-
  The proposed link between Type-I feature fraction and the capability gap is an original and high-value secondary hypothesis
  — if confirmed, it would mechanistically explain why abliterated models consistently underperform base models despite similar
  weight norms, a long-standing open question.
dimension_scores:
- dimension: soundness
  score: 2
  justification: >-
    The core technical claim rests on a Reversal_score whose definition is ambiguous in a way that could make the bimodality
    finding trivially true. Assumption 2 (that the community-produced abliterated model was created via standard difference-of-means
    + orthogonal projection) is unverified and potentially false. Success Criterion 4 (capability gap correlation across layers)
    is almost certainly underpowered. These are not polish issues — they could render the main results uninterpretable.
  improvements:
  - >-
    Clarify the Reversal_score computation: specify whether cosine similarity is computed (a) per-feature-per-prompt between
    two scalars (degenerates to sign agreement, making bimodality trivially guaranteed), or (b) across prompts as a vector
    correlation (e.g., Pearson r between the N-length vectors delta_instruct(f) and delta_abliterated(f)). Option (b) is scientifically
    meaningful; option (a) is not. Rewrite the definition unambiguously.
  - >-
    Verify the abliteration method: fetch the huihui-ai/Qwen3-8B-abliterated HuggingFace model card and confirm which specific
    technique was used (difference-of-means + orthogonal projection, fine-tuning-based, or other). If the method is not the
    standard one, either reproduce abliteration from scratch or weaken Assumption 2 and add robustness analysis.
  - >-
    Address the Qwen-Scope SAE out-of-distribution problem: the SFT-SAE paper (arXiv:2605.11426) shows that while raw activation
    cosine similarity stays high after SFT, sparse latents diverge substantially. This means the base-SAE may reconstruct
    instruct/abliterated activations poorly. Add a step that reports reconstruction MSE or explained variance for all three
    model variants through the base-SAE, and set a threshold for acceptable reconstruction quality before interpreting feature-level
    differences.
  - >-
    Repower Success Criterion 4: with only ~14 SAE groups from Qwen-Scope (across 7 model variants), a cross-layer correlation
    has N~14 points at best. Expand to multiple model families (e.g., also apply to Qwen3.5 or Llama abliterated variants)
    or explicitly reframe Criterion 4 as exploratory/preliminary rather than confirmatory.
- dimension: presentation
  score: 3
  justification: >-
    The hypothesis is well-organized, the terms section is thorough, and the motivation clearly explains the stakes. The neuroscience
    analogy aids understanding without overselling. The main presentation gap is the ambiguous Reversal_score formula and
    the bimodality assumption that is stated as expected outcome without justification.
  improvements:
  - >-
    Add a worked example of the Reversal_score computation (e.g., for a single feature f across 5 hypothetical prompts) to
    make the definition unambiguous.
  - >-
    Justify the bimodality prediction: reference any theoretical argument or prior empirical observation (e.g., the polysemanticity
    of the refusal direction from arXiv:2601.08489, or the distinct layer profiles found by arXiv:2605.11426) that makes two
    clusters a priori more plausible than a continuum. Without justification, this reads as an assumption masquerading as
    a prediction.
  - >-
    Add a section on what happens in the continuous case: if Reversal_score is distributed uniformly rather than bimodally,
    what does that imply, and how does the Type-R/Type-I classification scheme adapt?
- dimension: contribution
  score: 3
  justification: >-
    The feature-level decomposition of abliteration effects is the right next step after LatentBiopsy-style global geometry
    studies, and the Qwen-Scope SAE tool makes this tractable now. The secondary hypothesis (Type-I fraction predicts capability
    gap) is high-value if confirmed. The contribution is real but incremental relative to arXiv:2605.11426 (which already
    proposes the frozen-base-SAE diagnostic pipeline for fine-tuning) — the novelty lives specifically in the three-model
    trajectory extension and the Reversal_score classification.
  improvements:
  - >-
    Sharpen the unique contribution statement: explicitly note that NO prior work has classified individual SAE features by
    their abliteration-reversal behavior, as opposed to showing that some aggregate geometric signal survives — this is the
    key novelty and should be foregrounded more prominently.
  - >-
    If the Type-I → capability gap correlation is confirmed, this would be the strongest result in the paper. Elevate it from
    a secondary criterion to a co-primary hypothesis with its own experimental design, since it makes a mechanistic claim
    that no prior work has made at this granularity.
critiques:
- id: ''
  category: methodology
  severity: major
  description: >-
    The Reversal_score definition is ambiguous in a way that could make the bimodality finding (Success Criterion 1) trivially
    true by mathematical construction. The formula 'cosine_similarity(act_abliterated - act_base, act_instruct - act_base)
    averaged over i' has two possible interpretations: (a) for each prompt i and each scalar feature f, compute cosine similarity
    between two scalars (which degenerates to the sign of their product, giving values in {-1, 0, 1} and forcing bimodality),
    or (b) for each feature f, compute Pearson/cosine correlation between the N-length vector of delta_instruct(f) values
    and the N-length vector of delta_abliterated(f) values across all prompts. Only interpretation (b) is scientifically meaningful.
    If interpretation (a) is intended, the bimodal distribution in Criterion 1 is guaranteed by the definition, not an empirical
    finding, and the entire Type-R/Type-I framing loses its evidential value.
  suggested_action: >-
    Rewrite the Reversal_score as an explicit per-feature correlation across prompts: R(f) = PearsonR([act_instruct_i(f) -
    act_base_i(f) for i in 1..N], [act_abliterated_i(f) - act_base_i(f) for i in 1..N]). Add a sentence explaining that this
    measures whether the direction of feature change under abliteration matches the direction of change under RLHF, at the
    prompt-level granularity. Provide a worked example with concrete numbers.
- id: ''
  category: methodology
  severity: major
  description: >-
    Assumption 2 states that the huihui-ai/Qwen3-8B-abliterated model was 'produced this way' (difference-of-means + orthogonal
    projection). Community-produced abliterated models on HuggingFace use a variety of techniques, including multi-layer ablation,
    fine-tuning-based uncensoring, and nonstandard prompt sets. If huihui-ai used a different approach (e.g., averaging over
    different layers or using a non-standard harmful prompt set), the comparison between the 'abliteration direction' and
    individual SAE feature directions loses its mechanistic grounding. This could silently invalidate the entire Type-R/Type-I
    classification by introducing confounds.
  suggested_action: >-
    Check the huihui-ai/Qwen3-8B-abliterated model card and linked code to confirm the exact abliteration technique. If the
    technique matches (difference-of-means + orthogonal projection over a known layer set), document this explicitly. If it
    does not match, either: (a) perform abliteration from scratch using a reproducible script (llm-abliteration or similar)
    to ensure the method is controlled, or (b) add a robustness check comparing the community-abliterated model against a
    self-produced abliterated checkpoint.
- id: ''
  category: methodology
  severity: major
  description: >-
    The base-SAE (trained on base model activations) may reconstruct instruct and abliterated model activations poorly, especially
    after safety RLHF. arXiv:2605.11426 reports that sparse latents diverge substantially between base and SFT models even
    though raw activation cosine similarity stays high. A base-SAE will apply JumpReLU thresholds calibrated for base model
    activations; applied to instruct model activations, the sparsity pattern may be distorted (many features falsely zero,
    or anomalously active), making the feature-level comparison unreliable. This is an untested assumption that directly undermines
    the validity of all three model comparison steps.
  suggested_action: >-
    Before running the main experiment, report reconstruction MSE or fraction of variance explained (R²) for base, instruct,
    and abliterated model activations through the frozen base-SAE. Set an acceptance threshold (e.g., R² > 0.90 for all three
    variants). If the base-SAE reconstructs instruct activations poorly, consider using the SAE's encoder only (to get feature
    activations) without relying on reconstruction quality, and note the limitation. Alternatively, look at whether Qwen-Scope
    releases any instruct-model SAEs for cross-comparison.
- id: ''
  category: rigor
  severity: major
  description: >-
    Success Criterion 4 — the claim that Type-I fraction correlates with capability gap across SAE layers — is almost certainly
    underpowered as described. With Qwen-Scope providing SAEs for ~14 groups across 7 model variants, the effective N for
    a cross-layer correlation within a single model family is at most the number of distinct SAE layers (likely 5-7). A correlation
    from 5-7 data points provides no statistical reliability and cannot support the causal claim that 'Type-I features are
    entangled with general semantic processing.' This would be a misleading correlation even if it reaches p<0.05.
  suggested_action: >-
    Either (a) expand the correlation to multiple model families by running the same analysis on abliterated Qwen3.5-8B or
    Llama-3 variants (adding model families gives more data points with which to test cross-model generalizability); or (b)
    reframe Criterion 4 as exploratory/descriptive ('we report this correlation as a preliminary signal for future work')
    rather than a primary confirmatory criterion, and set an explicit minimum N for statistical validity in the paper.
- id: ''
  category: novelty
  severity: minor
  description: >-
    The space of 'harmful representations survive abliteration' is now crowded. LatentBiopsy (arXiv:2603.27412) demonstrates
    AUROC drops of ≤0.015 after abliteration using PCA geometry on exactly the Qwen model family. arXiv:2604.18901 shows harmful
    intent is linearly separable with AUROC 0.982 that survives abliteration within ±0.003, across 12 models spanning Qwen,
    Llama, and Gemma families. The proposed work's novelty rests entirely on 'which specific SAE features survive and whether
    they are semantically coherent harm-concept features.' If the SAE reconstruction quality issue (above) is not validated,
    the feature-level story may reduce to a restatement of what these geometric papers already show at a more granular vocabulary.
  suggested_action: >-
    In the introduction/framing, explicitly address why SAE-level feature characterization provides actionable insight beyond
    what LatentBiopsy and arXiv:2604.18901 already establish. The strongest argument is: (a) if Type-I features have interpretable
    semantic content (harm schemas, not just 'harmful geometry'), they become candidates for targeted restoration of safety
    without full retraining; (b) the capability gap link (Criterion 4) is a new mechanistic claim not made by geometric approaches.
    Frame these two specific downstream applications as the core motivation for needing feature-level, not just geometric,
    characterization.
- id: ''
  category: methodology
  severity: minor
  description: >-
    Assumption 5 claims that using raw text prompts without chat templates 'provides a fair comparison.' However, the instruct
    model (Qwen/Qwen3-8B) is trained with Qwen's chat template format. When presented with raw text (no [INST] or system prompt),
    the instruct model's behavior is off-distribution and its internal representations may not reflect its safety-trained
    state. The comparison may actually be instruct-OOD vs. base-in-distribution vs. abliterated-OOD rather than a fair three-way
    comparison.
  suggested_action: >-
    Run a pilot study comparing instruct model activations on raw vs. template-formatted harmful prompts and verify that the
    safety-relevant SAE features (high Safety_change) activate similarly in both formats. If they do not, use template-formatted
    prompts for the instruct model (with empty system prompt or safety system prompt) and document the formatting choice clearly.
    Cite the specific prior methodology papers that validate the no-template approach if that validation exists.
- id: ''
  category: rigor
  severity: minor
  description: >-
    The bimodality prediction (Success Criterion 1) is central but unjustified. The hypothesis asserts that high-Safety_change
    features will show a 'clearly bimodal' Reversal_score distribution rather than a continuum. There is no theoretical argument
    for why abliteration would produce a clean binary partition rather than a continuum of partial reversals. The Surgical
    Refusal Ablation paper (arXiv:2601.08489) suggests the refusal direction is polysemantic and entangled with multiple capability
    circuits — this entanglement would more naturally produce a continuous distribution of partial reversals.
  suggested_action: >-
    Weaken the bimodality claim from a success criterion to an empirical question: 'We will test whether the distribution
    is bimodal (consistent with two distinct mechanisms) or continuous (consistent with a single graded mechanism). Either
    result is interpretable.' Add a specification of the statistical test for bimodality (e.g., Hartigan's dip test) and a
    secondary analysis for the continuous case (e.g., treating Reversal_score as a continuous variable and examining its semantic
    correlates along the continuum).
- id: ''
  category: scope
  severity: minor
  description: >-
    The hypothesis is limited to a single model family (Qwen3-8B) and a single abliteration technique. The 'latent safety
    imprint' phenomenon, if confirmed, may be model-specific or technique-specific. The broader safety implications claimed
    in the motivation (evaluating 'the true robustness of safety alignment,' identifying 'candidate features for lightweight
    safety restoration') require generalization across at least two model families.
  suggested_action: >-
    Acknowledge in the motivation/scope section that this is a single-model proof-of-concept. Frame the Qwen3-8B study as
    demonstrating the measurement framework, with explicit plans for cross-model validation in follow-up work. This is acceptable
    for a first paper — but should not be oversold as a general result until replicated.
score: 6
confidence: 4
relation_type:
relation_rationale: ''
</previous_review_feedback><user_data>
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
    "TermDefinition": {
      "description": "A technical term and its definition.",
      "properties": {
        "term": {
          "description": "The technical term",
          "title": "Term",
          "type": "string"
        },
        "definition": {
          "description": "Clear definition of the term",
          "title": "Definition",
          "type": "string"
        }
      },
      "required": [
        "term",
        "definition"
      ],
      "title": "TermDefinition",
      "type": "object"
    }
  },
  "description": "A research hypothesis with validation approach.",
  "properties": {
    "title": {
      "description": "Hypothesis title in plain, everyday language \u2014 short and jargon-free so a non-expert grasps it at a glance and it fits the run visualizations. Aim for about 4-8 words (~40 characters); name the idea, not a status.",
      "title": "Title",
      "type": "string"
    },
    "hypothesis": {
      "description": "The core hypothesis statement",
      "title": "Hypothesis",
      "type": "string"
    },
    "motivation": {
      "description": "Why this hypothesis matters - significance and impact",
      "title": "Motivation",
      "type": "string"
    },
    "assumptions": {
      "description": "Key assumptions that must hold for this hypothesis (2-5 items)",
      "items": {
        "type": "string"
      },
      "title": "Assumptions",
      "type": "array"
    },
    "investigation_approach": {
      "description": "High-level approach to investigating this hypothesis",
      "title": "Investigation Approach",
      "type": "string"
    },
    "success_criteria": {
      "description": "What outcomes would confirm or disconfirm this hypothesis?",
      "title": "Success Criteria",
      "type": "string"
    },
    "related_works": {
      "description": "The most similar existing works found during research. Each entry describes one related work: what it does and how the proposed hypothesis fundamentally differs from it.",
      "items": {
        "type": "string"
      },
      "title": "Related Works",
      "type": "array"
    },
    "inspiration": {
      "description": "What inspired this hypothesis - which patterns, techniques, or cross-field insights were adapted (from the explicit inspiration seeds if your prompt included any, otherwise from your own cross-domain exploration)",
      "title": "Inspiration",
      "type": "string"
    },
    "terms": {
      "description": "Definitions of key technical terms used in the hypothesis",
      "items": {
        "$ref": "#/$defs/TermDefinition"
      },
      "title": "Terms",
      "type": "array"
    },
    "summary": {
      "description": "Brief summary of the hypothesis in 1-2 sentences",
      "title": "Summary",
      "type": "string"
    }
  },
  "required": [
    "title",
    "hypothesis",
    "motivation",
    "assumptions",
    "investigation_approach",
    "success_criteria",
    "related_works",
    "inspiration",
    "terms",
    "summary"
  ],
  "title": "Hypothesis",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-07-17 09:41:59 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [3] SKILL-INPUT — aii-web-tools · 2026-07-17 09:42:21 UTC

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
