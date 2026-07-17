# gen_hypo_1 — create_idea

> Phase: `hypo_loop` · round 1 · `gen_hypo`
> Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B
>
> Full, verbatim transcript of this agent task — every system/user prompt, assistant response, thinking block, tool call and tool result — in the order they occurred. Nothing truncated.

## Task: `gen_hypo_1` (terminal_claude_agent, claude-sonnet-4-6)

### [1] CONFIG · 2026-07-17 09:29:19 UTC

```
model: claude-sonnet-4-6 | effort: medium | permission: bypassPermissions | cwd: /ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_1/gen_hypo/claude_agent
```

### [2] SYSTEM-USER prompt · 2026-07-17 09:29:25 UTC

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

<user_data>
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

### [3] HUMAN-USER prompt · 2026-07-17 09:29:25 UTC

```
qwen3 on huggingface has base model also official safety finetuned version and there is a community finetuned uncensored model called abliterated. take these models and do a mech interp analysis of ho
```

### [4] ASSISTANT · 2026-07-17 09:30:00 UTC

```
I'll start by loading the web tools skill and doing parallel research on the existing landscape before brainstorming.
```

### [5] TOOL CALL — Skill · 2026-07-17 09:30:00 UTC

```
aii-web-tools
```

### [6] TOOL RESULT — Skill · 2026-07-17 09:30:00 UTC

```
{"success": true, "commandName": "aii-web-tools"}
```

### [7] ASSISTANT · 2026-07-17 09:30:04 UTC

```
Loading the web tools and then running parallel searches on the landscape.
```

### [8] TOOL CALL — ToolSearch · 2026-07-17 09:30:04 UTC

```
{'query': 'select:WebSearch,WebFetch'}
```

### [9] TOOL RESULT — ToolSearch · 2026-07-17 09:30:04 UTC

```
{"matches": ["WebSearch", "WebFetch"], "query": "select:WebSearch,WebFetch", "total_deferred_tools": 21}
```

### [10] ASSISTANT · 2026-07-17 09:30:10 UTC

```
Running parallel searches across multiple relevant directions.
```

### [11] TOOL CALL — WebSearch · 2026-07-17 09:30:10 UTC

```
abliteration mechanistic interpretability refusal direction LLM safety fine-tuning 2024 2025
```

### [12] TOOL CALL — WebSearch · 2026-07-17 09:30:10 UTC

```
Qwen3 safety fine-tuning mechanistic interpretability representation engineering 2025 2026
```

### [13] TOOL CALL — WebSearch · 2026-07-17 09:30:10 UTC

```
abliterated model uncensored LLM refusal direction orthogonal projection FailSpy huggingface
```

### [14] TOOL CALL — WebSearch · 2026-07-17 09:30:10 UTC

```
safety fine-tuning feature entanglement capability representation superposition mech interp 2025
```

### [15] TOOL RESULT — WebSearch · 2026-07-17 09:30:18 UTC

```
{"query": "Qwen3 safety fine-tuning mechanistic interpretability representation engineering 2025 2026", "results": [{"tool_use_id": "srvtoolu_018kDJsW3n8r2f6geAqq2PSf", "content": [{"title": "Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis", "url": "https://arxiv.org/pdf/2502.11812"}, {"title": "Understanding and Preserving Safety in Fine-Tuned LLMs", "url": "https://arxiv.org/pdf/2601.10141"}, {"title": "Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/html/2605.11887"}, {"title": "Beyond Static Alignment: Hierarchical Policy Control for LLM Safety via Risk-Aware Chain-of-Thought", "url": "https://arxiv.org/pdf/2602.06650"}, {"title": "AgentLens: Interpretable Safety Steering via Mechanistic Subspaces for Multi-Turn Coding Agent", "url": "https://arxiv.org/pdf/2606.22673"}, {"title": "Matching Ranks Over Probability Yields Truly Deep Safety Alignment", "url": "https://arxiv.org/pdf/2512.05518"}, {"title": "Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models", "url": "https://arxiv.org/pdf/2604.25011"}, {"title": "Two to Tango: Coupled Task-Reference Selection for Safe LLM Fine-tuning", "url": "https://arxiv.org/pdf/2606.09866"}, {"title": "sudoLLM: On Multi-role Alignment of Language Models", "url": "https://arxiv.org/pdf/2505.14607"}, {"title": "Safety-Aligned Weights Are Not Enough: Refusal-Teacher-Guided Finetuning Enhances Safety and Downstream Performance under Harmful Finetuning Attacks", "url": "https://arxiv.org/pdf/2506.07356"}]}, "The search results provide relevant information on these interconnected topics. Here's what I found:\n\n## Qwen3 and Related Models\n\nThe Qwen Team published a Qwen3 technical report in 2025 (arXiv:2505.09388), which appears to be a key reference document for the latest Qwen model.\n\n## Mechanistic Interpretability and Safety\n\nMechanistic interpretability investigates how components in large language models process and represent information, with many MI studies being applied in various fields of AI Safety.\n\nRecent work has applied sparse autoencoders (SAEs) to steering, targeted unlearning, and reasoning-related representations. An SAE toolkit was built for the Qwen family to support both mechanistic analysis and practical downstream use.\n\n## Safety Fine-Tuning and Representation Engineering\n\nFine-tuning aligned language models can compromise safety, even when users do not intend to. Safety alignment should be made more than just a few tokens deep, indicating deeper representations matter for safety.\n\nRepresentation noising has been proposed as a defence mechanism against harmful finetuning.\n\nThese 2025-2026 developments show active research connecting mechanistic interpretability, representation engineering, and safety in both general LLMs and specifically in the Qwen model family."], "durationSeconds": 9.933656551, "searchCount": 1}
```

### [16] TOOL RESULT — WebSearch · 2026-07-17 09:30:18 UTC

```
{"query": "abliteration mechanistic interpretability refusal direction LLM safety fine-tuning 2024 2025", "results": [{"tool_use_id": "srvtoolu_01X2zMJRYdcoKePR4FpS2PCV", "content": [{"title": "LLM Refusal Abliteration Mechanisms", "url": "https://www.emergentmind.com/topics/refusal-abliteration"}, {"title": "Refusal Direction in LLM Safety", "url": "https://www.emergentmind.com/topics/refusal-direction"}, {"title": "Safety Targeted Embedding Exploit via Refinement", "url": "https://arxiv.org/pdf/2607.01859"}, {"title": "Beyond I’m Sorry, I Can’t: Dissecting Large-Language-Model Refusal", "url": "https://arxiv.org/html/2509.09708v2"}, {"title": "Turning the Spell Around: Lightweight Alignment Amplification via Rank-One Safety Injection", "url": "https://arxiv.org/pdf/2508.20766"}, {"title": "An Embarrassingly Simple Defense Against LLM Abliteration Attacks", "url": "https://arxiv.org/pdf/2505.19056"}, {"title": "Abliteration - Learn AI - Miraheze", "url": "https://ai.miraheze.org/wiki/Abliteration"}, {"title": "COSMIC: Generalized Refusal Direction Identification in LLM Activations", "url": "https://arxiv.org/pdf/2506.00085"}, {"title": "Uncensor any LLM with abliteration | by Maxime Labonne | Medium", "url": "https://medium.com/@mlabonne/uncensor-any-llm-with-abliteration-d30148b7d43e"}, {"title": "OBLITERATUS: Mapping the Geometry of Refusal Inside Large Language Models", "url": "https://themenonlab.blog/blog/obliteratus-abliteration-llm-refusal-removal/"}]}, "I found substantial research on this topic. Here are the key findings:\n\n## Overview of Abliteration and Mechanistic Interpretability in LLM Safety\n\nAbliteration is a portmanteau of \"ablate\" and \"obliterate,\" coined by FailSpy, emerging from foundational research by Andy Arditi and colleagues published in June 2024 and later presented at NeurIPS 2024.\n\n## Core Mechanism\n\nSafety fine-tuning concentrates the refusal mechanism into a single linear direction in the residual stream—a clean, interpretable structure and a single point of failure. Refusal direction abliteration is a weight-surgery attack that neutralizes safety guardrails by removing a single direction in the model's residual stream, dramatically reducing the refusal rate of LLAMA-2-7B-CHAT from 100% to approximately 20% while preserving general utility.\n\n## Recent Defenses (2024-2025)\n\nExtended-refusal fine-tuning trains models on richer refusal responses distributed across several latent dimensions, maintaining refusal rates above 90% when abliteration is applied, compared to baseline models with rates reduced by 70–80%.\n\nRefusal Feature Adversarial Training (ReFAT) simulates refusal feature ablation during training to achieve adversarial robustness.\n\nThe ROSI method injects the safety direction into model weights permanently as a fine-tuning-free mechanism, providing comparable robustness to fine-tuning approaches with much lower cost.\n\n## Ongoing Challenges\n\nRefusal mechanisms are often shallow and brittle because they encode in low dimensions, requiring more robust, multi-factorial safety alignment strategies with multidimensional and mechanistically independent refusal features."], "durationSeconds": 10.929005456999999, "searchCount": 1}
```

### [17] TOOL RESULT — WebSearch · 2026-07-17 09:30:20 UTC

```
{"query": "abliterated model uncensored LLM refusal direction orthogonal projection FailSpy huggingface", "results": [{"tool_use_id": "srvtoolu_01Mp7iE4ZYztQJG55R9eUYrd", "content": [{"title": "andrevp/Ornith-1.0-9B-Heretic-Uncensored · Hugging Face", "url": "https://huggingface.co/andrevp/Ornith-1.0-9B-Heretic-Uncensored"}, {"title": "Uncensor any LLM with abliteration | by Maxime Labonne | Medium", "url": "https://medium.com/@mlabonne/uncensor-any-llm-with-abliteration-d30148b7d43e"}, {"title": "AEON-7/Step-3.7-Flash-AEON-Ultimate-Abliterated-BF16 · Hugging Face", "url": "https://huggingface.co/AEON-7/Step-3.7-Flash-AEON-Ultimate-Abliterated-BF16"}, {"title": "WTF Are Abliterated Models? Uncensored LLMs Explained - WebDecoy", "url": "https://webdecoy.com/blog/wtf-are-abliterated-models-uncensored-llms-explained/"}, {"title": "Comparative Analysis of LLM Abliteration Methods:", "url": "https://arxiv.org/pdf/2512.13655"}, {"title": "Not All Refusals Are Equal: How Safety Alignment Fails Cybersecurity at Scale", "url": "https://arxiv.org/pdf/2607.02714"}, {"title": "Five layers of LLM safety filters: where abliterated and uncensored models actually intervene | lilting channel", "url": "https://lilting.ch/en/articles/llm-safety-filters-abliterated-uncensored"}, {"title": "How to Use Uncensored (Abliterated) AI Models on Mobile and Desktop with \"Privacy AI\" and \"LM Studio\" | Privacy AI Documentation", "url": "https://privacyai.acmeup.com/docs/article_32_how_to_use_uncensored_ai_models_on_mobile_and_desktop_with_privacy_ai.html"}, {"title": "Uncensor any LLM with abliteration", "url": "https://huggingface.co/blog/mlabonne/abliteration"}, {"title": "LLMs Unrestrained", "url": "https://medium.com/@david.chew/llms-unrestrained-43820eff85c1"}]}, "Based on the search results, here's what I found about abliterated models, uncensored LLMs, refusal direction, orthogonal projection, FailSpy, and Hugging Face:\n\n## Overview\n\nAbliteration is a post-training modification technique that removes an LLM's refusal behavior by orthogonalizing model weights against the \"refusal direction\" in activation space. The term was coined by FailSpy, creator of the abliterator library.\n\n## How It Works\n\nRefusal behavior in language models is mediated by a single direction in the residual stream activation space, which can be identified by contrasting activations on harmful versus harmless prompts and computing the mean difference. By orthogonalizing the component weights with respect to the refusal direction, it prevents the model from writing to this direction altogether by adjusting the matrices that write to the residual stream.\n\n## Key Technical Details\n\nThe refusal direction is computed by difference-of-means on residual streams, then a single best direction is applied to every layer via weight orthogonalization of the attention out-projection and the MLP down-projection. Abliteration is a permanent modification to the model itself, and removes safety guardrails without retraining.\n\n## Hugging Face Presence\n\nWhen a new model drops, abliterated versions appear on Hugging Face within hours. Multiple abliterated models are available on the platform, including variants from different creators using FailSpy's orthogonalization techniques."], "durationSeconds": 10.296378641000002, "searchCount": 1}
```

### [18] TOOL RESULT — WebSearch · 2026-07-17 09:30:20 UTC

```
{"query": "safety fine-tuning feature entanglement capability representation superposition mech interp 2025", "results": [{"tool_use_id": "srvtoolu_015mhArm8UZh1TCWrsWVMqxF", "content": [{"title": "Mechanistic Interpretability: Circuits, Superposition, and Sparse Autoencoders | Probably Aligned", "url": "https://probablyaligned.ai/safety/interpretability/mechanistic-interpretability/"}, {"title": "A Mechanistic Investigation of Supervised Fine Tuning", "url": "https://arxiv.org/html/2605.11426v1"}, {"title": "Purifying Finetuning Data to Preserve LLM Safety Alignment", "url": "https://aclanthology.org/2025.emnlp-main.406.pdf"}, {"title": "Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation", "url": "https://arxiv.org/pdf/2501.18100"}, {"title": "AI Alignment: A Comprehensive Survey", "url": "https://arxiv.org/pdf/2310.19852"}, {"title": "Published as a conference paper at ICLR 2026 MECHANISTIC INTERPRETABILITY", "url": "https://arxiv.org/pdf/2510.02917"}, {"title": "Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation", "url": "https://arxiv.org/html/2512.23260v1"}, {"title": "Feature Geometry of LoRA Adapters: A Sparse Autoencoder Analysis of Representational Divergence in Fine-Tuned Language Models", "url": "https://arxiv.org/pdf/2605.28896"}, {"title": "What Makes and Breaks Safety Fine-tuning? A Mechanistic Study", "url": "https://arxiv.org/pdf/2407.10264"}]}, "Based on the search results, here's what I found on this topic:\n\n## Key Findings\n\n**Mechanistic Interpretability Overview:**\nMechanistic interpretability (mech interp) aims to reverse-engineer neural networks into human-understandable components. Mechanistic interpretability is the most active frontier in AI safety research.\n\n**Feature Entanglement and Superposition:**\nAccording to the superposition hypothesis, LLMs encode more semantic features than they have dimensions, resulting in individual neurons responding to multiple unrelated concepts. This semantic entanglement makes it difficult to isolate subspaces associated with specific tasks or behaviors directly in the model's hidden space.\n\n**Safety Fine-tuning and Representational Analysis:**\nA novel mechanistic analysis pipeline leverages pretrained SAEs as a frozen semantic basis. Since these SAEs were trained prior to fine-tuning and therefore are strictly invariant to it, they enable post-hoc attribution of representational change to the fine-tuning process itself.\n\nA discrete semantic taxonomy consisting of seven functional clusters was derived: Persona, Structure, Code, Reasoning, Safety, Multilingual, and Collateral (unrelated domain-specific trivia).\n\n**Addressing Feature Entanglement:**\nSparse Autoencoders (SAEs) have emerged as a powerful tool for mechanistic interpretability, learning to decompose model activations into sparse, monosemantic features. One method proposes interpretable safety alignment via SAE-constructed low-rank subspace adaptation, with the key insight that the difficulty of subspace identification can be resolved by operating in the disentangled SAE feature space, where task-relevant directions become readily separable."], "durationSeconds": 10.270467135000006, "searchCount": 1}
```

### [19] ASSISTANT · 2026-07-17 09:30:34 UTC

```
Fetching the most relevant papers in parallel to understand what's already been done.
```

### [20] TOOL CALL — WebFetch · 2026-07-17 09:30:34 UTC

```
URL: https://arxiv.org/abs/2407.10264
Prompt: What is the core mechanistic finding about what makes and breaks safety fine-tuning? What methods were used, what model families, and what are the key results and gaps?
```

### [21] TOOL CALL — WebFetch · 2026-07-17 09:30:34 UTC

```
URL: https://arxiv.org/abs/2512.13655
Prompt: What is the comparative analysis of LLM abliteration methods? What models were compared, what methods, and what were the key findings and gaps?
```

### [22] TOOL CALL — WebFetch · 2026-07-17 09:30:34 UTC

```
URL: https://arxiv.org/abs/2605.11887
Prompt: What is Qwen-Scope? What SAE analysis was done on Qwen models, what were the findings, and what gaps remain?
```

### [23] TOOL CALL — WebFetch · 2026-07-17 09:30:34 UTC

```
URL: https://arxiv.org/abs/2506.00085
Prompt: What is COSMIC? What is the method for identifying refusal directions, what models were tested, and what were the key findings?
```

### [24] TOOL RESULT — WebFetch · 2026-07-17 09:30:34 UTC

```
{"bytes": 42783, "code": 200, "codeText": "OK", "result": "# Comparative Analysis of LLM Abliteration Methods\n\n## Study Overview\nThis research evaluates techniques for removing safety alignment mechanisms from large language models to enable legitimate research applications like adversarial testing and security analysis.\n\n## Models and Methods Compared\n\n**Models:** 16 instruction-tuned models ranging from 7B to 14B parameters\n\n**Abliteration Tools Evaluated:**\n- Heretic\n- DECCP\n- ErisForge\n- FailSpy\n\n## Key Findings\n\nThe study found that \"single-pass methods demonstrated superior capability preservation\" compared to Bayesian-optimized approaches. Specifically, ErisForge showed minimal performance degradation (-0.28 percentage points) on mathematical reasoning tasks, while DECCP showed -0.13 pp change.\n\nMathematical reasoning emerged as particularly vulnerable: \"GSM8K change ranging from +1.51 pp to -18.81 pp (-26.5% relative) depending on tool selection.\"\n\n## Notable Gaps\n\nThe abstract reveals significant limitations—compatibility assessments covered all 16 models, but \"quantitative metrics on subsets dictated by tool support\" indicates inconsistent implementation across tools. This fragmentation prevents comprehensive comparative analysis and suggests researchers face uneven capability across platforms when deploying these techniques.", "durationMs": 6314, "url": "https://arxiv.org/abs/2512.13655"}
```

### [25] TOOL RESULT — WebFetch · 2026-07-17 09:30:36 UTC

```
{"bytes": 42032, "code": 200, "codeText": "OK", "result": "# COSMIC: Refusal Direction Identification\n\n## What is COSMIC?\n\nCOSMIC (Cosine Similarity Metrics for Inversion of Concepts) is an automated framework for identifying how LLMs encode refusal behaviors within their internal activation spaces. The approach operates independently of model outputs, making it broadly applicable across different alignment conditions.\n\n## Identification Method\n\nThe framework employs cosine similarity metrics to select viable steering directions and target layers. Notably, it requires no assumptions about specific refusal tokens or predefined templates, distinguishing it from earlier approaches that depend on observable output patterns.\n\n## Models Tested\n\nThe paper does not explicitly specify which particular model architectures were evaluated in the experiments section provided here.\n\n## Key Findings\n\nAccording to the abstract, COSMIC demonstrated several important capabilities:\n\n- Achieved \"steering performance comparable to prior methods without requiring assumptions about a model's refusal behavior\" \n- Reliably identified refusal directions even in adversarial contexts and weakly aligned models\n- Successfully steered models toward safer behavior while maintaining minimal false refusal rates\n- Showed robustness across varying alignment conditions\n\nThe research was accepted to ACL 2025 Findings, suggesting peer-reviewed validation of these results.", "durationMs": 5794, "url": "https://arxiv.org/abs/2506.00085"}
```

### [26] TOOL RESULT — WebFetch · 2026-07-17 09:30:36 UTC

```
{"bytes": 46038, "code": 200, "codeText": "OK", "result": "# Qwen-Scope Overview\n\n**What is Qwen-Scope?**\n\nQwen-Scope is an open-source collection of sparse autoencoders (SAEs) developed for the Qwen model family. It comprises \"14 groups of SAEs across 7 model variants from the Qwen3 and Qwen3.5 series, covering both dense and mixture-of-expert architectures.\"\n\n**SAE Analysis Conducted**\n\nThe researchers demonstrated that SAEs function as practical development interfaces across four key directions:\n\n1. **Inference-time steering**: Using SAE features to control language, concepts, and preferences without modifying model weights\n2. **Evaluation analysis**: Employing activated SAE features to assess benchmark redundancy and capability coverage\n3. **Data-centric workflows**: Leveraging SAE features for multilingual toxicity classification and safety-focused data synthesis\n4. **Post-training optimization**: Incorporating SAE-derived signals into fine-tuning and reinforcement learning to address issues like code-switching and repetition\n\n**Key Finding**\n\nThe research shows that \"SAEs can serve not only as post-hoc analysis tools, but also as reusable representation-level interfaces for diagnosing, controlling, evaluating, and improving large language models.\"\n\n**Remaining Gaps**\n\nThe paper does not explicitly identify unresolved limitations or gaps within the provided content.", "durationMs": 6357, "url": "https://arxiv.org/abs/2605.11887"}
```

### [27] TOOL RESULT — WebFetch · 2026-07-17 09:30:40 UTC

```
{"bytes": 43267, "code": 200, "codeText": "OK", "result": "# Core Mechanistic Finding\n\nThe research reveals that safety fine-tuning methods work by projecting unsafe inputs into the weight null space of MLP layers, creating a clustering effect where the model learns to distinguish safe from unsafe samples based on activation proximity.\n\n## Methods & Models\n\nThe study examined three safety approaches:\n- Supervised safety fine-tuning\n- Direct preference optimization\n- Unlearning\n\nThey validated findings on real-world models: Llama-2 7B and Llama-3 8B.\n\n## Key Results\n\nThe authors developed a synthetic data framework capturing unsafe inputs by modeling \"the interaction between the task the model is asked to perform (e.g., 'design') versus the specific concepts the task is asked to be performed upon (e.g., a 'cycle' vs. a 'bomb').\"\n\nTheir central discovery: these methods achieve safety through minimal weight transformation. Crucially, \"when an adversarial input (e.g., a jailbreak) is provided, its activations are closer to safer samples, leading the model to process such an input as if it were safe\"—revealing a fundamental vulnerability.\n\n## Key Gaps\n\nThe paper doesn't detail comprehensive evaluation across diverse jailbreak types or discuss mitigation strategies. The scope appears limited primarily to mechanistic analysis rather than proposing robustness improvements or exploring why this projection-based approach persists as the learned mechanism.", "durationMs": 11955, "url": "https://arxiv.org/abs/2407.10264"}
```

### [28] ASSISTANT · 2026-07-17 09:31:08 UTC

```
Fetching more key papers to check overlap with my forming hypotheses.
```

### [29] TOOL CALL — WebFetch · 2026-07-17 09:31:08 UTC

```
URL: https://arxiv.org/abs/2605.11426
Prompt: What is the mechanistic investigation of supervised fine-tuning? What SAE-based analysis was done, what were the key findings about how fine-tuning changes feature space, and what are the key gaps or limitations?
```

### [30] TOOL CALL — WebFetch · 2026-07-17 09:31:08 UTC

```
URL: https://arxiv.org/abs/2510.02917
Prompt: What is this ICLR 2026 mechanistic interpretability paper about? What are the key contributions, methods, and findings?
```

### [31] TOOL CALL — WebSearch · 2026-07-17 09:31:08 UTC

```
base model safety RLHF abliterated three-way comparison residual stream feature activation difference 2025 2026
```

### [32] TOOL CALL — WebSearch · 2026-07-17 09:31:08 UTC

```
"vestigial safety" OR "safety representation persists" abliteration uncensored model hidden states 2025
```

### [33] TOOL RESULT — WebFetch · 2026-07-17 09:31:10 UTC

```
{"bytes": 40504, "code": 200, "codeText": "OK", "result": "# Mechanistic Investigation of Supervised Fine-Tuning\n\n## Overview\nThe research examines how Supervised Fine-Tuning (SFT) alters language model representations at a mechanistic level, revealing that while surface-level activation geometry appears stable, deeper representational changes occur.\n\n## SAE-Based Analysis Method\nThe researchers employed Sparse Autoencoders (SAEs) pretrained on base models as diagnostic tools. They projected both pre- and post-SFT activations through these SAEs to detect changes in sparse latent representations that cosine similarity measures alone would miss.\n\n## Key Findings\nAccording to the abstract, the analysis discovered:\n\n- **Feature divergence**: \"The underlying sparse latents diverge significantly\" despite high cosine similarity between raw activations\n- **Task and layer specificity**: \"Task-specific and layer-specific distributions of the precise semantic features that are systematically altered during supervised fine-tuning\"\n- **Safety alignment patterns**: Identification of \"a layer-wise update profile specific to safety alignment\"\n\n## Limitations\nThe provided content doesn't explicitly discuss methodological gaps or limitations. The paper appears focused on presenting the diagnostic pipeline and findings rather than critically evaluating constraints on the analysis.", "durationMs": 5793, "url": "https://arxiv.org/abs/2605.11426"}
```

### [34] TOOL RESULT — WebFetch · 2026-07-17 09:31:12 UTC

```
{"bytes": 42558, "code": 200, "codeText": "OK", "result": "# Paper Summary: Mechanistic Interpretability of Code Correctness in LLMs\n\n**Note:** This appears to be an arXiv preprint (submitted October 2025), not confirmed as an ICLR 2026 paper.\n\n## Key Contributions\n\nThe research uses sparse autoencoders to analyze how LLMs represent and generate correct code. The authors identify specific neural directions associated with code correctness and study how these mechanisms operate.\n\n## Methods\n\nResearchers employed:\n- Sparse autoencoders to decompose LLM representations\n- T-statistics for selecting predictor directions\n- Separation scores for identifying steering directions\n- Mechanistic analysis through steering, attention examination, and weight orthogonalization\n\n## Main Findings\n\nSeveral important discoveries emerged:\n\n1. **Predictive accuracy:** \"Code correctness directions in LLMs reliably predict incorrect code\" while correction has tradeoffs between fixing errors and preserving working code.\n\n2. **Attention patterns:** Success depends on attending to test cases rather than problem descriptions.\n\n3. **Transfer learning:** Correctness mechanisms learned during pre-training persist after instruction-tuning.\n\n## Practical Applications\n\nThe authors propose three uses: prioritizing test examples in prompts, deploying correctness directions as error detection systems, and using selective steering interventions only when errors are anticipated.", "durationMs": 6346, "url": "https://arxiv.org/abs/2510.02917"}
```

### [35] TOOL RESULT — WebSearch · 2026-07-17 09:31:20 UTC

```
{"query": "\"vestigial safety\" OR \"safety representation persists\" abliteration uncensored model hidden states 2025", "results": [{"tool_use_id": "srvtoolu_01TgBjB1gnz7xXHrAeKENbhE", "content": [{"title": "How to Use Uncensored (Abliterated) AI Models on Mobile and Desktop with \"Privacy AI\" and \"LM Studio\" | Privacy AI Documentation", "url": "https://privacyai.acmeup.com/docs/article_32_how_to_use_uncensored_ai_models_on_mobile_and_desktop_with_privacy_ai.html"}, {"title": "Heretic Strips AI Censorship—1,000+ Uncensored LLMs Made | byteiota", "url": "https://byteiota.com/heretic-strips-ai-censorship-1000-uncensored-llms-made/"}, {"title": "ponpoke/ERNIE-Image-Abliterated-GGUF · Hugging Face", "url": "https://huggingface.co/ponpoke/ERNIE-Image-Abliterated-GGUF"}, {"title": "Uncensored and Abliterated Models: What They Are and How to Use Them | RunThisModel · RunThisModel", "url": "https://runthismodel.com/guides/uncensored-models-guide"}, {"title": "The Uncensored AI: What’s Actually Happening When People Strip Safety From AI | by Rekhi | Medium", "url": "https://medium.com/@Reiki32/whats-actually-happening-when-people-strip-safety-from-open-source-ai-06214672fa1c"}, {"title": "GitHub - NousResearch/llm-abliteration: Make abliterated models with transformers, easy and fast · GitHub", "url": "https://github.com/NousResearch/llm-abliteration"}, {"title": "GitHub - jim-plus/llm-abliteration: Make abliterated models with transformers, easy and fast · GitHub", "url": "https://github.com/jim-plus/llm-abliteration/"}, {"title": "HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States", "url": "https://arxiv.org/pdf/2502.14744"}, {"title": "WTF Are Abliterated Models? Uncensored LLMs Explained - WebDecoy", "url": "https://webdecoy.com/blog/wtf-are-abliterated-models-uncensored-llms-explained/"}, {"title": "Uncensor any LLM with abliteration", "url": "https://huggingface.co/blog/mlabonne/abliteration"}]}, "I found relevant results about abliteration and model safety. However, I didn't find specific results about \"vestigial safety\" or \"safety representation persists\" in the context you mentioned. Let me search more specifically for those terms.", {"tool_use_id": "srvtoolu_01FcoQYrbD5K4k92XCmzKqd7", "content": [{"title": "Federal Register :: Request for Comment on Vestigial Vehicle Safety Regulations", "url": "https://www.federalregister.gov/documents/2026/01/23/2026-01272/request-for-comment-on-vestigial-vehicle-safety-regulations"}, {"title": "Vestigial Structures: Understanding Evolutionary Remnants - ScienceInsights", "url": "https://scienceinsights.org/vestigial-structures-understanding-evolutionary-remnants/"}, {"title": "Comments to NHTSA on Vestigial Regulations", "url": "https://www.autosinnovate.org/posts/agency-comments/comments-to-nhtsa-on-vestigial-regulations"}, {"title": "CR comments to the NHTSA on the Request for Comments on Vestigial Vehicle Safety Regulations", "url": "https://advocacy.consumerreports.org/research/cr-comments-to-the-nhtsa-on-the-request-for-comments-on-vestigial-vehicle-safety-regulations/"}, {"title": "Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks", "url": "https://arxiv.org/pdf/2605.05995"}, {"title": "It's Time to Cut These! Safety Blanket &#038; Vestigial Cards | EDHRECast 359 | Magic the Gathering", "url": "https://edhrec.com/articles/its-time-to-cut-these-safety-blanket-vestigial-cards-edhrecast-359-magic-the-gathering"}, {"title": "Representation Bending for Large Language Model Safety", "url": "https://aclanthology.org/2025.acl-long.1173.pdf"}, {"title": "The persistence and evolutionary consequences of vestigial behaviours - PMC", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9540461/"}, {"title": "DeepSight: An All-in-One LM Safety Toolkit", "url": "https://arxiv.org/pdf/2602.12092"}, {"title": "ReWorld: Learning Better Representations for World Action Models", "url": "https://arxiv.org/pdf/2606.27504"}]}, "Based on my search results, I found substantial information about **abliteration and uncensored models**, but the specific terms \"vestigial safety\" and \"safety representation persists\" as you've framed them don't appear in recent AI research literature from 2025.\n\n## What I Found:\n\n### Abliteration and Hidden States\nAbliteration works by collecting activations of the model on harmful versus harmless instructions and taking the mean difference to identify the vector in the model's hidden state that drives refusal. Orthogonalising the weights or projecting out this direction at inference time yields an uncensored model that no longer refuses prompts but retains the underlying linguistic ability.\n\n### Safety Limitations\nImportantly, abliteration does not guarantee full removal of censorship, as a properly abliterated model will not explicitly refuse, theoretically, based on the nature of refusals captured in datasets used for abliteration. This suggests safety may persist in ways not captured by the abliteration process.\n\n### Related Research on Representation and Safety\nI found research on defenses like Vaccine and T-Vaccine that operate on the Representation Stability Hypothesis, positing that safety collapse stems from embedding drift, though this focuses on defending against attacks rather than vestigial safety remnants.\n\n**Note:** The specific combination of terms you searched for (\"vestigial safety\" + \"safety representation persists\" + \"abliteration\") doesn't appear in current 2025 publications. If you're looking for a specific paper or research discussing lingering safety representations after abliteration, you may need to refine your search or check recent preprints on arXiv."], "durationSeconds": 13.666566585000007, "searchCount": 2}
```

### [36] TOOL RESULT — WebSearch · 2026-07-17 09:31:26 UTC

```
{"query": "base model safety RLHF abliterated three-way comparison residual stream feature activation difference 2025 2026", "results": [{"tool_use_id": "srvtoolu_011tAneGmiWwPvnwbmFTVMGz", "content": [{"title": "Abliterated Models 2026: Free Uncensored AI Downloads (Qwen 3.6, Gemma 4, Llama)", "url": "https://locallyuncensored.com/blog/abliterated-models-guide.html"}, {"title": "SPECTER ABLITERATE — Open-Weight Model Alignment Removal Engine | Red Specter", "url": "https://red-specter.co.uk/nightfall/specter-abliterate/"}, {"title": "Heretic vs Abliterated LLMs: Refusal Rates & Benchmarks (2026)", "url": "https://aithinkerlab.com/heretic-ai-abliteration-benchmarks-2026/"}, {"title": "An Embarrassingly Simple Defense Against LLM Abliteration Attacks", "url": "https://arxiv.org/html/2505.19056v1"}, {"title": "The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams", "url": "https://arxiv.org/pdf/2603.27412"}, {"title": "1 Introduction", "url": "https://arxiv.org/html/2512.13655v1"}, {"title": "Theory of Mind and Self-Attributions of Mentality are Dissociable in LLMs", "url": "https://arxiv.org/pdf/2603.28925"}, {"title": "Different Paths to Harmful Compliance: Behavioral Side Effects and Mechanistic Divergence Across LLM Jailbreaks", "url": "https://arxiv.org/pdf/2604.18510"}, {"title": "Not All Refusals Are Equal: How Safety Alignment Fails Cybersecurity at Scale", "url": "https://arxiv.org/html/2607.02714"}]}, "I found relevant search results about abliteration and safety alignment in language models. Here's what the research shows:\n\n## Core Concept\n\nSafety-trained models learn a linear direction in their residual stream that mediates refusal behaviour, consistent across prompts, model families, and training recipes, and representing the mechanism by which RLHF/DPO/SFT alignment operates at the representation level.\n\n## Abliteration Technique\n\nThe technique involves taking a corpus of refused prompts, isolating the activation pattern that distinguishes them from accepted prompts, and projecting that direction out of every weight matrix in the model. This involves surgical removal of RLHF/DPO/SFT safety alignment from open-weight models by projecting all weight matrices orthogonal to the refusal direction.\n\n## Three-Way Comparisons (Base vs. RLHF vs. Abliterated)\n\nResearch has examined differences between these model variants. Within model families, base models match or slightly exceed their instruction-tuned counterparts in harmful detection, with safety geometry being present before alignment fine-tuning. Abliterated models show striking results—in Qwen3.5-0.8B, the abliterated model's harm detection capability lies within noise of the chat model.\n\n## Recent Research (2025-2026)\n\nDPO-only models show higher abliteration susceptibility than RLHF-aligned models. As of June 2026, more than 1,000 Heretic-derived models have been published on HuggingFace."], "durationSeconds": 20.089816877999997, "searchCount": 1}
```

### [37] ASSISTANT · 2026-07-17 09:31:57 UTC

```
Fetching the most directly relevant papers to my forming hypothesis.
```

### [38] TOOL CALL — WebFetch · 2026-07-17 09:31:57 UTC

```
URL: https://arxiv.org/abs/2603.27412
Prompt: What is this paper about? Does it compare base, safety-tuned, and abliterated models? What are the key findings about safety geometry in Qwen models? What SAE analysis was done?
```

### [39] TOOL CALL — WebFetch · 2026-07-17 09:31:57 UTC

```
URL: https://arxiv.org/abs/2604.18510
Prompt: What is the mechanistic divergence analysis across LLM jailbreaks? Does it compare base, safety-tuned, and abliterated model internals? What are key findings?
```

### [40] TOOL CALL — WebSearch · 2026-07-17 09:31:57 UTC

```
SAE sparse autoencoder Qwen3 base safety fine-tuned feature activation comparison abliteration 2025 2026
```

### [41] TOOL CALL — WebSearch · 2026-07-17 09:31:57 UTC

```
safety alignment collateral feature modification capability entanglement abliteration reversible irreversible 2025
```

### [42] TOOL RESULT — WebFetch · 2026-07-17 09:32:01 UTC

```
{"bytes": 45249, "code": 200, "codeText": "OK", "result": "# Summary of \"The Geometry of Harmful Intent\"\n\n## Paper Overview\nThis research introduces **LatentBiopsy**, a method that detects harmful prompts by examining how large language models represent them internally, without requiring harmful examples during training.\n\n## Model Comparison\nYes, the paper evaluates three model variants across two Qwen families:\n- Base models\n- Instruction-tuned (safety-aligned) models\n- \"Abliterated\" models (where refusal mechanisms were surgically removed)\n\nTesting covered six total variants: Qwen3.5-0.8B and Qwen2.5-0.5B in all three forms.\n\n## Key Findings on Safety Geometry\n\n1. **Geometric Persistence**: \"Geometry survives refusal ablation: both abliterated variants achieve AUROC at most 0.015 below their instruction-tuned counterparts,\" showing that harmful-intent representation exists independently of the refusal mechanism.\n\n2. **Tighter Harmful Distribution**: Harmful prompts cluster much more tightly (σθ ≈ 0.03 rad) compared to normal prompts (σθ ≈ 0.27 rad)—this difference persists across all alignment stages.\n\n3. **Family-Specific Orientations**: The two Qwen families show opposite spatial patterns at the same layer depth, motivating direction-agnostic detection.\n\n## SAE Analysis\nThe provided content contains **no information about Sparse Autoencoders (SAE) analysis**. The paper focuses on principal component analysis of residual streams rather than SAE-based methods.", "durationMs": 7217, "url": "https://arxiv.org/abs/2603.27412"}
```

### [43] TOOL RESULT — WebFetch · 2026-07-17 09:32:01 UTC

```
{"bytes": 44255, "code": 200, "codeText": "OK", "result": "# Mechanistic Divergence Analysis in LLM Jailbreaks\n\nBased on the paper's abstract, the mechanistic divergence analysis examines how different jailbreaking methods produce distinct internal changes despite achieving similar harmful compliance levels.\n\n## Key Findings on Internal Mechanisms:\n\nThe research identifies three different mechanistic pathways:\n\n1. **Abliteration**: Described as \"consistent with localized refusal-feature deletion,\" suggesting targeted removal of specific safety mechanisms within model internals.\n\n2. **RLVR (Reinforcement Learning with Verifiable Rewards)**: Associated with \"preserved safety geometry but retargeted policy behavior,\" meaning safety structures remain intact internally while the model's decision-making gets redirected.\n\n3. **SFT (Supervised Fine-Tuning)**: Shows \"broader distributed drift,\" indicating widespread changes across the model's internal representations rather than localized modifications.\n\n## Critical Distinction:\n\nA notable finding is that RLVR-jailbroken models remain remarkably similar to base models internally while still complying with harmful requests. This contrasts sharply with SFT models, which exhibit \"the largest collapse in explicit safety judgments\" and substantial capability loss.\n\nThe analysis suggests that mechanistic understanding reveals how identical harmful behavior can emerge through fundamentally different routes within model architecture—a distinction crucial for developing targeted defenses.", "durationMs": 6123, "url": "https://arxiv.org/abs/2604.18510"}
```

### [44] TOOL RESULT — WebSearch · 2026-07-17 09:32:07 UTC

```
{"query": "safety alignment collateral feature modification capability entanglement abliteration reversible irreversible 2025", "results": [{"tool_use_id": "srvtoolu_01A1STdRe1V2aTDWSQkHijov", "content": [{"title": "Refusal Direction in LLM Safety", "url": "https://www.emergentmind.com/topics/refusal-direction"}, {"title": "Surgical Refusal Ablation: Disentangling Safety from Intelligence via Concept-Guided Spectral Cleaning", "url": "https://arxiv.org/html/2601.08489v1"}, {"title": "Different Paths to Harmful Compliance: Behavioral Side Effects and Mechanistic Divergence Across LLM Jailbreaks", "url": "https://arxiv.org/pdf/2604.18510"}, {"title": "Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation", "url": "https://arxiv.org/pdf/2512.23260"}, {"title": "Measuring Alignment-Induced Activation Shifts Correctly: A Template-Controlled Difference-in-Differences Protocol", "url": "https://arxiv.org/pdf/2605.24583"}, {"title": "On the Inseparability of Instructions and Data in Shared-Embedding Sequence Models", "url": "https://arxiv.org/pdf/2606.27567"}, {"title": "(PDF) Surgical Refusal Ablation: Disentangling Safety from Intelligence via Concept-Guided Spectral Cleaning", "url": "https://www.researchgate.net/publication/399754691_Surgical_Refusal_Ablation_Disentangling_Safety_from_Intelligence_via_Concept-Guided_Spectral_Cleaning"}, {"title": "[Literature Review] Surgical Refusal Ablation: Disentangling Safety from Intelligence via Concept-Guided Spectral Cleaning", "url": "https://www.themoonlight.io/en/review/surgical-refusal-ablation-disentangling-safety-from-intelligence-via-concept-guided-spectral-cleaning"}, {"title": "Surgical Refusal Ablation: Disentangling Safety from ...", "url": "https://arxiv.org/pdf/2601.08489"}]}, "Based on the search results, I found several recent research papers from 2025-2026 addressing the topics you queried. Here are the key findings:\n\n## Safety Alignment and Feature Entanglement\n\nThe raw refusal vector is polysemantic: it entangles the refusal signal with unrelated linguistic structure (syntax, formatting) and core capability circuits (math, coding, reasoning).\n\n## Collateral Damage and Capability Modification\n\nThe near-zero drift and stable math/code perplexity under SRA suggest refusal-related behavior can be edited with minimal collateral damage when the intervention is sufficiently precise. Much of the \"tax\" observed in prior ablations may be attributable to dirty vectors and imprecise interventions.\n\n## Ablation Techniques and Reversibility\n\nAbliteration exhibits moderate, family-dependent behavioral drift but remains the most easily repairable via targeted feature patching. In comparison, RLVR preserves the underlying safety geometry and general capabilities while retargeting policy behavior, but is only partially repairable. SFT induces the most severe catastrophic forgetting, failing across all evaluated safety and capability dimensions.\n\n## Addressing Polysemantic Entanglement\n\nSRA constructs a registry of independent Concept Atoms representing protected capabilities and stylistic confounds, then uses ridge-regularized spectral residualization to orthogonalize the refusal vector against these directions.\n\nThe research from early 2025-2026 suggests that the reversibility and collateral damage associated with safety alignment modifications depends heavily on the technique used and the precision with which capability-irrelevant features are targeted."], "durationSeconds": 11.219655994, "searchCount": 1}
```

### [45] TOOL RESULT — WebSearch · 2026-07-17 09:32:07 UTC

```
{"query": "SAE sparse autoencoder Qwen3 base safety fine-tuned feature activation comparison abliteration 2025 2026", "results": [{"tool_use_id": "srvtoolu_01TXp6BGUBct9566ooEcNZzs", "content": [{"title": "Discovering Millions of Interpretable Features with Sparse Autoencoders", "url": "https://arxiv.org/html/2606.26620v1"}, {"title": "Auditing Games for Sandbagging", "url": "https://arxiv.org/pdf/2512.07810"}, {"title": "Qwen-Scope: Open-Source Sparse AutoEncoders for LLM Interpretability and Steering • Dev|Journal", "url": "https://earezki.com/ai-news/2026-05-01-qwen-ai-releases-qwen-scope-an-open-source-sparse-autoencoders-sae-suite-that-turns-llm-internal-features-into-practical-development-tools/"}, {"title": "Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers", "url": "https://arxiv.org/pdf/2512.15674"}, {"title": "Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/html/2605.11887"}, {"title": "Visual Self-Fulfilling Alignment: Shaping Safety-Oriented Personas via Threat-Related Images", "url": "https://arxiv.org/pdf/2603.08486"}, {"title": "Safe-SAIL: Towards a Fine-grained Safety Landscape of Large Language Models via Sparse Autoencoder Interpretation Framework Warning: this paper contains data, prompts, and model outputs that are offensive in nature.", "url": "https://arxiv.org/html/2509.18127v3"}, {"title": "Scalable Circuit Learning for Interpreting Large Language Models", "url": "https://arxiv.org/pdf/2606.16939"}, {"title": "SAEExplainer: Interpreting SAE Features with Activation-Guided Preference Optimization", "url": "https://arxiv.org/pdf/2606.08496"}, {"title": "Perplexity Can Miss SAE Feature Damage Under Quantization", "url": "https://arxiv.org/pdf/2606.03002"}]}, "Based on the search results, I found several relevant resources on SAEs, Qwen3 models, safety analysis, and feature activation:\n\n## Key Findings:\n\n**Qwen3-Instruct SAE Release:**\nA comprehensive public release of Sparse Autoencoders was trained on the Qwen3 instruction-tuning model, covering Qwen3-1.7B, Qwen3-4B, and Qwen3-8B. The project adopts JumpReLU as the primary training method for Qwen3-Instruct SAE.\n\n**Qwen-Scope Suite (2026):**\nQwen Team launched Qwen-Scope, an open-source suite of sparse autoencoders trained on Qwen3 and Qwen3.5 families, including 14 groups of SAE weights across 7 model variants. SAEs trained on Qwen3/3.5 serve as a common interface for four practical directions: interpretable steering at inference time, capability-aware benchmark analysis, feature-guided data workflows, and targeted post-training for model improvement.\n\n**Safety Analysis:**\nA comprehensive suite of SAEs was trained on Qwen2.5-3B-Instruct and generated human-readable explanations for safety-related features across four subdomains: pornography, politics, violence, and terror. Feature-driven safety data synthesis generates targeted prompt-completion pairs for missing safety features, with random safety synthesis resulting in significantly lower coverage of target features compared to SAE-guided methods.\n\nThe search results provide recent work (2025-2026) on SAEs for Qwen models with focus on interpretability, safety analysis, and feature activation patterns."], "durationSeconds": 12.089603367000004, "searchCount": 1}
```

### [46] ASSISTANT · 2026-07-17 09:33:19 UTC

```
Fetching the most relevant competing papers to check for overlap.
```

### [47] TOOL CALL — WebFetch · 2026-07-17 09:33:19 UTC

```
URL: https://arxiv.org/abs/2509.18127
Prompt: What does this paper do with SAEs for safety? Does it compare base model, safety-tuned, and abliterated model features? What are the key findings? What are the gaps?
```

### [48] TOOL CALL — WebFetch · 2026-07-17 09:33:19 UTC

```
URL: https://arxiv.org/abs/2601.08489
Prompt: What does Surgical Refusal Ablation do exactly? Does it compare SAE features across base, safety-tuned, and abliterated variants? What specific Qwen models are used? What gaps remain?
```

### [49] TOOL CALL — WebSearch · 2026-07-17 09:33:19 UTC

```
Qwen3 abliterated uncensored huggingface model 2025 2026 site:huggingface.co
```

### [50] TOOL RESULT — WebFetch · 2026-07-17 09:33:23 UTC

```
{"bytes": 45098, "code": 200, "codeText": "OK", "result": "# Safe-SAIL: SAE Safety Analysis\n\n## What the paper does with SAEs:\n\nThe research applies sparse autoencoders to interpret safety-critical aspects of large language models. The framework \"decomposes entangled model activations into monosemantic features\" specifically focused on safety domains.\n\n## Model comparisons:\n\nThe provided abstract does **not specify** whether the study compares base models, safety-tuned variants, or abliterated versions. This information isn't disclosed in the available content.\n\n## Key findings:\n\nThe work identifies and explains 1,758 safety-related features across four risk categories: pornography, politics, violence, and terror. The researchers demonstrate how \"safety-critical entities and concepts are encoded across model layers\" and develop methods to efficiently identify high-quality safety-domain features.\n\n## Notable contribution:\n\nSafe-SAIL introduces \"a pre-explanation evaluation metric\" to identify the most promising SAEs and reduces interpretation costs by 55% through a segment-level simulation approach.\n\n## Gaps in available information:\n\n- Specific performance metrics or accuracy rates aren't provided in this abstract\n- Comparative analysis methodology (if any) between different model types remains unclear\n- Detailed findings about which layers encode safety concepts most effectively aren't summarized here\n\nThe paper promises public release of models, explanations, and tools via an open-source toolkit.", "durationMs": 6519, "url": "https://arxiv.org/abs/2509.18127"}
```

### [51] TOOL RESULT — WebFetch · 2026-07-17 09:33:23 UTC

```
{"bytes": 42788, "code": 200, "codeText": "OK", "result": "# Surgical Refusal Ablation: Summary\n\n## What SRA Does\n\nSurgical Refusal Ablation addresses a key problem: standard refusal vector ablation damages model capabilities. The technique works by:\n\n1. **Identifying the core issue**: The raw refusal vector is \"polysemantic, entangling the refusal signal with core capability circuits and linguistic style\"\n2. **Creating a registry**: Building \"Concept Atoms representing protected capabilities and stylistic confounds\"\n3. **Cleaning the vector**: Using \"ridge-regularized spectral residualization to orthogonalize the refusal vector against these directions\"\n\nThe result is a refined ablation that achieves near-zero refusal while preserving model behavior.\n\n## SAE Feature Comparison\n\nThe paper does **not** compare Sparse Autoencoders (SAEs) across base, safety-tuned, and abliterated variants based on the abstract provided.\n\n## Specific Models Tested\n\nThe research evaluates **Qwen3-VL** and the **Ministral series** across five total models. Only Qwen3-VL-4B receives detailed discussion in the abstract.\n\n## Key Gaps\n\nThe document notes that \"common 'model damage' is often 'Ghost Noise'\" but doesn't elaborate on remaining limitations. Additional gaps likely include understanding which specific capability dimensions prove most vulnerable and whether the approach generalizes to future model architectures—though these specifics aren't detailed in the abstract.", "durationMs": 6772, "url": "https://arxiv.org/abs/2601.08489"}
```

### [52] TOOL RESULT — WebSearch · 2026-07-17 09:33:31 UTC

```
{"query": "Qwen3 abliterated uncensored huggingface model 2025 2026 site:huggingface.co", "results": [{"tool_use_id": "srvtoolu_01UxCCVFbNcCs9EWeDHRogHt", "content": [{"title": "Qwen3-VL Abliteration - a prithivMLmods Collection", "url": "https://huggingface.co/collections/prithivMLmods/qwen3-vl-abliteration"}, {"title": "huihui-ai/Qwen3-8B-abliterated · Hugging Face", "url": "https://huggingface.co/huihui-ai/Qwen3-8B-abliterated"}, {"title": "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated · Hugging Face", "url": "https://huggingface.co/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated"}, {"title": "huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated · Hugging Face", "url": "https://huggingface.co/huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated"}, {"title": "Youssofal/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF · Hugging Face", "url": "https://huggingface.co/Youssofal/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF"}, {"title": "Qwen3-abliterated - a huihui-ai Collection", "url": "https://huggingface.co/collections/huihui-ai/qwen3-abliterated"}, {"title": "Josiefied and Abliterated Qwen3 - a Goekdeniz-Guelmez Collection", "url": "https://huggingface.co/collections/Goekdeniz-Guelmez/josiefied-and-abliterated-qwen3"}, {"title": "DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF · Hugging Face", "url": "https://huggingface.co/DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF"}, {"title": "huihui-ai/Huihui-Qwen3-Coder-Next-abliterated · Hugging Face", "url": "https://huggingface.co/huihui-ai/Huihui-Qwen3-Coder-Next-abliterated"}, {"title": "huihui-ai/Huihui-Qwen3-VL-235B-A22B-Instruct-abliterated-GGUF · Hugging Face", "url": "https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-235B-A22B-Instruct-abliterated-GGUF"}]}, "Based on the web search results, here are the findings for Qwen3 abliterated uncensored models on Hugging Face:\n\n## Available Models\n\nThere are uncensored versions of Qwen/Qwen3-8B created with abliteration, with multiple variations available:\n\n1. **Qwen3-8B**: There's a new version available, please try using the new version huihui-ai/Huihui-Qwen3-8B-abliterated-v2\n\n2. **Qwen3.5 and Qwen3.6 Models**: There are multiple versions including Qwen3.5 abliterated v2-MAX and various multimodal collections dated through February 2026\n\n3. **Qwen3.6-35B**: An uncensored version of Qwen/Qwen3.5-35B-A3B created with abliteration\n\n4. **Qwen3.6-27B**: A GGUF release of an abliterated, uncensored version of Qwen's Qwen3.6-27B with vision, made with Heretic\n\n5. **Vision Models**: Uncensored versions of Qwen/Qwen3-VL-235B-A22B-Instruct created with abliteration\n\n## What Abliteration Does\n\nThis model has had refusal behavior attenuated at the weight level and will answer prompts that the base model would normally refuse\n\n## Safety Note\n\nThis model's safety filtering has been significantly reduced, potentially generating sensitive, controversial, or inappropriate content. Users should exercise caution and rigorously review generated outputs"], "durationSeconds": 12.975267431999994, "searchCount": 1}
```

### [53] ASSISTANT · 2026-07-17 09:34:52 UTC

```
Good. Let me verify the specific SAE availability for Qwen3-8B and check the Qwen-Scope repository.
```

### [54] TOOL CALL — WebFetch · 2026-07-17 09:34:52 UTC

```
URL: https://huggingface.co/Qwen/Qwen-Scope
Prompt: What SAE models are available for which Qwen3 model sizes? Specifically is there a Qwen3-8B SAE? What layers are covered?
```

### [55] TOOL RESULT — WebFetch · 2026-07-17 09:34:52 UTC

```
{"bytes": 0, "code": 401, "codeText": "Unauthorized", "result": "The server returned HTTP 401 Unauthorized.\n\nThe response body was not retrieved. If this URL requires authentication, use an authenticated tool (e.g. `gh` for GitHub, or an MCP-provided fetch tool) instead of WebFetch.", "durationMs": 400, "url": "https://huggingface.co/Qwen/Qwen-Scope"}
```

### [56] TOOL CALL — WebFetch · 2026-07-17 09:34:52 UTC

```
URL: https://arxiv.org/abs/2605.24583
Prompt: What is this paper about regarding alignment-induced activation shifts? Does it track changes across base, safety-tuned, and abliterated model variants? What are the key methods and findings?
```

### [57] TOOL RESULT — WebFetch · 2026-07-17 09:34:52 UTC

```
{"bytes": 44330, "code": 200, "codeText": "OK", "result": "# Paper Summary: Measuring Alignment-Induced Activation Shifts\n\n## Core Topic\n\nThis paper addresses methodological issues in comparing neural network activations before and after safety alignment training. The research identifies a fundamental confound: aligned models are evaluated using chat templates that base models never encounter, conflating alignment effects with formatting effects.\n\n## Key Methods\n\nThe authors introduce a four-variant decomposition framework:\n\n1. **Naive approach** - Direct difference (problematic)\n2. **Template-controlled** - Accounts for chat formatting\n3. **Within-aligned** - Examines variation within aligned models\n4. **Difference-in-differences (DiD)** - Isolates true alignment shifts\n\n## Major Findings\n\nThe template-controlled method substantially reduces inflated measurements. As noted in the abstract: \"Template control alone removes a 2.0-3.9x inflation of the measured effective rank\" across three model families (Llama-3.1-8B, Gemma-2-9B, Qwen-2.5-7B).\n\nCritically, the DiD approach recovered the refusal direction previously identified by Arditi et al., improving cosine alignment from 0.18-0.39 to 0.50-0.86—a significant improvement demonstrating the method's validity.\n\n## Scope\n\nRather than tracking multiple model variants, this work focuses on methodological correction. The authors validate findings through projection-ablation studies confirming behavioral relevance and provide measurement recommendations for future activation-difference research.", "durationMs": 7869, "url": "https://arxiv.org/abs/2605.24583"}
```

### [58] TOOL CALL — WebSearch · 2026-07-17 09:34:58 UTC

```
Qwen-Scope SAE huggingface Qwen3-8B sparse autoencoder available models 2026
```

### [59] TOOL RESULT — WebSearch · 2026-07-17 09:35:06 UTC

```
{"query": "Qwen-Scope SAE huggingface Qwen3-8B sparse autoencoder available models 2026", "results": [{"tool_use_id": "srvtoolu_01EP3FBm3E4jvvVBvwi9Ktea", "content": [{"title": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"}, {"title": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100"}, {"title": "Qwen/SAE-Res-Qwen3-30B-A3B-Base-W128K-L0_100 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3-30B-A3B-Base-W128K-L0_100"}, {"title": "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"}, {"title": "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 · Hugging Face", "url": "https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"}, {"title": "Qwen AI Releases Qwen-Scope: An Open-Source Sparse AutoEncoders (SAE) Suite That Turns LLM Internal Features into Practical Development Tools - MarkTechPost", "url": "https://www.marktechpost.com/2026/05/01/qwen-ai-releases-qwen-scope-an-open-source-sparse-autoencoders-sae-suite-that-turns-llm-internal-features-into-practical-development-tools/"}, {"title": "Qwen-Scope: Open-Source Sparse AutoEncoders for LLM Interpretability and Steering • Dev|Journal", "url": "https://earezki.com/ai-news/2026-05-01-qwen-ai-releases-qwen-scope-an-open-source-sparse-autoencoders-sae-suite-that-turns-llm-internal-features-into-practical-development-tools/"}, {"title": "Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models", "url": "https://arxiv.org/pdf/2605.11887"}, {"title": "Discovering Millions of Interpretable Features with Sparse Autoencoders", "url": "https://arxiv.org/html/2606.26620v1"}]}, "Based on the search results, here's what I found about Qwen-Scope SAE models available on Hugging Face in 2026:\n\n## Qwen-Scope Overview\n\nQwen-Scope is an interpretability module trained on the Qwen3 and Qwen3.5 series models, with Sparse Autoencoders (SAEs) integrated and trained within Qwen's hidden layers. By implementing sparsity constraints, the SAEs automatically extract data features that are highly decoupled, low-redundancy, and significantly more interpretable.\n\n## Available Models\n\nThe release comprises 14 groups of SAE weights across 7 model variants — five dense models (Qwen3-1.7B, Qwen3-8B, Qwen3.5-2B, Qwen3.5-9B, and Qwen3.5-27B) and two mixture-of-experts (MoE) models (Qwen3-30B-A3B and Qwen3.5-35B-A3B).\n\nSpecifically for the Qwen3-8B models mentioned in your query, available SAE variants include:\n- SAE-Res-Qwen3-8B-Base-W64K-L0_50\n- SAE-Res-Qwen3-8B-Base-W64K-L0_100\n\n## Applications\n\nQwen-Scope can be used to analyze the internal mechanisms of Qwen's behavior and holds potential for model optimization, with application scenarios including steerable inference control, evaluation sample distribution analysis and comparison, data classification and synthesis, and model training and optimization."], "durationSeconds": 9.727137731000024, "searchCount": 1}
```

### [60] ASSISTANT · 2026-07-17 09:37:20 UTC

```
I now have a comprehensive picture of the landscape. Let me synthesize and write the hypothesis.

**Summary of what's already done:**
- LatentBiopsy: 3-way comparison (base/instruct/abliterated Qwen3) but only using PCA/angular geometry of residual streams — no SAE feature analysis
- Qwen-Scope: SAEs trained on Qwen3-8B BASE model, available on HuggingFace
- SFT investigation (2605.11426): uses pretrained SAEs as frozen diagnostic to study base→instruct, but doesn't study the abliteration direction
- SRA (2601.08489): addresses polysemanticity of refusal vector, but doesn't track individual feature reversibility
- Safe-SAIL: SAE safety features on Qwen2.5, focused on toxicity categories
- Abliterated Qwen3-8B: publicly available (huihui-ai/Qwen3-8B-abliterated)

**The genuine gap:** No paper has used a fixed SAE basis to classify RLHF-induced feature changes by their **abliteration-reversibility**, revealing which features represent "refusal circuitry" (reversed by abliteration) vs. "harm semantic understanding" (permanently imprinted by RLHF).
```

### [61] TOOL CALL — Write · 2026-07-17 09:37:20 UTC

```
File: /ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_1/gen_hypo/claude_agent/.terminal_claude_agent_struct_out.json

{
  "title": "Latent Safety Imprinting in RLHF",
  "hypothesis": "Safety RLHF fine-tuning modifies Qwen3-8B's internal representations in two mechanistically distinct ways, distinguishable only when comparing all three model variants (base, instruct, abliterated) through a fixed Sparse Autoencoder (SAE) basis. The first type — 'Refusal-Coupled' (Type-R) features — are reversed when abliteration is applied; they implement the explicit refusal machinery and their activation pattern in the abliterated model reverts close to the base model's. The second type — 'Semantically-Imprinted' (Type-I) features — represent the model's understanding of harmful content and persist structurally unchanged after abliteration; the abliterated model's activations on these features remain closer to the safety-tuned instruct model than to the base model. This creates a 'latent safety imprint': despite identical compliance behavior to the base model on harmful prompts, the abliterated model retains the instruct model's representational fingerprint on harm-concept features in the SAE basis. The degree of imprinting (fraction of safety-induced feature changes that survive abliteration) further predicts the residual capability gap between abliterated and base models, because Type-I features are also entangled with general semantic processing.",
  "motivation": "The standard narrative treats abliteration as 'undoing' safety fine-tuning, restoring the model to a base-like state. This has major implications for AI safety: if abliteration truly erases RLHF, then safety alignment is a superficial behavioral patch that can be trivially removed. If, however, RLHF writes two kinds of changes — one reversible, one permanent — then safety training has a deeper effect on the model's world model than previously recognized, and abliteration only removes the behavioral surface layer while leaving a 'cryptomnesic' safety trace. Understanding which aspects of safety survive abliteration is critical for: (1) evaluating the true robustness of safety alignment, (2) identifying candidate features for lightweight safety restoration without full retraining, and (3) providing a feature-level account of why abliterated models consistently underperform base models despite theoretically similar weight norms. The Qwen3-8B ecosystem is uniquely positioned for this study: official base and instruct variants, publicly released Qwen-Scope base-SAEs (SAE-Res-Qwen3-8B-Base-W64K-L0_50), and community-created abliterated models (huihui-ai/Qwen3-8B-abliterated) all exist and are freely accessible.",
  "assumptions": [
    "The Qwen-Scope base-SAE captures enough of the relevant semantic feature space that projecting instruct and abliterated model activations through it produces interpretable, non-random feature activations (the base and instruct models share a broadly similar residual-stream geometry, which is supported by prior work showing high cosine similarity between pre- and post-alignment activations).",
    "Abliteration using the standard difference-of-means + orthogonal projection technique produces a well-defined 'abliteration direction' that can be compared against individual SAE feature directions, and the publicly available abliterated Qwen3-8B was produced this way.",
    "The distinction between Type-R and Type-I features is stable across prompt phrasings (i.e., a feature classified as Type-I shows consistent imprinting across diverse harmful prompts, not just for specific trigger phrases).",
    "Mid-layer residual stream activations (at the layers covered by Qwen-Scope SAEs) carry enough information to characterize the semantic changes induced by RLHF, consistent with prior findings that safety-relevant representations are concentrated in middle layers.",
    "Using raw text prompts without chat templates for all three variants provides a fair comparison that isolates safety-alignment effects from chat-template formatting artifacts, as validated by the difference-in-differences approach of prior methodology papers."
  ],
  "investigation_approach": "Three-step experiment using all-Python implementation on a single GPU (16+ GB VRAM):\n\n**Step 1 — Setup and data collection.**\nLoad three Qwen3-8B model checkpoints: base (Qwen/Qwen3-8B-Base), instruct/safety-tuned (Qwen/Qwen3-8B), and abliterated (huihui-ai/Qwen3-8B-abliterated). Load the Qwen-Scope base SAE (Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 — 64K features, JumpReLU) as a frozen measurement instrument. Collect 500 harmful prompts (from WildGuard/BeaverTails/HarmBench) and 500 benign prompts. All prompts are provided as raw text without chat templates to avoid formatting confounds. Run forward passes through each model, extract residual stream activations at the SAE-covered layer (typically a mid-layer). Project through the frozen SAE encoder to obtain 64K-dimensional sparse feature activation vectors for each (prompt, model) triple.\n\n**Step 2 — Feature classification.**\nFor each of the 64K SAE features f, compute over the harmful prompt set:\n- Safety_change(f) = mean_i |act_instruct_i(f) - act_base_i(f)| (how much RLHF changed this feature)\n- Reversal_score(f) = cosine_similarity(act_abliterated - act_base, act_instruct - act_base) averaged over i (does abliteration push the feature back toward base or retain the instruct pattern?)\n\nFeatures with high Safety_change(f) are RLHF-sensitive. Among these:\n- Type-R: low Reversal_score (abliteration reverses the change, abliterated ≈ base)\n- Type-I: high Reversal_score (abliteration does NOT reverse the change, abliterated ≈ instruct)\n\n**Step 3 — Semantic interpretation and distance analysis.**\nFor the top-100 Type-R and top-100 Type-I features by Safety_change magnitude: use the maximum-activating prompt segments as context, then call a cheap LLM via OpenRouter (e.g., Qwen2.5-7B or gemini-flash at ~$0.001/1K tokens) to generate natural-language labels for each feature. Compute centroid distances between model variants in the full feature space, Type-R subspace, and Type-I subspace separately, for harmful vs. benign prompts. Test the key prediction: d_TypeI(abliterated, instruct) < d_TypeI(abliterated, base) for harmful prompts but not for benign prompts.",
  "success_criteria": "CONFIRM hypothesis if ALL of the following hold:\n1. The Reversal_score distribution for RLHF-sensitive features (high Safety_change) is clearly bimodal — a cluster near 0 (Type-R, reversed) and a cluster near 1 (Type-I, imprinted) — rather than uniform. This rules out a continuum where every feature is partially reversed.\n2. Type-I features receive semantic labels from the LLM that are recognizably harm-concept related (e.g., weapon synthesis, deception schemas, manipulation tactics), while Type-R features receive refusal/hedging/polite-declination labels.\n3. In the Type-I feature subspace, d(abliterated, instruct) < d(abliterated, base) specifically for harmful prompts (not benign prompts). This is the core quantitative test: the abliterated model's harm-concept representations look more like the instruct model than the base model.\n4. The fraction of RLHF-sensitive features that are Type-I (vs. Type-R) positively correlates across multiple SAE layers with the capability gap between abliterated and base models on a standard benchmark (e.g., GSM8K or MMLU) — linking semantic imprinting to the capability cost of abliteration.\n\nDISCONFIRM hypothesis if: (a) the Reversal_score distribution is unimodal near 0 (all changes reversed — abliteration fully restores base model) or near 1 (all changes imprinted — abliteration does nothing semantically); (b) Type-I features receive no coherent harm-related semantic labels; or (c) d(abliterated, instruct) ≥ d(abliterated, base) in the Type-I subspace for harmful prompts.",
  "related_works": [
    "LatentBiopsy (arXiv:2603.27412, 2025): Compares base, instruct, and abliterated Qwen3.5-0.8B and Qwen2.5-0.5B through PCA-based angular analysis of residual streams, finding that harmful-prompt geometry survives refusal ablation (AUROC drops by at most 0.015). Our work differs fundamentally: LatentBiopsy uses a holistic geometric measure (PCA components, angular deviation) that collapses all feature information into a single detection score, while we decompose the change at the individual SAE feature level to identify WHICH specific semantic features survive. LatentBiopsy shows that SOMETHING survives; we characterize WHAT survives and WHY.",
    "Mechanistic Investigation of Supervised Fine-Tuning via SAEs (arXiv:2605.11426, 2025): Uses pretrained base-SAEs as a frozen diagnostic to study how SFT changes feature activations, finding significant feature divergence despite high cosine similarity of raw activations. Our work extends this approach in two key ways: (1) we apply it to the three-model trajectory base→instruct→abliterated rather than just base→instruct, and (2) we specifically characterize features by whether the abliteration operation reverses or preserves the fine-tuning change — a dimension this paper does not address.",
    "Surgical Refusal Ablation (arXiv:2601.08489, 2025): Identifies that the raw refusal vector is polysemantic, entangling refusal signals with capability circuits, and proposes spectral residualization to create a clean, targeted ablation. This work addresses which CAPABILITY directions are confounded with the refusal direction. Our hypothesis is complementary but distinct: we address which SEMANTIC KNOWLEDGE features (in the SAE feature space) survive standard abliteration regardless of whether the abliteration vector was clean or polysemantic — i.e., we study the aftermath rather than the mechanism of abliteration.",
    "What Makes and Breaks Safety Fine-tuning (arXiv:2407.10264, 2024): Studies how supervised safety fine-tuning, DPO, and unlearning project unsafe activations into the MLP null space, revealing that safety is achieved through minimal weight transformation. Our work complements this by using the SAE basis to characterize what the 'minimal transformation' actually modifies semantically, and specifically identifies which of these modifications are robust to the abliteration attack.",
    "Qwen-Scope (arXiv:2605.11887, 2026): Releases 14 groups of SAEs across 7 Qwen model variants including Qwen3-8B, demonstrating their use for steering, safety classification, and post-training. We directly use these SAEs as the measurement instrument for our three-model comparison. Qwen-Scope provides the tool but does not itself perform the base/instruct/abliterated comparison or the feature reversibility analysis we propose."
  ],
  "inspiration": "The core inspiration comes from cognitive neuroscience, specifically the **implicit vs. explicit memory dissociation** (Tulving, 1972; Schacter, 1987; Squire, 1992). In human memory, there are two distinct systems: declarative/explicit memory (consciously recallable facts and episodes, dependent on the hippocampus) and implicit/procedural memory (unconscious influences on behavior — priming, skills — dependent on striatum and neocortex). Classic amnesia studies (H.M., Clive Wearing) show that hippocampal damage abolishes declarative memory while leaving implicit memory intact. A patient who cannot remember learning to play piano may still improve with practice because the skill trace persists implicitly.\n\nApplied to RLHF: safety fine-tuning is analogous to encoding both explicit and implicit memories. The **explicit safety memory** is the refusal behavior — the model can consciously 'recall' that it should refuse and does so. The **implicit safety memory** is the semantic imprint — how the model internally represents and processes harmful content. Abliteration is analogous to hippocampal damage: it removes the explicit refusal behavior (the model can no longer 'recall' to refuse) while leaving the implicit semantic traces intact. The abliterated model exhibits **cryptomnesia** — it acts as if it has forgotten its safety training (it complies with harmful requests) but retains an unconscious imprint of that training in how it represents harmful concepts.\n\nAt the methodological level, the approach of using a fixed SAE basis as a 'measurement instrument' across all three model variants is inspired by **isotopic tracing in chemistry**: in metabolic labeling experiments, you introduce isotope-labeled molecules and then track where the isotope ends up after a chemical reaction. The SAE features are the 'isotopes' — a fixed vocabulary that lets us track which representations change through the reaction (RLHF fine-tuning) and which survive the second reaction (abliteration).",
  "terms": [
    {
      "term": "Sparse Autoencoder (SAE)",
      "definition": "A neural network trained to decompose model activations into a large set of sparse, near-monosemantic features. Given a residual stream activation vector (e.g., 4096-dimensional for Qwen3-8B), the SAE's encoder maps it to a high-dimensional sparse vector (e.g., 64,000 features) where typically only 50-100 features are non-zero. Each active feature ideally corresponds to a single interpretable concept. Here we use Qwen-Scope's pre-trained base-SAE as a fixed vocabulary for comparing activations across model variants."
    },
    {
      "term": "Abliteration",
      "definition": "A weight-editing technique that removes a model's refusal behavior by identifying the 'refusal direction' in the residual stream (computed as the mean difference between activations on harmful vs. harmless prompts), then orthogonalizing all weight matrices that write to the residual stream against this direction. The result is a modified model that can no longer activate the refusal direction and therefore complies with previously-refused prompts, without retraining."
    },
    {
      "term": "Type-R feature (Refusal-Coupled)",
      "definition": "An SAE feature whose activation was changed by safety RLHF fine-tuning and is reversed (restored toward the base model's activation level) when abliteration is applied. These features implement or support the explicit refusal mechanism. Operationally: features with high Safety_change and low Reversal_score in our measurement framework."
    },
    {
      "term": "Type-I feature (Semantically-Imprinted)",
      "definition": "An SAE feature whose activation was changed by safety RLHF fine-tuning and is NOT reversed when abliteration is applied — the abliterated model's activation on this feature remains close to the instruct model's, not the base model's. These features represent the model's understanding of harmful content that persists after abliteration. Operationally: features with high Safety_change and high Reversal_score."
    },
    {
      "term": "Latent Safety Imprint",
      "definition": "The set of Type-I feature modifications that safety RLHF fine-tuning writes into the model's representational space, which survive abliteration. A model with a latent safety imprint behaves like the base model (complies with harmful requests) but internally represents harmful content more like the safety-tuned instruct model. Analogous to implicit memory in neuroscience: the behavior is forgotten but the trace persists."
    },
    {
      "term": "Reversal_score",
      "definition": "For a given SAE feature f, the cosine similarity between two vectors in activation space: (act_abliterated(f) - act_base(f)) and (act_instruct(f) - act_base(f)), averaged over a set of prompts. A high reversal score (near 1) means the abliteration change goes in the SAME direction as the RLHF change — i.e., abliteration does NOT reverse the feature, it is imprinted. A low score (near 0 or negative) means abliteration reverses the RLHF change."
    },
    {
      "term": "Qwen-Scope",
      "definition": "An open-source suite of Sparse Autoencoders released by the Qwen team (May 2026) for interpretability and development use with Qwen3 and Qwen3.5 model families. Includes base-model SAEs for Qwen3-8B with 64,000 features (SAE-Res-Qwen3-8B-Base-W64K-L0_50), which we use as the fixed measurement instrument in this study."
    }
  ],
  "summary": "We hypothesize that RLHF safety fine-tuning in Qwen3-8B writes two mechanistically distinct types of changes into the model's SAE feature space: 'Refusal-Coupled' features (reversed by abliteration) and 'Semantically-Imprinted' features (permanently changed, not reversed by abliteration). Using the Qwen-Scope base SAE as a fixed measurement instrument across base, instruct, and abliterated Qwen3-8B variants, we test whether the abliterated model retains the instruct model's representational fingerprint on harm-concept features despite exhibiting base-model-like compliance behavior — a 'latent safety imprint' analogous to implicit memory surviving hippocampal damage in cognitive neuroscience."
}
```

### [62] TOOL RESULT — Write · 2026-07-17 09:37:20 UTC

```
{"type": "create", "filePath": "/ai-inventor/aii_data/runs/run_hql72-TZXK98/iter_1/gen_hypo/claude_agent/.terminal_claude_agent_struct_out.json", "content": "{\n  \"title\": \"Latent Safety Imprinting in RLHF\",\n  \"hypothesis\": \"Safety RLHF fine-tuning modifies Qwen3-8B's internal representations in two mechanistically distinct ways, distinguishable only when comparing all three model variants (base, instruct, abliterated) through a fixed Sparse Autoencoder (SAE) basis. The first type — 'Refusal-Coupled' (Type-R) features — are reversed when abliteration is applied; they implement the explicit refusal machinery and their activation pattern in the abliterated model reverts close to the base model's. The second type — 'Semantically-Imprinted' (Type-I) features — represent the model's understanding of harmful content and persist structurally unchanged after abliteration; the abliterated model's activations on these features remain closer to the safety-tuned instruct model than to the base model. This creates a 'latent safety imprint': despite identical compliance behavior to the base model on harmful prompts, the abliterated model retains the instruct model's representational fingerprint on harm-concept features in the SAE basis. The degree of imprinting (fraction of safety-induced feature changes that survive abliteration) further predicts the residual capability gap between abliterated and base models, because Type-I features are also entangled with general semantic processing.\",\n  \"motivation\": \"The standard narrative treats abliteration as 'undoing' safety fine-tuning, restoring the model to a base-like state. This has major implications for AI safety: if abliteration truly erases RLHF, then safety alignment is a superficial behavioral patch that can be trivially removed. If, however, RLHF writes two kinds of changes — one reversible, one permanent — then safety training has a deeper effect on the model's world model than previously recognized, and abliteration only removes the behavioral surface layer while leaving a 'cryptomnesic' safety trace. Understanding which aspects of safety survive abliteration is critical for: (1) evaluating the true robustness of safety alignment, (2) identifying candidate features for lightweight safety restoration without full retraining, and (3) providing a feature-level account of why abliterated models consistently underperform base models despite theoretically similar weight norms. The Qwen3-8B ecosystem is uniquely positioned for this study: official base and instruct variants, publicly released Qwen-Scope base-SAEs (SAE-Res-Qwen3-8B-Base-W64K-L0_50), and community-created abliterated models (huihui-ai/Qwen3-8B-abliterated) all exist and are freely accessible.\",\n  \"assumptions\": [\n    \"The Qwen-Scope base-SAE captures enough of the relevant semantic feature space that projecting instruct and abliterated model activations through it produces interpretable, non-random feature activations (the base and instruct models share a broadly similar residual-stream geometry, which is supported by prior work showing high cosine similarity between pre- and post-alignment activations).\",\n    \"Abliteration using the standard difference-of-means + orthogonal projection technique produces a well-defined 'abliteration direction' that can be compared against individual SAE feature directions, and the publicly available abliterated Qwen3-8B was produced this way.\",\n    \"The distinction between Type-R and Type-I features is stable across prompt phrasings (i.e., a feature classified as Type-I shows consistent imprinting across diverse harmful prompts, not just for specific trigger phrases).\",\n    \"Mid-layer residual stream activations (at the layers covered by Qwen-Scope SAEs) carry enough information to characterize the semantic changes induced by RLHF, consistent with prior findings that safety-relevant representations are concentrated in middle layers.\",\n    \"Using raw text prompts without chat templates for all three variants provides a fair comparison that isolates safety-alignment effects from chat-template formatting artifacts, as validated by the difference-in-differences approach of prior methodology papers.\"\n  ],\n  \"investigation_approach\": \"Three-step experiment using all-Python implementation on a single GPU (16+ GB VRAM):\\n\\n**Step 1 — Setup and data collection.**\\nLoad three Qwen3-8B model checkpoints: base (Qwen/Qwen3-8B-Base), instruct/safety-tuned (Qwen/Qwen3-8B), and abliterated (huihui-ai/Qwen3-8B-abliterated). Load the Qwen-Scope base SAE (Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 — 64K features, JumpReLU) as a frozen measurement instrument. Collect 500 harmful prompts (from WildGuard/BeaverTails/HarmBench) and 500 benign prompts. All prompts are provided as raw text without chat templates to avoid formatting confounds. Run forward passes through each model, extract residual stream activations at the SAE-covered layer (typically a mid-layer). Project through the frozen SAE encoder to obtain 64K-dimensional sparse feature activation vectors for each (prompt, model) triple.\\n\\n**Step 2 — Feature classification.**\\nFor each of the 64K SAE features f, compute over the harmful prompt set:\\n- Safety_change(f) = mean_i |act_instruct_i(f) - act_base_i(f)| (how much RLHF changed this feature)\\n- Reversal_score(f) = cosine_similarity(act_abliterated - act_base, act_instruct - act_base) averaged over i (does abliteration push the feature back toward base or retain the instruct pattern?)\\n\\nFeatures with high Safety_change(f) are RLHF-sensitive. Among these:\\n- Type-R: low Reversal_score (abliteration reverses the change, abliterated ≈ base)\\n- Type-I: high Reversal_score (abliteration does NOT reverse the change, abliterated ≈ instruct)\\n\\n**Step 3 — Semantic interpretation and distance analysis.**\\nFor the top-100 Type-R and top-100 Type-I features by Safety_change magnitude: use the maximum-activating prompt segments as context, then call a cheap LLM via OpenRouter (e.g., Qwen2.5-7B or gemini-flash at ~$0.001/1K tokens) to generate natural-language labels for each feature. Compute centroid distances between model variants in the full feature space, Type-R subspace, and Type-I subspace separately, for harmful vs. benign prompts. Test the key prediction: d_TypeI(abliterated, instruct) < d_TypeI(abliterated, base) for harmful prompts but not for benign prompts.\",\n  \"success_criteria\": \"CONFIRM hypothesis if ALL of the following hold:\\n1. The Reversal_score distribution for RLHF-sensitive features (high Safety_change) is clearly bimodal — a cluster near 0 (Type-R, reversed) and a cluster near 1 (Type-I, imprinted) — rather than uniform. This rules out a continuum where every feature is partially reversed.\\n2. Type-I features receive semantic labels from the LLM that are recognizably harm-concept related (e.g., weapon synthesis, deception schemas, manipulation tactics), while Type-R features receive refusal/hedging/polite-declination labels.\\n3. In the Type-I feature subspace, d(abliterated, instruct) < d(abliterated, base) specifically for harmful prompts (not benign prompts). This is the core quantitative test: the abliterated model's harm-concept representations look more like the instruct model than the base model.\\n4. The fraction of RLHF-sensitive features that are Type-I (vs. Type-R) positively correlates across multiple SAE layers with the capability gap between abliterated and base models on a standard benchmark (e.g., GSM8K or MMLU) — linking semantic imprinting to the capability cost of abliteration.\\n\\nDISCONFIRM hypothesis if: (a) the Reversal_score distribution is unimodal near 0 (all changes reversed — abliteration fully restores base model) or near 1 (all changes imprinted — abliteration does nothing semantically); (b) Type-I features receive no coherent harm-related semantic labels; or (c) d(abliterated, instruct) ≥ d(abliterated, base) in the Type-I subspace for harmful prompts.\",\n  \"related_works\": [\n    \"LatentBiopsy (arXiv:2603.27412, 2025): Compares base, instruct, and abliterated Qwen3.5-0.8B and Qwen2.5-0.5B through PCA-based angular analysis of residual streams, finding that harmful-prompt geometry survives refusal ablation (AUROC drops by at most 0.015). Our work differs fundamentally: LatentBiopsy uses a holistic geometric measure (PCA components, angular deviation) that collapses all feature information into a single detection score, while we decompose the change at the individual SAE feature level to identify WHICH specific semantic features survive. LatentBiopsy shows that SOMETHING survives; we characterize WHAT survives and WHY.\",\n    \"Mechanistic Investigation of Supervised Fine-Tuning via SAEs (arXiv:2605.11426, 2025): Uses pretrained base-SAEs as a frozen diagnostic to study how SFT changes feature activations, finding significant feature divergence despite high cosine similarity of raw activations. Our work extends this approach in two key ways: (1) we apply it to the three-model trajectory base→instruct→abliterated rather than just base→instruct, and (2) we specifically characterize features by whether the abliteration operation reverses or preserves the fine-tuning change — a dimension this paper does not address.\",\n    \"Surgical Refusal Ablation (arXiv:2601.08489, 2025): Identifies that the raw refusal vector is polysemantic, entangling refusal signals with capability circuits, and proposes spectral residualization to create a clean, targeted ablation. This work addresses which CAPABILITY directions are confounded with the refusal direction. Our hypothesis is complementary but distinct: we address which SEMANTIC KNOWLEDGE features (in the SAE feature space) survive standard abliteration regardless of whether the abliteration vector was clean or polysemantic — i.e., we study the aftermath rather than the mechanism of abliteration.\",\n    \"What Makes and Breaks Safety Fine-tuning (arXiv:2407.10264, 2024): Studies how supervised safety fine-tuning, DPO, and unlearning project unsafe activations into the MLP null space, revealing that safety is achieved through minimal weight transformation. Our work complements this by using the SAE basis to characterize what the 'minimal transformation' actually modifies semantically, and specifically identifies which of these modifications are robust to the abliteration attack.\",\n    \"Qwen-Scope (arXiv:2605.11887, 2026): Releases 14 groups of SAEs across 7 Qwen model variants including Qwen3-8B, demonstrating their use for steering, safety classification, and post-training. We directly use these SAEs as the measurement instrument for our three-model comparison. Qwen-Scope provides the tool but does not itself perform the base/instruct/abliterated comparison or the feature reversibility analysis we propose.\"\n  ],\n  \"inspiration\": \"The core inspiration comes from cognitive neuroscience, specifically the **implicit vs. explicit memory dissociation** (Tulving, 1972; Schacter, 1987; Squire, 1992). In human memory, there are two distinct systems: declarative/explicit memory (consciously recallable facts and episodes, dependent on the hippocampus) and implicit/procedural memory (unconscious influences on behavior — priming, skills — dependent on striatum and neocortex). Classic amnesia studies (H.M., Clive Wearing) show that hippocampal damage abolishes declarative memory while leaving implicit memory intact. A patient who cannot remember learning to play piano may still improve with practice because the skill trace persists implicitly.\\n\\nApplied to RLHF: safety fine-tuning is analogous to encoding both explicit and implicit memories. The **explicit safety memory** is the refusal behavior — the model can consciously 'recall' that it should refuse and does so. The **implicit safety memory** is the semantic imprint — how the model internally represents and processes harmful content. Abliteration is analogous to hippocampal damage: it removes the explicit refusal behavior (the model can no longer 'recall' to refuse) while leaving the implicit semantic traces intact. The abliterated model exhibits **cryptomnesia** — it acts as if it has forgotten its safety training (it complies with harmful requests) but retains an unconscious imprint of that training in how it represents harmful concepts.\\n\\nAt the methodological level, the approach of using a fixed SAE basis as a 'measurement instrument' across all three model variants is inspired by **isotopic tracing in chemistry**: in metabolic labeling experiments, you introduce isotope-labeled molecules and then track where the isotope ends up after a chemical reaction. The SAE features are the 'isotopes' — a fixed vocabulary that lets us track which representations change through the reaction (RLHF fine-tuning) and which survive the second reaction (abliteration).\",\n  \"terms\": [\n    {\n      \"term\": \"Sparse Autoencoder (SAE)\",\n      \"definition\": \"A neural network trained to decompose model activations into a large set of sparse, near-monosemantic features. Given a residual stream activation vector (e.g., 4096-dimensional for Qwen3-8B), the SAE's encoder maps it to a high-dimensional sparse vector (e.g., 64,000 features) where typically only 50-100 features are non-zero. Each active feature ideally corresponds to a single interpretable concept. Here we use Qwen-Scope's pre-trained base-SAE as a fixed vocabulary for comparing activations across model variants.\"\n    },\n    {\n      \"term\": \"Abliteration\",\n      \"definition\": \"A weight-editing technique that removes a model's refusal behavior by identifying the 'refusal direction' in the residual stream (computed as the mean difference between activations on harmful vs. harmless prompts), then orthogonalizing all weight matrices that write to the residual stream against this direction. The result is a modified model that can no longer activate the refusal direction and therefore complies with previously-refused prompts, without retraining.\"\n    },\n    {\n      \"term\": \"Type-R feature (Refusal-Coupled)\",\n      \"definition\": \"An SAE feature whose activation was changed by safety RLHF fine-tuning and is reversed (restored toward the base model's activation level) when abliteration is applied. These features implement or support the explicit refusal mechanism. Operationally: features with high Safety_change and low Reversal_score in our measurement framework.\"\n    },\n    {\n      \"term\": \"Type-I feature (Semantically-Imprinted)\",\n      \"definition\": \"An SAE feature whose activation was changed by safety RLHF fine-tuning and is NOT reversed when abliteration is applied — the abliterated model's activation on this feature remains close to the instruct model's, not the base model's. These features represent the model's understanding of harmful content that persists after abliteration. Operationally: features with high Safety_change and high Reversal_score.\"\n    },\n    {\n      \"term\": \"Latent Safety Imprint\",\n      \"definition\": \"The set of Type-I feature modifications that safety RLHF fine-tuning writes into the model's representational space, which survive abliteration. A model with a latent safety imprint behaves like the base model (complies with harmful requests) but internally represents harmful content more like the safety-tuned instruct model. Analogous to implicit memory in neuroscience: the behavior is forgotten but the trace persists.\"\n    },\n    {\n      \"term\": \"Reversal_score\",\n      \"definition\": \"For a given SAE feature f, the cosine similarity between two vectors in activation space: (act_abliterated(f) - act_base(f)) and (act_instruct(f) - act_base(f)), averaged over a set of prompts. A high reversal score (near 1) means the abliteration change goes in the SAME direction as the RLHF change — i.e., abliteration does NOT reverse the feature, it is imprinted. A low score (near 0 or negative) means abliteration reverses the RLHF change.\"\n    },\n    {\n      \"term\": \"Qwen-Scope\",\n      \"definition\": \"An open-source suite of Sparse Autoencoders released by the Qwen team (May 2026) for interpretability and development use with Qwen3 and Qwen3.5 model families. Includes base-model SAEs for Qwen3-8B with 64,000 features (SAE-Res-Qwen3-8B-Base-W64K-L0_50), which we use as the fixed measurement instrument in this study.\"\n    }\n  ],\n  \"summary\": \"We hypothesize that RLHF safety fine-tuning in Qwen3-8B writes two mechanistically distinct types of changes into the model's SAE feature space: 'Refusal-Coupled' features (reversed by abliteration) and 'Semantically-Imprinted' features (permanently changed, not reversed by abliteration). Using the Qwen-Scope base SAE as a fixed measurement instrument across base, instruct, and abliterated Qwen3-8B variants, we test whether the abliterated model retains the instruct model's representational fingerprint on harm-concept features despite exhibiting base-model-like compliance behavior — a 'latent safety imprint' analogous to implicit memory surviving hippocampal damage in cognitive neuroscience.\"\n}\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [63] ASSISTANT · 2026-07-17 09:37:37 UTC

```
The hypothesis has been written to `.terminal_claude_agent_struct_out.json`. Here's the core idea:

---

**Hypothesis: "Latent Safety Imprinting in RLHF"**

RLHF safety fine-tuning in Qwen3-8B writes two mechanistically distinct types of changes into the model's SAE feature space:

- **Type-R (Refusal-Coupled)**: Features that implement explicit refusal behavior — reversed when abliteration is applied (abliterated model reverts to base)
- **Type-I (Semantically-Imprinted)**: Features encoding the model's *understanding* of harmful content — **persist after abliteration** (abliterated model stays close to instruct model in this subspace)

The abliterated model thus carries a **"latent safety imprint"**: it behaves like the base model (complies with harmful requests) but its internal representations of harmful topics remain closer to the safety-tuned model than the original base model.

**Why this is novel**: LatentBiopsy (2603.27412) showed *something* survives abliteration geometrically; no work has decomposed *which specific SAE features* survive and characterized them semantically. The three-way feature reversibility taxonomy (base → instruct → abliterated) using a fixed SAE basis hasn't been done.

**What makes it feasible**: All three Qwen3-8B variants are on HuggingFace, Qwen-Scope base SAEs (64K features) are publicly released, LLM labeling costs under $1 with cheap OpenRouter models.

**Cross-domain inspiration**: The implicit vs. explicit memory dissociation in cognitive neuroscience — abliteration removes explicit safety behavior (like hippocampal damage removes declarative memory) while leaving implicit representational traces intact.
```
