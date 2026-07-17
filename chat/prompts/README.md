# Prompts

Complete, auto-generated record of **every prompt the AI Inventor system gave each agent** across this run — generated at repository-upload time so it captures all steps. For the full conversation (assistant turns, thinking, tool calls and results) see the sibling `../messages/` folder.

- Run: `run_hql72-TZXK98` — Abliteration Cannot Erase RLHF: Geometric Evidence from SAE Encoder Directions in Qwen3-8B

Each prompt is labelled by type and timestamped, with its full untruncated body:

- **SYSTEM-USER** — the pipeline-generated role/instruction prompt placed in the user slot.
- **HUMAN-USER** — the task / human-typed message into the agent stream.
- **SKILL-INPUT** — a skill the agent loaded; its `SKILL.md` instructions, verbatim.

Layout mirrors the run's module tree: one folder per high-level phase, a `round_N/` per iteration where the phase iterates, then each module — a single-task module is one `.md` file, a parallel module (gen_plan / gen_art / gen_viz / gen_demo_art) is a folder with one `.md` per task.

## Index

- **1. create_idea** — `hypo_loop`
  - round_1
    - `chat/prompts/1_create_idea/round_1/1_gen_hypo.md` — 3 prompts
    - `chat/prompts/1_create_idea/round_1/2_review_hypo.md` — 3 prompts
  - round_2
    - `chat/prompts/1_create_idea/round_2/1_gen_hypo.md` — 3 prompts
    - `chat/prompts/1_create_idea/round_2/2_review_hypo.md` — 5 prompts
  - round_3
    - `chat/prompts/1_create_idea/round_3/1_gen_hypo.md` — 3 prompts
    - `chat/prompts/1_create_idea/round_3/2_review_hypo.md` — 4 prompts
- **2. test_idea** — `invention_loop`
  - round_1
    - `chat/prompts/2_test_idea/round_1/1_gen_strat.md` — 2 prompts
    - `2_gen_plan/` — 2 task(s)
      - `chat/prompts/2_test_idea/round_1/2_gen_plan/gen_plan_dataset_1.md` — 4 prompts
      - `chat/prompts/2_test_idea/round_1/2_gen_plan/gen_plan_research_1.md` — 3 prompts
    - `3_gen_art/` — 2 task(s)
      - `chat/prompts/2_test_idea/round_1/3_gen_art/gen_art_dataset_1.md` — 13 prompts
      - `chat/prompts/2_test_idea/round_1/3_gen_art/gen_art_research_1.md` — 5 prompts
    - `chat/prompts/2_test_idea/round_1/4_gen_paper_text.md` — 5 prompts
    - `chat/prompts/2_test_idea/round_1/5_review_paper.md` — 3 prompts
    - `chat/prompts/2_test_idea/round_1/6_upd_hypo.md` — 3 prompts
  - round_2
    - `chat/prompts/2_test_idea/round_2/1_gen_strat.md` — 3 prompts
    - `2_gen_plan/` — 2 task(s)
      - `chat/prompts/2_test_idea/round_2/2_gen_plan/gen_plan_experiment_1.md` — 3 prompts
      - `chat/prompts/2_test_idea/round_2/2_gen_plan/gen_plan_experiment_2.md` — 2 prompts
    - `3_gen_art/` — 2 task(s)
      - `chat/prompts/2_test_idea/round_2/3_gen_art/gen_art_experiment_1.md` — 16 prompts
      - `chat/prompts/2_test_idea/round_2/3_gen_art/gen_art_experiment_2.md` — 16 prompts
    - `chat/prompts/2_test_idea/round_2/4_gen_paper_text.md` — 4 prompts
    - `chat/prompts/2_test_idea/round_2/5_review_paper.md` — 2 prompts
    - `chat/prompts/2_test_idea/round_2/6_upd_hypo.md` — 4 prompts
  - round_3
    - `chat/prompts/2_test_idea/round_3/1_gen_strat.md` — 3 prompts
    - `2_gen_plan/` — 1 task(s)
      - `chat/prompts/2_test_idea/round_3/2_gen_plan/gen_plan_experiment_1.md` — 2 prompts
    - `3_gen_art/` — 1 task(s)
      - `chat/prompts/2_test_idea/round_3/3_gen_art/gen_art_experiment_1.md` — 14 prompts
    - `chat/prompts/2_test_idea/round_3/4_gen_paper_text.md` — 5 prompts
    - `chat/prompts/2_test_idea/round_3/5_review_paper.md` — 3 prompts
    - `chat/prompts/2_test_idea/round_3/6_upd_hypo.md` — 3 prompts
  - round_4
    - `chat/prompts/2_test_idea/round_4/1_gen_strat.md` — 3 prompts
    - `2_gen_plan/` — 1 task(s)
      - `chat/prompts/2_test_idea/round_4/2_gen_plan/gen_plan_experiment_1.md` — 3 prompts
    - `3_gen_art/` — 1 task(s)
      - `chat/prompts/2_test_idea/round_4/3_gen_art/gen_art_experiment_1.md` — 9 prompts
    - `chat/prompts/2_test_idea/round_4/4_gen_paper_text.md` — 5 prompts
    - `chat/prompts/2_test_idea/round_4/5_review_paper.md` — 3 prompts
    - `chat/prompts/2_test_idea/round_4/6_upd_hypo.md` — 3 prompts
  - round_5
    - `chat/prompts/2_test_idea/round_5/1_gen_strat.md` — 3 prompts
    - `2_gen_plan/` — 2 task(s)
      - `chat/prompts/2_test_idea/round_5/2_gen_plan/gen_plan_experiment_1.md` — 2 prompts
      - `chat/prompts/2_test_idea/round_5/2_gen_plan/gen_plan_research_1.md` — 3 prompts
    - `chat/prompts/2_test_idea/round_5/3_gen_paper_text.md` — 5 prompts
    - `chat/prompts/2_test_idea/round_5/4_review_paper.md` — 3 prompts
    - `chat/prompts/2_test_idea/round_5/5_upd_hypo.md` — 3 prompts
- **3. report_results** — `gen_paper_repo`
  - `chat/prompts/3_report_results/1_gen_full_paper.md` — 4 prompts
