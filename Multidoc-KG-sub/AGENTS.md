# AGENTS.md

This file provides repo-level working rules for coding agents operating in this project.

## Project Scope

This repository is a **biomedical literature knowledge graph** project.

Active mainline:

```text
preprocess
-> paper loading
-> subdomain classification
-> per-batch subdomain refinement
-> extraction
-> grounding
-> validation
-> evolution
-> Neo4j
```

Do not treat this repository as an ancient-text / TCM project unless the user explicitly asks to revisit legacy work.

## Primary Source Of Truth

Before making substantial changes, check these files first:

- `/Users/joer/Gitroom/Multidoc-KG-zya/README.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/CLAUDE.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/reports/current_experiment_pipeline.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/reports/progress_tracker.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/reports/各阶段核心Prompt设计总结_2026-04-03.md`

## Major-Step Documentation Rule

Whenever an agent finishes a **major step**, it must update the relevant docs in the same session.

Examples of major steps:

- wiring a new stage into the main pipeline
- changing subdomain behavior, especially batch refinement behavior
- changing extraction prompt strategy or relation schema
- changing grounding / validation / Neo4j write semantics
- changing run commands, debugging scripts, or setup workflow
- materially changing experimental conclusions

At minimum, the agent should evaluate whether these need updates:

- `/Users/joer/Gitroom/Multidoc-KG-zya/README.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/CLAUDE.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/AGENTS.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/reports/current_experiment_pipeline.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/reports/progress_tracker.md`

And if prompt design changed:

- `/Users/joer/Gitroom/Multidoc-KG-zya/reports/各阶段核心Prompt设计总结_2026-04-03.md`

## Current Behavioral Notes

- Subdomain classification is hierarchy-aware.
- Mainline now performs **automatic refinement after each classification batch**.
- `SubdomainCandidate` nodes may remain in Neo4j as lifecycle / audit records after refinement.
- Extraction uses controlled biomedical relation labels.
- Default vector model is `BAAI/bge-m3`.
- LLM client now has automatic retry and exponential backoff for transient failures.

## Operational Notes

- Main entry point: `/Users/joer/Gitroom/Multidoc-KG-zya/main.py`
- Full run:

```bash
python main.py --data-dir data/cleaned_papers --batch-size 5
```

- Clean-start full run:

```bash
python main.py --data-dir data/cleaned_papers --batch-size 5 --clear-db
```

- Embedding model test:

```bash
python scripts/test_vector_model.py --model BAAI/bge-m3
```

- Advisor-facing stage export:

```bash
python scripts/export_stage_outputs.py --clear-db
```

This export script should be preferred when the user asks to inspect stage outputs for a small fixed set of papers.

## Change Discipline

- Prefer aligning code behavior and documentation in the same change set.
- If code changes but docs are intentionally left unchanged, state why.
- Prefer preserving biomedical naming consistency across prompts, config, and Neo4j schema.
