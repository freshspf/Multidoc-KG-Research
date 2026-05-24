# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

This repository is a **biomedical literature knowledge graph construction framework**.

The current target is:

- ingest biomedical paper JSON converted from PDF
- classify each paper into a biomedical subdomain
- maintain a batch-evolving subdomain taxonomy
- extract biomedical triples
- ground entities against the graph
- validate claims against evidence and graph context
- write validated claims into Neo4j

This is **not** an ancient-text / TCM project anymore.

## Current Pipeline

The current mainline is:

```text
raw PDFs
-> preprocessing / cleaning
-> paper JSON
-> PaperDataLoader
-> subdomain classification
-> per-batch subdomain refinement
-> extraction
-> grounding
-> validation
-> evolution
-> Neo4j
```

## Main Entry Point

Primary orchestration is in:

- `/Users/joer/Gitroom/Multidoc-KG-zya/main.py`

Typical full run:

```bash
python main.py --data-dir data/cleaned_papers --batch-size 5
```

Start from a clean Neo4j database:

```bash
python main.py --data-dir data/cleaned_papers --batch-size 5 --clear-db
```

## Important Modes

Subdomain only:

```bash
python main.py --data-dir data/cleaned_papers --subdomain-only
```

Subdomain graph only:

```bash
python main.py --data-dir data/cleaned_papers --batch-size 5 --subdomain-graph-only
```

Extraction only:

```bash
python main.py --data-dir data/cleaned_papers --extraction-only
```

Standalone refinement:

```bash
python main.py --refine-subdomains-only
```

Embedding model load test:

```bash
python scripts/test_vector_model.py --model BAAI/bge-m3
```

Stage export for advisor review:

```bash
python scripts/export_stage_outputs.py --clear-db
```

Purpose:
- export 5-paper stage-by-stage outputs for manual inspection
- include preprocess, subdomain, extraction, grounding, validation, and evolution results
- write outputs under `reports/stage_outputs_<timestamp>/`

## Core Agents

### 1. Subdomain Classifier

- file: `/Users/joer/Gitroom/Multidoc-KG-zya/agents/subdomain.py`
- purpose: assign a biomedical subdomain using the current hierarchy
- output: `SubdomainAssignment`

Key point:
- classification is hierarchy-aware
- new subdomains become candidates first

### 2. Subdomain Refinement

- file: `/Users/joer/Gitroom/Multidoc-KG-zya/agents/subdomain_refinement.py`
- purpose: decide whether each candidate should be merged or promoted

Key point:
- refinement now runs automatically after each classification batch in the mainline

### 3. Extraction Agent

- file: `/Users/joer/Gitroom/Multidoc-KG-zya/agents/extraction.py`
- purpose: extract ontology-layer and instance-layer biomedical claims

Key point:
- relations are now controlled biomedical relations
- prompt uses paper subdomain only as a soft prior

### 4. Grounding Agent

- file: `/Users/joer/Gitroom/Multidoc-KG-zya/agents/grounding.py`
- purpose: align entities to existing graph nodes

Key point:
- embedding retrieval first
- LLM decides `merge` or `new`

### 5. Validation Agent

- file: `/Users/joer/Gitroom/Multidoc-KG-zya/agents/validation.py`
- purpose: decide whether a grounded claim is acceptable

Key point:
- evidence-first validation
- uses graph context plus paper context

### 6. Evolution Agent

- file: `/Users/joer/Gitroom/Multidoc-KG-zya/agents/evolution.py`
- purpose: write validated claims to Neo4j

## Graph Model Notes

The graph currently contains both final objects and lifecycle objects.

Final objects:

- `Paper`
- `Subdomain`
- `Entity`

Lifecycle objects:

- `SubdomainCandidate`

Important relations:

- `CLASSIFIED_AS`
- `SUBCLASS_OF`
- `SUGGESTS_SUBDOMAIN`
- `CANDIDATE_SUBCLASS_OF`
- `PROMOTED_TO`
- `MERGED_INTO`

Important implication:

- seeing `SubdomainCandidate` nodes in Neo4j does **not** mean refinement failed
- candidates may remain as audit / lifecycle records after refinement

## Storage / Infrastructure

### Neo4j

- store implementation: `/Users/joer/Gitroom/Multidoc-KG-zya/core/neo4j_store.py`
- environment variables:
  - `NEO4J_URI`
  - `NEO4J_USER`
  - `NEO4J_PASSWORD`

### LLM Client

- file: `/Users/joer/Gitroom/Multidoc-KG-zya/core/llm_client.py`

Current behavior:

- supports custom OpenAI-compatible providers
- now includes automatic retry and exponential backoff for transient failures

Relevant env vars:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `LLM_MAX_RETRIES`
- `LLM_RETRY_BASE_DELAY`
- `LLM_RETRY_MAX_DELAY`

### Vector Store

- file: `/Users/joer/Gitroom/Multidoc-KG-zya/core/vector_store.py`
- default embedding model: `BAAI/bge-m3`

Mainline fallback:

- if vector model initialization fails, mainline falls back to `MockVectorStore`
- this is a resilience fallback, not the preferred production path

## Data Model

Main schema definitions live in:

- `/Users/joer/Gitroom/Multidoc-KG-zya/schema.py`

Key models:

- `Paper`
- `KnowledgeClaim`
- `SubdomainAssignment`
- `ClaimStatus`

Current `ClaimStatus` progression:

- `EXTRACTED`
- `GROUNDED`
- `VALIDATED`
- `REJECTED`

## Documentation To Check First

- `/Users/joer/Gitroom/Multidoc-KG-zya/reports/current_experiment_pipeline.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/reports/progress_tracker.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/reports/各阶段核心Prompt设计总结_2026-04-03.md`

## Documentation Maintenance Rules

When finishing any **major step** in this repository, Claude must update the relevant docs in the same working session before stopping.

A "major step" includes, but is not limited to:

- a pipeline stage being newly wired into the mainline
- a stage changing from manual to automatic behavior
- prompt strategy changing in a meaningful way
- graph schema / Neo4j behavior changing
- run commands or debug workflow changing
- adding or changing stage export / advisor review workflow
- experiment conclusions materially changing

At minimum, Claude should check and update these files when relevant:

- `/Users/joer/Gitroom/Multidoc-KG-zya/README.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/CLAUDE.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/AGENTS.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/reports/current_experiment_pipeline.md`
- `/Users/joer/Gitroom/Multidoc-KG-zya/reports/progress_tracker.md`

If prompt logic changes materially, Claude should also update:

- `/Users/joer/Gitroom/Multidoc-KG-zya/reports/各阶段核心Prompt设计总结_2026-04-03.md`

If a file does not need changes for the current step, Claude should still explicitly check whether it remains accurate before deciding not to edit it.

## Important Repository Notes

- `bookmark_based_splitter.py` is still the real preprocessing implementation.
- The repo has already been cleaned of most legacy TCM / ancient-text artifacts.
- When updating docs or prompts, assume the biomedical pipeline is the only active mainline unless the user explicitly asks to revive legacy work.
