# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FRESH-bench** is a research benchmark dataset for evaluating AI/LLM systems on temporally-changing factual information. It collects Wikipedia article snapshots at multiple timestamps (2023-2025) across 598 entities in 8 categories, enabling evaluation of how well models handle knowledge that changes over time.

## Commands

```bash
# Install dependencies
pip install requests jupyter

# Run the data collection notebook
jupyter notebook wikidata_fetch.ipynb

# Diff classifier — demo on Andrej Karpathy
python diff_classifier.py

# Diff classifier — compare consecutive snapshots for one entity
python diff_classifier.py data/wikipedia/people/Elon_Musk

# Diff classifier — compare two specific snapshot files
python diff_classifier.py data/wikipedia/people/Elon_Musk/2023-01-01.json data/wikipedia/people/Elon_Musk/2024-01-01.json

# Diff classifier — batch export all entities
python diff_classifier.py --batch-all                    # → changes_full.jsonl
python diff_classifier.py --batch-all custom_output.jsonl

# QA generator — generate QA pairs from change records
python qa_generator.py                                    # changes.jsonl → qa_pairs.jsonl
python qa_generator.py --input changes.jsonl --limit 5    # test run
python qa_generator.py --input changes_full.jsonl         # full dataset
python qa_generator.py --run-diff-first                   # batch diff + QA generation

# Retrieval scenarios — generate eval instances
python retrieval_scenarios.py                              # qa_pairs_210.jsonl → eval_instances_210.jsonl
python retrieval_scenarios.py --input qa_pairs.jsonl       # custom input
python retrieval_scenarios.py --scenarios S1,S3             # subset of scenarios
python retrieval_scenarios.py --max-doc-chars 2000          # truncate doc content (default)

# Eval harness — score predictions with 6 metrics
python eval_harness.py --instances eval_instances_210.jsonl --predictions predictions.jsonl
python eval_harness.py --instances eval_instances_210.jsonl --run-predictions          # generate + score
python eval_harness.py --instances eval_instances_210.jsonl --run-predictions --predictions-only
python eval_harness.py --instances eval_instances_210.jsonl --run-predictions --model gpt-4o --limit 10
```

The `.env` file contains `OPENAI_API_KEY` — load it before running any cells that call OpenAI APIs.

## Architecture

Three main components plus retrieval scenario generation:

### `wikidata_fetch.ipynb` — Data Collection
A single Jupyter notebook (3 cells, Cell 0 is ~75K chars containing entity definitions and collection logic). Key function:
- `get_article_at_timestamp(title, timestamp, max_retries=3)` — Fetches a Wikipedia article's content at a specific timestamp via the Wikipedia API. Returns a dict with `title`, `page_id`, `revision_id`, `revision_timestamp`, and `content`.

Entity definitions (in Cell 0) have fields: `name`, `wikipedia_title`, `fame_level` (`"high"/"medium"/"low"`), and `change_type` (e.g., `"roles/companies"`, `"policy positions"`).

### `diff_classifier.py` — Change Detection & Classification
Standalone Python script implementing a pipeline to detect and classify factual changes between Wikipedia snapshots:

1. **Myers diff** (`myers_diff`) — line-level shortest edit script between two texts
2. **Change classification** (`classify_change`) — categorizes diffs as `FACTUAL_UPDATE`, `NUMERIC_UPDATE`, `ADDITION`, or `DELETION` (priority in that order)
3. **Diff context extraction** (`extract_diff_context`) — selects the most significant changed block, preferring infobox fields over prose
4. **Semantic subtyping** (`infer_subtype`) — labels changes with subtypes like `leadership_change`, `numeric_stat`, `status_change`, etc. via regex patterns
5. **Temporal gradient** (`build_temporal_gradient`) — maps all snapshots to T-N…T+M labels relative to the change event
6. **Batch processing** (`run_batch`) — writes change records as JSONL in the benchmark schema

The classifier distinguishes factual changes from cosmetic edits using wiki markup stripping (`_strip_wiki_markup`), number extraction (`_numbers_differ`), and infobox key matching (`_same_infobox_key`).

### `qa_generator.py` — QA Pair Generation
Standalone CLI script that reads change records from JSONL and generates QA pairs with mechanism hypotheses via OpenAI:

1. **`generate_qa_pairs(client, record)`** — calls OpenAI (gpt-4o-mini) to produce 3 QA pairs per change record, each with question, old/new answers, and predicted mechanism (KC/TS/PO/C)
2. **`process_changes(input_path, output_path, limit)`** — batch runner with resume support (skips already-processed change_ids)
3. **`run_diff_first(output_path)`** — convenience wrapper that runs `diff_classifier.run_batch()` on all entities before QA generation
4. CLI flags: `--input`, `--output`, `--limit`, `--model`, `--run-diff-first`

### `retrieval_scenarios.py` — Corpus Index & Retrieval Scenarios
Standalone CLI script that builds a corpus index and generates eval instances under 5 retrieval scenarios:

1. **`build_corpus(qa_path, data_root)`** — runtime index mapping `doc_id → file path` by scanning `data/wikipedia/` and resolving doc_ids from QA temporal_gradient fields
2. **`extract_passage(snapshot_path, answer_hint, max_chars)`** — extracts relevant passage from a snapshot, finding the paragraph containing the answer hint ± 1 surrounding paragraph
3. **`select_docs(qa_record, scenario)`** — selects and ranks documents per scenario: S1 (fresh only), S2 (stale only), S3 (mixed fresh-dominant), S4 (mixed stale-dominant), S5 (mixed equal)
4. **`generate_eval_instances(qa_path, ...)`** — produces JSONL eval instances (QA pair × scenario) with resolved document content
5. CLI flags: `--input`, `--output`, `--scenarios`, `--max-doc-chars`, `--data-root`

### `eval_harness.py` — Evaluation Harness
Standalone CLI script that scores model predictions using 6 metrics and optionally generates predictions via OpenAI:

1. **`token_f1(prediction, reference)`** — SQuAD-style token F1 score
2. **`score_instance(instance, prediction)`** — computes 5 per-instance metrics: FA (Factual Accuracy), HR (Hallucination Rate), AU (Answer Uncertainty), CRS (Change Recognition Score), POS (Preference for Outdated Score)
3. **`run_predictions(instances_path, output_path, model)`** — generates predictions by calling OpenAI with RAG prompt (question + retrieved docs), resume support
4. **`aggregate_results(scored)`** — groups metrics by scenario, domain, mechanism; computes TGS (Temporal Gradient Score) as Pearson correlation of FA vs temporal distance
5. **`evaluate(instances_path, predictions_path)`** — main pipeline: load, score, aggregate, write scored JSONL + summary JSON report
6. CLI flags: `--instances`, `--predictions`, `--run-predictions`, `--predictions-only`, `--model`, `--limit`, `--output-dir`

## Data Layout

```
data/wikipedia/
  {category}/
    {entity_name}/
      {timestamp}.json      # e.g., 2023-01-01.json, 2024-06-01.json
```

**Categories:** `organizations`, `people`, `policy`, `products`, `science`, `science_biology_medicine`, `science_physics_chemistry`, `sports`

**Timestamps per entity:** `2023-01-01`, `2023-06-01`, `2024-01-01`, `2024-06-01`, `2025-01-01`

**JSON file schema:**
```json
{
  "title": "string",
  "page_id": 123456,
  "revision_id": 123456,
  "revision_timestamp": "2023-01-01T00:00:00Z",
  "content": "full Wikipedia markup text"
}
```

The `data/` directory is ~440 MB (598 entities × ~5 snapshots each = ~2,792 files) and is tracked in git.

## FreshRAG Experiment Steps

### Phase 1: Data Collection — DONE
- [x] 1.1 Collect Wikipedia snapshots (598 entities × 5 timestamps, 8 categories)
- [x] 1.2 Define entity metadata (name, wikipedia_title, fame_level, change_type)
- [x] 1.3 Build diff pipeline (`diff_classifier.py`: Myers diff, change classification, context extraction)

### Phase 2: Dataset Construction (FreshRAG-QA)
- [x] 2.1 Run batch diff on all entities — produce change records JSONL via `run_batch()` (10 entities done; `--batch-all` flag for full 598)
- [x] 2.2 Generate QA pairs from detected changes using LLM (`qa_generator.py`, 3 questions per change)
- [x] 2.3 Generate mechanism hypotheses (KC/TS/PO/C) for each QA pair via LLM (integrated into qa_generator.py)
- [ ] 2.4 Human validation of ~5K sample (10% quality check)

### Phase 3: Benchmark Infrastructure
- [x] 3.1 Build FreshRAG-Corpus: multi-temporal document store indexed by entity (`retrieval_scenarios.build_corpus`)
- [x] 3.2 Implement 5 retrieval scenarios (S1: fresh only, S2: stale only, S3: mixed fresh-dominant, S4: mixed stale-dominant, S5: mixed equal) (`retrieval_scenarios.py`)
- [ ] 3.3 Implement timestamp manipulation variants (with timestamps, without, misleading)
- [x] 3.4 Build evaluation harness with metrics (FA, HR, AU, CRS, TGS, POS) (`eval_harness.py`)

### Phase 4: Experiments
- [ ] 4.1 Experiment 1 — Mechanism Isolation: full factorial S1-S5 × timestamp conditions, ANOVA
- [ ] 4.2 Experiment 2 — Temporal Gradient: vary document age T-4→T+3, regression analysis
- [ ] 4.3 Experiment 3 — Domain Heterogeneity: stratified by domain, mixed-effects models
- [ ] 4.4 Experiment 4 — Model Scaling: compare across model sizes within families
- [ ] 4.5 Experiment 5 — Adversarial Freshness: old content with fake recent timestamps

### Phase 5: Analysis & Writing
- [ ] 5.1 Causal effect decomposition (TE = DE + IE via KC + IE via TS + IE via PO)
- [ ] 5.2 Validate mechanism hypotheses against experimental results
- [ ] 5.3 Derive practical guidelines for content refresh strategies
- [ ] 5.4 Write paper

### Phase 6: Release
- [ ] 6.1 Open-source benchmark, documentation, community feedback
