# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FRESH-bench** is a research benchmark for evaluating AI/LLM systems on temporally-changing factual information. It collects Wikipedia article snapshots at multiple timestamps (2023-2025) across 598 entities in 6 categories, enabling evaluation of how well models handle knowledge that changes over time. The current working dataset uses 210 sampled entities (30 per group × 7 groups).

## Commands

```bash
# Install dependencies
pip install requests jupyter openai

# Data collection notebook
jupyter notebook wikidata_fetch.ipynb

# Diff classifier
python diff_classifier.py                                              # demo on Andrej Karpathy
python diff_classifier.py data/wikipedia/people/Elon_Musk              # compare consecutive snapshots
python diff_classifier.py data/wikipedia/people/Elon_Musk/2023-01-01.json data/wikipedia/people/Elon_Musk/2024-01-01.json
python diff_classifier.py --batch-all                                  # → changes_full.jsonl

# Scale pipeline — 210-entity subset (30 per group × 7 groups)
python scale_pipeline.py                                               # → data/benchmark/changes_210.jsonl
python scale_pipeline.py --with-qa                                     # → + data/benchmark/qa_pairs_210.jsonl

# QA generator
python qa_generator.py                                                 # data/benchmark/changes_210.jsonl → qa_pairs_210.jsonl
python qa_generator.py --input data/benchmark/changes_210.jsonl --limit 5
python qa_generator.py --run-diff-first                                # batch diff + QA generation

# Retrieval scenarios — generate eval instances (scenario × timestamp condition)
python retrieval_scenarios.py                                          # → data/benchmark/eval_instances_210.jsonl
python retrieval_scenarios.py --scenarios S1,S3                        # subset of scenarios
python retrieval_scenarios.py --ts-conditions actual,none              # subset of timestamp conditions
python retrieval_scenarios.py --max-doc-chars 2000                     # truncate doc content (default)

# Eval harness — score predictions with 6 metrics
python eval_harness.py --instances data/benchmark/eval_instances_210.jsonl --predictions data/benchmark/predictions_210.jsonl
python eval_harness.py --instances data/benchmark/eval_instances_210.jsonl --run-predictions
python eval_harness.py --instances data/benchmark/eval_instances_210.jsonl --run-predictions --predictions-only
python eval_harness.py --instances data/benchmark/eval_instances_210.jsonl --run-predictions --model gpt-4o --limit 10

# Experiment analysis — statistical tests on scored instances
python experiment_analysis.py --experiment 4.1 --scored data/benchmark/scored_210.jsonl
```

The `.env` file contains `OPENAI_API_KEY` — required by `qa_generator.py` and `eval_harness.py --run-predictions`.

## Architecture

### Pipeline Data Flow

```
wikidata_fetch.ipynb      → data/wikipedia/{category}/{entity}/{date}.json   (raw snapshots)
        ↓
diff_classifier.py        → data/benchmark/changes_210.jsonl                 (293 change records)
        ↓
qa_generator.py           → data/benchmark/qa_pairs_210.jsonl                (641 QA pairs with mechanism labels)
        ↓
retrieval_scenarios.py    → data/benchmark/eval_instances_210.jsonl           (9,615 instances: 641 QA × 5 scenarios × 3 ts conditions)
        ↓
eval_harness.py           → data/benchmark/predictions_210.jsonl             (model answers)
                          → data/benchmark/scored_210.jsonl                  (per-instance metric scores)
                          → data/benchmark/report_210.json                   (aggregated summary)
        ↓
experiment_analysis.py    → data/benchmark/experiment_4_1.json               (ANOVA results)
```

`scale_pipeline.py` orchestrates the first two steps for the 210-entity subset (deterministic sampling with seed 42, splitting science into AI/Space and Life/Physical subgroups).

### `wikidata_fetch.ipynb` — Data Collection
Single Jupyter notebook (Cell 0 is ~75K chars with entity definitions and collection logic). Key function: `get_article_at_timestamp(title, timestamp)` fetches Wikipedia content at a specific timestamp via the API.

Entity definitions have fields: `name`, `wikipedia_title`, `fame_level` (`"high"/"medium"/"low"`), and `change_type`.

### `diff_classifier.py` — Change Detection & Classification
Detects and classifies factual changes between Wikipedia snapshots:

1. **Myers diff** (`myers_diff`) — line-level shortest edit script
2. **Change classification** (`classify_change`) — categorizes as `FACTUAL_UPDATE`, `NUMERIC_UPDATE`, `ADDITION`, or `DELETION`
3. **Diff context extraction** (`extract_diff_context`) — selects most significant changed block, preferring infobox over prose
4. **Semantic subtyping** (`infer_subtype`) — labels like `leadership_change`, `numeric_stat`, `status_change` via regex
5. **Temporal gradient** (`build_temporal_gradient`) — maps snapshots to T-N…T+M labels relative to change event
6. **Batch processing** (`run_batch`) — writes JSONL change records

Distinguishes factual changes from cosmetic edits via `_strip_wiki_markup`, `_numbers_differ`, and `_same_infobox_key`. Entity key normalization: `re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")`.

### `qa_generator.py` — QA Pair Generation
Calls OpenAI (gpt-4o-mini) to produce 3 QA pairs per change record with predicted failure mechanisms (KC/TS/PO/C). Has resume support — skips already-processed change_ids.

### `retrieval_scenarios.py` — Corpus Index & Retrieval Scenarios
Builds a runtime `doc_id → file path` index (no data duplication), extracts relevant passages using answer hints, and generates eval instances under 5 scenarios × 3 timestamp conditions:

| Scenario | Description | Selection |
|----------|-------------|-----------|
| S1 | Fresh only | All docs with fact_state="new" |
| S2 | Stale only | All docs with fact_state="old" |
| S3 | Mixed fresh-dominant | Up to 2 fresh + 1 stale |
| S4 | Mixed stale-dominant | Up to 1 fresh + 2 stale |
| S5 | Mixed equal | Exactly T-1 (stale) + T+1 (fresh) |

Timestamp conditions: `actual` (real dates shown), `none` (no dates), `misleading` (fresh/stale dates swapped).

### `eval_harness.py` — Evaluation Harness
Scores model predictions using 6 metrics:

| Metric | Description |
|--------|-------------|
| FA | Factual Accuracy — token F1 vs new_answer |
| HR | Hallucination Rate — matches neither answer (both F1 < 0.3) |
| AU | Answer Uncertainty — hedging language detected |
| CRS | Change Recognition Score — acknowledges information changed |
| POS | Preference for Outdated — prefers old_answer over new_answer |
| TGS | Temporal Gradient Score — Pearson correlation of FA vs temporal distance |

Aggregates results by scenario, domain, mechanism, and timestamp condition. Optionally generates predictions via OpenAI.

### `experiment_analysis.py` — Statistical Analysis
Runs statistical tests on scored instances. Currently supports:
- **Experiment 4.1**: Two-way ANOVA (scenario × timestamp_condition) for each metric, post-hoc pairwise comparisons (Bonferroni), cell means tables, stratified by mechanism. Uses scipy + pandas.

## Data Layout

```
data/
  wikipedia/                  # Raw Wikipedia snapshots (~440 MB)
    {category}/
      {entity_name}/
        {timestamp}.json      # 2023-01-01, 2023-06-01, 2024-01-01, 2024-06-01, 2025-01-01
  benchmark/                  # Pipeline outputs (JSONL)
    changes_210.jsonl
    qa_pairs_210.jsonl
    eval_instances_210.jsonl
    predictions_210.jsonl     # generated by eval_harness
    scored_210.jsonl          # generated by eval_harness
    report_210.json           # generated by eval_harness
```

**Active categories:** `organizations`, `people`, `policy`, `products`, `science`, `sports`
**Legacy categories** (moved into `science/`): `science_biology_medicine`, `science_physics_chemistry`

Snapshot JSON schema: `{title, page_id, revision_id, revision_timestamp, content}` where `content` is full Wikipedia markup.

## JSONL Record Schemas

**Change record** (`changes_210.jsonl`): `{change_id, entity_id, domain, change_detection: {old_snapshot, new_snapshot, diff_context: {old_text, new_text}}, change_classification: {type, subtype}, temporal_gradient: {T-N..T+M: {date, fact_state, doc_id}}}`

**QA pair** (`qa_pairs_210.jsonl`): `{qa_id, change_id, entity_id, domain, question, old_answer, new_answer, old_snapshot, new_snapshot, change_type, change_subtype, predicted_mechanism, mechanism_rationale, temporal_gradient}`

**Eval instance** (`eval_instances_210.jsonl`): `{instance_id, qa_id, scenario, timestamp_condition, question, new_answer, old_answer, retrieved_docs: [{doc_id, content, date, fact_state, rank, display_date?}], domain, change_type, predicted_mechanism}`

## FreshRAG Experiment Steps

### Phase 1: Data Collection — DONE
- [x] 1.1 Collect Wikipedia snapshots (598 entities × 5 timestamps)
- [x] 1.2 Define entity metadata
- [x] 1.3 Build diff pipeline (`diff_classifier.py`)

### Phase 2: Dataset Construction — DONE
- [x] 2.1 Run batch diff → change records JSONL (210 entities via `scale_pipeline.py`)
- [x] 2.2 Generate QA pairs from changes (`qa_generator.py`, 641 QA pairs)
- [x] 2.3 Generate mechanism hypotheses (KC/TS/PO/C) per QA pair
- [ ] 2.4 Human validation of ~5K sample (10% quality check)

### Phase 3: Benchmark Infrastructure
- [x] 3.1 Build FreshRAG-Corpus (`retrieval_scenarios.build_corpus`)
- [x] 3.2 Implement 5 retrieval scenarios (`retrieval_scenarios.py`)
- [x] 3.3 Implement timestamp manipulation variants: actual, none, misleading (`retrieval_scenarios.apply_timestamp_condition`)
- [x] 3.4 Build evaluation harness with metrics (`eval_harness.py`)

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
