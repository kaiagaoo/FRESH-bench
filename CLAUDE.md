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

# Retrieval scenarios — temporal gradient instances (Experiment 4.2)
python retrieval_scenarios.py --temporal-gradient                     # → data/benchmark/eval_instances_tg_210.jsonl

# Experiment analysis — statistical tests on scored instances
python experiment_analysis.py --experiment 4.1 --scored data/benchmark/scored_210.jsonl
python experiment_analysis.py --experiment 4.2 --scored data/benchmark/scored_tg_210.jsonl
python experiment_analysis.py --experiment 4.3 --scored data/benchmark/scored_210.jsonl
python experiment_analysis.py --experiment 5.1 --scored data/benchmark/scored_210.jsonl
python experiment_analysis.py --experiment 5.2 --scored data/benchmark/scored_210.jsonl
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
experiment_analysis.py    → data/benchmark/experiment_4_*.json               (statistical results)
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
Runs statistical tests on scored instances. Supports experiments 4.1–4.3, 4.5, 5.1–5.3. Uses scipy + pandas.

```bash
python experiment_analysis.py --experiment 4.1 --scored data/benchmark/scored_210.jsonl
python experiment_analysis.py --experiment 4.2 --scored data/benchmark/scored_tg_210.jsonl
python experiment_analysis.py --experiment 4.3 --scored data/benchmark/scored_210.jsonl
python experiment_analysis.py --experiment 4.5 --scored data/benchmark/scored_210.jsonl
python experiment_analysis.py --experiment 5.1 --scored data/benchmark/scored_210.jsonl
python experiment_analysis.py --experiment 5.2 --scored data/benchmark/scored_210.jsonl
python experiment_analysis.py --experiment 5.3 --scored data/benchmark/scored_210.jsonl --scored-tg data/benchmark/scored_tg_210.jsonl
```

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
    eval_instances_tg_210.jsonl   # temporal gradient instances (3,137)
    predictions_210.jsonl         # gpt-4o-mini predictions (9,615)
    predictions_tg_210.jsonl      # gpt-4o-mini TG predictions (3,137)
    scored_210.jsonl              # scored standard instances
    scored_tg_210.jsonl           # scored TG instances
    report_210.json               # aggregated summary
    experiment_4_1.json           # Experiment 4.1 results
    experiment_4_2.json           # Experiment 4.2 results
    experiment_4_3.json           # Experiment 4.3 results
    experiment_5_1.json           # Experiment 5.1 results (causal decomposition)
    experiment_5_2.json           # Experiment 5.2 results (mechanism validation)
    experiment_4_5.json           # Experiment 4.5 results (adversarial freshness)
    experiment_5_3.json           # Experiment 5.3 results (practical guidelines)
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
- [x] 4.1 Experiment 1 — Mechanism Isolation: full factorial S1-S5 × timestamp conditions, ANOVA
- [x] 4.2 Experiment 2 — Temporal Gradient: vary document age T-4→T+4, regression analysis
- [x] 4.3 Experiment 3 — Domain Heterogeneity: stratified by domain, ANOVA + Kruskal-Wallis
- [x] 4.4 Experiment 4 — Adversarial Freshness: old content with fake recent timestamps

## Experiment Results (gpt-4o-mini, 210 entities)

### Experiment 4.1 — Mechanism Isolation (n=9,615)
Two-way ANOVA: scenario × timestamp_condition. Results in `experiment_4_1.json`.

**Key findings:**
- **Scenario (retrieval composition) is the dominant factor.** All 5 metrics show highly significant scenario effects (p<0.001): FA (F=50.2, η²=0.021), HR (F=22.3), AU (F=59.6), CRS (F=7.9), POS (F=27.8).
- **Timestamp condition has NO effect.** None of the 5 metrics show significant timestamp_condition effects (all p>0.27). Showing actual dates, no dates, or misleading dates makes no difference. No scenario×timestamp interaction either.
- **S2 (stale-only) is the worst scenario; all others are statistically equivalent.** Post-hoc (Bonferroni): S2 vs every other scenario is significant (p<0.001) for FA. S1/S3/S4/S5 are not significantly different from each other.
- **Scenario means (FA):** S1=0.433, S2=0.334, S3=0.428, S4=0.431, S5=0.432. Having even one fresh document is sufficient — mixing ratios don't matter.

### Experiment 4.2 — Temporal Gradient (n=3,137)
OLS regression: FA vs temporal offset (T-4 to T+4, single-doc retrieval). Results in `experiment_4_2.json`.

**Key findings:**
- **Document freshness significantly predicts FA.** Overall regression: slope=+0.017/step, r=0.15, p<0.0001. One-way ANOVA across 8 positions: F(7)=16.9, p<0.0001.
- **Discrete jump at the change boundary, not a smooth gradient.** Stale side (T-4 to T-1): flat, slope≈0, p=0.49. Fresh side (T+1 to T+4): slight negative slope=-0.018, p=0.008. The effect is primarily a step function: stale FA=0.331 vs fresh FA=0.422 (diff=+0.091, t=10.3, p<0.0001).
- **Effect is consistent across all domains and mechanisms.** All 7 domains show significant positive slopes (p<0.05). TS and PO mechanisms show the strongest effects (p<0.001).
- **All metrics respond to freshness.** HR, AU, CRS, POS all show significant slopes (p<0.001) in expected directions (fresher docs → lower HR, lower AU, lower CRS, lower POS).

### Experiment 4.3 — Domain Heterogeneity (n=9,615)
One-way ANOVA + Kruskal-Wallis per metric across 7 domains. Results in `experiment_4_3.json`.

**Key findings:**
- **All metrics show significant domain effects** (all p<0.001). Largest effect: AU (F=71.2, η²=0.043).
- **Domain FA ranking:** products (0.464) > science_ai_space (0.437) > organizations (0.427) > policy (0.420) > people (0.400) > science_life_physical (0.368) > sports (0.360).
- **Highest hallucination rates:** science_life_physical (HR=0.379), people (0.309); lowest: organizations (0.239).
- **Highest uncertainty:** organizations (AU=0.452), sports (0.430); lowest: science_ai_space (0.183).
- **Domain × scenario interaction is NOT significant** (p=0.99 for FA), meaning the scenario effect is consistent across domains.
- **Sports has highest outdated-preference** (POS=0.235), suggesting models are most likely to prefer stale sports facts.

### Experiment 5.1 — Causal Effect Decomposition (n=9,615)
Decomposes TE (fresh-vs-stale FA gap) into mechanism-mediated indirect effects. Results in `experiment_5_1.json`.

**Key findings:**
- **Total Effect: TE=+0.099** (S1 FA=0.433 vs S2 FA=0.334). All domains significant (p<0.001).
- **TS dominates the effect** (68.6% of TE): TS mechanism instances show the largest fresh-stale gap (TE_m=+0.115, w=0.59). PO contributes 23.5% (TE_m=+0.101, w=0.23). KC contributes 3.7%. C contributes 4.2% (not significant, p=0.20).
- **Mediation analysis** (Baron & Kenny, Sobel test): 20.2% of TE is mediated through behavioural metrics:
  - AU mediates 23.1% (stale→higher uncertainty→lower FA, Sobel z=−7.9, p<0.001)
  - POS mediates 10.3% (stale→prefer outdated→lower FA, Sobel z=−6.1, p<0.001)
  - CRS shows a suppression effect (+13.2%): stale triggers change recognition which slightly *helps* FA
  - 79.8% of the effect is direct (not captured by these mediators)
- **Mixed scenarios recover nearly all of the TE:** S3 recovers 94.3%, S4 recovers 98.1%, S5 recovers 98.7% — even one fresh document largely neutralizes staleness.

### Experiment 5.2 — Mechanism Hypothesis Validation (n=9,615)
Tests whether predicted mechanism labels (KC/TS/PO/C) match observed metric signatures. Results in `experiment_5_2.json`.

**Key findings:**
- **Overall: 5/9 (56%) hypotheses confirmed.** TS best validated (2/2), PO partially (2/3), C partially (1/2), KC not confirmed (0/2).
- **TS validation (2/2):** TS instances show significantly higher AU (+0.087, p<0.001) and a larger S1-S2 FA gap than non-TS instances, confirming temporal sensitivity.
- **PO validation (2/3):** PO instances have higher POS in S2 (+0.057, p=0.018) and overall (+0.051, p<0.001). However, POS does not differ between S4 (stale-dominant) and S3 (fresh-dominant) within PO instances (p=0.80).
- **KC not validated (0/2):** KC instances do NOT show higher CRS in mixed scenarios or higher HR in S2 vs other mechanisms. KC labels may not reliably capture knowledge conflict.
- **C partially validated (1/2):** C instances have much higher CRS (+0.492, p<0.001) but unexpectedly *higher* FA than others (+0.074, p<0.001), not lower. C may capture "recognizable changes" rather than "hardest" instances.
- **Discriminant validity:** Mechanism labels significantly predict all metrics (ANOVA p<0.001). Largest effect on CRS (η²=0.14) and POS (η²=0.04). FA discrimination is modest (η²=0.016).

### Experiment 4.4 — Adversarial Freshness (n=9,615)
Tests whether fake timestamps can fool models into trusting stale content. Results in `experiment_4_5.json`.

**Key findings:**
- **Models are completely ROBUST to adversarial timestamps.** 0/5 metrics show significant differences between actual and misleading timestamps (all p>0.21).
- **Within S2 (stale-only), misleading dates have zero effect:** FA actual=0.335 vs misleading=0.335 (p=0.99). No effect on HR, AU, CRS, or POS either (all F<1.1, p>0.35).
- **Robustness holds across all scenarios, mechanisms, and domains.** Not a single significant comparison found in any stratum.
- **Verdict: gpt-4o-mini relies on content semantics, not timestamp metadata**, for temporal reasoning. Adversarial timestamp manipulation is not a viable attack vector.

### Experiment 5.3 — Practical Guidelines (n=9,615 + 3,137 TG)
Synthesizes all experimental findings into actionable content refresh recommendations. Results in `experiment_5_3.json`.

**Five guidelines with quantitative evidence:**
1. **One fresh document neutralizes staleness.** S5 (1 fresh + 1 stale) recovers 98.7% of the TE. Marginal value of additional fresh docs is negligible (S3 FA=0.427 ≈ S5 FA=0.432).
2. **Timestamp metadata has no measurable impact.** Max FA difference across actual/none/misleading is 0.004. Even adversarial timestamps are ineffective (p=0.99).
3. **Prioritize refreshing policy and organizations content.** Highest vulnerability scores (TE×(1+HR)): policy=0.156, organizations=0.151. science_life_physical has worst hallucination rate (HR=0.38).
4. **TS + PO mechanisms drive 92% of staleness failures.** TS contributes 69% (temporal sensitivity), PO contributes 24% (outdated preference). Event-driven refresh for TS-type facts; recency-weighted scoring for PO-type.
5. **Refresh should be event-driven, not calendar-driven.** Freshness is a step function at the change boundary — within-stale range=0.015, within-fresh range=0.053, but cross-boundary step=+0.091. A 6-month-old fresh doc performs the same as a 1-month-old one.

**Decision matrix:** High priority (TS/PO + policy/orgs, +0.099 FA gain) → Medium (any domain with 1 fresh doc, +0.097) → Low (C mechanism/products) → Not needed (timestamp updates).

### Phase 5: Analysis & Writing
- [x] 5.1 Causal effect decomposition (TE = DE + IE via KC + IE via TS + IE via PO)
- [x] 5.2 Validate mechanism hypotheses against experimental results
- [x] 5.3 Derive practical guidelines for content refresh strategies
- [ ] 5.4 Write paper

### Phase 6: Release
- [ ] 6.1 Open-source benchmark, documentation, community feedback
