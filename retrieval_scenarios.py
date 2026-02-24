"""
FreshRAG Corpus & Retrieval Scenarios (Phase 3.1 + 3.2 + 3.3).

Builds a corpus index from QA pairs' temporal_gradient fields, implements
5 retrieval scenarios with 3 timestamp conditions, and generates eval instances.

Usage:
    python retrieval_scenarios.py                              # data/benchmark/qa_pairs_210.jsonl → data/benchmark/eval_instances_210.jsonl
    python retrieval_scenarios.py --scenarios S1,S3             # subset of scenarios
    python retrieval_scenarios.py --ts-conditions actual,none   # subset of timestamp conditions
    python retrieval_scenarios.py --max-doc-chars 2000          # truncate doc content (default)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_DATA_ROOT = _SCRIPT_DIR / "data" / "wikipedia"

_ALL_SCENARIOS = ("S1", "S2", "S3", "S4", "S5")
_ALL_TS_CONDITIONS = ("actual", "none", "misleading")
_T_LABEL_RE = re.compile(r"^T([+-]\d+)$")


def _strip_wiki_markup(text: str) -> str:
    """Remove common wiki markup (mirrors diff_classifier._strip_wiki_markup)."""
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"{{[^}]*}}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _normalize_entity_key(name: str) -> str:
    """Normalize an entity directory name to a key (mirrors diff_classifier logic)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ---------------------------------------------------------------------------
# 2. Corpus index  (Phase 3.1)
# ---------------------------------------------------------------------------

def build_corpus(qa_path: Path, data_root: Path = _DATA_ROOT) -> dict[str, Path]:
    """
    Build a runtime index mapping doc_id → snapshot file path.

    Scans data_root for entity directories, then resolves every doc_id
    referenced in the QA file's temporal_gradient fields.
    """
    # Build entity_key → (category, entity_dir_name) lookup
    key_to_path: dict[str, Path] = {}
    for cat_dir in sorted(data_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        for entity_dir in sorted(cat_dir.iterdir()):
            if not entity_dir.is_dir():
                continue
            key = _normalize_entity_key(entity_dir.name)
            key_to_path[key] = entity_dir

    # Collect all doc_ids from QA records
    corpus: dict[str, Path] = {}
    with open(qa_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tg = rec.get("temporal_gradient", {})
            for _t_label, info in tg.items():
                doc_id = info.get("doc_id")
                if not doc_id or doc_id in corpus:
                    continue
                # Parse doc_id: doc_{entity_key}_{date}
                # Date is always YYYY-MM-DD at the end
                m = re.match(r"^doc_(.+)_(\d{4}-\d{2}-\d{2})$", doc_id)
                if not m:
                    continue
                entity_key, date_str = m.group(1), m.group(2)
                entity_dir = key_to_path.get(entity_key)
                if entity_dir is None:
                    continue
                snapshot_path = entity_dir / f"{date_str}.json"
                if snapshot_path.exists():
                    corpus[doc_id] = snapshot_path

    return corpus


def extract_passage(
    snapshot_path: Path,
    answer_hint: str,
    max_chars: int = 2000,
) -> str:
    """
    Extract a relevant passage from a Wikipedia snapshot JSON file.

    Looks for the paragraph containing answer_hint (case-insensitive),
    returns that paragraph ± 1 surrounding paragraph. Falls back to the
    first max_chars of stripped content.
    """
    with open(snapshot_path, encoding="utf-8") as fh:
        data = json.load(fh)
    raw_content = data.get("content", "")
    stripped = _strip_wiki_markup(raw_content)

    # Split into paragraphs (double newline or section boundaries)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", stripped) if p.strip()]

    if not paragraphs:
        return stripped[:max_chars]

    # Try to find the paragraph containing the answer hint
    if answer_hint and answer_hint.lower() not in ("information not available",
                                                     "information no longer documented"):
        hint_lower = answer_hint.lower()
        # Try progressively shorter hint fragments
        for frag_len in (len(hint_lower), 60, 30):
            hint_frag = hint_lower[:frag_len]
            for idx, para in enumerate(paragraphs):
                if hint_frag in para.lower():
                    start = max(0, idx - 1)
                    end = min(len(paragraphs), idx + 2)
                    passage = "\n\n".join(paragraphs[start:end])
                    return passage[:max_chars]

    # Fallback: first max_chars
    return "\n\n".join(paragraphs)[:max_chars]


# ---------------------------------------------------------------------------
# 3. Retrieval scenarios  (Phase 3.2)
# ---------------------------------------------------------------------------

def select_docs(qa_record: dict, scenario: str) -> list[dict]:
    """
    Select and rank documents for a QA pair under the given scenario.

    Returns list of {doc_id, date, fact_state, rank} sorted by rank (1 = top).
    """
    tg = qa_record.get("temporal_gradient", {})

    # Separate docs by fact_state
    fresh_docs = []  # fact_state == "new"
    stale_docs = []  # fact_state == "old"

    for t_label, info in sorted(tg.items()):
        if "doc_id" not in info:
            continue
        entry = {
            "doc_id": info["doc_id"],
            "date": info["date"],
            "fact_state": info["fact_state"],
            "t_label": t_label,
        }
        if info["fact_state"] == "new":
            fresh_docs.append(entry)
        elif info["fact_state"] == "old":
            stale_docs.append(entry)

    # Sort by t_label proximity to T0: T+1 before T+2, T-1 before T-2
    fresh_docs.sort(key=lambda d: d["t_label"])   # T+1, T+2, T+3...
    stale_docs.sort(key=lambda d: d["t_label"], reverse=True)  # T-1, T-2, T-3...

    selected: list[dict] = []

    if scenario == "S1":
        # Fresh only — all docs with fact_state="new"
        selected = fresh_docs

    elif scenario == "S2":
        # Stale only — all docs with fact_state="old"
        selected = stale_docs

    elif scenario == "S3":
        # Mixed fresh-dominant: up to 2 fresh + 1 stale
        selected = fresh_docs[:2] + stale_docs[:1]
        # Rank: fresh first, then stale
        selected = fresh_docs[:2] + stale_docs[:1]

    elif scenario == "S4":
        # Mixed stale-dominant: up to 1 fresh + 2 stale
        # Rank: stale first, then fresh
        selected = stale_docs[:2] + fresh_docs[:1]

    elif scenario == "S5":
        # Mixed equal: exactly T-1 (stale) + T+1 (fresh)
        t_minus_1 = [d for d in stale_docs if d["t_label"] == "T-1"]
        t_plus_1 = [d for d in fresh_docs if d["t_label"] == "T+1"]
        # Alternating: stale first, then fresh
        selected = t_minus_1[:1] + t_plus_1[:1]

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Assign ranks
    result = []
    for rank, doc in enumerate(selected, 1):
        result.append({
            "doc_id": doc["doc_id"],
            "date": doc["date"],
            "fact_state": doc["fact_state"],
            "rank": rank,
        })
    return result


# ---------------------------------------------------------------------------
# 4. Timestamp manipulation  (Phase 3.3)
# ---------------------------------------------------------------------------

def apply_timestamp_condition(
    retrieved_docs: list[dict],
    condition: str,
) -> list[dict]:
    """
    Apply a timestamp condition to retrieved docs.

    Conditions:
      - "actual":     keep real dates (no change)
      - "none":       remove dates (set to None)
      - "misleading": swap dates between fresh and stale docs — fresh docs
                      get the oldest stale date, stale docs get the newest
                      fresh date
    """
    if condition == "actual":
        return retrieved_docs

    docs = [dict(d) for d in retrieved_docs]  # shallow copy

    if condition == "none":
        for d in docs:
            d["display_date"] = None
        return docs

    if condition == "misleading":
        fresh_dates = sorted(d["date"] for d in docs if d["fact_state"] == "new")
        stale_dates = sorted(d["date"] for d in docs if d["fact_state"] == "old")
        # Fresh docs get the oldest stale date; stale docs get the newest fresh date
        fake_old = stale_dates[0] if stale_dates else "2023-01-01"
        fake_new = fresh_dates[-1] if fresh_dates else "2025-01-01"
        for d in docs:
            if d["fact_state"] == "new":
                d["display_date"] = fake_old
            else:
                d["display_date"] = fake_new
        return docs

    raise ValueError(f"Unknown timestamp condition: {condition}")


# ---------------------------------------------------------------------------
# 5. Eval instance generation
# ---------------------------------------------------------------------------

def generate_eval_instances(
    qa_path: Path,
    data_root: Path = _DATA_ROOT,
    scenarios: tuple[str, ...] = _ALL_SCENARIOS,
    ts_conditions: tuple[str, ...] = _ALL_TS_CONDITIONS,
    output_path: Path | None = None,
    max_doc_chars: int = 2000,
) -> int:
    """
    Generate eval instances for each QA pair × scenario × timestamp condition.
    Writes JSONL to output_path. Returns number of instances written.
    """
    if output_path is None:
        stem = qa_path.stem
        output_path = qa_path.parent / f"eval_instances_{stem.replace('qa_pairs_', '')}.jsonl"

    print(f"Building corpus index from {qa_path}...")
    corpus = build_corpus(qa_path, data_root)
    print(f"  Indexed {len(corpus)} documents")

    # Load QA records
    records: list[dict] = []
    with open(qa_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records)} QA pairs")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print(f"  Timestamp conditions: {', '.join(ts_conditions)}")

    # Short labels for instance IDs
    _ts_label = {"actual": "Ta", "none": "Tn", "misleading": "Tm"}

    written = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            qa_id = rec["qa_id"]
            for scenario in scenarios:
                docs = select_docs(rec, scenario)
                if not docs:
                    skipped += len(ts_conditions)
                    continue

                # Resolve doc content (shared across timestamp conditions)
                retrieved_docs = []
                for doc in docs:
                    doc_id = doc["doc_id"]
                    snapshot_path = corpus.get(doc_id)
                    if snapshot_path is None:
                        continue
                    # Use new_answer as hint for fresh docs, old_answer for stale
                    hint = (rec["new_answer"] if doc["fact_state"] == "new"
                            else rec["old_answer"])
                    content = extract_passage(snapshot_path, hint, max_doc_chars)
                    retrieved_docs.append({
                        "doc_id": doc_id,
                        "content": content,
                        "date": doc["date"],
                        "fact_state": doc["fact_state"],
                        "rank": doc["rank"],
                    })

                if not retrieved_docs:
                    skipped += len(ts_conditions)
                    continue

                for ts_cond in ts_conditions:
                    ts_docs = apply_timestamp_condition(retrieved_docs, ts_cond)
                    ts_tag = _ts_label[ts_cond]
                    instance = {
                        "instance_id": f"{qa_id}_{scenario}_{ts_tag}",
                        "qa_id": qa_id,
                        "scenario": scenario,
                        "timestamp_condition": ts_cond,
                        "question": rec["question"],
                        "new_answer": rec["new_answer"],
                        "old_answer": rec["old_answer"],
                        "retrieved_docs": ts_docs,
                        "domain": rec["domain"],
                        "change_type": rec["change_type"],
                        "predicted_mechanism": rec["predicted_mechanism"],
                    }
                    fh.write(json.dumps(instance, ensure_ascii=False) + "\n")
                    written += 1

    print(f"\nDone. Wrote {written} eval instances to {output_path}")
    if skipped:
        print(f"  ({skipped} scenario/ts combos skipped — no matching docs)")
    return written


# ---------------------------------------------------------------------------
# 6. Temporal gradient instance generation  (Experiment 4.2)
# ---------------------------------------------------------------------------

def generate_temporal_gradient_instances(
    qa_path: Path,
    data_root: Path = _DATA_ROOT,
    output_path: Path | None = None,
    max_doc_chars: int = 2000,
) -> int:
    """
    Generate single-doc eval instances for each QA pair × temporal position.

    Each instance contains exactly one document at a specific temporal distance
    from the change event (T-4..T+4, excluding T0). Used for Experiment 4.2
    to measure how FA varies with document age.

    Instance IDs: {qa_id}_TG_{t_label}  (e.g. freshrag_00011_1_TG_T-1)
    """
    if output_path is None:
        stem = qa_path.stem.replace("qa_pairs_", "")
        output_path = qa_path.parent / f"eval_instances_tg_{stem}.jsonl"

    print(f"Building corpus index from {qa_path}...")
    corpus = build_corpus(qa_path, data_root)
    print(f"  Indexed {len(corpus)} documents")

    records: list[dict] = []
    with open(qa_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records)} QA pairs")

    written = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            qa_id = rec["qa_id"]
            tg = rec.get("temporal_gradient", {})

            for t_label, info in sorted(tg.items()):
                m = _T_LABEL_RE.match(t_label)
                if not m:
                    continue  # skip T0 or invalid labels
                t_offset = int(m.group(1))

                doc_id = info.get("doc_id")
                if not doc_id:
                    skipped += 1
                    continue

                snapshot_path = corpus.get(doc_id)
                if not snapshot_path:
                    skipped += 1
                    continue

                hint = (rec["new_answer"] if info["fact_state"] == "new"
                        else rec["old_answer"])
                content = extract_passage(snapshot_path, hint, max_doc_chars)

                instance = {
                    "instance_id": f"{qa_id}_TG_{t_label}",
                    "qa_id": qa_id,
                    "scenario": "TG",
                    "timestamp_condition": "actual",
                    "t_label": t_label,
                    "t_offset": t_offset,
                    "question": rec["question"],
                    "new_answer": rec["new_answer"],
                    "old_answer": rec["old_answer"],
                    "retrieved_docs": [{
                        "doc_id": doc_id,
                        "content": content,
                        "date": info["date"],
                        "fact_state": info["fact_state"],
                        "rank": 1,
                    }],
                    "domain": rec["domain"],
                    "change_type": rec["change_type"],
                    "predicted_mechanism": rec["predicted_mechanism"],
                }
                fh.write(json.dumps(instance, ensure_ascii=False) + "\n")
                written += 1

    print(f"\nDone. Wrote {written} temporal gradient instances to {output_path}")
    if skipped:
        print(f"  ({skipped} positions skipped — no doc or not resolved)")
    return written


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate FreshRAG eval instances with retrieval scenarios",
    )
    parser.add_argument(
        "--input", "-i",
        default="data/benchmark/qa_pairs_210.jsonl",
        help="Input JSONL file with QA pairs (default: data/benchmark/qa_pairs_210.jsonl)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSONL file (default: eval_instances_{stem}.jsonl)",
    )
    parser.add_argument(
        "--scenarios", "-s",
        default=None,
        help="Comma-separated scenario list, e.g. S1,S3 (default: all)",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=2000,
        help="Max characters per retrieved document passage (default: 2000)",
    )
    parser.add_argument(
        "--ts-conditions", "-t",
        default=None,
        help="Comma-separated timestamp conditions: actual,none,misleading (default: all)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to data/wikipedia/ directory (default: auto-detect)",
    )
    parser.add_argument(
        "--temporal-gradient",
        action="store_true",
        help="Generate single-doc temporal gradient instances (Experiment 4.2)",
    )

    args = parser.parse_args()

    qa_path = Path(args.input)
    if not qa_path.exists():
        sys.exit(f"Error: input file not found: {qa_path}")

    output_path = Path(args.output) if args.output else None
    data_root = Path(args.data_root) if args.data_root else _DATA_ROOT

    if args.temporal_gradient:
        generate_temporal_gradient_instances(
            qa_path=qa_path,
            data_root=data_root,
            output_path=output_path,
            max_doc_chars=args.max_doc_chars,
        )
        return

    scenarios = _ALL_SCENARIOS
    if args.scenarios:
        scenarios = tuple(s.strip() for s in args.scenarios.split(","))
        for s in scenarios:
            if s not in _ALL_SCENARIOS:
                sys.exit(f"Error: unknown scenario '{s}'. Must be one of {_ALL_SCENARIOS}")

    ts_conditions = _ALL_TS_CONDITIONS
    if args.ts_conditions:
        ts_conditions = tuple(t.strip() for t in args.ts_conditions.split(","))
        for t in ts_conditions:
            if t not in _ALL_TS_CONDITIONS:
                sys.exit(f"Error: unknown timestamp condition '{t}'. "
                         f"Must be one of {_ALL_TS_CONDITIONS}")

    generate_eval_instances(
        qa_path=qa_path,
        data_root=data_root,
        scenarios=scenarios,
        ts_conditions=ts_conditions,
        output_path=output_path,
        max_doc_chars=args.max_doc_chars,
    )


if __name__ == "__main__":
    main()
