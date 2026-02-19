"""
QA pair generator for FreshRAG benchmark.

Reads change records from JSONL (produced by diff_classifier.py) and generates
question-answer pairs with mechanism hypotheses using OpenAI.

Usage:
    python qa_generator.py                                    # data/benchmark/changes_210.jsonl → data/benchmark/qa_pairs_210.jsonl
    python qa_generator.py --input data/benchmark/changes_210.jsonl --limit 5 # test run on first 5
    python qa_generator.py --input changes_full.jsonl         # full dataset
    python qa_generator.py --run-diff-first                   # run batch diff then generate QA
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent

_SYSTEM_PROMPT = """\
You are a research assistant generating question-answer pairs for a temporal \
knowledge benchmark. Given a factual change detected between two Wikipedia \
snapshots, generate exactly 3 natural questions that a user might ask, where \
the answer changed between the old and new snapshot.

For each question, also predict the dominant mechanism by which a RAG system \
might fail on this question:
- KC (Knowledge Conflict): the model's parametric knowledge conflicts with retrieved context
- TS (Temporal Sensitivity): the model fails to account for the temporal validity of information
- PO (Preference for Outdated): the model prefers older/more familiar information
- C (Conflation): the model conflates or merges old and new information

Return a JSON array of exactly 3 objects with these fields:
- "question": a natural, self-contained question (no references to "the article" or "the change")
- "old_answer": concise factual answer based on the OLD text (1-2 sentences)
- "new_answer": concise factual answer based on the NEW text (1-2 sentences)
- "predicted_mechanism": one of "KC", "TS", "PO", "C"
- "mechanism_rationale": brief explanation (1 sentence) of why this mechanism applies

Guidelines:
- Questions should be ones a real user would naturally ask (e.g., "Who is the CEO of X?")
- Questions must be answerable from the provided text excerpts
- Each question MUST have a DIFFERENT old_answer vs new_answer — if the answer is the same in both snapshots, do NOT include that question
- Vary the question types (who/what/when/where/how many)
- Vary the predicted mechanisms across the 3 questions when reasonable
- If old_text is empty (new content added), the old_answer should be "Information not available"
- If new_text is empty (content removed), the new_answer should be "Information no longer documented"
- If the old and new text convey the same factual information (e.g., only punctuation, formatting, or minor wording differs), return an EMPTY JSON array: []
- Return ONLY the JSON array, no other text"""


def _build_user_prompt(record: dict) -> str:
    detection = record["change_detection"]
    classification = record["change_classification"]
    return f"""\
Entity: {record["entity_id"]}
Domain: {record["domain"]}
Change type: {classification["type"]} ({classification["subtype"]})
Old snapshot date: {detection["old_snapshot"]}
New snapshot date: {detection["new_snapshot"]}

OLD TEXT:
{detection["diff_context"]["old_text"]}

NEW TEXT:
{detection["diff_context"]["new_text"]}"""


# ---------------------------------------------------------------------------
# 2. OpenAI interaction
# ---------------------------------------------------------------------------

def _init_client() -> OpenAI:
    """Load API key from .env and return OpenAI client."""
    env_path = _SCRIPT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY not found in .env or environment")
    return OpenAI(api_key=api_key)


def generate_qa_pairs(
    client: OpenAI,
    record: dict,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> list[dict]:
    """
    Call OpenAI to generate 3 QA pairs from a single change record.
    Returns a list of dicts with question, old_answer, new_answer,
    predicted_mechanism, and mechanism_rationale.
    """
    user_prompt = _build_user_prompt(record)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)

            # Handle both {"questions": [...]} and direct [...] formats
            if isinstance(parsed, dict):
                for key in ("questions", "qa_pairs", "items", "data"):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
                else:
                    # If dict has no recognizable list key, wrap in list
                    parsed = [parsed]

            if not isinstance(parsed, list):
                raise ValueError(f"Expected list, got {type(parsed).__name__}")

            # Validate each QA pair
            valid = []
            required_keys = {"question", "old_answer", "new_answer",
                             "predicted_mechanism", "mechanism_rationale"}
            valid_mechanisms = {"KC", "TS", "PO", "C"}

            for item in parsed[:3]:
                if not isinstance(item, dict):
                    continue
                if not required_keys.issubset(item.keys()):
                    continue
                # Filter: old_answer must differ from new_answer
                if item["old_answer"].strip() == item["new_answer"].strip():
                    continue
                # Normalize mechanism
                mech = item["predicted_mechanism"].upper().strip()
                if mech not in valid_mechanisms:
                    mech = "KC"  # default fallback
                item["predicted_mechanism"] = mech
                valid.append(item)

            if valid:
                return valid

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            print(f"  Warning: failed to parse response for {record['change_id']}: {exc}",
                  file=sys.stderr)
            return []

        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  Warning: API error for {record['change_id']}: {exc}",
                  file=sys.stderr)
            return []

    return []


# ---------------------------------------------------------------------------
# 3. Batch processing
# ---------------------------------------------------------------------------

def _load_processed_ids(output_path: Path) -> set[str]:
    """Read already-processed change IDs from the output file for resume support."""
    ids: set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        ids.add(rec.get("change_id", ""))
                    except json.JSONDecodeError:
                        continue
    return ids


def process_changes(
    input_path: Path,
    output_path: Path,
    limit: int | None = None,
    model: str = "gpt-4o-mini",
) -> int:
    """
    Read change records from *input_path*, generate QA pairs, write to *output_path*.
    Supports resuming: skips change_ids already in the output file.
    Returns the number of QA pairs written.
    """
    # Load input
    records: list[dict] = []
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit is not None:
        records = records[:limit]

    # Check for already-processed records
    processed = _load_processed_ids(output_path)
    remaining = [r for r in records if r["change_id"] not in processed]

    if not remaining:
        print(f"All {len(records)} records already processed. Nothing to do.")
        return 0

    print(f"Processing {len(remaining)} change records "
          f"({len(processed)} already done, {len(records)} total)")

    client = _init_client()
    written = 0
    qa_counter = len(processed) * 3  # approximate starting counter

    with open(output_path, "a", encoding="utf-8") as fh:
        for i, record in enumerate(remaining, 1):
            change_id = record["change_id"]
            print(f"  [{i}/{len(remaining)}] {change_id} ({record['entity_id']})...",
                  end=" ", flush=True)

            qa_pairs = generate_qa_pairs(client, record, model=model)

            if not qa_pairs:
                print("no QA pairs generated")
                continue

            for j, qa in enumerate(qa_pairs):
                qa_counter += 1
                # Use the derived_qa_ids from the change record if available
                derived_ids = record.get("derived_qa_ids", [])
                if j < len(derived_ids):
                    qa_id = derived_ids[j]
                else:
                    qa_id = f"freshrag_{qa_counter:05d}_{j + 1}"

                output_record = {
                    "qa_id": qa_id,
                    "change_id": change_id,
                    "entity_id": record["entity_id"],
                    "domain": record["domain"],
                    "question": qa["question"],
                    "old_answer": qa["old_answer"],
                    "new_answer": qa["new_answer"],
                    "old_snapshot": record["change_detection"]["old_snapshot"],
                    "new_snapshot": record["change_detection"]["new_snapshot"],
                    "change_type": record["change_classification"]["type"],
                    "change_subtype": record["change_classification"]["subtype"],
                    "predicted_mechanism": qa["predicted_mechanism"],
                    "mechanism_rationale": qa["mechanism_rationale"],
                    "temporal_gradient": record.get("temporal_gradient", {}),
                }
                fh.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                written += 1

            fh.flush()
            print(f"{len(qa_pairs)} QA pairs")

            # Rate limiting: ~0.5s between calls
            if i < len(remaining):
                time.sleep(0.5)

    print(f"\nDone. Wrote {written} QA pairs to {output_path}")
    return written


# ---------------------------------------------------------------------------
# 4. Full dataset batch diff helper
# ---------------------------------------------------------------------------

def run_diff_first(output_path: Path) -> Path:
    """
    Run diff_classifier.run_batch() on all entities under data/wikipedia/,
    writing to changes_full.jsonl. Returns the path to the changes file.
    """
    from diff_classifier import run_batch

    data_root = _SCRIPT_DIR / "data" / "wikipedia"
    categories = [
        "organizations", "people", "policy", "products",
        "science", "science_biology_medicine", "science_physics_chemistry", "sports",
    ]

    entity_dirs: list[tuple[str, Path]] = []
    for cat in categories:
        cat_dir = data_root / cat
        if not cat_dir.is_dir():
            continue
        for entity_dir in sorted(cat_dir.iterdir()):
            if entity_dir.is_dir():
                entity_dirs.append((cat, entity_dir))

    changes_path = output_path.parent / "changes_full.jsonl"
    print(f"Running batch diff on {len(entity_dirs)} entities...")
    written = run_batch(entity_dirs, changes_path)
    print(f"Wrote {written} change records to {changes_path}")
    return changes_path


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from Wikipedia change records",
    )
    parser.add_argument(
        "--input", "-i",
        default="data/benchmark/changes_210.jsonl",
        help="Input JSONL file with change records (default: data/benchmark/changes_210.jsonl)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/benchmark/qa_pairs_210.jsonl",
        help="Output JSONL file for QA pairs (default: data/benchmark/qa_pairs_210.jsonl)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Process only the first N change records",
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--run-diff-first",
        action="store_true",
        help="Run batch diff on all entities before generating QA pairs",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.run_diff_first:
        input_path = run_diff_first(output_path)
        if args.output == "data/benchmark/qa_pairs_210.jsonl":
            output_path = Path("qa_pairs_full.jsonl")

    if not input_path.exists():
        sys.exit(f"Error: input file not found: {input_path}")

    process_changes(input_path, output_path, limit=args.limit, model=args.model)


if __name__ == "__main__":
    main()
