"""
FreshRAG Evaluation Harness (Phase 3.4).

Scores model predictions against ground truth using 6 metrics:
FA (Factual Accuracy), HR (Hallucination Rate), AU (Answer Uncertainty),
CRS (Change Recognition Score), TGS (Temporal Gradient Score),
POS (Preference for Outdated Score).

Optionally generates predictions by calling an LLM with eval instances.

Usage:
    python eval_harness.py --instances eval_instances_210.jsonl --predictions predictions.jsonl
    python eval_harness.py --instances eval_instances_210.jsonl --run-predictions
    python eval_harness.py --instances eval_instances_210.jsonl --run-predictions --predictions-only
    python eval_harness.py --instances eval_instances_210.jsonl --run-predictions --model gpt-4o --limit 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent

_RAG_SYSTEM_PROMPT = """\
You are a helpful assistant. Answer the user's question using ONLY the \
provided documents. Be concise (1-3 sentences). If the documents contain \
conflicting information, note the conflict and state which information \
appears most current. If the documents do not contain enough information \
to answer, say so."""

_HEDGING_PATTERNS = re.compile(
    r"(?:i'?m not sure|it'?s unclear|it is unclear|not certain|cannot determine"
    r"|can'?t determine|i don'?t know|uncertain|conflicting information"
    r"|contradictory|it'?s? (?:possible|difficult to)"
    r"|may have (?:changed|been updated)|information (?:varies|conflicts)"
    r"|hard to say|no (?:clear|definitive) answer"
    r"|the documents (?:do not|don'?t) (?:contain|provide|mention)"
    r"|not enough information|insufficient information"
    r"|based on (?:available|the provided) (?:information|documents),"
    r" (?:it is|it'?s) (?:unclear|not clear))",
    re.IGNORECASE,
)

_TEMPORAL_KEYWORDS = re.compile(
    r"(?:previously|formerly|used to be|was previously|has (?:since )?changed"
    r"|has been updated|no longer|changed (?:from|to)|updated (?:from|to)"
    r"|as of \d{4}|more recently|in the past|originally"
    r"|however,? (?:more )?recent|the (?:latest|newest|most recent)"
    r"|now (?:is|has|serves?|holds?)|currently)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 2. Scoring primitives
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase text, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 between prediction and reference (SQuAD-style).
    Returns 0.0 if either is empty.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_bag = defaultdict(int)
    for t in pred_tokens:
        pred_bag[t] += 1
    ref_bag = defaultdict(int)
    for t in ref_tokens:
        ref_bag[t] += 1

    # Count common tokens (min of counts)
    common = 0
    for t, count in pred_bag.items():
        common += min(count, ref_bag.get(t, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _detect_uncertainty(text: str) -> bool:
    """Check if text contains hedging/uncertainty language."""
    return bool(_HEDGING_PATTERNS.search(text))


def _detect_change_recognition(prediction: str, old_answer: str, new_answer: str) -> bool:
    """
    Check if prediction acknowledges that information changed.
    Either: mentions both old and new facts, or uses temporal keywords.
    """
    # Check temporal keywords
    if _TEMPORAL_KEYWORDS.search(prediction):
        return True

    # Check if prediction has token overlap with BOTH old and new answers
    old_f1 = token_f1(prediction, old_answer)
    new_f1 = token_f1(prediction, new_answer)
    if old_f1 > 0.2 and new_f1 > 0.2:
        return True

    return False


def score_instance(instance: dict, prediction: str) -> dict:
    """
    Score a single prediction against an eval instance.
    Returns dict with FA, HR, AU, CRS, POS metrics.
    """
    new_answer = instance["new_answer"]
    old_answer = instance["old_answer"]

    fa = token_f1(prediction, new_answer)
    old_f1 = token_f1(prediction, old_answer)

    # HR: hallucination — matches neither answer
    hr = 1 if (fa < 0.3 and old_f1 < 0.3) else 0

    # AU: answer uncertainty
    au = 1 if _detect_uncertainty(prediction) else 0

    # CRS: change recognition
    crs = 1 if _detect_change_recognition(prediction, old_answer, new_answer) else 0

    # POS: preference for outdated
    pos = 1 if (old_f1 > fa and old_f1 > 0.3) else 0

    return {"FA": round(fa, 4), "HR": hr, "AU": au, "CRS": crs, "POS": pos}


# ---------------------------------------------------------------------------
# 3. Prediction generation
# ---------------------------------------------------------------------------

def _init_client():
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

    from openai import OpenAI
    return OpenAI(api_key=api_key)


def _build_rag_prompt(instance: dict) -> str:
    """Build the user prompt with question and retrieved documents."""
    docs_text = []
    for doc in instance["retrieved_docs"]:
        docs_text.append(f"[Document — {doc['date']}]\n{doc['content']}")

    return (
        f"Documents:\n\n{'\\n\\n'.join(docs_text)}\n\n"
        f"Question: {instance['question']}"
    )


def _load_predicted_ids(output_path: Path) -> set[str]:
    """Read already-predicted instance IDs for resume support."""
    ids: set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        ids.add(rec.get("instance_id", ""))
                    except json.JSONDecodeError:
                        continue
    return ids


def run_predictions(
    instances_path: Path,
    output_path: Path,
    model: str = "gpt-4o-mini",
    limit: int | None = None,
) -> int:
    """
    Generate predictions for eval instances by calling OpenAI.
    Writes {instance_id, prediction} JSONL. Supports resume.
    Returns number of predictions written.
    """
    # Load instances
    instances: list[dict] = []
    with open(instances_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                instances.append(json.loads(line))

    if limit is not None:
        instances = instances[:limit]

    # Check for resume
    predicted = _load_predicted_ids(output_path)
    remaining = [inst for inst in instances if inst["instance_id"] not in predicted]

    if not remaining:
        print(f"All {len(instances)} instances already predicted. Nothing to do.")
        return 0

    print(f"Generating predictions for {len(remaining)} instances "
          f"({len(predicted)} already done, {len(instances)} total)")

    client = _init_client()
    written = 0

    with open(output_path, "a", encoding="utf-8") as fh:
        for i, inst in enumerate(remaining, 1):
            instance_id = inst["instance_id"]
            print(f"  [{i}/{len(remaining)}] {instance_id}...", end=" ", flush=True)

            user_prompt = _build_rag_prompt(inst)

            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": _RAG_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=300,
                    )
                    prediction = response.choices[0].message.content.strip()
                    break
                except Exception as exc:
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
                        continue
                    print(f"ERROR: {exc}")
                    prediction = ""
                    break

            rec = {"instance_id": instance_id, "prediction": prediction}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            print(f"done ({len(prediction)} chars)")

            if i < len(remaining):
                time.sleep(0.3)

    print(f"\nWrote {written} predictions to {output_path}")
    return written


# ---------------------------------------------------------------------------
# 4. Aggregation
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    """Compute mean, return 0 if empty."""
    return sum(values) / len(values) if values else 0.0


def _pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient. Returns None if insufficient data."""
    n = len(x)
    if n < 3:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return None
    return cov / denom


def aggregate_results(scored: list[dict]) -> dict:
    """
    Aggregate scored instances into summary tables.
    Returns dict with overall, by_scenario, by_domain, by_mechanism, and tgs.
    """
    metrics = ("FA", "HR", "AU", "CRS", "POS")

    def _group_stats(items: list[dict]) -> dict:
        if not items:
            return {m: 0.0 for m in metrics} | {"n": 0}
        result = {}
        for m in metrics:
            result[m] = round(_mean([it["scores"][m] for it in items]), 4)
        result["n"] = len(items)
        return result

    # Overall
    report: dict = {"overall": _group_stats(scored)}

    # By scenario
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for item in scored:
        by_scenario[item["scenario"]].append(item)
    report["by_scenario"] = {
        k: _group_stats(v) for k, v in sorted(by_scenario.items())
    }

    # By domain
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for item in scored:
        by_domain[item["domain"]].append(item)
    report["by_domain"] = {
        k: _group_stats(v) for k, v in sorted(by_domain.items())
    }

    # By mechanism
    by_mechanism: dict[str, list[dict]] = defaultdict(list)
    for item in scored:
        by_mechanism[item["predicted_mechanism"]].append(item)
    report["by_mechanism"] = {
        k: _group_stats(v) for k, v in sorted(by_mechanism.items())
    }

    # TGS: compute FA by temporal distance
    # Group instances by the temporal distance of their retrieved docs
    # For S5 instances, use the scenario to infer distance isn't straightforward.
    # Instead, compute TGS from S1 (fresh) and S2 (stale) scenario instances
    # by mapping doc dates to temporal distances via the original QA temporal_gradient.
    # Simpler approach: group S1/S2 instances by their doc dates' distance from change.
    # We approximate by using doc fact_state: fresh docs have positive distance, stale negative.
    # For a proper TGS, we group by the number of distinct fresh/stale docs.
    distance_fa: dict[str, list[float]] = defaultdict(list)
    for item in scored:
        # Use the fact_states of retrieved docs to approximate temporal distance
        states = [d["fact_state"] for d in item.get("retrieved_docs", [])]
        fresh_count = states.count("new")
        stale_count = states.count("old")
        # Map to a distance score: more fresh = positive, more stale = negative
        if fresh_count > 0 and stale_count == 0:
            dist_label = f"+{fresh_count}"
        elif stale_count > 0 and fresh_count == 0:
            dist_label = f"-{stale_count}"
        else:
            dist_label = f"+{fresh_count}-{stale_count}"
        distance_fa[dist_label].append(item["scores"]["FA"])

    tgs_distances: list[str] = []
    tgs_fa_means: list[float] = []
    tgs_numeric_x: list[float] = []
    tgs_numeric_y: list[float] = []

    for label in sorted(distance_fa.keys(), key=lambda s: (s[0] != "-", s)):
        fa_mean = round(_mean(distance_fa[label]), 4)
        tgs_distances.append(label)
        tgs_fa_means.append(fa_mean)
        # Try to convert to numeric for correlation
        try:
            # Parse distance: "+2" -> 2, "-3" -> -3
            if "+" in label and "-" in label:
                continue  # skip mixed for correlation
            numeric = int(label)
            tgs_numeric_x.append(numeric)
            tgs_numeric_y.append(fa_mean)
        except ValueError:
            pass

    tgs_corr = _pearson_correlation(tgs_numeric_x, tgs_numeric_y)

    report["tgs"] = {
        "distances": tgs_distances,
        "fa_means": tgs_fa_means,
        "correlation": round(tgs_corr, 4) if tgs_corr is not None else None,
    }

    return report


# ---------------------------------------------------------------------------
# 5. Main evaluate pipeline
# ---------------------------------------------------------------------------

def evaluate(
    instances_path: Path,
    predictions_path: Path,
    output_dir: Path | None = None,
) -> dict:
    """
    Load instances + predictions, score each, aggregate, write results.
    Returns the aggregated report.
    """
    if output_dir is None:
        output_dir = instances_path.parent

    stem = instances_path.stem.replace("eval_instances_", "")

    # Load instances
    instances: dict[str, dict] = {}
    with open(instances_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                instances[rec["instance_id"]] = rec

    # Load predictions
    predictions: dict[str, str] = {}
    with open(predictions_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                predictions[rec["instance_id"]] = rec["prediction"]

    # Match and score
    matched = 0
    missing = 0
    scored: list[dict] = []

    scored_path = output_dir / f"scored_{stem}.jsonl"
    with open(scored_path, "w", encoding="utf-8") as fh:
        for instance_id, inst in instances.items():
            prediction = predictions.get(instance_id)
            if prediction is None:
                missing += 1
                continue

            scores = score_instance(inst, prediction)
            matched += 1

            record = {
                "instance_id": instance_id,
                "qa_id": inst["qa_id"],
                "scenario": inst["scenario"],
                "domain": inst["domain"],
                "predicted_mechanism": inst["predicted_mechanism"],
                "change_type": inst["change_type"],
                "question": inst["question"],
                "new_answer": inst["new_answer"],
                "old_answer": inst["old_answer"],
                "prediction": prediction,
                "scores": scores,
                "retrieved_docs": inst.get("retrieved_docs", []),
            }
            scored.append(record)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Scored {matched} instances ({missing} missing predictions)")
    print(f"  Scored output: {scored_path}")

    # Aggregate
    report = aggregate_results(scored)
    report_path = output_dir / f"report_{stem}.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"  Summary report: {report_path}")

    # Print summary table
    _print_report(report)

    return report


def _print_report(report: dict) -> None:
    """Print a formatted summary of the evaluation report."""
    metrics = ("FA", "HR", "AU", "CRS", "POS")
    header = f"{'Group':<25s} {'n':>5s}  " + "  ".join(f"{m:>6s}" for m in metrics)
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    # Overall
    o = report["overall"]
    print(f"{'OVERALL':<25s} {o['n']:>5d}  " +
          "  ".join(f"{o[m]:>6.3f}" for m in metrics))
    print(sep)

    # By scenario
    for key, vals in sorted(report.get("by_scenario", {}).items()):
        print(f"{key:<25s} {vals['n']:>5d}  " +
              "  ".join(f"{vals[m]:>6.3f}" for m in metrics))
    print(sep)

    # By domain
    for key, vals in sorted(report.get("by_domain", {}).items()):
        print(f"{key:<25s} {vals['n']:>5d}  " +
              "  ".join(f"{vals[m]:>6.3f}" for m in metrics))
    print(sep)

    # By mechanism
    for key, vals in sorted(report.get("by_mechanism", {}).items()):
        print(f"{key:<25s} {vals['n']:>5d}  " +
              "  ".join(f"{vals[m]:>6.3f}" for m in metrics))
    print(sep)

    # TGS
    tgs = report.get("tgs", {})
    if tgs.get("correlation") is not None:
        print(f"\nTGS (Temporal Gradient Score): r = {tgs['correlation']:.4f}")
    if tgs.get("distances"):
        print("  Distance  |  FA mean")
        for d, fa in zip(tgs["distances"], tgs["fa_means"]):
            print(f"  {d:>8s}  |  {fa:.4f}")


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FreshRAG evaluation harness — score predictions with 6 metrics",
    )
    parser.add_argument(
        "--instances", "-i",
        required=True,
        help="Input JSONL file with eval instances",
    )
    parser.add_argument(
        "--predictions", "-p",
        default=None,
        help="JSONL file with predictions ({instance_id, prediction})",
    )
    parser.add_argument(
        "--run-predictions",
        action="store_true",
        help="Generate predictions via OpenAI before scoring",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Only generate predictions, skip scoring",
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="OpenAI model for prediction generation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Process only the first N instances",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory for output files (default: same as instances)",
    )

    args = parser.parse_args()

    instances_path = Path(args.instances)
    if not instances_path.exists():
        sys.exit(f"Error: instances file not found: {instances_path}")

    output_dir = Path(args.output_dir) if args.output_dir else instances_path.parent
    stem = instances_path.stem.replace("eval_instances_", "")

    # Determine predictions path
    predictions_path = Path(args.predictions) if args.predictions else output_dir / f"predictions_{stem}.jsonl"

    # Generate predictions if requested
    if args.run_predictions:
        run_predictions(instances_path, predictions_path, model=args.model, limit=args.limit)
        if args.predictions_only:
            return

    # Score
    if not predictions_path.exists():
        sys.exit(f"Error: predictions file not found: {predictions_path}\n"
                 f"Use --run-predictions to generate predictions first.")

    evaluate(instances_path, predictions_path, output_dir)


if __name__ == "__main__":
    main()
