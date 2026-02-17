"""
Myers diff algorithm + factual change classifier for Wikipedia snapshots.

Usage:
    python diff_classifier.py                          # demo on Andrej Karpathy
    python diff_classifier.py <entity_dir>             # compare first two snapshots
    python diff_classifier.py <old.json> <new.json>    # compare two specific files
    python diff_classifier.py --batch-all              # export all entities to changes_full.jsonl
    python diff_classifier.py --batch-all out.jsonl    # export to custom path
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator, NamedTuple


# ---------------------------------------------------------------------------
# 1. Myers diff
# ---------------------------------------------------------------------------

class Edit(NamedTuple):
    op: str        # 'equal' | 'insert' | 'delete'
    lines: list[str]


def myers_diff(a: list[str], b: list[str]) -> list[Edit]:
    """
    Pure Python implementation of the Myers diff algorithm (1986).

    Returns the shortest edit script as a list of Edit named-tuples with
    op in {'equal', 'insert', 'delete'}.  Consecutive runs of the same op
    are merged into a single Edit.

    Algorithm reference:
        Myers, E.W. (1986). "An O(ND) Difference Algorithm and Its
        Variations". Algorithmica 1(2), 251-266.
    """
    N, M = len(a), len(b)

    if N == 0 and M == 0:
        return []
    if N == 0:
        return [Edit('insert', list(b))]
    if M == 0:
        return [Edit('delete', list(a))]

    MAX = N + M
    offset = MAX          # V is indexed as V[k + offset]
    V = [0] * (2 * MAX + 2)

    # trace[d] = snapshot of V after processing edit distance d
    trace: list[list[int]] = []

    found = False
    for d in range(MAX + 1):
        trace.append(list(V))
        for k in range(-d, d + 1, 2):
            # Choose whether to move right (delete from a) or down (insert from b)
            if k == -d or (k != d and V[k - 1 + offset] < V[k + 1 + offset]):
                x = V[k + 1 + offset]        # move down  → insert b[y]
            else:
                x = V[k - 1 + offset] + 1    # move right → delete a[x]
            y = x - k
            # Follow the diagonal (equal elements)
            while x < N and y < M and a[x] == b[y]:
                x += 1
                y += 1
            V[k + offset] = x
            if x >= N and y >= M:
                found = True
                break
        if found:
            break

    # Back-track through the saved trace snapshots to recover the edit path
    path: list[tuple[str, str]] = []   # (op, item)
    x, y = N, M
    for d in range(len(trace) - 1, 0, -1):
        V = trace[d]
        k = x - y
        # Determine which branch was taken at step d
        if k == -d or (k != d and V[k - 1 + offset] < V[k + 1 + offset]):
            prev_k = k + 1   # came from a down move
        else:
            prev_k = k - 1   # came from a right move
        prev_x = V[prev_k + offset]
        prev_y = prev_x - prev_k
        # Walk back along the snake (equal elements)
        while x > prev_x and y > prev_y:
            path.append(('equal', a[x - 1]))
            x -= 1
            y -= 1
        # Record the single non-diagonal step
        if x > prev_x:
            path.append(('delete', a[x - 1]))
            x -= 1
        elif y > prev_y:
            path.append(('insert', b[y - 1]))
            y -= 1

    # Any remaining prefix is equal
    while x > 0 and y > 0:
        path.append(('equal', a[x - 1]))
        x -= 1
        y -= 1

    path.reverse()

    # Merge consecutive same-op steps
    merged: list[Edit] = []
    for op, item in path:
        if merged and merged[-1].op == op:
            merged[-1].lines.append(item)
        else:
            merged.append(Edit(op, [item]))

    return merged


# ---------------------------------------------------------------------------
# 2. Helpers for classification
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"""
    (?<!\w)          # not preceded by a word char
    -?               # optional minus
    (?:
        \d{1,3}(?:,\d{3})+   # comma-grouped integers:  1,234,567
      | \d+\.\d+              # decimals:                3.14
      | \d+                   # plain integers
    )
    (?!\w)           # not followed by a word char
    """,
    re.VERBOSE,
)

# Infobox field line:  "| key = value"  or  "| key ="
_INFOBOX_RE = re.compile(r"^\|\s*([\w_ ]+?)\s*=\s*(.*)")


def _extract_numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _strip_wiki_markup(text: str) -> str:
    """Remove common wiki markup for comparison purposes."""
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)  # [[link|label]]
    text = re.sub(r"'''?", "", text)                                  # bold/italic
    text = re.sub(r"{{[^}]*}}", "", text)                            # templates
    text = re.sub(r"<[^>]+>", "", text)                              # HTML tags
    return text.strip()


def _normalize_for_comparison(text: str) -> str:
    """Aggressively normalize text for cosmetic-change detection."""
    text = _strip_wiki_markup(text)
    # Normalize all quote variants to ASCII
    text = re.sub(r'[\u2018\u2019\u201A\u201B\u2032\u0060\u00B4]', "'", text)
    text = re.sub(r'[\u201C\u201D\u201E\u201F\u2033\u00AB\u00BB\u2039\u203A]', '"', text)
    # Normalize dashes
    text = re.sub(r'[\u2013\u2014\u2015]', '-', text)
    # Strip punctuation entirely for word-level comparison
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _numbers_differ(old: str, new: str) -> bool:
    """Return True when the two strings differ only (or primarily) in numeric values."""
    old_nums = _extract_numbers(old)
    new_nums = _extract_numbers(new)
    if not old_nums and not new_nums:
        return False
    # At least one side has numbers and they differ
    return old_nums != new_nums


def _same_infobox_key(old_line: str, new_line: str) -> bool:
    """Return True if both lines are infobox fields with the same key."""
    m_old = _INFOBOX_RE.match(old_line)
    m_new = _INFOBOX_RE.match(new_line)
    if m_old and m_new:
        return m_old.group(1).lower() == m_new.group(1).lower()
    return False


def _is_wording_only(old_lines: list[str], new_lines: list[str]) -> bool:
    """
    Return True when the change is cosmetic — same factual content with only
    wording, punctuation, or formatting differences.

    Uses two checks:
    1. After aggressive normalization (strip punctuation, quotes, markup),
       if the word sequences are identical → cosmetic.
    2. If the word-level overlap is >= 90% (Jaccard) and the only differences
       are common filler/function words → cosmetic.
    """
    old_norm = _normalize_for_comparison(" ".join(old_lines))
    new_norm = _normalize_for_comparison(" ".join(new_lines))

    if not old_norm and not new_norm:
        return True
    if not old_norm or not new_norm:
        return False   # one side empty = real addition/deletion

    # Check 1: identical after normalization
    if old_norm == new_norm:
        return True

    # Check 2: high token overlap — catches minor wording tweaks
    old_words = old_norm.split()
    new_words = new_norm.split()
    old_set = set(old_words)
    new_set = set(new_words)

    if not old_set or not new_set:
        return False

    intersection = old_set & new_set
    union = old_set | new_set
    jaccard = len(intersection) / len(union)

    if jaccard < 0.85:
        return False

    # The differing words must all be function/filler words
    _FILLER = {
        "a", "an", "the", "of", "to", "in", "for", "on", "at", "by",
        "and", "or", "is", "was", "are", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "will", "would",
        "shall", "should", "may", "might", "can", "could",
        "that", "which", "who", "whom", "whose", "this", "these",
        "it", "its", "as", "with", "from", "into", "also", "but",
        "not", "no", "so", "if", "then", "than", "both", "each",
        "serve", "served", "serving", "current", "currently",
        "continue", "continued", "continues", "continuing",
    }
    diff_words = (old_set ^ new_set)  # symmetric difference
    non_filler_diff = diff_words - _FILLER
    if not non_filler_diff:
        return True

    # Even with some non-filler diffs, if Jaccard >= 0.95 and numbers
    # are the same, it's very likely cosmetic
    if jaccard >= 0.95:
        old_nums = _extract_numbers(" ".join(old_lines))
        new_nums = _extract_numbers(" ".join(new_lines))
        if old_nums == new_nums:
            return True

    # For long texts where a few words were dropped/added but all facts
    # (names, numbers, dates) are preserved: check Jaccard similarity.
    # This catches e.g. "continued to serve as ... until current chair X"
    # → "continued as ... until X" where most content is identical.
    if len(old_words) >= 10 and len(new_words) >= 10 and jaccard >= 0.90:
        old_nums = _extract_numbers(" ".join(old_lines))
        new_nums = _extract_numbers(" ".join(new_lines))
        if old_nums == new_nums:
            return True

    return False


# ---------------------------------------------------------------------------
# 3. Diff context extraction & semantic annotation
# ---------------------------------------------------------------------------

# Ordered list of (subtype, pattern) — first match wins
_SUBTYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("leadership_change",  re.compile(
        r"\b(CEO|president|director|chairman|chair|founder|head of|minister|"
        r"chancellor|prime minister|secretary|governor|commissioner)\b", re.I)),
    ("numeric_stat",       re.compile(
        r"\b(revenue|sales|profit|loss|employees|staff|population|subscribers|"
        r"users|valuation|market cap|members|attendance|capacity|net worth)\b", re.I)),
    ("date_change",        re.compile(
        r"\b(founded|established|born|died|created|launched|released|published|"
        r"opened|closed|discontinued|introduced)\b", re.I)),
    ("location_change",    re.compile(
        r"\b(headquarters|headquartered|located|based in|moved to|relocated)\b", re.I)),
    ("status_change",      re.compile(
        r"\b(merged|acquired|dissolved|bankrupt|IPO|listed|delisted|renamed|"
        r"rebranded|split|spun off)\b", re.I)),
    ("award_change",       re.compile(
        r"\b(award|prize|won|champion|title|medal|nominated|record)\b", re.I)),
]


def extract_diff_context(
    old_text: str,
    new_text: str,
    max_chars: int = 500,
) -> tuple[str, str]:
    """
    Return (old_excerpt, new_excerpt): the most significant changed block,
    stripped of wiki markup and truncated to *max_chars*.

    "Most significant" = largest block by total character count; infobox
    lines are boosted so structured facts are preferred over prose edits.
    """
    a = old_text.splitlines()
    b = new_text.splitlines()
    edits = myers_diff(a, b)

    best_deleted: list[str] = []
    best_inserted: list[str] = []
    best_score = -1

    for deleted, inserted in _iter_change_groups(edits):
        raw_len = sum(len(l) for l in deleted) + sum(len(l) for l in inserted)
        # Boost infobox blocks so structured facts beat boilerplate prose
        infobox_boost = 2 if any(_INFOBOX_RE.match(l) for l in deleted + inserted) else 1
        score = raw_len * infobox_boost
        if score > best_score:
            best_score = score
            best_deleted = deleted
            best_inserted = inserted

    def _clean(lines: list[str]) -> str:
        joined = " ".join(_strip_wiki_markup(l) for l in lines)
        # Collapse excess whitespace
        joined = re.sub(r"\s+", " ", joined).strip()
        return joined[:max_chars]

    return _clean(best_deleted), _clean(best_inserted)


def infer_subtype(old_excerpt: str, new_excerpt: str, change_type: str) -> str:
    """Return a semantic subtype label for the change."""
    if change_type == "ADDITION":
        return "content_addition"
    if change_type == "DELETION":
        return "content_removal"
    combined = old_excerpt + " " + new_excerpt
    for subtype, pattern in _SUBTYPE_PATTERNS:
        if pattern.search(combined):
            return subtype
    return "content_update"


def infer_complexity(edits: list[Edit]) -> str:
    """'simple' if ≤ 5 lines were changed, 'complex' otherwise."""
    changed = sum(len(e.lines) for e in edits if e.op != "equal")
    return "simple" if changed <= 5 else "complex"


def build_temporal_gradient(
    entity_dir: Path,
    old_stem: str,
    new_stem: str,
) -> dict:
    """
    Map all available snapshots for *entity_dir* onto a temporal gradient
    relative to the change event:

        T-(N+1) … T-2  pre-change snapshots before old_snapshot
        T-1             old_snapshot (last known old state)
        T0              inferred change event (midpoint between old and new)
        T+1             new_snapshot (first known new state)
        T+2  … T+M      post-change snapshots after new_snapshot
    """
    stems = sorted(p.stem for p in entity_dir.glob("*.json"))
    entity_key = re.sub(r"[^a-z0-9]+", "_", entity_dir.name.lower()).strip("_")

    old_idx = stems.index(old_stem)
    new_idx = stems.index(new_stem)

    # Midpoint date as the inferred change event
    old_date = date.fromisoformat(old_stem)
    new_date = date.fromisoformat(new_stem)
    t0_date = old_date + timedelta(days=(new_date - old_date).days // 2)

    gradient: dict = {}

    # Pre-change snapshots: earliest → most-negative T label
    for i, stem in enumerate(stems[:old_idx]):
        t_label = f"T-{old_idx - i + 1}"
        gradient[t_label] = {
            "date": stem,
            "fact_state": "old",
            "doc_id": f"doc_{entity_key}_{stem}",
        }

    gradient["T-1"] = {
        "date": old_stem,
        "fact_state": "old",
        "doc_id": f"doc_{entity_key}_{old_stem}",
    }
    gradient["T0"] = {
        "date": t0_date.isoformat(),
        "fact_state": "change_event",
    }
    gradient["T+1"] = {
        "date": new_stem,
        "fact_state": "new",
        "doc_id": f"doc_{entity_key}_{new_stem}",
    }

    # Post-change snapshots
    for i, stem in enumerate(stems[new_idx + 1:]):
        gradient[f"T+{i + 2}"] = {
            "date": stem,
            "fact_state": "new",
            "doc_id": f"doc_{entity_key}_{stem}",
        }

    return gradient


def build_change_record(
    change_id: str,
    category: str,
    entity_dir: Path,
    old_path: Path,
    new_path: Path,
) -> dict | None:
    """
    Build a full change record in the benchmark schema.
    Returns None if the two snapshots are identical or the change is cosmetic.
    """
    old = load_snapshot(old_path)
    new = load_snapshot(new_path)

    if old["content"] == new["content"]:
        return None

    edits = myers_diff(old["content"].splitlines(), new["content"].splitlines())
    change_type = classify_change(old["content"], new["content"])

    if change_type == "COSMETIC":
        return None
    old_excerpt, new_excerpt = extract_diff_context(old["content"], new["content"])

    # Also check if the selected excerpt is cosmetic — the full-article
    # classifier may pass (real changes elsewhere) but the chosen excerpt
    # block may be a cosmetic diff that would produce useless QA pairs.
    if _is_wording_only([old_excerpt], [new_excerpt]):
        return None

    subtype = infer_subtype(old_excerpt, new_excerpt, change_type)
    complexity = infer_complexity(edits)
    gradient = build_temporal_gradient(entity_dir, old_path.stem, new_path.stem)

    entity_key = re.sub(r"[^a-z0-9]+", "_", entity_dir.name.lower()).strip("_")
    entity_id = f"entity_{entity_key}"

    # Extract numeric suffix from change_id for QA ID generation
    num = re.search(r"\d+$", change_id)
    base = int(num.group()) if num else 0
    derived_qa_ids = [f"freshrag_{base:05d}_{i + 1}" for i in range(3)]

    return {
        "change_id": change_id,
        "entity_id": entity_id,
        "domain": category,
        "change_detection": {
            "old_snapshot": old_path.stem,
            "new_snapshot": new_path.stem,
            "detection_method": "myers_diff",
            "diff_context": {
                "old_text": old_excerpt,
                "new_text": new_excerpt,
            },
        },
        "change_classification": {
            "type": change_type,
            "subtype": subtype,
            "complexity": complexity,
            "reversal": False,
        },
        "temporal_gradient": gradient,
        "derived_qa_ids": derived_qa_ids,
        "verification": {
            "wikidata_verified": False,
            "human_verified": False,
            "confidence": 0.0,
        },
    }


# ---------------------------------------------------------------------------
# 4. Change classifier
# ---------------------------------------------------------------------------

def _iter_change_groups(edits: list[Edit]) -> Iterator[tuple[list[str], list[str]]]:
    """
    Walk the edit list and yield (deleted_lines, inserted_lines) pairs that
    represent a single logical change (a contiguous block of non-equal ops).
    """
    i = 0
    while i < len(edits):
        if edits[i].op == 'equal':
            i += 1
            continue
        deleted: list[str] = []
        inserted: list[str] = []
        while i < len(edits) and edits[i].op != 'equal':
            if edits[i].op == 'delete':
                deleted.extend(edits[i].lines)
            else:
                inserted.extend(edits[i].lines)
            i += 1
        yield deleted, inserted


def classify_change(old_text: str, new_text: str) -> str:
    """
    Run Myers diff on *old_text* vs *new_text* (compared line-by-line) and
    return the dominant change type:

        FACTUAL_UPDATE  – core fact changed (name, role, place, …)
        NUMERIC_UPDATE  – a number changed (stat, date, count, …)
        ADDITION        – new information added, nothing removed
        DELETION        – information removed, nothing added
        COSMETIC        – only punctuation, formatting, or wording tweaks

    Priority when multiple types are present:
        FACTUAL_UPDATE > NUMERIC_UPDATE > ADDITION > DELETION > COSMETIC
    """
    a = old_text.splitlines()
    b = new_text.splitlines()

    edits = myers_diff(a, b)

    has_addition = False
    has_deletion = False
    has_numeric = False
    has_factual = False

    for deleted, inserted in _iter_change_groups(edits):
        if deleted and not inserted:
            has_deletion = True
        elif inserted and not deleted:
            has_addition = True
        else:
            # Substitution: lines both deleted and inserted
            old_block = " ".join(deleted)
            new_block = " ".join(inserted)

            # Skip pure wording/markup changes (same semantic content)
            if _is_wording_only(deleted, inserted):
                continue

            # Check if any infobox field has the same key but different value
            numeric_in_pair = False
            for old_line, new_line in zip(deleted, inserted):
                if _same_infobox_key(old_line, new_line):
                    if _numbers_differ(old_line, new_line):
                        numeric_in_pair = True
                    else:
                        has_factual = True

            # Fall back to block-level number check
            if numeric_in_pair or _numbers_differ(old_block, new_block):
                has_numeric = True
            else:
                has_factual = True

    # Return the highest-priority type seen
    if has_factual:
        return "FACTUAL_UPDATE"
    if has_numeric:
        return "NUMERIC_UPDATE"
    if has_addition:
        return "ADDITION"
    if has_deletion:
        return "DELETION"
    return "COSMETIC"   # non-empty diff with only wording/formatting changes


# ---------------------------------------------------------------------------
# 5. Batch utilities
# ---------------------------------------------------------------------------

TIMESTAMPS = [
    "2023-01-01",
    "2023-06-01",
    "2024-01-01",
    "2024-06-01",
    "2025-01-01",
]


def load_snapshot(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def compare_entity(entity_dir: Path) -> list[dict]:
    """
    Compare consecutive snapshots for one entity directory.
    Returns a list of result dicts.
    """
    snapshots = sorted(entity_dir.glob("*.json"))
    results = []
    for i in range(len(snapshots) - 1):
        old_path = snapshots[i]
        new_path = snapshots[i + 1]
        old = load_snapshot(old_path)
        new = load_snapshot(new_path)
        if old["content"] == new["content"]:
            change_type = None          # identical – no diff needed
        else:
            change_type = classify_change(old["content"], new["content"])
        results.append({
            "entity": entity_dir.name,
            "from": old_path.stem,
            "to": new_path.stem,
            "change_type": change_type,
        })
    return results


def diff_summary(old_text: str, new_text: str, context: int = 2) -> str:
    """Return a human-readable unified-diff-style summary of the changes."""
    a = old_text.splitlines()
    b = new_text.splitlines()
    edits = myers_diff(a, b)

    lines = []
    for edit in edits:
        if edit.op == 'equal':
            for line in edit.lines[-context:]:
                lines.append(f"  {line}")
        elif edit.op == 'delete':
            for line in edit.lines:
                lines.append(f"- {line}")
        elif edit.op == 'insert':
            for line in edit.lines:
                lines.append(f"+ {line}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. CLI entry point
# ---------------------------------------------------------------------------

def _demo():
    data_root = Path(__file__).parent / "data" / "wikipedia"
    entity_dir = data_root / "people" / "Andrej_Karpathy"

    print("=== Myers diff + factual change classifier demo ===")
    print(f"Entity: {entity_dir.name}\n")

    snapshots = sorted(entity_dir.glob("*.json"))
    for i in range(len(snapshots) - 1):
        old = load_snapshot(snapshots[i])
        new = load_snapshot(snapshots[i + 1])
        label = f"{snapshots[i].stem} → {snapshots[i+1].stem}"

        if old["content"] == new["content"]:
            print(f"[{label}]  (no change)")
            continue

        change = classify_change(old["content"], new["content"])
        print(f"[{label}]  {change}")

        # Show a small excerpt of the diff
        summary = diff_summary(old["content"], new["content"], context=1)
        excerpt = "\n".join(summary.splitlines()[:30])
        print(excerpt)
        print()


def run_batch(
    entity_dirs: list[tuple[str, Path]],
    out_path: Path,
    start_id: int = 1,
) -> int:
    """
    Process a list of (category, entity_dir) pairs, write change records to
    *out_path* as JSONL in the benchmark schema.  Returns the next unused ID.
    """
    counter = start_id
    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for category, entity_dir in entity_dirs:
            snapshots = sorted(entity_dir.glob("*.json"))
            for i in range(len(snapshots) - 1):
                change_id = f"change_{counter:05d}"
                record = build_change_record(
                    change_id=change_id,
                    category=category,
                    entity_dir=entity_dir,
                    old_path=snapshots[i],
                    new_path=snapshots[i + 1],
                )
                counter += 1
                if record is None:
                    continue
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    return written


def _batch_all(out_path: str = "changes_full.jsonl"):
    """Run batch diff on all entities across all categories."""
    data_root = Path(__file__).parent / "data" / "wikipedia"
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

    print(f"Processing {len(entity_dirs)} entities across {len(categories)} categories...")
    written = run_batch(entity_dirs, Path(out_path))
    print(f"Done. Wrote {written} change records to {out_path}")


def main():
    args = sys.argv[1:]

    if args and args[0] == "--batch-all":
        out = args[1] if len(args) > 1 else "changes_full.jsonl"
        _batch_all(out)

    elif len(args) == 0:
        _demo()

    elif len(args) == 1:
        entity_dir = Path(args[0])
        if not entity_dir.is_dir():
            sys.exit(f"Error: {entity_dir} is not a directory")
        for result in compare_entity(entity_dir):
            ct = result["change_type"] or "NO_CHANGE"
            print(f"{result['from']} → {result['to']}  {ct}")

    elif len(args) == 2:
        old_path, new_path = Path(args[0]), Path(args[1])
        old = load_snapshot(old_path)
        new = load_snapshot(new_path)
        if old["content"] == new["content"]:
            print("NO_CHANGE")
        else:
            print(classify_change(old["content"], new["content"]))

    else:
        sys.exit("Usage: diff_classifier.py [--batch-all [output.jsonl] | entity_dir | old.json new.json]")


if __name__ == "__main__":
    main()
