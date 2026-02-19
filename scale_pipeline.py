"""
Scale the diff + QA pipeline to 210 entities (30 per group × 7 groups).

Groups:
  1. people          (30 / 100)
  2. organizations   (30 / 100)
  3. products        (30 / 99)
  4. sports          (30 / 100)
  5. policy          (30 / 100)
  6. science_ai_space        (30 / 50)  — AI/ML + Space/Astronomy
  7. science_life_physical   (30 / 49)  — Physics/Chemistry + Biology/Medicine

Usage:
    python scale_pipeline.py                  # diff only → data/benchmark/changes_210.jsonl
    python scale_pipeline.py --with-qa        # diff + QA  → data/benchmark/changes_210.jsonl + data/benchmark/qa_pairs_210.jsonl
    python scale_pipeline.py --with-qa --qa-limit 5   # QA on first 5 changes only
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from diff_classifier import run_batch

_ROOT = Path(__file__).parent
_DATA = _ROOT / "data" / "wikipedia"
_SEED = 42
_PER_GROUP = 30


# ---------------------------------------------------------------------------
# 1. Build science subcategory mapping from notebook
# ---------------------------------------------------------------------------

def _load_science_subcategories() -> dict[str, str]:
    """Return {wikipedia_title: subcategory} for all science entities."""
    nb_path = _ROOT / "wikidata_fetch.ipynb"
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    source = "".join(nb["cells"][0]["source"])
    m = re.search(r"science\s*=\s*\[(.*?)\n\]", source, re.DOTALL)
    if not m:
        sys.exit("Error: could not find science entity list in notebook")
    parts = re.split(
        r"#\s*(AI/ML|Space/Astronomy|Physics/Chemistry|Biology/Medicine)",
        m.group(1),
    )
    mapping: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        subcat = parts[i]
        titles = re.findall(r'"wikipedia_title":\s*"([^"]+)"', parts[i + 1])
        for t in titles:
            mapping[t] = subcat
    return mapping


# ---------------------------------------------------------------------------
# 2. Sample entities
# ---------------------------------------------------------------------------

def sample_entities() -> list[tuple[str, Path]]:
    """
    Return a list of (group_name, entity_dir) for 210 entities,
    30 per group, deterministically sampled.
    """
    rng = random.Random(_SEED)
    science_map = _load_science_subcategories()

    # Non-science domains: sample 30 from each
    simple_domains = ["people", "organizations", "products", "sports", "policy"]
    result: list[tuple[str, Path]] = []

    for domain in simple_domains:
        domain_dir = _DATA / domain
        entities = sorted(d for d in domain_dir.iterdir() if d.is_dir())
        sampled = rng.sample(entities, min(_PER_GROUP, len(entities)))
        for e in sampled:
            result.append((domain, e))

    # Science: split into two groups
    science_dir = _DATA / "science"
    all_science = sorted(d for d in science_dir.iterdir() if d.is_dir())

    ai_space = [d for d in all_science if science_map.get(d.name) in ("AI/ML", "Space/Astronomy")]
    life_phys = [d for d in all_science if science_map.get(d.name) in ("Physics/Chemistry", "Biology/Medicine")]

    for group_name, pool in [("science_ai_space", ai_space), ("science_life_physical", life_phys)]:
        sampled = rng.sample(pool, min(_PER_GROUP, len(pool)))
        for e in sampled:
            result.append((group_name, e))

    return result


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scale pipeline to 210 entities")
    parser.add_argument("--with-qa", action="store_true", help="Also run QA generation")
    parser.add_argument("--qa-limit", type=int, default=None, help="Limit QA generation to N changes")
    args = parser.parse_args()

    # Step 1: Sample and run diff
    entities = sample_entities()

    # Print summary
    from collections import Counter
    group_counts = Counter(g for g, _ in entities)
    print(f"Sampled {len(entities)} entities across {len(group_counts)} groups:")
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count}")

    changes_path = _ROOT / "data/benchmark/changes_210.jsonl"
    print(f"\nRunning batch diff...")
    written = run_batch(entities, changes_path)
    print(f"Wrote {written} change records to {changes_path}")

    # Step 2: QA generation (optional)
    if args.with_qa:
        from qa_generator import process_changes
        qa_path = _ROOT / "data/benchmark/qa_pairs_210.jsonl"
        print(f"\nRunning QA generation...")
        process_changes(changes_path, qa_path, limit=args.qa_limit)


if __name__ == "__main__":
    main()
