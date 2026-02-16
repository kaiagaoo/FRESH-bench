# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FRESH-bench** is a research benchmark dataset for evaluating AI/LLM systems on temporally-changing factual information. It collects Wikipedia article snapshots at multiple timestamps (2023-2025) across 598 entities in 8 categories, enabling evaluation of how well models handle knowledge that changes over time.

## Running the Notebook

```bash
# Install dependencies
pip install requests jupyter

# Run the data collection notebook
jupyter notebook wikidata_fetch.ipynb
```

The `.env` file contains `OPENAI_API_KEY` — load it before running any cells that call OpenAI APIs.

## Architecture

All logic lives in a single Jupyter notebook: `wikidata_fetch.ipynb`

**Key function:**
- `get_article_at_timestamp(title, timestamp, max_retries=3)` — Fetches a Wikipedia article's content at a specific timestamp via the Wikipedia API. Returns a dict with `title`, `page_id`, `revision_id`, `revision_timestamp`, and `content`.

**Entity definitions:** Each entity has fields: `name`, `wikipedia_title`, `fame_level` (`"high"/"medium"/"low"`), and `change_type` (e.g., `"roles/companies"`, `"policy positions"`).

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
