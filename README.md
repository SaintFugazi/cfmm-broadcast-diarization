# CFMM Broadcast Diarization

## Description

This pipeline processes raw broadcast transcript exports from Critical Mention and produces a structured, speaker-attributed dialogue dataset in Excel format. Given that Critical Mention transcripts are unpunctuated, unstructured, and lack speaker labels, the pipeline cleans, filters, segments, and diarizes them using a combination of local ML models and Gemini LLM agents.

The output is a windowed Excel file where each row represents a contiguous broadcast window — consecutive transcript groups merged within a configurable time threshold — enriched with stitched dialogue and a speaker-labelled diarized version.

---

## Pipeline Workflow

The pipeline executes the following steps in order:

| Step | Name | Description |
|------|------|-------------|
| 1 | **Ingest** | Loads the Critical Mention CSV export and filters rows to known news broadcast windows |
| 2 | **Group** | Consolidates consecutive 1-minute transcript fragments from the same channel and date into unified blocks (up to `GROUP_MAX_MINUTES`) |
| 3 | **Relevance Filtering** | Uses a Gemini LLM to discard non-news content (ads, weather, sport, etc.), keeping only groups relevant to the research focus |
| 4 | **Punctuation Restoration** | Uses a Gemini LLM to restore punctuation and casing to raw unpunctuated transcripts; falls back to a local model on safety blocks |
| 5 | **Token Count** | Classifies each group as `OVER` or `UNDER` a token threshold; `OVER` groups are routed to an indexed variant of the segmenter and attributor |
| 6 | **Segmentation** | Uses a Gemini LLM to split each group at speaker-change boundaries, producing ordered verbatim segments with no attribution yet |
| 7 | **Attribution** | Uses a Gemini LLM to assign each segment a speaker name, role, speaker type, confidence score, and a cleaned dialogue text |
| 8 | **Verification** | Re-checks attributions below the confidence threshold using a Gemini LLM; corrects speaker and role where needed |
| 9 | **Boundary Stitching** | Merges the last dialogue of one group with the first of the next when the speaker is the same, eliminating artificial cross-group splits |
| 10 | **Export** | Merges consecutive groups into broadcast windows, builds the `final` table, and writes it to Excel |

Progress is persisted to a local SQLite database after each step, so interrupted runs resume where they left off rather than restarting from scratch.

---

## Project Structure

```
cfmm-broadcast-diarization/
│
├── main.py                     # Entry point — parses CLI args and runs the pipeline
│
├── pipeline/
│   ├── orchestrator.py         # Coordinates all steps end-to-end
│   ├── group.py                # Step 2: transcript grouping logic
│   ├── punctuation.py          # Step 4: punctuation restoration agent
│   ├── relevance.py            # Step 3: relevance filtering agent
│   ├── count.py                # Step 5: transcript size classification
│   ├── segmentation.py         # Step 6: speaker-change segmentation agent
│   ├── attribution.py          # Step 7: speaker attribution agent
│   └── verify.py               # Step 8: low-confidence verification agent
│
├── utils/
│   ├── db_manager.py           # SQLite persistence layer (including final table builder)
│   ├── csv_loader.py           # Critical Mention CSV ingestion and news-window filtering
│   ├── prompt_loader.py        # Loads YAML prompt templates
│   ├── progress.py             # Async progress bar utility
│   └── logger.py               # Coloured console + file logger
│
├── config/
│   ├── constants.py            # All tunable settings (concurrency, costs, thresholds, etc.)
│   └── news_sched.csv          # News broadcast schedule used to filter input rows
│
├── prompts/                    # YAML prompt templates for each Gemini agent
├── model/                      # DeBERTa checkpoint (not tracked in git)
├── data/
│   ├── input/                  # Place Critical Mention CSV exports here
│   └── output/                 # Excel output written here
└── logs/                       # Per-run log files (not tracked in git)
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.11 or higher
- A Google Gemini API key

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-lite
```

### 4. Add your input data

Place your Critical Mention CSV export into the `data/input/` folder. The CSV must contain these columns:

```
Broadcast Date, Broadcast Time, Program Title, Channel Name, Plain Text
```

---

## Running the Pipeline

```bash
python main.py --filename "your_filename"
```

The `--filename` argument should be the name of your CSV file **without** the `.csv` extension.

### Optional flags

| Flag | Description |
|------|-------------|
| `--limit [n]` | Process only the first `n` rows — useful for testing on a small sample |
| `--test [step]` | Stop the pipeline after a specific step — useful for debugging a single stage |

### `--test` step options

```
group | relevance | punctuation | count | segmentation | attribution
```

**Example — run only through the relevance filter on 50 rows:**

```bash
python main.py --filename "my_export" --limit 50 --test relevance
```

---

## Output

A `.xlsx` file is written to `data/output/` with one row per broadcast window. A window is formed by merging consecutive grouped transcript blocks whose broadcast times are within `GROUP_MAX_MINUTES` of each other.

| Column | Description |
|--------|-------------|
| `window_id` | Auto-incremented unique identifier for the window |
| `program_title` | Most frequent program title among all groups in the window |
| `channel_name` | Broadcast channel |
| `broadcast_date` | Date of the broadcast |
| `broadcast_time` | Time range of the window (`HH:MM:SS - HH:MM:SS`) |
| `duration_min` | Duration of the window in minutes |
| `n_segments` | Total number of attributed dialogue turns in the window |
| `speakers` | Unique speakers in order of first appearance, comma-separated |
| `stitched_dialogue` | Full dialogue text, grouped by transcript block with timestamps |
| `diarized_dialogue` | Speaker-labelled dialogue; consecutive turns from the same speaker are merged under one heading |

If `--limit` was used, the output filename will include the limit value (e.g. `my_export_limit_50_dialogues.xlsx`) so test runs do not overwrite full runs.

---

## Notes

- The pipeline is resumable. If a run is interrupted, re-running with the same `--filename` will pick up from where it left off using the SQLite database.
- LLM cost is tracked per stage and printed in full at the end of each run. See `config/constants.py` to update pricing if Gemini rates change.
- All configurable settings (concurrency, batch sizes, confidence thresholds, relevance keywords, window size) are centralised in `config/constants.py`.
- `GROUP_MAX_MINUTES` controls both how many 1-minute clips are merged into a single group (Step 2) and the maximum gap allowed when merging groups into export windows (Step 10).
