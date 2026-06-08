# CFMM Broadcast Diarization

## Description

This pipeline processes raw broadcast transcript exports from Critical Mention and produces a structured, speaker-attributed dialogue dataset in Excel format. Given that Critical Mention transcripts are unpunctuated, unstructured, and lack speaker labels, the pipeline cleans, filters, and diarizes them using a combination of local ML models (BERT/DeBERTa) and Gemini LLM agents.

The output is a row-per-dialogue Excel file where each line of speech is attributed to a named speaker, enriched with metadata (program, channel, date, speaker role).

---

## Pipeline Workflow

The pipeline executes the following steps in order:

| Step | Name | Description |
|------|------|-------------|
| 1 | **Ingest** | Loads the Critical Mention CSV export |
| 2 | **Group** | Consolidates consecutive transcript fragments (within 1 minute) from the same program and channel into unified dialogue blocks |
| 3 | **Punctuation Restoration** | Uses a Gemini LLM to restore punctuation and casing to raw unpunctuated transcripts |
| 4 | **Relevance Filtering** | Uses a Gemini LLM to discard non-news content (ads, weather, sport, etc.), keeping only groups relevant to the research focus (Islam/Muslim discourse) |
| 5 | **Name Extraction** | Runs a two-stage process: a local BERT NER model (`dslim/bert-base-NER`) identifies person mentions, then a Gemini LLM assigns each person a role and context-of-mention |
| 6 | **Diarization** | Uses a Gemini LLM to attribute each sentence in the dialogue to a named speaker, producing a structured dialogue with confidence scores |
| 7 | **Verification** | Re-checks diarized dialogues where confidence fell below the threshold, using a Gemini LLM to correct uncertain attributions |
| 8 | **Export** | Writes the final speaker-attributed dialogue table to an `.xlsx` file in `data/output/` |

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
│   ├── punctuation.py          # Step 3: punctuation restoration agent
│   ├── relevance.py            # Step 4: relevance filtering agent
│   ├── name_extraction.py      # Step 5: NER + who/why Gemini agent
│   ├── diarize.py              # Step 6: speaker diarization agent
│   └── verify.py               # Step 7: low-confidence verification agent
│
├── utils/
│   ├── db_manager.py           # SQLite persistence layer
│   ├── csv_loader.py           # Critical Mention CSV ingestion
│   ├── prompt_loader.py        # Loads YAML prompt templates
│   ├── progress.py             # Async progress bar utility
│   └── logger.py               # Coloured console + file logger
│
├── config/
│   └── constants.py            # All tunable settings (concurrency, costs, model paths, etc.)
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
- DeBERTa model checkpoint file (not included in the repo, must be obtained separately)

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

### 4. Add the DeBERTa model checkpoint

Place the trained checkpoint at:

```
model/best_deberta_checkpoint.pt
```

This file is not tracked in git and must be obtained separately.

### 5. Add your input data

Place your Critical Mention CSV export into the `data/input/` folder.

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
group | punctuation | relevance | names | diarize
```

**Example — run only through the relevance filter on 50 rows:**

```bash
python main.py --filename "my_export" --limit 50 --test relevance
```

---

## Output

A `.xlsx` file is written to `data/output/` with one row per attributed dialogue line, including speaker name, role, program, channel, broadcast date, and confidence score.

If `--limit` was used, the output filename will include the limit value (e.g. `my_export_limit_50_dialogues.xlsx`) so test runs do not overwrite full runs.

---

## Notes

- The pipeline is resumable. If a run is interrupted, re-running with the same `--filename` will pick up from where it left off using the SQLite database.
- LLM cost is tracked per stage and printed in full at the end of each run. See `config/constants.py` to update pricing if Gemini rates change.
- All configurable settings (concurrency, batch sizes, confidence thresholds, relevance keywords) are centralised in `config/constants.py`.
