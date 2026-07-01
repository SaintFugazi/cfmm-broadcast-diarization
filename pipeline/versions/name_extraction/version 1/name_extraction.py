import os
import json
import uuid
import asyncio

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from google import genai
from google.genai import types

from utils.progress import gather_with_progress
from utils.rate_limit import (
    AsyncRateLimiter,
    async_call_with_retry,
    validate_response,
    close_async_client,
    PIPELINE_SAFETY_SETTINGS,
    SafetyBlockedError,
)

from utils.logger import get_logger
from utils.prompt_loader import load_prompt
from config.constants import RowStatus, BlockStatus
from config.constants import (
    NER_MODEL_NAME,
    NER_BATCH_SIZE,
    NER_MAX_LENGTH,
    NER_DEVICE,
    NAME_EXTRACTION_CONCURRENCY,
    NAME_EXTRACTION_MAX_RETRIES,
    NAME_EXTRACTION_BACKOFF_BASE,
    NAME_EXTRACTION_BACKOFF_CAP,
    WHO_AND_WHY_CHUNK_SIZE,
    GEMINI_RPM_LIMIT,
    GEMINI_INPUT_COST_PER_1M,
    GEMINI_OUTPUT_COST_PER_1M,
)

logger = get_logger(__name__)

PROMPT_PATH = "prompts/who_and_why_prompt.yaml"


class NameExtractor:
    """Two-stage name extraction:
      1. NER (dslim/bert-base-NER) finds person mentions per grouped row.
      2. An async, cost-aware Gemini agent assigns each (text, person) a role and
         context_of_mention, written one row per person into the person_bio table.
    """

    def __init__(self, db_manager, df):
        self.db = db_manager
        self.df = df

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL")

        prompt = load_prompt(PROMPT_PATH)
        self.who_why_prompt = prompt["prompt"]

        self.semaphore = asyncio.Semaphore(NAME_EXTRACTION_CONCURRENCY)
        self.limiter = None  # AsyncRateLimiter, created per run inside the event loop

        # NER model loaded lazily on first use
        self.device = torch.device(
            NER_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._tokenizer = None
        self._ner_model = None
        self._id2label = {}

        # Cost / progress accumulators
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.processed = 0      # groups marked NAMED
        self.skipped = 0        # groups with empty text / no persons
        self.failed = 0         # groups marked FAILED_N
        self.blocked = 0        # groups safety-blocked → NAMED w/ NULL role/context
        self.persons_found = 0

    # ------------------------------------------------------------------ #
    # Entrypoint
    # ------------------------------------------------------------------ #
    def extract(self):
        """Sync entrypoint: NER → who/why → persist, then log the cost breakdown."""
        rows = self.df if isinstance(self.df, list) else self.df.to_dict("records")
        if not rows:
            logger.info("No rows to extract names from. Skipping.")
            return

        logger.info(f"Starting name extraction on {len(rows)} rows "
                    f"(NER device={self.device}, concurrency={NAME_EXTRACTION_CONCURRENCY})")

        # Stage 1: NER over every row's punctuated text
        entries = self._run_ner(rows)

        # Stage 2: who/why for rows that contain person mentions
        results, failed_groups, blocked_groups = {}, set(), set()
        if entries:
            chunks = [entries[i:i + WHO_AND_WHY_CHUNK_SIZE]
                      for i in range(0, len(entries), WHO_AND_WHY_CHUNK_SIZE)]
            self.limiter = AsyncRateLimiter(GEMINI_RPM_LIMIT)
            results, failed_groups, blocked_groups = asyncio.run(self._run_all(chunks))

        # Stage 3: persist person rows + update grouped statuses
        self._persist(rows, entries, results, failed_groups, blocked_groups)

        self._log_cost_summary(len(rows))

    # ------------------------------------------------------------------ #
    # Stage 1: NER
    # ------------------------------------------------------------------ #
    def _load_ner_model(self):
        if self._ner_model is not None:
            return
        logger.info(f"Loading NER model '{NER_MODEL_NAME}'...")
        self._tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
        self._ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
        self._ner_model.eval().to(self.device)
        raw_id2label = getattr(self._ner_model.config, "id2label", {}) or {}
        self._id2label = {int(k): v for k, v in raw_id2label.items()}
        logger.info(f"NER model loaded. id2label={self._id2label}")

    def _label_from_idx(self, idx: int) -> str:
        return self._id2label.get(int(idx), "O")

    def _run_ner(self, rows) -> list:
        """Returns a flat list of {group_id, program_id, broadcast_time, plain_text, person}."""
        self._load_ner_model()

        texts = [(r.get("plain_text") or "") for r in rows]
        names_per_row = self._extract_names(texts)

        entries = []
        for row, names in zip(rows, names_per_row):
            if not names:
                continue
            for person in names:
                entries.append({
                    "group_id": row["group_id"],
                    "program_id": row.get("program_id"),
                    "broadcast_time": row.get("broadcast_time"),
                    "plain_text": row.get("plain_text") or "",
                    "person": person,
                })
        self.persons_found = len(entries)
        logger.info(f"NER complete: {self.persons_found} person mention(s) across "
                    f"{len({e['group_id'] for e in entries})} group(s)")
        return entries

    def _extract_names(self, texts) -> list:
        """Batched token-classification → list (per text) of unique person names."""
        from tqdm import tqdm
        all_names = []
        num_batches = (len(texts) + NER_BATCH_SIZE - 1) // NER_BATCH_SIZE
        for i in tqdm(range(0, len(texts), NER_BATCH_SIZE), total=num_batches, desc="NER batches", unit="batch"):
            batch = texts[i:i + NER_BATCH_SIZE]
            all_names.extend(self._predict_batch(batch))
        return all_names

    def _predict_batch(self, texts) -> list:
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=NER_MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._ner_model(**inputs)

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        input_ids = inputs["input_ids"].cpu().numpy()
        attention_mask = inputs["attention_mask"].cpu().numpy()
        special_ids = set(self._tokenizer.all_special_ids or [])

        results = []
        for i in range(len(texts)):
            tokens = self._tokenizer.convert_ids_to_tokens(input_ids[i])
            pred_labels = [self._label_from_idx(p) for p in preds[i]]
            mask = attention_mask[i]

            person_spans = []
            current = None
            prev_was_per = False   # True after any B/I-PER word token (not ##)

            for tok_id, tok, lab, attn in zip(input_ids[i], tokens, pred_labels, mask):
                if attn == 0 or int(tok_id) in special_ids:
                    if current is not None:
                        person_spans.append(current.strip())
                        current = None
                    prev_was_per = False
                    continue

                # Subword continuations (##...) always belong to the preceding token's
                # word. The NER model labels them O even when the root token is B/I-PER,
                # so we must attach them regardless of their label.
                # They do NOT change prev_was_per — they are not word boundaries.
                if tok.startswith("##"):
                    if current is not None:
                        current += tok[2:]
                    continue

                if lab.startswith("B-PER"):
                    if current is not None and prev_was_per:
                        # No O-labeled word between this span and the previous one:
                        # treat as a continuation (first + last name of the same person).
                        # Risk: two distinct names with no separator merge into one.
                        # Mitigated by the fact that separators ("and", ",", verbs) are
                        # nearly always present between different people in broadcast text.
                        current += " " + tok
                    else:
                        if current is not None:
                            person_spans.append(current.strip())
                        current = tok
                    prev_was_per = True
                elif lab.startswith("I-PER") and current is not None:
                    current += " " + tok
                    prev_was_per = True
                else:
                    if current is not None:
                        person_spans.append(current.strip())
                        current = None
                    prev_was_per = False

            if current is not None:
                person_spans.append(current.strip())

            cleaned = []
            seen = set()
            for name in person_spans:
                nm = name.replace(" ##", "").strip()
                nm = nm.replace(" ,", ",").replace(" .", ".")
                if nm and nm not in seen:
                    seen.add(nm)
                    cleaned.append(nm)
            results.append(cleaned)

        return results

    # ------------------------------------------------------------------ #
    # Stage 2: who/why (async Gemini)
    # ------------------------------------------------------------------ #
    async def _run_all(self, chunks):
        results = {}            # (group_id, person_lower) -> (role, context)
        failed_groups = set()   # group_ids whose chunk transiently failed (retryable)
        blocked_groups = set()  # group_ids whose text the core filter rejected (terminal)

        tasks = [asyncio.create_task(self._process_chunk(chunk)) for chunk in chunks]
        try:
            outcomes = await gather_with_progress(tasks, desc="Chunks", unit="chk")
        finally:
            await close_async_client(self.client, logger)
        for chunk, outcome in zip(chunks, outcomes):
            if isinstance(outcome, Exception) or outcome is None:
                failed_groups.update(e["group_id"] for e in chunk)
                continue
            results.update(outcome["results"])
            failed_groups.update(outcome["failed"])
            blocked_groups.update(outcome["blocked"])
        # A group resolved by any chunk (or terminally blocked) shouldn't also count
        # as transiently failed.
        resolved_groups = {group_id for (group_id, _person) in results}
        failed_groups -= (resolved_groups | blocked_groups)
        return results, failed_groups, blocked_groups

    async def _process_chunk(self, chunk):
        async with self.semaphore:
            return await self._classify_entries(chunk, allow_isolation=True)

    async def _classify_entries(self, chunk, allow_isolation: bool):
        """Resolve role/context for a batch of person-entries in one LLM call.

        Returns {"results": {...}, "failed": set(), "blocked": set()} — or None when the
        whole batch failed transiently (retryable). On a safety block of a multi-entry
        batch, re-resolve each entry individually so only the genuinely blocked group(s)
        are degraded (NAMED with NULL role/context) while the rest get real verdicts.
        """
        prompt = self._build_chunk_prompt(chunk)
        try:
            text, in_tokens, out_tokens = await self._call_llm_with_retry(prompt)
            self.total_input_tokens += in_tokens
            self.total_output_tokens += out_tokens
            return {"results": self._parse_response(text), "failed": set(), "blocked": set()}
        except SafetyBlockedError as e:
            if allow_isolation and len(chunk) > 1:
                logger.info(f"Who/Why batch safety-blocked ({len(chunk)} entries); "
                            f"isolating the offending group(s) one entry at a time.")
                merged = {"results": {}, "failed": set(), "blocked": set()}
                for entry in chunk:
                    outcome = await self._classify_entries([entry], allow_isolation=False)
                    if outcome is None:
                        merged["failed"].add(entry["group_id"])
                        continue
                    merged["results"].update(outcome["results"])
                    merged["failed"] |= outcome["failed"]
                    merged["blocked"] |= outcome["blocked"]
                return merged
            # Single entry (or already isolated) is the genuine offender.
            logger.warning(f"Who/Why safety-blocked for group_id={chunk[0]['group_id']}; "
                           f"keeping name(s) with NULL role/context ({e}).")
            return {"results": {}, "failed": set(),
                    "blocked": {e2["group_id"] for e2 in chunk}}
        except Exception as e:
            logger.warning(f"Who/Why chunk failed permanently "
                           f"({len(chunk)} entries): {e}")
            return None

    def _build_chunk_prompt(self, chunk) -> str:
        entries = []
        for idx, e in enumerate(chunk, 1):
            entries.append(
                f"\n--------\nEntry {idx}\nGroup ID: {e['group_id']}\n"
                f"Person: {e['person']}\nText: {e['plain_text'].strip()}\n"
            )
        return f"{''.join(entries)}\n\n{self.who_why_prompt}"

    def _parse_response(self, text) -> dict:
        """Parse Gemini JSON → {(group_id, person_lower): (role, context)}."""
        mapping = {}
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Skipping invalid JSON from Who/Why response: {e}")
            return mapping

        for item in data.get("results", []) or []:
            # The prompt asks for "Group ID"; accept "Text ID" too for robustness.
            group_id = item.get("Group ID") or item.get("Text ID")
            person = item.get("Person")
            if not group_id or not person:
                continue
            mapping[(group_id, person.lower().strip())] = (
                item.get("Role"),
                item.get("Context_of_mention"),
            )

        if (data.get("results") or []) and not mapping:
            logger.warning("Who/Why response contained results but none were usable — "
                           f"possible output-key mismatch. First item keys: "
                           f"{list((data['results'][0] or {}).keys())}")
        return mapping

    async def _call_llm_with_retry(self, prompt):
        async def make_call():
            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    safety_settings=PIPELINE_SAFETY_SETTINGS,
                ),
            )
            text = validate_response(resp)
            usage = resp.usage_metadata
            return text, usage.prompt_token_count or 0, usage.candidates_token_count or 0

        return await async_call_with_retry(
            make_call,
            limiter=self.limiter,
            logger=logger,
            backoff_base=NAME_EXTRACTION_BACKOFF_BASE,
            backoff_cap=NAME_EXTRACTION_BACKOFF_CAP,
            transient_max_retries=NAME_EXTRACTION_MAX_RETRIES,
        )

    # ------------------------------------------------------------------ #
    # Stage 3: persist
    # ------------------------------------------------------------------ #
    def _persist(self, rows, entries, results, failed_groups, blocked_groups):
        person_rows = []
        for e in entries:
            # Blocked groups never received an LLM verdict → keep the BERT-extracted
            # person but leave role/context NULL.
            role, context = results.get((e["group_id"], e["person"].lower().strip()),
                                        (None, None))
            person_rows.append({
                "row_id": str(uuid.uuid4()),
                "group_id": e["group_id"],
                "program_id": e["program_id"],
                "broadcast_time": e["broadcast_time"],
                "plain_text": e["plain_text"],
                "person": e["person"],
                "role": role,
                "context_of_mention": context,
            })

        if person_rows:
            self.db.insert_person_bio(person_rows)

        groups_with_persons = {e["group_id"] for e in entries}
        for row in rows:
            group_id = row["group_id"]
            if group_id in failed_groups:
                self.db.update_status("grouped", group_id, RowStatus.FAILED_N)
                self.failed += 1
            elif group_id in blocked_groups:
                # Terminal: names kept (NULL role/context), advance so reruns skip it.
                self.db.update_status("grouped", group_id, RowStatus.NAMED)
                self.db.set_block_status("grouped", group_id, BlockStatus.NAMES)
                self.blocked += 1
                if group_id not in groups_with_persons:
                    self.skipped += 1
            else:
                self.db.update_status("grouped", group_id, RowStatus.NAMED)
                self.processed += 1
                if group_id not in groups_with_persons:
                    self.skipped += 1

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def _log_cost_summary(self, total_rows):
        input_cost = self.total_input_tokens / 1_000_000 * GEMINI_INPUT_COST_PER_1M
        output_cost = self.total_output_tokens / 1_000_000 * GEMINI_OUTPUT_COST_PER_1M
        total_cost = input_cost + output_cost
        avg_cost = total_cost / self.processed if self.processed else 0.0

        logger.info("=" * 60)
        logger.info("NAME EXTRACTION - COST BREAKDOWN")
        logger.info("=" * 60)
        logger.info(f"Rows total:           {total_rows}")
        logger.info(f"  named:              {self.processed}")
        logger.info(f"  safety-blocked:     {self.blocked} (NULL role/context)")
        logger.info(f"  (no persons):       {self.skipped}")
        logger.info(f"  failed:             {self.failed}")
        logger.info(f"Persons extracted:    {self.persons_found}")
        logger.info(f"Input tokens:         {self.total_input_tokens:,}")
        logger.info(f"Output tokens:        {self.total_output_tokens:,}")
        logger.info(f"Input cost:           ${input_cost:.6f}")
        logger.info(f"Output cost:          ${output_cost:.6f}")
        logger.info(f"TOTAL COST:           ${total_cost:.6f}")
        logger.info(f"Avg cost/named row:   ${avg_cost:.6f}")
        logger.info("=" * 60)
