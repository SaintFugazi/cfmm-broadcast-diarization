import os
import re
import json
import uuid
import asyncio

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
    SEGMENTATION_CONCURRENCY,
    SEGMENTATION_MAX_RETRIES,
    SEGMENTATION_BACKOFF_BASE,
    SEGMENTATION_BACKOFF_CAP,
    SEGMENTATION_CACHE_TTL,
    GEMINI_RPM_LIMIT,
    GEMINI_INPUT_COST_PER_1M,
    GEMINI_OUTPUT_COST_PER_1M,
    GEMINI_CACHE_INPUT_DISCOUNT,
)

logger = get_logger(__name__)

# Sentence-level split: after ./!/? followed by whitespace (indexed agent).
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

# API-enforced output shape for the indexed (OVER) agent: unit ranges only, no text.
_INDEXED_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "segments": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "start": types.Schema(type=types.Type.INTEGER),
                    "end": types.Schema(type=types.Type.INTEGER),
                },
                required=["start", "end"],
            ),
        ),
    },
    required=["segments"],
)


class SegmentationAgent:
    """Step 6 — split a group's transcript into speaker-change segments (NO attribution).

    For each PUNCTUATED group (count = UNDER, or unclassified), the agent feeds the entire
    `plain_text` to Gemini, which returns the text split into consecutive verbatim segments at
    each speaker change. Segments are written to the `segments` table (one row per segment,
    ordered); the group advances to SEGMENTED (or FAILED_S on permanent failure). Speaker
    attribution is a later stage.

    Subclasses (IndexedSegmentationAgent for OVER-sized transcripts) override PROMPT_PATH /
    STAGE_LABEL and the per-group processing to use unit-index ranges instead of echoed text.
    """

    PROMPT_PATH = "prompts/segmentation_prompt.yaml"
    STAGE_LABEL = "SEGMENTATION"

    def __init__(self, db_manager, df):
        self.db = db_manager
        self.df = df

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL")

        prompt = load_prompt(self.PROMPT_PATH)
        self.system_instruction = prompt["system_prompt"]
        self.user_template = prompt["user_prompt"]

        self.semaphore = asyncio.Semaphore(SEGMENTATION_CONCURRENCY)
        self.limiter = None  # AsyncRateLimiter, created per run inside the event loop

        # Explicit context cache for the (large, fixed) system instruction.
        self.cache_name = None

        # Cost / progress accumulators
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0  # input tokens served from cache (billed at a discount)
        self.processed = 0          # groups marked SEGMENTED
        self.failed = 0             # groups marked FAILED_S
        self.blocked = 0            # groups safety-blocked → single whole-text segment
        self.segments_total = 0     # segments written to `segments`

    # ------------------------------------------------------------------ #
    # Entrypoint
    # ------------------------------------------------------------------ #
    def segment(self):
        """Sync entrypoint: segment every group concurrently, then log the cost breakdown."""
        rows = self.df if isinstance(self.df, list) else self.df.to_dict("records")
        if not rows:
            logger.info("No groups to segment. Skipping.")
            return

        logger.info(f"Starting segmentation on {len(rows)} group(s) "
                    f"(concurrency={SEGMENTATION_CONCURRENCY}, rpm={GEMINI_RPM_LIMIT})")

        self.limiter = AsyncRateLimiter(GEMINI_RPM_LIMIT)
        self._setup_cache()
        try:
            asyncio.run(self._run_all(rows))
        finally:
            self._teardown_cache()
        self._log_cost_summary(len(rows))

    # ------------------------------------------------------------------ #
    # Context cache lifecycle
    # ------------------------------------------------------------------ #
    def _setup_cache(self):
        """Cache the fixed system instruction once so it isn't re-sent on every group call.

        Falls back silently to inline system instruction if caching is unavailable
        (e.g. the instruction is below the model's minimum cacheable size).
        """
        try:
            cache = self.client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    system_instruction=self.system_instruction,
                    ttl=SEGMENTATION_CACHE_TTL,
                ),
            )
            self.cache_name = cache.name
            cached = cache.usage_metadata.total_token_count
            logger.info(f"Context cache created ({cached} tokens, ttl={SEGMENTATION_CACHE_TTL}): "
                        f"{self.cache_name}")
        except Exception as e:
            self.cache_name = None
            logger.info(f"Context caching unavailable; sending system instruction inline. ({e})")

    def _teardown_cache(self):
        """Delete the context cache so we don't keep paying storage after the run."""
        if not self.cache_name:
            return
        try:
            self.client.caches.delete(name=self.cache_name)
            logger.info(f"Context cache deleted: {self.cache_name}")
        except Exception as e:
            logger.warning(f"Failed to delete context cache {self.cache_name}: {e}")
        finally:
            self.cache_name = None

    # ------------------------------------------------------------------ #
    # Async LLM segmentation
    # ------------------------------------------------------------------ #
    async def _run_all(self, rows):
        tasks = [asyncio.create_task(self._process_group(row)) for row in rows]
        try:
            await gather_with_progress(tasks, desc="Groups", unit="grp")
        finally:
            await close_async_client(self.client, logger)

    async def _process_group(self, row):
        async with self.semaphore:
            group_id = row["group_id"]
            user_prompt = self.user_template.format(full_text=row.get("plain_text") or "")

            try:
                text, in_tokens, out_tokens, cached_tokens = await self._call_llm_with_retry(user_prompt)
                self.total_input_tokens += in_tokens
                self.total_output_tokens += out_tokens
                self.total_cached_tokens += cached_tokens

                texts = self._parse_response(text)
                self._persist_segments(row, texts)
                self.segments_total += len(texts)

                self.db.update_status("grouped", group_id, RowStatus.SEGMENTED)
                self.processed += 1
            except SafetyBlockedError as e:
                self._handle_safety_block(row, e)
            except Exception as e:
                self.failed += 1
                self.db.update_status("grouped", group_id, RowStatus.FAILED_S)
                logger.warning(f"Failed to segment group_id={group_id}: {e}")

    def _handle_safety_block(self, row, exc):
        """Core filter rejected the transcript: the LLM cannot segment it. Persist the whole
        text as a single segment (block_status set) so attribution still has something to work
        with, then advance to SEGMENTED (terminal — reruns skip it)."""
        group_id = row["group_id"]
        whole = (row.get("plain_text") or "").strip()
        self._persist_segments(row, [whole] if whole else [])
        self.segments_total += 1 if whole else 0
        self.db.set_block_status("grouped", group_id, BlockStatus.SEGMENTATION)
        self.db.update_status("grouped", group_id, RowStatus.SEGMENTED)
        self.blocked += 1
        logger.warning(f"Segmentation safety-blocked for group_id={group_id}; "
                       f"wrote single whole-text segment and continued ({exc}).")

    def _parse_response(self, text) -> list:
        """Parse Gemini JSON → list of verbatim segment strings. Returns [] on bad output."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Invalid JSON from segmentation response: {e}")
            return []
        out = []
        for seg in data.get("segments", []) or []:
            t = (seg.get("text") or "").strip()
            if t:
                out.append(t)
        return out

    def _persist_segments(self, row, texts) -> None:
        """Write the ordered segment texts to the `segments` table (deterministic ids)."""
        group_id = row["group_id"]
        segment_rows = []
        for i, t in enumerate(texts):
            segment_rows.append({
                "segment_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{group_id}|{i}")),
                "group_id": group_id,
                "program_id": row.get("program_id"),
                "broadcast_time": row.get("broadcast_time"),
                "order_index": i,
                "text": t,
            })
        if segment_rows:
            self.db.insert_segments(segment_rows)

    def _config_kwargs(self) -> dict:
        """GenerateContentConfig kwargs; references the cached system instruction when
        available, otherwise inlines it (cached_content and system_instruction are
        mutually exclusive). Subclasses extend this (e.g. to add a response_schema)."""
        kwargs = dict(
            temperature=0,
            response_mime_type="application/json",
            safety_settings=PIPELINE_SAFETY_SETTINGS,
        )
        if self.cache_name:
            kwargs["cached_content"] = self.cache_name
        else:
            kwargs["system_instruction"] = self.system_instruction
        return kwargs

    async def _call_llm_with_retry(self, user_prompt):
        async def make_call():
            config = types.GenerateContentConfig(**self._config_kwargs())

            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )
            text = validate_response(resp)
            usage = resp.usage_metadata
            cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
            return (text, usage.prompt_token_count or 0,
                    usage.candidates_token_count or 0, cached_tokens)

        return await async_call_with_retry(
            make_call,
            limiter=self.limiter,
            logger=logger,
            backoff_base=SEGMENTATION_BACKOFF_BASE,
            backoff_cap=SEGMENTATION_BACKOFF_CAP,
            transient_max_retries=SEGMENTATION_MAX_RETRIES,
        )

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def _log_cost_summary(self, total_rows):
        full_input_tokens = max(0, self.total_input_tokens - self.total_cached_tokens)
        input_cost = (
            full_input_tokens / 1_000_000 * GEMINI_INPUT_COST_PER_1M
            + self.total_cached_tokens / 1_000_000 * GEMINI_INPUT_COST_PER_1M * GEMINI_CACHE_INPUT_DISCOUNT
        )
        output_cost = self.total_output_tokens / 1_000_000 * GEMINI_OUTPUT_COST_PER_1M
        total_cost = input_cost + output_cost
        avg_cost = total_cost / self.processed if self.processed else 0.0

        logger.info("=" * 60)
        logger.info(f"{self.STAGE_LABEL} - COST BREAKDOWN")
        logger.info("=" * 60)
        logger.info(f"Groups total:          {total_rows}")
        logger.info(f"  segmented:           {self.processed}")
        logger.info(f"  safety-blocked:      {self.blocked} (single whole-text segment)")
        logger.info(f"  failed:              {self.failed}")
        logger.info(f"Segments written:      {self.segments_total}")
        logger.info(f"Input tokens:          {self.total_input_tokens:,}")
        logger.info(f"  of which cached:     {self.total_cached_tokens:,} (billed @ {GEMINI_CACHE_INPUT_DISCOUNT:.0%})")
        logger.info(f"Output tokens:         {self.total_output_tokens:,}")
        logger.info(f"Input cost:            ${input_cost:.6f}")
        logger.info(f"Output cost:           ${output_cost:.6f}")
        logger.info(f"TOTAL COST:            ${total_cost:.6f}")
        logger.info(f"Avg cost/segmented:    ${avg_cost:.6f}")
        logger.info("=" * 60)


class IndexedSegmentationAgent(SegmentationAgent):
    """Segmentation for OVER-sized transcripts (grouped."count" = 'OVER').

    The standard agent asks the model to re-emit every segment's text verbatim, so its output
    scales with transcript length and long rows hit the generation ceiling. This agent instead
    pre-splits the transcript into numbered sentence-level units, shows the model the WHOLE
    numbered transcript (full context, no chunking), and has it return unit index RANGES per
    segment. Segment text is reconstructed from the ranges locally, so:
      - output size is O(number of segments), independent of transcript length;
      - verbatim fidelity is guaranteed (the model never rewrites text);
      - full coverage is verifiable — ranges must tile units 1..N exactly, and violations are
        repaired or rejected rather than silently corrupting the segments table.
    """

    PROMPT_PATH = "prompts/segmentation_indexed_prompt.yaml"
    STAGE_LABEL = "INDEXED SEGMENTATION (OVER rows)"

    def _config_kwargs(self) -> dict:
        kwargs = super()._config_kwargs()
        kwargs["response_schema"] = _INDEXED_RESPONSE_SCHEMA
        return kwargs

    async def _process_group(self, row):
        async with self.semaphore:
            group_id = row["group_id"]
            units = self._split_into_units(row.get("plain_text") or "")
            user_prompt = self.user_template.format(
                unit_count=len(units),
                numbered_transcript=self._build_numbered_block(units),
            )

            try:
                text, in_tokens, out_tokens, cached_tokens = await self._call_llm_with_retry(user_prompt)
                self.total_input_tokens += in_tokens
                self.total_output_tokens += out_tokens
                self.total_cached_tokens += cached_tokens

                ranges = self._parse_ranges(text)
                ranges = self._validate_and_repair(ranges, len(units), group_id)
                texts = self._reconstruct_segments(ranges, units)
                self._persist_segments(row, texts)
                self.segments_total += len(texts)

                self.db.update_status("grouped", group_id, RowStatus.SEGMENTED)
                self.processed += 1
            except SafetyBlockedError as e:
                self._handle_safety_block(row, e)
            except Exception as e:
                self.failed += 1
                self.db.update_status("grouped", group_id, RowStatus.FAILED_S)
                logger.warning(f"Failed to segment (indexed) group_id={group_id}: {e}")

    # ------------------------------------------------------------------ #
    # Unit splitting / numbering
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_into_units(text: str) -> list:
        """Split a transcript into sentence-level units, preserving order and content.

        Splits on ' >> ' speaker markers first so each marker starts a fresh unit (the
        '>> ' prefix is kept — it is a segmentation cue), then on sentence boundaries within
        each part. Reconstruction joins units with single spaces, so units are stored
        whitespace-stripped.
        """
        units = []
        for i, part in enumerate(text.split(" >> ")):
            part = part.strip()
            if not part:
                continue
            if i > 0 and not part.startswith(">>"):
                part = ">> " + part
            for sentence in _SENTENCE_SPLIT.split(part):
                sentence = sentence.strip()
                if sentence:
                    units.append(sentence)
        return units or [text.strip()]

    @staticmethod
    def _build_numbered_block(units: list) -> str:
        return "\n".join(f"[{i}] {unit}" for i, unit in enumerate(units, start=1))

    # ------------------------------------------------------------------ #
    # Range parsing / coverage validation / reconstruction
    # ------------------------------------------------------------------ #
    def _parse_ranges(self, text) -> list:
        """Parse Gemini JSON → list of raw range dicts. Raises on unusable output."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid JSON from indexed segmentation response: {e}")
        segments = data.get("segments", []) or []
        if not segments:
            raise ValueError("Indexed segmentation response contained no segments")
        return segments

    @staticmethod
    def _validate_and_repair(ranges, n_units: int, group_id: str) -> list:
        """Enforce the coverage contract: segments must tile units 1..n_units exactly.

        Minor violations are repaired deterministically (clamping out-of-range indices,
        extending a segment backwards over a gap, trimming overlaps, extending the last
        segment to N). Every repair is logged. If nothing usable remains, raises so the
        group is retried/failed rather than persisted incomplete.
        """
        cleaned = []
        for r in ranges:
            try:
                start, end = int(r.get("start")), int(r.get("end"))
            except (TypeError, ValueError):
                logger.warning(f"group_id={group_id}: dropping segment with non-integer "
                               f"range: {r!r}")
                continue
            if end < start:
                start, end = end, start
            start = max(1, min(start, n_units))
            end = max(1, min(end, n_units))
            cleaned.append({"start": start, "end": end})

        if not cleaned:
            raise ValueError("No valid segment ranges in indexed segmentation response")

        cleaned.sort(key=lambda s: (s["start"], s["end"]))

        repaired = []
        cursor = 1  # next unit number that still needs an owner
        for seg in cleaned:
            if seg["end"] < cursor:
                logger.warning(f"group_id={group_id}: dropping fully-overlapped segment "
                               f"[{seg['start']}-{seg['end']}]")
                continue
            if seg["start"] != cursor:
                logger.warning(f"group_id={group_id}: repairing segment start "
                               f"{seg['start']} -> {cursor} (gap/overlap)")
                seg["start"] = cursor
            repaired.append(seg)
            cursor = seg["end"] + 1

        if cursor <= n_units:
            logger.warning(f"group_id={group_id}: extending last segment end "
                           f"{repaired[-1]['end']} -> {n_units} (trailing gap)")
            repaired[-1]["end"] = n_units

        return repaired

    @staticmethod
    def _reconstruct_segments(ranges, units) -> list:
        """Materialize each range into its verbatim text span."""
        texts = []
        for seg in ranges:
            text = " ".join(units[seg["start"] - 1:seg["end"]])
            # Strip the leading '>>' segmentation marker — it is not spoken dialogue.
            text = re.sub(r'^>>\s*', '', text).strip()
            if text:
                texts.append(text)
        return texts
