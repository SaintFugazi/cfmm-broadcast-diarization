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
from config.constants import RowStatus, BlockStatus, COUNT_OVER
from config.constants import (
    ATTRIBUTION_CONCURRENCY,
    ATTRIBUTION_MAX_RETRIES,
    ATTRIBUTION_BACKOFF_BASE,
    ATTRIBUTION_BACKOFF_CAP,
    ATTRIBUTION_CACHE_TTL,
    ATTRIBUTION_OVER_CHUNK_SEGMENTS,
    GEMINI_RPM_LIMIT,
    GEMINI_INPUT_COST_PER_1M,
    GEMINI_OUTPUT_COST_PER_1M,
    GEMINI_CACHE_INPUT_DISCOUNT,
)

logger = get_logger(__name__)

PROMPT_PATH = "prompts/attribution_prompt.yaml"

# API-enforced output shape: one object per resulting turn.
_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "turns": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "speaker": types.Schema(type=types.Type.STRING),
                    "role": types.Schema(type=types.Type.STRING, nullable=True),
                    "speaker_type": types.Schema(type=types.Type.STRING),
                    "text": types.Schema(type=types.Type.STRING),
                    "confidence": types.Schema(type=types.Type.NUMBER),
                    "note": types.Schema(type=types.Type.STRING, nullable=True),
                },
                required=["speaker", "speaker_type", "text", "confidence"],
            ),
        ),
    },
    required=["turns"],
)


class AttributionAgent:
    """Step 7 — attribute a group's segments to speakers, clean the text, score confidence.

    For each SEGMENTED group, the agent reads the ordered `segments` (Step 6 output), renders
    them as "[broadcast_time] <text>" lines, and asks Gemini to (a) assign each turn an actual
    speaker name or role, (b) clean/format the dialogue, and (c) emit a per-turn confidence.
    Each resulting turn is written to the `dialogues` table; the group advances to ATTRIBUTED
    (or FAILED_A on permanent failure).

    OVER-sized groups are attributed in chunks of ATTRIBUTION_OVER_CHUNK_SEGMENTS so the
    cleaned-text output stays under the model's generation ceiling; turns are concatenated
    across chunks with a continuous order_index.
    """

    def __init__(self, db_manager, df):
        self.db = db_manager
        self.df = df

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL")

        prompt = load_prompt(PROMPT_PATH)
        self.system_instruction = prompt["system_prompt"]
        self.user_template = prompt["user_prompt"]

        self.semaphore = asyncio.Semaphore(ATTRIBUTION_CONCURRENCY)
        self.limiter = None  # AsyncRateLimiter, created per run inside the event loop

        # Explicit context cache for the (large, fixed) system instruction.
        self.cache_name = None

        # Cost / progress accumulators
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0  # input tokens served from cache (billed at a discount)
        self.processed = 0          # groups marked ATTRIBUTED
        self.failed = 0             # groups marked FAILED_A
        self.blocked = 0            # groups safety-blocked → single UNKNOWN turn
        self.turns_total = 0        # speaker turns written to `dialogues`

    # ------------------------------------------------------------------ #
    # Entrypoint
    # ------------------------------------------------------------------ #
    def attribute(self):
        """Sync entrypoint: attribute every group concurrently, then log the cost breakdown."""
        rows = self.df if isinstance(self.df, list) else self.df.to_dict("records")
        if not rows:
            logger.info("No groups to attribute. Skipping.")
            return

        logger.info(f"Starting attribution on {len(rows)} group(s) "
                    f"(concurrency={ATTRIBUTION_CONCURRENCY}, rpm={GEMINI_RPM_LIMIT})")

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
                    ttl=ATTRIBUTION_CACHE_TTL,
                ),
            )
            self.cache_name = cache.name
            cached = cache.usage_metadata.total_token_count
            logger.info(f"Context cache created ({cached} tokens, ttl={ATTRIBUTION_CACHE_TTL}): "
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
    # Async LLM attribution
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
            segments = self.db.get_segments_by_group_id(group_id)
            if not segments:
                # Nothing to attribute (e.g. empty transcript) — advance cleanly.
                self.db.update_status("grouped", group_id, RowStatus.ATTRIBUTED)
                self.processed += 1
                return

            broadcast_time = row.get("broadcast_time")
            # OVER groups: chunk segments so the cleaned-text OUTPUT stays under the ceiling.
            if row.get("count") == COUNT_OVER:
                chunk_size = ATTRIBUTION_OVER_CHUNK_SEGMENTS
            else:
                chunk_size = len(segments)
            chunks = [segments[i:i + chunk_size]
                      for i in range(0, len(segments), chunk_size)]

            try:
                all_turns = []
                for chunk in chunks:
                    user_prompt = self.user_template.format(
                        segments=self._build_segments_block(chunk, broadcast_time)
                    )
                    text, in_t, out_t, cached_t = await self._call_llm_with_retry(user_prompt)
                    self.total_input_tokens += in_t
                    self.total_output_tokens += out_t
                    self.total_cached_tokens += cached_t
                    all_turns.extend(self._parse_response(text))

                self._persist_turns(row, all_turns)
                self.turns_total += len(all_turns)

                self.db.update_status("grouped", group_id, RowStatus.ATTRIBUTED)
                self.processed += 1
            except SafetyBlockedError as e:
                self._handle_safety_block(row, segments, e)
            except Exception as e:
                self.failed += 1
                self.db.update_status("grouped", group_id, RowStatus.FAILED_A)
                logger.warning(f"Failed to attribute group_id={group_id}: {e}")

    def _handle_safety_block(self, row, segments, exc):
        """Core filter rejected the content: the LLM cannot attribute it. Emit a single
        UNKNOWN-speaker turn carrying the whole text (confidence 0, block_status set) so the
        content still reaches the export, then advance to ATTRIBUTED (terminal — reruns skip
        it, and the block flag keeps it out of verification)."""
        group_id = row["group_id"]
        dialogue = " ".join((s.get("text") or "").strip() for s in segments).strip()
        if dialogue:
            self.db.insert_dialogues([{
                "dialogue_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"blocked-attribute::{group_id}")),
                "group_id": group_id,
                "program_id": row.get("program_id"),
                "broadcast_time": row.get("broadcast_time"),
                "dialogue": dialogue,
                "speaker": "UNKNOWN",
                "role": None,
                "confidence_score": 0.0,
                "block_status": BlockStatus.ATTRIBUTION,
                "order_index": 0,
                "speaker_type": None,
                "note": None,
            }])
            self.turns_total += 1
        self.db.set_block_status("grouped", group_id, BlockStatus.ATTRIBUTION)
        self.db.update_status("grouped", group_id, RowStatus.ATTRIBUTED)
        self.blocked += 1
        logger.warning(f"Attribution safety-blocked for group_id={group_id}; "
                       f"wrote single UNKNOWN-speaker turn and continued ({exc}).")

    @staticmethod
    def _build_segments_block(segments, broadcast_time) -> str:
        """Render the ordered segments as '[broadcast_time] <verbatim text>' lines."""
        prefix = f"[{broadcast_time}] " if broadcast_time else ""
        return "\n".join(f"{prefix}{(s.get('text') or '').strip()}" for s in segments)

    def _parse_response(self, text) -> list:
        """Parse Gemini JSON → list of turn dicts. Returns [] on bad output."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Invalid JSON from attribution response: {e}")
            return []
        return data.get("turns", []) or []

    def _persist_turns(self, row, turns) -> None:
        group_id = row["group_id"]
        dialogue_rows = []
        order = 0
        for turn in turns:
            dialogue = (turn.get("text") or "").strip()
            if not dialogue:
                continue

            speaker = turn.get("speaker") or None
            role = turn.get("role") or None
            speaker_type = turn.get("speaker_type") or None
            note = turn.get("note") or None

            try:
                confidence = float(turn.get("confidence"))
                confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]
            except (TypeError, ValueError):
                confidence = None

            # Backstop: if the assigned speaker's name appears in the turn text ONLY as a
            # vocative (direct address), cap confidence at 0.5 — the addressee is not the speaker.
            if (
                speaker
                and confidence is not None
                and confidence > 0.5
                and self._is_only_vocative(speaker, dialogue)
            ):
                logger.debug(
                    f"Vocative backstop triggered for speaker={speaker!r}; "
                    f"capping confidence {confidence:.2f} -> 0.50"
                )
                confidence = 0.5

            dialogue_rows.append({
                "dialogue_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{group_id}|{order}")),
                "group_id": group_id,
                "program_id": row.get("program_id"),
                "broadcast_time": row.get("broadcast_time"),
                "dialogue": dialogue,
                "speaker": speaker,
                "role": role,
                "confidence_score": confidence,
                "order_index": order,
                "speaker_type": speaker_type,
                "note": note,
            })
            order += 1

        if dialogue_rows:
            self.db.insert_dialogues(dialogue_rows)

    @staticmethod
    def _is_only_vocative(speaker: str, text: str) -> bool:
        """Return True if every occurrence of `speaker` in `text` is in vocative
        (direct-address) position — i.e. the name is being spoken TO, not spoken BY.
        Returns False if the name is absent or any occurrence is non-vocative.
        """
        pattern = re.compile(r'\b' + re.escape(speaker) + r'\b', re.IGNORECASE)
        matches = list(pattern.finditer(text))
        if not matches:
            return False

        for m in matches:
            before = text[:m.start()].rstrip()
            after = text[m.end():].lstrip()

            before_comma = before.endswith(',')
            after_end = (not after) or after[0] in '.!?,'
            at_sent_start = (not before) or (before[-1] in '.!?')
            after_comma = after.startswith(',')

            is_vocative = (before_comma and after_end) or (at_sent_start and after_comma)
            if not is_vocative:
                return False

        return True

    def _config_kwargs(self) -> dict:
        """GenerateContentConfig kwargs; references the cached system instruction when
        available, otherwise inlines it (cached_content and system_instruction are
        mutually exclusive)."""
        kwargs = dict(
            temperature=0,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
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
            backoff_base=ATTRIBUTION_BACKOFF_BASE,
            backoff_cap=ATTRIBUTION_BACKOFF_CAP,
            transient_max_retries=ATTRIBUTION_MAX_RETRIES,
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
        logger.info("ATTRIBUTION - COST BREAKDOWN")
        logger.info("=" * 60)
        logger.info(f"Groups total:          {total_rows}")
        logger.info(f"  attributed:          {self.processed}")
        logger.info(f"  safety-blocked:      {self.blocked} (UNKNOWN-speaker turn)")
        logger.info(f"  failed:              {self.failed}")
        logger.info(f"Speaker turns written: {self.turns_total}")
        logger.info(f"Input tokens:          {self.total_input_tokens:,}")
        logger.info(f"  of which cached:     {self.total_cached_tokens:,} (billed @ {GEMINI_CACHE_INPUT_DISCOUNT:.0%})")
        logger.info(f"Output tokens:         {self.total_output_tokens:,}")
        logger.info(f"Input cost:            ${input_cost:.6f}")
        logger.info(f"Output cost:           ${output_cost:.6f}")
        logger.info(f"TOTAL COST:            ${total_cost:.6f}")
        logger.info(f"Avg cost/attributed:   ${avg_cost:.6f}")
        logger.info("=" * 60)
