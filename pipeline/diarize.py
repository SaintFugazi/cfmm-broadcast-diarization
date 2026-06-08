import os
import re
import json
import uuid
import random
import asyncio

from google import genai
from google.genai import types

from utils.progress import gather_with_progress

from utils.logger import get_logger
from utils.prompt_loader import load_prompt
from config.constants import RowStatus
from config.constants import (
    DIARIZATION_CONCURRENCY,
    DIARIZATION_MAX_RETRIES,
    DIARIZATION_BACKOFF_BASE,
    DIARIZATION_BACKOFF_CAP,
    DIARIZATION_CACHE_TTL,
    GEMINI_INPUT_COST_PER_1M,
    GEMINI_OUTPUT_COST_PER_1M,
    GEMINI_CACHE_INPUT_DISCOUNT,
)

logger = get_logger(__name__)

PROMPT_PATH = "prompts/group_diarization_prompt.yaml"


class DiarizationAgent:
    """Whole-group diarization with an async, cost-aware Gemini agent.

    For each NAMED group, the agent feeds the entire `plain_text` (one broadcast minute)
    plus the group's person_bio candidate speakers to Gemini, which splits the passage into
    consecutive speaker turns. Each turn is written to the `dialogues` table; groups advance
    to DIARIZED (or FAILED_D on permanent failure).
    """

    def __init__(self, db_manager, df):
        self.db = db_manager
        self.df = df

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL")

        prompt = load_prompt(PROMPT_PATH)
        self.system_instruction = prompt["system_prompt"]
        self.user_template = prompt["user_prompt"]

        self.semaphore = asyncio.Semaphore(DIARIZATION_CONCURRENCY)

        # Explicit context cache for the (large, fixed) system instruction.
        self.cache_name = None

        # Cost / progress accumulators
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0  # input tokens served from cache (billed at a discount)
        self.processed = 0          # groups marked DIARIZED
        self.failed = 0             # groups marked FAILED_D
        self.segments_total = 0     # speaker turns written to `dialogues`

    # ------------------------------------------------------------------ #
    # Entrypoint
    # ------------------------------------------------------------------ #
    def diarize(self):
        """Sync entrypoint: diarize every group concurrently, then log the cost breakdown."""
        rows = self.df if isinstance(self.df, list) else self.df.to_dict("records")
        if not rows:
            logger.info("No groups to diarize. Skipping.")
            return

        logger.info(f"Starting group diarization on {len(rows)} group(s) "
                    f"(concurrency={DIARIZATION_CONCURRENCY})")

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
                    ttl=DIARIZATION_CACHE_TTL,
                ),
            )
            self.cache_name = cache.name
            cached = cache.usage_metadata.total_token_count
            logger.info(f"Context cache created ({cached} tokens, ttl={DIARIZATION_CACHE_TTL}): "
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
    # Async LLM diarization
    # ------------------------------------------------------------------ #
    async def _run_all(self, rows):
        tasks = [asyncio.create_task(self._process_group(row)) for row in rows]
        await gather_with_progress(tasks, desc="Groups", unit="grp")

    async def _process_group(self, row):
        async with self.semaphore:
            group_id = row["group_id"]
            people = self.db.get_person_bio_by_group_id(group_id)
            user_prompt = self.user_template.format(
                people_data=self._build_people_data(people),
                full_text=row.get("plain_text") or "",
            )

            try:
                text, in_tokens, out_tokens, cached_tokens = await self._call_llm_with_retry(user_prompt)
                self.total_input_tokens += in_tokens
                self.total_output_tokens += out_tokens
                self.total_cached_tokens += cached_tokens

                segments = self._parse_response(text)
                self._persist_segments(row, segments, people)
                self.segments_total += len(segments)

                self.db.update_status("grouped", group_id, RowStatus.DIARIZED)
                self.processed += 1
            except Exception as e:
                self.failed += 1
                self.db.update_status("grouped", group_id, RowStatus.FAILED_D)
                logger.warning(f"Failed to diarize group_id={group_id}: {e}")

    @staticmethod
    def _build_people_data(rows) -> str:
        """Render person_bio rows as a newline list of candidate speakers."""
        if not rows:
            return "(no people extracted for this group)"
        lines = []
        for r in rows:
            person = (r.get("person") or "").strip()
            role = (r.get("role") or "Unknown").strip()
            context = (r.get("context_of_mention") or "Unknown").strip()
            lines.append(f"- Person: {person} | Role: {role} | Context: {context}")
        return "\n".join(lines)

    def _parse_response(self, text) -> list:
        """Parse Gemini JSON → list of segment dicts. Returns [] on bad output."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Invalid JSON from diarization response: {e}")
            return []
        return data.get("segments", []) or []

    # Diarizer fallback speakers are not in person_bio; map them to their canonical role.
    _FALLBACK_ROLES = {
        "news anchor": "Anchor",
        "interviewee": "Interviewee",
        "guest": "Guest",
        "reporter": "Reporter",
    }

    @classmethod
    def _resolve_role(cls, speaker, people) -> str:
        """Look up a segment speaker's role from person_bio (no LLM).

        Strips a trailing ' (Clip)' marker, matches the speaker (case-insensitively) against the
        group's person_bio names, and returns that person's role. Falls back to the canonical role
        for the diarizer's fallback labels (News Anchor/Interviewee/...). Returns None when nothing
        matches.
        """
        if not speaker:
            return None
        name = re.sub(r'\s*\(Clip\)\s*$', '', str(speaker), flags=re.IGNORECASE).strip()
        key = name.lower()

        for r in people or []:
            person = (r.get("person") or "").strip()
            if person and person.lower() == key:
                role = (r.get("role") or "").strip()
                return role or None

        return cls._FALLBACK_ROLES.get(key)

    def _persist_segments(self, row, segments, people) -> None:
        dialogue_rows = []
        for seg in segments:
            dialogue = (seg.get("text") or "").strip()
            if not dialogue:
                continue

            speaker = seg.get("speaker")

            try:
                confidence = float(seg.get("confidence"))
                confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]
            except (TypeError, ValueError):
                confidence = None

            # Backstop: if the assigned speaker's name appears in the segment text
            # ONLY as a vocative (direct address), cap confidence at 0.5.
            # This catches cases where the LLM attributed a turn to an addressee
            # ("Don't rush me, Jeff.") rather than the actual speaker.
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
                "dialogue_id": str(uuid.uuid4()),
                "group_id": row["group_id"],
                "program_id": row.get("program_id"),
                "broadcast_time": row.get("broadcast_time"),
                "dialogue": dialogue,
                "speaker": speaker,
                "role": self._resolve_role(speaker, people),
                "confidence_score": confidence,
            })

        if dialogue_rows:
            self.db.insert_dialogues(dialogue_rows)

    @staticmethod
    def _is_only_vocative(speaker: str, text: str) -> bool:
        """Return True if every occurrence of `speaker` in `text` is in vocative
        (direct-address) position — i.e. the name is being spoken TO, not spoken BY.

        Vocative patterns recognised:
          - Comma before + sentence-ender/comma after:  "..., Jeff."  "..., Jeff,"
          - Sentence-start + comma after:               "Jeff, don't..."
          - Mid-sentence surrounded by commas:          "Well, Jeff, I think..."

        Returns False if the name does not appear in the text, or if ANY occurrence
        is in a non-vocative position (subject, object, referent, etc.).
        """
        # Word-boundary search so "Jeff" doesn't match "Jefferson"
        pattern = re.compile(r'\b' + re.escape(speaker) + r'\b', re.IGNORECASE)
        matches = list(pattern.finditer(text))

        if not matches:
            return False  # Name absent — no vocative issue to flag

        for m in matches:
            before = text[:m.start()].rstrip()
            after  = text[m.end():].lstrip()

            before_comma    = before.endswith(',')
            after_end       = (not after) or after[0] in '.!?,'
            at_sent_start   = (not before) or (before[-1] in '.!?')
            after_comma     = after.startswith(',')

            # Pattern A: "..., Name."  /  "..., Name,"
            # Pattern B: "Name,"  at the start of a sentence
            is_vocative = (before_comma and after_end) or (at_sent_start and after_comma)

            if not is_vocative:
                return False  # At least one non-vocative use — don't cap

        return True  # Every occurrence is a direct address

    async def _call_llm_with_retry(self, user_prompt):
        last_exc = None
        for attempt in range(DIARIZATION_MAX_RETRIES):
            try:
                # Reference the cached system instruction when available; otherwise inline it.
                # (cached_content and system_instruction are mutually exclusive.)
                if self.cache_name:
                    config = types.GenerateContentConfig(
                        cached_content=self.cache_name,
                        temperature=0,
                        response_mime_type="application/json",
                    )
                else:
                    config = types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        temperature=0,
                        response_mime_type="application/json",
                    )

                resp = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=config,
                )
                usage = resp.usage_metadata
                in_tokens = usage.prompt_token_count or 0
                out_tokens = usage.candidates_token_count or 0
                cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
                return resp.text, in_tokens, out_tokens, cached_tokens
            except Exception as e:
                last_exc = e
                sleep_s = min(DIARIZATION_BACKOFF_BASE ** attempt, DIARIZATION_BACKOFF_CAP)
                sleep_s += random.uniform(0, 1)  # jitter
                if self._is_rate_limit(e):
                    logger.info(f"Rate limit hit (attempt {attempt + 1}); "
                                f"backing off {sleep_s:.1f}s")
                else:
                    logger.info(f"Error (attempt {attempt + 1}): {e}; "
                                f"retrying in {sleep_s:.1f}s")
                await asyncio.sleep(sleep_s)

        raise last_exc

    @staticmethod
    def _is_rate_limit(exc) -> bool:
        msg = str(exc).lower()
        return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def _log_cost_summary(self, total_rows):
        # Cached input tokens are billed at a discount; the rest at the full input rate.
        full_input_tokens = max(0, self.total_input_tokens - self.total_cached_tokens)
        input_cost = (
            full_input_tokens / 1_000_000 * GEMINI_INPUT_COST_PER_1M
            + self.total_cached_tokens / 1_000_000 * GEMINI_INPUT_COST_PER_1M * GEMINI_CACHE_INPUT_DISCOUNT
        )
        output_cost = self.total_output_tokens / 1_000_000 * GEMINI_OUTPUT_COST_PER_1M
        total_cost = input_cost + output_cost
        avg_cost = total_cost / self.processed if self.processed else 0.0

        logger.info("=" * 60)
        logger.info("GROUP DIARIZATION - COST BREAKDOWN")
        logger.info("=" * 60)
        logger.info(f"Groups total:          {total_rows}")
        logger.info(f"  diarized:            {self.processed}")
        logger.info(f"  failed:              {self.failed}")
        logger.info(f"Speaker turns written: {self.segments_total}")
        logger.info(f"Input tokens:          {self.total_input_tokens:,}")
        logger.info(f"  of which cached:     {self.total_cached_tokens:,} (billed @ {GEMINI_CACHE_INPUT_DISCOUNT:.0%})")
        logger.info(f"Output tokens:         {self.total_output_tokens:,}")
        logger.info(f"Input cost:            ${input_cost:.6f}")
        logger.info(f"Output cost:           ${output_cost:.6f}")
        logger.info(f"TOTAL COST:            ${total_cost:.6f}")
        logger.info(f"Avg cost/diarized:     ${avg_cost:.6f}")
        logger.info("=" * 60)
