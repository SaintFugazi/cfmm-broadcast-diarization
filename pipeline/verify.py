import os
import re
import json
import asyncio
from collections import defaultdict

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
from config.constants import BlockStatus
from config.constants import (
    VERIFICATION_CONCURRENCY,
    VERIFICATION_MAX_RETRIES,
    VERIFICATION_BACKOFF_BASE,
    VERIFICATION_BACKOFF_CAP,
    VERIFICATION_CACHE_TTL,
    GEMINI_RPM_LIMIT,
    GEMINI_INPUT_COST_PER_1M,
    GEMINI_OUTPUT_COST_PER_1M,
    GEMINI_CACHE_INPUT_DISCOUNT,
)

logger = get_logger(__name__)

PROMPT_PATH = "prompts/verification_prompt.yaml"


class VerificationAgent:
    """Re-checks low-confidence diarized dialogues with an async, cost-aware Gemini agent.

    Uncertain dialogues (confidence < threshold, or NULL) are grouped by their broadcast minute
    (group_id) and verified ONE GROUP PER API CALL: the whole minute's context is sent once, and all
    of that minute's uncertain dialogues are reviewed together. This avoids re-sending the same
    (expensive) context for every dialogue and lets the model reason about turn-taking across the
    uncertain turns jointly. The agent confirms or corrects each speaker and writes a
    verification_score back to the `dialogues` table. High-confidence dialogues are never sent here,
    so their verification_score stays empty.
    """

    def __init__(self, db_manager, rows):
        self.db = db_manager
        self.rows = rows

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL")

        prompt = load_prompt(PROMPT_PATH)
        self.system_instruction = prompt["system_prompt"]
        self.user_template = prompt["user_prompt"]

        self.semaphore = asyncio.Semaphore(VERIFICATION_CONCURRENCY)
        self.limiter = None  # AsyncRateLimiter, created per run inside the event loop

        # Explicit context cache for the (large, fixed) system instruction.
        self.cache_name = None

        # Cost / progress accumulators
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0  # input tokens served from cache (billed at a discount)
        self.groups = 0         # broadcast minutes verified (one API call each)
        self.processed = 0      # dialogues verified (speaker confirmed or corrected)
        self.corrected = 0      # dialogues whose speaker changed
        self.blocked = 0        # dialogues safety-blocked → flagged, left unverified
        self.failed = 0         # dialogues that permanently failed

        # Per-group response log — flushed to JSONL at the end of verify()
        self._response_log = []

    # ------------------------------------------------------------------ #
    # Entrypoint
    # ------------------------------------------------------------------ #
    def verify(self):
        """Sync entrypoint: verify every low-confidence dialogue (batched by group), then log cost."""
        rows = self.rows if isinstance(self.rows, list) else self.rows.to_dict("records")
        if not rows:
            logger.info("No dialogues to verify. Skipping.")
            return

        # Group the uncertain dialogues by their broadcast minute (group_id).
        groups = defaultdict(list)
        for row in rows:
            groups[row["group_id"]].append(row)

        logger.info(f"Starting verification on {len(rows)} dialogue(s) across "
                    f"{len(groups)} group(s) (concurrency={VERIFICATION_CONCURRENCY}, "
                    f"rpm={GEMINI_RPM_LIMIT})")

        self.limiter = AsyncRateLimiter(GEMINI_RPM_LIMIT)
        self._setup_cache()
        try:
            asyncio.run(self._run_all(groups))
        finally:
            self._teardown_cache()
        self._flush_response_log()
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
                    ttl=VERIFICATION_CACHE_TTL,
                ),
            )
            self.cache_name = cache.name
            cached = cache.usage_metadata.total_token_count
            logger.info(f"Context cache created ({cached} tokens, ttl={VERIFICATION_CACHE_TTL}): "
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
    # Async LLM verification
    # ------------------------------------------------------------------ #
    async def _run_all(self, groups):
        tasks = [
            asyncio.create_task(self._process_group(group_id, group_rows))
            for group_id, group_rows in groups.items()
        ]
        try:
            await gather_with_progress(tasks, desc="Groups", unit="grp")
        finally:
            await close_async_client(self.client, logger)

    async def _process_group(self, group_id, group_rows):
        async with self.semaphore:
            # Stable index → row mapping so we can map verdicts back to dialogue_ids.
            indexed = list(enumerate(group_rows, start=1))
            full_text = group_rows[0].get("plain_text") or ""

            user_prompt = self.user_template.format(
                full_text=full_text,
                dialogues=self._build_dialogues_block(indexed),
            )

            try:
                text, in_tokens, out_tokens, cached_tokens = await self._call_llm_with_retry(user_prompt)
                self.total_input_tokens += in_tokens
                self.total_output_tokens += out_tokens
                self.total_cached_tokens += cached_tokens

                verdicts = self._parse_response(text)  # {index: {speaker, score, ...}}
                self._persist_verdicts(indexed, verdicts)
                self.groups += 1

                # Record this group's per-dialogue verdicts (incl. reasoning) for the .txt log.
                dialogues = []
                for idx, row in indexed:
                    current_speaker = row.get("speaker")
                    verdict = verdicts.get(idx) or {}
                    verified = verdict.get("speaker") or current_speaker
                    dialogues.append({
                        "index": idx,
                        "dialogue_id": row.get("dialogue_id"),
                        "current_speaker": current_speaker,
                        "dialogue": (row.get("dialogue") or "").strip(),
                        "verified_speaker": verified,
                        "verified_role": verdict.get("role"),
                        "prior_confidence": verdict.get("prior_confidence"),
                        "verification_score": verdict.get("score"),
                        "reasoning": verdict.get("reasoning") or "",
                        "has_verdict": idx in verdicts,
                    })
                self._response_log.append({
                    "group_id": group_id,
                    "dialogues": dialogues,
                })
            except SafetyBlockedError as e:
                # Core filter rejected this minute's context; verification is impossible.
                # Flag all of the group's dialogues as blocked so future runs exclude them
                # from the low-confidence re-check (otherwise they'd be re-sent and re-blocked).
                self.db.set_dialogues_block_status_by_group(group_id, BlockStatus.VERIFICATION)
                self.blocked += len(group_rows)
                logger.warning(f"Verification safety-blocked for group_id={group_id} "
                               f"({len(group_rows)} dialogue(s)); flagged, left unverified ({e}).")
            except Exception as e:
                self.failed += len(group_rows)
                logger.warning(f"Failed to verify group_id={group_id} "
                               f"({len(group_rows)} dialogue(s)): {e}")

    def _persist_verdicts(self, indexed, verdicts):
        """Apply the parsed verdicts to each dialogue in the group."""
        for idx, row in indexed:
            dialogue_id = row["dialogue_id"]
            current_speaker = row.get("speaker")
            dialogue = row.get("dialogue") or ""

            verdict = verdicts.get(idx)
            if verdict is None:
                # No verdict returned for this dialogue — keep current speaker, leave unverified.
                self.failed += 1
                logger.warning(f"No verdict returned for dialogue_id={dialogue_id}; "
                               f"leaving verification_score empty.")
                continue

            speaker = verdict["speaker"]
            role = verdict.get("role")
            score = verdict["score"]
            if not speaker or not str(speaker).strip():
                speaker = current_speaker
            else:
                speaker = str(speaker).strip()

            # Backstop: if the verified speaker's name appears in the dialogue ONLY as a
            # vocative (direct address), it cannot be the speaker — floor the score.
            if (
                speaker
                and score is not None
                and score > 0.5
                and self._is_only_vocative(speaker, dialogue)
            ):
                logger.debug(
                    f"Vocative backstop triggered for verified speaker={speaker!r}; "
                    f"capping verification_score {score:.2f} -> 0.50"
                )
                score = 0.5

            if speaker != current_speaker:
                # Speaker changed → also write the verifier-assigned role.
                self.db.update_dialogue_verification(dialogue_id, speaker, score, role=role)
                self.corrected += 1
            else:
                # Speaker unchanged → leave the role populated by diarization untouched.
                self.db.update_dialogue_verification(dialogue_id, speaker, score)
            self.processed += 1

    @staticmethod
    def _build_dialogues_block(indexed) -> str:
        """Render the numbered batch of dialogues under review."""
        blocks = []
        for idx, row in indexed:
            current_speaker = row.get("speaker")
            current_speaker = current_speaker if current_speaker not in (None, "") else "(none)"
            dialogue = (row.get("dialogue") or "").strip()
            blocks.append(
                f"[{idx}] CURRENTLY ASSIGNED SPEAKER: {current_speaker}\n"
                f"DIALOGUE: {dialogue}"
            )
        return "\n\n".join(blocks)

    def _parse_response(self, text) -> dict:
        """Parse Gemini JSON → {index: {speaker, score, prior_confidence, reasoning}}.

        Returns {} on bad output (callers then leave those dialogues unverified).
        """
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Invalid JSON from verification response: {e}.")
            return {}

        verdicts = {}
        for item in data.get("verifications", []) or []:
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue

            speaker = item.get("verified_speaker")
            speaker = str(speaker).strip() if speaker and str(speaker).strip() else None

            role = item.get("verified_role")
            role = str(role).strip() if role and str(role).strip() else None

            score = self._coerce_score(item.get("verification_score"))
            prior_confidence = self._coerce_score(item.get("prior_confidence"))

            reasoning = item.get("reasoning")
            reasoning = str(reasoning).strip() if reasoning is not None else ""

            verdicts[idx] = {
                "speaker": speaker,
                "role": role,
                "score": score,
                "prior_confidence": prior_confidence,
                "reasoning": reasoning,
            }

        return verdicts

    @staticmethod
    def _coerce_score(value):
        """Float in [0, 1], or None if missing/unparseable."""
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    async def _call_llm_with_retry(self, user_prompt):
        async def make_call():
            # Reference the cached system instruction when available; otherwise inline it.
            # (cached_content and system_instruction are mutually exclusive.)
            if self.cache_name:
                config = types.GenerateContentConfig(
                    cached_content=self.cache_name,
                    temperature=0,
                    response_mime_type="application/json",
                    safety_settings=PIPELINE_SAFETY_SETTINGS,
                )
            else:
                config = types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0,
                    response_mime_type="application/json",
                    safety_settings=PIPELINE_SAFETY_SETTINGS,
                )

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
            backoff_base=VERIFICATION_BACKOFF_BASE,
            backoff_cap=VERIFICATION_BACKOFF_CAP,
            transient_max_retries=VERIFICATION_MAX_RETRIES,
        )

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

    # ------------------------------------------------------------------ #
    # Response log
    # ------------------------------------------------------------------ #
    def _flush_response_log(self):
        """Write each group's verdicts + model reasoning to a human-readable .txt file."""
        if not self._response_log:
            return
        os.makedirs("data/output", exist_ok=True)
        out_path = "data/output/verification_responses.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for record in self._response_log:
                f.write("=" * 80 + "\n")
                f.write(f"GROUP: {record['group_id']}\n")
                f.write("=" * 80 + "\n\n")
                for d in record["dialogues"]:
                    f.write(self._format_dialogue_block(d))
                    f.write("\n")
                f.write("\n")
        logger.info(f"Verification responses written to {out_path} "
                    f"({len(self._response_log)} group(s))")

    @staticmethod
    def _format_dialogue_block(d) -> str:
        """Render one dialogue's verdict block for the .txt log."""
        if not d.get("has_verdict"):
            outcome = "NO VERDICT"
        elif d["verified_speaker"] != d["current_speaker"]:
            outcome = "CORRECTED"
        else:
            outcome = "confirmed"

        def fmt(score):
            return f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"

        reasoning = d.get("reasoning") or "(none provided)"
        role = d.get("verified_role") or "n/a"
        # Role is only written to the DB on a correction; flag display-only when confirmed.
        role_note = role if outcome == "CORRECTED" else f"{role} (not written; speaker unchanged)"
        return (
            f"[{d['index']}] dialogue_id={d.get('dialogue_id')}\n"
            f"    DIALOGUE:          {d.get('dialogue')}\n"
            f"    CURRENT SPEAKER:   {d.get('current_speaker')}\n"
            f"    VERIFIED SPEAKER:  {d.get('verified_speaker')}   ({outcome})\n"
            f"    VERIFIED ROLE:     {role_note}\n"
            f"    PRIOR CONFIDENCE:  {fmt(d.get('prior_confidence'))}\n"
            f"    VERIFICATION:      {fmt(d.get('verification_score'))}\n"
            f"    REASONING:         {reasoning}\n"
        )

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
        logger.info("SPEAKER VERIFICATION - COST BREAKDOWN")
        logger.info("=" * 60)
        logger.info(f"Dialogues total:       {total_rows}")
        logger.info(f"  groups (API calls):  {self.groups}")
        logger.info(f"  verified:            {self.processed}")
        logger.info(f"  speaker corrected:   {self.corrected}")
        logger.info(f"  safety-blocked:      {self.blocked} (left unverified)")
        logger.info(f"  failed:              {self.failed}")
        logger.info(f"Input tokens:          {self.total_input_tokens:,}")
        logger.info(f"  of which cached:     {self.total_cached_tokens:,} (billed @ {GEMINI_CACHE_INPUT_DISCOUNT:.0%})")
        logger.info(f"Output tokens:         {self.total_output_tokens:,}")
        logger.info(f"Input cost:            ${input_cost:.6f}")
        logger.info(f"Output cost:           ${output_cost:.6f}")
        logger.info(f"TOTAL COST:            ${total_cost:.6f}")
        logger.info(f"Avg cost/verified:     ${avg_cost:.6f}")
        logger.info("=" * 60)
