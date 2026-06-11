import os
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

from utils.local_punctuation import restore_punctuation
from utils.logger import get_logger
from utils.prompt_loader import load_prompt
from config.constants import RowStatus, BlockStatus, PunctuationSource
from config.constants import (
    PUNCTUATION_CONCURRENCY,
    PUNCTUATION_MAX_RETRIES,
    PUNCTUATION_BACKOFF_BASE,
    PUNCTUATION_BACKOFF_CAP,
    PUNCTUATION_MAX_CHUNK_CHARS,
    GEMINI_RPM_LIMIT,
    GEMINI_INPUT_COST_PER_1M,
    GEMINI_OUTPUT_COST_PER_1M,
)

logger = get_logger(__name__)

PROMPT_PATH = "prompts/punctuation_restore_prompt.yaml"

# If the punctuated output is shorter than this fraction of the input, something
# went wrong (truncation, hallucination) — fail the row rather than overwrite plain_text.
_MIN_OUTPUT_RATIO = 0.4


class PunctuationRestoration:
    """Restores punctuation of raw transcripts using an async, cost-aware LLM agent"""

    def __init__(self, db_manager, df):
        self.db = db_manager
        self.df = df
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL")

        prompt = load_prompt(PROMPT_PATH)
        self.system_prompt = prompt["system_prompt"]
        self.user_prompt = prompt["user_prompt"]

        self.semaphore = asyncio.Semaphore(PUNCTUATION_CONCURRENCY)
        self.limiter = None  # AsyncRateLimiter, created per run inside the event loop

        # Cost / progress accumulators
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.processed = 0
        self.skipped = 0
        self.failed = 0
        self.blocked = 0   # rows safety-blocked → local/raw punctuation fallback

    def punctuate(self):
        """Sync entrypoint: runs the async batch then logs the cost breakdown"""
        rows = self.df if isinstance(self.df, list) else self.df.to_dict("records")
        if not rows:
            logger.info("No rows to punctuate. Skipping.")
            return

        logger.info(f"Starting punctuation restoration on {len(rows)} rows "
                    f"(concurrency={PUNCTUATION_CONCURRENCY}, rpm={GEMINI_RPM_LIMIT})")
        self.limiter = AsyncRateLimiter(GEMINI_RPM_LIMIT)
        asyncio.run(self._run_all(rows))
        self._log_cost_summary(len(rows))

    async def _run_all(self, rows):
        tasks = [asyncio.create_task(self._process_row(row)) for row in rows]
        try:
            await gather_with_progress(tasks, desc="Rows", unit="row")
        finally:
            await close_async_client(self.client, logger)

    async def _process_row(self, row):
        async with self.semaphore:
            group_id = row["group_id"]
            raw_text = row.get("plain_text") or ""

            if not raw_text.strip():
                self.skipped += 1
                return

            try:
                if len(raw_text) > PUNCTUATION_MAX_CHUNK_CHARS:
                    text, in_tokens, out_tokens = await self._punctuate_chunked(raw_text)
                else:
                    text, in_tokens, out_tokens = await self._call_llm_with_retry(raw_text)
                self.total_input_tokens += in_tokens
                self.total_output_tokens += out_tokens
                self.db.update_punctuated(group_id, text, source=PunctuationSource.LLM)
                self.db.update_status("grouped", group_id, RowStatus.PUNCTUATED)
                self.processed += 1
            except SafetyBlockedError as e:
                # Non-configurable core filter rejected this content; the LLM can never
                # punctuate it. Fall back to a local (no-filter) model so the group still
                # advances, and flag it as degraded so reruns skip it (terminal).
                self._degrade_with_local(group_id, raw_text, e)
            except Exception as e:
                self.failed += 1
                self.db.update_status("grouped", group_id, RowStatus.FAILED_P)
                logger.warning(f"Failed to punctuate group_id={group_id}: {e}")

    def _degrade_with_local(self, group_id, raw_text, exc):
        """Safety-blocked row: restore punctuation locally (or keep raw if that fails),
        mark PUNCTUATED + block_status so the pipeline continues without retrying."""
        restored, ok = restore_punctuation(raw_text)
        source = PunctuationSource.LOCAL if ok else PunctuationSource.RAW
        self.db.update_punctuated(group_id, restored, source=source)
        self.db.set_block_status("grouped", group_id, BlockStatus.PUNCTUATION)
        self.db.update_status("grouped", group_id, RowStatus.PUNCTUATED)
        self.blocked += 1
        logger.warning(f"Punctuation safety-blocked for group_id={group_id}; "
                       f"used {source} fallback and continued ({exc}).")

    # ------------------------------------------------------------------ #
    # Chunked punctuation for oversized transcripts
    # ------------------------------------------------------------------ #

    async def _punctuate_chunked(self, raw_text: str):
        """Split a long transcript on >> speaker boundaries, punctuate each piece,
        then rejoin. Holds the semaphore for the entire duration so the slot isn't
        stolen mid-row; each chunk still goes through the rate limiter.
        """
        chunks = self._split_for_chunking(raw_text, PUNCTUATION_MAX_CHUNK_CHARS)
        logger.info(f"Long transcript ({len(raw_text)} chars) split into "
                    f"{len(chunks)} chunk(s) for punctuation")

        total_in = total_out = 0
        punctuated_parts = []

        for chunk in chunks:
            text, in_t, out_t = await self._call_llm_with_retry(chunk)
            punctuated_parts.append(text)
            total_in += in_t
            total_out += out_t

        return " >> ".join(punctuated_parts), total_in, total_out

    @staticmethod
    def _split_for_chunking(text: str, max_chars: int) -> list:
        """Split on ' >> ' speaker markers into groups of at most max_chars each."""
        segments = text.split(" >> ")
        chunks = []
        current = []
        current_len = 0

        for seg in segments:
            seg_len = len(seg) + 4  # account for ' >> ' separator when rejoined
            if current and current_len + seg_len > max_chars:
                chunks.append(" >> ".join(current))
                current = [seg]
                current_len = seg_len
            else:
                current.append(seg)
                current_len += seg_len

        if current:
            chunks.append(" >> ".join(current))

        return chunks or [text]

    # ------------------------------------------------------------------ #
    # LLM call
    # ------------------------------------------------------------------ #

    async def _call_llm_with_retry(self, raw_text):
        async def make_call():
            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=self.user_prompt.format(raw_transcript=raw_text),
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    safety_settings=PIPELINE_SAFETY_SETTINGS,
                ),
            )
            text = validate_response(resp)
            if len(text.strip()) < len(raw_text.strip()) * _MIN_OUTPUT_RATIO:
                raise ValueError(
                    f"Punctuated output suspiciously short "
                    f"({len(text)} chars vs {len(raw_text)} input) — possible truncation"
                )
            usage = resp.usage_metadata
            return text, usage.prompt_token_count or 0, usage.candidates_token_count or 0

        return await async_call_with_retry(
            make_call,
            limiter=self.limiter,
            logger=logger,
            backoff_base=PUNCTUATION_BACKOFF_BASE,
            backoff_cap=PUNCTUATION_BACKOFF_CAP,
            transient_max_retries=PUNCTUATION_MAX_RETRIES,
        )

    def _cost(self, input_tokens, output_tokens) -> float:
        return (
            input_tokens / 1_000_000 * GEMINI_INPUT_COST_PER_1M
            + output_tokens / 1_000_000 * GEMINI_OUTPUT_COST_PER_1M
        )

    def _log_cost_summary(self, total_rows):
        input_cost = self.total_input_tokens / 1_000_000 * GEMINI_INPUT_COST_PER_1M
        output_cost = self.total_output_tokens / 1_000_000 * GEMINI_OUTPUT_COST_PER_1M
        total_cost = input_cost + output_cost
        avg_cost = total_cost / self.processed if self.processed else 0.0

        logger.info("=" * 60)
        logger.info("PUNCTUATION RESTORATION - COST BREAKDOWN")
        logger.info("=" * 60)
        logger.info(f"Rows total:        {total_rows}")
        logger.info(f"  processed:       {self.processed}")
        logger.info(f"  skipped (empty): {self.skipped}")
        logger.info(f"  safety-blocked:  {self.blocked} (local/raw fallback)")
        logger.info(f"  failed:          {self.failed}")
        logger.info(f"Input tokens:      {self.total_input_tokens:,}")
        logger.info(f"Output tokens:     {self.total_output_tokens:,}")
        logger.info(f"Input cost:        ${input_cost:.6f}")
        logger.info(f"Output cost:       ${output_cost:.6f}")
        logger.info(f"TOTAL COST:        ${total_cost:.6f}")
        logger.info(f"Avg cost/row:      ${avg_cost:.6f}")
        logger.info("=" * 60)
