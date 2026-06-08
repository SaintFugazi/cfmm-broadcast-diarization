import os
import asyncio
import random

from google import genai
from google.genai import types

from utils.progress import gather_with_progress

from utils.logger import get_logger
from utils.prompt_loader import load_prompt
from config.constants import RowStatus
from config.constants import (
    PUNCTUATION_CONCURRENCY,
    PUNCTUATION_MAX_RETRIES,
    PUNCTUATION_BACKOFF_BASE,
    PUNCTUATION_BACKOFF_CAP,
    GEMINI_INPUT_COST_PER_1M,
    GEMINI_OUTPUT_COST_PER_1M,
)

logger = get_logger(__name__)

PROMPT_PATH = "prompts/punctuation_restore_prompt.yaml"

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

        # Cost / progress accumulators
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.processed = 0
        self.skipped = 0
        self.failed = 0

    def punctuate(self):
        """Sync entrypoint: runs the async batch then logs the cost breakdown"""
        rows = self.df if isinstance(self.df, list) else self.df.to_dict("records")
        if not rows:
            logger.info("No rows to punctuate. Skipping.")
            return

        logger.info(f"Starting punctuation restoration on {len(rows)} rows "
                    f"(concurrency={PUNCTUATION_CONCURRENCY})")
        asyncio.run(self._run_all(rows))
        self._log_cost_summary(len(rows))

    async def _run_all(self, rows):
        tasks = [asyncio.create_task(self._process_row(row)) for row in rows]
        await gather_with_progress(tasks, desc="Rows", unit="row")

    async def _process_row(self, row):
        async with self.semaphore:
            group_id = row["group_id"]
            raw_text = row.get("plain_text") or ""

            if not raw_text.strip():
                self.skipped += 1
                return

            try:
                text, in_tokens, out_tokens = await self._call_llm_with_retry(raw_text)
                self.total_input_tokens += in_tokens
                self.total_output_tokens += out_tokens
                self.db.update_punctuated(group_id, text)
                self.db.update_status("grouped", group_id, RowStatus.PUNCTUATED)
                self.processed += 1
            except Exception as e:
                self.failed += 1
                self.db.update_status("grouped", group_id, RowStatus.FAILED)
                logger.warning(f"Failed to punctuate group_id={group_id}: {e}")


    async def _call_llm_with_retry(self, raw_text):
        last_exc = None
        for attempt in range(PUNCTUATION_MAX_RETRIES):
            try:
                resp = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=self.user_prompt.format(raw_transcript=raw_text),
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt
                    ),
                )
                usage = resp.usage_metadata
                in_tokens = usage.prompt_token_count or 0
                out_tokens = usage.candidates_token_count or 0
                return resp.text, in_tokens, out_tokens
            except Exception as e:
                last_exc = e
                sleep_s = min(PUNCTUATION_BACKOFF_BASE ** attempt, PUNCTUATION_BACKOFF_CAP)
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
        logger.info(f"  failed:          {self.failed}")
        logger.info(f"Input tokens:      {self.total_input_tokens:,}")
        logger.info(f"Output tokens:     {self.total_output_tokens:,}")
        logger.info(f"Input cost:        ${input_cost:.6f}")
        logger.info(f"Output cost:       ${output_cost:.6f}")
        logger.info(f"TOTAL COST:        ${total_cost:.6f}")
        logger.info(f"Avg cost/row:      ${avg_cost:.6f}")
        logger.info("=" * 60)
