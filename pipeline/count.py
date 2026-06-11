import os
import asyncio

from google import genai

from utils.progress import gather_with_progress
from utils.rate_limit import AsyncRateLimiter, async_call_with_retry, close_async_client

from utils.logger import get_logger
from config.constants import RowStatus
from config.constants import (
    COUNT_OVER,
    COUNT_UNDER,
    COUNT_CONCURRENCY,
    COUNT_MAX_RETRIES,
    COUNT_BACKOFF_BASE,
    COUNT_BACKOFF_CAP,
    DIARIZATION_TOKEN_THRESHOLD,
    GEMINI_RPM_LIMIT,
)

logger = get_logger(__name__)


class TokenCounter:
    """Classifies each PUNCTUATED group's transcript size for the diarization split.

    Counts plain_text tokens via the Gemini count_tokens endpoint and writes OVER/UNDER
    into grouped."count": UNDER rows are safe for the standard (text-echoing) diarizer;
    OVER rows must go through the indexed diarizer, whose output size does not scale with
    transcript length. count_tokens is free (no token billing) but still rate-limited, so
    calls share the same limiter/retry strategy as the LLM agents.

    On success the row's status is (re)set to PUNCTUATED so FAILED_C reruns rejoin the
    main flow; permanent failures are marked FAILED_C.
    """

    def __init__(self, db_manager, df):
        self.db = db_manager
        self.df = df

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL")

        self.semaphore = asyncio.Semaphore(COUNT_CONCURRENCY)
        self.limiter = None  # AsyncRateLimiter, created per run inside the event loop

        # count_tokens is free; these exist so the orchestrator's cost accumulator
        # can treat this agent like any other stage.
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        self.under = 0      # groups classified UNDER
        self.over = 0       # groups classified OVER
        self.skipped = 0    # groups with empty text (classified UNDER without an API call)
        self.failed = 0     # groups marked FAILED_C

    # ------------------------------------------------------------------ #
    # Entrypoint
    # ------------------------------------------------------------------ #
    def count(self):
        """Sync entrypoint: classify every group concurrently, then log the summary."""
        rows = self.df if isinstance(self.df, list) else self.df.to_dict("records")
        if not rows:
            logger.info("No groups to classify by size. Skipping.")
            return

        logger.info(f"Classifying transcript size on {len(rows)} group(s) "
                    f"(threshold={DIARIZATION_TOKEN_THRESHOLD} tokens, "
                    f"concurrency={COUNT_CONCURRENCY}, rpm={GEMINI_RPM_LIMIT})")

        self.limiter = AsyncRateLimiter(GEMINI_RPM_LIMIT)
        asyncio.run(self._run_all(rows))
        self._log_summary(len(rows))

    async def _run_all(self, rows):
        tasks = [asyncio.create_task(self._process_row(row)) for row in rows]
        try:
            await gather_with_progress(tasks, desc="Counting", unit="grp")
        finally:
            await close_async_client(self.client, logger)

    async def _process_row(self, row):
        async with self.semaphore:
            group_id = row["group_id"]
            text = row.get("plain_text") or ""

            if not text.strip():
                self.db.update_count(group_id, COUNT_UNDER)
                self.db.update_status("grouped", group_id, RowStatus.PUNCTUATED)
                self.skipped += 1
                return

            try:
                tokens = await self._count_with_retry(text)
                label = COUNT_OVER if tokens > DIARIZATION_TOKEN_THRESHOLD else COUNT_UNDER
                self.db.update_count(group_id, label)
                self.db.update_status("grouped", group_id, RowStatus.PUNCTUATED)
                if label == COUNT_OVER:
                    self.over += 1
                    logger.info(f"group_id={group_id}: {tokens} tokens -> {label}")
                else:
                    self.under += 1
            except Exception as e:
                self.failed += 1
                self.db.update_status("grouped", group_id, RowStatus.FAILED_C)
                logger.warning(f"Failed to count tokens for group_id={group_id}: {e}")

    async def _count_with_retry(self, text) -> int:
        async def make_call():
            resp = await self.client.aio.models.count_tokens(
                model=self.model,
                contents=text,
            )
            return resp.total_tokens or 0

        return await async_call_with_retry(
            make_call,
            limiter=self.limiter,
            logger=logger,
            backoff_base=COUNT_BACKOFF_BASE,
            backoff_cap=COUNT_BACKOFF_CAP,
            transient_max_retries=COUNT_MAX_RETRIES,
        )

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def _log_summary(self, total_rows):
        logger.info("=" * 60)
        logger.info("TRANSCRIPT SIZE CLASSIFICATION - SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Groups total:        {total_rows}")
        logger.info(f"  UNDER (standard):  {self.under}")
        logger.info(f"  OVER (indexed):    {self.over}")
        logger.info(f"  skipped (empty):   {self.skipped}")
        logger.info(f"  failed:            {self.failed}")
        logger.info(f"Threshold:           {DIARIZATION_TOKEN_THRESHOLD} tokens")
        logger.info(f"Cost:                $0.000000 (count_tokens is free)")
        logger.info("=" * 60)
