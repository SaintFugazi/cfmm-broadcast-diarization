import os
import json
import asyncio

from google import genai
from google.genai import types

from utils.logger import get_logger
from utils.prompt_loader import load_prompt
from utils.progress import gather_with_progress
from utils.rate_limit import (
    AsyncRateLimiter,
    async_call_with_retry,
    validate_response,
    close_async_client,
    PIPELINE_SAFETY_SETTINGS,
    SafetyBlockedError,
)
from config.constants import RowStatus, BlockStatus
from config.constants import (
    RELEVANCE_CONCURRENCY,
    RELEVANCE_MAX_RETRIES,
    RELEVANCE_BACKOFF_BASE,
    RELEVANCE_BACKOFF_CAP,
    RELEVANCE_CHUNK_SIZE,
    RELEVANCE_DELETE_THRESHOLD,
    RELEVANCE_CACHE_TTL,
    RELEVANCE_KEYWORDS,
    GEMINI_RPM_LIMIT,
    GEMINI_INPUT_COST_PER_1M,
    GEMINI_OUTPUT_COST_PER_1M,
    GEMINI_CACHE_INPUT_DISCOUNT,
)

logger = get_logger(__name__)

PROMPT_PATH = "prompts/relevance_filter_prompt.yaml"


class RelevanceAgent:
    """Filters non-news groups with an async, cost-aware Gemini agent.

    Runs on raw PENDING groups (before punctuation) so non-news content is dropped
    before any punctuation spend is incurred. Decides whether each group's dominant theme
    SUBSTANTIVELY discusses news/current-affairs (kept RELEVANT) or is non-news filler —
    commercials, promos, passing mentions, entertainment/sport/leisure (marked NOT_RELEVANT).

    Punctuation is not required for theme classification — the words themselves are sufficient.
    Groups are batched (RELEVANCE_CHUNK_SIZE per call) and the fixed system instruction is
    cached once per run.

    A group is only marked NOT_RELEVANT when the model is confident (>= RELEVANCE_DELETE_THRESHOLD);
    uncertain groups are kept (favor recall). Soft delete only — no rows are removed.
    """

    def __init__(self, db_manager, df):
        self.db = db_manager
        self.df = df

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL")

        prompt = load_prompt(PROMPT_PATH)
        # Inject the keyword list as topical CONTEXT for the model (not a deterministic override).
        # Use replace (not format): the system prompt contains literal JSON braces that would
        # break str.format.
        keywords_block = ", ".join(RELEVANCE_KEYWORDS)
        self.system_instruction = prompt["system_prompt"].replace("{keywords}", keywords_block)
        self.user_template = prompt["user_prompt"]

        self.semaphore = asyncio.Semaphore(RELEVANCE_CONCURRENCY)
        self.limiter = None  # AsyncRateLimiter, created per run inside the event loop

        # Explicit context cache for the (large, fixed) system instruction.
        self.cache_name = None

        # Cost / progress accumulators
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0  # input tokens served from cache (billed at a discount)
        self.relevant = 0             # groups marked RELEVANT (news)
        self.not_relevant = 0         # groups marked NOT_RELEVANT (confident non-news)
        self.kept_uncertain = 0       # non-news but low confidence → kept RELEVANT
        self.blocked = 0              # groups safety-blocked → kept RELEVANT + flagged
        self.failed = 0               # groups marked FAILED_R

    # ------------------------------------------------------------------ #
    # Entrypoint
    # ------------------------------------------------------------------ #
    def filter(self):
        """Sync entrypoint: batched LLM classification of PENDING groups → log cost breakdown."""
        rows = self.df if isinstance(self.df, list) else self.df.to_dict("records")
        if not rows:
            logger.info("No groups to filter for relevance. Skipping.")
            return

        logger.info(f"Relevance filter on {len(rows)} group(s) sent to LLM "
                    f"(concurrency={RELEVANCE_CONCURRENCY}, rpm={GEMINI_RPM_LIMIT})")

        chunks = [rows[i:i + RELEVANCE_CHUNK_SIZE]
                  for i in range(0, len(rows), RELEVANCE_CHUNK_SIZE)]
        self.limiter = AsyncRateLimiter(GEMINI_RPM_LIMIT)
        self._setup_cache()
        try:
            asyncio.run(self._run_all(chunks))
        finally:
            self._teardown_cache()

        self._log_cost_summary(len(rows))

    # ------------------------------------------------------------------ #
    # Context cache lifecycle
    # ------------------------------------------------------------------ #
    def _setup_cache(self):
        """Cache the fixed system instruction once so it isn't re-sent on every chunk call.

        Falls back silently to inline system instruction if caching is unavailable
        (e.g. the instruction is below the model's minimum cacheable size).
        """
        try:
            cache = self.client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    system_instruction=self.system_instruction,
                    ttl=RELEVANCE_CACHE_TTL,
                ),
            )
            self.cache_name = cache.name
            cached = cache.usage_metadata.total_token_count
            logger.info(f"Context cache created ({cached} tokens, ttl={RELEVANCE_CACHE_TTL}): "
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
    # Async LLM classification
    # ------------------------------------------------------------------ #
    async def _run_all(self, chunks):
        tasks = [asyncio.create_task(self._process_chunk(chunk)) for chunk in chunks]
        try:
            await gather_with_progress(tasks, desc="Chunks", unit="chk")
        finally:
            await close_async_client(self.client, logger)

    async def _process_chunk(self, chunk):
        async with self.semaphore:
            await self._classify(chunk, allow_isolation=True)

    async def _classify(self, chunk, allow_isolation: bool):
        """Classify a batch of groups in one LLM call.

        On a safety block of a multi-group batch, re-classify each group individually
        (allow_isolation) so a single offending transcript doesn't force the whole
        batch to be kept-and-flagged — only the genuinely blocked group(s) are degraded.
        """
        indexed = list(enumerate(chunk, start=1))
        try:
            user_prompt = self.user_template.replace("{entries}", self._build_entries(indexed))
            text, in_tokens, out_tokens, cached_tokens = await self._call_llm_with_retry(user_prompt)
            self.total_input_tokens += in_tokens
            self.total_output_tokens += out_tokens
            self.total_cached_tokens += cached_tokens

            verdicts = self._parse_response(text)  # {index: (is_relevant, confidence)}
            self._apply_verdicts(indexed, verdicts)
        except SafetyBlockedError as e:
            if allow_isolation and len(chunk) > 1:
                logger.info(f"Relevance batch safety-blocked ({len(chunk)} groups); "
                            f"isolating the offender(s) one group at a time.")
                for row in chunk:
                    await self._classify([row], allow_isolation=False)
                return
            # Single group (or already isolated) is the genuine offender. Never silently
            # drop: keep RELEVANT (favor recall — blocked content is often the most
            # newsworthy) and flag it so it advances terminally instead of retrying.
            for _, row in indexed:
                group_id = row["group_id"]
                self.db.update_status("grouped", group_id, RowStatus.RELEVANT)
                self.db.set_block_status("grouped", group_id, BlockStatus.RELEVANCE)
                self.relevant += 1
                self.blocked += 1
            logger.warning(f"Relevance safety-blocked for group_id="
                           f"{indexed[0][1]['group_id']}; kept RELEVANT, flagged ({e}).")
        except Exception as e:
            # Never mark NOT_RELEVANT on error — flag for retry instead.
            for _, row in indexed:
                self.db.update_status("grouped", row["group_id"], RowStatus.FAILED_R)
                self.failed += 1
            logger.warning(f"Relevance chunk failed permanently "
                           f"({len(chunk)} group(s)): {e}")

    def _apply_verdicts(self, indexed, verdicts):
        """Apply the per-group decision rule (LLM relevance → confidence gate)."""
        for idx, row in indexed:
            group_id = row["group_id"]

            verdict = verdicts.get(idx)
            if verdict is None:
                # No verdict returned for this index — retry it rather than guess.
                self.db.update_status("grouped", group_id, RowStatus.FAILED_R)
                self.failed += 1
                logger.warning(f"No relevance verdict for group_id={group_id}; marking FAILED_R.")
                continue

            is_relevant, confidence = verdict
            if is_relevant:
                self.db.update_status("grouped", group_id, RowStatus.RELEVANT)
                self.relevant += 1
            elif confidence is not None and confidence >= RELEVANCE_DELETE_THRESHOLD:
                self.db.update_status("grouped", group_id, RowStatus.NOT_RELEVANT)
                self.not_relevant += 1
            else:
                # Non-news but not confident enough — keep it (favor recall).
                self.db.update_status("grouped", group_id, RowStatus.RELEVANT)
                self.relevant += 1
                self.kept_uncertain += 1

    @staticmethod
    def _build_entries(indexed) -> str:
        """Render the numbered batch of minutes to classify."""
        blocks = []
        for idx, row in indexed:
            text = (row.get("plain_text") or "").strip()
            blocks.append(f"[{idx}] TEXT:\n{text}")
        return "\n\n".join(blocks)

    def _parse_response(self, text) -> dict:
        """Parse Gemini JSON → {index: (is_relevant, confidence)}. {} on bad output."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Invalid JSON from relevance response: {e}.")
            return {}

        verdicts = {}
        for item in data.get("results", []) or []:
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue

            is_relevant = bool(item.get("is_relevant"))

            try:
                confidence = float(item.get("confidence"))
                confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]
            except (TypeError, ValueError):
                confidence = None

            verdicts[idx] = (is_relevant, confidence)

        return verdicts

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
            backoff_base=RELEVANCE_BACKOFF_BASE,
            backoff_cap=RELEVANCE_BACKOFF_CAP,
            transient_max_retries=RELEVANCE_MAX_RETRIES,
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

        logger.info("=" * 60)
        logger.info("RELEVANCE FILTER - COST BREAKDOWN")
        logger.info("=" * 60)
        logger.info(f"Groups total:          {total_rows}")
        logger.info(f"  relevant (kept):     {self.relevant}")
        logger.info(f"    kept (uncertain):  {self.kept_uncertain}")
        logger.info(f"    kept (blocked):    {self.blocked}")
        logger.info(f"  not relevant:        {self.not_relevant}")
        logger.info(f"  failed:              {self.failed}")
        logger.info(f"Input tokens:          {self.total_input_tokens:,}")
        logger.info(f"  of which cached:     {self.total_cached_tokens:,} (billed @ {GEMINI_CACHE_INPUT_DISCOUNT:.0%})")
        logger.info(f"Output tokens:         {self.total_output_tokens:,}")
        logger.info(f"Input cost:            ${input_cost:.6f}")
        logger.info(f"Output cost:           ${output_cost:.6f}")
        logger.info(f"TOTAL COST:            ${total_cost:.6f}")
        logger.info("=" * 60)
