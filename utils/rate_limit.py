"""Shared rate-limiting, retry, safety, and response-validation strategy for every
Gemini agent in the pipeline.

Three layers, defined once here and reused by all agents:

  1. AsyncRateLimiter (proactive): paces requests to stay under a per-minute budget,
     so the quota is never exhausted in a burst.

  2. async_call_with_retry (reactive): wraps every call with a hard timeout, handles
     safety blocks as non-retryable, backs off >= 60s on 429s (honoring Gemini's own
     retryDelay hint), and uses full jitter to break the thundering herd.

  3. validate_response / PIPELINE_SAFETY_SETTINGS (defensive): relaxed safety thresholds
     for news-monitoring content and explicit validation of every response before it
     reaches the caller (catches safety blocks, MAX_TOKENS truncation, empty text).
"""

import re
import time
import random
import asyncio

from google.genai import types

from config.constants import (
    RATE_LIMIT_MIN_BACKOFF,
    RATE_LIMIT_MAX_RETRIES,
    REQUEST_TIMEOUT,
)

# Window (seconds) over which the RPM budget is enforced and the quota resets.
_WINDOW_SECONDS = 60.0

# ------------------------------------------------------------------ #
# Custom exceptions
# ------------------------------------------------------------------ #

class SafetyBlockedError(Exception):
    """Raised when Gemini blocks a response due to safety filters.
    Non-retryable: re-sending the same content will produce the same block.
    """

class MaxTokensError(Exception):
    """Raised when Gemini truncates a response at the token limit.
    Non-retryable with the same input: the caller should split the input instead.
    """

# ------------------------------------------------------------------ #
# Safety settings (applied to every agent call)
# ------------------------------------------------------------------ #

# This pipeline processes broadcast news transcripts that routinely cover topics
# (religion, politics, crime, conflict) that default safety thresholds block.
# The task is purely analytical (punctuation, relevance, diarization) over factual
# news reporting, so we set every *configurable* category to OFF — the most
# permissive threshold — to stop legitimate dark-but-newsworthy content from being
# blocked. Note: HARM_CATEGORY_CIVIC_INTEGRITY plus the non-configurable core
# filters (PROHIBITED_CONTENT, RECITATION) cannot be disabled via safety_settings.
PIPELINE_SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",         threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",        threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",  threshold="OFF"),
]

# ------------------------------------------------------------------ #
# Response validation
# ------------------------------------------------------------------ #

def validate_response(resp) -> str:
    """Validate a Gemini response and return its text.

    Raises SafetyBlockedError if the response was blocked by safety filters.
    Raises MaxTokensError if the response was cut off at the token limit.
    Raises SafetyBlockedError for any other empty/inaccessible response.
    """
    # A prompt rejected by the non-configurable core filters (e.g. child-safety)
    # comes back with no candidates; the reason lives in prompt_feedback, not in
    # any candidate finish_reason. Surface it explicitly so these are not mistaken
    # for a generic "empty response".
    block_reason = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
    if block_reason:
        raise SafetyBlockedError(
            f"Prompt blocked before generation (block_reason={block_reason})"
        )

    if resp.candidates:
        finish_reason = str(getattr(resp.candidates[0], "finish_reason", ""))
        if any(s in finish_reason for s in ("SAFETY", "PROHIBITED_CONTENT", "RECITATION")):
            raise SafetyBlockedError(
                f"Response blocked by safety filters (finish_reason={finish_reason})"
            )
        if "MAX_TOKENS" in finish_reason:
            raise MaxTokensError(
                f"Response truncated at token limit (finish_reason={finish_reason})"
            )

    try:
        text = resp.text
    except ValueError as e:
        raise SafetyBlockedError(f"Response text unavailable: {e}") from e

    if not text or not text.strip():
        raise SafetyBlockedError("Empty response text")

    # The model occasionally wraps JSON in a markdown code fence even when
    # response_mime_type="application/json" is set. Strip it so json.loads
    # never sees "Extra data" from the closing fence.
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop the opening fence line (```json or ```) and the closing ``` line.
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner)

    return text

# ------------------------------------------------------------------ #
# Rate limiter
# ------------------------------------------------------------------ #

class AsyncRateLimiter:
    """Sliding-window limiter enforcing at most `rpm` acquisitions per 60 seconds.

    Shared across all concurrent tasks of a single agent run so the *aggregate* request
    rate stays under budget regardless of the concurrency semaphore. Construct one per
    agent run (inside the agent's asyncio.run entrypoint) so it binds to that event loop.
    """

    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self._timestamps = []          # monotonic times of recent acquisitions, oldest first
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a request slot is available within the rolling 60s window."""
        while True:
            async with self._lock:
                now = time.monotonic()
                cutoff = now - _WINDOW_SECONDS
                self._timestamps = [t for t in self._timestamps if t > cutoff]

                if len(self._timestamps) < self.rpm:
                    self._timestamps.append(now)
                    return

                wait_s = self._timestamps[0] + _WINDOW_SECONDS - now

            await asyncio.sleep(max(wait_s, 0.01))

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def is_rate_limit(exc: Exception) -> bool:
    """True if `exc` looks like a Gemini rate-limit / quota error."""
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg


async def close_async_client(client, logger=None) -> None:
    """Close a genai client's async connection pool while the event loop is still alive.

    Each agent drives its calls with asyncio.run(), which closes its event loop on
    return. If the client's pooled connections are left for garbage collection, the
    google.genai/httpx teardown coroutine runs after that loop is gone and asyncio
    logs "Task exception was never retrieved ... RuntimeError: Event loop is closed"
    at exit. Call this at the end of every agent's _run_all (in a finally block).
    """
    try:
        await client.aio.aclose()
    except Exception as e:
        if logger:
            logger.debug(f"Async client close failed (harmless): {e}")


def parse_retry_delay(exc: Exception):
    """Extract Gemini's suggested retry delay (seconds) from a 429, or None."""
    match = re.search(r"retry[_-]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s?",
                      str(exc), re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None

# ------------------------------------------------------------------ #
# Core retry wrapper
# ------------------------------------------------------------------ #

async def async_call_with_retry(
    make_call,
    *,
    limiter: AsyncRateLimiter,
    logger,
    backoff_base: int,
    backoff_cap: int,
    transient_max_retries: int,
):
    """Run `make_call` (an async, no-arg callable) with rate-aware retry.

    - Wraps every attempt in a REQUEST_TIMEOUT deadline; a hung call is cancelled
      and treated as a transient error.
    - SafetyBlockedError and MaxTokensError are non-retryable: re-raised immediately
      so the caller marks the row failed without burning further API quota.
    - On a rate-limit (429) error: backs off >= RATE_LIMIT_MIN_BACKOFF (or Gemini's
      retryDelay hint, whichever is larger) with full jitter.
    - On any other transient error: fast exponential backoff.
    - Rate-limit and transient retries are counted separately.
    """
    last_exc = None
    rate_attempts = 0
    transient_attempts = 0

    while True:
        await limiter.acquire()
        try:
            return await asyncio.wait_for(make_call(), timeout=REQUEST_TIMEOUT)
        except (SafetyBlockedError, MaxTokensError):
            raise  # non-retryable — propagate immediately
        except Exception as e:
            last_exc = e
            timed_out = isinstance(e, asyncio.TimeoutError)

            if not timed_out and is_rate_limit(e):
                rate_attempts += 1
                if rate_attempts >= RATE_LIMIT_MAX_RETRIES:
                    break
                hint = parse_retry_delay(e)
                base = max(RATE_LIMIT_MIN_BACKOFF, hint or 0)
                sleep_s = base + random.uniform(0, _WINDOW_SECONDS)
                logger.info(
                    f"Rate limit hit (rate attempt {rate_attempts}); "
                    f"backing off {sleep_s:.1f}s"
                    + (f" (hint {hint:.0f}s)" if hint else "")
                )
            else:
                transient_attempts += 1
                if transient_attempts >= transient_max_retries:
                    break
                sleep_s = min(backoff_base ** transient_attempts, backoff_cap)
                sleep_s += random.uniform(0, 1)
                if timed_out:
                    logger.warning(
                        f"Request timed out after {REQUEST_TIMEOUT}s "
                        f"(transient attempt {transient_attempts}); retrying in {sleep_s:.1f}s"
                    )
                else:
                    logger.info(
                        f"Error (transient attempt {transient_attempts}): {e}; "
                        f"retrying in {sleep_s:.1f}s"
                    )

            await asyncio.sleep(sleep_s)

    raise last_exc
