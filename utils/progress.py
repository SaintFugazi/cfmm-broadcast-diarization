import asyncio
from tqdm import tqdm


async def gather_with_progress(tasks, desc, unit="it", return_exceptions=True):
    """Await `tasks` like asyncio.gather, advancing a tqdm bar as each one completes.

    Unlike tqdm.asyncio.gather, this preserves asyncio.gather's semantics — ordered
    results and `return_exceptions` — while still showing per-item progress. The bar
    ticks on completion regardless of success or failure.
    """
    tasks = list(tasks)
    pbar = tqdm(total=len(tasks), desc=desc, unit=unit)

    async def _wrap(task):
        try:
            return await task
        finally:
            pbar.update(1)

    try:
        return await asyncio.gather(
            *[_wrap(t) for t in tasks], return_exceptions=return_exceptions
        )
    finally:
        pbar.close()
