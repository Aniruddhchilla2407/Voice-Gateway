"""Shared Groq client and error translation."""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import groq

from vgw.providers.base import ProviderError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> groq.AsyncGroq:
    """One client for the whole process, so HTTPS connections stay warm across turns."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Put it in .env (see .env.example).")
    # max_retries=0: hidden SDK retries would add latency and hide failures from our trace.
    return groq.AsyncGroq(api_key=api_key, max_retries=0)


def translate_error(exc: groq.APIError, provider: str) -> ProviderError:
    """Map an SDK exception to ProviderError. The message never includes the request or key."""
    logger.warning("%s call failed: %s", provider, exc)  # the API's own message helps debugging
    if isinstance(exc, groq.APIStatusError):
        transient = exc.status_code == 429 or exc.status_code >= 500
        return ProviderError(
            f"{provider}: HTTP {exc.status_code}", provider=provider, transient=transient
        )
    if isinstance(exc, groq.APIConnectionError):  # includes timeouts
        return ProviderError(f"{provider}: connection problem", provider=provider, transient=True)
    return ProviderError(f"{provider}: {type(exc).__name__}", provider=provider, transient=False)