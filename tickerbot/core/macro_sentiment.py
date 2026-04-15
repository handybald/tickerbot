"""
macro_sentiment.py — market-wide and country-level news sentiment.

Fetches yfinance news for index proxy tickers and returns a [-1, 1] sentiment
score for each context (global, market-specific). These scores are applied as
a small modifier on top of per-ticker indicator signals.

Proxy tickers used:
  Global risk-on/off : SPY, ^GSPC
  US / Nasdaq        : QQQ, ^NDX
  Turkey / BIST      : GARAN.IS, THYAO.IS  (most liquid, most news coverage)
  Forex / Dollar     : DX-Y.NYB, GC=F

Cache TTL: 4 hours per context key.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

# Proxy tickers per macro context.
_PROXIES: dict[str, list[str]] = {
    "global": ["SPY", "^GSPC"],
    "us":     ["QQQ", "^NDX"],
    "bist":   ["GARAN.IS", "THYAO.IS"],
    "forex":  ["DX-Y.NYB", "GC=F"],
}

_SCOPE_TO_MARKET: dict[str, str] = {
    "BIST30": "bist",
    "BIST":   "bist",
    "NASDAQ": "us",
    "US":     "us",
    "FOREX":  "forex",
    "ALL":    "us",        # ALL scope uses US proxy for market context
}

# In-memory cache: key → (score: float, fetched_at: float)
_cache: dict[str, tuple[float, float]] = {}
_TTL = 4.0 * 3600.0   # 4 hours


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_macro_sentiment(scope: str) -> dict[str, float]:
    """
    Return macro news sentiment for the given market scope.

    Returns a dict with two keys:
        "global"  : world-level sentiment based on SPY/^GSPC news
        "market"  : scope-specific sentiment (BIST, US, Forex …)

    Both values are in [-1, 1].  0.0 means neutral or unavailable.

    Usage:
        macro = get_macro_sentiment("BIST30")
        net = macro["global"] * 0.4 + macro["market"] * 0.6
        signal_score += net * 0.3   # small additive tilt
    """
    market_key = _SCOPE_TO_MARKET.get(scope.upper().strip(), "bist")
    return {
        "global": _get_cached("global"),
        "market": _get_cached(market_key),
    }


def invalidate_cache() -> None:
    """Force-expire the macro sentiment cache (e.g. after /sync)."""
    _cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_cached(key: str) -> float:
    now = time.time()
    entry = _cache.get(key)
    if entry and (now - entry[1]) < _TTL:
        return entry[0]

    score = _fetch(key)
    _cache[key] = (score, now)
    log.debug("macro_sentiment[%s] refreshed → %.3f", key, score)
    return score


def _fetch(key: str) -> float:
    proxies = _PROXIES.get(key, [])
    for ticker in proxies:
        try:
            from tickerbot.data_fetcher import DataFetcher
            score = DataFetcher(ticker).get_sentiment()
            if score != 0.0:
                return score
        except Exception as exc:
            log.debug("macro_sentiment fetch failed for %s: %s", ticker, exc)
    return 0.0
