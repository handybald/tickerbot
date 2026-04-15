from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from tickerbot.indicators import calculate_rsi, calculate_sma

try:
    from tradingview_screener import Query
except Exception:
    Query = None


BIST30_TICKERS = [
    "AKBNK.IS", "ALARK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "EKGYO.IS", "ENJSA.IS",
    "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS",
    "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS",
    "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TSKB.IS", "TUPRS.IS", "YKBNK.IS",
    "BRSAN.IS", "ODAS.IS",
]

NASDAQ100_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "ADBE", "PEP", "CSCO", "TMUS", "INTC", "CMCSA", "QCOM", "AMGN",
    "TXN", "INTU", "HON", "AMAT", "BKNG", "GILD", "ADP", "SBUX", "VRTX", "MDLZ",
]

# Forex / commodity currency pairs (yfinance format).
# Each entry is a yfinance symbol that returns OHLCV data.
FOREX_PAIRS = [
    # Major pairs
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
    "AUDUSD=X", "NZDUSD=X", "USDCAD=X",
    # Cross pairs
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "EURCHF=X",
    # Metals vs USD and EUR (popular with Turkish traders)
    "XAUUSD=X", "XAGUSD=X", "XAUEUR=X",
    # USD/TRY pair
    "USDTRY=X", "EURTRY=X",
]

CACHE_PATH = Path("data/universe_cache.json")


@dataclass
class CandleRequest:
    ticker: str
    period: str
    interval: str


def fetch_history(req: CandleRequest):
    data = yf.download(
        tickers=req.ticker,
        period=req.period,
        interval=req.interval,
        progress=False,
        threads=False,
        auto_adjust=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        # yfinance can return a MultiIndex even for one symbol.
        try:
            data = data.xs(req.ticker, axis=1, level=-1)
        except Exception:
            try:
                data.columns = data.columns.get_level_values(0)
            except Exception:
                pass

    return data


def get_last_price(ticker: str) -> float | None:
    history = fetch_history(CandleRequest(ticker=ticker, period="5d", interval="1d"))
    if history.empty:
        return None
    return float(history["Close"].iloc[-1])


def get_universe(scope: str, refresh_minutes: int = 720, force_sync: bool = False) -> list[str]:
    normalized = scope.upper().strip()
    if normalized == "BIST30":
        return BIST30_TICKERS.copy()
    if normalized == "BIST":
        return load_or_sync_bist_universe(refresh_minutes=refresh_minutes, force_sync=force_sync)
    if normalized in {"NASDAQ", "US"}:
        return load_or_sync_us_universe(refresh_minutes=refresh_minutes, force_sync=force_sync)
    if normalized == "FOREX":
        return FOREX_PAIRS.copy()
    return BIST30_TICKERS.copy()


def is_forex_ticker(ticker: str) -> bool:
    """Returns True if the ticker is a forex/commodity pair (ends with =X or =F)."""
    return ticker.endswith("=X") or ticker.endswith("=F")


def load_or_sync_bist_universe(refresh_minutes: int = 720, force_sync: bool = False) -> list[str]:
    cache = _read_cache()
    tickers = cache.get("bist", []) if isinstance(cache, dict) else []

    if force_sync or _cache_stale(cache, refresh_minutes):
        synced = sync_bist_universe()
        if synced:
            return synced

    if tickers:
        return tickers

    synced = sync_bist_universe()
    if synced:
        return synced

    # Hard fallback keeps service alive even if sources are unavailable.
    return BIST30_TICKERS.copy()


def sync_bist_universe() -> list[str]:
    providers = (_sync_from_tradingview, _sync_from_hedeffiyat)

    for provider in providers:
        tickers = provider()
        if len(tickers) >= 25:
            return tickers

    return []


def load_or_sync_us_universe(refresh_minutes: int = 720, force_sync: bool = False) -> list[str]:
    cache = _read_cache()
    tickers = cache.get("us", []) if isinstance(cache, dict) else []

    if force_sync or _cache_stale(cache, refresh_minutes, key="updated_at_us"):
        synced = sync_us_universe()
        if synced:
            return synced

    if tickers:
        return tickers

    synced = sync_us_universe()
    if synced:
        return synced

    return NASDAQ100_TICKERS.copy()


def sync_us_universe() -> list[str]:
    tickers = _sync_us_from_tradingview()
    if len(tickers) >= 50:
        return tickers
    return []


def universe_cache_info() -> dict:
    cache = _read_cache()
    if not isinstance(cache, dict):
        return {"size": 0, "updated_at": "", "source": "none", "us_size": 0, "us_updated_at": ""}
    return {
        "size": len(cache.get("bist", [])),
        "updated_at": cache.get("updated_at", ""),
        "source": cache.get("source", "mixed"),
        "us_size": len(cache.get("us", [])),
        "us_updated_at": cache.get("updated_at_us", ""),
    }


def _sync_from_tradingview() -> list[str]:
    if Query is None:
        return []

    try:
        screener = Query().set_screener("turkey")
        _, data = screener.get_scanner_data()
        symbols = data.get("name", [])
    except Exception:
        return []

    tickers = {_normalize_ticker(sym) for sym in symbols}
    cleaned = sorted(t for t in tickers if t)
    if cleaned:
        _write_cache({"updated_at": _now_iso(), "source": "tradingview", "bist": cleaned})
    return cleaned


def _sync_from_hedeffiyat() -> list[str]:
    try:
        response = requests.get("https://hedeffiyat.com.tr/", timeout=20)
        response.raise_for_status()
    except Exception:
        return []

    text = response.text
    # Very small parser via split patterns to avoid adding more parsing deps here.
    symbols = []
    marker = "<td>"
    idx = 0
    while True:
        idx = text.find(marker, idx)
        if idx == -1:
            break
        end = text.find("</td>", idx)
        if end == -1:
            break
        raw = text[idx + len(marker):end].strip()
        if raw and raw.isupper() and 2 <= len(raw) <= 6 and raw.isalpha():
            symbols.append(raw)
        idx = end + 5

    tickers = {_normalize_ticker(sym) for sym in symbols}
    cleaned = sorted(t for t in tickers if t)
    if cleaned:
        _write_cache({"updated_at": _now_iso(), "source": "hedeffiyat", "bist": cleaned})
    return cleaned


def _sync_us_from_tradingview() -> list[str]:
    if Query is None:
        return []
    try:
        screener = Query().set_screener("america")
        _, data = screener.get_scanner_data()
        symbols = data.get("name", [])
    except Exception:
        return []

    cleaned = sorted(_normalize_us_ticker(sym) for sym in symbols)
    cleaned = [s for s in cleaned if s]
    if cleaned:
        cache = _read_cache()
        if not isinstance(cache, dict):
            cache = {}
        cache.update(
            {
                "updated_at_us": _now_iso(),
                "source_us": "tradingview",
                "us": cleaned,
            }
        )
        _write_cache(cache)
    return cleaned


def _normalize_ticker(symbol: str) -> str:
    if not symbol:
        return ""
    normalized = symbol.strip().upper().replace("BIST:", "")
    if "." in normalized:
        # Keep existing suffix if already present.
        return normalized
    if not normalized.isalnum():
        return ""
    return f"{normalized}.IS"


def _normalize_us_ticker(symbol: str) -> str:
    if not symbol:
        return ""
    normalized = symbol.strip().upper()
    normalized = normalized.replace("NASDAQ:", "").replace("NYSE:", "").replace("AMEX:", "")
    if "." in normalized:
        return ""
    if not normalized.isalnum():
        return ""
    return normalized


def _read_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}


def _write_cache(payload: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload))


def _cache_stale(cache: dict, refresh_minutes: int, key: str = "updated_at") -> bool:
    if not cache:
        return True
    updated_at = cache.get(key, "")
    if not updated_at:
        return True
    try:
        dt = datetime.fromisoformat(updated_at)
    except Exception:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - dt > timedelta(minutes=max(5, refresh_minutes))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def forex_volatility_ok(pair: str = "EURUSD=X", max_atr_pct: float = 0.015) -> dict:
    """
    Forex regime check: reject trading when intraday volatility (ATR%) is extreme.
    Returns {"ok": True, "tradeable": bool, "atr_pct": float}.
    """
    from tickerbot.indicators import calculate_atr  # local import to avoid circular
    data = fetch_history(CandleRequest(ticker=pair, period="30d", interval="1h"))
    if data.empty or len(data) < 20:
        return {"ok": False, "reason": "no_data"}
    atr = calculate_atr(data)
    if atr.empty or atr.isna().all():
        return {"ok": False, "reason": "no_atr"}
    price = float(data["Close"].iloc[-1])
    atr_val = float(atr.iloc[-1])
    atr_pct = atr_val / max(price, 1e-9)
    return {"ok": True, "tradeable": atr_pct <= max_atr_pct, "atr_pct": atr_pct}


def regime_snapshot(
    ticker: str = "XU100.IS",
    fast_sma: int = 50,
    slow_sma: int = 200,
    rsi_min: float = 45.0,
) -> dict:
    data = fetch_history(CandleRequest(ticker=ticker, period="2y", interval="1d"))
    if data.empty or len(data) < max(fast_sma, slow_sma) + 5:
        return {"ok": False, "reason": "no_data"}

    price = float(data["Close"].iloc[-1])
    fast = float(calculate_sma(data, fast_sma).iloc[-1])
    slow = float(calculate_sma(data, slow_sma).iloc[-1])
    rsi = float(calculate_rsi(data).iloc[-1])

    bullish = bool(price > fast and fast > slow and rsi >= rsi_min)
    return {
        "ok": True,
        "bullish": bullish,
        "price": price,
        "fast_sma": fast,
        "slow_sma": slow,
        "rsi": rsi,
    }
