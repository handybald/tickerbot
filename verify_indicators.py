#!/usr/bin/env python3
"""
verify_indicators.py — cross-check our indicator implementations against pandas_ta.

Usage:
    python verify_indicators.py [TICKER] [--timeframe 1d|1h|15m]

Prints per-indicator last-value comparison and last-100-bar max deviation.
Max diff of 0.000000 in steady state = our formula is identical to the reference.
"""
from __future__ import annotations
import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

try:
    import pandas_ta as pta
except ImportError:
    raise SystemExit("Run: pip install pandas-ta")

from tickerbot.core.market import CandleRequest, fetch_history
from tickerbot.core.strategy import TIMEFRAME_CONFIG
from tickerbot.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    calculate_stochastic,
    calculate_williams_r,
)

_W = 16  # column width


def _row(name: str, ours: float, ref: float, max_diff: float, note: str = "") -> None:
    status = "OK" if max_diff < 1e-6 else ("WARN" if max_diff < 0.1 else "FAIL")
    print(f"  {name:<18} our={ours:>10.4f}  ref={ref:>10.4f}  last-100 max_diff={max_diff:.8f}  [{status}]  {note}")


def verify(ticker: str, timeframe: str) -> None:
    period, interval = TIMEFRAME_CONFIG[timeframe]
    print(f"\nFetching {ticker}  ({period} / {interval}) …")
    data = fetch_history(CandleRequest(ticker=ticker, period=period, interval=interval))
    if data.empty:
        print("  ERROR: no data")
        return

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.dropna()
    print(f"Bars loaded: {len(data)}  ({data.index[0].date()} → {data.index[-1].date()})\n")

    def diff100(a: pd.Series, b: pd.Series) -> float:
        """Max abs diff over last 100 non-NaN aligned bars."""
        both = pd.concat([a, b], axis=1).dropna()
        both.columns = ["a", "b"]
        return float((both["a"] - both["b"]).abs().iloc[-100:].max())

    # ── RSI ──────────────────────────────────────────────────────────────
    our = calculate_rsi(data)
    ref = pta.rsi(data["Close"], length=14)
    _row("RSI(14)", our.iloc[-1], ref.iloc[-1], diff100(our, ref),
         "Wilder EMA (com=13)")

    # ── MACD ─────────────────────────────────────────────────────────────
    our_m, our_s = calculate_macd(data)
    ref_df = pta.macd(data["Close"], fast=12, slow=26, signal=9)
    ref_m = ref_df["MACD_12_26_9"]
    ref_s = ref_df["MACDs_12_26_9"]
    _row("MACD line", our_m.iloc[-1], ref_m.iloc[-1], diff100(our_m, ref_m),
         "EMA(12)-EMA(26)")
    _row("MACD signal", our_s.iloc[-1], ref_s.iloc[-1], diff100(our_s, ref_s),
         "EMA(9) of MACD line")

    # ── Bollinger Bands ───────────────────────────────────────────────────
    our_u, our_l = calculate_bollinger_bands(data)
    bb = pta.bbands(data["Close"], length=20, std=2)
    # pandas_ta column name varies by version; find it
    ucol = next((c for c in bb.columns if c.startswith("BBU")), None)
    lcol = next((c for c in bb.columns if c.startswith("BBL")), None)
    if ucol and lcol:
        _row("BB upper", our_u.iloc[-1], bb[ucol].iloc[-1], diff100(our_u, bb[ucol]),
             "SMA(20) + 2*std  ddof=1")
        _row("BB lower", our_l.iloc[-1], bb[lcol].iloc[-1], diff100(our_l, bb[lcol]),
             "SMA(20) - 2*std  ddof=1")
    else:
        print("  BB: could not find reference columns")

    # ── ATR ───────────────────────────────────────────────────────────────
    our_atr = calculate_atr(data)
    ref_atr = pta.atr(data["High"], data["Low"], data["Close"], length=14)
    _row("ATR(14)", our_atr.iloc[-1], ref_atr.iloc[-1], diff100(our_atr, ref_atr),
         "Wilder EMA of True Range")

    # ── ADX / DI ──────────────────────────────────────────────────────────
    our_adx, our_dip, our_dim = calculate_adx(data)
    ref_adx_df = pta.adx(data["High"], data["Low"], data["Close"], length=14)
    ref_adx = ref_adx_df["ADX_14"]
    ref_dip  = ref_adx_df["DMP_14"]
    ref_dim  = ref_adx_df["DMN_14"]
    _row("ADX(14)", our_adx.iloc[-1], ref_adx.iloc[-1], diff100(our_adx, ref_adx),
         "Wilder EMA of DX")
    _row("DI+(14)", our_dip.iloc[-1], ref_dip.iloc[-1], diff100(our_dip, ref_dip))
    _row("DI-(14)", our_dim.iloc[-1], ref_dim.iloc[-1], diff100(our_dim, ref_dim))

    # ── Williams %R ───────────────────────────────────────────────────────
    our_wr = calculate_williams_r(data)
    ref_wr = pta.willr(data["High"], data["Low"], data["Close"], length=14)
    _row("Williams %R", our_wr.iloc[-1], ref_wr.iloc[-1], diff100(our_wr, ref_wr))

    # ── Stochastic ────────────────────────────────────────────────────────
    our_k, our_d = calculate_stochastic(data)
    ref_stoch = pta.stoch(data["High"], data["Low"], data["Close"], k=14, d=3, smooth_k=1)
    ref_k = ref_stoch.iloc[:, 0]
    ref_d = ref_stoch.iloc[:, 1]
    _row("Stoch %K", our_k.iloc[-1], ref_k.iloc[-1], diff100(our_k, ref_k),
         "fast (smooth_k=1)")
    _row("Stoch %D", our_d.iloc[-1], ref_d.iloc[-1], diff100(our_d, ref_d),
         "SMA(3) of %K")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify indicator implementations against pandas_ta.")
    parser.add_argument("tickers", nargs="*", default=["AVGO"], help="Ticker(s) to test (default: AVGO)")
    parser.add_argument("--timeframe", "-t", choices=list(TIMEFRAME_CONFIG), default="1d")
    args = parser.parse_args()

    for ticker in args.tickers:
        verify(ticker.upper(), args.timeframe)


if __name__ == "__main__":
    main()
