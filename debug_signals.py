#!/usr/bin/env python3
"""
debug_signals.py — inspect indicator values and strategy decision for any ticker.

Usage:
    python debug_signals.py TICKER [TICKER2 ...] [--timeframe 15m|1h|1d] [--strategy balanced|momentum|mean_reversion]

Examples:
    python debug_signals.py GARAN.IS
    python debug_signals.py EURUSD=X --timeframe 1h
    python debug_signals.py AAPL MSFT --strategy momentum
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from tickerbot.core.market import CandleRequest, fetch_history
from tickerbot.core.strategy import StrategyManager, TIMEFRAME_CONFIG, _ADX_TREND_THRESHOLD, _MAX_ATR_PCT
from tickerbot.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    calculate_stochastic,
    calculate_vwap,
    calculate_williams_r,
)

_BAR = "─" * 56


def _val(series: pd.Series, fmt: str = ".4f") -> str:
    try:
        v = float(series.dropna().iloc[-1])
        return format(v, fmt)
    except Exception:
        return "n/a"


def _flag(cond: bool, true_label: str = "YES", false_label: str = "NO") -> str:
    return true_label if cond else false_label


def analyze(ticker: str, timeframe: str, strategy: str) -> None:
    cfg = TIMEFRAME_CONFIG.get(timeframe)
    if not cfg:
        print(f"Unknown timeframe {timeframe!r}. Choose from: {list(TIMEFRAME_CONFIG)}")
        return

    period, interval = cfg
    print(f"\n{_BAR}")
    print(f"  {ticker}  |  timeframe={timeframe}  |  strategy={strategy}")
    print(_BAR)
    print(f"  Fetching {period} of {interval} data …")

    data = fetch_history(CandleRequest(ticker=ticker, period=period, interval=interval))
    if data.empty:
        print("  ERROR: no data returned — check ticker symbol or internet connection")
        return

    print(f"  Bars loaded : {len(data)}  ({data.index[0].date()} → {data.index[-1].date()})")
    price = float(data["Close"].iloc[-1])
    print(f"  Last close  : {price:.4f}")

    # ── Moving averages ────────────────────────────────────────────────────
    sma20  = _val(calculate_sma(data, 20))
    sma50  = _val(calculate_sma(data, 50))
    sma200 = _val(calculate_sma(data, 200))
    print(f"\n  SMA(20)={sma20}  SMA(50)={sma50}  SMA(200)={sma200}")

    # ── RSI ───────────────────────────────────────────────────────────────
    rsi = calculate_rsi(data)
    rsi_val = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50.0
    rsi_label = (
        "OVERSOLD ← BUY signal" if rsi_val < 30
        else "OVERBOUGHT ← SELL signal" if rsi_val > 70
        else "neutral"
    )
    print(f"\n  RSI(14)     : {rsi_val:.2f}  [{rsi_label}]")

    # ── MACD ──────────────────────────────────────────────────────────────
    macd, sig_line = calculate_macd(data)
    macd_last = float(macd.dropna().iloc[-1]) if not macd.dropna().empty else 0.0
    sig_last  = float(sig_line.dropna().iloc[-1]) if not sig_line.dropna().empty else 0.0
    cross = macd_last - sig_last
    cross_label = "BULLISH (MACD above signal)" if cross > 0 else "BEARISH (MACD below signal)"
    print(f"  MACD        : {macd_last:.4f}  Signal={sig_last:.4f}  Cross={cross:+.4f}  [{cross_label}]")

    # ── Bollinger Bands ───────────────────────────────────────────────────
    upper_bb, lower_bb = calculate_bollinger_bands(data)
    upper_val = float(upper_bb.dropna().iloc[-1]) if not upper_bb.dropna().empty else price * 1.05
    lower_val = float(lower_bb.dropna().iloc[-1]) if not lower_bb.dropna().empty else price * 0.95
    bb_pos = (
        "ABOVE upper band ← overbought" if price > upper_val
        else "BELOW lower band ← oversold" if price < lower_val
        else "inside bands"
    )
    print(f"  BB(20,2)    : lower={lower_val:.4f}  price={price:.4f}  upper={upper_val:.4f}  [{bb_pos}]")

    # ── ATR ───────────────────────────────────────────────────────────────
    atr = calculate_atr(data)
    if not atr.dropna().empty and price > 0:
        atr_val = float(atr.dropna().iloc[-1])
        atr_pct = atr_val / price
        atr_penalty = min(0.25, (atr_pct - _MAX_ATR_PCT) / _MAX_ATR_PCT * 0.25) if atr_pct > _MAX_ATR_PCT else 0.0
        atr_note = f"penalty={atr_penalty:.2f}" if atr_penalty > 0 else "within normal range"
        print(f"  ATR(14)     : {atr_val:.4f}  ({atr_pct*100:.2f}% of price)  [{atr_note}]")
    else:
        atr_val = 0.0
        atr_penalty = 0.0
        print("  ATR(14)     : n/a")

    # ── ADX ───────────────────────────────────────────────────────────────
    adx_val = 25.0
    di_trend_bullish = True
    try:
        adx_series, di_plus, di_minus = calculate_adx(data)
        if not adx_series.dropna().empty:
            adx_val = float(adx_series.dropna().iloc[-1])
            dip = float(di_plus.dropna().iloc[-1])
            dim = float(di_minus.dropna().iloc[-1])
            di_trend_bullish = dip > dim
            trend_str = "TRENDING" if adx_val >= _ADX_TREND_THRESHOLD else "RANGING/SIDEWAYS"
            dir_str = f"DI+={dip:.1f} > DI-={dim:.1f} → BULLISH" if di_trend_bullish else f"DI+={dip:.1f} < DI-={dim:.1f} → BEARISH"
            print(f"  ADX(14)     : {adx_val:.2f}  [{trend_str}]  {dir_str}")
    except Exception as exc:
        print(f"  ADX(14)     : error ({exc})")

    is_trending = adx_val >= _ADX_TREND_THRESHOLD

    # ── Williams %R ───────────────────────────────────────────────────────
    wr_val = -50.0
    try:
        wr = calculate_williams_r(data)
        if not wr.dropna().empty:
            wr_val = float(wr.dropna().iloc[-1])
            wr_label = (
                "OVERBOUGHT (>-20)" if wr_val > -20
                else "OVERSOLD (<-80)" if wr_val < -80
                else "neutral"
            )
            print(f"  Williams %R : {wr_val:.2f}  [{wr_label}]")
    except Exception as exc:
        print(f"  Williams %R : error ({exc})")

    # ── Stochastic ────────────────────────────────────────────────────────
    stoch_k_val = 50.0
    try:
        stoch_k, stoch_d = calculate_stochastic(data)
        if not stoch_k.dropna().empty:
            stoch_k_val = float(stoch_k.dropna().iloc[-1])
            stoch_d_val = float(stoch_d.dropna().iloc[-1]) if not stoch_d.dropna().empty else stoch_k_val
            stoch_label = (
                "OVERBOUGHT (>80)" if stoch_k_val > 80
                else "OVERSOLD (<20)" if stoch_k_val < 20
                else "neutral"
            )
            print(f"  Stoch %K/%D : K={stoch_k_val:.2f}  D={stoch_d_val:.2f}  [{stoch_label}]")
    except Exception as exc:
        print(f"  Stoch %K/%D : error ({exc})")

    # ── VWAP (only meaningful for intraday) ───────────────────────────────
    if timeframe in ("15m", "1h"):
        try:
            vwap = calculate_vwap(data)
            if not vwap.dropna().empty:
                vwap_val = float(vwap.dropna().iloc[-1])
                vwap_label = "price ABOVE vwap ← bullish bias" if price > vwap_val else "price BELOW vwap ← bearish bias"
                print(f"  VWAP        : {vwap_val:.4f}  [{vwap_label}]")
        except Exception:
            pass

    # ── Strategy decision ─────────────────────────────────────────────────
    print(f"\n  {_BAR[2:]}")
    print(f"  Strategy scoring  ({strategy})")
    print(f"  {_BAR[2:]}")

    manager = StrategyManager()
    raw = manager._raw_score(
        strategy=strategy,
        price=price,
        rsi=rsi_val,
        macd_cross=cross,
        upper_bb=upper_val,
        lower_bb=lower_val,
        adx=adx_val,
        is_trending=is_trending,
        di_trend_bullish=di_trend_bullish,
        wr=wr_val,
        stoch_k=stoch_k_val,
    )
    confidence = min(abs(raw) / 3.5, 1.0) - atr_penalty
    confidence = max(0.0, confidence)

    if raw >= 1.3:
        action = "BUY"
    elif raw <= -1.3:
        action = "SELL"
    else:
        action = "HOLD"

    print(f"  Raw score   : {raw:+.4f}  (threshold ±1.3)")
    print(f"  Confidence  : {confidence:.4f}  (ATR penalty={atr_penalty:.4f})")
    print(f"\n  >>> DECISION: {action}  (conf={confidence:.2f}) <<<")

    # Score breakdown
    print(f"\n  Contributing factors:")
    if strategy == "momentum":
        if not is_trending:
            print("  ✗ ADX < 20 — market not trending, score zeroed")
        else:
            print(f"  • MACD cross {cross:+.4f} → {'  +1.5' if cross > 0 else '  -1.0'}")
            print(f"  • RSI {rsi_val:.1f} {'> 55 → +0.8' if rsi_val > 55 else '(no boost)'}")
            if rsi_val > 78: print(f"  • RSI {rsi_val:.1f} > 78 → -0.7 (overbought)")
            print(f"  • Williams %R {wr_val:.1f} → {'+0.4 (near overbought, momentum)' if wr_val > -30 else '-0.4 (losing steam)' if wr_val < -70 else 'no signal'}")
            print(f"  • DI direction+MACD → {'+0.5 (aligned bullish)' if di_trend_bullish and cross > 0 else '-0.5 (aligned bearish)' if not di_trend_bullish and cross < 0 else 'mixed'}")
            print(f"  • Stoch %K {stoch_k_val:.1f} → {'+0.3' if stoch_k_val > 60 else 'no signal'}{' -0.3 (short-term top)' if stoch_k_val > 85 else ''}")
    elif strategy == "mean_reversion":
        if is_trending and adx_val > 35:
            print("  ✗ ADX > 35 in strong trend — mean reversion blocked")
        else:
            bb_note = "+1.5 (price below lower band)" if price < lower_val else "-1.5 (price above upper band)" if price > upper_val else "0 (inside bands)"
            print(f"  • Bollinger Bands → {bb_note}")
            print(f"  • RSI {rsi_val:.1f} → {'  +1.0 (oversold)' if rsi_val < 30 else '-1.0 (overbought)' if rsi_val > 70 else 'no signal'}")
            print(f"  • Williams %R {wr_val:.1f} → {'+0.8 (oversold)' if wr_val < -80 else '-0.8 (overbought)' if wr_val > -20 else 'no signal'}")
            print(f"  • Stoch %K {stoch_k_val:.1f} → {'+0.5 (oversold)' if stoch_k_val < 20 else '-0.5 (overbought)' if stoch_k_val > 80 else 'no signal'}")
    else:  # balanced
        print(f"  • MACD cross {cross:+.4f} → {'  +1.0' if cross > 0 else '  -1.0'}")
        print(f"  • RSI {rsi_val:.1f} → {'  +1.0 (oversold <35)' if rsi_val < 35 else '-1.0 (overbought >70)' if rsi_val > 70 else 'no signal'}")
        bb_note = "+0.7 (below lower)" if price < lower_val else "-0.7 (above upper)" if price > upper_val else "0 (inside)"
        print(f"  • Bollinger Bands → {bb_note}")
        if is_trending:
            print(f"  • ADX trend boost → {'+0.4 (bullish aligned)' if cross > 0 and di_trend_bullish else '-0.4 (bearish aligned)' if cross < 0 and not di_trend_bullish else '0 (mixed)'}")
        print(f"  • Williams %R {wr_val:.1f} → {'+0.4 (oversold <-75)' if wr_val < -75 else '-0.4 (overbought >-25)' if wr_val > -25 else 'no signal'}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect indicator values and strategy decision for a ticker."
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbol(s), e.g. GARAN.IS AAPL EURUSD=X")
    parser.add_argument(
        "--timeframe", "-t",
        choices=list(TIMEFRAME_CONFIG.keys()),
        default="1h",
        help="Timeframe to use (default: 1h)",
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=["balanced", "momentum", "mean_reversion", "all"],
        default="all",
        help="Strategy to score (default: all three)",
    )
    args = parser.parse_args()

    strategies = ["balanced", "momentum", "mean_reversion"] if args.strategy == "all" else [args.strategy]

    for ticker in args.tickers:
        for strat in strategies:
            analyze(ticker, args.timeframe, strat)


if __name__ == "__main__":
    main()
