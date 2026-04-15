#!/usr/bin/env python3
"""
debug_signals.py — inspect indicator values, volume, macro context,
                   and multi-timeframe decision for any ticker.

Usage:
    python debug_signals.py TICKER [TICKER2 ...] [--strategy balanced|momentum|mean_reversion|all]
    python debug_signals.py GARAN.IS
    python debug_signals.py AVGO --strategy momentum
    python debug_signals.py EURUSD=X --no-sentiment
"""
from __future__ import annotations

import argparse

import pandas as pd

from tickerbot.core.macro_sentiment import get_macro_sentiment
from tickerbot.core.market import CandleRequest, fetch_history
from tickerbot.core.strategy import (
    StrategyManager,
    TIMEFRAME_CONFIG,
    _ADX_TREND_THRESHOLD,
    _MAX_ATR_PCT,
)
from tickerbot.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    calculate_sma,
    calculate_stochastic,
    calculate_volume_ratio,
    calculate_vwap,
    calculate_williams_r,
)

_BAR = "─" * 60


def _v(series: pd.Series, fmt: str = ".4f") -> str:
    try:
        return format(float(series.dropna().iloc[-1]), fmt)
    except Exception:
        return "n/a"


def _fetch_sentiment(ticker: str) -> float:
    try:
        from tickerbot.data_fetcher import DataFetcher
        return DataFetcher(ticker).get_sentiment()
    except Exception:
        return 0.0


def _fetch_macro(scope: str) -> dict[str, float]:
    try:
        return get_macro_sentiment(scope)
    except Exception:
        return {"global": 0.0, "market": 0.0}


def analyze_ticker(
    ticker: str,
    strategy: str,
    ticker_sentiment: float,
    macro: dict[str, float],
    data_by_tf: dict[str, pd.DataFrame],
) -> None:
    """Print full indicator + MTF breakdown for one ticker / strategy combo."""
    macro_net = macro.get("global", 0.0) * 0.4 + macro.get("market", 0.0) * 0.6
    combined_sent = ticker_sentiment * 0.6 + macro_net * 0.4

    # Use selected analysis TF (default 1h) for indicator display
    for tf_pref in ("1h", "1d", "15m"):
        if tf_pref in data_by_tf:
            display_tf = tf_pref
            break
    else:
        print(f"  No data available for {ticker}")
        return

    data = data_by_tf[display_tf]
    price = float(data["Close"].iloc[-1])

    print(f"\n{_BAR}")
    print(f"  {ticker}  |  strategy={strategy}")
    print(f"  Bars: {display_tf}={len(data)}" +
          "".join(f"  {tf}={len(data_by_tf[tf])}" for tf in ("15m", "1d") if tf in data_by_tf))
    print(_BAR)
    print(f"  Last close  : {price:.4f}")

    # ── Moving Averages ───────────────────────────────────────────────────
    print(f"  SMA(20/50/200): {_v(calculate_sma(data, 20))} / {_v(calculate_sma(data, 50))} / {_v(calculate_sma(data, 200))}")

    # ── RSI ───────────────────────────────────────────────────────────────
    rsi = calculate_rsi(data)
    rsi_val = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50.0
    rsi_lbl = "OVERSOLD" if rsi_val < 30 else ("OVERBOUGHT" if rsi_val > 70 else "neutral")
    print(f"  RSI(14)       : {rsi_val:.2f}  [{rsi_lbl}]")

    # ── MACD ──────────────────────────────────────────────────────────────
    macd, sig_line = calculate_macd(data)
    cross = (float(macd.dropna().iloc[-1]) if not macd.dropna().empty else 0.0) - \
            (float(sig_line.dropna().iloc[-1]) if not sig_line.dropna().empty else 0.0)
    print(f"  MACD cross    : {cross:+.4f}  [{'BULLISH' if cross > 0 else 'BEARISH'}]")

    # ── Bollinger Bands ───────────────────────────────────────────────────
    upper_bb, lower_bb = calculate_bollinger_bands(data)
    upper_val = float(upper_bb.dropna().iloc[-1]) if not upper_bb.dropna().empty else price * 1.05
    lower_val = float(lower_bb.dropna().iloc[-1]) if not lower_bb.dropna().empty else price * 0.95
    bb_pos = ("ABOVE upper" if price > upper_val else "BELOW lower" if price < lower_val else "inside")
    print(f"  BB(20,2)      : {bb_pos}  lo={lower_val:.4f}  price={price:.4f}  hi={upper_val:.4f}")

    # ── ATR ───────────────────────────────────────────────────────────────
    atr = calculate_atr(data)
    atr_penalty = 0.0
    if not atr.dropna().empty and price > 0:
        atr_pct = float(atr.dropna().iloc[-1]) / price
        atr_penalty = min(0.25, (atr_pct - _MAX_ATR_PCT) / _MAX_ATR_PCT * 0.25) if atr_pct > _MAX_ATR_PCT else 0.0
        print(f"  ATR(14)       : {float(atr.dropna().iloc[-1]):.4f}  ({atr_pct*100:.2f}% of price)  "
              f"[penalty={atr_penalty:.2f}]")
    else:
        print("  ATR(14)       : n/a")

    # ── ADX ───────────────────────────────────────────────────────────────
    adx_val, di_trend_bullish = 25.0, True
    try:
        adx_s, dip_s, dim_s = calculate_adx(data)
        if not adx_s.dropna().empty:
            adx_val = float(adx_s.dropna().iloc[-1])
            dip = float(dip_s.dropna().iloc[-1])
            dim = float(dim_s.dropna().iloc[-1])
            di_trend_bullish = dip > dim
            t_lbl = "TRENDING" if adx_val >= _ADX_TREND_THRESHOLD else "RANGING"
            d_lbl = f"DI+={dip:.1f}>DI-={dim:.1f} BULL" if di_trend_bullish else f"DI-={dim:.1f}>DI+={dip:.1f} BEAR"
            print(f"  ADX(14)       : {adx_val:.2f}  [{t_lbl}]  {d_lbl}")
    except Exception:
        pass

    is_trending = adx_val >= _ADX_TREND_THRESHOLD

    # ── Williams %R ───────────────────────────────────────────────────────
    wr_val = -50.0
    try:
        wr = calculate_williams_r(data)
        if not wr.dropna().empty:
            wr_val = float(wr.dropna().iloc[-1])
            wr_lbl = "OVERBOUGHT" if wr_val > -20 else ("OVERSOLD" if wr_val < -80 else "neutral")
            print(f"  Williams %R   : {wr_val:.2f}  [{wr_lbl}]")
    except Exception:
        pass

    # ── Stochastic ────────────────────────────────────────────────────────
    stoch_k_val = 50.0
    try:
        sk, sd = calculate_stochastic(data)
        if not sk.dropna().empty:
            stoch_k_val = float(sk.dropna().iloc[-1])
            sd_val = float(sd.dropna().iloc[-1]) if not sd.dropna().empty else stoch_k_val
            s_lbl = "OVERBOUGHT" if stoch_k_val > 80 else ("OVERSOLD" if stoch_k_val < 20 else "neutral")
            print(f"  Stoch %K/%D   : K={stoch_k_val:.2f}  D={sd_val:.2f}  [{s_lbl}]")
    except Exception:
        pass

    # ── VWAP (intraday only) ──────────────────────────────────────────────
    if display_tf in ("15m", "1h"):
        try:
            vwap = calculate_vwap(data)
            if not vwap.dropna().empty:
                vv = float(vwap.dropna().iloc[-1])
                print(f"  VWAP          : {vv:.4f}  [price {'ABOVE' if price > vv else 'BELOW'} vwap]")
        except Exception:
            pass

    # ── OBV (On-Balance Volume) ───────────────────────────────────────────
    print()
    obv_rising = None
    try:
        obv = calculate_obv(data)
        if not obv.dropna().empty and len(obv.dropna()) >= 20:
            obv_sma = obv.rolling(20).mean()
            obv_rising = float(obv.iloc[-1]) > float(obv_sma.dropna().iloc[-1])
            lbl = "OBV RISING ↑ — volume confirms up move" if obv_rising else "OBV FALLING ↓ — volume confirms down move"
            print(f"  OBV trend     : {lbl}")
    except Exception:
        pass

    # ── Volume Ratio ──────────────────────────────────────────────────────
    vol_ratio_val = None
    try:
        vr = calculate_volume_ratio(data)
        if not vr.dropna().empty:
            vol_ratio_val = float(vr.dropna().iloc[-1])
            if vol_ratio_val > 2.0:
                vr_lbl = f"VERY HIGH ({vol_ratio_val:.2f}x avg) — strong confirmation"
            elif vol_ratio_val > 1.5:
                vr_lbl = f"HIGH ({vol_ratio_val:.2f}x avg) — above-average activity"
            elif vol_ratio_val < 0.5:
                vr_lbl = f"LOW ({vol_ratio_val:.2f}x avg) — signal less reliable"
            else:
                vr_lbl = f"normal ({vol_ratio_val:.2f}x avg)"
            print(f"  Volume ratio  : {vr_lbl}")
    except Exception:
        pass

    # ── News & Macro Sentiment ────────────────────────────────────────────
    print()
    ts_lbl = "BULLISH" if ticker_sentiment > 0.25 else ("BEARISH" if ticker_sentiment < -0.25 else "neutral")
    ms_lbl = "BULLISH" if macro_net > 0.15 else ("BEARISH" if macro_net < -0.15 else "neutral")
    print(f"  Ticker news   : {ticker_sentiment:+.3f}  [{ts_lbl}]")
    print(f"  Macro global  : {macro.get('global', 0.0):+.3f}   Market: {macro.get('market', 0.0):+.3f}   Net: {macro_net:+.3f}  [{ms_lbl}]")
    print(f"  Combined sent : {combined_sent:+.3f}  (ticker×0.6 + macro×0.4)")

    # ── Multi-Timeframe Analysis ──────────────────────────────────────────
    print(f"\n  {'─'*54}")
    print(f"  Multi-Timeframe Analysis  ({strategy})")
    print(f"  {'─'*54}")

    manager = StrategyManager()
    mtf = manager.mtf_signal(
        strategy=strategy,
        data_by_tf=data_by_tf,
        ticker_sentiment=ticker_sentiment,
        macro_sentiment=macro_net,
    )

    tf_labels = {"1d": "trend frame", "1h": "setup frame", "15m": "entry frame"}
    for tf_key in ("1d", "1h", "15m"):
        if tf_key in mtf.breakdown:
            s = mtf.breakdown[tf_key]
            lbl = tf_labels.get(tf_key, "")
            bar = "▓" * int(s["confidence"] * 10)
            print(f"  {tf_key} ({lbl:<12}): {s['action']:4s}  conf={s['confidence']:.2f}  {bar}")

    align_str = f"{mtf.alignment}/{mtf.total_tfs} timeframes agree"
    if mtf.alignment == mtf.total_tfs and mtf.total_tfs > 1:
        align_str += "  ← FULL ALIGNMENT"
    elif mtf.action == "HOLD" and mtf.total_tfs > 1:
        align_str += "  ← BLOCKED (timeframes conflict)"

    print(f"\n  Alignment     : {align_str}")
    print(f"\n  >>> MTF DECISION: {mtf.action}  (conf={mtf.confidence:.2f}) <<<")

    # ── Single-TF score breakdown for reference ───────────────────────────
    print(f"\n  Single-TF score breakdown ({display_tf}):")
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
        sentiment=combined_sent,
    )
    print(f"  Raw score: {raw:+.4f}  (threshold ±1.3)")
    if strategy == "momentum":
        if not is_trending:
            print("  ✗ ADX<20 — not trending, score zeroed")
        else:
            print(f"  • MACD {cross:+.4f} → {'  +1.5' if cross > 0 else '  -1.0'}")
            print(f"  • RSI {rsi_val:.1f} → {'+0.8 (>55)' if rsi_val > 55 else 'no boost'}{' -0.7 (>78)' if rsi_val > 78 else ''}")
            print(f"  • Williams %R {wr_val:.1f} → {'+0.4' if wr_val > -30 else '-0.4 (losing steam)' if wr_val < -70 else 'no signal'}")
            print(f"  • DI align → {'+0.5' if di_trend_bullish and cross > 0 else '-0.5' if not di_trend_bullish and cross < 0 else 'mixed'}")
            print(f"  • Stoch {stoch_k_val:.1f} → {'+0.3' if stoch_k_val > 60 else 'no signal'}{' -0.3 (top)' if stoch_k_val > 85 else ''}")
    elif strategy == "mean_reversion":
        if is_trending and adx_val > 35:
            print("  ✗ ADX>35 strong trend — mean reversion blocked")
        else:
            bb_c = "+1.5 (below lower)" if price < lower_val else "-1.5 (above upper)" if price > upper_val else "0 (inside)"
            print(f"  • BB → {bb_c}")
            print(f"  • RSI {rsi_val:.1f} → {'+1.0 (<30)' if rsi_val < 30 else '-1.0 (>70)' if rsi_val > 70 else 'no signal'}")
            print(f"  • Williams %R {wr_val:.1f} → {'+0.8 (<-80)' if wr_val < -80 else '-0.8 (>-20)' if wr_val > -20 else 'no signal'}")
            print(f"  • Stoch {stoch_k_val:.1f} → {'+0.5 (<20)' if stoch_k_val < 20 else '-0.5 (>80)' if stoch_k_val > 80 else 'no signal'}")
    else:  # balanced
        print(f"  • MACD {cross:+.4f} → {'  +1.0' if cross > 0 else '  -1.0'}")
        print(f"  • RSI {rsi_val:.1f} → {'+1.0 (<35)' if rsi_val < 35 else '-1.0 (>70)' if rsi_val > 70 else 'no signal'}")
        print(f"  • BB → {'+0.7 (below lower)' if price < lower_val else '-0.7 (above upper)' if price > upper_val else '0 (inside)'}")
        if is_trending:
            print(f"  • ADX boost → {'+0.4' if cross > 0 and di_trend_bullish else '-0.4' if cross < 0 and not di_trend_bullish else '0 (mixed)'}")
        print(f"  • Williams %R {wr_val:.1f} → {'+0.4 (<-75)' if wr_val < -75 else '-0.4 (>-25)' if wr_val > -25 else 'no signal'}")
    if combined_sent != 0.0:
        contrib = min(0.5, combined_sent * 0.8) if combined_sent > 0.25 else (max(-0.5, combined_sent * 0.8) if combined_sent < -0.25 else 0.0)
        if contrib != 0.0:
            print(f"  • Sentiment {combined_sent:+.3f} → {contrib:+.3f}")
    if obv_rising is not None:
        print(f"  • OBV {'rising ↑ +0.25' if obv_rising else 'falling ↓ -0.25'} (volume confirmation)")
    if vol_ratio_val is not None:
        if vol_ratio_val > 2.0:
            print(f"  • Volume ratio {vol_ratio_val:.2f}x → conf +0.10")
        elif vol_ratio_val > 1.5:
            print(f"  • Volume ratio {vol_ratio_val:.2f}x → conf +0.05")
        elif vol_ratio_val < 0.5:
            print(f"  • Volume ratio {vol_ratio_val:.2f}x → conf -0.10 (thin volume)")
    print()


def _infer_scope(ticker: str) -> str:
    """Infer the market scope from the ticker symbol."""
    t = ticker.upper()
    if t.endswith(".IS"):
        return "BIST30"
    if t.endswith("=X") or t.endswith("=F"):
        return "FOREX"
    return "NASDAQ"


def main() -> None:
    parser = argparse.ArgumentParser(description="Full indicator + MTF debug for any ticker.")
    parser.add_argument("tickers", nargs="+", help="Ticker(s), e.g. GARAN.IS AAPL EURUSD=X")
    parser.add_argument("--strategy", "-s",
                        choices=["balanced", "momentum", "mean_reversion", "all"],
                        default="all")
    parser.add_argument("--scope", default=None,
                        help="Market scope for macro sentiment (BIST30/NASDAQ/FOREX). "
                             "Auto-detected from ticker if not set.")
    parser.add_argument("--no-sentiment", action="store_true",
                        help="Skip news fetches (faster, works offline)")
    args = parser.parse_args()

    strategies = (
        ["balanced", "momentum", "mean_reversion"]
        if args.strategy == "all"
        else [args.strategy]
    )

    # Auto-detect scope from the first ticker if not explicitly set
    scope = args.scope or _infer_scope(args.tickers[0])

    # Fetch macro sentiment once for the whole run
    if args.no_sentiment:
        macro = {"global": 0.0, "market": 0.0}
    else:
        print(f"Fetching macro sentiment ({scope}) …", end=" ", flush=True)
        macro = _fetch_macro(scope)
        print(f"global={macro.get('global', 0.0):+.3f}  market={macro.get('market', 0.0):+.3f}")

    for ticker in args.tickers:
        # Fetch news sentiment
        if args.no_sentiment:
            ticker_sentiment = 0.0
        else:
            print(f"Fetching news for {ticker} …", end=" ", flush=True)
            ticker_sentiment = _fetch_sentiment(ticker)
            print(f"{ticker_sentiment:+.3f}")

        # Fetch all timeframes once
        print(f"Fetching data for {ticker} …")
        data_by_tf: dict[str, pd.DataFrame] = {}
        for tf, (period, interval) in TIMEFRAME_CONFIG.items():
            d = fetch_history(CandleRequest(ticker=ticker, period=period, interval=interval))
            if not d.empty:
                data_by_tf[tf] = d
                print(f"  {tf}: {len(d)} bars")

        if not data_by_tf:
            print(f"  ERROR: no data for {ticker}")
            continue

        for strat in strategies:
            analyze_ticker(ticker, strat, ticker_sentiment, macro, data_by_tf)


if __name__ == "__main__":
    main()
