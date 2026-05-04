#!/usr/bin/env python3
"""
debug_signals.py — inspect indicator values, volume, macro context,
                   and multi-timeframe decision for any ticker.

Usage:
    python debug_signals.py TICKER [TICKER2 ...] [--strategy balanced|momentum|mean_reversion|all]
    python debug_signals.py GARAN.IS
    python debug_signals.py AVGO --strategy momentum
    python debug_signals.py AVGO --strategy momentum --params optimized_params_momentum.json
    python debug_signals.py EURUSD=X --no-sentiment
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def print_longterm_context(ticker: str) -> None:
    """Long-horizon (weekly + monthly) context for swing/position decisions.

    Independent of the bot's MTF aggregator — purely informational for human
    holds of 2+ weeks. Fetches 1wk and 1mo candles directly.
    """
    print(f"\n{_BAR}")
    print(f"  {ticker}  |  Long-term context (for 2+ week holds)")
    print(_BAR)

    for tf_label, period, interval in (("Weekly", "5y", "1wk"), ("Monthly", "max", "1mo")):
        d = fetch_history(CandleRequest(ticker=ticker, period=period, interval=interval))
        if d.empty or len(d) < 20:
            print(f"  {tf_label:<8}: insufficient data")
            continue

        price = float(d["Close"].iloc[-1])
        sma_short = calculate_sma(d, 20)
        sma_long = calculate_sma(d, 50 if tf_label == "Weekly" else 12)
        rsi = calculate_rsi(d)
        macd, sig = calculate_macd(d)

        s_short = float(sma_short.dropna().iloc[-1]) if not sma_short.dropna().empty else float("nan")
        s_long = float(sma_long.dropna().iloc[-1]) if not sma_long.dropna().empty else float("nan")
        rsi_v = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50.0
        macd_v = float(macd.dropna().iloc[-1]) if not macd.dropna().empty else 0.0
        sig_v = float(sig.dropna().iloc[-1]) if not sig.dropna().empty else 0.0
        cross = macd_v - sig_v

        trend = "UPTREND" if price > s_short > s_long else \
                "DOWNTREND" if price < s_short < s_long else "MIXED"
        rsi_lbl = "OVERSOLD" if rsi_v < 30 else ("OVERBOUGHT" if rsi_v > 70 else "neutral")
        macd_lbl = "BULLISH" if cross > 0 else "BEARISH"

        sma_short_lbl = "20" if tf_label == "Weekly" else "20"
        sma_long_lbl = "50" if tf_label == "Weekly" else "12"

        print(f"\n  {tf_label} ({len(d)} bars):")
        print(f"    Price        : {price:.4f}")
        print(f"    SMA{sma_short_lbl}/SMA{sma_long_lbl} : {s_short:.4f} / {s_long:.4f}  [{trend}]")
        print(f"    RSI(14)      : {rsi_v:.2f}  [{rsi_lbl}]")
        print(f"    MACD cross   : {cross:+.4f}  [{macd_lbl}]")

        # 52-period range position (52 weeks for weekly, 12 months for monthly)
        lookback = min(52 if tf_label == "Weekly" else 12, len(d))
        window = d["Close"].tail(lookback)
        hi, lo = float(window.max()), float(window.min())
        if hi > lo:
            pos = (price - lo) / (hi - lo) * 100
            range_lbl = "near HIGH" if pos > 80 else ("near LOW" if pos < 20 else "mid-range")
            range_name = "52-week" if tf_label == "Weekly" else "12-month"
            print(f"    {range_name} range: {lo:.4f} … {hi:.4f}  position={pos:.0f}%  [{range_lbl}]")


def analyze_ticker(
    ticker: str,
    strategy: str,
    ticker_sentiment: float,
    macro: dict[str, float],
    data_by_tf: dict[str, pd.DataFrame],
    params: dict | None = None,
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

    params_label = "  [optimised params]" if params else "  [default params]"
    print(f"\n{_BAR}")
    print(f"  {ticker}  |  strategy={strategy}{params_label}")
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
        params=params,
    )

    tf_labels = {
        "1mo": "macro trend",
        "1wk": "swing trend",
        "1d":  "trend frame",
        "1h":  "setup frame",
        "15m": "entry frame",
    }
    for tf_key in ("1mo", "1wk", "1d", "1h", "15m"):
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
    p = params or {}
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
        params=params,
    )
    threshold = p.get("signal_threshold", 1.3)
    obv_w     = p.get("obv_weight", 0.25)
    sent_cap  = p.get("sentiment_cap", 0.5)
    print(f"  Raw score: {raw:+.4f}  (threshold ±{threshold:.2f})")
    if strategy == "momentum":
        macd_bull = p.get("m_macd_bull", 1.5)
        macd_bear = p.get("m_macd_bear", 1.0)
        rsi_thresh = p.get("m_rsi_thresh", 55.0)
        rsi_bull = p.get("m_rsi_bull", 0.8)
        rsi_ob_pen = p.get("m_rsi_ob_penalty", 0.7)
        wr_score = p.get("m_wr_score", 0.4)
        di_score = p.get("m_di_score", 0.5)
        stoch_thresh = p.get("m_stoch_thresh", 60.0)
        stoch_score = p.get("m_stoch_score", 0.3)
        if not is_trending:
            print("  ✗ ADX<20 — not trending, score zeroed")
        else:
            print(f"  • MACD {cross:+.4f} → {f'+{macd_bull:.2f}' if cross > 0 else f'-{macd_bear:.2f}'}")
            rsi_note = f"+{rsi_bull:.2f} (>{rsi_thresh:.0f})" if rsi_val > rsi_thresh else "no boost"
            if rsi_val > 78:
                rsi_note += f" -{rsi_ob_pen:.2f} (>78 overbought)"
            print(f"  • RSI {rsi_val:.1f} → {rsi_note}")
            print(f"  • Williams %R {wr_val:.1f} → {f'+{wr_score:.2f}' if wr_val > -30 else f'-{wr_score:.2f} (losing steam)' if wr_val < -70 else 'no signal'}")
            print(f"  • DI align → {f'+{di_score:.2f}' if di_trend_bullish and cross > 0 else f'-{di_score:.2f}' if not di_trend_bullish and cross < 0 else 'mixed'}")
            print(f"  • Stoch {stoch_k_val:.1f} → {f'+{stoch_score:.2f}' if stoch_k_val > stoch_thresh else 'no signal'}{f' -{stoch_score:.2f} (top)' if stoch_k_val > 85 else ''}")
    elif strategy == "mean_reversion":
        adx_block = p.get("mr_adx_block", 35.0)
        bb_score  = p.get("mr_bb_score", 1.5)
        rsi_score = p.get("mr_rsi_score", 1.0)
        wr_score  = p.get("mr_wr_score", 0.8)
        stoch_score = p.get("mr_stoch_score", 0.5)
        if is_trending and adx_val > adx_block:
            print(f"  ✗ ADX>{adx_block:.0f} strong trend — mean reversion blocked")
        else:
            bb_c = f"+{bb_score:.2f} (below lower)" if price < lower_val else f"-{bb_score:.2f} (above upper)" if price > upper_val else "0 (inside)"
            print(f"  • BB → {bb_c}")
            print(f"  • RSI {rsi_val:.1f} → {f'+{rsi_score:.2f} (<30)' if rsi_val < 30 else f'-{rsi_score:.2f} (>70)' if rsi_val > 70 else 'no signal'}")
            print(f"  • Williams %R {wr_val:.1f} → {f'+{wr_score:.2f} (<-80)' if wr_val < -80 else f'-{wr_score:.2f} (>-20)' if wr_val > -20 else 'no signal'}")
            print(f"  • Stoch {stoch_k_val:.1f} → {f'+{stoch_score:.2f} (<20)' if stoch_k_val < 20 else f'-{stoch_score:.2f} (>80)' if stoch_k_val > 80 else 'no signal'}")
    else:  # balanced
        macd_score  = p.get("b_macd_score", 1.0)
        rsi_os      = p.get("b_rsi_os_thresh", 35.0)
        rsi_score   = p.get("b_rsi_score", 1.0)
        bb_score    = p.get("b_bb_score", 0.7)
        adx_boost   = p.get("b_adx_boost", 0.4)
        wr_score    = p.get("b_wr_score", 0.4)
        print(f"  • MACD {cross:+.4f} → {f'+{macd_score:.2f}' if cross > 0 else f'-{macd_score:.2f}'}")
        print(f"  • RSI {rsi_val:.1f} → {f'+{rsi_score:.2f} (<{rsi_os:.0f})' if rsi_val < rsi_os else f'-{rsi_score:.2f} (>70)' if rsi_val > 70 else 'no signal'}")
        print(f"  • BB → {f'+{bb_score:.2f} (below lower)' if price < lower_val else f'-{bb_score:.2f} (above upper)' if price > upper_val else '0 (inside)'}")
        if is_trending:
            print(f"  • ADX boost → {f'+{adx_boost:.2f}' if cross > 0 and di_trend_bullish else f'-{adx_boost:.2f}' if cross < 0 and not di_trend_bullish else '0 (mixed)'}")
        print(f"  • Williams %R {wr_val:.1f} → {f'+{wr_score:.2f} (<-75)' if wr_val < -75 else f'-{wr_score:.2f} (>-25)' if wr_val > -25 else 'no signal'}")
    if combined_sent != 0.0:
        contrib = min(sent_cap, combined_sent * 0.8) if combined_sent > 0.25 else (max(-sent_cap, combined_sent * 0.8) if combined_sent < -0.25 else 0.0)
        if contrib != 0.0:
            print(f"  • Sentiment {combined_sent:+.3f} → {contrib:+.3f}  (cap={sent_cap:.2f})")
    if obv_rising is not None:
        print(f"  • OBV {'rising ↑' if obv_rising else 'falling ↓'} {'+' if obv_rising else '-'}{obv_w:.2f} (volume confirmation)")
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
    parser.add_argument("--params", "-p", default=None,
                        help="Path to optimized params JSON (from optimize.py --save). "
                             "If omitted, default weights are used.")
    args = parser.parse_args()

    # Load optimized params if provided
    opt_params: dict | None = None
    if args.params:
        path = Path(args.params)
        if not path.exists():
            print(f"ERROR: params file not found: {path}")
            raise SystemExit(1)
        data = json.loads(path.read_text())
        # Support both raw dict and the format saved by optimize.py (has "params" key)
        opt_params = data.get("params", data)
        strategy_hint = data.get("strategy")
        print(f"Loaded params from {path}" +
              (f"  [strategy={strategy_hint}]" if strategy_hint else ""))
        if args.strategy == "all" and strategy_hint:
            print(f"  (tip: run with --strategy {strategy_hint} to match the optimised strategy)")

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

        # Fetch all timeframes once. Add 1wk/1mo for swing/position context —
        # the bot itself only fetches the frames in TIMEFRAME_CONFIG.
        print(f"Fetching data for {ticker} …")
        data_by_tf: dict[str, pd.DataFrame] = {}
        extra_tfs = {"1wk": ("5y", "1wk"), "1mo": ("max", "1mo")}
        for tf, (period, interval) in {**TIMEFRAME_CONFIG, **extra_tfs}.items():
            d = fetch_history(CandleRequest(ticker=ticker, period=period, interval=interval))
            if not d.empty:
                data_by_tf[tf] = d
                print(f"  {tf}: {len(d)} bars")

        if not data_by_tf:
            print(f"  ERROR: no data for {ticker}")
            continue

        print_longterm_context(ticker)

        for strat in strategies:
            analyze_ticker(ticker, strat, ticker_sentiment, macro, data_by_tf, params=opt_params)


if __name__ == "__main__":
    main()
