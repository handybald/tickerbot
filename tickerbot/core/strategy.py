from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tickerbot.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    calculate_stochastic,
    calculate_volume_ratio,
    calculate_williams_r,
)


TIMEFRAME_CONFIG = {
    "15m": ("60d", "15m"),
    "1h":  ("180d", "1h"),
    "1d":  ("2y", "1d"),
}

# Timeframe weights for multi-timeframe combination.
# Higher timeframe carries more weight in the final decision.
_MTF_WEIGHTS = {"1d": 0.50, "1h": 0.35, "15m": 0.15}

# ADX threshold: below this value the market is considered range-bound.
_ADX_TREND_THRESHOLD = 20.0
# ATR% cap: when volatility is extreme, scale down confidence.
_MAX_ATR_PCT = 0.04   # 4 % of price


@dataclass
class StrategyChoice:
    name: str
    timeframe: str
    score: float


@dataclass
class MTFSignal:
    """Result of a multi-timeframe signal analysis."""
    action: str          # "BUY" | "SELL" | "HOLD"
    confidence: float    # [0, 1]
    alignment: int       # number of timeframes agreeing with the final direction
    total_tfs: int       # total timeframes evaluated
    breakdown: dict = field(default_factory=dict)
    # breakdown: {"1d": {"action": "BUY", "confidence": 0.7}, ...}


class StrategyManager:
    def __init__(self) -> None:
        self.candidates = ["balanced", "momentum", "mean_reversion"]

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def choose(
        self,
        timeframe_data: dict[str, dict[str, pd.DataFrame]],
        performance_bias: dict[str, float] | None = None,
    ) -> StrategyChoice:
        best = StrategyChoice(name="balanced", timeframe="1h", score=-(10 ** 9))
        for strategy in self.candidates:
            for timeframe, ticker_to_data in timeframe_data.items():
                score = self._score_candidate(strategy, ticker_to_data)
                if performance_bias:
                    key = f"{strategy}|{timeframe}"
                    score += performance_bias.get(key, 0.0)
                if score > best.score:
                    best = StrategyChoice(name=strategy, timeframe=timeframe, score=score)
        return best

    # ------------------------------------------------------------------
    # Multi-timeframe signal (triple-screen)
    # ------------------------------------------------------------------

    def mtf_signal(
        self,
        strategy: str,
        data_by_tf: dict[str, pd.DataFrame],
        ticker_sentiment: float = 0.0,
        macro_sentiment: float = 0.0,
    ) -> MTFSignal:
        """
        Triple-screen multi-timeframe signal.

        Screen 1 (1d) — Trend direction:
            The daily chart sets the allowed trading direction.
            If 1d is clearly bearish, no new BUY entries are made.
            If 1d is clearly bullish, no new SELL entries are made.

        Screen 2 (1h) — Setup confirmation:
            The hourly chart identifies the specific setup within the trend.

        Screen 3 (15m) — Entry timing:
            The 15-minute chart times the precise entry.
            Acts only when it aligns with higher-TF direction.

        Combined sentiment = ticker_sentiment * 0.6 + macro_sentiment * 0.4
        (macro context dilutes but does not dominate ticker-specific news)
        """
        # Blend ticker + macro sentiment
        combined_sentiment = ticker_sentiment * 0.6 + macro_sentiment * 0.4

        # Generate per-timeframe signals
        tf_results: dict[str, dict] = {}
        for tf, data in data_by_tf.items():
            if data.empty or len(data) < 50:
                continue
            action, conf = self.signal(
                strategy, tf, data, sentiment=combined_sentiment
            )
            tf_results[tf] = {"action": action, "confidence": conf}

        if not tf_results:
            return MTFSignal(action="HOLD", confidence=0.0, alignment=0,
                             total_tfs=0, breakdown={})

        # ── Triple-screen rule ─────────────────────────────────────────
        # The highest available timeframe sets the allowed direction.
        trend_tf = next(
            (tf for tf in ("1d", "1h", "15m") if tf in tf_results), None
        )
        trend_action = tf_results[trend_tf]["action"] if trend_tf else "HOLD"

        # Weighted directional score
        weighted_score = 0.0
        total_weight = 0.0
        for tf, sig in tf_results.items():
            w = _MTF_WEIGHTS.get(tf, 0.15)
            vote = 1.0 if sig["action"] == "BUY" else (-1.0 if sig["action"] == "SELL" else 0.0)
            weighted_score += vote * sig["confidence"] * w
            total_weight += w
        if total_weight > 0:
            weighted_score /= total_weight

        # Block signals that fight the dominant trend TF
        if trend_action == "SELL" and weighted_score > 0.05:
            return MTFSignal(action="HOLD", confidence=0.0, alignment=0,
                             total_tfs=len(tf_results), breakdown=tf_results)
        if trend_action == "BUY" and weighted_score < -0.05:
            return MTFSignal(action="HOLD", confidence=0.0, alignment=0,
                             total_tfs=len(tf_results), breakdown=tf_results)

        # Determine final action
        if weighted_score >= 0.20:
            final_action = "BUY"
        elif weighted_score <= -0.20:
            final_action = "SELL"
        else:
            final_action = "HOLD"

        # Count alignment
        alignment = sum(
            1 for s in tf_results.values()
            if s["action"] == final_action
        )

        # Confidence: normalized weighted score + alignment bonus
        base_conf = min(abs(weighted_score) / 0.50, 1.0)
        alignment_bonus = (alignment - 1) * 0.08 if alignment > 1 else 0.0
        confidence = min(base_conf + alignment_bonus, 1.0)

        return MTFSignal(
            action=final_action,
            confidence=confidence,
            alignment=alignment,
            total_tfs=len(tf_results),
            breakdown=tf_results,
        )

    # ------------------------------------------------------------------
    # Single-timeframe signal generation
    # ------------------------------------------------------------------

    def signal(
        self,
        strategy: str,
        timeframe: str,
        data: pd.DataFrame,
        sentiment: float = 0.0,
        params: dict | None = None,
    ) -> tuple[str, float]:
        """Return (action, confidence) for a single timeframe bar series.

        action    : "BUY" | "SELL" | "HOLD"
        confidence: [0, 1]
        sentiment : [-1, 1] combined news sentiment (ticker + macro already blended)
        """
        if data.empty or len(data) < 50:
            return "HOLD", 0.0

        price = float(data["Close"].iloc[-1])

        # ── Core price indicators ──────────────────────────────────────
        rsi = calculate_rsi(data)
        rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50.0

        macd, sig_line = calculate_macd(data)
        macd_last = float(macd.iloc[-1]) if not macd.empty else 0.0
        sig_last  = float(sig_line.iloc[-1]) if not sig_line.empty else 0.0
        macd_cross = macd_last - sig_last   # positive = bullish

        upper, lower = calculate_bollinger_bands(data)
        upper_val = float(upper.iloc[-1]) if not upper.empty else price * 1.05
        lower_val = float(lower.iloc[-1]) if not lower.empty else price * 0.95

        # ADX
        adx_val = 25.0
        di_trend_bullish = True
        try:
            adx_series, di_plus, di_minus = calculate_adx(data)
            if not adx_series.dropna().empty:
                adx_val = float(adx_series.iloc[-1])
                di_trend_bullish = float(di_plus.iloc[-1]) > float(di_minus.iloc[-1])
        except Exception:
            pass

        # Williams %R
        wr_val = -50.0
        try:
            wr = calculate_williams_r(data)
            if not wr.dropna().empty:
                wr_val = float(wr.iloc[-1])
        except Exception:
            pass

        # Stochastic
        stoch_k_val = 50.0
        try:
            stoch_k, _ = calculate_stochastic(data)
            if not stoch_k.dropna().empty:
                stoch_k_val = float(stoch_k.iloc[-1])
        except Exception:
            pass

        # ── ATR confidence penalty ─────────────────────────────────────
        atr_penalty = 0.0
        try:
            atr = calculate_atr(data)
            if not atr.dropna().empty and price > 0:
                atr_pct = float(atr.iloc[-1]) / price
                if atr_pct > _MAX_ATR_PCT:
                    atr_penalty = min(0.25, (atr_pct - _MAX_ATR_PCT) / _MAX_ATR_PCT * 0.25)
        except Exception:
            pass

        # ── Volume analysis ────────────────────────────────────────────
        # OBV trend: rising OBV = volume confirms the up move; falling = confirms down move
        obv_contribution = 0.0
        vol_conf_mod = 0.0
        _obv_w = (params or {}).get("obv_weight", 0.25)
        try:
            obv = calculate_obv(data)
            if not obv.dropna().empty and len(obv.dropna()) >= 20:
                obv_sma = obv.rolling(20).mean()
                obv_rising = float(obv.iloc[-1]) > float(obv_sma.dropna().iloc[-1])
                obv_contribution = _obv_w if obv_rising else -_obv_w
        except Exception:
            pass

        try:
            vol_ratio = calculate_volume_ratio(data)
            if not vol_ratio.dropna().empty:
                vr = float(vol_ratio.dropna().iloc[-1])
                if vr > 2.0:
                    vol_conf_mod = 0.10    # very high activity → confirms direction
                elif vr > 1.5:
                    vol_conf_mod = 0.05
                elif vr < 0.5:
                    vol_conf_mod = -0.10   # thin volume → signal less reliable
        except Exception:
            pass

        is_trending = adx_val >= _ADX_TREND_THRESHOLD

        score = self._raw_score(
            strategy=strategy,
            price=price,
            rsi=rsi_val,
            macd_cross=macd_cross,
            upper_bb=upper_val,
            lower_bb=lower_val,
            adx=adx_val,
            is_trending=is_trending,
            di_trend_bullish=di_trend_bullish,
            wr=wr_val,
            stoch_k=stoch_k_val,
            sentiment=sentiment,
            params=params,
        )

        # OBV contributes in the same direction as the current score tendency
        if score > 0:
            score += obv_contribution
        elif score < 0:
            score -= obv_contribution

        threshold = (params or {}).get("signal_threshold", 1.3)
        confidence = min(abs(score) / (threshold * 2.7), 1.0) - atr_penalty + vol_conf_mod
        confidence = max(0.0, confidence)

        if score >= threshold:
            return "BUY", confidence
        if score <= -threshold:
            return "SELL", confidence
        return "HOLD", confidence

    def _raw_score(
        self,
        strategy: str,
        price: float,
        rsi: float,
        macd_cross: float,
        upper_bb: float,
        lower_bb: float,
        adx: float,
        is_trending: bool,
        di_trend_bullish: bool,
        wr: float,
        stoch_k: float,
        sentiment: float = 0.0,
        params: dict | None = None,
    ) -> float:
        p = params or {}
        score = 0.0

        if strategy == "momentum":
            if not is_trending:
                return 0.0
            if macd_cross > 0:
                score += p.get("m_macd_bull", 1.5)
            else:
                score -= p.get("m_macd_bear", 1.0)
            if rsi > p.get("m_rsi_thresh", 55.0):
                score += p.get("m_rsi_bull", 0.8)
            if rsi > 78:
                score -= p.get("m_rsi_ob_penalty", 0.7)
            if wr > -30:
                score += p.get("m_wr_score", 0.4)
            if wr < -70:
                score -= p.get("m_wr_score", 0.4)
            if di_trend_bullish and macd_cross > 0:
                score += p.get("m_di_score", 0.5)
            elif not di_trend_bullish and macd_cross < 0:
                score -= p.get("m_di_score", 0.5)
            if stoch_k > p.get("m_stoch_thresh", 60.0):
                score += p.get("m_stoch_score", 0.3)
            if stoch_k > 85:
                score -= p.get("m_stoch_score", 0.3)

        elif strategy == "mean_reversion":
            if is_trending and adx > p.get("mr_adx_block", 35.0):
                return 0.0
            if price < lower_bb:
                score += p.get("mr_bb_score", 1.5)
            elif price > upper_bb:
                score -= p.get("mr_bb_score", 1.5)
            if rsi < 30:
                score += p.get("mr_rsi_score", 1.0)
            elif rsi > 70:
                score -= p.get("mr_rsi_score", 1.0)
            if wr < -80:
                score += p.get("mr_wr_score", 0.8)
            elif wr > -20:
                score -= p.get("mr_wr_score", 0.8)
            if stoch_k < 20:
                score += p.get("mr_stoch_score", 0.5)
            elif stoch_k > 80:
                score -= p.get("mr_stoch_score", 0.5)

        else:   # balanced
            if macd_cross > 0:
                score += p.get("b_macd_score", 1.0)
            else:
                score -= p.get("b_macd_score", 1.0)
            if rsi < p.get("b_rsi_os_thresh", 35.0):
                score += p.get("b_rsi_score", 1.0)
            elif rsi > 70:
                score -= p.get("b_rsi_score", 1.0)
            if price < lower_bb:
                score += p.get("b_bb_score", 0.7)
            elif price > upper_bb:
                score -= p.get("b_bb_score", 0.7)
            if is_trending:
                if macd_cross > 0 and di_trend_bullish:
                    score += p.get("b_adx_boost", 0.4)
                elif macd_cross < 0 and not di_trend_bullish:
                    score -= p.get("b_adx_boost", 0.4)
            if wr < -75:
                score += p.get("b_wr_score", 0.4)
            elif wr > -25:
                score -= p.get("b_wr_score", 0.4)

        # Sentiment tilt
        sent_cap = p.get("sentiment_cap", 0.5)
        if sentiment > 0.25:
            score += min(sent_cap, sentiment * 0.8)
        elif sentiment < -0.25:
            score += max(-sent_cap, sentiment * 0.8)

        return score

    # ------------------------------------------------------------------
    # Strategy scoring helpers (for auto-selection)
    # ------------------------------------------------------------------

    def _score_candidate(
        self, strategy: str, ticker_to_data: dict[str, pd.DataFrame]
    ) -> float:
        scores = []
        for _, data in ticker_to_data.items():
            if data.empty or len(data) < 80:
                continue
            series_score = self._simulate_quality(strategy, data)
            scores.append(series_score)
        if not scores:
            return -999.0
        return float(np.mean(scores))

    def _simulate_quality(self, strategy: str, data: pd.DataFrame) -> float:
        future_ret = data["Close"].pct_change(fill_method=None).shift(-1)
        rsi = calculate_rsi(data)
        macd, sig = calculate_macd(data)
        upper, lower = calculate_bollinger_bands(data)

        try:
            adx_s, di_plus, di_minus = calculate_adx(data)
            trending = adx_s >= _ADX_TREND_THRESHOLD
        except Exception:
            trending = pd.Series(True, index=data.index)

        try:
            wr = calculate_williams_r(data)
        except Exception:
            wr = pd.Series(-50.0, index=data.index)

        signal_series = pd.Series(0, index=data.index, dtype=float)

        if strategy == "momentum":
            buy_cond = (macd > sig) & (rsi > 50) & trending
            sell_cond = (macd < sig) & (rsi < 50) & trending
            signal_series[buy_cond] = 1
            signal_series[sell_cond] = -1
        elif strategy == "mean_reversion":
            buy_cond = (data["Close"] < lower) & (rsi < 40) & (wr < -75)
            sell_cond = (data["Close"] > upper) & (rsi > 60) & (wr > -25)
            signal_series[buy_cond] = 1
            signal_series[sell_cond] = -1
        else:   # balanced
            buy_cond = (macd > sig) & (rsi < 70)
            sell_cond = (macd < sig) & (rsi > 30)
            signal_series[buy_cond] = 1
            signal_series[sell_cond] = -1

        aligned = signal_series * future_ret.fillna(0)
        hit_rate = (aligned > 0).sum() / max((signal_series != 0).sum(), 1)
        edge = aligned.mean() * 100
        return float(hit_rate * 10.0 + edge)
