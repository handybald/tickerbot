from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tickerbot.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    calculate_stochastic,
    calculate_williams_r,
)


TIMEFRAME_CONFIG = {
    "15m": ("60d", "15m"),
    "1h":  ("180d", "1h"),
    "1d":  ("2y", "1d"),
}

# ADX threshold: below this value the market is considered range-bound.
_ADX_TREND_THRESHOLD = 20.0
# ATR% cap: when volatility is extreme, scale down confidence.
_MAX_ATR_PCT = 0.04   # 4 % of price


@dataclass
class StrategyChoice:
    name: str
    timeframe: str
    score: float


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
    # Signal generation
    # ------------------------------------------------------------------

    def signal(
        self,
        strategy: str,
        timeframe: str,
        data: pd.DataFrame,
    ) -> tuple[str, float]:
        """Return (action, confidence) for a single bar series.

        action    : "BUY" | "SELL" | "HOLD"
        confidence: [0, 1]
        """
        if data.empty or len(data) < 50:
            return "HOLD", 0.0

        price = float(data["Close"].iloc[-1])

        # Core indicators
        rsi = calculate_rsi(data)
        rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50.0

        macd, sig_line = calculate_macd(data)
        macd_last = float(macd.iloc[-1]) if not macd.empty else 0.0
        sig_last = float(sig_line.iloc[-1]) if not sig_line.empty else 0.0
        macd_cross = macd_last - sig_last   # positive = bullish

        upper, lower = calculate_bollinger_bands(data)
        upper_val = float(upper.iloc[-1]) if not upper.empty else price * 1.05
        lower_val = float(lower.iloc[-1]) if not lower.empty else price * 0.95

        # ADX – only valid when High/Low columns exist
        adx_val = 25.0   # default: assume trending if data unavailable
        di_trend_bullish = True
        try:
            adx_series, di_plus, di_minus = calculate_adx(data)
            if not adx_series.dropna().empty:
                adx_val = float(adx_series.iloc[-1])
                di_trend_bullish = float(di_plus.iloc[-1]) > float(di_minus.iloc[-1])
        except Exception:
            pass

        # Williams %R
        wr_val = -50.0   # default neutral
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

        # ATR-based confidence penalty (high volatility → lower confidence)
        atr_penalty = 0.0
        try:
            atr = calculate_atr(data)
            if not atr.dropna().empty and price > 0:
                atr_pct = float(atr.iloc[-1]) / price
                if atr_pct > _MAX_ATR_PCT:
                    atr_penalty = min(0.25, (atr_pct - _MAX_ATR_PCT) / _MAX_ATR_PCT * 0.25)
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
        )

        confidence = min(abs(score) / 3.5, 1.0) - atr_penalty
        confidence = max(0.0, confidence)

        if score >= 1.3:
            return "BUY", confidence
        if score <= -1.3:
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
    ) -> float:
        score = 0.0

        if strategy == "momentum":
            # Momentum: rides trends – require at least moderate trend strength.
            if not is_trending:
                return 0.0
            if macd_cross > 0:
                score += 1.5
            else:
                score -= 1.0
            if rsi > 55:
                score += 0.8
            if rsi > 78:
                score -= 0.7   # too overbought even for momentum
            if wr > -30:       # near overbought – momentum continuation
                score += 0.4
            if wr < -70:       # losing steam
                score -= 0.4
            if di_trend_bullish and macd_cross > 0:
                score += 0.5
            elif not di_trend_bullish and macd_cross < 0:
                score -= 0.5
            if stoch_k > 60:
                score += 0.3
            if stoch_k > 85:
                score -= 0.3   # short-term top

        elif strategy == "mean_reversion":
            # Mean Reversion: fades extremes – blocked in very strong trends.
            if is_trending and adx > 35:
                return 0.0
            if price < lower_bb:
                score += 1.5
            elif price > upper_bb:
                score -= 1.5
            if rsi < 30:
                score += 1.0
            elif rsi > 70:
                score -= 1.0
            if wr < -80:
                score += 0.8
            elif wr > -20:
                score -= 0.8
            if stoch_k < 20:
                score += 0.5
            elif stoch_k > 80:
                score -= 0.5

        else:   # balanced (default)
            if macd_cross > 0:
                score += 1.0
            else:
                score -= 1.0
            if rsi < 35:
                score += 1.0
            elif rsi > 70:
                score -= 1.0
            if price < lower_bb:
                score += 0.7
            elif price > upper_bb:
                score -= 0.7
            if is_trending:
                if macd_cross > 0 and di_trend_bullish:
                    score += 0.4
                elif macd_cross < 0 and not di_trend_bullish:
                    score -= 0.4
            if wr < -75:
                score += 0.4
            elif wr > -25:
                score -= 0.4

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
