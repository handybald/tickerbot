from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tickerbot.indicators import calculate_bollinger_bands, calculate_macd, calculate_rsi


TIMEFRAME_CONFIG = {
    "15m": ("60d", "15m"),
    "1h": ("180d", "1h"),
    "1d": ("2y", "1d"),
}


@dataclass
class StrategyChoice:
    name: str
    timeframe: str
    score: float


class StrategyManager:
    def __init__(self) -> None:
        self.candidates = ["balanced", "momentum", "mean_reversion"]

    def choose(
        self,
        timeframe_data: dict[str, dict[str, pd.DataFrame]],
        performance_bias: dict[str, float] | None = None,
    ) -> StrategyChoice:
        best = StrategyChoice(name="balanced", timeframe="1h", score=-10**9)
        for strategy in self.candidates:
            for timeframe, ticker_to_data in timeframe_data.items():
                score = self._score_candidate(strategy, ticker_to_data)
                if performance_bias:
                    key = f"{strategy}|{timeframe}"
                    score += performance_bias.get(key, 0.0)
                if score > best.score:
                    best = StrategyChoice(name=strategy, timeframe=timeframe, score=score)
        return best

    def signal(self, strategy: str, timeframe: str, data: pd.DataFrame) -> tuple[str, float]:
        if data.empty or len(data) < 50:
            return "HOLD", 0.0

        rsi = calculate_rsi(data).iloc[-1]
        macd, signal_line = calculate_macd(data)
        macd_last = macd.iloc[-1]
        sig_last = signal_line.iloc[-1]
        upper, lower = calculate_bollinger_bands(data)
        price = data["Close"].iloc[-1]

        score = 0.0
        if strategy == "momentum":
            if macd_last > sig_last:
                score += 1.5
            if rsi > 55:
                score += 0.8
            if rsi > 75:
                score -= 0.7
        elif strategy == "mean_reversion":
            if price < lower.iloc[-1]:
                score += 1.5
            if price > upper.iloc[-1]:
                score -= 1.5
            if rsi < 35:
                score += 0.8
            if rsi > 65:
                score -= 0.8
        else:  # balanced
            if macd_last > sig_last:
                score += 1.0
            else:
                score -= 1.0
            if rsi < 35:
                score += 1.0
            if rsi > 70:
                score -= 1.0
            if price < lower.iloc[-1]:
                score += 0.7
            if price > upper.iloc[-1]:
                score -= 0.7

        confidence = min(abs(score) / 3.0, 1.0)
        if score >= 1.3:
            return "BUY", confidence
        if score <= -1.3:
            return "SELL", confidence
        return "HOLD", confidence

    def _score_candidate(self, strategy: str, ticker_to_data: dict[str, pd.DataFrame]) -> float:
        scores = []
        for _, data in ticker_to_data.items():
            if data.empty or len(data) < 80:
                continue
            series_score = self._simulate_quality(strategy, data)
            scores.append(series_score)
        if not scores:
            return -999.0
        return sum(scores) / len(scores)

    def _simulate_quality(self, strategy: str, data: pd.DataFrame) -> float:
        future_ret = data["Close"].pct_change(fill_method=None).shift(-1)
        rsi = calculate_rsi(data)
        macd, signal = calculate_macd(data)
        upper, lower = calculate_bollinger_bands(data)

        signal_series = pd.Series(0, index=data.index, dtype=float)

        if strategy == "momentum":
            signal_series[(macd > signal) & (rsi > 50)] = 1
            signal_series[(macd < signal) & (rsi < 50)] = -1
        elif strategy == "mean_reversion":
            signal_series[(data["Close"] < lower) & (rsi < 40)] = 1
            signal_series[(data["Close"] > upper) & (rsi > 60)] = -1
        else:
            signal_series[(macd > signal) & (rsi < 70)] = 1
            signal_series[(macd < signal) & (rsi > 30)] = -1

        aligned = signal_series * future_ret.fillna(0)
        hit_rate = (aligned > 0).sum() / max((signal_series != 0).sum(), 1)
        edge = aligned.mean() * 100
        return (hit_rate * 10.0) + edge
