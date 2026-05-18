"""
backtester.py — bar-by-bar event-driven backtester.

No look-ahead bias:
  - Signal computed using data[0 .. i] (close of bar i)
  - Entry at open of bar i+1
  - Exit when stop-loss / take-profit / trailing-stop triggers at open of next bar

This is realistic for EOD strategies: you see today's close, place a market order
that fills at tomorrow's open.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    pnl: float          # absolute, after fees
    pnl_pct: float      # relative to entry cost
    reason: str         # stop_loss | take_profit | trailing_stop | signal_exit | eod


@dataclass
class BacktestResult:
    total_return: float      # (final_equity - initial) / initial
    hit_rate: float          # profitable trades / total trades
    avg_pnl_pct: float       # mean trade P&L %
    max_drawdown: float      # peak-to-trough (negative)
    sharpe: float            # annualised Sharpe on daily equity returns
    profit_factor: float     # sum(wins) / sum(losses)
    num_trades: int
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Trades={self.num_trades:3d}  "
            f"Win={self.hit_rate*100:5.1f}%  "
            f"Return={self.total_return*100:+7.2f}%  "
            f"Sharpe={self.sharpe:5.2f}  "
            f"MaxDD={self.max_drawdown*100:6.2f}%  "
            f"PF={self.profit_factor:.2f}"
        )


_EMPTY = BacktestResult(
    total_return=0.0, hit_rate=0.0, avg_pnl_pct=0.0,
    max_drawdown=0.0, sharpe=0.0, profit_factor=0.0, num_trades=0,
)


class Backtester:
    """
    Runs a single-asset bar-by-bar backtest using StrategyManager.signal().

    Parameters
    ----------
    fee_pct      : round-trip fee as fraction of trade value (0.001 = 0.1%)
    slippage_pct : one-way slippage as fraction of price (0.0005 = 0.05%)
    """

    def __init__(self, fee_pct: float = 0.001, slippage_pct: float = 0.0005) -> None:
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

    def run(
        self,
        data,                          # pd.DataFrame with OHLCV, DatetimeIndex
        strategy_name: str,
        timeframe: str = "1d",
        params: dict | None = None,
        stop_loss: float = 0.01,
        take_profit: float = 0.015,
        trailing_stop: float = 0.008,
        min_confidence: float = 0.45,
        cash_pct: float = 0.10,        # fraction of equity per trade
        initial_cash: float = 100_000.0,
    ) -> BacktestResult:
        import pandas as pd
        from tickerbot.core.strategy import StrategyManager

        if data.empty or len(data) < 55:
            return _EMPTY

        mgr = StrategyManager()
        has_open = "Open" in data.columns

        cash = initial_cash
        pos: dict | None = None
        trades: list[Trade] = []
        equity_curve: list[float] = [initial_cash]

        for i in range(50, len(data) - 1):
            hist = data.iloc[: i + 1]          # strictly past data at close of bar i
            close_i   = float(data["Close"].iloc[i])
            open_next = float(data["Open"].iloc[i + 1]) if has_open else close_i

            # ── Manage open position ─────────────────────────────────────
            if pos is not None:
                pos["high"] = max(pos["high"], close_i)
                pnl_pct     = (close_i - pos["entry"]) / pos["entry"]
                trail_pct   = (close_i - pos["high"]) / pos["high"]

                reason = None
                if pnl_pct <= -abs(stop_loss):
                    reason = "stop_loss"
                elif pnl_pct >= abs(take_profit):
                    reason = "take_profit"
                elif trail_pct <= -abs(trailing_stop):
                    reason = "trailing_stop"

                if reason:
                    exit_px = open_next * (1 - self.slippage_pct)
                    fee     = pos["qty"] * exit_px * self.fee_pct
                    proceeds = pos["qty"] * exit_px - fee
                    cash    += proceeds
                    pnl     = proceeds - pos["cost"]
                    trades.append(Trade(
                        entry_bar=pos["entry_bar"], exit_bar=i + 1,
                        entry_price=pos["entry"], exit_price=exit_px,
                        pnl=pnl, pnl_pct=pnl / pos["cost"], reason=reason,
                    ))
                    pos = None

            # ── Signal on closed bar i ───────────────────────────────────
            if pos is None:
                action, conf = mgr.signal(
                    strategy_name, timeframe, hist,
                    sentiment=0.0, params=params,
                )
                if action == "BUY" and conf >= min_confidence:
                    entry_px = open_next * (1 + self.slippage_pct)
                    equity   = cash + (pos["qty"] * close_i if pos else 0.0)
                    budget   = equity * cash_pct
                    qty      = budget / entry_px
                    fee      = qty * entry_px * self.fee_pct
                    cost     = qty * entry_px + fee
                    if cost <= cash * 0.99:   # leave 1% buffer
                        cash -= cost
                        pos = {
                            "entry_bar": i + 1,
                            "entry": entry_px,
                            "qty": qty,
                            "cost": cost,
                            "high": entry_px,
                        }

            # Equity snapshot
            mark = pos["qty"] * close_i if pos else 0.0
            equity_curve.append(cash + mark)

        # ── Close any open position at end of data ───────────────────────
        if pos is not None:
            final_px = float(data["Close"].iloc[-1]) * (1 - self.slippage_pct)
            fee      = pos["qty"] * final_px * self.fee_pct
            proceeds = pos["qty"] * final_px - fee
            cash    += proceeds
            pnl      = proceeds - pos["cost"]
            trades.append(Trade(
                entry_bar=pos["entry_bar"], exit_bar=len(data) - 1,
                entry_price=pos["entry"], exit_price=final_px,
                pnl=pnl, pnl_pct=pnl / pos["cost"], reason="eod",
            ))
            equity_curve.append(cash)

        return _metrics(trades, equity_curve, initial_cash)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(
    trades: list[Trade],
    equity: list[float],
    initial: float,
) -> BacktestResult:
    import numpy as np

    n = len(trades)
    if n == 0:
        return _EMPTY

    wins   = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]

    total_return = (equity[-1] - initial) / initial if initial > 0 else 0.0
    hit_rate     = len(wins) / n
    avg_pnl_pct  = float(np.mean([t.pnl_pct for t in trades]))
    profit_factor = (sum(wins) / -sum(losses)) if losses else float("inf")

    # Max drawdown from equity curve
    eq = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / np.where(peak > 0, peak, 1)
    max_drawdown = float(dd.min())

    # Annualised Sharpe on daily equity returns
    daily_ret = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * math.sqrt(252))
    else:
        sharpe = 0.0

    return BacktestResult(
        total_return=total_return,
        hit_rate=hit_rate,
        avg_pnl_pct=avg_pnl_pct,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        profit_factor=profit_factor,
        num_trades=n,
        trades=trades,
        equity_curve=list(eq),
    )
