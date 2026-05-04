from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta

from .broker import BrokerAdapter
from .store import TradeStore


def format_daily_report(now: datetime, store: TradeStore, broker: BrokerAdapter, latest_prices: dict[str, float]) -> str:
    trades = store.get_trades_for_day(now)
    return _format_report_from_trades(
        title=f"Daily Paper Trading Report ({now.strftime('%Y-%m-%d')})",
        trades=trades,
        broker=broker,
        latest_prices=latest_prices,
    )


def format_weekly_report(
    now: datetime,
    store: TradeStore,
    broker: BrokerAdapter,
    latest_prices: dict[str, float],
) -> str:
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    trades = store.get_trades_between(week_start, now)
    iso = now.isocalendar()
    return _format_report_from_trades(
        title=f"Weekly Paper Trading Report ({iso.year}-W{iso.week:02d})",
        trades=trades,
        broker=broker,
        latest_prices=latest_prices,
    )


def _format_report_from_trades(
    title: str,
    trades: list,
    broker: BrokerAdapter,
    latest_prices: dict[str, float],
) -> str:
    buys = [t for t in trades if t.side == "BUY"]
    sells = [t for t in trades if t.side == "SELL"]

    realized, closed_pnls = _realized_pnl_and_closed_trades(trades)
    unrealized = _unrealized_pnl(broker, latest_prices)

    equity = broker.get_equity(latest_prices)
    positions = broker.get_positions()

    wins = [p for p in closed_pnls if p > 0]
    losses = [p for p in closed_pnls if p < 0]
    win_rate = (len(wins) / len(closed_pnls) * 100.0) if closed_pnls else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    expectancy = (len(wins) / len(closed_pnls) * avg_win + len(losses) / len(closed_pnls) * avg_loss) if closed_pnls else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf") if wins else 0.0
    best_trade = max(closed_pnls) if closed_pnls else 0.0
    worst_trade = min(closed_pnls) if closed_pnls else 0.0
    max_drawdown = _max_drawdown(closed_pnls)

    lines = [
        title,
        f"Trades: {len(trades)} (BUY: {len(buys)}, SELL: {len(sells)})",
        f"Closed Trades: {len(closed_pnls)} | Win Rate: {win_rate:.1f}%",
        f"Realized PnL: {realized:.2f}",
        f"Unrealized PnL: {unrealized:.2f}",
        f"Expectancy/Trade: {expectancy:.2f}",
        f"Profit Factor: {profit_factor:.2f}" if profit_factor != float("inf") else "Profit Factor: inf",
        f"Best/Worst Closed Trade: {best_trade:.2f} / {worst_trade:.2f}",
        f"Max Drawdown (closed equity): {max_drawdown:.2f}",
        f"Cash: {broker.get_cash():.2f}",
        f"Equity: {equity:.2f}",
        f"Open Positions: {len(positions)}",
    ]

    if wins or losses:
        lines.append(f"Avg Win/Loss: {avg_win:.2f} / {avg_loss:.2f}")

    if positions:
        lines.append("Positions:")
        for ticker, pos in sorted(positions.items()):
            mark = latest_prices.get(ticker, pos.average_price)
            upnl = (mark - pos.average_price) * pos.quantity
            lines.append(
                f"- {ticker}: qty={pos.quantity}, avg={pos.average_price:.2f}, mark={mark:.2f}, uPnL={upnl:.2f}"
            )

    if trades:
        lines.append("Recent Trades:")
        for trade in trades[-8:]:
            lines.append(
                f"- {trade.timestamp.strftime('%H:%M')} {trade.side} {trade.ticker} x{trade.quantity} @ {trade.price:.2f} fee={trade.fee:.2f}"
            )

    return "\n".join(lines)


def format_status_report(strategy: str, timeframe: str, broker: BrokerAdapter) -> str:
    positions = broker.get_positions()
    lines = [
        "Bot Status",
        f"Strategy: {strategy}",
        f"Timeframe: {timeframe}",
        f"Cash: {broker.get_cash():.2f}",
        f"Open Positions: {len(positions)}",
    ]
    return "\n".join(lines)


def _realized_pnl_and_closed_trades(trades):
    lots = defaultdict(deque)
    realized = 0.0
    closed_pnls = []

    for t in trades:
        if t.quantity <= 0:
            continue
        if t.side == "BUY":
            unit_cost = (t.gross_value + t.fee) / t.quantity if t.gross_value else (t.price + (t.fee / t.quantity))
            lots[t.ticker].append([t.quantity, unit_cost])
            continue

        if t.side != "SELL":
            continue

        qty_left = t.quantity
        unit_proceeds = (t.gross_value - t.fee) / t.quantity if t.gross_value else (t.price - (t.fee / t.quantity))
        trade_realized = 0.0

        while qty_left > 0 and lots[t.ticker]:
            lot_qty, lot_cost = lots[t.ticker][0]
            match_qty = min(qty_left, lot_qty)
            trade_realized += (unit_proceeds - lot_cost) * match_qty
            lot_qty -= match_qty
            qty_left -= match_qty

            if lot_qty == 0:
                lots[t.ticker].popleft()
            else:
                lots[t.ticker][0][0] = lot_qty

        if t.quantity > qty_left:
            realized += trade_realized
            closed_pnls.append(trade_realized)

    return realized, closed_pnls


def _unrealized_pnl(broker: BrokerAdapter, latest_prices: dict[str, float]) -> float:
    total = 0.0
    for ticker, pos in broker.get_positions().items():
        mark = latest_prices.get(ticker, pos.average_price)
        total += (mark - pos.average_price) * pos.quantity
    return total


def _max_drawdown(closed_pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in closed_pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd
