from __future__ import annotations

import json
import logging
import threading
import time
from csv import DictWriter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .analytics import closed_trade_stats
from .broker import PaperBrokerAdapter
from .market import (
    CandleRequest,
    fetch_history,
    get_last_price,
    get_universe,
    is_forex_ticker,
    regime_snapshot,
    sync_bist_universe,
    sync_us_universe,
    universe_cache_info,
)
from .reporting import format_daily_report, format_status_report, format_weekly_report
from .settings import RuntimeSettings, SettingsStore
from .store import TradeStore
from .strategy import StrategyManager, TIMEFRAME_CONFIG
from .telegram_interface import TelegramInterface


log = logging.getLogger(__name__)


class BotService:
    def __init__(self, token: str, chat_id: str, initial_cash: float = 100_000.0) -> None:
        self.settings_store = SettingsStore()
        self.trade_store = TradeStore()
        self.broker = PaperBrokerAdapter(initial_cash=initial_cash)
        self.strategy_manager = StrategyManager()
        self.telegram = TelegramInterface(token=token, chat_id=chat_id)
        self.istanbul_tz = ZoneInfo("Europe/Istanbul")

        self.selected_strategy = "balanced"
        self.selected_timeframe = "1h"
        self.trading_enabled = True

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_trade_cycle_at = 0.0
        self._last_signal_snapshot: list[dict] = []
        self._last_signal_time: datetime | None = None
        self._position_high_watermark: dict[str, float] = {}
        self._no_data_failures: dict[str, int] = {}
        self._no_data_block_until: dict[str, str] = {}

        self._state_key = "runtime_state_v1"
        # Day key (YYYY-MM-DD) on which the profit target halt was triggered;
        # empty string means no halt is active.
        self._profit_halt_day: str = ""
        self._restore_runtime_state()

    def run(self) -> None:
        log.info("Bot service starting in paper mode")
        self.telegram.send_message("TickerBot started in paper-trading mode.")

        polling_thread = threading.Thread(
            target=self.telegram.poll_commands,
            args=(self._handle_command,),
            daemon=True,
        )
        polling_thread.start()

        try:
            while not self._stop.is_set():
                self._run_once()
                time.sleep(20)
        finally:
            self.telegram.stop()

    def stop(self) -> None:
        self._stop.set()

    def _run_once(self) -> None:
        now_local = datetime.now(self.istanbul_tz)
        settings = self.settings_store.load()
        self._apply_execution_settings(settings)

        # Auto-clear profit halt on a new calendar day.
        today = now_local.strftime("%Y-%m-%d")
        if self._profit_halt_day and self._profit_halt_day != today:
            log.info("New day %s - clearing profit halt from %s", today, self._profit_halt_day)
            self._profit_halt_day = ""
            self._persist_runtime_state()

        self._maybe_send_daily_report(now_local, settings)
        self._maybe_send_weekly_report(now_local, settings)
        self._maybe_flatten_before_close(now_local, settings)

        if not self.trading_enabled:
            return

        now_ts = time.time()
        if now_ts - self._last_trade_cycle_at < settings.trade_period_minutes * 60:
            return

        with self._lock:
            log.info(
                "Starting trading cycle | market=%s | auto_strategy=%s",
                settings.market_scope,
                settings.auto_strategy,
            )
            self._trading_cycle(settings, now_local)
            self._last_trade_cycle_at = now_ts

    def _trading_cycle(self, settings: RuntimeSettings, now_local: datetime) -> None:
        universe = get_universe(settings.market_scope, refresh_minutes=settings.universe_refresh_minutes)
        log.info("Universe loaded | symbols=%d", len(universe))
        timeframe_data: dict[str, dict] = {}

        if settings.auto_strategy:
            candidate_timeframes = list(TIMEFRAME_CONFIG.keys())
        else:
            selected = settings.timeframe_override or self.selected_timeframe
            candidate_timeframes = [selected] if selected in TIMEFRAME_CONFIG else ["1h"]

        for tf in candidate_timeframes:
            period, interval = TIMEFRAME_CONFIG[tf]
            ticker_data = {}
            for ticker in universe:
                if self._is_ticker_blocked(ticker, now_local):
                    continue
                try:
                    data = fetch_history(CandleRequest(ticker=ticker, period=period, interval=interval))
                    if not data.empty:
                        self._clear_no_data_failure(ticker)
                        ticker_data[ticker] = data
                    else:
                        self._record_no_data_failure(ticker, settings, now_local)
                except Exception as exc:
                    log.warning("history fetch failed for %s (%s): %s", ticker, tf, exc)
                    self._record_no_data_failure(ticker, settings, now_local)
            if ticker_data:
                timeframe_data[tf] = ticker_data

        if not timeframe_data:
            log.warning("No timeframe data available; cycle skipped")
            return

        perf_bias = self._strategy_performance_bias(now_local)
        if settings.auto_strategy:
            choice = self.strategy_manager.choose(timeframe_data, performance_bias=perf_bias)
            self.selected_strategy = choice.name
            self.selected_timeframe = choice.timeframe
        else:
            self.selected_strategy = settings.strategy_override or self.selected_strategy
            if settings.timeframe_override:
                self.selected_timeframe = settings.timeframe_override
        log.info(
            "Strategy selected | strategy=%s | timeframe=%s",
            self.selected_strategy,
            self.selected_timeframe,
        )

        tf_period, tf_interval = TIMEFRAME_CONFIG.get(self.selected_timeframe, TIMEFRAME_CONFIG["1h"])
        live_data = {}
        for ticker in universe:
            if self._is_ticker_blocked(ticker, now_local):
                continue
            try:
                data = fetch_history(CandleRequest(ticker=ticker, period=tf_period, interval=tf_interval))
                if not data.empty:
                    self._clear_no_data_failure(ticker)
                    live_data[ticker] = data
                else:
                    self._record_no_data_failure(ticker, settings, now_local)
            except Exception as exc:
                log.warning("live fetch failed for %s: %s", ticker, exc)
                self._record_no_data_failure(ticker, settings, now_local)

        latest_prices = {t: float(df["Close"].iloc[-1]) for t, df in live_data.items() if not df.empty}
        if not latest_prices:
            log.warning("No live prices available; cycle skipped")
            return

        day_key = now_local.strftime("%Y-%m-%d")
        equity = self.broker.get_equity(latest_prices)
        start_equity_key = f"start_equity:{day_key}"
        day_start_equity_raw = self.trade_store.get_meta(start_equity_key)
        if not day_start_equity_raw:
            self.trade_store.set_meta(start_equity_key, str(equity))
            day_start_equity = equity
        else:
            day_start_equity = float(day_start_equity_raw)

        daily_drawdown = (day_start_equity - equity) / max(day_start_equity, 1.0)
        if daily_drawdown >= settings.max_daily_loss_pct:
            log.warning(
                "Risk halt | drawdown=%.4f limit=%.4f day=%s",
                daily_drawdown,
                settings.max_daily_loss_pct,
                day_key,
            )
            self.telegram.send_message(
                f"Risk halt triggered for {day_key}. "
                f"Drawdown {daily_drawdown:.2%} exceeded "
                f"{settings.max_daily_loss_pct:.2%}. Use /startbot tomorrow."
            )
            self.trading_enabled = False
            self._persist_runtime_state()
            return

        # Daily profit target: pause new entries once the goal is reached.
        if settings.daily_profit_target_enabled:
            daily_gain = (equity - day_start_equity) / max(day_start_equity, 1.0)
            if daily_gain >= settings.daily_profit_target_pct:
                log.info(
                    "Profit target reached | gain=%.4f target=%.4f day=%s",
                    daily_gain,
                    settings.daily_profit_target_pct,
                    day_key,
                )
                if self._profit_halt_day != day_key:
                    self._profit_halt_day = day_key
                    self.telegram.send_message(
                        f"Daily profit target reached! "
                        f"Gain {daily_gain:.2%} >= {settings.daily_profit_target_pct:.2%} "
                        f"on {day_key}. No new entries until tomorrow."
                    )
                    self._persist_runtime_state()
                # Allow exits to continue running but skip new entries.
                # We set a flag and fall through to the exit-only pass below.

        regime_ok = True
        if settings.regime_filter_enabled:
            regime_ticker = self._effective_regime_ticker(settings)
            reg = regime_snapshot(
                ticker=regime_ticker,
                fast_sma=settings.regime_fast_sma,
                slow_sma=settings.regime_slow_sma,
                rsi_min=settings.regime_rsi_min,
            )
            if reg.get("ok"):
                regime_ok = bool(reg.get("bullish"))
                log.info(
                    "Regime | ticker=%s bullish=%s price=%.2f fast=%.2f slow=%.2f rsi=%.2f",
                    regime_ticker,
                    regime_ok,
                    reg.get("price", 0.0),
                    reg.get("fast_sma", 0.0),
                    reg.get("slow_sma", 0.0),
                    reg.get("rsi", 0.0),
                )
            else:
                log.warning("Regime snapshot unavailable; proceeding without block")

        dynamic_min_conf = self._dynamic_min_confidence(settings, now_local)
        positions = self.broker.get_positions()
        buys = 0
        sells = 0
        signal_snapshot = []
        buys_today = self._buys_today(now_local)
        exits_by_rule = 0

        for ticker, data in live_data.items():
            action, confidence = self.strategy_manager.signal(self.selected_strategy, self.selected_timeframe, data)
            price = float(latest_prices[ticker])
            positions = self.broker.get_positions()
            pos = positions.get(ticker)

            # Rule-based exits run before signal gating.
            exit_reason = self._exit_reason(pos, price, now_local, settings) if pos else ""
            if exit_reason:
                result = self.broker.place_market_sell(
                    ticker=ticker,
                    price=price,
                    quantity=pos.quantity,
                    note=(
                        f"strategy={self.selected_strategy};tf={self.selected_timeframe};"
                        f"reason={exit_reason};conf={confidence:.2f}"
                    ),
                )
                if result.success and result.trade:
                    self.trade_store.record_trade(result.trade)
                    self._touch_ticker_trade_time(ticker, now_local)
                    self._persist_runtime_state()
                    sells += 1
                    exits_by_rule += 1
                    self._position_high_watermark.pop(ticker, None)
                    log.info(
                        "SELL(rule) | ticker=%s reason=%s qty=%s price=%.4f",
                        ticker,
                        exit_reason,
                        result.trade.quantity,
                        result.trade.price,
                    )
                    self.telegram.send_message(
                        f"SELL {ticker} x{result.trade.quantity} @ {result.trade.price:.2f} ({exit_reason})"
                    )
                continue

            if pos:
                self._position_high_watermark[ticker] = max(
                    self._position_high_watermark.get(ticker, pos.average_price),
                    price,
                )

            signal_snapshot.append(
                {
                    "ticker": ticker,
                    "action": action,
                    "confidence": float(confidence),
                    "price": price,
                }
            )
            if action == "HOLD" or confidence < dynamic_min_conf:
                continue

            positions = self.broker.get_positions()
            has_position = ticker in positions

            if action == "BUY":
                if self._profit_halt_day == day_key:
                    continue   # profit target reached today - no new entries
                if not regime_ok:
                    continue
                if has_position:
                    continue
                if len(positions) >= settings.max_open_positions:
                    continue
                if buys >= settings.max_new_entries_per_cycle:
                    continue
                if buys_today + buys >= settings.max_new_entries_per_day:
                    continue
                if self._in_cooldown(ticker, settings, now_local):
                    continue
                if not self._passes_liquidity_gate(ticker, data, settings):
                    continue

                trade_budget = self.broker.get_equity(latest_prices) * settings.per_trade_cash_pct
                max_position_budget = self.broker.get_equity(latest_prices) * settings.max_position_pct
                cash_budget = min(trade_budget, max_position_budget)
                result = self.broker.place_market_buy(
                    ticker=ticker,
                    price=price,
                    cash_budget=cash_budget,
                    note=f"strategy={self.selected_strategy};tf={self.selected_timeframe};conf={confidence:.2f}",
                )
                if result.success and result.trade:
                    self.trade_store.record_trade(result.trade)
                    self._touch_ticker_trade_time(ticker, now_local)
                    self._persist_runtime_state()
                    buys += 1
                    self._position_high_watermark[ticker] = result.trade.price
                    log.info(
                        "BUY filled | ticker=%s qty=%s price=%.4f fee=%.4f status=%s",
                        ticker,
                        result.trade.quantity,
                        result.trade.price,
                        result.trade.fee,
                        result.message,
                    )
                    self.telegram.send_message(
                        f"BUY {ticker} x{result.trade.quantity} @ {result.trade.price:.2f} fee={result.trade.fee:.2f} ({result.message})"
                    )

            if action == "SELL" and has_position:
                qty = positions[ticker].quantity
                result = self.broker.place_market_sell(
                    ticker=ticker,
                    price=price,
                    quantity=qty,
                    note=f"strategy={self.selected_strategy};tf={self.selected_timeframe};conf={confidence:.2f}",
                )
                if result.success and result.trade:
                    self.trade_store.record_trade(result.trade)
                    self._touch_ticker_trade_time(ticker, now_local)
                    self._persist_runtime_state()
                    sells += 1
                    self._position_high_watermark.pop(ticker, None)
                    log.info(
                        "SELL filled | ticker=%s qty=%s price=%.4f fee=%.4f status=%s",
                        ticker,
                        result.trade.quantity,
                        result.trade.price,
                        result.trade.fee,
                        result.message,
                    )
                    self.telegram.send_message(
                        f"SELL {ticker} x{result.trade.quantity} @ {result.trade.price:.2f} fee={result.trade.fee:.2f} ({result.message})"
                    )

        signal_snapshot.sort(key=lambda x: x["confidence"], reverse=True)
        self._last_signal_snapshot = signal_snapshot
        self._last_signal_time = now_local
        self._persist_runtime_state()

        log.info(
            "Cycle completed | live_symbols=%d | open_positions=%d | buys=%d | sells=%d | rule_exits=%d | min_conf=%.2f | cash=%.2f",
            len(live_data),
            len(self.broker.get_positions()),
            buys,
            sells,
            exits_by_rule,
            dynamic_min_conf,
            self.broker.get_cash(),
        )

    def _maybe_send_daily_report(self, now_local: datetime, settings: RuntimeSettings) -> None:
        if not self._schedule_reached(now_local, settings.report_hour_gmt3, settings.report_minute_gmt3):
            return

        today = now_local.strftime("%Y-%m-%d")
        last_sent = self.trade_store.get_meta("last_report_day")
        if last_sent == today:
            return

        latest_prices = self._latest_position_prices()

        report = format_daily_report(now_local, self.trade_store, self.broker, latest_prices)
        self.telegram.send_message(report)
        self.trade_store.set_meta("last_report_day", today)

    def _maybe_send_weekly_report(self, now_local: datetime, settings: RuntimeSettings) -> None:
        if now_local.weekday() != settings.weekly_report_weekday:
            return
        if not self._schedule_reached(now_local, settings.report_hour_gmt3, settings.report_minute_gmt3):
            return

        iso = now_local.isocalendar()
        week_key = f"{iso.year}-W{iso.week:02d}"
        last_sent = self.trade_store.get_meta("last_weekly_report_key")
        if last_sent == week_key:
            return

        latest_prices = self._latest_position_prices()
        report = format_weekly_report(now_local, self.trade_store, self.broker, latest_prices)
        self.telegram.send_message(report)
        self.trade_store.set_meta("last_weekly_report_key", week_key)

    @staticmethod
    def _schedule_reached(now_local: datetime, hour: int, minute: int) -> bool:
        target = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return now_local >= target

    def _maybe_flatten_before_close(self, now_local: datetime, settings: RuntimeSettings) -> None:
        if not settings.intraday_flatten_enabled:
            return
        if not self._schedule_reached(now_local, settings.flatten_hour_gmt3, settings.flatten_minute_gmt3):
            return

        today = now_local.strftime("%Y-%m-%d")
        key = "last_flatten_day"
        if self.trade_store.get_meta(key) == today:
            return

        positions = self.broker.get_positions()
        if not positions:
            self.trade_store.set_meta(key, today)
            return

        flattened = 0
        for ticker, pos in list(positions.items()):
            price = get_last_price(ticker)
            if price is None:
                continue
            result = self.broker.place_market_sell(
                ticker=ticker,
                price=price,
                quantity=pos.quantity,
                note="strategy=flatten;tf=eod;reason=pre_close_flatten",
            )
            if result.success and result.trade:
                self.trade_store.record_trade(result.trade)
                self._touch_ticker_trade_time(ticker, now_local)
                self._position_high_watermark.pop(ticker, None)
                flattened += 1

        if flattened > 0:
            self.telegram.send_message(f"Pre-close flatten executed. Closed positions: {flattened}")
            log.info("Pre-close flatten executed | count=%d", flattened)
        self.trade_store.set_meta(key, today)
        self._persist_runtime_state()

    def _handle_command(self, text: str) -> str:
        with self._lock:
            return self._handle_command_locked(text)

    def _handle_command_locked(self, text: str) -> str:
        settings = self.settings_store.load()
        cmd = text.strip()

        if cmd in {"/help", "/start"}:
            return (
                "TickerBot commands:\n"
                "/status              - strategy, positions, cash\n"
                "/positions           - open positions with PnL\n"
                "/report              - today's trading report\n"
                "/weeklyreport        - this week's summary\n"
                "/debugsignals [n]    - top N signals by confidence\n"
                "/settings            - all runtime settings\n"
                "/set <key> <value>   - update a setting live\n"
                "  markets: market_scope = BIST30 | BIST | NASDAQ | FOREX\n"
                "  risk:    stop_loss_pct, take_profit_pct, trailing_stop_pct\n"
                "  target:  daily_profit_target_pct (default 0.03 = 3%)\n"
                "           daily_profit_target_enabled (true/false)\n"
                "  loss:    max_daily_loss_pct (default 0.02 = 2%)\n"
                "/syncuniverse        - force refresh market universe\n"
                "/tradescsv [days]    - export trades to CSV\n"
                "/startbot            - enable trading\n"
                "/stopbot             - disable trading"
            )

        if cmd == "/status":
            return format_status_report(self.selected_strategy, self.selected_timeframe, self.broker)

        if cmd == "/positions":
            positions = self.broker.get_positions()
            if not positions:
                return "No open positions."
            lines = ["Open Positions:"]
            for ticker, pos in sorted(positions.items()):
                lines.append(f"- {ticker}: qty={pos.quantity}, avg={pos.average_price:.2f}")
            return "\n".join(lines)

        if cmd == "/settings":
            cache = universe_cache_info()
            return "\n".join(
                [f"{k}={v}" for k, v in vars(settings).items()]
                + [
                    f"universe_cache_size={cache['size']}",
                    f"universe_cache_updated_at={cache['updated_at']}",
                    f"us_universe_cache_size={cache['us_size']}",
                    f"us_universe_cache_updated_at={cache['us_updated_at']}",
                ]
            )

        if cmd == "/report":
            latest_prices = self._latest_position_prices()
            return format_daily_report(datetime.now(self.istanbul_tz), self.trade_store, self.broker, latest_prices)

        if cmd == "/weeklyreport":
            latest_prices = self._latest_position_prices()
            return format_weekly_report(datetime.now(self.istanbul_tz), self.trade_store, self.broker, latest_prices)

        if cmd.startswith("/debugsignals"):
            parts = cmd.split(maxsplit=1)
            limit = 10
            if len(parts) == 2:
                try:
                    limit = max(1, int(parts[1]))
                except ValueError:
                    return "Usage: /debugsignals [n]"
            return self._format_debug_signals(limit=limit)

        if cmd == "/syncuniverse":
            scope = settings.market_scope.upper().strip()
            if scope in {"NASDAQ", "US"}:
                synced = sync_us_universe()
                label = "US"
            else:
                synced = sync_bist_universe()
                label = "BIST"
            if not synced:
                return "Universe sync failed; keeping cached/fallback universe."
            return f"Universe synced. {label} symbols cached: {len(synced)}"

        if cmd.startswith("/tradescsv"):
            parts = cmd.split(maxsplit=1)
            days = 7
            if len(parts) == 2:
                try:
                    days = max(1, int(parts[1]))
                except ValueError:
                    return "Usage: /tradescsv [days]"

            file_path = self._export_trades_csv(days)
            self.telegram.send_document(
                file_path=file_path,
                caption=f"Trades export for last {days} day(s)",
            )
            return f"Sent CSV export for last {days} day(s)."

        if cmd == "/startbot":
            self.trading_enabled = True
            self._persist_runtime_state()
            return "Trading enabled."

        if cmd == "/stopbot":
            self.trading_enabled = False
            self._persist_runtime_state()
            return "Trading disabled."

        if cmd.startswith("/set "):
            parts = cmd.split(maxsplit=2)
            if len(parts) < 3:
                return "Usage: /set <key> <value>"

            key, raw = parts[1], parts[2]
            try:
                cast_value = self._cast_setting_value(settings, key, raw)
                updated = self.settings_store.update(key, cast_value)
                return f"Updated {key}={getattr(updated, key)}"
            except Exception as exc:
                return f"Failed to update setting: {exc}"

        return "Unknown command. Send /help"

    def _latest_position_prices(self) -> dict[str, float]:
        latest_prices = {}
        for ticker in self.broker.get_positions().keys():
            price = get_last_price(ticker)
            if price is not None:
                latest_prices[ticker] = price
        return latest_prices

    def _format_debug_signals(self, limit: int = 10) -> str:
        if not self._last_signal_snapshot:
            return "No signal snapshot yet. Wait for at least one trading cycle."

        ts = self._last_signal_time.strftime("%Y-%m-%d %H:%M:%S") if self._last_signal_time else "unknown"
        top = self._last_signal_snapshot[:limit]
        counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for row in self._last_signal_snapshot:
            counts[row["action"]] = counts.get(row["action"], 0) + 1

        lines = [
            f"Signal Debug ({ts} GMT+3)",
            f"Strategy={self.selected_strategy} Timeframe={self.selected_timeframe}",
            f"Universe evaluated={len(self._last_signal_snapshot)} | BUY={counts.get('BUY',0)} SELL={counts.get('SELL',0)} HOLD={counts.get('HOLD',0)}",
            f"Top {len(top)} by confidence:",
        ]
        for row in top:
            lines.append(
                f"- {row['ticker']} {row['action']} conf={row['confidence']:.2f} price={row['price']:.2f}"
            )
        return "\n".join(lines)

    def _strategy_performance_bias(self, now_local: datetime) -> dict[str, float]:
        start = now_local - timedelta(days=30)
        trades = self.trade_store.get_trades_between(start, now_local)
        stats = closed_trade_stats(trades)
        bias = {}
        for bucket, row in stats.get("by_bucket", {}).items():
            if row.get("trades", 0) < 3:
                continue
            bias[bucket] = max(-2.0, min(2.0, row.get("score", 0.0) / 5.0))
        return bias

    def _dynamic_min_confidence(self, settings: RuntimeSettings, now_local: datetime) -> float:
        base = settings.min_signal_confidence
        start = now_local - timedelta(days=30)
        trades = self.trade_store.get_trades_between(start, now_local)
        stats = closed_trade_stats(trades)
        overall = stats.get("overall", {})
        closed_count = overall.get("closed_count", 0)
        avg_pnl = overall.get("avg_pnl", 0.0)
        wins = overall.get("wins", 0)
        losses = overall.get("losses", 0)

        if closed_count >= 8:
            hit_rate = wins / max(wins + losses, 1)
            if hit_rate < 0.45 or avg_pnl < 0:
                base += 0.07
            elif hit_rate > 0.58 and avg_pnl > 0:
                base -= 0.05
        return max(0.30, min(0.80, base))

    def _passes_liquidity_gate(self, ticker: str, data, settings: RuntimeSettings) -> bool:
        if data.empty or len(data) < 20:
            return False
        # Forex pairs don't have meaningful volume data in yfinance.
        if is_forex_ticker(ticker) or "Volume" not in data.columns:
            return True
        avg_volume = float(data["Volume"].tail(20).mean())
        price = float(data["Close"].iloc[-1])
        turnover = avg_volume * price
        return avg_volume >= settings.min_avg_volume and turnover >= settings.min_turnover_try

    def _exit_reason(self, pos, price: float, now_local: datetime, settings: RuntimeSettings) -> str:
        if not pos:
            return ""
        pnl_pct = (price - pos.average_price) / max(pos.average_price, 1e-9)
        if pnl_pct <= -abs(settings.stop_loss_pct):
            return "stop_loss"
        if pnl_pct >= abs(settings.take_profit_pct):
            return "take_profit"

        high = self._position_high_watermark.get(pos.ticker, max(pos.average_price, price))
        drawdown_from_high = (price - high) / max(high, 1e-9)
        if drawdown_from_high <= -abs(settings.trailing_stop_pct):
            return "trailing_stop"

        opened_local = pos.opened_at
        if opened_local.tzinfo is None:
            opened_local = opened_local.replace(tzinfo=ZoneInfo("UTC")).astimezone(self.istanbul_tz)
        else:
            opened_local = opened_local.astimezone(self.istanbul_tz)
        holding_minutes = (now_local - opened_local).total_seconds() / 60.0
        if holding_minutes >= max(1, settings.max_holding_minutes):
            return "max_holding"
        return ""

    def _touch_ticker_trade_time(self, ticker: str, now_local: datetime) -> None:
        self.trade_store.set_meta(f"last_trade_ts:{ticker}", now_local.isoformat())

    def _in_cooldown(self, ticker: str, settings: RuntimeSettings, now_local: datetime) -> bool:
        raw = self.trade_store.get_meta(f"last_trade_ts:{ticker}")
        if not raw:
            return False
        try:
            ts = datetime.fromisoformat(raw)
        except Exception:
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=self.istanbul_tz)
        elapsed = (now_local - ts).total_seconds() / 60.0
        return elapsed < max(0, settings.ticker_cooldown_minutes)

    def _buys_today(self, now_local: datetime) -> int:
        trades = self.trade_store.get_trades_for_day(now_local)
        return sum(1 for t in trades if t.side == "BUY")

    def _record_no_data_failure(self, ticker: str, settings: RuntimeSettings, now_local: datetime) -> None:
        count = self._no_data_failures.get(ticker, 0) + 1
        self._no_data_failures[ticker] = count
        if count >= max(1, settings.no_data_fail_threshold):
            until = now_local + timedelta(minutes=max(1, settings.no_data_cooldown_minutes))
            self._no_data_block_until[ticker] = until.isoformat()

    def _clear_no_data_failure(self, ticker: str) -> None:
        self._no_data_failures.pop(ticker, None)
        self._no_data_block_until.pop(ticker, None)

    def _is_ticker_blocked(self, ticker: str, now_local: datetime) -> bool:
        raw = self._no_data_block_until.get(ticker)
        if not raw:
            return False
        try:
            until = datetime.fromisoformat(raw)
        except Exception:
            self._no_data_block_until.pop(ticker, None)
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=self.istanbul_tz)
        if now_local >= until:
            self._no_data_block_until.pop(ticker, None)
            self._no_data_failures.pop(ticker, None)
            return False
        return True

    def _export_trades_csv(self, days: int) -> str:
        now_local = datetime.now(self.istanbul_tz)
        start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)

        start = start - timedelta(days=days - 1)
        trades = self.trade_store.get_trades_between(start, now_local)

        export_dir = Path("data/exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        file_path = export_dir / f"trades_{now_local.strftime('%Y%m%d_%H%M%S')}_{days}d.csv"

        with file_path.open("w", newline="") as handle:
            writer = DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "ticker",
                    "side",
                    "quantity",
                    "price",
                    "gross_value",
                    "fee",
                    "note",
                ],
            )
            writer.writeheader()
            for t in trades:
                writer.writerow(
                    {
                        "timestamp": t.timestamp.isoformat(),
                        "ticker": t.ticker,
                        "side": t.side,
                        "quantity": t.quantity,
                        "price": round(t.price, 6),
                        "gross_value": round(t.gross_value, 6),
                        "fee": round(t.fee, 6),
                        "note": t.note,
                    }
                )

        return str(file_path)

    def _apply_execution_settings(self, settings: RuntimeSettings) -> None:
        self.broker.configure_execution(
            fee_bps=settings.fee_bps,
            slippage_bps=settings.slippage_bps,
            partial_fill_enabled=settings.partial_fill_enabled,
            partial_fill_min_ratio=settings.partial_fill_min_ratio,
            partial_fill_max_ratio=settings.partial_fill_max_ratio,
        )

    def _persist_runtime_state(self) -> None:
        payload = {
            "selected_strategy": self.selected_strategy,
            "selected_timeframe": self.selected_timeframe,
            "trading_enabled": self.trading_enabled,
            "profit_halt_day": self._profit_halt_day,
            "last_signal_time": self._last_signal_time.isoformat() if self._last_signal_time else "",
            "position_high_watermark": self._position_high_watermark,
            "no_data_failures": self._no_data_failures,
            "no_data_block_until": self._no_data_block_until,
            "broker": self.broker.export_state(),
        }
        self.trade_store.set_meta(self._state_key, json.dumps(payload))

    @staticmethod
    def _effective_regime_ticker(settings: RuntimeSettings) -> str:
        scope = settings.market_scope.upper().strip()
        if scope == "FOREX" and settings.regime_ticker in {"XU100.IS", "QQQ"}:
            # DXY (US Dollar Index) as a macro regime filter for forex.
            return "DX-Y.NYB"
        if scope in {"NASDAQ", "US"} and settings.regime_ticker == "XU100.IS":
            return "QQQ"
        return settings.regime_ticker

    def _restore_runtime_state(self) -> None:
        raw = self.trade_store.get_meta(self._state_key)
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except Exception:
            return

        if not isinstance(payload, dict):
            return

        self.selected_strategy = str(payload.get("selected_strategy", self.selected_strategy))
        self.selected_timeframe = str(payload.get("selected_timeframe", self.selected_timeframe))
        self.trading_enabled = bool(payload.get("trading_enabled", self.trading_enabled))
        self._profit_halt_day = str(payload.get("profit_halt_day", ""))

        last_signal_time = payload.get("last_signal_time")
        if isinstance(last_signal_time, str) and last_signal_time:
            try:
                self._last_signal_time = datetime.fromisoformat(last_signal_time)
            except Exception:
                self._last_signal_time = None

        broker_state = payload.get("broker", {})
        self.broker.import_state(broker_state)
        self._position_high_watermark = {
            str(k): float(v) for k, v in payload.get("position_high_watermark", {}).items()
        } if isinstance(payload.get("position_high_watermark"), dict) else {}
        self._no_data_failures = {
            str(k): int(v) for k, v in payload.get("no_data_failures", {}).items()
        } if isinstance(payload.get("no_data_failures"), dict) else {}
        self._no_data_block_until = {
            str(k): str(v) for k, v in payload.get("no_data_block_until", {}).items()
        } if isinstance(payload.get("no_data_block_until"), dict) else {}
        log.info(
            "Runtime state restored | cash=%.2f | open_positions=%d | strategy=%s/%s",
            self.broker.get_cash(),
            len(self.broker.get_positions()),
            self.selected_strategy,
            self.selected_timeframe,
        )

    @staticmethod
    def _cast_setting_value(settings: RuntimeSettings, key: str, raw: str):
        if not hasattr(settings, key):
            raise ValueError(f"Unknown key: {key}")

        current = getattr(settings, key)
        if isinstance(current, bool):
            return raw.lower() in {"1", "true", "yes", "on"}
        if isinstance(current, int):
            return int(raw)
        if isinstance(current, float):
            return float(raw)
        return raw
