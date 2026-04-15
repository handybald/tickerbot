from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RuntimeSettings:
    market_scope: str = "BIST30"
    auto_strategy: bool = True
    strategy_override: str = ""
    timeframe_override: str = ""
    max_open_positions: int = 5
    max_position_pct: float = 0.1
    max_daily_loss_pct: float = 0.02
    per_trade_cash_pct: float = 0.05
    trade_period_minutes: int = 60
    report_hour_gmt3: int = 18
    report_minute_gmt3: int = 0
    weekly_report_weekday: int = 4
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    partial_fill_enabled: bool = True
    partial_fill_min_ratio: float = 0.6
    partial_fill_max_ratio: float = 1.0
    universe_refresh_minutes: int = 720
    min_signal_confidence: float = 0.45
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.015
    trailing_stop_pct: float = 0.008
    max_holding_minutes: int = 1440
    max_new_entries_per_cycle: int = 3
    max_new_entries_per_day: int = 10
    ticker_cooldown_minutes: int = 60
    min_avg_volume: int = 200000
    min_turnover_try: float = 2_000_000.0
    regime_filter_enabled: bool = True
    regime_ticker: str = "XU100.IS"
    regime_fast_sma: int = 50
    regime_slow_sma: int = 200
    regime_rsi_min: float = 45.0
    intraday_flatten_enabled: bool = False
    flatten_hour_gmt3: int = 17
    flatten_minute_gmt3: int = 55
    no_data_fail_threshold: int = 3
    no_data_cooldown_minutes: int = 240
    # Daily profit target: stop new entries once this gain is reached for the day.
    daily_profit_target_enabled: bool = True
    daily_profit_target_pct: float = 0.03   # 3 % default


class SettingsStore:
    def __init__(self, db_path: str = "data/bot_state.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def load(self) -> RuntimeSettings:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM settings WHERE id = 1").fetchone()
            if not row:
                defaults = RuntimeSettings()
                self.save(defaults)
                return defaults
            payload = json.loads(row[0])
            # Field-filtered deserialization handles schema evolution safely.
            known = {f for f in RuntimeSettings.__dataclass_fields__}
            filtered = {k: v for k, v in payload.items() if k in known}
            return RuntimeSettings(**filtered)

    def save(self, settings: RuntimeSettings) -> None:
        payload = json.dumps(asdict(settings))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO settings(id, payload)
                VALUES (1, ?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload
                """,
                (payload,),
            )
            conn.commit()

    def update(self, key: str, value) -> RuntimeSettings:
        settings = self.load()
        if not hasattr(settings, key):
            raise ValueError(f"Unknown setting: {key}")
        setattr(settings, key, value)
        self.save(settings)
        return settings
