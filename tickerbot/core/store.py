from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .models import Trade


class TradeStore:
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
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    gross_value REAL NOT NULL DEFAULT 0,
                    fee REAL NOT NULL DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    note TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

            columns = self._table_columns(conn, "trades")
            if "gross_value" not in columns:
                conn.execute("ALTER TABLE trades ADD COLUMN gross_value REAL NOT NULL DEFAULT 0")
            if "fee" not in columns:
                conn.execute("ALTER TABLE trades ADD COLUMN fee REAL NOT NULL DEFAULT 0")

            conn.commit()

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row[1] for row in rows}

    def record_trade(self, trade: Trade) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades(ticker, side, quantity, price, gross_value, fee, timestamp, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.ticker,
                    trade.side,
                    trade.quantity,
                    trade.price,
                    trade.gross_value,
                    trade.fee,
                    trade.timestamp.isoformat(),
                    trade.note,
                ),
            )
            conn.commit()

    def get_trades_for_day(self, day: datetime) -> list[Trade]:
        day_prefix = day.strftime("%Y-%m-%d")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ticker, side, quantity, price, gross_value, fee, timestamp, note
                FROM trades
                WHERE timestamp LIKE ?
                ORDER BY timestamp ASC
                """,
                (f"{day_prefix}%",),
            ).fetchall()
        return [
            Trade(
                ticker=row[0],
                side=row[1],
                quantity=row[2],
                price=row[3],
                gross_value=row[4] if row[4] is not None else row[2] * row[3],
                fee=row[5] if row[5] is not None else 0.0,
                timestamp=datetime.fromisoformat(row[6]),
                note=row[7],
            )
            for row in rows
        ]

    def get_trades_between(self, start: datetime, end: datetime) -> list[Trade]:
        start_iso = self._normalize_for_storage(start).isoformat()
        end_iso = self._normalize_for_storage(end).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ticker, side, quantity, price, gross_value, fee, timestamp, note
                FROM trades
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (start_iso, end_iso),
            ).fetchall()
        return [
            Trade(
                ticker=row[0],
                side=row[1],
                quantity=row[2],
                price=row[3],
                gross_value=row[4] if row[4] is not None else row[2] * row[3],
                fee=row[5] if row[5] is not None else 0.0,
                timestamp=datetime.fromisoformat(row[6]),
                note=row[7],
            )
            for row in rows
        ]

    @staticmethod
    def _normalize_for_storage(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt
        return dt.astimezone(timezone.utc).replace(tzinfo=None)

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meta(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, value),
            )
            conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def list_recent_trades(self, limit: int = 20) -> Iterable[Trade]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ticker, side, quantity, price, gross_value, fee, timestamp, note
                FROM trades
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            Trade(
                ticker=row[0],
                side=row[1],
                quantity=row[2],
                price=row[3],
                gross_value=row[4] if row[4] is not None else row[2] * row[3],
                fee=row[5] if row[5] is not None else 0.0,
                timestamp=datetime.fromisoformat(row[6]),
                note=row[7],
            )
            for row in rows
        ]
