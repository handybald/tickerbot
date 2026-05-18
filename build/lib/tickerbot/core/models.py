from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    ticker: str
    quantity: int
    average_price: float
    opened_at: datetime


@dataclass
class Trade:
    ticker: str
    side: str
    quantity: int
    price: float
    timestamp: datetime
    gross_value: float = 0.0
    fee: float = 0.0
    note: str = ""


@dataclass
class Signal:
    ticker: str
    action: str
    confidence: float
    strategy: str
    timeframe: str
