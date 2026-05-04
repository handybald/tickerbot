from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from .models import Position, Trade


@dataclass
class OrderResult:
    success: bool
    message: str
    trade: Trade | None = None


class BrokerAdapter(ABC):
    @abstractmethod
    def get_cash(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        raise NotImplementedError

    @abstractmethod
    def place_market_buy(self, ticker: str, price: float, cash_budget: float, note: str = "") -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def place_market_sell(self, ticker: str, price: float, quantity: int, note: str = "") -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def get_equity(self, latest_prices: dict[str, float]) -> float:
        raise NotImplementedError

    @abstractmethod
    def export_state(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def import_state(self, payload: dict) -> None:
        raise NotImplementedError


class PaperBrokerAdapter(BrokerAdapter):
    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self._cash = initial_cash
        self._positions: dict[str, Position] = {}

        self.fee_bps = 10.0
        self.slippage_bps = 5.0
        self.partial_fill_enabled = True
        self.partial_fill_min_ratio = 0.6
        self.partial_fill_max_ratio = 1.0

    def configure_execution(
        self,
        fee_bps: float,
        slippage_bps: float,
        partial_fill_enabled: bool,
        partial_fill_min_ratio: float,
        partial_fill_max_ratio: float,
    ) -> None:
        self.fee_bps = max(0.0, float(fee_bps))
        self.slippage_bps = max(0.0, float(slippage_bps))
        self.partial_fill_enabled = bool(partial_fill_enabled)
        lo = min(float(partial_fill_min_ratio), float(partial_fill_max_ratio))
        hi = max(float(partial_fill_min_ratio), float(partial_fill_max_ratio))
        self.partial_fill_min_ratio = max(0.1, lo)
        self.partial_fill_max_ratio = min(1.0, hi)

    def get_cash(self) -> float:
        return self._cash

    def get_positions(self) -> dict[str, Position]:
        return self._positions.copy()

    def place_market_buy(self, ticker: str, price: float, cash_budget: float, note: str = "") -> OrderResult:
        if price <= 0:
            return OrderResult(False, "Invalid price")

        budget = min(self._cash, cash_budget)
        fill_price = price * (1.0 + (self.slippage_bps / 10_000.0))
        fee_rate = self.fee_bps / 10_000.0

        max_qty = int(budget // (fill_price * (1.0 + fee_rate)))
        if max_qty <= 0:
            return OrderResult(False, "Insufficient budget for 1 share")

        qty = self._apply_partial_fill(max_qty)
        gross = qty * fill_price
        fee = gross * fee_rate
        total_cost = gross + fee
        if total_cost > self._cash:
            return OrderResult(False, "Insufficient cash")

        self._cash -= total_cost

        existing = self._positions.get(ticker)
        now = datetime.utcnow()
        if existing:
            total_qty = existing.quantity + qty
            avg_price = ((existing.average_price * existing.quantity) + total_cost) / total_qty
            self._positions[ticker] = Position(
                ticker=ticker,
                quantity=total_qty,
                average_price=avg_price,
                opened_at=existing.opened_at,
            )
        else:
            self._positions[ticker] = Position(
                ticker=ticker,
                quantity=qty,
                average_price=total_cost / qty,
                opened_at=now,
            )

        trade = Trade(
            ticker=ticker,
            side="BUY",
            quantity=qty,
            price=fill_price,
            timestamp=now,
            gross_value=gross,
            fee=fee,
            note=note,
        )
        msg = "Partially filled" if qty < max_qty else "Filled"
        return OrderResult(True, msg, trade=trade)

    def place_market_sell(self, ticker: str, price: float, quantity: int, note: str = "") -> OrderResult:
        if price <= 0 or quantity <= 0:
            return OrderResult(False, "Invalid sell request")
        pos = self._positions.get(ticker)
        if not pos:
            return OrderResult(False, "No open position")

        target_qty = min(quantity, pos.quantity)
        qty = self._apply_partial_fill(target_qty)
        fill_price = price * (1.0 - (self.slippage_bps / 10_000.0))
        fee_rate = self.fee_bps / 10_000.0

        gross = qty * fill_price
        fee = gross * fee_rate
        proceeds = gross - fee
        self._cash += proceeds

        remaining = pos.quantity - qty
        if remaining == 0:
            self._positions.pop(ticker, None)
        else:
            self._positions[ticker] = Position(
                ticker=ticker,
                quantity=remaining,
                average_price=pos.average_price,
                opened_at=pos.opened_at,
            )

        trade = Trade(
            ticker=ticker,
            side="SELL",
            quantity=qty,
            price=fill_price,
            timestamp=datetime.utcnow(),
            gross_value=gross,
            fee=fee,
            note=note,
        )
        msg = "Partially filled" if qty < target_qty else "Filled"
        return OrderResult(True, msg, trade=trade)

    def get_equity(self, latest_prices: dict[str, float]) -> float:
        holdings_value = 0.0
        for ticker, pos in self._positions.items():
            price = latest_prices.get(ticker, pos.average_price)
            holdings_value += price * pos.quantity
        return self._cash + holdings_value

    def _apply_partial_fill(self, requested_qty: int) -> int:
        if requested_qty <= 1:
            return requested_qty
        if not self.partial_fill_enabled:
            return requested_qty

        ratio = random.uniform(self.partial_fill_min_ratio, self.partial_fill_max_ratio)
        qty = int(requested_qty * ratio)
        return max(1, min(requested_qty, qty))

    def export_state(self) -> dict:
        return {
            "cash": self._cash,
            "positions": {
                ticker: {
                    "quantity": pos.quantity,
                    "average_price": pos.average_price,
                    "opened_at": pos.opened_at.isoformat(),
                }
                for ticker, pos in self._positions.items()
            },
        }

    def import_state(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        cash = payload.get("cash")
        if isinstance(cash, (int, float)):
            self._cash = float(cash)

        loaded_positions: dict[str, Position] = {}
        raw_positions = payload.get("positions", {})
        if isinstance(raw_positions, dict):
            for ticker, item in raw_positions.items():
                if not isinstance(item, dict):
                    continue
                try:
                    qty = int(item.get("quantity", 0))
                    avg = float(item.get("average_price", 0.0))
                    opened = datetime.fromisoformat(item.get("opened_at"))
                except Exception:
                    continue
                if qty <= 0 or avg <= 0:
                    continue
                loaded_positions[ticker] = Position(
                    ticker=ticker,
                    quantity=qty,
                    average_price=avg,
                    opened_at=opened,
                )
        self._positions = loaded_positions
