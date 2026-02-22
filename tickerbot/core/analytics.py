from __future__ import annotations

from collections import defaultdict, deque


def parse_trade_note(note: str) -> dict[str, str]:
    out = {}
    if not note:
        return out
    for part in note.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def closed_trade_stats(trades: list) -> dict:
    lots = defaultdict(deque)
    closed = []

    for t in trades:
        if t.quantity <= 0:
            continue
        meta = parse_trade_note(t.note)
        bucket = f"{meta.get('strategy', 'na')}|{meta.get('tf', 'na')}"

        if t.side == "BUY":
            unit_cost = (t.gross_value + t.fee) / t.quantity if t.gross_value else (t.price + (t.fee / t.quantity))
            lots[t.ticker].append([t.quantity, unit_cost, bucket])
            continue

        if t.side != "SELL":
            continue

        unit_proceeds = (t.gross_value - t.fee) / t.quantity if t.gross_value else (t.price - (t.fee / t.quantity))
        qty_left = t.quantity

        while qty_left > 0 and lots[t.ticker]:
            lot_qty, lot_cost, lot_bucket = lots[t.ticker][0]
            match_qty = min(qty_left, lot_qty)
            pnl = (unit_proceeds - lot_cost) * match_qty
            closed.append({"bucket": lot_bucket, "pnl": pnl})

            lot_qty -= match_qty
            qty_left -= match_qty
            if lot_qty == 0:
                lots[t.ticker].popleft()
            else:
                lots[t.ticker][0][0] = lot_qty

    by_bucket = defaultdict(list)
    for row in closed:
        by_bucket[row["bucket"]].append(row["pnl"])

    bucket_stats = {}
    for bucket, pnls in by_bucket.items():
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        hit_rate = len(wins) / len(pnls) if pnls else 0.0
        avg = sum(pnls) / len(pnls) if pnls else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses else (2.0 if wins else 0.0)
        bucket_stats[bucket] = {
            "trades": len(pnls),
            "avg_pnl": avg,
            "hit_rate": hit_rate,
            "profit_factor": profit_factor,
            "score": (hit_rate * 5.0) + avg + min(profit_factor, 3.0),
        }

    overall = {
        "closed_count": len(closed),
        "avg_pnl": (sum(x["pnl"] for x in closed) / len(closed)) if closed else 0.0,
        "wins": sum(1 for x in closed if x["pnl"] > 0),
        "losses": sum(1 for x in closed if x["pnl"] < 0),
    }

    return {"overall": overall, "by_bucket": bucket_stats}
