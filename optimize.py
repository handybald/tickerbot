"""
optimize.py — Walk-forward parameter optimization using Optuna + Backtester.

Usage:
    python optimize.py AVGO AAPL --strategy momentum --holdout 30 --trials 100
    python optimize.py GARAN.IS THYAO.IS --strategy balanced --holdout 45 --save

Workflow:
    1. Download 1d OHLCV for each ticker (2y history)
    2. Split: train = everything before the last `holdout` days
              test  = last `holdout` days (never touched during optimization)
    3. Baseline: run backtester with DEFAULT params on both splits
    4. Optimize: Optuna minimises -Sharpe on the TRAIN split (Bayesian search)
    5. Validate: run best params on the held-out TEST split
    6. Print side-by-side comparison table (default vs optimised, train vs test)
    7. Optionally save best params to optimized_params_<strategy>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # suppress yfinance noise


# ─────────────────────────────────────────────────────────────────────────────
# Parameter spaces per strategy
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_params(trial, strategy: str) -> dict:
    """Build the params dict for one Optuna trial."""
    p: dict = {}

    # Common across all strategies
    p["signal_threshold"] = trial.suggest_float("signal_threshold", 0.8, 2.5)
    p["obv_weight"]       = trial.suggest_float("obv_weight",       0.0, 0.5)
    p["sentiment_cap"]    = trial.suggest_float("sentiment_cap",    0.1, 0.8)

    if strategy == "momentum":
        p["m_macd_bull"]      = trial.suggest_float("m_macd_bull",      0.5, 3.0)
        p["m_macd_bear"]      = trial.suggest_float("m_macd_bear",      0.3, 2.0)
        p["m_rsi_thresh"]     = trial.suggest_float("m_rsi_thresh",     45.0, 65.0)
        p["m_rsi_bull"]       = trial.suggest_float("m_rsi_bull",       0.3, 2.0)
        p["m_rsi_ob_penalty"] = trial.suggest_float("m_rsi_ob_penalty", 0.1, 1.0)
        p["m_wr_score"]       = trial.suggest_float("m_wr_score",       0.0, 1.0)
        p["m_di_score"]       = trial.suggest_float("m_di_score",       0.0, 1.5)
        p["m_stoch_thresh"]   = trial.suggest_float("m_stoch_thresh",   50.0, 75.0)
        p["m_stoch_score"]    = trial.suggest_float("m_stoch_score",    0.0, 0.8)

    elif strategy == "mean_reversion":
        p["mr_adx_block"]  = trial.suggest_float("mr_adx_block",  20.0, 50.0)
        p["mr_bb_score"]   = trial.suggest_float("mr_bb_score",   0.5, 3.0)
        p["mr_rsi_score"]  = trial.suggest_float("mr_rsi_score",  0.3, 2.0)
        p["mr_wr_score"]   = trial.suggest_float("mr_wr_score",   0.0, 2.0)
        p["mr_stoch_score"]= trial.suggest_float("mr_stoch_score",0.0, 1.5)

    elif strategy == "balanced":
        p["b_macd_score"]   = trial.suggest_float("b_macd_score",   0.3, 2.5)
        p["b_rsi_os_thresh"]= trial.suggest_float("b_rsi_os_thresh",25.0, 45.0)
        p["b_rsi_score"]    = trial.suggest_float("b_rsi_score",    0.3, 2.0)
        p["b_bb_score"]     = trial.suggest_float("b_bb_score",     0.0, 2.0)
        p["b_adx_boost"]    = trial.suggest_float("b_adx_boost",    0.0, 1.0)
        p["b_wr_score"]     = trial.suggest_float("b_wr_score",     0.0, 1.0)

    return p


_DEFAULT_PARAMS: dict[str, dict] = {
    "momentum": {
        "signal_threshold": 1.3,
        "obv_weight": 0.25,
        "sentiment_cap": 0.5,
        "m_macd_bull": 1.5,
        "m_macd_bear": 1.0,
        "m_rsi_thresh": 55.0,
        "m_rsi_bull": 0.8,
        "m_rsi_ob_penalty": 0.7,
        "m_wr_score": 0.4,
        "m_di_score": 0.5,
        "m_stoch_thresh": 60.0,
        "m_stoch_score": 0.3,
    },
    "mean_reversion": {
        "signal_threshold": 1.3,
        "obv_weight": 0.25,
        "sentiment_cap": 0.5,
        "mr_adx_block": 35.0,
        "mr_bb_score": 1.5,
        "mr_rsi_score": 1.0,
        "mr_wr_score": 0.8,
        "mr_stoch_score": 0.5,
    },
    "balanced": {
        "signal_threshold": 1.3,
        "obv_weight": 0.25,
        "sentiment_cap": 0.5,
        "b_macd_score": 1.0,
        "b_rsi_os_thresh": 35.0,
        "b_rsi_score": 1.0,
        "b_bb_score": 0.7,
        "b_adx_boost": 0.4,
        "b_wr_score": 0.4,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Backtest helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_all(backtester, data_splits: list[tuple], strategy: str, params: dict | None) -> dict:
    """
    Run backtest on every (ticker, data) pair and aggregate results.
    Returns averaged metrics dict.
    """
    from tickerbot.core.backtester import BacktestResult, _EMPTY
    results: list[BacktestResult] = []
    for _ticker, data in data_splits:
        if len(data) < 60:
            continue
        r = backtester.run(data, strategy_name=strategy, timeframe="1d", params=params)
        results.append(r)

    if not results:
        return {"sharpe": 0.0, "total_return": 0.0, "hit_rate": 0.0,
                "max_drawdown": 0.0, "profit_factor": 0.0, "num_trades": 0}

    return {
        "sharpe":        sum(r.sharpe        for r in results) / len(results),
        "total_return":  sum(r.total_return  for r in results) / len(results),
        "hit_rate":      sum(r.hit_rate      for r in results) / len(results),
        "max_drawdown":  sum(r.max_drawdown  for r in results) / len(results),
        "profit_factor": sum(r.profit_factor for r in results) / len(results),
        "num_trades":    sum(r.num_trades    for r in results),
    }


def _objective(trial, backtester, train_splits: list[tuple], strategy: str) -> float:
    """Optuna objective: maximise Sharpe on train set."""
    params = _suggest_params(trial, strategy)
    m = _run_all(backtester, train_splits, strategy, params)
    # Penalise if too few trades (less than 2 per ticker on average)
    min_trades = len(train_splits) * 2
    if m["num_trades"] < min_trades:
        return -10.0
    # Composite score: Sharpe + profit_factor bonus − drawdown penalty
    score = m["sharpe"] + 0.3 * min(m["profit_factor"], 5.0) + 0.2 * m["max_drawdown"]
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_pct(v: float) -> str:
    return f"{v*100:+7.2f}%"

def _fmt_float(v: float, decimals: int = 2) -> str:
    return f"{v:7.{decimals}f}"

def _bar(v: float, width: int = 20) -> str:
    """ASCII bar for a value in [-1, 1]."""
    filled = int(abs(v) * width / 1.0)
    filled = min(filled, width)
    symbol = "█" if v >= 0 else "░"
    return (symbol * filled).ljust(width)

def _print_table(label: str, default_m: dict, opt_m: dict) -> None:
    """Print a comparison table for one split (train or test)."""
    header = f"{'Metric':<18}  {'Default':>10}  {'Optimised':>10}  {'Delta':>10}"
    sep = "─" * len(header)
    print(f"\n  ── {label} ──")
    print(f"  {header}")
    print(f"  {sep}")
    metrics = [
        ("Sharpe",         "sharpe",        "float"),
        ("Total Return",   "total_return",  "pct"),
        ("Hit Rate",       "hit_rate",      "pct"),
        ("Max Drawdown",   "max_drawdown",  "pct"),
        ("Profit Factor",  "profit_factor", "float"),
        ("Num Trades",     "num_trades",    "int"),
    ]
    for name, key, fmt in metrics:
        d = default_m[key]
        o = opt_m[key]
        delta = o - d
        if fmt == "pct":
            ds, os, ds2 = _fmt_pct(d), _fmt_pct(o), _fmt_pct(delta)
        elif fmt == "int":
            ds, os, ds2 = f"{d:>10d}", f"{o:>10d}", f"{delta:>+10d}"
        else:
            ds, os, ds2 = _fmt_float(d), _fmt_float(o), _fmt_float(delta)
        sign = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        print(f"  {name:<18}  {ds}  {os}  {sign}{ds2}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward parameter optimisation for TickerBot strategies."
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AVGO AAPL MSFT")
    parser.add_argument("--strategy", "-s", default="balanced",
                        choices=["balanced", "momentum", "mean_reversion"],
                        help="Strategy to optimise (default: balanced)")
    parser.add_argument("--holdout", "-H", type=int, default=30,
                        help="Hold-out days for out-of-sample test (default: 30)")
    parser.add_argument("--trials", "-n", type=int, default=80,
                        help="Number of Optuna trials (default: 80)")
    parser.add_argument("--save", action="store_true",
                        help="Save best params to optimized_params_<strategy>.json")
    parser.add_argument("--fee", type=float, default=0.001,
                        help="Round-trip fee fraction (default: 0.001 = 0.1%%)")
    parser.add_argument("--slippage", type=float, default=0.0005,
                        help="One-way slippage fraction (default: 0.0005)")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    from tickerbot.core.backtester import Backtester

    strategy  = args.strategy
    holdout   = args.holdout
    n_trials  = args.trials
    tickers   = [t.upper() for t in args.tickers]

    print(f"\n{'━'*60}")
    print(f"  TickerBot Parameter Optimiser")
    print(f"  Strategy : {strategy}")
    print(f"  Tickers  : {', '.join(tickers)}")
    print(f"  Hold-out : last {holdout} trading days")
    print(f"  Trials   : {n_trials}")
    print(f"{'━'*60}")

    # ── 1. Download data ─────────────────────────────────────────────────────
    print("\n[1/4] Downloading data …")
    raw: dict[str, object] = {}
    for tkr in tickers:
        print(f"      {tkr} …", end=" ", flush=True)
        try:
            df = yf.download(tkr, period="2y", interval="1d",
                             auto_adjust=True, progress=False)
            if df.empty:
                print("SKIP (empty)")
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            raw[tkr] = df
            print(f"{len(df)} bars")
        except Exception as exc:
            print(f"FAIL ({exc})")

    if not raw:
        print("ERROR: No data downloaded. Check ticker symbols and internet connection.")
        sys.exit(1)

    # ── 2. Split into train / test ───────────────────────────────────────────
    print(f"\n[2/4] Splitting train / test (last {holdout} bars = test) …")
    train_splits: list[tuple] = []
    test_splits:  list[tuple] = []
    for tkr, df in raw.items():
        if len(df) <= holdout + 60:
            print(f"      {tkr}: too short ({len(df)} bars), skipped")
            continue
        train_df = df.iloc[:-holdout].copy()
        test_df  = df.iloc[-holdout:].copy()
        train_splits.append((tkr, train_df))
        test_splits.append((tkr, test_df))
        print(f"      {tkr}: train={len(train_df)}, test={len(test_df)}")

    if not train_splits:
        print("ERROR: All tickers skipped (insufficient data).")
        sys.exit(1)

    # ── 3. Baseline ──────────────────────────────────────────────────────────
    print("\n[3/4] Running baseline with default params …")
    bt = Backtester(fee_pct=args.fee, slippage_pct=args.slippage)
    default_params = _DEFAULT_PARAMS[strategy]

    default_train = _run_all(bt, train_splits, strategy, default_params)
    default_test  = _run_all(bt, test_splits,  strategy, default_params)
    print(f"      Train  — Sharpe={default_train['sharpe']:.2f}  "
          f"Return={default_train['total_return']*100:+.2f}%  "
          f"Trades={default_train['num_trades']}")
    print(f"      Test   — Sharpe={default_test['sharpe']:.2f}  "
          f"Return={default_test['total_return']*100:+.2f}%  "
          f"Trades={default_test['num_trades']}")

    # ── 4. Optimise on train ─────────────────────────────────────────────────
    print(f"\n[4/4] Optimising on train set ({n_trials} trials) …")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Seed with default params so Optuna has a good starting point
    study.enqueue_trial(default_params)

    import functools
    study.optimize(
        functools.partial(_objective,
                          backtester=bt,
                          train_splits=train_splits,
                          strategy=strategy),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_value  = study.best_value

    print(f"\n  Best composite score on train: {best_value:.4f}")

    # ── 5. Validate best params on test set ─────────────────────────────────
    opt_train = _run_all(bt, train_splits, strategy, best_params)
    opt_test  = _run_all(bt, test_splits,  strategy, best_params)

    # ── 6. Print comparison tables ───────────────────────────────────────────
    print(f"\n{'━'*60}")
    print("  RESULTS")
    print(f"{'━'*60}")

    _print_table("TRAIN (in-sample)",  default_train, opt_train)
    _print_table("TEST  (out-of-sample, NEVER seen during optimisation)",
                 default_test, opt_test)

    # ── 7. Print best params ─────────────────────────────────────────────────
    print(f"\n{'━'*60}")
    print(f"  Best params ({strategy})")
    print(f"{'━'*60}")
    max_klen = max(len(k) for k in best_params)
    def_p = default_params
    for k, v in sorted(best_params.items()):
        default_v = def_p.get(k, "—")
        change = ""
        if isinstance(default_v, float):
            delta = v - default_v
            arrow = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "=")
            change = f"  (was {default_v:.4f}, {arrow}{abs(delta):.4f})"
        print(f"  {k:<{max_klen}}  {v:.4f}{change}")

    # ── 8. Overfit warning ───────────────────────────────────────────────────
    train_gain = opt_train["sharpe"] - default_train["sharpe"]
    test_gain  = opt_test["sharpe"]  - default_test["sharpe"]
    print(f"\n  Sharpe delta  — train: {train_gain:+.2f}   test: {test_gain:+.2f}")

    # If the test set had 0 trades, overfit analysis isn't meaningful
    if opt_test["num_trades"] == 0 and default_test["num_trades"] == 0:
        print(f"  ⚠  No trades fired in the test window ({holdout} bars).")
        print(f"     Try a larger --holdout (e.g. --holdout 60) or add more tickers.")
    else:
        overfit_ratio = abs(train_gain) / (abs(test_gain) + 1e-9)
        if overfit_ratio > 3.0 and train_gain > 0:
            print(f"  ⚠  Overfit signal: train improved {overfit_ratio:.1f}x more than test.")
            print(f"     Consider using fewer trials or a larger training set.")
        elif test_gain >= 0:
            print(f"  ✓  Optimised params generalise to held-out test set.")
        else:
            print(f"  ⚠  Test Sharpe declined — optimised params may overfit.")

    # ── 9. Save ──────────────────────────────────────────────────────────────
    if args.save:
        out_path = Path(f"optimized_params_{strategy}.json")
        with open(out_path, "w") as f:
            json.dump({"strategy": strategy, "tickers": tickers,
                       "holdout_days": holdout, "params": best_params,
                       "train_metrics": opt_train, "test_metrics": opt_test}, f, indent=2)
        print(f"\n  Saved → {out_path}")

    print()


if __name__ == "__main__":
    main()
