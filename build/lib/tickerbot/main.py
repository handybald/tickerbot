import argparse
import logging
import os
from pathlib import Path

import yfinance as yf
from .indicators import calculate_sma, calculate_rsi, calculate_macd, calculate_bollinger_bands
from .data_fetcher import DataFetcher
from .suggestion import generate_suggestion
from .trader import Trader
from .backtester import Backtester
from .core import BotService

try:
    from tradingview_screener import Query
except Exception:
    Query = None


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
            value = value[1:-1]
        os.environ.setdefault(key, value)

def get_stock_data(ticker, period="500d", interval="1d"):
    """
    Gets historical stock data.
    """
    stock = yf.Ticker(ticker)
    return stock.history(period=period, interval=interval)

def generate_interval_suggestion(ticker, interval, period):
    """
    Generates a suggestion for a specific interval.
    """
    data = get_stock_data(ticker, period=period, interval=interval)
    
    if not data.empty:
        price = data["Close"].iloc[-1]
        sma_20 = calculate_sma(data, 20).iloc[-1]
        sma_400 = calculate_sma(data, 400).iloc[-1] if len(data) > 400 else None
        rsi_14 = calculate_rsi(data).iloc[-1]
        macd, signal_line = calculate_macd(data)
        upper_band, lower_band = calculate_bollinger_bands(data)
        
        data_fetcher = DataFetcher(ticker)
        recommendations = data_fetcher.get_recommendations()
        news = data_fetcher.get_stock_news()
        avg_sentiment = news['sentiment_title'].mean()

        suggestion = generate_suggestion(rsi_14, macd.iloc[-1], signal_line.iloc[-1], upper_band.iloc[-1], lower_band.iloc[-1], price, sma_400, recommendations, avg_sentiment)
        print(f"{interval} Suggestion for {ticker}: {suggestion}")

def suggestion_mode(tickers):
    """
    Runs the suggestion mode.
    """
    intervals = {
        "15m": "1mo",
        "1h": "1mo",
        "1d": "2y",
        "1wk": "5y"
    }
    
    for ticker in tickers:
        print(f"\n--- Suggestions for {ticker} ---")
        for interval, period in intervals.items():
            generate_interval_suggestion(ticker, interval, period)

def automatic_mode(daily_limit, aggressiveness, target_profit):
    """
    Runs the automatic trading mode.
    """
    trader = Trader(daily_limit, aggressiveness, target_profit)
    trader.start_trading()

def backtest_mode(strategy, ticker, initial_investment):
    """
    Runs the backtesting mode.
    """
    data = get_stock_data(ticker, period="1y") # Use 1 year of data for backtesting
    backtester = Backtester(strategy, data, initial_investment)
    backtester.run()
    
def scrape_mode():
    """
    Runs the scraping mode.
    """
    data_fetcher = DataFetcher(None)
    stocks = data_fetcher.scrape_hedeffiyat()
    for stock in stocks:
        print(stock)
        
def discover_mode(market):
    """
    Runs the discover mode.
    """
    if Query is None:
        raise RuntimeError("discover mode requires 'tradingview-screener' package")
    screener = Query().set_screener(market)
    total, tickers = screener.get_scanner_data()
    print(f"Found {total} tickers in the {market} market.")
    for ticker in tickers['name']:
        print(ticker)


def run_mode(initial_cash):
    load_env_file()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        raise RuntimeError(
            "run mode requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables"
        )

    service = BotService(token=token, chat_id=chat_id, initial_cash=initial_cash)
    service.run()

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="A stock trading bot.")
    subparsers = parser.add_subparsers(dest="mode", help="Operating modes")

    # Suggestion mode
    suggestion_parser = subparsers.add_parser("suggestion", help="Get a trading suggestion for a stock.")
    suggestion_parser.add_argument("tickers", type=str, nargs='+', help="Stock ticker symbol(s) (e.g., AAPL GOOG)")

    # Automatic mode
    automatic_parser = subparsers.add_parser("automatic", help="Run the bot in automatic trading mode.")
    automatic_parser.add_argument("--daily-limit", type=float, default=1000.0, help="Daily transaction limit.")
    automatic_parser.add_argument("--aggressiveness", type=float, default=0.5, help="Aggressiveness of the trading strategy (0.0 to 1.0).")
    automatic_parser.add_argument("--target-profit", type=float, default=100.0, help="Target profit for the day.")

    # Backtest mode
    backtest_parser = subparsers.add_parser("backtest", help="Backtest a trading strategy.")
    backtest_parser.add_argument("strategy", type=str, help="The trading strategy to backtest.")
    backtest_parser.add_argument("ticker", type=str, help="The stock ticker to backtest on.")
    backtest_parser.add_argument("--initial-investment", type=float, default=10000.0, help="Initial investment amount.")
    
    # Scrape mode
    scrape_parser = subparsers.add_parser("scrape", help="Scrape data from hedeffiyat.com.tr.")
    
    # Discover mode
    discover_parser = subparsers.add_parser("discover", help="Discover tickers in a given market.")
    discover_parser.add_argument("market", type=str, help="Market to discover tickers in (e.g., america, turkey)")

    # Run mode (Telegram + paper trading)
    run_parser = subparsers.add_parser("run", help="Run Telegram-driven paper trading bot service.")
    run_parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial paper trading cash.")
    
    args = parser.parse_args()
    
    if args.mode == "suggestion":
        suggestion_mode(args.tickers)
    elif args.mode == "automatic":
        automatic_mode(args.daily_limit, args.aggressiveness, args.target_profit)
    elif args.mode == "backtest":
        backtest_mode(args.strategy, args.ticker, args.initial_investment)
    elif args.mode == "scrape":
        scrape_mode()
    elif args.mode == "discover":
        discover_mode(args.market)
    elif args.mode == "run":
        run_mode(args.initial_cash)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
