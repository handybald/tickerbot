
import logging

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# Positive / negative keyword sets used for simple title-level sentiment.
_POS_WORDS = frozenset({
    "up", "gain", "rise", "rose", "beat", "beats", "strong", "buy", "upgrade",
    "upgraded", "profit", "growth", "surge", "record", "rally", "bull", "bullish",
    "outperform", "positive", "higher", "rebound", "recovery",
})
_NEG_WORDS = frozenset({
    "down", "fall", "fell", "drop", "miss", "misses", "weak", "sell", "downgrade",
    "downgraded", "loss", "losses", "decline", "slump", "crash", "bear", "bearish",
    "underperform", "negative", "lower", "warning", "cut", "cuts",
})


def _keyword_sentiment(text: str) -> float:
    """Returns a [-1, 1] sentiment score based on keyword counts."""
    words = text.lower().split()
    pos = sum(1 for w in words if w in _POS_WORDS)
    neg = sum(1 for w in words if w in _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


class DataFetcher:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self._yf_ticker = yf.Ticker(ticker) if ticker else None
        # StockNews is optional; import lazily so the bot still works without it.
        self._stocknews = None
        if ticker:
            try:
                from stocknews import StockNews  # type: ignore
                self._stocknews = StockNews(ticker, save_news=False)
            except Exception:
                log.debug("stocknews not available for %s", ticker)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stock_news(self) -> pd.DataFrame:
        """Returns a DataFrame of recent news articles with sentiment scores.

        Tries StockNews first; falls back to yfinance news with keyword-based
        sentiment so the result is always usable.
        """
        df = self._try_stocknews()
        if df is not None and not df.empty:
            return df

        df = self._try_yfinance_news()
        if df is not None and not df.empty:
            return df

        return pd.DataFrame()

    def get_sentiment(self) -> float:
        """Returns a single [-1, 1] sentiment score for the ticker.

        > 0.2  → bullish signal
        < -0.2 → bearish signal
        """
        try:
            df = self.get_stock_news()
            if df.empty:
                return 0.0
            if "sentiment_title" in df.columns:
                scores = pd.to_numeric(df["sentiment_title"], errors="coerce").dropna()
                if not scores.empty:
                    return float(scores.mean())
            if "sentiment" in df.columns:
                scores = pd.to_numeric(df["sentiment"], errors="coerce").dropna()
                if not scores.empty:
                    return float(scores.mean())
        except Exception as exc:
            log.debug("get_sentiment error for %s: %s", self.ticker, exc)
        return 0.0

    def get_recommendations(self) -> pd.DataFrame | None:
        """Gets analyst buy/sell recommendations from yfinance."""
        if self._yf_ticker is None:
            return None
        try:
            return self._yf_ticker.recommendations
        except Exception as exc:
            log.debug("recommendations fetch failed for %s: %s", self.ticker, exc)
            return None

    def scrape_hedeffiyat(self) -> list[dict]:
        """Scrapes target price data from hedeffiyat.com.tr."""
        try:
            response = requests.get("https://hedeffiyat.com.tr/", timeout=20)
            response.raise_for_status()
        except Exception as exc:
            log.warning("hedeffiyat scrape failed: %s", exc)
            return []

        stocks = []
        try:
            soup = BeautifulSoup(response.content, "html.parser")
            rows = soup.select("table.table-bordered tbody tr")
            for row in rows:
                cols = row.select("td")
                if len(cols) > 4:
                    stocks.append({
                        "ticker": cols[0].text.strip(),
                        "last_price": cols[1].text.strip(),
                        "target_price": cols[2].text.strip(),
                        "potential": cols[4].text.strip(),
                    })
        except Exception as exc:
            log.warning("hedeffiyat parse error: %s", exc)

        return stocks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_stocknews(self) -> pd.DataFrame | None:
        if self._stocknews is None:
            return None
        try:
            df = self._stocknews.read_rss()
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            log.debug("stocknews RSS failed for %s: %s", self.ticker, exc)
        return None

    def _try_yfinance_news(self) -> pd.DataFrame | None:
        if self._yf_ticker is None:
            return None
        try:
            news = self._yf_ticker.news
            if not news:
                return None
            rows = []
            for item in news[:20]:
                title = item.get("title", "")
                summary = item.get("summary", "")
                text = f"{title} {summary}"
                rows.append({
                    "title": title,
                    "summary": summary,
                    "sentiment_title": _keyword_sentiment(title),
                    "sentiment": _keyword_sentiment(text),
                    "published": item.get("providerPublishTime", 0),
                    "source": item.get("publisher", ""),
                })
            return pd.DataFrame(rows)
        except Exception as exc:
            log.debug("yfinance news failed for %s: %s", self.ticker, exc)
            return None
