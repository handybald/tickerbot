
import yfinance as yf
from stocknews import StockNews
import requests
from bs4 import BeautifulSoup

class DataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker
        if ticker:
            self.yf_ticker = yf.Ticker(ticker)
            self.stock_news = StockNews(ticker, save_news=False)

    def get_stock_news(self):
        """
        Gets stock news and performs sentiment analysis.
        """
        df_news = self.stock_news.read_rss()
        return df_news

    def get_recommendations(self):
        """
        Gets analyst recommendations.
        """
        return self.yf_ticker.recommendations
        
    def scrape_hedeffiyat(self):
        """
        Scrapes data from hedeffiyat.com.tr.
        """
        url = "https://hedeffiyat.com.tr/"
        response = requests.get(url)
        print(f"Response status code: {response.status_code}")
        soup = BeautifulSoup(response.content, "html.parser")
        
        stocks = []
        rows = soup.select("table.table-bordered tbody tr")
        print(f"Found {len(rows)} rows in the table.")
        for row in rows:
            cols = row.select("td")
            if len(cols) > 5:
                stock = {
                    "ticker": cols[0].text.strip(),
                    "last_price": cols[1].text.strip(),
                    "target_price": cols[2].text.strip(),
                    "potential": cols[4].text.strip(),
                }
                stocks.append(stock)
        return stocks
