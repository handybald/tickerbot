
import pandas as pd
from .indicators import calculate_rsi

class Backtester:
    def __init__(self, strategy, data, initial_investment):
        self.strategy = strategy
        self.data = data
        self.initial_investment = initial_investment
        self.cash = initial_investment
        self.shares = 0
        self.portfolio_value = initial_investment

    def run(self):
        """
        Runs the backtest.
        """
        print("Running backtest...")
        
        rsi = calculate_rsi(self.data)
        
        for i in range(len(self.data)):
            # Get the signal from the strategy
            signal = self._get_signal(rsi.iloc[i])
            
            # Execute the trade
            self._execute_trade(signal, self.data['Close'].iloc[i])
            
            # Update portfolio value
            self.portfolio_value = self.cash + self.shares * self.data['Close'].iloc[i]

        print("Backtest finished.")
        print(f"Final portfolio value: {self.portfolio_value:.2f}")

    def _get_signal(self, rsi_value):
        """
        Gets the trading signal from the strategy.
        """
        if self.strategy == "rsi":
            if rsi_value < 30:
                return "BUY"
            elif rsi_value > 70:
                return "SELL"
            else:
                return "HOLD"
        else:
            return "HOLD"

    def _execute_trade(self, signal, price):
        """
        Executes a trade.
        """
        if signal == "BUY":
            if self.cash > price:
                self.shares += 1
                self.cash -= price
        elif signal == "SELL":
            if self.shares > 0:
                self.shares -= 1
                self.cash += price
