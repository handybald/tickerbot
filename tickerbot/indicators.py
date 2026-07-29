
import pandas as pd
import numpy as np


def calculate_sma(data, window):
    """Simple Moving Average."""
    return data['Close'].rolling(window=window).mean()


def calculate_ema(data, window):
    """Exponential Moving Average."""
    return data['Close'].ewm(span=window, adjust=False).mean()


def calculate_rsi(data, window=14):
    """
    Relative Strength Index using Wilder's EMA smoothing (the standard method).
    Previous implementation used SMA which gave inaccurate values.
    """
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Wilder's smoothing: alpha = 1/window  <=>  com = window - 1
    avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0, other=1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_macd(data, slow=26, fast=12, signal=9):
    """Moving Average Convergence Divergence."""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    """Bollinger Bands."""
    sma = calculate_sma(data, window)
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band


def calculate_atr(data, window=14):
    """
    Average True Range - measures volatility.
    Used for dynamic stop-loss sizing and confidence scaling.
    Requires 'High', 'Low', 'Close' columns.
    """
    if not {'High', 'Low', 'Close'}.issubset(data.columns):
        return pd.Series(dtype=float, index=data.index)
    high = data['High']
    low = data['Low']
    close = data['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    return atr


def calculate_stochastic(data, k_window=14, d_window=3):
    """
    Stochastic Oscillator (%K and %D).
    %K < 20 = oversold (bullish), %K > 80 = overbought (bearish).
    Requires 'High', 'Low', 'Close' columns.
    """
    if not {'High', 'Low', 'Close'}.issubset(data.columns):
        empty = pd.Series(dtype=float, index=data.index)
        return empty, empty
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    denom = (high_max - low_min).replace(0, np.nan)
    stoch_k = 100.0 * (data['Close'] - low_min) / denom
    stoch_d = stoch_k.rolling(window=d_window).mean()
    return stoch_k, stoch_d


def calculate_adx(data, window=14):
    """
    Average Directional Index - measures trend strength (not direction).
    ADX > 25: trending market; ADX < 20: ranging/sideways market.
    Also returns DI+ and DI- for direction.
    Requires 'High', 'Low', 'Close' columns.
    """
    if not {'High', 'Low', 'Close'}.issubset(data.columns):
        empty = pd.Series(dtype=float, index=data.index)
        return empty, empty, empty

    high = data['High']
    low = data['Low']
    close = data['Close']

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    dm_plus = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = tr.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    sm_dm_plus = dm_plus.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    sm_dm_minus = dm_minus.ewm(com=window - 1, min_periods=window, adjust=False).mean()

    atr_safe = atr.replace(0, np.nan)
    di_plus = 100.0 * sm_dm_plus / atr_safe
    di_minus = 100.0 * sm_dm_minus / atr_safe

    di_sum = (di_plus + di_minus).replace(0, np.nan)
    dx = 100.0 * (di_plus - di_minus).abs() / di_sum
    adx = dx.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    return adx, di_plus, di_minus


def calculate_williams_r(data, window=14):
    """
    Williams %R - momentum oscillator.
    Values near 0 = overbought; values near -100 = oversold.
    Typical thresholds: > -20 overbought, < -80 oversold.
    Requires 'High', 'Low', 'Close' columns.
    """
    if not {'High', 'Low', 'Close'}.issubset(data.columns):
        return pd.Series(dtype=float, index=data.index)
    high_max = data['High'].rolling(window=window).max()
    low_min = data['Low'].rolling(window=window).min()
    denom = (high_max - low_min).replace(0, np.nan)
    wr = -100.0 * (high_max - data['Close']) / denom
    return wr


def calculate_vwap(data):
    """
    Volume Weighted Average Price.
    Price above VWAP = bullish bias; price below VWAP = bearish bias.
    Best suited for intraday timeframes.
    Requires 'High', 'Low', 'Close', 'Volume' columns.
    """
    if not {'High', 'Low', 'Close', 'Volume'}.issubset(data.columns):
        return pd.Series(dtype=float, index=data.index)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3.0
    cum_tp_vol = (typical_price * data['Volume']).cumsum()
    cum_vol = data['Volume'].cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol
