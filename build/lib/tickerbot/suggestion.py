
def generate_suggestion(rsi, macd, signal_line, upper_band, lower_band, price, sma_400, recommendations, avg_sentiment):
    """
    Generates a trading suggestion based on a variety of indicators.
    """
    score = 0

    # RSI
    if rsi < 30:
        score += 1
    elif rsi > 70:
        score -= 1

    # MACD
    if macd > signal_line:
        score += 1
    else:
        score -= 1

    # Bollinger Bands
    if price < lower_band:
        score += 1
    elif price > upper_band:
        score -= 1

    # 400-day SMA
    if sma_400 is not None:
        if price > sma_400:
            score += 1
        else:
            score -= 1

    # Analyst Recommendations
    if recommendations is not None and not recommendations.empty:
        recommendations = recommendations.iloc[0]
        if recommendations['strongBuy'] > 0 or recommendations['buy'] > 0:
            score += 1
        elif recommendations['strongSell'] > 0 or recommendations['sell'] > 0:
            score -= 1
            
    # News Sentiment
    if avg_sentiment > 0.2:
        score += 1
    elif avg_sentiment < -0.2:
        score -=1

    if score >= 2:
        return "BUY"
    elif score <= -2:
        return "SELL"
    else:
        return "HOLD"
