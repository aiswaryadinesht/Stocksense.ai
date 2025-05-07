import requests
import pandas as pd
import matplotlib.pyplot as plt
import math
import talib
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from transformers import pipeline
import os
alpha_vantage_api_key = 'QT2ND911G9114QCZ'
news_api_key = '7f18a8b195754744b76048cc184febb7'
# Fetch stock data from Alpha Vantage
def fetch_stock_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=compact"
    response = requests.get(url)
    data = response.json()
    if 'Time Series (Daily)' not in data:
        return None

    df = pd.DataFrame(data['Time Series (Daily)']).T
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    df = df.astype(float).sort_index().reset_index()
    df.rename(columns={'index': 'Date'}, inplace=True)
    df['Code'] = symbol
    return df

# Fetch news using News API
def fetch_news(symbol, api_key):
    url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return [article['title'] + ' ' + article.get('description', '') for article in articles[:5]]

# Sentiment analysis using Hugging Face
def sentiment_analysis(article):
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=0)
    result = sentiment_analyzer(article)
    label = result[0]['label']
    score = result[0]['score']
    sentiment_score = round(score * 10) if label == 'POSITIVE' else round(-score * 10)
    return label, sentiment_score

# Summarize news articles into a paragraph
def summarize_news(articles):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
    combined_text = " ".join(articles)
    summary = summarizer(combined_text, max_length=200, min_length=80, do_sample=False)
    return summary[0]['summary_text']

# Moving Average calculation
def moving_average(df, window):
    return df['Close'].rolling(window=window).mean().iloc[-1]

# Identify support and resistance levels
def support_resistance(df):
    support = min(df['Low'].tail(30))
    resistance = max(df['High'].tail(30))
    return support, resistance

# ARIMA model for stock price prediction
def ARIMA_ALGO(df):
    
    def arima_model(train, test):
        history = [x for x in train]
        predictions = []
        for t in range(len(test)):
            model = ARIMA(history, order=(6, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()[0]
            predictions.append(output)
            history.append(test[t])
        return predictions

    df['Price'] = df['Close']

    # Save price trends
    os.makedirs('static', exist_ok=True)
    plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(df['Price'])
    plt.savefig('static/Trends.png')
    plt.close()

    # ARIMA prediction
    quantity = df['Price'].values
    size = int(len(quantity) * 0.80)
    train, test = quantity[:size], quantity[size:]

    predictions = arima_model(train, test)

    plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(test, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.legend(loc=4)
    plt.savefig('static/ARIMA.png')
    plt.close()

    arima_pred = predictions[-1]
    error_arima = math.sqrt(mean_squared_error(test, predictions))

    return arima_pred, error_arima
# Detect Candlestick Patterns
def detect_candlestick_patterns(df):
    patterns = {
        "Doji": talib.CDLDOJI, "Hammer": talib.CDLHAMMER, "Inverted Hammer": talib.CDLINVERTEDHAMMER,
        "Engulfing": talib.CDLENGULFING, "Shooting Star": talib.CDLSHOOTINGSTAR, "Hanging Man": talib.CDLHANGINGMAN,
        "Morning Star": talib.CDLMORNINGSTAR, "Evening Star": talib.CDLEVENINGSTAR, "Abandoned Baby": talib.CDLABANDONEDBABY,
        "Dark Cloud Cover": talib.CDLDARKCLOUDCOVER, "Piercing Line": talib.CDLPIERCING, "Three White Soldiers": talib.CDL3WHITESOLDIERS,
        "Three Black Crows": talib.CDL3BLACKCROWS, "Three Inside Up/Down": talib.CDL3INSIDE, "Three Outside Up/Down": talib.CDL3OUTSIDE,
        "Belt-hold": talib.CDLBELTHOLD, "Marubozu": talib.CDLMARUBOZU, "Harami": talib.CDLHARAMI,
        "Harami Cross": talib.CDLHARAMICROSS, "High Wave": talib.CDLHIGHWAVE, "Doji Star": talib.CDLDOJISTAR,
        "Dragonfly Doji": talib.CDLDRAGONFLYDOJI, "Gravestone Doji": talib.CDLGRAVESTONEDOJI, "Long-legged Doji": talib.CDLLONGLEGGEDDOJI,
        "Spinning Top": talib.CDLSPINNINGTOP, "Upside Gap Two Crows": talib.CDLUPSIDEGAP2CROWS, "Tasuki Gap": talib.CDLTASUKIGAP,
        "Unique Three River": talib.CDLUNIQUE3RIVER, "Gap Side-By-Side White Lines": talib.CDLGAPSIDESIDEWHITE, "Matching Low": talib.CDLMATCHINGLOW
    }

    signals = []
    for pattern_name, pattern_function in patterns.items():
        result = pattern_function(df['Open'], df['High'], df['Low'], df['Close'])
        last_signal = result.iloc[-1]
        if last_signal > 0:
            signals.append((pattern_name, "Bullish"))
        elif last_signal < 0:
            signals.append((pattern_name, "Bearish"))

    return signals


# Generate the stock analysis report
def generate_report(symbol, alpha_vantage_api_key, news_api_key):
    df = fetch_stock_data(symbol, alpha_vantage_api_key)

    if df is None or df.empty:
        return f"No data found for {symbol}. Please check the stock symbol."

    # Fetch and summarize news
    news_articles = fetch_news(symbol, news_api_key)
    news_summary = summarize_news(news_articles)

    # Sentiment analysis
    sentiments = [sentiment_analysis(article) for article in news_articles]
    avg_sentiment = sum(score for _, score in sentiments) / len(sentiments)

    # Predict stock price
    
    arima_pred, error_arima = ARIMA_ALGO(df)

    # Moving averages
    ma_50 = moving_average(df, 50)
    ma_200 = moving_average(df, 200)

    # Support and resistance
    support, resistance = support_resistance(df)

     # Detect Candlestick Patterns
    patterns = detect_candlestick_patterns(df)

    # Predict Stock Price
    #arima_pred, error_arima = ARIMA_ALGO(df)

    # Determine trade signal from patterns
    bullish_count = sum(1 for _, signal in patterns if signal == "Bullish")
    bearish_count = sum(1 for _, signal in patterns if signal == "Bearish")


    recommendation = "BUY" if avg_sentiment > 0 and arima_pred > df['Close'].iloc[-1] else "SELL"

    report = f"""
    📊 Advanced Stock Analysis Report

    Stock Symbol: {symbol}
    Current Price: {df['Close'].iloc[-1]:.2f}

    🔮 Predictions:
    ARIMA Predicted Price (Next Day): {arima_pred:.2f}
    ARIMA RMSE: {error_arima:.4f}

    📈 Technical Indicators:
    50-Day Moving Average: {ma_50:.2f}
    200-Day Moving Average: {ma_200:.2f}

    Support Level: {support:.2f}
    Resistance Level: {resistance:.2f}

    Sentiment Analysis (Avg Score): {avg_sentiment:.2f}
    Recommendation: {recommendation}

     Candlestick Patterns Detected:
    {', '.join(f"{pattern} ({signal})" for pattern, signal in patterns) if patterns else 'None'}

    Recommendation: {recommendation}
    

    🗞 News Summary:
    {news_summary}
    """

    return report
