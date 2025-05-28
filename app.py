from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from yahooquery import Ticker
from transformers import pipeline
from dotenv import load_dotenv
import requests, os

load_dotenv()
app = FastAPI()

# Load API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "7f18a8b195754744b76048cc184febb7")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_API_URL = 'https://www.alphavantage.co/query'

# Jinja templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")  # if you serve CSS/JS

# Sentiment Pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    stock = "AAPL"
    stocks = {
        "AAPL": "Apple Inc.",
        "TSLA": "Tesla, Inc.",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com, Inc.",
        "MSFT": "Microsoft Corporation",
    }
    news = get_stock_news(stock)
    support_resistance = get_support_resistance(stock)
    return templates.TemplateResponse("web.html", {
        "request": request,
        "stock": stock,
        "news": news,
        "stocks": stocks,
        "support_resistance": support_resistance
    })

# News fetch
def get_stock_news(stock):
    url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={NEWS_API_KEY}&sortBy=publishedAt&language=en"
    response = requests.get(url)
    return response.json().get("articles", [])[:5] if response.status_code == 200 else []

# Support/Resistance
def get_support_resistance(symbol):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(ALPHA_VANTAGE_API_URL, params=params)
    data = response.json()
    if 'Time Series (Daily)' not in data:
        return {"support": "N/A", "resistance": "N/A"}
    prices = [float(v['4. close']) for v in data['Time Series (Daily)'].values()]
    return {
        "support": round(min(prices), 2),
        "resistance": round(max(prices), 2)
    }

# Yahoo Stock Data
def get_stock_data(ticker):
    try:
        stock = Ticker(ticker)
        data = stock.summary_detail[ticker]
        return {
            "price": data.get("regularMarketPrice", "N/A"),
            "change": data.get("regularMarketChangePercent", "N/A"),
            "marketCap": data.get("marketCap", "N/A"),
            "volume": data.get("regularMarketVolume", "N/A"),
        }
    except Exception as e:
        return {"error": str(e)}

# Sentiment analysis
def sentiment_analysis(article):
    result = sentiment_analyzer(article)[0]
    score = round(result["score"] * 10) if result["label"] == "POSITIVE" else round(-result["score"] * 10)
    return result["label"], score

# API - Stock Data
@app.get("/stock/{ticker}")
async def stock_data(ticker: str):
    return get_stock_data(ticker)

# API - Support/Resistance
@app.get("/stock/{ticker}/sr")
async def stock_sr(ticker: str):
    return get_support_resistance(ticker)

# API - Sentiment Analysis
@app.post("/analyze")
async def analyze(stock: str = Form(...)):
    news_articles = get_stock_news(stock)
    sentiment_results = [sentiment_analysis(article["title"] + " " + article.get("description", "")) for article in news_articles]
    total_score = sum(score for _, score in sentiment_results)
    avg_score = total_score / len(sentiment_results) if sentiment_results else 0
    sentiment_label = "Positive" if avg_score > 0 else "Negative" if avg_score < 0 else "Neutral"
    prediction = "Stock likely to rise 📈" if avg_score > 0 else "Stock likely to fall 📉" if avg_score < 0 else "Stock likely to remain stable ➖"
    return JSONResponse({
        "sentiment_label": sentiment_label,
        "prediction": prediction,
        "articles": news_articles,
        "sentiments": sentiment_results,
        "average_score": avg_score
    })

# ✅ Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
