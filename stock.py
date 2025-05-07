import os
import json
from datetime import datetime
from transformers import pipeline  # For LLM calls
from dotenv import load_dotenv

# Import your existing stock analysis functions
from report import fetch_stock_data, fetch_news, sentiment_analysis, summarize_news, moving_average, support_resistance, ARIMA_ALGO, detect_candlestick_patterns, generate_report

# Load environment variables
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize the LLM pipeline (e.g., Mistral-7B or Zephyr-7B)
llm_pipeline = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", max_new_tokens=200)

class StockAssistantChatbot:
    def __init__(self):
        self.conversation_history = []
        self.context = {
            "current_stock": None,
            "analysis_performed": False,
        }
        
        # CHANGE: Add prompt templates for different intents
        self.prompt_templates = {
            "STOCK_LOOKUP": """
                You are a stock market assistant. The user has asked for information about the stock {symbol}.
                Provide the current price, recent performance, and key metrics for {symbol}.
            """,
            "TECHNICAL_ANALYSIS": """
                You are a stock market assistant. The user has asked for technical analysis of {symbol}.
                Provide the 50-day and 200-day moving averages, support/resistance levels, and candlestick patterns for {symbol}.
            """,
            "NEWS_QUERY": """
                You are a stock market assistant. The user has asked for news about {symbol}.
                Summarize the latest news articles related to {symbol} and highlight any significant events.
            """,
            "PREDICTION_QUERY": """
                You are a stock market assistant. The user has asked for price predictions for {symbol}.
                Provide a prediction for the next day's price and explain the reasoning behind it.
            """,
            "SENTIMENT_QUERY": """
                You are a stock market assistant. The user has asked for sentiment analysis of {symbol}.
                Analyze the sentiment of recent news articles and provide an overall sentiment score for {symbol}.
            """,
            "RECOMMENDATION_QUERY": """
                You are a stock market assistant. The user has asked for a buy/sell recommendation for {symbol}.
                Based on technical analysis, sentiment, and price predictions, provide a recommendation for {symbol}.
            """,
            "GENERAL_QUERY": """
                You are a stock market assistant. The user has asked a general question about the stock market.
                Provide an educational and accurate response to the user's question.
            """,
            "OTHER": """
                You are a stock market assistant. The user's query is not related to stocks.
                Politely inform the user that you can only assist with stock-related queries.
            """
        }
    
    def add_to_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
        # Keep history manageable (last 20 exchanges)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def analyze_intent(self, user_message):
        """Use LLM to understand user's intent"""
        system_prompt = """
        You are an AI assistant that helps analyze stock-related queries. 
        Identify the user's intent from these categories:
        1. STOCK_LOOKUP - User wants information about a specific stock (extract the ticker symbol)
        2. TECHNICAL_ANALYSIS - User wants technical analysis
        3. NEWS_QUERY - User wants news about a stock
        4. PREDICTION_QUERY - User wants price predictions
        5. SENTIMENT_QUERY - User wants sentiment analysis
        6. RECOMMENDATION_QUERY - User wants buy/sell recommendations
        7. COMPARISON_QUERY - User wants to compare stocks (extract ticker symbols)
        8. GENERAL_QUERY - General question about stocks or the market
        9. OTHER - Not related to stocks

        Output format: JSON with keys "intent" and "symbols" (list of ticker symbols if applicable)
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        prompt = system_prompt + "\nUser: " + user_message + "\nAssistant:"
        response = llm_pipeline(prompt)[0]["generated_text"]

        try:
            intent_data = json.loads(response.split("Assistant:")[-1].strip())
        except json.JSONDecodeError:
            intent_data = {"intent": "OTHER", "symbols": []}
    
        return intent_data
    
    def execute_stock_analysis(self, symbol):
        """Execute full stock analysis for a given symbol"""
        try:
            # First verify if we can fetch the stock data
            df = fetch_stock_data(symbol, ALPHA_VANTAGE_API_KEY)
            if df is None or df.empty:
                return f"Sorry, I couldn't find data for the symbol {symbol}. Please check if it's a valid stock ticker."
            
            # Store current stock in context
            self.context["current_stock"] = symbol
            self.context["analysis_performed"] = True
            
            # Generate the full report
            report = generate_report(symbol, ALPHA_VANTAGE_API_KEY, NEWS_API_KEY)
            return report
            
        except Exception as e:
            return f"I encountered an error while analyzing {symbol}: {str(e)}"
    
    def execute_specific_analysis(self, analysis_type, symbol=None):
        """Execute a specific type of analysis"""
        if not symbol:
            symbol = self.context.get("current_stock")
            if not symbol:
                return "Please specify a stock symbol for analysis."
        
        try:
            df = fetch_stock_data(symbol, ALPHA_VANTAGE_API_KEY)
            if df is None or df.empty:
                return f"Sorry, I couldn't find data for the symbol {symbol}."
            
            if analysis_type == "TECHNICAL_ANALYSIS":
                ma_50 = moving_average(df, 50)
                ma_200 = moving_average(df, 200)
                support, resistance = support_resistance(df)
                patterns = detect_candlestick_patterns(df)
                
                result = f"""
                Technical Analysis for {symbol}:
                Current Price: {df['Close'].iloc[-1]:.2f}
                50-Day Moving Average: {ma_50:.2f}
                200-Day Moving Average: {ma_200:.2f}
                Support Level: {support:.2f}
                Resistance Level: {resistance:.2f}
                
                Candlestick Patterns:
                {', '.join(f"{pattern} ({signal})" for pattern, signal in patterns) if patterns else 'None'}
                """
                return result
                
            elif analysis_type == "NEWS_QUERY":
                news_articles = fetch_news(symbol, NEWS_API_KEY)
                news_summary = summarize_news(news_articles)
                return f"News Summary for {symbol}:\n{news_summary}"
                
            elif analysis_type == "PREDICTION_QUERY":
                arima_pred, error_arima = ARIMA_ALGO(df)
                return f"Prediction for {symbol}:\nARIMA Predicted Price (Next Day): {arima_pred:.2f}\nARIMA RMSE: {error_arima:.4f}"
                
            elif analysis_type == "SENTIMENT_QUERY":
                news_articles = fetch_news(symbol, NEWS_API_KEY)
                sentiments = [sentiment_analysis(article) for article in news_articles]
                avg_sentiment = sum(score for _, score in sentiments) / len(sentiments)
                
                sentiment_text = "positive" if avg_sentiment > 0 else "negative"
                return f"Sentiment Analysis for {symbol}:\nOverall sentiment is {sentiment_text} with score: {avg_sentiment:.2f}"
                
            elif analysis_type == "RECOMMENDATION_QUERY":
                news_articles = fetch_news(symbol, NEWS_API_KEY)
                sentiments = [sentiment_analysis(article) for article in news_articles]
                avg_sentiment = sum(score for _, score in sentiments) / len(sentiments)
                
                arima_pred, _ = ARIMA_ALGO(df)
                recommendation = "BUY" if avg_sentiment > 0 and arima_pred > df['Close'].iloc[-1] else "SELL"
                
                return f"Recommendation for {symbol}: {recommendation}"
                
        except Exception as e:
            return f"I encountered an error during the {analysis_type} for {symbol}: {str(e)}"
    
    # CHANGE: Add llm_call method for prompt-based LLM responses
    def llm_call(self, prompt):
        """Calls the LLM with the given prompt"""
        try:
            response = llm_pipeline(prompt)[0]["generated_text"]
            return response
        except Exception as e:
            return f"Error in generating response: {e}"
    
    def get_response(self, user_message):
        """Main function to process user message and generate a response"""
        self.add_to_history("user", user_message)
        
        # Analyze intent
        intent_data = self.analyze_intent(user_message)
        intent = intent_data.get("intent")
        symbols = intent_data.get("symbols", [])
        
        # Handle different intents
        response = ""
        
        if intent == "STOCK_LOOKUP" and symbols:
            symbol = symbols[0]
            prompt = self.prompt_templates["STOCK_LOOKUP"].format(symbol=symbol)
            response = self.llm_call(prompt)
            
        elif intent == "TECHNICAL_ANALYSIS":
            symbol = symbols[0] if symbols else self.context.get("current_stock")
            if symbol:
                prompt = self.prompt_templates["TECHNICAL_ANALYSIS"].format(symbol=symbol)
                response = self.llm_call(prompt)
            else:
                response = "Please specify a stock symbol for technical analysis."
                
        elif intent == "NEWS_QUERY":
            symbol = symbols[0] if symbols else self.context.get("current_stock")
            if symbol:
                prompt = self.prompt_templates["NEWS_QUERY"].format(symbol=symbol)
                response = self.llm_call(prompt)
            else:
                response = "Please specify a stock symbol for news analysis."
                
        elif intent == "PREDICTION_QUERY":
            symbol = symbols[0] if symbols else self.context.get("current_stock")
            if symbol:
                prompt = self.prompt_templates["PREDICTION_QUERY"].format(symbol=symbol)
                response = self.llm_call(prompt)
            else:
                response = "Please specify a stock symbol for price predictions."
                
        elif intent == "SENTIMENT_QUERY":
            symbol = symbols[0] if symbols else self.context.get("current_stock")
            if symbol:
                prompt = self.prompt_templates["SENTIMENT_QUERY"].format(symbol=symbol)
                response = self.llm_call(prompt)
            else:
                response = "Please specify a stock symbol for sentiment analysis."
                
        elif intent == "RECOMMENDATION_QUERY":
            symbol = symbols[0] if symbols else self.context.get("current_stock")
            if symbol:
                prompt = self.prompt_templates["RECOMMENDATION_QUERY"].format(symbol=symbol)
                response = self.llm_call(prompt)
            else:
                response = "Please specify a stock symbol for recommendations."
                
        elif intent == "GENERAL_QUERY":
            prompt = self.prompt_templates["GENERAL_QUERY"]
            response = self.llm_call(prompt)
            
        else:
            prompt = self.prompt_templates["OTHER"]
            response = self.llm_call(prompt)
        
        self.add_to_history("assistant", response)
        return response