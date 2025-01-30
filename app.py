from flask import Flask, render_template, request, jsonify
import openai
from transformers import pipeline
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import base64

# Configure OpenAI API key securely
openai.api_key = os.getenv("ADD YOUR OPENAI KEY")  # Or set directly as openai.api_key = "your_openai_api_key"
#gemini_api_key = "your_gemini_api_key"  # Replace with Gemini's API Key

# Initialize the Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Initialize Flask
app = Flask(__name__)

# Fetch stock data
def fetch_stock_data(ticker, period="1mo"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

# Sentiment analysis
def analyze_sentiment(texts):
    try:
        return sentiment_pipeline(texts)
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return None

# Predict stock trends
def predict_trends(ticker_data):
    try:
        summarized_data = ticker_data[['Close']].describe().to_string()
        prompt = f"Given this stock data summary:\n{summarized_data}\nPredict future trends for the next week."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error in trend prediction: {e}")
        return None

# Generate investment recommendation
def generate_recommendation(sentiments, trends):
    try:
        prompt = f"The sentiment analysis is: {sentiments}. The trends are: {trends}. Provide investment advice."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error in recommendation generation: {e}")
        return None

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    period = request.form['period']
    headlines = request.form.getlist('headlines')

    # Fetch and plot stock data
    stock_data = fetch_stock_data(ticker, period)
    if stock_data is None:
        return jsonify({"error": "Failed to fetch stock data"}), 500

    # Perform sentiment analysis
    sentiments = analyze_sentiment(headlines)
    trends = predict_trends(stock_data.tail(5))
    recommendation = generate_recommendation(sentiments, trends)

    # Plot the stock data
    fig, ax = plt.subplots()
    stock_data['Close'].plot(ax=ax, title=f"{ticker} Stock Closing Prices", xlabel="Date", ylabel="Price (USD)")

    # Save plot to a string buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode()

    return render_template('result.html', plot_url=plot_url, sentiments=sentiments, trends=trends, recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
