import yfinance as yf
import csv


filepath="C:/Users/ranab/OneDrive/Desktop/Twitter Sentiment Analysis/Tsla_stock_price.csv"

ticker_symbol="TSLA"

ticker=yf.Ticker(ticker_symbol)

historical_data=ticker.history(period="20y")

historical_data[["Open","High","Low","Close","Volume"]].to_csv(filepath)