import yfinance as yf
import pandas as pd

print("Downloading Apple stock data...")
data = yf.download("AAPL", period="1mo")
print("Data shape:", data.shape)
print("First 5 rows:")
print(data.head())