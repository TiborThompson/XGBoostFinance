import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Create synthetic stock data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
price = 100 + np.cumsum(np.random.randn(500) * 2)

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Close': price,
    'Open': price + np.random.randn(500),
    'High': price + abs(np.random.randn(500) * 2),
    'Low': price - abs(np.random.randn(500) * 2),
    'Volume': np.random.randint(1000000, 10000000, size=500)
})
df.set_index('Date', inplace=True)

print("Created synthetic stock data")
print("Data shape:", df.shape)
print("First 5 rows:")
print(df.head())

# Plot the stock data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'])
plt.title('Synthetic Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.savefig('stock_plot.png')
print("Created plot: stock_plot.png")

# Simple moving average
window = 20
df['SMA'] = df['Close'].rolling(window=window).mean()

# Plot with SMA
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Price')
plt.plot(df.index, df['SMA'], label=f'{window}-day SMA', color='orange')
plt.title('Stock Price with Simple Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('stock_with_sma.png')
print("Created plot: stock_with_sma.png")

print("\nSuccess! The basic data processing and visualization is working.")