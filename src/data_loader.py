import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockDataLoader:
    def __init__(self, data_dir=None):
        # Use absolute path to ensure correct location
        if data_dir is None:
            # Get the project root directory (where src, data, and models are)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(project_root, 'data')
        else:
            self.data_dir = data_dir
            
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Data will be stored in: {self.data_dir}")
        
    def download_stock_data(self, ticker, period='2y', interval='1d'):
        """
        Download stock data for the given ticker
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Interval of data (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            pd.DataFrame: DataFrame with stock data
        """
        print(f"Downloading {ticker} data...")
        try:
            # Download data
            stock_data = yf.download(ticker, period=period, interval=interval)
            
            # Ensure the index is DatetimeIndex
            if not isinstance(stock_data.index, pd.DatetimeIndex):
                stock_data.index = pd.to_datetime(stock_data.index)
                
            # Ensure numeric columns
            for col in stock_data.columns:
                stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
            
            # Drop any rows with NaN values
            stock_data = stock_data.dropna()
            
            # Save to CSV
            file_path = os.path.join(self.data_dir, f"{ticker}_{interval}.csv")
            stock_data.to_csv(file_path)
            print(f"Data saved to {file_path}")
            
            return stock_data
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    def load_stock_data(self, ticker, interval='1d'):
        """
        Load stock data from CSV file or download if not available
        
        Args:
            ticker (str): Stock ticker symbol
            interval (str): Interval of data
            
        Returns:
            pd.DataFrame: DataFrame with stock data
        """
        file_path = os.path.join(self.data_dir, f"{ticker}_{interval}.csv")
        
        if os.path.exists(file_path):
            # Read CSV with flexible date parsing
            df = pd.read_csv(file_path, index_col=0)
            
            # Convert index to datetime with flexible format but suppress warnings
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    # Try to convert with flexible format detection
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    # Check for NaT values and replace them
                    if df.index.isna().any():
                        # Create a date range for the NaT values
                        valid_idx = ~df.index.isna()
                        if valid_idx.any():
                            # If we have some valid dates, use the first one as reference
                            start_date = df.index[valid_idx][0]
                        else:
                            # Otherwise, use a default date
                            start_date = pd.Timestamp('2021-01-01')
                        # Replace NaT with sequential dates
                        nat_count = (~valid_idx).sum()
                        if nat_count > 0:
                            new_dates = pd.date_range(start=start_date, periods=nat_count, freq='D')
                            df.index.values[~valid_idx] = new_dates
                except Exception as e:
                    print(f"Warning: Error parsing dates: {e}")
                    # If there's an issue with the index, create a new date range index
                    df.index = pd.date_range(start='2021-01-01', periods=len(df), freq='D')
            
            # Ensure numeric columns are properly converted
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        else:
            return self.download_stock_data(ticker, interval=interval)
    
    def generate_synthetic_data(self, periods=500, volatility=0.02, trend=0.0001, seed=42):
        """
        Generate synthetic stock data for testing
        
        Args:
            periods (int): Number of periods (days) to generate
            volatility (float): Daily volatility parameter
            trend (float): Daily trend parameter
            seed (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: DataFrame with synthetic stock data
        """
        np.random.seed(seed)
        dates = pd.date_range(start='2020-01-01', periods=periods, freq='D')
        
        # Generate random walk with drift
        returns = np.random.normal(trend, volatility, periods)
        price = 100 * np.exp(np.cumsum(returns))
        
        # Add some seasonality and cyclicality
        time = np.arange(periods)
        seasonal_component = 2 * np.sin(2 * np.pi * time / 252)  # Annual seasonality
        cyclical_component = 5 * np.sin(2 * np.pi * time / 1260)  # 5-year cycle
        
        # Add components to price
        price = price + seasonal_component + cyclical_component
        
        # Create DataFrame with OHLC data
        df = pd.DataFrame({
            'Open': price * (1 - 0.005 * np.random.rand(periods)),
            'High': price * (1 + 0.01 * np.random.rand(periods)),
            'Low': price * (1 - 0.01 * np.random.rand(periods)),
            'Close': price,
            'Volume': np.random.randint(100000, 10000000, periods)
        }, index=dates)
        
        # Ensure High is the highest, Low is the lowest
        df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
        df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
        
        return df
        
    def prepare_data_for_buckets(self, df, lookback_periods=[5, 10, 20, 30, 60], test_size=0.2):
        """
        Prepare stock data for bucket prediction model
        
        Args:
            df (pd.DataFrame): DataFrame with stock data (OHLCV)
            lookback_periods (list): List of lookback periods for feature engineering
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: (X, y, X_train, X_test, y_train, y_test, feature_names)
        """
        # Create features DataFrame
        features = pd.DataFrame(index=df.index)
        
        # Calculate returns
        features['daily_return'] = df['Close'].pct_change()
        
        # Recent returns over different periods
        for period in lookback_periods:
            # Cumulative return over period
            features[f'return_{period}d'] = df['Close'].pct_change(period)
            
            # Rolling statistics
            features[f'return_mean_{period}d'] = features['daily_return'].rolling(period).mean()
            features[f'return_std_{period}d'] = features['daily_return'].rolling(period).std()
            features[f'return_max_{period}d'] = features['daily_return'].rolling(period).max()
            features[f'return_min_{period}d'] = features['daily_return'].rolling(period).min()
            
        # Momentum indicators
        for short_period in [5, 10, 20]:
            for long_period in [30, 60]:
                if short_period < long_period:
                    features[f'momentum_{short_period}_{long_period}'] = (
                        features[f'return_{short_period}d'] - features[f'return_{long_period}d']
                    )
        
        # Volatility measures
        for period in lookback_periods:
            features[f'volatility_{period}d'] = features['daily_return'].rolling(period).std() * np.sqrt(252)  # Annualized
            
        # Volume indicators
        if 'Volume' in df.columns:
            features['volume_change'] = df['Volume'].pct_change()
            for period in lookback_periods:
                features[f'volume_mean_{period}d'] = df['Volume'].rolling(period).mean()
                features[f'volume_std_{period}d'] = df['Volume'].rolling(period).std()
                features[f'volume_change_mean_{period}d'] = features['volume_change'].rolling(period).mean()
        
        # Price-based indicators
        features['close_to_high_ratio'] = df['Close'] / df['High'].rolling(20).max()
        features['close_to_low_ratio'] = df['Close'] / df['Low'].rolling(20).min()
        
        # Simple moving averages and crossovers
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = df['Close'].rolling(period).mean()
            
        # SMA crossovers
        features['sma_5_cross_10'] = features['sma_5'] / features['sma_10'] - 1
        features['sma_10_cross_20'] = features['sma_10'] / features['sma_20'] - 1
        features['sma_20_cross_50'] = features['sma_20'] / features['sma_50'] - 1
        features['sma_50_cross_200'] = features['sma_50'] / features['sma_200'] - 1
        
        # Drop rows with NaN values resulting from rolling calculations
        features = features.dropna()
        
        # Target variable: next day's return (to be converted to buckets)
        features['next_return'] = features['daily_return'].shift(-1)
        
        # Drop the last row that will have NaN for next_return
        features = features.dropna()
        
        # Remove the daily_return column as it's just used for calculations
        X = features.drop(['daily_return', 'next_return'], axis=1)
        y = features['next_return']
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        feature_names = X.columns.tolist()
        
        return X, y, X_train, X_test, y_train, y_test, feature_names
    
    def prepare_data_for_lstm(self, df, sequence_length=60, train_split=0.8):
        """
        Legacy method for preparing data for LSTM model
        
        Args:
            df (pd.DataFrame): Stock data DataFrame
            sequence_length (int): Number of time steps in each sequence
            train_split (float): Ratio of training data
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, scaler)
        """
        # Extract closing prices and convert to numpy array
        data = df['Close'].values.reshape(-1, 1)
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length])
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_test, y_test, scaler