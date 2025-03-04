"""
Very simple backtest for the stock prediction model.
Just uses the basic features from original model.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm import tqdm
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import ProbabilisticBucketPredictor

def run_backtest(ticker='TSLA', start_date='2018-01-01', end_date=None, 
               initial_train_days=500, walk_forward_days=30):
    """Run a simple backtest of the model."""
    print(f"Running backtest for {ticker}")
    print(f"Initial training period: {initial_train_days} days")
    print(f"Walk-forward window: {walk_forward_days} days")
    
    # Download data
    print(f"Downloading {ticker} data...")
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    print(f"Data loaded with {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
    
    # Initialize model
    model = ProbabilisticBucketPredictor()
    
    # Prepare data
    from src.data_loader import StockDataLoader
    data_loader = StockDataLoader()
    X, y, X_train, X_test, y_train, y_test, feature_names = data_loader.prepare_data_for_buckets(df)
    
    # Define buckets
    model.define_return_buckets([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
    
    # Set up backtest tracking variables
    predictions = []
    actuals = []
    dates = []
    
    # Calculate total number of testing windows
    total_windows = (len(X) - initial_train_days + walk_forward_days - 1) // walk_forward_days
    
    # Create progress bar
    for i in range(total_windows):
        # Define window
        train_end = initial_train_days + i * walk_forward_days
        test_start = train_end
        test_end = min(test_start + walk_forward_days, len(X))
        
        # Skip if we've reached the end of data
        if test_start >= len(X):
            break
        
        # Get training data for this window
        window_X_train = X.iloc[:train_end]
        window_y_train = y.iloc[:train_end]
        
        # Get testing data for this window
        window_X_test = X.iloc[test_start:test_end]
        window_y_test = y.iloc[test_start:test_end]
        
        # Convert returns to buckets
        window_y_train_buckets = model._returns_to_buckets(window_y_train.values)
        
        # Build and train model
        model.build_model(num_classes=len(model.bucket_edges) + 1)
        model.train(window_X_train, window_y_train_buckets, ticker=f"{ticker}_window_{i}")
        
        # Make predictions for this window
        for j in range(len(window_X_test)):
            current_X = window_X_test.iloc[[j]]
            current_date = window_X_test.index[j]
            current_actual = window_y_test.iloc[j]
            
            # Get prediction with explanation
            prediction = model.predict_with_explanation(current_X)
            
            # Store results
            predictions.append(prediction['expected_return'])
            actuals.append(current_actual)
            dates.append(current_date)
        
        print(f"Completed window {i+1}/{total_windows}")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'date': dates,
        'actual_return': actuals,
        'predicted_return': predictions
    })
    results_df.set_index('date', inplace=True)
    
    # Calculate directional accuracy
    results_df['actual_direction'] = np.sign(results_df['actual_return'])
    results_df['predicted_direction'] = np.sign(results_df['predicted_return'])
    directional_accuracy = np.mean(results_df['actual_direction'] == results_df['predicted_direction'])
    
    # Calculate cumulative returns
    results_df['direction_strategy_return'] = results_df['actual_return'] * np.sign(results_df['predicted_return'])
    results_df['direction_strategy_cumulative'] = (1 + results_df['direction_strategy_return']).cumprod() - 1
    results_df['buy_hold_cumulative'] = (1 + results_df['actual_return']).cumprod() - 1
    
    # Print results
    print("\nBacktest Results:")
    print(f"Number of predictions: {len(results_df)}")
    print(f"Directional Accuracy: {directional_accuracy:.2%}")
    print(f"Direction Strategy Cumulative Return: {results_df['direction_strategy_cumulative'].iloc[-1]:.2%}")
    print(f"Buy & Hold Cumulative Return: {results_df['buy_hold_cumulative'].iloc[-1]:.2%}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(results_df.index, results_df['buy_hold_cumulative'], label='Buy & Hold')
    plt.plot(results_df.index, results_df['direction_strategy_cumulative'], label='Strategy')
    plt.title(f'{ticker} - Strategy vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{ticker}_strategy_performance.png')
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a simple backtest')
    parser.add_argument('--ticker', type=str, default='TSLA')
    parser.add_argument('--start', type=str, default='2020-01-01')
    parser.add_argument('--initial_train', type=int, default=500)
    parser.add_argument('--window', type=int, default=30)
    
    args = parser.parse_args()
    
    run_backtest(
        ticker=args.ticker,
        start_date=args.start,
        initial_train_days=args.initial_train,
        walk_forward_days=args.window
    )