"""
Backtest the probabilistic bucket prediction model to evaluate its performance.
This script walks forward through time, training on historical data and 
evaluating predictions on out-of-sample data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import StockDataLoader
from src.model import ProbabilisticBucketPredictor

def calculate_expected_returns(probabilities, bucket_edges):
    """
    Calculate expected return from probability distribution across buckets.
    
    Args:
        probabilities (np.array): Predicted probabilities for each bucket
        bucket_edges (list): Edges that define the buckets
        
    Returns:
        float: Expected return
    """
    # Define representative values for each bucket
    bucket_values = []
    for i in range(len(bucket_edges) + 1):
        if i == 0:  # < first edge (e.g., < -3%)
            bucket_values.append(bucket_edges[0] - 0.01)  # e.g., -4%
        elif i == len(bucket_edges):  # > last edge (e.g., > 3%)
            bucket_values.append(bucket_edges[-1] + 0.01)  # e.g., 4%
        else:  # Between two edges
            # Middle point between two edges
            bucket_values.append((bucket_edges[i-1] + bucket_edges[i]) / 2)
    
    # Calculate expected value
    expected_return = 0.0
    for i in range(len(bucket_values)):
        expected_return += bucket_values[i] * probabilities[i]
    
    return expected_return

def run_backtest(ticker, start_date=None, end_date=None, 
                initial_train_days=252, walk_forward_days=21, 
                bucket_edges=None, verbose=True):
    """
    Run a walk-forward backtest of the probabilistic bucket prediction model.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for data in format 'YYYY-MM-DD'
        end_date (str): End date for data in format 'YYYY-MM-DD'
        initial_train_days (int): Number of days for initial training period
        walk_forward_days (int): Number of days to predict before retraining
        bucket_edges (list): Custom bucket edges
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Backtest results
    """
    if verbose:
        print(f"Running backtest for {ticker}")
        print(f"Initial training period: {initial_train_days} days")
        print(f"Walk-forward window: {walk_forward_days} days")
    
    # Load data
    data_loader = StockDataLoader()
    
    # Download the data directly with yfinance since our loader doesn't support date ranges
    if start_date or end_date:
        import yfinance as yf
        print(f"Downloading {ticker} data from {start_date} to {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    else:
        df = data_loader.load_stock_data(ticker, interval='1d')
    
    if verbose:
        print(f"Data loaded with {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
    
    # Prepare data for bucket prediction
    # We need to implement a simplified version of prepare_data_for_buckets
    # since we can't access _engineer_features directly
    lookback_periods = [5, 10, 20, 30, 60]
    
    # Create features DataFrame
    features_df = pd.DataFrame(index=df.index)
    
    # Calculate returns
    features_df['daily_return'] = df['Close'].pct_change()
    
    # Recent returns over different periods
    for period in lookback_periods:
        # Cumulative return over period
        features_df[f'return_{period}d'] = df['Close'].pct_change(period)
        
        # Rolling statistics
        features_df[f'return_mean_{period}d'] = features_df['daily_return'].rolling(period).mean()
        features_df[f'return_std_{period}d'] = features_df['daily_return'].rolling(period).std()
        features_df[f'return_max_{period}d'] = features_df['daily_return'].rolling(period).max()
        features_df[f'return_min_{period}d'] = features_df['daily_return'].rolling(period).min()
        
    # Momentum indicators
    for short_period in [5, 10, 20]:
        for long_period in [30, 60]:
            if short_period < long_period:
                features_df[f'momentum_{short_period}_{long_period}'] = (
                    features_df[f'return_{short_period}d'] - features_df[f'return_{long_period}d']
                )
    
    # Volatility measures
    for period in lookback_periods:
        features_df[f'volatility_{period}d'] = features_df['daily_return'].rolling(period).std() * np.sqrt(252)
    
    # Target variable: next day's return
    features_df['next_return'] = features_df['daily_return'].shift(-1)
    
    # Drop rows with NaN values
    features_df = features_df.dropna()
    
    # Initialize model
    model = ProbabilisticBucketPredictor()
    
    # Define return buckets
    if bucket_edges is not None:
        model.define_return_buckets(bucket_edges)
    else:
        model.define_return_buckets()
    
    # Set up backtest tracking variables
    predictions = []
    actuals = []
    dates = []
    all_probabilities = []
    
    # Calculate total number of testing windows
    total_days = len(features_df) - initial_train_days
    total_windows = (total_days + walk_forward_days - 1) // walk_forward_days
    
    # Create progress bar
    pbar = tqdm(total=total_windows, disable=not verbose)
    
    # Walk forward through time
    for i in range(total_windows):
        train_end = initial_train_days + i * walk_forward_days
        test_start = train_end
        test_end = min(test_start + walk_forward_days, len(features_df))
        
        # Skip if we've reached the end of data
        if test_start >= len(features_df):
            break
            
        # Get training data
        X_train = features_df.drop('next_return', axis=1).iloc[:train_end]
        y_train = features_df['next_return'].iloc[:train_end]
        
        # Get testing data
        X_test = features_df.drop('next_return', axis=1).iloc[test_start:test_end]
        y_test = features_df['next_return'].iloc[test_start:test_end]
        
        # Convert returns to buckets
        y_train_buckets = model._returns_to_buckets(y_train.values)
        
        # Build and train model
        model.build_model(num_classes=len(model.bucket_edges) + 1)
        model.train(X_train, y_train_buckets, use_calibration=True, ticker=ticker + "_backtest")
        
        # Make predictions
        for j in range(len(X_test)):
            current_X = X_test.iloc[[j]]
            current_date = X_test.index[j]
            current_actual = y_test.iloc[j]
            
            # Predict probabilities
            probs = model.predict_probabilities(current_X)[0]
            
            # Calculate expected return
            expected_return = calculate_expected_returns(probs, model.bucket_edges)
            
            # Store results
            predictions.append(expected_return)
            actuals.append(current_actual)
            dates.append(current_date)
            all_probabilities.append(probs)
        
        pbar.update(1)
    
    pbar.close()
    
    # Convert to DataFrames
    results_df = pd.DataFrame({
        'date': dates,
        'actual_return': actuals,
        'predicted_return': predictions
    })
    results_df.set_index('date', inplace=True)
    
    # Calculate metrics
    mse = np.mean((results_df['actual_return'] - results_df['predicted_return'])**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(results_df['actual_return'] - results_df['predicted_return']))
    
    # Calculate directional accuracy (sign of return)
    results_df['actual_direction'] = np.sign(results_df['actual_return'])
    results_df['predicted_direction'] = np.sign(results_df['predicted_return'])
    directional_accuracy = np.mean(results_df['actual_direction'] == results_df['predicted_direction'])
    
    # Calculate Win Rate (correct prediction of positive vs negative)
    results_df['actual_positive'] = results_df['actual_return'] > 0
    results_df['predicted_positive'] = results_df['predicted_return'] > 0
    win_rate = np.mean(results_df['actual_positive'] == results_df['predicted_positive'])
    
    # Calculate return from a trading strategy that invests based on prediction
    results_df['strategy_return'] = results_df['actual_return'] * np.sign(results_df['predicted_return'])
    strategy_cumulative_return = (1 + results_df['strategy_return']).cumprod() - 1
    buy_hold_cumulative_return = (1 + results_df['actual_return']).cumprod() - 1
    
    # Plot results
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Actual vs Predicted Returns
    plt.subplot(3, 1, 1)
    plt.plot(results_df.index, results_df['actual_return'], label='Actual Return', alpha=0.7)
    plt.plot(results_df.index, results_df['predicted_return'], label='Predicted Return', alpha=0.7)
    plt.title(f'{ticker} - Actual vs Predicted Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Returns
    plt.subplot(3, 1, 2)
    plt.plot(buy_hold_cumulative_return.index, buy_hold_cumulative_return, 
             label='Buy & Hold', alpha=0.7)
    plt.plot(strategy_cumulative_return.index, strategy_cumulative_return, 
             label='Strategy', alpha=0.7)
    plt.title('Cumulative Returns - Strategy vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Return Distribution
    plt.subplot(3, 1, 3)
    plt.hist(results_df['actual_return'], bins=20, alpha=0.5, label='Actual Return')
    plt.hist(results_df['predicted_return'], bins=20, alpha=0.5, label='Predicted Return')
    plt.title('Return Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_backtest_results.png')
    
    # Print results if verbose
    if verbose:
        print("\nBacktest Results:")
        print(f"Number of predictions: {len(results_df)}")
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"Root Mean Squared Error: {rmse:.6f}")
        print(f"Mean Absolute Error: {mae:.6f}")
        print(f"Directional Accuracy: {directional_accuracy:.2%}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Strategy Cumulative Return: {strategy_cumulative_return.iloc[-1]:.2%}")
        print(f"Buy & Hold Cumulative Return: {buy_hold_cumulative_return.iloc[-1]:.2%}")
    
    # Return results
    results = {
        'results_df': results_df,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'win_rate': win_rate,
            'strategy_return': strategy_cumulative_return.iloc[-1],
            'buy_hold_return': buy_hold_cumulative_return.iloc[-1]
        },
        'all_probabilities': all_probabilities
    }
    
    return results

def evaluate_multiple_tickers(tickers, start_date=None, end_date=None, 
                             initial_train_days=252, walk_forward_days=21,
                             bucket_edges=None):
    """
    Run backtest on multiple tickers and compare results.
    
    Args:
        tickers (list): List of stock ticker symbols
        start_date (str): Start date for data in format 'YYYY-MM-DD'
        end_date (str): End date for data in format 'YYYY-MM-DD'
        initial_train_days (int): Number of days for initial training period
        walk_forward_days (int): Number of days to predict before retraining
        bucket_edges (list): Custom bucket edges
        
    Returns:
        dict: Results for each ticker
    """
    all_results = {}
    
    for ticker in tickers:
        print(f"\n{'='*40}")
        print(f"Backtesting {ticker}")
        print(f"{'='*40}")
        
        results = run_backtest(
            ticker, 
            start_date=start_date, 
            end_date=end_date,
            initial_train_days=initial_train_days,
            walk_forward_days=walk_forward_days,
            bucket_edges=bucket_edges
        )
        
        all_results[ticker] = results
    
    # Compare results across tickers
    comparison = {}
    for metric in ['mse', 'rmse', 'mae', 'directional_accuracy', 'win_rate', 'strategy_return', 'buy_hold_return']:
        comparison[metric] = {ticker: all_results[ticker]['metrics'][metric] for ticker in tickers}
    
    comparison_df = pd.DataFrame(comparison)
    
    print("\nComparison across tickers:")
    print(comparison_df)
    
    # Plot comparison of strategy returns
    plt.figure(figsize=(12, 6))
    
    # Strategy Returns
    plt.subplot(1, 2, 1)
    plt.bar(comparison_df.index, comparison_df['strategy_return'], color='skyblue')
    plt.title('Strategy Returns by Ticker')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Directional Accuracy
    plt.subplot(1, 2, 2)
    plt.bar(comparison_df.index, comparison_df['directional_accuracy'], color='lightgreen')
    plt.title('Directional Accuracy by Ticker')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tickers_comparison.png')
    
    return all_results, comparison_df

def main():
    parser = argparse.ArgumentParser(description='Backtest the probabilistic bucket prediction model')
    parser.add_argument('--ticker', type=str, default='AAPL', 
                        help='Stock ticker symbol (or comma-separated list)')
    parser.add_argument('--start', type=str, default='2020-01-01', 
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, default=None, 
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--initial_train_days', type=int, default=252, 
                        help='Number of days for initial training period')
    parser.add_argument('--walk_forward', type=int, default=21, 
                        help='Days to predict before retraining')
    
    args = parser.parse_args()
    
    # Check if multiple tickers
    if ',' in args.ticker:
        tickers = [t.strip() for t in args.ticker.split(',')]
        
        # Run backtest on multiple tickers
        all_results, comparison = evaluate_multiple_tickers(
            tickers=tickers,
            start_date=args.start,
            end_date=args.end,
            initial_train_days=args.initial_train_days,
            walk_forward_days=args.walk_forward
        )
    else:
        # Run backtest on single ticker
        results = run_backtest(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            initial_train_days=args.initial_train_days,
            walk_forward_days=args.walk_forward
        )

if __name__ == "__main__":
    main()