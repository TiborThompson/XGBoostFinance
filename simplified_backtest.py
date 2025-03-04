"""
Simplified backtest for the probabilistic bucket prediction model.
This version includes:
1. More data (longer training window)
2. Basic technical indicators
3. Feature selection
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm import tqdm
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import ProbabilisticBucketPredictor

def calculate_expected_returns(probabilities, bucket_edges):
    """Calculate expected return from probability distribution across buckets."""
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

def prepare_features(df):
    """Prepare feature set for the model."""
    # Create a copy of the DataFrame
    df_copy = df.copy()
    
    # Create features DataFrame
    features = pd.DataFrame(index=df_copy.index)
    
    # Calculate daily returns
    features['daily_return'] = df_copy['Close'].pct_change()
    
    # Recent returns over different periods
    for period in [1, 2, 3, 5, 10, 20, 30, 60]:
        features[f'return_{period}d'] = df_copy['Close'].pct_change(period)
    
    # Rolling statistics for returns
    for period in [5, 10, 20, 30, 60]:
        features[f'return_mean_{period}d'] = features['daily_return'].rolling(period).mean()
        features[f'return_std_{period}d'] = features['daily_return'].rolling(period).std()
        features[f'return_max_{period}d'] = features['daily_return'].rolling(period).max()
        features[f'return_min_{period}d'] = features['daily_return'].rolling(period).min()
    
    # Simple moving averages
    for period in [10, 20, 50, 200]:
        features[f'sma_{period}'] = df_copy['Close'].rolling(period).mean()
        # Calculate price to SMA ratio directly in pandas
        features[f'price_to_sma_{period}'] = df_copy['Close'] / features[f'sma_{period}'].replace(0, np.nan)
    
    # Volatility measures
    features['volatility_historic'] = features['return_std_20d'] * np.sqrt(252)
    
    # Volume features if volume data is available
    if 'Volume' in df_copy.columns:
        features['volume_change'] = df_copy['Volume'].pct_change()
        features['volume_change_mean_10d'] = features['volume_change'].rolling(10).mean()
        features['volume_to_avg_volume_20d'] = df_copy['Volume'] / df_copy['Volume'].rolling(20).mean()
    
    # Target variable: next day's return
    features['next_return'] = features['daily_return'].shift(-1)
    
    # Drop rows with NaN values
    features = features.dropna()
    
    return features

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
    
    # Download data
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    
    if verbose:
        print(f"Data loaded with {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
    
    # Prepare features
    features_df = prepare_features(df)
    
    if verbose:
        print(f"Feature engineering complete. {features_df.shape[1]} features created.")
    
    # Initialize model
    model = ProbabilisticBucketPredictor()
    
    # Define return buckets
    if bucket_edges is not None:
        model.define_return_buckets(bucket_edges)
    else:
        model.define_return_buckets([-0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03])
    
    # Set up backtest tracking variables
    predictions = []
    actuals = []
    dates = []
    all_probabilities = []
    
    # Calculate total number of testing windows
    total_days = len(features_df) - initial_train_days
    total_windows = max(1, (total_days + walk_forward_days - 1) // walk_forward_days)
    
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
        try:
            num_classes = len(model.bucket_edges) + 1
            
            # Use default model with some improvements
            xgb_model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=num_classes,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.2,
                reg_lambda=1.0,
                n_estimators=200,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            
            # Train the model
            xgb_model.fit(
                X_train, y_train_buckets,
                eval_set=[(X_train, y_train_buckets)],
                verbose=False
            )
            
            # Make predictions
            for j in range(len(X_test)):
                current_X = X_test.iloc[[j]]
                current_date = X_test.index[j]
                current_actual = y_test.iloc[j]
                
                # Predict probabilities
                probs = xgb_model.predict_proba(current_X)[0]
                
                # Calculate expected return
                expected_return = calculate_expected_returns(probs, model.bucket_edges)
                
                # Store results
                predictions.append(expected_return)
                actuals.append(current_actual)
                dates.append(current_date)
                all_probabilities.append(probs)
        
        except Exception as e:
            print(f"Error in window {i}: {e}")
            continue
            
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
    
    # Strategy that invests only when prediction is positive
    results_df['strategy_return'] = np.where(
        results_df['predicted_return'] > 0.002,  # Only take trades with expected return > 0.2%
        results_df['actual_return'],
        0
    )
    
    # Strategy that follows the direction (long/short)
    results_df['direction_strategy_return'] = results_df['actual_return'] * np.sign(results_df['predicted_return'])
    
    # Calculate cumulative returns
    results_df['strategy_cumulative'] = (1 + results_df['strategy_return']).cumprod() - 1
    results_df['direction_strategy_cumulative'] = (1 + results_df['direction_strategy_return']).cumprod() - 1
    results_df['buy_hold_cumulative'] = (1 + results_df['actual_return']).cumprod() - 1
    
    # Calculate Sharpe ratio (assuming 252 trading days per year)
    strategy_daily_returns = results_df['strategy_return']
    strategy_sharpe = np.sqrt(252) * np.mean(strategy_daily_returns) / np.std(strategy_daily_returns) if np.std(strategy_daily_returns) > 0 else 0
    
    direction_daily_returns = results_df['direction_strategy_return']
    direction_sharpe = np.sqrt(252) * np.mean(direction_daily_returns) / np.std(direction_daily_returns) if np.std(direction_daily_returns) > 0 else 0
    
    buyhold_daily_returns = results_df['actual_return']
    buyhold_sharpe = np.sqrt(252) * np.mean(buyhold_daily_returns) / np.std(buyhold_daily_returns) if np.std(buyhold_daily_returns) > 0 else 0
    
    # Plot results
    plt.figure(figsize=(16, 20))
    
    # Plot 1: Actual vs Predicted Returns
    plt.subplot(4, 1, 1)
    plt.plot(results_df.index, results_df['actual_return'], label='Actual Return', alpha=0.7)
    plt.plot(results_df.index, results_df['predicted_return'], label='Predicted Return', alpha=0.7)
    plt.title(f'{ticker} - Actual vs Predicted Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Returns
    plt.subplot(4, 1, 2)
    plt.plot(results_df.index, results_df['buy_hold_cumulative'], 
             label=f'Buy & Hold (Sharpe: {buyhold_sharpe:.2f})', alpha=0.7)
    plt.plot(results_df.index, results_df['strategy_cumulative'], 
             label=f'Threshold Strategy (Sharpe: {strategy_sharpe:.2f})', alpha=0.7)
    plt.plot(results_df.index, results_df['direction_strategy_cumulative'], 
             label=f'Direction Strategy (Sharpe: {direction_sharpe:.2f})', alpha=0.7)
    plt.title('Cumulative Returns - Strategies vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Return Distribution
    plt.subplot(4, 1, 3)
    plt.hist(results_df['actual_return'], bins=50, alpha=0.5, label='Actual Return')
    plt.hist(results_df['predicted_return'], bins=50, alpha=0.5, label='Predicted Return')
    plt.title('Return Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Confusion Matrix
    confusion_matrix = pd.crosstab(
        results_df['actual_direction'], 
        results_df['predicted_direction'],
        rownames=['Actual'], 
        colnames=['Predicted']
    )
    
    # Handle missing values in confusion matrix
    for direction in [-1.0, 0.0, 1.0]:
        if direction not in confusion_matrix.index:
            confusion_matrix.loc[direction] = 0
        if direction not in confusion_matrix.columns:
            confusion_matrix[direction] = 0
    
    confusion_matrix = confusion_matrix.sort_index().sort_index(axis=1)
    
    plt.subplot(4, 1, 4)
    im = plt.matshow(confusion_matrix, fignum=False, cmap='Blues')
    plt.colorbar(im)
    plt.title('Direction Prediction Confusion Matrix')
    
    # Add text annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix.iloc[i, j], 'd'),
                    horizontalalignment="center", color="white" if confusion_matrix.iloc[i, j] > confusion_matrix.values.max() / 2 else "black")
    
    plt.ylabel('Actual Direction')
    plt.xlabel('Predicted Direction')
    plt.xticks(range(len(confusion_matrix.columns)), confusion_matrix.columns)
    plt.yticks(range(len(confusion_matrix.index)), confusion_matrix.index)
    
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
        print(f"Threshold Strategy Cumulative Return: {results_df['strategy_cumulative'].iloc[-1]:.2%}")
        print(f"Direction Strategy Cumulative Return: {results_df['direction_strategy_cumulative'].iloc[-1]:.2%}")
        print(f"Buy & Hold Cumulative Return: {results_df['buy_hold_cumulative'].iloc[-1]:.2%}")
        print(f"Threshold Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
        print(f"Direction Strategy Sharpe Ratio: {direction_sharpe:.2f}")
        print(f"Buy & Hold Sharpe Ratio: {buyhold_sharpe:.2f}")
    
    # Return results
    results = {
        'results_df': results_df,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'win_rate': win_rate,
            'strategy_return': results_df['strategy_cumulative'].iloc[-1],
            'direction_strategy_return': results_df['direction_strategy_cumulative'].iloc[-1],
            'buy_hold_return': results_df['buy_hold_cumulative'].iloc[-1],
            'strategy_sharpe': strategy_sharpe,
            'direction_sharpe': direction_sharpe,
            'buyhold_sharpe': buyhold_sharpe
        },
        'all_probabilities': all_probabilities
    }
    
    return results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest the probabilistic bucket prediction model')
    parser.add_argument('--ticker', type=str, default='TSLA', 
                        help='Stock ticker symbol')
    parser.add_argument('--start', type=str, default='2020-01-01', 
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, default=None, 
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--initial_train_days', type=int, default=500, 
                        help='Number of days for initial training period')
    parser.add_argument('--walk_forward', type=int, default=30, 
                        help='Days to predict before retraining')
    
    args = parser.parse_args()
    
    # Run backtest
    run_backtest(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        initial_train_days=args.initial_train_days,
        walk_forward_days=args.walk_forward
    )

if __name__ == "__main__":
    main()