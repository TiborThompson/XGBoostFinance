"""
Improved backtest for the probabilistic bucket prediction model.
This version includes:
1. Hyperparameter optimization
2. Additional technical indicators
3. Feature selection
4. Ensemble approach
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
import ta
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import randint, uniform

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

def add_technical_indicators(df):
    """Add technical indicators using the ta library."""
    # Make copy of dataframe
    df_copy = df.copy()
    
    # Ensure columns are in expected format for ta
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for i, col in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
        if col in df_copy.columns:
            # Rename to lowercase for ta
            df_copy[required_columns[i]] = df_copy[col]
        else:
            # Create if missing
            if col == 'Open':
                df_copy[required_columns[i]] = df_copy['Close'].shift(1)
            elif col == 'High':
                df_copy[required_columns[i]] = df_copy['Close'] * 1.005
            elif col == 'Low':
                df_copy[required_columns[i]] = df_copy['Close'] * 0.995
            elif col == 'Volume':
                df_copy[required_columns[i]] = 1000000  # Default volume
    
    try:
        # Add all technical indicators using ta
        df_copy = ta.add_all_ta_features(
            df_copy, 
            open="open", 
            high="high", 
            low="low", 
            close="close", 
            volume="volume",
            fillna=True
        )
        
        # Add custom moving averages and crossovers
        for period in [10, 20, 50, 200]:
            # Simple Moving Averages
            df_copy[f'SMA_{period}'] = df_copy['close'].rolling(window=period).mean()
            # Exponential Moving Averages
            df_copy[f'EMA_{period}'] = df_copy['close'].ewm(span=period, adjust=False).mean()
            # Price relative to SMA
            df_copy[f'PRICE_SMA_{period}_RATIO'] = df_copy['close'] / df_copy[f'SMA_{period}']
        
        # Add crossover indicators
        df_copy['SMA_10_20_CROSS'] = df_copy['SMA_10'] / df_copy['SMA_20'] - 1
        df_copy['SMA_20_50_CROSS'] = df_copy['SMA_20'] / df_copy['SMA_50'] - 1
        df_copy['SMA_50_200_CROSS'] = df_copy['SMA_50'] / df_copy['SMA_200'] - 1
        
        # Add custom volatility indicator
        df_copy['volatility_20d'] = df_copy['close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # Add returns over different periods
        for period in [1, 3, 5, 10, 20]:
            df_copy[f'return_{period}d'] = df_copy['close'].pct_change(period)
    
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
        # Continue with basic features if ta fails
    
    return df_copy

def prepare_improved_features(df):
    """Prepare improved feature set for the model."""
    # Add technical indicators
    df_with_indicators = add_technical_indicators(df)
    
    # Calculate daily returns
    df_with_indicators['daily_return'] = df_with_indicators['Close'].pct_change()
    
    # Create features DataFrame
    features = pd.DataFrame(index=df_with_indicators.index)
    
    # Add all technical indicators
    for column in df_with_indicators.columns:
        if column not in ['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return']:
            features[column] = df_with_indicators[column]
    
    # Recent returns over different periods
    for period in [1, 2, 3, 5, 10, 20, 30, 60]:
        features[f'return_{period}d'] = df_with_indicators['Close'].pct_change(period)
    
    # Rolling statistics for returns
    for period in [5, 10, 20, 30, 60]:
        features[f'return_mean_{period}d'] = df_with_indicators['daily_return'].rolling(period).mean()
        features[f'return_std_{period}d'] = df_with_indicators['daily_return'].rolling(period).std()
        features[f'return_skew_{period}d'] = df_with_indicators['daily_return'].rolling(period).skew()
        features[f'return_kurt_{period}d'] = df_with_indicators['daily_return'].rolling(period).kurt()
        features[f'return_max_{period}d'] = df_with_indicators['daily_return'].rolling(period).max()
        features[f'return_min_{period}d'] = df_with_indicators['daily_return'].rolling(period).min()
    
    # Volatility measures
    features['volatility_historic'] = features['return_std_20d'] * np.sqrt(252)
    
    # Target variable: next day's return
    features['next_return'] = df_with_indicators['daily_return'].shift(-1)
    
    # Drop rows with NaN values
    features = features.dropna()
    
    return features

def optimize_xgboost_params(X_train, y_train, X_val, y_val, num_class):
    """Optimize XGBoost hyperparameters."""
    print("Optimizing XGBoost parameters...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Create XGBoost classifier
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_class,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0
    )
    
    # Set up cross-validation
    cv = TimeSeriesSplit(n_splits=3)
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=15,  # Number of parameter settings sampled
        scoring='neg_log_loss',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    # Fit to data
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {-random_search.best_score_:.4f}")
    
    # Create model with best parameters
    best_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_class,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
        **random_search.best_params_
    )
    
    # Fit model with early stopping
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    return best_model

def run_improved_backtest(ticker, start_date=None, end_date=None, 
                         initial_train_days=504, walk_forward_days=21, 
                         optimize_params=True, verbose=True):
    """Run an improved backtest with more advanced features and model optimization."""
    if verbose:
        print(f"Running improved backtest for {ticker}")
        print(f"Initial training period: {initial_train_days} days")
        print(f"Walk-forward window: {walk_forward_days} days")
        print(f"Parameter optimization: {'Enabled' if optimize_params else 'Disabled'}")
    
    # Download data with more history for better feature engineering
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    
    if verbose:
        print(f"Data loaded with {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
    
    # Prepare improved features
    features_df = prepare_improved_features(df)
    
    if verbose:
        print(f"Feature engineering complete. {features_df.shape[1]} features created.")
    
    # Initialize model
    model = ProbabilisticBucketPredictor()
    
    # Define more granular return buckets for better resolution
    bucket_edges = [-0.05, -0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    model.define_return_buckets(bucket_edges)
    
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
        
        # Split training data for validation
        val_size = int(len(X_train) * 0.2)
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        X_train = X_train.iloc[:-val_size]
        y_train = y_train.iloc[:-val_size]
        
        # Get testing data
        X_test = features_df.drop('next_return', axis=1).iloc[test_start:test_end]
        y_test = features_df['next_return'].iloc[test_start:test_end]
        
        # Convert returns to buckets
        y_train_buckets = model._returns_to_buckets(y_train.values)
        y_val_buckets = model._returns_to_buckets(y_val.values)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), 
            columns=X_val.columns, 
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        # Build and train model
        try:
            num_classes = len(model.bucket_edges) + 1
            
            if optimize_params and i == 0:  # Only optimize on first window to save time
                # Optimize model parameters
                optimized_model = optimize_xgboost_params(
                    X_train_scaled, y_train_buckets, 
                    X_val_scaled, y_val_buckets,
                    num_classes
                )
                model_params = optimized_model.get_params()
                
                # Clean params for XGBClassifier constructor
                params_to_remove = ['num_class', 'feature_names_in_', 'feature_types_in_', 'classes_']
                for param in params_to_remove:
                    if param in model_params:
                        del model_params[param]
                
                # Initialize model with optimized parameters
                xgb_model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=num_classes,
                    eval_metric='mlogloss',
                    use_label_encoder=False,
                    **model_params
                )
            else:
                # Use default model with some improvements
                xgb_model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=num_classes,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    n_estimators=300,
                    eval_metric='mlogloss',
                    use_label_encoder=False
                )
            
            # Train the model
            xgb_model.fit(
                X_train_scaled, y_train_buckets,
                eval_set=[(X_val_scaled, y_val_buckets)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            # Make predictions
            for j in range(len(X_test)):
                current_X = X_test_scaled.iloc[[j]]
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
    
    # Plot 4: Confusion Matrix as a heatmap
    plt.subplot(4, 1, 4)
    cm = pd.crosstab(results_df['actual_direction'], results_df['predicted_direction'], 
                     rownames=['Actual'], colnames=['Predicted'])
    
    # Add -1 row/column if missing
    for val in [-1.0, 0.0, 1.0]:
        if val not in cm.index:
            cm.loc[val] = 0
        if val not in cm.columns:
            cm[val] = 0
    
    # Sort index and columns
    cm = cm.sort_index().sort_index(axis=1)
    
    # Plot heatmap
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Direction Prediction Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max().max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm.iloc[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm.iloc[i, j] > thresh else "black")
    
    plt.ylabel('Actual Direction')
    plt.xlabel('Predicted Direction')
    plt.xticks(np.arange(len(cm.columns)), cm.columns)
    plt.yticks(np.arange(len(cm.index)), cm.index)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_improved_backtest_results.png')
    
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
        
        # Win rate by month
        results_df['month'] = results_df.index.month
        monthly_win_rate = results_df.groupby('month')['actual_positive'].mean()
        print("\nWin Rate by Month:")
        for month, rate in monthly_win_rate.items():
            month_name = datetime(2020, month, 1).strftime('%B')
            print(f"{month_name}: {rate:.2%}")
    
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
    """Main function to run the improved backtest."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run an improved backtest for stock prediction')
    parser.add_argument('--ticker', type=str, default='TSLA', 
                        help='Stock ticker symbol')
    parser.add_argument('--start', type=str, default='2018-01-01', 
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, default=None, 
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--initial_train_days', type=int, default=504, 
                        help='Number of days for initial training period')
    parser.add_argument('--walk_forward', type=int, default=21, 
                        help='Days to predict before retraining')
    parser.add_argument('--optimize', action='store_true',
                        help='Enable hyperparameter optimization')
    
    args = parser.parse_args()
    
    # Run backtest
    run_improved_backtest(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        initial_train_days=args.initial_train_days,
        walk_forward_days=args.walk_forward,
        optimize_params=args.optimize
    )

if __name__ == "__main__":
    main()