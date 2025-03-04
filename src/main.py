import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, brier_score_loss

from src.data_loader import StockDataLoader
from src.model import ProbabilisticBucketPredictor
from src.visualization import plot_probabilities, plot_returns_distribution, plot_feature_importance

def run_model(ticker='AAPL', use_synthetic=False, bucket_edges=None):
    """
    Run the probabilistic bucket prediction model pipeline
    
    Args:
        ticker (str): Stock ticker to analyze
        use_synthetic (bool): Whether to use synthetic data
        bucket_edges (list): Custom bucket edges, if None uses default
        
    Returns:
        tuple: (model, results)
    """
    print(f"{'='*40}")
    print(f"Running Probabilistic Bucket Prediction for {ticker}")
    print(f"{'='*40}")
    
    # Initialize data loader
    data_loader = StockDataLoader()
    
    # Load data or generate synthetic data
    if use_synthetic:
        print("Using synthetic data...")
        df = data_loader.generate_synthetic_data(periods=1000)
    else:
        print(f"Loading real data for {ticker}...")
        df = data_loader.load_stock_data(ticker, interval='1d')
    
    print(f"Data shape: {df.shape}")
    try:
        print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    except:
        print("Date range: Could not determine date range")
    
    # Prepare data for bucket prediction
    X, y, X_train, X_test, y_train, y_test, feature_names = data_loader.prepare_data_for_buckets(df)
    print(f"Features: {len(feature_names)}")
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    
    # Initialize model
    model = ProbabilisticBucketPredictor()
    
    # Define return buckets
    if bucket_edges is not None:
        model.define_return_buckets(bucket_edges)
    else:
        model.define_return_buckets()
    
    # Convert continuous returns to bucket indices
    y_train_buckets = model._returns_to_buckets(y_train.values)
    y_test_buckets = model._returns_to_buckets(y_test.values)
    
    # Print bucket distribution
    bucket_counts = np.bincount(y_train_buckets)
    total_samples = len(y_train_buckets)
    
    print("\nReturn bucket distribution (training set):")
    for i, count in enumerate(bucket_counts):
        bucket_range = model.bucket_to_return_range(i)
        print(f"Bucket {i} ({bucket_range}): {count} samples ({count/total_samples:.1%})")
    
    # Train model
    model.build_model(num_classes=len(model.bucket_edges) + 1)
    model.train(X_train, y_train_buckets, ticker=ticker)
    
    # Cross-validate
    print("\nPerforming time-series cross-validation...")
    cv_results = model.cross_validate(X, model._returns_to_buckets(y.values))
    print(f"Log Loss: {cv_results['log_loss_mean']:.4f} ± {cv_results['log_loss_std']:.4f}")
    print(f"Brier Score: {cv_results['brier_score_mean']:.4f} ± {cv_results['brier_score_std']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = model.evaluate(X_test, y_test_buckets)
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    
    # Plot feature importance
    model.plot_feature_importance(feature_names)
    print(f"Feature importance plot saved to {os.path.join(model.model_dir, 'feature_importance.png')}")
    
    # Plot calibration curves
    model.plot_calibration_curve(X_test, y_test_buckets)
    print(f"Calibration curves saved to {os.path.join(model.model_dir, 'calibration_curve.png')}")
    
    # Predict on the most recent data
    most_recent_X = X.iloc[-1:].copy()
    prediction_result = model.predict_with_explanation(most_recent_X)
    
    print("\nMost recent prediction:")
    print(f"Date: {X.index[-1]}")
    print(f"Most likely outcome: {prediction_result['return_range']} with {prediction_result['probability']:.2%} probability")
    print(f"Expected return (point forecast): {prediction_result['expected_return_pct']}")
    
    print("\nFull probability distribution:")
    for bucket_range, prob in prediction_result['sorted_distribution'].items():
        print(f"{bucket_range}: {prob:.2%}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.bar(
        list(prediction_result['sorted_distribution'].keys()),
        list(prediction_result['sorted_distribution'].values()),
        color='skyblue'
    )
    plt.title(f"{ticker} Return Probability Distribution")
    plt.ylabel('Probability')
    plt.xlabel('Return Range')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(model.model_dir, 'probability_distribution.png'))
    print(f"Probability distribution plot saved to {os.path.join(model.model_dir, 'probability_distribution.png')}")
    
    # Return model and results
    results = {
        'metrics': metrics,
        'cv_results': cv_results,
        'prediction': prediction_result,
        'feature_names': feature_names
    }
    
    return model, results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Probabilistic Bucket Prediction for Stock Returns')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data instead of real data')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'], 
                        help='Mode: train or predict')
    args = parser.parse_args()
    
    if args.mode == 'train':
        model, results = run_model(ticker=args.ticker, use_synthetic=args.synthetic)
    else:
        # Load model and make prediction only
        model = ProbabilisticBucketPredictor()
        
        # Define return buckets (same as during training)
        model.define_return_buckets()
        
        # Load trained model
        model.load_model(args.ticker)
        
        if model.model is None:
            print(f"No trained model found for {args.ticker}. Please train a model first.")
            return
            
        # Initialize data loader and get most recent data
        data_loader = StockDataLoader()
        if args.synthetic:
            df = data_loader.generate_synthetic_data(periods=1000)
        else:
            df = data_loader.load_stock_data(args.ticker, interval='1d')
        
        # Prepare features for most recent day
        X, y, _, _, _, _, feature_names = data_loader.prepare_data_for_buckets(df)
        most_recent_X = X.iloc[-1:].copy()
        
        # Make prediction
        prediction_result = model.predict_with_explanation(most_recent_X)
        
        print("\nPrediction for next trading day:")
        print(f"Date: {X.index[-1]}")
        print(f"Most likely outcome: {prediction_result['return_range']} with {prediction_result['probability']:.2%} probability")
        print(f"Expected return (point forecast): {prediction_result['expected_return_pct']}")
        
        print("\nFull probability distribution:")
        for bucket_range, prob in prediction_result['sorted_distribution'].items():
            print(f"{bucket_range}: {prob:.2%}")

if __name__ == "__main__":
    main()