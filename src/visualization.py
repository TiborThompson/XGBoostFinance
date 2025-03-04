import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_probabilities(distribution, title="Return Probability Distribution", save_path=None):
    """
    Plot probability distribution across return buckets
    
    Args:
        distribution (dict): Dictionary mapping return ranges to probabilities
        title (str): Plot title
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    plt.figure(figsize=(12, 6))
    
    # Sort by return ranges
    labels = list(distribution.keys())
    values = list(distribution.values())
    
    # Create bar plot
    bars = plt.bar(labels, values, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom', rotation=0)
    
    plt.title(title)
    plt.ylabel('Probability')
    plt.xlabel('Return Range')
    plt.ylim(0, max(values) * 1.2)  # Add some space for labels
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return plt.gcf()

def plot_returns_distribution(returns, bucket_edges, title="Historical Returns Distribution", save_path=None):
    """
    Plot histogram of historical returns with bucket edges
    
    Args:
        returns (np.array): Array of return values
        bucket_edges (list): List of bucket edge values
        title (str): Plot title
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    plt.figure(figsize=(12, 6))
    
    # Create histogram
    n, bins, patches = plt.hist(returns, bins=30, alpha=0.7, color='skyblue')
    
    # Add vertical lines for bucket edges
    for edge in bucket_edges:
        plt.axvline(x=edge, color='r', linestyle='--', alpha=0.7)
        plt.text(edge, max(n)*0.9, f'{edge:.1%}', rotation=90, 
                 verticalalignment='top', color='r')
    
    plt.title(title)
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return plt.gcf()

def plot_feature_importance(feature_importances, feature_names, top_n=20, title="Feature Importance", save_path=None):
    """
    Plot feature importance from the model
    
    Args:
        feature_importances (np.array): Array of feature importance values
        feature_names (list): List of feature names
        top_n (int): Number of top features to plot
        title (str): Plot title
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create DataFrame for sorting
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # Select top N features
    feat_imp = feat_imp.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp['feature'][::-1], feat_imp['importance'][::-1], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return plt.gcf()

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", cmap=plt.cm.Blues, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm (np.array): Confusion matrix
        class_names (list): List of class names
        title (str): Plot title
        cmap: Colormap
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    plt.figure(figsize=(10, 8))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return plt.gcf()

def plot_calibration_curves(probs, y_true, class_names, n_bins=10, title="Calibration Curves", save_path=None):
    """
    Plot calibration curves for each class
    
    Args:
        probs (np.array): Predicted probabilities (shape: n_samples, n_classes)
        y_true (np.array): True labels (shape: n_samples)
        class_names (list): List of class names
        n_bins (int): Number of bins for calibration curve
        title (str): Plot title
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    plt.figure(figsize=(12, 8))
    
    # Number of classes
    n_classes = probs.shape[1]
    
    # Get colormap
    colors = plt.cm.jet(np.linspace(0, 1, n_classes))
    
    # Plot each class
    for i in range(n_classes):
        # Create binary targets for this class
        y_true_binary = (y_true == i).astype(int)
        
        # Get predicted probabilities for this class
        y_proba = probs[:, i]
        
        # Sort predictions and true values by prediction probability
        sorted_indices = np.argsort(y_proba)
        sorted_proba = y_proba[sorted_indices]
        sorted_true = y_true_binary[sorted_indices]
        
        # Create bins
        bin_size = len(sorted_proba) // n_bins
        bins = []
        
        for j in range(n_bins):
            start_idx = j * bin_size
            end_idx = (j + 1) * bin_size if j < n_bins - 1 else len(sorted_proba)
            
            # Calculate average predicted probability in this bin
            avg_pred_prob = np.mean(sorted_proba[start_idx:end_idx])
            
            # Calculate fraction of positives in this bin
            fraction_pos = np.mean(sorted_true[start_idx:end_idx])
            
            bins.append((avg_pred_prob, fraction_pos))
        
        # Extract x and y values for plotting
        x_vals = [b[0] for b in bins]
        y_vals = [b[1] for b in bins]
        
        # Plot calibration curve for this class
        plt.plot(x_vals, y_vals, 'o-', color=colors[i], label=class_names[i])
    
    # Plot diagonal perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability (Fraction of Positives)')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        
    return plt.gcf()

def plot_stock_with_predictions(df, predictions_df, ticker="Stock", window=30, save_path=None):
    """
    Plot stock prices with prediction probabilities
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
        predictions_df (pd.DataFrame): DataFrame with predictions and probabilities
        ticker (str): Stock ticker
        window (int): Number of days to plot
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    plt.figure(figsize=(14, 10))
    
    # Plot the stock price
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index[-window:], df['Close'][-window:], label='Close Price')
    ax1.set_title(f'{ticker} Stock Price and Return Predictions')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Plot the prediction probabilities
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    
    # Select probability columns
    prob_columns = [col for col in predictions_df.columns if col.startswith('prob_')]
    
    # Create stacked area plot
    ax2.stackplot(predictions_df.index[-window:], 
                  [predictions_df[col][-window:] for col in prob_columns],
                  labels=[col.replace('prob_', '') for col in prob_columns],
                  alpha=0.7)
    
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return plt.gcf()

class StockVisualizer:
    """Legacy class for backward compatibility"""
    def __init__(self):
        pass
        
    def plot_stock_history(self, df, ticker):
        """
        Plot historical stock data
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            ticker (str): Stock ticker symbol
        """
        plt.figure(figsize=(14, 7))
        plt.plot(df['Close'])
        plt.title(f'{ticker} Stock Price History')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('stock_plot.png')
        print("Stock price history saved to stock_plot.png")
        
    def plot_training_history(self, history):
        """
        Plot training history
        
        Args:
            history: Training history from model.fit()
        """
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history saved to training_history.png")
        
    def plot_predictions(self, actual, predictions, ticker, days=None):
        """
        Plot actual vs predicted stock prices
        
        Args:
            actual (np.array): Actual stock prices
            predictions (np.array): Predicted stock prices
            ticker (str): Stock ticker symbol
            days (int): Number of days to plot (None for all)
        """
        if days is not None:
            actual = actual[-days:]
            predictions = predictions[-days:]
            
        plt.figure(figsize=(14, 7))
        plt.plot(actual, label='Actual Price')
        plt.plot(predictions, label='Predicted Price', linestyle='--')
        
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Days')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('prediction_results.png')
        print("Prediction results saved to prediction_results.png")
        
    def plot_future_prediction(self, historical_data, future_prediction, ticker):
        """
        Plot historical data with future prediction
        
        Args:
            historical_data (np.array): Historical stock prices
            future_prediction (float): Predicted future price
            ticker (str): Stock ticker symbol
        """
        plt.figure(figsize=(14, 7))
        
        # Plot historical data
        plt.plot(range(len(historical_data)), historical_data, label='Historical Data')
        
        # Plot prediction
        plt.plot(len(historical_data), future_prediction, 'ro', markersize=8, label='Next Day Prediction')
        
        # Draw a dotted line to the prediction
        plt.plot([len(historical_data)-1, len(historical_data)], 
                 [historical_data[-1], future_prediction], 
                 'r--', alpha=0.7)
        
        # Add the prediction value as text
        plt.text(len(historical_data), future_prediction, 
                 f'${future_prediction:.2f}', 
                 fontsize=12, ha='left')
        
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Days')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('future_prediction.png')
        print("Future prediction saved to future_prediction.png")