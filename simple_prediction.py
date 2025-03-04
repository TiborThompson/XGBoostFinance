"""
Simple example of the probabilistic bucket prediction approach.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

print("Creating synthetic stock data...")
# Create synthetic stock data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')

# Generate random walk with drift
returns = np.random.normal(0.0001, 0.015, 1000)
price = 100 * np.exp(np.cumsum(returns))

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Close': price
})
df.set_index('Date', inplace=True)

# Calculate returns
df['daily_return'] = df['Close'].pct_change()
df = df.dropna()

# Create features from returns
def engineer_features(df, lookback_periods=[5, 10, 20, 30, 60]):
    """Engineer features for return prediction"""
    features = pd.DataFrame(index=df.index)
    features['daily_return'] = df['daily_return']
    
    # Recent returns over different periods
    for period in lookback_periods:
        features[f'return_{period}d'] = df['Close'].pct_change(period)
        features[f'return_mean_{period}d'] = features['daily_return'].rolling(period).mean()
        features[f'return_std_{period}d'] = features['daily_return'].rolling(period).std()
    
    # Momentum indicators
    for short_period in [5, 10, 20]:
        for long_period in [30, 60]:
            if short_period < long_period:
                features[f'momentum_{short_period}_{long_period}'] = (
                    features[f'return_{short_period}d'] - features[f'return_{long_period}d']
                )
    
    # Volatility measures
    for period in lookback_periods:
        features[f'volatility_{period}d'] = features['daily_return'].rolling(period).std() * np.sqrt(252)
        
    # Target variable: next day's return
    features['next_return'] = features['daily_return'].shift(-1)
    
    # Drop NaN values
    features = features.dropna()
    
    return features

# Engineer features
print("Engineering features...")
features_df = engineer_features(df)

# Define return buckets
def define_return_buckets(returns, edges=None):
    """Convert continuous returns to bucket indices"""
    if edges is None:
        edges = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    
    return np.digitize(returns, edges), edges

# Create target variable: next day's return bucket
X = features_df.drop(['daily_return', 'next_return'], axis=1)
y = features_df['next_return']

# Convert returns to buckets
y_buckets, bucket_edges = define_return_buckets(y)

# Print bucket distribution
bucket_counts = np.bincount(y_buckets)
total_samples = len(y_buckets)

print("\nReturn bucket distribution:")
for i, count in enumerate(bucket_counts):
    if i == 0:
        bucket_range = f"< {bucket_edges[0]*100:.1f}%"
    elif i == len(bucket_edges):
        bucket_range = f"> {bucket_edges[-1]*100:.1f}%"
    else:
        bucket_range = f"{bucket_edges[i-1]*100:.1f}% to {bucket_edges[i]*100:.1f}%"
    
    print(f"Bucket {i} ({bucket_range}): {count} samples ({count/total_samples:.1%})")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_buckets, test_size=0.2, shuffle=False
)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Train an XGBoost model
print("\nTraining XGBoost model with probability outputs...")
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(bucket_edges) + 1,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=100,
    eval_metric='mlogloss'
)

# Train the model
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# Calibrate the probabilities
print("\nApplying probability calibration...")
from sklearn.base import clone
calibrated_model = CalibratedClassifierCV(
    estimator=clone(model),
    method='isotonic', 
    cv=5  # Use 5-fold cross-validation
)
calibrated_model.fit(X_train, y_train)

# Make predictions on test set
print("\nMaking predictions...")
y_pred = model.predict(X_test)
y_pred_proba = calibrated_model.predict_proba(X_test)

# Evaluate
accuracy = np.mean(y_pred == y_test)
ll = log_loss(y_test, y_pred_proba)

# Calculate Brier score for each class and average
bs_scores = []
for i in range(y_pred_proba.shape[1]):
    y_test_binary = (y_test == i).astype(int)
    bs = brier_score_loss(y_test_binary, y_pred_proba[:, i])
    bs_scores.append(bs)

print(f"Accuracy: {accuracy:.2%}")
print(f"Log Loss: {ll:.4f}")
print(f"Brier Score: {np.mean(bs_scores):.4f}")

# Plot feature importance
plt.figure(figsize=(12, 8))
xgb.plot_importance(model, max_num_features=20)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved to feature_importance.png")

# Make a prediction for the most recent data point
most_recent_X = X.iloc[-1:].copy()
prediction_probas = calibrated_model.predict_proba(most_recent_X)[0]
most_likely_bucket = np.argmax(prediction_probas)

# Convert bucket index to human-readable range
if most_likely_bucket == 0:
    return_range = f"< {bucket_edges[0]*100:.1f}%"
elif most_likely_bucket == len(bucket_edges):
    return_range = f"> {bucket_edges[-1]*100:.1f}%"
else:
    return_range = f"{bucket_edges[most_likely_bucket-1]*100:.1f}% to {bucket_edges[most_likely_bucket]*100:.1f}%"

print("\nMost recent prediction:")
print(f"Date: {X.index[-1]}")
print(f"Most likely outcome: {return_range} with {prediction_probas[most_likely_bucket]:.2%} probability")

print("\nFull probability distribution:")
for i in range(len(prediction_probas)):
    if i == 0:
        bucket_range = f"< {bucket_edges[0]*100:.1f}%"
    elif i == len(bucket_edges):
        bucket_range = f"> {bucket_edges[-1]*100:.1f}%"
    else:
        bucket_range = f"{bucket_edges[i-1]*100:.1f}% to {bucket_edges[i]*100:.1f}%"
    
    print(f"{bucket_range}: {prediction_probas[i]:.2%}")

# Plot probability distribution
plt.figure(figsize=(12, 6))
bucket_labels = []
for i in range(len(bucket_edges) + 1):
    if i == 0:
        bucket_labels.append(f"< {bucket_edges[0]*100:.1f}%")
    elif i == len(bucket_edges):
        bucket_labels.append(f"> {bucket_edges[-1]*100:.1f}%")
    else:
        bucket_labels.append(f"{bucket_edges[i-1]*100:.1f}% to {bucket_edges[i]*100:.1f}%")

bars = plt.bar(bucket_labels, prediction_probas, color='skyblue')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.1%}', ha='center', va='bottom', rotation=0)

plt.title('Return Probability Distribution')
plt.ylabel('Probability')
plt.xlabel('Return Range')
plt.ylim(0, max(prediction_probas) * 1.2)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('probability_distribution.png')
print("Probability distribution plot saved to probability_distribution.png")

print("\nProbabilistic bucket prediction model successfully trained!")