import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

class ProbabilisticBucketPredictor:
    def __init__(self, model_dir=None):
        # Use absolute path to ensure correct location
        if model_dir is None:
            # Get the project root directory (where src, data, and models are)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_dir = os.path.join(project_root, 'models')
        else:
            self.model_dir = model_dir
            
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"Models will be stored in: {self.model_dir}")
        
        self.model = None
        self.calibrated_model = None
        self.bucket_edges = None
        self.scaler = None
        
    def define_return_buckets(self, custom_edges=None):
        """
        Define the return buckets for classification.
        
        Args:
            custom_edges (list, optional): Custom bucket edge values. 
                                          Default is [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
                                          
        Returns:
            list: Bucket edge values
        """
        if custom_edges is None:
            # Default bucket edges: Large negative to large positive
            self.bucket_edges = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
        else:
            self.bucket_edges = custom_edges
            
        return self.bucket_edges
    
    def _returns_to_buckets(self, returns):
        """
        Convert continuous returns to bucket indices.
        
        Args:
            returns (np.array): Array of return values
            
        Returns:
            np.array: Array of bucket indices
        """
        bucket_indices = np.digitize(returns, self.bucket_edges)
        return bucket_indices
    
    def _engineer_features(self, df, lookback_periods=[5, 10, 20, 30, 60]):
        """
        Engineer features from raw time series data.
        
        Args:
            df (pd.DataFrame): DataFrame with at least 'Close' column
            lookback_periods (list): List of lookback periods for feature engineering
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
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
            
        # Momentum indicators (comparing short vs long term performance)
        for short_period in [5, 10, 20]:
            for long_period in [30, 60]:
                if short_period < long_period:
                    features[f'momentum_{short_period}_{long_period}'] = (
                        features[f'return_{short_period}d'] - features[f'return_{long_period}d']
                    )
        
        # Volatility measures
        for period in lookback_periods:
            features[f'volatility_{period}d'] = features['daily_return'].rolling(period).std() * np.sqrt(252)  # Annualized
        
        # Drop rows with NaN values resulting from rolling calculations
        features = features.dropna()
        
        # Target variable: next day's return
        features['next_return'] = features['daily_return'].shift(-1)
        
        # Drop the last row that will have NaN for next_return
        features = features.dropna()
        
        # Remove the daily_return column as it's just used for calculations
        features = features.drop('daily_return', axis=1)
        
        return features
    
    def prepare_data(self, df, test_size=0.2):
        """
        Prepare data for training including feature engineering and bucket assignment.
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, feature_names)
        """
        # Define buckets if not already defined
        if self.bucket_edges is None:
            self.define_return_buckets()
        
        # Engineer features
        features_df = self._engineer_features(df)
        
        # Separate features and target
        X = features_df.drop('next_return', axis=1)
        y = features_df['next_return']
        
        # Convert continuous returns to bucket indices
        y_buckets = self._returns_to_buckets(y.values)
        
        # Split into train and test sets using time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_buckets[:split_idx], y_buckets[split_idx:]
        
        feature_names = X.columns.tolist()
        
        return X_train, y_train, X_test, y_test, feature_names
    
    def build_model(self, num_classes=8):
        """
        Build and initialize XGBoost model for multi-class probability prediction.
        
        Args:
            num_classes (int): Number of prediction buckets
            
        Returns:
            xgb.XGBClassifier: Initialized XGBoost model
        """
        # Initialize XGBoost with multi:softprob objective for probability outputs
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_estimators=200,
            eval_metric='mlogloss'
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, ticker='STOCK', use_calibration=True):
        """
        Train the XGBoost model with optional calibration.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.array): Training targets (bucket indices)
            X_val (pd.DataFrame, optional): Validation features
            y_val (np.array, optional): Validation targets
            ticker (str): Stock ticker for model naming
            use_calibration (bool): Whether to apply probability calibration
            
        Returns:
            model: Trained model
        """
        if self.model is None:
            num_classes = len(self.bucket_edges) + 1  # Number of buckets = edges + 1
            self.build_model(num_classes=num_classes)
            
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train the model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        # Apply probability calibration if requested
        if use_calibration:
            print("Applying probability calibration...")
            from sklearn.base import clone
            # Use a frozen estimator approach instead of deprecated 'prefit'
            # Check for minimum samples per class for CV
            from sklearn.model_selection import StratifiedKFold
            import numpy as np
            
            # Get counts per class
            class_counts = np.bincount(y_train)
            # Determine max possible CV folds (minimum 2)
            min_samples_per_class = np.min(class_counts[class_counts > 0])
            max_cv_folds = min(5, min_samples_per_class)
            
            if max_cv_folds >= 2:
                cv = StratifiedKFold(n_splits=max_cv_folds)
            else:
                # Fall back to using the model as-is without calibration
                print(f"Warning: Not enough samples for calibration (minimum {min_samples_per_class}). Using model without calibration.")
                self.calibrated_model = self.model
                return self.model
                
            self.calibrated_model = CalibratedClassifierCV(
                estimator=clone(self.model),
                method='isotonic', 
                cv=cv
            )
            self.calibrated_model.fit(X_train, y_train)
        
        # Save the model
        model_path = os.path.join(self.model_dir, f"{ticker}_xgboost_model.json")
        self.model.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        return self.model
    
    def cross_validate(self, X, y, n_splits=5):
        """
        Perform time-series cross-validation.
        
        Args:
            X (pd.DataFrame): Features
            y (np.array): Target values
            n_splits (int): Number of time-series splits
            
        Returns:
            dict: Dictionary with cross-validation metrics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        if self.model is None:
            num_classes = len(self.bucket_edges) + 1
            self.build_model(num_classes=num_classes)
        
        cv_scores = {
            'log_loss': [],
            'brier_score': []
        }
        
        # Get the unique classes in the entire dataset
        unique_classes = np.unique(y)
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Skip this fold if some classes are missing in either train or test set
            train_classes = np.unique(y_train)
            test_classes = np.unique(y_test)
            
            # Check if any classes are missing in training data
            if len(train_classes) < len(unique_classes) or len(test_classes) < 1:
                print(f"Skipping fold due to missing classes. Train has {len(train_classes)}/{len(unique_classes)} classes")
                continue
                
            try:
                # Train model on this fold with the same classes that were used in the main model
                self.model.fit(X_train, y_train, verbose=False)
                
                # Get probability predictions
                y_pred_proba = self.model.predict_proba(X_test)
                
                # Calculate metrics
                ll = log_loss(y_test, y_pred_proba)
                cv_scores['log_loss'].append(ll)
                
                # For Brier score, need to calculate for each class and average
                bs_scores = []
                for i in range(y_pred_proba.shape[1]):
                    # Create binary targets for this class (if this class exists in test set)
                    if i in test_classes:
                        y_test_binary = (y_test == i).astype(int)
                        bs = brier_score_loss(y_test_binary, y_pred_proba[:, i])
                        bs_scores.append(bs)
                
                if bs_scores:  # Only append if we have scores
                    cv_scores['brier_score'].append(np.mean(bs_scores))
            except Exception as e:
                print(f"Error in cross-validation fold: {e}")
                continue
        
        # Calculate average scores
        cv_results = {
            'log_loss_mean': np.mean(cv_scores['log_loss']),
            'log_loss_std': np.std(cv_scores['log_loss']),
            'brier_score_mean': np.mean(cv_scores['brier_score']),
            'brier_score_std': np.std(cv_scores['brier_score']),
            'all_log_loss': cv_scores['log_loss'],
            'all_brier_score': cv_scores['brier_score']
        }
        
        return cv_results
    
    def predict_probabilities(self, X):
        """
        Predict probability distribution across buckets.
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.array: Array of probabilities for each bucket
        """
        if self.model is None:
            print("Model not trained or loaded")
            return None
        
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        else:
            return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (np.array): Test targets (bucket indices)
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if self.model is None:
            print("Model not trained or loaded")
            return None
        
        # Get probability predictions
        y_pred_proba = self.predict_probabilities(X_test)
        
        # Calculate log loss
        ll = log_loss(y_test, y_pred_proba)
        
        # Calculate Brier score for each class and average
        bs_scores = []
        for i in range(y_pred_proba.shape[1]):
            # Create binary targets for this class
            y_test_binary = (y_test == i).astype(int)
            bs = brier_score_loss(y_test_binary, y_pred_proba[:, i])
            bs_scores.append(bs)
        
        # Get predicted classes
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        
        # Return metrics
        metrics = {
            'log_loss': ll,
            'brier_score': np.mean(bs_scores),
            'accuracy': accuracy
        }
        
        return metrics
    
    def load_model(self, ticker):
        """
        Load a previously trained model.
        
        Args:
            ticker (str): Stock ticker of the model to load
            
        Returns:
            model: Loaded XGBoost model
        """
        model_path = os.path.join(self.model_dir, f"{ticker}_xgboost_model.json")
        if os.path.exists(model_path):
            # Initialize model if not already done
            if self.model is None:
                # We don't know num_classes yet, will be loaded from file
                self.model = xgb.XGBClassifier()
            
            self.model.load_model(model_path)
            print(f"Model loaded from {model_path}")
            return self.model
        else:
            print(f"No trained model found for {ticker}")
            return None
    
    def plot_feature_importance(self, feature_names=None, top_n=20):
        """
        Plot feature importance from the trained model.
        
        Args:
            feature_names (list): List of feature names
            top_n (int): Number of top features to plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if self.model is None:
            print("Model not trained or loaded")
            return None
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # If feature names not provided, create generic names
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Create DataFrame for plotting
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        plt.barh(feat_imp['feature'][:top_n][::-1], feat_imp['importance'][:top_n][::-1])
        plt.xlabel('Feature Importance')
        plt.title('Top Features by Importance')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.model_dir, 'feature_importance.png'))
        
        return plt.gcf()
    
    def plot_calibration_curve(self, X_test, y_test, n_bins=10):
        """
        Plot calibration curve to evaluate probability calibration.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (np.array): Test targets (bucket indices)
            n_bins (int): Number of bins for calibration curve
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if self.model is None:
            print("Model not trained or loaded")
            return None
        
        # Get probability predictions
        y_pred_proba = self.predict_probabilities(X_test)
        
        # Create improved calibration plot
        plt.figure(figsize=(16, 12))
        
        # Create subplots - one for overall calibration, one for individual classes
        plt.subplot(2, 1, 1)
        
        # Plot the overall reliability diagram (aggregating across all classes)
        from sklearn.calibration import calibration_curve
        
        # Define the classes we have enough samples for
        class_counts = np.bincount(y_test)
        valid_classes = [i for i, count in enumerate(class_counts) if count >= 5]
        
        # Prepare data for reliability diagram
        y_true_flat = np.zeros(len(y_test) * len(valid_classes))
        y_pred_flat = np.zeros(len(y_test) * len(valid_classes))
        
        for idx, i in enumerate(valid_classes):
            # Binary classification for each class
            start_idx = idx * len(y_test)
            end_idx = (idx + 1) * len(y_test)
            
            y_true_flat[start_idx:end_idx] = (y_test == i).astype(int)
            y_pred_flat[start_idx:end_idx] = y_pred_proba[:, i]
        
        # Calculate and plot reliability curve
        prob_true, prob_pred = calibration_curve(y_true_flat, y_pred_flat, n_bins=n_bins)
        plt.plot(prob_pred, prob_true, 's-', label='Overall Model Calibration')
        
        # Plot the ideal curve
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Format the plot
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Actual Probability (Fraction of Positives)')
        plt.title('Overall Calibration (Reliability Diagram)')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Plot calibration curve for each class
        plt.subplot(2, 1, 2)
        
        num_classes = y_pred_proba.shape[1]
        colors = plt.cm.jet(np.linspace(0, 1, num_classes))
        
        bucket_ranges = []
        for i in range(num_classes):
            if i == 0:
                bucket_ranges.append(f"< {self.bucket_edges[0]*100:.1f}%")
            elif i == len(self.bucket_edges):
                bucket_ranges.append(f"> {self.bucket_edges[-1]*100:.1f}%")
            else:
                bucket_ranges.append(f"{self.bucket_edges[i-1]*100:.1f}% to {self.bucket_edges[i]*100:.1f}%")
        
        for i in valid_classes:
            # Only plot classes with sufficient samples
            if class_counts[i] < 5:
                continue
                
            # Create binary targets for this class
            y_test_binary = (y_test == i).astype(int)
            
            # Get predicted probabilities for this class
            y_proba = y_pred_proba[:, i]
            
            # Calculate calibration curve for this class using sklearn's function
            try:
                prob_true, prob_pred = calibration_curve(y_test_binary, y_proba, n_bins=5)
                plt.plot(prob_pred, prob_true, 'o-', color=colors[i], 
                         label=f"{bucket_ranges[i]} (n={class_counts[i]})")
            except Exception as e:
                print(f"Could not plot calibration for class {i}: {e}")
        
        # Plot diagonal perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability (Fraction of Positives)')
        plt.title('Calibration Curve by Return Bucket')
        plt.legend(loc='best')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Add explanation text at the bottom
        plt.figtext(0.5, 0.01, 
                   "Calibration curves show how well the predicted probabilities match actual observed frequencies.\n"
                   "Points near the diagonal line indicate good calibration. Points above the line indicate underconfidence.\n"
                   "Points below the line indicate overconfidence. Ideally, all curves should be close to the diagonal.",
                   ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        # Save the plot
        plt.savefig(os.path.join(self.model_dir, 'calibration_curve.png'))
        
        return plt.gcf()
    
    def bucket_to_return_range(self, bucket_idx):
        """
        Convert bucket index to human-readable return range.
        
        Args:
            bucket_idx (int): Bucket index
            
        Returns:
            str: Human-readable return range
        """
        if self.bucket_edges is None:
            print("Bucket edges not defined")
            return None
        
        if bucket_idx == 0:
            return f"< {self.bucket_edges[0]*100:.1f}%"
        elif bucket_idx == len(self.bucket_edges):
            return f"> {self.bucket_edges[-1]*100:.1f}%"
        else:
            return f"{self.bucket_edges[bucket_idx-1]*100:.1f}% to {self.bucket_edges[bucket_idx]*100:.1f}%"
    
    def predict_with_explanation(self, X, return_proba=True):
        """
        Make a prediction with human-readable explanation.
        
        Args:
            X (pd.DataFrame): Features for prediction
            return_proba (bool): Whether to return full probability distribution
            
        Returns:
            dict: Dictionary with prediction and explanation
        """
        if self.model is None:
            print("Model not trained or loaded")
            return None
        
        # Get probability predictions
        probabilities = self.predict_probabilities(X)
        
        # Get most likely bucket
        most_likely_bucket = np.argmax(probabilities, axis=1)[0]
        
        # Convert to return range
        return_range = self.bucket_to_return_range(most_likely_bucket)
        
        # Calculate expected return (point prediction) from the probability distribution
        expected_return = 0.0
        
        # Define representative values for each bucket
        bucket_values = []
        for i in range(len(self.bucket_edges) + 1):
            if i == 0:  # < first edge (e.g., < -3%)
                bucket_values.append(self.bucket_edges[0] - 0.01)  # e.g., -4%
            elif i == len(self.bucket_edges):  # > last edge (e.g., > 3%)
                bucket_values.append(self.bucket_edges[-1] + 0.01)  # e.g., 4%
            else:  # Between two edges
                # Middle point between two edges
                bucket_values.append((self.bucket_edges[i-1] + self.bucket_edges[i]) / 2)
        
        # Calculate expected value
        for i in range(len(bucket_values)):
            expected_return += bucket_values[i] * probabilities[0, i]
        
        # Prepare result dictionary
        result = {
            'most_likely_bucket': int(most_likely_bucket),
            'return_range': return_range,
            'probability': float(probabilities[0, most_likely_bucket]),
            'expected_return': float(expected_return),
            'expected_return_pct': f"{expected_return*100:.2f}%"
        }
        
        # Include full probability distribution if requested
        if return_proba:
            result['full_distribution'] = {
                self.bucket_to_return_range(i): float(probabilities[0, i])
                for i in range(probabilities.shape[1])
            }
            
            # Sort by probability in descending order
            result['sorted_distribution'] = {
                k: v for k, v in sorted(
                    result['full_distribution'].items(), 
                    key=lambda item: item[1], 
                    reverse=True
                )
            }
        
        return result