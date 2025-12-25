"""
Evaluation utilities for streaming algorithms.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from river import metrics
import time


class OnlineEvaluator:
    """
    Evaluator for online/streaming algorithms.
    
    This class tracks performance metrics during online learning,
    compatible with River's prequential evaluation approach.
    """
    
    def __init__(self, metrics_list: Optional[List] = None):
        """
        Initialize evaluator.
        
        Args:
            metrics_list: List of River metrics to track
        """
        if metrics_list is None:
            # Default metrics for regression
            self.metrics_dict = {
                'MAE': metrics.MAE(),
                'RMSE': metrics.RMSE(),
                'R2': metrics.R2()
            }
        else:
            self.metrics_dict = {
                metric.__class__.__name__: metric 
                for metric in metrics_list
            }
        
        # Tracking
        self.predictions = []
        self.actuals = []
        self.errors = []
        self.timestamps = []
        self.latencies = []
    
    def update(self, y_true: float, y_pred: float):
        """
        Update metrics with a new prediction.
        
        Args:
            y_true: True value
            y_pred: Predicted value
        """
        # Update all metrics
        for metric in self.metrics_dict.values():
            metric.update(y_true, y_pred)
        
        # Store for later analysis
        self.predictions.append(y_pred)
        self.actuals.append(y_true)
        self.errors.append(y_true - y_pred)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metric values.
        
        Returns:
            Dictionary of metric names and values
        """
        return {
            name: metric.get() 
            for name, metric in self.metrics_dict.items()
        }
    
    def get_results_df(self) -> pd.DataFrame:
        """
        Get results as a DataFrame.
        
        Returns:
            DataFrame with predictions, actuals, and errors
        """
        return pd.DataFrame({
            'actual': self.actuals,
            'predicted': self.predictions,
            'error': self.errors
        })
    
    def print_metrics(self):
        """Print current metrics."""
        metrics_dict = self.get_metrics()
        print("\nCurrent Metrics:")
        print("-" * 40)
        for name, value in metrics_dict.items():
            print(f"{name}: {value:.6f}")
        print("-" * 40)


def prequential_evaluation(
    model,
    stream_data,
    n_samples: Optional[int] = None,
    metrics_list: Optional[List] = None,
    verbose: bool = True,
    warm_start: int = 0
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Perform prequential (test-then-train) evaluation.
    
    This is the standard evaluation method for streaming algorithms:
    1. Test on current sample
    2. Train on current sample
    3. Move to next sample
    
    Args:
        model: Model with predict_one and learn_one methods
        stream_data: Iterable of (x, y) tuples
        n_samples: Maximum number of samples to process (None = all)
        metrics_list: List of metrics to track
        verbose: Whether to print progress
        warm_start: Number of samples to train before testing
        
    Returns:
        Tuple of (final metrics dict, results DataFrame)
    """
    evaluator = OnlineEvaluator(metrics_list)
    
    sample_count = 0
    start_time = time.time()
    
    for x, y in stream_data:
        if n_samples and sample_count >= n_samples:
            break
        
        # Warm start: train without testing
        if sample_count < warm_start:
            model.learn_one(x, y)
            sample_count += 1
            continue
        
        # Test-then-train
        pred_start = time.time()
        y_pred = model.predict_one(x)
        pred_time = time.time() - pred_start
        
        model.learn_one(x, y)
        
        # Update metrics
        evaluator.update(y, y_pred)
        evaluator.latencies.append(pred_time)
        
        sample_count += 1
        
        # Print progress
        if verbose and sample_count % 100 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = sample_count / elapsed
            print(f"Processed {sample_count} samples "
                  f"({samples_per_sec:.1f} samples/sec) - "
                  f"MAE: {evaluator.metrics_dict['MAE'].get():.4f}")
    
    # Final results
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\nEvaluation complete:")
        print(f"Total samples: {sample_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average latency: {np.mean(evaluator.latencies)*1000:.2f}ms")
        evaluator.print_metrics()
    
    return evaluator.get_metrics(), evaluator.get_results_df()


def compare_algorithms(
    algorithms: Dict[str, object],
    stream_data,
    n_samples: Optional[int] = None,
    metrics_list: Optional[List] = None
) -> pd.DataFrame:
    """
    Compare multiple algorithms on the same stream.
    
    Args:
        algorithms: Dictionary of {name: model} pairs
        stream_data: Iterable of (x, y) tuples
        n_samples: Maximum samples to process
        metrics_list: Metrics to track
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, model in algorithms.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"{'='*50}")
        
        # Need to recreate stream for each algorithm
        # This assumes stream_data is re-iterable
        metrics_dict, _ = prequential_evaluation(
            model,
            stream_data,
            n_samples=n_samples,
            metrics_list=metrics_list,
            verbose=True
        )
        
        metrics_dict['Algorithm'] = name
        results.append(metrics_dict)
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    df = df.set_index('Algorithm')
    
    return df


def calculate_directional_accuracy(
    actuals: np.ndarray,
    predictions: np.ndarray
) -> float:
    """
    Calculate directional accuracy (correct up/down predictions).
    
    This is the main metric used in the paper:
    What percentage of times did we correctly predict the direction
    of the price movement?
    
    Args:
        actuals: Array of actual values
        predictions: Array of predicted values
        
    Returns:
        Directional accuracy (0-1)
    """
    # Calculate actual and predicted directions
    actual_direction = np.diff(actuals) > 0
    predicted_direction = np.diff(predictions) > 0
    
    # Calculate accuracy
    correct = np.sum(actual_direction == predicted_direction)
    total = len(actual_direction)
    
    return correct / total if total > 0 else 0.0


def evaluate_directional_accuracy_online(
    model,
    stream_data,
    n_samples: Optional[int] = None,
    verbose: bool = True
) -> Tuple[float, List[float]]:
    """
    Evaluate directional accuracy in online setting.
    
    Args:
        model: Model with predict_one and learn_one
        stream_data: Iterable of (x, y) tuples
        n_samples: Maximum samples to process
        verbose: Print progress
        
    Returns:
        Tuple of (final accuracy, accuracy history)
    """
    predictions = []
    actuals = []
    accuracy_history = []
    
    sample_count = 0
    correct_directions = 0
    total_directions = 0
    
    prev_actual = None
    prev_pred = None
    
    for x, y in stream_data:
        if n_samples and sample_count >= n_samples:
            break
        
        # Predict and learn
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        
        # Check direction if we have previous values
        if prev_actual is not None and prev_pred is not None:
            actual_dir = y > prev_actual
            pred_dir = y_pred > prev_pred
            
            if actual_dir == pred_dir:
                correct_directions += 1
            total_directions += 1
            
            # Calculate running accuracy
            if total_directions > 0:
                accuracy = correct_directions / total_directions
                accuracy_history.append(accuracy)
        
        prev_actual = y
        prev_pred = y_pred
        predictions.append(y_pred)
        actuals.append(y)
        sample_count += 1
        
        if verbose and sample_count % 100 == 0:
            if total_directions > 0:
                acc = correct_directions / total_directions
                print(f"Samples: {sample_count}, Directional Accuracy: {acc:.4f}")
    
    final_accuracy = correct_directions / total_directions if total_directions > 0 else 0.0
    
    if verbose:
        print(f"\nFinal Directional Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    return final_accuracy, accuracy_history


def sliding_window_evaluation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 100,
    step_size: int = 10
) -> List[Dict[str, float]]:
    """
    Evaluate model performance on sliding windows.
    
    Useful for analyzing performance over time and detecting
    concept drift.
    
    Args:
        model: Model with predict_one and learn_one
        X: Feature array
        y: Target array
        window_size: Size of evaluation window
        step_size: Step size between windows
        
    Returns:
        List of metric dictionaries for each window
    """
    results = []
    
    for start_idx in range(0, len(X) - window_size, step_size):
        end_idx = start_idx + window_size
        
        # Reset metrics for this window
        mae = metrics.MAE()
        rmse = metrics.RMSE()
        
        # Evaluate on window
        for i in range(start_idx, end_idx):
            if isinstance(X[i], dict):
                x = X[i]
            else:
                x = {f"f{j}": X[i][j] for j in range(len(X[i]))}
            
            y_pred = model.predict_one(x)
            model.learn_one(x, y[i])
            
            mae.update(y[i], y_pred)
            rmse.update(y[i], y_pred)
        
        results.append({
            'window_start': start_idx,
            'window_end': end_idx,
            'MAE': mae.get(),
            'RMSE': rmse.get()
        })
    
    return results
