"""
Simple example demonstrating KAF algorithms with synthetic data.
"""
import sys
sys.path.append('..')

import numpy as np
from src.algorithms import KLMS, KNLMS, KAPA, KRLS
from src.stream import KAFRegressor
from src.evaluation import prequential_evaluation
from src.data import generate_sample_data
from river import metrics
import matplotlib.pyplot as plt


def main():
    print("="*60)
    print("KAF Algorithms - Simple Example with Synthetic Data")
    print("="*60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y = generate_sample_data(n_samples=500, n_features=5)
    
    # Convert to stream format (dict of features)
    stream_data = []
    for i in range(len(X)):
        x_dict = {f"feature_{j}": X[i, j] for j in range(X.shape[1])}
        stream_data.append((x_dict, y[i]))
    
    print(f"   Generated {len(stream_data)} samples with {X.shape[1]} features")
    
    # Test different KAF algorithms
    algorithms = {
        'KLMS': KAFRegressor(
            algorithm='KLMS',
            learning_rate=0.1,
            kernel='gaussian',
            kernel_size=1.0,
            max_dictionary_size=100
        ),
        'KNLMS': KAFRegressor(
            algorithm='KNLMS',
            learning_rate=0.5,
            kernel='gaussian',
            kernel_size=1.0,
            max_dictionary_size=100
        ),
        'KAPA': KAFRegressor(
            algorithm='KAPA',
            learning_rate=0.1,
            kernel='gaussian',
            kernel_size=1.0,
            max_dictionary_size=100,
            projection_order=3
        ),
        'KRLS': KAFRegressor(
            algorithm='KRLS',
            learning_rate=0.1,
            kernel='gaussian',
            kernel_size=1.0,
            max_dictionary_size=100,
            forgetting_factor=0.99
        )
    }
    
    # Evaluate each algorithm
    results = {}
    predictions_dict = {}
    
    for name, model in algorithms.items():
        print(f"\n2. Evaluating {name}...")
        print("-" * 60)
        
        # Create fresh copy of stream data
        stream_copy = list(stream_data)
        
        # Evaluate
        metrics_dict, results_df = prequential_evaluation(
            model,
            stream_copy,
            metrics_list=[metrics.MAE(), metrics.RMSE(), metrics.R2()],
            verbose=False,
            warm_start=10  # Train on first 10 samples without testing
        )
        
        results[name] = metrics_dict
        predictions_dict[name] = results_df['predicted'].values
        
        print(f"   MAE:  {metrics_dict['MAE']:.4f}")
        print(f"   RMSE: {metrics_dict['RMSE']:.4f}")
        print(f"   R2:   {metrics_dict['R2']:.4f}")
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Algorithm':<10} {'MAE':<10} {'RMSE':<10} {'R2':<10}")
    print("-" * 60)
    for name, metrics_dict in results.items():
        print(f"{name:<10} {metrics_dict['MAE']:<10.4f} "
              f"{metrics_dict['RMSE']:<10.4f} {metrics_dict['R2']:<10.4f}")
    
    # Find best algorithm
    best_algo = min(results.items(), key=lambda x: x[1]['MAE'])
    print(f"\nBest algorithm by MAE: {best_algo[0]}")
    
    # Plot results
    print("\n3. Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('KAF Algorithms Performance Comparison', fontsize=16)
    
    # Plot 1: Predictions vs Actual for best algorithm
    ax = axes[0, 0]
    actual = y[10:]  # Skip warm start samples
    predicted = predictions_dict[best_algo[0]]
    ax.plot(actual[:100], label='Actual', linewidth=2, alpha=0.7)
    ax.plot(predicted[:100], label=f'{best_algo[0]} Prediction', linewidth=2, alpha=0.7)
    ax.set_title(f'{best_algo[0]} - Predictions vs Actual (First 100 samples)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution for best algorithm
    ax = axes[0, 1]
    errors = actual - predicted
    ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'{best_algo[0]} - Error Distribution')
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: MAE comparison
    ax = axes[1, 0]
    names = list(results.keys())
    mae_values = [results[name]['MAE'] for name in names]
    ax.bar(names, mae_values, alpha=0.7, edgecolor='black')
    ax.set_title('MAE Comparison')
    ax.set_ylabel('MAE')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Scatter plot actual vs predicted for best algorithm
    ax = axes[1, 1]
    ax.scatter(actual, predicted, alpha=0.5, s=20)
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_title(f'{best_algo[0]} - Actual vs Predicted')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/simple_example_results.png', dpi=150, bbox_inches='tight')
    print("   Saved plot to: results/simple_example_results.png")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
