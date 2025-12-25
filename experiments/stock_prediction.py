"""
Stock price prediction experiment using KAF algorithms.

This script demonstrates the main application from the paper:
predicting mid-price movements in financial time series.
"""
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from src.stream import KAFRegressor
from src.data import load_stock_data, calculate_technical_indicators, calculate_mid_price
from src.evaluation import (
    prequential_evaluation, 
    evaluate_directional_accuracy_online,
    OnlineEvaluator
)
from river import metrics, linear_model
import matplotlib.pyplot as plt
import argparse
from datetime import datetime, timedelta


def prepare_stock_stream(symbol, start_date, end_date, interval='1d'):
    """
    Load and prepare stock data for streaming.
    
    Args:
        symbol: Stock ticker
        start_date: Start date
        end_date: End date
        interval: Time interval
        
    Returns:
        List of (features_dict, target) tuples
    """
    print(f"Loading data for {symbol}...")
    
    # Load data
    df = load_stock_data(symbol, start_date, end_date, interval)
    
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    
    print(f"Loaded {len(df)} samples")
    
    # Add technical indicators
    print("Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    
    # Calculate mid-price
    df['mid_price'] = calculate_mid_price(df)
    
    # Create target: next mid-price
    df['target'] = df['mid_price'].shift(-1)
    
    # Select features (exclude NaN-heavy features)
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'returns', 'volatility_10', 'volume_ratio',
        'momentum_5', 'roc_5',
        'rsi', 'macd', 'macd_diff'
    ]
    
    # Keep only available features
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Drop NaN rows
    df_clean = df[feature_cols + ['target']].dropna()
    
    print(f"After cleaning: {len(df_clean)} samples with {len(feature_cols)} features")
    print(f"Features: {', '.join(feature_cols)}")
    
    # Convert to stream format
    stream_data = []
    for idx in range(len(df_clean)):
        x = df_clean[feature_cols].iloc[idx].to_dict()
        y = df_clean['target'].iloc[idx]
        stream_data.append((x, y))
    
    return stream_data, df_clean


def run_experiment(
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    interval='1d',
    algorithm='KLMS'
):
    """
    Run stock prediction experiment.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date
        end_date: End date
        interval: Time interval
        algorithm: KAF algorithm to use
    """
    print("="*70)
    print(f"Stock Price Prediction with {algorithm}")
    print("="*70)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    print("="*70)
    
    # Prepare data
    stream_data, df_clean = prepare_stock_stream(symbol, start_date, end_date, interval)
    
    # Create models
    print("\n" + "="*70)
    print("Creating models...")
    print("="*70)
    
    models = {
        algorithm: KAFRegressor(
            algorithm=algorithm,
            learning_rate=0.1,
            kernel='gaussian',
            kernel_size=1.0,
            max_dictionary_size=200,
            novelty_threshold=0.1
        ),
        'Linear (Baseline)': linear_model.LinearRegression()
    }
    
    # Evaluate each model
    results = {}
    predictions_dict = {}
    
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {name}")
        print(f"{'='*70}")
        
        # Create fresh stream
        stream_copy = list(stream_data)
        
        # Prequential evaluation
        metrics_dict, results_df = prequential_evaluation(
            model,
            stream_copy,
            metrics_list=[metrics.MAE(), metrics.RMSE(), metrics.R2()],
            verbose=True,
            warm_start=20
        )
        
        results[name] = metrics_dict
        predictions_dict[name] = results_df
        
        # Evaluate directional accuracy
        print("\nCalculating directional accuracy...")
        stream_copy = list(stream_data)
        dir_acc, _ = evaluate_directional_accuracy_online(
            model,
            stream_copy,
            verbose=False
        )
        results[name]['Directional_Accuracy'] = dir_acc
        print(f"Directional Accuracy: {dir_acc:.4f} ({dir_acc*100:.2f}%)")
    
    # Print comparison
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'R2':<12} {'Dir.Acc.':<12}")
    print("-" * 70)
    for name, metrics_dict in results.items():
        print(f"{name:<20} {metrics_dict['MAE']:<12.4f} "
              f"{metrics_dict['RMSE']:<12.4f} {metrics_dict['R2']:<12.4f} "
              f"{metrics_dict['Directional_Accuracy']:<12.4f}")
    
    # Plot results
    print("\n" + "="*70)
    print("Generating plots...")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{symbol} - {algorithm} vs Linear Baseline', fontsize=16)
    
    # Plot 1: Predictions comparison
    ax = axes[0, 0]
    actual = df_clean['target'].values[20:]  # Skip warm start
    kaf_pred = predictions_dict[algorithm]['predicted'].values
    linear_pred = predictions_dict['Linear (Baseline)']['predicted'].values
    
    plot_range = min(100, len(actual))
    ax.plot(actual[:plot_range], label='Actual', linewidth=2, alpha=0.8)
    ax.plot(kaf_pred[:plot_range], label=algorithm, linewidth=1.5, alpha=0.8)
    ax.plot(linear_pred[:plot_range], label='Linear', linewidth=1.5, alpha=0.8)
    ax.set_title('Predictions vs Actual (First 100 samples)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Mid-Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error comparison
    ax = axes[0, 1]
    kaf_errors = predictions_dict[algorithm]['error'].values
    linear_errors = predictions_dict['Linear (Baseline)']['error'].values
    
    ax.boxplot([kaf_errors, linear_errors], labels=[algorithm, 'Linear'])
    ax.set_title('Error Distribution Comparison')
    ax.set_ylabel('Error')
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Metrics comparison
    ax = axes[1, 0]
    metrics_names = ['MAE', 'RMSE', 'Directional_Accuracy']
    x = np.arange(len(metrics_names))
    width = 0.35
    
    kaf_values = [results[algorithm][m] for m in metrics_names]
    linear_values = [results['Linear (Baseline)'][m] for m in metrics_names]
    
    ax.bar(x - width/2, kaf_values, width, label=algorithm, alpha=0.8)
    ax.bar(x + width/2, linear_values, width, label='Linear', alpha=0.8)
    ax.set_ylabel('Value')
    ax.set_title('Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Scatter plot
    ax = axes[1, 1]
    ax.scatter(actual, kaf_pred, alpha=0.5, s=20, label=algorithm)
    ax.scatter(actual, linear_pred, alpha=0.5, s=20, label='Linear')
    min_val = actual.min()
    max_val = actual.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_title('Actual vs Predicted')
    ax.set_xlabel('Actual Mid-Price')
    ax.set_ylabel('Predicted Mid-Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'../results/stock_prediction_{symbol}_{algorithm}_{interval}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results).T
    csv_filename = f'../results/results_{symbol}_{algorithm}_{interval}.csv'
    results_df.to_csv(csv_filename)
    print(f"Saved results to: {csv_filename}")
    
    print("\n" + "="*70)
    print("Experiment completed successfully!")
    print("="*70)
    
    return results, predictions_dict


def main():
    parser = argparse.ArgumentParser(description='Stock price prediction with KAF')
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock ticker symbol')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-01-01',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1d',
                       help='Time interval (1m, 5m, 15m, 1h, 1d)')
    parser.add_argument('--algorithm', type=str, default='KLMS',
                       choices=['KLMS', 'KNLMS', 'KAPA', 'KRLS'],
                       help='KAF algorithm to use')
    
    args = parser.parse_args()
    
    try:
        run_experiment(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
            algorithm=args.algorithm
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
