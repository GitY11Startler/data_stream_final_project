"""
Multi-Stock Experiment
======================

This script tests all KAF and CapyMOA algorithms across multiple stocks
to demonstrate generalizability and reproduce experiments from the original paper.

The original paper tested on Nifty-50 Indian stocks. We use US stocks
across different sectors to demonstrate algorithm performance across diverse markets.

Usage:
    python multi_stock_experiment.py
    python multi_stock_experiment.py --stocks AAPL,GOOGL,MSFT,AMZN,TSLA
    python multi_stock_experiment.py --stock-list tech
    python multi_stock_experiment.py --interval 1h
    
Output:
    - results/multi_stock_results.csv: Detailed results per stock
    - results/multi_stock_summary.csv: Aggregated performance across all stocks
    - results/multi_stock_results.png: Visualization
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.stream import KAFRegressor, CapyMOARegressor
from src.data.stock_data import (
    load_stock_data, 
    adjust_dates_for_interval,
    STOCK_LISTS
)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate directional accuracy (percentage of correct direction predictions)."""
    if len(y_true) < 2:
        return 0.0
    
    # Calculate actual and predicted directions
    actual_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    # Calculate accuracy
    return np.mean(actual_direction == pred_direction)


# Default stocks representing different sectors
DEFAULT_STOCKS = [
    'AAPL',   # Tech - Apple
    'GOOGL',  # Tech - Alphabet
    'MSFT',   # Tech - Microsoft
    'JPM',    # Finance - JPMorgan
    'JNJ',    # Healthcare - Johnson & Johnson
    'WMT',    # Retail - Walmart
    'XOM',    # Energy - Exxon
    'TSLA',   # Auto/Tech - Tesla
]


def create_fresh_algorithms() -> Dict:
    """
    Create fresh instances of all algorithms.
    Must be called for each stock to ensure independence.
    
    Returns:
        Dictionary of algorithm name -> algorithm instance
    """
    algorithms = {
        # KAF Algorithms (implemented)
        'KLMS': KAFRegressor(
            algorithm='KLMS',
            learning_rate=0.1,
            kernel='gaussian',
            kernel_size=1.0,
            max_dictionary_size=100
        ),
        'KNLMS': KAFRegressor(
            algorithm='KNLMS',
            learning_rate=0.1,
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
            epsilon=0.1
        ),
        'KRLS': KAFRegressor(
            algorithm='KRLS',
            forgetting_factor=0.99,
            kernel='gaussian',
            kernel_size=1.0,
            max_dictionary_size=100,
            ald_threshold=0.1
        ),
        # CapyMOA Algorithms (library)
        'ARF': CapyMOARegressor(
            algorithm='AdaptiveRandomForestRegressor',
            ensemble_size=10
        ),
        'KNN': CapyMOARegressor(
            algorithm='KNNRegressor',
            k=5
        ),
        'SGBR': CapyMOARegressor(
            algorithm='StreamingGradientBoostedRegression'
        )
    }
    return algorithms


def load_and_prepare_data(symbol: str, start_date: str, end_date: str,
                          interval: str = '1d') -> Tuple[List, pd.DataFrame, Tuple]:
    """
    Load stock data and prepare for streaming evaluation.
    
    Args:
        symbol: Stock ticker
        start_date: Start date
        end_date: End date
        interval: Data interval
        
    Returns:
        Tuple of (stream_data, dataframe, normalization_params)
    """
    try:
        df = load_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            use_cache=True
        )
        
        if df.empty:
            return None, None, None
        
        # Handle multi-level columns
        if hasattr(df.columns, 'levels'):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                          for col in df.columns]
        
        # Find target column (Close price)
        target_col = None
        for col in df.columns:
            if 'Close' in str(col):
                target_col = col
                break
        
        if target_col is None:
            return None, None, None
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Compute normalization statistics
        feature_means = df[feature_cols].mean()
        feature_stds = df[feature_cols].std().replace(0, 1)
        target_mean = df[target_col].mean()
        target_std = df[target_col].std()
        if target_std == 0:
            target_std = 1
        
        # Create normalized stream data
        stream_data = []
        for idx, row in df.iterrows():
            y = (row[target_col] - target_mean) / target_std
            x_dict = {col: (row[col] - feature_means[col]) / feature_stds[col] 
                      for col in feature_cols}
            stream_data.append((x_dict, y))
        
        return stream_data, df, (target_mean, target_std)
        
    except Exception as e:
        print(f"  ⚠️ Error loading {symbol}: {e}")
        return None, None, None


def evaluate_algorithm(algorithm, stream_data: List, warm_start: int = 20) -> Dict:
    """
    Evaluate a single algorithm on stream data using prequential evaluation.
    
    Args:
        algorithm: Algorithm instance with predict_one/learn_one
        stream_data: List of (x_dict, y) tuples
        warm_start: Samples to use for warm-up
        
    Returns:
        Dictionary with metrics
    """
    y_true = []
    y_pred = []
    start_time = time.time()
    
    for i, (x, y) in enumerate(stream_data):
        # Predict (after warm-up)
        if i >= warm_start:
            pred = algorithm.predict_one(x)
            if pred is not None:
                y_true.append(y)
                y_pred.append(pred)
        
        # Learn
        algorithm.learn_one(x, y)
    
    elapsed_time = time.time() - start_time
    
    if len(y_true) < 2:
        return {
            'MAE': np.nan,
            'RMSE': np.nan,
            'R2': np.nan,
            'directional_accuracy': np.nan,
            'time_seconds': elapsed_time,
            'samples': len(y_true)
        }
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return {
        'MAE': calculate_mae(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred),
        'directional_accuracy': directional_accuracy(y_true, y_pred),
        'time_seconds': elapsed_time,
        'samples': len(y_true)
    }


def run_experiment_for_stock(symbol: str, start_date: str, end_date: str,
                             interval: str = '1d', warm_start: int = 20) -> pd.DataFrame:
    """
    Run full experiment for a single stock.
    
    Args:
        symbol: Stock ticker
        start_date: Start date
        end_date: End date
        interval: Data interval
        warm_start: Warm-up samples
        
    Returns:
        DataFrame with results for all algorithms
    """
    # Load data
    stream_data, df, norm_params = load_and_prepare_data(
        symbol, start_date, end_date, interval
    )
    
    if stream_data is None or len(stream_data) < warm_start + 10:
        print(f"  ⚠️ Insufficient data for {symbol}")
        return None
    
    results = []
    
    # Create fresh algorithms for this stock
    algorithms = create_fresh_algorithms()
    
    for algo_name, algo in algorithms.items():
        metrics = evaluate_algorithm(algo, stream_data, warm_start)
        metrics['algorithm'] = algo_name
        metrics['stock'] = symbol
        metrics['type'] = 'KAF' if algo_name in ['KLMS', 'KNLMS', 'KAPA', 'KRLS'] else 'CapyMOA'
        results.append(metrics)
    
    return pd.DataFrame(results)


def run_multi_stock_experiment(stocks: List[str], interval: str = '1d',
                               verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run experiment across multiple stocks.
    
    Args:
        stocks: List of stock symbols
        interval: Data interval
        verbose: Print progress
        
    Returns:
        Tuple of (detailed_results_df, summary_df)
    """
    # Adjust dates for interval
    start_date, end_date = adjust_dates_for_interval(interval)
    
    print("\n" + "="*70)
    print("MULTI-STOCK EXPERIMENT")
    print("="*70)
    print(f"  Stocks: {stocks}")
    print(f"  Interval: {interval}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Algorithms: KLMS, KNLMS, KAPA, KRLS (KAF) + ARF, KNN, SGBR (CapyMOA)")
    print("="*70)
    
    all_results = []
    successful_stocks = []
    
    for i, symbol in enumerate(stocks):
        print(f"\n[{i+1}/{len(stocks)}] Processing {symbol}...")
        
        result_df = run_experiment_for_stock(
            symbol, start_date, end_date, interval
        )
        
        if result_df is not None:
            all_results.append(result_df)
            successful_stocks.append(symbol)
            
            # Print quick summary for this stock
            if verbose:
                best_mae = result_df.loc[result_df['MAE'].idxmin()]
                best_da = result_df.loc[result_df['directional_accuracy'].idxmax()]
                print(f"  ✅ {symbol}: {len(result_df)} algorithms evaluated")
                print(f"     Best MAE: {best_mae['algorithm']} ({best_mae['MAE']:.4f})")
                print(f"     Best Dir.Acc: {best_da['algorithm']} ({best_da['directional_accuracy']:.2%})")
    
    if not all_results:
        print("\n❌ No stocks could be processed!")
        return None, None
    
    # Combine all results
    detailed_df = pd.concat(all_results, ignore_index=True)
    
    # Create summary - average performance across all stocks
    summary_df = detailed_df.groupby(['algorithm', 'type']).agg({
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'directional_accuracy': ['mean', 'std'],
        'time_seconds': ['mean', 'sum'],
        'samples': 'sum'
    }).round(4)
    
    # Flatten column names
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns]
    summary_df = summary_df.reset_index()
    
    # Sort by MAE
    summary_df = summary_df.sort_values('MAE_mean')
    
    return detailed_df, summary_df


def display_results(detailed_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Display formatted results."""
    print("\n" + "="*70)
    print("AGGREGATED RESULTS ACROSS ALL STOCKS")
    print("="*70)
    
    # Count stocks
    n_stocks = detailed_df['stock'].nunique()
    stocks = detailed_df['stock'].unique()
    
    print(f"\nStocks analyzed: {n_stocks}")
    print(f"  {', '.join(stocks)}")
    
    print("\n" + "-"*70)
    print("AVERAGE PERFORMANCE (sorted by MAE)")
    print("-"*70)
    print(f"{'Algorithm':<10} {'Type':<8} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'Dir.Acc':>10}")
    print("-"*70)
    
    for _, row in summary_df.iterrows():
        print(f"{row['algorithm']:<10} {row['type']:<8} "
              f"{row['MAE_mean']:>10.4f} {row['RMSE_mean']:>10.4f} "
              f"{row['R2_mean']:>10.4f} {row['directional_accuracy_mean']:>9.2%}")
    
    # Find best algorithm
    best_mae_algo = summary_df.loc[summary_df['MAE_mean'].idxmin(), 'algorithm']
    best_r2_algo = summary_df.loc[summary_df['R2_mean'].idxmax(), 'algorithm']
    best_da_algo = summary_df.loc[summary_df['directional_accuracy_mean'].idxmax(), 'algorithm']
    
    print("\n" + "-"*70)
    print("BEST ALGORITHMS")
    print("-"*70)
    print(f"  Best MAE: {best_mae_algo}")
    print(f"  Best R²: {best_r2_algo}")
    print(f"  Best Directional Accuracy: {best_da_algo}")
    
    # KAF vs CapyMOA comparison
    print("\n" + "-"*70)
    print("KAF vs CAPYMOA COMPARISON")
    print("-"*70)
    
    kaf_df = summary_df[summary_df['type'] == 'KAF']
    capymoa_df = summary_df[summary_df['type'] == 'CapyMOA']
    
    kaf_avg_mae = kaf_df['MAE_mean'].mean()
    capymoa_avg_mae = capymoa_df['MAE_mean'].mean()
    kaf_avg_da = kaf_df['directional_accuracy_mean'].mean()
    capymoa_avg_da = capymoa_df['directional_accuracy_mean'].mean()
    
    print(f"  KAF Average MAE: {kaf_avg_mae:.4f}")
    print(f"  CapyMOA Average MAE: {capymoa_avg_mae:.4f}")
    print(f"  Improvement: {((capymoa_avg_mae - kaf_avg_mae) / capymoa_avg_mae * 100):.1f}%")
    print()
    print(f"  KAF Average Dir.Acc: {kaf_avg_da:.2%}")
    print(f"  CapyMOA Average Dir.Acc: {capymoa_avg_da:.2%}")
    print(f"  Improvement: {((kaf_avg_da - capymoa_avg_da) / capymoa_avg_da * 100):+.1f}%")
    
    # Per-stock best algorithm
    print("\n" + "-"*70)
    print("BEST ALGORITHM PER STOCK (by Directional Accuracy)")
    print("-"*70)
    
    for stock in detailed_df['stock'].unique():
        stock_data = detailed_df[detailed_df['stock'] == stock]
        best = stock_data.loc[stock_data['directional_accuracy'].idxmax()]
        print(f"  {stock:<6}: {best['algorithm']:<6} ({best['directional_accuracy']:.2%})")


def create_visualization(detailed_df: pd.DataFrame, summary_df: pd.DataFrame,
                         save_path: Optional[str] = None):
    """Create visualization of multi-stock results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color scheme
    colors = {
        'KLMS': '#2E86AB', 'KNLMS': '#1E5F74', 'KAPA': '#145369', 'KRLS': '#0D3B4F',
        'ARF': '#F77F00', 'KNN': '#FCBF49', 'SGBR': '#EAE2B7'
    }
    
    algorithms = summary_df['algorithm'].tolist()
    x = np.arange(len(algorithms))
    bar_colors = [colors.get(algo, '#999999') for algo in algorithms]
    
    # Plot 1: Average MAE
    ax1 = axes[0, 0]
    mae_means = summary_df['MAE_mean'].values
    mae_stds = summary_df['MAE_std'].values
    bars1 = ax1.bar(x, mae_means, yerr=mae_stds, color=bar_colors, capsize=3)
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('MAE (normalized)')
    ax1.set_title('Average MAE Across All Stocks\n(lower is better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Average R²
    ax2 = axes[0, 1]
    r2_means = summary_df['R2_mean'].values
    r2_stds = summary_df['R2_std'].values
    bars2 = ax2.bar(x, r2_means, yerr=r2_stds, color=bar_colors, capsize=3)
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('R²')
    ax2.set_title('Average R² Across All Stocks\n(higher is better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Average Directional Accuracy
    ax3 = axes[1, 0]
    da_means = summary_df['directional_accuracy_mean'].values * 100
    da_stds = summary_df['directional_accuracy_std'].values * 100
    bars3 = ax3.bar(x, da_means, yerr=da_stds, color=bar_colors, capsize=3)
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Directional Accuracy (%)')
    ax3.set_title('Average Directional Accuracy Across All Stocks\n(higher is better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Random (50%)')
    ax3.legend(loc='lower right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Per-stock heatmap for Directional Accuracy
    ax4 = axes[1, 1]
    
    # Create pivot table for heatmap
    pivot_da = detailed_df.pivot(index='stock', columns='algorithm', values='directional_accuracy')
    # Reorder columns
    col_order = ['KLMS', 'KNLMS', 'KAPA', 'KRLS', 'ARF', 'KNN', 'SGBR']
    pivot_da = pivot_da[[c for c in col_order if c in pivot_da.columns]]
    
    im = ax4.imshow(pivot_da.values * 100, cmap='RdYlGn', aspect='auto', vmin=40, vmax=70)
    ax4.set_xticks(np.arange(len(pivot_da.columns)))
    ax4.set_yticks(np.arange(len(pivot_da.index)))
    ax4.set_xticklabels(pivot_da.columns, rotation=45, ha='right')
    ax4.set_yticklabels(pivot_da.index)
    ax4.set_title('Directional Accuracy per Stock (%)')
    
    # Add text annotations
    for i in range(len(pivot_da.index)):
        for j in range(len(pivot_da.columns)):
            val = pivot_da.values[i, j] * 100
            text_color = 'white' if val < 50 or val > 60 else 'black'
            ax4.text(j, i, f'{val:.1f}', ha='center', va='center', 
                     color=text_color, fontsize=8)
    
    plt.colorbar(im, ax=ax4, label='Dir. Acc (%)')
    
    # Add legend for algorithm types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='KAF Algorithms'),
        Patch(facecolor='#F77F00', label='CapyMOA Algorithms')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
               bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('Multi-Stock Experiment Results\nKAF vs CapyMOA Algorithms', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"\n✅ Visualization saved to: {save_path}")
    
    plt.close()


def save_results(detailed_df: pd.DataFrame, summary_df: pd.DataFrame, 
                 output_dir: str = 'results'):
    """Save results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    detailed_path = os.path.join(output_dir, 'multi_stock_results.csv')
    detailed_df.to_csv(detailed_path, index=False)
    print(f"✅ Detailed results saved to: {detailed_path}")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'multi_stock_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary saved to: {summary_path}")
    
    # Save timestamped backup
    backup_detailed = os.path.join(output_dir, f'multi_stock_results_{timestamp}.csv')
    backup_summary = os.path.join(output_dir, f'multi_stock_summary_{timestamp}.csv')
    detailed_df.to_csv(backup_detailed, index=False)
    summary_df.to_csv(backup_summary, index=False)
    print(f"✅ Backups saved with timestamp: {timestamp}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Stock Experiment for KAF vs CapyMOA comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multi_stock_experiment.py
  python multi_stock_experiment.py --stocks AAPL,GOOGL,MSFT
  python multi_stock_experiment.py --stock-list tech
  python multi_stock_experiment.py --interval 1h
        """
    )
    
    parser.add_argument('--stocks', type=str, default=None,
                        help='Comma-separated list of stock symbols')
    parser.add_argument('--stock-list', type=str, choices=list(STOCK_LISTS.keys()),
                        help=f'Pre-configured stock list: {list(STOCK_LISTS.keys())}')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (1d, 1h, 5m, etc.)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Determine stocks to use
    if args.stocks:
        stocks = [s.strip().upper() for s in args.stocks.split(',')]
    elif args.stock_list:
        stocks = STOCK_LISTS[args.stock_list]
    else:
        stocks = DEFAULT_STOCKS
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run experiment
    detailed_df, summary_df = run_multi_stock_experiment(
        stocks=stocks,
        interval=args.interval,
        verbose=True
    )
    
    if detailed_df is None:
        print("\n❌ Experiment failed - no results to save")
        return
    
    # Display results
    display_results(detailed_df, summary_df)
    
    # Save results
    save_results(detailed_df, summary_df, args.output_dir)
    
    # Create visualization
    plot_path = os.path.join(args.output_dir, 'multi_stock_results.png')
    create_visualization(detailed_df, summary_df, save_path=plot_path)
    
    print("\n" + "="*70)
    print("🎉 MULTI-STOCK EXPERIMENT COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
