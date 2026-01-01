"""
CapyMOA Comparison Experiment
=============================

This script compares KAF algorithms (KLMS, KNLMS, KAPA, KRLS) against 
CapyMOA algorithms (AdaptiveRandomForestRegressor, KNNRegressor, 
StreamingGradientBoostedRegression) on stock price prediction tasks.

This is the KEY DELIVERABLE for the project - comparing implemented 
algorithms against those available in the CapyMOA library.

Usage:
    python capymoa_comparison.py --symbol AAPL --interval 1d --period 1y
    
Output:
    - results/capymoa_comparison.csv: Detailed comparison metrics
    - results/capymoa_comparison.png: Visualization of results
    - Console: Summary statistics and rankings
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.stream import KAFRegressor, CapyMOARegressor
from src.data.stock_data import load_stock_data, adjust_dates_for_interval
from src.evaluation.comparisons import (
    compare_algorithms, 
    rank_algorithms,
    multi_metric_ranking,
    export_comparison_results
)
from src.evaluation.metrics import prequential_evaluation


def setup_kaf_algorithms() -> Dict[str, KAFRegressor]:
    """
    Initialize all 4 KAF algorithms with default hyperparameters.
    
    Returns:
        Dictionary of algorithm name -> KAFRegressor instance
    """
    print("\n" + "="*70)
    print("INITIALIZING KAF ALGORITHMS")
    print("="*70)
    
    kaf_algorithms = {
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
        )
    }
    
    for name in kaf_algorithms:
        print(f"  ✅ {name} initialized")
    
    return kaf_algorithms


def setup_capymoa_algorithms() -> Dict[str, CapyMOARegressor]:
    """
    Initialize the 3 working CapyMOA algorithms.
    
    Returns:
        Dictionary of algorithm name -> CapyMOARegressor instance
    """
    print("\n" + "="*70)
    print("INITIALIZING CAPYMOA ALGORITHMS")
    print("="*70)
    
    capymoa_algorithms = {
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
    
    for name in capymoa_algorithms:
        print(f"  ✅ {name} initialized")
    
    return capymoa_algorithms


def prepare_stream_data(symbol: str, start_date: str, end_date: str, 
                       interval: str = '1d', use_cache: bool = True) -> Tuple[List, pd.DataFrame]:
    """
    Load and prepare stock data for streaming evaluation.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1h', '5m', etc.)
        use_cache: Whether to use cached data
        
    Returns:
        Tuple of (stream_data as list of (x_dict, y) tuples, original DataFrame)
    """
    print("\n" + "="*70)
    print(f"LOADING STOCK DATA: {symbol}")
    print("="*70)
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Interval: {interval}")
    print(f"  Cache: {'enabled' if use_cache else 'disabled'}")
    
    # Load data
    df = load_stock_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        use_cache=use_cache
    )
    
    # Handle multi-level columns if present
    if hasattr(df.columns, 'levels'):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    
    print(f"\n  ✅ Loaded {len(df)} samples")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Features: {list(df.columns)}")
    
    # Identify target column (Close price)
    target_col = None
    for col in df.columns:
        if 'Close' in str(col):
            target_col = col
            break
    
    if target_col is None:
        raise ValueError("Could not find 'Close' price column in data")
    
    # Normalize features for kernel-based methods
    # Use online normalization (compute mean/std incrementally)
    print("\n  Normalizing features for kernel methods...")
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Compute statistics for normalization
    feature_means = df[feature_cols].mean()
    feature_stds = df[feature_cols].std()
    feature_stds = feature_stds.replace(0, 1)  # Avoid division by zero
    
    # Also normalize target (for prediction comparison)
    target_mean = df[target_col].mean()
    target_std = df[target_col].std()
    
    print(f"  Feature means: {dict(feature_means.round(2))}")
    print(f"  Feature stds: {dict(feature_stds.round(2))}")
    print(f"  Target mean: {target_mean:.2f}, std: {target_std:.2f}")
    
    # Convert to stream format (list of (x_dict, y) tuples) with normalized features
    stream_data = []
    for idx, row in df.iterrows():
        # Split features (x) and target (y) - NORMALIZED
        y = (row[target_col] - target_mean) / target_std
        x_dict = {col: (row[col] - feature_means[col]) / feature_stds[col] 
                  for col in feature_cols}
        stream_data.append((x_dict, y))
    
    print(f"  ✅ Prepared {len(stream_data)} NORMALIZED samples for streaming")
    print(f"  Target: {target_col} (normalized)")
    print(f"  Features: {len(x_dict)} (normalized)")
    
    return stream_data, df, (target_mean, target_std)


def run_comparison(algorithms: Dict, stream_data: List, 
                  warm_start: int = 20) -> pd.DataFrame:
    """
    Run comparison of all algorithms on the stream data.
    
    Args:
        algorithms: Dictionary of algorithm name -> algorithm instance
        stream_data: List of (x_dict, y) tuples
        warm_start: Number of samples for warm-up
        
    Returns:
        DataFrame with comparison results
    """
    print("\n" + "="*70)
    print("RUNNING COMPARISON")
    print("="*70)
    print(f"  Algorithms: {len(algorithms)}")
    print(f"  Samples: {len(stream_data)}")
    print(f"  Warm-up: {warm_start} samples")
    
    # Run comparison
    results_df = compare_algorithms(
        algorithms=algorithms,
        stream_data=stream_data,
        warm_start=warm_start
    )
    
    print("\n  ✅ Comparison complete!")
    
    return results_df


def display_results(results_df: pd.DataFrame):
    """
    Display formatted results to console.
    
    Args:
        results_df: DataFrame with comparison results
    """
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    # Display full results
    print("\n" + results_df.to_string(index=False))
    
    # Rankings
    print("\n" + "="*70)
    print("RANKINGS BY MAE (Lower is Better)")
    print("="*70)
    mae_ranking = rank_algorithms(results_df, metric='MAE', ascending=True)
    print("\n" + mae_ranking.to_string(index=False))
    
    print("\n" + "="*70)
    print("RANKINGS BY DIRECTIONAL ACCURACY (Higher is Better)")
    print("="*70)
    da_ranking = rank_algorithms(results_df, metric='directional_accuracy', ascending=False)
    print("\n" + da_ranking.to_string(index=False))
    
    # Multi-metric ranking
    print("\n" + "="*70)
    print("MULTI-METRIC RANKING")
    print("="*70)
    multi_ranking = multi_metric_ranking(results_df)
    print("\n" + multi_ranking.to_string(index=False))
    
    # Best algorithm
    best_algo = multi_ranking.iloc[0]['algorithm']
    print(f"\n🏆 OVERALL BEST: {best_algo}")


def save_results(results_df: pd.DataFrame, output_dir: str = '../results'):
    """
    Save results to CSV file.
    
    Args:
        results_df: DataFrame with comparison results
        output_dir: Directory to save results
    """
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Create results directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, 'capymoa_comparison.csv')
    csv_filename_timestamped = os.path.join(output_dir, f'capymoa_comparison_{timestamp}.csv')
    
    # Save both versions
    results_df.to_csv(csv_filename, index=False)
    results_df.to_csv(csv_filename_timestamped, index=False)
    
    print(f"  ✅ Saved: {csv_filename}")
    print(f"  ✅ Saved: {csv_filename_timestamped}")
    
    return csv_filename


def plot_results(results_df: pd.DataFrame, symbol: str, 
                output_dir: str = '../results'):
    """
    Create visualization of comparison results.
    
    Args:
        results_df: DataFrame with comparison results
        symbol: Stock symbol for title
        output_dir: Directory to save plot
    """
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Algorithm Comparison - {symbol}', fontsize=16, fontweight='bold')
    
    # Sort by MAE for consistent ordering
    df_sorted = results_df.sort_values('MAE')
    
    # Color map: KAF algorithms in blue, CapyMOA in orange
    colors = []
    for algo in df_sorted['algorithm']:
        if algo in ['KLMS', 'KNLMS', 'KAPA', 'KRLS']:
            colors.append('#2E86AB')  # Blue for KAF
        else:
            colors.append('#F77F00')  # Orange for CapyMOA
    
    # Plot 1: MAE (lower is better)
    ax1 = axes[0, 0]
    ax1.barh(df_sorted['algorithm'], df_sorted['MAE'], color=colors)
    ax1.set_xlabel('MAE (Lower is Better)', fontweight='bold')
    ax1.set_title('Mean Absolute Error')
    ax1.invert_xaxis()  # Best (lowest) on right
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: RMSE (lower is better)
    ax2 = axes[0, 1]
    ax2.barh(df_sorted['algorithm'], df_sorted['RMSE'], color=colors)
    ax2.set_xlabel('RMSE (Lower is Better)', fontweight='bold')
    ax2.set_title('Root Mean Squared Error')
    ax2.invert_xaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Directional Accuracy (higher is better)
    df_sorted_da = results_df.sort_values('directional_accuracy', ascending=False)
    colors_da = []
    for algo in df_sorted_da['algorithm']:
        if algo in ['KLMS', 'KNLMS', 'KAPA', 'KRLS']:
            colors_da.append('#2E86AB')
        else:
            colors_da.append('#F77F00')
    
    ax3 = axes[1, 0]
    bars = ax3.barh(df_sorted_da['algorithm'], df_sorted_da['directional_accuracy'] * 100, color=colors_da)
    ax3.set_xlabel('Directional Accuracy % (Higher is Better)', fontweight='bold')
    ax3.set_title('Directional Accuracy')
    ax3.grid(axis='x', alpha=0.3)
    # Add 50% reference line (random guessing)
    ax3.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (50%)')
    ax3.legend()
    
    # Plot 4: Execution Time
    df_sorted_time = results_df.sort_values('time_seconds')
    colors_time = []
    for algo in df_sorted_time['algorithm']:
        if algo in ['KLMS', 'KNLMS', 'KAPA', 'KRLS']:
            colors_time.append('#2E86AB')
        else:
            colors_time.append('#F77F00')
    
    ax4 = axes[1, 1]
    ax4.barh(df_sorted_time['algorithm'], df_sorted_time['time_seconds'], color=colors_time)
    ax4.set_xlabel('Time (seconds)', fontweight='bold')
    ax4.set_title('Execution Time')
    ax4.grid(axis='x', alpha=0.3)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='KAF Algorithms'),
        Patch(facecolor='#F77F00', label='CapyMOA Algorithms')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, 'capymoa_comparison.png')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename_timestamped = os.path.join(output_dir, f'capymoa_comparison_{timestamp}.png')
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.savefig(plot_filename_timestamped, dpi=300, bbox_inches='tight')
    
    print(f"  ✅ Saved: {plot_filename}")
    print(f"  ✅ Saved: {plot_filename_timestamped}")
    
    plt.close()
    
    return plot_filename


def main():
    """Main execution function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Compare KAF and CapyMOA algorithms on stock prediction'
    )
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--interval', type=str, default='1d',
                       help='Data interval: 1d, 1h, 5m, etc. (default: 1d)')
    parser.add_argument('--period', type=str, default='1y',
                       help='Time period: 1y, 6mo, 3mo, etc. (default: 1y)')
    parser.add_argument('--warm-start', type=int, default=20,
                       help='Number of warm-up samples (default: 20)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable data caching')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    print("\n" + "="*70)
    print("CAPYMOA COMPARISON EXPERIMENT")
    print("="*70)
    print(f"  Stock: {args.symbol}")
    print(f"  Interval: {args.interval}")
    print(f"  Period: {args.period}")
    print(f"  Warm-up: {args.warm_start} samples")
    print(f"  Random seed: {args.seed}")
    print(f"  Cache: {'disabled' if args.no_cache else 'enabled'}")
    
    # Calculate dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    if args.period == '1y':
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    elif args.period == '6mo':
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    elif args.period == '3mo':
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    elif args.period == '1mo':
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    else:
        # Assume it's a number of days
        days = int(args.period.replace('d', ''))
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    try:
        # Step 1: Initialize algorithms
        kaf_algorithms = setup_kaf_algorithms()
        capymoa_algorithms = setup_capymoa_algorithms()
        
        # Combine all algorithms
        all_algorithms = {**kaf_algorithms, **capymoa_algorithms}
        
        # Step 2: Load data (now returns normalization stats too)
        stream_data, df, norm_stats = prepare_stream_data(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            interval=args.interval,
            use_cache=not args.no_cache
        )
        
        # Check if we have enough data
        if len(stream_data) < args.warm_start + 10:
            print(f"\n⚠️  WARNING: Only {len(stream_data)} samples available.")
            print(f"    Need at least {args.warm_start + 10} for meaningful comparison.")
            print("    Consider using a different period or interval.")
            return
        
        # Step 3: Run comparison
        results_df = run_comparison(
            algorithms=all_algorithms,
            stream_data=stream_data,
            warm_start=args.warm_start
        )
        
        # Step 4: Display results
        display_results(results_df)
        
        # Step 5: Save results
        csv_file = save_results(results_df)
        
        # Step 6: Generate plots
        plot_file = plot_results(results_df, symbol=args.symbol)
        
        # Final summary
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE! 🎉")
        print("="*70)
        print(f"\n📊 Results saved to:")
        print(f"   • CSV: {csv_file}")
        print(f"   • Plot: {plot_file}")
        print("\n📝 Next steps:")
        print("   1. Review the comparison results")
        print("   2. Check if metrics make sense")
        print("   3. Run with different stocks/intervals")
        print("   4. Include results in your report")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
