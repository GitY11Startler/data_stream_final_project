"""
Time Window Experiment
======================

This script compares algorithm performance across different time intervals
(1min, 5min, 1h, 1d) to reproduce experiments from the paper and understand
how prediction granularity affects accuracy.

Key Challenge: yfinance has different data availability limits per interval:
- 1m: max 7 days
- 5m: max 60 days  
- 1h: max 730 days
- 1d: unlimited

Usage:
    python time_window_experiment.py --symbol AAPL
    python time_window_experiment.py --symbol AAPL --intervals 1d,1h
    
Output:
    - results/time_window_results.csv: Comparison across intervals
    - results/time_window_results.png: Visualization
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.stream import KAFRegressor, CapyMOARegressor
from src.data.stock_data import load_stock_data, INTERVAL_LIMITS
from src.evaluation.comparisons import compare_algorithms
from src.evaluation.metrics import prequential_evaluation


# yfinance interval limitations
INTERVAL_CONFIG = {
    '1m': {
        'max_days': 7,
        'description': '1 minute',
        'samples_per_day': 390,  # ~6.5 hours of trading
    },
    '5m': {
        'max_days': 60,
        'description': '5 minutes',
        'samples_per_day': 78,
    },
    '15m': {
        'max_days': 60,
        'description': '15 minutes',
        'samples_per_day': 26,
    },
    '1h': {
        'max_days': 730,
        'description': '1 hour',
        'samples_per_day': 7,
    },
    '1d': {
        'max_days': 3650,  # ~10 years
        'description': '1 day',
        'samples_per_day': 1,
    }
}


def get_date_range_for_interval(interval: str, target_samples: int = 200) -> Tuple[str, str]:
    """
    Calculate appropriate date range for a given interval.
    
    Args:
        interval: Time interval (1m, 5m, 1h, 1d)
        target_samples: Desired number of samples
        
    Returns:
        Tuple of (start_date, end_date) strings
    """
    config = INTERVAL_CONFIG.get(interval, INTERVAL_CONFIG['1d'])
    
    # Calculate days needed for target samples
    samples_per_day = config['samples_per_day']
    days_needed = max(1, int(np.ceil(target_samples / samples_per_day)))
    
    # Respect yfinance limits
    max_days = config['max_days']
    days_to_use = min(days_needed, max_days)
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_use)
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def setup_algorithms() -> Dict:
    """
    Initialize algorithms for comparison.
    Using a subset for faster execution across multiple intervals.
    """
    algorithms = {
        # KAF algorithms (our implementations)
        'KLMS': KAFRegressor(
            algorithm='KLMS',
            learning_rate=0.1,
            kernel='gaussian',
            kernel_size=1.0,
            max_dictionary_size=100
        ),
        'KRLS': KAFRegressor(
            algorithm='KRLS',
            forgetting_factor=0.99,
            kernel='gaussian',
            kernel_size=1.0,
            max_dictionary_size=100
        ),
        # CapyMOA algorithms (library)
        'ARF': CapyMOARegressor(
            algorithm='AdaptiveRandomForestRegressor',
            ensemble_size=10
        ),
        'KNN': CapyMOARegressor(
            algorithm='KNNRegressor',
            k=5
        ),
    }
    return algorithms


def prepare_normalized_stream(symbol: str, start_date: str, end_date: str, 
                              interval: str) -> Tuple[List, pd.DataFrame, Tuple]:
    """
    Load and normalize stock data for streaming evaluation.
    
    Returns:
        Tuple of (stream_data, dataframe, (target_mean, target_std))
    """
    # Load data
    df = load_stock_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        use_cache=True
    )
    
    if df is None or len(df) == 0:
        return [], None, (0, 1)
    
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
        return [], df, (0, 1)
    
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


def run_interval_experiment(symbol: str, interval: str, 
                           warm_start: int = 20) -> Optional[pd.DataFrame]:
    """
    Run comparison for a specific interval.
    
    Returns:
        DataFrame with results or None if failed
    """
    config = INTERVAL_CONFIG.get(interval, INTERVAL_CONFIG['1d'])
    
    print(f"\n{'='*70}")
    print(f"INTERVAL: {interval} ({config['description']})")
    print(f"{'='*70}")
    
    # Get appropriate date range
    start_date, end_date = get_date_range_for_interval(interval)
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Max days allowed: {config['max_days']}")
    
    # Load data
    try:
        stream_data, df, norm_stats = prepare_normalized_stream(
            symbol, start_date, end_date, interval
        )
    except Exception as e:
        print(f"  ❌ Failed to load data: {e}")
        return None
    
    if len(stream_data) < warm_start + 10:
        print(f"  ⚠️ Insufficient data: {len(stream_data)} samples (need {warm_start + 10}+)")
        return None
    
    print(f"  ✅ Loaded {len(stream_data)} samples")
    
    # Initialize fresh algorithms for this interval
    algorithms = setup_algorithms()
    
    # Run comparison
    print(f"  Running comparison...")
    try:
        results_df = compare_algorithms(
            algorithms=algorithms,
            stream_data=stream_data,
            warm_start=warm_start,
            verbose=False
        )
        
        # Add interval column
        results_df['interval'] = interval
        results_df['interval_desc'] = config['description']
        results_df['num_samples'] = len(stream_data) - warm_start
        
        print(f"  ✅ Comparison complete!")
        
        # Print summary
        best_mae = results_df.loc[results_df['MAE'].idxmin()]
        best_da = results_df.loc[results_df['directional_accuracy'].idxmax()]
        print(f"  Best MAE: {best_mae['algorithm']} ({best_mae['MAE']:.4f})")
        print(f"  Best Dir.Acc: {best_da['algorithm']} ({best_da['directional_accuracy']*100:.1f}%)")
        
        return results_df
        
    except Exception as e:
        print(f"  ❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_interval_comparison(all_results: pd.DataFrame, symbol: str,
                            output_dir: str = '../results'):
    """
    Create visualization comparing performance across intervals.
    """
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")
    
    # Get unique intervals in order
    interval_order = ['1m', '5m', '15m', '1h', '1d']
    intervals = [i for i in interval_order if i in all_results['interval'].values]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Algorithm Performance Across Time Intervals - {symbol}', 
                 fontsize=14, fontweight='bold')
    
    # Color map for algorithms
    algo_colors = {
        'KLMS': '#2E86AB',
        'KRLS': '#1E5F74',
        'ARF': '#F77F00',
        'KNN': '#FCBF49',
    }
    
    algorithms = all_results['algorithm'].unique()
    
    # Plot 1: MAE by interval
    ax1 = axes[0, 0]
    for algo in algorithms:
        algo_data = all_results[all_results['algorithm'] == algo]
        algo_data = algo_data.set_index('interval').reindex(intervals)
        ax1.plot(intervals, algo_data['MAE'].values, 'o-', 
                label=algo, color=algo_colors.get(algo, 'gray'), linewidth=2, markersize=8)
    ax1.set_xlabel('Time Interval')
    ax1.set_ylabel('MAE (Normalized)')
    ax1.set_title('Mean Absolute Error by Interval')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R² by interval
    ax2 = axes[0, 1]
    for algo in algorithms:
        algo_data = all_results[all_results['algorithm'] == algo]
        algo_data = algo_data.set_index('interval').reindex(intervals)
        ax2.plot(intervals, algo_data['R2'].values, 'o-', 
                label=algo, color=algo_colors.get(algo, 'gray'), linewidth=2, markersize=8)
    ax2.set_xlabel('Time Interval')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score by Interval')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 3: Directional Accuracy by interval
    ax3 = axes[1, 0]
    for algo in algorithms:
        algo_data = all_results[all_results['algorithm'] == algo]
        algo_data = algo_data.set_index('interval').reindex(intervals)
        ax3.plot(intervals, algo_data['directional_accuracy'].values * 100, 'o-', 
                label=algo, color=algo_colors.get(algo, 'gray'), linewidth=2, markersize=8)
    ax3.set_xlabel('Time Interval')
    ax3.set_ylabel('Directional Accuracy (%)')
    ax3.set_title('Directional Accuracy by Interval')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    
    # Plot 4: Execution Time by interval
    ax4 = axes[1, 1]
    for algo in algorithms:
        algo_data = all_results[all_results['algorithm'] == algo]
        algo_data = algo_data.set_index('interval').reindex(intervals)
        ax4.plot(intervals, algo_data['time_seconds'].values, 'o-', 
                label=algo, color=algo_colors.get(algo, 'gray'), linewidth=2, markersize=8)
    ax4.set_xlabel('Time Interval')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Execution Time by Interval')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_file = os.path.join(output_dir, 'time_window_results.png')
    plot_file_ts = os.path.join(output_dir, f'time_window_results_{timestamp}.png')
    
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.savefig(plot_file_ts, dpi=300, bbox_inches='tight')
    
    print(f"  ✅ Saved: {plot_file}")
    plt.close()
    
    return plot_file


def create_summary_table(all_results: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table showing best algorithm per interval.
    """
    summary = []
    
    for interval in all_results['interval'].unique():
        interval_data = all_results[all_results['interval'] == interval]
        
        best_mae = interval_data.loc[interval_data['MAE'].idxmin()]
        best_r2 = interval_data.loc[interval_data['R2'].idxmax()]
        best_da = interval_data.loc[interval_data['directional_accuracy'].idxmax()]
        
        summary.append({
            'interval': interval,
            'samples': interval_data['num_samples'].iloc[0],
            'best_MAE_algo': best_mae['algorithm'],
            'best_MAE': best_mae['MAE'],
            'best_R2_algo': best_r2['algorithm'],
            'best_R2': best_r2['R2'],
            'best_DA_algo': best_da['algorithm'],
            'best_DA': best_da['directional_accuracy'],
        })
    
    return pd.DataFrame(summary)


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description='Compare algorithms across different time intervals'
    )
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--intervals', type=str, default='1d,1h,5m',
                       help='Comma-separated intervals to test (default: 1d,1h,5m)')
    parser.add_argument('--warm-start', type=int, default=20,
                       help='Number of warm-up samples (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Parse intervals
    intervals = [i.strip() for i in args.intervals.split(',')]
    
    print("="*70)
    print("TIME WINDOW EXPERIMENT")
    print("="*70)
    print(f"  Symbol: {args.symbol}")
    print(f"  Intervals: {intervals}")
    print(f"  Warm-up: {args.warm_start}")
    print(f"  Seed: {args.seed}")
    
    # Print interval limits
    print(f"\n  yfinance Interval Limits:")
    for interval in intervals:
        config = INTERVAL_CONFIG.get(interval, {})
        print(f"    {interval}: max {config.get('max_days', 'N/A')} days")
    
    # Run experiments for each interval
    all_results = []
    
    for interval in intervals:
        result = run_interval_experiment(
            symbol=args.symbol,
            interval=interval,
            warm_start=args.warm_start
        )
        
        if result is not None:
            all_results.append(result)
    
    if not all_results:
        print("\n❌ No successful experiments. Check data availability.")
        return
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Display results
    print(f"\n{'='*70}")
    print("COMBINED RESULTS")
    print(f"{'='*70}")
    print(combined_df.to_string(index=False))
    
    # Create summary
    print(f"\n{'='*70}")
    print("SUMMARY: BEST ALGORITHM PER INTERVAL")
    print(f"{'='*70}")
    summary_df = create_summary_table(combined_df)
    print(summary_df.to_string(index=False))
    
    # Save results
    output_dir = '../results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, 'time_window_results.csv')
    csv_file_ts = os.path.join(output_dir, f'time_window_results_{timestamp}.csv')
    
    combined_df.to_csv(csv_file, index=False)
    combined_df.to_csv(csv_file_ts, index=False)
    
    print(f"\n  ✅ Saved: {csv_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, 'time_window_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"  ✅ Saved: {summary_file}")
    
    # Generate plots
    plot_file = plot_interval_comparison(combined_df, args.symbol, output_dir)
    
    # Final summary
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE! 🎉")
    print(f"{'='*70}")
    print(f"\n📊 Results saved to:")
    print(f"   • CSV: {csv_file}")
    print(f"   • Summary: {summary_file}")
    print(f"   • Plot: {plot_file}")
    
    # Key findings
    print(f"\n📝 Key Findings:")
    for _, row in summary_df.iterrows():
        print(f"   • {row['interval']}: Best MAE={row['best_MAE_algo']}, "
              f"Best Dir.Acc={row['best_DA_algo']} ({row['best_DA']*100:.1f}%)")


if __name__ == '__main__':
    main()
