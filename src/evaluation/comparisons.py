"""
Algorithm comparison utilities for streaming regression.

This module provides functions to compare multiple algorithms on the same
streaming data with consistent evaluation and result formatting.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from river import metrics
import time
from .metrics import (
    prequential_evaluation,
    evaluate_directional_accuracy_online,
    OnlineEvaluator
)


def compare_algorithms(
    algorithms: Dict[str, Any],
    stream_data: List[Tuple[Dict, float]],
    metrics_list: Optional[List] = None,
    warm_start: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare multiple algorithms on the same streaming data.
    
    This function runs prequential evaluation for each algorithm and
    returns results in a consistent format for easy comparison.
    
    Args:
        algorithms: Dictionary of {name: model} pairs
        stream_data: List of (x, y) tuples (will be copied for each algorithm)
        metrics_list: List of River metrics to track
        warm_start: Number of initial samples for training only
        verbose: Whether to print progress
        
    Returns:
        DataFrame with comparison results (one row per algorithm)
        
    Example:
        >>> algorithms = {
        ...     'KLMS': KAFRegressor(algorithm='KLMS'),
        ...     'ARF': CapyMOARegressor(algorithm='AdaptiveRandomForestRegressor')
        ... }
        >>> results_df = compare_algorithms(algorithms, stream_data)
    """
    if metrics_list is None:
        metrics_list = [metrics.MAE(), metrics.RMSE(), metrics.R2()]
    
    results = []
    
    for name, model in algorithms.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"Evaluating: {name}")
            print(f"{'='*70}")
        
        # Create fresh copy of stream data
        stream_copy = list(stream_data)
        
        # Track time
        start_time = time.time()
        
        # Run prequential evaluation
        try:
            metrics_dict, results_df = prequential_evaluation(
                model,
                stream_copy,
                metrics_list=metrics_list,
                verbose=verbose,
                warm_start=warm_start
            )
            
            elapsed_time = time.time() - start_time
            
            # Add algorithm name and timing
            metrics_dict['algorithm'] = name
            metrics_dict['time_seconds'] = elapsed_time
            metrics_dict['samples'] = len(stream_data) - warm_start
            
            # Calculate directional accuracy
            if verbose:
                print("\nCalculating directional accuracy...")
            
            stream_copy = list(stream_data)
            dir_acc, _ = evaluate_directional_accuracy_online(
                model,
                stream_copy,
                verbose=False
            )
            metrics_dict['directional_accuracy'] = dir_acc
            
            if verbose:
                print(f"Directional Accuracy: {dir_acc:.4f} ({dir_acc*100:.2f}%)")
                print(f"Total time: {elapsed_time:.2f}s")
            
            results.append(metrics_dict)
            
        except Exception as e:
            print(f"❌ Error evaluating {name}: {e}")
            # Add placeholder results
            results.append({
                'algorithm': name,
                'MAE': np.nan,
                'RMSE': np.nan,
                'R2': np.nan,
                'directional_accuracy': np.nan,
                'time_seconds': np.nan,
                'samples': len(stream_data) - warm_start,
                'error': str(e)
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = ['algorithm', 'MAE', 'RMSE', 'R2', 'directional_accuracy', 
                   'time_seconds', 'samples']
    
    # Add any extra columns at the end
    extra_cols = [col for col in df.columns if col not in column_order]
    column_order.extend(extra_cols)
    
    # Filter to only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    return df


def compare_with_baseline(
    test_algorithms: Dict[str, Any],
    baseline_algorithm: Any,
    baseline_name: str,
    stream_data: List[Tuple[Dict, float]],
    metrics_list: Optional[List] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare test algorithms against a baseline.
    
    Args:
        test_algorithms: Dictionary of test algorithms
        baseline_algorithm: Baseline model
        baseline_name: Name for baseline
        stream_data: Streaming data
        metrics_list: Metrics to track
        verbose: Print progress
        
    Returns:
        Tuple of (results_df, improvement_df)
        - results_df: Raw comparison results
        - improvement_df: Percentage improvement over baseline
    """
    # Combine all algorithms
    all_algorithms = {baseline_name: baseline_algorithm}
    all_algorithms.update(test_algorithms)
    
    # Run comparison
    results_df = compare_algorithms(
        all_algorithms,
        stream_data,
        metrics_list=metrics_list,
        verbose=verbose
    )
    
    # Calculate improvement over baseline
    baseline_row = results_df[results_df['algorithm'] == baseline_name].iloc[0]
    
    improvement_data = []
    for _, row in results_df.iterrows():
        if row['algorithm'] == baseline_name:
            continue
        
        improvement = {
            'algorithm': row['algorithm'],
            'MAE_improvement': (baseline_row['MAE'] - row['MAE']) / baseline_row['MAE'] * 100,
            'RMSE_improvement': (baseline_row['RMSE'] - row['RMSE']) / baseline_row['RMSE'] * 100,
            'R2_improvement': (row['R2'] - baseline_row['R2']) / abs(baseline_row['R2']) * 100 if baseline_row['R2'] != 0 else 0,
            'dir_acc_improvement': (row['directional_accuracy'] - baseline_row['directional_accuracy']) / baseline_row['directional_accuracy'] * 100,
        }
        improvement_data.append(improvement)
    
    improvement_df = pd.DataFrame(improvement_data)
    
    return results_df, improvement_df


def detailed_comparison(
    algorithms: Dict[str, Any],
    stream_data: List[Tuple[Dict, float]],
    window_size: int = 100,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Detailed comparison with windowed performance tracking.
    
    This tracks how each algorithm performs over time in sliding windows.
    
    Args:
        algorithms: Dictionary of algorithms
        stream_data: Streaming data
        window_size: Size of sliding window
        verbose: Print progress
        
    Returns:
        Dictionary mapping algorithm names to performance DataFrames
    """
    results = {}
    
    for name, model in algorithms.items():
        if verbose:
            print(f"\nEvaluating {name} with windowed analysis...")
        
        stream_copy = list(stream_data)
        
        # Track metrics in windows
        window_results = []
        current_window = []
        predictions = []
        actuals = []
        
        mae_metric = metrics.MAE()
        rmse_metric = metrics.RMSE()
        
        for i, (x, y) in enumerate(stream_copy):
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            
            current_window.append((y, y_pred))
            predictions.append(y_pred)
            actuals.append(y)
            
            mae_metric.update(y, y_pred)
            rmse_metric.update(y, y_pred)
            
            # Record window results
            if (i + 1) % window_size == 0:
                window_preds = [p for _, p in current_window]
                window_actuals = [a for a, _ in current_window]
                
                window_mae = np.mean(np.abs(np.array(window_actuals) - np.array(window_preds)))
                window_rmse = np.sqrt(np.mean((np.array(window_actuals) - np.array(window_preds)) ** 2))
                
                window_results.append({
                    'window': (i + 1) // window_size,
                    'sample_end': i + 1,
                    'window_mae': window_mae,
                    'window_rmse': window_rmse,
                    'cumulative_mae': mae_metric.get(),
                    'cumulative_rmse': rmse_metric.get()
                })
                
                current_window = []
        
        results[name] = pd.DataFrame(window_results)
    
    return results


def print_comparison_table(
    results_df: pd.DataFrame,
    title: str = "Algorithm Comparison Results"
):
    """
    Print a formatted comparison table.
    
    Args:
        results_df: DataFrame with comparison results
        title: Table title
    """
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)
    
    # Format numeric columns
    display_df = results_df.copy()
    
    numeric_cols = ['MAE', 'RMSE', 'R2', 'directional_accuracy', 'time_seconds']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    if 'samples' in display_df.columns:
        display_df['samples'] = display_df['samples'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
    
    print(display_df.to_string(index=False))
    print("="*80)


def export_comparison_results(
    results_df: pd.DataFrame,
    filename: str,
    improvement_df: Optional[pd.DataFrame] = None
):
    """
    Export comparison results to CSV.
    
    Args:
        results_df: Main results DataFrame
        filename: Output filename (without extension)
        improvement_df: Optional improvement DataFrame
    """
    # Save main results
    results_path = f"{filename}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")
    
    # Save improvement if provided
    if improvement_df is not None:
        improvement_path = f"{filename}_improvement.csv"
        improvement_df.to_csv(improvement_path, index=False)
        print(f"✅ Improvement saved to: {improvement_path}")


def rank_algorithms(
    results_df: pd.DataFrame,
    metric: str = 'MAE',
    ascending: bool = True
) -> pd.DataFrame:
    """
    Rank algorithms by a specific metric.
    
    Args:
        results_df: Results DataFrame
        metric: Metric to rank by
        ascending: True for lower is better (MAE, RMSE), False for higher is better (R2, accuracy)
        
    Returns:
        DataFrame sorted by metric with rank column
    """
    ranked_df = results_df.copy()
    ranked_df = ranked_df.sort_values(metric, ascending=ascending)
    ranked_df['rank'] = range(1, len(ranked_df) + 1)
    
    # Move rank to first column after algorithm
    cols = ['algorithm', 'rank'] + [col for col in ranked_df.columns if col not in ['algorithm', 'rank']]
    ranked_df = ranked_df[cols]
    
    return ranked_df


def multi_metric_ranking(
    results_df: pd.DataFrame,
    metrics_config: Optional[Dict[str, bool]] = None
) -> pd.DataFrame:
    """
    Rank algorithms across multiple metrics.
    
    Args:
        results_df: Results DataFrame
        metrics_config: Dictionary of {metric: ascending} pairs
                       Default: {'MAE': True, 'RMSE': True, 'directional_accuracy': False}
        
    Returns:
        DataFrame with average rank across all metrics
    """
    if metrics_config is None:
        metrics_config = {
            'MAE': True,
            'RMSE': True,
            'directional_accuracy': False
        }
    
    ranking_df = results_df[['algorithm']].copy()
    
    # Calculate rank for each metric
    for metric, ascending in metrics_config.items():
        if metric in results_df.columns:
            ranks = results_df[metric].rank(ascending=ascending, method='min')
            ranking_df[f'{metric}_rank'] = ranks
    
    # Calculate average rank
    rank_cols = [col for col in ranking_df.columns if col.endswith('_rank')]
    ranking_df['average_rank'] = ranking_df[rank_cols].mean(axis=1)
    
    # Sort by average rank
    ranking_df = ranking_df.sort_values('average_rank')
    ranking_df['overall_rank'] = range(1, len(ranking_df) + 1)
    
    return ranking_df


def get_best_algorithm(
    results_df: pd.DataFrame,
    metric: str = 'MAE',
    ascending: bool = True
) -> Tuple[str, float]:
    """
    Get the best performing algorithm for a specific metric.
    
    Args:
        results_df: Results DataFrame
        metric: Metric to evaluate
        ascending: True for lower is better, False for higher is better
        
    Returns:
        Tuple of (algorithm_name, metric_value)
    """
    if ascending:
        best_row = results_df.loc[results_df[metric].idxmin()]
    else:
        best_row = results_df.loc[results_df[metric].idxmax()]
    
    return best_row['algorithm'], best_row[metric]


def create_comparison_summary(
    results_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Create a comprehensive summary of comparison results.
    
    Args:
        results_df: Results DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_algorithms': len(results_df),
        'algorithms': results_df['algorithm'].tolist(),
        'best_mae': get_best_algorithm(results_df, 'MAE', ascending=True),
        'best_rmse': get_best_algorithm(results_df, 'RMSE', ascending=True),
        'best_r2': get_best_algorithm(results_df, 'R2', ascending=False),
        'best_directional_accuracy': get_best_algorithm(results_df, 'directional_accuracy', ascending=False),
        'fastest': get_best_algorithm(results_df, 'time_seconds', ascending=True),
        'mean_mae': results_df['MAE'].mean(),
        'std_mae': results_df['MAE'].std(),
        'mean_rmse': results_df['RMSE'].mean(),
        'std_rmse': results_df['RMSE'].std(),
    }
    
    return summary


def print_comparison_summary(summary: Dict[str, Any]):
    """
    Print comparison summary in a nice format.
    
    Args:
        summary: Summary dictionary from create_comparison_summary
    """
    print("\n" + "="*80)
    print("COMPARISON SUMMARY".center(80))
    print("="*80)
    
    print(f"\nAlgorithms compared: {summary['n_algorithms']}")
    print(f"Algorithms: {', '.join(summary['algorithms'])}")
    
    print(f"\n{'Best Performers:':<30}")
    print(f"  {'MAE:':<25} {summary['best_mae'][0]} ({summary['best_mae'][1]:.4f})")
    print(f"  {'RMSE:':<25} {summary['best_rmse'][0]} ({summary['best_rmse'][1]:.4f})")
    print(f"  {'R²:':<25} {summary['best_r2'][0]} ({summary['best_r2'][1]:.4f})")
    print(f"  {'Directional Accuracy:':<25} {summary['best_directional_accuracy'][0]} ({summary['best_directional_accuracy'][1]:.4f})")
    print(f"  {'Fastest:':<25} {summary['fastest'][0]} ({summary['fastest'][1]:.2f}s)")
    
    print(f"\n{'Overall Statistics:':<30}")
    print(f"  {'Mean MAE:':<25} {summary['mean_mae']:.4f} (±{summary['std_mae']:.4f})")
    print(f"  {'Mean RMSE:':<25} {summary['mean_rmse']:.4f} (±{summary['std_rmse']:.4f})")
    
    print("="*80)
