"""
Visualization Module for KAF vs CapyMOA Experiments
====================================================

This module provides plotting utilities for:
1. Comparison bar charts - Compare algorithm metrics side-by-side
2. Time series plots - Show predictions vs actuals over time
3. Error distribution plots - Analyze prediction errors

All plots use consistent styling with:
- Blue tones for KAF algorithms
- Orange tones for CapyMOA algorithms

Usage:
    from src.evaluation.plots import (
        plot_metric_comparison,
        plot_predictions_timeseries,
        plot_error_distribution,
        create_experiment_report
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import os


# Style constants
KAF_ALGORITHMS = ['KLMS', 'KNLMS', 'KAPA', 'KRLS']
CAPYMOA_ALGORITHMS = ['ARF', 'KNN', 'SGBR']

# Color palettes
KAF_COLORS = {
    'KLMS': '#2E86AB',
    'KNLMS': '#1E5F74', 
    'KAPA': '#145369',
    'KRLS': '#0D3B4F',
}

CAPYMOA_COLORS = {
    'ARF': '#F77F00',
    'KNN': '#FCBF49',
    'SGBR': '#EAE2B7',
}

ALL_COLORS = {**KAF_COLORS, **CAPYMOA_COLORS}

# Default figure settings
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def get_algorithm_color(algorithm: str) -> str:
    """Get color for an algorithm based on its type."""
    return ALL_COLORS.get(algorithm, '#808080')


def get_algorithm_type(algorithm: str) -> str:
    """Get algorithm type (KAF or CapyMOA)."""
    if algorithm in KAF_ALGORITHMS:
        return 'KAF'
    elif algorithm in CAPYMOA_ALGORITHMS:
        return 'CapyMOA'
    return 'Other'


# =============================================================================
# 1. COMPARISON BAR CHARTS
# =============================================================================

def plot_metric_comparison(
    results_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    title: str = 'Algorithm Comparison',
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create bar charts comparing algorithms across multiple metrics.
    
    Args:
        results_df: DataFrame with columns [algorithm, MAE, RMSE, R2, directional_accuracy, ...]
        metrics: List of metrics to plot. Default: ['MAE', 'RMSE', 'R2', 'directional_accuracy']
        title: Main title for the figure
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    if metrics is None:
        metrics = ['MAE', 'RMSE', 'R2', 'directional_accuracy']
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    n_metrics = len(available_metrics)
    
    if n_metrics == 0:
        raise ValueError(f"No metrics found in DataFrame. Available: {results_df.columns.tolist()}")
    
    # Create subplots
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    # Sort by first metric
    df_sorted = results_df.sort_values(available_metrics[0])
    algorithms = df_sorted['algorithm'].tolist()
    colors = [get_algorithm_color(algo) for algo in algorithms]
    
    # Metric display settings
    metric_settings = {
        'MAE': {'label': 'MAE (Lower is Better)', 'invert': True},
        'RMSE': {'label': 'RMSE (Lower is Better)', 'invert': True},
        'R2': {'label': 'R² Score (Higher is Better)', 'invert': False},
        'directional_accuracy': {'label': 'Directional Accuracy %', 'invert': False, 'multiply': 100},
        'time_seconds': {'label': 'Execution Time (s)', 'invert': True},
    }
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        settings = metric_settings.get(metric, {'label': metric, 'invert': False})
        
        values = df_sorted[metric].values.copy()
        if settings.get('multiply'):
            values = values * settings['multiply']
        
        bars = ax.barh(algorithms, values, color=colors)
        ax.set_xlabel(settings['label'], fontweight='bold')
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}' if val < 100 else f'{val:.1f}',
                   ha='left', va='center', fontsize=8, color='black')
        
        # Add reference line for directional accuracy
        if metric == 'directional_accuracy':
            ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
            ax.legend(loc='lower right', fontsize=8)
        
        # Invert x-axis for "lower is better" metrics
        if settings.get('invert'):
            ax.invert_xaxis()
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#2E86AB', label='KAF Algorithms'),
        mpatches.Patch(facecolor='#F77F00', label='CapyMOA Algorithms')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    return fig


def plot_grouped_bar_comparison(
    results_df: pd.DataFrame,
    metric: str = 'MAE',
    group_by: str = 'interval',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create grouped bar chart comparing algorithms across categories (e.g., intervals).
    
    Args:
        results_df: DataFrame with algorithm results
        metric: Metric to plot
        group_by: Column to group by (e.g., 'interval', 'symbol')
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    if group_by not in results_df.columns:
        raise ValueError(f"Column '{group_by}' not found in DataFrame")
    
    groups = results_df[group_by].unique()
    algorithms = results_df['algorithm'].unique()
    
    x = np.arange(len(groups))
    width = 0.8 / len(algorithms)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, algo in enumerate(algorithms):
        algo_data = results_df[results_df['algorithm'] == algo]
        values = [algo_data[algo_data[group_by] == g][metric].values[0] 
                  if len(algo_data[algo_data[group_by] == g]) > 0 else 0 
                  for g in groups]
        
        offset = (i - len(algorithms)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=algo, 
                     color=get_algorithm_color(algo))
    
    ax.set_xlabel(group_by.replace('_', ' ').title())
    ax.set_ylabel(metric)
    ax.set_title(title or f'{metric} by {group_by.title()}')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    return fig


# =============================================================================
# 2. TIME SERIES PLOTS
# =============================================================================

def plot_predictions_timeseries(
    predictions: Dict[str, List[float]],
    actuals: List[float],
    timestamps: Optional[List] = None,
    title: str = 'Predictions vs Actuals',
    figsize: Tuple[int, int] = (14, 6),
    max_points: int = 200,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot predictions from multiple algorithms against actual values over time.
    
    Args:
        predictions: Dict of {algorithm_name: [predictions]}
        actuals: List of actual values
        timestamps: Optional list of timestamps/indices
        title: Plot title
        figsize: Figure size
        max_points: Maximum points to plot (for readability)
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    n_points = len(actuals)
    
    # Subsample if too many points
    if n_points > max_points:
        step = n_points // max_points
        indices = list(range(0, n_points, step))
        actuals = [actuals[i] for i in indices]
        predictions = {k: [v[i] for i in indices] for k, v in predictions.items()}
        if timestamps:
            timestamps = [timestamps[i] for i in indices]
    
    if timestamps is None:
        timestamps = list(range(len(actuals)))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actuals
    ax.plot(timestamps, actuals, 'k-', linewidth=2, label='Actual', alpha=0.8)
    
    # Plot predictions for each algorithm
    for algo, preds in predictions.items():
        color = get_algorithm_color(algo)
        ax.plot(timestamps, preds, '--', linewidth=1.5, label=algo, 
               color=color, alpha=0.7)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if timestamps are dates
    if timestamps and isinstance(timestamps[0], (datetime, pd.Timestamp)):
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    return fig


def plot_prediction_errors_timeseries(
    errors: Dict[str, List[float]],
    timestamps: Optional[List] = None,
    title: str = 'Prediction Errors Over Time',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot prediction errors over time for multiple algorithms.
    
    Args:
        errors: Dict of {algorithm_name: [errors]}
        timestamps: Optional list of timestamps
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    if not errors:
        raise ValueError("No error data provided")
    
    first_algo = list(errors.keys())[0]
    n_points = len(errors[first_algo])
    
    if timestamps is None:
        timestamps = list(range(n_points))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot errors for each algorithm
    for algo, err in errors.items():
        color = get_algorithm_color(algo)
        ax.plot(timestamps, err, '-', linewidth=1, label=algo, 
               color=color, alpha=0.7)
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Error (Actual - Predicted)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    return fig


def plot_cumulative_error(
    errors: Dict[str, List[float]],
    title: str = 'Cumulative Absolute Error',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cumulative absolute error over time.
    
    Args:
        errors: Dict of {algorithm_name: [errors]}
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for algo, err in errors.items():
        cumulative = np.cumsum(np.abs(err))
        color = get_algorithm_color(algo)
        ax.plot(cumulative, '-', linewidth=2, label=algo, color=color)
    
    ax.set_xlabel('Sample')
    ax.set_ylabel('Cumulative Absolute Error')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    return fig


# =============================================================================
# 3. ERROR DISTRIBUTION PLOTS
# =============================================================================

def plot_error_distribution(
    errors: Dict[str, List[float]],
    title: str = 'Error Distribution',
    figsize: Tuple[int, int] = (14, 5),
    bins: int = 30,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot error distribution histograms for multiple algorithms.
    
    Args:
        errors: Dict of {algorithm_name: [errors]}
        title: Plot title
        figsize: Figure size
        bins: Number of histogram bins
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    n_algos = len(errors)
    
    fig, axes = plt.subplots(1, n_algos, figsize=figsize, sharey=True)
    if n_algos == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for idx, (algo, err) in enumerate(errors.items()):
        ax = axes[idx]
        color = get_algorithm_color(algo)
        
        ax.hist(err, bins=bins, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=np.mean(err), color='green', linestyle='-', linewidth=1.5, 
                  alpha=0.7, label=f'Mean: {np.mean(err):.3f}')
        
        ax.set_xlabel('Error')
        ax.set_title(f'{algo}\nStd: {np.std(err):.3f}')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    axes[0].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    return fig


def plot_error_boxplot(
    errors: Dict[str, List[float]],
    title: str = 'Error Distribution Comparison',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create box plot comparing error distributions across algorithms.
    
    Args:
        errors: Dict of {algorithm_name: [errors]}
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    algorithms = list(errors.keys())
    data = [errors[algo] for algo in algorithms]
    colors = [get_algorithm_color(algo) for algo in algorithms]
    
    # Create box plot
    bp = ax.boxplot(data, labels=algorithms, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_ylabel('Error (Actual - Predicted)')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#2E86AB', label='KAF', alpha=0.7),
        mpatches.Patch(facecolor='#F77F00', label='CapyMOA', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    return fig


def plot_qq_errors(
    errors: Dict[str, List[float]],
    title: str = 'Q-Q Plot of Errors',
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create Q-Q plots to assess normality of errors.
    
    Args:
        errors: Dict of {algorithm_name: [errors]}
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    from scipy import stats
    
    n_algos = len(errors)
    fig, axes = plt.subplots(1, n_algos, figsize=figsize)
    if n_algos == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for idx, (algo, err) in enumerate(errors.items()):
        ax = axes[idx]
        color = get_algorithm_color(algo)
        
        # Q-Q plot
        stats.probplot(err, dist="norm", plot=ax)
        ax.get_lines()[0].set_color(color)
        ax.get_lines()[0].set_markersize(4)
        ax.set_title(algo)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    return fig


# =============================================================================
# 4. COMPREHENSIVE REPORT GENERATION
# =============================================================================

def create_experiment_report(
    results_df: pd.DataFrame,
    predictions: Optional[Dict[str, List[float]]] = None,
    actuals: Optional[List[float]] = None,
    errors: Optional[Dict[str, List[float]]] = None,
    title: str = 'Experiment Report',
    output_dir: str = '../results',
    prefix: str = 'report'
) -> List[str]:
    """
    Generate a comprehensive set of plots for an experiment.
    
    Args:
        results_df: DataFrame with algorithm comparison results
        predictions: Optional dict of predictions per algorithm
        actuals: Optional list of actual values
        errors: Optional dict of errors per algorithm
        title: Report title
        output_dir: Directory to save plots
        prefix: Filename prefix
        
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    print(f"\n{'='*70}")
    print(f"GENERATING EXPERIMENT REPORT: {title}")
    print(f"{'='*70}")
    
    # 1. Metric comparison bar charts
    try:
        path = os.path.join(output_dir, f'{prefix}_metrics_{timestamp}.png')
        plot_metric_comparison(results_df, title=f'{title} - Metrics', save_path=path)
        saved_files.append(path)
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not create metric comparison: {e}")
    
    # 2. Time series (if data provided)
    if predictions and actuals:
        try:
            path = os.path.join(output_dir, f'{prefix}_timeseries_{timestamp}.png')
            plot_predictions_timeseries(predictions, actuals, 
                                       title=f'{title} - Predictions', save_path=path)
            saved_files.append(path)
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not create time series: {e}")
    
    # 3. Error distribution (if data provided)
    if errors:
        try:
            # Histogram
            path = os.path.join(output_dir, f'{prefix}_error_hist_{timestamp}.png')
            plot_error_distribution(errors, title=f'{title} - Error Distribution', 
                                   save_path=path)
            saved_files.append(path)
            plt.close()
            
            # Box plot
            path = os.path.join(output_dir, f'{prefix}_error_box_{timestamp}.png')
            plot_error_boxplot(errors, title=f'{title} - Error Comparison', 
                              save_path=path)
            saved_files.append(path)
            plt.close()
            
            # Cumulative error
            path = os.path.join(output_dir, f'{prefix}_cumulative_{timestamp}.png')
            plot_cumulative_error(errors, title=f'{title} - Cumulative Error', 
                                 save_path=path)
            saved_files.append(path)
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Could not create error plots: {e}")
    
    print(f"\n📊 Generated {len(saved_files)} plots in {output_dir}")
    
    return saved_files


# =============================================================================
# 5. UTILITY FUNCTIONS
# =============================================================================

def save_figure(fig: plt.Figure, path: str, formats: List[str] = ['png']) -> List[str]:
    """
    Save figure in multiple formats.
    
    Args:
        fig: matplotlib Figure
        path: Base path (without extension)
        formats: List of formats to save ('png', 'pdf', 'svg')
        
    Returns:
        List of saved file paths
    """
    saved = []
    base_path = os.path.splitext(path)[0]
    
    for fmt in formats:
        file_path = f"{base_path}.{fmt}"
        fig.savefig(file_path, dpi=300, bbox_inches='tight', format=fmt)
        saved.append(file_path)
        print(f"✅ Saved: {file_path}")
    
    return saved


def set_plot_style(style: str = 'default'):
    """
    Set global plot style.
    
    Args:
        style: Style name ('default', 'dark', 'presentation')
    """
    if style == 'dark':
        plt.style.use('dark_background')
    elif style == 'presentation':
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['figure.dpi'] = 150
    else:
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == '__main__':
    # Quick test with sample data
    print("Testing plots module...")
    
    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample results DataFrame
    test_results = pd.DataFrame({
        'algorithm': ['KLMS', 'KRLS', 'ARF', 'KNN'],
        'MAE': [0.25, 0.21, 0.36, 0.43],
        'RMSE': [0.34, 0.29, 0.50, 0.58],
        'R2': [0.85, 0.89, 0.56, 0.40],
        'directional_accuracy': [0.54, 0.53, 0.46, 0.46],
        'time_seconds': [0.03, 0.04, 0.07, 0.01]
    })
    
    # Test metric comparison
    fig = plot_metric_comparison(test_results, title='Test Comparison')
    plt.savefig(os.path.join(output_dir, 'test_plots_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Metric comparison plot created")
    
    # Sample errors
    np.random.seed(42)
    test_errors = {
        'KLMS': np.random.randn(100) * 0.25,
        'KRLS': np.random.randn(100) * 0.21,
        'ARF': np.random.randn(100) * 0.36,
    }
    
    # Test error distribution
    fig = plot_error_distribution(test_errors, title='Test Error Distribution')
    plt.savefig(os.path.join(output_dir, 'test_plots_errors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Error distribution plot created")
    
    # Test box plot
    fig = plot_error_boxplot(test_errors, title='Test Error Boxplot')
    plt.savefig(os.path.join(output_dir, 'test_plots_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Error boxplot created")
    
    print(f"\n🎉 All plot tests passed! Plots saved to {output_dir}")
