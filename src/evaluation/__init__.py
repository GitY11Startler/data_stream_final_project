"""
Evaluation module for assessing online learning algorithms.
"""

from .metrics import (
    OnlineEvaluator,
    prequential_evaluation,
    compare_algorithms,
    calculate_directional_accuracy,
    evaluate_directional_accuracy_online,
    sliding_window_evaluation
)

from .plots import (
    plot_metric_comparison,
    plot_grouped_bar_comparison,
    plot_predictions_timeseries,
    plot_prediction_errors_timeseries,
    plot_cumulative_error,
    plot_error_distribution,
    plot_error_boxplot,
    plot_qq_errors,
    create_experiment_report,
)

__all__ = [
    # Metrics
    'OnlineEvaluator',
    'prequential_evaluation',
    'compare_algorithms',
    'calculate_directional_accuracy',
    'evaluate_directional_accuracy_online',
    'sliding_window_evaluation',
    # Plots
    'plot_metric_comparison',
    'plot_grouped_bar_comparison',
    'plot_predictions_timeseries',
    'plot_prediction_errors_timeseries',
    'plot_cumulative_error',
    'plot_error_distribution',
    'plot_error_boxplot',
    'plot_qq_errors',
    'create_experiment_report',
]
