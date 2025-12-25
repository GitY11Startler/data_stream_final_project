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

__all__ = [
    'OnlineEvaluator',
    'prequential_evaluation',
    'compare_algorithms',
    'calculate_directional_accuracy',
    'evaluate_directional_accuracy_online',
    'sliding_window_evaluation'
]
