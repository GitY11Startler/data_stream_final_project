"""
River-compatible wrapper for KAF algorithms.
"""
from river import base
from typing import Union, Dict
import numpy as np

from ..algorithms import KLMS, KNLMS, KAPA, KRLS


class KAFRegressor(base.Regressor):
    """
    River-compatible wrapper for Kernel Adaptive Filtering regressors.
    
    This class wraps KAF algorithms to make them compatible with River's
    streaming interface, allowing for easy integration with River's ecosystem.
    
    Example:
        >>> from river import stream, metrics
        >>> from src.stream.river_wrapper import KAFRegressor
        >>> 
        >>> model = KAFRegressor(algorithm='KLMS')
        >>> metric = metrics.MAE()
        >>> 
        >>> for x, y in stream.iter_csv('data.csv'):
        ...     y_pred = model.predict_one(x)
        ...     model.learn_one(x, y)
        ...     metric.update(y, y_pred)
    """
    
    def __init__(
        self,
        algorithm: str = 'KLMS',
        learning_rate: float = 0.1,
        kernel: str = 'gaussian',
        kernel_size: float = 1.0,
        max_dictionary_size: int = 200,
        novelty_threshold: float = 0.1,
        **kwargs
    ):
        """
        Initialize KAF regressor.
        
        Args:
            algorithm: Algorithm type ('KLMS', 'KNLMS', 'KAPA', 'KRLS')
            learning_rate: Learning rate/step size
            kernel: Kernel type ('gaussian', 'polynomial', 'linear')
            kernel_size: Kernel bandwidth parameter
            max_dictionary_size: Maximum number of dictionary elements
            novelty_threshold: Threshold for adding new dictionary elements
            **kwargs: Additional algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.max_dictionary_size = max_dictionary_size
        self.novelty_threshold = novelty_threshold
        self.kwargs = kwargs
        
        # Initialize the appropriate algorithm
        if algorithm == 'KLMS':
            self.model = KLMS(
                learning_rate=learning_rate,
                kernel=kernel,
                kernel_size=kernel_size,
                max_dictionary_size=max_dictionary_size,
                novelty_threshold=novelty_threshold,
                **kwargs
            )
        elif algorithm == 'KNLMS':
            self.model = KNLMS(
                learning_rate=learning_rate,
                kernel=kernel,
                kernel_size=kernel_size,
                max_dictionary_size=max_dictionary_size,
                novelty_threshold=novelty_threshold,
                **kwargs
            )
        elif algorithm == 'KAPA':
            self.model = KAPA(
                learning_rate=learning_rate,
                kernel=kernel,
                kernel_size=kernel_size,
                max_dictionary_size=max_dictionary_size,
                novelty_threshold=novelty_threshold,
                **kwargs
            )
        elif algorithm == 'KRLS':
            self.model = KRLS(
                learning_rate=learning_rate,
                kernel=kernel,
                kernel_size=kernel_size,
                max_dictionary_size=max_dictionary_size,
                novelty_threshold=novelty_threshold,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.feature_names = None
    
    def learn_one(self, x: dict, y: Union[float, int]) -> 'KAFRegressor':
        """
        Update the model with a single sample.
        
        Args:
            x: Feature dictionary
            y: Target value
            
        Returns:
            self
        """
        # Store feature names on first call
        if self.feature_names is None:
            self.feature_names = sorted(x.keys())
        
        # Convert dictionary to array using consistent ordering
        x_array = np.array([x[feat] for feat in self.feature_names])
        
        # Update model
        self.model.learn_one(x_array, float(y))
        
        return self
    
    def predict_one(self, x: dict) -> float:
        """
        Predict the target value for a single sample.
        
        Args:
            x: Feature dictionary
            
        Returns:
            Predicted value
        """
        # Handle case where model hasn't been trained yet
        if self.feature_names is None:
            self.feature_names = sorted(x.keys())
            return 0.0
        
        # Convert dictionary to array
        x_array = np.array([x.get(feat, 0.0) for feat in self.feature_names])
        
        return self.model.predict_one(x_array)
    
    @property
    def _multiclass(self):
        return False


class KAFClassifier(base.Classifier):
    """
    River-compatible wrapper for KAF-based classification.
    
    This uses KAF for regression on binary/multiclass targets,
    converting the problem to regression and applying thresholding.
    """
    
    def __init__(
        self,
        algorithm: str = 'KLMS',
        learning_rate: float = 0.1,
        kernel: str = 'gaussian',
        kernel_size: float = 1.0,
        max_dictionary_size: int = 200,
        novelty_threshold: float = 0.1,
        **kwargs
    ):
        """Initialize KAF classifier."""
        self.regressor = KAFRegressor(
            algorithm=algorithm,
            learning_rate=learning_rate,
            kernel=kernel,
            kernel_size=kernel_size,
            max_dictionary_size=max_dictionary_size,
            novelty_threshold=novelty_threshold,
            **kwargs
        )
        self.classes = set()
    
    def learn_one(self, x: dict, y: Union[bool, int]) -> 'KAFClassifier':
        """
        Update the classifier with a single sample.
        
        Args:
            x: Feature dictionary
            y: Class label
            
        Returns:
            self
        """
        self.classes.add(y)
        
        # Convert to numeric if boolean
        if isinstance(y, bool):
            y_numeric = 1.0 if y else -1.0
        else:
            y_numeric = float(y)
        
        self.regressor.learn_one(x, y_numeric)
        return self
    
    def predict_proba_one(self, x: dict) -> Dict[Union[bool, int], float]:
        """
        Predict class probabilities for a single sample.
        
        Args:
            x: Feature dictionary
            
        Returns:
            Dictionary of class probabilities
        """
        if not self.classes:
            return {}
        
        # Get regression prediction
        pred = self.regressor.predict_one(x)
        
        # Convert to probabilities (simple sigmoid for binary)
        if len(self.classes) == 2:
            classes = sorted(self.classes)
            prob_positive = 1.0 / (1.0 + np.exp(-pred))
            return {
                classes[0]: 1.0 - prob_positive,
                classes[1]: prob_positive
            }
        else:
            # For multiclass, use simple thresholding
            # This is a simplification; proper multiclass would need one-vs-all
            classes = sorted(self.classes)
            probs = {c: 0.0 for c in classes}
            
            # Find closest class
            min_dist = float('inf')
            closest_class = classes[0]
            for c in classes:
                dist = abs(pred - float(c))
                if dist < min_dist:
                    min_dist = dist
                    closest_class = c
            
            probs[closest_class] = 1.0
            return probs
    
    def predict_one(self, x: dict) -> Union[bool, int]:
        """
        Predict the class for a single sample.
        
        Args:
            x: Feature dictionary
            
        Returns:
            Predicted class
        """
        proba = self.predict_proba_one(x)
        if not proba:
            return list(self.classes)[0] if self.classes else 0
        
        return max(proba.items(), key=lambda item: item[1])[0]
    
    @property
    def _multiclass(self):
        return len(self.classes) > 2
