"""
CapyMOA-compatible wrapper for regression algorithms.

This module provides a unified interface for CapyMOA regressors,
making them compatible with River's streaming API (predict_one/learn_one).
"""
import numpy as np
from typing import Dict, Any, Optional
import warnings


class CapyMOARegressor:
    """
    Wrapper for CapyMOA regression algorithms to match River API.
    
    This class wraps CapyMOA regressors to provide the same interface
    as River models (predict_one, learn_one), allowing for seamless
    comparison and evaluation.
    
    Example:
        >>> from src.stream.capymoa_wrapper import CapyMOARegressor
        >>> model = CapyMOARegressor(algorithm='AdaptiveRandomForestRegressor')
        >>> 
        >>> for x, y in stream_data:
        ...     y_pred = model.predict_one(x)
        ...     model.learn_one(x, y)
    """
    
    SUPPORTED_ALGORITHMS = {
        'AdaptiveRandomForestRegressor': {
            'class': 'AdaptiveRandomForestRegressor',
            'default_params': {}
        },
        'KNNRegressor': {
            'class': 'KNNRegressor',
            'default_params': {'k': 10}
        },
        'SGDRegressor': {
            'class': 'SGDRegressor',
            'default_params': {}
        },
        'PassiveAggressiveRegressor': {
            'class': 'PassiveAggressiveRegressor',
            'default_params': {}
        },
        'StreamingGradientBoostedRegression': {
            'class': 'StreamingGradientBoostedRegression',
            'default_params': {}
        }
    }
    
    def __init__(
        self,
        algorithm: str = 'AdaptiveRandomForestRegressor',
        schema: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize CapyMOA regressor wrapper.
        
        Args:
            algorithm: Name of the CapyMOA algorithm
            schema: CapyMOA schema (auto-created if None)
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm_name = algorithm
        self.kwargs = kwargs
        self.model = None
        self.schema = schema
        self.n_samples = 0
        self.feature_names = None
        self.n_features = None
        
        # Track if model needs initialization
        self._initialized = False
        
        # Validate algorithm
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            available = ', '.join(self.SUPPORTED_ALGORITHMS.keys())
            raise ValueError(
                f"Algorithm '{algorithm}' not supported. "
                f"Available: {available}"
            )
        
        # Import CapyMOA
        try:
            import capymoa.regressor as regressor_module
            from capymoa.stream import Schema
            self.regressor_module = regressor_module
            self.Schema = Schema
        except ImportError as e:
            raise ImportError(
                f"Failed to import CapyMOA: {e}. "
                "Please ensure CapyMOA is installed: pip install capymoa"
            )
    
    def _create_schema(self):
        """Create CapyMOA schema from feature information."""
        if self.feature_names is None or self.n_features is None:
            raise ValueError("Feature information not available. Call learn_one first.")
        
        # Create schema with feature names + target
        # In CapyMOA, target must be in the features list
        all_attributes = self.feature_names + ['target']
        self.schema = self.Schema.from_custom(
            features=all_attributes,
            target='target'
        )
    
    def _initialize_model(self):
        """Initialize the CapyMOA model with schema."""
        if self._initialized:
            return
        
        if self.schema is None:
            self._create_schema()
        
        # Get algorithm class
        algo_info = self.SUPPORTED_ALGORITHMS[self.algorithm_name]
        algo_class = getattr(self.regressor_module, algo_info['class'])
        
        # Merge default params with user params
        params = algo_info['default_params'].copy()
        params.update(self.kwargs)
        
        # Initialize model
        try:
            self.model = algo_class(schema=self.schema, **params)
            self._initialized = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize {self.algorithm_name}: {e}"
            )
    
    def _dict_to_instance(self, x: Dict[str, float], y: Optional[float] = None):
        """
        Convert dictionary to CapyMOA instance.
        
        Args:
            x: Feature dictionary
            y: Target value (optional)
            
        Returns:
            CapyMOA Instance object
        """
        from capymoa.instance import Instance
        
        # Extract features in consistent order
        if self.feature_names is None:
            self.feature_names = sorted(x.keys())
            self.n_features = len(self.feature_names)
        
        # Convert to numpy array (features + target)
        x_values = [x[name] for name in self.feature_names]
        
        # CapyMOA Instance.from_array expects [features..., target]
        if y is not None:
            instance_array = np.array(x_values + [float(y)], dtype=np.float64)
        else:
            # For prediction only, use 0.0 as placeholder target
            instance_array = np.array(x_values + [0.0], dtype=np.float64)
        
        # Create instance
        instance = Instance.from_array(
            schema=self.schema,
            instance=instance_array
        )
        
        return instance
    
    def predict_one(self, x: Dict[str, float]) -> float:
        """
        Predict for a single instance.
        
        Args:
            x: Feature dictionary
            
        Returns:
            Predicted value
        """
        # If not initialized, return 0.0 (can't predict without training)
        if not self._initialized:
            warnings.warn(
                f"{self.algorithm_name} not initialized. Returning 0.0. "
                "Call learn_one first."
            )
            return 0.0
        
        try:
            # Convert to CapyMOA instance
            instance = self._dict_to_instance(x)
            
            # Get prediction
            prediction = self.model.predict(instance)
            
            return float(prediction)
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}. Returning 0.0")
            return 0.0
    
    def learn_one(self, x: Dict[str, float], y: float):
        """
        Update model with a single instance.
        
        Args:
            x: Feature dictionary
            y: Target value
        """
        # Extract feature names on first call
        if self.feature_names is None:
            self.feature_names = sorted(x.keys())
            self.n_features = len(self.feature_names)
        
        # Initialize model if needed
        if not self._initialized:
            self._initialize_model()
        
        try:
            # Convert to CapyMOA instance
            instance = self._dict_to_instance(x, y)
            
            # Train model
            self.model.train(instance)
            
            self.n_samples += 1
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'algorithm': self.algorithm_name,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'initialized': self._initialized
        }
    
    def __repr__(self) -> str:
        return f"CapyMOARegressor(algorithm='{self.algorithm_name}', n_samples={self.n_samples})"


def list_available_algorithms() -> list:
    """
    Get list of available CapyMOA algorithms.
    
    Returns:
        List of algorithm names
    """
    return list(CapyMOARegressor.SUPPORTED_ALGORITHMS.keys())


def test_algorithm(algorithm: str, n_samples: int = 50) -> bool:
    """
    Test if a CapyMOA algorithm can be loaded and used.
    
    Args:
        algorithm: Algorithm name
        n_samples: Number of test samples
        
    Returns:
        True if test passes
    """
    print(f"\nTesting {algorithm}...")
    
    try:
        # Create model
        model = CapyMOARegressor(algorithm=algorithm)
        
        # Generate simple test data
        np.random.seed(42)
        predictions = []
        
        for i in range(n_samples):
            x = {f'f{j}': np.random.randn() for j in range(5)}
            y = np.random.randn()
            
            # Predict and learn
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            
            predictions.append(y_pred)
        
        # Check model info
        info = model.get_info()
        
        print(f"  ✅ Success!")
        print(f"     Samples processed: {info['n_samples']}")
        print(f"     Features: {info['n_features']}")
        print(f"     Last prediction: {predictions[-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
