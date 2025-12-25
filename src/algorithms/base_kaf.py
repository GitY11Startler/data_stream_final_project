"""
Base class for Kernel Adaptive Filtering algorithms.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Callable


class KernelFunction:
    """Kernel function implementations."""
    
    @staticmethod
    def gaussian(x1: np.ndarray, x2: np.ndarray, sigma: float) -> float:
        """
        Gaussian (RBF) kernel.
        
        Args:
            x1: First input vector
            x2: Second input vector
            sigma: Kernel bandwidth parameter
            
        Returns:
            Kernel value
        """
        diff = x1 - x2
        return np.exp(-np.sum(diff ** 2) / (2 * sigma ** 2))
    
    @staticmethod
    def polynomial(x1: np.ndarray, x2: np.ndarray, degree: int = 3, 
                   coef0: float = 1.0) -> float:
        """
        Polynomial kernel.
        
        Args:
            x1: First input vector
            x2: Second input vector
            degree: Polynomial degree
            coef0: Independent term
            
        Returns:
            Kernel value
        """
        return (np.dot(x1, x2) + coef0) ** degree
    
    @staticmethod
    def linear(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Linear kernel.
        
        Args:
            x1: First input vector
            x2: Second input vector
            
        Returns:
            Kernel value
        """
        return np.dot(x1, x2)


class BaseKAF(ABC):
    """
    Base class for Kernel Adaptive Filtering algorithms.
    
    This implements the common functionality for online kernel-based
    learning algorithms suitable for streaming data.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        kernel: str = 'gaussian',
        kernel_size: float = 1.0,
        max_dictionary_size: int = 200,
        novelty_threshold: float = 0.1,
        **kwargs
    ):
        """
        Initialize the KAF algorithm.
        
        Args:
            learning_rate: Step size for weight updates
            kernel: Kernel function type ('gaussian', 'polynomial', 'linear')
            kernel_size: Kernel bandwidth (sigma for Gaussian)
            max_dictionary_size: Maximum size of the dictionary (memory)
            novelty_threshold: Threshold for adding new elements to dictionary
            **kwargs: Additional kernel parameters
        """
        self.learning_rate = learning_rate
        self.kernel_type = kernel
        self.kernel_size = kernel_size
        self.max_dictionary_size = max_dictionary_size
        self.novelty_threshold = novelty_threshold
        
        # Dictionary storage
        self.dictionary = []  # List of input vectors
        self.coefficients = []  # Corresponding weights
        
        # Performance tracking
        self.n_samples = 0
        self.error_history = []
        
        # Set up kernel function
        if kernel == 'gaussian':
            self.kernel_func = lambda x1, x2: KernelFunction.gaussian(
                x1, x2, self.kernel_size
            )
        elif kernel == 'polynomial':
            degree = kwargs.get('degree', 3)
            coef0 = kwargs.get('coef0', 1.0)
            self.kernel_func = lambda x1, x2: KernelFunction.polynomial(
                x1, x2, degree, coef0
            )
        elif kernel == 'linear':
            self.kernel_func = KernelFunction.linear
        else:
            raise ValueError(f"Unknown kernel type: {kernel}")
    
    def _compute_kernel_vector(self, x: np.ndarray) -> np.ndarray:
        """
        Compute kernel evaluations between input x and all dictionary elements.
        
        Args:
            x: Input vector
            
        Returns:
            Vector of kernel evaluations
        """
        if not self.dictionary:
            return np.array([])
        
        kernel_values = np.array([
            self.kernel_func(x, dict_elem) 
            for dict_elem in self.dictionary
        ])
        return kernel_values
    
    def _compute_novelty(self, x: np.ndarray) -> float:
        """
        Compute the novelty criterion for deciding whether to add x to dictionary.
        
        Args:
            x: Input vector
            
        Returns:
            Novelty measure (distance to dictionary)
        """
        if not self.dictionary:
            return float('inf')
        
        # Approximate linear independence (ALD) criterion
        kernel_vec = self._compute_kernel_vector(x)
        k_xx = self.kernel_func(x, x)
        
        if len(self.coefficients) > 0:
            projection = np.dot(kernel_vec, kernel_vec)
            novelty = k_xx - projection
        else:
            novelty = k_xx
            
        return novelty
    
    def _update_dictionary(self, x: np.ndarray, coefficient: float):
        """
        Update the dictionary with new element if necessary.
        
        Args:
            x: Input vector to potentially add
            coefficient: Initial coefficient for the new element
        """
        # Check if dictionary is full
        if len(self.dictionary) >= self.max_dictionary_size:
            # Simple strategy: remove oldest element
            self.dictionary.pop(0)
            self.coefficients.pop(0)
        
        # Add new element
        self.dictionary.append(x.copy())
        self.coefficients.append(coefficient)
    
    def predict(self, x: np.ndarray) -> float:
        """
        Make a prediction for input x.
        
        Args:
            x: Input vector (flattened)
            
        Returns:
            Predicted value
        """
        x = np.asarray(x).flatten()
        
        if not self.dictionary:
            return 0.0
        
        kernel_vec = self._compute_kernel_vector(x)
        coefficients = np.array(self.coefficients)
        
        prediction = np.dot(coefficients, kernel_vec)
        return prediction
    
    @abstractmethod
    def update(self, x: np.ndarray, y: float):
        """
        Update the model with a new sample.
        
        Args:
            x: Input vector
            y: Target value
        """
        pass
    
    def learn_one(self, x: np.ndarray, y: float) -> 'BaseKAF':
        """
        Learn from one sample (River-compatible interface).
        
        Args:
            x: Input features (dict or array)
            y: Target value
            
        Returns:
            self
        """
        # Convert dict to array if necessary
        if isinstance(x, dict):
            x = np.array(list(x.values()))
        else:
            x = np.asarray(x).flatten()
        
        self.update(x, y)
        self.n_samples += 1
        return self
    
    def predict_one(self, x) -> float:
        """
        Predict for one sample (River-compatible interface).
        
        Args:
            x: Input features (dict or array)
            
        Returns:
            Predicted value
        """
        # Convert dict to array if necessary
        if isinstance(x, dict):
            x = np.array(list(x.values()))
        
        return self.predict(x)
    
    def get_params(self) -> dict:
        """Get algorithm parameters."""
        return {
            'learning_rate': self.learning_rate,
            'kernel': self.kernel_type,
            'kernel_size': self.kernel_size,
            'max_dictionary_size': self.max_dictionary_size,
            'novelty_threshold': self.novelty_threshold,
            'dictionary_size': len(self.dictionary)
        }
    
    def reset(self):
        """Reset the model to initial state."""
        self.dictionary = []
        self.coefficients = []
        self.n_samples = 0
        self.error_history = []
