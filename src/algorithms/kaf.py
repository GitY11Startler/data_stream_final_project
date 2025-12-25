"""
Kernel Least Mean Square (KLMS) algorithm implementation.
"""
import numpy as np
from .base_kaf import BaseKAF


class KLMS(BaseKAF):
    """
    Kernel Least Mean Square (KLMS) algorithm.
    
    KLMS is one of the simplest kernel adaptive filtering algorithms,
    extending the classical LMS algorithm to nonlinear spaces using
    the kernel trick.
    
    Reference:
        W. Liu, P. P. Pokharel, and J. C. Principe, "The kernel least-mean-square 
        algorithm," IEEE Transactions on Signal Processing, 2008.
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
        Initialize KLMS.
        
        Args:
            learning_rate: Step size (eta)
            kernel: Kernel function type
            kernel_size: Kernel bandwidth
            max_dictionary_size: Maximum dictionary size
            novelty_threshold: Threshold for dictionary update
            **kwargs: Additional parameters
        """
        super().__init__(
            learning_rate=learning_rate,
            kernel=kernel,
            kernel_size=kernel_size,
            max_dictionary_size=max_dictionary_size,
            novelty_threshold=novelty_threshold,
            **kwargs
        )
        self.name = "KLMS"
    
    def update(self, x: np.ndarray, y: float):
        """
        Update KLMS with a new sample using the LMS update rule.
        
        The update rule is:
        coefficient = learning_rate * error
        
        Args:
            x: Input vector
            y: Target value
        """
        x = np.asarray(x).flatten()
        
        # Get prediction
        y_pred = self.predict(x)
        
        # Compute error
        error = y - y_pred
        self.error_history.append(error)
        
        # Compute novelty to decide if we should add to dictionary
        novelty = self._compute_novelty(x)
        
        if not self.dictionary or novelty > self.novelty_threshold:
            # Add new element to dictionary with LMS coefficient
            coefficient = self.learning_rate * error
            self._update_dictionary(x, coefficient)
        else:
            # Update existing coefficients
            kernel_vec = self._compute_kernel_vector(x)
            for i in range(len(self.coefficients)):
                self.coefficients[i] += self.learning_rate * error * kernel_vec[i]


class KNLMS(BaseKAF):
    """
    Kernel Normalized Least Mean Square (KNLMS) algorithm.
    
    KNLMS normalizes the update step by the input power, which can lead
    to faster convergence compared to KLMS.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        kernel: str = 'gaussian',
        kernel_size: float = 1.0,
        max_dictionary_size: int = 200,
        novelty_threshold: float = 0.1,
        epsilon: float = 1e-6,
        **kwargs
    ):
        """
        Initialize KNLMS.
        
        Args:
            learning_rate: Step size
            kernel: Kernel function type
            kernel_size: Kernel bandwidth
            max_dictionary_size: Maximum dictionary size
            novelty_threshold: Threshold for dictionary update
            epsilon: Regularization parameter to avoid division by zero
            **kwargs: Additional parameters
        """
        super().__init__(
            learning_rate=learning_rate,
            kernel=kernel,
            kernel_size=kernel_size,
            max_dictionary_size=max_dictionary_size,
            novelty_threshold=novelty_threshold,
            **kwargs
        )
        self.epsilon = epsilon
        self.name = "KNLMS"
    
    def update(self, x: np.ndarray, y: float):
        """
        Update KNLMS with normalized LMS rule.
        
        Args:
            x: Input vector
            y: Target value
        """
        x = np.asarray(x).flatten()
        
        # Get prediction
        y_pred = self.predict(x)
        
        # Compute error
        error = y - y_pred
        self.error_history.append(error)
        
        # Compute input power (kernel evaluation with itself)
        k_xx = self.kernel_func(x, x)
        
        # Normalized step size
        normalized_lr = self.learning_rate / (k_xx + self.epsilon)
        
        # Compute novelty
        novelty = self._compute_novelty(x)
        
        if not self.dictionary or novelty > self.novelty_threshold:
            # Add new element with normalized coefficient
            coefficient = normalized_lr * error
            self._update_dictionary(x, coefficient)
        else:
            # Update existing coefficients with normalization
            kernel_vec = self._compute_kernel_vector(x)
            for i in range(len(self.coefficients)):
                self.coefficients[i] += normalized_lr * error * kernel_vec[i]


class KAPA(BaseKAF):
    """
    Kernel Affine Projection Algorithm (KAPA).
    
    KAPA uses affine projection which considers multiple past samples
    for each update, potentially leading to faster convergence.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        kernel: str = 'gaussian',
        kernel_size: float = 1.0,
        max_dictionary_size: int = 200,
        novelty_threshold: float = 0.1,
        projection_order: int = 2,
        epsilon: float = 1e-6,
        **kwargs
    ):
        """
        Initialize KAPA.
        
        Args:
            learning_rate: Step size
            kernel: Kernel function type
            kernel_size: Kernel bandwidth
            max_dictionary_size: Maximum dictionary size
            novelty_threshold: Threshold for dictionary update
            projection_order: Number of past samples to use
            epsilon: Regularization parameter
            **kwargs: Additional parameters
        """
        super().__init__(
            learning_rate=learning_rate,
            kernel=kernel,
            kernel_size=kernel_size,
            max_dictionary_size=max_dictionary_size,
            novelty_threshold=novelty_threshold,
            **kwargs
        )
        self.projection_order = projection_order
        self.epsilon = epsilon
        self.name = "KAPA"
        
        # Store recent samples for affine projection
        self.recent_x = []
        self.recent_y = []
    
    def update(self, x: np.ndarray, y: float):
        """
        Update KAPA using affine projection.
        
        Args:
            x: Input vector
            y: Target value
        """
        x = np.asarray(x).flatten()
        
        # Get prediction
        y_pred = self.predict(x)
        
        # Compute error
        error = y - y_pred
        self.error_history.append(error)
        
        # Update recent samples buffer
        self.recent_x.append(x.copy())
        self.recent_y.append(y)
        
        if len(self.recent_x) > self.projection_order:
            self.recent_x.pop(0)
            self.recent_y.pop(0)
        
        # Compute novelty
        novelty = self._compute_novelty(x)
        
        if not self.dictionary or novelty > self.novelty_threshold:
            # Add new element
            coefficient = self.learning_rate * error
            self._update_dictionary(x, coefficient)
        else:
            # Affine projection update using recent samples
            if len(self.recent_x) > 0:
                # Build kernel matrix for recent samples
                K_recent = np.zeros((len(self.recent_x), len(self.dictionary)))
                for i, x_recent in enumerate(self.recent_x):
                    K_recent[i, :] = self._compute_kernel_vector(x_recent)
                
                # Compute predictions for recent samples
                predictions = K_recent @ np.array(self.coefficients)
                
                # Error vector
                error_vec = np.array(self.recent_y) - predictions
                
                # Regularized kernel matrix
                G = K_recent @ K_recent.T + self.epsilon * np.eye(len(self.recent_x))
                
                # Solve for coefficient update
                try:
                    coef_update = self.learning_rate * K_recent.T @ np.linalg.solve(G, error_vec)
                    
                    # Update coefficients
                    for i in range(len(self.coefficients)):
                        self.coefficients[i] += coef_update[i]
                except np.linalg.LinAlgError:
                    # Fallback to simple update if matrix is singular
                    kernel_vec = self._compute_kernel_vector(x)
                    for i in range(len(self.coefficients)):
                        self.coefficients[i] += self.learning_rate * error * kernel_vec[i]


class KRLS(BaseKAF):
    """
    Kernel Recursive Least Squares (KRLS) algorithm.
    
    KRLS is a recursive implementation of kernel-based least squares,
    providing exponentially weighted least squares solution.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        kernel: str = 'gaussian',
        kernel_size: float = 1.0,
        max_dictionary_size: int = 200,
        novelty_threshold: float = 0.1,
        forgetting_factor: float = 0.99,
        regularization: float = 0.01,
        **kwargs
    ):
        """
        Initialize KRLS.
        
        Args:
            learning_rate: Step size (not used in RLS but kept for consistency)
            kernel: Kernel function type
            kernel_size: Kernel bandwidth
            max_dictionary_size: Maximum dictionary size
            novelty_threshold: Threshold for dictionary update
            forgetting_factor: Exponential forgetting factor (lambda)
            regularization: Initial regularization parameter
            **kwargs: Additional parameters
        """
        super().__init__(
            learning_rate=learning_rate,
            kernel=kernel,
            kernel_size=kernel_size,
            max_dictionary_size=max_dictionary_size,
            novelty_threshold=novelty_threshold,
            **kwargs
        )
        self.forgetting_factor = forgetting_factor
        self.regularization = regularization
        self.name = "KRLS"
        
        # Inverse correlation matrix
        self.P = None
    
    def update(self, x: np.ndarray, y: float):
        """
        Update KRLS using recursive least squares.
        
        Args:
            x: Input vector
            y: Target value
        """
        x = np.asarray(x).flatten()
        
        # Get prediction
        y_pred = self.predict(x)
        
        # Compute error
        error = y - y_pred
        self.error_history.append(error)
        
        # Compute novelty
        novelty = self._compute_novelty(x)
        
        if not self.dictionary:
            # Initialize with first sample
            self._update_dictionary(x, y)
            k_xx = self.kernel_func(x, x)
            self.P = np.array([[1.0 / (k_xx + self.regularization)]])
        elif novelty > self.novelty_threshold and len(self.dictionary) < self.max_dictionary_size:
            # Add to dictionary and update P matrix
            n = len(self.dictionary)
            
            # Compute kernel vector
            k = self._compute_kernel_vector(x)
            k_xx = self.kernel_func(x, x)
            
            # Compute gain
            if self.P is not None and len(k) == n:
                Pk = self.P @ k
                denominator = self.forgetting_factor + k.T @ Pk
                
                # Update coefficient
                coefficient = error / denominator
                self._update_dictionary(x, coefficient)
                
                # Update P matrix - expand dimensions
                P_new = np.zeros((n + 1, n + 1))
                P_new[:n, :n] = (self.P - np.outer(Pk, Pk) / denominator) / self.forgetting_factor
                P_new[n, :n] = -Pk / denominator
                P_new[:n, n] = -Pk / denominator
                P_new[n, n] = 1.0 / denominator
                self.P = P_new
            else:
                # Fallback to simple update
                coefficient = self.learning_rate * error
                self._update_dictionary(x, coefficient)
                # Reinitialize P for safety
                self.P = np.eye(len(self.dictionary)) * (1.0 / self.regularization)
        else:
            # Update existing coefficients using RLS
            if self.P is not None and len(self.dictionary) > 0:
                k = self._compute_kernel_vector(x)
                
                # Check dimensions match
                if len(k) == len(self.coefficients) and self.P.shape[0] == len(k):
                    Pk = self.P @ k
                    denominator = self.forgetting_factor + k.T @ Pk
                    
                    # RLS gain
                    gain = Pk / denominator
                    
                    # Update coefficients
                    coef_array = np.array(self.coefficients)
                    coef_array = coef_array + gain * error
                    self.coefficients = coef_array.tolist()
                    
                    # Update P
                    self.P = (self.P - np.outer(Pk, Pk) / denominator) / self.forgetting_factor
                else:
                    # Dimension mismatch, use simple update
                    kernel_vec = self._compute_kernel_vector(x)
                    for i in range(min(len(self.coefficients), len(kernel_vec))):
                        self.coefficients[i] += self.learning_rate * error * kernel_vec[i]
