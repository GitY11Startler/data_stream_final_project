"""
Unit tests for KAF algorithms.
"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from src.algorithms import KLMS, KNLMS, KAPA, KRLS


class TestKAFAlgorithms:
    """Test suite for Kernel Adaptive Filtering algorithms."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 3
        
        # Generate simple test data
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.sin(self.X[:, 0]) + 0.5 * self.X[:, 1] + np.random.randn(self.n_samples) * 0.1
    
    def test_klms_initialization(self):
        """Test KLMS initialization."""
        model = KLMS(learning_rate=0.1, kernel='gaussian', kernel_size=1.0)
        assert model.learning_rate == 0.1
        assert model.kernel_type == 'gaussian'
        assert len(model.dictionary) == 0
    
    def test_klms_learning(self):
        """Test KLMS learning and prediction."""
        model = KLMS(learning_rate=0.1, max_dictionary_size=20)
        
        # Train on data
        for i in range(self.n_samples):
            model.update(self.X[i], self.y[i])
        
        # Check dictionary is populated
        assert len(model.dictionary) > 0
        assert len(model.dictionary) <= 20
        
        # Check predictions work
        pred = model.predict(self.X[0])
        assert isinstance(pred, (float, np.floating))
    
    def test_knlms_learning(self):
        """Test KNLMS learning."""
        model = KNLMS(learning_rate=0.5, epsilon=1e-6)
        
        for i in range(self.n_samples):
            model.update(self.X[i], self.y[i])
        
        assert len(model.dictionary) > 0
        pred = model.predict(self.X[0])
        assert isinstance(pred, (float, np.floating))
    
    def test_kapa_learning(self):
        """Test KAPA learning."""
        model = KAPA(learning_rate=0.1, projection_order=3)
        
        for i in range(self.n_samples):
            model.update(self.X[i], self.y[i])
        
        assert len(model.dictionary) > 0
        assert len(model.recent_x) <= 3
    
    def test_krls_learning(self):
        """Test KRLS learning."""
        model = KRLS(forgetting_factor=0.99)
        
        for i in range(self.n_samples):
            model.update(self.X[i], self.y[i])
        
        assert len(model.dictionary) > 0
        assert model.P is not None
    
    def test_different_kernels(self):
        """Test different kernel functions."""
        for kernel in ['gaussian', 'polynomial', 'linear']:
            model = KLMS(kernel=kernel, max_dictionary_size=10)
            
            for i in range(10):
                model.update(self.X[i], self.y[i])
            
            pred = model.predict(self.X[0])
            assert isinstance(pred, (float, np.floating))
    
    def test_dictionary_size_limit(self):
        """Test that dictionary size is limited."""
        max_size = 15
        model = KLMS(max_dictionary_size=max_size, novelty_threshold=0.0)
        
        # Feed many samples
        for i in range(50):
            model.update(self.X[i % self.n_samples], self.y[i % self.n_samples])
        
        assert len(model.dictionary) <= max_size
    
    def test_predict_before_training(self):
        """Test prediction before any training."""
        model = KLMS()
        pred = model.predict(self.X[0])
        assert pred == 0.0
    
    def test_river_interface(self):
        """Test River-compatible interface."""
        model = KLMS()
        
        # Test learn_one with dict
        x_dict = {f"f{i}": self.X[0, i] for i in range(self.n_features)}
        model.learn_one(x_dict, self.y[0])
        
        # Test predict_one
        pred = model.predict_one(x_dict)
        assert isinstance(pred, (float, np.floating))
    
    def test_get_params(self):
        """Test parameter retrieval."""
        model = KLMS(learning_rate=0.2, kernel='gaussian')
        params = model.get_params()
        
        assert params['learning_rate'] == 0.2
        assert params['kernel'] == 'gaussian'
        assert 'dictionary_size' in params
    
    def test_reset(self):
        """Test model reset."""
        model = KLMS()
        
        # Train model
        for i in range(10):
            model.update(self.X[i], self.y[i])
        
        assert len(model.dictionary) > 0
        
        # Reset
        model.reset()
        
        assert len(model.dictionary) == 0
        assert len(model.coefficients) == 0
        assert model.n_samples == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
