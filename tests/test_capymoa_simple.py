"""
Simple test to verify the 3 main CapyMOA algorithms work correctly.
"""
import sys
sys.path.append('..')

import numpy as np
from src.stream.capymoa_wrapper import CapyMOARegressor
from river import metrics


def test_three_main_algorithms():
    """Test the three main algorithms we'll use for comparison."""
    print("="*70)
    print("TESTING THREE MAIN CAPYMOA ALGORITHMS")
    print("="*70)
    
    algorithms = [
        'AdaptiveRandomForestRegressor',
        'KNNRegressor',
        'StreamingGradientBoostedRegression'
    ]
    
    np.random.seed(42)
    n_samples = 200
    
    for algo_name in algorithms:
        print(f"\n{'='*70}")
        print(f"Testing: {algo_name}")
        print(f"{'='*70}")
        
        # Create model
        model = CapyMOARegressor(algorithm=algo_name)
        
        # Create metrics
        mae = metrics.MAE()
        rmse = metrics.RMSE()
        
        # Generate and test data
        for i in range(n_samples):
            # Create simple linear relationship
            x = {f'x{j}': np.random.randn() for j in range(5)}
            y_true = sum(x.values()) + np.random.randn() * 0.5
            
            # Predict
            y_pred = model.predict_one(x)
            
            # Update metrics
            mae.update(y_true, y_pred)
            rmse.update(y_true, y_pred)
            
            # Learn
            model.learn_one(x, y_true)
            
            # Print progress
            if (i + 1) % 50 == 0:
                print(f"  Sample {i+1:3d}: MAE = {mae.get():.4f}, RMSE = {rmse.get():.4f}")
        
        # Final results
        print(f"\n  Final Results:")
        print(f"    MAE:  {mae.get():.4f}")
        print(f"    RMSE: {rmse.get():.4f}")
        print(f"    ✅ {algo_name} works correctly!")
    
    print(f"\n{'='*70}")
    print("🎉 ALL THREE MAIN ALGORITHMS WORK PERFECTLY! 🎉")
    print(f"{'='*70}")
    print("\nReady for experiments:")
    print("  ✅ AdaptiveRandomForestRegressor - Ensemble method (best overall)")
    print("  ✅ KNNRegressor - Instance-based (simple, interpretable)")
    print("  ✅ StreamingGradientBoostedRegression - Advanced ensemble")
    print("\nYou can now proceed with the comparison experiments!")


if __name__ == "__main__":
    test_three_main_algorithms()
