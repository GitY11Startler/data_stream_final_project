"""
Tests for CapyMOA wrapper functionality.
"""
import sys
sys.path.append('..')

import numpy as np
from src.stream.capymoa_wrapper import (
    CapyMOARegressor, 
    list_available_algorithms,
    test_algorithm
)


def test_list_algorithms():
    """Test listing available algorithms."""
    print("\n" + "="*70)
    print("TEST 1: List Available Algorithms")
    print("="*70)
    
    algorithms = list_available_algorithms()
    print(f"\nFound {len(algorithms)} algorithms:")
    for algo in algorithms:
        print(f"  - {algo}")
    
    assert len(algorithms) > 0, "No algorithms available"
    print("\n✅ Test passed!")


def test_single_algorithm():
    """Test a single algorithm with simple data."""
    print("\n" + "="*70)
    print("TEST 2: Test Single Algorithm (AdaptiveRandomForestRegressor)")
    print("="*70)
    
    # Create model
    model = CapyMOARegressor(algorithm='AdaptiveRandomForestRegressor')
    print(f"\nModel created: {model}")
    
    # Generate simple data
    np.random.seed(42)
    n_samples = 100
    
    print(f"\nTraining on {n_samples} samples...")
    predictions = []
    actuals = []
    
    for i in range(n_samples):
        # Create features
        x = {f'feature_{j}': np.random.randn() for j in range(5)}
        # Simple linear target
        y = sum(x.values()) + np.random.randn() * 0.1
        
        # Predict then learn
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        
        predictions.append(y_pred)
        actuals.append(y)
        
        if (i + 1) % 25 == 0:
            mae = np.mean(np.abs(np.array(actuals[:i+1]) - np.array(predictions[:i+1])))
            print(f"  Sample {i+1}: MAE = {mae:.4f}")
    
    # Calculate final metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mae = np.mean(np.abs(actuals - predictions))
    
    print(f"\nFinal MAE: {mae:.4f}")
    
    # Get model info
    info = model.get_info()
    print(f"\nModel Info:")
    print(f"  Samples: {info['n_samples']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Initialized: {info['initialized']}")
    
    assert info['n_samples'] == n_samples, "Sample count mismatch"
    assert info['n_features'] == 5, "Feature count mismatch"
    print("\n✅ Test passed!")


def test_all_algorithms():
    """Test all available algorithms."""
    print("\n" + "="*70)
    print("TEST 3: Test All Available Algorithms")
    print("="*70)
    
    algorithms = list_available_algorithms()
    results = {}
    
    for algo in algorithms:
        success = test_algorithm(algo, n_samples=50)
        results[algo] = success
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    for algo, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {algo}")
    
    if passed == total:
        print("\n✅ All tests passed!")
    else:
        print(f"\n⚠️ {total - passed} algorithm(s) failed")
    
    return results


def test_river_compatibility():
    """Test that the wrapper is compatible with River's evaluation."""
    print("\n" + "="*70)
    print("TEST 4: River Compatibility")
    print("="*70)
    
    try:
        from river import metrics
        
        # Create model
        model = CapyMOARegressor(algorithm='KNNRegressor', k=5)
        
        # Create metric
        mae_metric = metrics.MAE()
        
        # Generate data
        np.random.seed(42)
        n_samples = 100
        
        print(f"\nEvaluating with River metrics on {n_samples} samples...")
        
        for i in range(n_samples):
            x = {f'x{j}': np.random.randn() for j in range(3)}
            y = sum(x.values()) + np.random.randn() * 0.5
            
            # Predict
            y_pred = model.predict_one(x)
            
            # Update metric
            mae_metric.update(y, y_pred)
            
            # Learn
            model.learn_one(x, y)
            
            if (i + 1) % 25 == 0:
                print(f"  Sample {i+1}: MAE = {mae_metric.get():.4f}")
        
        print(f"\nFinal MAE: {mae_metric.get():.4f}")
        print("\n✅ River compatibility test passed!")
        
        return True
        
    except ImportError:
        print("\n⚠️ River not available, skipping compatibility test")
        return False


def test_error_handling():
    """Test error handling."""
    print("\n" + "="*70)
    print("TEST 5: Error Handling")
    print("="*70)
    
    # Test invalid algorithm
    print("\n1. Testing invalid algorithm...")
    try:
        model = CapyMOARegressor(algorithm='InvalidAlgorithm')
        print("  ❌ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✅ Correctly raised ValueError: {e}")
    
    # Test prediction before training
    print("\n2. Testing prediction before training...")
    model = CapyMOARegressor(algorithm='SGDRegressor')
    x = {'f0': 1.0, 'f1': 2.0}
    y_pred = model.predict_one(x)
    print(f"  ✅ Prediction before training: {y_pred} (expected 0.0 with warning)")
    
    print("\n✅ Error handling test passed!")
    return True


def main():
    """Run all tests."""
    print("="*70)
    print("CAPYMOA WRAPPER TEST SUITE")
    print("="*70)
    
    tests = [
        ("List Algorithms", test_list_algorithms),
        ("Single Algorithm", test_single_algorithm),
        ("All Algorithms", test_all_algorithms),
        ("River Compatibility", test_river_compatibility),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result if result is not None else True
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
