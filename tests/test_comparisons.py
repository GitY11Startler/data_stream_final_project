"""
Tests for comparisons module.
"""
import sys
sys.path.append('..')

import numpy as np
from src.stream import KAFRegressor, CapyMOARegressor
from src.evaluation.comparisons import (
    compare_algorithms,
    compare_with_baseline,
    rank_algorithms,
    multi_metric_ranking,
    get_best_algorithm,
    create_comparison_summary,
    print_comparison_summary,
    print_comparison_table
)


def generate_test_data(n_samples=200, n_features=5):
    """Generate simple test data."""
    np.random.seed(42)
    stream_data = []
    
    for i in range(n_samples):
        x = {f'f{j}': np.random.randn() for j in range(n_features)}
        y = sum(x.values()) + np.random.randn() * 0.5
        stream_data.append((x, y))
    
    return stream_data


def test_basic_comparison():
    """Test basic algorithm comparison."""
    print("\n" + "="*70)
    print("TEST 1: Basic Algorithm Comparison")
    print("="*70)
    
    # Generate data
    stream_data = generate_test_data(n_samples=150)
    print(f"\nGenerated {len(stream_data)} samples")
    
    # Create algorithms
    algorithms = {
        'KLMS': KAFRegressor(algorithm='KLMS', learning_rate=0.1),
        'KRLS': KAFRegressor(algorithm='KRLS', learning_rate=0.1),
        'ARF': CapyMOARegressor(algorithm='AdaptiveRandomForestRegressor'),
        'KNN': CapyMOARegressor(algorithm='KNNRegressor', k=10)
    }
    
    print(f"\nComparing {len(algorithms)} algorithms...")
    
    # Run comparison
    results_df = compare_algorithms(
        algorithms,
        stream_data,
        warm_start=10,
        verbose=False
    )
    
    # Print results
    print_comparison_table(results_df, "Algorithm Comparison Results")
    
    # Verify results
    assert len(results_df) == 4, "Should have 4 algorithms"
    assert 'MAE' in results_df.columns, "Should have MAE"
    assert 'directional_accuracy' in results_df.columns, "Should have directional accuracy"
    
    print("\n✅ Test passed!")
    return results_df


def test_baseline_comparison():
    """Test comparison with baseline."""
    print("\n" + "="*70)
    print("TEST 2: Comparison with Baseline")
    print("="*70)
    
    # Generate data
    stream_data = generate_test_data(n_samples=150)
    
    # Create algorithms
    test_algorithms = {
        'KLMS': KAFRegressor(algorithm='KLMS', learning_rate=0.1),
        'ARF': CapyMOARegressor(algorithm='AdaptiveRandomForestRegressor')
    }
    
    baseline = CapyMOARegressor(algorithm='KNNRegressor', k=10)
    
    print("\nComparing algorithms against KNN baseline...")
    
    # Run comparison
    results_df, improvement_df = compare_with_baseline(
        test_algorithms,
        baseline,
        'KNN_Baseline',
        stream_data,
        verbose=False
    )
    
    # Print results
    print_comparison_table(results_df, "Results vs Baseline")
    
    print("\n" + "="*70)
    print("Improvement Over Baseline (%)".center(70))
    print("="*70)
    print(improvement_df.to_string(index=False))
    print("="*70)
    
    assert len(results_df) == 3, "Should have 3 algorithms (2 test + 1 baseline)"
    assert len(improvement_df) == 2, "Should have 2 improvements"
    
    print("\n✅ Test passed!")
    return results_df, improvement_df


def test_ranking():
    """Test algorithm ranking."""
    print("\n" + "="*70)
    print("TEST 3: Algorithm Ranking")
    print("="*70)
    
    # Generate data and run comparison
    stream_data = generate_test_data(n_samples=150)
    
    algorithms = {
        'KLMS': KAFRegressor(algorithm='KLMS', learning_rate=0.1),
        'KNLMS': KAFRegressor(algorithm='KNLMS', learning_rate=0.1),
        'KRLS': KAFRegressor(algorithm='KRLS', learning_rate=0.1),
        'ARF': CapyMOARegressor(algorithm='AdaptiveRandomForestRegressor')
    }
    
    print("\nRunning comparison for ranking...")
    results_df = compare_algorithms(algorithms, stream_data, verbose=False)
    
    # Test single metric ranking
    print("\n" + "="*70)
    print("Ranking by MAE (Lower is Better)".center(70))
    print("="*70)
    ranked_df = rank_algorithms(results_df, metric='MAE', ascending=True)
    print(ranked_df[['rank', 'algorithm', 'MAE', 'RMSE', 'directional_accuracy']].to_string(index=False))
    
    # Test multi-metric ranking
    print("\n" + "="*70)
    print("Multi-Metric Ranking".center(70))
    print("="*70)
    multi_ranked_df = multi_metric_ranking(results_df)
    print(multi_ranked_df.to_string(index=False))
    
    # Get best algorithm
    best_algo, best_value = get_best_algorithm(results_df, 'MAE', ascending=True)
    print(f"\n🏆 Best algorithm by MAE: {best_algo} (MAE = {best_value:.4f})")
    
    print("\n✅ Test passed!")
    return ranked_df, multi_ranked_df


def test_summary():
    """Test comparison summary."""
    print("\n" + "="*70)
    print("TEST 4: Comparison Summary")
    print("="*70)
    
    # Generate data and run comparison
    stream_data = generate_test_data(n_samples=150)
    
    algorithms = {
        'KLMS': KAFRegressor(algorithm='KLMS', learning_rate=0.1),
        'KRLS': KAFRegressor(algorithm='KRLS', learning_rate=0.1),
        'ARF': CapyMOARegressor(algorithm='AdaptiveRandomForestRegressor'),
        'KNN': CapyMOARegressor(algorithm='KNNRegressor', k=10)
    }
    
    print("\nRunning comparison...")
    results_df = compare_algorithms(algorithms, stream_data, verbose=False)
    
    # Create summary
    summary = create_comparison_summary(results_df)
    
    # Print summary
    print_comparison_summary(summary)
    
    # Verify summary
    assert summary['n_algorithms'] == 4
    assert 'best_mae' in summary
    assert 'best_directional_accuracy' in summary
    
    print("\n✅ Test passed!")
    return summary


def test_consistent_format():
    """Test that results have consistent format."""
    print("\n" + "="*70)
    print("TEST 5: Result Format Consistency")
    print("="*70)
    
    stream_data = generate_test_data(n_samples=100)
    
    # Test with KAF only
    print("\n1. Testing with KAF algorithms only...")
    kaf_algorithms = {
        'KLMS': KAFRegressor(algorithm='KLMS'),
        'KRLS': KAFRegressor(algorithm='KRLS')
    }
    kaf_results = compare_algorithms(kaf_algorithms, stream_data, verbose=False)
    print(f"   Columns: {list(kaf_results.columns)}")
    
    # Test with CapyMOA only
    print("\n2. Testing with CapyMOA algorithms only...")
    capy_algorithms = {
        'ARF': CapyMOARegressor(algorithm='AdaptiveRandomForestRegressor'),
        'KNN': CapyMOARegressor(algorithm='KNNRegressor')
    }
    capy_results = compare_algorithms(capy_algorithms, stream_data, verbose=False)
    print(f"   Columns: {list(capy_results.columns)}")
    
    # Test with mixed
    print("\n3. Testing with mixed algorithms...")
    mixed_algorithms = {
        'KLMS': KAFRegressor(algorithm='KLMS'),
        'ARF': CapyMOARegressor(algorithm='AdaptiveRandomForestRegressor')
    }
    mixed_results = compare_algorithms(mixed_algorithms, stream_data, verbose=False)
    print(f"   Columns: {list(mixed_results.columns)}")
    
    # Verify consistency
    required_cols = ['algorithm', 'MAE', 'RMSE', 'R2', 'directional_accuracy', 'time_seconds', 'samples']
    
    for df, name in [(kaf_results, 'KAF'), (capy_results, 'CapyMOA'), (mixed_results, 'Mixed')]:
        for col in required_cols:
            assert col in df.columns, f"{col} missing in {name} results"
    
    print("\n✅ All results have consistent format!")
    print(f"\n   Required columns present in all results:")
    for col in required_cols:
        print(f"     ✓ {col}")
    
    print("\n✅ Test passed!")


def main():
    """Run all tests."""
    print("="*70)
    print("COMPARISONS MODULE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic Comparison", test_basic_comparison),
        ("Baseline Comparison", test_baseline_comparison),
        ("Ranking", test_ranking),
        ("Summary", test_summary),
        ("Format Consistency", test_consistent_format)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            test_func()
            results[test_name] = True
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\n✅ comparisons.py is ready for experiments!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
