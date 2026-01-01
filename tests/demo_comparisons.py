"""
Quick demo of the comparisons module.
"""
import sys
sys.path.append('..')

import numpy as np
from src.stream import KAFRegressor, CapyMOARegressor
from src.evaluation.comparisons import (
    compare_algorithms,
    print_comparison_table,
    create_comparison_summary,
    print_comparison_summary,
    export_comparison_results
)


def main():
    print("="*70)
    print("COMPARISONS MODULE - QUICK DEMO")
    print("="*70)
    
    # 1. Generate sample data
    print("\n1. Generating sample streaming data...")
    np.random.seed(42)
    n_samples = 200
    
    stream_data = []
    for i in range(n_samples):
        x = {f'feature_{j}': np.random.randn() for j in range(5)}
        y = sum(x.values()) + np.random.randn() * 0.5
        stream_data.append((x, y))
    
    print(f"   ✅ Generated {n_samples} samples with 5 features")
    
    # 2. Create algorithms to compare
    print("\n2. Creating algorithms...")
    algorithms = {
        # KAF algorithms
        'KLMS': KAFRegressor(
            algorithm='KLMS',
            learning_rate=0.1,
            kernel='gaussian',
            kernel_size=1.0
        ),
        'KRLS': KAFRegressor(
            algorithm='KRLS',
            learning_rate=0.1,
            kernel='gaussian',
            kernel_size=1.0
        ),
        # CapyMOA algorithms
        'ARF': CapyMOARegressor(
            algorithm='AdaptiveRandomForestRegressor'
        ),
        'KNN': CapyMOARegressor(
            algorithm='KNNRegressor',
            k=10
        ),
        'SGBR': CapyMOARegressor(
            algorithm='StreamingGradientBoostedRegression'
        )
    }
    
    print(f"   ✅ Created {len(algorithms)} algorithms")
    print(f"      - 2 KAF algorithms (KLMS, KRLS)")
    print(f"      - 3 CapyMOA algorithms (ARF, KNN, SGBR)")
    
    # 3. Run comparison
    print("\n3. Running comparison (this may take a minute)...")
    results_df = compare_algorithms(
        algorithms,
        stream_data,
        warm_start=20,
        verbose=False
    )
    
    # 4. Display results
    print_comparison_table(results_df, "Algorithm Comparison Results")
    
    # 5. Create and display summary
    summary = create_comparison_summary(results_df)
    print_comparison_summary(summary)
    
    # 6. Export results
    print("\n4. Exporting results...")
    export_comparison_results(results_df, '../results/demo_comparison')
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nKey takeaways:")
    print("  ✅ compare_algorithms() works with both KAF and CapyMOA")
    print("  ✅ Results are in consistent format")
    print("  ✅ Multiple metrics tracked automatically")
    print("  ✅ Easy to export and analyze")
    print("\n🎉 Ready for real experiments!")


if __name__ == "__main__":
    main()
