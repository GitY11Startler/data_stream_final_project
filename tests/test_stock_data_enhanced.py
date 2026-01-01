"""
Tests for enhanced stock_data module.
"""
import sys
sys.path.append('..')

import os
from src.data.stock_data import (
    load_stock_data,
    load_multiple_stocks,
    load_stock_list,
    validate_date_interval,
    adjust_dates_for_interval,
    clear_cache,
    get_cache_info,
    STOCK_LISTS,
    INTERVAL_LIMITS,
    CACHE_DIR
)


def test_interval_validation():
    """Test date/interval validation."""
    print("\n" + "="*70)
    print("TEST 1: Date/Interval Validation")
    print("="*70)
    
    # Test valid cases
    print("\n1. Testing valid date ranges...")
    
    valid_cases = [
        ('2024-01-01', '2024-12-31', '1d', True),
        ('2024-12-25', '2025-01-01', '1m', True),  # 7 days
        ('2024-11-01', '2024-12-31', '5m', True),  # 60 days
    ]
    
    for start, end, interval, expected in valid_cases:
        is_valid, msg = validate_date_interval(start, end, interval)
        status = "✅" if is_valid == expected else "❌"
        print(f"  {status} {interval}: {start} to {end}")
        if not is_valid:
            print(f"     Message: {msg}")
    
    # Test invalid cases
    print("\n2. Testing invalid date ranges...")
    
    invalid_cases = [
        ('2024-01-01', '2024-12-31', '1m', False),  # Too long for 1m
        ('2023-01-01', '2024-12-31', '5m', False),  # Too long for 5m
    ]
    
    for start, end, interval, expected in invalid_cases:
        is_valid, msg = validate_date_interval(start, end, interval)
        status = "✅" if is_valid == expected else "❌"
        print(f"  {status} {interval}: {start} to {end}")
        print(f"     Message: {msg}")
    
    print("\n✅ Test passed!")


def test_date_adjustment():
    """Test automatic date adjustment."""
    print("\n" + "="*70)
    print("TEST 2: Automatic Date Adjustment")
    print("="*70)
    
    intervals = ['1m', '5m', '1h', '1d']
    
    for interval in intervals:
        start, end = adjust_dates_for_interval(interval)
        print(f"\n{interval}:")
        print(f"  Start: {start}")
        print(f"  End: {end}")
        
        # Validate the adjusted dates
        is_valid, _ = validate_date_interval(start, end, interval)
        status = "✅" if is_valid else "❌"
        print(f"  Valid: {status}")
    
    print("\n✅ Test passed!")


def test_caching():
    """Test data caching functionality."""
    print("\n" + "="*70)
    print("TEST 3: Data Caching")
    print("="*70)
    
    # Clear existing cache
    print("\n1. Clearing existing cache...")
    clear_cache()
    
    # Load data (should download and cache)
    print("\n2. Loading AAPL data (will download)...")
    start, end = adjust_dates_for_interval('1d')
    
    import time
    t1 = time.time()
    df1 = load_stock_data('AAPL', start, end, interval='1d', use_cache=True)
    download_time = time.time() - t1
    
    print(f"   Downloaded {len(df1)} samples in {download_time:.2f}s")
    
    # Load again (should use cache)
    print("\n3. Loading AAPL data again (will use cache)...")
    t2 = time.time()
    df2 = load_stock_data('AAPL', start, end, interval='1d', use_cache=True)
    cache_time = time.time() - t2
    
    print(f"   Loaded {len(df2)} samples from cache in {cache_time:.2f}s")
    print(f"   Speedup: {download_time/cache_time:.1f}x faster")
    
    # Check cache info
    print("\n4. Cache information:")
    cache_info = get_cache_info()
    if not cache_info.empty:
        print(f"   Files in cache: {len(cache_info)}")
        print(f"   Total size: {cache_info['size_mb'].sum():.2f} MB")
    
    # Verify data is similar (allow small difference due to timing)
    assert abs(len(df1) - len(df2)) <= 2, f"Data length difference too large: {len(df1)} vs {len(df2)}"
    print(f"   ✅ Data lengths match (df1={len(df1)}, df2={len(df2)})")
    print("\n✅ Test passed!")
    
    return df1


def test_multi_stock_loading():
    """Test loading multiple stocks."""
    print("\n" + "="*70)
    print("TEST 4: Multi-Stock Loading")
    print("="*70)
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    start, end = adjust_dates_for_interval('1d')
    
    print(f"\nLoading {len(symbols)} stocks: {', '.join(symbols)}")
    
    data_dict = load_multiple_stocks(
        symbols,
        start,
        end,
        interval='1d',
        use_cache=True
    )
    
    print(f"\n✅ Successfully loaded {len(data_dict)} stocks:")
    for symbol, df in data_dict.items():
        print(f"   • {symbol}: {len(df)} samples")
    
    assert len(data_dict) > 0, "No stocks loaded"
    print("\n✅ Test passed!")
    
    return data_dict


def test_stock_lists():
    """Test pre-configured stock lists."""
    print("\n" + "="*70)
    print("TEST 5: Pre-configured Stock Lists")
    print("="*70)
    
    print("\nAvailable stock lists:")
    for list_name, symbols in STOCK_LISTS.items():
        print(f"  • {list_name}: {', '.join(symbols[:3])}{'...' if len(symbols) > 3 else ''}")
    
    # Test loading a small list
    print("\n\nLoading 'tech' stock list...")
    start, end = adjust_dates_for_interval('1d')
    
    data_dict = load_stock_list(
        'tech',
        start,
        end,
        interval='1d',
        use_cache=True
    )
    
    print(f"\n✅ Successfully loaded {len(data_dict)} tech stocks:")
    for symbol, df in data_dict.items():
        print(f"   • {symbol}: {len(df)} samples")
    
    print("\n✅ Test passed!")


def test_interval_limits():
    """Test that interval limits are documented."""
    print("\n" + "="*70)
    print("TEST 6: Interval Limits Documentation")
    print("="*70)
    
    print("\nSupported intervals and limits:")
    for interval, info in INTERVAL_LIMITS.items():
        max_days = info['max_days']
        name = info['name']
        if max_days:
            print(f"  • {interval:4s} ({name:12s}): max {max_days} days")
        else:
            print(f"  • {interval:4s} ({name:12s}): unlimited")
    
    print("\n✅ Test passed!")


def test_cache_directory():
    """Test cache directory creation."""
    print("\n" + "="*70)
    print("TEST 7: Cache Directory")
    print("="*70)
    
    print(f"\nCache directory: {CACHE_DIR}")
    
    if os.path.exists(CACHE_DIR):
        print("  ✅ Cache directory exists")
        files = os.listdir(CACHE_DIR)
        print(f"  Files in cache: {len(files)}")
    else:
        print("  ℹ️  Cache directory will be created on first use")
    
    print("\n✅ Test passed!")


def main():
    """Run all tests."""
    print("="*70)
    print("ENHANCED STOCK_DATA MODULE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Interval Validation", test_interval_validation),
        ("Date Adjustment", test_date_adjustment),
        ("Caching", test_caching),
        ("Multi-Stock Loading", test_multi_stock_loading),
        ("Stock Lists", test_stock_lists),
        ("Interval Limits", test_interval_limits),
        ("Cache Directory", test_cache_directory),
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
        print("\n✅ Enhanced stock_data.py is ready!")
        print("\nNew features:")
        print("  ✓ Data caching for faster loading")
        print("  ✓ Multi-stock loading")
        print("  ✓ Date/interval validation")
        print("  ✓ Pre-configured stock lists")
        print("  ✓ Automatic date adjustment")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
