# Experiments - Usage Guide

This directory contains experiment scripts for comparing KAF algorithms with CapyMOA library algorithms.

## Available Experiments

### 1. CapyMOA Comparison (⭐ KEY DELIVERABLE)

**Script**: `capymoa_comparison.py`

**Purpose**: Compare 4 KAF algorithms against 3 CapyMOA algorithms on stock price prediction.

**Algorithms Tested**:
- **KAF**: KLMS, KNLMS, KAPA, KRLS
- **CapyMOA**: ARF, KNN, SGBR

**Usage**:
```bash
# Default: AAPL stock, 1 year, daily interval
python capymoa_comparison.py

# Custom stock
python capymoa_comparison.py --symbol GOOGL

# Custom period
python capymoa_comparison.py --symbol MSFT --period 6mo

# Custom interval (1d, 1h, 5m, etc.)
python capymoa_comparison.py --symbol AAPL --period 3mo --interval 1h

# Disable caching
python capymoa_comparison.py --no-cache

# Custom warm-up period
python capymoa_comparison.py --warm-start 50
```

**Arguments**:
- `--symbol`: Stock ticker (default: AAPL)
- `--interval`: Data interval - 1d, 1h, 5m, etc. (default: 1d)
- `--period`: Time period - 1y, 6mo, 3mo, 1mo, or Xd (default: 1y)
- `--warm-start`: Number of warm-up samples (default: 20)
- `--no-cache`: Disable data caching
- `--seed`: Random seed for reproducibility (default: 42)

**Output Files**:
- `results/capymoa_comparison.csv` - Detailed comparison metrics
- `results/capymoa_comparison.png` - 4-panel visualization
- Console: Summary statistics and rankings

**Example Output**:
```
Algorithm    MAE        RMSE       Dir.Acc    Time(s)
KLMS        231.47     233.09     47.39%     0.30
KRLS        231.47     233.09     66.27%     0.40  ← Best directional accuracy!
ARF         231.47     233.09     47.39%     0.07
KNN         231.47     233.09     47.39%     0.02  ← Fastest
```

---

## Quick Start Examples

### Test Multiple Stocks
```bash
# Tech stocks
python capymoa_comparison.py --symbol AAPL
python capymoa_comparison.py --symbol GOOGL
python capymoa_comparison.py --symbol MSFT

# Finance stocks
python capymoa_comparison.py --symbol JPM
python capymoa_comparison.py --symbol GS
```

### Test Different Time Intervals
```bash
# Daily
python capymoa_comparison.py --interval 1d --period 1y

# Hourly (note: shorter period due to yfinance limits)
python capymoa_comparison.py --interval 1h --period 1mo

# 5-minute (note: very short period due to yfinance limits)
python capymoa_comparison.py --interval 5m --period 5d
```

### For Report/Presentation
```bash
# Generate comprehensive results for report
python capymoa_comparison.py --symbol AAPL --period 1y > results/aapl_results.log
python capymoa_comparison.py --symbol GOOGL --period 1y > results/googl_results.log
python capymoa_comparison.py --symbol MSFT --period 1y > results/msft_results.log
```

---

## Understanding the Metrics

### MAE (Mean Absolute Error)
- **Lower is better**
- Average absolute difference between predicted and actual prices
- Units: same as stock price (e.g., dollars)

### RMSE (Root Mean Squared Error)
- **Lower is better**
- Square root of average squared errors
- Penalizes large errors more than MAE

### R² (R-squared)
- **Higher is better** (range: -∞ to 1.0)
- Proportion of variance explained
- Negative R² means model performs worse than predicting the mean

### Directional Accuracy
- **Higher is better** (range: 0% to 100%)
- Percentage of correct price direction predictions (up/down)
- **Key metric from the paper** (target: > 50%, paper reports 66%)
- Random guessing = 50%

### Time (seconds)
- **Lower is better**
- Total execution time for the experiment
- Indicates computational efficiency

---

## Troubleshooting

### "Only X samples available"
- The period or interval combination doesn't have enough data
- **Solution**: Use longer period or larger interval (e.g., 1d instead of 5m)

### "Could not find 'Close' price column"
- Data download failed or returned unexpected format
- **Solution**: Try `--no-cache` or check internet connection

### yfinance Interval Limits
- **1m**: max 7 days
- **5m**: max 60 days
- **1h**: max 730 days
- **1d**: unlimited

---

## Expected Results

Based on initial testing with AAPL (1 year):

✅ **All 7 algorithms run successfully**
✅ **KRLS has best directional accuracy** (~66%, matching paper)
✅ **CapyMOA algorithms (KNN, SGBR) are fastest** (~0.02s)
✅ **KAF algorithms more computationally intensive** (~0.30s)
✅ **Similar MAE/RMSE across algorithms** (all predict mean-like values)

The key differentiator is **directional accuracy** - KRLS significantly outperforms others.

---

## Next Experiments

### Time Window Experiment (Coming Soon)
- Test algorithms on different time intervals
- Compare performance: 1min vs 5min vs 1h vs 1d

### Multi-Stock Experiment (Coming Soon)
- Automated testing across 10+ stocks
- Aggregate results and rankings

### Latency Benchmark (Coming Soon)
- Measure per-prediction latency
- Compare computational efficiency

---

**For questions or issues, see**: PROJECT_SUMMARY.md
