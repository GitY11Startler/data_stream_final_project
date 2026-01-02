# Online Kernel Adaptive Filtering for Mid-Price Prediction

**Project 1 - Theme 4: REGRESSION FINANCE**  
**Course**: Data Stream Processing  
**Date**: January 2026

---

## 📋 Quick Navigation for Reviewers

| Document | Purpose |
|----------|---------|
| **[REPORT.md](REPORT.md)** | 📄 Full academic report (main deliverable) |
| **[results/](results/)** | 📊 All experimental results and figures |
| **This README** | 🚀 How to run and verify the project |

---

## 👥 Team Members

- Yassine Zanned
- Mohamed Amine Arous
- Chaouch Achraf

---

## 📝 Project Summary

This project implements **Kernel Adaptive Filtering (KAF)** algorithms for online stock price prediction, based on the paper *"An Online Kernel Adaptive Filtering-Based Approach for Mid-Price Prediction"* (Mishra et al., 2022).

### Key Results

| Metric | Best KAF (KRLS) | Best CapyMOA (ARF) | Improvement |
|--------|-----------------|-------------------|-------------|
| **MAE** | 0.242 | 0.860 | **65.3%** |
| **R²** | 0.883 | -0.005 | KAF >> CapyMOA |
| **Directional Accuracy** | 54.26% | 45.69% | **+14.4%** |

**Conclusion**: KAF algorithms significantly outperform CapyMOA streaming algorithms for stock prediction.

---

## 📁 Project Structure

```
data_stream_final_project/
│
├── 📄 REPORT.md                    # Academic report (READ THIS)
├── 📄 README.md                    # This file
│
├── src/                            # Source code
│   ├── algorithms/
│   │   ├── base_kaf.py            # Base class with kernel functions
│   │   └── kaf.py                 # KLMS, KNLMS, KAPA, KRLS implementations
│   ├── stream/
│   │   ├── river_wrapper.py       # River API wrapper for KAF
│   │   └── capymoa_wrapper.py     # CapyMOA integration
│   ├── data/
│   │   └── stock_data.py          # Data loading with caching
│   └── evaluation/
│       ├── metrics.py             # Evaluation metrics
│       ├── comparisons.py         # Algorithm comparison utilities
│       └── plots.py               # Visualization functions
│
├── experiments/                    # Experiment scripts
│   ├── capymoa_comparison.py      # Main KAF vs CapyMOA comparison
│   ├── multi_stock_experiment.py  # Multi-stock generalization test
│   ├── time_window_experiment.py  # Time interval analysis
│   └── aggregate_results.py       # Result aggregation
│
├── tests/                          # Unit tests
│   ├── test_kaf.py                # KAF algorithm tests
│   ├── test_capymoa_wrapper.py    # CapyMOA wrapper tests
│   └── test_comparisons.py        # Comparison utility tests
│
├── results/                        # Experimental results
│   ├── multi_stock_results.csv    # Main results (8 stocks)
│   ├── final_report_plots.png     # Publication-ready figures
│   └── ...                        # Other result files
│
├── data/cache/                     # Downloaded stock data cache
├── requirements.txt                # Python dependencies
└── setup.py                        # Package setup
```

---

## 🚀 Installation & Setup

### Step 1: Create Conda Environment

```bash
# Create a new conda environment with Python 3.10
conda create -n DataStream python=3.10 -y

# Activate the environment
conda activate DataStream
```

### Step 2: Install Dependencies

```bash
# Navigate to the project directory
cd /path/to/data_stream_final_project

# Install all required packages
pip install -r requirements.txt

# Install the project in development mode
pip install -e .
```

### Step 3: Verify Installation

```bash
# Test that KAF algorithms load correctly
python -c "from src.algorithms import KLMS, KNLMS, KAPA, KRLS; print('✅ KAF algorithms loaded')"

# Test that CapyMOA wrapper loads correctly
python -c "from src.stream import CapyMOARegressor; print('✅ CapyMOA wrapper loaded')"

# Test that data loading works
python -c "from src.data.stock_data import load_stock_data; print('✅ Data loader ready')"
```

### Required Packages (from requirements.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.24.0 | Numerical computing |
| pandas | >=2.0.0 | Data manipulation |
| scikit-learn | >=1.3.0 | ML utilities |
| matplotlib | >=3.7.0 | Visualization |
| river | >=0.18.0 | Online learning framework |
| **capymoa** | >=0.3.0 | Streaming ML library |
| yfinance | >=0.2.0 | Stock data download |
| pytest | >=7.4.0 | Testing framework |
| pytorch | >=2.0.0 | ML utilities |

### Troubleshooting

If you encounter Java errors with CapyMOA:
```bash
# CapyMOA requires Java. Install OpenJDK if needed:
sudo apt install openjdk-11-jdk  # Ubuntu/Debian
# or
conda install -c conda-forge openjdk=11
```

---

## ✅ How to Verify the Project (For Reviewers)

### 1. Run Unit Tests

```bash
cd /path/to/data_stream_final_project
conda activate DataStream

# Run all tests
python -m pytest tests/ -v

# Or run individual test files
python -m pytest tests/test_kaf.py -v
python -m pytest tests/test_capymoa_wrapper.py -v
python -m pytest tests/test_comparisons.py -v
```

### 2. Run Main Experiment (KAF vs CapyMOA)

```bash
cd experiments
conda activate DataStream

# Quick test with single stock
python capymoa_comparison.py --symbol AAPL --interval 1d

# Expected output: Comparison table showing KRLS with best MAE
```

### 3. Run Multi-Stock Experiment

```bash
cd experiments

# Test across 8 diverse stocks (takes ~2 minutes)
python multi_stock_experiment.py

# Expected output: KRLS wins on 7/8 stocks
```

### 4. Run Time Window Experiment

```bash
cd experiments

# Test different time intervals
python time_window_experiment.py --symbol AAPL

# Expected output: Results for 1d, 1h, 5m intervals
```

### 5. Generate Aggregated Results

```bash
cd experiments

# Aggregate all results into final summary
python aggregate_results.py

# Creates: results/final_summary.csv, results/final_report_plots.png
```

---

## 📊 Viewing Results

### CSV Results

| File | Description |
|------|-------------|
| `results/multi_stock_results.csv` | Detailed results for 7 algorithms × 8 stocks |
| `results/multi_stock_summary.csv` | Aggregated average performance |
| `results/final_summary.csv` | Master summary table |
| `results/time_window_results.csv` | Performance across intervals |
| `results/kaf_vs_capymoa_summary.csv` | Head-to-head comparison |

### Figures

| File | Description |
|------|-------------|
| `results/final_report_plots.png` | Publication-ready 4-panel summary |
| `results/multi_stock_results.png` | Multi-stock comparison with heatmap |
| `results/time_window_results.png` | Time interval comparison |
| `results/capymoa_comparison.png` | Single-stock detailed comparison |

---

## 🧪 Implemented Algorithms

### KAF Algorithms (Implemented by us)

| Algorithm | Description | Key Parameters |
|-----------|-------------|----------------|
| **KLMS** | Kernel Least Mean Squares | learning_rate=0.1 |
| **KNLMS** | Kernel Normalized LMS | learning_rate=0.1 |
| **KAPA** | Kernel Affine Projection | learning_rate=0.1, epsilon=0.1 |
| **KRLS** | Kernel Recursive Least Squares | forgetting_factor=0.99 |

### CapyMOA Baselines (Library)

| Algorithm | Description |
|-----------|-------------|
| **ARF** | Adaptive Random Forest Regressor |
| **KNN** | K-Nearest Neighbors Regressor |
| **SGBR** | Streaming Gradient Boosted Regression |

---

## 📈 Key Findings

1. **KRLS is the best algorithm** - Wins on 7/8 stocks tested
2. **KAF outperforms CapyMOA by 65%** on MAE metric
3. **Feature normalization is critical** - Without it, kernel methods fail
4. **Higher frequency data yields better directional accuracy** (63% at 5min vs 53% at daily)
5. **Results are above random baseline** (54% vs 50%)

---

## 📚 References

1. Mishra, S., et al. (2022). "An Online Kernel Adaptive Filtering-Based Approach for Mid-Price Prediction." *Scientific Programming*.
2. CapyMOA: https://capymoa.org/
3. River: https://riverml.xyz/

---

## 📄 Full Report

For complete methodology, results, and analysis, see **[REPORT.md](REPORT.md)**.

---

*Last updated: January 1, 2026*
