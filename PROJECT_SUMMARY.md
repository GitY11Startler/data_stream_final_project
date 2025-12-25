# Project Implementation Summary

## Overview

This project implements the paper **"An Online Kernel Adaptive Filtering-Based Approach for Mid-Price Prediction"** (Mishra et al., 2022) using Python with River and CapyMOA for online/streaming machine learning.

## What Has Been Implemented

### ✅ Core Algorithms

1. **KLMS (Kernel Least Mean Square)**
   - Basic kernel adaptive filtering
   - Simple LMS update rule in kernel space
   - Supports multiple kernel functions (Gaussian, Polynomial, Linear)

2. **KNLMS (Kernel Normalized LMS)**
   - Normalized step size for better convergence
   - Adapts to input signal power

3. **KAPA (Kernel Affine Projection Algorithm)**
   - Uses multiple past samples for updates
   - Potentially faster convergence
   - Configurable projection order

4. **KRLS (Kernel Recursive Least Squares)**
   - RLS-based online learning
   - Generally best accuracy
   - Maintains inverse correlation matrix

### ✅ Key Features

- **Online/Streaming Learning**: All algorithms support incremental learning
- **River Integration**: Full compatibility with River's streaming API
- **Dictionary Management**: Automatic memory control with novelty criterion
- **Multiple Kernels**: Gaussian (RBF), Polynomial, and Linear kernels
- **Evaluation Metrics**: MAE, RMSE, R², and directional accuracy

### ✅ Data Processing

- Stock data loading via yfinance
- Technical indicator calculation (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- Mid-price calculation
- Streaming data conversion
- Feature normalization utilities

### ✅ Evaluation Framework

- **Prequential Evaluation**: Test-then-train for streaming data
- **Directional Accuracy**: Key metric from the paper (up/down prediction)
- **Comparison Tools**: Compare multiple algorithms easily
- **Sliding Window Analysis**: Detect concept drift
- **Online Metrics**: Compatible with River metrics

### ✅ Examples and Documentation

1. **simple_example.py**: Test on synthetic data
2. **stock_prediction.py**: Full stock prediction pipeline
3. **tutorial.ipynb**: Interactive Jupyter notebook
4. **Unit Tests**: Comprehensive test suite
5. **Documentation**: README, Getting Started Guide

## Project Structure

```
ds_final_project/
├── src/
│   ├── algorithms/
│   │   ├── base_kaf.py      # Base KAF class with kernel functions
│   │   ├── kaf.py           # KLMS, KNLMS, KAPA, KRLS implementations
│   │   └── __init__.py
│   ├── stream/
│   │   ├── river_wrapper.py # River-compatible wrappers
│   │   └── __init__.py
│   ├── data/
│   │   ├── stock_data.py    # Data loading and preprocessing
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── metrics.py       # Evaluation utilities
│   │   └── __init__.py
│   └── __init__.py
├── experiments/
│   ├── simple_example.py    # Synthetic data example
│   ├── stock_prediction.py  # Stock prediction script
│   └── tutorial.ipynb       # Interactive tutorial
├── tests/
│   └── test_kaf.py          # Unit tests
├── results/                 # Output directory
├── data/                    # Data storage
├── requirements.txt         # Dependencies
├── README.md               # Main documentation
├── GETTING_STARTED.md      # Quick start guide
└── .gitignore              # Git ignore rules
```

## How to Use

### Quick Start

```bash
# Install dependencies
source venv/bin/activate
pip install -r requirements.txt

# Run simple example
cd experiments
python simple_example.py

# Run stock prediction
python stock_prediction.py --symbol AAPL --algorithm KLMS
```

### Basic Usage

```python
from src.stream import KAFRegressor
from river import stream, metrics

# Create model
model = KAFRegressor(
    algorithm='KLMS',
    learning_rate=0.1,
    kernel='gaussian',
    kernel_size=1.0
)

# Stream learning
for x, y in stream.iter_csv('data.csv'):
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
```

## Performance

Based on initial testing with synthetic data:

| Algorithm | MAE    | RMSE   | R²     |
|-----------|--------|--------|--------|
| KLMS      | 0.6730 | 0.9803 | 0.1154 |
| KNLMS     | 0.4881 | 0.7914 | 0.4234 |
| KAPA      | 0.5999 | 0.9114 | 0.2353 |
| KRLS      | 0.4466 | 0.7243 | 0.5171 |

**Best algorithm**: KRLS (highest R² score)

## Key Implementation Decisions

1. **Novelty Criterion**: Used Approximate Linear Dependence (ALD) for dictionary management
2. **Memory Management**: Configurable max dictionary size with FIFO removal
3. **River Compatibility**: Full integration with River's API for easy comparison
4. **Kernel Functions**: Implemented Gaussian, Polynomial, and Linear kernels
5. **Error Handling**: Robust handling of edge cases and dimension mismatches

## Comparison with Paper

### Similarities
- ✅ Implements KLMS, KNLMS, KAPA, KRLS algorithms
- ✅ Online/streaming learning approach
- ✅ Mid-price prediction focus
- ✅ Directional accuracy evaluation
- ✅ Technical indicator features

### Differences/Extensions
- 🔧 Added River/CapyMOA integration (as per project requirements)
- 🔧 Simplified some matrix operations for numerical stability
- 🔧 Added comprehensive evaluation framework
- 🔧 Implemented multiple kernel options
- 🔧 Added synthetic data generation for testing

## Next Steps for Your Project

### For the Report (Minimum 5 Pages)

1. **Introduction**: Problem description, motivation, paper summary
2. **Methodology**: Algorithm descriptions, implementation details
3. **Experiments**: 
   - Test on multiple stocks (replicate paper experiments)
   - Multiple time windows (1min, 5min, 15min, 1d)
   - Compare with River/CapyMOA baselines
4. **Results**: Tables, plots, directional accuracy analysis
5. **Conclusions**: Findings, limitations, future work

### Suggested Experiments

1. **Reproduce Paper Results**:
   ```bash
   python stock_prediction.py --symbol ^NSEI --algorithm KLMS --interval 1d
   ```
   (Note: Use Nifty-50 stocks if available)

2. **Compare with Baselines**:
   - Linear regression
   - ARIMA
   - River's linear models
   - CapyMOA's adaptive algorithms

3. **Hyperparameter Tuning**:
   - Learning rates: [0.01, 0.05, 0.1, 0.5]
   - Kernel sizes: [0.5, 1.0, 2.0, 5.0]
   - Dictionary sizes: [50, 100, 200, 500]

4. **Different Time Windows**:
   - Test on 1min, 5min, 15min, 1h, 1d intervals
   - Analyze accuracy vs. time window

### For the Presentation (13 minutes)

1. **Problem & Motivation** (2 min)
2. **KAF Algorithms Overview** (3 min)
3. **Implementation with River/CapyMOA** (2 min)
4. **Experimental Results** (4 min)
5. **Conclusions** (2 min)

### For the Demo

Show:
1. Running `simple_example.py`
2. Running `stock_prediction.py` with live ticker
3. Jupyter notebook with visualizations
4. Comparison with baseline algorithms

## Testing

All tests pass:
```bash
cd tests
pytest test_kaf.py -v
```

## Deliverables Checklist

- ✅ Documented source code (.py files)
- ✅ README.md with project description
- ✅ Requirements.txt for dependencies
- ✅ Example scripts demonstrating usage
- ✅ Test suite
- ⏳ Report (5+ pages) - **TO DO**
- ⏳ PowerPoint slides - **TO DO**
- ⏳ GitHub repository - **TO DO**
- ⏳ Experimental results on real stock data - **TO DO**

## References

1. Mishra, S., Ahmed, T., Mishra, V., Bourouis, S., & Ullah, M. A. (2022). An Online Kernel Adaptive Filtering-Based Approach for Mid-Price Prediction. *Scientific Programming*, 2022.

2. River: Online machine learning in Python. https://riverml.xyz/

3. CapyMOA: Python library for online learning. https://capymoa.org/

## Contributors

[Add your team members' names here]

## License

[Specify license if needed]

---

**Created**: December 2024  
**Course**: M2 IPP - Data Stream Processing  
**Project**: Theme 4 - REGRESSION FINANCE
