# Online Kernel Adaptive Filtering for Mid-Price Prediction

**Project 1 - Theme 4: REGRESSION FINANCE**

Implementation of the paper "An Online Kernel Adaptive Filtering-Based Approach for Mid-Price Prediction" (Mishra et al., 2022) using River and CapyMOA for online/streaming machine learning.

## Team Members
- [Add your names here]

## Project Overview

This project implements an online kernel adaptive filtering (KAF) approach for stock mid-price prediction in a streaming/online learning setting. The original paper proposes using KAF algorithms to predict stock price movements, which is particularly suitable for the non-stationary nature of financial time series.

### Key Features
- Implementation of multiple Kernel Adaptive Filtering algorithms
- Online/streaming learning compatible with River and CapyMOA
- Support for real-time financial data processing
- Evaluation on multiple time windows (1 min, 5 min, 10 min, etc.)
- Comparison with baseline streaming algorithms

## Paper Summary

**Title:** An Online Kernel Adaptive Filtering-Based Approach for Mid-Price Prediction  
**Authors:** Shambhavi Mishra, Tanveer Ahmed, Vipul Mishra, Sami Bourouis, Mohammad Aman Ullah  
**Published:** Scientific Programming, 2022

The paper proposes using kernel adaptive filtering for online stock price prediction, achieving 66% accuracy in predicting upward/downward movements. The approach is tested on Nifty-50 stocks with various time windows.

## Project Structure

```
ds_final_project/
├── src/
│   ├── algorithms/          # KAF algorithm implementations
│   ├── data/                # Data loading and preprocessing
│   ├── evaluation/          # Evaluation metrics and utilities
│   └── stream/              # River/CapyMOA integration
├── experiments/             # Experimental scripts and notebooks
├── tests/                   # Unit tests
├── data/                    # Dataset storage
├── results/                 # Experimental results
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from src.algorithms.kaf import KLMS
from src.stream.river_wrapper import KAFRegressor
from river import stream

# Create KAF-based regressor
model = KAFRegressor(algorithm='KLMS', learning_rate=0.1, kernel_size=0.5)

# Stream learning
for x, y in stream.iter_csv('data/stock_data.csv'):
    # Make prediction
    y_pred = model.predict_one(x)
    
    # Update model
    model.learn_one(x, y)
```

### Running Experiments

```bash
# Run single experiment
python experiments/run_kaf_experiment.py --algorithm KLMS --dataset nifty50

# Run full evaluation
python experiments/evaluate_all.py
```

## Implemented Algorithms

The project implements the following Kernel Adaptive Filtering algorithms:

1. **KLMS** (Kernel Least Mean Square)
2. **KNLMS** (Kernel Normalized LMS)
3. **KAPA** (Kernel Affine Projection Algorithm)
4. **KRLS** (Kernel Recursive Least Squares)
5. Additional KAF variants as needed

## Datasets

The project uses financial time-series data:
- Stock market data (OHLCV - Open, High, Low, Close, Volume)
- Multiple time windows: 1 day, 60 min, 30 min, 25 min, 20 min, 15 min, 10 min, 5 min, 1 min
- Features: Technical indicators, order book features, etc.

## Evaluation Metrics

- **Accuracy:** Percentage of correct directional predictions
- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Square Error
- **Prequential Evaluation:** Online evaluation for streaming data
- **Comparison:** Benchmarking against River/CapyMOA baseline algorithms

## Results

Results will be documented in the `results/` directory, including:
- Performance comparison across different KAF algorithms
- Analysis across different time windows
- Comparison with baseline streaming algorithms
- Visualization of predictions vs. actual values

## Report

The project report (minimum 5 pages) will include:
- Problem description and motivation
- Implementation details and adaptations
- Experimental setup and methodology
- Results and analysis
- Comparison with CapyMOA/River baselines
- Conclusions and future work

## References

1. Mishra, S., Ahmed, T., Mishra, V., Bourouis, S., & Ullah, M. A. (2022). An Online Kernel Adaptive Filtering-Based Approach for Mid-Price Prediction. Scientific Programming, 2022.

2. CapyMOA: https://capymoa.org/
3. River: https://riverml.xyz/

## License

[Specify your license]

## Contact

For questions about this project, contact:
- mariam.sa.barry@gmail.com
- maurras.togbe@isep.fr
- sathiyapkr@gmail.com
