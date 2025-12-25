# Getting Started Guide

## Quick Start

This guide will help you get up and running with the Online Kernel Adaptive Filtering project.

## Installation

### 1. Clone the Repository

```bash
cd /path/to/ds_final_project
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running Examples

### Example 1: Simple Synthetic Data

Test the KAF algorithms on synthetic data:

```bash
cd experiments
python simple_example.py
```

This will:
- Generate synthetic nonlinear data
- Train KLMS, KNLMS, KAPA, and KRLS algorithms
- Compare their performance
- Save results to `results/simple_example_results.png`

### Example 2: Stock Price Prediction

Run stock prediction experiment:

```bash
cd experiments
python stock_prediction.py --symbol AAPL --start 2023-01-01 --end 2024-01-01 --algorithm KLMS
```

Available arguments:
- `--symbol`: Stock ticker (e.g., AAPL, GOOGL, MSFT)
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--interval`: Time interval (1d, 1h, 5m, etc.)
- `--algorithm`: KAF algorithm (KLMS, KNLMS, KAPA, KRLS)

### Example 3: Interactive Jupyter Notebook

```bash
cd experiments
jupyter notebook tutorial.ipynb
```

## Basic Usage

### Using KAF with River

```python
from src.stream import KAFRegressor
from river import stream, metrics

# Create model
model = KAFRegressor(
    algorithm='KLMS',
    learning_rate=0.1,
    kernel='gaussian',
    kernel_size=1.0,
    max_dictionary_size=200
)

# Stream learning
metric = metrics.MAE()

for x, y in stream.iter_csv('data.csv'):
    # Predict
    y_pred = model.predict_one(x)
    
    # Update metric
    metric.update(y, y_pred)
    
    # Learn
    model.learn_one(x, y)

print(f"Final MAE: {metric.get():.4f}")
```

### Using KAF Directly

```python
from src.algorithms import KLMS
import numpy as np

# Create model
model = KLMS(
    learning_rate=0.1,
    kernel='gaussian',
    kernel_size=1.0
)

# Generate data
X = np.random.randn(100, 5)
y = np.sin(X[:, 0]) + 0.5 * X[:, 1]

# Online learning
for i in range(len(X)):
    # Predict
    y_pred = model.predict(X[i])
    
    # Learn
    model.update(X[i], y[i])
```

## Running Tests

```bash
cd tests
pytest test_kaf.py -v
```

## Project Structure

```
ds_final_project/
├── src/
│   ├── algorithms/       # KAF algorithm implementations
│   │   ├── base_kaf.py   # Base class for KAF
│   │   └── kaf.py        # KLMS, KNLMS, KAPA, KRLS
│   ├── stream/           # River/CapyMOA integration
│   │   └── river_wrapper.py
│   ├── data/             # Data loading utilities
│   │   └── stock_data.py
│   └── evaluation/       # Evaluation metrics
│       └── metrics.py
├── experiments/          # Example scripts
│   ├── simple_example.py
│   ├── stock_prediction.py
│   └── tutorial.ipynb
├── tests/               # Unit tests
└── results/             # Output directory
```

## Common Tasks

### Load Stock Data

```python
from src.data import load_stock_data, calculate_technical_indicators

df = load_stock_data(
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    interval='1d'
)

df = calculate_technical_indicators(df)
```

### Evaluate Multiple Algorithms

```python
from src.evaluation import compare_algorithms

algorithms = {
    'KLMS': KAFRegressor(algorithm='KLMS'),
    'KNLMS': KAFRegressor(algorithm='KNLMS'),
}

results = compare_algorithms(algorithms, stream_data)
print(results)
```

### Calculate Directional Accuracy

```python
from src.evaluation import evaluate_directional_accuracy_online

accuracy, history = evaluate_directional_accuracy_online(
    model, stream_data, verbose=True
)
print(f"Directional Accuracy: {accuracy:.2%}")
```

## Tips and Best Practices

### Hyperparameter Tuning

1. **Learning Rate**: Start with 0.1, increase for faster adaptation
2. **Kernel Size**: Adjust based on data scale (0.5-2.0 typical range)
3. **Max Dictionary Size**: Balance memory vs. accuracy (100-500 typical)
4. **Novelty Threshold**: Lower = more dictionary updates (0.01-0.5)

### Choosing an Algorithm

- **KLMS**: Simple, fast, good starting point
- **KNLMS**: Better for varying input scales
- **KAPA**: Faster convergence, more memory
- **KRLS**: Best accuracy, higher computational cost

### Performance Tips

1. Normalize features before training
2. Use warm start period for initial training
3. Monitor dictionary size growth
4. Use smaller kernel size for high-dimensional data

## Troubleshooting

### ImportError: No module named 'river'

```bash
pip install river
```

### yfinance download fails

- Check internet connection
- Verify stock ticker symbol is correct
- Try different date range

### Out of Memory

- Reduce `max_dictionary_size`
- Increase `novelty_threshold`
- Process data in smaller batches

## Next Steps

1. Read the [full paper](Scientific%20Programming%20-%202022%20-%20Mishra%20-%20An%20Online%20Kernel%20Adaptive%20Filtering‐Based%20Approach%20for%20Mid‐Price%20Prediction.pdf)
2. Experiment with different datasets
3. Try different time windows (1min, 5min, etc.)
4. Compare with CapyMOA baseline algorithms
5. Optimize hyperparameters for your use case

## References

- River Documentation: https://riverml.xyz/
- CapyMOA Documentation: https://capymoa.org/
- Paper: Mishra et al., "An Online Kernel Adaptive Filtering-Based Approach for Mid-Price Prediction", Scientific Programming, 2022

## Getting Help

For questions or issues:
- Check the README.md
- Review the tutorial notebook
- Contact project instructors:
  - mariam.sa.barry@gmail.com
  - maurras.togbe@isep.fr
  - sathiyapkr@gmail.com
