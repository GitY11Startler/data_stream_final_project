# Quick Reference Card

## Installation
```bash
cd /home/movithepawy11/ds_final_project
source venv/bin/activate
pip install -r requirements.txt
```

## Run Examples

### 1. Simple Synthetic Data Test
```bash
cd experiments
python simple_example.py
```
**Output**: `results/simple_example_results.png`

### 2. Stock Price Prediction
```bash
python stock_prediction.py --symbol AAPL --start 2023-01-01 --end 2024-01-01 --algorithm KLMS
```
**Arguments**:
- `--symbol`: Stock ticker (AAPL, GOOGL, MSFT, etc.)
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--interval`: Time interval (1m, 5m, 15m, 1h, 1d)
- `--algorithm`: KAF algorithm (KLMS, KNLMS, KAPA, KRLS)

### 3. Interactive Tutorial
```bash
jupyter notebook tutorial.ipynb
```

## Run Tests
```bash
cd tests
pytest test_kaf.py -v
```

## Code Examples

### Basic KAF Usage
```python
from src.algorithms import KLMS
import numpy as np

model = KLMS(learning_rate=0.1, kernel='gaussian', kernel_size=1.0)

# Online learning
for x, y in data:
    y_pred = model.predict(x)
    model.update(x, y)
```

### River Integration
```python
from src.stream import KAFRegressor
from river import metrics

model = KAFRegressor(algorithm='KLMS', learning_rate=0.1)
metric = metrics.MAE()

for x, y in stream_data:
    y_pred = model.predict_one(x)
    metric.update(y, y_pred)
    model.learn_one(x, y)

print(f"MAE: {metric.get():.4f}")
```

### Stock Data Loading
```python
from src.data import load_stock_data, calculate_technical_indicators

df = load_stock_data('AAPL', '2023-01-01', '2024-01-01', '1d')
df = calculate_technical_indicators(df)
```

### Evaluation
```python
from src.evaluation import prequential_evaluation, evaluate_directional_accuracy_online
from river import metrics

# Standard metrics
results, df = prequential_evaluation(
    model, stream_data,
    metrics_list=[metrics.MAE(), metrics.RMSE()],
    verbose=True
)

# Directional accuracy (paper's main metric)
acc, history = evaluate_directional_accuracy_online(model, stream_data)
print(f"Directional Accuracy: {acc:.2%}")
```

## File Structure
```
ds_final_project/
├── src/
│   ├── algorithms/       # KAF implementations
│   ├── stream/           # River wrappers
│   ├── data/            # Data loading
│   └── evaluation/      # Metrics
├── experiments/         # Example scripts
├── tests/              # Unit tests
├── results/            # Outputs
└── data/               # Datasets
```

## Algorithm Comparison

| Algorithm | Speed | Accuracy | Memory | Use Case |
|-----------|-------|----------|--------|----------|
| KLMS      | Fast  | Good     | Low    | General purpose |
| KNLMS     | Fast  | Better   | Low    | Varying scales |
| KAPA      | Medium| Good     | Medium | Fast convergence |
| KRLS      | Slow  | Best     | High   | Best accuracy |

## Hyperparameters

### Learning Rate
- **Range**: 0.01 - 0.5
- **Default**: 0.1
- **Effect**: Higher = faster adaptation, less stable

### Kernel Size
- **Range**: 0.5 - 5.0
- **Default**: 1.0
- **Effect**: Controls smoothness of kernel function

### Max Dictionary Size
- **Range**: 50 - 500
- **Default**: 200
- **Effect**: Higher = better accuracy, more memory

### Novelty Threshold
- **Range**: 0.01 - 0.5
- **Default**: 0.1
- **Effect**: Lower = more dictionary updates

## Common Commands

```bash
# Activate environment
source venv/bin/activate

# Run tests
pytest tests/test_kaf.py -v

# Run simple example
python experiments/simple_example.py

# Run stock prediction
python experiments/stock_prediction.py --symbol AAPL --algorithm KLMS

# Start Jupyter
jupyter notebook experiments/tutorial.ipynb

# Check results
ls -lh results/

# Deactivate environment
deactivate
```

## Documentation Files

- **README.md** - Main project documentation
- **GETTING_STARTED.md** - Installation and setup guide
- **PROJECT_SUMMARY.md** - Implementation details and next steps
- **QUICK_REFERENCE.md** - This file (command reference)

## Paper Metrics to Replicate

1. **Directional Accuracy**: ~66% (paper's result on Nifty-50)
2. **Time Windows**: 1min, 5min, 10min, 15min, 20min, 25min, 30min, 60min, 1day
3. **Stocks**: Test on 50 stocks (paper used Nifty-50 index)

## For Your Report

### Experiments to Run
1. Multiple stocks across different time windows
2. Comparison with River/CapyMOA baselines
3. Hyperparameter sensitivity analysis
4. Performance under different market conditions

### Visualizations to Include
1. Predictions vs. actual prices
2. Error distributions
3. Directional accuracy over time
4. Algorithm comparison bar charts
5. Performance across time windows

### Tables to Include
1. Overall performance comparison
2. Per-stock results
3. Time window analysis
4. Computational efficiency metrics

## Contact

**Instructors**:
- mariam.sa.barry@gmail.com
- maurras.togbe@isep.fr
- sathiyapkr@gmail.com

## Submission Requirements

✅ Documented source code (ZIP + GitHub)
✅ Report (5+ pages)
✅ PowerPoint slides
✅ Demo preparation (20 min total: 13 min presentation + 7 min Q&A)
