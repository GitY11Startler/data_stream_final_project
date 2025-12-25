"""
Financial data loading and preprocessing utilities.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta


def load_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Load stock data using yfinance.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval ('1m', '5m', '15m', '1h', '1d', etc.)
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")
    
    # Download data
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False
    )
    
    return df


def calculate_mid_price(df: pd.DataFrame) -> pd.Series:
    """
    Calculate mid-price from OHLC data.
    
    Mid-price = (High + Low) / 2
    
    Args:
        df: DataFrame with High and Low columns
        
    Returns:
        Series with mid-prices
    """
    return (df['High'] + df['Low']) / 2


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for stock data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional technical indicator columns
    """
    df = df.copy()
    
    # Returns
    df['returns'] = df['Close'].pct_change()
    
    # Moving averages
    for window in [5, 10, 20]:
        df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # Volatility (rolling standard deviation)
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    
    # Volume indicators
    df['volume_sma_10'] = df['Volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_10']
    
    # Price momentum
    df['momentum_5'] = df['Close'].diff(5)
    df['momentum_10'] = df['Close'].diff(10)
    
    # Rate of change
    df['roc_5'] = df['Close'].pct_change(5)
    df['roc_10'] = df['Close'].pct_change(10)
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    return df


def prepare_features_target(
    df: pd.DataFrame,
    target_column: str = 'mid_price',
    feature_columns: Optional[List[str]] = None,
    forecast_horizon: int = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for machine learning.
    
    Args:
        df: DataFrame with stock data and indicators
        target_column: Name of the target column
        feature_columns: List of feature column names (None = auto-select)
        forecast_horizon: How many steps ahead to predict
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    df = df.copy()
    
    # Calculate mid-price if not present
    if 'mid_price' not in df.columns and 'High' in df.columns and 'Low' in df.columns:
        df['mid_price'] = calculate_mid_price(df)
    
    # Create target (future mid-price)
    df['target'] = df[target_column].shift(-forecast_horizon)
    
    # Auto-select features if not provided
    if feature_columns is None:
        # Exclude non-feature columns
        exclude_cols = ['target', 'Date', 'Datetime', 'Adj Close']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    # Drop rows with NaN values
    df_clean = df[feature_columns + ['target']].dropna()
    
    X = df_clean[feature_columns]
    y = df_clean['target']
    
    return X, y


def create_directional_target(df: pd.DataFrame, column: str = 'Close') -> pd.Series:
    """
    Create a binary directional target (up=1, down=0).
    
    Args:
        df: DataFrame with price data
        column: Column name to use for direction
        
    Returns:
        Series with binary direction (1=up, 0=down)
    """
    return (df[column].diff().shift(-1) > 0).astype(int)


def normalize_features(X: pd.DataFrame, method: str = 'standardize') -> pd.DataFrame:
    """
    Normalize features.
    
    Args:
        X: Feature DataFrame
        method: Normalization method ('standardize', 'minmax', 'none')
        
    Returns:
        Normalized DataFrame
    """
    if method == 'standardize':
        return (X - X.mean()) / X.std()
    elif method == 'minmax':
        return (X - X.min()) / (X.max() - X.min())
    elif method == 'none':
        return X
    else:
        raise ValueError(f"Unknown normalization method: {method}")


class StreamingStockDataset:
    """
    Iterator for streaming stock data simulation.
    
    This class simulates a streaming environment by yielding
    one sample at a time from historical data.
    """
    
    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d',
        forecast_horizon: int = 1,
        add_indicators: bool = True
    ):
        """
        Initialize streaming dataset.
        
        Args:
            symbol: Stock ticker
            start_date: Start date
            end_date: End date
            interval: Time interval
            forecast_horizon: Prediction horizon
            add_indicators: Whether to add technical indicators
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.forecast_horizon = forecast_horizon
        self.add_indicators = add_indicators
        
        # Load data
        self.df = load_stock_data(symbol, start_date, end_date, interval)
        
        if add_indicators:
            self.df = calculate_technical_indicators(self.df)
        
        # Calculate mid-price
        self.df['mid_price'] = calculate_mid_price(self.df)
        
        # Prepare features and target
        self.X, self.y = prepare_features_target(
            self.df,
            target_column='mid_price',
            forecast_horizon=forecast_horizon
        )
        
        self.feature_names = self.X.columns.tolist()
    
    def __iter__(self):
        """Iterate over samples."""
        for idx in range(len(self.X)):
            x = self.X.iloc[idx].to_dict()
            y = self.y.iloc[idx]
            yield x, y
    
    def __len__(self):
        """Return number of samples."""
        return len(self.X)


def generate_sample_data(n_samples: int = 1000, n_features: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic streaming data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        
    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(42)
    
    # Generate correlated features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with nonlinear relationship
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.1
    
    return X, y
