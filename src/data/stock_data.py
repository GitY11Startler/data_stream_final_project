"""
Financial data loading and preprocessing utilities.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import os
import hashlib
import warnings


# Cache directory for downloaded stock data
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'cache')


# Interval compatibility with yfinance
INTERVAL_LIMITS = {
    '1m': {'max_days': 7, 'name': '1 minute'},
    '2m': {'max_days': 60, 'name': '2 minutes'},
    '5m': {'max_days': 60, 'name': '5 minutes'},
    '15m': {'max_days': 60, 'name': '15 minutes'},
    '30m': {'max_days': 60, 'name': '30 minutes'},
    '60m': {'max_days': 730, 'name': '60 minutes'},
    '1h': {'max_days': 730, 'name': '1 hour'},
    '1d': {'max_days': None, 'name': '1 day'},
    '1wk': {'max_days': None, 'name': '1 week'},
    '1mo': {'max_days': None, 'name': '1 month'},
}


# Pre-configured stock lists for experiments
STOCK_LISTS = {
    'tech': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
    'finance': ['JPM', 'BAC', 'GS', 'MS', 'C'],
    'mixed': ['AAPL', 'JPM', 'TSLA', 'WMT', 'DIS'],
    'popular': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']
}


def _create_cache_dir():
    """Create cache directory if it doesn't exist."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def _get_cache_filename(symbol: str, start_date: str, end_date: str, interval: str) -> str:
    """
    Generate cache filename from parameters.
    
    Args:
        symbol: Stock ticker
        start_date: Start date
        end_date: End date
        interval: Time interval
        
    Returns:
        Cache filename
    """
    # Create hash from parameters
    params_str = f"{symbol}_{start_date}_{end_date}_{interval}"
    cache_key = hashlib.md5(params_str.encode()).hexdigest()[:12]
    
    return f"{symbol}_{interval}_{cache_key}.csv"


def validate_date_interval(start_date: str, end_date: str, interval: str) -> Tuple[bool, str]:
    """
    Validate if date range is compatible with interval.
    
    yfinance has limitations on historical data availability for smaller intervals.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Time interval
        
    Returns:
        Tuple of (is_valid, message)
    """
    if interval not in INTERVAL_LIMITS:
        return False, f"Invalid interval '{interval}'. Valid intervals: {list(INTERVAL_LIMITS.keys())}"
    
    max_days = INTERVAL_LIMITS[interval]['max_days']
    
    if max_days is None:
        return True, "Date range is valid"
    
    # Calculate date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    days_diff = (end - start).days
    
    if days_diff > max_days:
        interval_name = INTERVAL_LIMITS[interval]['name']
        return False, (
            f"Date range ({days_diff} days) exceeds maximum for {interval_name} data ({max_days} days). "
            f"For {interval}, use dates within the last {max_days} days."
        )
    
    return True, "Date range is valid"


def adjust_dates_for_interval(interval: str, end_date: Optional[str] = None) -> Tuple[str, str]:
    """
    Automatically adjust date range based on interval limitations.
    
    Args:
        interval: Time interval
        end_date: End date (None = today)
        
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    if end_date is None:
        end = datetime.now()
    else:
        end = datetime.strptime(end_date, '%Y-%m-%d')
    
    max_days = INTERVAL_LIMITS.get(interval, {}).get('max_days')
    
    if max_days is None:
        # No limit, use 1 year of data
        start = end - timedelta(days=365)
    else:
        # Use maximum available range (minus 1 day for safety)
        start = end - timedelta(days=max_days - 1)
    
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def load_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = '1d',
    use_cache: bool = True,
    validate_dates: bool = True
) -> pd.DataFrame:
    """
    Load stock data using yfinance with caching support.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval ('1m', '5m', '15m', '1h', '1d', etc.)
        use_cache: Whether to use cached data
        validate_dates: Whether to validate date/interval compatibility
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")
    
    # Validate date range if requested
    if validate_dates:
        is_valid, message = validate_date_interval(start_date, end_date, interval)
        if not is_valid:
            warnings.warn(message)
            # Auto-adjust dates
            start_date, end_date = adjust_dates_for_interval(interval, end_date)
            warnings.warn(f"Auto-adjusted dates to: {start_date} to {end_date}")
    
    # Check cache if enabled
    if use_cache:
        _create_cache_dir()
        cache_file = os.path.join(CACHE_DIR, _get_cache_filename(symbol, start_date, end_date, interval))
        
        if os.path.exists(cache_file):
            # Load from cache
            try:
                df = pd.read_csv(cache_file, index_col=0)
                df.index = pd.to_datetime(df.index)
                if not df.empty:
                    return df
            except Exception as e:
                warnings.warn(f"Failed to load cache: {e}. Re-downloading...")
    
    # Download data
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False
    )
    
    # Save to cache if enabled and data is not empty
    if use_cache and not df.empty:
        try:
            cache_file = os.path.join(CACHE_DIR, _get_cache_filename(symbol, start_date, end_date, interval))
            df.to_csv(cache_file)
        except Exception as e:
            warnings.warn(f"Failed to save cache: {e}")
    
    return df


def load_multiple_stocks(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1d',
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple stocks.
    
    Args:
        symbols: List of stock ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval
        use_cache: Whether to use cached data
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    results = {}
    
    for symbol in symbols:
        try:
            df = load_stock_data(
                symbol,
                start_date,
                end_date,
                interval=interval,
                use_cache=use_cache
            )
            
            if not df.empty:
                results[symbol] = df
            else:
                warnings.warn(f"No data retrieved for {symbol}")
        
        except Exception as e:
            warnings.warn(f"Failed to load {symbol}: {e}")
    
    return results


def load_stock_list(
    list_name: str,
    start_date: str,
    end_date: str,
    interval: str = '1d',
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Load data for a pre-configured stock list.
    
    Args:
        list_name: Name of stock list ('tech', 'finance', 'mixed', 'popular')
        start_date: Start date
        end_date: End date
        interval: Time interval
        use_cache: Whether to use cached data
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    if list_name not in STOCK_LISTS:
        raise ValueError(
            f"Unknown stock list '{list_name}'. "
            f"Available lists: {list(STOCK_LISTS.keys())}"
        )
    
    symbols = STOCK_LISTS[list_name]
    return load_multiple_stocks(symbols, start_date, end_date, interval, use_cache)


def clear_cache(symbol: Optional[str] = None):
    """
    Clear cached stock data.
    
    Args:
        symbol: Specific symbol to clear (None = clear all)
    """
    if not os.path.exists(CACHE_DIR):
        return
    
    files_removed = 0
    
    for filename in os.listdir(CACHE_DIR):
        if symbol is None or filename.startswith(f"{symbol}_"):
            filepath = os.path.join(CACHE_DIR, filename)
            try:
                os.remove(filepath)
                files_removed += 1
            except Exception as e:
                warnings.warn(f"Failed to remove {filename}: {e}")
    
    print(f"Removed {files_removed} cache file(s)")


def get_cache_info() -> pd.DataFrame:
    """
    Get information about cached files.
    
    Returns:
        DataFrame with cache information
    """
    if not os.path.exists(CACHE_DIR):
        return pd.DataFrame()
    
    cache_info = []
    
    for filename in os.listdir(CACHE_DIR):
        filepath = os.path.join(CACHE_DIR, filename)
        
        # Get file stats
        stat = os.stat(filepath)
        size_mb = stat.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(stat.st_mtime)
        
        cache_info.append({
            'filename': filename,
            'size_mb': size_mb,
            'modified': modified
        })
    
    return pd.DataFrame(cache_info).sort_values('modified', ascending=False)


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
