"""Technical indicators for trading analysis"""

import numpy as np
import pandas as pd

def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """
    Calculate the Relative Strength Index (RSI).
    
    Args:
        prices (np.ndarray): Array of price data
        period (int): RSI period (default: 14)
        
    Returns:
        float: RSI value between 0 and 100
    """
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down if down != 0 else 0
    rsi = 100 - (100/(1+rs))
    return rsi

def calculate_macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """
    Calculate the Moving Average Convergence Divergence (MACD).
    
    Args:
        prices (np.ndarray): Array of price data
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period
        
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    fast_ema = pd.Series(prices).ewm(span=fast_period, adjust=False).mean()
    slow_ema = pd.Series(prices).ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices (np.ndarray): Array of price data
        period (int): Moving average period
        num_std (float): Number of standard deviations
        
    Returns:
        tuple: (Upper band, Middle band, Lower band)
    """
    middle_band = pd.Series(prices).rolling(window=period).mean()
    std = pd.Series(prices).rolling(window=period).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    return upper_band.iloc[-1], middle_band.iloc[-1], lower_band.iloc[-1] 