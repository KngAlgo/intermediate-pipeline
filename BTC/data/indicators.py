"""
Technical Indicators for Bitcoin Data

This module adds technical analysis indicators to your OHLCV data.
These indicators help the model understand market momentum, trends, and sentiment.
"""

import pandas as pd
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange


def add_technical_indicators(df):
    """
    Adds technical indicators to a DataFrame with OHLCV data

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional indicator columns

    Indicators Added:
    -----------------
    1. RSI (Relative Strength Index) - Momentum oscillator
    2. MACD (Moving Average Convergence Divergence) - Trend following
    3. MACD Signal - Signal line for MACD
    4. MACD Histogram - Difference between MACD and Signal
    5. SMA_20 - Simple Moving Average (20 days)
    6. SMA_50 - Simple Moving Average (50 days)
    7. EMA_12 - Exponential Moving Average (12 days)
    8. Bollinger Bands (Upper, Middle, Lower) - Volatility
    9. ATR - Average True Range (volatility measure)
    10. Stochastic Oscillator - Momentum indicator
    """

    df = df.copy()

    # ==========================================
    # 1. RSI - Relative Strength Index
    # ==========================================
    # Measures speed and magnitude of price changes
    # Values: 0-100
    # > 70 = Overbought (might fall)
    # < 30 = Oversold (might rise)
    rsi = RSIIndicator(close=df['Close'], window=336)
    df['RSI'] = rsi.rsi()

    # ==========================================
    # 2. MACD - Moving Average Convergence Divergence
    # ==========================================
    # Shows relationship between two moving averages
    # When MACD crosses above Signal = Bullish
    # When MACD crosses below Signal = Bearish
    macd = MACD(
        close=df['Close'],
        window_slow=624,    # 26-day EMA
        window_fast=288,    # 12-day EMA
        window_sign=216      # 9-day Signal line
    )
    df['MACD'] = macd.macd()                    # MACD line
    df['MACD_Signal'] = macd.macd_signal()      # Signal line
    df['MACD_Histogram'] = macd.macd_diff()     # Histogram (MACD - Signal)

    # ==========================================
    # 3. Moving Averages
    # ==========================================
    # Smooth out price data to identify trends
    # SMA = Simple Moving Average (equal weights)
    # EMA = Exponential Moving Average (recent prices weighted more)

    # Short-term trend (20 days)
    sma_20 = SMAIndicator(close=df['Close'], window=480)
    df['SMA_20'] = sma_20.sma_indicator()

    # Medium-term trend (50 days)
    sma_50 = SMAIndicator(close=df['Close'], window=1200)
    df['SMA_50'] = sma_50.sma_indicator()

    # Exponential moving average (responds faster to price changes)
    ema_12 = EMAIndicator(close=df['Close'], window=288)
    df['EMA_12'] = ema_12.ema_indicator()

    # ==========================================
    # 5. ATR - Average True Range
    # ==========================================
    # Measures market volatility
    # High ATR = High volatility (big price swings)
    # Low ATR = Low volatility (stable prices)
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=336)
    df['ATR'] = atr.average_true_range()

    # ==========================================
    # 6. Stochastic Oscillator
    # ==========================================
    # Compares closing price to price range over time
    # Values: 0-100
    # > 80 = Overbought
    # < 20 = Oversold
    stoch = StochasticOscillator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14,
        smooth_window=3
    )
    df['Stochastic'] = stoch.stoch()
    df['Stochastic_Signal'] = stoch.stoch_signal()

    # ==========================================
    # 7. Price-based Features
    # ==========================================
    # Additional features derived from OHLC

    # Daily return (percentage change)
    df['Returns'] = df['Close'].pct_change()

    # Price range (High - Low)
    df['Range'] = df['High'] - df['Low']

    # Body size (Close - Open) - shows candle strength
    df['Body'] = df['Close'] - df['Open']

    # Upper shadow (High - max(Open, Close))
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)

    # Lower shadow (min(Open, Close) - Low)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    # ==========================================
    # Clean up NaN values
    # ==========================================
    # Technical indicators need historical data, so first rows will be NaN
    # We'll drop them (you need ~50 days of history for all indicators)
    print(f"Rows before cleaning: {len(df)}")
    df = df.dropna()
    print(f"Rows after cleaning: {len(df)}")
    print(f"Dropped {len(df) - len(df.dropna())} rows with NaN values")

    return df


def get_feature_list():
    """
    Returns list of all features (original + indicators)
    Useful for knowing what to scale and feed to the model
    """
    original_features = ['Open', 'High', 'Low', 'Close', 'Volume']

    indicator_features = [
        'RSI',
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        'SMA_20', 'SMA_50', 'EMA_12',
        'ATR',
        'Stochastic', 'Stochastic_Signal',
        'Returns', 'Range', 'Body', 'Upper_Shadow', 'Lower_Shadow'
    ]

    return original_features, indicator_features


if __name__ == '__main__':
    """
    Test the indicators on real BTC data
    """
    from api import get_train_data

    print("Fetching BTC data...")
    data = get_train_data()

    print("\nOriginal data shape:", data.shape)
    print("Original columns:", data.columns.tolist())

    print("\nAdding technical indicators...")
    data_with_indicators = add_technical_indicators(data)

    print("\nNew data shape:", data_with_indicators.shape)
    print("New columns:", data_with_indicators.columns.tolist())

    print("\nSample of indicator values (last 5 rows):")
    print(data_with_indicators.tail())

    print("\n" + "="*50)
    print("INDICATOR SUMMARY")
    print("="*50)
    orig, ind = get_feature_list()
    print(f"Original features: {len(orig)}")
    print(f"Indicator features: {len(ind)}")
    print(f"Total features: {len(orig) + len(ind)}")
