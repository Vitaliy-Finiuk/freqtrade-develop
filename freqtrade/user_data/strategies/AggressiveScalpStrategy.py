# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
    AnnotationType,
)

import talib.abstract as ta
from technical import qtpylib


class AggressiveScalpStrategy(IStrategy):
    """
    High-performance momentum strategy with tight risk control.
    Focus: Quick entries on strong momentum, cut losses fast, let winners run.
    """
    
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short: bool = False

    # Aggressive ROI - take profits on momentum
    minimal_roi = {
        "0": 0.04,     # 4% immediate target
        "20": 0.025,   # 2.5% after 20 min
        "40": 0.015,   # 1.5% after 40 min
        "80": 0.008    # 0.8% after 80 min
    }

    # TIGHT stop loss - cut losers fast!
    stoploss = -0.018  # 1.8% stop - быстро режем убытки

    # Aggressive trailing
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.008   # Trail at 0.8%
    trailing_stop_positive_offset = 0.015  # Offset 1.5%

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False  # Exit on any strong reversal signal
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 50

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }

    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "ema_fast": {"color": "blue"},
                "ema_slow": {"color": "red"},
            },
            "subplots": {
                "RSI": {"rsi": {"color": "red"}},
                "MACD": {
                    "macd": {"color": "blue"},
                    "macdsignal": {"color": "orange"},
                }
            }
        }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Fast indicators for momentum detection"""
        
        # Fast EMAs for trend
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_trend"] = ta.EMA(dataframe, timeperiod=50)

        # RSI for momentum
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        
        # Fast RSI for entries
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=7)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )

        # ADX for trend strength
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # Volume
        dataframe["volume_mean"] = dataframe["volume"].rolling(window=20).mean()

        # Price momentum
        dataframe["price_change"] = (dataframe["close"] - dataframe["close"].shift(5)) / dataframe["close"].shift(5) * 100

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enter on strong momentum with multiple confirmations.
        Strategy: Buy breakouts with volume in established uptrends.
        """
        dataframe.loc[
            (
                # TREND: Above 50 EMA (overall uptrend)
                (dataframe["close"] > dataframe["ema_trend"]) &
                
                # MOMENTUM: Fast EMA above slow (short-term uptrend)
                (dataframe["ema_fast"] > dataframe["ema_slow"]) &
                
                # MOMENTUM: RSI showing strength but not overbought
                (dataframe["rsi"] > 50) &
                (dataframe["rsi"] < 75) &
                
                # MOMENTUM: Fast RSI confirming
                (dataframe["rsi_fast"] > 45) &
                
                # MACD: Bullish momentum
                (dataframe["macd"] > dataframe["macdsignal"]) &
                (dataframe["macdhist"] > 0) &
                (dataframe["macdhist"] > dataframe["macdhist"].shift(1)) &  # Increasing
                
                # ADX: Strong trend
                (dataframe["adx"] > 20) &
                
                # ENTRY TIMING: Pullback or breakout
                (
                    # Option 1: Pullback to lower BB in uptrend
                    (
                        (dataframe["bb_percent"] < 0.3) &
                        (dataframe["close"] > dataframe["close"].shift(1))  # Starting to bounce
                    ) |
                    # Option 2: Breakout above middle BB with momentum
                    (
                        (dataframe["close"] > dataframe["bb_middleband"]) &
                        (dataframe["close"].shift(1) <= dataframe["bb_middleband"].shift(1)) &
                        (dataframe["price_change"] > 0.3)  # Strong price momentum
                    )
                ) &
                
                # VOLUME: Above average
                (dataframe["volume"] > dataframe["volume_mean"] * 1.1) &
                
                (dataframe["volume"] > 0)
            ),
            "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit on momentum reversal - protect profits quickly.
        """
        dataframe.loc[
            (
                (
                    # Momentum weakening: Fast EMA crosses below slow
                    (
                        (dataframe["ema_fast"] < dataframe["ema_slow"]) &
                        (dataframe["ema_fast"].shift(1) >= dataframe["ema_slow"].shift(1))
                    ) |
                    
                    # RSI showing weakness
                    (
                        (dataframe["rsi"] < 45) |
                        (dataframe["rsi"] > 80)  # Extreme overbought
                    ) |
                    
                    # MACD bearish cross
                    (
                        (dataframe["macd"] < dataframe["macdsignal"]) &
                        (dataframe["macd"].shift(1) >= dataframe["macdsignal"].shift(1))
                    ) |
                    
                    # Price rejecting upper BB (resistance)
                    (
                        (dataframe["close"] < dataframe["bb_upperband"]) &
                        (dataframe["high"] >= dataframe["bb_upperband"]) &
                        (dataframe["close"] < dataframe["open"])  # Red candle after hitting resistance
                    )
                ) &
                (dataframe["volume"] > 0)
            ),
            "exit_long"] = 1

        return dataframe