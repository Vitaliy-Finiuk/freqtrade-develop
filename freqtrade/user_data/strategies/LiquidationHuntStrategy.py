# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

import pandas as pd
from pandas import DataFrame
from typing import Optional
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import talib.abstract as ta


class LiquidationHuntStrategy(IStrategy):
    """
    Liquidation Hunt Strategy - Fixed version
    """
    
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = False

    # ROI configuration
    minimal_roi = {
        "0": 0.02,
        "30": 0.015,
        "60": 0.01,
        "120": 0.005
    }

    # Stoploss
    stoploss = -0.02

    # Trailing stop
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count = 100

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Volume analysis
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Price movements
        dataframe['price_change_3'] = (dataframe['close'] - dataframe['close'].shift(3)) / dataframe['close'].shift(3) * 100
        dataframe['price_change_6'] = (dataframe['close'] - dataframe['close'].shift(6)) / dataframe['close'].shift(6) * 100
        
        # Recovery detection
        dataframe['recovery_signal'] = (
            (dataframe['close'] > dataframe['open']) &  # Green candle
            (dataframe['close'] > dataframe['close'].shift(1))  # Higher than previous close
        )
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        
        # Bollinger Bands
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_lower'] = bb['lowerband']
        dataframe['bb_upper'] = bb['upperband']
        
        # EMA for trend
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        # Stochastic
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        # Candle patterns
        dataframe['lower_wick'] = (dataframe[['open', 'close']].min(axis=1) - dataframe['low']) / dataframe['close'] * 100
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Wait for recovery confirmation after drop
        """
        dataframe.loc[
            (
                # Price drop condition
                (dataframe['price_change_3'] < -1.5) &
                (dataframe['price_change_6'] < -2.5) &
                
                # Volume spike
                (dataframe['volume_ratio'] > 2.0) &
                
                # Oversold but not extreme
                (dataframe['rsi'] < 30) &
                (dataframe['rsi'] > 15) &
                (dataframe['stoch_k'] < 25) &
                
                # RECOVERY CONFIRMATION - MOST IMPORTANT
                (dataframe['recovery_signal']) &
                
                # Trend filter
                (dataframe['close'] > dataframe['ema_50'] * 0.9) &
                
                # Bollinger band position
                (dataframe['close'] <= dataframe['bb_lower'] * 1.02) &
                
                # Volume confirmation
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 60) |
                (dataframe['close'] >= dataframe['bb_upper'] * 0.98) |
                (dataframe['stoch_k'] > 80)
            ),
            'exit_long'
        ] = 1

        return dataframe

    def leverage(self, pair: str, current_time: pd.Timestamp, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], 
                 side: str, **kwargs) -> float:
        return 1.0