from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class MinimalPionexStrategy(IStrategy):
    timeframe = '5m'
    minimal_roi = {"0": 0.01}  # 1% ROI
    stoploss = -0.02           # стоплосс 2%
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['sma50'] > dataframe['sma200']),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['sma50'] < dataframe['sma200']),
            'sell'] = 1
        return dataframe
