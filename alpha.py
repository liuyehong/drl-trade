import numpy as np
import pandas as pd
import talib

class Alpha:

    def __init__(self, ohlc):
        df = ohlc
        self.open = df['Open'].values
        self.close = df['Close'].values
        self.high = df['High'].values
        self.low = df['Low'].values
        self.vol = df['Volume'].values
        self.upper = np.max([self.open, self.close], axis=0)
        self.lower = np.min([self.open, self.close], axis=0)
        self.T = len(self.close)


    def rolling_window(self, arr, window):
        shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
        strides = arr.strides + (arr.strides[-1],)
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    # alpha should have the same length with the data.

    def close_diff(self):
        signal = np.concatenate([[0], self.close[1:] - self.close[:-1]])
        return signal

    def open_diff(self):
        signal = np.concatenate([[0], self.open[1:] - self.open[:-1]])
        return signal

    def high_diff(self):
        signal = np.concatenate([[0], self.high[1:] - self.high[:-1]])
        return signal

    def low_diff(self):
        signal = np.concatenate([[0], self.low[1:] - self.low[:-1]])
        return signal

    def close_return(self):
        signal = np.concatenate([[0],
                                 (self.close[1:]-self.close[:-1])/self.close[:-1]])
        return signal

    def open_return(self):
        signal = np.concatenate([[0],
                                 (self.open[1:]-self.open[:-1])/self.open[:-1]])
        return signal

    def close_open_diff(self):
        signal = (self.close - self.open) / self.close
        return signal

    def upper_lower_diff(self):
        signal = (self.upper - self.lower) / self.close
        return signal

    def high_low_diff(self):
        signal = (self.high - self.low) / self.close
        return signal

    def high_upper_diff(self):
        signal = (self.high - self.upper) / self.close
        return signal

    def lower_low_diff(self):
        signal = (self.lower - self.low)/self.close
        return signal

    def moving_average(self, window=10):
        signal = np.concatenate([np.nan*np.ones(window-1), np.mean(self.rolling_window(self.close, window=window), axis=1)])
        return signal

    def moving_std(self, window=10):
        signal = np.concatenate(
            [np.nan * np.ones(window - 1), np.std(self.rolling_window(self.close, window=window), axis=1)])

        return signal

    def moving_var(self, window=10):
        signal = np.concatenate(
            [np.nan * np.ones(window - 1), np.var(self.rolling_window(self.close, window=window), axis=1)])

        return signal

    def moving_med(self, window=10):
        signal = np.concatenate(
            [np.nan * np.ones(window - 1), np.median(self.rolling_window(self.close, window=window), axis=1)])

        return signal

    def moving_max(self, window=10):
        signal = np.concatenate(
            [np.nan*np.ones(window-1), np.max(self.rolling_window(self.close, window=window), axis=1)])

        return signal

    def moving_min(self, window=10):
        signal = np.concatenate(
            [np.nan * np.ones(window - 1), np.min(self.rolling_window(self.close, window=window), axis=1)])

        return signal

    def moving_average_diff(self, window1=10, window2=20):
        signal1 = self.moving_average(window=window1)
        signal2 = self.moving_average(window=window2)
        signal = signal1-signal2
        return signal

    def bollinger_upper_bound(self, window=10, width=2):
        signal = self.moving_average(window=window) + width*self.moving_std(window=window)
        return signal

    def bollinger_lower_bound(self, window=10, width=2):
        signal = self.moving_average(window=window) - width * self.moving_std(window=window)
        return signal


    # TA-Lib
    def DEMA(self, window=30):
        real = talib.DEMA(self.close, timeperiod=window)
        return real

    def EMA(self, window=30):
        real = talib.EMA(close, timeperiod=window)
        return real

    def HT_TRENDLINE(self):
        real = talib.HT_TRENDLINE(self.close)
        return real

    def KAMA(self, window=30):
        real = talib.KAMA(self.close, timeperiod=window)
        return real

    def MA(self, window=30):
        real = talib.MA(self.close, timeperiod=window, matype=0)
        return real

    def MAMA(self):
        mama, fama = talib.MAMA(self.close, fastlimit=0, slowlimit=0)
        return mama, fama

    def MAVP(self, window1=2, window2=30):
        real = talib.MAVP(self.close, self.periods, minperiod=window1, maxperiod=window2, matype=0)
        return real


    def MIDPOINT(self, window=14):
        real = talib.MIDPOINT(close, timeperiod=window)
        return real

    def MIDPRICE(self, window=14):
        real = talib.MIDPRICE(high, low, timeperiod=window)
        return real

    def SAR(self):
        real = talib.SAR(self.high, self.low, acceleration=0, maximum=0)
        return real

    def SMA(self, window=30):
        real = talib.SMA(self.close, timeperiod=window)
        return real

    def T3(self, window=5):
        real = talib.T3(self.close, timeperiod=window, vfactor=0)
        return real

    def TEMA(self, window=30):
        real = talib.TEMA(self.close, timeperiod=window)
        return real

    def TRIMA(self, window=30):
        real = talib.TRIMA(self.close, timeperiod=window)
        return real

    def ADX(self, window=14):
        real = talib.ADX(self.high, self.low, self.close, timeperiod=window)
        return real

    def ADXR(self, window=14):
        real = talib.ADXR(self.high, self.low, self.close, timeperiod=window)
        return ADXR

    def APO(self, window1=12, window2=26):
        real = talib.APO(self.close, fastperiod=12, slowperiod=26, matype=0)
        return real

    def AROON(self, window=14):
        aroondown, aroonup = talib.AROON(self.high, self.low, timeperiod=window)
        return aroondown, aroonup

    def AROONOSC(self, window=14):
        real = talib.AROONOSC(self.high, self.low, timeperiod=window)
        return

    def BOP(self):
        real = talib.BOP(self.open, self.high, self.low, self.close)
        return real

    def CCI(self, window=14):
        real = talib.CCI(self.high, self.low, self.close, timeperiod=window)

    def CMP(self, window=14):
        real = talib.CMO(self.close, timeperiod=window)
        return real

    def DX(self, window=14):
        real = talib.DX(self.high, self.low, self.close, timeperiod=window)
        return real

    def MACD(self, window1=12, window2=26, window=9):
        macd, macdsignal, macdhist = talib.MACD(self.close, fastperiod=window1, slowperiod=window2, signalperiod=window)
        return macd, macdsignal, macdhist


    def MACDEXT(self, window1=12, window2=26, window=9):
        macd, macdsignal, macdhist = talib.MACDEXT(self.close, fastperiod=window1, fastmatype=0, slowperiod=window2, slowmatype=0,
                                             signalperiod=window, signalmatype=0)
        return macd, macdsignal, macdhist

    def MACDFIX(self, window=9):
        macd, macdsignal, macdhist = talib.MACDFIX(self.close, signalperiod=window)
        return macd, macdsignal, macdhist

    def MFI(self, window=14):
        real = talib.MFI(self.high, self.low, self.close, self.volume, timeperiod=window)
        return real

    def MINUS_DI(self, window=14):
        real = talib.MINUS_DI(self.high, self.low, self.close, timeperiod=window)
        return real

    def MINUS_DM(self, window=14):
        real = talib.MINUS_DM(self.high, self.low, timeperiod=window)
        return real

    def MOM(self, window=10):
        real = talib.MOM(self.close, timeperiod=window)
        return real

    def PLUS_DI(self, window=14):
        real = talib.PLUS_DI(self.high, self.low, self.close, timeperiod=window)
        return real

    def PLUS_DM(self, window=14):
        real = talib.PLUS_DM(self.high, self.low, timeperiod=window)
        return real

    def PPO(self, window1=12, window2=26):
        real = talib.PPO(self.close, fastperiod=window1, slowperiod=window2, matype=0)
        return real

    def ROC(self, window=10):
        real = talib.ROC(self.close, timeperiod=window)
        return real

    def ROCP(self, window=10):
        real = talib.ROCP(self.close, timeperiod=window)

    def ROCR(self, window=10):
        real = talib.ROCR(self.close, timeperiod=window)

    def ROCR100(self, window=10):
        real = talib.ROCR100(self.close, timeperiod=window)
        return real

    def RSI(self, window=14):
        real = talib.RSI(self.close, timeperiod=window)
        return real

    def STOCH(self,window1=5, window2=3, window=3):
        slowk, slowd = talib.STOCH(self.high, self.low, self.close, fastk_period=window1, slowk_period=window2, slowk_matype=0, slowd_period=window,
                             slowd_matype=0)
        return slowk, slowd

    def STOCHF(self, window1=5, window2=3):
        fastk, fastd = talib.STOCHF(self.high, self.low, self.close, fastk_period=window1, fastd_period=window2, fastd_matype=0)
        return fastk, fastd

    def STOCHRSI(self, window1=5, window2=3, window=14):
        fastk, fastd = talib.STOCHRSI(self.close, timeperiod=window, fastk_period=window1, fastd_period=window2, fastd_matype=0)
        return fastk, fastd

    def TRIX(self, window=30):
        real = talib.TRIX(self.close, timeperiod=window)
        return real

    def ULTOSC(self, window1=7, window2=14, window=28):
        real = talib.ULTOSC(self.high, self.low, self.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        return real

    def WILLR(self, window=14):
        real = talib.WILLR(self.high, self.low, self.close, timeperiod=window)
        return real

    def AD(self):
        real = talib.AD(self.high, self.low, self.close, self.vol)
        return real

    def ADOSC(self, window1=3, window2=10):
        real = talib.ADOSC(self.high, self.low, self.close, self.volume, fastperiod=window1, slowperiod=window2)
        return real

    def OBV(self):
        real = talib.OBV(self.close, self.volume)
        return real

    def ATR(self, window=14):
        real = talib.ATR(self.high, self.low, self.close, timeperiod=window)
        return real

    def NATR(self, window=14):
        real = talib.NATR(self.high, self.low, self.close, timeperiod=window)
        return real

    def TRANGE(self):
        real = talib.TRANGE(self.high, self.low, self.close)
        return real

    def AVGPRICE(self):
        real = talib.AVGPRICE(self.open, self.high, self.low, self.close)
        return real

    def MEDPRICE(self):
        real = talib.MEDPRICE(self.high, self.low)
        return real

    def TYPPRICE(self):
        real = talib.TYPPRICE(self.high, self.low, self.close)
        return real

    def WCLPRICE(self):
        real = talib.WCLPRICE(self.high, self.low, self.close)
        return real

    def HT_DCPERIOD(self):
        real = talib.HT_DCPERIOD(self.close)
        return real

    def HT_DCPHASE(self):
        real = talib.HT_DCPHASE(self.close)
        return real

    def HT_PHASOR(self):
        inphase, quadrature = talib.HT_PHASOR(self.close)
        return inphase, quadrature

    def HT_SINE(self):
        sine, leadsine = talib.HT_SINE(self.close)
        return sine, leadsine

    def HT_TRENDMODE(self):
        integer = talib.HT_TRENDMODE(self.close)
        return integer

    def CDL2CROWS(self):
        integer = talib.CDL2CROWS(self.open, self.high, self.low, self.close)
        return integer

    def CDL3BLACKCROWS(self):
        integer = talib.CDL3BLACKCROWS(self.open, self.high, self.low, self.close)
        return integer

    def CDL3INSIDE(self):
        integer = talib.CDL3INSIDE(self.open, self.high, self.low, self.close)
        return integer

    def CDL3LINESTRIKE(self):
        integer = talib.CDL3LINESTRIKE(self.open, self.high, self.low, self.close)
        return integer

    def CDL3OUTSIDE(self):
        integer = talib.CDL3OUTSIDE(self.open, self.high, self.low, self.close)
        return integer

    def CDL3STARSINSOUTH(self):
        integer = talib.CDL3STARSINSOUTH(self.open, self.high, self.low, self.close)
        return integer

    def CDL3WHITESOLDIERS(self):
        integer = talib.CDL3WHITESOLDIERS(self.open, self.high, self.low, self.close)
        return integer

    def CDLABANDONEDBABY(self):
        integer = talib.CDLABANDONEDBABY(self.open, self.high, self.low, self.close, penetration=0)
        return integer

    def CDLADVANCEBLOCK(self):
        integer = talib.CDLADVANCEBLOCK(self.open, self.high, self.low, self.close)
        return integer

    def CDLBELTHOLD(self):
        integer = talib.CDLBELTHOLD(self.open, self.high, self.low, self.close)
        return integer

    def CDLBREAKAWAY(self):
        integer = talib.CDLBREAKAWAY(self.open, self.high, self.low, self.close)
        return integer

    def CDLCLOSINGMARUBOZU(self):
        integer = talib.CDLCLOSINGMARUBOZU(self.open, self.high, self.low, self.close)
        return integer

    def CDLCONCEALBABYSWALL(self):
        integer = talib.CDLCONCEALBABYSWALL(self.open, self.high, self.low, self.close)
        return integer

    def CDLCOUNTERATTACK(self):
        integer = talib.CDLCOUNTERATTACK(self.open, self.high, self.low, self.close)
        return integer

    def CDLDARKCLOUDCOVER(self):
        integer = talib.CDLDARKCLOUDCOVER(self.open, self.high, self.low, self.close, penetration=0)
        return integer

    def CDLDOJI(self):
        integer = talib.CDLDOJI(self.open, self.high, self.low, self.close)
        return integer

    def CDLDOJISTAR(self):
        integer = talib.CDLDOJISTAR(self.open, self.high, self.low, self.close)
        return integer

    def CDLDRAGONFLYDOJI(self):
        integer = talib.CDLDRAGONFLYDOJI(self.open, self.high, self.low, self.close)
        return integer

    def CDLENGULFING(self):
        integer = talib.CDLENGULFING(self.open, self.high, self.low, self.close)
        return integer

    def CDLEVENINGDOJISTAR(self):
        integer = talib.CDLEVENINGDOJISTAR(self.open, self.high, self.low, self.close, penetration=0)
        return integer

    def CDLEVENINGSTAR(self):
        nteger = talib.CDLEVENINGSTAR(self.open, self.high, self.low, self.close, penetration=0)
        return nteger

    def CDLGAPSIDESIDEWHITE(self):
        integer = talib.CDLGAPSIDESIDEWHITE(self.open, self.high, self.low, self.close)
        return integer

    def CDLGRAVESTONEDOJI(self):
        integer = talib.CDLGRAVESTONEDOJI(self.open, self.high, self.low, self.close)
        return integer

    def CDLHAMMER(self):
        integer = talib.CDLHAMMER(self.open, self.high, self.low, self.close)
        return integer

    def CDLHANGINGMAN(self):
        integer = talib.CDLHANGINGMAN(self.open, self.high, self.low, self.close)
        return integer

    def CDLHARAMI(self):
        integer = talib.CDLHARAMI(self.open, self.high, self.low, self.close)
        return integer

    def CDLHARAMICROSS(self):
        integer = talib.CDLHARAMICROSS(self.open, self.high, self.low, self.close)
        return integer

    def CDLHIGHWAVE(self):
        integer = talib.CDLHIGHWAVE(self.open, self.high, self.low, self.close)
        return integer

    def CDLHIKKAKE(self):
        integer = talib.CDLHIKKAKE(self.open, self.high, self.low, self.close)
        return integer

    def CDLHIKKAKEMOD(self):
        integer = talib.CDLHIKKAKEMOD(self.open, self.high, self.low, self.close)
        return integer

    def CDLHOMINGPIGEON(self):
        integer = talib.CDLHOMINGPIGEON(self.open, self.high, self.low, self.close)
        return integer

    def CDLIDENTICAL3CROWS(self):
        integer = talib.CDLIDENTICAL3CROWS(self.open, self.high, self.low, self.close)
        return integer

    def CDLINNECK(self):
        integer = talib.CDLINNECK(self.open, self.high, self.low, self.close)
        return integer

    def CDLINVERTEDHAMMER(self):
        integer = talib.CDLINVERTEDHAMMER(self.open, self.high, self.low, self.close)
        return integer

    def CDLKICKING(self):
        integer = talib.CDLKICKING(self.open, self.high, self.low, self.close)
        return integer

    def CDLKICKINGBYLENGTH(self):
        integer = talib.CDLKICKINGBYLENGTH(self.open, self.high, self.low, self.close)
        return integer

    def CDLLADDERBOTTOM(self):
        integer = talib.CDLLADDERBOTTOM(self.open, self.high, self.low, self.close)
        return integer

    def CDLLONGLEGGEDDOJI(self):
        integer = talib.CDLLONGLEGGEDDOJI(self.open, self.high, self.low, self.close)
        return integer

    def CDLLONGLINE(self):
        integer = talib.CDLLONGLINE(self.open, self.high, self.low, self.close)
        return integer

    def CDLMARUBOZU(self):
        integer = talib.CDLMARUBOZU(self.open, self.high, self.low, self.close)
        return integer

    def CDLMATCHINGLOW(self):
        integer = talib.CDLMATCHINGLOW(self.open, self.high, self.low, self.close)
        return integer

    def CDLMATHOLD(self):
        integer = talib.CDLMATHOLD(self.open, self.high, self.low, self.close, penetration=0)
        return integer

    def CDLMORNINGDOJISTAR(self):
        integer = talib.DLMORNINGDOJISTAR(self.open, self.high, self.low, self.close, penetration=0)
        return integer

    def CDLMORNINGSTAR(self):
        integer = talib.CDLMORNINGSTAR(self.open, self.high, self.low, self.close, penetration=0)
        return integer

    def CDLONNECK(self):
        integer = talib.CDLONNECK(self.open, self.high, self.low, self.close)
        return integer

    def CDLPIERCING(self):
        integer = talib.CDLPIERCING(self.open, self.high, self.low, self.close)
        return integer

    def CDLRICKSHAWMAN(self):
        integer = talib.CDLRICKSHAWMAN(self.open, self.high, self.low, self.close)
        return integer

    def CDLRISEFALL3METHODS(self):
        integer = talib.CDLRISEFALL3METHODS(self.open, self.high, self.low, self.close)
        return integer

    def CDLSEPARATINGLINES(self):
        integer = talib.CDLSEPARATINGLINES(self.open, self.high, self.low, self.close)
        return integer

    def CDLSHOOTINGSTAR(self):
        integer = talib.CDLSHOOTINGSTAR(self.open, self.high, self.low, self.close)
        return integer

    def CDLSHORTLINE(self):
        integer = talib.CDLSHORTLINE(self.open, self.high, self.low, self.close)
        return integer

    def CDLSPINNINGTOP(self):
        integer = talib.CDLSPINNINGTOP(self.open, self.high, self.low, self.close)
        return integer

    def CDLSTALLEDPATTERN(self):
        integer = talib.CDLSTALLEDPATTERN(self.open, self.high, self.low, self.close)
        return integer

    def CDLSTICKSANDWICH(self):
        integer = talib.CDLSTICKSANDWICH(self.open, self.high, self.low, self.close)
        return integer

    def CDLTAKURI(self):
        integer = talib.CDLTAKURI(self.open, self.high, self.low, self.close)
        return integer

    def CDLTASUKIGAP(self):
        integer = talib.CDLTASUKIGAP(self.open, self.high, self.low, self.close)
        return integer

    def CDLTHRUSTING(self):
        integer = talib.CDLTHRUSTING(self.open, self.high, self.low, self.close)
        return integer

    def CDLTRISTAR(self):
        integer = talib.CDLTRISTAR(self.open, self.high, self.low, self.close)
        return integer

    def CDLUNIQUE3RIVER(self):
        integer = talib.CDLUNIQUE3RIVER(self.open, self.high, self.low, self.close)
        return integer

    def CDLUPSIDEGAP2CROWS(self):
        integer = talib.CDLUPSIDEGAP2CROWS(self.open, self.high, self.low, self.close)
        return integer

    def CDLXSIDEGAP3METHODS(self):
        integer = talib.CDLXSIDEGAP3METHODS(self.open, self.high, self.low, self.close)
        return integer

    def BETA(self, window=5):
        real = talib.BETA(self.high, self.low, timeperiod=window)
        return real

    def CORREL(self, window=30):
        real = talib.CORREL(self.high, self.low, timeperiod=window)
        return real

    def LINEARREG(self, window=14):
        real = talib.LINEARREG(self.close, timeperiod=window)
        return real

    def LINEARREG_ANGLE(self, window=14):
        real = talib.LINEARREG_ANGLE(self.close, timeperiod=window)
        return real

    def LINEARREG_INTERCEPT(self, window=14):
        real = talib.LINEARREG_INTERCEPT(self.close, timeperiod=window)
        return real

    def LINEARREG_SLOPE(self, window=14):
        real = talib.LINEARREG_SLOPE(self.close, timeperiod=window)
        return real

    def STDDEV(self, window=5):
        real = talib.STDDEV(self.close, timeperiod=window, nbdev=1)
        return real

    def TSF(self, window=14):
        real = talib.TSF(self.close, timeperiod=window)
        return real

    def VAR(self, window=5):
        real = talib.VAR(self.close, timeperiod=window, nbdev=1)
        return real



if __name__ == '__main__':
    df = pd.read_csv('./data/1Min/600008.csv')
    alpha = Alpha(df)
    print(alpha.TRIX())

