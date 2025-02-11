import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from sklearn.model_selection import TimeSeriesSplit

####### CLASS DEFINITIONS #######

#define FinanceData class
class FinanceData:
    
    def __init__(self, tickers, timePeriod, timeInterval): #initial init for searching tickers
        self.tickers = tickers
        self.timePeriod = timePeriod
        self.timeInterval = timeInterval
    

    def importData(self): #info on tickers
        self.yfRawData = yf.Tickers(self.tickers)
    

    def downloadData(self): #download raw data and separate into ticker:dataframe self.tickerDf dictionary
        
        #download market data for tickers
        self.yfRawDataDl = yf.download(self.tickers, period = self.timePeriod)

        #separate raw data into each ticker
        self.tickerDf = {} #init tickerdf dict
        for ticker in self.tickers:#parse raw data and make ticker:marketDataFrame dictionary
            # print(ticker)
            tickerdf_i = self.yfRawDataDl.xs(key=ticker, level='Ticker', axis=1)
            # print(tickerdf_i)
            self.tickerDf[ticker] = tickerdf_i
        # print(self.tickerDf)
        print("Data downloaded and inside ticker:dataframe dict (self.tickerDf)")
    

    def generateMetrics(self, 
                        metrics = ["sma", "ema", "macd", "adx", "rsi"],
                        metricTimePeriod = "50d",
                        macdParams = [12, 26, 9], # fast, slow, signal
                        rsiParams = 14, #how long RSI period is
                        adxParams = 14 #how long ADX period is
                        ): #generate statistics and metrics to append to ticker dataframes
        self.metricTimePeriod = metricTimePeriod
        numeric_period = int(str(metricTimePeriod).replace("d","")) #TODO: need to make it so d can be any interval

        self.cutTail = max(macdParams + [rsiParams, adxParams])


        for ticker in self.tickers: #iterate through each ticker
            tickerdf_i = self.tickerDf[ticker] 

            if "sma" in metrics: #simple moving average 
                tickerdf_i[("sma" + str(metricTimePeriod))] = tickerdf_i["Close"].rolling(window = metricTimePeriod).mean()

            if "ema" in metrics: #exponential moving average
                tickerdf_i[("ema" + str(metricTimePeriod))] = tickerdf_i["Close"].ewm(span = numeric_period, adjust=False).mean()
        
            if "macd" in metrics: #moving average convergence/divergence: trend following momentum indicator
                macdFast = tickerdf_i["Close"].ewm(span = macdParams[0], adjust= False).mean()
                macdSlow = tickerdf_i["Close"].ewm(span = macdParams[1], adjust= False).mean()

                tickerdf_i["macd"] = macdFast - macdSlow
                tickerdf_i["macd_signal"] = tickerdf_i["macd"].ewm(span = macdParams[2], adjust= False).mean()
                tickerdf_i["macd_diff"] = tickerdf_i["macd"] - tickerdf_i["macd_signal"]
            
            if "rsi" in metrics: #relative strength index: momentum oscillator to measure speed and magnitude of recent prices changes compared to period
                delta = tickerdf_i["Close"].diff()
                gains = delta.clip(lower = 0)
                losses = -1 * delta.clip(upper = 0)

                #calculate averages.
                avgGain = gains.ewm(alpha=1/rsiParams, adjust = False).mean()
                avgLoss = losses.ewm(alpha=1/rsiParams, adjust = False).mean()

                rs = avgGain/avgLoss#relative strength
                rsi = 100 - (100 / (1+rs))

                tickerdf_i["rsi"] = rsi
                # print(rsi)

            if "adx" in metrics: #average directional index: measures strength of trend
                #collect highs, lows, closes, and prev values
                high = tickerdf_i["High"]
                low = tickerdf_i["Low"]
                close = tickerdf_i["Close"]
                highPrev = high.shift(1)
                lowPrev = low.shift(1)
                closePrev = close.shift(1)

                #calculate true range
                tr1 = high - low
                tr2 = (high - closePrev).abs()
                tr3 = (low - closePrev).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                #calculate +/- directional movements
                dmPlus = high - highPrev
                dmMinus = lowPrev - low
                #remove -ve values
                dmPlus[dmPlus < 0 ] = 0
                dmMinus[dmMinus < 0 ] = 0

                #average true rate
                avgTR = tr.ewm(alpha = 1/adxParams, adjust = False).mean()
                #smoothed +/- directional movements
                dmPlusEWM = dmPlus.ewm(alpha = 1/adxParams, adjust = False).mean()
                dmMinusEWM = dmMinus.ewm(alpha = 1/adxParams, adjust = False).mean()

                #calculate +/- directional index
                diPlus = 100 * (dmPlusEWM / avgTR)
                diMinus = 100 * (dmMinusEWM / avgTR)

                #calculate directional index DX
                dx = 100 * ((diPlus - diMinus).abs() / (diPlus + diMinus).abs())

                #calculate average directional index ADX
                adx = dx.ewm(alpha=1/adxParams, adjust=False).mean()
                
                #place adx and +/- DI into tickerDf
                tickerdf_i["adx"] = adx
                tickerdf_i["+di"] = diPlus
                tickerdf_i["-di"] = diMinus


    def generatePlots(self,
                      cols = ["sma"],
                      plotTicker = None,
                      pairplot = 1): #generate plots with date on y axis, and chosen cols on x axis
        
        if plotTicker == None:#for default take first ticker in tickers list
            plotTicker = self.tickers[0]

        plotdf = self.tickerDf[plotTicker] 
        #gather columns for plotting
        plotvars = {}
        for col in cols:
            plotvars["Close"] = plotdf["Close"] 
            if col in ["sma", "ema"]: #include period for sma and ema
                plotvars[f"{col}{self.metricTimePeriod}"] = plotdf[f"{col}{self.metricTimePeriod}"]
            else:
                plotvars[f"{col}"] = plotdf[f"{col}"]
        plotvars = pd.DataFrame(plotvars,
                                 index = plotdf.index)
        plotvars.plot()
        plt.show()

        if pairplot == 1:
            sns.pairplot(data = self.tickerDf[plotTicker]) # plot pairplot
            #plot correlation matrix
            corrmatrix = self.tickerDf[plotTicker].corr()
            sns.heatmap(corrmatrix, annot = True)
        
    
    def kFoldTimeSeries(self,
                        kSplits= 5,
                        maxTrainSize = None,
                        testSize = None,
                        gapSize = 0,
                        predictionPeriod = 1, # how many invervals in future that classifier will predict direction on
                        tickerKFold = None
                        ): #split data for kfold cross validation using TimeSeriesSplit
        
        if tickerKFold == None:
            tickerKFold = self.tickers[0]

        #get target values according to predictionPeriod
        tickerdf_i = self.tickerDf[tickerKFold].copy()
        tickerdf_i["target"] = tickerdf_i["Close"].shift(predictionPeriod)

        y = tickerdf_i["target"] # target is close price after specific predictionPeriod
        X = tickerdf_i.drop(columns=["Close", "High", "Low", "Open"]) # predicotrs are everything except these cols

        tss = TimeSeriesSplit(n_splits = kSplits, max_train_size=maxTrainSize, test_size=testSize, gap=gapSize)#init tss from scikit learn

        #now split and store indexes in self.kFoldIdx
        self.kFoldIdx = enumerate(tss.split(X))
        self.kFoldX = X
        self.kFoldy = y
        
        # print(self.kFoldIdx)
        # for i, (x, y) in self.kFoldIdx:
        #     print(i)
        #     print(x)
        #     print(y)



###################################






####### SETTINGS #######
#data.downloadData
stock_tickers = ["SPY", "GOOG", "AAPL", "MSFT", "NVDA"]
timePeriod = "1y"
timeInterval = "1d"
#data.generateMetrics
metrics = ["sma", "ema", "macd", "adx", "rsi"]
metricTimePeriod = "50d"
macdParams = [12, 26, 9] # 12 fast, 26 slow, 9 signal
rsiParams = 14
adxParams =14
pairplot = 0
#data.generatePlots
plotCols = ["sma", "ema", "macd", "macd_signal", "rsi", "adx"]
plotTicker = None
#data.kFoldTimeSeries
kSplits = 5
maxTrainSize = None
testSize = None
gapSize = 0
predictionPeriod = 1
tickerKFold = None

####### PIPELINE #######

#create the FinanceData object
data = FinanceData(tickers=stock_tickers, 
                   timePeriod=timePeriod,
                   timeInterval = timeInterval)

#import the data and display info
data.importData()
data.yfRawData.tickers["SPY"].info

#download market data
data.downloadData()

#generate statistics and metrics
data.generateMetrics(metrics = metrics,
                     metricTimePeriod = metricTimePeriod,
                     macdParams= macdParams,
                     rsiParams=rsiParams,
                     adxParams=adxParams)

#generate plots
data.generatePlots(cols = plotCols,
                   plotTicker = plotTicker,
                   pairplot = pairplot)

#split data via TimeSeriesSplit
data.kFoldTimeSeries(kSplits = kSplits,
                     maxTrainSize = maxTrainSize,
                     testSize= testSize,
                     gapSize=gapSize,
                     predictionPeriod= predictionPeriod,
                     tickerKFold = None)

###################################