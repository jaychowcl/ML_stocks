import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


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
                        rsiParams = 14 #how long RSI period is
                        ): #generate statistics and metrics to append to ticker dataframes
        self.metricTimePeriod = metricTimePeriod
        numeric_period = int(str(metricTimePeriod).replace("d","")) #TODO: need to make it so d can be any interval


        for ticker in self.tickers: #iterate through each ticker
            tickerdf_i = self.tickerDf[ticker] 

            if "sma" in metrics: #simple moving average 
                tickerdf_i[("sma" + str(metricTimePeriod))] = tickerdf_i["Close"].rolling(window = metricTimePeriod).mean()
                print("smadone")

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

                #calculate averages. use ewm instead of wilder smoothing to achieve basically the same result
                avgGain = gains.ewm(alpha=1/rsiParams, adjust = False).mean()
                avgLoss = losses.ewm(alpha=1/rsiParams, adjust = False).mean()

                rs = avgGain/avgLoss#relative strength
                rsi = 100 - (100 / (1+rs))

                tickerdf_i["rsi"] = rsi
                print(rsi)

            

    def generatePlots(self,
                      cols = ["sma"],
                      plotTicker = None): #generate plots with date on y axis, and chosen cols on x axis
        
        if plotTicker == None:#for default take first ticker in tickers list
            plotTicker = self.tickers[0]

        plotdf = self.tickerDf[plotTicker] 
        #gather columns for plotting
        plotvars = {}
        for col in cols:
            plotvars["Close"] = plotdf["Close"] 
            if col in ["sma", "ema"]:
                plotvars[f"{col}{self.metricTimePeriod}"] = plotdf[f"{col}{self.metricTimePeriod}"]
            else:
                plotvars[f"{col}"] = plotdf[f"{col}"]
        plotvars = pd.DataFrame(plotvars,
                                 index = plotdf.index)
        plotvars.plot()
        plt.show()
        


        print("end")


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
#data.generatePlots
plotCols = ["sma", "ema", "macd", "macd_signal"]
plotTicker = None

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
                     rsiParams=rsiParams)

#generate plots
data.generatePlots(cols = plotCols,
                   plotTicker = plotTicker)

###################################