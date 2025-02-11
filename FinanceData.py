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
                        metricTimePeriod = "50d" ): #generate statistics and metrics to append to ticker dataframes
        self.metricTimePeriod = metricTimePeriod
        numeric_period = int(str(metricTimePeriod).replace("d","")) #TODO: need to make it so d can be any interval


        for ticker in self.tickers: #iterate through each ticker
            tickerdf_i = self.tickerDf[ticker] 

            for metric in metrics: #iterate through each metric and check if need to calculate

                if metric == "sma": #simple moving average 
                    tickerdf_i[("sma" + str(metricTimePeriod))] = tickerdf_i["Close"].rolling(window = metricTimePeriod).mean()
                    print("smadone")

                elif metric == "ema": #exponential moving average
                    tickerdf_i[("ema" + str(metricTimePeriod))] = tickerdf_i["Close"].ewm(span = numeric_period, adjust=False).mean()
            # print(tickerdf_i)


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
            plotvars[f"{col}{self.metricTimePeriod}"] = plotdf[f"{col}{self.metricTimePeriod}"]
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
#data.generatePlots
plotCols = ["sma", "ema"]
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
                     metricTimePeriod = metricTimePeriod)

#generate plots
data.generatePlots(cols = plotCols,
                   plotTicker = plotTicker)

###################################