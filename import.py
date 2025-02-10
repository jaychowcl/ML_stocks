import yfinance as yf
import pandas as pd
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
    
    def downloadData(self): #download market data for tickers
        self.yfRawDataDl = yf.download(self.tickers, period = self.timePeriod)
        #separate raw data into each ticker
        self.tickerDf = {} #init tickerdf dict

        for ticker in self.tickers:#parse raw data and make ticker:marketDataFrame dictionar
            # print(ticker)
            tickerdf_i = self.yfRawDataDl.xs(key=ticker, level='Ticker', axis=1)
            # print(tickerdf_i)
            self.tickerDf[ticker] = tickerdf_i

        # print(self.tickerDf)
    
    def generateMetrics(self): #


###################################






####### SETTINGS #######
stock_tickers = ["SPY", "GOOG", "AAPL", "MSFT", "NVDA"]
timePeriod = "1mo"
timeInterval = "1d"

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


###################################