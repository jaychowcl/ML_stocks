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
        self.yfRawData = yf.download(self.tickers, period = self.timePeriod)

    def sep(self):
        #separate raw data into each ticker
        for ticker in self.tickers:
            print(ticker)


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
data.yfRawData

#separate data into tickerdata dataframe
data.sep()

###################################