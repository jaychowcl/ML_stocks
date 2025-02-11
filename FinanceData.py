import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

        self.cutTail = max(macdParams + [rsiParams, adxParams]) # identify period that need to be cut due to insufficient past data TODO: cut these periods from data



        for ticker in self.tickers: #iterate through each ticker
            tickerdf_i = self.tickerDf[ticker] 

            #calculate delta and %delta
            tickerdf_i["delta"] = self.tickerDf[ticker]["Close"].diff()
            tickerdf_i["deltaPerc"] = 100 * (tickerdf_i["delta"] / tickerdf_i["Close"].copy().shift(1))


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


            #trim tail of dataset to remove NaN metrics and clean
            tickerdf_i.drop(tickerdf_i.index[:self.cutTail], inplace = True)


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
        
    
    def kFoldSplitTimeSeries(self,
                        kSplits= 5,
                        maxTrainSize = None,
                        testSize = None,
                        gapSize = 0,
                        predictionPeriod = 1, # how many invervals in future that classifier will predict direction on
                        tickerKFold = None,
                        yTarget = "targetDirection"
                        ): #split data for kfold cross validation using TimeSeriesSplit
        
        if tickerKFold == None:
            tickerKFold = self.tickers[0]

        #get target values according to predictionPeriod
        tickerdf_i = self.tickerDf[tickerKFold]
        tickerdf_i["target"] = tickerdf_i["deltaPerc"].copy().shift(predictionPeriod)
        tickerdf_i["targetDirection"] = [1 if x >=0 else -1 for x in tickerdf_i["target"]]

        #make targetDirection the target if needed for binary classifiers
        if yTarget == "targetDirection":
            tickerdf_i["target"] = tickerdf_i["targetDirection"]

        #remove first row to remove NaN target since will always be in future
        tickerdf_i.drop(tickerdf_i.index[:1], inplace = True)


        y = tickerdf_i["target"] # target is close price after specific predictionPeriod
        X = tickerdf_i.drop(columns=["Close", "High", "Low", "Open", "target", "targetDirection"]) # predicotrs are everything except these cols

        tss = TimeSeriesSplit(n_splits = kSplits, max_train_size=maxTrainSize, test_size=testSize, gap=gapSize)#init tss from scikit learn

        #now split and store indexes in self.kFoldIdx
        #store split indexes into k:(train, test) dict
        self.kFoldIdx = {}
        for i, (trainIdx, testIdx) in enumerate(tss.split(X)):
            self.kFoldIdx[i] = {"trainIdx": trainIdx, "testIdx": testIdx}

        self.kFoldX = X
        self.kFoldy = y
        
        # print(self.kFoldIdx)
        # for i, (x, y) in self.kFoldIdx:
        #     print(i)
        #     print(x)
        #     print(y)

    def runStrategy(self, 
                    strategy = ["logistic"],
                    X_train = None,
                    y_train = None,
                    X_test = None,
                    y_test = None
                    ): #input training data and testing predictors to output prediction
        
        if X_train is None or y_train is None or X_test is None:
            raise ValueError("ERROR: Please specify training and testing datasets")

        #scale data to mean 0 and var 1
        scaler = StandardScaler()
        scaler.fit(X_train) #train scaler with training predictors

        #scale data for relevant methods
        X_train_Scale = scaler.transform(X_train)
        X_test_Scale = scaler.transform(X_test)

        if "logistic" in strategy: # logistic regression with L2 ridge regression regularization and liblinear solver
            print("Logistic regression:")
            lr = LogisticRegression(fit_intercept= True, 
                                    solver = "liblinear", 
                                    penalty = "l2") # first init logistic regression model
            lr.fit(X_train_Scale, y_train) # train model

            #use lr model to predict test cases
            y_predict = lr.predict(X_test_Scale)

            if y_test is None: # return only predictions if without y test
                self.predictions = {"y_predict": y_predict, "scores": "NA"}
                return {"y_predict": y_predict, "scores": "NA"}
            else: # return predictions and accuracy score with y test
                y_predict_score = lr.score(X_test_Scale, y_test)
                self.predictions = {"y_predict": y_predict, "scores": y_predict_score}
                return {"y_predict": y_predict, "scores": y_predict_score}
            
    
    def kFoldValidation(self,
                        ):
        pass





        

        

        



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
#data.kFoldSplitTimeSeries
kSplits = 5
maxTrainSize = None
testSize = None
gapSize = 0
predictionPeriod = 1
tickerKFold = None
yTarget = "targetDirection"
#data.runStrategy
strategy = ["logistic"]
X_train = None
y_train = None
X_test = None
y_test = None

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
data.kFoldSplitTimeSeries(kSplits = kSplits,
                          maxTrainSize = maxTrainSize,
                          testSize= testSize,
                          gapSize=gapSize,
                          predictionPeriod= predictionPeriod,
                          tickerKFold = None,
                          yTarget= yTarget)

#run a strategy and predict/evaluate target
k_folds = 3
data.runStrategy(strategy = strategy,
                 X_train = data.kFoldX.iloc[data.kFoldIdx[k_folds]["trainIdx"]],
                 y_train = data.kFoldy.iloc[data.kFoldIdx[k_folds]["trainIdx"]],
                 X_test = data.kFoldX.iloc[data.kFoldIdx[k_folds]["testIdx"]],
                 y_test = data.kFoldy.iloc[data.kFoldIdx[k_folds]["testIdx"]]
                 )

#kfold cross validation

###################################