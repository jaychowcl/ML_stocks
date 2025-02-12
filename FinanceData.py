import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix


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
                        predictionPeriod = 1, # how many invervals in future the mode will predict y on
                        tickerKFold = None
                        ): #split data for kfold cross validation using TimeSeriesSplit
        
        if tickerKFold is None:
            tickerKFold = self.tickers[0]

        #get target values according to predictionPeriod
        tickerdf_i = self.tickerDf[tickerKFold]
        tickerdf_i["target"] = tickerdf_i["deltaPerc"].copy().shift(predictionPeriod)
        tickerdf_i["targetDirection"] = [1 if x >=0 else -1 for x in tickerdf_i["target"]]

        #make targetDirection the target if needed for binary classifiers
        tickerdf_i["target"] = tickerdf_i["targetDirection"]

        #remove first row to remove NaN target since will always be in future
        tickerdf_i.drop(tickerdf_i.index[:1], inplace = True)


        y = tickerdf_i["target"] # target is close price after specific predictionPeriod
        X = tickerdf_i.drop(columns=["Close", "High", "Low", "Open", "target", "targetDirection", "delta", "deltaPerc"]) # predicotrs are everything except these cols

        tss = TimeSeriesSplit(n_splits = kSplits, max_train_size=maxTrainSize, test_size=testSize, gap=gapSize)#init tss from scikit learn

        #now split and store indexes in self.kFoldIdx
        #store split indexes into k:(train, test) dict
        self.kFoldIdx = {}
        for i, (trainIdx, testIdx) in enumerate(tss.split(X)):
            self.kFoldIdx[i] = {"trainIdx": trainIdx, "testIdx": testIdx}

        self.kFoldX = X
        self.kFoldy = y
        self.kFoldDf = tickerdf_i[["Close", "High", "Low", "Open", "target", "targetDirection", "delta", "deltaPerc"]]
        
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

        #store scaler
        self.scaler = scaler

        #scale data for relevant methods
        X_train_Scale = scaler.transform(X_train)
        X_test_Scale = scaler.transform(X_test)

        if "logistic" in strategy: # logistic regression with L2 ridge regression regularization and liblinear solver
            # print("Logistic regression:")
            lr = LogisticRegression(fit_intercept= True, 
                                    solver = "liblinear", 
                                    penalty = "l2") # first init logistic regression model
            lr.fit(X_train_Scale, y_train) # train model
            #store model
            self.model = lr

            #use lr model to predict test cases
            y_predict = lr.predict(X_test_Scale)

            if y_test is None: # return only predictions if without y test
                self.predictions = {"y_predict": y_predict, "accuracy": "NA"}
            else: # return predictions and accuracy score with y test
                y_predict_score = lr.score(X_test_Scale, y_test)
                self.predictions = {"y_predict": y_predict, "accuracy": y_predict_score}
            
        elif "svm" in strategy: #support vector machine with radial basis function kernel 
            svm = SVC(kernel = "rbf")
            svm.fit(X_train_Scale, y_train) # train model

            self.model = svm

            #use svm model to predict test cases
            y_predict = svm.predict(X_test_Scale)

            if y_test is None: # return only predictions if without y test
                self.predictions = {"y_predict": y_predict, "accuracy": "NA"}
            else: # return predictions and accuracy score with y test
                y_predict_score = svm.score(X_test_Scale, y_test)
                self.predictions = {"y_predict": y_predict, "accuracy": y_predict_score}

        elif "dt" in strategy: #decision trees
            dt = DecisionTreeClassifier(criterion= "entropy")
            dt.fit(X_train, y_train)

            self.model = dt

            y_predict = dt.predict(X_test) # no need to scale since dt uses feature threshods

            if y_test is None: # return only predictions if without y test
                self.predictions = {"y_predict": y_predict, "accuracy": "NA"}
            else: # return predictions and accuracy score with y test
                y_predict_score = dt.score(X_test, y_test)
                self.predictions = {"y_predict": y_predict, "accuracy": y_predict_score}

        elif "rf" in strategy: #random forest
            rf = RandomForestClassifier(n_estimators=100, criterion= "entropy")
            rf.fit(X_train, y_train)

            self.model = rf

            y_predict = rf.predict(X_test)

            if y_test is None: # return only predictions if without y test
                self.predictions = {"y_predict": y_predict, "accuracy": "NA"}
            else: # return predictions and accuracy score with y test
                y_predict_score = rf.score(X_test, y_test)
                self.predictions = {"y_predict": y_predict, "accuracy": y_predict_score}
        
        #calculate f1 scores 
        self.predictions["f1"] = f1_score(y_true= y_test,
                                          y_pred= self.predictions["y_predict"])
        # and confusion matrix
        self.predictions["cm"] = confusion_matrix(y_true= y_test,
                                                  y_pred= self.predictions["y_predict"])
        # and normalized confusion matrix
        self.predictions["cmNorm"] = self.predictions["cm"] / self.predictions["cm"].sum(axis=1)[:, np.newaxis]
        
        #plot and report
        if y_test is not None:
            print(f"Accuracy: {self.predictions["accuracy"]}")
            print(f"F1 Score: {self.predictions["f1"]}")
            sns.heatmap(self.predictions["cmNorm"], xticklabels= [0 , 1], yticklabels= [0,1], annot = True, vmin= 0, vmax=1)
            plt.show()
            sns.heatmap(self.predictions["cm"], xticklabels= [0 , 1], yticklabels= [0,1], annot = True, vmin= 0, vmax=1)
            plt.show()
        

    def kFoldValidation(self,
                        ):
        pass


    def backtest(self,
                 model = "svm", # "classif"  "regres"
                 kFoldDf = None,
                 X_test = None,
                 seedMoney = 1000,
                 scaler = 0
                 ): # use a strategy to buy and sell the stock. chart the gains/losses against the testing set
        
        if kFoldDf is None or X_test is None:
            raise ValueError("Please input y_train, y_test and X_test datasets")

        if model is not None: #lets see how much money we lose if we follow the classifiers!
            backtestX = X_test
            backtestX_test = backtestX

            cash = [seedMoney]
            stock = [0]
            stockValue = [0]

            #apply scaler to test data
            if scaler == 1:
                backtestX_test = self.scaler.transform(backtestX)
                print("SCALINGSCALINGSCALING")
                print(backtestX_test)
                self.backtestX_scale = backtestX_test
            
            print(backtestX)

            # Apply model to each row of X_test to obtain direction in prediction col
            backtestX["prediction"] = self.model.predict(backtestX_test)
            self.backtestX = backtestX
            
            # buy / sell according to predicted direction
            stock_i = 0
            cash_i = seedMoney
            stockValue_i = 0
            for i in range(0, backtestX.shape[0]):
                if backtestX["prediction"][i] == 1:
                    stock_i = stock_i + (cash[i] / kFoldDf["Close"][i])
                    cash_i = 0

                    stockValue_i = (stock_i * kFoldDf["Close"][i])

                    stock.append(stock_i)
                    cash.append(cash_i)
                    stockValue.append(stockValue_i)

                if backtestX["prediction"][i] == -1:
                    cash_i = cash_i + (stock_i * kFoldDf["Close"][i])
                    stock_i = 0

                    stockValue_i = (stock_i * kFoldDf["Close"][i])

                    stock.append(stock_i)
                    cash.append(cash_i)
                    stockValue.append(stockValue_i)

            holdings = np.array(cash) + np.array(stockValue) # calculate total holdings, cash + stock value
            X_testIndex = X_test.index.to_numpy() # get date index from X test

            holdings = pd.DataFrame({"date": pd.to_datetime(X_testIndex), 
                                     "total_holdings": holdings[1:]}) # add dates to holdings. remove first holding since init from seedMoney
            holdings.set_index("date", inplace=True)

            self.backtestHoldings = holdings

            #plot holdings vs time
            # plt.plot(holdings.index, holdings["total_holdings"])
            # plt.show()

            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Plot total holdings
            ax1.plot(holdings.index, holdings["total_holdings"], color='blue', label="Total Holdings")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Total Holdings ($)")
            ax1.legend(loc="upper left")

            # Create second y-axis for stock price
            ax2 = ax1.twinx()
            ax2.plot(holdings.index, kFoldDf["Close"].loc[holdings.index], color='red', linestyle='dashed', label="Stock Price")
            ax2.set_ylabel("Stock Price ($)")
            ax2.legend(loc="upper right")

            plt.title(f"Backtest Performance for {tickerKFold}")
            plt.show()

            #get stats
            #avg returns over test period]

            model_returns = 100 * (holdings.iloc[len(holdings) - 1] - holdings.iloc[0]) / holdings.iloc[0]
            stock_returns = 100 * (kFoldDf["Close"].iloc[len(kFoldDf) - 1] - kFoldDf["Close"].iloc[0]) / kFoldDf["Close"].iloc[0]
            print(f"Model avg returns: {model_returns}%")
            print(f"Stock buy&hold returns: {stock_returns}%")

            #TODO: implement more metrics eg. Sharpe ratio, VAR, ES, alpha, 







        

        

        



###################################






####### SETTINGS #######
#data.downloadData
stock_tickers = ["SPY", "GOOG", "AAPL", "MSFT", "NVDA"]
timePeriod = "3y"
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
tickerKFold = "GOOG"  #SELECT TICKER FOR ANALYSIS
#data.runStrategy
strategy = ["logistic"] # "logistic"  "svm"  "dt"  "rf"
X_train = None
y_train = None
X_test = None
y_test = None

#data.backtest
seedMoneyy = 1000
scaler = 1 # SCALER FOR LOGISTIC AND SVM

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
                          tickerKFold = tickerKFold)

#run a strategy and predict/evaluate target
k_folds = 3
data.runStrategy(strategy = strategy,
                 X_train = data.kFoldX.iloc[data.kFoldIdx[k_folds]["trainIdx"]],
                 y_train = data.kFoldy.iloc[data.kFoldIdx[k_folds]["trainIdx"]],
                 X_test = data.kFoldX.iloc[data.kFoldIdx[k_folds]["testIdx"]],
                 y_test = data.kFoldy.iloc[data.kFoldIdx[k_folds]["testIdx"]]
                 )

#kfold cross validation

#backtest strat
data.backtest(model = strategy,
              kFoldDf= data.kFoldDf.iloc[data.kFoldIdx[k_folds]["testIdx"]],
              X_test= data.kFoldX.iloc[data.kFoldIdx[k_folds]["testIdx"]],
              seedMoney= seedMoneyy,
              scaler = scaler)

###################################