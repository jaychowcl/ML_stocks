[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Picture1](https://github.com/user-attachments/assets/27dcaa8b-2732-4e95-8b87-e97d1a80d573)

Introducing the FinanceData object!

## Description

This project is a financial data analysis and machine learning-based trading strategy framework. It collects stock market data, calculates metrics and techinical indicators, applies machine learning techniques for price movement prediction and evaluates a given strategy via backtesting.

### This is _**NOT**_ a serious trading algo. 
### Use at your own risk, for it is only meant to be a playground for me to analyse some financial data! 
### THIS WILL LOSE YOU MONEY!

### General Overview

1. Initialize FinanceData object with chosen tickers and time periods via FinanceData()
2. Download market data from Yahoo! Finance
3. Generate financtial metrics and indicators (sma, ema, macd, rsi, adx) with tunable parameters
4. Generate plots for pairwise and corerlation analysis
5. Conducts kfold splits and splits data into a training and testing set via TimeSeriesSplit into each k fold
6. Train and test a model to gather performance metrics
7. Backtest a strategy using the model.
8. Plot and gather backtesting metrics (TBA)


## Getting Started

from FinanceData import FinanceData

1. Create a FinanceData object via FinanceData()
(Optional): use FinanceData.importData to gather ticker info before downloading
2. Download ticker(s) market data from Yahoo! Finance via FinanceData.downloadData()
3. Generate statistics and techinical indicators with tunable parameters via  FinanceData.generateMetrics()

   (sma, ema, macd, adx, rsi)
   
5. Generate pair and correlation plots, as well as price vs any chosen metrics via FinanceData.generatePlots()
6. Split dataset into training/testing sets in k folds for a chosen ticker via FinanceData.kFoldSplitTimeSeries()
7. Train and evaluate the performance of different machine learning implementations via FinanceData.runStrategy()
   
   Classifiers: Logistic regression, support vector machines, decision trees, random forest
   
   Regression (TBA): Multiple linear regression, polynomial regression, support vector machines, knn
   
8. Backtest a strategy using a ML model via FinanceData.backtest()
   Will plot holdings and stock price vs time, and return avg returns
9. Generate metrics from a backtest via FinanceData.backtestStats() (TBA)
    
    (alpha, beta, Sharpe ratio, Sortina ratio, value added risk, expected shortfall,)

### Dependencies

yfinance

pandas

numpy

matplotlib

seaborn

sklearn

### Installing
Place FinanceData.py in workspace

Then:
from FinanceData import FinanceData

## Authors

[Jay Chow (Chi Lung)](https://github.com/jaychowcl)

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the MIT License
