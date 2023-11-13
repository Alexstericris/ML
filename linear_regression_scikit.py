from _datetime import datetime, timedelta
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from dotenv import load_dotenv
import requests

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model,preprocessing,model_selection
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

load_dotenv()
style.use('ggplot')
if __name__ == '__main__':
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=IBM&apikey='+os.getenv('ALPHAVANTAGE_KEY')
    r = requests.get(url)
    if r.status_code == 200:
        # Parse the JSON response
        df = pd.DataFrame.from_dict(r.json()['Monthly Time Series'], orient='index',dtype=float)
        df['HL_PCT'] = (df['2. high'] - df['4. close']) / df['4. close'] * 100
        df['PCT_CHANGE'] = (df['4. close']-df['1. open'])/df['1. open']*100
        df=df[['4. close','HL_PCT','PCT_CHANGE','5. volume']]

        forecast_col='4. close'
        df.fillna(-9999,inplace=True)
        forecast_out=int(math.ceil(0.1*len(df)))
        df['label']=df[forecast_col].shift(-forecast_out)
        df.dropna(inplace=True)

        X=np.array(df.drop(['label'],axis=1))
        y=np.array(df['label'])
        X=preprocessing.scale(X)
        X=X[:-forecast_out]
        y=y[:-forecast_out]
        X_lately=X[-forecast_out:]
        X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,test_size=0.2)

        clf=LinearRegression()
        clf.fit(X_train,y_train)
        accuracy=clf.score(X_test,y_test)

        forecast_set=clf.predict(X_lately)
        print(accuracy)
        print(forecast_set)
        df['forecast']=np.nan
        last_date = datetime.strptime(df.iloc[0].name, '%Y-%m-%d')
        next_month=last_date+timedelta(days=30)
        for forcasted in forecast_set:
            df.loc[next_month]=[np.nan]*(len(df.columns)-1)+[forcasted]
            next_month = next_month + timedelta(days=30)

        df['4. close'].plot()
        df['forecast'].plot()
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

        # clf2=SVR()
        # clf2.fit(X_train,y_train)
        # accuracy2=clf2.score(X_test,y_test)

        # print(accuracy2)

    #diabetes linear_reg
    # diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    # diabetes_X = diabetes_X[:, np.newaxis, 2]
    # diabetes_X_train = diabetes_X[:-20]
    # diabetes_X_test = diabetes_X[-20:]
    # diabetes_y_train = diabetes_y[:-20]
    # diabetes_y_test = diabetes_y[-20:]
    # regr = linear_model.LinearRegression()
    # regr.fit(diabetes_X_train, diabetes_y_train)
    # diabetes_y_pred = regr.predict(diabetes_X_test)
    # print("Coefficients: \n", regr.coef_)
    # print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
    # plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    # plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()