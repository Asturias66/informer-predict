import math

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
import numpy as np
import statsmodels.api as sm

df_raw = pd.read_csv(os.path.join('../data/', '601398.csv'))
# df = df.iloc[0:2402,:]
# df_test = df.iloc[2402:,:]

df1=df_raw.iloc[:,1:]
# min_max_scaler = preprocessing.MinMaxScaler()
# df0=min_max_scaler.fit_transform(df1)
object = StandardScaler()
df0 = object.fit_transform(df1)
df = pd.DataFrame(df0, columns=df1.columns)

df = df.dropna()
df = df["close"]
df.reset_index()
# df1 = np


# object = StandardScaler()
# object.fit(np)
# print(object.mean())
# print(object.std())
# df_np = object.fit_transform(np)
# print(df_np[0][0])

# min_max_scaler = preprocessing.MinMaxScaler()
# df_np = min_max_scaler.fit_transform(np)

# df = pd.DataFrame()
# df['close'] = pd.Series(df_np[0])
# print(df)

# 进行一阶差分
D_ts1 = df.diff().dropna()
D_ts1.columns = [u'close差分']


size = int(len(df) * 0.8)
train, test = df[0:size], df[size:len(df)]

print(len(train), len(test))
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(size,size + len(test)):
    model = ARIMA(history[-10:-1], order=(4,1,4), missing='drop')
    model_fit = model.fit()
    output = model_fit.forecast()
    print(output)
    pre = output[0]
    predictions.append(pre)
    print(t)
    print(test)
    history.append(test[t])
    print('predicted=%f, expected=%f' % (pre, test[t]))

rmse = math.sqrt(mean_squared_error(test, predictions))
mse = mean_squared_error(test, predictions)
mae = mean_absolute_error(test, predictions)
mape = mean_absolute_percentage_error(test, predictions)
print('Test RMSE: %.3f' % rmse)
print('Test MSE: %.3f' % mse)
print('Test MAE: %.3f' % mae)
print('Test MAPE: %.3f' % mape)

# fig, ax = plt.subplots(1, 1)
plt.figure()
plt.plot(test.values, label='GroundTruth')
plt.plot(predictions, label='Prediction')
# plt.title('Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Price')
# ax.xaxis.set_major_locator(ticker.MultipleLocator(base=100))
# plt.xticks(range(len(test)), test.index, rotation=90)
plt.legend()
plt.show()

# ratio=0.2
# testsize=int(len(D_ts1)*ratio) #0.2
# trainsize=len(D_ts1)-testsize  #0.8
# test=D_ts1[-testsize:].values
# train=D_ts1[:trainsize].values
# history=list(train)
# predictions=list()
#
# #预测添加，训练再预测
# for t in range(int(len(test)/10)):
#     print(t)
#     model=ARIMA(history,order=(4,1,4))
#     model_fit=model.fit()
#     output=model_fit.forecast()
#     if len(output) >= 10 :
#         yhat = output[:10]
#     else:
#         yhat = output[:len(output)]
#     print(yhat)
#     # if len(predictions) == 0:
#     #     predictions.append(y)
#     predictions = predictions + yhat
#     print(predictions)
#     history = history + yhat
#     print(history)
#
# mse = mean_squared_error(test,predictions)
# mae = mean_absolute_error(test,predictions)
# mape = mean_absolute_percentage_error(test,predictions)
# rmse = math.sqrt(mse)
# print("mse",mse)
# print("mae",mae)
# print("rmse",rmse)
# print("mape",mape)
# print(testsize)
# x1 = df.index[-testsize:].values
# plt.plot(x1,test,label='真实值')
# plt.plot(x1,predictions,color='red',label='预测值')
# plt.legend()
# plt.title('模型评估')
# plt.show()


# fig, ax = plt.subplots(1,1)
# plt.plot(test.values, color='blue', label='Origin')
# plt.plot(predictions, color='red', label='Predcition RMSE:%.3f' % rmse)
# plt.title('sz000789 Price Prediction')
# plt.xlabel('Time(Weeks)')
# plt.ylabel('Price')
# ax.xaxis.set_major_locator(ticker.MultipleLocator(base=100))
# plt.xticks(range(len(test)), test.index, rotation=90)
# plt.legend()
# plt.show()

# plt.plot(np.arange(2105),D_ts1)
# plt.title("一阶差分图")  #添加图标题
# plt.xticks(rotation=45)    #横坐标旋转45度
# plt.xlabel('date')   #添加图的标签（x轴，y轴）
# plt.ylabel("close")
# plt.show()

# acf
# plot_acf(D_ts1,lags=80)
# plt.show()

#pacf
# plot_pacf(D_ts1,lags=80)
# plt.show()

# from arch.unitroot import ADF
# adf = ADF(D_ts1)
# print(adf.pvalue)
# print(adf.summary().as_text())

#
#aic
# (p,q) = (sm.tsa.arma_order_select_ic(D_ts1,max_ar=3,max_ma=3,ic='aic'))['aic_min_order']
# print(p,q)

# from statsmodels.tsa.arima_model import ARIMA
# #假设检验
# from statsmodels.tsa.arima.model import ARIMA
#
# model = ARIMA(df, order=(4,1,4))
# model_fit = model.fit()
# print(model_fit.summary())

# from statsmodels.graphics.api import qqplot
# from statsmodels.stats.stattools import durbin_watson #DW检验
#
# resid = model_fit.resid
# plt.figure(figsize=(12,8))
#
# print(qqplot(resid,line='q',fit=True))
#
# print('D-W检验值为{}'.format(durbin_watson(resid.values)))

