import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA


source = pd.read_csv("Req_JB_BP.csv")
source['Date'] = pd.to_datetime(source['Date'], format='%d/%m/%Y')

Vehicle_Type = source.columns[1:]
parameters = np.genfromtxt('BestParameters.csv', delimiter=',')

train = source[source['Date'] <= pd.to_datetime("2024-01-15", format='%Y-%m-%d')]
test = source[source['Date'] > pd.to_datetime("2024-01-15", format='%Y-%m-%d')]

Pred_result = test.copy()
Pred_result.loc[:,Pred_result.columns != 'Date']=0

counter=0
#StatModel SARIMAX
for vehicle in Vehicle_Type:
    y = train[vehicle]
    p, d, q, P, D, Q, s = parameters[counter,0:]
    SARIMAXmodel = SARIMAX(y, order = (p, d, q), seasonal_order=(P,D,Q,s))
    SARIMAXmodel = SARIMAXmodel.fit()

    y_pred_SARIMAX = SARIMAXmodel.get_forecast(len(test.index))
    y_pred_df_SARIMAX = y_pred_SARIMAX.conf_int(alpha = 0.05) 
    y_pred_df_SARIMAX["Predictions"] = SARIMAXmodel.predict(start = y_pred_df_SARIMAX.index[0], end = y_pred_df_SARIMAX.index[-1])
    y_pred_df_SARIMAX.index = test.index
    y_pred_out_SARIMAX = y_pred_df_SARIMAX["Predictions"]
    Pred_result.loc[:,vehicle] = y_pred_out_SARIMAX 
    counter=counter+1

Pred_result.to_csv('DailyForecast.csv', index=False)

'''
sns.set()
plt.plot(train['Date'],train['Economy'], color = "black",label = 'Training')
plt.plot(test['Date'],test['Economy'], color = "red",label = 'Testing')
plt.plot(test['Date'],y_pred_out_ARMA, color='green', label = 'ARMA Predictions')
plt.plot(test['Date'],y_pred_out_ARIMA, color='yellow', label = 'ARIMA Predictions')
plt.plot(test['Date'],y_pred_out_SARIMAX, color='blue', label = 'SARIMAX Predictions')
plt.ylabel('Number of Request')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.2)
plt.xlim(test['Date'].min(), test['Date'].max())
plt.title("Train/Test split for Request")
plt.legend()
plt.show()
'''
