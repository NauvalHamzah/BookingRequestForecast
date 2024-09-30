import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA


source = pd.read_csv("Req_Hourly_JB_BP.csv")
source['Date Hour'] = pd.to_datetime(source['Date Hour'], format='%d/%m/%Y %H:%M')
print(source)

Vehicle_Type = source.columns[1:]
best_parameters = np.zeros((len(Vehicle_Type),7))

counter=0
for item in Vehicle_Type:
    print(item)
    source1 = source[['Date Hour',item]]

    train = source1[source1['Date Hour'] <= pd.to_datetime("2024-01-15 00:00:00", format='%Y-%m-%d %H:%M:%S')]
    test = source1[source1['Date Hour'] > pd.to_datetime("2024-01-15 00:00:00", format='%Y-%m-%d %H:%M:%S')]

    import pmdarima as pm

    # Assuming your time series data is in 'data'
    model = pm.auto_arima(train[item], seasonal=True, m=24,  # assuming monthly seasonality
                        start_p=1, max_p=3,  # range for AR order
                        start_q=1, max_q=3,  # range for MA order
                        start_P=1, max_P=2,  # range for seasonal AR order
                        start_Q=1, max_Q=2,  # range for seasonal MA order
                        max_order=None,  # max order of differencing
                        d=1, D=1,  # differencing order
                        trace=True,  # print status of the fit
                        error_action='ignore',  # ignore orders that don't converge
                        suppress_warnings=True,  # suppress warnings
                        stepwise=True)  # use stepwise algorithm for faster computation

    print(model.summary())

    # Getting the best parameters
    best_order = model.order
    best_seasonal_order = model.seasonal_order

    best_parameters[counter,0:3] = best_order
    best_parameters[counter,3:7] = best_seasonal_order
    counter=counter+1


np.savetxt('BestParametersHour.csv', best_parameters, delimiter=',')
