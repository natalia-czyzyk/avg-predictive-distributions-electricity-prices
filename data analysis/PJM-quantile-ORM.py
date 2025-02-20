import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
import csv

# PJM, QRM, point windows 56-728, quantiles 0.01-0.99, probabilistic windows 14, 182, 364

df = pd.DataFrame()
for i in [56, 84, 112, 714, 721, 728]:
    df[f"{i}"] = pd.read_csv(f'PJMforecast{i}.csv')

df['mean'] = df.iloc[:, 0:6].mean(axis=1)
X = df.drop(df.columns[[0,1,2,3,4,5]],axis=1)

Y = pd.read_csv('PJM.csv', header=None, sep=';', parse_dates=[0])
Y = Y[37344:]
Y = Y.reset_index(drop=True)

quantiles = [x / 100 for x in range(1, 100)]

num_days_hours = 728 * 24

for win in [14,182,364]:
    with open(f"PJM-QRM-quantile{win}.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for day in range(364, 728+364):
            X_fut_end = X[364 * 24:]
            X_fut_end = X_fut_end.reset_index(drop=True)
            X_day = X_fut_end[day * 24:(day * 24) + 24]
            for hour in range(24):
                X_fut = X_day.iloc[hour]
                Xregress = X[(day - win) * 24 + hour:(day * 24) + hour]
                Yregress = Y[(day - win) * 24 + hour:(day * 24) + hour]
                Yregress = Yregress.iloc[:, 2]
                hour_list = []
                for i, quantile in enumerate(quantiles):
                    qr = QuantileRegressor(quantile=quantile, alpha=0, solver='highs').fit(Xregress, Yregress)
                    prediction = qr.intercept_ + np.dot(X_fut, qr.coef_)
                    hour_list.append(prediction)
                csv_writer.writerow(hour_list)