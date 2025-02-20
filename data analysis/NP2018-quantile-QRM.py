import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
import csv

# Nord Pool, QRM, point windows 56-728, quantiles 0.01-0.99, probabilistic windows 14, 182, 364

df = pd.DataFrame()
for i in [56, 84, 112, 714, 721, 728]:
    df[f"{i}"] = pd.read_csv(f'NP2018forecast{i}.csv')

df['mean'] = df.iloc[:, 0:6].mean(axis=1)
X = df.drop(df.columns[[0,1,2,3,4,5]],axis=1)

Y = pd.read_csv('NP2018.csv', header=None, parse_dates=[0])
Y = Y[26208:]
Y = Y.reset_index(drop=True)

quantiles = [x / 100 for x in range(1, 100)]

num_days_hours = 582 * 24

for win in [70,308,336,364]:
    with open(f"NP2018-QRM-quantile{win}.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for day in range(364,582+364):
            print('win:',win,'day:',day)
            X_day = X[day*24:(day*24)+24]
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
                hour_list.sort()
                csv_writer.writerow(hour_list)
                
