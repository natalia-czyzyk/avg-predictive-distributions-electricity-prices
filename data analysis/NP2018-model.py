import pandas as pd
import numpy as np
from scipy.stats import linregress
import csv
import os

df = pd.read_csv('NP2018.csv', header=None, parse_dates=[0])
df = df[8736:]
df = df.reset_index(drop=True)

df['X_min'] = np.zeros(len(df))
for i in range(0, len(df) - 23, 24):
    df['X_min'].loc[i:i + 24] = np.min(df.loc[i:i + 24,2])
    
df['X_max'] = np.zeros(len(df))
for i in range(0, len(df) - 23, 24):
    df['X_max'].loc[i:i + 24] = np.max(df.loc[i:i + 24,2])
    
df['X_mid'] = np.zeros(len(df))
for i in range(0, len(df) - 23, 24):
    df['X_mid'].loc[i:i + 24] = df.loc[23+i,2]
    
df['Monday'] = np.where(df[0].dt.dayofweek == 0, 1, 0)
df['Tuesday'] = np.where(df[0].dt.dayofweek == 1, 1, 0)
df['Wednesday'] = np.where(df[0].dt.dayofweek == 2, 1, 0)
df['Thursday'] = np.where(df[0].dt.dayofweek == 3, 1, 0)
df['Friday'] = np.where(df[0].dt.dayofweek == 4, 1, 0)
df['Saturday'] = np.where(df[0].dt.dayofweek == 5, 1, 0)
df['Sunday'] = np.where(df[0].dt.dayofweek == 6, 1, 0)

df[0] = df[0].view('int64')

forecast_dict = {}
errors_dict = {}

errors = []

for win_length in [56,84,112,714,721,728]:
    forecast_dict[f"forecast{win_length}"] = []
    errors_dict[f"forecast{win_length}"] = []
    error_win = []
    
    for hour in range(24):
        p_hour = df.loc[hour::24].values
        err_h = []
        for day in range(728, len(p_hour)):
            cal_data = p_hour[(day-win_length):day]
        
            Y = cal_data[7:win_length, 2]
            
            X1 = cal_data[6:win_length-1, 2]
            X2 = cal_data[5:win_length-2, 2]
            X3 = cal_data[:win_length-7, 2]
            X4 = cal_data[6:win_length-1, 5]
            X5 = cal_data[6:win_length-1, 6]
            X6 = cal_data[6:win_length-1, 7]
            X7 = cal_data[7:win_length, 3]
    
            D1 = cal_data[7:win_length, 8]
            D2 = cal_data[7:win_length, 9]
            D3 = cal_data[7:win_length, 10]
            D4 = cal_data[7:win_length, 11]
            D5 = cal_data[7:win_length, 12]
            D6 = cal_data[7:win_length, 13]
            D7 = cal_data[7:win_length, 14]
            
            X = np.column_stack([X1, X2, X3, X4, X5, X6, X7, D1, D2, D3, D4, D5, D6, D7])
    
            betas, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            
            x_fut = np.array([cal_data[win_length-1,2],cal_data[win_length-2,2],cal_data[win_length-7,2],cal_data[win_length-1,5],cal_data[win_length-1,6],cal_data[win_length-1,7],p_hour[day,3],p_hour[day,8],p_hour[day,9],p_hour[day,10],p_hour[day,11],p_hour[day,12],p_hour[day,13],p_hour[day,14]])
            forecast = np.dot(x_fut, betas)
            forecast_dict[f"forecast{win_length}"].append(forecast)
            real = p_hour[day,2]
            errors_dict[f"forecast{win_length}"].append(np.abs(forecast-real))

    
transformed_dict = {}
for key, original_list in forecast_dict.items():
    matrix = np.array(original_list).reshape(946, 24, order='F')
    matrix2 = matrix.reshape(-1, 1)
    transformed_dict[key] = matrix2

transformed_errors_dict = {}
for key, original_list in errors_dict.items():
    matrix_e = np.array(original_list).reshape(946, 24, order='F')
    matrix2_e = matrix_e.reshape(-1, 1)
    transformed_errors_dict[key] = matrix2_e
    
mae_values = []
for i, j in transformed_errors_dict.items():
    mae = np.mean(j)
    mae_values.append(mae)
    
print(mae_values)
                
for key, values in transformed_dict.items():
    output_file = f"NP2018{key}.csv"
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([key])
        for value in values:
            csv_writer.writerow([str(v) for v in value])
#osobno mae dla kazdego win length
