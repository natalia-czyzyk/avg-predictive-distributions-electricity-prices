import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


forecasts = {}
for num in [14, 21, 28, 42, 70, 308, 336, 364]:
    #file_path = f'NP2018-QRA-quantile{num}.csv'
    #file_path = f'NP2018-QRM-quantile{num}.csv'
    file_path = f'PJM-QRA-quantile{num}.csv'
    #file_path = f'PJM-QRM-quantile{num}.csv'
    key = f'forecast{num}'

    forecasts[key] = pd.read_csv(file_path, header=None)

real = pd.read_csv('PJM.csv', header=None, sep=';', parse_dates=[0])
real = real.iloc[46080:, 2].reset_index(drop=True)
#real = pd.read_csv('NP2018.csv', header=None, parse_dates=[0])
#real = real.iloc[34944:,2].reset_index(drop=True)

aps_all = []

# grey points
for key, forecast in forecasts.items():
    #print(key)
    ps_list = []
    for i in range(0, len(forecast)):
        real_value = real[i]
        for j in range(1, 100):
            forecast_value = forecast.iloc[i, j - 1]
            diff = real_value - forecast_value  # if + or 0 then 2nd condition, if - then 1st
            if diff >= 0:
                PS = diff * (j / 100)
            else:
                PS = (forecast_value - real_value) * (1 - (j / 100))
            ps_list.append(PS)

    ps_array = np.array(ps_list)
    aps = ps_array.mean()
    print(key, 'Grey', aps)
    aps_all.append(aps)

# blue points
avg_ps_list = []
for i in range(0, len(forecasts[key])):
    #print(i)
    real_value = real[i]
    for j in range(1, 100):
        value_list = []
        for key in ["forecast14", "forecast21", "forecast28"]:
            forecast_value = forecasts[f"{key}"].iloc[i, j - 1]
            value_list.append(forecast_value)
        value_list_array = np.array(value_list)
        avg_value = value_list_array.mean()

        diff = real_value - avg_value  # if + or 0 then 2nd condition, if - then 1st
        if diff >= 0:
            PS = diff * (j / 100)
        else:
            PS = (avg_value - real_value) * (1 - (j / 100))
        avg_ps_list.append(PS)

avg_ps_array = np.array(avg_ps_list)
avg_aps = avg_ps_array.mean()
print('Blue', avg_aps)

# red points
avg_ps_list_2 = []
for i in range(0, len(forecasts[key])):
    #print(i)
    real_value = real[i]
    for j in range(1, 100):
        value_list = []
        for key in ["forecast14", "forecast42", "forecast70", "forecast308", "forecast336", "forecast364"]:
            forecast_value = forecasts[f"{key}"].iloc[i, j - 1]
            value_list.append(forecast_value)
        value_list_array = np.array(value_list)
        avg_value = value_list_array.mean()

        diff = real_value - avg_value  # if + or 0 then 2nd condition, if - then 1st
        if diff >= 0:
            PS = diff * (j / 100)
        else:
            PS = (avg_value - real_value) * (1 - (j / 100))
        avg_ps_list_2.append(PS)

avg_ps_array_2 = np.array(avg_ps_list_2)
avg_aps_2 = avg_ps_array_2.mean()
print('Red', avg_aps_2)

# black points
avg_ps_list_3 = []
for i in range(0, len(forecasts[key])):
    #print(i)
    real_value = real[i]
    for j in range(1, 100):
        value_list = []
        for key in ["forecast14", "forecast21", "forecast28", "forecast308", "forecast336", "forecast364"]:
            forecast_value = forecasts[f"{key}"].iloc[i, j - 1]
            value_list.append(forecast_value)
        value_list_array = np.array(value_list)
        avg_value = value_list_array.mean()

        diff = real_value - avg_value  # if + or 0 then 2nd condition, if - then 1st
        if diff >= 0:
            PS = diff * (j / 100)
        else:
            PS = (avg_value - real_value) * (1 - (j / 100))
        avg_ps_list_3.append(PS)

avg_ps_array_3 = np.array(avg_ps_list_3)
avg_aps_3 = avg_ps_array_3.mean()
print('Black', avg_aps_3)

plt.figure(figsize=(10, 8))
plt.scatter([14, 21, 28, 42, 70, 308, 336, 364], aps_all, color='gray', label='QRM(T)', s=50)

plt.scatter([14, 21, 28], [avg_aps]*3, color='blue', label='QRM(14:7:28)', s=50)
plt.plot([14, 21, 28], [avg_aps]*3, color='blue', linestyle='-', linewidth=2)

plt.scatter([14, 42, 70, 308, 336, 364], [avg_aps_2]*6, color='red', label='QRM(14:28:70,308:28:364)', s=50)
plt.plot([14, 42, 70, 308, 336, 364], [avg_aps_2]*6, color='red', linestyle='-', linewidth=2)

plt.scatter([14, 21, 28, 308, 336, 364], [avg_aps_3]*6, color='black', label='QRM(14:7:28,308:28:364)', s=50)
plt.plot([14, 21, 28, 308, 336, 364], [avg_aps_3]*6, color='black', linestyle='-', linewidth=2)

plt.xlabel('Probabilistic calibration window lenght (T)')
plt.ylabel('Aggregate Pinball Score')
plt.title('Aggregate Pinball Score (APS) for PJM QRM, win lengths = [14, 21, 28, 42, 70, 308, 336, 364]')
plt.legend()

plt.show()