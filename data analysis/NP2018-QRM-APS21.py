import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

aps_all = []

for num in [14, 21, 28, 42, 70]:
    print(num)
    filename = f'NP2018-QRM-quantile{num}.csv'
    forecast = pd.read_csv(filename, header=None)
    
    real = pd.read_csv('NP2018.csv', header=None, parse_dates=[0])
    real = real.iloc[34944:,2].reset_index(drop=True)
    
    ps_list = []
    for i in range(0,len(forecast)):
        real_value = real[i]
        for j in range(1,100):
            forecast_value = forecast.iloc[i,j-1]
            diff = real_value - forecast_value  #if + or 0 then 2nd condition, if - then 1st
            if diff >= 0:
                PS = diff * (j/100)
            else:
                PS = (forecast_value - real_value) * (1-(j/100)) 
            ps_list.append(PS)
            
    ps_array = np.array(ps_list)
    aps = ps_array.mean()
    aps_all.append(aps)
    
plt.figure(figsize=(10, 6))
plt.scatter([14, 21, 28, 42, 70], aps_all, color='gray', label='APS', s=50)

# Calculate mean of 14, 21, 28 APS
mean_aps_14_21_28 = np.mean(aps_all[:3])
plt.scatter([14, 21, 28], [mean_aps_14_21_28] * 3, color='blue', label='Mean of 14, 21, 28 APS', s=50)

# Add labels and legend
plt.xlabel('Probabilistic calibration window lenght (T)')
plt.ylabel('Aggregate Pinball Score')
plt.title('Aggregate Pinball Score (APS) for Nord Pool QRA, win lengths = [14, 21, 28, 42, 70]')
plt.legend()

# Show the plot
plt.show()