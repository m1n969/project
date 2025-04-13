import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

path = '/Volumes/Extreme SSD/20241230/Vsub0mV_Vd2000mV/RawData/1e5_pulses/'
file_111 = path + 'pattern_1101.csv'
file_011 = path + 'pattern_0101.csv'
output_p_values = path + 'p_values_11010101.csv'
output_effect_size = path + 'effect_sizes_11010101.csv' 

data_111 = pd.read_csv(file_111, header=None)
data_011 = pd.read_csv(file_011, header=None)

if data_111.shape[0] != data_011.shape[0]:
    raise ValueError("error")

p_values = []
effect_sizes = []

for i in range(data_111.shape[0]):  
    row_111 = data_111.iloc[i, :].values 
    row_011 = data_011.iloc[i, :].values  

    t_stat, p_value = ttest_ind(row_111, row_011, equal_var=False)  
    p_values.append(f"{p_value:.10e}")  

    mean_111 = np.mean(row_111)
    mean_011 = np.mean(row_011)

    std_111 = np.std(row_111, ddof=1)
    std_011 = np.std(row_011, ddof=1)

    n1 = len(row_111)
    n2 = len(row_011)

    pooled_std = np.sqrt(((n1 - 1) * std_111**2 + (n2 - 1) * std_011**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        effect_size = 0  
    else:
        effect_size = abs(mean_111 - mean_011) / pooled_std

    effect_sizes.append(f"{effect_size:.6f}")  

p_values_df = pd.DataFrame(p_values, columns=["p_value"])
p_values_df.to_csv(output_p_values, index=False, header=None)

effect_sizes_df = pd.DataFrame(effect_sizes, columns=["effect_size"])
effect_sizes_df.to_csv(output_effect_size, index=False, header=None)

print(f" {output_p_values}")
print(f"{output_effect_size}")