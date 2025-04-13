import pandas as pd
import numpy as np

Bit = 3
Nodes = 200
Voltage = 'Tru'
path = '/Volumes/Extreme SSD/20250327/Vgswing2500mV_Vd500mV/RawData/1e5_pulses/'

data_df = pd.DataFrame()

for i in range(10):
    if Voltage == 'True':
        filename = path + f'Seed{i}_trimed_averaged.csv'
    else:
        filename = path + f'Id{i}_out_Channel202.csv'
    df = pd.read_csv(filename, header=None)
    column_data = df.iloc[:, 1]
    data_df[i] = column_data

print("Data completed")
print(data_df.shape)

seed_df = pd.read_csv('/Volumes/Extreme SSD/20241108/Vsub2TimeStepDelayVd500mV/SourceCode/Seed.csv', header=None, skiprows=1)
print(seed_df.shape)

patterns = ["".join(seq) for seq in np.array(np.meshgrid(*[['0', '1']] * Bit)).T.reshape(-1, Bit)]
classified_data = {pattern: [] for pattern in patterns}

for i in range(seed_df.shape[1]):
    for j in range(501, seed_df.shape[0]):
        key = "".join(seed_df.iloc[j-Bit+1:j+1, i].astype(str).values)
        start_row = Nodes * j
        end_row = Nodes * (j + 1)
        chunk_data = data_df.iloc[start_row:end_row, i].values
        classified_data[key].append(chunk_data)
for pattern, data_list in classified_data.items():
    if data_list:  
        pattern_matrix = np.column_stack(data_list)  
        output_df = pd.DataFrame(pattern_matrix)
        output_file = path + f'pattern_{pattern}.csv'
        output_df.to_csv(output_file, index=False, header=False)
        print(f"Pattern {pattern} saved to {output_file}")

print(f"{Bit}bit processed")