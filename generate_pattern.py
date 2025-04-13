import pandas as pd
import numpy as np

Bit = 3
Nodes = 200

#Zip = 'True'
Voltage = 'Tru'

#Date = '20240913'
#DeviceType = 'nMOS_Hf'
#Category = 'Widthswing_new_High3.5V_Low-3.5V_Cent0.8V_Width4e-06s_0_0_'
#Type = 'Width_8e-06'


path = '/Volumes/Extreme SSD/20250410_3000_0/Vgswing3000mV_Vd300mV/RawData/1e5_pulses/'


#Gen_Data

# 新しいDataFrameを初期化
data_df = pd.DataFrame()
# 10個のCSVファイルから2列目のデータを読み取って結合
for i in range(10):
    # ファイル名を作成
    if Voltage == 'True':
        filename = path + f'Seed{i}_trimed_averaged.csv'
    else:
        filename = path + f'Id{i}_out_Channel202.csv'
    #if Zip == 'True':
        #filename = filename + '.zip'
    # CSVファイルを読み込む
    df = pd.read_csv(filename, header=None)
    
    # 2列目のデータを取得（列インデックスは1）
    column_data = df.iloc[:, 1]
    
    # 列データをDataFrameに追加
    data_df[i] = column_data

print("Dataの作成が完了しました。")
print(data_df.shape)

#Gen_Pattern
# CSVファイルの読み込み
seed_df = pd.read_csv('/Volumes/Extreme SSD/20241108/Vsub2TimeStepDelayVd500mV/SourceCode/Seed.csv', header=None, skiprows=1)

# Seed.csvのデータをもとにData.csvのデータを分類
print(seed_df.shape)
if Bit == 3:
    # 分類用の辞書を初期化
    classified_data = {
        '000': np.zeros((Nodes, 0)),
        '001': np.zeros((Nodes, 0)),
        '010': np.zeros((Nodes, 0)),
        '011': np.zeros((Nodes, 0)),
        '100': np.zeros((Nodes, 0)),
        '101': np.zeros((Nodes, 0)),
        '110': np.zeros((Nodes, 0)),
        '111': np.zeros((Nodes, 0))
    }
    for i in range(seed_df.shape[1]):
        for j in range(501,seed_df.shape[0]):
            previous2_value = seed_df.iloc[j-2, i]
            previous_value = seed_df.iloc[j-1, i]
            current_value = seed_df.iloc[j, i]
            key = f"{previous2_value}{previous_value}{current_value}"
            start_row = Nodes * j
            end_row = Nodes * (j + 1)
            chunk_data = data_df.iloc[start_row:end_row, i].values.reshape(Nodes, 1)
            classified_data[key] = np.hstack((classified_data[key], chunk_data))
    
    
elif Bit == 2:
    classified_data = {
        '00': np.zeros((Nodes, 0)),
        '01': np.zeros((Nodes, 0)),
        '10': np.zeros((Nodes, 0)),
        '11': np.zeros((Nodes, 0)),
    }
    for i in range(seed_df.shape[1]):
        for j in range(501,seed_df.shape[0]):
            previous_value = seed_df.iloc[j-1, i]
            current_value = seed_df.iloc[j, i]
            key = f"{previous_value}{current_value}"
            start_row = Nodes * j
            end_row = Nodes * (j + 1)
            chunk_data = data_df.iloc[start_row:end_row, i].values.reshape(Nodes, 1)
            classified_data[key] = np.hstack((classified_data[key], chunk_data))

# 各分類のデータを出力
pattern_df = pd.DataFrame()
stats_df = pd.DataFrame()
for key, data_matrix in classified_data.items():
    averaged_data = data_matrix.mean(axis=1)
    pattern_df[key] = averaged_data

    mean = data_matrix.mean()  
    std_dev = data_matrix.std()  
    min_value = data_matrix.min()  
    max_value = data_matrix.max()  
    median = np.median(data_matrix, axis=1)  


    stats = {
        'mean': mean,
        'std': std_dev,
        'min': min_value,
        'max': max_value,
        'median': median.mean() 
    }

    stats_df = stats_df._append(pd.DataFrame(stats, index=[key]))

stats_df = stats_df[['mean', 'std', 'min', 'max', 'median']]


# まとめたデータをCSVとして出力
pattern_df.to_csv(path + 'pattern_3bit.csv', index=False, header=True)

stats_df.to_csv(path +'pattern_stats_data.csv', index_label='pattern', header=True)
print(f"{Bit}bitでの分類と出力が完了しました。")
