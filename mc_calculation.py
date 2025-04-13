import os
import shutil
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rename_and_copy_csv(file_path, target_folder, keyword, folder_suffix):

    original_name = os.path.basename(file_path)
    name, ext = os.path.splitext(original_name)
    
    new_name = f"{keyword}_{folder_suffix}{ext}"
    new_file_path = os.path.join(target_folder, new_name)
    
    shutil.copy2(file_path, new_file_path)
    print(f"{new_file_path}")
    
    return new_file_path

def process_csv(file_path):

    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f" {file_path}, error: {e}")
        return

    if df.shape[1] < 7:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=range(df.shape[1], 7))], axis=1)

    if len(df) < 12:
        missing_rows = 12 - len(df)
        df = pd.concat([df, pd.DataFrame(np.nan, index=range(missing_rows), columns=df.columns)], ignore_index=True)
    
    for row in range(1, 11):
        try:
            numeric_values = pd.to_numeric(df.iloc[row, 1:6], errors='coerce')
            df.iloc[row, 6] = numeric_values.mean()
        except Exception as e:
            print(f" {row} error: {e}")
            continue

    try:
        sum_value = pd.to_numeric(df.iloc[2:11, 6], errors='coerce').sum()
        df.iloc[11, 6] = sum_value
    except Exception as e:
        print(f"error calculation {e}")

    df.to_csv(file_path, index=False, header=False)
    print(f" {file_path}")

def find_and_copy_csv_files(base_folder, target_folder):

    for k in ["STM", "PC", "XOR"]:
        keyword = f"RateRidge_1e5_pulses_{k}_Ch202301_VN200_avg_woTh_Td10"
        for i in[0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000]:
            for j in [300]:
                folder_name = f"Vgswing{i}mV_Vd{j}mV"
                analysis_folder = os.path.join(base_folder, folder_name, "AnalysisResult")
                
                if not os.path.isdir(analysis_folder):
                    print(f" AnalysisResult not found: {analysis_folder}")
                    continue

                folder_suffix = f"Vgswing{i}mV_Vd{j}mV"
                
                search_pattern = os.path.join(analysis_folder, f"*{keyword}*.csv")
                csv_files = [f for f in glob.glob(search_pattern) if "CC" not in f and "PCA" not in f]
                
                if not csv_files:
                    print(f"CSV not found：{folder_suffix}")
                    continue
                
                for file_path in csv_files:
                    print(f"processed: {file_path}")
                    
                    new_file_path = rename_and_copy_csv(file_path, target_folder, keyword, folder_suffix)
                    process_csv(new_file_path)

base_folder = "/Volumes/Extreme SSD/20250410_3000_0"
target_folder = "/Volumes/Extreme SSD/20250410_3000_0/mc_IdIs"  

find_and_copy_csv_files(base_folder, target_folder)