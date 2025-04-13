import os
import shutil

def organize_folder_structure(base_folder):
   
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        print(f" {folder_path}")
        
        analysis_result_path = os.path.join(folder_path, "AnalysisResult")
        os.makedirs(analysis_result_path, exist_ok=True)
        
        raw_data_path = os.path.join(folder_path, "RawData")
        pulses_path = os.path.join(raw_data_path, "1e5_pulses")
        os.makedirs(pulses_path, exist_ok=True)
        
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            
            if item_name in ["AnalysisResult", "RawData"]:
                continue
            
            new_path = os.path.join(pulses_path, item_name)
            shutil.move(item_path, new_path)
            print(f"{item_path} -> {new_path}")

base_folder = "/Volumes/Extreme SSD/20250411_3000_0"

organize_folder_structure(base_folder)