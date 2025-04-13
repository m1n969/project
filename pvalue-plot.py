import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

base_path = "/Volumes/Extreme SSD/20250118_3/Vsub{i}mV_Vd300mV/RawData/1e5_pulses/"
file_names = [
    base_path.format(i="0") + "p_values_100000.csv",
    base_path.format(i="-1000") + "p_values_100000.csv",
    base_path.format(i="-2000") + "p_values_100000.csv",
    base_path.format(i="-3000") + "p_values_100000.csv",
    base_path.format(i="-5000") + "p_values_100000.csv",
]
labels = ["Vsub 0V", "Vsub -1V", "Vsub -2V", "Vsub -3V", "Vsub -5V"]
colors = ["tab:red", "tab:orange", "tab:green", "tab:purple", "deeppink"]

global_min_pvalue = np.inf 

for file_name in file_names:
    try:
        data = pd.read_csv(file_name, header=None)
        column_data = data.iloc[:200, 0].dropna().values.astype(np.float64)
        file_min_pvalue = np.min(column_data[column_data > 0])  
        global_min_pvalue = min(global_min_pvalue, file_min_pvalue)
    except Exception as e:
        print(f"⚠️ Error processing file {file_name}: {e}")


y_min = max(global_min_pvalue / 10, 1e-300)  
y_max = 1  

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(22, 5), sharex=True, sharey=True)
axes = axes.flatten()  

for idx, (file_name, label, color) in enumerate(zip(file_names, labels, colors)):
    ax = axes[idx] 
    try:
        data = pd.read_csv(file_name, header=None)


        column_data = data.iloc[:200, 0].dropna().values.astype(np.float64)
        column_data[column_data == 0] = y_min  

        x = np.arange(len(column_data))  

        ax.plot(x, column_data , color=color, linewidth=3, marker='o', markersize=5)

        ax.set_yscale("log")

        ax.set_ylim(y_min, y_max)

        ax.set_title(label, fontsize=16, fontweight='bold', color=color, pad=10)

        ax.set_xlabel("Virtual Node", fontsize=14, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("P-Value", fontsize=14, fontweight='bold')

        ax.set_xlim(0, 200)
        xticks = [0, 40, 80, 120, 160, 200]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(tick) for tick in xticks], fontsize=14, fontweight='bold')

        ax.tick_params(axis='y', labelsize=14, width=2, length=6)
        yticks = np.logspace(np.log10(y_min), np.log10(y_max), num=5)  
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{tick:.1e}" for tick in yticks], fontsize=14, fontweight='bold')
 
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

    except Exception as e:
        print(f"⚠️ Error processing file {file_name}: {e}")

fig.suptitle("p for Vd 0.3V (100000)", fontsize=22, fontweight='bold')
plt.subplots_adjust(top=0.85)  

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("/Volumes/Extreme SSD/20250118_3/plot/p_100000_plot_Vd0.3V_STM_0-200.png", dpi=600)
plt.show()