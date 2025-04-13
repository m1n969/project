import numpy as np
import matplotlib.pyplot as plt

vsub_values = ["0","0.2","0.4","0.6","0.8","1","1.2","1.4","1.6","1.8","2", "2.2", "2.4","2.6","2.8","3"]
y1 = [1.554159324,1.788295344,1.894477335,1.942898695,1.999584279,2.028386897,2.026165944,2.027949824,2.034470829,2.03248935,2.033967268,2.023224103,2.026625427,2.037916954,2.034617388,2.031443107]  # 互换后的 Vsw 3V (10mA range)
y2 = [1.867209304	,1.880596936	,1.879328716,	1.880769258,	1.902649158	,1.901762646]  # 互换后的 Vsw 3V (1mA range)
y3 = [1.915966518	,2.00365957	,2.220103636,	2.25415426	,2.104004547 ]  # Vsw 3.5V (1mA range)
yerr1 = [0.064093633,0.015428227,0.02598357,0.019006297,0.035469616,0.031860103,0.026746451,0.024613677,0.03382473,0.030698866,0.034119883,0.034044374,0.039500942,0.044617261,0.037509026,0.040007356]
yerr2 = [0.018246223	,0.01015337	,0.016999087,	0.007852396,	0.01210663	,0.010175807]
x = np.arange(len(vsub_values))

plt.figure(figsize=(18, 8))

plt.errorbar(x, y1,  yerr=yerr1,fmt='o-', linewidth=10, markersize=36, color='black', label='Vd 0.3V (1mA range)')  
#plt.errorbar(x, y2, yerr=yerr2,fmt='s-', linewidth=6, markersize=30, color='black', label='Vd 0.5V (1mA range)')  
#plt.plot(x, y3, '^-', linewidth=6, markersize=30, color='black', label='Vsw 3.5V (1mA range)')

plt.title('XOR(Measured From 0V to 3V)', fontsize=30, fontweight='bold')
plt.ylabel('MC', fontsize=28, fontweight='bold')
plt.xlabel('Vgswing (V)',fontsize=28, fontweight='bold')

plt.xticks(x, vsub_values, fontsize=16, fontweight='bold')  # 加大字体
plt.yticks(fontsize=28, fontweight='bold')  # 加大字体
plt.tick_params(axis='both', which='major', labelsize=28, width=6, length=8)  # 加粗刻度

plt.gca().spines['top'].set_linewidth(8)
plt.gca().spines['right'].set_linewidth(8)
plt.gca().spines['left'].set_linewidth(8)
plt.gca().spines['bottom'].set_linewidth(8)
plt.tight_layout()

legend = plt.legend(fontsize=30, markerscale=0.6,loc="lower right")

plt.savefig('/Volumes/Extreme SSD/20250410_0_3000/MC_XOR_IdIs.png', dpi=600)
plt.show()
