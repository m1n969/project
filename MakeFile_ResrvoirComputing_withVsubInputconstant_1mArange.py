import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import csv
import shutil
import os

def File_Header(FileName):
    with open(FileName, 'a') as fout:
        fout.write('<Application>\n')
        fout.write('SAMPLING\n')
        fout.write('</Application>\n')
        fout.write('<Setting>\n')

def File_Footer(FileName):
    with open(FileName, 'a') as fout:
        fout.write('</Setting>\n')

def ChSetting(FileName,TabChNo,SystemChNo,IMeasRange,Meas=True,OperationMode=0,MeasMode=0,ForceRange=0,VMeasRange=0,HwSkew=0.0,RawData=False):
    if TabChNo < 0 or TabChNo > 4:
        print("TabChNo is incorrect")
        exit()
    with open(FileName, 'a') as fout:
        CheckBoxChNo = TabChNo if TabChNo != 1 else 11
        Flag = 'True' if Meas else 'False'
        fout.write('checkBoxCh' + str(CheckBoxChNo) + 'Meas:' + Flag + '\n')
        Ch_dic = {201:2, 202:3, 301:4, 302:5}
        fout.write('comboBoxCh' + str(TabChNo) + 'Ch:' + str(Ch_dic[SystemChNo]) +':' + str(SystemChNo) + '\n') 
        OpMode_dic = {0:'Fast IV mode', 1:'PG Mmode'}
        if CheckBoxChNo != 2:
            fout.write('comboBoxCh' + str(TabChNo) + 'OperationMode:' + str(OperationMode) + ':' + str(OpMode_dic[OperationMode]) + '\n')
        else:
            fout.write('comboBoxSource' + str(TabChNo) + 'OperationMode:' + str(OperationMode) + ':' + str(OpMode_dic[OperationMode]) + '\n')
        MeasMode_dic = {0:'V Measurement',1:'I Measurement'}
        fout.write('comboBoxCh' + str(TabChNo) + 'MeasMode:' + str(MeasMode) + ':' + str(MeasMode_dic[MeasMode]) + '\n')
        if TabChNo == 1 or 2:
            ForceRange_dic = {0:'Auto', 1:'+/- 3 V', 2:'+/- 5 V', 3:'-10 V to 0 V', 4:'0 V to +10 V'}
        else:
            ForceRange_dic = {0:'+/- 5 V', 1:'-10 V to 0 V', 2:'0 V to +10 V'}
        fout.write('comboBoxCh' + str(TabChNo) + 'VForceRange:'+ str(ForceRange) + ':' + str(ForceRange_dic[ForceRange]) + '\n')
        IMeasRange_dic = {0:'+/-1 uA Fixed',1:'+/-10 uA Fixed',2:'+/-100 uA Fixed',3:'+/-1 mA Fixed',4:'+/-10 mA Fixed'}
        fout.write('comboBoxCh' + str(TabChNo) + 'IMeasRange:'+ str(IMeasRange) + ':' + str(IMeasRange_dic[IMeasRange]) + '\n')
        VMeasRange_dic = {0:'+/- 5 V'}
        fout.write('comboBoxCh' + str(TabChNo) + 'VMeasRange:'+ str(VMeasRange) + ':' + str(VMeasRange_dic[VMeasRange]) + '\n')
        fout.write('textBoxCh' + str(TabChNo) + 'HwSkew:' + str(HwSkew) + '\n')
        fout.write('checkBoxCh' + str(TabChNo) + 'RawData:' + ( 'True' if RawData else 'False') + '\n')
    return 0

def BiasSetting(FileName):
    with open(FileName, 'a') as fout:
        fout.write("comboBoxBiasSource1Type:0:N/A\n")
        fout.write("comboBoxBiasSource1Ch:0:N/A\n")
        fout.write("textBoxBiasSource1BiasV:0.0\n")
        fout.write("textBoxBiasSource1Compliance:0.0\n")
        fout.write("comboBoxBiasSource2Type:0:N/A\n")
        fout.write("comboBoxBiasSource2Ch:0:N/A\n")
        fout.write("textBoxBiasSource2BiasV:0.0\n")
        fout.write("textBoxBiasSource2Compliance:0.0\n")

def WaveformSetting(FileName,TabChNo,WaveformDf):
    WaveformDf['new_col_1'] = ''
    with open(FileName, 'a') as fout:
        fout.write('dataGridViewCh' + str(TabChNo)  + ':' + str(len(WaveformDf) + 1) + '\n')
    WaveformDf.to_csv(FileName,index=False,header=False,mode='a')
    with open(FileName, 'a') as fout:    
        fout.write('null,null,\n')

def MeasurementSetting(FileName,MeasurementDf):
    MeasurementDf['new_col_1'] = ''
    MeasurementDf[1] = MeasurementDf[1].astype(int)
    with open(FileName, 'a') as fout:
        fout.write('dataGridViewMeasurement:' + str(len(MeasurementDf) + 1) + '\n')
    MeasurementDf.to_csv(FileName,index=False,header=False,mode='a')
    with open(FileName, 'a') as fout:    
        fout.write('null,null,null,null,\n')

def TriangleWaveMake(Width, HighV, MidV, LowV, ZeroOneVector):
    Data = np.zeros((ZeroOneVector.shape[0] * 2 + 1, 2))
    for i in range(ZeroOneVector.shape[0]):
        Data[2 * i + 0][0] = round(i * Width, 7)
        Data[2 * i + 1][0] = round(i * Width + Width / 2, 7)
        Data[2 * i + 0][1] = MidV
        Data[2 * i + 1][1] = HighV if ZeroOneVector[i] == 1 else LowV
    Data[-1][0] = round(Width * ZeroOneVector.shape[0], 7)
    Data[-1][1] = MidV
    return Data

def RepeatSetting(FileName,Repeat=1):
    with open(FileName, 'a') as fout:
        fout.write('textRepeat:' + str(int(Repeat)) + '\n')

def SequenceSetting(FileName,SeuquenceDf=pd.DataFrame()):
    with open(FileName, 'a') as fout:    
        fout.write('dataGridViewSeq:' + str(len(SeuquenceDf) + 1) + '\n')
    SeuquenceDf.to_csv(FileName,index=False,header=False,mode='a')
    with open(FileName, 'a') as fout:    
        fout.write('null,null,\n')

def ReadRandomData(SeedNo):
    RandomDataFrame = pd.read_csv("/Volumes/Extreme SSD/20241114/SourceCode_Measurement/Seed.csv")
    RandomMatrix = RandomDataFrame.values.T
    return RandomMatrix[SeedNo]

def MakeFile(FileName):
    File_Header(FileName)
    ChSetting(FileName=FileName,TabChNo=1,SystemChNo=201,IMeasRange=4,MeasMode=0,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=2,SystemChNo=202,IMeasRange=3,MeasMode=1,ForceRange=2)
    ChSetting(FileName=FileName,TabChNo=3,SystemChNo=301,IMeasRange=3,MeasMode=1,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=4,SystemChNo=302,IMeasRange=3,MeasMode=1,ForceRange=0)
    BiasSetting(FileName)
    Vector = ReadRandomData(0)
    Wave = pd.DataFrame(TriangleWaveMake(4e-6,3.5,0.5,-2.5,Vector))
    WaveformSetting(FileName = FileName,TabChNo=1,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0.3],[0.004,0.3]])
    WaveformSetting(FileName = FileName,TabChNo=2,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.004,0]])
    WaveformSetting(FileName = FileName,TabChNo=3,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.004,0]])
    WaveformSetting(FileName = FileName,TabChNo=4,WaveformDf=Wave)

    Meas = pd.DataFrame(np.array([[0,int(0),2.00E-08,2.00E-08]]))
    MeasurementSetting(FileName=FileName,MeasurementDf=Meas)

    RepeatSetting(FileName=FileName)
    SequenceSetting(FileName=FileName)
    File_Footer(FileName)

def MakeFile_aging(FileName,Repeatcount):
    File_Header(FileName)
    ChSetting(FileName=FileName,TabChNo=1,SystemChNo=201,IMeasRange=4,MeasMode=0,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=2,SystemChNo=202,IMeasRange=3,MeasMode=1,ForceRange=2)
    ChSetting(FileName=FileName,TabChNo=3,SystemChNo=301,IMeasRange=3,MeasMode=1,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=4,SystemChNo=302,IMeasRange=3,MeasMode=1,ForceRange=0)
    BiasSetting(FileName)
    Vector = ReadRandomData(0)
    Wave = pd.DataFrame(TriangleWaveMake(4e-6,3.5,0.5,-2.5,Vector))
    WaveformSetting(FileName = FileName,TabChNo=1,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0.3],[0.004,0.3]])
    WaveformSetting(FileName = FileName,TabChNo=2,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.004,0]])
    WaveformSetting(FileName = FileName,TabChNo=3,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.004,0]])
    WaveformSetting(FileName = FileName,TabChNo=4,WaveformDf=Wave)

    Meas = pd.DataFrame(np.array([[0,int(0),2.00E-08,2.00E-08]]))
    MeasurementSetting(FileName=FileName,MeasurementDf=Meas)

    RepeatSetting(FileName=FileName,Repeat=Repeatcount)
    SequenceSetting(FileName=FileName)
    File_Footer(FileName)

def MakeFile_PV1kHz(FileName,MiddleVoltage,SwingVoltage):
    File_Header(FileName)
    ChSetting(FileName=FileName,TabChNo=1,SystemChNo=201,IMeasRange=4,MeasMode=0,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=2,SystemChNo=202,IMeasRange=1,MeasMode=1,ForceRange=2)
    ChSetting(FileName=FileName,TabChNo=3,SystemChNo=301,IMeasRange=1,MeasMode=1,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=4,SystemChNo=302,IMeasRange=1,MeasMode=1,ForceRange=0)
    BiasSetting(FileName)
    Wave = pd.DataFrame([[0,0],
                         [0.00025,MiddleVoltage-SwingVoltage],[0.00075,MiddleVoltage+SwingVoltage],[0.001,MiddleVoltage],
                         [0.00125,MiddleVoltage-SwingVoltage],[0.00175,MiddleVoltage+SwingVoltage],[0.002,MiddleVoltage],
                         [0.00225,MiddleVoltage-SwingVoltage],[0.00275,MiddleVoltage+SwingVoltage],[0.003,MiddleVoltage],
                         [0.00325,MiddleVoltage-SwingVoltage],[0.00375,MiddleVoltage+SwingVoltage],[0.004,MiddleVoltage],
                         [0.00425,MiddleVoltage-SwingVoltage],[0.00475,MiddleVoltage+SwingVoltage],[0.005,MiddleVoltage]])
    WaveformSetting(FileName = FileName,TabChNo=1,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.005,0]])
    WaveformSetting(FileName = FileName,TabChNo=2,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.005,0]])
    WaveformSetting(FileName = FileName,TabChNo=3,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.005,0]])
    WaveformSetting(FileName = FileName,TabChNo=4,WaveformDf=Wave)

    Meas = pd.DataFrame(np.array([[0,int(5000),1.00E-06,1.00E-06]]))
    MeasurementSetting(FileName=FileName,MeasurementDf=Meas)

    RepeatSetting(FileName=FileName)
    SequenceSetting(FileName=FileName)
    File_Footer(FileName)

def MakeFile_PV125kHz(FileName,MiddleVoltage,SwingVoltage):
    File_Header(FileName)
    ChSetting(FileName=FileName,TabChNo=1,SystemChNo=201,IMeasRange=4,MeasMode=0,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=2,SystemChNo=202,IMeasRange=3,MeasMode=1,ForceRange=2)
    ChSetting(FileName=FileName,TabChNo=3,SystemChNo=301,IMeasRange=3,MeasMode=1,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=4,SystemChNo=302,IMeasRange=3,MeasMode=1,ForceRange=0)
    BiasSetting(FileName)
    Wave = pd.DataFrame([[0,0],
                         [0.000002,MiddleVoltage-SwingVoltage],[0.000006,MiddleVoltage+SwingVoltage],[0.000008,MiddleVoltage],
                         [0.000010,MiddleVoltage-SwingVoltage],[0.000014,MiddleVoltage+SwingVoltage],[0.000016,MiddleVoltage],
                         [0.000018,MiddleVoltage-SwingVoltage],[0.000022,MiddleVoltage+SwingVoltage],[0.000024,MiddleVoltage],
                         [0.000026,MiddleVoltage-SwingVoltage],[0.000030,MiddleVoltage+SwingVoltage],[0.000032,MiddleVoltage],
                         [0.000034,MiddleVoltage-SwingVoltage],[0.000038,MiddleVoltage+SwingVoltage],[0.000040,MiddleVoltage]])
    WaveformSetting(FileName = FileName,TabChNo=1,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.000040,0]])
    WaveformSetting(FileName = FileName,TabChNo=2,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.000040,0]])
    WaveformSetting(FileName = FileName,TabChNo=3,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.000040,0]])
    WaveformSetting(FileName = FileName,TabChNo=4,WaveformDf=Wave)

    Meas = pd.DataFrame(np.array([[0,int(4000),1.00E-08,1.00E-08]]))
    MeasurementSetting(FileName=FileName,MeasurementDf=Meas)

    RepeatSetting(FileName=FileName)
    SequenceSetting(FileName=FileName)
    File_Footer(FileName)

def MakeFile_cycling_triangular(FileName,Repeatcount):
    File_Header(FileName)
    ChSetting(FileName=FileName,TabChNo=1,SystemChNo=201,IMeasRange=4,MeasMode=0,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=2,SystemChNo=202,IMeasRange=3,MeasMode=1,ForceRange=2)
    ChSetting(FileName=FileName,TabChNo=3,SystemChNo=301,IMeasRange=3,MeasMode=1,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=4,SystemChNo=302,IMeasRange=3,MeasMode=1,ForceRange=0)
    BiasSetting(FileName)
    WaveData = np.zeros((202,2))
    WaveData[:,1] = (np.arange(202) % 2) * 6 - 2.5
    WaveData[0,1] = 0.5
    WaveData[201,1] = 0.5
    WaveData[:,0] = np.arange(202) * 4e-6 - 2e-6
    WaveData[0,0] = 0
    WaveData[201,0] = 8e-4
    Wave = pd.DataFrame(WaveData)
    WaveformSetting(FileName = FileName,TabChNo=1,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0.3],[0.0008,0]])
    WaveformSetting(FileName = FileName,TabChNo=2,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.0008,0]])
    WaveformSetting(FileName = FileName,TabChNo=3,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.0008,0]])
    WaveformSetting(FileName = FileName,TabChNo=4,WaveformDf=Wave)

    Meas = pd.DataFrame(np.array([[0,int(0),2.00E-08,2.00E-08]]))
    MeasurementSetting(FileName=FileName,MeasurementDf=Meas)

    RepeatSetting(FileName=FileName,Repeat=Repeatcount)
    SequenceSetting(FileName=FileName)
    File_Footer(FileName)

def MakeFile_cycling_rectangular(FileName,Repeatcount):
    File_Header(FileName)
    ChSetting(FileName=FileName,TabChNo=1,SystemChNo=201,IMeasRange=4,MeasMode=0,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=2,SystemChNo=202,IMeasRange=3,MeasMode=1,ForceRange=2)
    ChSetting(FileName=FileName,TabChNo=3,SystemChNo=301,IMeasRange=3,MeasMode=1,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=4,SystemChNo=302,IMeasRange=3,MeasMode=1,ForceRange=0)
    BiasSetting(FileName)
    WaveData = np.zeros((401,2))

    repeat_times = 100
    small_array = np.array([3.5, 3.5, -2.5, -2.5])
    large_array = np.tile(small_array, repeat_times)
    small_array2 = np.array([0, 3.99e-6, 4e-6, 7.99e-6])
    large_array2 = np.tile(small_array2, repeat_times) + np.array(np.arange(400) / 4, dtype=np.int32) * 8e-6

    WaveData[0:400,0] = large_array2
    WaveData[0:400,1] = large_array
    WaveData[400,0] = 8e-4
    WaveData[400,1] = 3.5
    Wave = pd.DataFrame(WaveData)
    WaveformSetting(FileName = FileName,TabChNo=1,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0.3],[0.0008,0]])
    WaveformSetting(FileName = FileName,TabChNo=2,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.0008,0]])
    WaveformSetting(FileName = FileName,TabChNo=3,WaveformDf=Wave)
    Wave = pd.DataFrame([[0,0],[0.0008,0]])
    WaveformSetting(FileName = FileName,TabChNo=4,WaveformDf=Wave)

    Meas = pd.DataFrame(np.array([[0,int(0),2.00E-08,2.00E-08]]))
    MeasurementSetting(FileName=FileName,MeasurementDf=Meas)

    RepeatSetting(FileName=FileName,Repeat=Repeatcount)
    SequenceSetting(FileName=FileName)
    File_Footer(FileName)

def MakeFile_ReservoirComputing(FileName,SeedId,Vg_mid,Vg_swing1,Vg_swing2,Vsub_mid,Vsub_swing,Vsub_delay,Vd):
    PulseWidth = 4e-6
    
    File_Header(FileName)
    ChSetting(FileName=FileName,TabChNo=1,SystemChNo=201,IMeasRange=4,MeasMode=0,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=2,SystemChNo=202,IMeasRange=3,MeasMode=1,ForceRange=2)
    ChSetting(FileName=FileName,TabChNo=3,SystemChNo=301,IMeasRange=3,MeasMode=1,ForceRange=0)
    ChSetting(FileName=FileName,TabChNo=4,SystemChNo=302,IMeasRange=4,MeasMode=0,ForceRange=0)
    BiasSetting(FileName)

    Vector = ReadRandomData(SeedId)
    Wave = pd.DataFrame(TriangleWaveMake(PulseWidth, Vg_mid + Vg_swing1, Vg_mid, Vg_mid - Vg_swing2, Vector))
    WaveformSetting(FileName = FileName,TabChNo=1,WaveformDf=Wave)

    Wave = pd.DataFrame([[0,Vd],[0.004,Vd]])
    WaveformSetting(FileName = FileName,TabChNo=2,WaveformDf=Wave)

    Wave = pd.DataFrame([[0,0],[0.004,0]])
    WaveformSetting(FileName = FileName,TabChNo=3,WaveformDf=Wave)
    
    SubVector = ReadRandomData(SeedId)
    SubVector[Vsub_delay:] = SubVector[:SubVector.shape[0]-Vsub_delay]
    Wave = pd.DataFrame(TriangleWaveMake(PulseWidth,Vsub_mid+Vsub_swing,Vsub_mid,Vsub_mid-Vsub_swing,SubVector))
    WaveformSetting(FileName = FileName,TabChNo=4,WaveformDf=Wave)

    Meas = pd.DataFrame(np.array([[0,int(200000),2.00E-08,2.00E-08]]))
    MeasurementSetting(FileName=FileName,MeasurementDf=Meas)

    RepeatSetting(FileName=FileName)
    SequenceSetting(FileName=FileName)
    File_Footer(FileName)

##MakeFile_ReservoirComputing(FileName,SeedId,Vg_mid,Vg_swing1,Vg_swing2,Vsub_mid,Vsub_swing,Vsub_delay,Vd)
for Vsub_mid in [0]:
    for Vgswing_2 in [3,2.8,2.6,2.4,2.2,2,1.8,1.6,1.4,1.2,1,0.8,0.6,0.4,0.2,0]:
        for Vd in [0.5]:
            DirName = "Vgswing" + str(int(Vgswing_2 * 1000)) + "mV" + "_" + "Vd" + str(int(Vd * 1000)) + "mV"
            os.mkdir("/Volumes/Extreme SSD/20250411//From_0mV_to_3000mV//RawData//" + DirName)  # 这里要缩进到 Vd 循环内
            for i in range(10):
                MakeFile_ReservoirComputing(
                    "/Volumes/Extreme SSD/20250411//From_0mV_to_3000mV//RawData//" + DirName + "//" + "Id" + str(i) + ".cf",
                    i, 0.5, 3, Vgswing_2, Vsub_mid, 0, 0, Vd
                )