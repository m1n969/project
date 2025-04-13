##This code runs without Reservoir Computing Framework
##OutputFile
##~.csv :Normal output
##~_CC.csv :Accuracy of similar pattern
##~_Accuracy.csv :Accuracy version of normal output
##~_PCA.csv :Accuracy of each input pattern


from email import header
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import csv

def ShuffleSeedGenerator(N):
    x = np.arange(N)
    np.random.shuffle(x)
    return x

def ShuffleBySeed(X,seed):
    DataLength = X.shape[0]
    SeedLength = seed.shape[0]
    OutData = np.zeros(X.shape)
    for i in range(DataLength):
        OutData[i] = X[seed[i]]
    return OutData

#ShortTermMmemoryタスクのベクトルの作成
def CreateSTMVector(Y,Length,Tdelay):
    Ysize = Y.size
    Vector = Y[ Ysize-(Tdelay + Length) : Ysize-Tdelay]
    return Vector

#ParityCheckタスクのベクトルの作成
def CreatePCVector(Y,Length,Tdelay):
    Ysize = Y.size
    BaseVector = Y[Ysize -Tdelay - Length :Ysize -Tdelay]
    for i in range(Tdelay):
        BaseVector = BaseVector + Y[Ysize - (i + Length) :Ysize - i]
    return BaseVector % 2

#TemporalXORタスクのベクトルの作成
def CreateTemporalXORVector(Y,Length,Tdelay):
    Ysize = Y.size
    BaseVector = (Y[Ysize -Tdelay - Length :Ysize -Tdelay] + Y[Ysize - Length :Ysize])%2
    return BaseVector

#SUMベクトルの作成
def CreateSUMVector(Y,Length,Tdelay):
    Ysize = Y.size
    BaseVector = np.zeros(Length)
    for i in range(Tdelay+1):
        BaseVector += (2 ** (Tdelay - i)) * Y[Ysize - (i + Length) :Ysize - i]
    return BaseVector 

## Load training file (return ndarray)
def GetVirtualNodesData(DirectoryName,FilePrefix="Id",VirtualNodes=200,PullingNo=2,ChannelNo=202,OneVectorFlag=True,MaxFileIndex=10,AveragingFlag=False):
    if AveragingFlag:
        print("Averaging Mode")
    else:
        print("Subsampling Mode")

    for i in range(MaxFileIndex):
        Path = "/Volumes/Extreme SSD/20250410_3000_0/Vgswing1000mV_Vd300mV/RawData/" + DirectoryName + "/" + FilePrefix + str(i) + "_out_Channel" + str(ChannelNo) + ".csv"
        InputDf = pd.read_csv(Path,header = None)
        if AveragingFlag:
            RawInputDfdata = np.squeeze(InputDf.values[:,1])
            PullingInput1DArray = np.average(RawInputDfdata.reshape(-1,PullingNo),axis=1)
        else:
            PullingInputDf = InputDf[::PullingNo]
            PullingInput1DArray = np.squeeze(PullingInputDf.values[:,1])
        PullingInputArray = PullingInput1DArray.reshape([int(PullingInput1DArray.size/VirtualNodes) , VirtualNodes ])
        if i==0:
            if OneVectorFlag:
                DataArray = np.empty((MaxFileIndex,int(PullingInput1DArray.size/VirtualNodes),VirtualNodes + 1))
            else:
                DataArray = np.empty((MaxFileIndex,int(PullingInput1DArray.size/VirtualNodes),VirtualNodes))
        if OneVectorFlag:
            OneVec = np.ones([ int(PullingInput1DArray.size/VirtualNodes) , 1 ])
            PullingInputArray = np.hstack((PullingInputArray,OneVec))
        DataArray[i]=PullingInputArray
    return DataArray

## Load seed file (return ndarray)
def GetSeedData(SeedFileName="Seed.csv"):
    Data = pd.read_csv("/Volumes/Extreme SSD/20241108/Vsub2TimeStepDelayVd500mV/SourceCode/Seed.csv")
    return Data.values.T

## Load tarrget vector (return ndarray)
def GetTaskDataFromSeedData(SeedData,Task,Tdelay=0,FullLength=5000):
    TaskData = np.empty(FullLength)
    Length = int(FullLength/SeedData.shape[0])
    for i in range(SeedData.shape[0]):
        if Task == "STM":
            TaskData[i * Length:(i+1) * Length] = CreateSTMVector(SeedData[i],Length,Tdelay)
        elif Task == "PC":
            TaskData[i * Length:(i+1) * Length] = CreatePCVector(SeedData[i],Length,Tdelay)
        elif Task == "XOR":
            TaskData[i * Length:(i+1) * Length] = CreateTemporalXORVector(SeedData[i],Length,Tdelay)
        elif Task == "SUM":
            TaskData[i * Length:(i+1) * Length] = CreateSUMVector(SeedData[i],Length,Tdelay)
        else:
            print("Error:unknown task")
            exit()
    return TaskData

def MergeVirtualNodesData(DataArray,FullLength):
    Length = int(FullLength / DataArray.shape[0])
    OutputArray = np.empty((FullLength,DataArray.shape[2]))
    for i in range(DataArray.shape[0]):
        OutputArray[i*Length:(i+1)*Length] = DataArray[i,-Length:]
    return OutputArray

def get_ridgelsqW_np(x, t, RidgeArray):##x=(L,VN),t(Tasks,L),RidgeArray(RidgeL) Output(Ridge,tasks,VN)
    RidgeL = RidgeArray.shape[0]
    VN = x.shape[1]
    xx = np.dot(x.T, x)
    xx_Ridge = np.empty((RidgeL,VN,VN))##(RidgeL,L,L)
    for i in range(RidgeArray.shape[0]):
        xx_Ridge[i] = xx + RidgeArray[i] * np.identity(VN)
    xx_inv = np.linalg.inv(xx_Ridge)
    xt = np.dot(x.T, t.T)
    W = np.dot(xx_inv, xt)
    W = W.transpose(0,2,1)
    return W

## Ridge regression and crosscheck
def RidgeCrossCheck(Vectors,Matrix,RidgeMax,RidgeMin,RidgeDense,CrossCheckParameter=5):
    Length = int(Vectors.shape[1])
    Domain = int(Length/CrossCheckParameter)
    VirtualNode = int(Matrix.size/Length)
    OutWeight = np.empty((CrossCheckParameter,RidgeDense,Vectors.shape[0],VirtualNode))
    OutTargetVector = np.empty((CrossCheckParameter,RidgeDense,Vectors.shape[0],Domain))##(CC,Ridge,10,50)
    OutInferenceResult = np.empty((CrossCheckParameter,RidgeDense,Vectors.shape[0],Domain))

    for TestNo in range(CrossCheckParameter):
        TestVector = Vectors[:,TestNo*Domain : (TestNo+1)*Domain]
        TestMatrix = Matrix[TestNo*Domain : (TestNo+1)*Domain]
        TrainingVector = np.hstack([Vectors[:,:TestNo*Domain],Vectors[:,(TestNo+1)*Domain:]])
        TrainingMatrix = np.vstack([Matrix[:TestNo*Domain],Matrix[(TestNo+1)*Domain:]])
        Weight = get_ridgelsqW_np(TrainingMatrix,TrainingVector,np.geomspace(RidgeMax, RidgeMin, RidgeDense))##x=(L,VN),t(L,Tasks),RidgeArray(RidgeL) Output(Ridge,Tasks,VN)
        OutTargetVector[TestNo] = TestVector
        Weight = Weight.transpose(0,2,1)
        OutInferenceResult[TestNo] = np.dot(TestMatrix,Weight).transpose(1,2,0)
        Weight = Weight.transpose(0,2,1)
        OutWeight[TestNo] = Weight
    return OutTargetVector.transpose(2,1,0,3),OutInferenceResult.transpose(2,1,0,3),OutWeight.transpose(2,1,0,3)

def CapacityCalculation(TargetVectorArray,InferenceResultArray,ThresholdFlag=False,Fill0inTdelay0=False):
    ##TdelayMax,RidgeDense,CrossCheckParameter,FullLength
    TdelayMax = TargetVectorArray.shape[0]
    RidgeDense = TargetVectorArray.shape[1]
    CrossCheckParameter = TargetVectorArray.shape[2]
    CorArrayMatrix = np.zeros((RidgeDense,TdelayMax,CrossCheckParameter))
    AccuracyArrayMatrix = np.zeros((RidgeDense,TdelayMax,CrossCheckParameter))
    for Tdelay in range(TdelayMax):
        for Ridge in range(RidgeDense):
            for CrossCheckNo in range(CrossCheckParameter):
                if ThresholdFlag == False:
                    Cor = np.corrcoef(InferenceResultArray[Tdelay,Ridge,CrossCheckNo],TargetVectorArray[Tdelay,Ridge,CrossCheckNo])[0,1]
                    if Fill0inTdelay0 == False or Tdelay != 0:
                        CorArrayMatrix[Ridge,Tdelay,CrossCheckNo] = Cor**2
                else:
                    Cor = np.corrcoef(np.where(InferenceResultArray[Tdelay,Ridge,CrossCheckNo] < 0.5,0,1),TargetVectorArray[Tdelay,Ridge,CrossCheckNo])[0,1]
                    Accuracy = np.count_nonzero(np.where(InferenceResultArray[Tdelay,Ridge,CrossCheckNo] < 0.5,0,1)==TargetVectorArray[Tdelay,Ridge,CrossCheckNo])/len(InferenceResultArray[Tdelay,Ridge,CrossCheckNo])
                    if Fill0inTdelay0 == False or Tdelay != 0:
                        CorArrayMatrix[Ridge,Tdelay,CrossCheckNo] = Cor**2
                        AccuracyArrayMatrix[Ridge,Tdelay,CrossCheckNo] = Accuracy
                
    
    RidgeCapacityArray = np.empty((RidgeDense))
    for Ridge_i in range(RidgeDense):
        RidgeCapacityArray[Ridge_i] = np.sum(CorArrayMatrix[Ridge_i,1:,:])/CrossCheckParameter
    MaxCapacityIndex = RidgeCapacityArray.argmax()
    
    return MaxCapacityIndex,CorArrayMatrix,AccuracyArrayMatrix

def SUMtoHistory(X,Bit):
    if X < 0 or X > 2**Bit-1:
        print("Error:SUMtoHistory",X)
        exit()
    FormatString = '0' + str(Bit) + 'b'
    BinX = format(X,FormatString)
    BinXInv = BinX[::-1]
    result = "->".join([char for char in BinXInv])
    return result

def PatternCheckAccuracy(TargetDelay,RawVector,TargetVectorArray,InferenceResultArray,RandomSeed,MaxCapacityIndex):
    SUMVector = GetTaskDataFromSeedData(RawVector,"SUM",TargetDelay)
    SUMVector = ShuffleBySeed(SUMVector,RandomSeed)
    CheckTVA = TargetVectorArray[TargetDelay,MaxCapacityIndex].ravel()
    CheckIRA = InferenceResultArray[TargetDelay,MaxCapacityIndex].ravel()
    CheckIRAwTh = np.where(CheckIRA < 0.5,0,1)
    cols = ['input_history','Acc']
    df = pd.DataFrame(index=[], columns=cols)
    for i in range(2**(TargetDelay+1)):
        TVAsp = np.zeros(np.count_nonzero(SUMVector == i))
        IRAwThsp = np.zeros(np.count_nonzero(SUMVector == i))
        Count = 0
        for j in range(5000):
            if SUMVector[j] == i:
                TVAsp[Count] = CheckTVA[j]
                IRAwThsp[Count] = CheckIRAwTh[j]
                Count+=1
        Acc = np.count_nonzero(np.where(IRAwThsp==TVAsp))/len(IRAwThsp)
        record = pd.Series([SUMtoHistory(i,TargetDelay+1),Acc], index=df.columns)
        df = df._append(record, ignore_index=True)
    return df

def ClassificationCheck(TargetDelay,RawVector,TargetVectorArray,InferenceResultArray,RandomSeed,MaxCapacityIndex):
    SUMVector = GetTaskDataFromSeedData(RawVector,"SUM",TargetDelay)
    SUMVector = ShuffleBySeed(SUMVector,RandomSeed)
    CheckTVA = TargetVectorArray[TargetDelay,MaxCapacityIndex].ravel()
    CheckIRA = InferenceResultArray[TargetDelay,MaxCapacityIndex].ravel()
    CheckIRAwTh = np.where(CheckIRA < 0.5,0,1)
    cols = ['input_even', 'input_odd','Cor2','Acc']
    df = pd.DataFrame(index=[], columns=cols)
    for i in range(2**TargetDelay):
        TVAsp = np.zeros(np.count_nonzero(SUMVector == 2*i)+np.count_nonzero(SUMVector == 2*i+1))
        IRAsp = np.zeros(np.count_nonzero(SUMVector == 2*i)+np.count_nonzero(SUMVector == 2*i+1))
        IRAwThsp = np.zeros(np.count_nonzero(SUMVector == 2*i)+np.count_nonzero(SUMVector == 2*i+1))
        Count = 0
        for j in range(5000):
            if SUMVector[j] == 2*i or SUMVector[j] == 2*i+1:
                TVAsp[Count] = CheckTVA[j]
                IRAsp[Count] = CheckIRA[j]
                IRAwThsp[Count] = CheckIRAwTh[j]
                Count+=1
        Cor2 = np.corrcoef(TVAsp,IRAsp)[0,1] **2
        Acc = np.count_nonzero(np.where(IRAwThsp==TVAsp))/len(IRAwThsp)
        record = pd.Series([SUMtoHistory(2*i,TargetDelay+1),SUMtoHistory(2*i+1,TargetDelay+1),Cor2,Acc], index=df.columns)
        df = df._append(record, ignore_index=True)
    return df



def STMandPC(DirectoryName,Task,CN,VirtualNode,Pulling,ThresholdFlag,AvgFlag=False):
    RidgeDense = 51
    CrossCheckParameter = 5
    TdelayMax = 10
    RidgeMax = 1e-7
    RidgeMin = 1e-12
    ##RandomSeed = np.load("/Volumes/Extreme SSD/20241108/Vsub2TimeStepDelayVd500mV/SourceCode/Rand5000.npy")##ShuffleSeedGenerator(5000)
    print("Loading:",DirectoryName,CN)
    RawMatrixId = GetVirtualNodesData(DirectoryName,FilePrefix="Id",VirtualNodes=VirtualNode,ChannelNo=202,PullingNo=Pulling,OneVectorFlag=False,AveragingFlag=AvgFlag)
    RawMatrixIs = GetVirtualNodesData(DirectoryName,FilePrefix="Id",VirtualNodes=VirtualNode,ChannelNo=301,PullingNo=Pulling,OneVectorFlag=False,AveragingFlag=AvgFlag)
    RawMatrix = np.concatenate([RawMatrixId,RawMatrixIs],axis=2)
    Matrix = MergeVirtualNodesData(RawMatrix,5000)
    RawVector = GetSeedData()
    TargetVectorArray = np.empty((TdelayMax,RidgeDense,CrossCheckParameter,int(5000/CrossCheckParameter)))
    InferenceResultArray = np.empty((TdelayMax,RidgeDense,CrossCheckParameter,int(5000/CrossCheckParameter)))
    Vectors = np.empty((TdelayMax,5000))
    print("Task=",Task)
    for Tdelay in range(TdelayMax):
        Vector = GetTaskDataFromSeedData(RawVector,Task,Tdelay)
        Vectors[Tdelay] = Vector
    TargetVectorArray,InferenceResultArray,WeightArray = RidgeCrossCheck(Vectors,Matrix,RidgeMax,RidgeMin,RidgeDense)
    if Task == "XOR":
        MaxCapacityIndex,CorArrayMatrix,AccuracyArrayMatrix = CapacityCalculation(TargetVectorArray,InferenceResultArray,ThresholdFlag,Fill0inTdelay0=True)
    else:
        MaxCapacityIndex,CorArrayMatrix,AccuracyArrayMatrix = CapacityCalculation(TargetVectorArray,InferenceResultArray,ThresholdFlag,Fill0inTdelay0=False)
    # Ridge 参数优化
    RidgeScale = np.geomspace(RidgeMax, RidgeMin, RidgeDense)
    OptimizedRidge = RidgeScale[MaxCapacityIndex]
    print(f"Optimized Ridge Parameter: {OptimizedRidge}")

    # 保存每个 Tdelay 的权重
    for Tdelay in range(TdelayMax):
        OptimizedWeights = WeightArray[Tdelay, MaxCapacityIndex]  # 提取优化的 Ridge 对应的权重
        WeightDf = pd.DataFrame(OptimizedWeights).T
        while WeightDf.shape[1] < 7:
          WeightDf[WeightDf.shape[1]] = None
        WeightDf[6] = WeightDf.iloc[:, :5].mean(axis=1)
        WeightFileName = f"/Volumes/Extreme SSD/20250410_3000_0/Vgswing1000mV_Vd300mV/AnalysisResult/OptimizedWeights_{DirectoryName}_{Task}_Tdelay{Tdelay}_Ridge{OptimizedRidge:.2e}.csv"
        WeightDf.to_csv(WeightFileName, index=False, header=False)
        print(f"Saved optimized weights for Tdelay={Tdelay} to {WeightFileName}")
    #CorArrayMatrix=(Ridgedense,TdelayMax,CrossCheckParameter)
    RidgeScale = np.geomspace(RidgeMax, RidgeMin, RidgeDense)
    print(RidgeScale[MaxCapacityIndex])
    ResultDf = pd.DataFrame(CorArrayMatrix[MaxCapacityIndex],columns=["Test0","Test1","Test2","Test3","Test4"],index=["Tdelay0","Tdelay1","Tdelay2","Tdelay3","Tdelay4","Tdelay5","Tdelay6","Tdelay7","Tdelay8","Tdelay9"])
    AccuracyDf = pd.DataFrame(AccuracyArrayMatrix[MaxCapacityIndex],columns=["Test0","Test1","Test2","Test3","Test4"],index=["Tdelay0","Tdelay1","Tdelay2","Tdelay3","Tdelay4","Tdelay5","Tdelay6","Tdelay7","Tdelay8","Tdelay9"])
    now = datetime.datetime.now()
    Time=now.strftime('%Y%m%d_%H%M%S')
    Targetdelay = 3
    PCAdf = PatternCheckAccuracy(Targetdelay,RawVector,TargetVectorArray,InferenceResultArray,np.arange(5000),MaxCapacityIndex)
    CCdf = ClassificationCheck(Targetdelay,RawVector,TargetVectorArray,InferenceResultArray,np.arange(5000),MaxCapacityIndex)

    MainNamePrefix = "/Volumes/Extreme SSD/20250410_3000_0/Vgswing1000mV_Vd300mV/AnalysisResult/RateRidge_" + DirectoryName + "_" + Task + "_Ch" + str(CN) + "_VN" + str(VirtualNode)
    if AvgFlag:
        MainNamePrefix = MainNamePrefix + "_avg"
    else:
        MainNamePrefix = MainNamePrefix + "_sub"
    if ThresholdFlag:
        MainName = MainNamePrefix + "_wTh_Td10_" + Time
        AccuracyName = MainName + "_Accuracy.csv"
        AccuracyDf.to_csv(AccuracyName)
    else:
        MainName = MainNamePrefix + "_woTh_Td10_" + Time
    print(MainName)
    np.save(MainName,CorArrayMatrix[MaxCapacityIndex])
    OutName = MainName + ".csv"
    ResultDf.to_csv(OutName)
    PCAName = MainName + "_PCA.csv"
    PCAdf.to_csv(PCAName)
    CCName = MainName + "_CC.csv"
    CCdf.to_csv(CCName)
    
Condition = "1e5_pulses"
for VN in [200]:
        CurrentNo = 202301
        SubAvg = False
        STMandPC(Condition,"STM",CurrentNo,VirtualNode=VN,Pulling=int(200/VN),ThresholdFlag=False,AvgFlag=SubAvg)
        STMandPC(Condition,"PC",CurrentNo,VirtualNode=VN,Pulling=int(200/VN),ThresholdFlag=False,AvgFlag=SubAvg)
        STMandPC(Condition,"XOR",CurrentNo,VirtualNode=VN,Pulling=int(200/VN),ThresholdFlag=False,AvgFlag=SubAvg)
        SubAvg = True
        STMandPC(Condition,"STM",CurrentNo,VirtualNode=VN,Pulling=int(200/VN),ThresholdFlag=False,AvgFlag=SubAvg)
        STMandPC(Condition,"PC",CurrentNo,VirtualNode=VN,Pulling=int(200/VN),ThresholdFlag=False,AvgFlag=SubAvg)
        STMandPC(Condition,"XOR",CurrentNo,VirtualNode=VN,Pulling=int(200/VN),ThresholdFlag=False,AvgFlag=SubAvg)

