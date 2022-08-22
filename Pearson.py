import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame

dataset= r'Bandgap.xlsx'
data=pd.DataFrame(pd.read_excel(dataset))
featureData=data.iloc[:,:]
corMat = DataFrame(featureData.corr())  #corr 求相关系数矩阵
print(corMat)
writer = pd.ExcelWriter('output9.xlsx')
corMat.to_excel(writer,'Sheet1')
writer.save()
plt.figure(figsize=(20, 30))
sns.heatmap(corMat, annot=False, vmax=1, square=True, cmap="Blues",linewidths=0)
plot.show()
'''
# dataset= r'ML stable3.xlsx'
# data=pd.DataFrame(pd.read_excel(dataset))
# 
# 
# #画出所有特征关于目标值的相关系数排名
# featureData=data.iloc[:,:]
# corMat = DataFrame(featureData.corr())  #corr 求相关系数矩阵
# print(corMat)
# writer = pd.ExcelWriter('output.xlsx')
# corMat.to_excel(writer,'Sheet1')
# writer.save()

'''