import matplotlib.pyplot as plot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn import preprocessing
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from gplearn import genetic
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
dataset= r'0107.xlsx'
data=pd.DataFrame(pd.read_excel(dataset))
featureData=data.iloc[:,:]
corMat = DataFrame(featureData.corr())  #corr 求相关系数矩阵
print(corMat)
#writer = pd.ExcelWriter('output.xlsx')
#corMat.to_excel(writer,'Sheet1')
#writer.save()
plt.figure(figsize=(20, 30))
sns.heatmap(corMat, annot=False, vmax=1, square=True, cmap="Blues",linewidths=0)
plot.show()
'''

#读取文件
dataset= r'Bandgap After.xlsx'
# dataset1= r'Formation energy Predict.xlsx'
data=pd.DataFrame(pd.read_excel(dataset))
# data1=pd.DataFrame(pd.read_excel(dataset1))



#画出所有特征关于目标值的相关系数排名
'''
featureData=data.iloc[:,:]
corMat = DataFrame(featureData.corr())  #corr 求相关系数矩阵
print(corMat)
writer = pd.ExcelWriter('output.xlsx')
corMat.to_excel(writer,'Sheet1')
writer.save()
'''

#读取原数据集的特征和目标值
X = data.values[:, :-1]
y = data.values[:, -1]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=1)


for i in range(X_train.shape[1]):
    X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])



for i in range(X_test.shape[1]):
    X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])



'''
px=X[:,0:2]
pkk=X[:,10]
ppo=(X[:,15]+X[:,1])*(X[:,6]/X[:,11])
ppi=(X[:,8]*X[:,19])*(X[:,10]/X[:,4])
ppu=(X[:,21]+X[:,1])/(X[:,12]+X[:,6])
ppq=(X[:,9]+X[:,3])/(X[:,8]+X[:,4])
ppw=(X[:,3]-X[:,16])/(X[:,12]/X[:,13])

px[:,0]=pkk
px[:,1]=y
pxx=pd.DataFrame(px)
print(px[:,:])
'''
'''
px[:,2]=ppi
px[:,3]=ppu
px[:,4]=ppq
px[:,5]=ppw
px[:,6]=y
per=np.corrcoef(px[:,2], y)
print(per[0,1])
pxx=pd.DataFrame(px)
corMat = DataFrame(pxx.corr())  #corr 求相关系数矩阵
print(corMat)
#writer = pd.ExcelWriter('output.xlsx')
#corMat.to_excel(writer,'Sheet1')
#writer.save()
plt.figure(figsize=(20, 30))
sns.heatmap(corMat, annot=False, vmax=1, square=True, cmap="Blues",linewidths=0)
plot.show()
'''
'''
for i in range(0,23):
 for j in range(0,23):
  for k in range(0,23):
    for n in range(0,23):
     px=(X[:,i]-X[:,j])*(X[:,k]-X[:,n])
     per=np.corrcoef(px, y)
     if per[0,1]>0.45 or per[0,1]<-0.45:
      print(per[0,1])
      print(i,j,k,n)
'''

'''                                                                     
#读取自定义的训练集和测试集
X=data.values[:54,:22]
for i in range(X.shape[1]):
    X[:,[i]] = preprocessing.MinMaxScaler().fit_transform(X[:,[i]])
y=data.values[:54,22]
testX=data.values[54:,:22]
for i in range(testX.shape[1]):
    testX[:,[i]] = preprocessing.MinMaxScaler().fit_transform(testX[:,[i]])
testy=data.values[54:,22]
'''


'''lrTool=RandomForestRegressor()
lrTool.fit(X,y)
print(lrTool.score(X, y))
mse = mean_squared_error(y, lrTool.predict(X))
rmse = mse ** (1/2)
sse = np.sum((y - lrTool.predict(X)) ** 2)
sst = np.sum((y - np.mean(X)) ** 2)
R2= 1 - sse / sst
print(R2)
print(rmse)
print(pearsonr(y,lrTool.predict(X)))
# 画图显示
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.rcParams['font.sans-serif'] = 'Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(y,y,label='Real Data')
plt.scatter(y,lrTool.predict(X),label='Predict',c='r')
ax=plt.gca()
ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2)

plt.tick_params(width=2)
ax.xaxis.set_tick_params(labelsize=20)
plt.tick_params(which='major',length=8)
plt.tick_params(which='minor',length=4,width=2)
ax.yaxis.set_tick_params(labelsize=20)
xminorLocator   = MultipleLocator(1000)
yminorLocator   = MultipleLocator(1000)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
plt.show()'''


#选取4种分类算法
# clf = RandomForestRegressor(n_estimators=160,random_state=80, min_samples_split=2,max_features=4,max_depth=7)
# #clf = svm.SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=4.0, epsilon=0.15 , shrinking=True, cache_size=200, verbose=False, max_iter=- 1)
# #clf=ExtraTreesRegressor(max_depth=2, n_estimators=200,random_state=20, min_samples_split=2)
# #kernel = 1.0 * RBF([1.0])
# #clf=GaussianProcessRegressor()
# clf.fit(X_train, y_train)
# y_prediction=clf.predict(X_test)
# #print(py)
# '''
# for i in range(0,10000):
#  clf = RandomForestRegressor(random_state=i,n_estimators=19)
#  clf.fit(X,y)
#  sse = np.sum((ty - clf.predict(tx)) ** 2)
#  sst = np.sum((ty - np.mean(tx)) ** 2)
#  R2 = 1 - sse / sst
#  o=pearsonr(ty,clf.predict(tx))
#  if o[0]>0.85:
# #print(clf.score(X, y))
#   print(o[0],R2,i)
# '''
#
# mse = mean_squared_error(y_test, y_prediction)
# rmse = mse ** (1/2)
# sse = np.sum((y_test - y_prediction) ** 2)
# sst = np.sum((y_test - np.mean(X_test)) ** 2)
# R2= 1 - sse / sst
# print(R2)
# print(rmse)
# print(pearsonr(y_test, y_prediction))
# plt.yticks(fontproperties = 'Times New Roman', size = 14)
# plt.xticks(fontproperties = 'Times New Roman', size = 14)
# plt.rcParams['font.sans-serif'] = 'Roman'
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.plot(y_test, y_test, label='Real Data')
# plt.scatter(y_test, y_prediction, label='Predict', c='r')
# ax=plt.gca()
# ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
# ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
# ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
# ax.spines['top'].set_linewidth(2)
#
# plt.tick_params(width=2)
# ax.xaxis.set_tick_params(labelsize=24)
# plt.tick_params(which='major',length=8)
# plt.tick_params(which='minor',length=4,width=2)
# ax.yaxis.set_tick_params(labelsize=24)
# xminorLocator   = MultipleLocator(1000)
# yminorLocator   = MultipleLocator(1000)
# ax.xaxis.set_minor_locator(xminorLocator)
# ax.yaxis.set_minor_locator(yminorLocator)
# plt.show()

#使用KFold交叉验证

# for nk in range(2,10):
#  kfolder = KFold(n_splits=nk)
#  score=0
#  for train, test in kfolder.split(X_train, y_train):
#    train_data = np.array(data)[train]
#    test_data = np.array(data)[test]
#    trany=train_data[:,14]
#    tranx=train_data[:,:14]
#    testx=test_data[:,:14]
#    testy=test_data[:,14]
#    clf.fit(tranx,trany)
#    pu=pearsonr(testy,clf.predict(testx))
#    score=score+pu[0]
#  print(score/nk)


#画出ROC曲线
'''y_score = clf.fit(X, y).predict_proba(testX)
fpr,tpr,threshold = roc_curve(testy, y_score[:, 1])
roc_auc = auc(fpr,tpr)
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
print(fpr)
print(tpr)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()'''


#画出混淆矩阵
'''clf.fit(X, y)
prey=clf.predict(testX)
true=0
for i in range(0,len(testy)):
 if prey[i]==testy[i]:
     true=true+1
print(true/55)
C = confusion_matrix(testy, prey, labels=[0,1])
plt.imshow(C, cmap=plt.cm.Blues)
indices = range(len(C))
plt.xticks(indices, [0, 1],fontsize=20)
plt.yticks(indices, [0, 1],fontsize=20)
plt.colorbar()
for first_index in range(len(C)):    #第几行
    for second_index in range(len(C)):    #第几列
        plt.text(first_index, second_index, C[first_index][second_index],fontsize=20,horizontalalignment='center')
plt.show()'''


#预测

'''
for i in range(0,300):
 result=0
 clf = RandomForestRegressor(max_depth=3, n_estimators=93,random_state=i, min_samples_split=3)
 clf.fit(X, y)
 py = clf.predict(tx)
 pu=pearsonr(ty,py)
 if pu[0] > 0.84:
  print(i,pu[0])

'''
'''
px=preX[:,0:2]
qq = clf.fit(X, y)
predy = qq.predict(preX)
px[:,0]=predy
print(px[:,:6])
'''

#符号回归

est_gp = genetic.SymbolicTransformer(population_size=100,
                           generations=15, stopping_criteria=0.01,
                           p_crossover=0.8, p_subtree_mutation=0.05,
                           p_hoist_mutation=0.05, p_point_mutation=0.05,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=None,n_components=100)
V=est_gp.fit(X, y)
print(V)
px=V.transform(X)
for i in range(0,1001):
  pear=np.corrcoef(px[:,i], y)
  pea=pear[0,1]
  if abs(pea)>0.5:
   print(pea,V)

'''
dataset='Formation energy Predict.xlsx'
data=pd.DataFrame(pd.read_excel(dataset))
featureData1 = data.values[:,:]
# StandardScaler.fit(featureData1)
# featureData2 = StandardScaler.transform(featureData1)
# print(featureData2)
predict = clf.predict(featureData1)

predict_Ef = pd.DataFrame(predict)
writer = pd.ExcelWriter('Formation energy Predict newst.xlsx')
predict_Ef.to_excel(writer,'Sheet1')
writer.save()
'''