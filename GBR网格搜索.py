from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor

dataset= r'C:\Users\86151\Desktop\Graphene\Train1.xlsx'
data=pd.DataFrame(pd.read_excel(dataset))
featureData=data.values[:,0:-1]
# 目标值
band_gap= data.values[:,-1]
'''
# 划分数据集
X_train,X_test,y_train,y_test=train_test_split(featureData, band_gap, test_size=0.2,random_state=1)
print("Size of training set:{} size of testing set:{}".format(X_train.shape[0],X_test.shape[0]))
#   grid search start
best_score = 0

for learning_rate in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]:
        for n_estimators in range(40,100):
            for min_samples_split in [2,3,4]:
                for min_samples_leaf in [1,2,3]:
                    for max_depth in [2,3,4]:
                        lrtool = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, n_estimators=
                        n_estimators, min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_depth=max_depth)  # 对于每种参数可能的组合，进行一次训练；
                        lrtool.fit(X_train, y_train)
                        score = r2_score(y_test, lrtool.predict(X_test))
                        if score > best_score:  # 找到表现最好的参数
                            best_score = score
                            best_parameters = {'loss':loss, 'learning_rate':learning_rate, 'n_estimators':
                                n_estimators, 'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf,'max_depth':max_depth}
#   grid search end
print("Best score:{:.3f}".format(best_score))
print("Best parameters:{}".format(best_parameters))
'''


# 加入验证集的网格搜索
X_trainval,X_test,y_trainval,y_test = train_test_split(featureData,band_gap,random_state=1)
X_train,X_val,y_train,y_val = train_test_split(X_trainval,y_trainval,random_state=1)
print("Size of training set:{} size of validation set:{} size of testing set:{}".format(X_train.shape[0],X_val.shape[0],X_test.shape[0]))
#   grid search start
best_score = 0
for loss in ['ls', 'lad', 'huber', 'quantile']:
    for learning_rate in [0.1]:
        for n_estimators in range(40,100):
            for min_samples_split in [2,3,4,5]:
                for min_samples_leaf in [1,2,3,4]:
                    for max_depth in [2,3,4]:
                        lrtool = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, n_estimators=
                        n_estimators, min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_depth=max_depth)  # 对于每种参数可能的组合，进行一次训练；
                        lrtool.fit(X_train, y_train)
                        score = r2_score(y_val, lrtool.predict(X_val))
                        if score > best_score:  # 找到表现最好的参数
                            best_score = score
                            best_parameters = {'loss': loss, 'learning_rate': learning_rate, 'n_estimators':
                                n_estimators, 'min_samples_split': min_samples_split,
                                               'min_samples_leaf': min_samples_leaf, 'max_depth': max_depth}
GBR = GradientBoostingRegressor(**best_parameters) #使用最佳参数，构建新的模型
GBR.fit(X_trainval,y_trainval) #使用训练集和验证集进行训练，more data always results in good performance.
test_score = r2_score(y_test, lrtool.predict(X_test)) # evaluation模型评估
print("Best score on validation set:{:.3f}".format(best_score))
print("Best parameters:{}".format(best_parameters))
print("Best score on test set:{:.3f}".format(test_score))
#   grid search end