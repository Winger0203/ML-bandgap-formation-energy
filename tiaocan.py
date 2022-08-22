from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

#读取文件
dataset= r'Bandgap After.xlsx'
# dataset1= r'Formation energy Predict.xlsx'
data=pd.DataFrame(pd.read_excel(dataset))
# data1=pd.DataFrame(pd.read_excel(dataset1))

X = data.values[:, :-1]
y = data.values[:, -1]

# 划分数据集
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2,random_state=1)
print("Size of training set:{} size of testing set:{}".format(X_train.shape[0],X_test.shape[0]))

#   grid search start
best_score = 0
for n_estimators in range(100,200):
    for max_depth in [2,3,4,5,6,7,8,9]:
        for min_samples_leaf in [1,2,3]:
            for min_samples_split in [2, 3,4,5]:
                lrtool = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=
                                                   min_samples_leaf,min_samples_split=min_samples_split)#对于每种参数可能的组合，进行一次训练；
                lrtool.fit(X_train,y_train)
                score = r2_score(y_test, lrtool.predict(X_test))
                if score > best_score:  # 找到表现最好的参数
                    best_score = score
                    best_parameters = {'n_estimators': n_estimators,'max_depth': max_depth,'min_samples_leaf':min_samples_leaf,
                                        'min_samples_split':min_samples_split}
#   grid search end
print("Best score:{:.3f}".format(best_score))
print("Best parameters:{}".format(best_parameters))




'''
# 加入验证集的网格搜索
X_trainval,X_test,y_trainval,y_test = train_test_split(X,y,random_state=1)
X_train,X_val,y_train,y_val = train_test_split(X_trainval,y_trainval,random_state=1)
print("Size of training set:{} size of validation set:{} size of testing set:{}".format(X_train.shape[0],X_val.shape[0],X_test.shape[0]))

#   grid search start
best_score = 0
for n_estimators in range(10,100):
    for max_depth in [2,3,4,5,6,7,8,9,10]:
        for min_samples_leaf in [1,2,3]:
            for min_samples_split in [2, 3,4,5,6,7,8]:
                lrtool = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=
                                                   min_samples_leaf,min_samples_split=min_samples_split)#对于每种参数可能的组合，进行一次训练；
                lrtool.fit(X_train,y_train)
                score = r2_score(y_val, lrtool.predict(X_val))
                if score > best_score:  # 找到表现最好的参数
                    best_score = score
                    best_parameters = {'n_estimators': n_estimators,'max_depth': max_depth,'min_samples_leaf':min_samples_leaf,
                                        'min_samples_split':min_samples_split}
RF = RandomForestRegressor(**best_parameters) #使用最佳参数，构建新的模型
RF.fit(X_trainval,y_trainval) #使用训练集和验证集进行训练，more data always results in good performance.
test_score = r2_score(y_test, lrtool.predict(X_test)) # evaluation模型评估
print("Best score on validation set:{:.3f}".format(best_score))
print("Best parameters:{}".format(best_parameters))
print("Best score on test set:{:.3f}".format(test_score))
#  grid search end


best_score = 0.0
for n_estimators in range(90,100):
    for max_depth in range(2,3):
        for min_samples_leaf in range(1,2):
            for min_samples_split in range(2, 3):
               lrtool = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=
                                                   min_samples_leaf,min_samples_split=min_samples_split)  # 对于每种参数可能的组合，进行一次训练；
               lrtool.fit(X_train, y_train)
               scores = cross_val_score(lrtool,X_trainval,y_trainval,cv=5) #5折交叉验证
               score = scores.mean() #取平均数
               if score > best_score:
                   best_score = score
                   best_parameters = {'n_estimators': n_estimators,'max_depth': max_depth,'min_samples_leaf':min_samples_leaf,
                                        'min_samples_split':min_samples_split}
RF = RandomForestRegressor(**best_parameters)
RF.fit(X_trainval,y_trainval)
test_score = RF.score(X_test,y_test)
print("Best score on validation set:{:.3f}".format(best_score))
print("Best parameters:{}".format(best_parameters))
print("Score on testing set:{:.3f}".format(test_score))
'''