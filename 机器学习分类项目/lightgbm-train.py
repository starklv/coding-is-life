#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import r2_score,confusion_matrix,f1_score,cohen_kappa_score,precision_score,recall_score
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import time 


# In[2]:


get_ipython().run_line_magic('pinfo', 'LGBMClassifier')


# In[47]:


##或者使用 alldata=pd.read_csv('E:/研究生相关/统计学习1课件/数据分析项目与小论文/数据分析项目/trainset.csv',index_col=0) 
##将第一列变为index的值
alldata=pd.read_csv('E:/研究生相关/统计学习1课件/数据分析项目与小论文/数据分析项目/trainset.csv')
alldata=alldata.drop(alldata.columns[0],axis=1)
test=pd.read_csv('E:/研究生相关/统计学习1课件/数据分析项目与小论文/数据分析项目/testxset.csv')
test=test.drop('Unnamed: 0',axis=1)


# 碰到字符串变量中含有nan(缺失值),用LabelEncoder函数将字符串转化为数字时,就会出现报错现象,将这些变量转化为np.string_类型

# In[48]:


#lightgbm不接受字符串，故先将字符串转化为数字，通过LabelEncoder函数实现
#找出类别变量
alldata['emp_length']=alldata['emp_length'].astype(np.string_)
categorical_features=np.where(alldata.dtypes==object)[0]
class_le=LabelEncoder()
#用class_le先fit，之后再实行transform
for i in categorical_features:
    class_le.fit(alldata.iloc[:,i].values)
    alldata.iloc[:,i]=class_le.transform(alldata.iloc[:,i].values)
alldata.head()


# In[49]:


x_train=alldata.iloc[:,1:]
y_train=alldata.iloc[:,0]


# In[59]:


start = time.time()
estimator=LGBMClassifier(objective='binary',colsample_bytree=0.8,subsample=0.8,
                         eval_metric='auc',learning_rate=0.3,n_estimators=25)
param_grid={'max_depth':range(6,18,3),
           'num_leaves':range(1000,10000,2000)}
gs=GridSearchCV(estimator,param_grid,cv=3)
print(gs.fit(x_train.head(100000),y_train.head(100000)))
print('{:.2f}'.format(time.time()-start)+' sec')


# In[10]:


#网格化搜索max_depth,num_leaves，且num_leaves<2**max_depth
estimator=LGBMClassifier(objective='binary',colsample_bytree=0.8,subsample=0.8,
                         eval_metric='auc',learning_rate=0.3,n_estimators=25)
param_grid={'max_depth':range(6,18,3),
           'num_leaves':range(1000,10000,2000)}
gs=GridSearchCV(estimator,param_grid,cv=3)
gs.fit(x_train,y_train)


# In[11]:


gs.best_score_


# In[12]:


gs.best_params_


# In[13]:


param_grid={'max_depth':range(3,7,1),
           'num_leaves':range(50,220,30)}
gs=GridSearchCV(estimator,param_grid,cv=3)
gs.fit(x_train,y_train)
#由于num_leaves为下界，再进行调优


# In[14]:


gs.best_score_


# In[15]:


gs.best_params_


# In[13]:


estimator=LGBMClassifier(binary='objective',eval_metric='auc',colsample_bytree=0.8,
                        subsample=0.8,learning_rate=0.3,n_estimators=25,max_depth=4)
param_grid={'num_leaves':range(6,10,1)}
gs=GridSearchCV(estimator,param_grid,cv=5)
gs.fit(x_train,y_train)
#最后优化时缩小间距，逐渐中心化调优


# In[14]:


gs.best_score_


# In[15]:


gs.best_params_


# In[18]:


# 最优 max_depth=4,num_leaves=8
#网格化搜索 learning_rate 和 n_estimators
estimator=LGBMClassifier(objective='binary',eval_metric='auc',subsample=0.8,
                        colsample_bytree=0.8,max_depth=4,num_leaves=8)
param_grid={'learning_rate':np.arange(0.05,0.35,0.05),
           'n_estimators':[20,30,40]}
gs=GridSearchCV(estimator,param_grid,cv=5)
gs.fit(x_train,y_train)


# In[19]:


gs.best_score_


# In[20]:


gs.best_params_


# In[22]:


param_grid={'learning_rate':np.arange(0.3,0.7,0.1),
           'n_estimators':range(40,80,10)}
gs=GridSearchCV(estimator,param_grid,cv=5)
gs.fit(x_train,y_train)


# In[24]:


gs.best_score_


# In[26]:


gs.best_params_


# In[12]:


#最优 learning_rate=0.3,n_estimators=70
#网格化搜索 min_child_samples和max_bin
from time import *
begin_time=time()
estimator=LGBMClassifier(objective='binary',colsample_bytree=0.8,subsample=0.8,
                        max_depth=4,num_leaves=8,learning_rate=0.3,n_estimators=70,
                        save_binary='true',max_bin=240)
param_grid={'min_child_samples':range(18,23,1),
           'max_bin':range(230,270,10)}
gs=GridSearchCV(estimator,param_grid,cv=5)
gs.fit(x_train,y_train)
end_time=time()
run_time=end_time-begin_time


# In[18]:


run_time/60


# In[15]:


gs.best_score_


# In[19]:


param_grid={
    'min_child_samples':range(14,19,2),
    'max_bin':range(253,257,1)
}
gs=GridSearchCV(estimator,param_grid,cv=5)
gs.fit(x_train,y_train)
gs.best_params_


# In[20]:


gs.best_score_


# In[29]:


#最优 max_bin=255,min_child_samples=16
#网格化搜索 colsample_bytree和subsample
estimator=LGBMClassifier(objective='binary',max_depth=4,num_leaves=8,learning_rate=0.3,
                        n_estimators=70,min_child_samples=16,max_bin=255,
                         save_binary='true')
param_grid={'colsample_bytree':np.arange(0.7,1.0,0.1),
           'subsample':np.arange(0.7,1.0,0.1)}
gs=GridSearchCV(estimator,param_grid,cv=5)
gs.fit(x_train,y_train)


# In[30]:


gs.best_score_


# In[31]:


gs.best_params_


# In[40]:


#subsample取在边界，继续搜索
estimator=LGBMClassifier(objective='binary',max_depth=4,num_leaves=8,learning_rate=0.3,
                        n_estimators=70,min_child_samples=16,max_bin=255,
                         save_binary='true',colsample_bytree=0.8)
param_grid={'subsample':np.arange(0.6,0.8,0.05)}
gs=GridSearchCV(estimator,param_grid,cv=5)
gs.fit(x_train,y_train)


# In[41]:


gs.best_score_


# In[42]:


gs.best_params_


# In[127]:


#最优参数如下
gbm=LGBMClassifier(objective='binary',max_depth=4,num_leaves=8,learning_rate=0.3,
                  n_estimators=70,min_child_samples=16,max_bin=255,
                  colsample_bytree=0.8,subsample=0.7)


# In[37]:


#第一个抽样
x_train1,x_test1,y_train1,y_test1=train_test_split(x_train,y_train,test_size=1/3,random_state=1)


# In[134]:


gbm1=LGBMClassifier(objective='binary',max_depth=4,num_leaves=8,
                   learning_rate=0.3,n_estimators=70,min_child_samples=16,max_bin=255,
                   colsample_bytree=0.8,subsample=0.7)
gbm1.fit(x_train1,y_train1,eval_metric={'binary_logloss','auc'},early_stopping_rounds=5,eval_set=[(x_train,y_train)])


# In[135]:


pred_test1=gbm1.predict(x_test1)
kappa1=cohen_kappa_score(pred_test1,y_test1)
f1=f1_score(y_true=y_test1,y_pred=pred_test1)
print('kappa1:',kappa1,'F-measure1:',f1)


# In[56]:


#第二个抽样
x_train2,x_test2,y_train2,y_test2=train_test_split(x_train,y_train,test_size=1/3,random_state=2)


# In[136]:


gbm1.fit(x_train2,y_train2,eval_metric={'binary_logloss','auc'},early_stopping_rounds=5,eval_set=[(x_train,y_train)])


# In[137]:


pred_test2=gbm1.predict(x_test2)
kappa2=cohen_kappa_score(pred_test2,y_test2)
f2=f1_score(y_true=y_test2,y_pred=pred_test2)
print('kappa2:',kappa2,'F-measure2:',f2)


# In[93]:


#第三个抽样
x_train3,x_test3,y_train3,y_test3=train_test_split(x_train,y_train,test_size=1/3,random_state=3)


# In[143]:


gbm1.fit(x_train3,y_train3,eval_metric='auc',early_stopping_rounds=5,eval_set=[(x_train,y_train)])


# In[144]:


pred_test3=gbm1.predict(x_test3)
kappa3=cohen_kappa_score(pred_test3,y_test3)
f3=f1_score(y_true=y_test3,y_pred=pred_test3)
print('kappa3:',kappa3,'F-meaure3:',f3)


# In[141]:


mean_kappa=(kappa1+kappa2+kappa3)/3
mean_Fmeasure=(f1+f2+f3)/3


# In[129]:


#最终预测,categorical_features最后的那个[0]不要忘记,本身对象是个列表
categorical_features_test=np.where(test.dtypes==np.object)[0]
test['emp_length']=test['emp_length'].astype(np.string_)
class_le=LabelEncoder()
for i in categorical_features_test:
    class_le.fit(test.iloc[:,i].values)
    test.iloc[:,i]=class_le.transform(test.iloc[:,i].values)
test.head()   


# In[145]:


gbm=LGBMClassifier(objective='binary',max_depth=4,num_leaves=8,learning_rate=0.3,
                  n_estimators=70,min_child_samples=16,max_bin=255,
                  colsample_bytree=0.8,subsample=0.7)
gbm.fit(x_train,y_train,eval_metric='auc',early_stopping_rounds=5,eval_set=[(x_train,y_train)])


# In[155]:


#将预测出的test_pred输出
pred_test=gbm.predict(test)
result_lightgbm=pd.DataFrame(pred_test)
result_lightgbm.to_csv('E:/研究生相关/统计学习1课件/数据分析项目与小论文/数据分析项目/result_lightgbm1.csv')


# In[165]:


#画出重要性变量
import matplotlib.pylab as plt
import lightgbm as lgb


# In[167]:


plt.figure(figsize=(12,6))
lgb.plot_importance(gbm,max_num_features=15)
plt.title('Featuretances')
plt.show()

