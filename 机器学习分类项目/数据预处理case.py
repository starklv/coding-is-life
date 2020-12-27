#!/usr/bin/env python
# coding: utf-8

# In[431]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
os.chdir("../data/")
alldata = pd.read_csv("audit.csv", )


# In[432]:


alldata.drop(alldata.columns[0], axis = 1, inplace = True)
alldata.head()


# # 2.1

# In[433]:


alldata.info()


# In[434]:


alldata.columns


# # 2.2

# In[435]:


alldata.drop(['ID', 'IGNORE_Accounts', 'RISK_Adjustment'], axis = 1, inplace = True)
alldata['TARGET_Adjusted'] = alldata['TARGET_Adjusted'].astype('object')


# # 2.3

# In[436]:


alldata.columns


# In[437]:


from sklearn.model_selection import train_test_split
y = alldata[['TARGET_Adjusted']]
X = alldata[['Age', 'Employment', 'Education', 'Marital', 
             'Occupation', 'Income', 'Gender', 'Deductions', 'Hours']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 1/3, random_state = 1, stratify = y)
X_train.reset_index(drop = True, inplace = True)
y_train.reset_index(drop = True, inplace = True)


# # 2.4

# In[438]:


X_train.describe()


# # 2.5

# In[439]:


def lowvariance(X_train):
    res = []
    for column in X_train.columns:
        n = len(X_train[[column]])
        dic1 = dict(X_train[[column]].value_counts())
        lst1 = sorted(dic1.items(), key = lambda x:x[1], reverse = True)
        if lst1[0][1]/lst1[1][1] > 20 and len(lst1)/n < 0.1:
            res.append(column)
    return res


# In[440]:


lst = lowvariance(X_train)
X_train.drop(lst, axis = 1, inplace = True)


# # 2.8

# In[441]:


d = copy.deepcopy(X_train['Age'])
for i in range(len(d)):
    if d[i] <= 30:
        d[i] = 'youth'
    elif 31<= d[i] <= 50:
        d[i] = 'middleaged'
    else:
        d[i] = 'senior'
X_train['Age'] = d


# # 2.6

# In[442]:


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
numeric_features = np.where(X_train.dtypes != np.object)[0]
X_train.iloc[:, numeric_features] = stdsc.fit_transform(X_train.iloc[:, numeric_features])


# # 2.7

# In[443]:


X_train.isnull().sum()


# In[444]:


###通过观察发现employment和occupation列的缺失值几乎是同时出现的,除了行索引为49的那一行employment为unemployment,occupation为nan
##由于onehot独热编码不容许缺失值，因此先对应填充为UnKnown独热编码之后再调整为nan
null_employment = X_train['Employment'][X_train['Employment'].isnull().values == True].index.tolist()
null_occupation = X_train['Occupation'][X_train['Occupation'].isnull().values == True].index.tolist()
set(null_occupation).difference(null_employment)


# In[475]:


X_train.iloc[null_employment,:]


# In[445]:


from sklearn.impute import SimpleImputer
imputesim = SimpleImputer
X_train.loc[1204, 'Employment'] = 'UnKnown'        


# In[446]:


from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(handle_unknown = 'ignore')
categoriacl_features = np.where(X_train.dtypes == np.object)[0]
X_train_cat_onehot = pd.DataFrame(onehot.fit_transform(X_train.iloc[:, categoriacl_features]).toarray())


# In[448]:


import pandas as pd
X_train_dum = pd.get_dummies(X_train)
X_train_cat_onehot.columns = X_train_dum.columns[2:]
position = X_train_cat_onehot['Employment_UnKnown'] == 1.0
lst = list(position[position.values == 1].index) ###缺失值所在的行数
X_train_cat_onehot.iloc[lst, [9, 47]] = np.nan ###代表Employment_UnKnown和Occupation_UnKnown所在列的索引,重新改为缺失值


# In[449]:


from sklearn.impute import KNNImputer
X_train.reset_index(drop = True, inplace = True)
new_X_train = pd.concat([X_train.iloc[:,numeric_features],X_train_cat_onehot],axis = 1)
imputeknn = KNNImputer(n_neighbors = 10, weights = "distance")
new_X_train  = pd.DataFrame(imputeknn.fit_transform(new_X_train), columns = new_X_train.columns)
new_X_train.loc[lst,]###插补出来的值均为0,因为那一列其它行记录的值为0,所以用独热编码来处理不太妥当


# # 2.9

# In[450]:


import re
pattern = re.compile(r"^Edu|Gen", re.S)
res = []
for x in new_X_train.columns:
    if pattern.findall(x):
        res.append(x)
new_X_train.loc[:, res].corr()


# # 2.10

# In[451]:


import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors = 20, contamination = 'auto')
lofscores = lof.fit_predict(new_X_train.iloc[:,1:3])
lofscores


# In[452]:


X_scores = lof.negative_outlier_factor_ #LOF值的负数。LOF越大，越离群；因此，X_scores越小，越离群。
X_scores


# In[453]:


lof.offset_


# # 2.11

# In[454]:


import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
import random
svr = SVR(kernel = "linear")
random.seed(1)
rfecv = RFECV(estimator = svr,step = 1, cv = 5,
             scoring="neg_mean_squared_error")
y_train.reset_index(drop = True, inplace = True)
rfecv.fit(new_X_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[455]:


rfecv.support_


# In[456]:


rfecv.ranking_


# In[457]:


rfecv.grid_scores_


# In[460]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators = 50)
y_train = y_train.astype("int64")#必须把y_train的类型转换为指定类型才能成功
clf = clf.fit(new_X_train, y_train)
clf.feature_importances_


# In[463]:


selector = SelectFromModel(clf, prefit = True)
X_new = selector.transform(new_X_train)
selector.get_support()


# In[474]:


lst = selector.get_support()
for i in range(len(lst)):
    if  lst[i] == True:
        print(new_X_train.columns[i])


# In[ ]:





# In[ ]:





# In[ ]:




