#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data=pd.read_csv('E:/研究生相关/统计学习1课件/数据分析项目与小论文/数据分析项目/train_copy.csv',index_col=0)


# In[5]:


data.shape


# In[6]:


data.info()


# # earliest_cr_line 提取月份

# In[7]:


month=data['earliest_cr_line'].str.split('-',expand=True)
month.columns=['month','year']
data['earliest_cr_line']=month['month']


# In[8]:


data.drop(['zip_code'],axis=1,inplace=True)


# # 分类变量缺失值填充

# In[27]:


# replace函数可以用来将变量值取代,将定性变量转化为定量变量
mapping_dict={'emp_length':{'10+ years':10,'9 years':9,'8 years':8,'7 years':7,'6 years':6,'5 years':5,
                           '4 years':4,'3 years':3,'2 years':2,'1 years':1,'< 1 years':1},
              'grade':{'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}}
data.replace(mapping_dict)


# In[9]:


data.emp_length.describe()
data['emp_length']=data['emp_length'].fillna('10+ years')
data['emp_length'].isnull().sum() ###计算缺失值数量


# In[10]:


data.earliest_cr_line.value_counts()
data['earliest_cr_line']=data['earliest_cr_line'].fillna('Aug')
data['earliest_cr_line'].isnull().sum()


# 计算缺失值比例

# In[11]:


categorical_col=data.select_dtypes(include=np.object).columns  ##根据列的类型选取特定行和列
data[categorical_col].isnull().sum().sort_values(ascending=False)


# In[12]:


data.drop('emp_title',axis=1,inplace=True)


# In[13]:


data['title'].value_counts()
data['title'].fillna('Debt consolidation',inplace=True)


# # 数值型变量缺失值填充

# In[42]:


number_col=data.select_dtypes(include=np.number).columns
(data[number_col].isnull().sum()/1130298).sort_values(ascending=False)


# 定量变量用中位数填补

# In[14]:


from sklearn.impute import SimpleImputer  ### 官方文档https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
imp_median=SimpleImputer(missing_values=np.nan,strategy='median')
number_col=data.select_dtypes(include=np.number).columns 
imp_median.fit_transform(data[number_col])          ### 必须是2D-array,如果选取dataframe的话要选取至少两行,fit_transform就是先fit然后transform
data[number_col]=imp_median.transform(data[number_col])
for col in number_col:
    data[col]=round(data[col],2)


# # 特征衍生

# 在数值型变量中,'installment'代表贷款每月分期的金额,我们将annual_inc除以12得到贷款申请人的月收入金额,然后再把'installment'(月负债)与('annual_inc'/12)(月收入)相除生成新的特征'installment_rate',新特征'installment_rate'代表客户每月还款支出与收入的比,'installment_rate'的值越大,说明贷款人的还款压力越大,违约的可能性越大。

# In[15]:


sns.set()                    ####若是在1个单元格中画图，就是画在一张表上,也可以创建多张表,作多张图fig,axes=plt.subplots(1,2)
index1=data['y']==1
index2=data['y']==0
sns.kdeplot(data=data['installment'][index1])
sns.kdeplot(data=data['installment'][index2])
plt.title('Relationship between Installment_rate and Default')
plt.show()


# 但从图形上来看,其实新特征'installment_rate',在违约客户和非违约客户之间并没有明显的联系

# # 特征编码

# 数值进行标准化,经过处理的数据符合标准正态分布,均值为0,标准差为1

# In[16]:


tmp=data.drop(['y'],axis=1)
number_col=tmp.select_dtypes(include=np.number).columns
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data[number_col]=sc.fit_transform(data[number_col])
data.head()


# 对分类变量进行独热编码,使其能够在模型中运用

# In[25]:


object_col=data.select_dtypes(include=np.object).columns  ###自变量的因子数较多会导致独热编码时内存不够,尽量删减自变量的因子数
dummy=pd.get_dummies(data[object_col])
data=pd.concat([dummy,data],axis=1)


# In[18]:


object_col


# In[23]:


data.drop('title',axis=1,inplace=True)


# In[ ]:




