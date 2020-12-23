#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LR
# from sklearn.externals import joblib
import joblib
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,cohen_kappa_score,f1_score
import pandas as pd
import numpy as np
import jieba

import glob#文件名匹配模式
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


# In[3]:


# 训练文件
filepath_p = 'data_health.csv'
data = pd.read_csv(filepath_p)
data.head()


# In[4]:


#删去重复值
data=data.drop_duplicates()


# ## 数据分析和可视化

# In[39]:


import matplotlib.pyplot as plt
import seaborn as sns
#很关键显示中文
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['ytick.labelsize'] = 20 #设置字体大小
plt.rcParams['xtick.labelsize'] = 20 #设置字体大小
plt.rcParams['font.size'] = 15  #设置标题和图例字体大小


# In[6]:


#查看各个医院的评分情况
party_counts = pd.crosstab(data['hospital'],data['attitude'])
party_counts


# In[7]:


#转换成百分比更加清晰
party_pcts = party_counts.div(party_counts.sum(1),axis = 0)
party_pcts


# In[40]:


fig,ax=plt.subplots(1,1,figsize=(10,5)) 
party_pcts.plot.barh(ax=ax)
ax.set(title='各医院评分差异表')
ax.legend(loc = 'best')
plt.savefig('各医院评分差异表.jpg',dpi=400,bbox_inches='tight') #必须得放在plt.show之前
plt.show()


# In[9]:


data['district']='静安区'
data.loc[(data['hospital']=='上海第六人民医院')|(data['hospital']=='复旦大学附属中山医院'),'district']='徐汇区'
data.loc[(data['hospital']=='上海交通大学附属第一人民医院'),'district']='虹口区'
data.loc[(data['hospital']=='上海长征医院'),'district']='黄浦区'


# In[10]:


data=data.reindex(columns=['attitude','hospital','district','comment'])


# In[11]:


#查看各个区域的评分
party_counts1 = pd.crosstab(data['district'],data['attitude'])
party_counts1


# In[12]:


party_pcts1 = party_counts1.div(party_counts1.sum(1),axis = 0)
party_pcts1


# In[41]:


fig,ax=plt.subplots(1,1,figsize=(10,5)) 
party_pcts1.plot.barh(ax=ax)
ax.set(title='各区评分差异表')
ax.legend(loc = 'best')
plt.savefig('各区评分差异表.jpg',dpi=400,bbox_inches='tight') #必须得放在plt.show之前
plt.show()


# In[3]:


# 加载停词表
def stopwordslist(filepath = 'stopwords.txt'):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]#line.strip()删除分词前后的空格
    stopwords.append(' ')
    return stopwords


# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(str(sentence).strip())#jieba
    
    stopwords = stopwordslist()  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        word = word.strip()                             #jieba
        
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


def gen_data(path):
    data = pd.read_csv(path)
    texts = []
    for i, text in enumerate(data['comment']):
        text = seg_sentence(text)
        texts.append(text)

    vector = vectorizer.transform(texts)
    vector = vector.toarray()
    labels = data['attitude']

    return vector, np.array(labels)


# In[4]:


# 词向量
texts = []
for i, text in enumerate(data['comment']):
    text = seg_sentence(text)
    texts.append(text)
    

#创建transform
vectorizer = CountVectorizer()


#分词并建立词汇表
vectorizer.fit(texts)


# In[5]:


#数据加载
data,label = gen_data(filepath_p)


# In[34]:


#模型定义
#clf = LR(penalty="l2", solver='newton-cg', multi_class='multinomial', max_iter=200, n_jobs=-1)
clf = MLPClassifier(solver='sgd', alpha=1e-4,hidden_layer_sizes=(50, 50), random_state=1,
                    max_iter=100,verbose=10,learning_rate_init=.1,validation_fraction=0.3)


# In[9]:


get_ipython().run_line_magic('pinfo', 'MLPClassifier')


# In[35]:


#普通检验
train_X, test_X, train_y, test_y = train_test_split(data,
                                                    label,
                                                    test_size = 0.2,
                                                    random_state = 2020)

#模型训练
clf.fit(data, label)
#clf = joblib.load("clf")

print("开始预测")
pred_y = clf.predict(test_X)
print(pred_y[:100])


# In[39]:


macro_f1 = f1_score(test_y, pred_y,average='macro')
kappa = cohen_kappa_score(test_y,pred_y)
print("kappa:", kappa,";macro-f1:", macro_f1)


# In[ ]:




