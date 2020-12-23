#!/usr/bin/env python
# coding: utf-8

# # 医疗LDA模型

# In[7]:


import pandas as pd
import numpy as np
import re
import nltk
import os


# ## 整体语料的主题词提取

# In[8]:


theme = pd.read_csv ("E:/lv python/text mining/文本挖掘大作业/data&code/data_weibo.csv",sep=',',header=None)
theme


# ### 数据预处理
# 删除缺失值

# In[9]:


theme=theme.dropna()  ###删除缺失值
theme.shape


# 只保留中文

# In[10]:


###特殊符号去除
import re
import warnings

warnings.filterwarnings("ignore")

r1 = "[^\u4e00-\u9fa5]"

theme['clean_text'] = ''
text_len = theme.shape[0]
for i in range(text_len):
    a = re.sub(r1,'',theme.iloc[i,1])
    theme.iloc[i,2] = a


# In[11]:


theme


# 获取停用词

# In[16]:


stopwords1= [line.rstrip() for line in open('./data/中文停用词表.txt', 'r', encoding='utf-8')]
stopwords2= [line.rstrip() for line in open('./data/哈工大停用词表.txt', 'r', encoding='utf-8')]
stopwords3 = [line.rstrip() for line in open('./data/停用词库.txt', 'r', encoding='utf-8')]
stopwords = stopwords1 + stopwords2 + stopwords3


# In[17]:


stopwords = stopwords+['网页','链接','微博','视频','收起','全文']


# 定义分词和停用词去除的函数

# In[18]:


import jieba
from bs4 import BeautifulSoup
ch_stopwords = set(stopwords)

def clean_Ctext(text):
    text = BeautifulSoup(text,'html.parser').get_text()
    words = text.split()
    words = [w for w in words if w not in ch_stopwords]     #如果有需要中文停用词，要自己去下
    return ' '.join(words)

def split_sentences(review):
    raw_sentences = jieba.cut(review,cut_all = True)
    sentences = [clean_Ctext(s) for s in raw_sentences if s]
    return ' '.join(sentences)


theme['clean_text'] = theme.clean_text.apply(split_sentences)


# 处理好的文本

# In[19]:


theme


# 使用 TfidVectorizer 进行 TF-idf 词袋模型的构建

# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer

Tf = TfidfVectorizer(use_idf=True)
Tf.fit(theme.clean_text)
vocs = Tf.get_feature_names()
corpus_array = Tf.transform(theme.clean_text).toarray()
corpus_norm_df = pd.DataFrame(corpus_array, columns=vocs)
corpus_norm_df


# ### 构建LDA主题模型

# In[21]:


from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=2,learning_method = 'batch', max_iter=1000, random_state=1,verbose=0)

docres = LDA.fit_transform(corpus_array)


# 类别所属概率

# In[22]:


LDA_corpus = np.array(docres)

LDA_corpus


# In[23]:


LDA_corpus.shape


# 每篇文章中对每个特征词的所属概率矩阵：list长度等于分类数量

# In[24]:


print('主题词所属矩阵：\n', LDA.components_)
print(LDA.components_.shape)


# 确定所属类别

# In[25]:


LDA_corpus_one = np.zeros([LDA_corpus.shape[0]])

LDA_corpus_one[LDA_corpus[:, 0] < LDA_corpus[:, 1]] = 1

print('所属类别：', LDA_corpus_one)


# ### 从LDA的主题分类结果中挑出关于医疗的主题，再进行主题词的提取

# In[26]:


LDA_corpus_one = np.zeros([LDA_corpus.shape[0]])
corpus_norm_df['LDA_labels'] = LDA_corpus.argmax(axis=1)
corpus_norm_df.head()


# 每个单词的主题权重值

# In[27]:


tt_matrix = LDA.components_

i = 0
for tt_m in tt_matrix:
    tt_dict_1= [(name, tt) for name, tt in zip(vocs, tt_m)]
    tt_dict_1 = sorted(tt_dict_1, key=lambda x: x[1], reverse=True)
    
    # 打印权重值大于0.6的主题词
    tt_dict_1 = [tt_threshold for tt_threshold in tt_dict_1 if tt_threshold[1] > 0.6]
    
    if(i==1):
        print(tt_dict_1)
    i = i+1


# 所有主题的每个单词的主题权重值

# In[28]:


tt_matrix = LDA.components_
id = 0
for tt_m in tt_matrix:
    tt_dict = [(name, tt) for name, tt in zip(vocs, tt_m)]
    tt_dict = sorted(tt_dict, key=lambda x: x[1], reverse=True)
    
    # 打印权重值大于0.6的主题词：
    # tt_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0.6]
    
    # 打印每个类别前20个主题词：
    tt_dict = tt_dict[:20]
    print('主题%d:' % (id), tt_dict,'\n')
    id += 1


# ## 关于医疗的主题词提取

# In[29]:


med_topic_array = corpus_array[corpus_norm_df.LDA_labels==1,]


# In[30]:


LDA_med = LatentDirichletAllocation(n_components=2, max_iter=1000, random_state=1)
LDA_med_corpus = np.array(LDA_med.fit_transform(med_topic_array))
LDA_med_corpus


# In[21]:


Tf.fit(theme.clean_text)
vocs = Tf.get_feature_names()


# 每个单词的主题权重值

# In[31]:


tt_med_matrix = LDA_med.components_
i = 0
for tt_m in tt_med_matrix:
    tt_dict = [(name, tt) for name, tt in zip(vocs, tt_m)]
    tt_dict = sorted(tt_dict, key=lambda x: x[1], reverse=True)
    #打印权重值大于0.6的主题词
    tt_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0.6]
    if(i==1):
        print(tt_dict)
    i = i+1


# 所有主题的每个单词的主题权重值

# In[32]:


tt_med_matrix = LDA_med.components_
id = 0
for tt_m in tt_med_matrix:
    tt_dict = [(name, tt) for name, tt in zip(vocs, tt_m)]
    tt_dict = sorted(tt_dict, key=lambda x: x[1], reverse=True)
    
    # 打印每个类别前20个主题词：
    tt_dict_1 = tt_dict[:20]
    print('主题%d:' % (id), tt_dict_1,'\n')
    id += 1


# ## 制作词云

# In[34]:


word_dict = dict(tt_dict[:100])


# In[35]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

wordcloud = WordCloud(font_path = 'Part I Text Mining Basic/simdata/simhei.ttf',max_font_size=200, 
                      background_color="white",relative_scaling=0.3,scale=5,width=1000, height=600).fit_words(word_dict)
plt.figure(figsize=(20,15),dpi=200)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:




