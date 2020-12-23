#!/usr/bin/env python
# coding: utf-8

# # 1 爬取许嵩歌词数据库

# In[2]:


from selenium import webdriver
import time
import json
import pandas as pd
import numpy as np
from selenium.common.exceptions import NoSuchElementException
import requests
import re


# In[1]:


class Spider_wangyiyun(object):
    
    def __init__(self):
        
        self.start_url = 'https://music.163.com/#/search/m/?s=%E8%AE%B8%E5%B5%A9&type=1'
        self.browser = webdriver.Chrome()
        self.data = []
        self.page = 1
       
    def parse_content(self):
        infos = self.browser.find_elements_by_xpath('//div[@class="srchsongst"]/div') ##要用/div,而不是//div,要用绝对路径而不是相对路径
        for info in infos:
            song_id = info.find_element_by_xpath('./div[2]/div/div/a').get_attribute('href').split('=')[-1]
            song = info.find_element_by_xpath('./div[2]/div/div/a/b').text
            try:
                singer = info.find_element_by_xpath('./div[4]/div[@class="text"]/a').text
            except NoSuchElementException:
                singer = info.find_element_by_xpath('./div[4]/div[@class="text"]').text
            album = info.find_element_by_xpath('div[5]/div/a').get_attribute('title')
            self.data.append([song_id,song,singer,album])
        while self.page < 5: #用while循环,不能用for循环
            self.page+=1
            print(self.page)
            ##翻页按钮
            #通过link文字定位元素定位元素,也可以模糊定位
            pages = self.browser.find_element_by_link_text('下一页')
            self.browser.execute_script('arguments[0].click();',pages)
            self.parse_content()
            
    #将爬好的数据保存到csv文件中     
    def save_content(self):
        colname = ['song_id','song','singer','album']
        df=pd.DataFrame(self.data,columns=colname)
        df.to_csv('music.csv',encoding='utf_8_sig',index=False) ##保存为csv文件,删去索引列
        
    ##利用网易云上的歌词API接口爬取歌词
    def get_info(self,id):
        headers = {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'
}
        res=requests.get('http://music.163.com/api/song/lyric?id={}&lv=1&kv=1&tv=-1'.format(id),headers=headers)
        json_data = json.loads(res.text)
        try:
            lyric = json_data['lrc']['lyric']
            lyric = re.sub('\[.*\]','',lyric)
        except KeyError :
            lyric = ''
        return  str(lyric)
    
    def txt(self):
        data = pd.read_csv('music.csv')
        for i in range(len(data['song_id'])):
            fp = open('XuSong_lyrics.txt'.format(data['song'][i]),'a+',encoding='utf-8')
            fp.write(self.get_info(data['song_id'][i]))
            fp.close()
            
    def run(self):
        self.browser.get(self.start_url)
        self.browser.switch_to.frame('g_iframe')
        self.browser.implicitly_wait(5) #隐式等待,针对所有元素
        self.parse_content()
        self.save_content()
        self.txt()
        
if __name__ == '__main__':        
    apl = Spider_wangyiyun()
    apl.run()              


# # 2  数据来源和数据分析

# In[4]:


import numpy as np
import pandas as pd
df=pd.read_csv('E:/lv python/统计学习2论文/music.csv')


# ## 2.1 可视化分析

# In[7]:


kind,count=np.unique(df['album'],return_counts=True) #用unique函数来计算数据中几个不同的id,以及每个id出现的次数
data={'album':kind,'count':count}
data=pd.DataFrame(data)
data = data.loc[data['count']>1,]
data.head()


# In[51]:


import matplotlib.pyplot as plt
import seaborn as sns
#很关键显示中文
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# In[81]:


#统计各张专辑中的单曲数,可见许嵩是一个高产的网络歌手
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)  
ax.bar(data['album'],data['count'])
ax.set(title='专辑单曲数')
ax.set_xticklabels(data['album'],rotation=30)
plt.savefig('E:/lv python/统计学习2论文/专辑单曲数.jpg',dpi=400,bbox_inches='tight') #必须得放在plt.show之前
plt.show()


# In[ ]:


df.to_csv('E:/lv python/统计学习2论文/music.csv',encoding='utf_8_sig') ##保存为csv文件


# ## 2.2 对歌词进行分词,统计词频,并制作词云

# In[2]:


import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import jieba.analyse
from PIL import Image


# In[1]:


with open('E:/lv python/统计学习2论文/xusong.txt','r',encoding='utf-8') as f:
    sentences=f.read()
    f.close()


# In[5]:


seg_list = jieba.cut(sentences, cut_all=False)
word_dict={}
for word in seg_list:
    if len(word)>1:
        prev_counts=word_dict.get(word,0)
        word_dict[word]=prev_counts+1  


# In[6]:


word=sorted(word_dict.items(),key=lambda item:item[1],reverse=True)
word[:20]


# In[17]:


bg=np.array(Image.open('star.jpg'))
segDict={k:v for k,v in word}
wordcloud=WordCloud(font_path='E:/lv python/text mining/Part I  Text Mining Basic/simdata/simhei.ttf',
                   max_font_size=60,relative_scaling=.5,mask=bg,background_color='white').fit_words(segDict)
plt.figure(figsize=(20,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('E:/lv python/统计学习2论文/歌词词云图.jpg',dpi=400,bbox_inches='tight') #必须得放在plt.show之前
plt.show()
plt.close()


# # 3  构造建模过程

# In[18]:


from mxnet import nd
import random


# In[19]:


#建立字符索引,来方便之后的数据处理
def load_data_xusong_lyrics():
    """Load the XuSong lyric data set (available in the Chinese book)"""
    with open ('E:/lv python/统计学习2论文/xusong.txt',encoding='utf-8') as f:
        corpus_chars = f.read()
        corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
        idx_to_char = list(set(corpus_chars))
        char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
        vocab_size = len(char_to_idx)
        corpus_indices = [char_to_idx[char] for char in corpus_chars]
        return corpus_indices, char_to_idx, idx_to_char, vocab_size


# In[20]:


import d2lzh as d2l
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import time


# ## 3.1 循环网络的从零开始

# In[21]:


#载入许嵩歌词,很关键
(corpus_indices, char_to_idx, idx_to_char, vocab_size)=load_data_xusong_lyrics() 


# ###  初始化模型参数

# In[62]:


num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
print('will use', ctx)

def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    # 附上梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


# ### 定义模型 

# In[ ]:


def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


# ### 相邻采样

# In[ ]:


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


# ### 剪裁梯度

# In[ ]:


def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# ### 定义模型训练函数

# In[ ]:


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()
    
    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n,start=0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)
                # 拼接之后形状为(num_steps * batch_size, vocab_size)
                outputs = nd.concat(*outputs, dim=0)
                # Y的形状是(batch_size, num_steps)，转置后再变成长度为
                # batch * num_steps 的向量，这样跟输出的行一一对应
                y = Y.T.reshape((-1,))
                # 使用交叉熵损失计算平均分类误差
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))
                print(y)


# ## 3.2 循环神经网络的简洁实现

# ### 定义模型

# In[ ]:


ctx = d2l.try_gpu()
model = d2lzh.RNNModel(rnn_layer, vocab_size)
model.initialize(force_reinit=True, ctx=ctx)
predict_rnn_gluon('分开', 10, model, vocab_size, ctx, idx_to_char, char_to_idx)


# In[ ]:


# 本类已保存在d2lzh包中方便以后使用
class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        # 将输入转置成(num_steps, batch_size)后获取one-hot向量表示
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


# ### 训练模型

# In[ ]:


# 本函数已保存在d2lzh包中方便以后使用
def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char,
                      char_to_idx):
    # 使用model的成员函数来初始化隐藏状态
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])


# ### 预测模型

# In[69]:


num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period,pred_len, prefixes)


# ## GRU 门控循环神经网络预测

# In[83]:


num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()


# In[72]:


num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']


# In[74]:


gru_layer = rnn.GRU(num_hiddens)
model = d2l.RNNModel(gru_layer, vocab_size)
d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)


# ## LSTM 长短期记忆预测

# In[ ]:


num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()


# In[84]:


num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']


# In[85]:


lstm_layer = rnn.LSTM(num_hiddens)
model = d2l.RNNModel(lstm_layer, vocab_size)
d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)


# # 4 调参过程-选择GRU模型

# In[ ]:


#主要是对num_steps，batch_size，clipping_rate，num_epochs,lr这5个参数进行调整
#batch_size:每个小批量的样本数
#num_epochs:迭代次数
#clipping_theta:裁剪梯度
#lr:学习率
#num_steps:每个样本所包含的时间步数


# In[132]:


num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()


# In[ ]:


num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']


# In[ ]:


gru_layer = rnn.GRU(num_hiddens)
model = d2l.RNNModel(gru_layer, vocab_size)
d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)


# ## 4.1  先确定循环周期次数,num_epochs

# In[ ]:


#如果随着num_epochs的增加，困惑度perplexity在一定迭代次数内变化很小，就可以停止迭代.


# In[116]:


start=time.time()
for num_epochs in np.arange(200,400,50):
        num_steps, batch_size,clipping_theta,lr= 35, 32, 1e-2,80
        pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
        gru_layer = rnn.GRU(num_hiddens)
        model = d2l.RNNModel(gru_layer, vocab_size)
        print('lr: {0} num_epochs: {1}'.format(lr,num_epochs))
        d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
print('{:.2f} sec'.format(time.time()-start))


# In[112]:


#最优的循环周期数num_epochs为300,随着epochs的增大,困惑度perplexity并没有明显的下降,所以可以停止Epoch


# ## 4.2 调整学习率lr和batch_size批量大小

# In[ ]:


#lr:学习率直接影响模型的收敛状态,过大则导致模型不收敛,过小则导致模型收敛特别慢
#batch_size:每个小批量的样本数,大的batch_size减少训练时间,提高稳定性,导致模型泛化能力下降;
#小的batch_size会增加训练时间,但得到的模型泛化能力较好


# In[121]:


start=time.time()
for lr in np.arange(80,130,10):
     for batch_size in np.logspace(4,8,5,endpoint=True,base=2)::
        num_steps,clipping_theta,num_epochs= 35, 1e-2,300
        pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
        gru_layer = rnn.GRU(num_hiddens)
        model = d2l.RNNModel(gru_layer, vocab_size)
        print('lr: {0} batch_size: {1}'.format(lr,int(batch_size)))
        d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                int(batch_size), pred_period, pred_len, prefixes)
print('{:.2f} sec'.format(time.time()-start))


# In[ ]:


#最优的batch_size为32,lr为100


# ## 4.3 调整剪裁梯度clipping_theta和num_steps时间步数

# In[ ]:


#clipping_theta:裁剪后的梯度的L2范数不超过theta
#num_steps:时间步数


# In[133]:


start=time.time()
for clipping_theta in [0.005,0.01,0.05]:
    for num_steps in np.arange(30,50,5):
        lr,num_epochs,batch_size= 100,300,32
        pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
        gru_layer = rnn.GRU(num_hiddens)
        model = d2l.RNNModel(gru_layer, vocab_size)
        print('clipping_theta: {0} num_steps: {1}'.format(clipping_theta,num_steps))
        d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
print('{:.2f} sec'.format(time.time()-start))


# In[137]:


#缩小间距,逐渐中心化调优clipping_theta
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
start=time.time()
for clipping_theta in np.arange(0.008,0.012,0.001):
    for num_steps in np.arange(35,50,5):
        lr,num_epochs,batch_size= 100,300,32
        pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
        gru_layer = rnn.GRU(num_hiddens)
        model = d2l.RNNModel(gru_layer, vocab_size)
        print('clipping_theta: {0} num_steps: {1}'.format(clipping_theta,num_steps))
        d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
print('{:.2f} sec'.format(time.time()-start))


# In[138]:


#最优的clipping_theta为0.01,继续调优num_steps
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
start=time.time()
for num_steps in np.arange(25,40,5):
        lr,num_epochs,batch_size,clipping_theta= 100,300,32,0.01
        pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
        gru_layer = rnn.GRU(num_hiddens)
        model = d2l.RNNModel(gru_layer, vocab_size)
        print('num_steps: {0}'.format(num_steps))
        d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
print('{:.2f} sec'.format(time.time()-start))


# In[139]:


#最优的num_steps为35
#最优参数如下
num_epochs, num_steps, batch_size, lr, clipping_theta = 300, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']


# # 5 GRU门控循环单元与LSTM长短期记忆之间的区别
# 
# ## 1.对memory的控制
# LSTM : 用output gate 控制, 传输给下一个unit
# GRU : 直接传递给下一个unit, 不做任何控制,GRU更节省时间
# 
# ## 2.input gate 和reset gate 作用位置不同
# LSTM : 计算new memory $\hat{c}^{(t)}$时,不对上一时刻的信息做任何控制,而是用forget gate独立的实现这一点
# GRU ：计算new memory $\hat{h}^{(t)}$时利用reset gate对上一时刻的信息进行控制.
# 
# ## 3.相似
# 最大的相似之处就是,在从$t$到$t-1$的更新时都引入了加法
# 这个加法的好处在于能防止梯度弥散,因此LSTM和GRU都比一般的RNN效果好

# ### 普通RNN模型

# In[39]:


num_epochs, num_steps, batch_size, lr, clipping_theta = 300, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['只是', '回忆']
ctx = d2l.try_gpu()
start=time.time()
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
model = d2lzh.RNNModel(rnn_layer, vocab_size)
model.initialize(force_reinit=True, ctx=ctx)
d2lzh.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period,pred_len, prefixes)
print('{:.2f} sec'.format(time.time()-start))


# ### GRU门控循环单元模型

# In[40]:


start=time.time()
num_epochs, num_steps, batch_size, lr, clipping_theta = 300, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['只是', '回忆']
ctx = d2l.try_gpu()
gru_layer = rnn.GRU(num_hiddens)
model = d2l.RNNModel(gru_layer, vocab_size)
d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
print('{:.2f} sec'.format(time.time()-start))


# ### LSTM 长短期记忆

# In[37]:


num_epochs, num_steps, batch_size, lr, clipping_theta = 300, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['只是', '回忆']
ctx = d2l.try_gpu()
start=time.time()
lstm_layer = rnn.LSTM(num_hiddens)
model = d2l.RNNModel(lstm_layer, vocab_size)
d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
print('{:.2f} sec'.format(time.time()-start))


# ### 由以上模型的困惑度perplexity可得,GRU门控循环单元的perplexity最小,为1.04左右,所以效果最好
