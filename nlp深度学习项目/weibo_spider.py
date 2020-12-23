#!/usr/bin/env python
# coding: utf-8

# In[7]:


from selenium import webdriver
import time
from tqdm import tqdm


# In[49]:


try:
    print(u'登陆新浪微博手机端...')
    browser = webdriver.Chrome()
    ##给定登陆的网址
    url = 'https://passport.weibo.cn/signin/login'
    browser.get(url)
    time.sleep(3)
    #找到输入用户名的地方
    username = browser.find_element_by_css_selector('#loginName')
    time.sleep(2)
    username.clear()
    username.send_keys('180****6088')#输入自己的账号
    #找到输入密码的地方
    password = browser.find_element_by_css_selector('#loginPassword')
    time.sleep(2)
    password.send_keys('fjj******.com')
    #点击登录
    browser.find_element_by_css_selector('#loginAction').click()
    time.sleep(2)
except:
    print('########出现Error########')
finally:
    print('完成登陆!')


# In[64]:


#获取微博医疗话题下的内容
def get_page(i):
    url = 'https://s.weibo.com/weibo?q=%23%E5%8C%BB%E7%96%97%23&Refer=index&page='+str(i) 
    browser.get(url)
    time.sleep(3)


# In[72]:


#将微博折叠隐藏的内容展开
def show_all():
    num = len(browser.find_elements_by_partial_link_text('展开全文'))   
    for i in range(num):
        browser.find_element_by_partial_link_text('展开全文').click()
        time.sleep(1)


# In[71]:


#提取微博文本内容
text = []
def get_info():
    info = browser.find_elements_by_xpath("//p[@class='txt']")
    for value in info: 
        text.append(value.text)


# In[82]:


for i in tqdm(range(1,50)):
    get_page(i)
    show_all()
    get_info()


# In[92]:


len(text)


# In[85]:


text


# In[91]:


with open("text.txt","w") as f:
    for i in range(len(text)):
        f.write(text[i])


# In[90]:


text[6]


# In[93]:


import pandas as pd


# In[99]:


pd.DataFrame(text).to_csv('text.csv')


# In[100]:


pd.read_csv('text.csv')

