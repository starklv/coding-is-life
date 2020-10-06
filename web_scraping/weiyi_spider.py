#!/usr/bin/env python
# coding: utf-8

# # 爬取医生患者数据

# In[1]:


from selenium import webdriver
import time
import json
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import numpy as np


# ## 获取Cookies寻求免验证登陆

# In[9]:


#填写webdriver的保存目录
driver = webdriver.Chrome()

#记得写完整的url 包括http和https
driver.get('https://www.guahao.com/commentslist/e-bd34313a-da4b-4a25-a1d3-3f15a5e92214000/1-0')
driver.maximize_window()

#程序打开网页后20秒内手动登陆账户
time.sleep(20)

with open('cookies.txt','w') as cookief:
    #将cookies保存为json格式
    cookief.write(json.dumps(driver.get_cookies()))

driver.close()


# In[578]:


start=time.time()
# #复旦大学附属中山医院-呼吸内科高级专家门诊
url='https://www.guahao.com/department/125982448692445000?isStd='
#填写webdriver的保存目录
browser = webdriver.Chrome()
#记得写完整的url 包括http和https
browser.get(url)


#首先清除由于浏览器打开已有的cookies
browser.delete_all_cookies()

with open('cookies.txt','r') as cookief:
    #使用json读取cookies 注意读取的是文件 所以用load而不是loads
    cookieslist = json.load(cookief)
    for cookie in cookieslist:
        browser.add_cookie(cookie)
browser.refresh()
time.sleep(1) ##页面刷新后防止找不到按钮键
comment_button=browser.find_element_by_xpath('//a[@class="more-comment-num"]')
comment_button.click()


# ## 爬虫源程序

# In[573]:


def parse_content():
    tr_list=browser.find_elements_by_xpath('//div[@class="comment-list"]//div[@class="comment-list-box"]')
    for tr in tr_list:
        comment=tr.find_element_by_class_name('detail').text
        attitude=tr.find_elements_by_xpath('.//div[@class="stars"]//span[@class="giS giS-star-0"]') 
        #加.是为了加个前缀路径
        products.append([len(attitude),comment])


# In[140]:


start=time.time()
# #复旦大学附属中山医院-呼吸内科高级专家门诊
url='https://www.guahao.com/department/125982448692445000?isStd='
#填写webdriver的保存目录
browser = webdriver.Chrome()
#记得写完整的url 包括http和https
browser.get(url)


#首先清除由于浏览器打开已有的cookies
browser.delete_all_cookies()

with open('cookies.txt','r') as cookief:
    #使用json读取cookies 注意读取的是文件 所以用load而不是loads
    cookieslist = json.load(cookief)
    for cookie in cookieslist:
        browser.add_cookie(cookie)
browser.refresh()
time.sleep(1) ##页面刷新后防止找不到按钮键
pages=browser.find_element_by_xpath('//a[@class="next J_pageNext_gh"]')
browser.execute_script('arguments[0].click();',pages) 
products=[] ##先爬去第一页内容
parse_content()
while True:
    try:
        nextpagebutton=browser.find_element_by_xpath('//a[@class="next J_pageNext_gh"]')
        nextpagebutton.click()
        time.sleep(2) #防止NoSuchElementException,更新太块,找不到
        parse_content()
#     except ElementClickInterceptedException:
#         nextpagebutton=browser.find_element_by_xpath('//a[@class="next J_pageNext_gh"]')
#         nextpagebutton.click()
#         parse_content()
    except NoSuchElementException: ##设置这个是为了在爬完最后一页时自动退出,
        browser.close()             ##最好设置成能够读取页数,然后运用while语句退出
        break
print('{:.2f}'.format(time.time()-start)+'sec')


# ## 爬虫程序封装在对象Spider中

# In[6]:


class Spider:
    
    
    
    def __init__(self):
        
        self.start_url = 'https://www.guahao.com/department/9aabfcdf-b9f5-4f44-95b4-88441d2f7fca000'
        #首先得装一个google chrome的驱动,为了成功运行webdriver.chrome(),打开chrome浏览器,firfox也有相对应的教程,可网上查找
        #具体chrome驱动安装过程可参考网址:https://blog.csdn.net/weixin_44318830/article/details/103339273
        #驱动下载地址可参考:http://npm.taobao.org/mirrors/chromedriver/85.0.4183.87/,这个网址下载很快,上面网址中推荐的下载地址很慢
        self.browser = webdriver.Chrome()
        self.start=time.time()
        self.data=[]
        self.page=1
        
        
    def parse_content(self):
        tr_list = self.browser.find_elements_by_xpath('//div[@class="comment-list"]//div[@class="comment-list-box"]')
        for tr in tr_list:
            comment=tr.find_element_by_class_name('detail').text
            attitude=tr.find_elements_by_xpath('.//div[@class="stars"]//span[@class="giS giS-star-0"]') 
            #加.是为了加个前缀路径
            self.data.append([len(attitude),comment])
        while self.page<60:#每个网页中都是60页
            self.page+=1
            try:
                time.sleep(2) ##防止NoSuchElementException,更新太块,找不到,如果依旧报错,就
                              ##延长sleep的时间,有些网页sleep(2)就可以了
                pages=self.browser.find_element_by_xpath('//a[@class="next J_pageNext_gh"]')
                self.browser.execute_script('arguments[0].click();',pages) 
                #ElementClickInterceptedException,会报错,元素被掩盖
                time.sleep(2) #防止NoSuchElementException,更新太块,找不到,如果依旧报错,就
                              #延长sleep的时间,有些网页sleep(2)就可以了
                self.parse_content()
            except NoSuchElementException: 
                time.sleep(2) ##防止NoSuchElementException,更新太块,找不到,如果依旧报错,就
                              ##延长sleep的时间,有些网页sleep(2)就可以了
                pages=self.browser.find_element_by_xpath('//a[@class="next J_pageNext_gh"]')
                self.browser.execute_script('arguments[0].click();',pages) 
                #ElementClickInterceptedException,会报错,元素被掩盖
                time.sleep(2) #防止NoSuchElementException,更新太块,找不到,如果依旧报错,就
                              ##延长sleep的时间,有些网页sleep(2)就可以了  
                self.parse_content()   
            

    def get_post_cookies(self):
        self.browser.maximize_window()

        #程序打开网页后20秒内手动登陆账户
        time.sleep(20)
        #登陆账户后会获取账户相关的cookies
        with open('cookies.txt','w') as cookief:
            #将cookies保存为json格式
            cookief.write(json.dumps(self.browser.get_cookies()))
        #首先清除由于浏览器打开已有的cookies
        self.browser.delete_all_cookies()
        with open('cookies.txt','r') as cookief:
            #使用json读取cookies 注意读取的是文件 所以用load而不是loads
            cookieslist = json.load(cookief)
            for cookie in cookieslist:
                self.browser.add_cookie(cookie)
        self.browser.refresh()
        time.sleep(2) ##页面刷新后防止找不到按钮键
        button=self.browser.find_element_by_xpath('//a[@class="more-comment-num"]')
        self.browser.execute_script('arguments[0].click();',button)
        ##替代按钮键的功能
    
    
    def run(self):
        self.browser.get(self.start_url)
        self.get_post_cookies()
        self.parse_content()
        print('{:.2f}'.format(time.time()-self.start) + 'sec')
        self.browser.close()
        return self.data

#加入这段代码是对象直接启动   
if __name__ == '__main__':
     apl = Spider()
     products = apl.run()


# In[ ]:


products


# # 最后数据的存取

# ## 转化成pandas中的DataFrame形式

# In[4]:


colname=['attitude','comment']
df=pd.DataFrame(products,columns=colname)
df.tail()


# ### 观察爬取的内容是否重复现象过多

# In[478]:


import numpy as np
kind,count=np.unique(df['comment'],return_counts=True) #用unique函数来计算数据中几个不同的id,以及每个id出现的次数
print(kind)
print(count)
print(len(kind)) #不同id的数量


# In[5]:


df['hospital']='上海交通大学附属瑞金医院'
df=df.reindex(columns=['attitude','hospital','comment'])
df.head()


# ## 将数据写入到csv文件中

# In[479]:


df.to_csv('E:/lv python/text mining/文本挖掘大作业/data_复旦大学附属中山医院.csv',index=False,
          encoding="utf_8_sig")
df.to_csv('E:/lv python/text mining/文本挖掘大作业/data_上海交通大学附属瑞金医院.csv',index=False,mode='a',header=False,
          encoding="utf_8_sig")
#index=False 不要索引,mode='a'表示追加,header=False 表示不要列名

