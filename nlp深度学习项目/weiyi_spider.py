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

# In[4]:


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


# ## 遍历6家医院7个科室-共42个科室

# In[574]:


#url_list1为复旦大学大学附属中山医院的各个科室的网址
#从上到下,依次为呼吸内科高级专家门诊,心内科高级专家门诊,普外科,心外科,内分泌科,消化科，
# 肝肿瘤外科
url_list1=['https://www.guahao.com/department/125982448692445000',
         'https://www.guahao.com/department/125982446272843000',
          'https://www.guahao.com/department/125809950754835000',
          'https://www.guahao.com/department/125810615177463000',
          'https://www.guahao.com/department/125809911628817000',
          'https://www.guahao.com/department/125810110336729000',
          'https://www.guahao.com/department/125810512426329000']
#url_list2为复旦大学附属华山医院的各个科室的网址
#从上到下,依次为皮肤科,神经内科,消化科,感染发热,泌尿科
#中医内科,呼吸科
url_list2=['https://www.guahao.com/department/125617797246614000',
          'https://www.guahao.com/department/125617811675219000',
          'https://www.guahao.com/department/126708416116682000',
          'https://www.guahao.com/department/126708756338798000',
          'https://www.guahao.com/department/126707963142052000',
          'https://www.guahao.com/department/3aa9c9a6-ddf3-4364-89d9-1a22f2f84510000',
          'https://www.guahao.com/department/126708315613209000']
#url_list3为上海市第六人民医院的各个科室的网址
#从上到下,依次为骨科,内分泌代谢科,心血管内科,呼吸内科,神经内科,
##消化内科,呼吸内科
url_list3=['https://www.guahao.com/department/1686d06f-5018-485e-bdc7-c988725bbe1a000',
          'https://www.guahao.com/department/a9382d5e-fb4a-48f0-bf07-e37cbf4894ae000',
          'https://www.guahao.com/department/0fc9393e-45f5-439c-8d6c-765fbe45d7e4000',
          'https://www.guahao.com/department/4ae38d43-5398-43fd-a8fe-7f238969dd06000',
          'https://www.guahao.com/department/a61f76ef-7d16-4ede-8a16-bfa6d5771fed000',
          'https://www.guahao.com/department/4873bc82-55d5-494e-b154-5d827b2a3b99000',
          'https://www.guahao.com/department/4ae38d43-5398-43fd-a8fe-7f238969dd06000']


# In[575]:


#url_list4为上海长征医院的各个科室的网址
#从上到下,依次为消化内科,神经内科,肾内科,耳鼻咽喉科,泌尿外科
# 肾移植科,脊柱外科(page<39 soecial)
url_list4=['https://www.guahao.com/department/b90e0fdc-7b42-45af-952b-3733457a8886000',
          'https://www.guahao.com/department/cf308ae2-f03c-414d-9a7f-7f9e70030270000',
          'https://www.guahao.com/department/715dc555-9cac-4dab-8429-7ce894731d73000',
          'https://www.guahao.com/department/6a9975cd-0e03-4d85-a3f6-0dc9647da40e000',
          'https://www.guahao.com/department/48e7fc9e-63de-4f47-b54d-0444927ec0a5000',
          'https://www.guahao.com/department/9d4c70fb-0f86-4ce0-aa9d-c90f4186aa34000',
          'https://www.guahao.com/department/b2af1eb0-c662-4d8d-a9a0-21734e427aa2000']

#url_list5为上海交通大学附属第一人民医院的各个科室的网址
#从上到下,依次为心内科,消化科,神经内科,内分泌科,眼科,中医科,
# 口腔科(page<44 special)
url_list5=['https://www.guahao.com/department/55b91013-e4ef-4fb1-87d5-dee2eede7c2e000',
          'https://www.guahao.com/department/9f45fbb1-9054-4a9a-b3bd-d16f8bf13fca000',
          'https://www.guahao.com/department/ba21c39d-85ad-40e2-9b0f-64932a9ea219000',
          'https://www.guahao.com/department/a31e0193-120b-4fd9-8553-25cf481e730a000',
          'https://www.guahao.com/department/eefb1083-778a-4565-93f4-13edd2389942000',
          'https://www.guahao.com/department/02366af8-3f71-4de7-946d-f109e3794e05000',
          'https://www.guahao.com/department/526f8544-fab2-418e-85f0-98b1626e2d77000']

#url_list6为上海师第十人民医院的各个科室的网址
#从上到下,依次为心血管内科门诊,呼吸内科门诊(page<45),泌尿外科门诊,神经内科门诊(page<52)
# 肾内科(page<59, special),内分泌科,骨科
url_list6=['https://www.guahao.com/department/24da18dc-eb4a-4ed6-bcf4-168279cbec18000',
          'https://www.guahao.com/department/cc365591-f4b7-4649-9609-9e04f4c42dd5000',
          'https://www.guahao.com/department/f6dccb40-9ffb-4719-8c1a-79101458054f000',
          'https://www.guahao.com/department/e9bfddf3-76af-41f6-b598-a89e854aabed000',
          'https://www.guahao.com/department/8398269e-1272-4aa8-9cd9-40435c04e637000',
          'https://www.guahao.com/department/08cb68d3-4907-4dcf-b643-fb0d03113d1c000',
          'https://www.guahao.com/department/c53dd910-e3c0-4949-ba7e-17fdb571ab90000']


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

# In[1]:


class Spider:
    
    
    
    def __init__(self):
        
        self.start_url = 'https://www.guahao.com/department/9aabfcdf-b9f5-4f44-95b4-88441d2f7fca000'
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
            

    def post_cookies(self):
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
        self.post_cookies()
        self.parse_content()
        print('{:.2f}'.format(time.time()-self.start)+'sec')
        self.browser.close()
        return self.data

#加入这段代码是对象直接启动   
if __name__ == '__main__':
     apl = Spider()
     products = apl.run()


# In[3]:


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


# ## 将6家医院的数据整合

# In[576]:


dfa=pd.read_csv('data_中山.csv')
dfb=pd.read_csv('data_华山.csv')
dfc=pd.read_csv('data_上海长征医院.csv')
dfd=pd.read_csv('data_上海市第六人民医院.csv')
dfe=pd.read_csv('data_上海交通大学附属第一人民医院.csv')
dff=pd.read_csv('data_上海市第十人民医院.csv')


# In[547]:


for i in [dfa,dfb,dfc,dfd,dfe,dff]:
    i.to_csv('E:/lv python/text mining/文本挖掘大作业/data_health.csv',index=False,
           mode='a',header=False,encoding="utf_8_sig")


# In[565]:


dfall=pd.read_csv('data_health.csv')
dfall.head()


# In[585]:


dfall.info()

