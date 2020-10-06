#!/usr/bin/env python
# coding: utf-8

# In[2]:


from selenium import webdriver
import time
import json
import pandas as pd
import numpy as np
from selenium.common.exceptions import NoSuchElementException
import requests
import re


# In[ ]:


class Spider_wangyiyun:
    
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
        self.browser.switch_to.frame('g_iframe')#要转换到g_iframe
        self.browser.implicitly_wait(5) #隐式等待,针对所有元素
        self.parse_content()
        self.save_content()
        self.txt()
        
if __name__ == '__main__':        
    apl = Spider_wangyiyun()
    apl.run()              


# In[ ]:


# 隐式等待
implicitly_wait()
# 比如：
driver.implicitly_wait(10)
# 说明：隐式等待是全局的是针对所有元素，设置等待时间如10秒，如果10秒内出现，则继续向下，
# 否则抛异常。可以理解为在10秒以内，不停刷新看元素是否加载出来


# In[ ]:


#页数
pagecount_button = browser.find_element_by_css_selector('a[class^="zpgi zpg9 "]') 
#只用类名前面部分定位,因为后面是动态随机生成的,
##css可以灵活地选择控件的任意属性,比如id,class,
#属性值包含wd：适用于由空格分隔的属性值,find_element_by_css_selector("[name~='wd']")
# 组合定位元素
# 标签名#id属性值：指的是该input标签下id属性为kw的元素
# find_element_by_css_selector("input#kw")
# 标签名.class属性值：指的是该input标签下class属性为s_ipt的元素
# find_element_by_css_selector("input.s_ipt")
# 标签名[属性=’属性值‘]：指的是该input标签下name属性为wd的元素
#find_element_by_css_selector("input[name='wd']")
pagecount = int(pagecount_button.text)
pagecount


# In[ ]:


#通过link文字定位元素
pages=browser.find_element_by_link_text('下一页')
browser.execute_script('arguments[0].click();',pages)
#通过link文字定位元素模糊定位元素
#使用：find_element_by_partial_link_text("部分text_vaule")
#实例：find_element_by_partial_link_text("登")


# In[49]:


#无界面启动模式
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
browser = webdriver.Chrome(options=chrome_options)


# In[ ]:


##显示启动模式
url='https://music.163.com/#/search/m/?s=%E8%AE%B8%E5%B5%A9&type=1'
browser=webdriver.Chrome()

##隐式等待3s
browser.implicitly_wait(5)
browser.get(url)

#切换子页面,很关键
browser.switch_to.frame('g_iframe')

#爬取第一页内容
products=[]
parse_content()

#确定页数
pagecount_button = browser.find_element_by_css_selector('a[class^="zpgi zpg9 "]') 
pagecount = int(pagecount_button.text)
for page in range(1,5):
    print(page+1)
    ##翻页按钮
    pages=browser.find_element_by_link_text('下一页')
    browser.execute_script('arguments[0].click();',pages)
    parse_content()    


# In[ ]:




