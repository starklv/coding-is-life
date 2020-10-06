#!/usr/bin/env python
# coding: utf-8

# # requests
# 
# ### 爬取网页内容一般步骤：有多条路径，本课程安排的是先requests请求下载网页，然后lxml+xpath解析网页内容。

# ## 实例引入

# In[1]:


import requests

response = requests.get('https://www.baidu.com/')
print(type(response))
print(response.status_code)
print(type(response.text))
print(response.text)
print(response.cookies)


# ## 各种请求方式
import requests
requests.get('http://httpbin.org/get'
requests.post('http://httpbin.org/post')
requests.put('http://httpbin.org/put')
requests.delete('http://httpbin.org/delete')
requests.head('http://httpbin.org/get')
requests.options('http://httpbin.org/get')
# # 请求

# ## 基本GET请求

# ### 基本写法

# In[3]:


import requests

response = requests.get('http://httpbin.org/get')
print(response.text)


# #### requests中提供的类和方法

# In[4]:


dir(requests)


# In[5]:


help(requests.get)

requests.get(url, params=None, headers=None, cookies=None, auth=None, timeout=None)
Sends a GET request. Returns Response object.
Parameters:

    url – URL for the new Request object.
    params – (optional) Dictionary of GET Parameters to send with the Request.
    headers – (optional) Dictionary of HTTP Headers to send with the Request.
    cookies – (optional) CookieJar object to send with the Request.
    auth – (optional) AuthObject to enable Basic HTTP Auth.
    timeout – (optional) Float describing the timeout of the request.

# ### 带参数GET请求

# In[6]:


import requests
response = requests.get("http://httpbin.org/get?name=germey&age=22")
print(response.text)


# In[7]:


import requests

data = {
    'name': 'germey',
    'age': 22
}
response = requests.get("http://httpbin.org/get", params=data)
print(response.text)


# ### 解析json

# In[8]:


import requests
import json

response = requests.get("http://httpbin.org/get")
print(type(response.text))
print(response.json())
print(json.loads(response.text))
print(type(response.json()))


# ### 获取二进制数据

# In[9]:


import requests

response = requests.get("https://github.com/favicon.ico")
print(type(response.text), type(response.content))
print(response.text)
print(response.content)


# In[10]:


import requests

response = requests.get("https://github.com/favicon.ico")
with open('favicon.ico', 'wb') as f:
    f.write(response.content)
    f.close()


# ### 添加headers

# In[11]:


import requests

response = requests.get("https://www.zhihu.com/explore")
print(response.text)


# In[12]:


import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'
}
response = requests.get("https://www.zhihu.com/explore", headers=headers)
print(response.text)


# ## 基本POST请求

# In[1]:


import requests

data = {'name': 'germey', 'age': '22'}
response = requests.post("http://httpbin.org/post", data=data)
print(response.text)


# In[1]:


import requests

data = {'name': 'germey', 'age': '22'}
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'
}
response = requests.post("http://httpbin.org/post", data=data, headers=headers)
print(response.json())


# # 响应

# ## reponse属性

# In[2]:


import requests

response = requests.get('http://www.jianshu.com')
print(type(response.status_code), response.status_code)
print(type(response.headers), response.headers)
print(type(response.cookies), response.cookies)
print(type(response.url), response.url)
print(type(response.history), response.history)


# ## 状态码判断

# In[3]:


import requests

response = requests.get('http://www.jianshu.com/hello.html')

print(response.status_code) if not response.status_code == requests.codes.not_found else print('404 Not Found')


# In[4]:


import requests

response = requests.get('http://www.jianshu.com')

print(response.status_code) if not response.status_code == 200 else print('Request Successfully')

100: ('continue',),
101: ('switching_protocols',),
102: ('processing',),
103: ('checkpoint',),
122: ('uri_too_long', 'request_uri_too_long'),
200: ('ok', 'okay', 'all_ok', 'all_okay', 'all_good', '\\o/', '✓'),
201: ('created',),
202: ('accepted',),
203: ('non_authoritative_info', 'non_authoritative_information'),
204: ('no_content',),
205: ('reset_content', 'reset'),
206: ('partial_content', 'partial'),
207: ('multi_status', 'multiple_status', 'multi_stati', 'multiple_stati'),
208: ('already_reported',),
226: ('im_used',),

# Redirection.
300: ('multiple_choices',),
301: ('moved_permanently', 'moved', '\\o-'),
302: ('found',),
303: ('see_other', 'other'),
304: ('not_modified',),
305: ('use_proxy',),
306: ('switch_proxy',),
307: ('temporary_redirect', 'temporary_moved', 'temporary'),
308: ('permanent_redirect',
      'resume_incomplete', 'resume',), # These 2 to be removed in 3.0

# Client Error.
400: ('bad_request', 'bad'),
401: ('unauthorized',),
402: ('payment_required', 'payment'),
403: ('forbidden',),
404: ('not_found', '-o-'),
405: ('method_not_allowed', 'not_allowed'),
406: ('not_acceptable',),
407: ('proxy_authentication_required', 'proxy_auth', 'proxy_authentication'),
408: ('request_timeout', 'timeout'),
409: ('conflict',),
410: ('gone',),
411: ('length_required',),
412: ('precondition_failed', 'precondition'),
413: ('request_entity_too_large',),
414: ('request_uri_too_large',),
415: ('unsupported_media_type', 'unsupported_media', 'media_type'),
416: ('requested_range_not_satisfiable', 'requested_range', 'range_not_satisfiable'),
417: ('expectation_failed',),
418: ('im_a_teapot', 'teapot', 'i_am_a_teapot'),
421: ('misdirected_request',),
422: ('unprocessable_entity', 'unprocessable'),
423: ('locked',),
424: ('failed_dependency', 'dependency'),
425: ('unordered_collection', 'unordered'),
426: ('upgrade_required', 'upgrade'),
428: ('precondition_required', 'precondition'),
429: ('too_many_requests', 'too_many'),
431: ('header_fields_too_large', 'fields_too_large'),
444: ('no_response', 'none'),
449: ('retry_with', 'retry'),
450: ('blocked_by_windows_parental_controls', 'parental_controls'),
451: ('unavailable_for_legal_reasons', 'legal_reasons'),
499: ('client_closed_request',),

# Server Error.
500: ('internal_server_error', 'server_error', '/o\\', '✗'),
501: ('not_implemented',),
502: ('bad_gateway',),
503: ('service_unavailable', 'unavailable'),
504: ('gateway_timeout',),
505: ('http_version_not_supported', 'http_version'),
506: ('variant_also_negotiates',),
507: ('insufficient_storage',),
509: ('bandwidth_limit_exceeded', 'bandwidth'),
510: ('not_extended',),
511: ('network_authentication_required', 'network_auth', 'network_authentication'),
# # 高级操作

# ## 文件上传

# In[1]:


import requests

files = {'file': open('favicon.ico', 'rb')}
response = requests.post("http://httpbin.org/post", files=files)
print(response.text)


# ## 获取cookie

# In[2]:


import requests

response = requests.get("https://www.baidu.com")
print(response.cookies)
for key, value in response.cookies.items():
    print(key + '=' + value)


# ## 会话维持

# 模拟登录

# In[3]:


import requests

requests.get('http://httpbin.org/cookies/set/number/123456789')
response = requests.get('http://httpbin.org/cookies')
print(response.text)


# In[4]:


import requests

s = requests.Session()
s.get('http://httpbin.org/cookies/set/number/123456789')
response = s.get('http://httpbin.org/cookies')
print(response.text)


# ### 使用会话抓取豆瓣影评数据

# In[5]:


import requests
 
baseurl='https://movie.douban.com/subject/25823277/comments?status=P'
user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0" 
headers={"user_agent":user_agent}   

html = requests.get(baseurl,headers=headers).content
print(html.decode('utf-8'))


# In[6]:


import requests
 
url = 'https://movie.douban.com/subject/25823277/comments?status=P'
ua_headers = { "User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0'}
      
s = requests.session()
response = s.get(url=url, headers=ua_headers)
print(response.content.decode('utf-8'))


# ## 用户名/密码模拟登录

# #### 在开发者工具（F12）中的“网络”监控中找到登录页面，以及登录的一些参数。下图豆瓣的登录信息供参考！

# In[7]:


from IPython.display import display, Image
display( Image( filename = "imgs/douban_login_F.png" ))
display( Image( filename = "imgs/douban_login.png" ))


# In[8]:


import requests
 
url_basic = 'https://accounts.douban.com/j/mobile/login/basic'
url = 'https://movie.douban.com/subject/25823277/comments?status=P'
ua_headers = { "User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0'}
data = {
        'ck': '',
        'name': 'your username',
        'password': 'your password',
        'remember': 'false',
        'ticket': ''
    }     


# In[ ]:


requests.post(url=url_basic, headers=ua_headers, data=data)


# In[ ]:


response = requests.get(url=url, headers=ua_headers)
print(response.content.decode('utf-8'))


# ## 证书验证

# In[9]:


import requests

response = requests.get('https://www.12306.cn')
print(response.status_code)


# In[10]:


import requests
from requests.packages import urllib3
urllib3.disable_warnings()
response = requests.get('https://www.12306.cn', verify=False)
print(response.status_code)


# ###### 目录下需要证书文件，否则报错

# In[ ]:


import requests

response = requests.get('https://www.12306.cn', cert=('/path/server.crt', '/path/key'))
print(response.status_code)


# ## 代理设置

# #####  代理地址有问题会报错

# In[ ]:


import requests

proxies = {
  "http": "http://127.0.0.1:9743",
  "https": "https://127.0.0.1:9743",
}

response = requests.get("https://www.taobao.com", proxies=proxies)
print(response.status_code)


# In[ ]:


import requests

proxies = {
    "http": "http://user:password@127.0.0.1:9743/",
}
response = requests.get("https://www.taobao.com", proxies=proxies)
print(response.status_code)


# ## 超时设置

# In[11]:


import requests
from requests.exceptions import ReadTimeout
try:
    response = requests.get("http://www.baidu.com", timeout = 0.5)
    print(response.status_code)
except ReadTimeout:
    print('Timeout')


# ## 异常处理

# In[ ]:


import requests
from requests.exceptions import ReadTimeout, ConnectionError, RequestException
try:
    response = requests.get("http://httpbin.org/get", timeout = 0.5)
    print(response.status_code)
except ReadTimeout:
    print('Timeout')
except ConnectionError:
    print('Connection error')
except RequestException:
    print('Error')


# In[ ]:




