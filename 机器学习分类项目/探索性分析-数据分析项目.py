#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data1=pd.read_csv('E:/研究生相关/统计学习1课件/数据分析项目与小论文/数据分析项目/train.csv',index_col=0)


# In[3]:


data1.drop([788816,1068958],inplace=True)


# In[23]:


df=pd.DataFrame({'Not Default':[980788],'Default':[149512]})
data1['y'].value_counts()


# In[26]:


sns.barplot(data=df)


# # 处理年份数据

# In[48]:


month=data1['earliest_cr_line'].str.split('-',expand=True)
month.columns=['month','year']
data1['earliest_cr_line']=month['month']


# In[38]:


import re
data1['']
pattern=r'([A-Za-z.]+)\-([0-9.]+)'
regex=re.compile(pattern)
regex.findall(data1['earliest_cr_line'])


# # 单变量情况

# In[176]:


sns.set
sns.distplot(data1.loan_amnt) #绘制直方图和密度图
plt.ylabel('rate')
plt.title('loan_amnt distribution')
plt.show()


# In[101]:


data1['loan_amnt'].max()


# 贷款金额集中在10000元左右

# In[103]:


a=data1.grade.value_counts(1) #换上1就是比例
sns.set #设定sns时的默认参数格式
plt.rc('font', family='SimHei', size=13)# 提前设置好默认参数格式，rc第一个参数是想要定义的组件，family字体格式，size字体大小
sns.barplot(x=a.index,y=a.values)
plt.ylabel('Percentage')
plt.title('Loan Grade')
plt.show() #去除一些额外的文字


# 贷款等级中B、C两等级占比较多

# In[107]:


year=data1['emp_length'].value_counts()
sns.set
plt.rc('font',family='SimHei',size=13)
sns.barplot(x=year.index,y=year.values)
plt.xticks(rotation=90)
plt.title('Employment length')
plt.ylabel('People Amount')
plt.show()


# 发现工作10年以上的样本占大多数

# In[196]:


#作图前得提前删除异常值，不然图的效果会很难看
fig=plt.figure(figsize=(12,10))
data1['annual_inc']=data1['annual_inc'].astype(float)
annual_avg=data1['annual_inc'].mean()
data1['annual_inc']=data1['annual_inc'].fillna(annual_avg)
sns.set
sns.distplot(data1.annual_inc)
plt.ylabel('rate')
plt.title('Income distribution')
plt.xlim([0,60000])
plt.show()


# In[195]:


home=data1.home_ownership.value_counts()
sns.set
fig=plt.figure(figsize=(12,10)) #default(6.4,4.8) width,height
sns.barplot(x=home.index,y=home.values)
plt.xticks(rotation=90)
plt.ylabel('People Amount')
plt.title('Home Ownership')
plt.show()


# 按揭和租房的占大多数，有房产的人占第三

# In[22]:


annual_avg=data1['inq_last_6mths'].median()
data1['inq_last_6mths']=data1['inq_last_6mths'].fillna(annual_avg)
inquest=data1['inq_last_6mths'].value_counts()
sns.set
fig=plt.figure(figsize=(12,10))
sns.barplot(x=inquest.index,y=inquest.values)
plt.ylabel('Frquency')
plt.title('Inquest distribution')
plt.show()


# 客户过去6个月征信次数，基本都在2次以内

# In[108]:


fig=plt.figure(figsize=(16,12))
#explode 一个数组用以表示每个图形离开中心的距离，autopct百分数保留两位小数，不然不会有每个fraction代表的数值
plt.pie(data1['term'].value_counts(),labels=['36 months','60 months'],explode=[0,0.1],autopct='%0.2f%%') 
plt.title('Loan Term')
plt.show()


# 贷款期限有两种，一种是36个月，一种是60个月，其中36个月的占比71.2%，说明该信贷公司主要办理的贷款是短期贷款

# In[45]:


fig=plt.figure(figsize=(10,12))
plt.pie(data1['application_type'].value_counts(),labels=['Individual','Joint App'],explode=[0,0.1],autopct='%.2f%%')
plt.title('Application Type Proportion')
plt.savefig('E:/研究生相关/统计学习1课件/1.pdf',dpi=100)
plt.show()


# 个人来公司办理贷款的居多，受理的共同贷款业务很少

# # 双变量情况

# In[48]:


sns.set()
sns.boxplot(x=data1['grade'].values,y=data1['int_rate'])
plt.title('Relationship between Grade')
plt.show()


# AB贷款利率较低约为7-12%，贷款人数较多；CD贷款利率适中，约为14-18%，贷款人数也较多；EFG贷款利率偏高，贷款人数较少；风险等级依次为A<B<C<D<E<F<G

# In[49]:


sns.set
sns.boxplot(x=data1.grade,y=data1.loan_amnt)
plt.title('Relationship between Grade and Loan Amount')
plt.show()


# 从贷款金额上可以大致分为3个层次，AB等级，CD等级，EFG等级对应低中高的贷款金额水平；结合之前利率与等级的相关分析，可以猜测适中的贷款利率和贷款水平会是热门选择，如BC等级，这与之前单变量分析时，贷款等级中B、C两等级占比较多相符合

# In[76]:


sns.set
sns.barplot(x='y',y='total_pymnt',data=data1)
plt.title('Relationship between Default and total_pymnt')
plt.show()


# 从还贷金额上来说，明显不违约的借款人还贷金额更多，说明不违约的人有固定还贷的意识和还贷能力

# In[96]:


plt.rc('font', family='SimHei', size=13)
fig=plt.figure(figsize=(15,20))
sns.set
#hue在这里是一个分类参数用于将数值分离，且x和y不能同时使用,palette是一个调色板，比color的颜色效果明显，color太淡
sns.countplot(y='addr_state',hue='y',data=data1,palette='Greens_d') 
plt.title('Relationship Between State and Default')
plt.show()


# 可以看出各州的违约贷款分布其实相差不大，其中CA(加利福尼亚州)最为突出，说明加州人民通过该信贷公司贷款最多，所以违约与不违约的人数也最多。因为加利福尼亚拥有硅谷和洛杉矶等繁华现代都市，也可进一步推测，该信贷公司可能位于加利福尼亚州。其次也可以看出贷款数量最多的前三名是CA(加利福尼亚州)，NY(纽约州)，TX(德克萨斯州)

# In[106]:


fig=plt.figure(figsize=(10,15))
sns.countplot(y='purpose',hue='y',palette='Purples_d',data=data1)
plt.title('Relationship Between Purpose and Default')
plt.show()
plt.savefig('11.pdf',bbox_inches='tight')


# 可以看到该信贷公司主要办理债务合并类的贷款，即一种将所有个人债务打包后按某一利率进行还款的贷款项目。其背后理论是一种支付比几种更容易管理，且达到整合降低利率和每月付款的目标。所以信用风险在该信贷公司尤其重要。同时，该公司也承接大量的信用卡贷款业务。可以推测，该信贷公司应该是一家新兴的信贷公司，大概率会做贷款证券化业务。

# # 探索性分析总结

#   从该信贷公司的业务信息进行分析,办理1万美元贷款项目最多,贷款额最高是4万美元。贷款期限有两种,一种是36个月,一种是60个月,且办理短期贷款业务居多。主要办理个人申请贷款,受理的共同贷款业务很少。贷款利率集中在10%-16%。对客户的信用评级为A~G,其代表的信用风险排序为A<B<C<D<E<F<G。所以对AB等级的客户利率最低,CD较之升高,EFG等级的贷款利率升至22%及以上。同时也发现,等级与贷款金额也有一定关系,贷款金额越少,等级越高。
#   对该信贷公司的贷款客户分析。可以看到,贷款人的年收入分布较为平均,大部分工作10年以上,租房或者有按揭,贷款的目的是债务合并,信用评级大部分在A-D,过去6个月征信次数3次以内。综上,贷款客户质量较好。
#   从业务涉及的地区可以推测出,该贷款公司位于美国,且大概率位于加利福尼亚州。因为该公司涉及CA(加利福尼亚州)的业务明显多于其他州,同时加利福尼亚州拥有硅谷和洛杉矶等繁华现代都市,很适合发展金融。观察各州的贷款违约分布,发现违约情况相差不大,同时也可以看出贷款笔数最多的州前三名是CA、NY(纽约州)、TX(德克萨斯州)。
#   最后因为该信贷公司主要是办理债务合并类的贷款,同时也承接了大量的信用卡贷款业务。故合理推测,该信贷公司很可能是一家新兴的信贷公司。所以信用风险分析和客户违约率确实是该信贷公司应该关注的目标。这与我们的分析相一致。
