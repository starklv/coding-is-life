####导入数据
alldata<-read.csv('E:/研究生相关/统计学习1课件/数据分析项目与小论文/数据分析项目/train.csv')

####基于数据的初步清洗，删除第788816，1068958条观测值
summary(alldata)
alldata<-alldata[-c(788816,1068958),]
alldata$y<-factor(alldata$y)
train<-alldata[,-1]
trainx<-train[,-1]
trainy<-train[,1]
test<-read.csv('E:/研究生相关/统计学习1课件/数据分析项目与小论文/数据分析项目/testx.csv')
test<-test[,-1]

########################### 根据变量实际含义转变数据类型
###将三个原本属于定量的变量进行转化
trainx$annual_inc<-as.numeric(trainx$annual_inc)
trainx$delinq_2yrs<-as.numeric(trainx$delinq_2yrs)
trainx$total_acc<-as.numeric(trainx$total_acc)

###################提取日期变量中的月份数据,消除具体日期影响，降低因子数
if(!require(stringr)){
  install.packages(stringr)
  library(stringr)
}
trainx$earliest_cr_line<-as.factor(str_extract(as.character(trainx$earliest_cr_line),
                                               '[A-Za-z]+'))
test$earliest_cr_line<-as.factor(str_extract(as.character(test$earliest_cr_line),
                                             '[A-Za-z]+'))

#####################不相关变量和冗余变量筛选
Dele_number1<-c(9,16,17)
Dele_name1<-c('zip_code','title','emp_title')
trainx<-trainx[,-Dele_number1]
test<-test[,-Dele_number1]

####################删除缺失数量较多的变量
N<-dim(trainx)[2]
c<-rep(0,N)
for (i in 1:N){
  c[i]<-mean(is.na(train[,i]))
}
which(c>0.35)
Dele_number2<-c(20,33,34,37,38,39,40,41,42,43,44,45,46,47,48,49,50,62)
Dele_name2<-c('mths_since_last_record','annual_inc_joint','dti_joint',
              'open_acc_6m','open_act_il','open_il_12m','open_il_24m','mths_since_rcnt_il',
              'total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc',
              'all_util','inq_fi','total_cu_tl','inq_last_12m','mths_since_recent_bc_dlq')
trainx<-trainx[,-Dele_number2]
test<-test[,-Dele_number2]

###################删除方差几乎为0的变量(不包括两个定性变量)
if(!require(caret)){
  install.packages(caret)
  library(caret)
}
nzv<-nearZeroVar(trainx,saveMetrics=T)
which(nzv$nzv==TRUE)
row.names(nzv)
Dele_number3<-c(22,32,38,55,56,57,59,62)
Dele_name3<-c('revol_bal','tot_coll_amt','chargeoff_within_12_mths',
              'num_tl_120dpd_2m','num_tl_30dpd','num_tl_90g_dpd_24m','pct_tl_nvr_dlq',
              'tax_liens')
trainx<-trainx[,-Dele_number3]
test<-test[,-Dele_number3]



##################分离定性变量和定量变量
ID_cat<-c(4,7,8,9,10,12,13,14,15,18,24,30,59)
name_cat<-c('term','grade','sub_grade','emp_length','home_ownership',
            'verification_status','pymnt_plan','purpose','addr_state','earliest_cr_line',
            'initial_list_status','application_type','disbursement_method')
trainx_catvar<-trainx[,ID_cat]
trainx_quavar<-trainx[,-ID_cat]

###########################删除自变量间高度相关的变量
#####作箱线图和相关系数来判定取值的相关性
Dele_number4<-c(2,3,26,28)
Dele_name4<-c('funded_amnt','funded_amnt_inv','out_prncp_inv','total_pymnt_inv')
trainx<-trainx[,-Dele_number4]
test<-test[,-Dele_number4]


#############查看缺失值情况,并删除缺失值较多的观测值
if(!require(mice)){
  install.packages(mice)
  libraray(mice)
}
if(!require(VIM)){
  install.packages(VIM)
  library(VIM)
}
md.pattern(trainx)
aggr(trainx,prop=T,numbers=F)
###可以用来代替下面一部分，减少运算时间
### f1<-function(x)
###       sum(is.na(x))
###case_na_num<-apply(trainx,1,f1)
###case_na_num<-data.frame(case_na_num)
###pr<-(case_na/ncol(trainx))*100
case_na_num<-apply(trainx,1,is.na)   # 1 indicates rows,2 indicates columns
case_na_num<-apply(case_na_num,2,mean) #各观测值缺失个数
Dele_number5<-which(case_na_num>=0.48)
trainx<-trainx[-Dele_number5,]
trainy<-trainy[-Dele_number5]
aggr(trainx,prop=T,numbers=T)

#############最后对定性数据进行整理,缩减因子变量的类型
#1.将home_ownership因子归结为'ANY','RENT','OWN','MORTGAGE'
trainx$home_ownership[which(trainx$home_ownership=='OTHER')]<-'ANY'
trainx$home_ownership[which(trainx$home_ownership=='NONE')]<-'ANY'

############清洗定性变量的levels
ID_cat1<-c(2,5,6,7,8,10,11,12,13,16,22,26,55)
trainx[,ID_cat1]<-droplevels(trainx[,ID_cat1])
test[,ID_cat1]<-droplevels(test[,ID_cat1])

###查看每一行缺失值的个数,客户有效性度量，生成新变量effective
###客户信息越完善,违约的可能性越小
f1<-function(x)
  sum(is.na(x))
na.r<-apply(train_copy,1,f1)
na.r<-data.frame(na.r)
pr<-(na.r/ncol(train_copy))*100
cbind(na.r,pr)
effective<-rep(0,nrow(train_copy))
effective[which(pr>60)]<-1 #最少客户信息
effective[which(pr>20&pr<=60)]<-2 #较少客户信息
effective[which(pr<=20)]<-3 #较多客户信息
train_copy<-cbind(train_copy,effective)
train_copy$effective<-as.factor(train_copy$effective)


###########变量取值变换
# int_rate 分箱
int_rate3<-rep(0,nrow(train_copy))
pr<-train_copy$int_rate
int_rate3[which(pr<10)]<-'low' #低贷款利率
int_rate3[which(10<=pr & pr<16)]<-'middle' #中等贷款利率 
int_rate3[which(pr>=16)]<-'high' #高贷款利率
train_copy$int_rate<-as.factor(train_copy$int_rate)
names(train_copy)[6]<-'int_rate_level'

#############保存数据
trainset<-cbind(trainy,trainx)
names(trainset)[1]<-'y'
write.csv(trainset,file='E:/研究生相关/统计学习1课件/数据分析项目与小论文/数据分析项目/trainset.csv')
write.csv(test,file='E:/研究生相关/统计学习1课件/数据分析项目与小论文/数据分析项目/testxset.csv')


#--------------------重要性变量作图---------------
library(ggplot2)
data1<-read.csv('E:/研究生相关/统计学习1课件/数据分析项目与小论文/数据分析项目/trainset.csv')
names(data1)[2]<-'status'
data1$status<-factor(data1$status,labels=c('违约','未违约'))
####对因变量y的分布作条形图观察分布
ggplot(data=data1,aes(x=status,fill=status))+
  geom_bar()
#### loan_amnt(借款人申请的贷款总额)作箱线图和密度图
ggplot(data=data1,aes(x=loan_amnt,fill=status))+
  geom_density()+
  labs(x='贷款金额',y='分布密度',title='贷款金额与违约情况的关系')+
  facet_grid(.~status)
ggplot(data=data1,aes(x=status,y=loan_amnt,color=status))+
  geom_boxplot()+
  labs(x='违约情况',y='借款人收到的贷款金额',title='贷款金额与违约情况的关系')

######total_rec_prncp(当前收到客户本金)作箱线图
ggplot(data=data1,aes(x=status,y=total_rec_prncp,color=status))+
  geom_boxplot()+
  labs(title='违约情况与当前收到本金的关系',x='违约情况',y='当前收到本金')

ggplot(data=data1,aes(x=total_rec_prncp,fill=status))+
  geom_density()+
  labs(title='违约情况与当前收到本金的关系',x='违约情况',y='当前收到本金')+
  facet_grid(.~status)   ###每个status的独立图，配置成一个单行

###### sub_grade(指定贷款次等级)作条形图
ggplot(data=data1,aes(x=sub_grade,fill=status))+
  geom_bar()+
  facet_grid(status~.) ###每个status的独立图，配置成一个单列

