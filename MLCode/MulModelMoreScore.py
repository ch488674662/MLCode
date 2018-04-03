import numpy as np
from sklearn.linear_model import LinearRegression,BayesianRidge,ElasticNet
from sklearn.svm import SVR
#集成算法
from sklearn.ensemble.gradient_boosting import  GradientBoostingRegressor
from sklearn.cross_validation import cross_val_score
#导入指标算法
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
import pandas as pd
import matplotlib as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def cal_corrcoef(float_df,y_train,float_col):
    corr_values = []
    print(y)
    print(float_df)
    for col in float_col:
        corr_values.append(abs(np.corrcoef(float_df[col].values,y_train)\
                [0,1]))
    corr_df = pd.DataFrame({'col':float_col,'corr_value':corr_values})
    corr_df = corr_df.sort_values(by='corr_value',ascending=False)
    return corr_df
##data prepare
raw_data=pd.read_csv(u'data_feature.csv',encoding='gbk')
y=pd.read_csv('y.csv',header=None,encoding='gbk')
print(raw_data)
# x_train=raw_data.ix[:,:-2]
# y_train=raw_data.ix[:,-1:]
# raw_data.replace('??',1)
# print(raw_data.dtypes)
# y_train=raw_data.ix[:,-1]
##分割数据集
# print(raw_data.ix[:,1:])
# print(raw_data.columns)

y_scalar=preprocessing.StandardScaler()
data_scale=preprocessing.StandardScaler().fit_transform(raw_data)
y_scale=y_scalar.fit_transform(y)
data_train=data_scale[0:5642]
data_train=pd.DataFrame(data_train,columns=raw_data.columns)
y=pd.DataFrame(y_scale)
print(data_train)

# print(data.values)
#计算各个属性和血糖浓度的相关性
corr_list=[]
for col in data_train.columns:
    print(col)
    print(data_train[col].values)
    print(y.values)
    corr=np.corrcoef(np.reshape(data_train[col].values,(5642,1)),y.values)
    corr_list.append(corr)

    # print(corr)

# col=cal_corrcoef(data,np.array(y),raw_data.columns)

# print(y_scale)
# print(data_scale)
data_scale=pd.DataFrame(data_scale)
y=pd.DataFrame(y_scale)

x=data_scale.ix[0:5641,:]

# print(np.shape(y))
x_test=data_scale.ix[5642:,:]
# x_train=raw_data.ix[:,:]
# y_train=raw_data.ix[:,:]
#
#
#########################train model
# n_folds=5
# model_br=BayesianRidge()#贝叶斯领回归模型
# model_lr=LinearRegression()
# model_etc=ElasticNet()
# model_svr=SVR()
# model_gbr=GradientBoostingRegressor(learning_rate=0.001)#梯度增强回归模型
# model_mlp=MLPRegressor(hidden_layer_sizes=(20,20,20))
#
# model_names=['BayesianRidge','LinearRegression','ElasticNet','SVR','GradientBoostingRegressor','MLPRegressor']
# #不同模型的对象集合
# model_dic=[model_br,model_lr,model_etc,model_svr,model_gbr,model_mlp]
# #交叉验证结果列表
# cv_score_list=[]
# pre_y_list=[]
#
# for model in model_dic:
#     #将每个回归模型导入到交叉验证模型中做训练验证
#     scores=cross_val_score(model,x,y,cv=n_folds)
#     cv_score_list.append(scores)
#     pre_y_list.append(model.fit(x,y).predict(x))
#
# #模型评估
# n_samples,n_features=np.shape(x)
# #回归评估指标对象集
# model_metrics_name=[explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
# model_metrics_list = []  # 回归评估指标列表
# y=np.reshape(np.array(y),(np.shape(y)[0],1))
# for i in range(6):
#     tmp_list=[]
#     for m in model_metrics_name:
#         # print('y.shape:',np.shape(y))
#         # print('pre_y_list.shape:',np.shape(pre_y_list[i]))
#
#         tmp_score=m(y,pre_y_list[i]) # 计算每个回归指标结果
#         tmp_list.append(tmp_score)
#     model_metrics_list.append(tmp_list)
# df1=pd.DataFrame(cv_score_list,index=model_names)
# df2=pd.DataFrame(model_metrics_list,index=model_names)
# #建立回归指标的数据框
# # ames,columns=['ev', 'mae', 'mse', 'r2']
# print('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
# print (70 * '-')  # 打印分隔线
# print ('cross validation result:')  # 打印输出标题
# print (df1)  # 打印输出交叉检验的数据框
# print (70 * '-')  # 打印分隔线
# print ('regression metrics:')  # 打印输出标题
# print (df2)  # 打印输出回归指标的数据框
# print (70 * '-')  # 打印分隔线
# print ('short name \t full name')  # 打印输出缩写和全名标题
# print ('ev \t explained_variance')
# print ('mae \t mean_absolute_error')
# print ('mse \t mean_squared_error')
# print ('r2 \t r2')
# print (70 * '-')  #打印分隔线
# predict=model_mlp.predict(x_test)
# predict=y_scalar.inverse_transform(predict)
# data_test=pd.DataFrame(predict)

# data_test.to_csv('test.csv',index=None,header=None)
#效果展示
# plt.figure()# 创建画布
# plt.plot(np.arange(x.shape()[0],y,color='k',label='true_y'))# 画出原始值的曲线
# color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
# linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
#
# #画出通过回归模型预测得到的索引及结果
# for i,pre_y in enumerate(pre_y_list):
#     plt.plot(np.arange(x.shape()[0],pre_y_list[i],color_list[i],label=model_names[i]))
# plt.title('regression result comparison')  # 标题
# plt.legend(loc='upper right')  # 图例位置
# plt.ylabel('real and predicted value')  # y轴标题
# plt.show()  # 展示图像

# #用算法拟合进行缺失值填充，适用于缺失值较少时
# def set_missing_browse_his(df):
#     # 把已有的数值型特征取出来输入到RandomForestRegressor中
#     process_df = df[['browse_his', 'gender', 'job', 'edu', 'marriage', 'family_type']]
#     # 乘客分成已知该特征和未知该特征两部分
#     known = process_df[process_df.browse_his.notnull()].as_matrix()
#     unknown = process_df[process_df.browse_his.isnull()].as_matrix()
#     # X为特征属性值
#     X = known[:, 1:]
#     # y为结果标签值
#     y = known[:, 0]
#     # fit到RandomForestRegressor之中
#     rfr = RandomForestRegressor(random_state=0, n_estimators=2000,  n_jobs=-1)
#     rfr.fit(X,y)
#     # 用得到的模型进行未知特征值预测
#     predicted = rfr.predict(unknown[:, 1::])
#     # 用得到的预测结果填补原缺失数据
#     df.loc[(df.browse_his.isnull()), 'browse_his'] = predicted
#     return df, rfr

