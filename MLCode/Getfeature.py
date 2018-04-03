

import numpy as np
import pandas as pd
import xlrd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score

#### calculate miss values
def col_miss(train_df):
    col_missing_df = train_df.isnull().sum(axis=0).reset_index()
    col_missing_df.columns = ['col','missing_count']
    col_missing_df = col_missing_df.sort_values(by='missing_count')
    return col_missing_df

#### obtain cols of XX type
def obtain_x(train_df,xtype):
    dtype_df = train_df.dtypes.reset_index()
    dtype_df.columns = ['col','type']
    return dtype_df[dtype_df.type==xtype].col.values

def date_cols(train_df,float_col):
    float_date_col = []
    for col in float_col:
        if train_df[col].min() > 1e13:
            float_date_col.append(col)
    return float_date_col

def float_uniq(float_df,float_col):
    float_uniq_col = []
    for col in tqdm(float_col):
        uniq = float_df[col].unique()
        if len(uniq) == 1:
            float_uniq_col.append(col)
    return float_uniq_col

def cal_corrcoef(float_df,y_train,float_col):
    corr_values = []
    for col in float_col:
        corr_values.append(abs(np.corrcoef(float_df[col].values,y_train)\
                [0,1]))
    corr_df = pd.DataFrame({'col':float_col,'corr_value':corr_values})
    corr_df = corr_df.sort_values(by='corr_value',ascending=False)
    return corr_df

def build_model(x_train,y_train):
    reg_model =linear_model.Lasso(alpha=0.0018)
    reg_model.fit(X=x_train,y=y_train)
    return reg_model
def accurate_rate(y_net,y_true):
    loss=np.linalg.norm(y_net-y_true)/99.0
    return loss
if __name__ == '__main__':
    # read train data
    print('read train...')
    train_df = pd.read_excel('train.xlsx')
    print('train shape:',train_df.shape)
    # calculate the number of miss values
    col_missing_df = col_miss(train_df)
    # del cols of all nan
    all_nan_columns = col_missing_df[col_missing_df.missing_count==499].\
                        col.values
    print('number of all nan col:',len(all_nan_columns))
    train_df.drop(all_nan_columns,axis=1,inplace=True)
    print('deleted,and train shape:', train_df.shape)
    # obtain float cols
    float64_col = obtain_x(train_df,'float64')
    print('obtained float cols, and count:',len(float64_col))
    # del cols that miss number greater than 200
    miss_float = train_df[float64_col].isnull().sum(axis=0).reset_index()
    miss_float.columns = ['col','count']
    miss_float_almost = miss_float[miss_float['count']>300].col.values
    float64_col = float64_col.tolist()
    float64_col = [col for col in float64_col if col not in \
                    miss_float_almost]
    print('deleted cols that miss number > 200')
    # del date cols
    float64_date_col = date_cols(train_df,float64_col)
    float64_col = [col for col in float64_col if col not in\
                    float64_date_col]
    print('deleted date cols, and number of float cols:',len(float64_col))
    # fill nan
    print('get float cols data and fill nan...')
    float_df = train_df[float64_col]
    float_df.fillna(float_df.median(),inplace=True)
    print('filled nan')
    # del cols which unique eq. 1
    float64_uniq_col = float_uniq(float_df,float64_col)
    float64_col = [col for col in float64_col if col not in\
                    float64_uniq_col]
    print('deleted unique cols, and float cols count:',len(float64_col))
    # obtained corrcoef greater than 0.2
    float64_col.remove('Y')
    y_train = train_df.Y.values
    corr_df = cal_corrcoef(float_df,y_train,float64_col)
    corr02 = corr_df[corr_df.corr_value>=0.2]
    corr02_col = corr02['col'].values.tolist()
    print('get x_train')
    x_train = float_df[corr02_col].values
    print('get test data...')
    test_df = pd.read_csv('testB.csv')
    sub_test = test_df[corr02_col]
    sub_test.fillna(sub_test.median(),inplace=True)
    x_test = sub_test.values
    print('x_train shape:',x_train.shape)
    print('x_test shape:',x_test.shape)
    print('build model...')
    X = np.vstack((x_train,x_test))
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y_train_scale=preprocessing.MinMaxScaler().fit_transform(y_train)
    X=pd.DataFrame(X)
    y_train_scale = pd.DataFrame(y_train_scale)
    # y_train=pd.DataFrame(y_train)
    X.to_csv('x_b.csv',header=None,index=None)

    y_train_scale.to_csv('y_scale.csv',header=None,index=None)

    # # X=preprocessing.MinMaxScaler().fit_transform(X)
    # # y_prepro=preprocessing.MinMaxScaler()
    # # y_train=preprocessing.MinMaxScaler().fit_transform(y_train)
    # x_train = X[0:len(x_train)]
    # x_val=X[100:199]
    #
    # x_test = X[len(x_train):]
    # model = build_model(x_train,y_train)
    # print('predict and submit...')
    # y_net = model.predict(x_val)
    # y_val = y_train[100:199]
    # # print(np.shape(y_net))
    # # print(np.shape(y_val))
    # print('val:', accurate_rate(np.reshape(y_net,(99,1)), np.reshape(y_val,(99,1))))
    # # print(cross_val_score(model,x_train,y_train,cv=5,scoring='mean_squared_error'))
    #
    #
    # subA = model.predict(x_test)
    # # read submit data
    # # subA=y_prepro.inverse_transform(subA)
    # sub_df = pd.read_csv('subA.csv',header=None)
    # sub_df['Y'] = subA
    # sub_df.to_csv('github.csv',header=None,index=False)
