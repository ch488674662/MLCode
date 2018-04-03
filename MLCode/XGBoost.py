import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



titanc=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
X=titanc[['pclass','age','sex']]
y=titanc['survived']
X['age'].fillna(X['age'].mean(),inplace=True)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)
vec=DictVectorizer(sparse=True)
x_train=vec.fit_transform(x_train.to_dict(orient='record'))
x_test=vec.transform(x_test.to_dict(orient='record'))
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
print('RFC score:',rfc.score(x_train,y_train))
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print('xgb score:',xgb.score(x_train,y_train))


