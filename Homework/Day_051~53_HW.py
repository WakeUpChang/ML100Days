# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy, time
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data_path = 'data/'
df_train = pd.read_csv(data_path + 'train_data.csv')
df_test = pd.read_csv(data_path + 'test_features.csv')


train_Y = df_train['poi']
names = df_test['name']
df_train = df_train.drop(['name', 'poi','email_address'] , axis=1)
df_test = df_test.drop(['name','email_address'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()


def na_check(df_data):
    data_na = (df_data.isnull().sum() / len(df_data))
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    display(missing_data)
    return missing_data
na_check(df)

##與email相關的次數填為0(代表沒往來?
df["from_messages"] = df["from_messages"].fillna(0)
df["from_poi_to_this_person"] = df["from_poi_to_this_person"].fillna(0)
df["from_this_person_to_poi"] = df["from_this_person_to_poi"].fillna(0)
df["to_messages"] = df["to_messages"].fillna(0)
df["shared_receipt_with_poi"] = df["shared_receipt_with_poi"].fillna(0)

missing_data = na_check(df)


df = df[missing_data.index].fillna(0)

train_num = train_Y.shape[0]
train_X = df[:train_num]
test_X = df[train_num:]

# 使用三種模型 : 邏輯斯迴歸 / 梯度提升機 / 隨機森林, 參數使用 Random Search 尋找
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
lr = LogisticRegression(tol=0.001, penalty='l2', fit_intercept=True, C=1.0)
gdbt = GradientBoostingClassifier(tol=100, subsample=0.75, n_estimators=250, max_features=20,
                                  max_depth=6, learning_rate=0.03)
rf = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, 
                            max_features='sqrt', max_depth=6, bootstrap=True)

# 線性迴歸預測檔 (結果有部分隨機, 請以 Kaggle 計算的得分為準, 以下模型同理)
lr.fit(train_X, train_Y)
lr_pred = lr.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'name': names, 'poi': lr_pred})
#sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('exam_lr.csv', index=False) 

# 梯度提升機預測檔 
#gdbt.fit(train_X, train_Y)
#gdbt_pred = gdbt.predict_proba(test_X)[:,1]
#sub = pd.DataFrame({'name': names, 'poi': gdbt_pred})
##sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
#sub.to_csv('exam_gdbt.csv', index=False)

# 隨機森林預測檔
rf.fit(train_X, train_Y)
rf_pred = rf.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'name': names, 'poi': rf_pred})
#sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('exam_rf.csv', index=False)