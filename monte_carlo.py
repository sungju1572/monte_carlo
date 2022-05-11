import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import talib



#삼성데이터 
#train 데이터 (2016~2020.12)
train = fdr.DataReader(symbol='KS11', start='2015', end='2021')

#모멘텀지수 사용해야 하므로 이전데이터 몇개 추가(2015년 데이터)
train = train[220:]


#test 데이터 (2021.1~12)
test = fdr.DataReader(symbol='KS11', start='2020', end='2022')

test = test[150:]


#로그 수익률 생성 함수(input 값: dataframe)
def log_rtn(train):
    train['log_rtn'] = np.log(train.Close/train.Close.shift(1))
    train = train.dropna()[['log_rtn','Close']]

    #퍼센트로 계산
    train['log_rtn'] = train['log_rtn']*100
    train_log = train['log_rtn']
    return train_log

#정규분포 난수 생성 함수
def random_normal():
    r_n = np.random.normal(size = len(train))
    return r_n



#변동률, 평균수익률, 종가 데이터 생성
def new_data(train):
    train_log = log_rtn(train)
    r_n = random_normal()
    
    #변동률
    roc = math.sqrt(((train_log - train_log.mean())**2).sum()/(len(train_log)-1))
    
    #평균수익률
    earning_rate_mean = train_log.mean() -0.5*((roc/100)*(roc/100))
    
    #수익률 
    rtn = earning_rate_mean + roc*r_n
    
    #새로 생성한 종가 데이터
    data = (100*np.exp(rtn/100))
    
    
    return data    



#라벨링 리스트 생성 함수(input : (data, 만들 행 갯수))
def label_list(data, num):
    data_df = pd.DataFrame(data, columns = ["data"])

    ###라벨링
    label_shift = data_df['data'].diff().shift(-1)[:num]
    label_shift_list = []

    for i in range(len(label_shift)):
        if label_shift[i] > 0 :
            label_shift_list.append(1)
        else:
            label_shift_list.append(0)
    

    return label_shift_list


#기술지표 생성함수(input : (data,  만들 행 갯수,time(25)))
def tal(data, num, time):
    label = label_list(data, num)
        
    #기술지표들
    train_apo = talib.APO(data[:num])
    train_cmo = talib.CMO(data[:num])
    train_macd , train_macdsignal , train_macdhist = talib.MACD(data[:num])
    train_mom = talib.MOM(data[:num])
    train_ppo = talib.PPO(data[:num])
    train_roc = talib.ROC(data[:num])
    train_rocp = talib.ROCP(data[:num])
    train_rocr = talib.ROCR(data[:num])
    train_rocr100 = talib.ROCR100(data[:num])
    train_rsi = talib.RSI(data[:num])
    train_fasrk, train_fasrd = talib.STOCHRSI(data[:num])
    train_trix = talib.TRIX(data[:num])

    data = {'APO' : train_cmo[time:],
            'CMO' : train_cmo[time:],
            'MACD' : train_macd[time:],
            'MACDSIGNAL' : train_macdsignal[time:],
            'MACDHIST' : train_macdhist[time:],
            'MOM' : train_mom[time:],
            'PPO' : train_ppo[time:],
            'ROC' : train_roc[time:],
            'ROCP' : train_rocp[time:],
            'ROCR' : train_rocr[time:],
            'ROCR100' : train_rocr100[time:],
            'RSI' : train_rsi[time:],
            'FASRK' : train_fasrk[time:],
            'FASRD' : train_fasrd[time:],
            'TRIX' : train_trix[time:],
            'label' : label[time:]}
    
    #train_data 생성(종가 제외)
    train_data = pd.DataFrame(data)
    train_data = train_data.reset_index(drop=True)
    
    return train_data


num =275

data = new_data(train)

train_apo = talib.APO(data[:num])
train_cmo = talib.CMO(data[:num])
train_macd , train_macdsignal , train_macdhist = talib.MACD(data[:num])
train_mom = talib.MOM(data[:num])
train_ppo = talib.PPO(data[:num])
train_roc = talib.ROC(data[:num])
train_rocp = talib.ROCP(data[:num])
train_rocr = talib.ROCR(data[:num])
train_rocr100 = talib.ROCR100(data[:num])
train_rsi = talib.RSI(data[:num])
train_stochrsi = talib.STOCHRSI(data[:num])
train_trix = talib.TRIX(data[:num])



#데이터 생성(랜덤 남수)    
data = new_data(train)   
data1 = new_data(train)   

#학습 데이터 생성
b = tal(data, 338, 88)
b1 = tal(data1, 275, 25)


#train 데이터 생성 (20개)
train_data = pd.DataFrame()

for i in range(200):
    data = new_data(train)
    df = tal(data, 338, 88)
    train_data = pd.concat([train_data, df])
     
train_data = train_data.reset_index(drop=True)   


#실제 train 생성
train_df = train["Close"].diff().shift(-1)
train_df = train_df.dropna()


train_df_list = []

for i in range(len(train_df)):
    if train_df[i] > 0 :
        train_df_list.append(1)
    else:
        train_df_list.append(0)



train_apo = talib.APO(train['Close'])
train_cmo = talib.CMO(train['Close'])
train_macd , train_macdsignal , train_macdhist = talib.MACD(train['Close'])
train_mom = talib.MOM(train['Close'])
train_ppo = talib.PPO(train['Close'])
train_roc = talib.ROC(train['Close'])
train_rocp = talib.ROCP(train['Close'])
train_rocr = talib.ROCR(train['Close'])
train_rocr100 = talib.ROCR100(train['Close'])
train_rsi = talib.RSI(train['Close'])
train_fasrk, train_fasrd = talib.STOCHRSI(train["Close"])
train_trix = talib.TRIX(train['Close'])


len(train_trix[248:-1])

len(train_df_list[248:])

len(train_trix)
len(train_df_list)

time = 248

data = {'APO' : train_cmo[time:-1],
        'CMO' : train_cmo[time:-1],
        'MACD' : train_macd[time:-1],
        'MACDSIGNAL' : train_macdsignal[time:-1],
        'MACDHIST' : train_macdhist[time:-1],
        'MOM' : train_mom[time:-1],
        'PPO' : train_ppo[time:-1],
        'ROC' : train_roc[time:-1],
        'ROCP' : train_rocp[time:-1],
        'ROCR' : train_rocr[time:-1],
        'ROCR100' : train_rocr100[time:-1],
        'RSI' : train_rsi[time:-1],
        'FASRK' : train_fasrk[time:-1],
        'FASRD' : train_fasrd[time:-1],
        'TRIX' : train_trix[time:-1],
        'label' : train_df_list[time:]
        }

#train_data 생성(종가 제외)
train_data_real = pd.DataFrame(data)
train_data_real = train_data_real.reset_index(drop=True)



#test 데이터 생성
test_df = test["Close"].diff().shift(-1)
test_df = test_df.dropna()

test_df = test_df[99:]


test_df_list = []

for i in range(len(test_df)):
    if test_df[i] > 0 :
        test_df_list.append(1)
    else:
        test_df_list.append(0)


len(test_df_list)


time = 99

test_apo = talib.APO(test['Close'])
test_cmo = talib.CMO(test['Close'])
test_macd , test_macdsignal , test_macdhist = talib.MACD(test['Close'])
test_mom = talib.MOM(test['Close'])
test_ppo = talib.PPO(test['Close'])
test_roc = talib.ROC(test['Close'])
test_rocp = talib.ROCP(test['Close'])
test_rocr = talib.ROCR(test['Close'])
test_rocr100 = talib.ROCR100(test['Close'])
test_rsi = talib.RSI(test['Close'])
test_fasrk, test_fasrd = talib.STOCHRSI(test["Close"])
test_trix = talib.TRIX(test['Close'])


len(test_df)
len(test_trix[99:-1])


data = {'APO' : test_cmo[time:-1],
        'CMO' : test_cmo[time:-1],
        'MACD' : test_macd[time:-1],
        'MACDSIGNAL' : test_macdsignal[time:-1],
        'MACDHIST' : test_macdhist[time:-1],
        'MOM' : test_mom[time:-1],
        'PPO' : test_ppo[time:-1],
        'ROC' : test_roc[time:-1],
        'ROCP' : test_rocp[time:-1],
        'ROCR' : test_rocr[time:-1],
        'ROCR100' : test_rocr100[time:-1],
        'RSI' : test_rsi[time:-1],
        'FASRK' : test_fasrk[time:-1],
        'FASRD' : test_fasrd[time:-1],
        'TRIX' : test_trix[time:-1],
        'label' : test_df_list
        }

#test_data 생성(종가 제외)
test_data = pd.DataFrame(data)
test_data = test_data.reset_index(drop=True)




#train /test 라벨 나누기
X_train = train_data.drop(["label"], axis = 1 ) #학습데이터
y_train = train_data["label"] #정답라벨
X_test = test_data.drop(['label'], axis=1) #test데이터
y_test = test_data["label"]


#train_real / test 라벨나누기
X_train = train_data_real.drop(["label"], axis = 1 ) #학습데이터
y_train = train_data_real["label"] #정답라벨
X_test = test_data.drop(['label'], axis=1) #test데이터
y_test = test_data["label"]



##로지스틱 회귀분석
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

model = LogisticRegression()
model.fit(X_train, y_train)


print(model.score(X_train, y_train))

y_pred = model.predict(X_test)

accuracy_score(y_pred, y_test) #0.5121951219512195


confusion_matrix(y_pred, y_test)



#DT
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy_score(y_pred, y_test) #0.4552

#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy_score(y_pred, y_test) #0.5203252032520326



#RF
from sklearn.ensemble import RandomForestClassifier
# instantiate the classifier 
rfc = RandomForestClassifier(random_state=14)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

accuracy_score(y_pred, y_test) #0.532520325203252



#xgboost
from sklearn import preprocessing
import xgboost as xgb
from xgboost import XGBClassifier
import datetime
from sklearn.model_selection import GridSearchCV


X_train = train_data.drop(["label"], axis = 1 ) #학습데이터
y_train = train_data["label"] #정답라벨
X_test = test_data.drop(['label'], axis=1) #test데이터
y_test = test_data["label"]

xgb1 = XGBClassifier()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [3, 4, 5],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500],
              "random_state" : [26]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5, 
                        verbose=True)

xgb_grid.fit(X_train,
         y_train)


print(xgb_grid.best_score_)
print(xgb_grid.best_params_)



#prediction
y_pred = xgb_grid.predict(X_test)

accuracy_score(y_pred, y_test) #0.540650406504065
