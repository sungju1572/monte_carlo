import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import talib

#삼성데이터 
#train 데이터 (2016~2020.12)
train = fdr.DataReader(symbol='005930', start='2015', end='2021')

#모멘텀지수 사용해야 하므로 이전데이터 몇개 추가(2015년 데이터)
train = train[220:]



#test 데이터 (2021.1~12)
test = fdr.DataReader(symbol='005930', start='2021', end='2022')


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
    earning_rate_mean = train_log.mean() -0.5*((train_log.mean())*(train_log.mean()))
    
    #수익률 
    rtn = earning_rate_mean + roc*r_n
    
    #새로 생성한 종가 데이터
    data = (100*np.exp(rtn/100))
    
    
    return data

#데이터 생성    
data = new_data(train)    


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


a = label_list(data, 299)
len(a)



#기술지표들
real = talib.CMO(a, timeperiod=20)

plt.plot(talib.CMO(test["Close"], timeperiod=20))
plt.plot(talib.CMO(train['new_data'][:300], timeperiod=20))

plt.plot(talib.PPO(test["Close"]))
plt.plot(talib.PPO(train['new_data'][:300]))

plt.plot(talib.ROC(test["Close"]))
plt.plot(talib.ROC(train['new_data'][:300]))

plt.plot(talib.RSI(test["Close"]))
plt.plot(talib.RSI(train['new_data'][:300]))

train_cmo = talib.CMO(data[:299], timeperiod=25)
train_ppo = talib.PPO(data[:299])
train_roc = talib.ROC(data[:299])
train_rsi = talib.RSI(data[:299])

train_cmo = train_cmo.reset_index(drop=True)
train_ppo = train_ppo.reset_index(drop=True)
train_roc = train_roc.reset_index(drop=True)
train_rsi = train_rsi.reset_index(drop=True)



len(train_cmo)
len(label_shift_list)

data = {'CMO' : train_cmo[25:],
        'PPO' : train_ppo[25:],
        'ROC' : train_roc[25:],
        'RSI' : train_rsi[25:],
        'label' : label_shift_list[25:]}

#train_data 생성(종가 제외)
train_data = pd.DataFrame(data)
train_data = train_data.reset_index(drop=True)




# 5) eager execution 기능 끄기
tf.compat.v1.disable_eager_execution()

# 실제 데이터 준비
real_data = np.random.normal(size=1000)
real_data = real_data.reshape(real_data.shape[0], 1)

# 가짜 데이터 생성
def makeZ(m, n):
    z = np.random.uniform(-1.0, 1.0, size=[m, n])
    return z

# 모델 파라미터 설정
d_input = real_data.shape[1]
d_hidden = 32
d_output = 1 # 주의
g_input = 16
g_hidden = 32
g_output = d_input # 주의

def build_GAN(discriminator, generator):
    discriminator.trainable = False # discriminator 업데이트 해제
    z = Input(batch_shape=(None, g_input))
    Gz = generator(z)
    DGz = discriminator(Gz)
    
    gan_model = Model(z, DGz)
    gan_model.compile(loss='binary_crossentropy', optimizer=myOptimizer(0.0005))
    
    return gan_model