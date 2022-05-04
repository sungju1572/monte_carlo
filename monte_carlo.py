import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import math

#삼성데이터 
#train 데이터 (2016~2020.12)
train = fdr.DataReader(symbol='005930', start='2015', end='2021')

#모멘텀지수 사용해야 하므로 이전데이터 몇개 추가(2015년 데이터)
train = train[220:]



#test 데이터 (2021.1~12)
test = fdr.DataReader(symbol='005930', start='2021', end='2022')



#로그 수익률
train['log_rtn'] = np.log(train.Close/train.Close.shift(1))
train = train.dropna()[['log_rtn','Close']]

#퍼센트로 계산
train['log_rtn'] = train['log_rtn']*100


plt.plot(train["Close"])

train_close = train['Close']
train_log = train['log_rtn']


#정규분포 난수 
r_n = np.random.normal(size = len(train))

len(r_n)

#train 데이터에 정규분포난수 추가
train['r_n'] = r_n


#변동률 계산
roc = math.sqrt(((train_log - train_log.mean())**2).sum()/(len(train_log)-1))

#평균 수익률(다시계산?)
earning_rate_mean = train_log.mean() -0.5*((train_log.mean())*(train_log.mean()))

#수익률 추가
train["rtn"] = earning_rate_mean + roc*train["r_n"]

plt.plot((100*np.exp(train["rtn"]/100))[:300])

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