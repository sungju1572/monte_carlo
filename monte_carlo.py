import FinanceDataReader as fdr


#삼성데이터 
#train 데이터 (2016~2020.12)
train = fdr.DataReader(symbol='005930', start='2016', end='2021')

train = train["Close"]

#test 데이터 (2021.1~12)
test = fdr.DataReader(symbol='005930', start='2021', end='2022')

test = test["Close"]




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

# 옵티마이저 설정