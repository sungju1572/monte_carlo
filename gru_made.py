import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import FinanceDataReader as fdr


start = dt.datetime(2017, 1, 1)
end = dt.datetime(2022, 12, 31)

data = fdr.DataReader(symbol= '005930', start= start, end=end)

data['pct_change'] = (data['Close'] - data['Close'].shift(1))/ data['Close'].shift(1)
data['log_change'] = np.log(data['Close']/ data['Close'].shift(1))

class Generator():
    def __init__(self):
        pass

    def SMA(self, data, windows):
        res = data.rolling(window = windows).mean()
        return res

    def EMA(self, data, windows):
        res = data.ewm(span = windows).mean()
        return res

    def MACD(self, data, long, short, windows):
        short_ = data.ewm(span = short).mean()
        long_ = data.ewm(span = long).mean()
        macd_ = short_ - long_
        res = macd_.ewm(span = windows).mean()
        return res

    def RSI(self, data, windows):
        delta = data.diff(1)
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_up = up.rolling(window = windows).mean()
        avg_down = down.rolling(window = windows).mean()
        rs = avg_up/ avg_down
        rsi = 100. -(100./ (1. + rs))
        return rsi

    def atr(self, data_high, data_low, windows):
        range_ = data_high - data_low
        res = range_.rolling(window = windows).mean()
        return res

    def bollinger_band(self, data, windows):
        sma = data.rolling(window = windows).mean()
        std = data.rolling(window = windows).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper, lower

    def rsv(self, data, windows):
        min_ = data.rolling(window = windows).min()
        max_ = data.rolling(window = windows).max()
        res = (data - min_)/ (max_ - min_) * 100
        return res

Generator = Generator()

data['7ma'] = Generator.EMA(data['Close'], 7)
data['14ma'] = Generator.EMA(data['Close'], 14)
data['21ma'] = Generator.EMA(data['Close'], 21)
data['7macd'] = Generator.MACD(data['Close'], 3, 11, 7)
data['14macd'] = Generator.MACD(data['Close'], 7, 21, 14)
data['7rsi'] = Generator.RSI(data['Close'], 7)
data['14rsi'] = Generator.RSI(data['Close'], 14)
data['21rsi'] = Generator.RSI(data['Close'], 21)
data['7atr'] = Generator.atr(data['High'], data['Low'], 7)
data['14atr'] = Generator.atr(data['High'], data['Low'], 14)
data['21atr'] = Generator.atr(data['High'], data['Low'], 21)
data['7upper'], data['7lower'] = Generator.bollinger_band(data['Close'], 7)
data['14upper'], data['14lower'] = Generator.bollinger_band(data['Close'], 14)
data['21upper'], data['21lower'] = Generator.bollinger_band(data['Close'], 21)
data['7rsv'] = Generator.rsv(data['Close'], 7)
data['14rsv'] = Generator.rsv(data['Close'], 14)
data['21rsv'] = Generator.rsv(data['Close'], 21)

data_combine = data.dropna()

x_ = np.arange(data_combine.shape[0])
plt.figure(figsize=(12, 6))
plt.plot(data_combine['7ma'].values, label = 'MA 7', color = 'g', linestyle = '--')
plt.plot(data_combine['Close'].values, label = 'Closing price', color = 'b')
plt.plot(data_combine['21ma'].values, label = 'MA 21', color = 'r', linestyle = '--')
plt.plot(data_combine['7upper'].values, label = 'Upper Bound', color = 'c')
plt.plot(data_combine['7lower'].values, label = 'Lower Bound', color = 'c')
plt.fill_between(x_, data_combine['7lower'].values, data_combine['7upper'].values, alpha = 0.35)
plt.title('Technical indicators')
plt.ylabel('TWD')
plt.xlabel('Days')
plt.legend()


close_fft = np.fft.fft(np.asarray(data_combine['Close'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

plt.figure(figsize=(14, 7), dpi=100)
fft_list = np.asarray(fft_df['fft'].tolist())
for num_ in [3, 6, 9, 27, 81, 100]:
    fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_] = 0
    data_combine[f'FT_{num_}components'] = np.fft.ifft(fft_list_m10)
    plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
plt.plot(data_combine['Close'].values,  label='Real')
plt.xlabel('Days')
plt.ylabel('TWD')
plt.title('TSMC (close) stock prices & Fourier transforms')
plt.legend()
plt.show()

data_combine['FT_3components'] = data_combine['FT_3components'].astype('float')
data_combine['FT_6components'] = data_combine['FT_6components'].astype('float')
data_combine['FT_9components'] = data_combine['FT_9components'].astype('float')
data_combine['FT_27components'] = data_combine['FT_27components'].astype('float')
data_combine['FT_81components'] = data_combine['FT_81components'].astype('float')
data_combine['FT_100components'] = data_combine['FT_100components'].astype('float')
data_combine.head()


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math


data = data_combine.iloc[:2665, :]
data = data.dropna()

data = data[np.isfinite(data).all(1)]


#data = data.reset_index()
#data[data["Date"]=="2021-12-30"].index


data['y'] = data['Close']


x = data.iloc[:, :34].values
y = data.iloc[:, 34].values


#split = int(data.shape[0]* 0.8)
#train_x, test_x = x[: split, :], x[split - 20:, :]
#train_y, test_y = y[: split, ], y[split - 20: , ]

#split = int(data.shape[0]* 0.8)
train_x, test_x = x[:1171], x[1171:]
train_y, test_y = y[:1171], y[1171:]

print(f'trainX: {train_x.shape} trainY: {train_y.shape}')
print(f'testX: {test_x.shape} testY: {test_y.shape}')

x_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))

train_x = x_scaler.fit_transform(train_x)
test_x = x_scaler.transform(test_x)

train_y = y_scaler.fit_transform(train_y.reshape(-1, 1))
test_y = y_scaler.transform(test_y.reshape(-1, 1))

class VAE(nn.Module):
    def __init__(self, config, latent_dim):
        super().__init__()

        modules = []
        for i in range(1, len(config)):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i - 1], config[i]),
                    nn.ReLU()
                )
            )
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(config[-1], latent_dim)
        self.fc_var = nn.Linear(config[-1], latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, config[-1])

        for i in range(len(config) - 1, 1, -1):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i], config[i - 1]),
                    nn.ReLU()
                )
            )       
        modules.append(
            nn.Sequential(
                nn.Linear(config[1], config[0]),
                nn.Sigmoid()
            )
        ) 

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logVar = self.fc_var(result)
        return mu, logVar

    def decode(self, x):
        result = self.decoder(x)
        return result

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5* logVar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        output = self.decode(z)
        return output, z, mu, logVar

train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float()), batch_size = 128, shuffle = False)
model = VAE([34, 400, 400, 400, 10], 10)


use_cuda = 1
device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
num_epochs = 500
learning_rate = 0.00001

model = model.to(device)   
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

hist = np.zeros(num_epochs) 
for epoch in range(num_epochs):
    total_loss = 0
    loss_ = []
    for (x, ) in train_loader:
        x = x.to(device)
        output, z, mu, logVar = model(x)
        kl_divergence = 0.5* torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = F.binary_cross_entropy(output, x) + kl_divergence
        loss.backward()
        optimizer.step()
        loss_.append(loss.item())
    hist[epoch] = sum(loss_)
    print('[{}/{}] Loss:'.format(epoch+1, num_epochs), sum(loss_))

plt.figure(figsize=(12, 6))
plt.plot(hist)

model.eval()
_, VAE_train_x, train_x_mu, train_x_var = model(torch.from_numpy(train_x).float().to(device))
_, VAE_test_x, test_x_mu, test_x_var = model(torch.from_numpy(test_x).float().to(device))


def sliding_window(x, y, window):
    x_ = []
    y_ = []
    y_gan = []
    for i in range(window, x.shape[0]):
        tmp_x = x[i - window: i, :]
        tmp_y = y[i]
        tmp_y_gan = y[i - window: i + 1]
        x_.append(tmp_x)
        y_.append(tmp_y)
        y_gan.append(tmp_y_gan)
    x_ = torch.from_numpy(np.array(x_)).float()
    y_ = torch.from_numpy(np.array(y_)).float()
    y_gan = torch.from_numpy(np.array(y_gan)).float()
    return x_, y_, y_gan


train_x = np.concatenate((train_x, VAE_train_x.cpu().detach().numpy()), axis = 1)
test_x = np.concatenate((test_x, VAE_test_x.cpu().detach().numpy()), axis = 1)


train_x_slide, train_y_slide, train_y_gan = sliding_window(train_x, train_y, 3)
test_x_slide, test_y_slide, test_y_gan = sliding_window(test_x, test_y, 3)
print(f'train_x: {train_x_slide.shape} train_y: {train_y_slide.shape} train_y_gan: {train_y_gan.shape}')
print(f'test_x: {test_x_slide.shape} test_y: {test_y_slide.shape} test_y_gan: {test_y_gan.shape}')

class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru_1 = nn.GRU(input_size, 1024, batch_first = True)
        self.gru_2 = nn.GRU(1024, 512, batch_first = True)
        self.gru_3 = nn.GRU(512, 256, batch_first = True)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        

    def forward(self, x):
        use_cuda = 1
        device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
        h0 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1, _ = self.gru_1(x, h0)
        out_1 = self.dropout(out_1)
        h1 = torch.zeros(1, x.size(0), 512).to(device)
        out_2, _ = self.gru_2(out_1, h1)
        out_2 = self.dropout(out_2)
        h2 = torch.zeros(1, x.size(0), 256).to(device)
        out_3, _ = self.gru_3(out_2, h2)
        out_3 = self.dropout(out_3)
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out_6 = self.linear_3(out_5)
        return out_6

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size = 5, stride = 1, padding = 'same')
        self.conv2 = nn.Conv1d(32, 64, kernel_size = 5, stride = 1, padding = 'same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size = 5, stride = 1, padding = 'same')
        self.linear1 = nn.Linear(128, 220)
        self.linear2 = nn.Linear(220, 220)
        self.linear3 = nn.Linear(220, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.leaky(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.leaky(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.leaky(conv3)
        flatten_x =  conv3.reshape(conv3.shape[0], conv3.shape[1])
        out_1 = self.linear1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        out = self.linear3(out_2)
        return out
    
    
use_cuda = 1
device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")

batch_size = 128
learning_rate = 0.000115
num_epochs = 300
critic_iterations = 9
weight_clip = 0.01

trainDataloader = DataLoader(TensorDataset(train_x_slide, train_y_gan), batch_size = batch_size, shuffle = False)

modelG = Generator(44).to(device)
modelD = Discriminator().to(device)

optimizerG = torch.optim.Adam(modelG.parameters(), lr = learning_rate, betas = (0.0, 0.9), weight_decay = 1e-3)
optimizerD = torch.optim.Adam(modelD.parameters(), lr = learning_rate, betas = (0.0, 0.9), weight_decay = 1e-3)

histG = np.zeros(num_epochs)
histD = np.zeros(num_epochs)
count = 0
for epoch in range(num_epochs):
    loss_G = []
    loss_D = []
    for (x, y) in trainDataloader:
        x = x.to(device)
        y = y.to(device)

        fake_data = modelG(x)
        fake_data = torch.cat([y[:, :3, :], fake_data.reshape(-1, 1, 1)], axis = 1)
        critic_real = modelD(y)
        critic_fake = modelD(fake_data)
        lossD = -(torch.mean(critic_real) - torch.mean(critic_fake))
        modelD.zero_grad()
        lossD.backward(retain_graph = True)
        optimizerD.step()

        output_fake = modelD(fake_data)
        lossG = -torch.mean(output_fake)
        modelG.zero_grad()
        lossG.backward()
        optimizerG.step()

        loss_D.append(lossD.item())
        loss_G.append(lossG.item()) 
    histG[epoch] = sum(loss_G) 
    histD[epoch] = sum(loss_D)    
    print(f'[{epoch+1}/{num_epochs}] LossD: {sum(loss_D)} LossG:{sum(loss_G)}')
    
plt.figure(figsize = (12, 6))
plt.plot(histG, color = 'blue', label = 'Generator Loss')
plt.plot(histD, color = 'black', label = 'Discriminator Loss')
plt.title('WGAN-GP Loss')
plt.xlabel('Days')
plt.legend(loc = 'upper right')



modelG.eval()
pred_y_train = modelG(train_x_slide.to(device))
pred_y_test = modelG(test_x_slide.to(device))

y_train_true = y_scaler.inverse_transform(train_y_slide)
y_train_pred = y_scaler.inverse_transform(pred_y_train.cpu().detach().numpy())

y_test_true = y_scaler.inverse_transform(test_y_slide)
y_test_pred = y_scaler.inverse_transform(pred_y_test.cpu().detach().numpy())


plt.figure(figsize=(12, 8))
plt.plot(y_train_true, color = 'black', label = 'Acutal Price')
plt.plot(y_train_pred, color = 'blue', label = 'Predict Price')
plt.title('WGAN-GP prediction training dataset')
plt.ylabel('TWD')
plt.xlabel('Days')
plt.legend(loc = 'upper right')

MSE = mean_squared_error(y_train_true, y_train_pred)
RMSE = math.sqrt(MSE)
print(f'Training dataset RMSE:{RMSE}')



plt.figure(figsize=(12, 8))
plt.plot(y_test_true, color = 'black', label = 'Acutal Price')
plt.plot(y_test_pred, color = 'blue', label = 'Predict Price')
plt.title('WGAN-GP prediction testing dataset')
plt.ylabel('TWD')
plt.xlabel('Days')
plt.legend(loc = 'upper right')

MSE = mean_squared_error(y_test_true, y_test_pred)
RMSE = math.sqrt(MSE)
print(f'Training dataset RMSE:{RMSE}')


data_df  = pd.DataFrame(y_test_pred, columns = ["Close"])



#%% 라벨링

#라벨링 리스트 생성 함수(input : (data, 만들 행 갯수))

label_shift = data_df['Close'].diff().shift(-1)
label_shift_list = []

for i in range(len(label_shift)):
    if label_shift[i] > 0 :
        label_shift_list.append(1)
    else:
        label_shift_list.append(0)

len(label_shift_list)

data_df['label'] =  label_shift_list


#실제종가 + 예측한 종가 라벨링
test_data_pred = pd.DataFrame(data[1174:]["Close"], columns = ["Close"])
test_data_pred['label'] = label_shift_list



test_data_rtn = test_data_pred['Close'].pct_change() * 100


test_data_pred["rtn"] = test_data_rtn  



test_data_pred['position'] = None


for i in range(0, len(test_data_pred)):
        try:
            if test_data_pred['label'][i]+test_data_pred['label'][i+1]==0:
                test_data_pred['position'][i+1]='no action'
            elif test_data_pred['label'][i]+test_data_pred['label'][i+1]==2:
                test_data_pred['position'][i+1]='holding'
            elif test_data_pred['label'][i] > test_data_pred['label'][i+1]:
                test_data_pred['position'][i+1]='sell'
            else:
                test_data_pred['position'][i+1]='buy'
        except:
            pass


test_data_pred['position'].iloc[-1] ="sell"

if test_data_pred['position'][0] == None:
    if test_data_pred['label'][0] ==   1:
        test_data_pred['position'][0] = "buy"




test_data_pred["new_rtn"] = 0.0

for i in range(len(test_data_pred)):
    if test_data_pred["position"][i] == "buy" or test_data_pred["position"][i] == "no action" :
        test_data_pred["new_rtn"][i] = 0
    else : 
        test_data_pred["new_rtn"][i] = test_data_pred['rtn'][i] 
         
        
test_data_pred["new_rtn"].sum()


#diff
test_data_pred["diff"] = test_data_pred["Close"].diff()


#거래 횟수
test_data_pred['new_diff'] = 0.0

for i in range(len(test_data_pred)):
    if test_data_pred["position"][i] == "buy" or test_data_pred["position"][i] == "no action" :
        test_data_pred["new_diff"][i] = 0
    else : 
        test_data_pred["new_diff"][i] = test_data_pred['diff'][i] 
         

#test_data_pred = test_data_pred[1:]
#test_data_pred = test_data_pred.reset_index(drop=True)

#sell 기준 합치기

test_data_pred["diff_sum"] = 0.0

a = []

for i in range(1, len(test_data_pred)):
    if test_data_pred["position"][i] == "holding" :
            a.append(test_data_pred["new_diff"][i])
    elif test_data_pred["position"][i] == "sell"  :
        a.append(test_data_pred["new_diff"][i])
        test_data_pred["diff_sum"][i] = sum(a)
        a=[]


#rmfovm


test_data_pred["index"] = test_data_pred.index

buy_index =[]
sell_index = []




for i in range(len(test_data_pred)):
    if test_data_pred["position"][i] == "buy":
        buy_index.append(test_data_pred['index'][i])
    elif test_data_pred['position'][i] == "sell" :       
        sell_index.append(test_data_pred['index'][i])

len(sell_index)



#win rate

win_rate_count = 0

for i in range(len(test_data_pred)): 
    if test_data_pred["diff_sum"][i] > 0:
        win_rate_count +=1        
    

#win_rate_count / len(sell_index)



#payoff_rate
gain_list = []
loss_list = []
all_list = []

for i in range(len(test_data_pred)):
    if test_data_pred["diff_sum"][i] > 0:
        gain_list.append(test_data_pred["diff_sum"][i])
        all_list.append(test_data_pred["diff_sum"][i])
    elif test_data_pred["diff_sum"][i] <0:
        loss_list.append(test_data_pred["diff_sum"][i])
        all_list.append(test_data_pred["diff_sum"][i])
        
np.mean(loss_list) / np.mean(gain_list)


#profit factor

#sum(loss_list) / sum(gain_list)

#average gain & loss
np.mean(gain_list)
np.mean(loss_list)

#총손실
sum(loss_list)

#총수익
sum(gain_list)


#수익률 붙이기





#지표들
print("거래횟수 : ", len(sell_index))
print("winning ratio :", win_rate_count / len(sell_index))
print("평균 수익 :", np.mean(gain_list))
print("평균 손실 :", np.mean(loss_list))
print("payoff ratio :", abs(np.mean(gain_list)/ np.mean(loss_list)))
print("총수익:", sum(gain_list))
print("총손실:", sum(loss_list))
print("profit factor:", abs(sum(gain_list)/  sum(loss_list)))



