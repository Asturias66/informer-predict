import math

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
plt.ticklabel_format(style='plain')
import torch
import torch.nn as nn

df_raw = pd.read_csv('../data/601398.csv',index_col='date')

print(df_raw)

# 绘制收盘价格走势图
# df_raw[['close']].plot()
# plt.ylabel("stock_price")
# plt.title("aaxj ETFs")
# plt.show()

sel_col = ['open', 'high', 'low', 'close','volume']

# 数据缩放
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = StandardScaler()
# scaler = MinMaxScaler(feature_range=(-1, 1))
for col in sel_col:                           # 这里不能进行统一进行缩放，因为fit_transform返回值是numpy类型
    df_raw[col] = scaler.fit_transform(df_raw[col].values.reshape(-1,1))

# 将下一日的收盘价作为本日的标签
df_raw['target'] = df_raw['close'].shift(-1)

print(df_raw.head())

df_raw.dropna()                      # 使用了shift函数，在最后必然是有缺失值的，这里去掉缺失值所在行
df_raw = df_raw.astype(np.float32)  # 修改数据类型

print(df_raw)



input_dim = 5      # 数据的特征数
hidden_dim = 32    # 隐藏层的神经元个数
num_layers = 2     # LSTM的层数
output_dim = 1     # 预测值的特征数
                   #（这是预测股票价格，所以这里特征数是1，如果预测一个单词，那么这里是one-hot向量的编码长度）
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer 在LSTM后再加一个全连接层，因为是回归问题，所以不能在线性层后加激活函数
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 这里x.size(0)就是batch_size

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out)

        return out


# 创建两个列表，用来存储数据的特征和标签
data_feat, data_target = [],[]

# 设每条数据序列有60组数据
seq = 60

for index in range(len(df_raw) - seq):
    # 构建特征集
    data_feat.append(df_raw[['open', 'high', 'low', 'close','volume']][index: index + seq].values)
    # 构建target集
    data_target.append(df_raw['target'][index:index + seq])

# 将特征集和标签集整理成numpy数组
data_feat = np.array(data_feat)
data_target = np.array(data_target)

# 这里按照8:2的比例划分训练集和测试集
test_set_size = int(np.round(0.2*df_raw.shape[0]))  # np.round(1)是四舍五入，
train_size = data_feat.shape[0] - (test_set_size)
print(test_set_size)  # 输出测试集大小
print(train_size)     # 输出训练集大小

trainX = torch.from_numpy(data_feat[:train_size].reshape(-1,seq,5)).type(torch.Tensor)
# 这里第一个维度自动确定，我们认为其为batch_size，因为在LSTM类的定义中，设置了batch_first=True
testX  = torch.from_numpy(data_feat[train_size:].reshape(-1,seq,5)).type(torch.Tensor)
trainY = torch.from_numpy(data_target[:train_size].reshape(-1,seq,1)).type(torch.Tensor)
testY  = torch.from_numpy(data_target[train_size:].reshape(-1,seq,1)).type(torch.Tensor)

print('x_train.shape = ',trainX.shape)
print('y_train.shape = ',trainY.shape)
print('x_test.shape = ',testX.shape)
print('y_test.shape = ',testY.shape)

batch_size= 64
train = torch.utils.data.TensorDataset(trainX,trainY)
test = torch.utils.data.TensorDataset(testX,testY)
train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=batch_size,
                                          shuffle=False)

# 实例化模型
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

# 定义优化器和损失函数
optimiser = torch.optim.Adam(model.parameters(), lr=0.01) # 使用Adam优化算法
loss_fn = torch.nn.MSELoss(size_average=True)             # 使用均方差作为损失函数
loss_mae = torch.nn.L1Loss()

# 设定数据遍历次数
num_epochs = 10

# 打印模型结构
print(model)

# 打印模型各层的参数尺寸
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# train model
hist = np.zeros(num_epochs)
iter=0
for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    # model.hidden = model.init_hidden()

    # Forward pass
    # y_train_pred = model(trainX)

    for i, (batch_x, batch_y) in enumerate(train_loader):
        y_train_pred = model(batch_x)
        optimiser.zero_grad()  # 将每次传播时的梯度累积清除
        # print(outputs.shape, batch_y.shape)
        loss = loss_fn(y_train_pred, batch_y)  # 计算损失
        loss.backward()  # 反向传播
        optimiser.step()
        iter += 1
        if iter % 100 == 0:
            print("iter: %d, loss: %1.5f" % (iter, loss.item()))

    # loss = loss_fn(y_train_pred, trainY)
    # if t % 10 == 0 and t != 0:  # 每训练十次，打印一次均方差
    #     print("Epoch ", t, "MSE: ", loss.item())
    # hist[t] = loss.item()
    #
    # # Zero out gradient, else they will accumulate between epochs 将梯度归零
    # optimiser.zero_grad()
    #
    # # Backward pass
    # loss.backward()
    #
    # # Update parameters
    # optimiser.step()

# 计算训练得到的模型在训练集上的均方差
y_train_pred = model(trainX)
print(y_train_pred)
print(trainY)
print('mse:' + str(loss_fn(y_train_pred, trainY).item()))
print('mae:' + str(loss_mae(y_train_pred, trainY).item()))
print('rmse:' + str(math.sqrt(loss_fn(y_train_pred, trainY).item())))

# make predictions
y_test_pred = model(testX)
print('mse:' + str(loss_fn(y_test_pred, testY).item()))
print('mae:' + str(loss_mae(y_test_pred, testY).item()))
print('rmse:' + str(math.sqrt(loss_fn(y_test_pred, testY).item())))
# print(loss_fn(y_test_pred, testY).item())

# 无论是真实值，还是模型的输出值，它们的维度均为（batch_size, seq, 1），seq=20
# 我们的目的是用前20天的数据预测今天的股价，所以我们只需要每个数据序列中第20天的标签即可
# 因为前面用了使用DataFrame中shift方法，所以第20天的标签，实际上就是第21天的股价
pred_value = y_train_pred.detach().numpy()[:,-1,0]
true_value = trainY.detach().numpy()[:,-1,0]

# plt.plot(pred_value, label="Preds")    # 预测值
# plt.plot(true_value, label="Data")    # 真实值
# plt.legend()
# plt.show()

# 纵坐标还有负的，因为前面进行缩放，现在让数据还原成原来的大小
# invert predictions
pred_value = scaler.inverse_transform(pred_value.reshape(-1, 1))
true_value = scaler.inverse_transform(true_value.reshape(-1, 1))

# plt.plot(pred_value, label="Preds")    # 预测值
# plt.plot(true_value, label="Data")    # 真实值
# plt.legend()
# plt.show()

#测试集
pred_value = y_test_pred.detach().numpy()[:,-1,0]
true_value = testY.detach().numpy()[:,-1,0]


plt.plot(pred_value, label="Preds")    # 预测值
plt.plot(true_value, label="Data")    # 真实值
plt.legend()
plt.show()

pred_value = scaler.inverse_transform(pred_value.reshape(-1, 1))
true_value = scaler.inverse_transform(true_value.reshape(-1, 1))

np.around(pred_value,2)
np.around(true_value,2)



plt.plot(pred_value, label="Preds")    # 预测值
plt.plot(true_value, label="Data")    # 真实值
plt.legend()
plt.show()



