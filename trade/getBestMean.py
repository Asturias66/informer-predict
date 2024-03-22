import datetime
import pandas as pd
import numpy as np


timebegin = datetime.datetime.now()  # 记录程序开始运行时间
fileName = '../data/601398.csv'  # 设置输入文件名
sheetName = '000001'  # 设置输入sheet名
df = pd.read_csv(fileName)  # 读入数据
DateBS = df['date'].values  # 获取交易日期
OpenP = df['open'].values  # 获取开盘价
# CloseP = df['前复权收盘价'].values  # 获取前复权收盘价
CloseP = df['close'].values  # 获取收盘价
n_records = df.shape[0]  # 记录总数量


# 定义计算均线函数，data是原始数据，ma_range是待计算的均线范围
def get_ma(data, ma_range):
    n_data = len(data)
    n_ma = len(ma_range)
    ma = np.zeros((n_data, n_ma))  # ma用于保存计算结果，保存均线的矩阵
    # 计算均线
    for j in range(n_ma):
        for i in range(ma_range[j] - 1, n_data):
            ma[i, j] = data[(i - ma_range[j] + 1):(i + 1)].mean()
    return ma

# 判断表格中多少行至多少行为第几年
def findYears(DateBS):
    year = []
    for i in range(1, n_records):
        dataStr0 = str(DateBS[i - 1]).split('/')
        dataStr1 = str(DateBS[i]).split('/')
        if dataStr0[0] != dataStr1[0]:
            year.append(i)
    year.append(n_records - 1)
    return year

# 寻找最佳均线
def bestMean(day_ma, maRange, sum_result, ibegin, iend, money0, netAsset0):
    num_bs = 0  # 买卖次数
    nma = len(maRange)
    for j in range(nma):
        position = 0
        flag_bs = 0
        money = money0
        record_bs = [['序号', '日期', '买卖', '价格', '证券数量', '资金余额', '净资产', '平仓盈利'],
                     [1, DateBS[ibegin], 0, 0.0, 0, money, money, 0.0]]
        for i in range(ibegin, iend):
            if (i == ibegin):
                cl_1day = CloseP[i - 1]
                ma_1day = day_ma[i - 1, j]
                if cl_1day > ma_1day:
                    flag_bs = 1
                if cl_1day < ma_1day:
                    flag_bs = -1
            else:
                cl_2day = CloseP[i - 2]
                ma_2day = day_ma[i - 2, j]
                cl_1day = CloseP[i - 1]
                ma_1day = day_ma[i - 1, j]
                if (cl_2day <= ma_2day) and (cl_1day > ma_1day):
                    flag_bs = 1
                if (cl_2day >= ma_2day) and (cl_1day < ma_1day):
                    flag_bs = -1
            if (flag_bs != 0):
                if (flag_bs == 1) and (position == 0):
                    price = OpenP[i]
                    amount = int(money / price / (1 + rateComm))
                    money = money - price * amount * (1 + rateComm)
                    position = 1
                    netAsset = money + price * amount
                    num_bs += 1  # 买卖记录序号加一
                    date_bs = DateBS[i]
                    record_bs.append([num_bs, date_bs, 1, price, amount, money, netAsset, 0.0])
                if (flag_bs == -1) and (position == 1):
                    price = OpenP[i]
                    money = money + price * amount * (1 - rateComm)
                    position = 0
                    amount = 0
                    netAsset = money
                    profit = netAsset - netAsset0
                    netAsset0 = netAsset
                    num_bs += 1  # 买卖记录序号加一
                    date_bs = DateBS[i]
                    record_bs.append([num_bs, date_bs, -1, price, amount, money, netAsset, profit])
                flag_bs = 0
            # 如果持仓不为0，用最后一天的收盘价计算净资产
        if (position != 0):
            price = CloseP[i]
            netAsset9 = money + price * amount
        else:
            netAsset9 = money
        sum_result.iat[j, 1] = netAsset9
    return sum_result

# 1.设置初始值
ibegin = 0  # 循环起始值
iend = 2543  # 循环终止值
ibegin_test = 5365  # 2014年1月2日在数据表中所在位置
iend_test = 7268  # 2021年10月29日在数据表中所在位置
money0 = 100000.0  # 初始资金100w
rateComm = 0.0003  # 手续费费率，万分之五
money = money0  # 资金余额
position = 0  # 持仓情况：0为零仓
netAsset0 = money0  # 上次卖出后的净资产
record_bs = [['序号', '日期', '买卖', '价格', '证券数量', '资金余额', '净资产', '平仓盈亏'],
             [1, DateBS[ibegin], 0, 0.0, 0, money, money, 0.0]]  # 交易记录
record_all = [['日期', '价格', '证券数量', '资金余额', '净资产', '资产涨跌幅', '股价涨跌幅']]  # 记录每一天的资产状况
num_bs = 1  # 买卖次数
amount = 0  # 买入数量
flag_bs = 0  # 买卖标志：1为买入；-1为卖出
maRange = range(120, 241)
day_ma = get_ma(CloseP, maRange)
nma = len(maRange)
sum_result = pd.DataFrame({'MA': maRange, 'netAsset': [0.0] * nma})

# 2.获取数据中的所有年份开始行
years = findYears(DateBS)
print(years)

# 3.滑动窗口获取2014年-2021年每年的最佳均线
# 前22年用于第一次寻找，从0开始，循环years-22次

bestMas = []  # 存储每年的最佳均线

sum_result = bestMean(day_ma, maRange, sum_result, ibegin, iend, money0,
                          netAsset0)  # 计算前22年数据预测出的最佳均线
imax = sum_result['netAsset'].idxmax()
bestMa = sum_result.iloc[imax, 0]
asset = sum_result.iloc[imax, 1]
print("最好均线是：" + str(bestMa) + '值为' + str(asset))

# for i in range(len(years) - 22 - 1):
#     start = years[0 + i]  # 获取开始行的序号
#     end = years[8 + i]  # 获取结束行的序号
#     # print(day_ma)
#     sum_result = bestMean(day_ma, maRange, sum_result, years[0 + i], years[22 + i], money0,
#                           netAsset0)  # 计算前22年数据预测出的最佳均线
#
#     currYear = str(2014 + i)  # 获取计算均线的当年年份
#
#     imax = sum_result['netAsset'].idxmax()
#     bestMa = sum_result.iloc[imax, 0]
#     print(currYear + "年最好均线是：", bestMa)
#     bestMas.append(bestMa)
