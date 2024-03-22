import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_raw = pd.read_csv(os.path.join('../data/',
                                          '601398.csv'))
df_raw.plot(x='date',y='close',figsize=(12, 6))
# df_raw.iloc[:,-2].plot(figsize=(12, 6))
plt.legend()
plt.show()

# plt.figure()
# plt.plot(df_raw.iloc[:,0],df_raw.iloc[:,-1], label='close')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(trues[:,0,-1].reshape(-1), label='GroundTruth')
# plt.plot(preds[:,0,-1].reshape(-1), label='Prediction')
# plt.legend()
# # plt.show()
# plt.savefig("../result_picture/"+ "informer" + "_fic.png")
