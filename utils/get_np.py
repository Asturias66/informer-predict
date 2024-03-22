import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# preds = np.load('../results/' + 'informer_601398_ftMS_sl20_ll10_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0' + '/pred.npy')
# trues = np.load('../results/' + 'informer_601398_ftMS_sl20_ll10_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0' + '/true.npy')

# plt.figure()
# plt.plot(trues[:,0,-1].reshape(-1), label='GroundTruth')
# plt.plot(preds[:,0,-1].reshape(-1), label='Prediction')
# plt.legend()
# # plt.show()
# plt.savefig("../result_picture/"+ "informer" + "_fic.png")
# metrics = np.load('../results/' + 'informer_601398_ftMS_sl20_ll10_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0' + '/metrics.npy')
from utils.tools import StandardScaler

setting = 'informer_601398_ftMS_sl60_ll30_pl10_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1'

preds = np.load('../results/' + setting + '/pred.npy')
trues = np.load('../results/' + setting + '/true.npy')
real = np.load('../results/' + setting + '/real_prediction.npy')

print(preds)
# print(trues)

print(preds.shape)
print(preds[:, 0, -1].reshape(-1))

plt.figure()
plt.plot(trues[:, 0, -1].reshape(-1), label='GroundTruth')
plt.plot(preds[:, 0, -1].reshape(-1), label='Prediction')
plt.legend()
plt.savefig(f"./result_picture/{setting}_test.png")
# per_10_list = []
# for i in range(0,len(preds)):
#     pre_10 = preds[i, :, -1]
#     print(pre_10)
#     per_10_list.append(pre_10)
#
# predictions_10 = pd.DataFrame(per_10_list)
# predictions_10.to_csv('../predictions_10.csv')

# print(preds.shape)

# print(preds[:, 0, -1].reshape(-1))
#
# preds = preds[:, 0, -1].reshape(-1)
#
# predictions = pd.DataFrame(preds)
# predictions.to_csv('../predictions.csv')

# scaler = StandardScaler()
#
# print(preds)
# print(trues)
#
# inverse_preds = scaler.inverse_transform(preds)
# inverse_trues = scaler.inverse_transform(trues)
#
# plt.figure()
# plt.plot(inverse_trues[:, 0, -1].reshape(-1), label='GroundTruth')
# plt.plot(inverse_preds[:, 0, -1].reshape(-1), label='Prediction')
# plt.legend()
# plt.savefig(f"./result_picture/{setting}_inverse_fic.png")

# print(metrics)