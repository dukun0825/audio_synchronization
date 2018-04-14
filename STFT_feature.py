import os
import librosa
import librosa.display
import types
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
path = 'E:/workcode/python/synchronization/footbal_unorganized'
files = os.listdir(path)
file_name = []
L = 6
order = 0
for file in files:
    if not os.path.isdir(file):
        # print(file)
        file_name.append(file)
        path_file = os.path.join(path, file)
        # print(path_file)
        y, sr = librosa.load(path_file, sr=8000)
        #print('y=', y, 'sr=', sr)
        D = librosa.stft(y, hop_length=512)  # 短时傅里叶变换
        print(file, D.shape)
        S = np.abs(D)
        feature = S[128:257]  # 500到1000hz
        sum_col = np.sum(feature, axis=0)  # 整列求和
        sum_0 = np.zeros([1, L])
        sum_zero = np.append(sum_0, sum_col)
        sum_zero = np.append(sum_zero, sum_0)  # 前后补L个0
        # 算法生成的特征,默认第一个时间帧特征为1
        sum_row = np.empty_like(sum_col)
        sum_row[0] = 1
        col = sum_col.shape[0]
        x = sum(sum_zero[0:0 + L + L])

        for i in range(col - 1):
            y = sum(sum_zero[i + 1:i + L + L + 1])
            if y >= x:
                sum_row[i + 1] = 1
            else:
                sum_row[i + 1] = -1
            x = y

        frame = pd.DataFrame(sum_row)
        outputpath='{0}{1}{2}'.format("E:/workcode/python/synchronization/output/feature/",order,".csv")
        print(outputpath)
        frame.to_csv(outputpath)
        order = order+1