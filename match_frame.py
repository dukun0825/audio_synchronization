import os
import numpy as np
import pandas as pd
import types

path = 'E:/workcode/python/synchronization/footbal_unorganized'
files = os.listdir(path)
file_name = []
file_dict={}
i=0
#获取文件路径及文件名列表
for file in files:
    if not os.path.isdir(file):
        print(file)
        file_name.append(file)
        file_dict[i]=file
        i=i+1
#print(i)
match_strength_list=[]
frame_list=[]
match_name_list=[]

for k in range(i-1):
    path_feature1 = '{0}{1}{2}'.format("E:/workcode/python/synchronization/output/feature/",k,".csv")
    print(path_feature1)
    for j in range(k+1,i):
        #value_list=[]
        path_feature2 = '{0}{1}{2}'.format("E:/workcode/python/synchronization/output/feature/",j,".csv")
        print(path_feature2)
        x=pd.read_csv(path_feature1)
        t1=x.iloc[:,1].values
        y=pd.read_csv(path_feature2)
        t2=y.iloc[:,1].values
        out=np.correlate(t1, t2, 'full')
        #print(out)
        match_strength, frame = out.max(), out.argmax()
        key_name=file_dict[k]+' '+file_dict[j]
        match_strength_list.append(match_strength)
        frame_list.append(frame)
        match_name_list.append(key_name)
        #output_match[key_name] = value_list
        #print(file_dict[k],file_dict[j],match_strength, frame)

output_match = {'match_strength':match_strength_list,'frame':frame_list}#输出,匹配系数，时移帧数
output = pd.DataFrame(output_match,index=match_name_list)
output.to_csv('E:/workcode/python/synchronization/output/result.csv')



