import numpy as np, pandas as pd, sys, os
import sys
D=sys.argv[1].rstrip('/')+'/'
files=[D+'NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_1.txt']+[D+f'NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_{k}.txt' for k in range(1,10)]
for day,f in enumerate(files,1):
    a=pd.read_csv(f,sep=r'\s+',header=None,engine='c',dtype=np.float32).values
    print(day,a.shape,np.unique(a[144:149]).tolist()[:6],flush=True)
    np.save(sys.argv[2]+f'/day{day:02d}.npy',a)
