import os
import importlib
import pysrisk03 as jl
import pandas as pd
from datetime import datetime as dt
from multiprocessing import Pool, cpu_count
#%%
h5 = 'Result.h5'
d5 = '.\\Data\\Data.h5'
tic = dt.now()
if __name__ == '__main__':
    d1 = dt(1993,12,31); d2 = dt(2017,4,30)
    days = pd.bdate_range(start=d1,end=d2,freq='m')
    d = days
    with Pool(cpu_count) as pool:
        slist = pool.map(jl.getSRisk,d)
toc = dt.now()
time_used = toc-tic
print(time_used)