import h5py
import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt



h5 = h5py.File('water_level.h5','w')
SP_csv = pd.read_csv(glob('station_8726520.csv')[0])


times=np.array(SP_csv["Time (Timezone lst)"],dtype='datetime64[ms]')

h5["t"]=times.astype('float')

h5.create_group("SP")
h5["SP"]["p"]=SP_csv.iloc[:,2].to_numpy(dtype=np.float32)
SP_csv.iloc[:,1] = SP_csv.iloc[:,1].replace('-', np.nan)
h5["SP"]["v"] =SP_csv.iloc[:,1].to_numpy(dtype=np.float32)

h5.close()
print('done')


hh = h5py.File('water_level.h5')

ttt = hh['t'][:].astype('datetime64[ms]')
wll = hh['SP']['v'][:]
tide = hh['SP']['p'][:]
surge = wll - tide
plt.plot(ttt,surge)

np.max(surge)
np.min(surge)

surge.shape