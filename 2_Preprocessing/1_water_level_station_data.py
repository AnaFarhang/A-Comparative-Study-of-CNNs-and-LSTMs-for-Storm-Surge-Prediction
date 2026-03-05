import h5py
import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from io import StringIO
import requests

#### Get station waterlevel and tide data

def get_noaa_wl(station,start_date,end_date,interval='6',datum='MSL',units='metric',time_zone='lst',parameter='verified_wl'):
    delta=end_date-start_date
    if interval=='6':
        lim=np.timedelta64(30,'D')
        if parameter == 'verified_wl':
            product='water_level'
        elif parameter == 'prediction':
            product='predictions'
        else:
            print('error')
            exit()
    else:
        lim=np.timedelta64(365,'D')
        if parameter == 'verified_wl':
            product='hourly_height'
        elif parameter == 'prediction':
            product='predictions'
        else:
            print('error')
            exit()

    if delta>lim:
        start=start_date
        end=start_date+lim
    else:
        start=start_date
        end=end_date
    loop=True
    time=[]
    wl=[]
    while loop:
        print('Downloading from ',str(start),' to ',str(end),'....', end='')
        url = f'https://tidesandcurrents.noaa.gov/api/datagetter'
        params = {
            'begin_date': np.datetime_as_string(start, unit='m').replace('T',' ').replace('-',''),
            'end_date': np.datetime_as_string(end, unit='m').replace('T',' ').replace('-',''),    
            'station': station,
            'product': product,
            'interval' : interval,
            'datum' : datum,
            'units': 'metric',
            'time_zone': 'lst',
            'format': 'csv',
            'application': 'DataAPI_Sample',
        }
        response = requests.get(url, params=params)
        csv_data = StringIO(response.text)
        print(' ...done')
        df = pd.read_csv(csv_data)
        time.extend(df['Date Time'].to_list())
        if parameter == 'verified_wl':
            wl.extend(df[' Water Level'].to_list())
        elif parameter == 'prediction':
            wl.extend(df[' Prediction'].to_list())
        else:
            print('error')
            exit()
        if interval=='6':
            start=end+np.timedelta64(6,'m')
        elif interval=='h':
            start=end+np.timedelta64(60,'m')
        end=min(start+lim,end_date)
        if (end<=start):
            loop=False
        out_t = np.array(time,dtype=np.datetime64)
        out_wl = np.array(wl)
    return out_t,out_wl

datum = 'MSL'
station='8726520'
start_time=np.datetime64('2011-01-01T01:00:00')
end_time  =np.datetime64('2024-01-01T00:00:00')
twl,wl=get_noaa_wl(station,start_time,end_time,interval='h',parameter='verified_wl')
ttide,tide=get_noaa_wl(station,start_time,end_time,interval='h',parameter='prediction')
np.savetxt(f'station_{station}.csv', np.column_stack((twl.astype(str),wl,tide)), delimiter=',', header=f'Time (Timezone lst),Verified Water Level,Prediction ', fmt='%s', comments='')

