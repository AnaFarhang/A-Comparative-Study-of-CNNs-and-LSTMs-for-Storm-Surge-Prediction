import os
import xarray as xr
import gcsfs
import h5py


lon_west=-85
lon_east=-82
lat_south=26
lat_north=29
cs="EPSG:4326"

lon_name='lon'
lat_name='lat'
t_name='time'
u_name='U_GRD_L103'
v_name='V_GRD_L103'
p_name='PRES_L1'


U1 = xr.open_dataset("wind_u_combined.nc")
P1 = xr.open_dataset("wind_p_combined.nc")

u1 = U1[u_name].values
v1 = U1[v_name].values
p1 = P1[p_name].values

lat1=U1[lat_name].values
lon1=U1[lon_name].values


ltemp=lon1[lon1>180]
ltemp=ltemp-360
lon1[lon1>180]=ltemp


uu = u1
vv = v1
pp = p1[1:,:]
tt =U1[t_name].values
ttp=P1[t_name].values[1:]


tt = tt.astype('datetime64[ms]')

h5=h5py.File('wind.h5','w')

h5['t']=tt.astype('float')
h5['u']=uu
h5['v']=vv
h5['p']=pp
h5['lon']=lon1
h5['lat']=lat1
h5.close()
