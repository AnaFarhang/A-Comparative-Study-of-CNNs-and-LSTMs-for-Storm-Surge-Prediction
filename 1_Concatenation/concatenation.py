import xarray as xr
import os


print ("Cancatating files...")
ds = xr.open_mfdataset('pressfc*')
ds.to_netcdf("wind_p_combined.nc")
ds = xr.open_mfdataset('wnd10m*')
ds.to_netcdf("wind_u_combined.nc")