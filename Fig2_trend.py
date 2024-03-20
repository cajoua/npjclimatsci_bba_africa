#data manipulation
import numpy as np
import xarray as xr
import pandas as pd

#data statistics
import pymannkendall as mk

#data reggrid
import xesmf as xe

#warning
import warnings
warnings.filterwarnings("ignore")

###################
# 1- Download multiple CERES netcdf files
# 2- Select time date range 
# 3- Change longitude 0-360 to -180-180
# 4- Deseasonalizing and yearly Mean
# 5- Reggrid to 5x5 deg
# 6- Convert 3D xarray into 2D xarray
# 7- Convert xarray into dataframe
# 8- Aplly MK trends from pymannkendall
# 9- go back into xarray
# 10- Create a netcdf file with the MK calculated trend, signif, p, std_error
###################

# Open sevral Netcdf files 
ceres_dir='path-to-ceres-input'
ceres_ds = xr.open_mfdataset(ceres_dir)

# Select time date range
ceres_ds_sel = ceres_ds.sel(time=slice('2002-03-01','2022-03-01'))

# Change longitude from 0-360 to -180-180
ceres_ds_sel.coords['lon'] = (ceres_ds_sel.coords['lon'].values + 180) % 360 - 180
ceres_ds_sel = ceres_ds_sel.sortby(ceres_ds_sel.lon)

# Select subdomain
# Africa
domain=['Africa']
lat_bnd = [-30, 10] 
lon_bnd = [-25, 35] 

ceres_ds_afr = ceres_ds_sel.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)

# Select variable
sw_all_mon_afr  = ceres_ds_afr["toa_sw_all_mon"]
sw_clr_mon_afr  = ceres_ds_afr["toa_sw_clr_c_mon"]

# yearly mean each year
sw_all_mon = sw_all_mon_afr.sel(time=sw_all_mon_afr.time.dt.month.isin([7,8,9]))
sw_all_year_mean  = sw_all_mon.groupby('time.year').mean('time',skipna=False)
sw_clr_mon = sw_clr_mon_afr.sel(time=sw_clr_mon_afr.time.dt.month.isin([7,8,9]))
sw_clr_year_mean  = sw_clr_mon.groupby('time.year').mean('time',skipna=False)

# Convert 3D dimension (lon, lat, year)  into 2D dimension (lonxlat, year)
sw_all_year_2d  = sw_all_year_mean.values.reshape((len(sw_all_year_mean.year), -1))
sw_clr_year_2d  = sw_clr_year_mean.values.reshape((len(sw_clr_year_mean.year), -1))

#convert np.array into dataframe with column for each reff_latxlon, and index as time (time, latxlon)
sw_all_year_df  = pd.DataFrame(sw_all_year_2d,columns=np.arange(0,len(sw_all_year_2d[0,:])),index = sw_all_year_mean.year)
sw_clr_year_df  = pd.DataFrame(sw_clr_year_2d,columns=np.arange(0,len(sw_clr_year_2d[0,:])),index = sw_clr_year_mean.year)

def MKTFPW(ds):
    trend, h, p, z, Tau, s, var_s, slope, intercept = mk.trend_free_pre_whitening_modification_test(ds)     
    return slope, p

MK_trend_sw_all  = sw_all_year_df.apply(MKTFPW)
MK_trend_sw_clr  = sw_clr_year_df.apply(MKTFPW)

# Convert dataframe into xarray

mk_slope_sw_all = xr.DataArray(
    name= "slope",
    data= MK_trend_sw_all.values.reshape((2,) + sw_all_year_mean.data.shape[1:])[0],
    coords=(
        ("y", sw_all_year_mean.lat.data),
        ("x", sw_all_year_mean.lon.data),
    ),
)

mk_p_sw_all = xr.DataArray(
    name= "p",
    data= MK_trend_sw_all.values.reshape((2,) + sw_all_year_mean.data.shape[1:])[1],
    coords=(
        ("y", sw_all_year_mean.lat.data),
        ("x", sw_all_year_mean.lon.data),
    ),
)


mk_slope_sw_clr = xr.DataArray(
    name= "slope",
    data= MK_trend_sw_clr.values.reshape((2,) + sw_clr_year_mean.data.shape[1:])[0],
    coords=(
        ("y", sw_clr_year_mean.lat.data),
        ("x", sw_clr_year_mean.lon.data),
    ),
)

mk_p_sw_clr = xr.DataArray(
    name= "p",
    data= MK_trend_sw_clr.values.reshape((2,) + sw_clr_year_mean.data.shape[1:])[1],
    coords=(
        ("y", sw_clr_year_mean.lat.data),
        ("x", sw_clr_year_mean.lon.data),
    ),
)


MK_trends_sw_all = xr.merge([mk_slope_sw_all,mk_p_sw_all])
MK_trends_sw_clr = xr.merge([mk_slope_sw_clr,mk_p_sw_clr])

# Create netcdf file
new_filename_1 = './trend_article/CERES_TOA_SW_ALL_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc'
print ('saving to ', new_filename_1)
MK_trends_sw_all.to_netcdf(path=new_filename_1)
MK_trends_sw_all.close()
new_filename_4 = './trend_article/CERES_TOA_SW_CLR_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc'
print ('saving to ', new_filename_4)
MK_trends_sw_clr.to_netcdf(path=new_filename_4)
MK_trends_sw_clr.close()
print ('finished saving')
