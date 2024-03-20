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
# 1- Download multiple MODIS Terra and Aqua netcdf files
# 2- Select time date range 
# 3- Combine Terra and Aqua data (mean)
# 4- Select a region
# 5- Yearly mean
# 6- Convert 3D xarray into 2D xarray
# 7- COnvert xarray into dataframe
# 8- Aplly MK trends from pymannkendall
# 9- go back into xarray
# 10- Create a netcdf file with the MK calculated trend, signif, p, std_error
###################

def TREND(rname,vname,input_dir,output_dir):
    # CERL, COTL, CTP. CTT, CWPL, CF
    # Open sevral Netcdf files 
    modis_y_dir= '/monthly/scrubbed.MY' + str(vname) + '.*.nc'
    modis_y_fn = "".join([input_dir,rname,modis_y_dir])
    modis_y_ds = xr.open_mfdataset(modis_y_fn)
    
    modis_o_dir= '/monthly/scrubbed.MO' + str(vname) + '.*.nc'
    modis_o_fn = "".join([input_dir,rname,modis_o_dir])
    modis_o_ds = xr.open_mfdataset(modis_o_fn)
    
    # Select time date range
    modis_y_ds_sel = modis_y_ds.sel(time=slice('2002-03-01','2022-03-01'))
    modis_o_ds_sel = modis_o_ds.sel(time=slice('2002-03-01','2022-03-01'))
    
    # Select variable
    modis_y_var = modis_y_ds_sel["MY" + str(vname)]
    modis_o_var = modis_o_ds_sel["MO" + str(vname)]
    
    # Merge the Terra and Aqua modis DataArray
    modis_yo_var = xr.merge([modis_y_var,modis_o_var])
    # Mean over the Terra and Aqua DataArray
    mean = modis_yo_var.to_array(dim='new').mean('new',skipna=True)
    # Assign the new variable to the DataArray
    modis_yom_var = modis_yo_var.assign(new_var=mean)
    # Select the new variable
    modis_monthly_var = modis_yom_var["new_var"]
    
    # Select subdomain
    # Africa
    domain=['Africa']
    lat_bnd = [-30, 10]
    lon_bnd = [-25, 35]

    modis_monthly_afr = modis_monthly_var.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)

    # yearly mean each year
    modis_jaso_var = modis_monthly_afr.sel(time=modis_monthly_afr.time.dt.month.isin([7,8,9]))
    # Check the missing data in the bounded area
    nan_counts = modis_jaso_var.isnull().sum(dim='time')
    # Mask where nan_count > 5
    modis_jaso_var = modis_jaso_var.where(nan_counts <= 5)
    modis_year_mean = modis_jaso_var.groupby('time.year').mean('time',skipna=True)
    # total mean
    modis_year_mean_mean = modis_year_mean.mean(dim='year',skipna=False)
    # Put var into %
    modis_year_perc = ((modis_year_mean-modis_year_mean_mean)/modis_year_mean_mean)*100.    
    
    # Convert 3D dimension (lon, lat, year)  into 2D dimension (lonxlat, year)
    modis_year_mean_2d = modis_year_mean.values.reshape((len(modis_year_mean.year), -1))
    modis_year_perc_2d = modis_year_perc.values.reshape((len(modis_year_perc.year), -1))
    
    #convert np.array into dataframe with column for each reff_latxlon, and index as time (time, latxlon)
    modis_year_mean_df = pd.DataFrame(modis_year_mean_2d,columns=np.arange(0,len(modis_year_mean_2d[0,:])),index = modis_year_mean.year)
    modis_year_perc_df = pd.DataFrame(modis_year_perc_2d,columns=np.arange(0,len(modis_year_perc_2d[0,:])),index = modis_year_perc.year)
    
    def MKTFPW(ds):
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.trend_free_pre_whitening_modification_test(ds)     
        return slope, p
    
    MK_trend_year_mean = modis_year_mean_df.apply(lambda column: MKTFPW(column) if column.notnull().all() else [np.NaN,np.NaN], axis=0)
    MK_trend_year_perc = modis_year_perc_df.apply(lambda column: MKTFPW(column) if column.notnull().all() else [np.NaN,np.NaN], axis=0)
    
    # Convert dataframe into xarray
    mk_slope_year_mean = xr.DataArray(
        name= "slope",
        data= MK_trend_year_mean.values.reshape((2,) + modis_year_mean.data.shape[1:])[0],
        coords=(
            ("y", modis_year_mean.lat.data),
            ("x", modis_year_mean.lon.data),
        ),
    )
    
    mk_p_year_mean = xr.DataArray(
        name= "p", 
        data= MK_trend_year_mean.values.reshape((2,) + modis_year_mean.data.shape[1:])[1],
        coords=(
            ("y", modis_year_mean.lat.data),
            ("x", modis_year_mean.lon.data),
        ),
    )
    
    mk_slope_year_perc = xr.DataArray(
        name= "slope",
        data= MK_trend_year_perc.values.reshape((2,) + modis_year_perc.data.shape[1:])[0],
        coords=(
            ("y", modis_year_perc.lat.data),
            ("x", modis_year_perc.lon.data),
        ),
    )
    
    mk_p_year_perc = xr.DataArray(
        name= "p",
        data= MK_trend_year_perc.values.reshape((2,) + modis_year_perc.data.shape[1:])[1],
        coords=(
            ("y", modis_year_perc.lat.data),
            ("x", modis_year_perc.lon.data),
        ),
    )
    
    MK_trends_year_mean = xr.merge([mk_slope_year_mean,mk_p_year_mean])
    MK_trends_year_perc = xr.merge([mk_slope_year_perc,mk_p_year_perc])
    
    # Create netcdf file
    new_filename_1 = output_dir + 'MYO' + str(vname) + '_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc'
    print ('saving to ', new_filename_1)
    MK_trends_year_mean.to_netcdf(path=new_filename_1)
    MK_trends_year_mean.close()
    
    new_filename_2 = output_dir + 'MYO' + str(vname) + '_TrendMod_percmean_AFR_1deg_JAS_2002_2021.nc'
    print ('saving to ', new_filename_2)
    MK_trends_year_perc.to_netcdf(path=new_filename_2)
    MK_trends_year_perc.close()
    
    print ('finished saving')

dir_in = 'path-to-modis-input'
dir_out = 'path-to-output-trend-repertory'
var = {'AOD':'D08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean'}

for vname in var:
    TREND(vname,var[vname],dir_in,dir_out)
