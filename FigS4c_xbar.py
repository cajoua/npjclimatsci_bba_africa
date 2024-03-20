import xarray as xr
import numpy as np
import warnings; warnings.filterwarnings(action='ignore')
from matplotlib import pyplot as plt


## MODIS
# AOD
# Open sevral Netcdf files 
modis_y_var_dir='path-to-modis-input'
modis_y_var_ds = xr.open_mfdataset(modis_y_var_dir)
modis_o_var_dir='path-to-modis-input'
modis_o_var_ds = xr.open_mfdataset(modis_o_var_dir)
# Select time date range
modis_y_var_ds_sel = modis_y_var_ds.sel(time=slice('2010-03-01','2018-03-01'))
modis_o_var_ds_sel = modis_o_var_ds.sel(time=slice('2010-03-01','2018-03-01'))
# Select variable
modis_y_monthly_var = modis_y_var_ds_sel["MYD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean"]
modis_o_monthly_var = modis_o_var_ds_sel["MOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean"]
# Merge the Terra and Aqua modis DataArray
modis_yo_monthly_var = xr.merge([modis_y_monthly_var,modis_o_monthly_var])
mean_var = modis_yo_monthly_var.to_array(dim='new').mean('new',skipna=True)
modis_yom_monthly_var = modis_yo_monthly_var.assign(MYOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean=mean_var)
modis_monthly = modis_yom_monthly_var["MYOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean"]

## OSLOCTM3
# ad550aer
osloCTM_dir= 'path-to-model-input'
osloCTM_ds = xr.open_mfdataset(osloCTM_dir)
# Change day 02 to day 01
osloCTM_ds.coords['time'] = xr.cftime_range(start="2010", periods=96, freq="1MS", calendar="noleap")
osloCTM_ds = osloCTM_ds.sortby(osloCTM_ds.time)
# Change the time range date.
osloCTM_ds_sel = osloCTM_ds.sel(time=slice('2010-01-02','2017-12-02'))
# Change longitude from 0-360 to -180-180
osloCTM_ds_sel.coords['lon'] = (osloCTM_ds_sel.coords['lon'].values + 180) % 360 - 180
osloCTM_ds_sel = osloCTM_ds_sel.sortby(osloCTM_ds_sel.lon)
# Select variable
osloCTM_monthly = osloCTM_ds_sel['od550aer']

## OSLO and MODIS
# Select august and september
modis_as_var = modis_monthly.sel(time=modis_monthly.time.dt.month.isin([7, 8, 9]))
modis_as_mean = modis_as_var.groupby('time.year').mean('time',skipna=True) 
osloCTM_as_var = osloCTM_monthly.sel(time=osloCTM_monthly.time.dt.month.isin([7, 8, 9]))
osloCTM_as_mean = osloCTM_as_var.groupby('time.year').mean('time',skipna=True)

# Select subdomain
# Africa
domain=['Red_Box']
lat_bnd = [-20, -2]
lon_bnd = [-10, 10]
modis_as_afr = modis_as_mean.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)
osloCTM_as_afr = osloCTM_as_mean.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)
# Average over latitude and longitude
avg_modis = modis_as_afr.mean(dim=('lat', 'lon'))
avg_osloCTM = osloCTM_as_afr.mean(dim=('lat', 'lon'))

# Export dataset into dataframe
df_avg_modis = avg_modis.to_dataframe()
df_avg_osloCTM = avg_osloCTM.to_dataframe()

rep_fig='path-to-figure-repertory'

# Make a time serie plot
color = ['black', 'gray']
fig, ax = plt.subplots(figsize=(16,8))
ind = np.arange(len(df_avg_modis.index))
width= 0.35
plt.bar(ind-width/2, df_avg_modis['MYOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean'], width, label='MODIS')
plt.bar(ind+width/2, df_avg_osloCTM['od550aer'], width, label='OsloCTM3')
plt.xlabel("Year", {'color': 'black', 'fontsize': 22})
ax.set_xticks(ind)
ax.set_xticklabels(('2010','2011','2012','2013','2014','2015','2016','2017'))
ax.xaxis.set_tick_params(labelsize=22)
plt.ylabel("AOD", {'color': 'black', 'fontsize': 22})
ax.yaxis.set_tick_params(labelsize=22)
plt.legend(loc='upper left', fancybox=True, shadow=True, ncol=2, bbox_to_anchor=(0, 1.1, 1, 0),fontsize =22)
plt.savefig(rep_fig+'FigS4c.jpeg', dpi=300, bbox_inches='tight')   
plt.savefig(rep_fig+'FigS4c.eps' , format='eps',bbox_inches='tight')
plt.show()

