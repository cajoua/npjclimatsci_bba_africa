import xarray as xr
import numpy as np
import warnings; warnings.filterwarnings(action='ignore')
from matplotlib import pyplot as plt


## MODIS
# AOD
# Open sevral Netcdf files 
ceres_dir='path-to-ceres-input'
ceres_ds = xr.open_mfdataset(ceres_dir)
# Select time date range
ceres_ds_sel = ceres_ds.sel(time=slice('2010-03-01','2018-03-01'))
# Change longitude from 0-360 to -180-180
ceres_ds_sel.coords['lon'] = (ceres_ds_sel.coords['lon'].values + 180) % 360 - 180
ceres_ds_sel = ceres_ds_sel.sortby(ceres_ds_sel.lon)
# Select variable
ceres_monthly  = ceres_ds_sel["toa_sw_clr_c_mon"]
#ceres_monthly  = ceres_ds_sel["toa_sw_all_mon"]

## OSLOCTM3
# ad550aer
osloCTM_dir= 'path-to-model-input'
#osloCTM_dir= 'path-to-model-input'
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
osloCTM_monthly = osloCTM_ds_sel['rsutcs']
#osloCTM_monthly = osloCTM_ds_sel['rsut']

## OSLO and CERES
# Select august and september
ceres_as_var = ceres_monthly.sel(time=ceres_monthly.time.dt.month.isin([7, 8, 9]))
ceres_as_mean = ceres_as_var.groupby('time.year').mean('time')
osloCTM_as_var = osloCTM_monthly.sel(time=osloCTM_monthly.time.dt.month.isin([7, 8, 9]))
osloCTM_as_mean = osloCTM_as_var.groupby('time.year').mean('time')

# Select subdomain
# Africa
domain=['Red_Box']
lat_bnd = [-20, -2]
lon_bnd = [-10, 10]
ceres_as_afr = ceres_as_mean.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)
osloCTM_as_afr = osloCTM_as_mean.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)
# Average over latitude and longitude
avg_ceres = ceres_as_afr.mean(dim=('lat', 'lon'))
avg_osloCTM = osloCTM_as_afr.mean(dim=('lat', 'lon'))

# Export dataset into dataframe
df_avg_ceres = avg_ceres.to_dataframe()
df_avg_osloCTM = avg_osloCTM.to_dataframe()

rep_fig='path-to-figure-repertory'

# Make a time serie plot
color = ['black', 'gray']
fig, ax = plt.subplots(figsize=(16,8))
ind = np.arange(len(df_avg_ceres.index))
width= 0.35
plt.bar(ind-width/2, df_avg_ceres['toa_sw_clr_c_mon'], width, label='CERES')
plt.bar(ind+width/2, df_avg_osloCTM['rsutcs'], width, label='OsloCTM3')
#plt.bar(ind-width/2, df_avg_ceres['toa_sw_all_mon'], width, label='CERES')
#plt.bar(ind+width/2, df_avg_osloCTM['rsut'], width, label='OsloCTM3')
plt.xlabel("Year", {'color': 'black', 'fontsize': 22})
ax.set_xticks(ind)
ax.set_xticklabels(('2010','2011','2012','2013','2014','2015','2016','2017'))
ax.xaxis.set_tick_params(labelsize=22)
plt.ylabel(r'SWR$_{\rm clr}$ [W m$^{⁻2}$]', {'color': 'black', 'fontsize': 22})
ax.yaxis.set_tick_params(labelsize=22)
plt.legend(loc='upper left', fancybox=True, shadow=True, ncol=2, bbox_to_anchor=(0, 1.1, 1, 0),fontsize =22)
plt.savefig(rep_fig+'FigS4a.jpeg', dpi=300, bbox_inches='tight')   
plt.savefig(rep_fig+'FigS4a.eps' , format='eps',bbox_inches='tight')
plt.show()

