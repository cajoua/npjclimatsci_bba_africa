#data manipulation
import numpy as np
import xarray as xr
import pandas as pd

#data statistics
import scipy.stats
import statsmodels.api as sm

#graphics
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
#plt.style.use('ggplot')

#warning
import warnings
warnings.filterwarnings("ignore")

###################
# 1- Download multiple MODIS Terra and Aqua netcdf files
# 2- Select time date range 
# 3- Combine Terra and Aqua data (mean)
# 4- Monthly mean
# 5- OLS Regression
###################

# Open sevral Netcdf files 
# CERES
ceres_dir='path-to-ceres-input'
ceres_ds = xr.open_mfdataset(ceres_dir)

# MODIS
modis_dir_cotl='path-to-modis-input'
modis_ds_cotl = xr.open_mfdataset(modis_dir_cotl)
modis_dir_crfl='path-to-modis-input'
modis_ds_crfl = xr.open_mfdataset(modis_dir_crfl)

# CERES (day=15) select time date range
ceres_ds_sel = ceres_ds.sel(time=slice('2002-07-15','2022-03-15'))
# Change day name to day=1 such as MODIS
ceres_ds_sel.coords['time'] = xr.date_range(start="2002-07-01", periods=237, freq="1MS", calendar="standard")
#ceres_ds_sel = ceres_ds_sel.sortby(ceres_ds_sel.time)
# Change longitude from 0-360 to -180-180 such as MODIS
ceres_ds_sel.coords['lon'] = (ceres_ds_sel.coords['lon'].values + 180) % 360 - 180
ceres_ds_sel = ceres_ds_sel.sortby(ceres_ds_sel.lon)

# MODIS (day=1) select time date range
modis_ds_sel_cotl = modis_ds_cotl.sel(time=slice('2002-07-01','2022-03-01'))
modis_ds_sel_crfl = modis_ds_crfl.sel(time=slice('2002-07-01','2022-03-01'))

# Select variable
sw_all_mon  = ceres_ds_sel["toa_sw_all_mon"]
modis_monthly_cotl = modis_ds_sel_cotl["MYOD08_M3_7_0_Cloud_Optical_Thickness_Liquid_Mean"]
modis_monthly_crfl = modis_ds_sel_crfl["MYOD08_M3_7_0_Cloud_Retrieval_Fraction_Liquid_Mean"]

# Select region
# Africa
domain=['Ocean_africa']
lat_bnd = [-20, -2.0]
lon_bnd = [-10, 10]
sw_all_mon_var = sw_all_mon.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)
modis_mon_cotl = modis_monthly_cotl.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)
modis_mon_crfl = modis_monthly_crfl.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)

# Select month in each year
sw_all_jas_var_m = sw_all_mon_var.sel(time=sw_all_mon_var.time.dt.month.isin([7, 8, 9]))
sw_all_jas_var = sw_all_jas_var_m.groupby('time.year').mean('time',skipna=False)
modis_jas_cotl_m = modis_mon_cotl.sel(time=modis_mon_cotl.time.dt.month.isin([7, 8, 9]))
modis_jas_cotl = modis_jas_cotl_m.groupby('time.year').mean('time',skipna=False)
modis_jas_crfl_m = modis_mon_crfl.sel(time=modis_mon_crfl.time.dt.month.isin([7, 8, 9]))
modis_jas_crfl = modis_jas_crfl_m.groupby('time.year').mean('time',skipna=False) 


# First reshape the xarray (time, lat, lon) into np.array (time,latxlon)
sw_all_var_1d = sw_all_jas_var.values.reshape(-1)
modis_cotl_1d = modis_jas_cotl.values.reshape(-1)
modis_crfl_1d = modis_jas_crfl.values.reshape(-1)

#convert np.array into dataframe with column for each reff_latxlon, and index as time (timexlatxlon)
df = pd.DataFrame({'sw_all': sw_all_var_1d, 'cotl': modis_cotl_1d, 'crfl': modis_crfl_1d})
x = df[['crfl','cotl']]
y = df['sw_all']
# with statsmodels
x = sm.add_constant(x) # adding a constant
model = sm.OLS(y, x).fit()

# Have information on the regression line
#print_model = model.summary()
#print(print_model)
const = model.params['const']
x1 = model.params['crfl']
x2 = model.params['cotl']

#repertory
fig_rep = 'path-to-figure-repertory'

# Figure pour voir le residu.
fig = plt.figure(figsize=(18,18),frameon=True)
ax = fig.add_subplot(1,1,1, projection='3d')
fig.tight_layout()
fig.subplots_adjust(top=0.80)
xx = df['crfl']
yy = df['cotl']
zz = df['sw_all']
ax.scatter(xx,yy,zz, marker='.', s=10, c="black")
ax.view_init(elev=15, azim=-45)
ax.set_xlabel('$x_1$: CF', fontsize=24, rotation=45, labelpad=30.0)
ax.set_ylabel('$x_2$: COT', fontsize=24, rotation=45, labelpad=30.0)
ax.set_zlabel(r'${{y}}$: SWR$_{\rm ALL}$ [W m$^{⁻2}$]', fontsize=24, rotation=45, labelpad=30.0)
ax.tick_params(labelsize=22)
plt.title("R$^2$ = 0.974",fontsize=22)
plt.savefig(fig_rep + 'Fig4a_corr_test.eps', format='eps')#,bbox_inches='tight')
plt.savefig(fig_rep + 'Fig4a_corr_test.jpeg', dpi=300)#,bbox_inches='tight')
plt.show()
