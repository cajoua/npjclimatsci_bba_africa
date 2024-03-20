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
# 5- Spearman correlation
# 6-Create a netcdf file with the Spearman correlation
###################

# Open sevral Netcdf files 
# CERES
ceres_dir='path-to-ceres-data'
ceres_ds = xr.open_mfdataset(ceres_dir)

# MODIS
modis_y_dir_aod='path-to-modis-data'
modis_y_ds_aod = xr.open_mfdataset(modis_y_dir_aod)
modis_o_dir_aod='path-to-modis-data'
modis_o_ds_aod = xr.open_mfdataset(modis_o_dir_aod)

modis_dir_cotl='path-to-modis-data'
modis_ds_cotl = xr.open_mfdataset(modis_dir_cotl)
modis_dir_crfl='path-to-modis-data'
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
modis_y_ds_sel_aod = modis_y_ds_aod.sel(time=slice('2002-07-01','2022-03-01'))
modis_o_ds_sel_aod = modis_o_ds_aod.sel(time=slice('2002-07-01','2022-03-01'))
modis_ds_sel_cotl = modis_ds_cotl.sel(time=slice('2002-07-01','2022-03-01'))
modis_ds_sel_crfl = modis_ds_crfl.sel(time=slice('2002-07-01','2022-03-01'))

# Select variable
sw_all_mon  = ceres_ds_sel["toa_sw_all_mon"]
modis_monthly_cotl = modis_ds_sel_cotl["MYOD08_M3_7_0_Cloud_Optical_Thickness_Liquid_Mean"]
modis_monthly_crfl = modis_ds_sel_crfl["MYOD08_M3_7_0_Cloud_Retrieval_Fraction_Liquid_Mean"]
modis_y_daily_aod = modis_y_ds_sel_aod["MYD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean"]
modis_o_daily_aod = modis_o_ds_sel_aod["MOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean"]

# Merge the Terra and Aqua modis DataArray
modis_yo_daily_aod = xr.merge([modis_y_daily_aod,modis_o_daily_aod])
mean_aod = modis_yo_daily_aod.to_array(dim='new').mean('new',skipna=True)
modis_yom_daily_aod = modis_yo_daily_aod.assign(MYOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean=mean_aod)
modis_monthly_aod = modis_yom_daily_aod["MYOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean"]

# Select region
# Africa
domain=['Ocean_africa']
lat_bnd = [-20, -2.0]
lon_bnd = [-10, 10]
modis_mon_aod = modis_monthly_aod.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)
sw_all_mon_var = sw_all_mon.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)
modis_mon_cotl = modis_monthly_cotl.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)
modis_mon_crfl = modis_monthly_crfl.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)

# Select month in each year
sw_all_jas_var_m = sw_all_mon_var.sel(time=sw_all_mon_var.time.dt.month.isin([7, 8, 9]))
sw_all_jas_var = sw_all_jas_var_m.groupby('time.year').mean('time',skipna=True)
modis_jas_cotl_m = modis_mon_cotl.sel(time=modis_mon_cotl.time.dt.month.isin([7, 8, 9]))
modis_jas_cotl = modis_jas_cotl_m.groupby('time.year').mean('time',skipna=True)
modis_jas_crfl_m = modis_mon_crfl.sel(time=modis_mon_crfl.time.dt.month.isin([7, 8, 9]))
modis_jas_crfl = modis_jas_crfl_m.groupby('time.year').mean('time',skipna=True) 
modis_jas_aod_m = modis_mon_aod.sel(time=modis_mon_aod.time.dt.month.isin([7, 8, 9]))
modis_jas_aod = modis_jas_aod_m.groupby('time.year').mean('time',skipna=True)

# First reshape the xarray (time, lat, lon) into np.array (time,latxlon)
sw_all_mask_var_1d = sw_all_jas_var.values.reshape(-1)
modis_mask_cotl_1d = modis_jas_cotl.values.reshape(-1)
modis_mask_crfl_1d = modis_jas_crfl.values.reshape(-1)
modis_mask_aod_1d = modis_jas_aod.values.reshape(-1)

#Suppress AOD < 0.25 to remove low AOD and add range of AOD based on the defenition below
aod_ones = xr.ones_like(modis_jas_aod)
mask0 = aod_ones.where((modis_jas_aod < 0.1))
mask1 = aod_ones.where((modis_jas_aod >= 0.1) & (modis_jas_aod < 0.2))
mask2 = aod_ones.where((modis_jas_aod >= 0.2) & (modis_jas_aod < 0.3))
mask3 = aod_ones.where((modis_jas_aod >= 0.3) & (modis_jas_aod < 0.4))
mask4 = aod_ones.where((modis_jas_aod >= 0.4) & (modis_jas_aod < 0.5))
mask5 = aod_ones.where((modis_jas_aod >= 0.5) & (modis_jas_aod < 0.6))
mask6 = aod_ones.where((modis_jas_aod >= 0.6) & (modis_jas_aod < 0.7))
mask7 = aod_ones.where((modis_jas_aod >= 0.7) & (modis_jas_aod < 0.8))
mask8 = aod_ones.where((modis_jas_aod >= 0.8) & (modis_jas_aod < 0.9))
mask9 = aod_ones.where((modis_jas_aod >= 0.9))

modis_mask0_aod = modis_jas_aod*mask0
sw_all_mask0_var = sw_all_jas_var*mask0
modis_mask0_cotl = modis_jas_cotl*mask0
modis_mask0_crfl = modis_jas_crfl*mask0

modis_mask1_aod = modis_jas_aod*mask1
sw_all_mask1_var = sw_all_jas_var*mask1
modis_mask1_cotl = modis_jas_cotl*mask1
modis_mask1_crfl = modis_jas_crfl*mask1

modis_mask2_aod = modis_jas_aod*mask2
sw_all_mask2_var = sw_all_jas_var*mask2
modis_mask2_cotl = modis_jas_cotl*mask2
modis_mask2_crfl = modis_jas_crfl*mask2

modis_mask3_aod = modis_jas_aod*mask3
sw_all_mask3_var = sw_all_jas_var*mask3
modis_mask3_cotl = modis_jas_cotl*mask3
modis_mask3_crfl = modis_jas_crfl*mask3

modis_mask4_aod = modis_jas_aod*mask4
sw_all_mask4_var = sw_all_jas_var*mask4
modis_mask4_cotl = modis_jas_cotl*mask4
modis_mask4_crfl = modis_jas_crfl*mask4

modis_mask5_aod = modis_jas_aod*mask5
sw_all_mask5_var = sw_all_jas_var*mask5
modis_mask5_cotl = modis_jas_cotl*mask5
modis_mask5_crfl = modis_jas_crfl*mask5

modis_mask6_aod = modis_jas_aod*mask6
sw_all_mask6_var = sw_all_jas_var*mask6
modis_mask6_cotl = modis_jas_cotl*mask6
modis_mask6_crfl = modis_jas_crfl*mask6

modis_mask7_aod = modis_jas_aod*mask7
sw_all_mask7_var = sw_all_jas_var*mask7
modis_mask7_cotl = modis_jas_cotl*mask7
modis_mask7_crfl = modis_jas_crfl*mask7

modis_mask8_aod = modis_jas_aod*mask8
sw_all_mask8_var = sw_all_jas_var*mask8
modis_mask8_cotl = modis_jas_cotl*mask8
modis_mask8_crfl = modis_jas_crfl*mask8

modis_mask9_aod = modis_jas_aod*mask9
sw_all_mask9_var = sw_all_jas_var*mask9
modis_mask9_cotl = modis_jas_cotl*mask9
modis_mask9_crfl = modis_jas_crfl*mask9

# First reshape the xarray (time, lat, lon) into np.array (timexlatxlon)
modis_mask0_aod_res = modis_mask0_aod.values.reshape(-1)
sw_all_mask0_var_res = sw_all_mask0_var.values.reshape(-1)
modis_mask0_cotl_res = modis_mask0_cotl.values.reshape(-1)
modis_mask0_crfl_res = modis_mask0_crfl.values.reshape(-1)

modis_mask1_aod_res = modis_mask1_aod.values.reshape(-1)
sw_all_mask1_var_res = sw_all_mask1_var.values.reshape(-1)
modis_mask1_cotl_res = modis_mask1_cotl.values.reshape(-1)
modis_mask1_crfl_res = modis_mask1_crfl.values.reshape(-1)

modis_mask2_aod_res = modis_mask2_aod.values.reshape(-1)
sw_all_mask2_var_res = sw_all_mask2_var.values.reshape(-1)
modis_mask2_cotl_res = modis_mask2_cotl.values.reshape(-1)
modis_mask2_crfl_res = modis_mask2_crfl.values.reshape(-1)

modis_mask3_aod_res = modis_mask3_aod.values.reshape(-1)
sw_all_mask3_var_res = sw_all_mask3_var.values.reshape(-1)
modis_mask3_cotl_res = modis_mask3_cotl.values.reshape(-1)
modis_mask3_crfl_res = modis_mask3_crfl.values.reshape(-1)

modis_mask4_aod_res = modis_mask4_aod.values.reshape(-1)
sw_all_mask4_var_res = sw_all_mask4_var.values.reshape(-1)
modis_mask4_cotl_res = modis_mask4_cotl.values.reshape(-1)
modis_mask4_crfl_res = modis_mask4_crfl.values.reshape(-1)

modis_mask5_aod_res = modis_mask5_aod.values.reshape(-1)
sw_all_mask5_var_res = sw_all_mask5_var.values.reshape(-1)
modis_mask5_cotl_res = modis_mask5_cotl.values.reshape(-1)
modis_mask5_crfl_res = modis_mask5_crfl.values.reshape(-1)

modis_mask6_aod_res = modis_mask6_aod.values.reshape(-1)
sw_all_mask6_var_res = sw_all_mask6_var.values.reshape(-1)
modis_mask6_cotl_res = modis_mask6_cotl.values.reshape(-1)
modis_mask6_crfl_res = modis_mask6_crfl.values.reshape(-1)

modis_mask7_aod_res = modis_mask7_aod.values.reshape(-1)
sw_all_mask7_var_res = sw_all_mask7_var.values.reshape(-1)
modis_mask7_cotl_res = modis_mask7_cotl.values.reshape(-1)
modis_mask7_crfl_res = modis_mask7_crfl.values.reshape(-1)

modis_mask8_aod_res = modis_mask8_aod.values.reshape(-1)
sw_all_mask8_var_res = sw_all_mask8_var.values.reshape(-1)
modis_mask8_cotl_res = modis_mask8_cotl.values.reshape(-1)
modis_mask8_crfl_res = modis_mask8_crfl.values.reshape(-1)

modis_mask9_aod_res = modis_mask9_aod.values.reshape(-1)
sw_all_mask9_var_res = sw_all_mask9_var.values.reshape(-1)
modis_mask9_cotl_res = modis_mask9_cotl.values.reshape(-1)
modis_mask9_crfl_res = modis_mask9_crfl.values.reshape(-1)

modis_mask0_aod_res = modis_mask0_aod_res[~np.isnan(modis_mask0_aod_res)]
sw_all_mask0_var_res = sw_all_mask0_var_res[~np.isnan(sw_all_mask0_var_res)]
modis_mask0_cotl_res = modis_mask0_cotl_res[~np.isnan(modis_mask0_cotl_res)]
modis_mask0_crfl_res = modis_mask0_crfl_res[~np.isnan(modis_mask0_crfl_res)]

modis_mask1_aod_res = modis_mask1_aod_res[~np.isnan(modis_mask1_aod_res)]
sw_all_mask1_var_res = sw_all_mask1_var_res[~np.isnan(sw_all_mask1_var_res)]
modis_mask1_cotl_res = modis_mask1_cotl_res[~np.isnan(modis_mask1_cotl_res)]
modis_mask1_crfl_res = modis_mask1_crfl_res[~np.isnan(modis_mask1_crfl_res)]

modis_mask2_aod_res = modis_mask2_aod_res[~np.isnan(modis_mask2_aod_res)]
sw_all_mask2_var_res = sw_all_mask2_var_res[~np.isnan(sw_all_mask2_var_res)]
modis_mask2_cotl_res = modis_mask2_cotl_res[~np.isnan(modis_mask2_cotl_res)]
modis_mask2_crfl_res = modis_mask2_crfl_res[~np.isnan(modis_mask2_crfl_res)]

modis_mask3_aod_res = modis_mask3_aod_res[~np.isnan(modis_mask3_aod_res)]
sw_all_mask3_var_res = sw_all_mask3_var_res[~np.isnan(sw_all_mask3_var_res)]
modis_mask3_cotl_res = modis_mask3_cotl_res[~np.isnan(modis_mask3_cotl_res)]
modis_mask3_crfl_res = modis_mask3_crfl_res[~np.isnan(modis_mask3_crfl_res)]

modis_mask4_aod_res = modis_mask4_aod_res[~np.isnan(modis_mask4_aod_res)]
sw_all_mask4_var_res = sw_all_mask4_var_res[~np.isnan(sw_all_mask4_var_res)]
modis_mask4_cotl_res = modis_mask4_cotl_res[~np.isnan(modis_mask4_cotl_res)]
modis_mask4_crfl_res = modis_mask4_crfl_res[~np.isnan(modis_mask4_crfl_res)]

modis_mask5_aod_res = modis_mask5_aod_res[~np.isnan(modis_mask5_aod_res)]
sw_all_mask5_var_res = sw_all_mask5_var_res[~np.isnan(sw_all_mask5_var_res)]
modis_mask5_cotl_res = modis_mask5_cotl_res[~np.isnan(modis_mask5_cotl_res)]
modis_mask5_crfl_res = modis_mask5_crfl_res[~np.isnan(modis_mask5_crfl_res)]

modis_mask6_aod_res = modis_mask6_aod_res[~np.isnan(modis_mask6_aod_res)]
sw_all_mask6_var_res = sw_all_mask6_var_res[~np.isnan(sw_all_mask6_var_res)]
modis_mask6_cotl_res = modis_mask6_cotl_res[~np.isnan(modis_mask6_cotl_res)]
modis_mask6_crfl_res = modis_mask6_crfl_res[~np.isnan(modis_mask6_crfl_res)]

modis_mask7_aod_res = modis_mask7_aod_res[~np.isnan(modis_mask7_aod_res)]
sw_all_mask7_var_res = sw_all_mask7_var_res[~np.isnan(sw_all_mask7_var_res)]
modis_mask7_cotl_res = modis_mask7_cotl_res[~np.isnan(modis_mask7_cotl_res)]
modis_mask7_crfl_res = modis_mask7_crfl_res[~np.isnan(modis_mask7_crfl_res)]

modis_mask8_aod_res = modis_mask8_aod_res[~np.isnan(modis_mask8_aod_res)]
sw_all_mask8_var_res = sw_all_mask8_var_res[~np.isnan(sw_all_mask8_var_res)]
modis_mask8_cotl_res = modis_mask8_cotl_res[~np.isnan(modis_mask8_cotl_res)]
modis_mask8_crfl_res = modis_mask8_crfl_res[~np.isnan(modis_mask8_crfl_res)]

modis_mask9_aod_res = modis_mask9_aod_res[~np.isnan(modis_mask9_aod_res)]
sw_all_mask9_var_res = sw_all_mask9_var_res[~np.isnan(sw_all_mask9_var_res)]
modis_mask9_cotl_res = modis_mask9_cotl_res[~np.isnan(modis_mask9_cotl_res)]
modis_mask9_crfl_res = modis_mask9_crfl_res[~np.isnan(modis_mask9_crfl_res)]

#convert np.array into dataframe with column for each reff_latxlon, and index as time (timexlatxlon)
df = pd.DataFrame({'sw_all': sw_all_mask_var_1d, 'cotl': modis_mask_cotl_1d, 'crfl': modis_mask_crfl_1d})
x = df[['crfl','cotl']]
y = df['sw_all']
# with statsmodels
x = sm.add_constant(x) # adding a constant
model = sm.OLS(y, x).fit()

#convert np.array into dataframe with column for each reff_latxlon, and index as time (timexlatxlon)
df0 = pd.DataFrame({'sw_all': sw_all_mask0_var_res, 'cotl': modis_mask0_cotl_res, 'crfl': modis_mask0_crfl_res})
x0 = df0[['crfl','cotl']]
y0 = df0['sw_all']
# with statsmodels
x0 = sm.add_constant(x0) # adding a constant
model0 = sm.OLS(y0, x0).fit()

#convert np.array into dataframe with column for each reff_latxlon, and index as time (timexlatxlon)
df1 = pd.DataFrame({'sw_all': sw_all_mask1_var_res, 'cotl': modis_mask1_cotl_res, 'crfl': modis_mask1_crfl_res})
x1 = df1[['crfl','cotl']]
y1 = df1['sw_all']
# with statsmodels
x1 = sm.add_constant(x1) # adding a constant
model1 = sm.OLS(y1, x1).fit()

#convert np.array into dataframe with column for each reff_latxlon, and index as time (timexlatxlon)
df2 = pd.DataFrame({'sw_all': sw_all_mask2_var_res, 'cotl': modis_mask2_cotl_res, 'crfl': modis_mask2_crfl_res})
x2 = df2[['crfl','cotl']]
y2 = df2['sw_all']
# with statsmodels
x2 = sm.add_constant(x2) # adding a constant
model2 = sm.OLS(y2, x2).fit()

#convert np.array into dataframe with column for each reff_latxlon, and index as time (timexlatxlon)
df3 = pd.DataFrame({'sw_all': sw_all_mask3_var_res, 'cotl': modis_mask3_cotl_res, 'crfl': modis_mask3_crfl_res})
x3 = df3[['crfl','cotl']]
y3 = df3['sw_all']
# with statsmodels
x3 = sm.add_constant(x3) # adding a constant
model3 = sm.OLS(y3, x3).fit()

#convert np.array into dataframe with column for each reff_latxlon, and index as time (timexlatxlon)
df4 = pd.DataFrame({'sw_all': sw_all_mask4_var_res, 'cotl': modis_mask4_cotl_res, 'crfl': modis_mask4_crfl_res})
x4 = df4[['crfl','cotl']]
y4 = df4['sw_all']
# with statsmodels
x4 = sm.add_constant(x4) # adding a constant
model4 = sm.OLS(y4, x4).fit()

#convert np.array into dataframe with column for each reff_latxlon, and index as time (timexlatxlon)
df5 = pd.DataFrame({'sw_all': sw_all_mask5_var_res, 'cotl': modis_mask5_cotl_res, 'crfl': modis_mask5_crfl_res})
x5 = df5[['crfl','cotl']]
y5 = df5['sw_all']
# with statsmodels
x5 = sm.add_constant(x5) # adding a constant
model5 = sm.OLS(y5, x5).fit()

#convert np.array into dataframe with column for each reff_latxlon, and index as time (timexlatxlon)
df6 = pd.DataFrame({'sw_all': sw_all_mask6_var_res, 'cotl': modis_mask6_cotl_res, 'crfl': modis_mask6_crfl_res})
x6 = df6[['crfl','cotl']]
y6 = df6['sw_all']
# with statsmodels
x6 = sm.add_constant(x6) # adding a constant
model6 = sm.OLS(y6, x6).fit()

#convert np.array into dataframe with column for each reff_latxlon, and index as time (timexlatxlon)
df7 = pd.DataFrame({'sw_all': sw_all_mask7_var_res, 'cotl': modis_mask7_cotl_res, 'crfl': modis_mask7_crfl_res})
x7 = df7[['crfl','cotl']]
y7 = df7['sw_all']
# with statsmodels
x7 = sm.add_constant(x7) # adding a constant
model7 = sm.OLS(y7, x7).fit()

#convert np.array into dataframe with column for each reff_latxlon, and index as time (timexlatxlon)
df8 = pd.DataFrame({'sw_all': sw_all_mask8_var_res, 'cotl': modis_mask8_cotl_res, 'crfl': modis_mask8_crfl_res})
x8 = df8[['crfl','cotl']]
y8 = df8['sw_all']
# with statsmodels
x8 = sm.add_constant(x8) # adding a constant
model8 = sm.OLS(y8, x8).fit()

#convert np.array into dataframe with column for each reff_latxlon, and index as time (timexlatxlon)
df9 = pd.DataFrame({'sw_all': sw_all_mask9_var_res, 'cotl': modis_mask9_cotl_res, 'crfl': modis_mask9_crfl_res})
x9 = df9[['crfl','cotl']]
y9 = df9['sw_all']
# with statsmodels
x9 = sm.add_constant(x9) # adding a constant
model9 = sm.OLS(y9, x9).fit()

# Have information on the regression line
#print_model = model.summary()
#print(print_model)
const = model.params['const']
x1 = model.params['crfl']
x2 = model.params['cotl']

#print_model0 = model0.summary()
#print(print_model0)
const0 = model0.params['const']
x10 = model0.params['crfl']
x20 = model0.params['cotl']

#print_model1 = model1.summary()
#print(print_model1)
const1 = model1.params['const']
x11 = model1.params['crfl']
x21 = model1.params['cotl']

#print_model2 = model2.summary()
#print(print_model2)
const2 = model2.params['const']
x12 = model2.params['crfl']
x22 = model2.params['cotl']

#print_model3 = model3.summary()
#print(print_model3)
const3 = model3.params['const']
x13 = model3.params['crfl']
x23 = model3.params['cotl']

#print_model4 = model4.summary()
#print(print_model4)
const4 = model4.params['const']
x14 = model4.params['crfl']
x24 = model4.params['cotl']

#print_model5 = model5.summary()
#print(print_model5)
const5 = model5.params['const']
x15 = model5.params['crfl']
x25 = model5.params['cotl']

#print_model6 = model6.summary()
#print(print_model6)
const6 = model6.params['const']
x16 = model6.params['crfl']
x26 = model6.params['cotl']

#print_model7 = model7.summary()
#print(print_model7)
const7 = model7.params['const']
x17 = model7.params['crfl']
x27 = model7.params['cotl']

#print_model8 = model8.summary()
#print(print_model8)
const8 = model8.params['const']
x18 = model8.params['crfl']
x28 = model8.params['cotl']

#print_model9 = model9.summary()
#print(print_model9)
const9 = model9.params['const']
x19 = model9.params['crfl']
x29 = model9.params['cotl']

#repertory
fig_rep = 'path-to-figure-repertory'

# Figure pour voir le residu.
fig = plt.figure(figsize=(22,10),frameon=True)
ax = fig.add_subplot(1,1,1, projection='3d')
fig.tight_layout()
xx0 = df0['crfl']
yy0 = df0['cotl']
zz0 = df0['sw_all']
xx1 = df1['crfl']
yy1 = df1['cotl']
zz1 = df1['sw_all']
xx2 = df2['crfl']
yy2 = df2['cotl']
zz2 = df2['sw_all']
xx3 = df3['crfl']
yy3 = df3['cotl']
zz3 = df3['sw_all']
xx4 = df4['crfl']
yy4 = df4['cotl']
zz4 = df4['sw_all']
xx5 = df5['crfl']
yy5 = df5['cotl']
zz5 = df5['sw_all']
xx6 = df6['crfl']
yy6 = df6['cotl']
zz6 = df6['sw_all']
xx7 = df7['crfl']
yy7 = df7['cotl']
zz7 = df7['sw_all']
xx8 = df8['crfl']
yy8 = df8['cotl']
zz8 = df8['sw_all']
xx9 = df9['crfl']
yy9 = df9['cotl']
zz9 = df9['sw_all']
line0 = f'         AOD<0.1 : ${{y}}$ = {const0:.2f} + {x10:.2f}$x_1$ + {x20:.2f}$x_2$, R$^2$ = 0.96, No. pts=36'
line1 = f'0.1$\geq$AOD<0.2 : ${{y}}$ = {const1:.2f} + {x11:.2f}$x_1$ + {x21:.2f}$x_2$, R$^2$ = 0.97, No. pts=947'
line2 = f'0.2$\geq$AOD<0.3 : ${{y}}$ =   {const2:.2f} + {x12:.2f}$x_1$ + {x22:.2f}$x_2$, R$^2$ = 0.97, No. pts=1239'
line3 = f'0.3$\geq$AOD<0.4 : ${{y}}$ = {const3:.2f} + {x13:.2f}$x_1$ + {x23:.2f}$x_2$, R$^2$ = 0.98, No. pts=1282'
line4 = f'0.4$\geq$AOD<0.5 : ${{y}}$ = {const4:.2f} + {x14:.2f}$x_1$ + {x24:.2f}$x_2$, R$^2$ = 0.98, No. pts=1060'
line5 = f'0.5$\geq$AOD<0.6 : ${{y}}$ = {const5:.2f} + {x15:.2f}$x_1$ + {x25:.2f}$x_2$, R$^2$ = 0.98, No. pts=746'
line6 = f'0.6$\geq$AOD<0.7 : ${{y}}$ = {const6:.2f} + {x16:.2f}$x_1$ + {x26:.2f}$x_2$, R$^2$ = 0.98, No. pts=501'
line7 = f'0.7$\geq$AOD<0.8 : ${{y}}$ = {const7:.2f} + {x17:.2f}$x_1$ + {x27:.2f}$x_2$, R$^2$ = 0.98, No. pts=416'
line8 = f'0.8$\geq$AOD<0.9 : ${{y}}$ = {const8:.2f} + {x18:.2f}$x_1$ + {x28:.2f}$x_2$, R$^2$ = 0.97, No. pts=317'
line9 = f'        AOD$\geq$0.9 : y = {const9:.2f} + {x19:.2f}$x_1$ + {x29:.2f}$x_2$, R$^2$ = 0.98, No. pts=656'
ax.scatter(xx0,yy0,zz0, marker='.', s=10, c='#050f2c', label=line0)
ax.scatter(xx1,yy1,zz1, marker='.', s=10, c='#003666', label=line1)
ax.scatter(xx2,yy2,zz2, marker='.', s=10, c='#00aeff', label=line2)
ax.scatter(xx3,yy3,zz3, marker='.', s=10, c='#3369e7', label=line3)
ax.scatter(xx4,yy4,zz4, marker='.', s=10, c='#8e43e7', label=line4)
ax.scatter(xx5,yy5,zz5, marker='.', s=10, c='#b84592', label=line5)
ax.scatter(xx6,yy6,zz6, marker='.', s=10, c='#ff4f81', label=line6)
ax.scatter(xx7,yy7,zz7, marker='.', s=10, c='#ff6c5f', label=line7)
ax.scatter(xx8,yy8,zz8, marker='.', s=10, c='#ffc168', label=line8)
ax.scatter(xx9,yy9,zz9, marker='.', s=10, c='#2dde98', label=line9)
ax.view_init(elev=15, azim=-45)
ax.set_xlabel('$x_1$: CF', fontsize=22, rotation=45, labelpad=30.0)
ax.set_ylabel('$x_2$: COT', fontsize=22, rotation=45, labelpad=30.0)
ax.set_zlabel(r'${{y}}$: SWR$_{\rm ALL}$ [W m$^{-2}$]', fontsize=22, rotation=45, labelpad=30.0)
ax.tick_params(labelsize=22)
ax.legend(bbox_to_anchor=(1.15,0.75),facecolor='white',markerscale=4.,fontsize=18,frameon=False)
plt.savefig(fig_rep + 'FigS3.eps', format='eps',bbox_inches='tight')
plt.savefig(fig_rep + 'FigS3.jpeg', dpi=300,bbox_inches='tight')
plt.show()
