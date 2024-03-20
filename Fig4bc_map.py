#data manipulation
import numpy as np
import xarray as xr

#data statistics
from scipy.stats import norm

#for plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyproj import Proj, transform
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings("ignore")
import copy

def make_figure():
    fig=plt.figure(figsize=(18,8), frameon=True) 
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))    
    ax.add_feature(cfeature.LAND.with_scale('50m')) 
    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                                facecolor='none', name='coastline')
    ax.add_feature(coast, edgecolor='black')
    return fig, ax

# Data repertory
data_rep = 'path-to-trend-input'
# Variable name
data_crfl = 'MYOD08_M3_7_0_Cloud_Retrieval_Fraction_Liquid_Mean_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc'
data_cotl = 'MYOD08_M3_7_0_Cloud_Optical_Thickness_Liquid_Mean_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc'
data_sw_all = 'CERES_TOA_SW_ALL_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc'

# Figure repertory
fig_rep = 'path-to-figure-repertory'

#Read data
ncfile_crfl = data_rep + data_crfl
ncfile_cotl = data_rep + data_cotl
ncfile_sw_all = data_rep + data_sw_all
crfl = xr.open_dataset(ncfile_crfl)
cotl = xr.open_dataset(ncfile_cotl)
sw_all = xr.open_dataset(ncfile_sw_all)

# cotl and crfl linear regression
sw_all_predict = 76.320*crfl + 4.291*cotl
sw_all_residu = sw_all - sw_all_predict

#Averaging the red box values
domain=['RedBox']
lat_red = [-20, -2.0]
lon_red = [-10, 10]
sw_all_sel = sw_all.sel(x=slice(*lon_red), y=slice(*lat_red),)
# Rename coordinates
sw_all_ds = sw_all_sel.rename({'x': 'longitude','y': 'latitude'})
### Weighted mean according to lat.
weights = np.cos(np.deg2rad(sw_all_ds.latitude))
weights.name = "weights"
sw_all_ds_lat_weighted = sw_all_ds.weighted(weights)
sw_all_ds_lat_weighted_mean = sw_all_ds_lat_weighted.mean(("longitude","latitude"))
sw_all_ds_lat_weighted_std = sw_all_ds_lat_weighted.std(("longitude","latitude"))
print('sw_all_lat_mean:',sw_all_ds_lat_weighted_mean)
print('sw_all_lat_std:',sw_all_ds_lat_weighted_std)

sw_all_predict_sel = sw_all_predict.sel(x=slice(*lon_red), y=slice(*lat_red),)
# Rename coordinates
sw_all_predict_ds = sw_all_predict_sel.rename({'x': 'longitude','y': 'latitude'})
### Weighted mean according to lat.
weights = np.cos(np.deg2rad(sw_all_predict_ds.latitude))
weights.name = "weights"
sw_all_predict_ds_lat_weighted = sw_all_predict_ds.weighted(weights)
sw_all_predict_ds_lat_weighted_mean = sw_all_predict_ds_lat_weighted.mean(("longitude","latitude"))
sw_all_predict_ds_lat_weighted_std = sw_all_predict_ds_lat_weighted.std(("longitude","latitude"))
print('sw_all_pred_lat_mean:',sw_all_predict_ds_lat_weighted_mean)
print('sw_all_pred_lat_std:',sw_all_predict_ds_lat_weighted_std)

sw_all_residu_sel = sw_all_residu.sel(x=slice(*lon_red), y=slice(*lat_red),)
# Rename coordinates
sw_all_residu_ds = sw_all_residu_sel.rename({'x': 'longitude','y': 'latitude'})
### Weighted mean according to lat.
weights = np.cos(np.deg2rad(sw_all_residu_ds.latitude))
weights.name = "weights"
sw_all_residu_ds_lat_weighted = sw_all_residu_ds.weighted(weights)
sw_all_residu_ds_lat_weighted_mean = sw_all_residu_ds_lat_weighted.mean(("longitude","latitude"))
sw_all_residu_ds_lat_weighted_std = sw_all_residu_ds_lat_weighted.std(("longitude","latitude"))
print('sw_all_res_lat_mean:',sw_all_residu_ds_lat_weighted_mean)
print('sw_all_res_lat_std:',sw_all_residu_ds_lat_weighted_std)

_, ax1 = make_figure()
# Keep the original linear cmap
cmap = mpl.cm.PRGn_r
# regrid 1D latitude and longitude to 2D grid
mm1 = ax1.pcolormesh(sw_all_residu.x,\
                   sw_all_residu.y,\
                   sw_all_residu.slope*-1,\
                   transform=ccrs.PlateCarree(),\
                   vmin = -0.5, \
                   vmax = 0.5, \
                   cmap=cmap)
coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',     # ajout de la couche cotière 
                                facecolor='none', name='coastline') 
ax1.add_feature(coast, edgecolor='black') 
# Add the gridlines
gl = ax1.gridlines(draw_labels=True, color="black", linestyle="dotted")
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
# Add patch
# Africa
patches_A = []
zone_A = np.array([[-10.0,-20.0],[-10.0,-2.0],[10.0,-2.0],[10.0,-20.0]])
patches_A.append(Polygon(zone_A))
ax1.add_collection(PatchCollection(patches_A, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidths=1.5))
cbar = plt.colorbar(mm1, extend='both', shrink=0.35, orientation='horizontal')
cbar.set_label(label='W m$^{-2}$ yr$^{-1}$', fontsize=22)
cbar.ax.tick_params(labelsize=22) 
plt.title(r'DARE$_{\rm all}$',fontsize=32)
plt.savefig(fig_rep + 'Figure_4c.eps', format='eps',bbox_inches='tight')
plt.savefig(fig_rep + 'Figure_4c.jpeg', dpi=300,bbox_inches='tight')
plt.show()

_, ax2 = make_figure()
# Keep the original linear cmap
cmap = mpl.cm.PRGn_r
# regrid 1D latitude and longitude to 2D grid
mm2 = ax2.pcolormesh(sw_all_predict.x,\
                   sw_all_predict.y,\
                   sw_all_predict.slope*-1.0,\
                   transform=ccrs.PlateCarree(),\
                   vmin = -1, \
                   vmax = 1, \
                   cmap=cmap)
coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',     # ajout de la couche cotière 
                                facecolor='none', name='coastline')
ax2.add_feature(coast, edgecolor='black')
# Add the gridlines
gl = ax2.gridlines(draw_labels=True, color="black", linestyle="dotted")
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
# Add patch
# Africa
patches_A = []
zone_A = np.array([[-10.0,-20.0],[-10.0,-2.0],[10.0,-2.0],[10.0,-20.0]])
patches_A.append(Polygon(zone_A))
ax2.add_collection(PatchCollection(patches_A, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidths=1.5))
cbar = plt.colorbar(mm2, extend='both', shrink=0.35, orientation='horizontal')
cbar.set_label(label='W m$^{-2}$ yr$^{-1}$', fontsize=22)
cbar.ax.tick_params(labelsize=22)
plt.title(r'SWR$_{\rm cld}$',fontsize=32)
plt.savefig(fig_rep + 'Figure_4b.eps', format='eps',bbox_inches='tight')
plt.savefig(fig_rep + 'Figure_4b.jpeg', dpi=300,bbox_inches='tight')
plt.show()
