#data manipulation
import numpy as np
import xarray as xr

#data statistics
from scipy.stats import norm

#data reggrid
import xesmf as xe

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

# guigui plot :-)
def make_figure():
    fig=plt.figure(figsize=(18,8), frameon=True) 
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))      # couche ocean
    ax.add_feature(cfeature.LAND.with_scale('50m'))       # couche land
    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',     # ajout de la couche cotière 
                                facecolor='none', name='coastline')
    ax.add_feature(coast, edgecolor='black')
    return fig, ax

# Data MKtrend repertory
data_rep = 'path-to-trend-data'
data_set = xr.open_dataset(data_rep)
data_set = data_set.rename({'y': 'lat', 'x':'lon'})

# NCFile names+repertory
ncfile_sw_all_toa_aero = 'path-to-model-input'
ncfile_sw_all_toa_noaero = 'path-to-model-input'
#ncfile_sw_all_toa_aero = 'path-to-model-input'
#ncfile_sw_all_toa_noaero = 'path-to-model-input'
ncfile_aod = 'path-to-model-input'

# Figure repertory
fig_rep = 'path-to-figure-repertory'

# Read data
aod_ds = xr.open_mfdataset(ncfile_aod)
sw_all_aero_ds = xr.open_mfdataset(ncfile_sw_all_toa_aero)
sw_all_noaero_ds = xr.open_mfdataset(ncfile_sw_all_toa_noaero)

# Change the time range date.
aod_ds.coords['time'] = xr.cftime_range(start="2010", periods=96, freq="1MS", calendar="noleap")
sw_all_aero_ds.coords['time'] = xr.cftime_range(start="2010", periods=96, freq="1MS", calendar="noleap")
sw_all_noaero_ds.coords['time'] = xr.cftime_range(start="2010", periods=96, freq="1MS", calendar="noleap")

# Change longitude from 0-360 to -180-180
aod_ds.coords['lon'] = (aod_ds.coords['lon'].values + 180) % 360 - 180
aod_ds = aod_ds.sortby(aod_ds.lon)
sw_all_aero_ds.coords['lon'] = (sw_all_aero_ds.coords['lon'].values + 180) % 360 - 180
sw_all_aero_ds = sw_all_aero_ds.sortby(sw_all_aero_ds.lon)
sw_all_noaero_ds.coords['lon'] = (sw_all_noaero_ds.coords['lon'].values + 180) % 360 - 180
sw_all_noaero_ds = sw_all_noaero_ds.sortby(sw_all_noaero_ds.lon)

aod_ds_sel = aod_ds.sel(time=slice('2010-01-02','2017-12-02'))
sw_all_aero_ds_sel = sw_all_aero_ds.sel(time=slice('2010-01-02','2017-12-02'))
sw_all_noaero_ds_sel = sw_all_noaero_ds.sel(time=slice('2010-01-02','2017-12-02'))

# Select variable
sw_all_aero_mon  = sw_all_aero_ds_sel["rsutcs"]
sw_all_noaero_mon  = sw_all_noaero_ds_sel["rsutcs"]
#sw_all_aero_mon  = sw_all_aero_ds_sel["rsut"]
#sw_all_noaero_mon  = sw_all_noaero_ds_sel["rsut"]
aod_mon = aod_ds_sel["od550aer"]

# Calculation
# Radiative Efficiency of AOD
delta_sw_on_aod = (sw_all_aero_mon-sw_all_noaero_mon)/aod_mon

# seasonaly mean each year
# yearly mean each year
delta_sw_on_aod_as = delta_sw_on_aod.sel(time=delta_sw_on_aod.time.dt.month.isin([7,8,9]))
delta_sw_on_aod_as_mean = delta_sw_on_aod_as.groupby('time.year').mean('time')
# mean year
delta_sw_on_aod_as_mean_mean = delta_sw_on_aod_as_mean.mean(dim='year')

# Curvilinear grid - method = bilinear - to fit with MODIS Linear Trend
dr_1deg = xr.Dataset(
    {
        "lat": (["lat"], np.arange(-89.5, 90, 1.0)),
        "lon": (["lon"], np.arange(-179.5, 180, 1.0)),
    }
)
regridder = xe.Regridder(delta_sw_on_aod_as_mean_mean, dr_1deg, "bilinear")
dsw_on_aod_regrid = regridder(delta_sw_on_aod_as_mean_mean)

# Africa
domain=['Africa']
lat_bnd = [-30, 10]
lon_bnd = [-25, 35]
dsw_on_aod = dsw_on_aod_regrid.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)

# Radiative Efficiency of OsloCTM AOD * MODIS AOD 
sw_aero_predict = dsw_on_aod * data_set.slope 

#Averaging the red box values
domain=['RedBox']
lat_red = [-20, -2.0]
lon_red = [-10, 10]
# Dask Array so np.nanmean
sw_aero_predict_red = sw_aero_predict.sel(lon=slice(*lon_red), lat=slice(*lat_red),)
# Rename coordinates
ds = sw_aero_predict_red.load()
ds = ds.rename({'lon': 'longitude','lat': 'latitude'})
### Weighted mean according to lat.
weights = np.cos(np.deg2rad(ds.latitude))
weights.name = "weights"
ds_lat_weighted = ds.weighted(weights)
ds_lat_weighted_mean = ds_lat_weighted.mean(("longitude","latitude"))
ds_lat_weighted_std = ds_lat_weighted.std(("longitude","latitude"))
print('sw_aero_pred_lat_mean:',ds_lat_weighted_mean)
print('sw_aero_pred_lat_std:',ds_lat_weighted_std)

dsw_on_aod_red = dsw_on_aod.sel(lon=slice(*lon_red), lat=slice(*lat_red),)
# Rename coordinates
ds2 = dsw_on_aod_red.load()
ds2 = dsw_on_aod_red.rename({'lon': 'longitude','lat': 'latitude'})
### Weighted mean according to lat.
weights = np.cos(np.deg2rad(ds.latitude))
weights.name = "weights"
ds2_lat_weighted = ds2.weighted(weights)
ds2_lat_weighted_mean = ds2_lat_weighted.mean(("longitude","latitude"))
ds2_lat_weighted_std = ds2_lat_weighted.std(("longitude","latitude"))
print('dsw_on_aod_lat_mean:',ds2_lat_weighted_mean)
print('dsw_on_aod_lat_std:',ds2_lat_weighted_std)

_, ax = make_figure()
# Keep the original linear cmap
cmap = mpl.cm.PRGn_r
# regrid 1D latitude and longitude to 2D grid
mm = ax.pcolormesh(sw_aero_predict.lon,\
                   sw_aero_predict.lat,\
                   sw_aero_predict*-1.0,\
                   transform=ccrs.PlateCarree(),\
                   vmin = -0.5, \
                   vmax = 0.5, \
                   cmap=cmap)

coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',     # ajout de la couche cotière 
                                facecolor='none', name='coastline') 
ax.add_feature(coast, edgecolor='black') 
# Add the gridlines
gl = ax.gridlines(draw_labels=True, color="black", linestyle="dotted")
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
# Add patch
# Africa
patches_A = []
zone_A = np.array([[-10.0,-20.0],[-10.0,-2.0],[10.0,-2.0],[10.0,-20.0]])
patches_A.append(Polygon(zone_A))
ax.add_collection(PatchCollection(patches_A, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidths=1.5))
cbar = plt.colorbar(mm, extend='both', shrink=0.35, orientation='horizontal')
cbar.set_label(label='W m$^{-2}$ yr$^{-1}$', fontsize=22)
cbar.ax.tick_params(labelsize=22) 
plt.title(r'DARE$_{\rm clr,OsloCTM3+MODIS}$',fontsize=22)
plt.savefig(fig_rep + 'Fig5c.eps', format='eps',bbox_inches='tight')
plt.savefig(fig_rep + 'Fig5c.jpeg', dpi=300,bbox_inches='tight')
plt.show()

_, ax2 = make_figure()
# Keep the original linear cmaP
cmap = mpl.cm.PRGn_r
mm = ax2.pcolormesh(dsw_on_aod.lon,\
                   dsw_on_aod.lat,\
                   dsw_on_aod*-1,\
                   transform=ccrs.PlateCarree(),\
                   vmin = -50, \
                   vmax = 50, \
                   cmap = 'seismic')

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
cbar = plt.colorbar(mm, extend='both', shrink=0.35, orientation='horizontal')
cbar.set_label(label='W m$^{-2}$ AOD$^{-1}$', fontsize=22)
cbar.ax.tick_params(labelsize=22)
plt.title('Clr AOD Radiative Efficiency',fontsize=22)
plt.savefig(fig_rep + 'Fig5a.eps', format='eps',bbox_inches='tight')
plt.savefig(fig_rep + 'Fig5a.jpeg', dpi=300,bbox_inches='tight')
plt.show()

