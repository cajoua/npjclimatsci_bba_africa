#data manipulation
import numpy as np
import xarray as xr

# data plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Proj, transform
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import copy

#warning
import warnings
warnings.filterwarnings("ignore")

###################
# 1- Download multiple MODIS Terra and Aqua netcdf files
# 2- Select time date range 
# 3- Combine Terra and Aqua data (mean)
# 4- Yearly seasonal spatial mean
###################

def make_figure():
    fig=plt.figure(figsize=(18,8), frameon=True)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                                facecolor='none', name='coastline')
    ax.add_feature(coast, edgecolor='black')
    return fig, ax


# Open sevral Netcdf files 
modis_y_var_dir='path-to-data'
modis_y_var_ds = xr.open_mfdataset(modis_y_var_dir)
modis_o_var_dir='path-to-data'
modis_o_var_ds = xr.open_mfdataset(modis_o_var_dir)

# Select time date range
modis_y_var_ds_sel = modis_y_var_ds.sel(time=slice('2002-03-01','2022-03-01'))
modis_o_var_ds_sel = modis_o_var_ds.sel(time=slice('2002-03-01','2022-03-01'))

del modis_y_var_ds, modis_o_var_ds

# Select variable
modis_y_monthly_var = modis_y_var_ds_sel["MYD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean"]
modis_o_monthly_var = modis_o_var_ds_sel["MOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean"]

del modis_y_var_ds_sel, modis_o_var_ds_sel

# Merge the Terra and Aqua modis DataArray
modis_yo_monthly_var = xr.merge([modis_y_monthly_var,modis_o_monthly_var])
mean_var = modis_yo_monthly_var.to_array(dim='new').mean('new',skipna=True)
modis_yom_monthly_var = modis_yo_monthly_var.assign(MYOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean=mean_var)
modis_monthly_var = modis_yom_monthly_var["MYOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean"]

# Select region
# Africa
domain=['Africa']
lat_bnd = [-30, 10]
lon_bnd = [-25, 35]

modis_ds_p0 = modis_monthly_var.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)

# yearly mean each year
modis_ds_jaso_p0  = modis_ds_p0.sel(time=modis_ds_p0.time.dt.month.isin([7, 8, 9]))
# Check the missing data in the bounded area
nan_counts = modis_ds_jaso_p0.isnull().sum(dim='time')
# Mask where nan_count > 5
modis_ds_jaso_p0 = modis_ds_jaso_p0.where(nan_counts <= 5)
#lat_bnd = [-20, 2]
#lon_bnd = [-10, 10]
#nan_counts_zoom = nan_counts.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)
#import matplotlib.pyplot as plt
#cmap = plt.get_cmap('jet',50)
#mm=plt.pcolormesh(nan_counts_zoom,cmap=cmap)
#plt.colorbar(mm)
modis_ds_jaso_p0m = modis_ds_jaso_p0.groupby('time.year').mean('time',skipna=True)

# Total mean for each series
modis_ds_y_p0m = modis_ds_jaso_p0m.mean(dim='year',skipna=False)

# Figure repertory
fig_rep = 'path-to-figure-repertory'

_, ax = make_figure()
# Get the colormap and set the under and bad colors (gist_rainbow is to reverse the color bar)
cmap0 = copy.copy(plt.cm.jet)
# Add the gridlines
gl = ax.gridlines(draw_labels=True, color="black", linestyle="dotted")
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
levels=np.arange(0,1.0,0.05)
# Make the figure
mm=plt.pcolormesh(modis_ds_y_p0m.lon, \
            modis_ds_y_p0m.lat, \
            modis_ds_y_p0m, \
            transform=ccrs.PlateCarree(), \
            vmin = levels[0], \
            vmax = levels[-1], \
            cmap=cmap0)

# Setting specific geographical extent
ax.set_extent([-27, 37, -32, 12], crs=ccrs.PlateCarree())

# Add patch
# Bounded area
patches_A = []
zone_A = np.array([[-10.0,-20.0],[-10.0,-2.0],[10.0,-2.0],[10.0,-20.0]])
patches_A.append(Polygon(zone_A))
ax.add_collection(PatchCollection(patches_A, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='white', linewidths=1.5))
# Add a color bar
cbar = plt.colorbar(mm, extend='both', shrink=0.35, orientation='horizontal')
cbar.set_label('AOD', fontsize=18)
cbar.ax.tick_params(labelsize=22)
plt.savefig(fig_rep + 'Figure_1a.eps', format='eps',bbox_inches='tight')
plt.savefig(fig_rep + 'Figure_1a.jpeg', format='jpeg',dpi=300,bbox_inches='tight')
plt.show()
