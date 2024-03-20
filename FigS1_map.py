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
import matplotlib as mpl
import copy

#warning
import warnings
warnings.filterwarnings("ignore")

###################
# 1- Download multiple MODIS Terra and Aqua netcdf files
# 2- Select time date range 
# 3- Combine Terra and Aqua data (mean)
# 4- Spatial distribution of seasonal means of AOD and CF
# 5- Map
###################

def make_figure():
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                                facecolor='none', name='coastline')
    ax.add_feature(coast, edgecolor='black')
    return fig, ax


# Open several Netcdf files 
modis_y_var_dir='path-to-AOD-modis-input'
modis_y_var_ds = xr.open_mfdataset(modis_y_var_dir)
modis_o_var_dir='path-to-AOD-modis-input'
modis_o_var_ds = xr.open_mfdataset(modis_o_var_dir)

modis_crfl_dir='path-to-CF-modis-input'
modis_crfl_ds = xr.open_mfdataset(modis_crfl_dir)

# Select time date range
modis_y_var_ds_sel = modis_y_var_ds.sel(time=slice('2002-03-01','2022-03-01'))
modis_o_var_ds_sel = modis_o_var_ds.sel(time=slice('2002-03-01','2022-03-01'))
modis_crfl_ds_sel = modis_crfl_ds.sel(time=slice('2002-03-01','2022-03-01'))

del modis_y_var_ds, modis_o_var_ds

# Select variable
modis_y_monthly_var = modis_y_var_ds_sel["MYD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean"]
modis_o_monthly_var = modis_o_var_ds_sel["MOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean"]
modis_monthly_crfl = modis_crfl_ds_sel["MYOD08_M3_7_0_Cloud_Retrieval_Fraction_Liquid_Mean"]

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
modis_crfl_p0 = modis_monthly_crfl.sel(lon=slice(*lon_bnd), lat=slice(*lat_bnd),)

# yearly mean each year
modis_ds_jaso_p0  = modis_ds_p0.sel(time=modis_ds_p0.time.dt.month.isin([7, 8, 9]))
# Check the missing data in the bounded area
nan_counts = modis_ds_jaso_p0.isnull().sum(dim='time')
# Mask where nan_count > 5
modis_ds_jaso_p0 = modis_ds_jaso_p0.where(nan_counts <= 5)
modis_ds_jaso_p0m = modis_ds_jaso_p0.groupby('time.year').mean('time',skipna=True) 

modis_crfl_jaso_p0  = modis_crfl_p0.sel(time=modis_crfl_p0.time.dt.month.isin([7, 8, 9]))
# Check the missing data in the bounded area
nan_counts2 = modis_crfl_jaso_p0.isnull().sum(dim='time')
# Mask where nan_count > 5
modis_crfl_jaso_p0 = modis_crfl_jaso_p0.where(nan_counts2 <= 5)
modis_crfl_jaso_p0m = modis_crfl_jaso_p0.groupby('time.year').mean('time',skipna=True) 

# Total mean for each series
modis_ds_y_p0m = modis_ds_jaso_p0m.mean(dim='year',skipna=False)
modis_crfl_y_p0m = modis_crfl_jaso_p0m.mean(dim='year',skipna=False)

# Figure repertory
fig_rep = 'path-to-figure-repertory'

fig, ax = make_figure()
# Get the colormap and set the under and bad colors (gist_rainbow is to reverse the color bar)
Y=np.array([[185,185,185],[195,195,195],[205,205,205],[215,215,215],[225,225,225],[235,235,235],[245,245,245],[250,250,250],[255,255,255]])/255.
cmap=mpl.colors.ListedColormap(Y)
bounds = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)#

cmap0 = copy.copy(plt.cm.YlOrRd)

# Add the gridlines
gl = ax.gridlines(draw_labels=True, color="black", linestyle="dotted")
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}

# Make the figure
test=modis_crfl_y_p0m.values
test[test < 0.5] = np.nan

#Plot CF
cf1 = ax.contourf(modis_crfl_y_p0m.lon, \
                   modis_crfl_y_p0m.lat, \
                   test, \
                   transform=ccrs.PlateCarree(), \
                   vmin = 0.5, \
                   vmax = 1.05, \
                   norm=norm, \
                   cmap=cmap)

#Plot AOD
cf2 = ax.contour(modis_ds_y_p0m.lon, \
               modis_ds_y_p0m.lat, \
               modis_ds_y_p0m, \
               transform=ccrs.PlateCarree(), \
               vmin = 0.0, \
               vmax = 1.0, \
               linewidths=2, \
               linestyles='solid', \
               cmap=cmap0)

# Add a color bar
# First add more white space below the figure
fig.subplots_adjust(bottom=0.2)
# Find the position (0,0,0,0) is corner bottom left
# rect: left bottom width height
cax1 = fig.add_axes([0.23, 0.12, 0.25, 0.025])
cb1 = fig.colorbar(cf1, cax=cax1, orientation = 'horizontal')
cb1.set_label('CF', fontsize=22)
cb1.ax.tick_params(labelsize=20,rotation=45)
cax2 = fig.add_axes([0.52, 0.12, 0.25, 0.025])
cb2 = fig.colorbar(cf2, cax=cax2, orientation = 'horizontal')
# To width the contour colorbar
cb2.lines[0].set_linewidth(18)
cb2.set_label('AOD', fontsize=22)
cb2.ax.tick_params(labelsize=20,rotation=45)

# Add patch
# Bounded Area
patches_A = []
zone_A = np.array([[-10.0,-20.0],[-10.0,-2.0],[10.0,-2.0],[10.0,-20.0]])
patches_A.append(Polygon(zone_A))
ax.add_collection(PatchCollection(patches_A, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidths=1.5))

# Title
plt.savefig(fig_rep + 'Figure_S1.eps', format='eps',bbox_inches='tight')
plt.savefig(fig_rep + 'Figure_S1.jpeg', format='jpeg',dpi=300,bbox_inches='tight')
plt.show()
