#data manipulation
import numpy as np
import xarray as xr
import matplotlib.ticker as ticker

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
data_rep = 'path-to-trend-repertory'
    
# Loop over variable name
data_name_CLD_perc = ['MYOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean_TrendMod_percmean_AFR_1deg_JAS_2002_2021.nc',\
                      'MYOD08_M3_7_0_Cloud_Particle_Size_Liquid_Mean_TrendMod_percmean_AFR_1deg_JAS_2002_2021.nc',\
                      'MYOD08_M3_7_0_Cloud_Optical_Thickness_Liquid_Mean_TrendMod_percmean_AFR_1deg_JAS_2002_2021.nc',\
                      'MYOD08_M3_7_0_Cloud_Retrieval_Fraction_Liquid_Mean_TrendMod_percmean_AFR_1deg_JAS_2002_2021.nc',\
                      'MYOD08_M3_7_0_Cloud_Water_Path_Liquid_Mean_TrendMod_percmean_AFR_1deg_JAS_2002_2021.nc']
    
data_name_CLD_unit = ['MYOD08_M3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc',\
                      'MYOD08_M3_7_0_Cloud_Particle_Size_Liquid_Mean_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc',\
                      'MYOD08_M3_7_0_Cloud_Optical_Thickness_Liquid_Mean_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc',\
                      'MYOD08_M3_7_0_Cloud_Retrieval_Fraction_Liquid_Mean_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc',\
                      'MYOD08_M3_7_0_Cloud_Water_Path_Liquid_Mean_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc']

data_name_TOA = ['CERES_TOA_SW_ALL_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc',\
                 'CERES_TOA_SW_CLR_TrendMod_unit_AFR_1deg_JAS_2002_2021.nc']
    
data_range_CLD_perc = np.array([[-4.0,4.0],[-2.0,2.0],[-2.0,2.0],[-2.0,2.0],[-2.0,2.0]])
data_range_CLD_unit = np.array([[-5e-3,5e-3],[-7e-2,7e-2],[-10e-2,10e-2],[-5e-3,5e-3],[-1.5,1.5]])
data_range_TOA =  np.array([[-1.0,1.0],[-0.4,0.4]])
title_CLD_name_unit = ('','Reff','COT','CF','LWP')
title_CLD_name_perc = ('AOD','Reff','COT','CF','LWP')
title_CLD_unit = ('unit yr$^{-1}$','$\mu$m yr$^{-1}$','unit yr$^{-1}$','unit yr$^{-1}$','g m$^{-2}$ yr$^{-1}$')
title_TOA_name = (r'SWR$_{\rm all}$',r'DARE$_{\rm clr}$')


# Figure repertory
fig_rep = 'path-to-figure-repertory'

i=0
# Lopp over experiments:
for dname in data_name_CLD_perc:
    i=i+1
    ncfile = data_rep + data_name_CLD_perc[i-1]
    data_set = xr.open_dataset(ncfile)
    print(dname)
    #Averaging the red box values
    domain=['RedBox']
    lat_red = [-20, -2.0]
    lon_red = [-10, 10]
    data_sel = data_set.sel(x=slice(*lon_red), y=slice(*lat_red),)

    # Rename coordinates
    ds = data_sel.rename({'x': 'longitude','y': 'latitude'})
    ### Weighted mean according to lat.
    weights = np.cos(np.deg2rad(ds.latitude))
    weights.name = "weights"
    ds_lat_weighted = ds.weighted(weights)
    ds_lat_weighted_mean = ds_lat_weighted.mean(("longitude","latitude"))
    ds_lat_weighted_std = ds_lat_weighted.std(("longitude","latitude"))
    print('weighted_lat_mean:',ds_lat_weighted_mean)
    print('weighted_lat_std:',ds_lat_weighted_std)

    _, ax = make_figure()
    # Keep the original linear cmap
    cmap = mpl.cm.PRGn_r
    lon2d, lat2d = np.meshgrid(data_set.x, data_set.y)
    p_ones = xr.ones_like(data_set.p)
    p_mask = p_ones.where(data_set.p < 0.05)
    lon2d_p = lon2d*p_mask
    lat2d_p = lat2d*p_mask
    gl = ax.gridlines(draw_labels=True, color="black", linestyle="dotted")
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}    

    vminmax = data_range_CLD_perc[i-1] 
    mm = ax.pcolormesh(data_set.x,\
                   data_set.y,\
                   data_set.slope,\
                   transform=ccrs.PlateCarree(),\
                   vmin = vminmax[0], \
                   vmax = vminmax[1], \
                   cmap=cmap) 
    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',     # ajout de la couche cotière 
                                facecolor='none', name='coastline') 
    ax.add_feature(coast, edgecolor='black') 
    ax.scatter(lon2d_p, lat2d_p, transform=ccrs.PlateCarree(), marker='x', color='black', s=30, linewidths=0.5)
    # Setting specific geographical extent
    ax.set_extent([-27, 37, -32, 12], crs=ccrs.PlateCarree())
    # Add patch
    # Africa
    patches_A = []
    zone_A = np.array([[-10.0,-20.0],[-10.0,-2.0],[10.0,-2.0],[10.0,-20.0]])
    patches_A.append(Polygon(zone_A))
    ax.add_collection(PatchCollection(patches_A, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidths=1.5))

    cbar = plt.colorbar(mm, extend='both', shrink=0.35, orientation='horizontal')
    cbar.set_label('% yr$^{-1}$', fontsize=22)
    cbar.ax.tick_params(labelsize=22) 
    plt.title(title_CLD_name_perc[i-1],fontsize=32)
    plt.savefig(fig_rep + data_name_CLD_perc[i-1][0:-3] + '.eps', format='eps',bbox_inches='tight')
    plt.savefig(fig_rep + data_name_CLD_perc[i-1][0:-3] + '.jpeg', dpi=300,bbox_inches='tight')
plt.show()

j=0
# Lopp over experiments:
for dname in data_name_CLD_unit:
    j=j+1
    ncfile = data_rep + data_name_CLD_unit[j-1]
    data_set = xr.open_dataset(ncfile)
    print(dname)
    #Averaging the red box values
    domain=['RedBox']
    lat_red = [-20, -2.0]
    lon_red = [-10, 10]
    data_sel = data_set.sel(x=slice(*lon_red), y=slice(*lat_red),)

    # Rename coordinates
    ds = data_sel.rename({'x': 'longitude','y': 'latitude'})
    ### Weighted mean according to lat.
    weights = np.cos(np.deg2rad(ds.latitude))
    weights.name = "weights"
    ds_lat_weighted = ds.weighted(weights)
    ds_lat_weighted_mean = ds_lat_weighted.mean(("longitude","latitude"))
    ds_lat_weighted_std = ds_lat_weighted.std(("longitude","latitude"))
    print('weighted_lat_mean:',ds_lat_weighted_mean)
    print('weighted_lat_std:',ds_lat_weighted_std)

    _, ax = make_figure()
    # Keep the original linear cmap
    cmap = mpl.cm.PRGn_r
    lon2d, lat2d = np.meshgrid(data_set.x, data_set.y)
    p_ones = xr.ones_like(data_set.p)
    p_mask = p_ones.where(data_set.p < 0.05)
    lon2d_p = lon2d*p_mask
    lat2d_p = lat2d*p_mask
    gl = ax.gridlines(draw_labels=True, color="black", linestyle="dotted")
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}

    vminmax = data_range_CLD_unit[j-1]
    mm = ax.pcolormesh(data_set.x,\
                   data_set.y,\
                   data_set.slope,\
                   transform=ccrs.PlateCarree(),\
                   vmin = vminmax[0], \
                   vmax = vminmax[1], \
                   cmap=cmap)
    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',     # ajout de la couche cotière 
                                facecolor='none', name='coastline')
    ax.add_feature(coast, edgecolor='black')
    ax.scatter(lon2d_p, lat2d_p, transform=ccrs.PlateCarree(), marker='x', color='black', s=30, linewidths=0.5)
    # Setting specific geographical extent
    ax.set_extent([-27, 37, -32, 12], crs=ccrs.PlateCarree())
    # Add patch
    # Africa
    patches_A = []
    zone_A = np.array([[-10.0,-20.0],[-10.0,-2.0],[10.0,-2.0],[10.0,-20.0]])
    patches_A.append(Polygon(zone_A))
    ax.add_collection(PatchCollection(patches_A, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidths=1.5))
    cbar = plt.colorbar(mm, extend='both', shrink=0.35, orientation='horizontal')
    cbar.set_label(title_CLD_unit[j-1], fontsize=22)
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.xaxis.get_major_formatter().set_powerlimits((0, 1))   
    cbar.ax.tick_params(labelsize=22)
    # to get 10^3 instead of 1e3
    cbar.formatter.set_useMathText(True)
    cbar.ax.xaxis.get_offset_text().set_fontsize(22)
    plt.title(title_CLD_name_unit[j-1],fontsize=32)
    plt.savefig(fig_rep + data_name_CLD_unit[j-1][0:-3] + '.eps' , format='eps',bbox_inches='tight')
    plt.savefig(fig_rep + data_name_CLD_unit[j-1][0:-3] + '.jpeg' , dpi=300, bbox_inches='tight')
plt.show()

k=0
# Lopp over experiments:
for dname in data_name_TOA:
    k=k+1
    ncfile = data_rep + data_name_TOA[k-1]
    data_set = xr.open_dataset(ncfile)
    print(dname)
    #Averaging the red box values
    domain=['RedBox']
    lat_red = [-20, -2.0]
    lon_red = [-10, 10]
    data_sel = data_set.sel(x=slice(*lon_red), y=slice(*lat_red),)

    # Rename coordinates
    ds = data_sel.rename({'x': 'longitude','y': 'latitude'})
    ### Weighted mean according to lat.
    weights = np.cos(np.deg2rad(ds.latitude))
    weights.name = "weights"
    ds_lat_weighted = ds.weighted(weights)
    ds_lat_weighted_mean = ds_lat_weighted.mean(("longitude","latitude"))
    ds_lat_weighted_std = ds_lat_weighted.std(("longitude","latitude"))
    print('weighted_lat_mean:',ds_lat_weighted_mean)
    print('weighted_lat_std:',ds_lat_weighted_std)

    _, ax = make_figure()
    # Keep the original linear cmap
    cmap = mpl.cm.PRGn_r
    lon2d, lat2d = np.meshgrid(data_set.x, data_set.y)
    p_ones = xr.ones_like(data_set.p)
    p_mask = p_ones.where(data_set.p < 0.05)
    lon2d_p = lon2d*p_mask
    lat2d_p = lat2d*p_mask

    gl = ax.gridlines(draw_labels=True, color="black", linestyle="dotted")
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}

    vminmax = data_range_TOA[k-1]
    mm = ax.pcolormesh(data_set.x,\
                   data_set.y,\
                   data_set.slope*-1.0,\
                   transform=ccrs.PlateCarree(),\
                   vmin = vminmax[0], \
                   vmax = vminmax[1], \
                   cmap=cmap)
    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',     # ajout de la couche cotière 
                                facecolor='none', name='coastline')
    ax.add_feature(coast, edgecolor='black')
    ax.scatter(lon2d_p, lat2d_p, transform=ccrs.PlateCarree(), marker='x', color='black', s=30, linewidths=0.5)
    # Setting specific geographical extent
    ax.set_extent([-27, 37, -32, 12], crs=ccrs.PlateCarree())
    # Add patch
    # Africa
    patches_A = []
    zone_A = np.array([[-10.0,-20.0],[-10.0,-2.0],[10.0,-2.0],[10.0,-20.0]])
    patches_A.append(Polygon(zone_A))
    ax.add_collection(PatchCollection(patches_A, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidths=1.5))

    cbar = plt.colorbar(mm, extend='both', shrink=0.35, orientation='horizontal')
    cbar.set_label('W m$^{-2}$ yr$^{-1}$', fontsize=22)
    cbar.ax.tick_params(labelsize=22)
    plt.title(title_TOA_name[k-1],fontsize=32)
    plt.savefig(fig_rep + data_name_TOA[k-1][0:-3] + '_opposit.eps' , format='eps',bbox_inches='tight')
    plt.savefig(fig_rep + data_name_TOA[k-1][0:-3] + '_opposit.jpeg' , dpi=300, bbox_inches='tight')
plt.show()
