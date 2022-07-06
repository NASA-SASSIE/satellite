#!/usr/bin/env python
#
#Severine Fournier - June 2022
#
##############################################
## Download and save locally:
##     - AMSR Sea ice concentration
##     - MASIE Sea ice extent
##     - OISST
##     - SMAP JPL SSS
##     - SMOS LOCEAN Arctic SSS
##     - CCMP winds
##     - Aviso SLA
##
## Plot these data in the Beaufort Sea and save the figures locally
## When force is:
##     - False, the files won't be redownloaded if they already exist locally and the figures won't be recreated if they already exist
##     - True, the files will be redownloaded and the figures recreated no matter if they already exist locally or not
##############################################
    
import numpy
import glob
import xarray as xr
from pathlib import Path
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

######### DEFINE PARAMETERS ###################

#day
year=2021
month=9
day=1

#force (true) or not (false) the redownload of the data and the creation of the figures (even if any already exists)
force=True

#region for data selection
lonmin=-180
lonmax=-100
latmin=60
latmax=90

#region for maps
lon0=-150
lonmapmin=-165
lonmapmax=-120
latmapmin=68
latmapmax=78

#paths
# SEVERINE
rawdata_path=Path('/Users/severinf/Data/SASSIE/satellite/') 
figures_path=Path('/Users/severinf/Figures/SASSIE/cruise/') 

# KYLA CLOUD
# rawdata_path=Path('/home/jovyan/data/SASSIE/satellite/')
# figures_path=Path('/home/jovyan/figures/SASSIE')





######## FUNCTIONS ##########################

def z_masked_overlap(axe, X, Y, Z, source_projection=None):
    """
    for data in projection axe.projection
    find and mask the overlaps (more 1/2 the axe.projection range)
    X, Y either the coordinates in axe.projection or longitudes latitudes
    Z the data
    operation one of 'pcorlor', 'pcolormesh', 'countour', 'countourf'
    if source_projection is a geodetic CRS data is in geodetic coordinates
    and should first be projected in axe.projection
    X, Y are 2D same dimension as Z for contour and contourf
    same dimension as Z or with an extra row and column for pcolor
    and pcolormesh
    return ptx, pty, Z
    """
    if not hasattr(axe, 'projection'):
        return Z
    if not isinstance(axe.projection, cartopy.crs.Projection):
        return Z
    if len(X.shape) != 2 or len(Y.shape) != 2:
        return Z

    if (source_projection is not None and
            isinstance(source_projection, cartopy.crs.Geodetic)):
        transformed_pts = axe.projection.transform_points(
            source_projection, X, Y)
        ptx, pty = transformed_pts[..., 0], transformed_pts[..., 1]
    else:
        ptx, pty = X, Y
    with numpy.errstate(invalid='ignore'):
        # diagonals have one less row and one less columns
        diagonal0_lengths = numpy.hypot(
            ptx[1:, 1:] - ptx[:-1, :-1],
            pty[1:, 1:] - pty[:-1, :-1]
        )
        diagonal1_lengths = numpy.hypot(
            ptx[1:, :-1] - ptx[:-1, 1:],
            pty[1:, :-1] - pty[:-1, 1:]
        )
        to_mask = (
            (diagonal0_lengths > (
                abs(axe.projection.x_limits[1]
                    - axe.projection.x_limits[0])) / 2) |
            numpy.isnan(diagonal0_lengths) |
            (diagonal1_lengths > (
                abs(axe.projection.x_limits[1]
                    - axe.projection.x_limits[0])) / 2) |
            numpy.isnan(diagonal1_lengths)
        )
        # TODO check if we need to do something about surrounding vertices
        # add one extra colum and row for contour and contourf
        if (to_mask.shape[0] == Z.shape[0] - 1 and
                to_mask.shape[1] == Z.shape[1] - 1):
            to_mask_extended = numpy.zeros(Z.shape, dtype=bool)
            to_mask_extended[:-1, :-1] = to_mask
            to_mask_extended[-1, :] = to_mask_extended[-2, :]
            to_mask_extended[:, -1] = to_mask_extended[:, -2]
            to_mask = to_mask_extended
        if numpy.any(to_mask):
            Z_mask = getattr(Z, 'mask', None)
            to_mask = to_mask if Z_mask is None else to_mask | Z_mask
            Z = ma.masked_where(to_mask, Z)
        return ptx, pty, Z
          

def map(x,y,data,vmin,vmax,**karg):
    
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=cartopy.crs.NorthPolarStereo(central_longitude=lon0))
    ax.set_extent([lonmapmin, lonmapmax, latmapmin, latmapmax], crs=cartopy.crs.PlateCarree()) 
    if karg['land']:
        ax.add_feature(cfeature.LAND, facecolor = '0.75',zorder=1)
    if karg['coastline']:
            ax.coastlines('10m',zorder=2)
    ax.add_feature(cfeature.RIVERS,facecolor='blue',zorder=3)
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.xlocator = mticker.FixedLocator(numpy.arange(lonmapmin-15,lonmapmax+15,15))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {'size': 14, 'color': 'k','rotation':0}
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylocator = mticker.FixedLocator(numpy.arange(latmapmin-15,latmapmax+15,2))
    gl.ylabel_style = {'size': 14, 'color': 'gray','rotation':0}

    pp = ax.pcolormesh(x, y, data, 
                        vmin=vmin, vmax=vmax,  # Set max and min values for plotting
                        cmap=karg['palette'], shading='auto',   # shading='auto' to avoid warning
                        transform=ccrs.PlateCarree())  # coords are lat,lon but map if NPS
    if 'level_contour' in karg:  
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        X, Y, masked_data = z_masked_overlap(ax, x.values, y.values, data.values, source_projection=cartopy.crs.Geodetic())    
        cc=ax.contour(X,Y,masked_data,levels=[karg['level_contour']],colors='m',linewidth=3)
        fmt = {}
        plt.clabel(cc, cc.levels, fmt=' {:.0f} '.format, inline=True, fontsize=10,colors='m')
    if 'u' in karg:  
        q=plt.quiver(x[::5].values,y[::2].values,karg['u'][::2,::5].values,karg['v'][::2,::5].values,scale=karg['scale'],transform=cartopy.crs.PlateCarree())
        qk= plt.quiverkey (q,0.95,-0.07,10,'10'+karg['unit_vector'],labelpos='N')
                    
    cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.04])
    h=plt.colorbar(pp, cax=cbar_ax,orientation='horizontal',ax=ax)
    h.ax.tick_params(labelsize=20)
    h.set_label(karg['unit'],fontsize=20)
    cmin,cmax = h.mappable.get_clim()
    ax.set_title(karg['title'],fontsize=20)
    plt.subplots_adjust(right=0.9,left=0.1,top=0.9,bottom=0.18)
    plt.savefig(karg['fileout'], dpi=fig.dpi)
    
    

######## DOWNLOAD Data ######################

my_datetime = datetime.datetime(year,month,day)
doy = my_datetime.strftime('%j')


#AMSR Sea ice
os.system('mkdir -p '+str(rawdata_path)+'/seaice_amsr')
filename='AMSR_U2_L3_SeaIce12km_B04_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.he5'
if force==True:
    os.system('wget -0 -P '+str(rawdata_path)+'/seaice_amsr/'+' https://n5eil01u.ecs.nsidc.org/AMSA/AU_SI12.001/'+str(year)+'.'+str(month).zfill(2)+'.'+str(day).zfill(2)+'/'+filename)   
else:
    if os.path.isfile(str(rawdata_path)+'/seaice_amsr/'+filename)==False:
        os.system('wget -P '+str(rawdata_path)+'/seaice_amsr/'+' https://n5eil01u.ecs.nsidc.org/AMSA/AU_SI12.001/'+str(year)+'.'+str(month).zfill(2)+'.'+str(day).zfill(2)+'/'+filename)   


#MASIE
os.system('mkdir -p '+str(rawdata_path)+'/iceextent_masie')
if force==True:
    os.system('wget -0 -P '+str(rawdata_path)+'/iceextent_masie/'+' https://masie_web.apps.nsidc.org/pub/DATASETS/NOAA/G02186/ancillary/masie_lat_lon_4km.nc')
else:
    if os.path.isfile(str(rawdata_path)+'/iceextent_masie/masie_lat_lon_4km.nc')==False:
        os.system('wget -P '+str(rawdata_path)+'/iceextent_masie/'+' https://masie_web.apps.nsidc.org/pub/DATASETS/NOAA/G02186/ancillary/masie_lat_lon_4km.nc')

filename='masie_all_r00_v01_'+str(year)+doy+'_4km.nc'
if force==True:
    os.system('wget -0 -P '+str(rawdata_path)+'/iceextent_masie/'+' https://masie_web.apps.nsidc.org/pub/DATASETS/NOAA/G02186/netcdf/4km/'+str(year)+'/'+filename)   
else:
    if os.path.isfile(str(rawdata_path)+'/iceextent_masie/'+filename)==False:
        os.system('wget -P '+str(rawdata_path)+'/iceextent_masie/'+' https://masie_web.apps.nsidc.org/pub/DATASETS/NOAA/G02186/netcdf/4km/'+str(year)+'/'+filename)   


#OISST L4
os.system('mkdir -p '+str(rawdata_path)+'/sst_oi')
filename='oisst-avhrr-v02r01.'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.nc'
if force==True:
    os.system('wget -0 -P '+str(rawdata_path)+'/sst_oi/'+' https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/'+str(year)+str(month).zfill(2)+'/'+filename)   
else:
    if os.path.isfile(str(rawdata_path)+'/sst_oi/'+filename)==False:
        os.system('wget -P '+str(rawdata_path)+'/sst_oi/'+' https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/'+str(year)+str(month).zfill(2)+'/'+filename)   


#SMAP JPL SSS L3 (file for 09/01 will be in the folder of 09/05 (8 day running mean))
os.system('mkdir -p '+str(rawdata_path)+'/sss_smapjpl')
filename='SMAP_L3_SSS_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_8DAYS_V5.0.nc'
if force==True:
    os.system('wget -0 -P '+str(rawdata_path)+'/sss_smapjpl/'+' https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5/'+str(year)+'/'+str(int(doy)-4)+'/'+filename)
else:
    if os.path.isfile(str(rawdata_path)+'/sss_smapjpl/'+filename)==False:
        os.system('wget -P '+str(rawdata_path)+'/sss_smapjpl/'+' https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5/'+str(year)+'/'+str(int(doy)-4)+'/'+filename)
        

#SMOS SSS L3
os.system('mkdir -p '+str(rawdata_path)+'/sss_smos')
filename='SMOS-arctic-LOCEAN-SSS-'+str(year)+'-'+str(month).zfill(2)+'-'+str(day).zfill(2)+'-v1.1AT-7days.nc'
if force==True:
    os.system('wget -0 -P '+str(rawdata_path)+'/sss_smos/'+' ftp://ext-catds-cecos-locean@ftp.ifremer.fr/Ocean_products/SMOS_ARCTIC_SSS_L3_LOCEAN/netcdf_weekly_v1_1/'+filename)   
else:
    if os.path.isfile(str(rawdata_path)+'/sss_smos/'+filename)==False:
        os.system('wget -P '+str(rawdata_path)+'/sss_smos/'+' ftp://ext-catds-cecos-locean@ftp.ifremer.fr/Ocean_products/SMOS_ARCTIC_SSS_L3_LOCEAN/netcdf_weekly_v1_1/'+filename)   


#CCMP winds
os.system('mkdir -p '+str(rawdata_path)+'/wind_ccmp')
filename_dt='CCMP_Wind_Analysis_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_V02.0_L3.0_RSS.nc'
filename_nrt='CCMP_RT_Wind_Analysis_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_V02.1_L3.0_RSS.nc'
if force==True:
    os.system('wget -0 -P '+str(rawdata_path)+'/wind_ccmp/'+' https://data.remss.com/ccmp/v02.0/Y'+str(year)+'/M'+str(month).zfill(2)+'/'+filename_dt)   
    if len(glob.glob(str(rawdata_path)+'/wind_ccmp/'+filename_dt))<1:
        os.system('wget -0 -P '+str(rawdata_path)+'/wind_ccmp/'+' https://data.remss.com/ccmp/v02.1.NRT/Y'+str(year)+'/M'+str(month).zfill(2)+'/'+filename_nrt)   
else:
    if len(glob.glob(str(rawdata_path)+'/wind_ccmp/'+filename_dt))<1:
        os.system('wget -P '+str(rawdata_path)+'/wind_ccmp/'+' https://data.remss.com/ccmp/v02.0/Y'+str(year)+'/M'+str(month).zfill(2)+'/'+filename_dt)   
        if len(glob.glob(str(rawdata_path)+'/wind_ccmp/'+filename_dt))<1 and len(glob.glob(str(rawdata_path)+'/wind_ccmp/'+filename_nrt))<1:
            os.system('wget -P '+str(rawdata_path)+'/wind_ccmp/'+' https://data.remss.com/ccmp/v02.1.NRT/Y'+str(year)+'/M'+str(month).zfill(2)+'/'+filename_nrt)   
if len(glob.glob(str(rawdata_path)+'/wind_ccmp/'+filename_dt))>0 and len(glob.glob(str(rawdata_path)+'/wind_ccmp/'+filename_nrt))>0:
    os.system('rm '+str(rawdata_path)+'/wind_ccmp/'+filename_nrt)


#AVISO SLA
os.system('mkdir -p '+str(rawdata_path)+'/sla_aviso')
filename_dt='dt_global_allsat_phy_l4_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_*nc'
filename_nrt='nrt_global_allsat_phy_l4_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_*nc'
if force==True:
    os.system('wget -0 -P '+str(rawdata_path)+'/sla_aviso/'+' ftp://my.cmems-du.eu/Core/SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D/'+str(year)+'/'+str(month).zfill(2)+'/'+filename_dt)   
    if len(glob.glob(str(rawdata_path)+'/sla_aviso/'+filename_dt))<1:
        os.system('wget -0 -P '+str(rawdata_path)+'/sla_aviso/'+' ftp://nrt.cmems-du.eu/Core/SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046/dataset-duacs-nrt-global-merged-allsat-phy-l4/'+str(year)+'/'+str(month).zfill(2)+'/'+filename_nrt)   
else:
    if len(glob.glob(str(rawdata_path)+'/sla_aviso/'+filename_dt))<1:
        os.system('wget -P '+str(rawdata_path)+'/sla_aviso/'+' ftp://my.cmems-du.eu/Core/SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D/'+str(year)+'/'+str(month).zfill(2)+'/'+filename_dt)   
        if len(glob.glob(str(rawdata_path)+'/sla_aviso/'+filename_dt))<1 and len(glob.glob(str(rawdata_path)+'/sla_aviso/'+filename_nrt))<1:
            os.system('wget -P '+str(rawdata_path)+'/sla_aviso/'+' ftp://nrt.cmems-du.eu/Core/SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046/dataset-duacs-nrt-global-merged-allsat-phy-l4/'+str(year)+'/'+str(month).zfill(2)+'/'+filename_nrt)   
if len(glob.glob(str(rawdata_path)+'/sla_aviso/'+filename_dt))>0 and len(glob.glob(str(rawdata_path)+'/sla_aviso/'+filename_nrt))>0:
    os.system('rm '+str(rawdata_path)+'/sla_aviso/'+filename_nrt)
        
        

######## Load and plot Data ######################
        
#AMSR Sea ice
if os.path.isfile(str(figures_path)+'/mapbeaufort_sic_amsr_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.png')==False:
    filename = str(rawdata_path)+'/seaice_amsr/AMSR_U2_L3_SeaIce12km_B04_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.he5'
    if os.path.isfile(filename):
        dsc = xr.open_dataset(filename,group='HDFEOS/GRIDS/NpPolarGrid12km')
        dsd = xr.open_dataset(filename,group='HDFEOS/GRIDS/NpPolarGrid12km/Data Fields')

        map(dsc.lon,dsc.lat,dsd.SI_12km_NH_ICECON_DAY.squeeze(),0,100,
            palette='Blues_r',unit='',land=True,coastline=True,
            level_contour=15,
            title='AMSR Sea ice concentration - '+str(year)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2),
            fileout=str(figures_path)+'/mapbeaufort_sic_amsr_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.png')


#MASIE
if os.path.isfile(str(figures_path)+'/mapbeaufort_iceextent_masie_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.png')==False:
    filename = str(rawdata_path)+'/iceextent_masie/masie_lat_lon_4km.nc'
    ds_masie = xr.open_dataset(filename)
    filename = str(rawdata_path)+'/iceextent_masie/masie_all_r00_v01_'+str(year)+doy+'_4km.nc'
    if os.path.isfile(filename):
        ds = xr.open_dataset(filename)

        map(ds_masie.longitude,ds_masie.latitude,ds.sea_ice_extent.squeeze()[:-1,:-1],0,5,
            palette='jet',unit='0:missing+not SI; 1:ocean; 2:land; 3:SI; 4:coastline; 5:lake',land=False,coastline=False,
            title='Ice extent MASIE - '+str(year)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2),
            fileout=str(figures_path)+'/mapbeaufort_iceextent_masie_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.png')


#OISST L4
if os.path.isfile(str(figures_path)+'/mapbeaufort_sst_oi_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.png')==False:
    filename = str(rawdata_path)+'/sst_oi/oisst-avhrr-v02r01.'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.nc'
    if os.path.isfile(filename):
        ds = xr.open_dataset(filename).sel(lon=slice(lonmin+360,lonmax+360), lat=slice(latmin,latmax))

        map(ds.lon,ds.lat,ds.sst.squeeze(),-2,10,
            palette='jet',unit='degC',land=True,coastline=True,
            title='SST OI - '+str(year)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2),
            fileout=str(figures_path)+'/mapbeaufort_sst_oi_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.png')
    
    
#SMAP JPL SSS L3
if os.path.isfile(str(figures_path)+'/mapbeaufort_sss_smapjpl_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.png')==False:
    filename = str(rawdata_path)+'/sss_smapjpl/SMAP_L3_SSS_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_8DAYS_V5.0.nc'
    if os.path.isfile(filename):
        ds = xr.open_dataset(filename).sel(longitude=slice(lonmin,lonmax), latitude=slice(latmax,latmin))

        map(ds.longitude,ds.latitude,ds.smap_sss.squeeze(),20,35,
            palette='jet',unit='psu',land=True,coastline=True,
            title='SSS SMAP JPL - '+str(year)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2),
            fileout=str(figures_path)+'/mapbeaufort_sss_smapjpl_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.png')
    

#SMOS SSS L3 #Updates to make for Alex and Jacqueline's NRT version?
if os.path.isfile(str(figures_path)+'/mapbeaufort_sss_smos_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.png')==False:
    filename = str(rawdata_path)+'/sss_smos/SMOS-arctic-LOCEAN-SSS-'+str(year)+'-'+str(month).zfill(2)+'-'+str(day).zfill(2)+'-v1.1AT-7days.nc'
    if os.path.isfile(filename):
        ds = xr.open_dataset(filename)
        
        map(ds.longitude,ds.latitude,ds.smos_sss.squeeze(),20,35,
            palette='jet',unit='psu',land=True,coastline=True,
            title='SSS SMOS - '+str(year)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2),
            fileout=str(figures_path)+'/mapbeaufort_sss_smos_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.png')


#CCMP wind
if os.path.isfile(str(figures_path)+'/mapbeaufort_wind_ccmp_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_0.png')==False:
    filename = glob.glob(str(rawdata_path)+'/wind_ccmp/*_Wind_Analysis_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_V*_L3.0_RSS.nc')[0]
    if os.path.isfile(filename):
        ds = xr.open_dataset(filename)
        for t in range(0,len(ds.time)):
            ts = datetime.datetime.strptime(str(ds.time[t].values),'%Y-%m-%dT%H:%M:%S.%f000').strftime("%Y-%m-%d %H:%M:%S") 
            
            map(ds.longitude,ds.latitude,numpy.sqrt(ds.uwnd[t,:,:].squeeze()**2+ds.vwnd[t,:,:].squeeze()**2).squeeze(),0,15,
                palette='jet',unit='m/s',land=True,coastline=True,
                u=ds.uwnd[t,:,:].squeeze(),v=ds.vwnd[t,:,:].squeeze(),unit_vector='m/s',scale=200,
                title='Wind CCMP - '+ts,
                fileout=str(figures_path)+'/mapbeaufort_wind_ccmp_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_'+str(t)+'.png')


#AVISO SLA
if os.path.isfile(str(figures_path)+'/mapbeaufort_sla_aviso_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.png')==False:
    filename = glob.glob(str(rawdata_path)+'/sla_aviso/*global_allsat_phy_l4_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_*nc')[0]
    if os.path.isfile(filename):
        ds = xr.open_dataset(filename).sel(longitude=slice(lonmin,lonmax), latitude=slice(latmin,latmax))

        map(ds.longitude,ds.latitude,ds.sla.squeeze()*10**2,-25,25,
            palette='seismic',unit='cm',land=True,coastline=True,
            u=ds.ugos.squeeze()*10**2,v=ds.vgos.squeeze()*10**2,unit_vector='cm/s',scale=400,
            title='SLA Aviso - '+str(year)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2),
            fileout=str(figures_path)+'/mapbeaufort_sla_aviso_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.png')







