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
##
## Push the data and figures to an ftp site
##
## The program will download and plot data for today (or yesterday, or the day before etc up to a week before if no data found)
##############################################
    
import numpy
import pandas as pd
from scipy import stats
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

######### TO DEFINE ###################

#paths
# SEVERINE
rawdata_path=Path('/Users/severinf/Data/SASSIE/satellite/') 
figures_path=Path('/Users/severinf/Figures/SASSIE/cruise/') 

# KYLA CLOUD
# rawdata_path=Path('/home/jovyan/data/SASSIE/satellite/')
# figures_path=Path('/home/jovyan/figures/SASSIE')


######### OTHER PARAMETERS ###################

#date
time=datetime.date.today()
year=time.year
month=time.month
day=time.day

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
    ax.set_extent([lonmapmin, lonmapmax, latmapmin, latmapmax], crs=ccrs.PlateCarree()) 
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
        X, Y, masked_data = z_masked_overlap(ax, karg['x_contour'].values, karg['y_contour'].values, karg['data_contour'].values, source_projection=cartopy.crs.Geodetic())    
        cc=ax.contour(X,Y,masked_data,levels=[karg['level_contour']],colors='m',linewidth=4)
        fmt = {}
        plt.clabel(cc, cc.levels, fmt=karg['label_contour'], inline=True, fontsize=10,colors='m')
    if 'u' in karg:  
        q=plt.quiver(x[::5].values,y[::2].values,karg['u'][::2,::5].values,karg['v'][::2,::5].values,scale=karg['scale'],transform=ccrs.PlateCarree())
        qk= plt.quiverkey (q,0.95,-0.07,10,'10'+karg['unit_vector'],labelpos='N')

    filename=str(rawdata_path)+'/Ship_track.xlsx'
    if os.path.isfile(filename):
        df = pd.read_excel(filename,header=None)
        plt.plot(df.iloc[:,4],df.iloc[:,3],linestyle='--',color='#808080',marker='o',markersize=6,markerfacecolor='#808080',markeredgecolor='k',transform=cartopy.crs.PlateCarree())                    
        for t in range(df.shape[0]):
            plt.annotate(str(df.iloc[t,1]).zfill(2)+'/'+str(df.iloc[t,2]).zfill(2), xy=(df.iloc[t,4],df.iloc[t,3]),fontsize=10,color='#808080',xycoords=ccrs.PlateCarree()._as_mpl_transform(ax))
    cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.04])
    h=plt.colorbar(pp, cax=cbar_ax,orientation='horizontal',ax=ax)
    h.ax.tick_params(labelsize=20)
    h.set_label(karg['unit'],fontsize=20)
    cmin,cmax = h.mappable.get_clim()
    ax.set_title(karg['title'],fontsize=20)
    plt.subplots_adjust(right=0.9,left=0.1,top=0.9,bottom=0.18)
    plt.savefig(karg['fileout'], dpi=fig.dpi)
    
    
######################################################
######## DOWNLOAD and plot Data ######################
######################################################

doy = time.timetuple().tm_yday

#Ship tracks from ftp site ######################
# os.system('wget -N -P '+str(rawdata_path)+'/'+' ftp:// /Ship_track.xlsx'



#AMSR Sea ice ############################################
os.system('mkdir -p '+str(rawdata_path)+'/seaice_amsr')
os.system('mkdir -p '+str(figures_path)+'/seaice_amsr')

filename_si='AMSR_U2_L3_SeaIce12km_B04_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'.he5'
i=0
time_tmp=time - datetime.timedelta(days=i)
while os.path.isfile(str(rawdata_path)+'/seaice_amsr/'+filename_si)==False and i<8:
    time_tmp=time - datetime.timedelta(days=i)
    filename_si='AMSR_U2_L3_SeaIce12km_B04_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'.he5'
    os.system('wget -N -P '+str(rawdata_path)+'/seaice_amsr/'+' https://n5eil01u.ecs.nsidc.org/AMSA/AU_SI12.001/'+str(time_tmp.year)+'.'+str(time_tmp.month).zfill(2)+'.'+str(time_tmp.day).zfill(2)+'/'+filename_si)   
    i=i+1

if os.path.isfile(str(rawdata_path)+'/seaice_amsr/'+filename_si):
    datestr_ice=datetime.datetime.strptime(str(time_tmp),'%Y-%m-%d').strftime("%m/%d") 
    dsc = xr.open_dataset(str(rawdata_path)+'/seaice_amsr/'+filename_si,group='HDFEOS/GRIDS/NpPolarGrid12km')
    dsd = xr.open_dataset(str(rawdata_path)+'/seaice_amsr/'+filename_si,group='HDFEOS/GRIDS/NpPolarGrid12km/Data Fields')

    map(dsc.lon,dsc.lat,dsd.SI_12km_NH_ICECON_DAY.squeeze(),0,100,
        palette='Blues_r',unit='',land=True,coastline=True,
        level_contour=15,label_contour=datestr_ice,x_contour=dsc.lon,y_contour=dsc.lat,data_contour=dsd.SI_12km_NH_ICECON_DAY.squeeze(),
        title='AMSR Sea ice concentration - '+str(time_tmp.year)+'/'+str(time_tmp.month).zfill(2)+'/'+str(time_tmp.day).zfill(2),
        fileout=str(figures_path)+'/seaice_amsr/sic_amsr_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'.png')


#MASIE ############################################
os.system('mkdir -p '+str(rawdata_path)+'/iceextent_masie')
os.system('mkdir -p '+str(figures_path)+'/iceextent_masie')

if os.path.isfile(str(rawdata_path)+'/iceextent_masie/masie_lat_lon_4km.nc')==False:
    os.system('wget -N -P '+str(rawdata_path)+'/iceextent_masie/'+' https://masie_web.apps.nsidc.org/pub/DATASETS/NOAA/G02186/ancillary/masie_lat_lon_4km.nc')

filename='masie_all_r00_v01_'+str(year)+str(doy)+'_4km.nc'
i=0
time_tmp=time - datetime.timedelta(days=i)
while os.path.isfile(str(rawdata_path)+'/iceextent_masie/'+filename)==False and i<8:
    time_tmp=time - datetime.timedelta(days=i)
    filename='masie_all_r00_v01_'+str(time_tmp.year)+str(time_tmp.timetuple().tm_yday)+'_4km.nc'
    os.system('wget -N -P '+str(rawdata_path)+'/iceextent_masie/'+' https://masie_web.apps.nsidc.org/pub/DATASETS/NOAA/G02186/netcdf/4km/'+str(time_tmp.year)+'/'+filename)   
    i=i+1

ds_masie = xr.open_dataset(str(rawdata_path)+'/iceextent_masie/masie_lat_lon_4km.nc')
if os.path.isfile(str(rawdata_path)+'/iceextent_masie/'+filename):
        ds = xr.open_dataset(str(rawdata_path)+'/iceextent_masie/'+filename)
        if os.path.isfile(str(rawdata_path)+'/seaice_amsr/'+filename_si):
            map(ds_masie.longitude,ds_masie.latitude,ds.sea_ice_extent.squeeze()[:-1,:-1],0,5,
                palette='jet',unit='0:missing+not SI; 1:ocean; 2:land; 3:SI; 4:coastline; 5:lake',land=False,coastline=False,
                level_contour=15,label_contour=datestr_ice,x_contour=dsc.lon,y_contour=dsc.lat,data_contour=dsd.SI_12km_NH_ICECON_DAY.squeeze(),
                title='Ice extent MASIE - '+str(time_tmp.year)+'/'+str(time_tmp.month).zfill(2)+'/'+str(time_tmp.day).zfill(2),
                fileout=str(figures_path)+'/iceextent_masie/iceextent_masie_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'.png')
        else:
            map(ds_masie.longitude,ds_masie.latitude,ds.sea_ice_extent.squeeze()[:-1,:-1],0,5,
                palette='jet',unit='0:missing+not SI; 1:ocean; 2:land; 3:SI; 4:coastline; 5:lake',land=False,coastline=False,
                title='Ice extent MASIE - '+str(time_tmp.year)+'/'+str(time_tmp.month).zfill(2)+'/'+str(time_tmp.day).zfill(2),
                fileout=str(figures_path)+'/iceextent_masie/iceextent_masie_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'.png')



#OISST L4 ############################################
os.system('mkdir -p '+str(rawdata_path)+'/sst_oi')
os.system('mkdir -p '+str(figures_path)+'/sst_oi')

filename='oisst-avhrr-v02r01.'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_preliminary.nc'
i=0
time_tmp=time - datetime.timedelta(days=i)
while os.path.isfile(str(rawdata_path)+'/sst_oi/'+filename)==False and i<8:
    time_tmp=time - datetime.timedelta(days=i)
    filename='oisst-avhrr-v02r01.'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'_preliminary.nc'
    os.system('wget -N -P '+str(rawdata_path)+'/sst_oi/'+' https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+'/'+filename)   
    i=i+1

if os.path.isfile(str(rawdata_path)+'/sst_oi/'+filename):
        ds = xr.open_dataset(str(rawdata_path)+'/sst_oi/'+filename).sel(lon=slice(lonmin+360,lonmax+360), lat=slice(latmin,latmax))
        if os.path.isfile(str(rawdata_path)+'/seaice_amsr/'+filename_si):
            map(ds.lon,ds.lat,ds.sst.squeeze(),-2,10,
                palette='jet',unit='degC',land=True,coastline=True,
                level_contour=15,label_contour=datestr_ice,x_contour=dsc.lon,y_contour=dsc.lat,data_contour=dsd.SI_12km_NH_ICECON_DAY.squeeze(),
                title='SST OI - '+str(time_tmp.year)+'/'+str(time_tmp.month).zfill(2)+'/'+str(time_tmp.day).zfill(2),
                fileout=str(figures_path)+'/sst_oi/sst_oi_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'.png')
        else:
            map(ds.lon,ds.lat,ds.sst.squeeze(),-2,10,
                palette='jet',unit='degC',land=True,coastline=True,
                title='SST OI - '+str(time_tmp.year)+'/'+str(time_tmp.month).zfill(2)+'/'+str(time_tmp.day).zfill(2),
                fileout=str(figures_path)+'/sst_oi/sst_oi_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'.png')


#SMAP JPL SSS L3 ############################################
#(file for 09/01 will be in the folder of 09/05 (8 day running mean))
os.system('mkdir -p '+str(rawdata_path)+'/sss_smapjpl')
os.system('mkdir -p '+str(figures_path)+'/sss_smapjpl')

filename='SMAP_L3_SSS_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_8DAYS_V5.0.nc'
i=0
time_tmp=time - datetime.timedelta(days=i)
while os.path.isfile(str(rawdata_path)+'/sss_smapjpl/'+filename)==False and i<8:
    time_tmp=time - datetime.timedelta(days=i)
    filename='SMAP_L3_SSS_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'_8DAYS_V5.0.nc'
    os.system('wget -N -P '+str(rawdata_path)+'/sss_smapjpl/'+' https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5/'+str(time_tmp.year)+'/'+str(int(time_tmp.timetuple().tm_yday)-4)+'/'+filename)
    i=i+1

if os.path.isfile(str(rawdata_path)+'/sss_smapjpl/'+filename):
        ds = xr.open_dataset(str(rawdata_path)+'/sss_smapjpl/'+filename).sel(longitude=slice(lonmin,lonmax), latitude=slice(latmax,latmin))
        if os.path.isfile(str(rawdata_path)+'/seaice_amsr/'+filename_si):
            map(ds.longitude,ds.latitude,ds.smap_sss.squeeze(),20,35,
                palette='jet',unit='psu',land=True,coastline=True,
                level_contour=15,label_contour=datestr_ice,x_contour=dsc.lon,y_contour=dsc.lat,data_contour=dsd.SI_12km_NH_ICECON_DAY.squeeze(),
                title='SSS SMAP JPL - '+str(time_tmp.year)+'/'+str(time_tmp.month).zfill(2)+'/'+str(time_tmp.day).zfill(2),
                fileout=str(figures_path)+'/sss_smapjpl/sss_smapjpl_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'.png')
        else:
            map(ds.longitude,ds.latitude,ds.smap_sss.squeeze(),20,35,
                palette='jet',unit='psu',land=True,coastline=True,
                title='SSS SMAP JPL - '+str(time_tmp.year)+'/'+str(time_tmp.month).zfill(2)+'/'+str(time_tmp.day).zfill(2),
                fileout=str(figures_path)+'/sss_smapjpl/sss_smapjpl_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'.png')
        


#SMOS SSS L3############################################
os.system('mkdir -p '+str(rawdata_path)+'/sss_smos')
os.system('mkdir -p '+str(figures_path)+'/sss_smos')

filename='SMOS-arctic-LOCEAN-SSS-'+str(year)+'-'+str(month).zfill(2)+'-'+str(day).zfill(2)+'-v1.1AT-7days.nc'
i=0
time_tmp=time - datetime.timedelta(days=i)
while os.path.isfile(str(rawdata_path)+'/sss_smos/'+filename)==False and i<8:
    time_tmp=time - datetime.timedelta(days=i)
    filename='SMOS-arctic-LOCEAN-SSS-'+str(time_tmp.year)+'-'+str(time_tmp.month).zfill(2)+'-'+str(time_tmp.day).zfill(2)+'-v1.1AT-7days.nc'
    os.system('wget -N -P '+str(rawdata_path)+'/sss_smos/'+' ftp://ext-catds-cecos-locean@ftp.ifremer.fr/Ocean_products/SMOS_ARCTIC_SSS_L3_LOCEAN/netcdf_weekly_v1_1/'+filename)   
    i=i+1

if os.path.isfile(str(rawdata_path)+'/sss_smos/'+filename):
        ds = xr.open_dataset(str(rawdata_path)+'/sss_smos/'+filename)
        if os.path.isfile(str(rawdata_path)+'/seaice_amsr/'+filename_si):
            map(ds.longitude,ds.latitude,ds.smos_sss.squeeze(),20,35,
                palette='jet',unit='psu',land=True,coastline=True,
                level_contour=15,label_contour=datestr_ice,x_contour=dsc.lon,y_contour=dsc.lat,data_contour=dsd.SI_12km_NH_ICECON_DAY.squeeze(),
                title='SSS SMOS - '+str(time_tmp.year)+'/'+str(time_tmp.month).zfill(2)+'/'+str(time_tmp.day).zfill(2),
                fileout=str(figures_path)+'/sss_smos/sss_smos_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'.png')
        else:
            map(ds.longitude,ds.latitude,ds.smos_sss.squeeze(),20,35,
                palette='jet',unit='psu',land=True,coastline=True,
                title='SSS SMOS - '+str(time_tmp.year)+'/'+str(time_tmp.month).zfill(2)+'/'+str(time_tmp.day).zfill(2),
                fileout=str(figures_path)+'/sss_smos/sss_smos_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'.png')



#CCMP winds############################################
os.system('mkdir -p '+str(rawdata_path)+'/wind_ccmp')
os.system('mkdir -p '+str(figures_path)+'/wind_ccmp')

# filename_dt='CCMP_Wind_Analysis_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_V02.0_L3.0_RSS.nc'
filename='CCMP_RT_Wind_Analysis_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_V02.1_L3.0_RSS.nc'
i=0
time_tmp=time - datetime.timedelta(days=i)
while os.path.isfile(str(rawdata_path)+'/wind_ccmp/'+filename)==False and i<8:
    time_tmp=time - datetime.timedelta(days=i)
    filename='CCMP_RT_Wind_Analysis_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'_V02.1_L3.0_RSS.nc'
    os.system('wget -N -P '+str(rawdata_path)+'/wind_ccmp/'+' https://data.remss.com/ccmp/v02.1.NRT/Y'+str(time_tmp.year)+'/M'+str(time_tmp.month).zfill(2)+'/'+filename)   
    i=i+1

if os.path.isfile(str(rawdata_path)+'/wind_ccmp/'+filename):
        ds = xr.open_dataset(str(rawdata_path)+'/wind_ccmp/'+filename)
        for t in range(0,len(ds.time)):
            ts = datetime.datetime.strptime(str(ds.time[t].values),'%Y-%m-%dT%H:%M:%S.%f000').strftime("%Y-%m-%d %H:%M:%S") 
            if os.path.isfile(str(rawdata_path)+'/seaice_amsr/'+filename_si):
                map(ds.longitude,ds.latitude,numpy.sqrt(ds.uwnd[t,:,:].squeeze()**2+ds.vwnd[t,:,:].squeeze()**2).squeeze(),0,15,
                    palette='jet',unit='m/s',land=True,coastline=True,
                    level_contour=15,label_contour=datestr_ice,x_contour=dsc.lon,y_contour=dsc.lat,data_contour=dsd.SI_12km_NH_ICECON_DAY.squeeze(),
                    u=ds.uwnd[t,:,:].squeeze(),v=ds.vwnd[t,:,:].squeeze(),unit_vector='m/s',scale=200,
                    title='Wind CCMP - '+ts,
                    fileout=str(figures_path)+'/wind_ccmp/wind_ccmp_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'_'+str(t)+'.png')
            else:
                map(ds.longitude,ds.latitude,numpy.sqrt(ds.uwnd[t,:,:].squeeze()**2+ds.vwnd[t,:,:].squeeze()**2).squeeze(),0,15,
                    palette='jet',unit='m/s',land=True,coastline=True,
                    u=ds.uwnd[t,:,:].squeeze(),v=ds.vwnd[t,:,:].squeeze(),unit_vector='m/s',scale=200,
                    title='Wind CCMP - '+ts,
                    fileout=str(figures_path)+'/wind_ccmp/wind_ccmp_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'_'+str(t)+'.png')
                


#AVISO SLA ############################################
os.system('mkdir -p '+str(rawdata_path)+'/sla_aviso')
os.system('mkdir -p '+str(figures_path)+'/sla_aviso')

# filename_dt='dt_global_allsat_phy_l4_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_*nc'
filename='nrt_global_allsat_phy_l4_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_*nc'
i=0
time_tmp=time - datetime.timedelta(days=i)
while len(glob.glob(str(rawdata_path)+'/sla_aviso/'+filename))<1 and i<8:
    time_tmp=time - datetime.timedelta(days=i)
    filename='nrt_global_allsat_phy_l4_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'_*nc'
    os.system('wget -N -P '+str(rawdata_path)+'/sla_aviso/'+' ftp://nrt.cmems-du.eu/Core/SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046/dataset-duacs-nrt-global-merged-allsat-phy-l4/'+str(time_tmp.year)+'/'+str(time_tmp.month).zfill(2)+'/'+filename)   
    i=i+1

if len(glob.glob(str(rawdata_path)+'/sla_aviso/'+filename))>=1:
        filename=glob.glob(str(rawdata_path)+'/sla_aviso/'+filename)[0]
        ds = xr.open_dataset(filename).sel(longitude=slice(lonmin,lonmax), latitude=slice(latmin,latmax))
        if os.path.isfile(str(rawdata_path)+'/seaice_amsr/'+filename_si):
            map(ds.longitude,ds.latitude,ds.sla.squeeze()*10**2,-25,25,
                palette='seismic',unit='cm',land=True,coastline=True,
                level_contour=15,label_contour=datestr_ice,x_contour=dsc.lon,y_contour=dsc.lat,data_contour=dsd.SI_12km_NH_ICECON_DAY.squeeze(),
                u=ds.ugos.squeeze()*10**2,v=ds.vgos.squeeze()*10**2,unit_vector='cm/s',scale=400,
                title='SLA Aviso - '+str(time_tmp.year)+'/'+str(time_tmp.month).zfill(2)+'/'+str(time_tmp.day).zfill(2),
                fileout=str(figures_path)+'/sla_aviso/sla_aviso_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'.png')
        else:
            map(ds.longitude,ds.latitude,ds.sla.squeeze()*10**2,-25,25,
                palette='seismic',unit='cm',land=True,coastline=True,
                u=ds.ugos.squeeze()*10**2,v=ds.vgos.squeeze()*10**2,unit_vector='cm/s',scale=400,
                title='SLA Aviso - '+str(time_tmp.year)+'/'+str(time_tmp.month).zfill(2)+'/'+str(time_tmp.day).zfill(2),
                fileout=str(figures_path)+'/sla_aviso/sla_aviso_'+str(time_tmp.year)+str(time_tmp.month).zfill(2)+str(time_tmp.day).zfill(2)+'.png')
            


############################################################
######## Push data and figures to ftp ######################
############################################################
# scp folder per variable for figures AND data


