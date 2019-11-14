'''
PURPOSE: To pre process CMIP model files containing vertical profiles of 
temperature, surface pressure and specific humidity to create a
two layer theta_e variables that can combine to produce a buoyancy-like
variable

AUTHOR: Fiaz Ahmed

DATE: 11/08/19
'''

import numpy as np
from netCDF4 import Dataset
from glob import glob

import datetime as dt
from dateutil.relativedelta import relativedelta
import time
import itertools
from mpi4py import MPI
from sys import exit, stdout
from numpy import dtype
from parameters import *
from vert_cython import vert_integ_variable_bl,vert_integ_exneri_variable_bl,vert_integ_lt_variable_bl
from vert_cython import find_closest_index
import os


# Import Python functions specific to Convective Transition Basic Statistics
from convecTransBasic_util import generate_region_mask
from convecTransBasic_util import convecTransBasic_calc_model
from convecTransBasic_util import convecTransLev2_calc_model
# from convecTransBasic_util import convecTransBasic_loadAnalyzedData
# from convecTransBasic_util import convecTransBasic_plot
print("**************************************************")
print("Executing Convective Transition Level 2 Statistics (convecTransLev2.py)......")
print("**************************************************")

print("Load user-specified binning parameters..."),

# Create and read user-specified parameters
# temporarily using os.getcwd instead of POD_HOME
os.system("python "+os.getcwd()+"/"+"convecTransBasic_usp_calc.py")
with open(os.getcwd()+"/"+"convecTransLev2_calc_parameters.json") as outfile:
    bin_data=json.load(outfile)
print("...Loaded!")


# ======================================================================
# Binned data, i.e., convective transition statistics binned in specified intervals of 
#  Bint, are saved to avoid redoing binning computation every time
# Check if binned data file exists in wkdir/MDTF_casename/ from a previous computation
#  if so, skip binning; otherwise, bin data using model output
#  (see convecTransBasic_usp_calc.py for where the model output locate)

if (len(bin_data["bin_output_list"])==0 or bin_data["BIN_ANYWAY"]):

    print("Starting binning procedure...")

    if bin_data["PREPROCESS_THETAE"]==1:
        print("   THETA_E-BASED pre-processing required")
#     if bin_data["SAVE_TAVE_QSAT_INT"]==1:
#         print("      Pre-processed temperature fields ("\
#             +os.environ["tave_var"]+" & "+os.environ["qsat_int_var"]\
#             +") will be saved to "+bin_data["PREPROCESSING_OUTPUT_DIR"]+"/")

    # Load & pre-process region mask
#     REGION=generate_region_mask(bin_data["REGION_MASK_DIR"]+"/"+bin_data["REGION_MASK_FILENAME"], bin_data["pr_list"][0],bin_data["LAT_VAR"],bin_data["LON_VAR"])

    # Pre-process temperature (if necessary) & bin & save binned results
#     binned_output=convecTransBasic_calc_model(REGION,bin_data["args1"])
        binned_output=convecTransLev2_calc_model(bin_data["args1"])

else: # Binned data file exists & BIN_ANYWAY=False
    print("Binned output detected..."),
#     binned_output=convecTransBasic_loadAnalyzedData(bin_data["args2"])
#     print("...Loaded!")

exit()

####### MASK ########

## choose the landsea  mask based on model ##

# model_name='c96L32_am4g9_fullaero_MDTF'
# model_name='c96L48_am4b6_DDFull_MDTF'
model_name='CAM5.3'
model_name='NASA-GISS'


# f=Dataset('/home/fiaz/MDTF/files/LandSeaMask_am4g9.nc','r')
# f=Dataset('/home/fiaz/MDTF/files/LandSeaMask_cam5.nc','r')
# lsm=np.asarray(f.variables['LSMASK'][:],dtype='float')
# latm,lonm=f.variables['lat'][:],f.variables['lon'][:]
# f.close()
# 

# mask_ocean=np.copy(lsm)
# mask_ocean[mask_ocean!=1]=np.nan
# 
# mask_land=np.copy(lsm)
# mask_land[mask_land!=0]=np.nan
# mask_land[mask_land==0]=1.

### Create list of file to process

strt_date=dt.datetime(2009,1,1)
# end_date=dt.datetime(2001,9,2)
end_date=dt.datetime(2010,12,31)

dts=[]

dirc='/scratch/neelin/MDTF_TIMESLICE_EXP/'+model_name+'/segmented_files/'

dirc1=dirc+'ta/'
dirc2=dirc+'hus/'
dirc3=dirc+'tas/'
dirc4=dirc+'huss/'
dirc5=dirc+'ps/'
dirc6=dirc+'pr/'
    
list1=[]
list2=[]
list3=[]
list4=[]
    
d1=strt_date.strftime("%Y%m%d")
# fname=dirc1+'atmos.'+d1+'*'
fname=dirc1+'*atmos.'+'*'
list1=(glob(fname))
list1.sort()

list1=list1[4:]

# print(list1)
# exit()

f=Dataset(list1[0],'r')
lev=f.variables['lev'][:]
# lev=f.variables['level'][::-1]
# lev=f.variables['pfull'][:]
# phalf=f.variables['phalf'][:]
lat=f.variables['lat'][:]
lon=f.variables['lon'][:]
f.close()

i200=np.argmin(abs(lev-200))
i500=np.argmin(abs(lev-500))
i1000=np.argmin(abs(lev-1000))

comm=MPI.COMM_WORLD
print(comm.rank)

def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]

if comm.rank == 0:
        jobs=list1
        jobs=split(jobs,comm.size)

else:
        jobs=None

# Scatter jobs across cores
jobs=comm.scatter(jobs,root=0)
# 
s=time.time()

# Use this if MPI is off:
# jobs=list1
ix=100

for j in jobs:
    print(j)
    t1=time.time()
    d1=j[-32:-3]
    header=j[-62:-34]
#     fname2=dirc2+'*'+header+'hus'+d1+'*'
#     fname3=dirc3+header+'tas'+'*'+d1+'*'
#     fname4=dirc4+header+'huss'+'*'+d1+'*'
#     fname5=dirc5+header+'ps'+'*'+d1+'*'
#     fname6=dirc6+header+'pr'+'*'+d1+'*'

    fname2=dirc2+'*'+header+'hus'+d1+'*'
    fname3=dirc3+'*'+header+'tas'+d1+'*'
    fname4=dirc4+'*'+header+'huss'+d1+'*'
    fname5=dirc5+'*'+header+'ps'+d1+'*'
    fname6=dirc6+'*'+header+'pr'+d1+'*'

        
    list2=(glob(fname2))
    list3=(glob(fname3))
    list4=(glob(fname4))
    list5=(glob(fname5))
    list6=(glob(fname6))
    
    print(list2)
        
    ## LOAD TEMPERATURE ##

    print('LOADING TEMP.')
    f=Dataset(j,'r')
    t=f.variables['ta'][:]#[:ix,...]#[:ix,...]
#     tyme=f.variables['time'][:]#[:ix,...]
#     t=f.variables['ta'][:,::-1,...]
    tyme=f.variables['time'][:]#[:ix,...]
    f.close()
    ## LOAD SPECIFIC HUMIDITY (UNITS OF K)##


    print('LOADING SP.HUM.')
    f=Dataset(list2[0],'r')
    q=f.variables['hus'][:]#[:ix,...]#[:ix,...]
#     q=f.variables['hus'][:,::-1,...]
    f.close()

    ## LOAD 2 meter TEMP. & HUMIDITY ##

    print('LOADING SURF. T & Q')
    f=Dataset(list3[0],'r')
    t2m=f.variables['tas'][:]#[:ix,...]#[:ix,...]
#     t2m=f.variables['tas'][:]
    f.close()
        
    f=Dataset(list4[0],'r')
    qps=f.variables['huss'][:]#[:ix,...]#[:ix,...]
#     qps=f.variables['huss'][:]
    f.close()
    qps=np.squeeze(qps)
    
    ## LOAD SURFACE PRESSURE ######

    print('LOADING SURF. PRESSURE')
    f=Dataset(list5[0],'r')
    pres=f.variables['ps'][:]*1e-2#[:ix,...]*1e-2 # Converting to hPa            
#     pres=f.variables['ps'][:ix,...]*1e-2 # Converting to hPa            
    f.close()    
    

    ### Some specific humidity values are < 0,
    ### possibly an interpolation issue.

    q[q<0]=0.0    
    qps[qps<0]=0.0        
        
    pbl_top=pres-100 ## The sub-cloud layer is 100 mb thick ##
    low_top=np.zeros_like(pres)
    low_top[:]=500
#     low_top=pbl_top-400 ## The next layer extends to 500 mb ##

    pbl_top=np.float_(pbl_top.flatten())
    low_top=np.float_(low_top.flatten())
    lev=np.float_(lev)

    print('PREPPING LAYER AVERAGING-2')
    
    pbl_ind=np.zeros(pbl_top.size,dtype=np.int64)
    low_ind=np.zeros(low_top.size,dtype=np.int64)

    find_closest_index(pbl_top,lev,pbl_ind)
    find_closest_index(low_top,lev,low_ind)

    pres_3d=np.zeros_like(t)
    pres_3d[:]=pres[:,None,:,:]

    levels=np.zeros_like(t)        
    levels[:]=lev[None,:,None,None]
    
     ##### Calculate qsat #####

    Tk0 = 273.15 # Reference temperature.
    Es0 = 610.7 # Vapor pressure at Tk0.
    Lv0 = 2500800 # Latent heat of evaporation at Tk0.
    cpv = 1869.4 # Isobaric specific heat capacity of water vapor at tk0.
    cl = 4218.0 # Specific heat capacity of liquid water at tk0.
    R = 8.3144 # Universal gas constant.
    Mw = 0.018015 # Molecular weight of water.
    Rv = R/Mw # Gas constant for water vapor.
    Ma = 0.028964 # Molecular weight of dry air.
    Rd = R/Ma # Gas constant for dry air.
    epsilon = Mw/Ma 
    g = 9.80665 
    cpd=1004.

    ### TROPOSPHERIC MOISTURE AND TEMPERATURE CALCULATIONS ###


    ### Saturation specific humidity and mixing ratio ###
    print('ESTIMATING qSAT.')
    stdout.flush()

#     Es=(Es0*(t/Tk0)**((cpv-cl)/Rv))*np.exp((Lv0+(cl-cpv)*Tk0)/Rv*(1/Tk0-1./t))
    Es=es_calc(t)
    ws=(epsilon)*(Es/levels)    
#     qsat=(Lv/Cp)*ws/(1+ws)
    qsat=ws/(1+ws)
    
    w=q/(1-q)
    e=w*levels/(epsilon+w) # vapor pressure
    
    print(Es.max(),Es.min())
    print(ws.max(),ws.min())
    print(qsat.max(),qsat.min())
    print(w.max(),w.min())
    print(e.max(),e.min())
    
    qsat[qsat<0]=0.0
    ws[ws<0]=0.0
    Es[Es<0]=0.0
    
    ### END ###

    ### SURFACE MOISTURE AND TEMPERATURE CALCULATIONS ###

    ### Get 2 metre specific humidity ###

    ### 2m specific humidity and mixing ratio ###
    print('ESTIMATING 2m SP.HUM.')
    stdout.flush()

#     eps=(Es0*(dt2m/Tk0)**((cpv-cl)/Rv))*np.exp((Lv0+(cl-cpv)*Tk0)/Rv*(1/Tk0-1./dt2m))
#     eps=eps*1e-2
#     wps=(epsilon)*(eps/pres)
# #     qps=(Lv/Cp)*wps/(1+wps) 
#     qps=wps/(1+wps) 
    wps=qps/(1-qps)
    eps=(qps/(1-qps))*(pres/epsilon)

    ### 2m qsat and saturation specific humidity ###

    print('ESTIMATING 2m qSAT')
    stdout.flush()

    Esat_ps=(Es0*(t2m/Tk0)**((cpv-cl)/Rv))*np.exp((Lv0+(cl-cpv)*Tk0)/Rv*(1/Tk0-1./t2m))
    Esat_ps=Esat_ps*1e-2
    wsat_ps=(epsilon)*(Esat_ps/pres)
#     qsat_ps=(Lv/Cp)*wsat_ps/(1+wsat_ps) 
    qsat_ps=wsat_ps/(1+wsat_ps) 
        
    #### END ########
    
    print('FILLING SUB-SURFACE PRESSURE LEVELS')
    stdout.flush()

    ##### Fill all pressure level below surface with surface values ###

    var_4d=np.zeros_like(t)

    ## Surface temperature
    var_4d[:]=t2m[:,None,:,:]
    t[levels>=pres_3d]=var_4d[levels>=pres_3d]

    ## Surface specific humidity 
    var_4d[:]=qps[:,None,:,:]
#     var_4d[:]=qps[:]

    q[levels>=pres_3d]=var_4d[levels>=pres_3d]
    
    ### Surface pressure
    dps=np.zeros_like(pres) ## Thickness of surface layer if surface_pres>1000 mb
    dps[pres>levels[:,-1,:,:]]=pres[pres>levels[:,-1,:,:]]-levels[:,-1,:,:][pres>levels[:,-1,:,:]]
    levels[levels>=pres_3d]=pres_3d[levels>=pres_3d]
    dp=np.diff(levels,axis=1)
    
    print('COMPUTING THETA_E')
    stdout.flush()

    ### Surface theta_e values ###
    
    psd=pres-eps # partial pressure of dry air
    rh_ps=qps/qsat_ps

#     print(t2m.size,wps.size,qps.size,pres.size,eps.size,qsat_ps.size)
#     exit()

    theta_e_ps=(t2m*(Po/psd)**(Rd/cpd))*((rh_ps)**(-wps*Rv/cpd))*np.exp(Lv0*wps/(cpd*t2m))
    theta_e_sat_ps=(t2m*(Po/psd)**(Rd/cpd))*np.exp(Lv0*wsat_ps/(cpd*t2m))
    theta_e_sub_sat_ps=theta_e_sat_ps-theta_e_ps

    ## RH ###
    rh=q/qsat
    pd=levels-e # partial pressure of dry air

   #Calculate theta_e
    theta_e=(t*(Po/pd)**(Rd/cpd))*((rh)**(-w*Rv/cpd))*np.exp(Lv0*w/(cpd*t))
   #Saturated theta_e
    theta_e_sat=(t*(Po/pd)**(Rd/cpd))*np.exp(Lv0*ws/(cpd*t))

    ###### INTEGRATED QUANTITIES ###### 
    mse=theta_e
    mse_sat=theta_e_sat
    mse_surf=theta_e_ps
    mse_sat_surf=theta_e_sat_ps
    
    print(np.nanmax(theta_e[:,i500:,...]),np.nanmin(theta_e[:,i500:,...]))
    print(np.nanmax(theta_e_sat[:,i500:,...]),np.nanmin(theta_e_sat[:,i500:,...]))
    print(np.nanmax(theta_e_ps),np.nanmin(theta_e_ps))
    print(np.nanmax(theta_e_sat_ps),np.nanmin(theta_e_sat_ps))
    print(np.nanmax((Po/pd)**(Rd/cpd)),np.nanmin((Po/pd)**(Rd/cpd)))
    
#     exit()
    
    print('PREPPING LAYER AVERAGING-3')
    stdout.flush()

    pbl_top=np.asarray([lev[i] for i in pbl_ind])
    low_top=np.asarray([lev[i] for i in low_ind])

    pbl_top=pbl_top.flatten()
    pbl_ind=pbl_ind.flatten()
    low_ind=low_ind.flatten()

#     print(pbl_top.max(),pbl_top.min())
#     print(low_top.max(),low_top.min())
#     exit()

    mse=np.swapaxes(mse,0,1)
    mse=mse.reshape(lev.size,-1)

    ### Get the lower layer temp. and qsat ###
    t=np.swapaxes(t,0,1)
    t=t.reshape(lev.size,-1)

#     temp=np.swapaxes(temp,0,1)
#     temp=temp.reshape(lev.size,-1)

    qsat=np.swapaxes(qsat,0,1)
    qsat=qsat.reshape(lev.size,-1)
    ##########################################

    mse_sat=np.swapaxes(mse_sat,0,1)
    mse_sat=mse_sat.reshape(lev.size,-1)
    mse_surf=mse_surf.flatten()

    pres=pres.flatten()

    dps=np.float64(dps.flatten())

    dp=np.swapaxes(dp,0,1)
    dp=dp.reshape(lev.size-1,-1)


    ### MSE VERT. ###
#     print(dps.size,qps.size)
#     print(dps.shape,mse[-1,:].size,mse_surf.size)
#     exit()

    mse_vert=(mse[1:,:]+mse[:-1,:])*0.5
    mse_lt_vert=(mse[1:,:]+mse[:-1,:])*0.5
    mse_sat_vert=(mse_sat[1:,:]+mse_sat[:-1,:])*0.5
    mse_last=(mse[-1,:]+mse_surf)*0.5*dps
#     exit()

    qsat_vert=(qsat[1:,:]+qsat[:-1,:])*0.5
    t_vert=(t[1:,:]+t[:-1,:])*0.5

    mse_bl=np.zeros((pres.size))
    mse_lt=np.zeros((pres.size))

    qsat_lt=np.zeros((pres.size))
    t_lt=np.zeros((pres.size))

    mse_sat_lt=np.zeros((pres.size))
    mse_mt=np.zeros(pres.size)
    mse_sat_mt=np.zeros(pres.size)

    mse_vert=(np.float_(mse_vert))
    qsat_vert=(np.float_(qsat_vert))
    t_vert=(np.float_(t_vert))

    mse_surf=(np.float_(mse_surf))
    mse_last=(np.float_(mse_last))
    mse_sat=(np.float_(mse_sat))
    pbl_ind=np.int_(pbl_ind)
    lev=np.float_(lev)
    dp=np.float_(dp)
    low_ind=np.int_(low_ind)

    print('VERTICAL INTEGRATION')
    stdout.flush()

#     vert_integ_variable_bl(mse_vert,mse_last,mse_sat,
#     pbl_ind,lev,dp,mse_bl,mse_lt,mse_sat_lt,low_ind)

#     vert_integ_variable_bl(mse_vert,mse_last,mse_sat,
#     pbl_ind,lev,dp,mse_bl,mse_lt,mse_sat_lt,mse_mt,mse_sat_mt,
#     i500,low_ind)

    vert_integ_variable_bl(mse_vert,mse_last,mse_sat,
    pbl_ind,lev,dp,mse_bl,mse_lt,mse_sat_lt,low_ind)
        
    mse_bl=mse_bl.reshape(-1,lat.size,lon.size)
    mse_lt=mse_lt.reshape(-1,lat.size,lon.size)
    mse_sat_lt=mse_sat_lt.reshape(-1,lat.size,lon.size)
    mse_mt=mse_mt.reshape(-1,lat.size,lon.size)
    mse_sat_mt=mse_sat_mt.reshape(-1,lat.size,lon.size)

    pbl_top=pbl_top.reshape(-1,lat.size,lon.size)
    low_top=low_top.reshape(-1,lat.size,lon.size)
    pres=pres.reshape(-1,lat.size,lon.size)

    mse_bl/=(pres-pbl_top)
    mse_lt/=(pbl_top-low_top)
    mse_sat_lt/=(pbl_top-low_top)
    
    mse_bl[mse_bl<0]=np.nan ### ~1% of the files have missing values
    mse_lt[mse_bl<0]=np.nan
    mse_sat_lt[mse_bl<0]=np.nan
    
    mse_bl[qps.mask]=np.nan
    mse_lt[qps.mask]=np.nan
    mse_sat_lt[qps.mask]=np.nan
    
#     ind=np.where(mse_bl<200)
#     print((qps[ind].mask.size,qps.mask.size))
#     exit()

    print('Maximum:')
    print('thetae BL:',np.nanmax(mse_bl),np.nanmin(mse_bl))
    print('thetae LT:',np.nanmax(mse_lt),np.nanmin(mse_lt))
    print('thetae SAT LT:',np.nanmax(mse_sat_lt),np.nanmin(mse_sat_lt))
    stdout.flush()

#     print('EXNERI BL:',np.nanmax(exneri_bl),np.nanmin(exneri_bl))
#     print('EXNERI LT:',np.nanmax(exneri_lt),np.nanmin(exneri_lt))
    
    print('---------------------')

    print('Mean:')

    print('thetae BL:',np.nanmean(mse_bl))
    print('thetae LT:',np.nanmean(mse_lt))
    print('thetae SAT LT:',np.nanmean(mse_sat_lt))
    stdout.flush()


#     print('MSE BL:',mse_bl.max(),mse_bl.min())
#     print('MSE LT:',mse_lt.max(),mse_lt.min())
#     print('MSE SAT LT:',mse_sat_lt.max(),mse_sat_lt.min())

    print('SAVING FILE')
    stdout.flush()

#     fout='/glade/p/univ/p35681102/fiaz/erai_data/layer_moist_static_energy/'+'era_layer_Lqcptgz_'+d1+'.nc'
#     fout='/scratch/neelin/layer_thetae/'+model_name+'/'+model_name+'.thetae'+d1+'.nc'
    fout='/scratch/neelin/layer_thetae/'+model_name+'/'+model_name+header+'thetae'+d1+'.nc'
#     fout='/scratch/neelin/layer_thetae/'+model_name+'layer_thetae_'+d1+'.nc'

    # fout='/glade/scratch/fiaz/era_processed/'+'era_trop_0.25_'+d1+'.nc'

    ##### SAVE FILE ######

    try:ncfile.close()
    except:pass

    ncfile = Dataset(fout, mode='w', format='NETCDF4')

    ncfile.createDimension('time',mse_bl.shape[0])
    ncfile.createDimension('lat',lat.size)
    ncfile.createDimension('lon',lon.size)

    dy = ncfile.createVariable('time','i4',('time'))
    lt = ncfile.createVariable('lat',dtype('float32').char,('lat'))
    ln = ncfile.createVariable('lon',dtype('float32').char,('lon'))

    dy.units = "hours since 1800-01-01 00:00" ;
    dy.long_name = "initial time" ;
    dy.calendar = "gregorian"

    mbl = ncfile.createVariable('thetae_bl',dtype('float32').char,('time','lat','lon'),zlib=True)
    mlt = ncfile.createVariable('thetae_lt',dtype('float32').char,('time','lat','lon'),zlib=True)
    mslt = ncfile.createVariable('thetae_sat_lt',dtype('float32').char,('time','lat','lon'),zlib=True)
    mmt = ncfile.createVariable('thetae_mt',dtype('float32').char,('time','lat','lon'),zlib=True)
    msmt = ncfile.createVariable('thetae_sat_mt',dtype('float32').char,('time','lat','lon'),zlib=True)

    mbl.description="thetae averaged from surface to 100 mb up" 
    mlt.description="thetae averaged from 100 mb above surface to 500 mb "
    mslt.description="SATURATED thetae averaged from 100 mb above surface to 500 mb "

    dy[:]=tyme
    lt[:]=lat
    ln[:]=lon

    mbl[:]=mse_bl
    mlt[:]=mse_lt
    mslt[:]=mse_sat_lt
    mmt[:]=mse_mt
    msmt[:]=mse_sat_mt

    ncfile.close()

    print('FILE WRITTEN')
    stdout.flush()
    et=time.time()
    print('Iteration took %.2f minutes'%((et-s)/60.))
