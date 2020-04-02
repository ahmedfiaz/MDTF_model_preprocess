# This file is part of the convective_transition_diag module of the MDTF code package (see mdtf/MDTF_v2.0/LICENSE.txt)

# ======================================================================
# convecTransBasic_util.py
#   
#   Provide functions called by convecTransBasic.py
#
#   This file is part of the Convective Transition Diagnostic Package 
#    and the MDTF code package. See LICENSE.txt for the license.
#
#   Including:
#    (1) convecTransBasic_binTave
#    (2) convecTransBasic_binQsatInt
#    (3) generate_region_mask
#    (4) convecTransBasic_calcTaveQsatInt
#    (5) convecTransBasic_calc_model
#    (6) convecTransBasic_loadAnalyzedData
#    (7) convecTransBasic_plot
#    
# ======================================================================
# Import standard Python packages
import numpy as np
import numba
import glob
import os
from numba import jit,autojit
import scipy.io
from scipy.interpolate import NearestNDInterpolator
from netCDF4 import Dataset
from vert_cython import find_closest_index_2D, compute_layer_thetae
# import matplotlib.pyplot as mp
# import matplotlib.cm as cm
import networkx
import datetime as dt
from sys import exit, stdout

# ======================================================================
# convecTransBasic_binTave
#  takes arguments and bins by CWV & tave bins

@jit(nopython=True)
def convecTransLev2_binThetae(lon_idx, NUMBER_CAPE_BIN, NUMBER_SUBSAT_BIN, 
 CAPE, SUBSAT, RAIN, p0, p1, p2):
    for lat_idx in np.arange(SUBSAT.shape[1]):
        subsat_idx=SUBSAT[:,lat_idx,lon_idx]
        cape_idx=CAPE[:,lat_idx,lon_idx]
        rain=RAIN[:,lat_idx,lon_idx]

        for time_idx in np.arange(SUBSAT.shape[0]):
            if (cape_idx[time_idx]<NUMBER_CAPE_BIN and cape_idx[time_idx]>=0 and subsat_idx[time_idx]<NUMBER_SUBSAT_BIN):
                p0[subsat_idx[time_idx],cape_idx[time_idx]]+=1
                p1[subsat_idx[time_idx],cape_idx[time_idx]]+=rain[time_idx]
                p2[subsat_idx[time_idx],cape_idx[time_idx]]+=rain[time_idx]**2
#                 if (rain[time_idx]>PRECIP_THRESHOLD):
#                     pe[subsat_idx[time_idx],cape_idx[time_idx]]+=1
#                 if (subsat_idx[time_idx]+1>(0.6/SUBSAT_BIN_WIDTH)*qsat_int[time_idx]):
#                     q0[cape_idx[time_idx]]+=1
#                     q1[cape_idx[time_idx]]+=qsat_int[time_idx]


# ======================================================================
# convecTransBasic_binTave
#  takes arguments and bins by CWV & tave bins

@jit(nopython=True)
def convecTransBasic_binTave(lon_idx, CWV_BIN_WIDTH, NUMBER_OF_REGIONS, NUMBER_TEMP_BIN, NUMBER_CWV_BIN, PRECIP_THRESHOLD, REGION, CWV, RAIN, temp, QSAT_INT, p0, p1, p2, pe, q0, q1):
    for lat_idx in np.arange(CWV.shape[1]):
        reg=REGION[lon_idx,lat_idx]
        if (reg>0 and reg<=NUMBER_OF_REGIONS):
            cwv_idx=CWV[:,lat_idx,lon_idx]
            rain=RAIN[:,lat_idx,lon_idx]
            temp_idx=temp[:,lat_idx,lon_idx]
            qsat_int=QSAT_INT[:,lat_idx,lon_idx]
            for time_idx in np.arange(CWV.shape[0]):
                if (temp_idx[time_idx]<NUMBER_TEMP_BIN and temp_idx[time_idx]>=0 and cwv_idx[time_idx]<NUMBER_CWV_BIN):
                    p0[reg-1,cwv_idx[time_idx],temp_idx[time_idx]]+=1
                    p1[reg-1,cwv_idx[time_idx],temp_idx[time_idx]]+=rain[time_idx]
                    p2[reg-1,cwv_idx[time_idx],temp_idx[time_idx]]+=rain[time_idx]**2
                    if (rain[time_idx]>PRECIP_THRESHOLD):
                        pe[reg-1,cwv_idx[time_idx],temp_idx[time_idx]]+=1
                    if (cwv_idx[time_idx]+1>(0.6/CWV_BIN_WIDTH)*qsat_int[time_idx]):
                        q0[reg-1,temp_idx[time_idx]]+=1
                        q1[reg-1,temp_idx[time_idx]]+=qsat_int[time_idx]

# ======================================================================
# convecTransBasic_binQsatInt
#  takes arguments and bins by CWV & qsat_int bins

@jit(nopython=True)
def convecTransBasic_binQsatInt(lon_idx, NUMBER_OF_REGIONS, NUMBER_TEMP_BIN, NUMBER_CWV_BIN, PRECIP_THRESHOLD, REGION, CWV, RAIN, temp, p0, p1, p2, pe):
    for lat_idx in np.arange(CWV.shape[1]):
        reg=REGION[lon_idx,lat_idx]
        if (reg>0 and reg<=NUMBER_OF_REGIONS):
            cwv_idx=CWV[:,lat_idx,lon_idx]
            rain=RAIN[:,lat_idx,lon_idx]
            temp_idx=temp[:,lat_idx,lon_idx]
            for time_idx in np.arange(CWV.shape[0]):
                if (temp_idx[time_idx]<NUMBER_TEMP_BIN and temp_idx[time_idx]>=0 and cwv_idx[time_idx]<NUMBER_CWV_BIN):
                    p0[reg-1,cwv_idx[time_idx],temp_idx[time_idx]]+=1
                    p1[reg-1,cwv_idx[time_idx],temp_idx[time_idx]]+=rain[time_idx]
                    p2[reg-1,cwv_idx[time_idx],temp_idx[time_idx]]+=rain[time_idx]**2
                    if (rain[time_idx]>PRECIP_THRESHOLD):
                        pe[reg-1,cwv_idx[time_idx],temp_idx[time_idx]]+=1


### Compute the saturation vapor pressure for a given temperature ###

def es_calc(temp):

    #This function gives sat. vap. pressure (in Pa) for a temp value (in K)
    
	#get some constants:
	tmelt  = 273.15

	#convert inputs to proper units, forms
	tempc = temp - tmelt # in C
	tempcorig = tempc
	c=np.array((0.6105851e+03,0.4440316e+02,0.1430341e+01,0.2641412e-01,0.2995057e-03,0.2031998e-05,0.6936113e-08,0.2564861e-11,-.3704404e-13))

	#calc. es in hPa (!!!)
	#es = 6.112*EXP(17.67*tempc/(243.5+tempc))
	es=c[0]+tempc*(c[1]+tempc*(c[2]+tempc*(c[3]+tempc*(c[4]+tempc*(c[5]+tempc*(c[6]+tempc*(c[7]+tempc*c[8])))))))
	return es


# ======================================================================
# generate_region_mask
#  generates a map of integer values that correspond to regions using
#  the file region_0.25x0.25_costal2.5degExcluded.mat 
#  in var_data/convective_transition_diag
# Currently, there are 4 regions corresponding to ocean-only grid points
#  in the Western Pacific (WPac), Eastern Pacific (EPac),
#  Atlantic (Atl), and Indian (Ind) Ocean basins
# Coastal regions (within 2.5 degree with respect to sup-norm) are excluded

def generate_region_mask(region_mask_filename, model_netcdf_filename, lat_var, lon_var):
    
    print("   Generating region mask..."),

    # Load & Pre-process Region Mask
    matfile=scipy.io.loadmat(region_mask_filename)
    lat_m=matfile["lat"]
    lon_m=matfile["lon"] # 0.125~359.875 deg
    region=matfile["region"]
    lon_m=np.append(lon_m,np.reshape(lon_m[0,:],(-1,1))+360,0)
    lon_m=np.append(np.reshape(lon_m[-2,:],(-1,1))-360,lon_m,0)
    region=np.append(region,np.reshape(region[0,:],(-1,lat_m.size)),0)
    region=np.append(np.reshape(region[-2,:],(-1,lat_m.size)),region,0)

    LAT,LON=np.meshgrid(lat_m,lon_m,sparse=False,indexing="xy")
    LAT=np.reshape(LAT,(-1,1))
    LON=np.reshape(LON,(-1,1))
    REGION=np.reshape(region,(-1,1))

    LATLON=np.squeeze(np.array((LAT,LON)))
    LATLON=LATLON.transpose()

    regMaskInterpolator=NearestNDInterpolator(LATLON,REGION)

    # Interpolate Region Mask onto Model Grid using Nearest Grid Value
    pr_netcdf=Dataset(model_netcdf_filename,"r")
    lon=np.asarray(pr_netcdf.variables[lon_var][:],dtype="float")
    lat=np.asarray(pr_netcdf.variables[lat_var][:],dtype="float")
    pr_netcdf.close()
    if lon[lon<0.0].size>0:
        lon[lon[lon<0.0]]+=360.0
    lat=lat[np.logical_and(lat>=-20.0,lat<=20.0)]

    LAT,LON=np.meshgrid(lat,lon,sparse=False,indexing="xy")
    LAT=np.reshape(LAT,(-1,1))
    LON=np.reshape(LON,(-1,1))
    LATLON=np.squeeze(np.array((LAT,LON)))
    LATLON=LATLON.transpose()
    REGION=np.zeros(LAT.size)
    for latlon_idx in np.arange(REGION.shape[0]):
        REGION[latlon_idx]=regMaskInterpolator(LATLON[latlon_idx,:])
    REGION=np.reshape(REGION.astype(int),(-1,lat.size))
    
    print("...Generated!")

    return REGION

    # Use the following 3 lines for plotting the resulting region mask
    #REGION=np.reshape(REGION.astype(int),(-1,lat.size))
    #mp.contourf(lon.squeeze(), lat.squeeze(), REGION.T)
    #mp.axes().set_aspect('equal')

# ======================================================================
# convecTransLev2_calcthetae_ML
#  takes in 3D tropospheric temperature and specific humidity fields on model levels, 
# and calculates: thetae_LFT, thetae_sat_LFT & thetae_BL.
# Calculations will be broken up into chunks of time-period corresponding
#  to time_idx_delta with a default of 1000 time steps


def convecTransLev2_calcthetae_ML(ta_netcdf_filename, TA_VAR, hus_netcdf_filename, HUS_VAR,\
                        LEV_VAR, PS_VAR, A_VAR, B_VAR, MODEL, p_lev_mid, time_idx_delta,\
                        START_DATE, END_DATE, PARENT_DATE, TIME_STEP,\
                        SAVE_THETAE,PREPROCESSING_OUTPUT_DIR, THETAE_OUT,\
                        THETAE_LT_VAR,THETAE_SAT_LT_VAR,THETAE_BL_VAR,\
                        TIME_VAR,LAT_VAR,LON_VAR):


    strt_dt=dt.datetime.strptime(str(START_DATE),"%Y%m%d%H")
    end_dt=dt.datetime.strptime(str(END_DATE),"%Y%m%d%H")
    pt_date=dt.datetime.strptime(str(PARENT_DATE),"%Y%m%d%H")

    ### LOAD T & q ###

    ta_netcdf=Dataset(ta_netcdf_filename,"r")
    time=np.asarray(ta_netcdf.variables[TIME_VAR][:],dtype="float")
    lev=np.asarray(ta_netcdf.variables[LEV_VAR][:],dtype="float")
    a=np.asarray(ta_netcdf.variables[A_VAR][:],dtype="float")
    b=np.asarray(ta_netcdf.variables[B_VAR][:],dtype="float")
    lat=np.asarray(ta_netcdf.variables[LAT_VAR][:],dtype="float")
    lon=np.asarray(ta_netcdf.variables[LON_VAR][:],dtype="float")

    ## Take latitudinal slice
    ilatx=np.where(np.logical_and(lat>=-20.0,lat<=20.0))[0]
    lat=lat[ilatx]

    ps=np.asarray(ta_netcdf.variables[PS_VAR][:,ilatx,:],dtype="float")
    ta=np.asarray(ta_netcdf.variables[TA_VAR][:,:,ilatx,:],dtype="float")
    time_units=str([(j.units) for i,j in ta_netcdf.variables.items() if i=='time'][0])
    ps_units=str([(j.units) for i,j in ta_netcdf.variables.items() if i==PS_VAR][0])
    ta_netcdf.close()
    
    dates_ta=[pt_date+dt.timedelta(**{TIME_STEP:ti}) for ti in time]
    dates_ta_max,dates_ta_min=max(dates_ta),min(dates_ta)
    dates_indx=np.asarray([i for (i,idt) in enumerate(dates_ta) if (idt<dates_ta_max and idt>dates_ta_min)])
    
    ### READ SP.HUMIDITY ### 
        
    hus_netcdf=Dataset(hus_netcdf_filename,"r")
    hus=np.asarray(hus_netcdf.variables[HUS_VAR][:,:,ilatx,:],dtype="float")
    time=np.asarray(hus_netcdf.variables[TIME_VAR][:],dtype="float")
    hus_netcdf.close()

    dates_hus=[pt_date+dt.timedelta(**{TIME_STEP:ti}) for ti in time]    
    assert(len(dates_ta)==len(dates_hus))
    
    ### CREATE PRESSURE LEVELS ###
    pres=b[:,None,None,None]*ps[None,...]+a[:,None,None,None]

    ### Define the layers ###    
    pbl_top=ps-100e2 ## The sub-cloud layer is 100 mb thick ##
    low_top=np.zeros_like(ps)
    low_top[:]=500e2  # the mid-troposphere is fixed at 500 mb

    pbl_top=np.float_(pbl_top.flatten())
    low_top=np.float_(low_top.flatten())
    lev=pres.reshape(*pres.shape[:1],-1)
    
    pbl_ind=np.zeros(pbl_top.size,dtype=np.int64)
    low_ind=np.zeros(low_top.size,dtype=np.int64)

    find_closest_index_2D(pbl_top,lev,pbl_ind)
    find_closest_index_2D(low_top,lev,low_ind)

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
    Po=1025e2
    REF_THETAE=340.

    ### Saturation specific humidity and mixing ratio ###
    stdout.flush()

    ### Swap pressure axes for following computation ###
#     pres=np.swapaxes(pres,0,1)
    ta_flat=np.swapaxes(ta,0,1)
    ta_flat=ta_flat.reshape(*ta_flat.shape[:1],-1)

    hus_flat=np.swapaxes(hus,0,1)
    hus_flat=hus_flat.reshape(*hus_flat.shape[:1],-1)

    thetae_bl=np.zeros_like(pbl_top)
    thetae_lt=np.zeros_like(pbl_top)
    thetae_sat_lt=np.zeros_like(pbl_top)
    wb=np.zeros_like(pbl_top)
    
    ### Use trapezoidal rule for approximating the vertical integral ###
    ### vert. integ.=(b-a)*(f(a)+f(b))/2
    compute_layer_thetae(ta_flat, hus_flat, lev, pbl_ind, low_ind, thetae_bl, thetae_lt, thetae_sat_lt, wb)

    thetae_bl[thetae_bl==0]=np.nan
    thetae_lt[thetae_lt==0]=np.nan
    thetae_sat_lt[thetae_sat_lt==0]=np.nan

    ### Reshape all arrays ###
    thetae_bl=thetae_bl.reshape(ps.shape)
    thetae_lt=thetae_lt.reshape(ps.shape)
    thetae_sat_lt=thetae_sat_lt.reshape(ps.shape)

    print('      '+ta_netcdf_filename+" & "+hus_netcdf_filename+" pre-processed!")

#     Save Pre-Processed tave & qsat_int Fields
    if SAVE_THETAE==1:
#         Create PREPROCESSING_OUTPUT_DIR
        os.system("mkdir -p "+PREPROCESSING_OUTPUT_DIR)

#         Get necessary coordinates/variables for netCDF files

        thetae_output_filename=PREPROCESSING_OUTPUT_DIR+ta_netcdf_filename.split('/')[-1].replace(TA_VAR,THETAE_OUT)
        thetae_output_netcdf=Dataset(thetae_output_filename,"w",format="NETCDF4",zlib='True')
        thetae_output_netcdf.description="Theta_e averaged over the BL (100 hPa above surface) "\
                                    +"Theta_e and Theta_e_sat averaged over the LT (100 hPa above surface to 500 hPa) for "+MODEL
        thetae_output_netcdf.source="Convective Onset Statistics Diagnostic Package \
        - as part of the NOAA Model Diagnostic Task Force (MDTF) effort"

        lon_dim=thetae_output_netcdf.createDimension(LON_VAR,len(lon))
        lon_val=thetae_output_netcdf.createVariable(LON_VAR,np.float64,(LON_VAR,))
        lon_val.units="degree"
        lon_val[:]=lon

        lat_dim=thetae_output_netcdf.createDimension(LAT_VAR,len(lat))
        lat_val=thetae_output_netcdf.createVariable(LAT_VAR,np.float64,(LAT_VAR,))
        lat_val.units="degree_north"
        lat_val[:]=lat

        time_dim=thetae_output_netcdf.createDimension(TIME_VAR,None)
        time_val=thetae_output_netcdf.createVariable(TIME_VAR,np.float64,(TIME_VAR,))
        time_val.units=time_units
        time_val[:]=time

        thetabl_val=thetae_output_netcdf.createVariable(THETAE_BL_VAR,np.float64,(TIME_VAR,LAT_VAR,LON_VAR))
        thetabl_val.units="K"
        thetabl_val[:]=thetae_bl

        thetalt_val=thetae_output_netcdf.createVariable(THETAE_LT_VAR,np.float64,(TIME_VAR,LAT_VAR,LON_VAR))
        thetalt_val.units="K"
        thetalt_val[:]=thetae_lt

        thetalt_sat_val=thetae_output_netcdf.createVariable(THETAE_SAT_LT_VAR,np.float64,(TIME_VAR,LAT_VAR,LON_VAR))
        thetalt_sat_val.units="K"
        thetalt_sat_val[:]=thetae_sat_lt

        ps_val=thetae_output_netcdf.createVariable(PS_VAR,np.float64,(TIME_VAR,LAT_VAR,LON_VAR))
        ps_val.units="K"
        ps_val[:]=ps

        thetae_output_netcdf.close()

        print('      '+thetae_output_filename+" saved!")

    
    
def convecTransLev2_matchpcpta(ta_netcdf_filename, TA_VAR, pr_list, PR_VAR,\
    prc_list, PRC_VAR, MODEL, time_idx_delta,\
    START_DATE, END_DATE, PARENT_DATE, TIME_STEP, PRECIP_FACTOR, SAVE_PRECIP,\
    PREPROCESSING_OUTPUT_DIR, TIME_VAR,LAT_VAR,LON_VAR):
    
    strt_dt=dt.datetime.strptime(str(START_DATE),"%Y%m%d%H")
    end_dt=dt.datetime.strptime(str(END_DATE),"%Y%m%d%H")
    pt_date=dt.datetime.strptime(str(PARENT_DATE),"%Y%m%d%H")

    ### LOAD T & q ###

    ta_netcdf=Dataset(ta_netcdf_filename,"r")
    time=np.asarray(ta_netcdf.variables[TIME_VAR][:],dtype="float")
    time_units=str([(j.units) for i,j in ta_netcdf.variables.items() if i=='time'][0])
    ta_netcdf.close()
    
    pt_date=dt.datetime.strptime(str(PARENT_DATE),"%Y%m%d%H")
    dates_pr=[pt_date+dt.timedelta(**{TIME_STEP:ti}) for ti in time]    

    for i in pr_list:
    
        pr_netcdf=Dataset(i,"r")
        time_pr=np.asarray(pr_netcdf.variables[TIME_VAR][:],dtype="float")
        lat=np.asarray(pr_netcdf.variables[LAT_VAR][:],dtype="float")
        lon=np.asarray(pr_netcdf.variables[LON_VAR][:],dtype="float")

        ## Take latitudinal slice
        ilatx=np.where(np.logical_and(lat>=-20.0,lat<=20.0))[0]
        lat=lat[ilatx]

        pr_var=np.asarray(pr_netcdf.variables[PR_VAR][:,ilatx,:],dtype="float")*PRECIP_FACTOR
        pr_units=str([(j.units) for i,j in pr_netcdf.variables.items() if i==PR_VAR][0])
#         print(pr_units)
#         print(np.max(pr_var)*1e3,np.min(pr_var))
#         exit()
        ### Extract time of closest approach ###
        ## Note that 0.0625 is the time duration in fraction of the day ###        
        time_ind=([j for j,k in enumerate(time_pr) if abs((k-time)).min()<=0.0625])
        ## Choosing time so that the 3hrly avg. precip. is centered 1.5 hrs after the
        ## T,q measurement.
        
        time_ind=time_ind[1::2]
        time_ind_new=np.zeros((max(len(time_ind),time.size)))
        ### time_ind_new ensures that any mismatch is size is accounted for
        ### for now it specifically targets the case where time.size-len(time_ind)=1 
        ### We can easily generalize this case to time.size-len(time_ind)=n
        
        diff=time.size-len(time_ind)
        
        try:
            assert len(time_ind)==len(time)
            time_ind_new[:]=time_ind
        except:
            time_ind_new[:-diff]=time_ind
            time_ind_new[-(diff+1):-1]=np.nan 
               
        time_ind_new_fin=(np.int_(time_ind_new[np.isfinite(time_ind_new)]))
        time_ind_new_nan=np.isnan(np.int_(time_ind_new))
    
        ### Assuming that time is index 0
        pr_var_temp=np.zeros((time_ind_new.size,pr_var.shape[1],pr_var.shape[2]))
        
        assert diff>=0
        
        if diff==0:
            pr_var_temp[:,...]=pr_var[time_ind_new_fin,...]        
        else:               
            pr_var_temp[:-diff,...]=pr_var[time_ind_new_fin,...]
            pr_var_temp[-(diff+1):-1,...]=np.nan
    
        pr_netcdf.close()
        
        if SAVE_PRECIP==1:

    #         Create PREPROCESSING_OUTPUT_DIR
            os.system("mkdir -p "+PREPROCESSING_OUTPUT_DIR)

    #        Get necessary coordinates/variables for netCDF files

            pr_filename=PREPROCESSING_OUTPUT_DIR+ta_netcdf_filename.split('/')[-1].replace(TA_VAR,PR_VAR)
            
            pr_output_netcdf=Dataset(pr_filename,"w",format="NETCDF4",zlib='True')
            pr_output_netcdf.description="Precipitation extracted and matched to the nearest"\
                                        +"thermodynamic variable"+MODEL
            pr_output_netcdf.source="Convective Onset Statistics Diagnostic Package \
            - as part of the NOAA Model Diagnostic Task Force (MDTF) effort"

            lon_dim=pr_output_netcdf.createDimension(LON_VAR,len(lon))
            lon_val=pr_output_netcdf.createVariable(LON_VAR,np.float64,(LON_VAR,))
            lon_val.units="degree"
            lon_val[:]=lon

            lat_dim=pr_output_netcdf.createDimension(LAT_VAR,len(lat))
            lat_val=pr_output_netcdf.createVariable(LAT_VAR,np.float64,(LAT_VAR,))
            lat_val.units="degree_north"
            lat_val[:]=lat

            time_dim=pr_output_netcdf.createDimension(TIME_VAR,None)
            time_val=pr_output_netcdf.createVariable(TIME_VAR,np.float64,(TIME_VAR,))
            time_val.units=time_units
            time_val[:]=time

            pr_val=pr_output_netcdf.createVariable(PR_VAR,np.float64,(TIME_VAR,LAT_VAR,LON_VAR))
            pr_val.units=pr_units
            pr_val[:]=pr_var_temp
#             prc_val=prc_output_netcdf.createVariable(PR_VAR,np.float64,(TIME_VAR,LAT_VAR,LON_VAR))
#             prc_val.units="mm/hr"
#             prc_val[:]=prc_var_temp

            pr_output_netcdf.close()

            print('      '+pr_filename+" saved!")

        
        
        print(' Precip time series matched to thermo time series')
        
        
    

# ======================================================================
# convecTransBasic_calc_model
#  takes in ALL 2D pre-processed fields (precip, CWV, and EITHER tave or qsat_int),
#  calculates the binned data, and save it as a netCDF file
#  in the var_data/convective_transition_diag directory

def convecTransLev2_preprocess(*argsv):
    # ALLOCATE VARIABLES FOR EACH ARGUMENT
            
    BINT_BIN_WIDTH,\
    BINT_RANGE_MAX,\
    BINT_RANGE_MIN,\
    CAPE_RANGE_MIN,\
    CAPE_RANGE_MAX,\
    CAPE_BIN_WIDTH,\
    SUBSAT_RANGE_MIN,\
    SUBSAT_RANGE_MAX,\
    SUBSAT_BIN_WIDTH,\
    NUMBER_OF_REGIONS,\
    START_DATE,\
    END_DATE,\
    PARENT_DATE,\
    TIME_STEP,\
    pr_list,\
    PR_VAR,\
    prc_list,\
    PRC_VAR,\
    MODEL_OUTPUT_DIR,\
    THETAE_OUT,\
    thetae_list,\
    LFT_THETAE_VAR,\
    LFT_THETAE_SAT_VAR,\
    BL_THETAE_VAR,\
    ta_list,\
    TA_VAR,\
    hus_list,\
    HUS_VAR,\
    LEV_VAR,\
    PS_VAR,\
    A_VAR,\
    B_VAR,\
    MODEL,\
    p_lev_mid,\
    time_idx_delta,\
    SAVE_THETAE,\
    PREPROCESSING_OUTPUT_DIR,\
    PRECIP_THRESHOLD,\
    BIN_OUTPUT_DIR,\
    BIN_OUTPUT_FILENAME,\
    TIME_VAR,\
    LAT_VAR,\
    LON_VAR=argsv[0]
    
    print("   Start pre-processing atmospheric temperature & moisture fields...")
    for li in np.arange(len(ta_list)):
        convecTransLev2_calcthetae_ML(ta_list[li], TA_VAR, hus_list[li], HUS_VAR,\
                            LEV_VAR, PS_VAR, A_VAR, B_VAR, MODEL, p_lev_mid, time_idx_delta,\
                            START_DATE, END_DATE, PARENT_DATE, TIME_STEP,\
                            SAVE_THETAE, PREPROCESSING_OUTPUT_DIR,THETAE_OUT,\
                            LFT_THETAE_VAR,LFT_THETAE_SAT_VAR,BL_THETAE_VAR,\
                            TIME_VAR,LAT_VAR,LON_VAR)
    
                                
                                
def convecTransLev2_extractprecip(*argsv):
    # ALLOCATE VARIABLES FOR EACH ARGUMENT
            
    BINT_BIN_WIDTH,\
    BINT_RANGE_MAX,\
    BINT_RANGE_MIN,\
    CAPE_RANGE_MIN,\
    CAPE_RANGE_MAX,\
    CAPE_BIN_WIDTH,\
    SUBSAT_RANGE_MIN,\
    SUBSAT_RANGE_MAX,\
    SUBSAT_BIN_WIDTH,\
    NUMBER_OF_REGIONS,\
    START_DATE,\
    END_DATE,\
    PARENT_DATE,\
    TIME_STEP,\
    pr_list,\
    PR_VAR,\
    prc_list,\
    PRC_VAR,\
    MODEL_OUTPUT_DIR,\
    THETAE_OUT,\
    thetae_list,\
    LFT_THETAE_VAR,\
    LFT_THETAE_SAT_VAR,\
    BL_THETAE_VAR,\
    ta_list,\
    TA_VAR,\
    hus_list,\
    HUS_VAR,\
    LEV_VAR,\
    PS_VAR,\
    A_VAR,\
    B_VAR,\
    MODEL,\
    p_lev_mid,\
    time_idx_delta,\
    SAVE_THETAE,\
    SAVE_PRECIP,\
    PREPROCESSING_OUTPUT_DIR,\
    PRECIP_THRESHOLD,\
    PRECIP_FACTOR,\
    BIN_OUTPUT_DIR,\
    BIN_OUTPUT_FILENAME,\
    TIME_VAR,\
    LAT_VAR,\
    LON_VAR=argsv[0]
    
    print("   Start pre-processing atmospheric temperature & moisture fields...")
    for li in np.arange(len(ta_list)):
        convecTransLev2_matchpcpta(ta_list[li], TA_VAR, pr_list, PR_VAR,\
        prc_list, PRC_VAR, MODEL, time_idx_delta,\
        START_DATE, END_DATE, PARENT_DATE, TIME_STEP,\
        PRECIP_FACTOR, SAVE_PRECIP, PREPROCESSING_OUTPUT_DIR, TIME_VAR, LAT_VAR,LON_VAR)
              
                                            
def convecTransLev2_bin(*argsv):
    # ALLOCATE VARIABLES FOR EACH ARGUMENT
            
    BINT_BIN_WIDTH,\
    BINT_RANGE_MAX,\
    BINT_RANGE_MIN,\
    CAPE_RANGE_MIN,\
    CAPE_RANGE_MAX,\
    CAPE_BIN_WIDTH,\
    SUBSAT_RANGE_MIN,\
    SUBSAT_RANGE_MAX,\
    SUBSAT_BIN_WIDTH,\
    NUMBER_OF_REGIONS,\
    START_DATE,\
    END_DATE,\
    PARENT_DATE,\
    TIME_STEP,\
    pr_list,\
    PR_VAR,\
    prc_list,\
    PRC_VAR,\
    MODEL_OUTPUT_DIR,\
    THETAE_OUT,\
    thetae_list,\
    LFT_THETAE_VAR,\
    LFT_THETAE_SAT_VAR,\
    BL_THETAE_VAR,\
    ta_list,\
    TA_VAR,\
    hus_list,\
    HUS_VAR,\
    LEV_VAR,\
    PS_VAR,\
    A_VAR,\
    B_VAR,\
    MODEL,\
    p_lev_mid,\
    time_idx_delta,\
    SAVE_THETAE,\
    PREPROCESSING_OUTPUT_DIR,\
    PRECIP_THRESHOLD,\
    BIN_OUTPUT_DIR,\
    BIN_OUTPUT_FILENAME,\
    TIME_VAR,\
    LAT_VAR,\
    LON_VAR=argsv[0]
    
    # Re-load file lists for thetae_ave & precip.
    thetae_list=sorted(glob.glob(PREPROCESSING_OUTPUT_DIR+"/"+THETAE_OUT+'*'))
    pr_list=sorted(glob.glob(PREPROCESSING_OUTPUT_DIR+"/"+PR_VAR+'*'))
        
    # Define Bin Centers
    cape_bin_center=np.arange(CAPE_RANGE_MIN,CAPE_RANGE_MAX+CAPE_BIN_WIDTH,CAPE_BIN_WIDTH)
    subsat_bin_center=np.arange(SUBSAT_RANGE_MIN,SUBSAT_RANGE_MAX+SUBSAT_BIN_WIDTH,SUBSAT_BIN_WIDTH)

    NUMBER_CAPE_BIN=cape_bin_center.size
    NUMBER_SUBSAT_BIN=subsat_bin_center.size
    
    # Allocate Memory for Arrays
    P0=np.zeros((NUMBER_SUBSAT_BIN,NUMBER_CAPE_BIN))
    P1=np.zeros((NUMBER_SUBSAT_BIN,NUMBER_CAPE_BIN))
    P2=np.zeros((NUMBER_SUBSAT_BIN,NUMBER_CAPE_BIN))
    PE=np.zeros((NUMBER_SUBSAT_BIN,NUMBER_CAPE_BIN))

    ### Internal constants ###

    ref_thetae=340 ## reference theta_e in K to convert buoy. to temp units
    gravity=9.8 ### accl. due to gravity
    thresh_pres=700 ## Filter all point below this surface pressure in hPa

    for i,(j,k) in enumerate(zip(thetae_list,pr_list)):
        print(j,k)
        
        pr_netcdf=Dataset(k,'r')
        lat=np.asarray(pr_netcdf.variables[LAT_VAR][:],dtype="float")
        pr=np.squeeze(np.asarray(pr_netcdf.variables[PR_VAR][:,:,:],dtype="float"))
        pr_netcdf.close()
        print("      "+k+" Loaded!")

        thetae_netcdf=Dataset(j,'r')
        lat=np.asarray(thetae_netcdf.variables[LAT_VAR][:],dtype="float")
        thetae_bl=np.asarray(thetae_netcdf.variables[BL_THETAE_VAR][:],dtype="float")
        thetae_lt=np.asarray(thetae_netcdf.variables[LFT_THETAE_VAR][:],dtype="float")
        thetae_sat_lt=np.asarray(thetae_netcdf.variables[LFT_THETAE_SAT_VAR][:],dtype="float")
        ps=np.asarray(thetae_netcdf.variables[PS_VAR][:],dtype="float")
        thetae_netcdf.close()
        print("      "+j+" Loaded!")

        
        ps=ps*1e-2 ## Convert surface pressure to hPa
        
        delta_pl=ps-100-500
        delta_pb=100
        wb=(delta_pb/delta_pl)*np.log((delta_pl+delta_pb)/delta_pb)
        wl=1-wb
    
        wb[ps<thresh_pres]=np.nan
        wl[ps<thresh_pres]=np.nan

        cape=ref_thetae*(thetae_bl-thetae_sat_lt)/thetae_sat_lt
        subsat=ref_thetae*(thetae_sat_lt-thetae_lt)/thetae_sat_lt
        bint=gravity*(wb*(thetae_bl-thetae_sat_lt)/thetae_sat_lt-wl*(thetae_sat_lt-thetae_lt)/thetae_sat_lt)

        cape[ps<thresh_pres]=np.nan
        subsat[ps<thresh_pres]=np.nan
        bint[ps<thresh_pres]=np.nan

        print("      Binning...")
        
        ### Start binning
        SUBSAT=subsat/SUBSAT_BIN_WIDTH-0.5
        SUBSAT=SUBSAT.astype(int)

        CAPE=subsat/CAPE_BIN_WIDTH-0.5
        CAPE=CAPE.astype(int)

        RAIN=pr        
        RAIN[RAIN<0]=0 # Sometimes models produce negative rain rates
        
        
        # Binning is structured in the following way to avoid potential round-off issue
        #  (an issue arise when the total number of events reaches about 1e+8)
        for lon_idx in np.arange(SUBSAT.shape[2]):
            p0=np.zeros((NUMBER_SUBSAT_BIN,NUMBER_CAPE_BIN))
            p1=np.zeros((NUMBER_SUBSAT_BIN,NUMBER_CAPE_BIN))
            p2=np.zeros((NUMBER_SUBSAT_BIN,NUMBER_CAPE_BIN))
            pe=np.zeros((NUMBER_SUBSAT_BIN,NUMBER_CAPE_BIN))
            
            convecTransLev2_binThetae(lon_idx, NUMBER_CAPE_BIN, NUMBER_SUBSAT_BIN, 
            CAPE, SUBSAT, RAIN, p0, p1, p2)

            P0+=p0
            P1+=p1
            P2+=p2

        print("...Complete for current files!")

    print("   Total binning complete!")
    
    # Save Binning Results
    bin_output_netcdf=Dataset(BIN_OUTPUT_DIR+BIN_OUTPUT_FILENAME+".nc","w",format="NETCDF4")
            
    bin_output_netcdf.description="Convective Onset Buoyancy Statistics for "+MODEL
    bin_output_netcdf.source="Convective Onset Buoyancy Statistics Diagnostic Package \
    - as part of the NOAA Model Diagnostic Task Force (MDTF) effort"

    subsat_dim=bin_output_netcdf.createDimension("subsat",len(subsat_bin_center))
    subsat_var=bin_output_netcdf.createVariable("subsat",np.float64,("subsat",))
    subsat_var.units="K"
    subsat_var[:]=subsat_bin_center

    cape_dim=bin_output_netcdf.createDimension("cape",len(cape_bin_center))
    cape=bin_output_netcdf.createVariable("cape",np.float64,("cape"))
    cape.units="K"
    cape[:]=cape_bin_center

    p0=bin_output_netcdf.createVariable("P0",np.float64,("subsat","cape"))
    print(p0.shape,P0.shape)
    p0[:,:]=P0

    p1=bin_output_netcdf.createVariable("P1",np.float64,("subsat","cape"))
    p1.units="mm/hr"
    p1[:,:]=P1

    p2=bin_output_netcdf.createVariable("P2",np.float64,("subsat","cape"))
    p2.units="mm^2/hr^2"
    p2[:,:]=P2
    
    bin_output_netcdf.close()

    print("   Binned results saved as "+BIN_OUTPUT_DIR+"/"+BIN_OUTPUT_FILENAME+".nc!")

    return subsat_bin_center,cape_bin_center,P0,P1,P2

        
                
#         QSAT_INT=qsat_int
#         if BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1:
#             TAVE=tave
#             temp=(TAVE-temp_offset)/temp_bin_width
#         elif BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2:
#             temp=(QSAT_INT-temp_offset)/temp_bin_width
#         temp=temp.astype(int)


#     # Bulk Tropospheric Temperature Measure (1:tave, or 2:qsat_int)
#     if BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1:
#         tave_bin_center=np.arange(T_RANGE_MIN,T_RANGE_MAX+T_BIN_WIDTH,T_BIN_WIDTH)
#         temp_bin_center=tave_bin_center
#         temp_bin_width=T_BIN_WIDTH
#     elif BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2:
#         qsat_int_bin_center=np.arange(Q_RANGE_MIN,Q_RANGE_MAX+Q_BIN_WIDTH,Q_BIN_WIDTH)
#         temp_bin_center=qsat_int_bin_center
#         temp_bin_width=Q_BIN_WIDTH
#     
#     NUMBER_CWV_BIN=cwv_bin_center.size
#     NUMBER_TEMP_BIN=temp_bin_center.size
#     temp_offset=temp_bin_center[0]-0.5*temp_bin_width
# 
#     # Allocate Memory for Arrays
#     P0=np.zeros((NUMBER_OF_REGIONS,NUMBER_CWV_BIN,NUMBER_TEMP_BIN))
#     P1=np.zeros((NUMBER_OF_REGIONS,NUMBER_CWV_BIN,NUMBER_TEMP_BIN))
#     P2=np.zeros((NUMBER_OF_REGIONS,NUMBER_CWV_BIN,NUMBER_TEMP_BIN))
#     PE=np.zeros((NUMBER_OF_REGIONS,NUMBER_CWV_BIN,NUMBER_TEMP_BIN))
#     if BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1:
#         Q0=np.zeros((NUMBER_OF_REGIONS,NUMBER_TEMP_BIN))
#         Q1=np.zeros((NUMBER_OF_REGIONS,NUMBER_TEMP_BIN))

                   
def convecTransLev2_loadAnalyzedData(*argsv):

    bin_output_list,\
    LFT_THETAE_VAR,\
    LFT_THETAE_SAT_VAR,\
    BL_THETAE_VAR=argsv[0]
    
    if (len(bin_output_list)!=0):

        bin_output_filename=bin_output_list[0]    
        if bin_output_filename.split('.')[-1]=='nc':
            bin_output_netcdf=Dataset(bin_output_filename,"r")

            P0=np.asarray(bin_output_netcdf.variables["P0"][:,:],dtype="float")
            P1=np.asarray(bin_output_netcdf.variables["P1"][:,:],dtype="float")
            P2=np.asarray(bin_output_netcdf.variables["P2"][:,:],dtype="float")
#             PRECIP_THRESHOLD=bin_output_netcdf.getncattr("PRECIP_THRESHOLD")

            cape_bin_center=np.asarray(bin_output_netcdf.variables["cape"][:],dtype="float")
            subsat_bin_center=np.asarray(bin_output_netcdf.variables["subsat"][:],dtype="float")

            print(cape_bin_center[-1],cape_bin_center[0],P0.shape)
            print(subsat_bin_center[-1],subsat_bin_center[0],P1.shape)

            bin_output_netcdf.close()
            
        # Return CWV_BIN_WIDTH & PRECIP_THRESHOLD to make sure that
        #  user-specified parameters are consistent with existing data
        return subsat_bin_center,cape_bin_center,P0,P1,P2

    else: # If the binned model/obs data does not exist (in practive, for obs data only)   
        return [],[],[],[],[]




# ======================================================================
# convecTransBasic_calcTaveQsatInt
#  takes in 3D tropospheric temperature fields and calculates tave & qsat_int
# Calculations will be broken up into chunks of time-period corresponding
#  to time_idx_delta with a default of 1000 time steps
# Definition of column can be changed through p_lev_bottom & p_lev_top,
#  but the default filenames for tave & qsat_int do not contain column info



def convecTransBasic_calcTaveQsatInt(ta_netcdf_filename,TA_VAR,PRES_VAR,MODEL,\
                        p_lev_bottom,p_lev_top,dp,time_idx_delta,\
                        SAVE_TAVE_QSAT_INT,PREPROCESSING_OUTPUT_DIR,\
                        TAVE_VAR,QSAT_INT_VAR,TIME_VAR,LAT_VAR,LON_VAR):
    # Constants for calculating saturation vapor pressure
    Tk0 = 273.15 # Reference temperature.
    Es0 = 610.7 # Vapor pressure [Pa] at Tk0.
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
    # Calculate tave & qsat_int
    #  Column: 1000-200mb (+/- dp mb)
    ta_netcdf=Dataset(ta_netcdf_filename,"r")
    lat=np.asarray(ta_netcdf.variables[LAT_VAR][:],dtype="float")
    pfull=np.asarray(ta_netcdf.variables[PRES_VAR][:],dtype="float")
    if (max(pfull)>2000): # If units: Pa
        pfull*=0.01
    FLIP_PRES=(pfull[1]-pfull[0]<0)
    if FLIP_PRES:
        pfull=np.flipud(pfull)
    tave=np.array([])
    qsat_int=np.array([])

    time_idx_start=0

    print("      Pre-processing "+ta_netcdf_filename)

    while (time_idx_start<ta_netcdf.variables[TA_VAR].shape[0]):
        if (time_idx_start+time_idx_delta<=ta_netcdf.variables[TA_VAR].shape[0]):
            time_idx_end=time_idx_start+time_idx_delta
        else:
            time_idx_end=ta_netcdf.variables[TA_VAR].shape[0]

        print("         Integrate temperature field over "\
            +str(p_lev_bottom)+"-"+str(p_lev_top)+" hPa "\
            +"for time steps "\
            +str(time_idx_start)+"-"+str(time_idx_end))

        p_min=np.sum(pfull<=p_lev_top)-1
        if (pfull[p_min+1]<p_lev_top+dp):
            p_min=p_min+1
        p_max=np.sum(pfull<=p_lev_bottom)-1
        if (p_max+1<pfull.size and pfull[p_max]<p_lev_bottom-dp):
            p_max=p_max+1
        plev=np.copy(pfull[p_min:p_max+1])
        # ta[time,p,lat,lon]
        if FLIP_PRES:
            ta=np.asarray(ta_netcdf.variables[TA_VAR][time_idx_start:time_idx_end,pfull.size-(p_max+1):pfull.size-p_min,np.logical_and(lat>=-20.0,lat<=20.0),:],dtype="float")
            ta=np.fliplr(ta)
        else:
            ta=np.asarray(ta_netcdf.variables[TA_VAR][time_idx_start:time_idx_end,p_min:p_max+1,np.logical_and(lat>=-20.0,lat<=20.0),:],dtype="float")
        time_idx_start=time_idx_end
        p_max=p_max-p_min
        p_min=0

        if (plev[p_min]<p_lev_top-dp):
            # Update plev(p_min) <-- p_lev_top
            #  AND ta(p_min) <-- ta(p_lev_top) by interpolation
            ta[:,p_min,:,:]=ta[:,p_min,:,:] \
                            +(p_lev_top-plev[p_min]) \
                            /(plev[p_min+1]-plev[p_min]) \
                            *(ta[:,p_min+1,:,:]-ta[:,p_min,:,:])
            plev[p_min]=p_lev_top

        if (plev[p_max]>p_lev_bottom+dp):
            # Update plev(p_max) <-- p_lev_bottom
            #  AND Update ta(p_max) <-- ta(p_lev_bottom) by interpolation
            ta[:,p_max,:,:]=ta[:,p_max,:,:] \
                            +(p_lev_bottom-plev[p_max]) \
                            /(plev[p_max-1]-plev[p_max]) \
                            *(ta[:,p_max-1,:,:]-ta[:,p_max,:,:])
            plev[p_max]=p_lev_bottom

        if (plev[p_max]<p_lev_bottom-dp):
            # Update plev(p_max+1) <-- p_lev_bottom
            #  AND ta(p_max+1) <-- ta(p_lev_bottom) by extrapolation
            ta=np.append(ta,np.expand_dims(ta[:,p_max,:,:] \
                            +(p_lev_bottom-plev[p_max]) \
                            /(plev[p_max]-plev[p_max-1]) \
                            *(ta[:,p_max,:,:]-ta[:,p_max-1,:,:]),1), \
                            axis=1)
            plev=np.append(plev,p_lev_bottom)
            p_max=p_max+1

        # Integrate between level p_min and p_max
        tave_interim=ta[:,p_min,:,:]*(plev[p_min+1]-plev[p_min])
        for pidx in range(p_min+1,p_max-1+1):
            tave_interim=tave_interim+ta[:,pidx,:,:]*(plev[pidx+1]-plev[pidx-1])
        tave_interim=tave_interim+ta[:,p_max,:,:]*(plev[p_max]-plev[p_max-1])
        tave_interim=np.squeeze(tave_interim)/2/(plev[p_max]-plev[p_min])
        if (tave.size==0):
            tave=tave_interim
        else:
            tave=np.append(tave,tave_interim,axis=0)

        # Integrate Saturation Specific Humidity between level p_min and p_max 
        Es=Es0*(ta/Tk0)**((cpv-cl)/Rv)*np.exp((Lv0+(cl-cpv)*Tk0)/Rv*(1/Tk0-1/ta))
        qsat_interim=Es[:,p_min,:,:]*(plev[p_min+1]-plev[p_min])/plev[p_min]
        for pidx in range(p_min+1,p_max-1+1):
            qsat_interim=qsat_interim+Es[:,pidx,:,:]*(plev[pidx+1]-plev[pidx-1])/plev[pidx]
        qsat_interim=qsat_interim+Es[:,p_max,:,:]*(plev[p_max]-plev[p_max-1])/plev[p_max]
        qsat_interim=(epsilon/2/g)*qsat_interim
        if (qsat_int.size==0):
            qsat_int=qsat_interim
        else:
            qsat_int=np.append(qsat_int,qsat_interim,axis=0)

    ta_netcdf.close()
    # End-while time_idx_start

    print('      '+ta_netcdf_filename+" pre-processed!")

    # Save Pre-Processed tave & qsat_int Fields
    if SAVE_TAVE_QSAT_INT==1:
        # Create PREPROCESSING_OUTPUT_DIR
        os.system("mkdir -p "+PREPROCESSING_OUTPUT_DIR)

        # Get necessary coordinates/variables for netCDF files
        ta_netcdf=Dataset(ta_netcdf_filename,"r")
        time=ta_netcdf.variables[TIME_VAR]
        longitude=np.asarray(ta_netcdf.variables[LON_VAR][:],dtype="float")
        latitude=np.asarray(ta_netcdf.variables[LAT_VAR][:],dtype="float")
        latitude=latitude[np.logical_and(latitude>=-20.0,latitude<=20.0)]

        # Save 1000-200mb Column Average Temperature as tave
        tave_output_filename=PREPROCESSING_OUTPUT_DIR+"/"+ta_netcdf_filename.split('/')[-1].replace("."+TA_VAR+".","."+TAVE_VAR+".")
        tave_output_netcdf=Dataset(tave_output_filename,"w",format="NETCDF4")
        tave_output_netcdf.description=str(p_lev_bottom)+"-"+str(p_lev_top)+" hPa "\
                                    +"Mass-Weighted Column Average Temperature for "+MODEL
        tave_output_netcdf.source="Convective Onset Statistics Diagnostic Package \
        - as part of the NOAA Model Diagnostic Task Force (MDTF) effort"

        lon_dim=tave_output_netcdf.createDimension(LON_VAR,len(longitude))
        lon_val=tave_output_netcdf.createVariable(LON_VAR,np.float64,(LON_VAR,))
        lon_val.units="degree"
        lon_val[:]=longitude

        lat_dim=tave_output_netcdf.createDimension(LAT_VAR,len(latitude))
        lat_val=tave_output_netcdf.createVariable(LAT_VAR,np.float64,(LAT_VAR,))
        lat_val.units="degree_north"
        lat_val[:]=latitude

        time_dim=tave_output_netcdf.createDimension(TIME_VAR,None)
        time_val=tave_output_netcdf.createVariable(TIME_VAR,np.float64,(TIME_VAR,))
        time_val.units=time.units
        time_val[:]=time[:]

        tave_val=tave_output_netcdf.createVariable(TAVE_VAR,np.float64,(TIME_VAR,LAT_VAR,LON_VAR))
        tave_val.units="K"
        tave_val[:,:,:]=tave

        tave_output_netcdf.close()

        print('      '+tave_output_filename+" saved!")

        # Save 1000-200mb Column-integrated Saturation Specific Humidity as qsat_int
        qsat_int_output_filename=PREPROCESSING_OUTPUT_DIR+"/"+ta_netcdf_filename.split('/')[-1].replace("."+TA_VAR+".","."+QSAT_INT_VAR+".")
        qsat_int_output_netcdf=Dataset(qsat_int_output_filename,"w",format="NETCDF4")
        qsat_int_output_netcdf.description=str(p_lev_bottom)+"-"+str(p_lev_top)+" hPa "\
                                    +"Column-integrated Saturation Specific Humidity for "+MODEL
        qsat_int_output_netcdf.source="Convective Onset Statistics Diagnostic Package \
        - as part of the NOAA Model Diagnostic Task Force (MDTF) effort"

        lon_dim=qsat_int_output_netcdf.createDimension(LON_VAR,len(longitude))
        lon_val=qsat_int_output_netcdf.createVariable(LON_VAR,np.float64,(LON_VAR,))
        lon_val.units="degree"
        lon_val[:]=longitude

        lat_dim=qsat_int_output_netcdf.createDimension(LAT_VAR,len(latitude))
        lat_val=qsat_int_output_netcdf.createVariable(LAT_VAR,np.float64,(LAT_VAR,))
        lat_val.units="degree_north"
        lat_val[:]=latitude

        time_dim=qsat_int_output_netcdf.createDimension(TIME_VAR,None)
        time_val=qsat_int_output_netcdf.createVariable(TIME_VAR,np.float64,(TIME_VAR,))
        time_val.units=time.units
        time_val[:]=time[:]

        qsat_int_val=qsat_int_output_netcdf.createVariable(QSAT_INT_VAR,np.float64,(TIME_VAR,LAT_VAR,LON_VAR))
        qsat_int_val.units="mm"
        qsat_int_val[:,:,:]=qsat_int

        qsat_int_output_netcdf.close()

        print('      '+qsat_int_output_filename+" saved!")

        ta_netcdf.close()
    # End-if SAVE_TAVE_QSAT_INT==1

    return tave, qsat_int

# ======================================================================
# convecTransBasic_calc_model
#  takes in ALL 2D pre-processed fields (precip, CWV, and EITHER tave or qsat_int),
#  calculates the binned data, and save it as a netCDF file
#  in the var_data/convective_transition_diag directory

def convecTransBasic_calc_model(REGION,*argsv):
    # ALLOCATE VARIABLES FOR EACH ARGUMENT
    
    BULK_TROPOSPHERIC_TEMPERATURE_MEASURE, \
    CWV_BIN_WIDTH, \
    CWV_RANGE_MAX, \
    T_RANGE_MIN, \
    T_RANGE_MAX, \
    T_BIN_WIDTH, \
    Q_RANGE_MIN, \
    Q_RANGE_MAX, \
    Q_BIN_WIDTH, \
    NUMBER_OF_REGIONS, \
    pr_list, \
    PR_VAR, \
    prw_list, \
    PRW_VAR, \
    PREPROCESS_TA, \
    MODEL_OUTPUT_DIR, \
    qsat_int_list, \
    QSAT_INT_VAR, \
    tave_list, \
    TAVE_VAR, \
    ta_list, \
    TA_VAR, \
    PRES_VAR, \
    MODEL, \
    p_lev_bottom, \
    p_lev_top, \
    dp, \
    time_idx_delta, \
    SAVE_TAVE_QSAT_INT, \
    PREPROCESSING_OUTPUT_DIR, \
    PRECIP_THRESHOLD, \
    BIN_OUTPUT_DIR, \
    BIN_OUTPUT_FILENAME, \
    TIME_VAR, \
    LAT_VAR, \
    LON_VAR = argsv[0]

    # Pre-process temperature field if necessary
    if PREPROCESS_TA==1:
        print("   Start pre-processing atmospheric temperature fields...")
        for li in np.arange(len(pr_list)):
            convecTransBasic_calcTaveQsatInt(ta_list[li],TA_VAR,PRES_VAR,MODEL,\
                                p_lev_bottom,p_lev_top,dp,time_idx_delta,\
                                SAVE_TAVE_QSAT_INT,PREPROCESSING_OUTPUT_DIR,\
                                TAVE_VAR,QSAT_INT_VAR,TIME_VAR,LAT_VAR,LON_VAR)
        # Re-load file lists for tave & qsat_int
        tave_list=sorted(glob.glob(PREPROCESSING_OUTPUT_DIR+"/"+os.environ["tave_file"]))
        qsat_int_list=sorted(glob.glob(PREPROCESSING_OUTPUT_DIR+"/"+os.environ["qsat_int_file"]))
    
    # Allocate Memory for Arrays for Binning Output
    
    # Define Bin Centers
    cwv_bin_center=np.arange(CWV_BIN_WIDTH,CWV_RANGE_MAX+CWV_BIN_WIDTH,CWV_BIN_WIDTH)
    
    # Bulk Tropospheric Temperature Measure (1:tave, or 2:qsat_int)
    if BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1:
        tave_bin_center=np.arange(T_RANGE_MIN,T_RANGE_MAX+T_BIN_WIDTH,T_BIN_WIDTH)
        temp_bin_center=tave_bin_center
        temp_bin_width=T_BIN_WIDTH
    elif BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2:
        qsat_int_bin_center=np.arange(Q_RANGE_MIN,Q_RANGE_MAX+Q_BIN_WIDTH,Q_BIN_WIDTH)
        temp_bin_center=qsat_int_bin_center
        temp_bin_width=Q_BIN_WIDTH
    
    NUMBER_CWV_BIN=cwv_bin_center.size
    NUMBER_TEMP_BIN=temp_bin_center.size
    temp_offset=temp_bin_center[0]-0.5*temp_bin_width

    # Allocate Memory for Arrays
    P0=np.zeros((NUMBER_OF_REGIONS,NUMBER_CWV_BIN,NUMBER_TEMP_BIN))
    P1=np.zeros((NUMBER_OF_REGIONS,NUMBER_CWV_BIN,NUMBER_TEMP_BIN))
    P2=np.zeros((NUMBER_OF_REGIONS,NUMBER_CWV_BIN,NUMBER_TEMP_BIN))
    PE=np.zeros((NUMBER_OF_REGIONS,NUMBER_CWV_BIN,NUMBER_TEMP_BIN))
    if BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1:
        Q0=np.zeros((NUMBER_OF_REGIONS,NUMBER_TEMP_BIN))
        Q1=np.zeros((NUMBER_OF_REGIONS,NUMBER_TEMP_BIN))

    # Binning by calling convecTransBasic_binTave or convecTransBasic_binQsatInt

    print("   Start binning...")

    for li in np.arange(len(pr_list)):

        pr_netcdf=Dataset(pr_list[li],"r")
        lat=np.asarray(pr_netcdf.variables[LAT_VAR][:],dtype="float")
        pr=np.squeeze(np.asarray(pr_netcdf.variables[PR_VAR][:,:,:],dtype="float"))
        pr_netcdf.close()
        # Units: mm/s --> mm/hr
        pr=pr[:,np.logical_and(lat>=-20.0,lat<=20.0),:]*3.6e3*float(os.environ["pr_conversion_factor"])
        print("      "+pr_list[li]+" Loaded!")

        prw_netcdf=Dataset(prw_list[li],"r")
        lat=np.asarray(prw_netcdf.variables[LAT_VAR][:],dtype="float")
        prw=np.squeeze(np.asarray(prw_netcdf.variables[PRW_VAR][:,:,:],dtype="float"))
        prw_netcdf.close()
        prw=prw[:,np.logical_and(lat>=-20.0,lat<=20.0),:]
        print("      "+prw_list[li]+" Loaded!")
        
        qsat_int_netcdf=Dataset(qsat_int_list[li],"r")
        lat=np.asarray(qsat_int_netcdf.variables[LAT_VAR][:],dtype="float")
        qsat_int=np.squeeze(np.asarray(qsat_int_netcdf.variables[QSAT_INT_VAR][:,:,:],dtype="float"))
        qsat_int_netcdf.close()
        qsat_int=qsat_int[:,np.logical_and(lat>=-20.0,lat<=20.0),:]
            
        print("      "+qsat_int_list[li]+" Loaded!")
            
        if BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1:
            tave_netcdf=Dataset(tave_list[li],"r")
            lat=np.asarray(tave_netcdf.variables[LAT_VAR][:],dtype="float")
            tave=np.squeeze(np.asarray(tave_netcdf.variables[TAVE_VAR][:,:,:],dtype="float"))
            tave_netcdf.close()
            tave=tave[:,np.logical_and(lat>=-20.0,lat<=20.0),:]
            
            print("      "+tave_list[li]+" Loaded!")
           
        print("      Binning..."),
        
        ### Start binning
        CWV=prw/CWV_BIN_WIDTH-0.5
        CWV=CWV.astype(int)
        RAIN=pr
        
        RAIN[RAIN<0]=0 # Sometimes models produce negative rain rates
        QSAT_INT=qsat_int
        if BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1:
            TAVE=tave
            temp=(TAVE-temp_offset)/temp_bin_width
        elif BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2:
            temp=(QSAT_INT-temp_offset)/temp_bin_width
        temp=temp.astype(int)

        # Binning is structured in the following way to avoid potential round-off issue
        #  (an issue arise when the total number of events reaches about 1e+8)
        for lon_idx in np.arange(CWV.shape[2]):
            p0=np.zeros((NUMBER_OF_REGIONS,NUMBER_CWV_BIN,NUMBER_TEMP_BIN))
            p1=np.zeros((NUMBER_OF_REGIONS,NUMBER_CWV_BIN,NUMBER_TEMP_BIN))
            p2=np.zeros((NUMBER_OF_REGIONS,NUMBER_CWV_BIN,NUMBER_TEMP_BIN))
            pe=np.zeros((NUMBER_OF_REGIONS,NUMBER_CWV_BIN,NUMBER_TEMP_BIN))
            if BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1:
                q0=np.zeros((NUMBER_OF_REGIONS,NUMBER_TEMP_BIN))
                q1=np.zeros((NUMBER_OF_REGIONS,NUMBER_TEMP_BIN))
                convecTransBasic_binTave(lon_idx, CWV_BIN_WIDTH, \
                            NUMBER_OF_REGIONS, NUMBER_TEMP_BIN, NUMBER_CWV_BIN, PRECIP_THRESHOLD, \
                            REGION, CWV, RAIN, temp, QSAT_INT, \
                            p0, p1, p2, pe, q0, q1)
            elif BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2:
                convecTransBasic_binQsatInt(lon_idx, \
                            NUMBER_OF_REGIONS, NUMBER_TEMP_BIN, NUMBER_CWV_BIN, PRECIP_THRESHOLD, \
                            REGION, CWV, RAIN, temp, \
                            p0, p1, p2, pe)
            P0+=p0
            P1+=p1
            P2+=p2
            PE+=pe
            if BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1:
                Q0+=q0
                Q1+=q1
        # end-for lon_idx

        print("...Complete for current files!")
        
    print("   Total binning complete!")

    # Save Binning Results
    bin_output_netcdf=Dataset(BIN_OUTPUT_DIR+"/"+BIN_OUTPUT_FILENAME+".nc","w",format="NETCDF4")
            
    bin_output_netcdf.description="Convective Onset Statistics for "+MODEL
    bin_output_netcdf.source="Convective Onset Statistics Diagnostic Package \
    - as part of the NOAA Model Diagnostic Task Force (MDTF) effort"
    bin_output_netcdf.PRECIP_THRESHOLD=PRECIP_THRESHOLD

    region=bin_output_netcdf.createDimension("region",NUMBER_OF_REGIONS)
    reg=bin_output_netcdf.createVariable("region",np.float64,("region",))
    reg=np.arange(1,NUMBER_OF_REGIONS+1)

    cwv=bin_output_netcdf.createDimension("cwv",len(cwv_bin_center))
    prw=bin_output_netcdf.createVariable("cwv",np.float64,("cwv",))
    prw.units="mm"
    prw[:]=cwv_bin_center

    if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
        tave=bin_output_netcdf.createDimension(TAVE_VAR,len(tave_bin_center))
        temp=bin_output_netcdf.createVariable(TAVE_VAR,np.float64,(TAVE_VAR,))
        temp.units="K"
        temp[:]=tave_bin_center

        p0=bin_output_netcdf.createVariable("P0",np.float64,("region","cwv",TAVE_VAR))
        p0[:,:,:]=P0

        p1=bin_output_netcdf.createVariable("P1",np.float64,("region","cwv",TAVE_VAR))
        p1.units="mm/hr"
        p1[:,:,:]=P1

        p2=bin_output_netcdf.createVariable("P2",np.float64,("region","cwv",TAVE_VAR))
        p2.units="mm^2/hr^2"
        p2[:,:,:]=P2

        pe=bin_output_netcdf.createVariable("PE",np.float64,("region","cwv",TAVE_VAR))
        pe[:,:,:]=PE

        q0=bin_output_netcdf.createVariable("Q0",np.float64,("region",TAVE_VAR))
        q0[:,:]=Q0

        q1=bin_output_netcdf.createVariable("Q1",np.float64,("region",TAVE_VAR))
        q1.units="mm"
        q1[:,:]=Q1

    elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
        qsat_int=bin_output_netcdf.createDimension(QSAT_INT_VAR,len(qsat_int_bin_center))
        temp=bin_output_netcdf.createVariable(QSAT_INT_VAR,np.float64,(QSAT_INT_VAR,))
        temp.units="mm"
        temp[:]=qsat_int_bin_center

        p0=bin_output_netcdf.createVariable("P0",np.float64,("region","cwv",QSAT_INT_VAR))
        p0[:,:,:]=P0

        p1=bin_output_netcdf.createVariable("P1",np.float64,("region","cwv",QSAT_INT_VAR))
        p1.units="mm/hr"
        p1[:,:,:]=P1

        p2=bin_output_netcdf.createVariable("P2",np.float64,("region","cwv",QSAT_INT_VAR))
        p2.units="mm^2/hr^2"
        p2[:,:,:]=P2

        pe=bin_output_netcdf.createVariable("PE",np.float64,("region","cwv",QSAT_INT_VAR))
        pe[:,:,:]=PE

    bin_output_netcdf.close()

    print("   Binned results saved as "+BIN_OUTPUT_DIR+"/"+BIN_OUTPUT_FILENAME+".nc!")

    if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):    
        return cwv_bin_center,tave_bin_center,P0,P1,P2,PE,Q0,Q1,CWV_BIN_WIDTH,PRECIP_THRESHOLD
    elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
        return cwv_bin_center,qsat_int_bin_center,P0,P1,P2,PE,[],[],CWV_BIN_WIDTH,PRECIP_THRESHOLD

# ======================================================================
# convecTransBasic_loadAnalyzedData
#  loads the binned output calculated from convecTransBasic_calc_model
#  and prepares it for plotting

def convecTransBasic_loadAnalyzedData(*argsv):

    bin_output_list,\
    TAVE_VAR,\
    QSAT_INT_VAR,\
    BULK_TROPOSPHERIC_TEMPERATURE_MEASURE=argsv[0]
    
    if (len(bin_output_list)!=0):

        bin_output_filename=bin_output_list[0]    
        if bin_output_filename.split('.')[-1]=='nc':
            bin_output_netcdf=Dataset(bin_output_filename,"r")

            cwv_bin_center=np.asarray(bin_output_netcdf.variables["cwv"][:],dtype="float")
            P0=np.asarray(bin_output_netcdf.variables["P0"][:,:,:],dtype="float")
            P1=np.asarray(bin_output_netcdf.variables["P1"][:,:,:],dtype="float")
            P2=np.asarray(bin_output_netcdf.variables["P2"][:,:,:],dtype="float")
            PE=np.asarray(bin_output_netcdf.variables["PE"][:,:,:],dtype="float")
            PRECIP_THRESHOLD=bin_output_netcdf.getncattr("PRECIP_THRESHOLD")
            if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
                temp_bin_center=np.asarray(bin_output_netcdf.variables[TAVE_VAR][:],dtype="float")
                Q0=np.asarray(bin_output_netcdf.variables["Q0"][:,:],dtype="float")
                Q1=np.asarray(bin_output_netcdf.variables["Q1"][:,:],dtype="float") 
            elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
                temp_bin_center=np.asarray(bin_output_netcdf.variables[QSAT_INT_VAR][:],dtype="float")
                Q0=[]
                Q1=[]
            CWV_BIN_WIDTH=cwv_bin_center[1]-cwv_bin_center[0]
            bin_output_netcdf.close()
            
        elif bin_output_filename.split('.')[-1]=='mat':
            matfile=scipy.io.loadmat(bin_output_filename)

            cwv_bin_center=matfile['cwv']
            P0=matfile['P0'].astype(float)
            P1=matfile['P1']
            P2=matfile['P2']
            PE=matfile['PE'].astype(float)
            PRECIP_THRESHOLD=matfile['PRECIP_THRESHOLD'][0,0]
            if BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1:
                temp_bin_center=matfile[TAVE_VAR]
                Q0=matfile['Q0'].astype(float)
                Q1=matfile['Q1']
            elif BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2:
                temp_bin_center=matfile[QSAT_INT_VAR]
                Q0=[]
                Q1=[]
            CWV_BIN_WIDTH=cwv_bin_center[1][0]-cwv_bin_center[0][0]
    
        # Return CWV_BIN_WIDTH & PRECIP_THRESHOLD to make sure that
        #  user-specified parameters are consistent with existing data
        return cwv_bin_center,temp_bin_center,P0,P1,P2,PE,Q0,Q1,CWV_BIN_WIDTH,PRECIP_THRESHOLD

    else: # If the binned model/obs data does not exist (in practive, for obs data only)   
        return [],[],[],[],[],[],[],[],[],[]

# ======================================================================
# convecTransBasic_plot
#  takes output from convecTransBasic_loadAnalyzedData and saves the figure as a ps file

def convecTransBasic_plot(ret,argsv1,argsv2,*argsv3):

    print("Plotting...")

    # Load binned model data with parameters
    #  CBW:CWV_BIN_WIDTH, PT:PRECIP_THRESHOLD
    cwv_bin_center,\
    temp_bin_center,\
    P0,\
    P1,\
    P2,\
    PE,\
    Q0,\
    Q1,\
    CBW,\
    PT=ret
    
    # Load plotting parameters from convecTransBasic_usp_plot.py
    fig_params=argsv1

    # Load parameters from convecTransBasic_usp_calc.py
    #  Checking CWV_BIN_WIDTH & PRECIP_THRESHOLD 
    #  against CBW & PT guarantees the detected binned result
    #  is consistent with parameters defined in 
    #  convecTransBasic_usp_calc.py
    CWV_BIN_WIDTH,\
    PDF_THRESHOLD,\
    CWV_RANGE_THRESHOLD,\
    CP_THRESHOLD,\
    MODEL,\
    REGION_STR,\
    NUMBER_OF_REGIONS,\
    BULK_TROPOSPHERIC_TEMPERATURE_MEASURE,\
    PRECIP_THRESHOLD,\
    FIG_OUTPUT_DIR,\
    FIG_OUTPUT_FILENAME,\
    OBS,\
    RES,\
    REGION_STR_OBS,\
    FIG_OBS_DIR,\
    FIG_OBS_FILENAME,\
    USE_SAME_COLOR_MAP,\
    OVERLAY_OBS_ON_TOP_OF_MODEL_FIG=argsv3[0]

    # Load binned OBS data (default: R2TMIv7)
    cwv_bin_center_obs,\
    temp_bin_center_obs,\
    P0_obs,\
    P1_obs,\
    P2_obs,\
    PE_obs,\
    Q0_obs,\
    Q1_obs,\
    CWV_BIN_WIDTH_obs,\
    PT_obs=convecTransBasic_loadAnalyzedData(argsv2)

    # Check whether the detected binned MODEL data is consistent with User-Specified Parameters
    #  (Not all parameters, just 3)
    if (CBW!=CWV_BIN_WIDTH):
        print("==> Caution! The detected binned output has a CWV_BIN_WIDTH value "\
                +"different from the value specified in convecTransBasic_usp_calc.py!")
    if (PT!=PRECIP_THRESHOLD):
        print("==> Caution! The detected binned output has a PRECIP_THRESHOLD value "\
                +"different from the value specified in convecTransBasic_usp_calc.py!")
    if (P0.shape[0]!=NUMBER_OF_REGIONS):
        print("==> Caution! The detected binned output has a NUMBER_OF_REGIONS "\
                +"different from the value specified in convecTransBasic_usp_calc.py!")
    if (CBW!=CWV_BIN_WIDTH or PT!=PRECIP_THRESHOLD or P0.shape[0]!=NUMBER_OF_REGIONS):
        print("Caution! The detected binned output is inconsistent with  "\
                +"User-Specified Parameter(s) defined in convecTransBasic_usp_calc.py!")
        print("   Please double-check convecTransBasic_usp_calc.py, "\
                +"or if the required MODEL output exist, set BIN_ANYWAY=True "\
                +"in convecTransBasic_usp_calc.py!")

    ### Process/Plot binned OBS data
    #  if the binned OBS data exists, checking by P0_obs==[]
    if (P0_obs!=[]):
        # Post-binning Processing before Plotting
        P0_obs[P0_obs==0.0]=np.nan
        P_obs=P1_obs/P0_obs
        CP_obs=PE_obs/P0_obs
        PDF_obs=np.zeros(P0_obs.shape)
        for reg in np.arange(P0_obs.shape[0]):
            PDF_obs[reg,:,:]=P0_obs[reg,:,:]/np.nansum(P0_obs[reg,:,:])/CWV_BIN_WIDTH_obs
        # Bins with PDF>PDF_THRESHOLD
        pdf_gt_th_obs=np.zeros(PDF_obs.shape)
        with np.errstate(invalid="ignore"):
            pdf_gt_th_obs[PDF_obs>PDF_THRESHOLD]=1

        # Indicator of (temp,reg) with wide CWV range
        t_reg_I_obs=(np.squeeze(np.sum(pdf_gt_th_obs,axis=1))*CWV_BIN_WIDTH_obs>CWV_RANGE_THRESHOLD)

        ### Connected Component Section
        # The CWV_RANGE_THRESHOLD-Criterion must be satisfied by a connected component
        #  Default: off for MODEL/on for OBS/on for Fitting
        # Fot R2TMIv7 (OBS) this doesn't make much difference
        #  But when models behave "funny" one may miss by turning on this section
        # For fitting procedure (finding critical CWV at which the precip picks up)
        #  Default: on
        for reg in np.arange(P0_obs.shape[0]):
            for Tidx in np.arange(P0_obs.shape[2]):
                if t_reg_I_obs[reg,Tidx]:
                    G=networkx.DiGraph()
                    for cwv_idx in np.arange(pdf_gt_th_obs.shape[1]-1):
                        if (pdf_gt_th_obs[reg,cwv_idx,Tidx]>0 and pdf_gt_th_obs[reg,cwv_idx+1,Tidx]>0):
                            G.add_path([cwv_idx,cwv_idx+1])
                    largest = max(networkx.weakly_connected_component_subgraphs(G),key=len)
                    bcc=largest.nodes() # Biggest Connected Component
                    if (sum(pdf_gt_th_obs[reg,bcc,Tidx])*CWV_BIN_WIDTH_obs>CWV_RANGE_THRESHOLD):
                        t_reg_I_obs[reg,Tidx]=True
                        #pdf_gt_th_obs[reg,:,Tidx]=0
                        #pdf_gt_th_obs[reg,bcc,Tidx]=1
                    else:
                        t_reg_I_obs[reg,Tidx]=False
                        #pdf_gt_th_obs[reg,:,Tidx]=0
        ### End of Connected Component Section    

        # Copy P1, CP into p1, cp for (temp,reg) with "wide CWV range" & "large PDF"
        p1_obs=np.zeros(P1_obs.shape)
        cp_obs=np.zeros(CP_obs.shape)
        for reg in np.arange(P1_obs.shape[0]):
            for Tidx in np.arange(P1_obs.shape[2]):
                if t_reg_I_obs[reg,Tidx]:
                    p1_obs[reg,:,Tidx]=np.copy(P_obs[reg,:,Tidx])
                    cp_obs[reg,:,Tidx]=np.copy(CP_obs[reg,:,Tidx])
        p1_obs[pdf_gt_th_obs==0]=np.nan
        cp_obs[pdf_gt_th_obs==0]=np.nan
        pdf_obs=np.copy(PDF_obs)

        for reg in np.arange(P1_obs.shape[0]):
            for Tidx in np.arange(P1_obs.shape[2]):
                if (t_reg_I_obs[reg,Tidx] and cp_obs[reg,:,Tidx][cp_obs[reg,:,Tidx]>=0.0].size>0):
                    if (np.max(cp_obs[reg,:,Tidx][cp_obs[reg,:,Tidx]>=0])<CP_THRESHOLD):
                        t_reg_I_obs[reg,Tidx]=False
                else:
                    t_reg_I_obs[reg,Tidx]=False
                    
        for reg in np.arange(P1_obs.shape[0]):
            for Tidx in np.arange(P1_obs.shape[2]):
                if (~t_reg_I_obs[reg,Tidx]):
                    p1_obs[reg,:,Tidx]=np.nan
                    cp_obs[reg,:,Tidx]=np.nan
                    pdf_obs[reg,:,Tidx]=np.nan
        pdf_pe_obs=pdf_obs*cp_obs

        # Temperature range for plotting
        TEMP_MIN_obs=np.where(np.sum(t_reg_I_obs,axis=0)>=1)[0][0]
        TEMP_MAX_obs=np.where(np.sum(t_reg_I_obs,axis=0)>=1)[0][-1]
        # ======================================================================
        # ======================Start Plot OBS Binned Data======================
        # ======================================================================
        NoC=TEMP_MAX_obs-TEMP_MIN_obs+1 # Number of Colors
        scatter_colors = cm.jet(np.linspace(0,1,NoC,endpoint=True))

        axes_fontsize,legend_fonsize,marker_size,xtick_pad,figsize1,figsize2 = fig_params['f0'] 

        print("   Plotting OBS Figure..."),
        # create figure canvas
        fig_obs = mp.figure(figsize=(figsize1,figsize2))

        title_text=fig_obs.text(s='Convective Transition Basic Statistics ('+OBS+', '+RES+'$^{\circ}$)', x=0.5, y=1.02, ha='center', va='bottom', transform=fig_obs.transFigure, fontsize=16)

        for reg in np.arange(NUMBER_OF_REGIONS):
            # create figure 1
            ax1 = fig_obs.add_subplot(NUMBER_OF_REGIONS,4,1+reg*NUMBER_OF_REGIONS)
            ax1.set_xlim(fig_params['f1'][0])
            ax1.set_ylim(fig_params['f1'][1])
            ax1.set_xticks(fig_params['f1'][4])
            ax1.set_yticks(fig_params['f1'][5])
            ax1.tick_params(labelsize=axes_fontsize)
            ax1.tick_params(axis="x", pad=10)
            for Tidx in np.arange(TEMP_MIN_obs,TEMP_MAX_obs+1):
                if t_reg_I_obs[reg,Tidx]:
                    if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
                        ax1.scatter(cwv_bin_center_obs,p1_obs[reg,:,Tidx],\
                                    edgecolor="none",facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                    s=marker_size,clip_on=True,zorder=3,\
                                    label="{:.0f}".format(temp_bin_center_obs[Tidx]))
                    elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
                        ax1.scatter(cwv_bin_center_obs,p1_obs[reg,:,Tidx],\
                                    edgecolor="none",facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                    s=marker_size,clip_on=True,zorder=3,\
                                    label="{:.1f}".format(temp_bin_center_obs[Tidx]))
            for Tidx in np.arange(TEMP_MIN_obs,TEMP_MAX_obs+1):
                if t_reg_I_obs[reg,Tidx]:
                    if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
                        ax1.scatter(Q1_obs[reg,Tidx]/Q0_obs[reg,Tidx],fig_params['f1'][1][1]*0.98,\
                                    edgecolor=scatter_colors[Tidx-TEMP_MIN_obs,:]/2,facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                    s=marker_size,clip_on=True,zorder=3,marker="^",\
                                    label=': $\widehat{q_{sat}}$ (Column-integrated Saturation Specific Humidity)')
                    elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
                        ax1.scatter(temp_bin_center_obs[Tidx],fig_params['f1'][1][1]*0.98,\
                                    edgecolor=scatter_colors[Tidx-TEMP_MIN_obs,:]/2,facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                    s=marker_size,clip_on=True,zorder=3,marker="^",\
                                    label=': $\widehat{q_{sat}}$ (Column-integrated Saturation Specific Humidity)')
            ax1.set_xlabel(fig_params['f1'][2], fontsize=axes_fontsize)
            ax1.set_ylabel(fig_params['f1'][3], fontsize=axes_fontsize)
            ax1.grid()
            ax1.set_axisbelow(True)

            handles, labels = ax1.get_legend_handles_labels()
            num_handles = sum(t_reg_I_obs[reg,:])
            leg = ax1.legend(handles[0:num_handles], labels[0:num_handles], fontsize=axes_fontsize, bbox_to_anchor=(0.05,0.95), \
                            bbox_transform=ax1.transAxes, loc="upper left", borderaxespad=0, labelspacing=0.1, \
                            fancybox=False,scatterpoints=1,  framealpha=0, borderpad=0, \
                            handletextpad=0.1, markerscale=1, ncol=1, columnspacing=0.25)
            ax1.add_artist(leg)
            if reg==0:
                ax1_text = ax1.text(s='Precip. cond. avg. on CWV', x=0.5, y=1.05, transform=ax1.transAxes, fontsize=12, ha='center', va='bottom')

            # create figure 2 (probability pickup)
            ax2 = fig_obs.add_subplot(NUMBER_OF_REGIONS,4,2+reg*NUMBER_OF_REGIONS)
            ax2.set_xlim(fig_params['f2'][0])
            ax2.set_ylim(fig_params['f2'][1])
            ax2.set_xticks(fig_params['f2'][4])
            ax2.set_yticks(fig_params['f2'][5])
            ax2.tick_params(labelsize=axes_fontsize)
            ax2.tick_params(axis="x", pad=xtick_pad)
            for Tidx in np.arange(TEMP_MIN_obs,TEMP_MAX_obs+1):
                if t_reg_I_obs[reg,Tidx]:
                    ax2.scatter(cwv_bin_center_obs,cp_obs[reg,:,Tidx],\
                                edgecolor="none",facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                s=marker_size,clip_on=True,zorder=3)
            for Tidx in np.arange(TEMP_MIN_obs,TEMP_MAX_obs+1):
                if t_reg_I_obs[reg,Tidx]:
                    if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
                        ax2.scatter(Q1_obs[reg,Tidx]/Q0_obs[reg,Tidx],fig_params['f2'][1][1]*0.98,\
                                    edgecolor=scatter_colors[Tidx-TEMP_MIN_obs,:]/2,facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                    s=marker_size,clip_on=True,zorder=3,marker="^")
                    elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
                        ax2.scatter(temp_bin_center_obs[Tidx],fig_params['f2'][1][1]*0.98,\
                                    edgecolor=scatter_colors[Tidx-TEMP_MIN_obs,:]/2,facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                    s=marker_size,clip_on=True,zorder=3,marker="^")
            ax2.set_xlabel(fig_params['f2'][2], fontsize=axes_fontsize)
            ax2.set_ylabel(fig_params['f2'][3], fontsize=axes_fontsize)
            #ax2.text(0.05, 0.95, OBS, transform=ax2.transAxes, fontsize=12, fontweight="bold", verticalalignment="top")
            #ax2.text(0.05, 0.85, RES+"$^{\circ}$", transform=ax2.transAxes, fontsize=12, fontweight="bold", verticalalignment="top")
            ax2.text(0.05, 0.95, REGION_STR_OBS[reg], transform=ax2.transAxes, fontsize=12, fontweight="bold", verticalalignment="top")
            ax2.grid()
            ax2.set_axisbelow(True)
            if reg==0:
                ax2_text = ax2.text(s='Prob. of Precip.>'+str(PT_obs)+'mm/hr', x=0.5, y=1.05, transform=ax2.transAxes, fontsize=12, ha='center', va='bottom')

            # create figure 3 (normalized PDF)
            ax3 = fig_obs.add_subplot(NUMBER_OF_REGIONS,4,3+reg*NUMBER_OF_REGIONS)
            ax3.set_yscale("log")
            ax3.set_xlim(fig_params['f3'][0])
            ax3.set_ylim(fig_params['f3'][1])
            ax3.set_xticks(fig_params['f3'][4])
            ax3.tick_params(labelsize=axes_fontsize)
            ax3.tick_params(axis="x", pad=xtick_pad)
            for Tidx in np.arange(TEMP_MIN_obs,TEMP_MAX_obs+1):
                if t_reg_I_obs[reg,Tidx]:
                    ax3.scatter(cwv_bin_center_obs,PDF_obs[reg,:,Tidx],\
                                edgecolor="none",facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                s=marker_size,clip_on=True,zorder=3)
            for Tidx in np.arange(TEMP_MIN_obs,TEMP_MAX_obs+1):
                if t_reg_I_obs[reg,Tidx]:
                    if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
                        ax3.scatter(Q1_obs[reg,Tidx]/Q0_obs[reg,Tidx],fig_params['f3'][1][1]*0.83,\
                                    edgecolor=scatter_colors[Tidx-TEMP_MIN_obs,:]/2,facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                    s=marker_size,clip_on=True,zorder=3,marker="^")
                    elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
                        ax3.scatter(temp_bin_center_obs[Tidx],fig_params['f3'][1][1]*0.83,\
                                    edgecolor=scatter_colors[Tidx-TEMP_MIN_obs,:]/2,facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                    s=marker_size,clip_on=True,zorder=3,marker="^")
            ax3.set_xlabel(fig_params['f3'][2], fontsize=axes_fontsize)
            ax3.set_ylabel(fig_params['f3'][3], fontsize=axes_fontsize)
            ax3.grid()
            ax3.set_axisbelow(True)
            if reg==0:
                ax3_text = ax3.text(s='PDF of CWV', x=0.5, y=1.05, transform=ax3.transAxes, fontsize=12, ha='center', va='bottom')

            # create figure 4 (normalized PDF - precipitation)
            ax4 = fig_obs.add_subplot(NUMBER_OF_REGIONS,4,4+reg*NUMBER_OF_REGIONS)
            ax4.set_yscale("log")
            ax4.set_xlim(fig_params['f4'][0])
            ax4.set_ylim(fig_params['f4'][1])
            ax4.set_xticks(fig_params['f4'][4])
            ax4.tick_params(labelsize=axes_fontsize)
            ax4.tick_params(axis="x", pad=xtick_pad)
            for Tidx in np.arange(TEMP_MIN_obs,TEMP_MAX_obs+1):
                if t_reg_I_obs[reg,Tidx]:
                    ax4.scatter(cwv_bin_center_obs,pdf_pe_obs[reg,:,Tidx],\
                                edgecolor="none",facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                s=marker_size,clip_on=True,zorder=3)
            for Tidx in np.arange(TEMP_MIN_obs,TEMP_MAX_obs+1):
                if t_reg_I_obs[reg,Tidx]:
                    if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
                        ax4.scatter(Q1_obs[reg,Tidx]/Q0_obs[reg,Tidx],fig_params['f4'][1][1]*0.83,\
                                    edgecolor=scatter_colors[Tidx-TEMP_MIN_obs,:]/2,facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                    s=marker_size,clip_on=True,zorder=3,marker="^")
                    elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
                        ax4.scatter(temp_bin_center_obs[Tidx],fig_params['f4'][1][1]*0.83,\
                                    edgecolor=scatter_colors[Tidx-TEMP_MIN_obs,:]/2,facecolor=scatter_colors[Tidx-TEMP_MIN_obs,:],\
                                    s=marker_size,clip_on=True,zorder=3,marker="^")
            ax4.set_xlabel(fig_params['f4'][2], fontsize=axes_fontsize)
            ax4.set_ylabel(fig_params['f4'][3], fontsize=axes_fontsize)
            ax4.text(0.05, 0.95, "Precip > "+str(PT_obs)+" mm hr$^-$$^1$" , transform=ax4.transAxes, fontsize=12, verticalalignment="top")
            ax4.grid()
            ax4.set_axisbelow(True)
            if reg==0:
                ax4_text = ax4.text(s='PDF of CWV for Precip.>'+str(PT_obs)+'mm/hr', x=0.49, y=1.05, transform=ax4.transAxes, fontsize=12, ha='center', va='bottom')

        # now add a separate legend for triangles that represent column saturation values
        leg2 = ax1.legend([handles[num_handles]], [labels[num_handles]], fontsize=axes_fontsize, bbox_to_anchor=(0.0,-0.00), \
                            bbox_transform=fig_obs.transFigure, loc="upper left", borderaxespad=0, labelspacing=0.1, \
                            fancybox=False, scatterpoints=1,  framealpha=0, borderpad=0, \
                            handletextpad=0.1, markerscale=1, ncol=1, columnspacing=0.25)
        if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
            footnote_str='$\widehat{T}$ (Mass-weighted Column Average Temperature) used as the bulk tropospheric temperature measure'
        elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
            footnote_str='$\widehat{q_{sat}}$ (Column-integrated Saturation Specific Humidity) used as the bulk tropospheric temperature measure'
        footnote = ax1.text(s=footnote_str, x=0, y=-0.02, transform=fig_obs.transFigure, ha='left', va='top', fontsize=12)

        leg2.legendHandles[0].set_color('black')

        # set layout to tight (so that space between figures is minimized)
        fig_obs.tight_layout()
        fig_obs.savefig(FIG_OBS_DIR+"/"+FIG_OBS_FILENAME, bbox_inches="tight", bbox_extra_artists=(leg,title_text,footnote,))
        
        print("...Completed!")
        print("      OBS Figure saved as "+FIG_OBS_DIR+"/"+FIG_OBS_FILENAME+"!")
        # ======================================================================
        # =======================End Plot OBS Binned Data=======================
        # ======================================================================
    ### End of Process/Plot binned OBS data    

    # Post-binning Processing before Plotting
    P0[P0==0.0]=np.nan
    P=P1/P0
    CP=PE/P0
    PDF=np.zeros(P0.shape)
    for reg in np.arange(P0.shape[0]):
        PDF[reg,:,:]=P0[reg,:,:]/np.nansum(P0[reg,:,:])/CBW
    # Bins with PDF>PDF_THRESHOLD
    pdf_gt_th=np.zeros(PDF.shape)
    with np.errstate(invalid="ignore"):
        pdf_gt_th[PDF>PDF_THRESHOLD]=1

    # Indicator of (temp,reg) with wide CWV range
    t_reg_I=(np.squeeze(np.sum(pdf_gt_th,axis=1))*CBW>CWV_RANGE_THRESHOLD)

    ### Connected Component Section
    # The CWV_RANGE_THRESHOLD-Criterion must be satisfied by a connected component
    #  Default: off for MODEL/on for OBS/on for Fitting
    # Fot R2TMIv7 (OBS) this doesn't make much difference
    #  But when models behave "funny" one may miss by turning on this section
    # For fitting procedure (finding critical CWV at which the precip picks up)
    #  Default: on
#    for reg in np.arange(P0.shape[0]):
#        for Tidx in np.arange(P0.shape[2]):
#            if t_reg_I[reg,Tidx]:
#                G=networkx.DiGraph()
#                for cwv_idx in np.arange(pdf_gt_th.shape[1]-1):
#                    if (pdf_gt_th[reg,cwv_idx,Tidx]>0 and pdf_gt_th[reg,cwv_idx+1,Tidx]>0):
#                        G.add_path([cwv_idx,cwv_idx+1])
#                largest = max(networkx.weakly_connected_component_subgraphs(G),key=len)
#                bcc=largest.nodes() # Biggest Connected Component
#                if (sum(pdf_gt_th[reg,bcc,Tidx])*CBW>CWV_RANGE_THRESHOLD):
#                    t_reg_I[reg,Tidx]=True
#                    #pdf_gt_th[reg,:,Tidx]=0
#                    #pdf_gt_th[reg,bcc,Tidx]=1
#                else:
#                    t_reg_I[reg,Tidx]=False
#                    #pdf_gt_th[reg,:,Tidx]=0
    ### End of Connected Component Section    

    # Copy P1, CP into p1, cp for (temp,reg) with "wide CWV range" & "large PDF"
    p1=np.zeros(P1.shape)
    cp=np.zeros(CP.shape)
    for reg in np.arange(P1.shape[0]):
        for Tidx in np.arange(P1.shape[2]):
            if t_reg_I[reg,Tidx]:
                p1[reg,:,Tidx]=np.copy(P[reg,:,Tidx])
                cp[reg,:,Tidx]=np.copy(CP[reg,:,Tidx])
    p1[pdf_gt_th==0]=np.nan
    cp[pdf_gt_th==0]=np.nan
    pdf=np.copy(PDF)

    for reg in np.arange(P1.shape[0]):
        for Tidx in np.arange(P1.shape[2]):
            if (t_reg_I[reg,Tidx] and cp[reg,:,Tidx][cp[reg,:,Tidx]>=0.0].size>0):
                if (np.max(cp[reg,:,Tidx][cp[reg,:,Tidx]>=0])<CP_THRESHOLD):
                    t_reg_I[reg,Tidx]=False
            else:
                t_reg_I[reg,Tidx]=False
                
    for reg in np.arange(P1.shape[0]):
        for Tidx in np.arange(P1.shape[2]):
            if (~t_reg_I[reg,Tidx]):
                p1[reg,:,Tidx]=np.nan
                cp[reg,:,Tidx]=np.nan
                pdf[reg,:,Tidx]=np.nan
    pdf_pe=pdf*cp

    # Temperature range for plotting
    TEMP_MIN=np.where(np.sum(t_reg_I,axis=0)>=1)[0][0]
    TEMP_MAX=np.where(np.sum(t_reg_I,axis=0)>=1)[0][-1]
    # Use OBS to set colormap (but if they don't exist or users don't want to...)
    if (P0_obs==[] or USE_SAME_COLOR_MAP==False): 
        TEMP_MIN_obs=TEMP_MIN
        TEMP_MAX_obs=TEMP_MAX

    # ======================================================================
    # =====================Start Plot MODEL Binned Data=====================
    # ======================================================================
    NoC=TEMP_MAX_obs-TEMP_MIN_obs+1 # Number of Colors
    scatter_colors = cm.jet(np.linspace(0,1,NoC,endpoint=True))

    axes_fontsize,legend_fonsize,marker_size,xtick_pad,figsize1,figsize2 = fig_params['f0'] 

    print("   Plotting MODEL Figure..."),

    # create figure canvas
    fig = mp.figure(figsize=(figsize1,figsize2))

    title_text=fig.text(s='Convective Transition Basic Statistics ('+MODEL+')', x=0.5, y=1.02, ha='center', va='bottom', transform=fig.transFigure, fontsize=16)

    for reg in np.arange(NUMBER_OF_REGIONS):
        # create figure 1
        ax1 = fig.add_subplot(NUMBER_OF_REGIONS,4,1+reg*NUMBER_OF_REGIONS)
        ax1.set_xlim(fig_params['f1'][0])
        ax1.set_ylim(fig_params['f1'][1])
        ax1.set_xticks(fig_params['f1'][4])
        ax1.set_yticks(fig_params['f1'][5])
        ax1.tick_params(labelsize=axes_fontsize)
        ax1.tick_params(axis="x", pad=10)
        for Tidx in np.arange(TEMP_MIN,TEMP_MAX+1):
            if t_reg_I[reg,Tidx]:
                if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
                    ax1.scatter(cwv_bin_center,p1[reg,:,Tidx],\
                                edgecolor="none",facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                                s=marker_size,clip_on=True,zorder=3,\
                                label="{:.0f}".format(temp_bin_center[Tidx]))
                elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
                    ax1.scatter(cwv_bin_center,p1[reg,:,Tidx],\
                                edgecolor="none",facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                                s=marker_size,clip_on=True,zorder=3,\
                                label="{:.1f}".format(temp_bin_center[Tidx]))
        for Tidx in np.arange(min(TEMP_MIN_obs,TEMP_MIN),max(TEMP_MAX_obs+1,TEMP_MAX+1)):
            if (OVERLAY_OBS_ON_TOP_OF_MODEL_FIG and \
                P0_obs!=[] and t_reg_I_obs[reg,Tidx]):
                ax1.scatter(cwv_bin_center_obs,p1_obs[reg,:,Tidx],\
                            edgecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:]/2,\
                            facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                            s=marker_size/5,clip_on=True,zorder=3,\
                            label='Statistics for '+OBS+' (spatial resolution: '+RES+'$^{\circ}$)')
        for Tidx in np.arange(TEMP_MIN,TEMP_MAX+1):
            if t_reg_I[reg,Tidx]:
                if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
                    ax1.scatter(Q1[reg,Tidx]/Q0[reg,Tidx],fig_params['f1'][1][1]*0.98,\
                                edgecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:]/2,facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                                s=marker_size,clip_on=True,zorder=4,marker="^",\
                                label=': $\widehat{q_{sat}}$ (Column-integrated Saturation Specific Humidity)')
                elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
                    ax1.scatter(temp_bin_center[Tidx],fig_params['f1'][1][1]*0.98,\
                                edgecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:]/2,facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                                s=marker_size,clip_on=True,zorder=4,marker="^",\
                                label=': $\widehat{q_{sat}}$ (Column-integrated Saturation Specific Humidity)')
        ax1.set_xlabel(fig_params['f1'][2], fontsize=axes_fontsize)
        ax1.set_ylabel(fig_params['f1'][3], fontsize=axes_fontsize)
        ax1.grid()
        ax1.set_axisbelow(True)

        handles, labels = ax1.get_legend_handles_labels()
        num_handles = sum(t_reg_I[reg,:])
        leg = ax1.legend(handles[0:num_handles], labels[0:num_handles], fontsize=axes_fontsize, bbox_to_anchor=(0.05,0.95), \
                        bbox_transform=ax1.transAxes, loc="upper left", borderaxespad=0, labelspacing=0.1, \
                        fancybox=False,scatterpoints=1,  framealpha=0, borderpad=0, \
                        handletextpad=0.1, markerscale=1, ncol=1, columnspacing=0.25)
        ax1.add_artist(leg)
        if reg==0:
            ax1_text = ax1.text(s='Precip. cond. avg. on CWV', x=0.5, y=1.05, transform=ax1.transAxes, fontsize=12, ha='center', va='bottom')

        # create figure 2 (probability pickup)
        ax2 = fig.add_subplot(NUMBER_OF_REGIONS,4,2+reg*NUMBER_OF_REGIONS)
        ax2.set_xlim(fig_params['f2'][0])
        ax2.set_ylim(fig_params['f2'][1])
        ax2.set_xticks(fig_params['f2'][4])
        ax2.set_yticks(fig_params['f2'][5])
        ax2.tick_params(labelsize=axes_fontsize)
        ax2.tick_params(axis="x", pad=xtick_pad)
        for Tidx in np.arange(TEMP_MIN,TEMP_MAX+1):
            if t_reg_I[reg,Tidx]:
                ax2.scatter(cwv_bin_center,cp[reg,:,Tidx],\
                            edgecolor="none",facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                            s=marker_size,clip_on=True,zorder=3)
        for Tidx in np.arange(min(TEMP_MIN_obs,TEMP_MIN),max(TEMP_MAX_obs+1,TEMP_MAX+1)):
            if (OVERLAY_OBS_ON_TOP_OF_MODEL_FIG and \
                P0_obs!=[] and t_reg_I_obs[reg,Tidx]):
                ax2.scatter(cwv_bin_center_obs,cp_obs[reg,:,Tidx],\
                            edgecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:]/2,\
                            facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                            s=marker_size/5,clip_on=True,zorder=3)
        for Tidx in np.arange(TEMP_MIN,TEMP_MAX+1):
            if t_reg_I[reg,Tidx]:
                if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
                    ax2.scatter(Q1[reg,Tidx]/Q0[reg,Tidx],fig_params['f2'][1][1]*0.98,\
                                edgecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:]/2,facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                                s=marker_size,clip_on=True,zorder=4,marker="^")
                elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
                    ax2.scatter(temp_bin_center[Tidx],fig_params['f2'][1][1]*0.98,\
                                edgecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:]/2,facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                                s=marker_size,clip_on=True,zorder=4,marker="^")
        ax2.set_xlabel(fig_params['f2'][2], fontsize=axes_fontsize)
        ax2.set_ylabel(fig_params['f2'][3], fontsize=axes_fontsize)
        #ax2.text(0.05, 0.95, MODEL, transform=ax2.transAxes, fontsize=12, fontweight="bold", verticalalignment="top")
        ax2.text(0.05, 0.95, REGION_STR[reg], transform=ax2.transAxes, fontsize=12, fontweight="bold", verticalalignment="top")
#       if (OVERLAY_OBS_ON_TOP_OF_MODEL_FIG and P0_obs!=[]):
#           ax2.text(0.05, 0.7, OBS, transform=ax2.transAxes, fontsize=12, fontweight="bold", verticalalignment="top", color="0.3")
#           ax2.text(0.05, 0.6, RES+"$^{\circ}$", transform=ax2.transAxes, fontsize=12, fontweight="bold", verticalalignment="top", color="0.3")
        ax2.grid()
        ax2.set_axisbelow(True)
        if reg==0:
            ax2_text = ax2.text(s='Prob. of Precip.>'+str(PT)+'mm/hr', x=0.5, y=1.05, transform=ax2.transAxes, fontsize=12, ha='center', va='bottom')

        # create figure 3 (normalized PDF)
        ax3 = fig.add_subplot(NUMBER_OF_REGIONS,4,3+reg*NUMBER_OF_REGIONS)
        ax3.set_yscale("log")
        ax3.set_xlim(fig_params['f3'][0])
        ax3.set_ylim(fig_params['f3'][1])
        ax3.set_xticks(fig_params['f3'][4])
        ax3.tick_params(labelsize=axes_fontsize)
        ax3.tick_params(axis="x", pad=xtick_pad)
        for Tidx in np.arange(TEMP_MIN,TEMP_MAX+1):
            if t_reg_I[reg,Tidx]:
                ax3.scatter(cwv_bin_center,PDF[reg,:,Tidx],\
                            edgecolor="none",facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                            s=marker_size,clip_on=True,zorder=3)
        for Tidx in np.arange(min(TEMP_MIN_obs,TEMP_MIN),max(TEMP_MAX_obs+1,TEMP_MAX+1)):
            if (OVERLAY_OBS_ON_TOP_OF_MODEL_FIG and \
                P0_obs!=[] and t_reg_I_obs[reg,Tidx]):
                ax3.scatter(cwv_bin_center_obs,PDF_obs[reg,:,Tidx],\
                            edgecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:]/2,\
                            facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                            s=marker_size/5,clip_on=True,zorder=3)
        for Tidx in np.arange(TEMP_MIN,TEMP_MAX+1):
            if t_reg_I[reg,Tidx]:
                if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
                    ax3.scatter(Q1[reg,Tidx]/Q0[reg,Tidx],fig_params['f3'][1][1]*0.83,\
                                edgecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:]/2,facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                                s=marker_size,clip_on=True,zorder=4,marker="^")
                elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
                    ax3.scatter(temp_bin_center[Tidx],fig_params['f3'][1][1]*0.83,\
                                edgecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:]/2,facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                                s=marker_size,clip_on=True,zorder=4,marker="^")
        ax3.set_xlabel(fig_params['f3'][2], fontsize=axes_fontsize)
        ax3.set_ylabel(fig_params['f3'][3], fontsize=axes_fontsize)
        ax3.grid()
        ax3.set_axisbelow(True)
        if reg==0:
            ax3_text = ax3.text(s='PDF of CWV', x=0.5, y=1.05, transform=ax3.transAxes, fontsize=12, ha='center', va='bottom')

        # create figure 4 (normalized PDF - precipitation)
        ax4 = fig.add_subplot(NUMBER_OF_REGIONS,4,4+reg*NUMBER_OF_REGIONS)
        ax4.set_yscale("log")
        ax4.set_xlim(fig_params['f4'][0])
        ax4.set_ylim(fig_params['f4'][1])
        ax4.set_xticks(fig_params['f4'][4])
        ax4.tick_params(labelsize=axes_fontsize)
        ax4.tick_params(axis="x", pad=xtick_pad)
        for Tidx in np.arange(TEMP_MIN,TEMP_MAX+1):
            if t_reg_I[reg,Tidx]:
                ax4.scatter(cwv_bin_center,pdf_pe[reg,:,Tidx],\
                            edgecolor="none",facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                            s=marker_size,clip_on=True,zorder=3)
        for Tidx in np.arange(min(TEMP_MIN_obs,TEMP_MIN),max(TEMP_MAX_obs+1,TEMP_MAX+1)):
            if (OVERLAY_OBS_ON_TOP_OF_MODEL_FIG and \
                P0_obs!=[] and t_reg_I_obs[reg,Tidx]):
                ax4.scatter(cwv_bin_center_obs,pdf_pe_obs[reg,:,Tidx],\
                            edgecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:]/2,\
                            facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                            s=marker_size/5,clip_on=True,zorder=3)
        for Tidx in np.arange(TEMP_MIN,TEMP_MAX+1):
            if t_reg_I[reg,Tidx]:
                if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
                    ax4.scatter(Q1[reg,Tidx]/Q0[reg,Tidx],fig_params['f4'][1][1]*0.83,\
                                edgecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:]/2,facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                                s=marker_size,clip_on=True,zorder=4,marker="^")
                elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
                    ax4.scatter(temp_bin_center[Tidx],fig_params['f4'][1][1]*0.83,\
                                edgecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:]/2,facecolor=scatter_colors[(Tidx-TEMP_MIN_obs)%NoC,:],\
                                s=marker_size,clip_on=True,zorder=4,marker="^")
        ax4.set_xlabel(fig_params['f4'][2], fontsize=axes_fontsize)
        ax4.set_ylabel(fig_params['f4'][3], fontsize=axes_fontsize)
        ax4.text(0.05, 0.95, "Precip > "+str(PT)+" mm hr$^-$$^1$" , transform=ax4.transAxes, fontsize=12, verticalalignment="top")
        ax4.grid()
        ax4.set_axisbelow(True)
        if reg==0:
            ax4_text = ax4.text(s='PDF of CWV for Precip.>'+str(PT)+'mm/hr', x=0.49, y=1.05, transform=ax4.transAxes, fontsize=12, ha='center', va='bottom')

    # now add a separate legend for triangles that represent column saturation values
    leg2 = ax1.legend([handles[num_handles]], [labels[num_handles]], fontsize=axes_fontsize, bbox_to_anchor=(0.0,-0.00), \
                        bbox_transform=fig.transFigure, loc="upper left", borderaxespad=0, labelspacing=0.1, \
                        fancybox=False, scatterpoints=1,  framealpha=0, borderpad=0, \
                        handletextpad=0.1, markerscale=1, ncol=1, columnspacing=0.25)
    ax1.add_artist(leg2)
    leg3 = ax1.legend([handles[-1]], [labels[-1]], fontsize=axes_fontsize, bbox_to_anchor=(0.0,-0.02), \
                        bbox_transform=fig.transFigure, loc="upper left", borderaxespad=0, labelspacing=0.1, \
                        fancybox=False, scatterpoints=1,  framealpha=0, borderpad=0, \
                        handletextpad=0.1, markerscale=1, ncol=1, columnspacing=0.25)
    if (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1):
        footnote_str='$\widehat{T}$ (Mass-weighted Column Average Temperature) used as the bulk tropospheric temperature measure'
    elif (BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2):
        footnote_str='$\widehat{q_{sat}}$ (Column-integrated Saturation Specific Humidity) used as the bulk tropospheric temperature measure'
    
    if (OVERLAY_OBS_ON_TOP_OF_MODEL_FIG and P0_obs!=[]):
        leg2.legendHandles[0].set_color('black')
        footnote = ax1.text(s=footnote_str, x=0, y=-0.04, transform=fig.transFigure, ha='left', va='top', fontsize=12)
    else:
        footnote = ax1.text(s=footnote_str, x=0, y=-0.03, transform=fig.transFigure, ha='left', va='top', fontsize=12)
    leg3.legendHandles[0].set_color('black')

    # set layout to tight (so that space between figures is minimized)
    fig.tight_layout()
    fig.savefig(FIG_OUTPUT_DIR+"/"+FIG_OUTPUT_FILENAME, bbox_inches="tight", bbox_extra_artists=(leg,title_text,footnote,))
    
    print("...Completed!")
    print("      Figure saved as "+FIG_OUTPUT_DIR+"/"+FIG_OUTPUT_FILENAME+"!")
    # ======================================================================
    # ======================End Plot MODEL Binned Data======================
    # ======================================================================