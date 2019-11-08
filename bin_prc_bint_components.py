'''
PURPOSE: To bin the precipitation vs. Bint and a two-dimensional binnning
         between its two components.

AUTHOR: Fiaz Ahmed  

DATE: 08/23/19 

'''

import datetime as dt
import numpy as np
from netCDF4 import Dataset
from dateutil.relativedelta import relativedelta
import glob,itertools
from sys import exit,stdout
#import h5py
from bin_parameters import * ## Import all the region indices and relevant variables
# import bin_cython1
from dateutil.relativedelta import relativedelta
import time
from mpi4py import MPI
import pickle

print('STARTING SCRIPT')

mmpersectommperhr=3600. ## Convert from m/s to mm/hr

ref_thetae=340 ## reference theta_e in K to convert buoy. to temp units
gravity=9.8 ##
thresh_pres=700 ## Filter all point below this surface pressure in hPa

#from ncdump import ncdump

### 0=ocean, 1=land, 2=lake, 3=small island, 4=ice shelf

##### Set Bin Option here ######
bin_ocean=1
bin_land=1
            
#############################

mask_land=np.copy(lsm)
mask_ocean=np.copy(lsm)

mask_land[mask_land!=0]=np.nan
mask_land[mask_land==0]=1.
mask_ocean[mask_ocean!=1]=np.nan

dts=[]

list1=[]
list2=[]
list_trmm=[]

months_jja=[6,7,8]
months_djf=[12,1,2]


## Load paths and file name information ##
# model_name='c96L32_am4g9_fullaero_MDTF'
# model_name='c96L48_am4b6_DDFull_MDTF'
model_name='CAM5.3'

dirc='/scratch/neelin/layer_thetae/'+model_name+'/'
fil=model_name

fname=dirc+fil+'*'+'thetae'+'*'
list1=(glob.glob(fname))
list1.sort()

print(fname)

 ##############################
 
dirc0='/scratch/neelin/MDTF_TIMESLICE_EXP/'+model_name+'/segmented_files/'
dirc1=dirc0+'pr/'
dirc2=dirc0+'ps/'

f=Dataset(list1[0],'r')
lat,lon=f.variables['lat'][:],f.variables['lon'][:]
f.close()

### Scatter Jobs #####
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

### Scatter jobs across cores
jobs=comm.scatter(jobs,root=0)

ocean_names=['io','wp','ep','at']
land_names_jja=['ism','wafr','easm']
land_names_djf=['sasm','asm','mc','arge']
land_names=land_names_djf+land_names_jja

precip_ca={}
counts={}
precip_counts={}

precip_ca_2d={}
counts_2d={}
precip_counts_2d={}

if bin_ocean==1:

    for i in ocean_names:
        precip_ca[i]=[]
        counts[i]=[]
        precip_counts[i]=[]

        precip_ca_2d[i]=[]
        counts_2d[i]=[]
        precip_counts_2d[i]=[]

if bin_land==1:

    for i in land_names:

        precip_ca[i]=[]
        counts[i]=[]
        precip_counts[i]=[]
        
        precip_ca_2d[i]=[]
        counts_2d[i]=[]
        precip_counts_2d[i]=[]

## If not using MPIPy: ##
# jobs=list1[:2]

# print(jobs)
# exit()

for j in jobs:

    d1=j[-31:-3]
    print(d1)
    
    ##### LOAD BINNING VARIABLES #####

    prc_model=[]
    Bint_model=[]
    cape_model=[]
    subsat_model=[]

    f=Dataset(j,"r")    
    thetae_lt=f.variables['thetae_lt'][:]
    thetae_sat_lt=f.variables['thetae_sat_lt'][:]
    thetae_bl=f.variables['thetae_bl'][:]
    f.close()
        
    fname1=dirc1+'*'+d1+'.nc'
    list_prc=(glob.glob(fname1))

    fname2=dirc2+'*'+d1+'.nc'
    list_ps=(glob.glob(fname2))
    
    f=Dataset(list_prc[0],'r')
    prc_model=f.variables['pr'][:]*mmpersectommperhr*1e6
    f.close()
        
    
    f=Dataset(list_ps[0],'r')
    surfp=f.variables['ps'][:]*1e-2
    f.close()

    ### Buoyancy computation ###
    
    delta_pl=surfp-100-500
    delta_pb=100
    
    wb=(delta_pb/delta_pl)*np.log((delta_pl+delta_pb)/delta_pb)
    wl=1-wb
    
    ### Filter all points whose surface pressure is < 750 mb ###
    ### these include high-altitude points. Orographic rain is dominated by convergence
    ### and perhaps less likely by the thermodynamics (?). A future framework to capture
    ### the dynamics will be useful.
    
    wb[surfp<thresh_pres]=np.nan
    wl[surfp<thresh_pres]=np.nan
    
    cape=ref_thetae*(thetae_bl-thetae_sat_lt)/thetae_sat_lt
    subsat=ref_thetae*(thetae_sat_lt-thetae_lt)/thetae_sat_lt    
    bint=gravity*(wb*(thetae_bl-thetae_sat_lt)/thetae_sat_lt-wl*(thetae_sat_lt-thetae_lt)/thetae_sat_lt)
    
    cape[surfp<thresh_pres]=np.nan
    subsat[surfp<thresh_pres]=np.nan
    bint[surfp<thresh_pres]=np.nan

    print(np.nanmax(thetae_bl),np.nanmin(thetae_bl))
    stdout.flush()

#     print(np.nanmax(bint),np.nanmin(bint))
#     print(np.nanmax(cape),np.nanmin(cape))
#     print(np.nanmax(subsat),np.nanmin(subsat))
#     print(np.nanmax(wb),np.nanmin(wb))
#     print(np.nanmax(wl),np.nanmin(wl))
        
    Bint_model=bint
    cape_model=cape
    subsat_model=subsat

#     Bint_model=np.asarray(Bint_model).reshape(-1,lat.size,lon.size)
#     cape_model=np.asarray(cape_model).reshape(-1,lat.size,lon.size)
#     subsat_model=np.asarray(subsat_model).reshape(-1,lat.size,lon.size)
#     prc_model=np.asarray(prc_model).reshape(-1,lat.size,lon.size)
# 

    ### Compute a value of integrated buoyancy

    Bint_model[prc_model<0]=np.nan
    cape_model[prc_model<0]=np.nan
    subsat_model[prc_model<0]=np.nan
    prc_model[prc_model<0]=np.nan
        
#     print(np.nanmax(Bint_model),np.nanmin(Bint_model))
#     print(np.nanmax(cape_model),np.nanmin(cape_model))
#     print(np.nanmax(subsat_model),np.nanmin(subsat_model))
    
    if bin_ocean==1:

        Bint_ocean=np.copy(Bint_model)*mask_land    
        prc_model_ocean=np.copy(prc_model)*mask_land
        cape_model_ocean=np.copy(cape_model)*mask_land    
        subsat_model_ocean=np.copy(subsat_model)*mask_land    
        
    if bin_land==1:

        Bint_land=np.copy(Bint_model)*mask_ocean    
        prc_model_land=np.copy(prc_model)*mask_ocean
        cape_model_land=np.copy(cape_model)*mask_ocean    
        subsat_model_land=np.copy(subsat_model)*mask_ocean    
        


    ### Buoyancy threshold ###
    bt=0.0
    print('FILES LOADED')
    
    ### APPLY MASK ####
    ### Create dictionaries that will hold the return values ###
    
    u={}       
    v={}
    w={}
    x={}
    z={}
    
    if bin_ocean==1:
                
        for ocn in ocean_names:
        
            indd=np.arange(Bint_ocean.shape[0])
            ixgrid1=np.ix_(indd,ilat1[ocn],ilon1[ocn])
            ixgrid2=np.ix_(indd,ilat2[ocn],ilon2[ocn])
            
            v[ocn]=np.concatenate((subsat_model_ocean[ixgrid1].flatten(),subsat_model_ocean[ixgrid2].flatten()))
            w[ocn]=np.concatenate((cape_model_ocean[ixgrid1].flatten(),cape_model_ocean[ixgrid2].flatten()))
            x[ocn]=np.concatenate((Bint_ocean[ixgrid1].flatten(),Bint_ocean[ixgrid2].flatten()))
            z[ocn]=np.concatenate((prc_model_ocean[ixgrid1].flatten(),prc_model_ocean[ixgrid2].flatten()))

            ret=bin_2d(w[ocn],v[ocn],z[ocn])
            precip_ca_2d[ocn].append(ret[0])
            counts_2d[ocn].append(ret[1])
            precip_counts_2d[ocn].append(ret[2])

            ret=bin_var_1d(x[ocn],z[ocn])
            precip_ca[ocn].append(ret[0])
            counts[ocn].append(ret[1])
            precip_counts[ocn].append(ret[2])


    if bin_land==1:
                
        reg_names=[]
        ### JJA ######        
#         if month in [6,7,8]: 
#             reg_names=land_names_jja
#             
#         elif month in [12,1,2]:
#             reg_names=land_names_djf

        ### Comment line below if jja vs. djf is required
        reg_names=land_names

        if reg_names:

            for lnd in reg_names:                
                    
                indd=np.arange(Bint_land.shape[0])
                ixgrid1=np.ix_(indd,ilat1[lnd],ilon1[lnd])

                v[lnd]=subsat_model_land[ixgrid1].flatten()
                w[lnd]=cape_model_land[ixgrid1].flatten()
                x[lnd]=Bint_land[ixgrid1].flatten()
                z[lnd]=prc_model_land[ixgrid1].flatten()

                ret=bin_2d(w[lnd],v[lnd],z[lnd])
                precip_ca_2d[lnd].append(ret[0])
                counts_2d[lnd].append(ret[1])
                precip_counts_2d[lnd].append(ret[2])
                
                ret=bin_var_1d(x[lnd],z[lnd])
                precip_ca[lnd].append(ret[0])
                counts[lnd].append(ret[1])
                precip_counts[lnd].append(ret[2])
                    

pcp_ca=comm.gather(precip_ca_2d,root=0)
cnts=comm.gather(counts_2d,root=0)
pcp_cnts=comm.gather(precip_counts_2d,root=0)

pcp_ca_1d=comm.gather(precip_ca,root=0)
cnts_1d=comm.gather(counts,root=0)
pcp_cnts_1d=comm.gather(precip_counts,root=0)

# #     pcp_ca=comm.gather(precip_ca,root=0)
# #     cnts=comm.gather(counts,root=0)
# #     pcp_cnts=comm.gather(precip_counts,root=0)
# #     qst=comm.gather(qsat,root=0)
# 
# ## Gathering, processing and saving to file ###
# 
if comm.rank==0:   

    if bin_ocean==1:

        for i in ocean_names:

            precip_ca_2d[i]=[]
            counts_2d[i]=[]
            precip_counts_2d[i]=[]
            
            precip_ca[i]=[]
            counts[i]=[]
            precip_counts[i]=[]

            precip_ca_2d[i].append([sl[i] for sl in pcp_ca ])
            counts_2d[i].append([sl[i] for sl in cnts])
            precip_counts_2d[i].append([sl[i] for sl in pcp_cnts])

            precip_ca_2d[i]=[k for sl in precip_ca_2d[i] for j in sl for k in j]
            counts_2d[i]=[k for sl in counts_2d[i] for j in sl for k in j]
            precip_counts_2d[i]=[k for sl in precip_counts_2d[i] for j in sl for k in j]

            precip_ca_2d[i]=np.sum(precip_ca_2d[i],0)   
            counts_2d[i]=np.sum(counts_2d[i],0)   
            precip_counts_2d[i]=np.sum(precip_counts_2d[i],0)   

            precip_ca[i].append([sl[i] for sl in pcp_ca_1d])
            counts[i].append([sl[i] for sl in cnts_1d])
            precip_counts[i].append([sl[i] for sl in pcp_cnts_1d])

            precip_ca[i]=[k for sl in precip_ca[i] for j in sl for k in j]
            counts[i]=[k for sl in counts[i] for j in sl for k in j]
            precip_counts[i]=[k for sl in precip_counts[i] for j in sl for k in j]

            precip_ca[i]=np.sum(precip_ca[i],0)   
            counts[i]=np.sum(counts[i],0)   
            precip_counts[i]=np.sum(precip_counts[i],0)   

            data=[precip_ca_2d,counts_2d,precip_counts_2d,
            precip_ca,counts,precip_counts,
            cape_bin_center,subsat_bin_center,buoy_bin_center]
    
        fname="ocean_"+model_name+"_binned_bint_components.dat"

        with open(fname, "wb") as f:
            pickle.dump(data, f)

    if bin_land==1:

        for i in land_names:

            precip_ca_2d[i]=[]
            counts_2d[i]=[]
            precip_counts_2d[i]=[]
            
            precip_ca[i]=[]
            counts[i]=[]
            precip_counts[i]=[]

            ##### 2D binning #######

            precip_ca_2d[i].append([sl[i] for sl in pcp_ca ])
            counts_2d[i].append([sl[i] for sl in cnts])
            precip_counts_2d[i].append([sl[i] for sl in pcp_cnts])

            precip_ca_2d[i]=[k for sl in precip_ca_2d[i] for j in sl for k in j]
            counts_2d[i]=[k for sl in counts_2d[i] for j in sl for k in j]
            precip_counts_2d[i]=[k for sl in precip_counts_2d[i] for j in sl for k in j]

            precip_ca_2d[i]=np.sum(precip_ca_2d[i],0)   
            counts_2d[i]=np.sum(counts_2d[i],0)   
            precip_counts_2d[i]=np.sum(precip_counts_2d[i],0)   

            ##### 1D binning #######
            
            precip_ca[i].append([sl[i] for sl in pcp_ca_1d])
            counts[i].append([sl[i] for sl in cnts_1d])
            precip_counts[i].append([sl[i] for sl in pcp_cnts_1d])

            precip_ca[i]=[k for sl in precip_ca[i] for j in sl for k in j]
            counts[i]=[k for sl in counts[i] for j in sl for k in j]
            precip_counts[i]=[k for sl in precip_counts[i] for j in sl for k in j]

            precip_ca[i]=np.sum(precip_ca[i],0)   
            counts[i]=np.sum(counts[i],0)   
            precip_counts[i]=np.sum(precip_counts[i],0) 
            

            data=[precip_ca_2d,counts_2d,precip_counts_2d,
            precip_ca,counts,precip_counts,
            cape_bin_center,subsat_bin_center,buoy_bin_center]
            
        fname="land_"+model_name+"_binned_bint_components.dat"

        with open(fname, "wb") as f:
            pickle.dump(data, f)
        
               
               

    
