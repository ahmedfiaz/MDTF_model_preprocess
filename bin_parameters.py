import numpy as np
from netCDF4 import Dataset
import bin_cython
### Input landsea mask ###
# f=Dataset('/home/fiaz/MDTF/files/LandSeaMask_am4g9.nc','r')
f=Dataset('/home/fiaz/MDTF/files/LandSeaMask_cam5.nc','r')
lsm=np.asarray(f.variables['LSMASK'][:],dtype='float')
lat,lon=f.variables['lat'][:],f.variables['lon'][:]
f.close()

print(lat,lon)


##### Setting the region indices #####

ilon1={}
ilat1={}

ilon2={}
ilat2={}

# print '****** IO *********'

ilon1['io']=np.where(np.logical_and(lon>=45,lon<100))[0]
ilat1['io']=np.where(np.logical_and(lat>=-25,lat<=25))[0]
# ilat1['io']=np.where(np.logical_and(lat>=-10,lat<=10))[0]

ilon2['io']=np.where(np.logical_and(lon>100,lon<125))[0]
ilat2['io']=np.where(np.logical_and(lat>=-25,lat<=-5))[0]
# ilat2['io']=np.where(np.logical_and(lat>=-10,lat<=-5))[0]

# print '****** WP *********'

ilon1['wp']=np.where(np.logical_and(lon>=125,lon<180))[0]
ilat1['wp']=np.where(np.logical_and(lat>=-25,lat<=25))[0]
# ilat1['wp']=np.where(np.logical_and(lat>=-10,lat<=10))[0]

ilon2['wp']=np.where(np.logical_and(lon>105,lon<125))[0]
ilat2['wp']=np.where(np.logical_and(lat<=15.,lat>=-5))[0]
# ilat2['wp']=np.where(np.logical_and(lat<=10.,lat>=-5))[0]

# print '****** EP *********'

# East Pacific (EPAC; 5-15 N, 100-120 W)

# ilon1['ep']=np.where(np.logical_and(lon>=240,lon<250))[0]
# ilat1['ep']=np.where(np.logical_and(lat>=5,lat<=15))[0]

ilon1['ep']=np.where(np.logical_and(lon>=180,lon<260))[0]
ilat1['ep']=np.where(np.logical_and(lat>=-25,lat<=25))[0]
# ilat1['ep']=np.where(np.logical_and(lat>=-10,lat<=10))[0]

ilon2['ep']=np.where(np.logical_and(lon>=260,lon<290))[0]
ilat2['ep']=np.where(np.logical_and(lat>=-25,lat<=10.))[0]
# ilat2['ep']=np.where(np.logical_and(lat>=-10,lat<=10.))[0]

# ilon2['ep']=np.where(np.logical_and(lon>=250,lon<260))[0]
# ilat2['ep']=np.where(np.logical_and(lat>=5,lat<=15))[0]


# print '******* AT ********'

# GATE ocean (GATE; 5-15 N, 20-40 W), and


# ilon1['at']=np.where(np.logical_or(lon>=320,lon<=330))[0]
# ilat1['at']=np.where(np.logical_and(lat>=5,lat<=15))[0]
# 
# ilon2['at']=np.where(np.logical_or(lon>=330,lon<=340))[0]
# ilat2['at']=np.where(np.logical_and(lat>=5,lat<=15))[0]

# ilon2['at']=[]
# ilat2['at']=[]

ilon1['at']=np.where(np.logical_or(lon>=300,lon<=15))[0]
ilat1['at']=np.where(np.logical_and(lat>=-25,lat<=25))[0]
# ilat1['at']=np.where(np.logical_and(lat>=-10,lat<=10))[0]

ilon2['at']=np.where(np.logical_and(lon>=290,lon<300))[0]
ilat2['at']=np.where(np.logical_and(lat>=-25,lat<=25))[0]
# ilat2['at']=np.where(np.logical_and(lat>=-10,lat<=10))[0]

##### Binning information #####

print('Setting BIN information')

pthresh=0.25

cwv_bins=np.arange(10,76,1)*1.
cwv_bins=np.append(cwv_bins,80)
cwv_bin_center=(cwv_bins[:-1]+cwv_bins[1:])*0.5
cwv_bin_width=np.diff(cwv_bins)

crh_bins=np.arange(0,0.98,.02)*1.
crh_bin_center=(crh_bins[:-1]+crh_bins[1:])*0.5
crh_bin_width=np.diff(crh_bins)


## Vertically integrated tropospheric temperature (might change if considering the reduced free troposphere)
# temp_bins=np.arange(260.5,272.5,1.)  
temp_bins=np.arange(266.5,277.5,1.)
# temp_bins=np.arange(20.,40.,4.)
temp_bin_center=(temp_bins[:-1]+temp_bins[1:])*0.5
temp_bin_width=np.diff(temp_bins)

##### Modified binning information #####

tsub_lt_bins=np.arange(0,30.5,0.5)*1.
tsub_lt_bin_center=(tsub_lt_bins[:-1]+tsub_lt_bins[1:])*0.5
tsub_lt_bin_width=np.diff(tsub_lt_bins)

#### BINS CURRENTLY IN USE FOR THEORETICAL VARIABLES ####

# tsub_lt_bins=np.arange(0,34.,4.)
# tsub_lt_bin_center=(tsub_lt_bins[:-1]+tsub_lt_bins[1:])*0.5
# tsub_lt_bin_width=np.diff(tsub_lt_bins)

# tsub_bins=np.arange(0,30.5,0.5)*1.
# tsub_bins=np.arange(0,.15,.05)*1.
# tsub_bins=np.arange(0,1.5,.05)*1e-1
# tsub_bins=np.arange(0,30.,2.)*1.
# tsub_bins=np.arange(0,16.,2.)
# tsub_bins=np.arange(0,28.,4)*1.

# tsub_bins=np.arange(-30.,6.,0.75)*1.
# tsub_bins=np.arange(-30.,6.,4.)*1.

# tsub_bins=np.arange(-10.,1.,.25)*1e-1
# tsub_bins=np.arange(0.,15,.25)*1e-1
# tsub_bins=np.arange(0,30.5,0.5)*1.

#### TSUB LT ####

# tsub_bins=np.arange(0,24.,2.)*1.
tsub_bins=np.arange(0,30.5,0.5)*1.
tsub_bin_center=(tsub_bins[:-1]+tsub_bins[1:])*0.5
tsub_bin_width=np.diff(tsub_bins)

#### TSUB BL ####

tsub_bl_bins=np.arange(0,24.,2.)
# tsub_bl_bins=np.arange(0,24.,4.)
# tsub_bl_bins=np.arange(0,30.5,.5)*1.
# tsub_bl_bins=np.arange(0,35.,1.)*1.
tsub_bl_bin_center=(tsub_bl_bins[:-1]+tsub_bl_bins[1:])*0.5
tsub_bl_bin_width=np.diff(tsub_bl_bins)

# tbl_bins=np.arange(336,358.,2.)
# tbl_bins=np.arange(310,362.5,2.5)
# tbl_bins=np.arange(326,358.,2.)
# tbl_bins=np.arange(325,350,.5)
# tbl_bins=np.arange(310,356.,8.)
# tbl_bins=np.arange(-35.,15.,4.)

# tbl_bins=np.arange(-10.,5.5,1.5)*1e-1
# tbl_bins=np.arange(-30.,16.,2.)
# tbl_bins=np.arange(310,355.,2.5)

# tbl_bins=np.arange(336,362.,4.)
tbl_bins=np.arange(334.0,356.,2.)  
# tbl_bins=np.arange(328,353.,.5)
tbl_bin_width=np.diff(tbl_bins)
tbl_bin_center=(tbl_bins[:-1]+tbl_bins[1:])*0.5

# ts_bins=np.arange(338.0,362.,2.)
# ts_bins=np.arange(336.0,352.,2.)  
# ts_bins=np.arange(324.0,358.,2.)  
# ts_bins=np.arange(330.0,358.,2.)  
# ts_bins=np.arange(338.0,342.,2.)  
# ts_bins=np.arange(310,362.5,2.5)
# ts_bins=np.arange(225,252.5,2.5)
# ts_bins=np.arange(325.0,360.,2.)  
# ts_bins=np.arange(325.0,360.,5.)
# ts_bins=np.arange(-24.0,16.,2.)
# ts_bins=np.arange(-1.4,.5,.05)   
# ts_bins=np.arange(-15.,15.5,1.)   
# ts_bins=np.arange(-15.,15.5,1.)   

# ts_bins=np.arange(-.3,.7,0.05)*1e-1
# ts_bins=np.arange(-12.,28.,2.)
# ts_bins=np.arange(310,355.,2.5)

#### TBL SAT ####

ts_bl_bins=np.arange(328,354.,2.)
# ts_bl_bins=np.arange(339.5,350.5,.5)  
ts_bl_bin_width=np.diff(ts_bl_bins)
ts_bl_bin_center=(ts_bl_bins[:-1]+ts_bl_bins[1:])*0.5

#### TS LT ####

ts_bins=np.arange(336.0,352.,2.)  
ts_bin_width=np.diff(ts_bins)
ts_bin_center=(ts_bins[:-1]+ts_bins[1:])*0.5

# ts_mt_bins=np.arange(330.0,358.,2.)  
ts_mt_bins=np.arange(-20.0,20.,4.)  
ts_mt_bin_width=np.diff(ts_mt_bins)
ts_mt_bin_center=(ts_mt_bins[:-1]+ts_mt_bins[1:])*0.5

ts_lt_bins=np.arange(315,370.,2.5)  
ts_lt_bin_width=np.diff(ts_lt_bins)
ts_lt_bin_center=(ts_lt_bins[:-1]+ts_lt_bins[1:])*0.5

#### TSUB MT ####

# tsub_mt_bins=np.arange(0,24.,4.)*1.
# tsub_mt_bins=np.arange(0,14.,2.)*1.
# tsub_mt_bins=np.arange(0,30.,2.5)*1.
# tsub_mt_bins=np.arange(0,32.,2.)
# tsub_mt_bins=np.arange(0,16.,2.)*1.
# tsub_mt_bins=np.arange(0,30.,2.)*1.
# tsub_mt_bins=np.arange(-34.,2.,4.)*1.
# tsub_mt_bins=np.arange(-34.,2.,.5)*1.
# tsub_mt_bins=np.arange(0.,5.5,.5)*1e-1
# tsub_mt_bins=np.arange(0,24.,4.)*1.
tsub_mt_bins=np.arange(0,14.,2.)*1.
tsub_mt_bin_center=(tsub_mt_bins[:-1]+tsub_mt_bins[1:])*0.5
tsub_mt_bin_width=np.diff(tsub_mt_bins)

### ORIGINAL ###
# buoy_bins=np.arange(-11,6.5,.125)*1e-1

### NO BL ###
# buoy_bins=np.arange(-5.,-3.5,.025)

### NO LT ###
# buoy_bins=np.arange(.35,.9,.0125)

### NO MT ###
# buoy_bins=np.arange(-.6,.975,.0125)

### NO TS ###
# buoy_bins=np.arange(-2.15,3.55,.025)
# buoy_bins=np.arange(-1.,1.,.125)

# buoy_bins=np.arange(1.8,3.5,.05)
# buoy_bins=np.arange(-5,15.125,.125)

# buoy_bins=np.arange(20,32.125,.125)
# buoy_bins=np.arange(12,22.125,.125)

# buoy_bins=np.arange(-9,4.,0.125)*1e-1
# buoy_bins=np.arange(-4.,-3.,0.0125)

buoy_bins=np.arange(-1.5,1.51,.01)
# buoy_bins=cwv_bins ### Use this if trying to bin by CWV
buoy_bin_center=(buoy_bins[1:]+buoy_bins[:-1])*0.5
buoy_bin_width=np.diff(buoy_bins)

cape_bins=np.arange(-40,17.0,1.0)
cape_bin_center=(cape_bins[1:]+cape_bins[:-1])*0.5
cape_bin_width=np.diff(cape_bins)

subsat_bins=np.arange(0,42.,1.0)
subsat_bin_center=(subsat_bins[1:]+subsat_bins[:-1])*0.5
subsat_bin_width=np.diff(subsat_bins)

#### BINS CURRENTLY IN USE FOR THEORETICAL VARIABLES ####

theta_es_ref=340. ## Reference theta_es in K

#########################

#### Precipitation bins for the joint PDFs
### Logarithmic binning - closely spaced near beginning
### and sparsely spaced near end

precip_bins=2**(np.arange(-1,5.,.3))
precip_bin_width=np.diff(precip_bins)
#precip_bins=np.arange(0,20.5,.5)
precip_bin_center=(precip_bins[1:]+precip_bins[:-1])*0.5
precip_bin_center=np.insert(precip_bin_center,0,0)

print('Getting region indices')

##### Get the region indices #####

ilon1['easm']=np.where(np.logical_and(lon>=105,lon<=125))[0]
ilat1['easm']=np.where(np.logical_and(lat>=15,lat<=25))[0]

ilon1['ism']=np.where(np.logical_and(lon>=75,lon<=90))[0]
ilat1['ism']=np.where(np.logical_and(lat>=5,lat<=25))[0]

ilon1['wafr']=np.where(np.logical_or(lon>=343,lon<=10))[0]
ilat1['wafr']=np.where(np.logical_and(lat>=0,lat<=15))[0]

ilon1['cong']=np.where(np.logical_or(lon>=15,lon<=30))[0]
ilat1['cong']=np.where(np.logical_and(lat>=-10,lat<=10))[0]

ilon1['mc']=np.where(np.logical_and(lon>=95,lon<=145))[0]
ilat1['mc']=np.where(np.logical_and(lat>=-10,lat<=10))[0]

ilon1['asm']=np.where(np.logical_and(lon>=125,lon<=145))[0]
ilat1['asm']=np.where(np.logical_and(lat>=-20,lat<=-10))[0]

ilon1['sasm']=np.where(np.logical_and(lon>=285,lon<=310))[0]
ilat1['sasm']=np.where(np.logical_and(lat>=-10,lat<=10))[0]

# ilat1['sasm']=np.where(np.logical_and(lat>=-15,lat<=-5))[0]
# ilat1['sasm']=np.where(np.logical_and(lat>=-20,lat<=10))[0]

ilon1['arge']=np.where(np.logical_and(lon>=285,lon<=310))[0]
ilat1['arge']=np.where(np.logical_and(lat>=-25,lat<=-15))[0]

print('Declaring functions')

def fin(x,ind):
    return x[ind]
    
def mask(y,x,func,thresh):
    y[func(x,thresh)]=np.nan
    
print('bin_2d')

bin1_2d=cape_bins
bin2_2d=subsat_bins

bin1_2d_width=cape_bin_width
bin2_2d_width=subsat_bin_width

bin1_2d_center=cape_bin_center
bin2_2d_center=subsat_bin_center

    
def bin_2d(x,y,z):

    ## This method is only valid if the bin width is invariant ##
    xind=np.int_((x-bin1_2d[0])/bin1_2d_width[0])
    yind=np.int_((y-bin2_2d[0])/bin2_2d_width[0])
    
    ### Make sure all the indices are within 0 to bin size ###

    ind=np.where((np.logical_and(xind>=0,xind<bin1_2d_width.size ))&
    (np.logical_and(yind>=0,yind<bin2_2d_width.size)))[0]

    z,xind,yind=map(fin,[z,xind,yind],[ind]*3)
    
    pcp_ret=np.zeros((bin1_2d_center.size,bin2_2d_center.size))
    cnts_ret=np.zeros((bin1_2d_center.size,bin2_2d_center.size))
    pcp_cnts_ret=np.zeros((bin1_2d_center.size,bin2_2d_center.size))

    if xind.size>0:
        bin_cython.bin_2d_precip_mod(z,xind,yind,pcp_ret,cnts_ret,pcp_cnts_ret,pthresh)

    return pcp_ret, cnts_ret, pcp_cnts_ret


print('bin_4d')

bin_1d=buoy_bins
bin_1d_width=buoy_bin_width
bin_1d_center=buoy_bin_center

def bin_var_1d(w,z):

    ### THIS FUNCTION WILL DO 1-D BINNING ###

    ## This method is only valid if the bin width is invariant ##
#     wind=np.int_((w-buoy_bins[0])/buoy_bin_width[0])
    wind=np.int_((w-bin_1d[0])/bin_1d_width[0])
    
    ### Make sure all the indices are within 0 to bin size ###
#     ind=np.where((np.logical_and(wind>=0,wind<buoy_bin_width.size)))[0]
    ind=np.where((np.logical_and(wind>=0,wind<bin_1d_width.size)))[0]

    ### Use map to extract only variables with valid indices ###
    
#     pcp_ret=np.zeros((buoy_bins.size-1))
#     cnts_ret=np.zeros((buoy_bins.size-1))
#     pcp_cnts_ret=np.zeros((buoy_bins.size-1))

    pcp_ret=np.zeros((bin_1d.size-1))
    cnts_ret=np.zeros((bin_1d.size-1))
    pcp_cnts_ret=np.zeros((bin_1d.size-1))


    if ind.size>1:
        w,z,wind=map(fin,[w,z,wind],[ind]*3)    
        bin_cython.bin_1d_precip_mod(w,z,wind,pcp_ret,cnts_ret,pcp_cnts_ret,pthresh)

    return pcp_ret, cnts_ret, pcp_cnts_ret

    
def bin_4d(u,w,x,y,z):

    print('Here')
    print('bin_4d, get bin indices')

    ## This method is only valid if the bin width is invariant ##
    
    uind=np.int_((u-tsub_bins[0])/tsub_bin_width[0])
    wind=np.int_((w-tsub_mt_bins[0])/tsub_mt_bin_width[0])
    xind=np.int_((x-ts_bins[0])/ts_bin_width[0])
    yind=np.int_((y-tbl_bins[0])/tbl_bin_width[0])
    
#     yind=np.int_((y-choke_bins[0])/choke_bin_width[0])
#     yind=np.int_((y-ts_lt_bins[0])/ts_lt_bin_width[0])
    
    ### Make sure all the indices are within 0 to bin size ###

    print('bin_4d, ensure bin indices are within legal bounds')

    ind=np.where((np.logical_and(uind>=0,uind<tsub_bin_width.size ))&
    (np.logical_and(wind>=0,wind<tsub_mt_bin_width.size))& 
    (np.logical_and(xind>=0,xind<ts_bin_width.size))&
    (np.logical_and(yind>=0,yind<tbl_bin_width.size)))[0]

    print('Assign finite indices')

    z,uind,wind,xind,yind=map(fin,[z,uind,wind,xind,yind],[ind]*5)

#     print 'Tsub:'
#     print np.histogram(u,bins=tsub_bins)
# 
#     print 'Tsub_mt:'
#     print np.histogram(w,bins=tsub_mt_bins)
# 
#     print 'Ts:'
#     print np.histogram(x,bins=ts_bins)
# 
#     print 'Tbl:'
#     print np.histogram(y,bins=tbl_bins)
#     
#     print np.where(np.logical_and(uind>=0,uind<tsub_bin_width.size))[0].size
#     print np.where(np.logical_and(wind>=0,wind<tsub_bl_bin_width.size))[0].size
#     print np.where(np.logical_and(xind>=0,xind<tsub_mt_bin_width.size))[0].size
#     print np.where(np.logical_and(xind>=0,yind<choke_bin_width.size))[0].size
#     print uind.size

    pcp_ret=np.zeros((tsub_bin_center.size,tsub_mt_bin_center.size,ts_bin_center.size,tbl_bin_center.size))
    cnts_ret=np.zeros((tsub_bin_center.size,tsub_mt_bin_center.size,ts_bin_center.size,tbl_bin_center.size))
    pcp_cnts_ret=np.zeros((tsub_bin_center.size,tsub_mt_bin_center.size,ts_bin_center.size,tbl_bin_center.size))


#     pcp_ret=np.zeros((tsub_lt_bin_center.size,ts_lt_bin_center.size,tsub_bl_bin_center.size,ts_bl_bin_center.size))
#     cnts_ret=np.zeros((tsub_lt_bin_center.size,ts_lt_bin_center.size,tsub_bl_bin_center.size,ts_bl_bin_center.size))
#     pcp_cnts_ret=np.zeros((tsub_lt_bin_center.size,ts_lt_bin_center.size,tsub_bl_bin_center.size,ts_bl_bin_center.size))
#     pcp_ret=np.zeros((ts_bin_center.size,ts_bin_center.size,tsub_bin_center.size,tbl_bin_center.size))
#     cnts_ret=np.zeros((ts_bin_center.size,ts_bin_center.size,tsub_bin_center.size,tbl_bin_center.size))
#     pcp_cnts_ret=np.zeros((ts_bin_center.size,ts_bin_center.size,tsub_bin_center.size,tbl_bin_center.size))

#     bin_cython.bin_4d_precip_mod(u,w,x,y,z,uind,wind,xind,yind,
#     pcp_ret,cnts_ret,pcp_cnts_ret,pthresh)
        
    if uind.size>0:
        bin_cython.bin_4d_precip_mod(z,uind,wind,xind,yind,pcp_ret,cnts_ret,pcp_cnts_ret,pthresh)


    return pcp_ret, cnts_ret, pcp_cnts_ret
    
def bin_4d_vertical(u,w,y,z,var1,var2,sat_vert,sub_sat_vert,cnts_ret,lev):


    # u,w,y,z are the "binants"
    # var1, var2 are the vertical variables

    ## This method is only valid if the bin width is invariant ##
    uind=np.int_((u-tsub_bins[0])/tsub_bin_width[0])
    wind=np.int_((w-tsub_mt_bins[0])/tsub_mt_bin_width[0])
    yind=np.int_((y-ts_bins[0])/ts_bin_width[0])
    zind=np.int_((z-tbl_bins[0])/tbl_bin_width[0])
        
    ### Make sure all the indices are within 0 to bin size ###

    ind=np.where((np.logical_and(uind>=0,uind<tsub_bin_width.size ))&
    (np.logical_and(wind>=0,wind<tsub_mt_bin_width.size))& 
    (np.logical_and(yind>=0,yind<ts_bin_width.size))&
    (np.logical_and(zind>=0,zind<tbl_bin_width.size)))[0]
    
#     print np.where(np.logical_and(uind>=0,uind<tsub_bin_width.size )) [0].size
#     print np.where(np.logical_and(wind>=0,wind<tsub_mt_bin_width.size))[0].size
#     print np.where(np.logical_and(yind>=0,yind<ts_bin_width.size))[0].size
#     print np.where(np.logical_and(zind>=0,zind<tbl_bin_width.size))[0].size
# 
#     uind=uind[ind]
#     wind=wind[ind]
#     yind=yind[ind]
#     zind=zind[ind]

    uind,wind,yind,zind=map(fin,[uind,wind,yind,zind],[ind]*4)
    
    var1=var1[:,ind]
    var2=var2[:,ind]
 
    ## So that CYTHON can recognize the format ###
    lev=np.int_(lev)

#     sat_vert=np.zeros((lev.size,tsub_bin_center.size,tsub_mt_bin_center.size,ts_bin_center.size,tbl_bin_center.size))
#     sub_sat_vert=np.zeros((lev.size,tsub_bin_center.size,tsub_mt_bin_center.size,ts_bin_center.size,tbl_bin_center.size))
#     cnts_ret=np.zeros((tsub_bin_center.size,tsub_mt_bin_center.size,ts_bin_center.size,tbl_bin_center.size))
    
#     pcp_cnts_ret=np.zeros((tsub_bin_center.size,tsub_mt_bin_center.size,ts_bin_center.size,tbl_bin_center.size))
#     pcp_ret=np.zeros((tsub_lt_bin_center.size,ts_lt_bin_center.size,tsub_bl_bin_center.size,ts_bl_bin_center.size))
#     cnts_ret=np.zeros((tsub_lt_bin_center.size,ts_lt_bin_center.size,tsub_bl_bin_center.size,ts_bl_bin_center.size))
#     pcp_cnts_ret=np.zeros((tsub_lt_bin_center.size,ts_lt_bin_center.size,tsub_bl_bin_center.size,ts_bl_bin_center.size))
#     pcp_ret=np.zeros((ts_bin_center.size,ts_bin_center.size,tsub_bin_center.size,tbl_bin_center.size))
#     cnts_ret=np.zeros((ts_bin_center.size,ts_bin_center.size,tsub_bin_center.size,tbl_bin_center.size))
#     pcp_cnts_ret=np.zeros((ts_bin_center.size,ts_bin_center.size,tsub_bin_center.size,tbl_bin_center.size))

#     bin_cython.bin_4d_precip_mod(u,w,x,y,z,uind,wind,xind,yind,
#     pcp_ret,cnts_ret,pcp_cnts_ret,pthresh)
        
    if uind.size>0:
        bin_cython.bin_4d_vert(uind,wind,yind,zind,var1,var2,sat_vert,sub_sat_vert,cnts_ret,lev)

#     return sat_vert, sub_sat_vert, cnts_ret
    
def bin_5d(u,v,w,x,y,z):

    ### BINNING BY LTSS, MTSS, BLSS, DEEP TEMP, BL TEMP

    ## This method is only valid if the bin width is invariant ##
    uind=np.int_((u-tsub_bins[0])/tsub_bin_width[0])
    vind=np.int_((v-tsub_mt_bins[0])/tsub_mt_bin_width[0])
    wind=np.int_((w-tbl_bins[0])/tbl_bin_width[0])
    xind=np.int_((x-ts_bins[0])/ts_bin_width[0])
    yind=np.int_((y-choke_bins[0])/choke_bin_width[0])

#     wind=np.int_((w-tsub_bl_bins[0])/tsub_bl_bin_width[0])
#     xind=np.int_((x-choke_bins[0])/choke_bin_width[0])

    
    ### Make sure all the indices are within 0 to bin size ###
    
    ind=np.where((np.logical_and(uind>=0,uind<tsub_bin_width.size ))&
    (np.logical_and(vind>=0,vind<tsub_mt_bin_width.size))&
    (np.logical_and(wind>=0,wind<tbl_bin_width.size))&
    (np.logical_and(xind>=0,xind<ts_bin_width.size))& 
    (np.logical_and(yind>=0,yind<choke_bin_width.size)))[0]

#     print np.where(np.logical_and(uind>=0,uind<tsub_bin_width.size))[0].size
#     print np.where(np.logical_and(vind>=0,vind<tsub_mt_bin_width.size))[0].size
#     print np.where(np.logical_and(wind>=0,wind<tsub_bl_bin_width.size))[0].size
#     print np.where(np.logical_and(xind>=0,xind<choke_bin_width.size))[0].size
#     print np.where(np.logical_and(yind>=0,yind<ts_bl_bin_width.size))[0].size
# 
#     print ind.size
#     exit()
#     ind=np.where((np.logical_and(uind>=0,uind<tsub_bin_width.size ))&
#     (np.logical_and(vind>=0,vind<tsub_mt_bin_width.size))&
#     (np.logical_and(wind>=0,wind<ts_bin_width.size))&
#     (np.logical_and(xind>=0,xind<buoy_bin_width.size))& 
#     (np.logical_and(yind>=0,yind<tbl_bin_width.size)))[0]

    z,uind,vind,wind,xind,yind=map(fin,[z,uind,vind,wind,xind,yind],[ind]*6)

#     pcp_ret=np.zeros((tsub_bin_width.size,tsub_mt_bin_width.size,ts_bin_width.size,buoy_bin_width.size,tbl_bin_width.size))
#     cnts_ret=np.zeros((tsub_bin_width.size,tsub_mt_bin_width.size,ts_bin_width.size,buoy_bin_width.size,tbl_bin_width.size))
#     pcp_cnts_ret=np.zeros((tsub_bin_width.size,tsub_mt_bin_width.size,ts_bin_width.size,buoy_bin_width.size,tbl_bin_width.size))

    pcp_ret=np.zeros((tsub_bin_width.size,tsub_mt_bin_width.size,tbl_bin_width.size,ts_bin_width.size,choke_bin_width.size))
    cnts_ret=np.zeros((tsub_bin_width.size,tsub_mt_bin_width.size,tbl_bin_width.size,ts_bin_width.size,choke_bin_width.size))
    pcp_cnts_ret=np.zeros((tsub_bin_width.size,tsub_mt_bin_width.size,tbl_bin_width.size,ts_bin_width.size,choke_bin_width.size))

#     print pcp_ret.shape

    if uind.size>0:

        bin_cython.bin_5d_precip(z,uind,vind,wind,xind,yind,pcp_ret,cnts_ret,pcp_cnts_ret,pthresh)

    return pcp_ret, cnts_ret, pcp_cnts_ret

def bin_new_var(w,x,y,z):

    ## This method is only valid if the bin width is invariant ##

#     xind=np.int_((x-ts_bins[0])/ts_bin_width[0])
    # wind=np.int_((w-tsub_bins[0])/tsub_bin_width[0])
#     xind=np.int_((x-tsub_mt_bins[0])/tsub_mt_bin_width[0])
#     
    wind=np.int_((w-ts_lt_bins[0])/ts_lt_bin_width[0])
    xind=np.int_((x-ts_mt_bins[0])/ts_mt_bin_width[0])
    yind=np.int_((y-tbl_bins[0])/tbl_bin_width[0])
    
#     print tsub_bins[0],tsub_bins[-1],np.nanmax(w),np.nanmin(w)
#     print ts_bins[0],ts_bins[-1],np.nanmax(x),np.nanmin(x)
#     print tbl_bins[0],tbl_bins[-1],np.nanmax(y),np.nanmin(y)
#     exit()

    ### Make sure all the indices are within 0 to bin size ###
  
    ind=np.where((np.logical_and(wind>=0,wind<tbl_bin_width.size)) & 
    (np.logical_and(xind>=0,xind<tbl_bin_width.size)) &
    (np.logical_and(yind>=0,yind<tbl_bin_width.size)))
    
    # print np.where(np.logical_and(wind>=0,wind<tsub_bin_width.size))[0].size
#     print np.where(np.logical_and(xind>=0,xind<tsub_mt_bin_width.size))[0].size
#     print np.where(np.logical_and(yind>=0,yind<tbl_bin_width.size))[0].size
#     
    ### Use map to extract only variables with valid indices ###
    z,wind,xind,yind=map(fin,[z,wind,xind,yind],[ind]*4)

    pcp_ret=np.zeros((ts_lt_bin_center.size,ts_mt_bin_center.size,tbl_bin_center.size))
    cnts_ret=np.zeros((ts_lt_bin_center.size,ts_mt_bin_center.size,tbl_bin_center.size))
    pcp_cnts_ret=np.zeros((ts_lt_bin_center.size,ts_mt_bin_center.size,tbl_bin_center.size))

    if wind.size>0:

        bin_cython.bin_3d_precip_mod(z,wind,xind,yind,pcp_ret,cnts_ret,pcp_cnts_ret,pthresh)

    return pcp_ret, cnts_ret, pcp_cnts_ret
    
def bin_var_1d_diurnal(w,z,lt):

    ### THIS FUNCTION WILL DO 1-D BINNING ###
    ### WITH THE ADDED FUNCTIONALITY OF THE DIURNAL CYCLE ###
    
    pcp_ret=np.zeros((4,buoy_bin_center.size))
    cnts_ret=np.zeros((4,buoy_bin_center.size))
    pcp_cnts_ret=np.zeros((4,buoy_bin_center.size))


#     print lt[np.isfinite(lt)].size
    for k in np.arange(4):
        
        if k==0:
           indt=np.where(np.logical_and(lt>=6,lt<=11))[0]
#             indt=np.where(np.logical_or( np.logical_and(lt>=0,lt<=11),np.logical_or(lt>=19,lt<11)))[0]
#             indt=np.where(np.logical_or(lt>=19,lt<11))[0]

#             print k, indt.size

        elif k==1:
           indt=np.where(np.logical_and(lt>=12,lt<=17))[0]
#             indt=np.where(np.logical_and(lt>=11,lt<=18))[0]
#             print k, indt.size

        elif k==2:
           indt=np.where(np.logical_and(lt>=18,lt<=23))[0]
        elif k==3:
           indt=np.where(np.logical_and(lt>=0,lt<=5))[0]

#         continue
#                    
        wt,zt=w[indt],z[indt]
# 
        ## This method is only valid if the bin width is invariant ##
        wind=np.int_((wt-buoy_bins[0])/buoy_bin_width[0])
        ### Make sure all the indices are within 0 to bin size ###
  
        ind=np.where((np.logical_and(wind>=0,wind<buoy_bin_width.size)))

        ### Use map to extract only variables with valid indices ###
        wt,zt,wind=map(fin,[wt,zt,wind],[ind]*3)
    

        if wt.size>0:

            bin_cython.bin_1d_precip_mod(wt,zt,wind,pcp_ret[k,...],cnts_ret[k,...],pcp_cnts_ret[k,...],pthresh)


    return pcp_ret, cnts_ret, pcp_cnts_ret
    
    
    
    
