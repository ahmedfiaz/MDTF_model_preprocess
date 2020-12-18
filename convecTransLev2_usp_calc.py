# This file is part of the convective_transition_diag module of the MDTF code package (see mdtf/MDTF_v2.0/LICENSE.txt)

# ======================================================================
# convecTransBasic_usp_calc.py
#
#   Called by convecTransBasic.py
#    Provides User-Specified Parameters for Binning and Plotting
#
#   This file is part of the Convective Transition Diagnostic Package 
#    and the MDTF code package. See LICENSE.txt for the license.
#
import json
import os
import glob

# ======================================================================
# START USER SPECIFIED SECTION
# ======================================================================
# Model name

# MODEL_NAME='KACE'  #data is in height co-ordinates

# MODEL_NAME='NESM'
# MODEL_NAME='MRI'
# MODEL_NAME='MPI-ESM1'
# MODEL_NAME='MIROC-E2SL'
# MODEL_NAME='MIROC6'
# MODEL_NAME='IPSL'
# MODEL_NAME='GFDL-CM4'
# MODEL_NAME='F-GOALS' 
# MODEL_NAME='CNRM-CM6-1-HR' ## issue with a(p) and b(p): these parameters seem correct for first file, but wrong afterward
# MODEL_NAME='CNRM-CM6-1'
# MODEL_NAME='ACCESS-ESM1' ### issue with a(p) and b(p), data is possibly in height co-ordinates
# MODEL_NAME='CESM' ## issue with the date which begins from 0001-01-01
# MODEL_NAME='NASA-GISS'
# MODEL_NAME='SNU.SAM0-UNICON'
# MODEL_NAME='MPI-ESM1'
# MODEL_NAME='BCC_3hr'
MODEL_NAME='UNICON'

# START_DATE='2013010106' ## TIME FORMAT: YYYYMMDDHH
# END_DATE='2014123118' 

START_DATE='1985010100' ## TIME FORMAT: YYYYMMDDHH
END_DATE='2004123100' 
DATE_FORMAT='%Y%m%d' ### If time is specified as a float or integer

# START_DATE='2096010103' ## TIME FORMAT: YYYYMMDDHH
# END_DATE='2097123121' 

## The latitudinal bounds are controlled by the region mask (which is bounded by 20 N- 20 S)
# LAT_BNDS=[-20,20]  ## Set latitudinal bounds for analysis


# MODEL=os.environ["CASENAME"]#os.environ["model"] # will show up in the figure
# Model output directory

# MODEL_OUTPUT_DIR='/scratch/neelin/CMIP6/SSP585/'+MODEL_NAME+'/' # where original model data are located
# FORCING='SSP585'

# MODEL_OUTPUT_DIR='/scratch/neelin/CMIP6/'+MODEL_NAME+'/' # where original model data are located
# FORCING='HIST'

MODEL_OUTPUT_DIR='/neelin2020/UNICON/unicon_stand_2deg_A159d_exp11/' # where original model data are located
FORCING='Omega=0.4'


# MODEL_OUTPUT_DIR=os.environ["MODEL_OUTPUT_DIR"] # where original model data are located
# Variable Names
# PR_VAR="pr"
# PRC_VAR="prc"
# TA_VAR="ta"
# HUS_VAR="hus"
# LEV_VAR="lev"
# PS_VAR="ps"
# A_VAR="a"
# B_VAR="b"

## For UNICON ###
PR_VAR="PRCP"
PRC_VAR="prc"
TA_VAR="T"
HUS_VAR="Q"
LEV_VAR="lev"
PS_VAR="ps"
A_VAR="a"
B_VAR="b"

VERT_TYPE='pres' # sigma or pres

## for F-GOALS ##
# A_VAR='ptop'

## for MPI-ESM, CNRM-CM6, IPSL, MPI-ESM1, GFDL-CM4
# A_VAR="ap"

## for IPSL
# LEV_VAR="presnivs" 

TIME_VAR="time"
LAT_VAR="lat"
LON_VAR="lon"



# PRES_VAR=os.environ["lev_coord"]
# TIME_VAR=os.environ["time_coord"]
# LAT_VAR=os.environ["lat_coord"]
# LON_VAR=os.environ["lon_coord"]

# ======================================================================
# Region mask directory & filename

REGION_MASK_DIR="/home/fiaz/MDTF/files/"
REGION_MASK_FILENAME="region_0.25x0.25_costal2.5degExcluded.mat"

# REGION_MASK_DIR=os.environ["OBS_DATA"]
# REGION_MASK_FILENAME="region_0.25x0.25_costal2.5degExcluded.mat"
# Number of regions
#  Use grids with 1<=region<=NUMBER_OF_REGIONS in the mask
NUMBER_OF_REGIONS=4 # default: 4
# Region names
REGION_STR=["WPac","EPac","Atl","Ind"]
time_idx_delta=1000
# ======================================================================
# Directory for saving pre-processed temperature fields
#  tave [K]: Mass-weighted column average temperature
#  qsat_int [mm]: Column-integrated saturation specific humidity
# USER MUST HAVE WRITE PERMISSION
#  If one changes PREPROCESSING_OUTPUT_DIR, one must also modify data["tave_list"]
#  & data["qsat_int_list"] below by replacing MODEL_OUTPUT_DIR with
#  PREPROCESSING_OUTPUT_DIR

# PREPROCESSING_OUTPUT_DIR="/scratch/neelin/layer_thetae/CMIP6/"+MODEL_NAME+"/" 
PREPROCESSING_OUTPUT_DIR="/neelin2020/UNICON/unicon_stand_2deg_A159d_exp11/layer_thetae/"

# PREPROCESSING_OUTPUT_DIR="/scratch/neelin/layer_thetae/CMIP6-SSP585/"+MODEL_NAME+"/" 

THETAE_OUT="data_layer_thetae_var"

LFT_THETAE_VAR="thetae_lt"
LFT_THETAE_SAT_VAR="thetae_sat_lt"
BL_THETAE_VAR="thetae_bl"


# ======================================================================
# Directory & Filename for saving binned results (netCDF4)
#  tave or qsat_int will be appended to BIN_OUTPUT_FILENAME
BIN_OUTPUT_DIR="/home/fiaz/MDTF/files/HIST/"
# BIN_OUTPUT_FILENAME=MODEL_NAME+"_"+FORCING+".prc.convecTransLev2"
BIN_OUTPUT_FILENAME=MODEL_NAME+"_"+FORCING+".convecTransLev2"

# BIN_OUTPUT_DIR=os.environ["WK_DIR"]+"/model/netCDF"
# BIN_OUTPUT_FILENAME=os.environ["CASENAME"]+".convecTransBasic"

# ======================================================================
# Re-do binning even if binned data file detected (default: True)
BIN_ANYWAY=True

# Override the pre-processing step ()
SKIP_PREPROCESS=True

# ======================================================================
# Column Water Vapor (CWV in mm) range & bin-width
#  CWV bin centers are integral multiples of cwv_bin_width

BINT_BIN_WIDTH=0.01 # default=0.3 (following satellite retrieval product)
BINT_RANGE_MAX=1.51 # default=90 (75 for satellite retrieval product)
BINT_RANGE_MIN=-1.5 # default=90 (75 for satellite retrieval product)

# Bin width and intervals for CAPE and SUBSAT.
# In units of K

CAPE_RANGE_MIN=-40.0
CAPE_RANGE_MAX=20.0

CAPE_BIN_WIDTH=1.0

SUBSAT_RANGE_MIN=-1.0
SUBSAT_RANGE_MAX=42.0

# CAPE_RANGE_MIN=-100.0
# CAPE_RANGE_MAX=-20.0

# SUBSAT_RANGE_MIN=25.0
# SUBSAT_RANGE_MAX=100.0

SUBSAT_BIN_WIDTH=1.0

# Column-integrated Saturation Specific Humidity qsat_int [mm] range & bin-width
#  with bin centers = Q_RANGE_MIN + integer*Q_BIN_WIDTH
# Satellite retrieval suggests T_BIN_WIDTH=1 
#  is approximately equivalent to Q_BIN_WIDTH=4.8
# Q_RANGE_MIN=16.0
# Q_RANGE_MAX=106.0
# Q_BIN_WIDTH=4.5

# Define column [hPa] (default: 1000-200 hPa)
#  One can re-define column by changing p_lev_bottom & p_lev_top,
#  but one must also delete/re-name existing tave & qsat_int files
#  since the default tave & qsat_int filenames do not contain conlumn info
p_lev_mid=500
# p_lev_top=200
# If model pressure levels are close to p_lev_bottom and/or p_lev_top
#  (within dp-hPa neighborhood), use model level(s) to define column instead
# dp=1.0

# Threshold value defining precipitating events [mm/hr]
PRECIP_THRESHOLD=1.0#0.25
# PRECIP_FACTOR=1e3 ## Factor to convert precip. units to mm/hr
# PRECIP_FACTOR=36e2 ## Factor to convert precip. units to mm/hr
PRECIP_FACTOR=1./24 ## Factor to convert precip. units to mm/hr

# ======================================================================
# END USER SPECIFIED SECTION
# ======================================================================
#
# ======================================================================
# DO NOT MODIFY CODE BELOW UNLESS
# YOU KNOW WHAT YOU ARE DOING
# ======================================================================
data={}

data["MODEL"]=MODEL_NAME
data["START_DATE"]=START_DATE
data["END_DATE"]=END_DATE
data["MODEL_OUTPUT_DIR"]=MODEL_OUTPUT_DIR
data["PREPROCESSING_OUTPUT_DIR"]=PREPROCESSING_OUTPUT_DIR

data["REGION_MASK_DIR"]=REGION_MASK_DIR
data["REGION_MASK_FILENAME"]=REGION_MASK_FILENAME

data["NUMBER_OF_REGIONS"]=NUMBER_OF_REGIONS
data["REGION_STR"]=REGION_STR

data["TIME_VAR"]=TIME_VAR
data["LAT_VAR"]=LAT_VAR
data["LON_VAR"]=LON_VAR
data["LEV_VAR"]=LEV_VAR
data["DATE_FORMAT"]=DATE_FORMAT

data["LFT_THETAE_VAR"]=LFT_THETAE_VAR
data["LFT_THETAE_SAT_VAR"]=LFT_THETAE_SAT_VAR
data["BL_THETAE"]=BL_THETAE_VAR

# data["PR_VAR"]=PR_VAR
# data["PRC_VAR"]=PRC_VAR
# data["TA_VAR"]=TA_VAR
# data["HUS_VAR"]=HUS_VAR
# data["PRES_VAR"]=PRES_VAR

data["PS_VAR"]=PS_VAR
data["A_VAR"]=A_VAR
data["B_VAR"]=B_VAR

data["VERT_TYPE"]=VERT_TYPE


# data["TAVE_VAR"]=TAVE_VAR
# data["QSAT_INT_VAR"]=QSAT_INT_VAR
# data["PRES_VAR"]=PRES_VAR
data["time_idx_delta"]=time_idx_delta
# data["BULK_TROPOSPHERIC_TEMPERATURE_MEASURE"]=BULK_TROPOSPHERIC_TEMPERATURE_MEASURE

data["BIN_OUTPUT_DIR"]=BIN_OUTPUT_DIR
data["BIN_OUTPUT_FILENAME"]=BIN_OUTPUT_FILENAME

# if BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==1:
#     data["BIN_OUTPUT_FILENAME"]+="_"+TAVE_VAR
#     data["TEMP_VAR"]=TAVE_VAR
# elif BULK_TROPOSPHERIC_TEMPERATURE_MEASURE==2:
#     data["BIN_OUTPUT_FILENAME"]+="_"+QSAT_INT_VAR
#     data["TEMP_VAR"]=QSAT_INT_VAR
# 
data["BIN_ANYWAY"]=BIN_ANYWAY
    
data["BINT_BIN_WIDTH"]=BINT_BIN_WIDTH 
data["BINT_RANGE_MAX"]=BINT_RANGE_MAX
data["BINT_RANGE_MIN"]=BINT_RANGE_MIN

data["CAPE_BIN_WIDTH"]=CAPE_BIN_WIDTH 
data["CAPE_RANGE_MAX"]=BINT_RANGE_MAX
data["CAPE_RANGE_MIN"]=BINT_RANGE_MIN

data["SUBSAT_BIN_WIDTH"]=SUBSAT_BIN_WIDTH 
data["SUBSAT_RANGE_MAX"]=BINT_RANGE_MAX
data["SUBSAT_RANGE_MIN"]=BINT_RANGE_MIN


data["PRECIP_THRESHOLD"]=PRECIP_THRESHOLD
data["PRECIP_FACTOR"]=PRECIP_FACTOR

data["p_lev_mid"]=p_lev_mid

# List binned data file (with filename corresponding to casename)
data["bin_output_list"]=sorted(glob.glob(data["BIN_OUTPUT_DIR"]+"/"+data["BIN_OUTPUT_FILENAME"]+".nc"))

# List available netCDF files
# Assumes that the corresponding files in each list
#  have the same spatial/temporal coverage/resolution

pr_list=sorted(glob.glob(MODEL_OUTPUT_DIR+PR_VAR+"/*"))
prc_list=sorted(glob.glob(MODEL_OUTPUT_DIR+PRC_VAR+"/*"))
ta_list=sorted(glob.glob(MODEL_OUTPUT_DIR+TA_VAR+"/*"))
hus_list=sorted(glob.glob(MODEL_OUTPUT_DIR+HUS_VAR+"/*"))

data["pr_list"] = pr_list
data["prc_list"] = prc_list
data["ta_list"] = ta_list
data["hus_list"] = hus_list
data["THETAE_OUT"] = THETAE_OUT
data["LFT_THETAE_VAR"]= LFT_THETAE_VAR
data["LFT_THETAE_SAT_VAR"]= LFT_THETAE_SAT_VAR
data["BL_THETAE"]= BL_THETAE_VAR

# Check for pre-processed tave & qsat_int data
# print(PREPROCESSING_OUTPUT_DIR+THETAE_OUT)
thetae_list=sorted(glob.glob(PREPROCESSING_OUTPUT_DIR+THETAE_OUT+'*'))
pr_save_list=sorted(glob.glob(PREPROCESSING_OUTPUT_DIR+PR_VAR+"*"))
prc_save_list=sorted(glob.glob(PREPROCESSING_OUTPUT_DIR+PRC_VAR+"*"))

# lft_thetae_sat_list=sorted(glob.glob(MODEL_OUTPUT_DIR+LFT_THETAE_SAT_VAR))
# bl_thetae_list=sorted(glob.glob(MODEL_OUTPUT_DIR+BL_THETAE_VAR))

data["thetae_list"]=thetae_list
data["pr_save_list"]=pr_save_list

if len(data["ta_list"])==0:
        exit('     No input files found...')

if (len(data["thetae_list"])<len(data["ta_list"])): 
    data["PREPROCESS_THETAE"]=1
    data["SAVE_THETAE"]=1
else:
    data["PREPROCESS_THETAE"]=0
    data["SAVE_THETAE"]=0

if SKIP_PREPROCESS:
    data["PREPROCESS_THETAE"]=0
    data["SAVE_THETAE"]=0


    
# Taking care of function arguments for binning
data["args1"]=[ \
BINT_BIN_WIDTH, \
BINT_RANGE_MAX, \
BINT_RANGE_MIN, \
CAPE_RANGE_MIN, \
CAPE_RANGE_MAX, \
CAPE_BIN_WIDTH, \
SUBSAT_RANGE_MIN, \
SUBSAT_RANGE_MAX, \
SUBSAT_BIN_WIDTH, \
NUMBER_OF_REGIONS, \
START_DATE,\
END_DATE,\
DATE_FORMAT,\
pr_list, \
PR_VAR, \
prc_list, \
PRC_VAR, \
MODEL_OUTPUT_DIR, \
THETAE_OUT,\
data["thetae_list"], \
LFT_THETAE_VAR, \
LFT_THETAE_SAT_VAR, \
BL_THETAE_VAR, \
ta_list, \
TA_VAR, \
hus_list, \
HUS_VAR, \
LEV_VAR, \
PS_VAR, \
A_VAR,\
B_VAR,\
VERT_TYPE,\
MODEL_NAME, \
p_lev_mid, \
time_idx_delta, \
data["SAVE_THETAE"], \
PREPROCESSING_OUTPUT_DIR, \
PRECIP_THRESHOLD, \
PRECIP_FACTOR,\
data["BIN_OUTPUT_DIR"], \
data["BIN_OUTPUT_FILENAME"], \
TIME_VAR, \
LAT_VAR, \
LON_VAR ]

data["args2"]=[ \
data["bin_output_list"]]

data["args3"]=[ \
BINT_BIN_WIDTH, \
BINT_RANGE_MAX, \
BINT_RANGE_MIN, \
CAPE_RANGE_MIN, \
CAPE_RANGE_MAX, \
CAPE_BIN_WIDTH, \
SUBSAT_RANGE_MIN, \
SUBSAT_RANGE_MAX, \
SUBSAT_BIN_WIDTH, \
NUMBER_OF_REGIONS, \
START_DATE,\
END_DATE,\
pr_list, \
PR_VAR, \
prc_list, \
PRC_VAR, \
MODEL_OUTPUT_DIR, \
THETAE_OUT,\
data["thetae_list"], \
LFT_THETAE_VAR, \
# data["lft_thetae_sat_list"], \
LFT_THETAE_SAT_VAR, \
# data["bl_thetae_list"], \
BL_THETAE_VAR, \
ta_list, \
TA_VAR, \
hus_list, \
HUS_VAR, \
LEV_VAR, \
PS_VAR, \
A_VAR,\
B_VAR,\
MODEL_NAME, \
p_lev_mid, \
# dp, \
time_idx_delta, \
data["SAVE_THETAE"], \
PREPROCESSING_OUTPUT_DIR, \
PRECIP_THRESHOLD, \
PRECIP_FACTOR,\
data["BIN_OUTPUT_DIR"], \
data["BIN_OUTPUT_FILENAME"], \
TIME_VAR, \
LAT_VAR, \
LON_VAR ]

print(len(data['args1']))

with open(os.getcwd()+'/'+'convecTransLev2_calc_parameters.json', "w") as outfile:
    json.dump(data, outfile)
