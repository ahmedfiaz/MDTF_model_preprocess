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
MODEL='NASA-GISS'
# MODEL=os.environ["CASENAME"]#os.environ["model"] # will show up in the figure
# Model output directory
MODEL_OUTPUT_DIR='/scratch/neelin/CMIP6/wget_scripts/NASA-GISS/' # where original model data are located
# MODEL_OUTPUT_DIR=os.environ["MODEL_OUTPUT_DIR"] # where original model data are located
# Variable Names
PR_VAR="pr"
PRC_VAR="prc"
TA_VAR="ta"
HUS_VAR="hus"
LEV_VAR="lev"
PS_VAR="ps"
A_VAR="a"
B_VAR="b"
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

# ======================================================================
# Directory for saving pre-processed temperature fields
#  tave [K]: Mass-weighted column average temperature
#  qsat_int [mm]: Column-integrated saturation specific humidity
# USER MUST HAVE WRITE PERMISSION
#  If one changes PREPROCESSING_OUTPUT_DIR, one must also modify data["tave_list"]
#  & data["qsat_int_list"] below by replacing MODEL_OUTPUT_DIR with
#  PREPROCESSING_OUTPUT_DIR

PREPROCESSING_OUTPUT_DIR="/scratch/neelin/layer_thetae/CMIP6/NASA-GISS/" 
LFT_THETAE_VAR="thetae_lt"
LFT_THETAE_SAT_VAR="thetae_sat_lt"
BL_THETAE_VAR="thetae_bl"

# BL_INT_VAR=os.environ["qsat_int_var"]

# PREPROCESSING_OUTPUT_DIR=os.environ["DATADIR"] 
# TAVE_VAR=os.environ["tave_var"]
# QSAT_INT_VAR=os.environ["qsat_int_var"]
# Number of time-steps in Temperature-preprocessing
#  Default: 1000 (use smaller numbers for limited memory)
# time_idx_delta=1000
# Use 1:tave, or 2:qsat_int as Bulk Tropospheric Temperature Measure 
# BULK_TROPOSPHERIC_TEMPERATURE_MEASURE=int(os.environ["BULK_TROPOSPHERIC_TEMPERATURE_MEASURE"])

# ======================================================================
# Directory & Filename for saving binned results (netCDF4)
#  tave or qsat_int will be appended to BIN_OUTPUT_FILENAME
BIN_OUTPUT_DIR="/home/fiaz/MDTF/"
BIN_OUTPUT_FILENAME=MODEL+".convecTransLev2"

# BIN_OUTPUT_DIR=os.environ["WK_DIR"]+"/model/netCDF"
# BIN_OUTPUT_FILENAME=os.environ["CASENAME"]+".convecTransBasic"

# ======================================================================
# Re-do binning even if binned data file detected (default: True)
BIN_ANYWAY=True

# ======================================================================
# Column Water Vapor (CWV in mm) range & bin-width
#  CWV bin centers are integral multiples of cwv_bin_width

# buoy_bins=np.arange(-1.5,1.51,.01)
# # buoy_bins=cwv_bins ### Use this if trying to bin by CWV
# buoy_bin_center=(buoy_bins[1:]+buoy_bins[:-1])*0.5
# buoy_bin_width=np.diff(buoy_bins)
# 
# cape_bins=np.arange(-40,17.0,1.0)
# cape_bin_center=(cape_bins[1:]+cape_bins[:-1])*0.5
# cape_bin_width=np.diff(cape_bins)
# 
# subsat_bins=np.arange(0,42.,1.0)
# subsat_bin_center=(subsat_bins[1:]+subsat_bins[:-1])*0.5
# subsat_bin_width=np.diff(subsat_bins)
# 
# Bin width and intervals for Bint in units of 0.01 m/s^2

BINT_BIN_WIDTH=0.01 # default=0.3 (following satellite retrieval product)
BINT_RANGE_MAX=1.51 # default=90 (75 for satellite retrieval product)
BINT_RANGE_MIN=-1.5 # default=90 (75 for satellite retrieval product)

# Bin width and intervals for CAPE and SUBSAT.
# In units of K

CAPE_RANGE_MIN=-40.0
CAPE_RANGE_MAX=17.0
CAPE_BIN_WIDTH=1.0

SUBSAT_RANGE_MIN=0.0
SUBSAT_RANGE_MAX=42.0
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
PRECIP_THRESHOLD=0.25

# ======================================================================
# END USER SPECIFIED SECTION
# ======================================================================
#
# ======================================================================
# DO NOT MODIFY CODE BELOW UNLESS
# YOU KNOW WHAT YOU ARE DOING
# ======================================================================
data={}

data["MODEL"]=MODEL
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

data["LFT_THETAE_VAR"]=LFT_THETAE_VAR
data["LFT_THETAE_SAT_VAR"]=LFT_THETAE_SAT_VAR
data["BL_THETAE"]=BL_THETAE_VAR

data["PR_VAR"]=PR_VAR
data["PRC_VAR"]=PRC_VAR
data["TA_VAR"]=TA_VAR
data["HUS_VAR"]=HUS_VAR

data["PS_VAR"]=PS_VAR
data["A_VAR"]=A_VAR
data["B_VAR"]=B_VAR

# data["TAVE_VAR"]=TAVE_VAR
# data["QSAT_INT_VAR"]=QSAT_INT_VAR
# data["PRES_VAR"]=PRES_VAR
# data["time_idx_delta"]=time_idx_delta
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

# 
# data["T_RANGE_MIN"]=T_RANGE_MIN
# data["T_RANGE_MAX"]=T_RANGE_MAX
# data["T_BIN_WIDTH"]=T_BIN_WIDTH
# 
# data["Q_RANGE_MIN"]=Q_RANGE_MIN
# data["Q_RANGE_MAX"]=Q_RANGE_MAX
# data["Q_BIN_WIDTH"]=Q_BIN_WIDTH
# 
# data["p_lev_bottom"]=p_lev_bottom
# data["p_lev_top"]=p_lev_top
# data["dp"]=dp

data["PRECIP_THRESHOLD"]=PRECIP_THRESHOLD
data["p_lev_mid"]=p_lev_mid

# List binned data file (with filename corresponding to casename)
data["bin_output_list"]=sorted(glob.glob(data["BIN_OUTPUT_DIR"]+"/"+data["BIN_OUTPUT_FILENAME"]+".nc"))

# List available netCDF files
# Assumes that the corresponding files in each list
#  have the same spatial/temporal coverage/resolution

pr_list=sorted(glob.glob(MODEL_OUTPUT_DIR+"/"+PR_VAR))
prc_list=sorted(glob.glob(MODEL_OUTPUT_DIR+"/"+PRC_VAR))
lft_thetae_list=sorted(glob.glob(MODEL_OUTPUT_DIR+"/"+LFT_THETAE_VAR))
lft_thetae_sat_list=sorted(glob.glob(MODEL_OUTPUT_DIR+"/"+LFT_THETAE_SAT_VAR))
bl_thetae_list=sorted(glob.glob(MODEL_OUTPUT_DIR+"/"+BL_THETAE_VAR))

# data["LFT_THETAE_VAR"]=LFT_THETAE_VAR
# data["LFT_THETAE_SAT_VAR"]=LFT_THETAE_SAT_VAR
# data["BL_THETAE"]=BL_THETAE
# prw_list=sorted(glob.glob(MODEL_OUTPUT_DIR+"/"+os.environ["prw_file"]))
# ta_list=sorted(glob.glob(MODEL_OUTPUT_DIR+"/"+os.environ["ta_file"]))
# data["pr_list"] = pr_list
# data["prw_list"] = prw_list
# data["ta_list"] = ta_list

# Check for pre-processed tave & qsat_int data
data["lft_thetae_list"]=lft_thetae_list#sorted(glob.glob(MODEL_OUTPUT_DIR+"/"+lft_thetae_list))
data["lft_thetae_sat_list"]=lft_thetae_sat_list#sorted(glob.glob(MODEL_OUTPUT_DIR+"/"+lft_thetae_sat_list))
data["bl_thetae_list"]=bl_thetae_list#sorted(glob.glob(MODEL_OUTPUT_DIR+"/"+bl_thetae_list))

if 0 in (len(data["lft_thetae_list"]), len(data["lft_thetae_sat_list"]), len(data["bl_thetae_list"])):
    data["PREPROCESS_THETAE"]=1
else:
    data["PREPROCESS_THETAE"]=0

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
pr_list, \
PR_VAR, \
prc_list, \
PRC_VAR, \
data["PREPROCESS_THETAE"], \
MODEL_OUTPUT_DIR, \
data["lft_thetae_list"], \
LFT_THETAE_VAR, \
data["lft_thetae_sat_list"], \
LFT_THETAE_SAT_VAR, \
data["bl_thetae_list"], \
BL_THETAE_VAR, \
LEV_VAR, \
MODEL, \
# p_lev_mid, \
# dp, \
# time_idx_delta, \
# data["SAVE_TAVE_QSAT_INT"], \
PREPROCESSING_OUTPUT_DIR, \
PRECIP_THRESHOLD, \
data["BIN_OUTPUT_DIR"], \
data["BIN_OUTPUT_FILENAME"], \
TIME_VAR, \
LAT_VAR, \
LON_VAR ]

# data["args2"]=[ \
# data["bin_output_list"],\
# TAVE_VAR,\
# QSAT_INT_VAR,\
# BULK_TROPOSPHERIC_TEMPERATURE_MEASURE ]
with open(os.getcwd()+'/'+'convecTransLev2_calc_parameters.json', "w") as outfile:
    json.dump(data, outfile)
