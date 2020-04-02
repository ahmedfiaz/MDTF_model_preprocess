# This file is part of the convective_transition_diag module of the MDTF code package (see mdtf/MDTF_v2.0/LICENSE.txt)

# ======================================================================
# convecTransLev2_usp_plot.py
#
#   Called by convecTransLev2.py
#    Provides User-Specified Parameters for Plotting
#
#   This file is part of the Convective Transition Diagnostic Package 
#    and the MDTF code package. See LICENSE.txt for the license.
#
import json
import os
import glob

with open(os.environ["WK_DIR"]+"/"+"convecTransLev2_calc_parameters.json") as outfile:
    bin_data=json.load(outfile)
    
# ======================================================================
# START USER SPECIFIED SECTION
# ======================================================================
# Don't plot bins with PDF<PDF_THRESHOLD
NUMBER_THRESHOLD=50 # Don't plot points where <100 samples are used for conditional 
                     # averaging

FIG_OUTPUT_DIR=os.environ["WK_DIR"]+"/model/PS"
FIG_OUTPUT_FILENAME=bin_data["BIN_OUTPUT_FILENAME"]
FIG_EXTENSION='pdf'

## Binned data filename & figure directory/filename for OBS (default: R2TMIv7) ##
OBS="ERA-I + TRMM3B42" # will show up in the MODEL figure
# REGION_STR_OBS=["WPac","EPac","Atl","Ind"]
bin_obs_list=sorted(glob.glob(os.environ["OBS_DATA"]\
                    +"trmm3B42_erai_2002_2014.convecTransLev2"
                    +".nc"))
FIG_OBS_DIR=os.environ["WK_DIR"]+"/obs/PS"
FIG_OBS_FILENAME="convecTransLev2_trmm3B42_erai_2002_2014"+".pdf"

# Force the OBS & MODEL figures to use the same color map
#  Will be ignored if binned OBS data does not exist
# Set to False if COLUMN is defined differently for OBS & MODEL
USE_SAME_COLOR_MAP=True

# Plot OBS results on top of MODEL results for comparison
#  Will be ignored if binned OBS data does not exist
# If REGION MASK and/or COLUMN DEFINITION are changed
#  set to False unless the corresponding binned OBS data exists
OVERLAY_OBS_ON_TOP_OF_MODEL_FIG=True

## Plot formatting ##
axes_fontsize = 13 # size of font in all plots
axes_elev= 20 # 30 elevation for 3D plot
axes_azim=290 # 300 azimuthal angle for 3D plot
# legend_fontsize = 9
# marker_size = 40 # size of markers in scatter plots
# xtick_pad = 10 # padding between x tick labels and actual plot
figsize1 = 9.5 # figure size set by figsize=(figsize1,figsize2)
figsize2 = 6 

### There is currently one figure in level 2 diagnostics ###
### Choose the plot parameters for each figure below ###
xlim1={}
xlim2={}

ylim1={}
ylim2={}

zlim1={}
zlim2={}

xlabel={}
ylabel={}
zlabel={}

xticks={}
yticks={}

#==========================================
###### Figure 1 : Precip vs. CWV #########
#==========================================
xlim1['f1']=0 
xlim2['f1']=40

ylim1['f1']=-40
ylim2['f1']=10

zlim1['f1']=0
zlim2['f1']=6


### Enter labels as strings; Latex mathtype is allowed within $...$ ##
xlabel['f1']="SUBSAT (K)"
ylabel['f1']="CAPE (K)"
zlabel['f1']="Precip (mm/h)"

### Enter ticks as lists ##
## Note: this option overrides axes limit options above ##
# xticks['f1']=[10,20,30,40,50,60,70,80]
# yticks['f1']=[0,1,2,3,4,5,6,7,8]
# 
# #========================================================
# ###### Figure 2 : Probability of precip vs. CWV #########
# #========================================================
# xlim1['f2']=10 
# xlim2['f2']=80
# 
# ylim1['f2']=0
# ylim2['f2']=1
# 
# ### Enter labels as strings; Latex mathtype is allowed within $...$ ##
# xlabel['f2']="CWV (mm)"
# ylabel['f2']="Probability of Precip"
# 
# ### Enter ticks as lists ##
# ## Note: this option overrides axes limit options above ##
# xticks['f2']=[10,20,30,40,50,60,70,80]
# yticks['f2']=[0.0,0.2,0.4,0.6,0.8,1.0]
# 
# #==============================================
# ###### Figure 3 : Total PDF vs. CWV #########
# #==============================================
# xlim1['f3']=10 
# xlim2['f3']=80
# 
# ylim1['f3']=1e-5
# ylim2['f3']=5e-2
# 
# ### Enter labels as strings; Latex mathtype is allowed within $...$ ##
# xlabel['f3']="CWV (mm)"
# ylabel['f3']="PDF (mm$^-$$^1$)"
# 
# ### Enter ticks as lists ##
# ## Note: this option overrides axes limit options above ##
# xticks['f3']=[10,20,30,40,50,60,70,80]
# yticks['f3']=[]
# 
# #====================================================
# ###### Figure 4 : Precipitating PDF vs. CWV #########
# #====================================================
# xlim1['f4']=10 
# xlim2['f4']=80
# 
# ylim1['f4']=1e-5
# ylim2['f4']=5e-2
# 
# ### Enter labels as strings; Latex mathtype is allowed within $...$ ##
# xlabel['f4']="CWV (mm)"
# ylabel['f4']="PDF (mm$^-$$^1$)"
# 
# ### Enter ticks as lists ##
# ## Note: this option overrides axes limit options above ##
# xticks['f4']=[10,20,30,40,50,60,70,80]
# yticks['f4']=[]

# ======================================================================
# END USER SPECIFIED SECTION
# ======================================================================
#
# ======================================================================
# DO NOT MODIFY CODE BELOW UNLESS
# YOU KNOW WHAT YOU ARE DOING
# ======================================================================
data={}

# data["PDF_THRESHOLD"]=PDF_THRESHOLD
# 
# data["CWV_RANGE_THRESHOLD"]=CWV_RANGE_THRESHOLD
# 
# data["CP_THRESHOLD"]=CP_THRESHOLD

data["FIG_OUTPUT_DIR"]=FIG_OUTPUT_DIR
data["FIG_OUTPUT_FILENAME"]=FIG_OUTPUT_FILENAME
data["FIG_OBS_FILENAME"]=FIG_OBS_FILENAME
data["FIG_EXTENSION"]=FIG_EXTENSION

data["args3"]=[ bin_obs_list]
print(data['args3'])

# ,\
#                 bin_data["TAVE_VAR"],\
#                 bin_data["QSAT_INT_VAR"],\
#                 bin_data["BULK_TROPOSPHERIC_TEMPERATURE_MEASURE"] ]

data["args4"]=[ NUMBER_THRESHOLD, FIG_OUTPUT_DIR,FIG_OUTPUT_FILENAME, FIG_EXTENSION,\
                OBS, FIG_OBS_DIR,FIG_OBS_FILENAME,\
                USE_SAME_COLOR_MAP,OVERLAY_OBS_ON_TOP_OF_MODEL_FIG ]
# 
fig_params={}

fig_params['f0']=[axes_fontsize,axes_elev,axes_azim,figsize1,figsize2]
# for i in ['f1','f2','f3','f4']:    
for i in ['f1']:    
    fig_params[i]=[[xlim1[i],xlim2[i]],[ylim1[i],ylim2[i]],[zlim1[i],zlim2[i]], xlabel[i],ylabel[i],zlabel[i]]
#                 
data["plot_params"]=fig_params
# 
with open(os.environ["WK_DIR"]+"/"+"convecTransLev2_plot_parameters.json", "w") as outfile:
    json.dump(data, outfile)
