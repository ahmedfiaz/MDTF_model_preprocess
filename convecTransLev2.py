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
import time as tt

t0=tt.time()

### SET PATH NAMES ###
### Note: path names in the actual package are set elsewhere ###
import argparse, os, util
parser = argparse.ArgumentParser()
cwd = os.path.dirname(os.path.realpath(__file__)) # gets dir of currently executing script
parser.add_argument('config_file', nargs='?', type=str, 
                    default=os.path.join(cwd, 'config.yml'),
                    help="Configuration file.")
args = parser.parse_args()
config = util.read_yaml(args.config_file)
os.environ["OBS_DATA"] = config['paths']['OBS_DATA_ROOT']
os.environ['POD_HOME']= cwd
os.environ['WK_DIR']= cwd
#########################


# Import Python functions specific to Convective Transition Basic Statistics
from convecTransLev2_util import generate_region_mask
from convecTransLev2_util import convecTransLev2_preprocess
from convecTransLev2_util import convecTransLev2_extractprecip
from convecTransLev2_util import convecTransLev2_bin
from convecTransLev2_util import convecTransLev2_loadAnalyzedData
from convecTransLev2_util import convecTransLev2_plot

# from convecTransBasic_util import convecTransBasic_plot
print("**************************************************")
print("Executing Convective Transition Level 2 Statistics (convecTransLev2.py)......")
print("**************************************************")

print("Load user-specified binning parameters..."),

# Create and read user-specified parameters
# temporarily using os.getcwd instead of POD_HOME
os.system("python "+os.getcwd()+"/"+"convecTransLev2_usp_calc.py")
with open(os.getcwd()+"/"+"convecTransLev2_calc_parameters.json") as outfile:
    bin_data=json.load(outfile)
print("...Loaded!")

print("Load user-specified plotting parameters..."),
os.system("python "+os.environ["POD_HOME"]+"/"+"convecTransLev2_usp_plot.py")
with open(os.environ["WK_DIR"]+"/"+"convecTransLev2_plot_parameters.json") as outfile:
    plot_data=json.load(outfile)
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
        convecTransLev2_preprocess(bin_data["args1"])

    ### Only for CMIP6 models with different output formats between precip
    ### and thetae-based variables ###
            
    if (bin_data["SAVE_PRECIP"]==1) :
        print("     Precip-thetae matching and saving required")
        convecTransLev2_extractprecip(bin_data["args3"])

    else:
        print("    Pre-processed data available...")
        print("     Now binning...")


    # Load & pre-process region mask
    REGION=generate_region_mask(bin_data["REGION_MASK_DIR"]+"/"+bin_data["REGION_MASK_FILENAME"], 
    bin_data["pr_list"][0],bin_data["LAT_VAR"],bin_data["LON_VAR"])

    # Pre-process temperature (if necessary) & bin & save binned results
    binned_output=convecTransLev2_bin(REGION, bin_data['args1'])

else: # Binned data file exists & BIN_ANYWAY=False
    print("Binned output detected...")    
    binned_output=convecTransLev2_loadAnalyzedData(bin_data["args2"])


plot_data["args4"].append(bin_data["MODEL"])
# Plot binning results & save the figure in wkdir/MDTF_casename/.../
convecTransLev2_plot(binned_output,plot_data["plot_params"],plot_data["args3"],plot_data["args4"])
print("**************************************************")
print("Convective Transition Basic Statistics (convecTransLev2.py) Executed!")

t1=tt.time()

print('Took a total of %.2f minutes'%((t1-t0)/60))
exit()

