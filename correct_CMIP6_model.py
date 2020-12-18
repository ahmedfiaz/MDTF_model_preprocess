import xarray as xr
from glob import glob
import os

fils=glob('/scratch/neelin/CMIP6/CNRM-CM6-1-HR/ta/ta_6hrLev_CNRM-CM6-1-HR_historical_r1i1p1f2_gr_*')
fils.sort()

ds_ref=xr.open_dataset(fils[0])
ds_ref['ap'],ds_ref['b']
print(ds_ref['ap'],ds_ref['b'])

for i in fils[5:]:
    print(i)
    ds=xr.open_mfdataset(i)
    ds=ds.assign_coords({"ap":ds_ref['ap'].values,
                      "b":ds_ref['b'].values})
    os.remove(i)
    ds.to_netcdf(i,mode='w')

    #print(ds['ap'].values,ds['b'].values)
    ds.close()

print(fils[0])

