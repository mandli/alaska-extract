#!/usr/bin/env python

import sys
from pathlib import Path
import netCDF4 as nc
import numpy as np

def fix_time(ds: nc.Dataset):
    # Parse Times char array -> seconds since reference
    t0 = np.datetime64("2011-11-01T00:00:00")
    times_raw = nc.chartostring(ds.variables["Times"][:])
    # WRF format: "2011-11-01_00:00:00" -> replace _ with T
    times_dt = np.array([np.datetime64(t.replace("_", "T")) for t in times_raw])
    seconds = (times_dt - t0).astype("timedelta64[s]").astype(float)

    # Replace the variable name 'time' with 'Time' to match the dimension
    if "Time" not in ds.variables:
        tvar = ds.createVariable("Time", "f8", ("Time",))
    else:
        # If you already wrote 'time', delete it and recreate (netCDF4 can't rename in-place)
        # Easier: just overwrite attributes on whatever variable works as the coord
        tvar = ds.variables["Time"]
    tvar[:] = seconds
    tvar.units = "seconds since 2011-11-01 00:00:00"
    tvar.calendar = "standard"
    tvar.standard_name = "time"
    tvar.axis = "T"

def fix_latlon(ds: nc.Dataset):
    # Do some other name fixups for lat/lon
    ds.variables["lon"].standard_name = "longitude"
    ds.variables["lon"].units = "degrees_east"
    ds.variables["lon"].axis = "X"
    ds.variables["lat"].standard_name = "latitude"
    ds.variables["lat"].units = "degrees_north"
    ds.variables["lat"].axis = "Y"

def check_pressure(ds: nc.Dataset):
    print(ds.variables["P"].units)       # should be "Pa"
    print(ds.variables["P"][:].mean())   # sanity check: ~1e5

def main():
    if sys.argv[1:]:
        path = sys.argv[1]
    else:        
        path = "uvp_latlon.nc"

    with nc.Dataset(path, "r+") as ds:
        fix_latlon(ds)
        fix_time(ds)
        check_pressure(ds)


if __name__ == "__main__":
    main()
