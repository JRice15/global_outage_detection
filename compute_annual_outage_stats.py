import multiprocessing as mp
import functools
import argparse
import glob
import pathlib
import os
import time
import warnings

import pandas as pd
import numpy as np
import xarray as xr
import scipy.stats
import numba
from tqdm import tqdm

from src.base import DATA_DIR, OUTPUT_DIR
from src.utils import downsampling_2d, timed
from src.counties import get_county_geometries
from src.download_nasa_nightlight import lonlat_to_nightlight_tiles, get_nightlight

counties = get_county_geometries()

tiles = list(sorted(lonlat_to_nightlight_tiles(
    counties.lon_approx,
    counties.lat_approx,
)))


def load_one(path, year):
    dayofyear = int(pathlib.PurePath(path).stem.split("_")[-1])
    result = get_nightlight(
        year, 
        dayofyear, 
        return_variables=[
            "DNB_BRDF-Corrected_NTL--high_quality_only",
            # "Sensor_Zenith",
        ],
        return_as="stitched-full",
        tiles=tiles, 
        allow_missing=True,
    )
    if result is None:
        return None
    
    nightlight, (lons, lats) = result
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # catch all-nan slice warnings from downsampling
        # downsample to landscan resolution
        nightlight = downsampling_2d(nightlight, 2, 2, agg_f=np.mean)
        # # median acts similar to mode when 3/4 are similar, or mean when 2/4 disagree
        # zenith = downsampling_2d(zenith, 2, 2, agg_f=np.median)
    # select pixel centers
    lons = lons[1::2]
    lats = lats[1::2]
    return nightlight, (lons, lats)

# @timed
@numba.njit(parallel=True)
def nanmean0(arr):
    """nanmean over axis zero of 3D array"""
    z,y,x = arr.shape
    output = np.empty((y,x), np.float32)
    for i in numba.prange(y):
        for j in range(x):
            output[i,j] = np.nanmean(arr[:,i,j])
    return output

# @timed
@numba.njit(parallel=True)
def nanstd0(arr):
    """nanstd over axis zero of 3D array"""
    z,y,x = arr.shape
    output = np.empty((y,x), np.float32)
    for i in numba.prange(y):
        for j in range(x):
            output[i,j] = np.nanstd(arr[:,i,j])
    return output

# @timed
@numba.njit(parallel=True)
def nanmedian0(arr):
    """nanmedian over axis zero of 3D array"""
    z,y,x = arr.shape
    output = np.empty((y,x), np.float32)
    for i in numba.prange(y):
        for j in range(x):
            output[i,j] = np.nanmedian(arr[:,i,j])
    return output

# @timed
@numba.njit(parallel=True)
def validcount0(arr):
    """count non-NaN over axis zero of 3D array"""
    z,y,x = arr.shape
    output = np.empty((y,x), np.int32)
    for i in numba.prange(y):
        for j in range(x):
            output[i,j] = np.sum(~np.isnan(arr[:,i,j]), np.int32)
    return output

# @timed
@numba.njit(parallel=True)
def nanmeddev0(arr, median):
    """nanstd over axis zero of 3D array"""
    z,y,x = arr.shape
    output = np.empty((y,x), np.float32)
    for i in numba.prange(y):
        for j in range(x):
            output[i,j] = np.nanmedian(np.abs(arr[:,i,j] - median[i,j]))
    return output







# @timed
@numba.njit(parallel=True)
def nanmean0_where(arr, mask):
    """nanmean over axis zero of 3D array"""
    z,y,x = arr.shape
    output = np.empty((y,x), np.float32)
    for i in numba.prange(y):
        for j in range(x):
            output[i,j] = np.nanmean(arr[:,i,j][mask[:,i,j]])
    return output

# @timed
@numba.njit(parallel=True)
def nanstd0_where(arr, mask):
    """nanstd over axis zero of 3D array"""
    z,y,x = arr.shape
    output = np.empty((y,x), np.float32)
    for i in numba.prange(y):
        for j in range(x):
            output[i,j] = np.nanstd(arr[:,i,j][mask[:,i,j]])
    return output

# @timed
@numba.njit(parallel=True)
def nanmedian0_where(arr, mask):
    """nanstd over axis zero of 3D array"""
    z,y,x = arr.shape
    output = np.empty((y,x), np.float32)
    for i in numba.prange(y):
        for j in range(x):
            output[i,j] = np.nanmedian(arr[:,i,j][mask[:,i,j]])
    return output

# @timed
@numba.njit(parallel=True)
def validcount0_where(arr, mask):
    """count non-NaN over axis zero of 3D array"""
    z,y,x = arr.shape
    output = np.empty((y,x), np.int32)
    for i in numba.prange(y):
        for j in range(x):
            is_valid = (~np.isnan(arr[:,i,j])) & mask[:,i,j]
            output[i,j] = np.sum(is_valid, np.int32)
    return output

# @timed
@numba.njit(parallel=True)
def nanmeddev0_where(arr, median, mask):
    """nanstd over axis zero of 3D array"""
    z,y,x = arr.shape
    output = np.empty((y,x), np.float32)
    for i in numba.prange(y):
        for j in range(x):
            output[i,j] = np.nanmedian(np.abs(arr[:,i,j] - median[i,j])[mask[:,i,j]])
    return output


def aggregate_stats(nightlight):
    count_full = validcount0(nightlight)
    mean_full = nanmean0(nightlight)
    std_full = nanstd0(nightlight)
    median_full = nanmedian0(nightlight)
    median_dev_full = nanmeddev0(nightlight, median_full)
    full_out = dict(
        count=count_full, 
        mean=mean_full, 
        std=std_full, 
        median=median_full, 
        median_dev=median_dev_full,
    )

    # angle binned stats
    # angles = list(range(0, 80, 5)) # no zenith angles larger than 75 appear in NASA data
    # pms = [5, 10] # plus-minuses
    # angular_out = {
    #     name: np.empty((len(angles), len(pms), *full_out[name].shape), dtype=full_out[name].dtype)
    #     for name in ["count", "mean", "std", "median", "median_dev"]
    # }
    # for i, angle in enumerate(tqdm(angles)):
    #     for j, pm in enumerate(pms):
    #         mask = (zenith >= angle - pm) & (zenith <= angle + pm)
    #         angular_out["count"][i,j] = validcount0_where(nightlight, mask)
    #         angular_out["mean"][i,j] = nanmean0_where(nightlight, mask)
    #         angular_out["std"][i,j] = nanstd0_where(nightlight, mask)
    #         angular_out["median"][i,j] = nanmedian0_where(nightlight, mask)
    #         angular_out["median_dev"][i,j] = nanmeddev0_where(nightlight, angular_out["median"][i,j], mask)
    return full_out #, angular_out, angles, pms


# precompile agg stats
aggregate_stats(
    np.empty((5, 10, 10), dtype=np.float32),
    # np.empty((5, 10, 10), dtype=np.float32),
)


def compute_annual_stats(year, n_workers=None):
    print(year)

    globpath = DATA_DIR / f"NASA_Nightlight/NASA_VNP46_{year}_???.nc"
    paths = list(glob.glob(str(globpath)))

    nightlight_stack = []
    # zenith_stack = []
    # parallelized I/O
    f_partial = functools.partial(load_one, year=year)
    with mp.Pool(n_workers) as pool:
        print(f"parallelized with {pool._processes} processes")
        for result in tqdm(pool.imap_unordered(f_partial, paths), total=len(paths)):
            if result is not None:
                nightlight, (lons, lats) = result
                nightlight_stack.append(nightlight)
                # zenith_stack.append(zenith)

    nightlight_stack = np.stack(nightlight_stack, axis=0)
    # zenith_stack = np.stack(zenith_stack, axis=0)

    # compute stats
    full_stats = aggregate_stats(nightlight_stack)    

    # save to dataarray
    del nightlight_stack
    ds = xr.Dataset(coords=dict(lon=lons, lat=lats))
    for name, stat in full_stats.items():
        ds[name] = (("lat", "lon"), stat)
    # for name, stat in angular_stats.items():
    #     ds[name+"_angular"] = (("angle", "pm", "lat", "lon"), stat)
        
    # save to nc file
    savepath = OUTPUT_DIR / f"annual_outage_stats/annual_outage_stats_{year}.nc"
    os.makedirs(savepath.parent, exist_ok=True)
    ds.to_netcdf(savepath)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=2012, type=int)
    parser.add_argument("--end", default=2025, type=int)
    parser.add_argument("--workers", default=None, type=int, help="defaults to all available processors")
    ARGS = parser.parse_args()

    for year in range(ARGS.start, ARGS.end+1):
        t = time.perf_counter()
        compute_annual_stats(year, n_workers=ARGS.workers)
        print(f"  took {time.perf_counter() - t} sec")
    


if __name__ == "__main__":
    main()