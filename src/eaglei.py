import os
import functools
import glob
import multiprocessing as mp

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import pathlib

from src.base import DATA_DIR
from src.counties import get_county_geometries


def _load_outage_year(path):
    counties = get_county_geometries()
    
    outages = pd.read_csv(
        path,
        engine="c",
        dtype={"fips_code": int, "customers_out": float, "run_start_time": str},
        parse_dates=False,
        usecols=["fips_code", "customers_out", "run_start_time"],
    ).dropna()
    outages = outages.rename(columns={"run_start_time": "time_utc"})
    
    # parse time string
    outages["time_utc"] = pd.to_datetime(outages["time_utc"], format="%Y-%m-%d %H:%M:%S")
    outages = counties.merge(outages)
    outages["time_local"] = outages["time_utc"] + outages["utc_offset_solar"]

    # get observation closest to 1:30am for all nights
    outages = outages[outages.time_local.dt.hour == 1]
    t = outages.time_local.dt
    minutes_offset = (t.hour * 60) + t.minute + (t.second / 60)
    diff_from_130am = minutes_offset - 90
    # with observations every 15 minutes, the closest to 1:30am will always be within 7.5 minutes or less
    outages = outages[np.abs(diff_from_130am) < 7.5]
    
    return outages


def load_nightly_eaglei(year_range=None, n_procs=12):
    """
    args:
        year_range: (min_year, max_year) inclusive, or None to load all data
    """
    globpath = DATA_DIR / "eaglei-data/all_outages_new/database_2024/eaglei_outages_20??.csv"
    found_paths = sorted(glob.glob(str(globpath)))
    
    # filter paths to those within year range
    if year_range is None:
        paths = found_paths
    else:
        paths = []
        for path in found_paths:
            year = int(pathlib.PurePath(path).stem.split("_")[-1])
            min_year, max_year = year_range
            if year >= min_year and year <= max_year:
                paths.append(path)
    
    all_outages = []
    n_procs = min(n_procs, len(paths)+1)
    print("loading eaglei with", n_procs, "processes")
    with mp.Pool(n_procs) as pool:
        for df in tqdm(pool.imap(_load_outage_year, paths), total=len(paths)):
            all_outages.append(df)

    all_outages = pd.concat(all_outages, axis=0)
    return all_outages