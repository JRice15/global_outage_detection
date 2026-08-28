import datetime
import os
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
from tqdm import tqdm

from src.counties import get_county_geometries
from src.base import OUTPUT_DIR, DATA_DIR


def flatten_night(df):
    # get constant data from first row of set
    row = df.iloc[0]
    row = row[[
        "fips_code",
        "date",
        "name",
        "censusarea",
        # "geometry",
        "lon_approx",
        "lat_approx",
        "utc_offset_solar",
        "modeled_total_customers",
    ]]
    
    df = df.set_index("minutes_from_local_midnight_round15")
    indices = list(range(30, 240+1, 15)) # 12:30a to 4:00a
    observations = []
    for i in indices:
        try:
            val = df.loc[i].customers_out
        except KeyError:
            val = 0
        observations.append(val)
    observations = pd.Series(observations, index=[f"customers_out_obs{i}" for i in indices])
    observations["customers_out_mean"] = observations.mean()
    return pd.concat([row, observations])


def process_year(year, pool):
    counties = get_county_geometries()
    
    outages = pd.read_csv(
        DATA_DIR / f"eaglei-data/all_outages_new/database_2024/eaglei_outages_{year}.csv",
        engine="c",
        dtype={"fips_code": int, "customers_out": float, "run_start_time": str},
        parse_dates=False,
        usecols=["fips_code", "customers_out", "run_start_time"],
    ).dropna()
    # runs start at this time, and take less than 5 minutes to complete (Brelsford 2024)
    outages["run_start_time"] = pd.to_datetime(outages["run_start_time"], format="%Y-%m-%d %H:%M:%S")
    outages = outages.rename(columns={"run_start_time": "time_utc"})
    
    # parse time string
    outages["date"] = outages.time_utc.dt.date
    outages = counties.merge(outages)
    outages["time_local"] = outages["time_utc"] + outages["utc_offset_solar"]
    
    outages = outages[
        (outages.time_local.dt.time >= datetime.time.fromisoformat("00:15"))
        &
        (outages.time_local.dt.time <= datetime.time.fromisoformat("04:15"))
    ]
    outages["minutes_from_local_midnight"] = (outages.time_local.dt.hour * 60) + outages.time_local.dt.minute
    outages["minutes_from_local_midnight_round15"] = np.round(outages["minutes_from_local_midnight"] / 15).astype(int) * 15

    # process each night-group
    args_generator = (group for name, group in outages.groupby(["fips_code", "date"]))
    iterator = pool.imap(flatten_night, args_generator)
    results = [result for result in tqdm(iterator)]
    results = pd.DataFrame(results)
    # for some reason these dtypes get converted to object
    results["date"] = pd.to_datetime(results["date"])
    results["name"] = results["name"].astype(str)
    return results


def main():
    out_dir = OUTPUT_DIR / "nightly_eaglei/"
    os.makedirs(out_dir, exist_ok=True)
    
    with multiprocessing.Pool() as pool:
        print(f"parallelized with {pool._processes} processes")
        for year in range(2014, 2024+1):
            print(year)
            results = process_year(year, pool)

            results.to_hdf(
                out_dir / f"nightly_eaglei_{year}.pandas.hdf", 
                key="data",
                # driver="GPKG",
            )
        

if __name__ == "__main__":
    main()
