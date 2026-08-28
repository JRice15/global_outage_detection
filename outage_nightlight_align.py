import argparse
import os
import sys
import multiprocessing
import functools
import warnings
import gc
import re
import time

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import numba
from exactextract import exact_extract

from src.base import OUTPUT_DIR
from src.utils import downsampling_2d, timed
from src.counties import get_county_geometries
from src.download_nasa_nightlight import lonlat_to_nightlight_tiles
from src.downsampled_nightlight import load_downsampled_nightlight
from src.landscan import load_landscan
from src.ghs_building_height import load_ghs_building_height
from compute_linear_correction import load_normalized_nightlight


@functools.cache
def load_landscan_eaglei_ratio():
    return xr.load_dataset(OUTPUT_DIR / f"landscan_eaglei_ratio_2024.nc").landscan_to_eaglei_ratio


@functools.cache
def load_annual_constants(year):
    ds = xr.Dataset()
    ds["landscan"] = load_landscan(year)
    annual_stats = xr.open_dataset(OUTPUT_DIR / f"annual_outage_stats/annual_outage_stats_{year}.nc")
    # now that angular stuff is handled by linear correction, no need to load it here
    annual_stats = annual_stats[["count", "mean", "std", "median", "median_dev"]]
    annual_stats = annual_stats.load()
    assert np.isclose(ds.x, annual_stats.lon).all()
    assert np.isclose(ds.y, annual_stats.lat).all()
    # add to ds
    for name, var in annual_stats.variables.items():
        if var.ndim == 2:
            ds[f"annual_{name}"] = (("y", "x"), var.values)
        if var.ndim == 4:
            raise NotImplementedError()

    # add landscan/eaglei ratio
    ds["landscan_to_eaglei_ratio"] = load_landscan_eaglei_ratio()

    # add ghs building height
    ds["building_height"] = load_ghs_building_height()

    return ds

def get_sample_nights(rows):
    N = 25 # number of percentile and peak sample each to take

    rows = rows.sort_values(by="customers_out_mean") # mean from 12:30a to 4:00a
    # mask of rows to select
    mask = np.zeros(len(rows), dtype=bool)
    # N equally spaced percentiles from 0 to 100%
    mask[np.linspace(0, len(rows), num=N, endpoint=False).astype(int)] = True
    # top N of nights by outage
    mask[-N:] = True
    return rows[mask]



def dtype32_like(data):
    if np.issubdtype(data.dtype, np.integer):
        return "int32"
    else:
        return "float32"


@numba.njit
def linear_weighted_binning_histogram(values, weights, bin_edges):
    """
    density histogram with points linearly interpolated between bins
    values: array of shape (n_samples,)
    weights: array of shape (n_samples,)
    bins: array of shape (n_bins + 1,) representing bin edges
    returns: array of shape (n_bins,) representing density histogram
    """
    n_bins = len(bin_edges) - 1
    hist = np.zeros(n_bins, dtype=np.float32)
    for i in range(len(values)):
        value = values[i]
        weight = weights[i]
        # find the two bins this value falls between
        for j in range(n_bins):
            if bin_edges[j] <= value < bin_edges[j+1]:
                # linearly interpolate between these two bins
                interp_weight_upper = (value - bin_edges[j]) / (bin_edges[j+1] - bin_edges[j])
                interp_weight_lower = 1 - interp_weight_upper
                hist[j] += interp_weight_lower * weight
                if j + 1 < n_bins:
                    hist[j+1] += interp_weight_upper * weight
    return hist


def process_one_county_year(eaglei, fips_code, year):
    """
    get all results for one county for one year

    args:
        eaglei: eaglei dataframe of outage samples for this county
    """
    assert len(pd.unique(eaglei.fips_code)) == 1
    assert len(pd.unique(eaglei.date.dt.year)) == 1

    # load annual constants
    constants = load_annual_constants(year)

    # GeoDataFrame/Geoseries for this county. some ops require one, some the other
    county_gdf = get_county_geometries()
    county_gdf = county_gdf[county_gdf.fips_code == fips_code]
    county_geoseries = county_gdf.iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        county_pts = county_gdf.buffer(0.02).get_coordinates()
        county_centroid = county_gdf.geometry.centroid.iloc[0]

    tiles = lonlat_to_nightlight_tiles(county_pts.x, county_pts.y)
    
    # load nightlight by day
    all_valid_rows = []
    data_stacks = []
    for row in eaglei.itertuples():
        result = load_downsampled_nightlight(
            row.date, 
            tiles=tiles, 
            varsets=["nightlight", "view", "moon", "snow", "clouds"],
        )
        if result is None:
            continue
        data, (date, nl_lons, nl_lats) = result

        all_valid_rows.append(row)
        data_stacks.append(data)

    if not len(data_stacks):
        return None
    # get valid eaglei rows
    eaglei = eaglei.set_index("date").loc[[row.date for row in all_valid_rows]].reset_index(drop=False)
    assert len(eaglei) == len(data_stacks)

    # stack data from all days together into arrays of shape (day, lat, lon)
    data_stacks = {
        k: np.stack([d[k] for d in data_stacks], axis=0)
        for k in data_stacks[0].keys()
    }
    all_dates = [row.date.date() for row in all_valid_rows]

    # minimum detectable brightness
    L_MIN = 0.29 # Roman et al. 2018 https://doi.org/10.1016/j.rse.2018.03.017
    data_stacks["nightlight"] = np.maximum(data_stacks["nightlight"], L_MIN)
    
    # load linear normalized data for same dates
    normalized_data, (norm_lons, norm_lats) = load_normalized_nightlight(
        dates=pd.DatetimeIndex(all_dates),
        lonlats=(county_pts.x, county_pts.y),
        variables=["normalized_frac", "normalized_zscore"],
    )
    
    # get constants for same area
    # rounding nudges pixels to perfect 10-degree lines to account for float imprecision
    constants = constants.sel(
        x=slice(nl_lons[0].round(), nl_lons[-1].round()),
        y=slice(nl_lats[0].round(), nl_lats[-1].round()),
        # pm=5, # just use the +/-5 degrees version
    )
    if constants.annual_mean.shape != data_stacks["nightlight"].shape[1:]:
        raise ValueError(f"shape mismatch: {constants.shape} {data_stacks['nightlight'].shape}")
    if normalized_data["normalized_frac"].shape != data_stacks["nightlight"].shape:
        raise ValueError(f"shape mismatch: {normalized_data['normalized_frac'].shape} {data_stacks['nightlight'].shape}")
        
    # create output dataset
    ds = xr.Dataset()
    ds["landscan"] = constants.landscan
    coords = dict(
        date=np.array(all_dates, dtype="datetime64[D]"),
        y=ds.y,
        x=ds.x,
    )
    # data stacks
    ds["nightlight"] = xr.DataArray(data_stacks["nightlight"], coords=coords)
    ds["sensor_zenith"] = xr.DataArray(data_stacks["zenith"], coords=ds.nightlight.coords)
    ds["sensor_azimuth"] = xr.DataArray(data_stacks["azimuth"], coords=ds.nightlight.coords)
    ds["moon_illum_frac"] = xr.DataArray(data_stacks["moon_illum_frac"], coords=ds.nightlight.coords)
    ds["lunar_zenith"] = xr.DataArray(data_stacks["lunar_zenith"], coords=ds.nightlight.coords)
    ds["lunar_azimuth"] = xr.DataArray(data_stacks["lunar_azimuth"], coords=ds.nightlight.coords)
    ds["viewhour"] = xr.DataArray(data_stacks["viewhour"], coords=ds.nightlight.coords)
    ds["snow"] = xr.DataArray(data_stacks["snow"], coords=ds.nightlight.coords)
    ds["clouds"] = xr.DataArray(data_stacks["clouds"], coords=ds.nightlight.coords)
    ds["cloud_mask_quality"] = xr.DataArray(data_stacks["cloud_mask_quality"], coords=ds.nightlight.coords)
    # normalized data
    ds["normalized_frac"] = xr.DataArray(normalized_data["normalized_frac"], coords=ds.nightlight.coords)
    ds["normalized_zscore"] = xr.DataArray(normalized_data["normalized_zscore"], coords=ds.nightlight.coords)
    
    # annual constants
    for stat in [
            "landscan_to_eaglei_ratio", 
            "building_height",
            "annual_count", 
            "annual_mean", 
            "annual_std", 
            "annual_median", 
            "annual_median_dev",
        ]:
        ds[stat] = constants[stat]


    #####
    ## Extract county boundaries
    #####
    
    def vals(values, coverage):
        return values
    def coverage_weight(values, coverage):
        return coverage

    # flip Y axis. required by exactextract for some reason
    ds = ds.reindex(y=ds.y[::-1])
    # extract pixel values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # extract for all variables
        df = exact_extract(
            ds,
            county_gdf,
            ops=vals,
            output="pandas",
        )
        df.columns = [re.sub(r"_vals_", "_count_", col)[:-5] for col in df.columns] # remove "_vals" suffix
        # get coverage weight (landscan is placeholder, any var could work)
        df["coverage_weight"] = exact_extract(
            ds.landscan,
            county_gdf,
            ops=coverage_weight,
            output="pandas",
        ).coverage_weight

    # to series
    df = df.squeeze()

    # format as xarray
    results = xr.Dataset()
    # spatial data
    temporal_prefixes = [
        "nightlight", 
        "sensor_zenith",
        "sensor_azimuth",
        "moon_illum_frac",
        "lunar_zenith",
        "lunar_azimuth",
        "viewhour",
        "snow",
        "clouds",
        "cloud_mask_quality",
        "normalized_frac",
        "normalized_zscore",
    ]
    is_temporal = lambda x: any(x.startswith(name) for name in temporal_prefixes)
    # stack temporal features together
    for prefix in temporal_prefixes:
        cols = [col for col in df.index if col.startswith(prefix)]
        results[prefix] = (("date", "pixels"), np.stack(df[cols].values, dtype=dtype32_like(df[cols[0]])))
    # other spatial-only features
    other_cols = [col for col in df.index if not is_temporal(col)]
    for col in other_cols:
        data = df[col]
        data = data.astype(dtype32_like(data))
        results[col] = (("pixels",), data)
    
    # County centroid
    results["lon_countycentroid"] = county_centroid.x
    results["lat_countycentroid"] = county_centroid.y

    results["date"] = np.array(all_dates, dtype="datetime64[D]")

    # scalars
    results["year"] = year
    results["fips_code"] = fips_code
    results["modeled_total_customers"] = eaglei.modeled_total_customers.iloc[0]
                       
    # pop coverage per night
    COVERED_LANDSCAN = (results.landscan * results.coverage_weight).fillna(0.0)
    total_pop = COVERED_LANDSCAN.sum(dim="pixels")
    pop_coverage = ((~np.isnan(results.normalized_zscore)) * COVERED_LANDSCAN).sum(dim="pixels")
    results["pop_coverage_frac"] = pop_coverage / total_pop

    # must have at least 90% pop coverage
    DATE_MEETS_POP_COV_MASK = results.pop_coverage_frac >= 0.90
    # return none if no dates with meet population coverage threshold
    if DATE_MEETS_POP_COV_MASK.sum() == 0:
        return None

    # mask away bad dates
    results = results.sel(date=DATE_MEETS_POP_COV_MASK)
    eaglei = eaglei[DATE_MEETS_POP_COV_MASK.values]

    # eaglei data
    minute_offsets = list(range(30, 240+1, 15))
    eaglei_outages_15minutely = eaglei[[f"customers_out_obs{k}" for k in minute_offsets]]
    results["customers_out_nightly_mean"] = (("date",), eaglei_outages_15minutely.values.mean(axis=1))
    results["customers_out_nightly_min"] = (("date",), eaglei_outages_15minutely.values.min(axis=1))
    results["customers_out_nightly_max"] = (("date",), eaglei_outages_15minutely.values.max(axis=1))

    # get customers out at exact viewing times
    # convert viewing time to county's local mean solar time, in minutes from midnight
    local_solar_viewing_time = (results["viewhour"] * 60) + (county_geoseries.utc_offset_solar.total_seconds() / 60)
    # round to nearest 15-minutes
    local_solar_viewing_time = np.round(local_solar_viewing_time / 15) * 15
    customers_out_atviewtime = []
    customers_out_atviewtime_min = []
    customers_out_atviewtime_max = []
    hist_bins = np.array(minute_offsets + [minute_offsets[-1]+15]) # extra for rightmost edge of histogram
    for i in range(len(results.date)):
        # get view times binned by 15-minute interval
        view_freqs = linear_weighted_binning_histogram(
            local_solar_viewing_time.isel(date=i).values,
            weights=COVERED_LANDSCAN.values, # weight by population coverage at each pixel
            bin_edges=hist_bins,
        )
        assert view_freqs.sum() > 0, f"view_freqs sum to {view_freqs.sum()} for date {all_dates[i]}, fips {fips_code}"
        view_freqs /= view_freqs.sum()  # normalize to sum to 1
        # get corresponding customers-out readings
        customersout_at_bins = eaglei_outages_15minutely.iloc[i].values
        # take the weighted average
        customers_out_atviewtime.append(
            np.sum(view_freqs * customersout_at_bins)
        )
        # get lower and upper bound
        customers_out_atviewtime_min.append(
            customersout_at_bins[view_freqs > 0].min()
        )
        customers_out_atviewtime_max.append(
            customersout_at_bins[view_freqs > 0].max()
        )
    results["customers_out_atviewtime"] = (("date",), customers_out_atviewtime)
    results["customers_out_atviewtime_min"] = (("date",), customers_out_atviewtime_min)
    results["customers_out_atviewtime_max"] = (("date",), customers_out_atviewtime_max)

    return results

    

def process_one_county_year_wrapper(kwargs):
    return process_one_county_year(**kwargs)


@timed
def process_one_year(process_kwargs_list, year, nworkers, maxtasksperchild):
    print(year, "-", nworkers, "workers")

    load_annual_constants(year) # preload constants
    
    results = dict()
    if nworkers == 1:
        for kwargs in tqdm(process_kwargs_list):
            ds = process_one_county_year(**kwargs)
            if ds is not None:
                results[str(int(ds.fips_code))] = xr.DataTree(ds)
    else:
        with multiprocessing.Pool(nworkers, maxtasksperchild=maxtasksperchild) as pool:
            iterator = pool.imap(process_one_county_year_wrapper, process_kwargs_list)
            for ds in tqdm(iterator, total=len(process_kwargs_list)):
                if ds is not None:
                    # ds.encoding = dict(zlib=True, complevel=7)
                    results[str(int(ds.fips_code))] = xr.DataTree(ds)

    # stack them all together
    results = xr.DataTree(children=results)

    # save to path
    path = OUTPUT_DIR / f"aligned_nightlight_outage_zarr/aligned_nightlight_outages_{year}.xarray.zarr"
    os.makedirs(path.parent, exist_ok=True)
    results.to_zarr(path, mode="w")

    gc.collect()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2014)
    parser.add_argument("--end", type=int, default=2024, help="inclusive")
    # --workers 64 --maxtasks 16 needed for 2012-2022
    # --workers 32 --maxtasks 16 needed for 2023-2024
    parser.add_argument("--workers", default=64, type=int, help="number of parallel workers to use (default 64. Too many => OOM)")
    parser.add_argument("--maxtasksperchild", default=100, type=int)
    parser.add_argument("--mp-start-method", default=None)
    ARGS = parser.parse_args()

    multiprocessing.set_start_method(ARGS.mp_start_method)

    # have to load all years even if only processing later ones
    eaglei = []
    for year in tqdm(range(2014, 2025), desc="loading eaglei"):
        eaglei.append(
            pd.read_hdf(OUTPUT_DIR / "nightly_eaglei" / f"nightly_eaglei_{year}.pandas.hdf", key="data")
        )
    eaglei = pd.concat(eaglei)

    print("identifying outage nights")
    eaglei = eaglei.groupby("fips_code").apply(get_sample_nights, include_groups=False)
    # save 'fips_code', drop the 'level_1' meaningless index that get created
    eaglei = eaglei.reset_index(level=0, drop=False).reset_index(drop=True)

    # groupby counties
    # format as kwargs for process_one_night(...)
    print("splitting into county-years")
    process_kwargs_list = []
    for name, group in tqdm(eaglei.groupby(["fips_code", eaglei.date.dt.year])):
        process_kwargs_list.append(dict(
            fips_code=group.fips_code.iloc[0],
            year=group.date.dt.year.iloc[0],
            eaglei=group,
        ))

    # run multiprocessing once for each year
    for year in range(ARGS.start, ARGS.end+1):
        process_one_year(
            [kw for kw in process_kwargs_list if kw["year"] == year],
            year,
            nworkers=ARGS.workers,
            maxtasksperchild=ARGS.maxtasksperchild,
        )



if __name__ == "__main__":
    main()