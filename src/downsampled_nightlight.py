import warnings
import functools

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils import downsampling_2d
from src.download_nasa_nightlight import get_nightlight, lonlat_to_nightlight_tiles

def load_downsampled_nightlight(date, tiles, varsets=["nightlight", "view", "moon", "snow"], subslice=None):
    """
    load 2x2 block-downsampled nightlight and associated variables for a given day and tile. 
    args:
        date: datetime.date or pd.Timestamp
        tiles: list of (h,v) tuples
        varsets: list of one or more of the following
            nightlight: DNB_BRDF-Corrected_NTL--high_quality_only
            view: Sensor_Zenith, Sensor_Azimuth, Viewing_UTC_Hour
            moon: Lunar_Zenith, Lunar_Azimuth, Moon_Illumination_Fraction
            snow: snow flag, 
            clouds: clouds, cloud_quality_mask
    returns: None if no data, else tuple of:
        data: dict(varname: 2D array of downsampled data) with keys:
            nightlight, zenith, azimuth, viewhour, lunar_zenith, lunar_azimuth, moon_illum_frac, snow, clouds, cloud_mask_quality
        (date, lons, lats): tuple of date and 1D arrays of lons and lats corresponding to pixel centers
    """
    return_variables = sum(
        (
            ("DNB_BRDF-Corrected_NTL--high_quality_only",) 
                if "nightlight" in varsets else (),
            ("Sensor_Zenith", "Sensor_Azimuth", "Viewing_UTC_Hour",) 
                if "view" in varsets else (),
            ("Lunar_Zenith", "Lunar_Azimuth", "Moon_Illumination_Fraction",) 
                if "moon" in varsets else (),
            ("QF_Cloud_Mask",) 
                if (("snow" in varsets) or ("clouds" in varsets)) else (),
        ), 
        start=()
    )
    data = get_nightlight(
        date.year, 
        date.dayofyear,
        return_variables=return_variables,
        return_as="stitched-full",
        allow_missing=True,
        tiles=tiles,
    )
    if data is None:
        return None
    else:
        data, (lons, lats) = data
        if len(return_variables) == 1:
            data = {return_variables[0]: data}
        else:
            data = {k:v for k,v in zip(return_variables, data)}
        output = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # catch all-nan slice warnings from downsampling
            if "nightlight" in varsets:
                output["nightlight"] = downsampling_2d(data["DNB_BRDF-Corrected_NTL--high_quality_only"], 2, 2, agg_f=np.mean)
            if "view" in varsets:
                output["zenith"] = downsampling_2d(data["Sensor_Zenith"], 2, 2, agg_f=np.median)
                output["azimuth"] = downsampling_2d(data["Sensor_Azimuth"], 2, 2, agg_f=np.median)
                output["viewhour"] = downsampling_2d(data["Viewing_UTC_Hour"], 2, 2, agg_f=np.mean)
            if "moon" in varsets:
                output["lunar_zenith"] = downsampling_2d(data["Lunar_Zenith"], 2, 2, agg_f=np.median)
                output["lunar_azimuth"] = downsampling_2d(data["Lunar_Azimuth"], 2, 2, agg_f=np.median)
                output["moon_illum_frac"] = downsampling_2d(data["Moon_Illumination_Fraction"], 2, 2, agg_f=np.mean) / 100 # rescale to [0,1]
            if "snow" in varsets:
                snow = (data["QF_Cloud_Mask"] >> 10) & 0b1
                output["snow"] = downsampling_2d(snow, 2, 2, agg_f=np.mean)
            if "clouds" in varsets:
                clouds = (data["QF_Cloud_Mask"] >> 6) & 0b11
                cloud_mask_quality = (data["QF_Cloud_Mask"] >> 4) & 0b11
                output["clouds"] = downsampling_2d(clouds, 2, 2, agg_f=np.mean) / 3 # rescale to [0,1]
                output["cloud_mask_quality"] = downsampling_2d(cloud_mask_quality, 2, 2, agg_f=np.mean) / 3 # rescale to [0,1]
        lons = lons[1::2]
        lats = lats[1::2]
        if subslice is not None:
            output = {k:v[subslice,subslice] for k,v in output.items()}
            lons = lons[subslice]
            lats = lats[subslice]
        return output, (date, lons, lats)



def load_downsampled_nightlight_stack(date_range, pool, tiles=None, lonlats=None, varsets=["nightlight", "view", "moon", "snow"], subslice=None):
    """
    args:
        date_range: iterable of datetime objects
        one of either:
            tiles: list of (h,v) tuples
            lonlats: list of (lon, lat) tuples
        pool: multiprocessing.Pool
    returns:
        data_stacks: dict of varname: 3D array of stacked downsampled data with shape (time, y, x)
        (valid_dates, lons, lats)
    """
    assert (tiles is None) != (lonlats is None), "Must provide either tile or lon/lats for load_downsampled_nightlight_stack"
    if lonlats is not None:
        tiles = lonlat_to_nightlight_tiles(*lonlats)

    valid_dates = []
    data_stacks = []
    
    # possibly parallelized I/O of nightlight data
    f_partial = functools.partial(load_downsampled_nightlight, tiles=tiles, varsets=varsets, subslice=subslice)
    if pool is None:
        iterator = map(f_partial, date_range)
    else:
        print(f"reading nightlight stack with {pool._processes} workers...")
        iterator = pool.imap(f_partial, date_range)
    for result in tqdm(iterator, total=len(date_range)):
        if result is not None:
            data, (date, lons, lats) = result
            valid_dates.append(date)
            data_stacks.append(data)

    valid_dates = pd.DatetimeIndex(valid_dates)
    data_stacks = {
        k: np.stack([d[k] for d in data_stacks], axis=0)
        for k in data_stacks[0].keys()
    }

    return data_stacks, (valid_dates, lons, lats)


