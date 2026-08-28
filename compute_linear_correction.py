import multiprocessing
import collections
import functools
import os
import datetime
import argparse
import warnings
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import netCDF4
import numba

from src.base import DATA_DIR
from src.download_nasa_nightlight import lonlat_to_nightlight_tiles
from src.downsampled_nightlight import load_downsampled_nightlight_stack
from src.utils import downsampling_2d
from src.counties import get_county_geometries
from src.landscan import load_landscan
from src.moon import get_moon_states

COMPRESSION_KWARGS = dict(
    compression="zlib",
    complevel=4, # 4 is default
)


def mask_all(mask, *args):
    for arr in args:
        yield arr[mask]

def unmask_all(mask, *args):
    for arr in args:
        out = np.full(shape=mask.shape, fill_value=np.nan, dtype=arr.dtype)
        out[mask] = arr
        yield out


@numba.njit
def _nb_compute_annual_stats(nightlight, data_years, unique_years):
    """
    compute annual stats for a given pixel timeseries
    """
    # initialize outputs
    count = np.full(len(unique_years), 0, dtype=np.int32)
    mean = np.full(len(unique_years), np.nan, dtype=np.float32)
    std = np.full(len(unique_years), np.nan, dtype=np.float32)
    median = np.full(len(unique_years), np.nan, dtype=np.float32)
    median_dev = np.full(len(unique_years), np.nan, dtype=np.float32)
    
    year_start_idx = 0
    for i, year in enumerate(unique_years):
        # data is already masked to only nightlight > 0 in fit_lr
        # years is sorted. get first and last index where years==year
        year_mask = (data_years == year)
        if not year_mask.any():
            continue
        year_end_idx = year_mask.size - np.argmax(year_mask[::-1]) - 1 # gets index of last True value
        values = nightlight[year_start_idx:year_end_idx+1]

        count[i] = len(values)
        if len(values) > 0:
            mean[i] = np.mean(values)
            std[i] = np.std(values)
            median[i] = np.median(values)
            median_dev[i] = np.median(np.abs(values - median[i]))

        # update start idx for next year
        year_start_idx = year_end_idx + 1

    return count, mean, std, median, median_dev


def compute_annual_stats(nightlight, data_years, unique_years):
    """
    wrapper around jitted function to return a dictionary
    """
    count, mean, std, median, median_dev = _nb_compute_annual_stats(nightlight, data_years, unique_years)
    return dict(
        count=count,
        mean=mean,
        std=std,
        median=median,
        median_dev=median_dev,
    )

# precompile jitted function
compute_annual_stats(np.arange(10.0), np.arange(10), np.arange(10))


def fit_lr(nightlight, zenith, azimuth, viewhour, landscan, snow, dates, ij, lonlat, unique_years):
    """
    accepts 1D arrays of nightlight, zenith, azimuth, viewhour, landscan, and datetime, representing timeseries
    of data at a given pixel
    """    
    # get valid portion of data
    valid_mask = (nightlight > 0) & (~np.isnan(viewhour)) # handles both 0 and NaN

    # get rid of invalid steps
    nightlight, zenith, azimuth, viewhour, landscan, snow, dates = mask_all(
        valid_mask, 
        nightlight, zenith, azimuth, viewhour, landscan, snow, dates,
    )

    # minimum detectable brightness
    L_MIN = 0.29 # Roman et al. 2018 https://doi.org/10.1016/j.rse.2018.03.017
    nightlight = np.maximum(nightlight, L_MIN)
    
    # build predictors dataframe
    predictors = pd.get_dummies(dates.dayofweek).add_prefix("dayofweek")
    predictors = predictors.drop(columns=["dayofweek0"]) # drop one to avoid dummy variable trap
    predictors["zenith"] = zenith
    zenith_sq = zenith ** 2
    predictors["zenith_sq"] = zenith_sq
    # azimuth should only matter when zenith substantially different from zero, so only include their interaction effect
    # predictors["azimuth_cos"] = np.cos(np.deg2rad(azimuth))
    # predictors["azimuth_sin"] = np.sin(np.deg2rad(azimuth))
    predictors["zenith*azimuth_cos"] = zenith * np.cos(np.deg2rad(azimuth))
    predictors["zenith*azimuth_sin"] = zenith * np.sin(np.deg2rad(azimuth))
    predictors["zenith_sq*azimuth_cos"] = zenith_sq * np.cos(np.deg2rad(azimuth))
    predictors["zenith_sq*azimuth_sin"] = zenith_sq * np.sin(np.deg2rad(azimuth))
    predictors["days_since_2000"] = (dates - pd.Timestamp("2000-01-01")).days
    predictors["population"] = landscan
    predictors["season_cos"] = np.cos(2*np.pi * dates.dayofyear / 365)
    predictors["season_sin"] = np.sin(2*np.pi * dates.dayofyear / 365)
    predictors["snow"] = snow
    # add moon brightness
    lon, lat = lonlat
    view_datetimes = dates + (viewhour * 3600).astype("timedelta64[s]")
    moons = get_moon_states(view_datetimes, lon=lon, lat=lat, variables=("mag", "alt"))
    predictors["moon_brightness_abovehorizon"] = moons.brightness * (moons.alt > 0)

    # predictors["snow*moonlight"] = predictors["snow"] * predictors["moon_brightness_abovehorizon"]

    assert not predictors.isna().any().any(), ij
    
    # because variance(nightlight) scales with nightlight, log transform target
    log_nightlight = np.log(nightlight)
    # compute multi linear regression
    lr = LinearRegression().fit(
        predictors, 
        log_nightlight,
    )
    # predicted timeseries
    log_pred = lr.predict(predictors)
    # enforce minimim
    log_pred = np.maximum(log_pred, np.log(L_MIN))
    # un-transform
    pred = np.exp(log_pred)

    # pixelwise metrics and coefficients
    metrics = dict(zip(
        [f"coef_{c}" for c in predictors.columns], 
        lr.coef_
    ))
    metrics["coef_intercept"] = lr.intercept_
    metrics["r2_log"] = r2_score(log_nightlight, log_pred)
    metrics["r2"] = r2_score(nightlight, pred)
    metrics["n"] = len(nightlight) # number of valid samples used in regression analysis
    
    # compute normalized and z-scores
    normalized_frac = nightlight / pred
    # compute z-score with normalized frac, so that variance is normalized by predicted value
    normfrac_deviation = normalized_frac - 1.0
    MAD = np.median(np.abs(normfrac_deviation - np.median(normfrac_deviation))) * 1.4286
    normalized_zscore = normfrac_deviation / MAD

    # unmask
    normalized_frac, normalized_zscore = unmask_all(
        valid_mask, 
        normalized_frac, normalized_zscore,
    )
    
    # annual stats
    annual_stats = compute_annual_stats(nightlight, dates.year.values, unique_years)
    
    return dict(
        normalized=dict(
            normalized_frac=normalized_frac,
            normalized_zscore=normalized_zscore,
        ), 
        metrics=metrics,
        annual_stats=annual_stats,
        ij=ij,
    )


def fit_lr_wrapper(kwargs):
    return fit_lr(**kwargs)


def load_landscan_stack_and_inhabited_mask(tile, valid_dates, isglobal=False):
    # get mask of which pixels are inhabited
    inhabited_mask = None
    landscan_dict = {}
    for year in range(valid_dates.year.min(), valid_dates.year.max()+1):
        landscan = load_landscan(year, conus_only=False, tile=tile)
        landscan_dict[year] = np.nan_to_num(landscan.values, nan=0)
        mask = landscan.values > 0
        if inhabited_mask is None:
            inhabited_mask = mask
        else:
            inhabited_mask = inhabited_mask | mask

    # make landscan stack
    landscan_stack = np.stack([
        landscan_dict[year] for year in valid_dates.year
    ]) # (time, lat, lon)

    return landscan_stack, inhabited_mask


def distributed_fit_lr(data_stacks, landscan_stack, valid_pixel_mask, valid_dates, lons, lats, pool, unique_years):
    # yield timeseries data for individual inhabited pixels
    def input_generator():
        for i in range(len(lats)):
            for j in range(len(lons)):
                if valid_pixel_mask[i,j]:
                    yield dict(
                        nightlight=data_stacks["nightlight"][:,i,j],
                        zenith=data_stacks["zenith"][:,i,j],
                        azimuth=data_stacks["azimuth"][:,i,j],
                        viewhour=data_stacks["viewhour"][:,i,j],
                        # lunar_zenith=data_stacks["lunar_zenith"][:,i,j],
                        # lunar_azimuth=data_stacks["lunar_azimuth"][:,i,j],
                        # moon_illum_frac=data_stacks["moon_illum_frac"][:,i,j],
                        snow=data_stacks["snow"][:,i,j],
                        # clouds=data_stacks["clouds"][:,i,j],
                        # cloud_mask_quality=data_stacks["cloud_mask_quality"][:,i,j],
                        landscan=landscan_stack[:,i,j],
                        dates=valid_dates,
                        ij=(i,j),
                        lonlat=(lons[j], lats[i]),
                        unique_years=unique_years,
                    )

    print("processing all timeseries")
    normalized = collections.defaultdict(
        lambda: np.full_like(data_stacks["nightlight"], np.nan) # (time,lat,lon)
    )
    metrics = collections.defaultdict(
        lambda: np.full_like(data_stacks["nightlight"][0], np.nan) # (lat,lon)
    )
    annual_stats = collections.defaultdict(
        lambda: np.full((len(unique_years),) + data_stacks["nightlight"][0].shape, np.nan, dtype=np.float32) # (years, lat,lon)
    )
    for result in tqdm(pool.imap(fit_lr_wrapper, input_generator()), total=valid_pixel_mask.sum()):
        i, j = result["ij"]
        for name,value in result["normalized"].items():
            normalized[name][:,i,j] = value
        for name,value in result["metrics"].items():
            metrics[name][i,j] = value
        for name,value in result["annual_stats"].items():
            annual_stats[name][:,i,j] = value

    metrics = dict(**metrics) # unpack defaultdict
    annual_stats = dict(**annual_stats) # unpack defaultdict

    return normalized, metrics, annual_stats


def process_tile(tile, date_range, min_samples, pool, isglobal=False):
    print("processing", tile)

    # get data stacks for this tile
    data_stacks, (valid_dates, lons, lats) = load_downsampled_nightlight_stack(
        date_range, 
        tiles=[tile], 
        pool=pool, 
        varsets=["nightlight", "view", "snow"],
    )

    # associated landscan
    landscan_stack, inhabited_mask = load_landscan_stack_and_inhabited_mask(tile, valid_dates, isglobal=isglobal)
    
    # get number of valid observations per pixel
    sufficient_samples_mask = (
        np.sum(
            ~(np.isnan(data_stacks["nightlight"]) | np.isnan(data_stacks["viewhour"])), 
            axis=0,
        ) >= min_samples
    )
    # combine the two
    valid_pixel_mask = (inhabited_mask & sufficient_samples_mask)
    print(f"  {inhabited_mask.sum() - valid_pixel_mask.sum()} excluded of {inhabited_mask.sum()} for insufficient samples")
    
    unique_years = np.sort(pd.unique(valid_dates.year))
    
    # fit linear regression and compute normalized values and metrics
    normalized, metrics, annual_stats = distributed_fit_lr(
        data_stacks=data_stacks,
        landscan_stack=landscan_stack,
        valid_pixel_mask=valid_pixel_mask,
        valid_dates=valid_dates,
        lons=lons,
        lats=lats,
        pool=pool,
        unique_years=unique_years,
    )

    # save to yearly files
    print("outputting to files...")
    h,v = tile
    year_start_idx = 0
    for i_year, year in enumerate(tqdm(unique_years)):
        # get slice of entries corresponding to this year
        year_mask = (valid_dates.year == year)
        year_end_idx = year_mask.size - np.argmax(year_mask[::-1]) - 1 # gets index of last True value
        year_slice = slice(year_start_idx, year_end_idx + 1)

        output_path = DATA_DIR / f"NASA_Nightlight_normalized/nightlight_normalized_{year}_h{h}v{v}.nc"
        with netCDF4.Dataset(output_path, "w") as nc:
            nc.creation_utc = datetime.datetime.now().isoformat()
            nc.author = "Julian Rice julian.rice@pnnl.gov"
            nc.data_year = year
            nc.data_nasa_tile = f"h{h}v{v}"
            
            nc.createDimension("date", len(valid_dates[year_slice]))
            nc.createDimension("lat", len(lats))
            nc.createDimension("lon", len(lons))
    
            nc.createVariable("lat", lats.dtype, ("lat",))[:] = lats
            nc.createVariable("lon", lons.dtype, ("lon",))[:] = lons
            nc.createVariable("days_since_2000", "int32", ("date",))[:] = (valid_dates[year_slice] - pd.Timestamp("2000-01-01")).days

            CHUNK_KWARGS = dict(
                chunksizes=(1, len(lats), len(lons))
            )
            
            # save primary data
            var = nc.createVariable(
                "normalized_frac", 
                "float32",
                ("date", "lat", "lon"),
                least_significant_digit=3, # preserve accuracy to 0.001 (0.1%)
                **COMPRESSION_KWARGS,
                **CHUNK_KWARGS,
            )
            var[:] = normalized["normalized_frac"][year_slice]
            var.description = "Fraction of nightlight observed relative to expected, given this observation's satellite viewing angle, day of week, and seasonality. For example, 0.7 means a 30% reduction from expected."
        
            var = nc.createVariable(
                "normalized_zscore", 
                "float32",
                ("date", "lat", "lon"),
                least_significant_digit=2, # preserve accuracy to 0.01
                **COMPRESSION_KWARGS,
                **CHUNK_KWARGS,
            )
            var[:] = normalized["normalized_zscore"][year_slice]
            var.description = "Z-score of this normalized anomaly, computed robustly from the median absolute deviation method. Negative numbers are potentially indicative of outage."
    
            # save supplementary pixelwise metrics
            for name,values in metrics.items():
                nc.createVariable(
                    name, 
                    values.dtype,
                    ("lat", "lon"),
                    **COMPRESSION_KWARGS,
                )[:] = values
                var.note = "This metric is computed for the entire data range, not just this data year"
            # annual stats
            for name,values in annual_stats.items():
                nc.createVariable(
                    "annual_"+name, 
                    values.dtype,
                    ("lat", "lon"),
                    **COMPRESSION_KWARGS,
                )[:] = values[i_year] # select this year's data only

        # update start idx for next year
        year_start_idx = year_end_idx + 1




def load_normalized_nightlight(dates, lonlats=None, tiles=None, variables=["normalized_frac"], pixelwise_variables=[]):
    """
    args:
        dates: pd.DatetimeIndex
        lonlats: (xs, ys) each np.array of locations of interest
    returns:
        out_vars: dict(variable_name: np.array shape (date,lat,lon) of data)
        (lons, lats): pixel center coordinates
    """
    from src.download_nasa_nightlight import lonlat_to_nightlight_tiles, make_nightlight_grid_for_tiles

    assert (lonlats is None) != (tiles is None), "must specify either lonlats or tiles, but not both"

    # all the same year
    year = dates.year[0]
    assert (dates.year == year).all()
    # convert to days since 2000
    requested_days_since_2000 = (dates - pd.Timestamp("2000-01-01")).days
    
    if tiles is None:
        tiles = lonlat_to_nightlight_tiles(*lonlats)
    lons, lats, extents = make_nightlight_grid_for_tiles(tiles)
    # downsample 2x. get pixel centers
    lons = lons[1::2]
    lats = lats[1::2]
    extents = {
        k: (
            slice(x_ext.start//2, x_ext.stop//2, x_ext.step), 
            slice(y_ext.start//2, y_ext.stop//2, y_ext.step), 
        )
        for k, (x_ext,y_ext) in extents.items()
    }
    
    out_vars = {
        name: np.full(
            shape=(len(dates), len(lats), len(lons)),
            dtype=np.float32,
            fill_value=np.nan,
        )
        for name in variables
    }
    if len(pixelwise_variables):
        out_vars.update({
            name: np.full(
                shape=(len(lats), len(lons)), # pixelwise metrics with no time component
                dtype=np.float32,
                fill_value=np.nan,
            )
            for name in pixelwise_variables
        })
    # for each tile
    for tile in tiles:
        try:
            h,v = tile
            tile_extent = extents[tile]
            # open the file for this year and tile
            path = DATA_DIR / f"NASA_Nightlight_normalized/nightlight_normalized_{year}_h{h}v{v}.nc"
            with netCDF4.Dataset(path) as nc:
                data_days = nc["days_since_2000"][:]
                for i,requested_day in enumerate(requested_days_since_2000):
                    # if day exists in data, load it, otherwise skip and keep all-nan
                    mask = (data_days == requested_day)
                    if mask.any():
                        date_idx = np.argmax(mask)
                        # load all the variables
                        for name in variables:
                            out_vars[name][(i,) + tile_extent] = nc[name][date_idx, :, :]
                # pixelwise vars independent of date
                for name in pixelwise_variables:
                    out_vars[name][tile_extent] = nc[name][:, :]
        except:
            print(f"Error: {year} {tile}")
            print(dates)
            raise 

    return out_vars, (lons, lats)
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2012-01-19")
    parser.add_argument("--end", default="2024-12-31", help="inclusive")
    parser.add_argument("--min-samples", default=365, type=int, help="minimum number of nightlight observations in order to compute regression analysis")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    parser.add_argument("--tile", nargs=2, type=int, help="if not specified, run all US CONUS tiles. format: --tile H V")
    parser.add_argument("--isglobal", action="store_true", help="run in global mode")
    parser.add_argument("--workers", default=64, type=int)
    ARGS = parser.parse_args()

    if ARGS.tile is None:
        # get tiles to process
        centroids = get_county_geometries().geometry.centroid
        tiles = sorted(lonlat_to_nightlight_tiles(centroids.x, centroids.y))
    else:
        tiles = [tuple(ARGS.tile)]
    print("tiles:", tiles)
    
    os.makedirs(DATA_DIR / "NASA_Nightlight_normalized/", exist_ok=True)

    date_range = pd.date_range(ARGS.start, ARGS.end)
    years = np.arange(date_range[0].year, date_range[-1].year + 1)
    
    # spinup the pool. Defaults to all available processes
    with multiprocessing.Pool(ARGS.workers) as pool: 
        print(f"pool with {pool._processes} processes")

        # process each tile
        for tile in tiles:
            h,v = tile
            # check all years are completed, otherwise overwrite
            matches = glob.glob(str(DATA_DIR / f"NASA_Nightlight_normalized/nightlight_normalized_????_h{h}v{v}.nc"))
            if (len(matches) == len(years)) and (not ARGS.overwrite):
                print(tile, "already complete, not overwriting")
            else:
                process_tile(
                    tile, 
                    date_range=date_range,
                    min_samples=ARGS.min_samples,
                    pool=pool,
                    isglobal=ARGS.isglobal,
                )

    os.system(f"chmod -R 755 {DATA_DIR / 'NASA_Nightlight_normalized/'}")


if __name__ == "__main__":
    main()