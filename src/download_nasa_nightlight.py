import os
import io
import time
import functools

import numpy as np
import h5py
import netCDF4
import requests

from src.base import DATA_DIR, REPO_ROOT

# NASA LAADS data store parameters and authentication
BASE_URL = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/"

# expected metadata of tiles downloaded from the source
TILE_SIZE_DEG = 10 # degrees
TILE_SHAPE = (2400, 2400)
TILE_DTYPE = "float32"
FILL_VALUE = -999.9

# Performance parameters
COMPRESSION_KWARGS = dict(
    compression="zlib",
    complevel=4, # 4 is default
)

# Tiles we may be interested in
# (horizontal,vertical)
CONUS_TILES = {
    (9,4), # Michigan
    (10,4), # NY
    (11,4), # Maine
    (8,5), # NW Texas / Louisiana
    (9,5), # Alabama
    (10,5), # Carolinas
    (8,6), # SW Texas
    (9,6), # Florida / W Cuba
}
CARIBBEAN_TILES = {
    (9,6), # Florida / W Cuba
    (10,6), # E Cuba / Bahamas
    (10,7), # Jamaica / Haiti
    (11,7), # Puerto Rico / DR
}


class UnavailableError(Exception):
    pass


@functools.cache
def get_session():
    """
    function which starts and caches session
    """
    key_path = REPO_ROOT / "secrets/LAADS_token.key"
    with open(str(key_path), "r") as f:
        TOKEN = f.read().strip()

    SESSION = requests.Session()
    SESSION.headers.update({
        'Authorization': 'Bearer ' + TOKEN,
    })
    return SESSION


def lonlat_to_nightlight_tiles(lon, lat):
    """
    args:
        lon,lat: number, or array of numbers
    returns:
        list(2-tuples), each tuple is a (horiz,vert) tile coordinate
    """
    lon = np.array(lon).flatten() % 360
    lat = np.array(lat).flatten()
    h = ((lon - 180) // 10).astype(int) % 36
    v = ((-lat + 90) // 10).astype(int) % 18
    tiles = list(set(zip(h,v)))
    tiles = [(int(h), int(v)) for h,v in tiles]
    return tiles

def nightlight_tile_to_lonlat(horiz, vert):
    if isinstance(horiz, np.ndarray):
        assert horiz.shape == vert.shape
    W = (horiz * 10) - 180
    E = ((horiz+1) * 10) - 180
    N = -((vert * 10) - 90)
    S = -(((vert+1) * 10) - 90)
    return N, S, E, W


def make_nightlight_grid_for_tiles(tiles):
    """
    make a grid for tiles
    args:
        tiles: list(tuple), each tuple is (h,v) ints
    returns:
        lons: 1D np.array
        lats: 1D np.array
        extents: dicts mapping tile (h,v) to 2-tuple of slices of where that tile fits on the grid
    """
    tile_height, tile_width = TILE_SHAPE
    
    tiles = np.array(tiles)
    h_min, v_min = tiles.min(axis=0)
    h_max, v_max = tiles.max(axis=0)
    n_tiles_vert = v_max - v_min + 1
    n_tiles_horiz = h_max - h_min + 1

    N, S, E, W = nightlight_tile_to_lonlat(tiles[:,0], tiles[:,1])
    N = N.max()
    S = S.min()
    E = E.max()
    W = W.min()

    x = np.linspace(W, E, num=n_tiles_horiz*tile_width, endpoint=False)
    y = np.linspace(S, N, num=n_tiles_vert*tile_height, endpoint=False)

    extents = {}
    for h in range(h_min, h_max + 1):
        for v in range(v_min, v_max + 1):
            h_ind = h - h_min
            v_ind = v_max - v
            extents[(h,v)] = (
                slice(v_ind*tile_height, (v_ind+1)*tile_height),
                slice(h_ind*tile_width,  (h_ind+1)*tile_width),
            )

    return x, y, extents



def _get_reponse(url, _cache={"last_request_time": time.perf_counter()}):
    """
    get http response from a url, with proper waiting and sleeping to not get rate limited
    """
    print("GET", url)
    wait_time = 1.1 # seconds
    if time.perf_counter() - _cache["last_request_time"] < wait_time:
        time.sleep(wait_time)
    SESSION = get_session()
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
    except requests.exceptions.ReadTimeout:
        print("  timeout, trying again once in 20 seconds...")
        time.sleep(20) # wait a few seconds
        r = SESSION.get(url, timeout=60)
        r.raise_for_status()
    _cache["last_request_time"] = time.perf_counter()
    return r

def _create_or_get_nc_var(nc, name, *args, **kwargs):
    try:
        return nc.createVariable(name, *args, **kwargs)
    except:
        return nc[name]

def _download_nightlight_tile(f_out, a1_url, a2_url, tile, year, day):
    """
    download and process and single nightlight tile
    args:
        f_out: file pointer to open output file
        a1/a2_url: url where h5 tile data is stored
        tile: VIIRS Black Marble tile coordinates (horiz, vert)
    """
    h, v = tile
    tile_name = f"h{h}v{v}"
    # if tile_name in f_out.groups.keys():
        # del f_out[tile_name]

    a1_content = _get_reponse(a1_url).content
    a2_content = _get_reponse(a2_url).content
    
    # create in-memory netcdf4 object from the raw binary content (which is an h5 file)
    with netCDF4.Dataset("a1", mode="r", memory=a1_content) as f_a1, netCDF4.Dataset("a2", mode="r", memory=a2_content) as f_a2:
        tilegroup = f_out.createGroup(tile_name)
        tilegroup.done = 0

        # A1
        zenith = f_a1["/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Sensor_Zenith"][:]
        azimuth = f_a1["/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Sensor_Azimuth"][:]
        moon_illumination = f_a1["/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Moon_Illumination_Fraction"][:]
        moon_phase = f_a1["/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Moon_Phase_Angle"][:]
        lunar_zenith = f_a1["/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Lunar_Zenith"][:]
        lunar_azimuth = f_a1["/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Lunar_Azimuth"][:]
        utc_offset = f_a1["/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/UTC_Time"][:]
        # A2
        nightlight_ds = f_a2["/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_BRDF-Corrected_NTL"]
        quality_flag = f_a2["/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Mandatory_Quality_Flag"][:]
        qf_cloud_mask = f_a2["/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/QF_Cloud_Mask"][:]

        # make sure the tile is how we expect it
        assert nightlight_ds.shape == TILE_SHAPE
        assert nightlight_ds.dtype == TILE_DTYPE
        assert np.isclose(nightlight_ds._FillValue, FILL_VALUE, atol=0.01)

        # read the data
        nightlight = nightlight_ds[:]
        # quality flag has 0 for high quality, and 1-5 are low quality (or not set)
        quality_mask = (quality_flag == 0)
        # set low quality values to missing
        set_nan_mask = nightlight.mask | (~quality_mask)
        nightlight[set_nan_mask] = np.nan
        # remove mask as all masked values are filled with NaN now
        nightlight = nightlight.data

        # handle angular data
        # remove the mostly-unused mask
        zenith[zenith.mask] = np.nan
        zenith = zenith.data
        azimuth[azimuth.mask] = np.nan
        azimuth = azimuth.data
        moon_illumination[moon_illumination.mask] = np.nan
        moon_illumination = moon_illumination.data
        moon_phase[moon_phase.mask] = np.nan
        moon_phase = moon_phase.data
        lunar_zenith[lunar_zenith.mask] = np.nan
        lunar_zenith = lunar_zenith.data
        lunar_azimuth[lunar_azimuth.mask] = np.nan
        lunar_azimuth = lunar_azimuth.data
        utc_offset[utc_offset.mask] = np.nan
        utc_offset = utc_offset.data
        # qf_cloud_mask is int so can't take NaN

        # NASA oriented these grids upside-down compared to the way we do it
        nightlight = nightlight[::-1]
        zenith = zenith[::-1]
        azimuth = azimuth[::-1]
        moon_illumination = moon_illumination[::-1]
        moon_phase = moon_phase[::-1]
        lunar_zenith = lunar_zenith[::-1]
        lunar_azimuth = lunar_azimuth[::-1]
        utc_offset = utc_offset[::-1]
        qf_cloud_mask = qf_cloud_mask[::-1]

        _create_or_get_nc_var(tilegroup,
            "DNB_BRDF-Corrected_NTL--high_quality_only", 
            TILE_DTYPE,
            ("lat", "lon"),
            least_significant_digit=1, # preserve accuracy to 0.1
            **COMPRESSION_KWARGS,
        )[:] = nightlight
        _create_or_get_nc_var(tilegroup,
            "Sensor_Zenith", 
            TILE_DTYPE,
            ("lat", "lon"),
            least_significant_digit=1, # preserve accuracy to 0.1 degrees
            **COMPRESSION_KWARGS,
        )[:] = zenith
        _create_or_get_nc_var(tilegroup,
            "Sensor_Azimuth", 
            TILE_DTYPE,
            ("lat", "lon"),
            least_significant_digit=1, # preserve accuracy to 0.1 degrees
            **COMPRESSION_KWARGS,
        )[:] = azimuth
        _create_or_get_nc_var(tilegroup,
            "Moon_Illumination_Fraction", 
            TILE_DTYPE,
            ("lat", "lon"),
            least_significant_digit=1, # preserve accuracy to 0.1. Data is in range [0-100]
            **COMPRESSION_KWARGS,
        )[:] = moon_illumination
        _create_or_get_nc_var(tilegroup,
            "Moon_Phase_Angle", 
            TILE_DTYPE,
            ("lat", "lon"),
            least_significant_digit=1, # preserve accuracy to 0.1 degrees
            **COMPRESSION_KWARGS,
        )[:] = moon_phase
        _create_or_get_nc_var(tilegroup,
            "Lunar_Zenith", 
            TILE_DTYPE,
            ("lat", "lon"),
            least_significant_digit=1, # preserve accuracy to 0.1 degrees
            **COMPRESSION_KWARGS,
        )[:] = lunar_zenith
        _create_or_get_nc_var(tilegroup,
            "Lunar_Azimuth", 
            TILE_DTYPE,
            ("lat", "lon"),
            least_significant_digit=1, # preserve accuracy to 0.1 degrees
            **COMPRESSION_KWARGS,
        )[:] = lunar_azimuth
        _create_or_get_nc_var(tilegroup,
            "Viewing_UTC_Hour", 
            TILE_DTYPE,
            ("lat", "lon"),
            least_significant_digit=1, # preserve accuracy to 0.1 hours (6 minutes)
            **COMPRESSION_KWARGS,
        )[:] = utc_offset
        _create_or_get_nc_var(tilegroup,
            "QF_Cloud_Mask", 
            qf_cloud_mask.dtype,
            ("lat", "lon"),
            **COMPRESSION_KWARGS,
        )[:] = qf_cloud_mask


        # final flag to signify completion
        tilegroup.done = 1

def _open_for_download(path):
    """
    open file for append mode, or overwriting if it is corrupted
    """
    if os.path.exists(path):
        try:
            return netCDF4.Dataset(path, "a")
        except:
            pass
    nc = netCDF4.Dataset(path, "w")
    nc.createDimension("lat", TILE_SHAPE[0])
    nc.createDimension("lon", TILE_SHAPE[1])
    return nc


def _download_nightlight(year, day, save_path, tiles):
    """
    download nightlight imagery for a day and year, for all specified tiles, to the save_path
    """
    a1_base_url = BASE_URL + f"VNP46A1/{year}/{day:03}"
    a2_base_url = BASE_URL + f"VNP46A2/{year}/{day:03}"

    with _open_for_download(save_path) as f_out:
        if "unavailable_tiles" in f_out.ncattrs():
            unavailable_tiles = list(f_out.unavailable_tiles)
        else:
            unavailable_tiles = []

        # get file listing from json file
        try:
            a2_files = _get_reponse(a2_base_url + ".json").json()["content"]
            a2_files = [file["name"] for file in a2_files if file["size"] > 0]
            # get associated a1 files
            a1_files = _get_reponse(a1_base_url + ".json").json()["content"]
            a1_files = [file["name"] for file in a1_files if file["size"] > 0]
            # properly align a1 files to a2 set and order
            keyfunc = lambda name: tuple(name.split(".")[1:4])
            a1_files_dict = {keyfunc(name): name for name in a1_files}
            # filter to files which have both a1 and a2 existing
            a2_files = [name for name in a2_files if keyfunc(name) in a1_files_dict.keys()]
            a1_files = [a1_files_dict[keyfunc(name)] for name in a2_files]

        except requests.exceptions.HTTPError as e:
            # 404 means the entire day is missing. mark as unavailable
            if e.response.status_code == 404:
                if "ALL" not in unavailable_tiles:
                    unavailable_tiles.append("ALL")
                    f_out.unavailable_tiles = unavailable_tiles
                return
            # otherwise, raise
            else:
                raise

        tiles_completed = []
        # find and download relevant tiles
        for a1_fname, a2_fname in zip(a1_files, a2_files):
            # get hXXvXX location from name string
            loc = a2_fname.split(".")[2]
            h = int(loc[1:3])
            v = int(loc[-2:])
            if (h,v) in tiles:
                tile_name = f"h{h}v{v}"
                # download data if it doesn't already exist
                try:
                    assert f_out[tile_name].done
                except:
                    _download_nightlight_tile(
                        f_out=f_out, 
                        a1_url=a1_base_url + "/" + a1_fname, 
                        a2_url=a2_base_url + "/" + a2_fname, 
                        tile=(h,v),
                        year=year,
                        day=day,
                    )
                tiles_completed.append(tile_name)
    
        for (h,v) in tiles:
            tile_name = f"h{h}v{v}"
            if tile_name not in tiles_completed and tile_name not in unavailable_tiles:
                unavailable_tiles.append(tile_name)
        f_out.unavailable_tiles = unavailable_tiles


def _filter_tiles_available(f, tiles, allow_missing=False):
    try:
        unavailable = list(f.unavailable_tiles)
    except:
        unavailable = []

    if "ALL" in unavailable:
        available = []
    else:
        available = [(h,v) for (h,v) in tiles if f"h{h}v{v}" not in unavailable]

    if len(available) != len(tiles) and (not allow_missing):
        raise UnavailableError(str(unavailable))
    return available


def _read_saved_nightlight(path, tiles, return_variables, return_as="stitched", allow_missing=False):
    all_results = []
    with netCDF4.Dataset(str(path), "r") as f:
        available_tiles = _filter_tiles_available(f, tiles, allow_missing=allow_missing)
        if not len(available_tiles):
            return None

        if return_as == "stitched":
            lons, lats, extents = make_nightlight_grid_for_tiles(available_tiles)
        if return_as == "stitched-full":
            lons, lats, extents = make_nightlight_grid_for_tiles(tiles)

        for varname in return_variables:
            if return_as == "tiles":
                result = {}
            elif return_as.startswith("stitched"):
                result = None # initialized later with proper dtype
            else:
                raise ValueError(f"unknown return kind '{return_as}'")
            
            # load each tile
            for h,v in available_tiles:
                dataset = f[f"h{h}v{v}/{varname}"]
                data = dataset[:]
                if return_as == "tiles":
                    # add to output dict
                    result[(h,v)] = data
                elif return_as.startswith("stitched"):
                    if len(tiles) == 1:
                        result = data
                    else:
                        # add tile to correct location in output grid
                        slices = extents[(h,v)]
                        if result is None:
                            if data.dtype == np.uint16:
                                fill_value = 255
                            else:
                                fill_value = np.nan
                            result = np.full((len(lats), len(lons)), fill_value=fill_value, dtype=data.dtype)
                        result[slices] = data
                else:
                    raise AssertionError(f"return_as={return_as} not handled")
                
            # save to full output list
            all_results.append(result)

    if len(return_variables) == 1:
        # unpack if only one variable
        all_results = all_results[0]
    if return_as.startswith("stitched"):
        return all_results, (lons, lats)
    return all_results


def _get_tiles(region=None, lonlat=None):
    """
    get relevant tiles
    """
    if lonlat is not None:
        lon, lat = lonlat
        tiles = lonlat_to_nightlight_tiles(lon, lat)
    elif region is not None:
        region = region.lower()
        if region == "conus":
            tiles = CONUS_TILES
        elif region == "caribbean":
            tiles = CARIBBEAN_TILES
        elif region == "all":
            tiles = CONUS_TILES.union(CARIBBEAN_TILES)
        else:
            raise ValueError(f'Unknown region {region}')
    else:
        raise ValueError("argument required")
    
    return tiles


def get_nightlight(year, day, region=None, lonlat=None, tiles=None, return_as="stitched-full", 
                   return_variables=("DNB_BRDF-Corrected_NTL--high_quality_only",), 
                   allow_missing=False, raise_on_error=True):
    '''
    reads or downloads nightlight imagery for a particular day
    
    satellite passes are done between 01:30am and 03:30am local time according to https://www.mdpi.com/2072-4292/9/3/286
    
    this data is defined on a different grid than GLOBALS.LON/LAT_GRID. See aggregated_interpolate_2d(mode="avg") 
    for recommended conversion
    
    args:
        year: int
        day: int, 1-indexed day of the year (1-365/366)
        one of region or lonlat is required:
        region: one of "CONUS", "Caribbean", "All" (case insensitive)
        lonlat: tuple(lons, lats), a set of coordinates that cover the region of interest and
                determine which tiles to download
        tiles: list of 2-tuples indicating tile coordinates in NASA grid
        return_as: str
            "tiles": returns dict((h,v): data)
            "stitched": returns data, (lons, lats) grid
            "stitched-full": same, but returns grid including space for missing tiles
        allow_missing: return partial data if some tiles are unavailable
        raise_on_error: raise if some other error occurs
    returns:
        if return_as=="stitched":
            data: 2-D np.array
            grid: tuple of (lons,lats)
        if return_as=="tiles":
            dict mapping tile coordinate tuples (h,v) to 2d np.arrays
        if no tiles are available:
            None
    raises:
        ValueError if date is before 2012 day 19 (earliest date that imagery is available)
        UnavailableError if not all requested tiles are available, and allow_missing is False
    '''
    if (year < 2012) or (year == 2012 and day < 19):
        raise ValueError(f"Nightlight imagery only available from 2012 day 19 onward. You requested {year} {day}")
    
    assert year % 1 == 0
    assert day % 1 == 0
    assert day > 0
    year = int(year)
    day = int(day)

    # create save path
    outdir = DATA_DIR / "NASA_Nightlight"
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    outfile = outdir.joinpath(f"NASA_VNP46_{year}_{day:03}.nc")

    # get tiles to download
    if tiles is not None:
        assert region is None
        assert lonlat is None
    else:
        tiles = _get_tiles(region=region, lonlat=lonlat)

    for h,v in tiles:
        assert 0 <= h <= 35
        assert 0 <= v <= 17
    
    read_kwargs = dict(
        return_variables=return_variables,
        return_as=return_as,
        allow_missing=allow_missing,
    )

    try:
        # load data, or download and load if that fails
        try:
            return _read_saved_nightlight(outfile, tiles, **read_kwargs)
        except UnavailableError as e:
            raise
        except Exception as e:
            print(f"Could not read nightlight:\n  {type(e)}: {e}\n  {outfile}\n  Downloading now...")

        print("  tiles:", tiles)
        _download_nightlight(year, day, outfile, tiles)
        return _read_saved_nightlight(outfile, tiles, **read_kwargs)

    except Exception as e:
        print(f"Error in get_nightlight({year} {day}, ...): {type(e)}: {e}")
        if raise_on_error:
            raise


