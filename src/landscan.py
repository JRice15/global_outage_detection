import functools

import numpy as np
import xarray as xr
import rioxarray

from src.base import DATA_DIR
from src.download_nasa_nightlight import nightlight_tile_to_lonlat


@functools.cache
def load_landscan(year, conus_only=True, tile=None):
    """
    load landscan covering the CONUS for a specific year
    lon/lats are referenced to the center of the pixel, I am pretty sure
    returns:
        xarray DataArray
    """
    landscan = rioxarray.open_rasterio(DATA_DIR / f"landscan/landscan-global-{year}-assets/landscan-global-{year}.tif").squeeze()
    if conus_only:
        # crop to CONUS
        landscan = landscan.sel(x=slice(-130, -60), y=slice(50, 10)).compute()
    # flip y-axis to be ascending
    landscan = landscan.reindex(y=landscan.y[::-1])
    landscan = landscan.where(landscan >= 0, np.nan)

    if tile is not None:
        h,v = tile
        N,S,E,W = nightlight_tile_to_lonlat(h,v)
        landscan = landscan.sel(
            x=slice(W, E),
            y=slice(S, N),
        ).compute()
        
    return landscan

