import functools
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from tqdm import tqdm
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays

from src.base import DATA_DIR, OUTPUT_DIR
from src.landscan import load_landscan


GHS_BUILDING_HEIGHT_DIR = DATA_DIR / "GHS" / "average_gross_building_height"
STEM = "GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_4326_3ss_V1_0"


def _cache_path():
    return OUTPUT_DIR / "AGBH" / f"ghs_ag_building_height_2018_landscan_grid.xarray.nc"


def _ghs_tile_paths():
    paths = sorted(GHS_BUILDING_HEIGHT_DIR.glob(f"{STEM}*/{STEM}*.tif"))
    if not paths:
        raise FileNotFoundError(f"No GHS building height GeoTIFFs found under {GHS_BUILDING_HEIGHT_DIR}")
    return paths


def _open_ghs_tile(path):
    tile = rioxarray.open_rasterio(path).squeeze(drop=True)
    nodata = tile.rio.nodata
    if nodata is not None:
        tile = tile.where(tile != nodata, np.nan)
    tile = tile.astype(np.float32)
    # convert from 3-arcsec to 30-arcsec
    tile = tile.coarsen(x=10, y=10).mean()
    return tile

def _load_ghs_building_height_mosaic():
    """
    Load and mosaic all GHS average gross building height tiles available under
    DATA_DIR / GHS / average_gross_building_height.

    Returns:
        xarray.DataArray with projected x/y coordinates in the source CRS.
    """
    tiles = [_open_ghs_tile(path) for path in tqdm(_ghs_tile_paths(), desc="Loading GHS tiles")]
    print("merging tiles...")
    mosaic = merge_arrays(tiles)
    print("mosaic created")
    return mosaic.astype(np.float32).rename("building_height")


def _reproject_to_landscan_grid(building_height):
    print(f"Reprojecting building height to match LandScan grid...")
    target = load_landscan(2024)
    reprojected = building_height.rio.reproject_match(
        target,
        resampling=Resampling.bilinear, # averaging was already done in _open_ghs_tile, so just interpolate here
        nodata=np.nan,
    )
    reprojected = reprojected.rename("building_height").astype(np.float32)
    reprojected = reprojected.assign_coords(x=target.x, y=target.y)
    print("reprojection complete")

    if reprojected.shape != target.shape:
        raise ValueError(
            f"Reprojected building height shape {reprojected.shape} does not match LandScan shape {target.shape}"
        )
    if not np.allclose(reprojected.x.values, target.x.values):
        raise ValueError("Reprojected building height x coordinates do not match LandScan grid")
    if not np.allclose(reprojected.y.values, target.y.values):
        raise ValueError("Reprojected building height y coordinates do not match LandScan grid")

    return reprojected.where(np.isfinite(reprojected), np.nan)


@functools.cache
def load_ghs_building_height():
    """
    Load GHS average gross building height on the exact CONUS LandScan grid.

    The first call mosaics the source Mollweide GeoTIFF tiles, reprojects them
    to match ``load_landscan``, and saves a netCDF cache.
    Future calls reuse that cache and also benefit from in-process memoization.
    Returns:
        xarray.DataArray named ``building_height`` with dims ``(y, x)`` and
        coordinates exactly matching ``load_landscan()``.
    """
    cache_path = _cache_path()
    if cache_path.exists():
        return xr.open_dataarray(cache_path).load()

    building_height = _load_ghs_building_height_mosaic()
    building_height = _reproject_to_landscan_grid(building_height)

    print(f"Caching reprojected building height to {cache_path}...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    building_height.to_netcdf(
        cache_path,
        encoding={"building_height": {"zlib": True, "complevel": 4}},
    )
    print("caching complete")
    return building_height

