import os, sys
from pathlib import Path, PurePath
import xarray as xr

# code expects these environment variables to be set, pointing to 
#  a location to load data from and store data to
DATA_DIR = Path(os.environ["HIGHRES_OUTAGE_DATA_DIR"])
OUTPUT_DIR = Path(os.environ["HIGHRES_OUTAGE_OUTPUT_DIR"])

# repository root, if ever explicitly needed by the code
REPO_ROOT = Path(__file__).absolute().parent.parent





### monkeypatch in a feature of xarray.DataTree that is planned for a future release
# https://github.com/pydata/xarray/pull/10400
def _subset(self, keys, errors="raise"):
    if isinstance(keys, str):
        keys = [keys]

    def getitem(ds):
        keys_for_ds = keys
        if errors == "ignore":
            keys_for_ds = [key for key in keys if key in ds.data_vars]

        return ds[keys_for_ds]

    return xr.map_over_datasets(getitem, self)

xr.DataTree.subset = _subset