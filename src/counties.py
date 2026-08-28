import os, json
import requests
import functools
import warnings

import pandas as pd
import numpy as np
import geopandas as gpd

from src.base import DATA_DIR

@functools.cache
def get_county_geometries():
    """
    US county geometries by FIPS
    returns:
        GeoDataFrame
    """
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    savepath = str(DATA_DIR / "us_counties_by_fips_plotly.geojson")
    if not os.path.exists(savepath):
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        with open(savepath, "w") as f:
            json.dump(data, f)
    df = gpd.read_file(savepath)
    df.columns = [x.lower().strip() for x in df.columns]
    # df = clean_columns(df)
    # we don't use these (state and county are id numbers, not strings)
    del df["state"], df["county"], df["geo_id"]
    df = df.rename(columns={"id": "fips_code"})
    df["fips_code"] = df["fips_code"].astype(int)

    # throws warning about the centroid results being imperfect, but that's ok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centroids = df.geometry.centroid

    df["lon_approx"] = centroids.x
    df["lat_approx"] = centroids.y

    # local solar time UTC offset
    utc_offset_solar = df.lon_approx / 360 * 24 # hours
    utc_offset_solar = utc_offset_solar * 60 * 60 # seconds
    utc_offset_solar = utc_offset_solar.astype("timedelta64[s]")
    df["utc_offset_solar"] = utc_offset_solar

    # filter to CONUS + PR
    df = df[(
        (df.lon_approx > -125)
        &
        (df.lon_approx < -65)
        &
        (df.lat_approx > 15)
        &
        (df.lat_approx < 50)
    )]

    # modeled customers
    path = str(DATA_DIR / "eaglei-data/all_outages_new/database_2024/MCC.csv")
    mcc_df = pd.read_csv(path)[:-1].astype(int)
    mcc_df = mcc_df.rename(columns={"County_FIPS": "fips_code", "Customers": "modeled_total_customers"})
    df = df.merge(mcc_df, on="fips_code", how="left")

    return df