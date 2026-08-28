import re

import pandas as pd
import numpy as np
import ephem

def _set_all(obj, **kwargs):
    for k,v in kwargs.items():
        setattr(obj, k, v)
    return obj

def get_moon_state(dt, lon, lat):
    """
    args:
        dt: UTC datetime
        lon: longitude in degrees
        lat: latitude in degrees
    """
    assert not pd.isna(dt), f"NaN in get_moon_state: {dt}"
    obs = _set_all(
        ephem.Observer(),
        date=re.sub(r"T", " ", str(dt)),
        lat=str(lat),
        lon=str(lon),
    )
    moon = ephem.Moon(obs)
    return moon


def get_moon_states(datetimes, lon, lat, variables=("mag", "az", "alt")):
    """
    get moon states DataFrame for a UTC timeseries at a specific location
    args:
        datetimes: list of UTC datetimes
        lon: longitude in degrees
        lat: latitude in degrees
        variables: one of: mag, moon_phase, az, alt, ra, dec. If "mag" selected, brightness is automatically 
            computed as well
    """
    moons = [get_moon_state(t, lon=lon, lat=lat) for t in datetimes]
    moons = [[getattr(m, name) for name in variables] for m in moons]
    moons = pd.DataFrame(moons, columns=variables)
    moons["datetime"] = datetimes
    if "mag" in variables:
        # magnitude is log-scaled with negative being brighter. Actual brightness is proportional to exp(-mag)
        # unitless
        moons["brightness"] = np.exp(-moons.mag) 
    return moons


def compute_moon_brightness(illum_frac):
    """
    approximate moon magnitude from illumination fraction
    does not account for (small) variations in earth-moon and moon-sun distance
    code shamelessly adapted from PyEphem

    args:
        illum_frac: moon illumination fraction in [0,1]
    returns:
        brightness: unitless, proportional to actual brightness of the moon
    """
    pang = np.rad2deg(np.arccos((illum_frac * 2) - 1.0))

    # for angles <= 40
    i_le40 = -12.72 + 0.0267 * (pang - 20) + 0.534
    # for angles > 40
    p = pang - 80.0
    i_gt40 = -12.72 + p * (0.03188 + p * (1.9621e-4 + p * 1.7256e-6)) + 2.14
    # select which to use based on angle
    i = np.where(pang <= 40, i_le40, i_gt40)
        
    # i is magntitude. convert to my unitless "brightness" by negating and exponentiating
    brightness = np.exp(-i)
    return brightness

