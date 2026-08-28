import functools
import os, json
import warnings
import math
import argparse
import datetime
import multiprocessing
import pickle
from pprint import pprint

import pandas as pd
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics
import xarray as xr
from tqdm import tqdm

# prevents OOM for some reason
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base import OUTPUT_DIR, DATA_DIR
from src.moon import compute_moon_brightness

# metadata variables required during loss computation
META_COLS = [
    "pixel_customers_scaled",
    "valid_pixel_mask",
]


# small constant to avoid log(0) issues, chosen to be near the smallest valid value
# variables not included in this dict will not have a log-scaled version available to them
LOG_LOWERBOUNDS = {
    # 'annual_count': ('pixels',),
    'annual_mean': 0.29,
    'annual_median': 0.29,
    'annual_median_dev': 0.01,
    'annual_std': 0.01,
    'building_height': 1e-5,
    # 'cloud_mask_quality': ('date', 'pixels'),
    # 'clouds': ('date', 'pixels'),
    # 'coverage_weight': ('pixels',),
    'customers_out_atviewtime': 0.1,
    'customers_out_atviewtime_max': 0.1,
    'customers_out_atviewtime_min': 0.1,
    'customers_out_nightly_max': 0.1,
    'customers_out_nightly_mean': 0.1,
    'customers_out_nightly_min': 0.1,
    # 'date': ('date',),
    # 'fips_code': (),
    'landscan': 0.1,
    # 'landscan_to_eaglei_ratio': 0.05,
    # 'lat_countycentroid': (),
    # 'lon_countycentroid': (),
    # 'lunar_azimuth': ('date', 'pixels'),
    # 'lunar_zenith': ('date', 'pixels'),
    # 'modeled_total_customers': (),
    # 'moon_illum_frac': ('date', 'pixels'),
    # 'nightlight': ('date', 'pixels'),
    'normalized_frac': 1e-4,
    # 'normalized_zscore': ('date', 'pixels'),
    # 'pop_coverage_frac': ('date',),
    # 'sensor_azimuth': ('date', 'pixels'),
    # 'sensor_zenith': ('date', 'pixels'),
    # 'snow': ('date', 'pixels'),
    # 'viewhour': ('date', 'pixels'),
    # 'year': ()
}


def print_nans(raiseit=False, **kwargs):
    """test tensors for NaN"""
    nans_found = False
    for name,arr in kwargs.items():
        if arr.isnan().any():
            nans_found = True
            if arr.ndim == 3:
                nans = arr.isnan().any(dim=0).any(dim=0)
            elif arr.ndim == 1:
                nans = arr.isnan()
            else:
                nans = arr.isnan().any()
            print("NaNs!")
            print(name)
            print("nans", nans)
            print("count", arr.isnan().sum())
    if nans_found and raiseit:
        raise ValueError("NaN encountered")


def _assert_radians_or_nan(x):
    assert(
        np.isnan(x) | ((-2*np.pi <= x) & (x <= 2*np.pi))
    ).all(), x

def angle_between_zen_az_points(z1, a1, z2, a2):
    """
    args: zeniths and aziumths in radians
    returns: great circle angle in radians
    """
    _assert_radians_or_nan(z1)
    _assert_radians_or_nan(a1)
    _assert_radians_or_nan(z2)
    _assert_radians_or_nan(a2)
    
    cos_angle = (
        (np.cos(z1) * np.cos(z2))
        +
        (np.sin(z1) * np.sin(z2) * np.cos(a1 - a2))
    )
    return np.arccos(cos_angle)



###
### Load dataset
###

def load_tree(year, min_pop_cov_frac):
    tree = xr.open_datatree(
        OUTPUT_DIR / f"aligned_nightlight_outage_zarr/aligned_nightlight_outages_{year}.xarray.zarr"
    ).load()

    # process
    results = []
    for node in tree.leaves:
        if node.fips_code.item() >= 60_000:
            # exclude Puerto Rico for now
            continue

        # get dataset view
        ds = node.to_dataset()
        
        # exclude days with insufficient nightlight coverage
        date_meets_pop_cov_mask = (ds.pop_coverage_frac >= min_pop_cov_frac)
        if date_meets_pop_cov_mask.sum() == 0:
            continue
        ds = ds.sel(date=date_meets_pop_cov_mask)

        # invalid pixels, shape (date, pixels) so we can't just mask them away entirely here
        ds["valid_pixel_mask"] = (
            (ds.landscan > 0)
            &
            (~np.isnan(ds.normalized_zscore))
        )
        
        # fill landscan nans
        ds["landscan"] = ds.landscan.fillna(0)
        # assume <0.3 are spurious, and fill missing with mean value ~0.55
        ds["landscan_to_eaglei_ratio"] = (
            ds.landscan_to_eaglei_ratio.where(
                (ds.landscan_to_eaglei_ratio >= 0.3),
                0.55,
            )
        )
        # limit high end of normalized values
        ds["normalized_frac"] = ds["normalized_frac"].clip(None, 3)
        ds["normalized_zscore"] = ds["normalized_zscore"].clip(None, 5)

        # log-scale certain variables
        log_ds_list = [
            np.log10(ds[key].clip(lowerbound, None)).rename("log_"+key)
            for key, lowerbound in LOG_LOWERBOUNDS.items()
        ]
        ds = xr.merge([ds] + log_ds_list)

        # clouds
        ds["clear_confidence"] = (1 - ds.clouds) * ds.cloud_mask_quality

        # transformed moon
        ds["moon_brightness"] = (tuple(ds.moon_illum_frac.dims), compute_moon_brightness(ds.moon_illum_frac))

        # date/seasonal cycle factors
        ds["day_of_year"] = ds["date"].dt.dayofyear
        ds["day_of_year_sin"] = np.sin(2 * np.pi * ds["day_of_year"] / 365)
        ds["day_of_year_cos"] = np.cos(2 * np.pi * ds["day_of_year"] / 365)
        ds["day_of_week"] = ds["date"].dt.dayofweek
        ds["day_of_week_sin"] = np.sin(2 * np.pi * ds["day_of_week"] / 7)
        ds["day_of_week_cos"] = np.cos(2 * np.pi * ds["day_of_week"] / 7)
        ds["days_since_2000"] = (ds["date"] - np.datetime64("2000-01-01")).dt.days

        # angles
        ds["sensor_azimuth_sin"] = np.sin(np.deg2rad(ds["sensor_azimuth"]))
        ds["sensor_azimuth_cos"] = np.cos(np.deg2rad(ds["sensor_azimuth"]))
        ds["lunar_azimuth_sin"] = np.sin(np.deg2rad(ds["lunar_azimuth"]))
        ds["lunar_azimuth_cos"] = np.cos(np.deg2rad(ds["lunar_azimuth"]))
        ds["moon_sensor_angle"] = np.rad2deg(angle_between_zen_az_points(
            np.deg2rad(ds["sensor_zenith"]),
            np.deg2rad(ds["sensor_azimuth"]),
            np.deg2rad(ds["lunar_zenith"]),
            np.deg2rad(ds["lunar_azimuth"]),
        ))

        # pop to customers conversion
        ds["pixel_customers_scaled"] = (
            ds.landscan 
            * ds.landscan_to_eaglei_ratio 
            * ds.coverage_weight 
            / ds.pop_coverage_frac
        )
        
        # test if prediction can meet target
        max_prediction = ds.pixel_customers_scaled.sum("pixels")
        bad_ratio = (ds.customers_out_atviewtime / max_prediction).values
        badmask = bad_ratio > 1.2
        # if it would not be possible to meet the observed outage count, note this fact
        if badmask.any():
            print("WARNING: observed customers_out_atviewtime exceeds max possible predicted customers_out")
            print(f"  {node.fips_code.item()} {ds.date.values}")
            print("  badratio", bad_ratio[badmask])
            print("  cust out", ds.customers_out_atviewtime.values[badmask])
            print("  max pred", max_prediction.values[badmask])
            print("  mtc     ", ds.modeled_total_customers.item())
            print("  ls2e rat", ds.landscan_to_eaglei_ratio.mean().item())

        results.append(ds)

    return results



class OutageDataset(torch.utils.data.Dataset):

    def __init__(self, min_pop_cov_frac):
        self.min_pop_cov_frac = min_pop_cov_frac

        years = range(2014, 2025)
        print("loading & processing datatree")
        self.datasets = []
        loader_f = functools.partial(load_tree, min_pop_cov_frac=min_pop_cov_frac)
        with multiprocessing.Pool(len(years)+1) as pool:
            iterator = pool.imap(loader_f, years)
            for result_list in tqdm(iterator, total=len(years)):
                self.datasets += result_list

        # norm data
        self.maxs = xr.concat(
            [ds.max() for ds in tqdm(self.datasets, desc="maxs")], 
            dim="node"
        ).max()
        
        self.mins = xr.concat(
            [ds.min() for ds in tqdm(self.datasets, desc="mins")], 
            dim="node"
        ).min()

    def set_predictors_target(self, predictors, target):
        self.predictors = predictors
        self.target = target

    def set_target_lowerbound(self, lowerbound):
        if self.target.startswith("customers_out"):
            self._lowerbound = lowerbound
        elif self.target.startswith("log_customers_out"):
            self._lowerbound = np.log10(lowerbound)
        else:
            raise ValueError(f"unknown target '{self.target}'")
    
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, index):
        ds = self.datasets[index]
        X = ds[self.predictors]
        X = (X - self.mins[self.predictors]) / (self.maxs[self.predictors] - self.mins[self.predictors])
        X = X.to_array()
        if "date" not in X.dims:
            print(X)
            raise ValueError("missing 'date' dim?")
        X = X.transpose("date", "pixels", "variable")
        X = X.fillna(0.0)

        meta = ds[META_COLS].to_array().transpose("date", "pixels", "variable")

        Y = ds[self.target]
        Y = Y.clip(self._lowerbound, None)
        ## Don't normalize Y, compute losses real space. Because preds are also multiplied 
        ##   by landscan population, I think this properly scales the gradients 
        ##   that propogate to the weights. 
        ## Y = (Y - self.mins[self.target]) / (self.maxs[self.target] - self.mins[self.target])

        attributes = {
            "fips_code": ds.fips_code.item(),
            "year": ds.year.item(),
            "date": ds.date.values,
        }

        return X.values, Y.values, meta.values, attributes


def stack_ragged_examples(X):
    """
    list of arrays each shape (dates, pixels, variables), with different n_pixels sizes, into homogeneous block
    """
    # infer shapes
    _, _, nvars = X[0].shape
    total_examples = sum(x.shape[0] for x in X)
    max_npixels = max(x.shape[1] for x in X)

    out = np.zeros((total_examples, max_npixels, nvars), dtype=X[0].dtype)
    i = 0
    for x in X:
        n_examples, n_pixels, _ = x.shape
        out[i:i+n_examples, :n_pixels, :] = x
        i += n_examples

    return torch.tensor(out, dtype=torch.float32)

def collate_custom_batch(batch):
    """
    X: list of arrays, each shape (dates, pixels, variables)
    Y: list of arrays shape (dates,)
    meta: list of arrays, shape (date, pixels, variables)
    attributes: list of dicts
    """
    X, Y, meta, attributes = zip(*batch)

    X = stack_ragged_examples(X)
    meta = stack_ragged_examples(meta)

    # each Y is shape (dates,)
    Y = np.concatenate(Y)
    Y = torch.tensor(Y, dtype=torch.float32)

    return X, Y, meta, attributes


@functools.cache
def load_cached_ds(min_pop_cov_frac, recache=False):
    """
    two levels of caching:
        store built version of dataset in pickle file (regenerated with recache=True)
        subsequent calls in the same session with cache that object in active memory
    """
    print("Loading dataset")
    ds_cache_path = OUTPUT_DIR / f"nl2outage/ds_cache_mpcf{min_pop_cov_frac}.pickle"
    if os.path.exists(ds_cache_path) and (not recache):
        print("using cached version:", ds_cache_path)
        with open(ds_cache_path, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = OutageDataset(min_pop_cov_frac=min_pop_cov_frac)
        print("saving to cache")
        with open(ds_cache_path, "wb") as f:
            pickle.dump(dataset, f)

    return dataset


def build_dataloaders(predictors, target, valyears, testyears, min_pop_cov_frac, batchsize, lowerbound, workers=2, recache=False):
    """
    load train/val/test dataloaders, with 2023 as val year and 2024 as test year

    workers=2 is optimal for single NERSC GPU
    """
    dataset = load_cached_ds(min_pop_cov_frac=min_pop_cov_frac,recache=recache)
    print("sizes and mins/maxs:")
    mins = dataset.mins.to_dict()["data_vars"]
    maxs = dataset.maxs.to_dict()["data_vars"]
    pprint({
        k: (
            tuple(dataset.datasets[0][k].dims),
            (mins[k]['data'], maxs[k]['data'])
        )
        for k in mins.keys()
    })
    
    # set the proper predictors and target variables
    dataset.set_predictors_target(predictors, target)

    # set target lower bound
    dataset.set_target_lowerbound(lowerbound)

    # split and create data loaders
    indices = np.arange(len(dataset))
    years = np.array([ds.year.item() for ds in dataset.datasets])
    print(pd.Series(years).value_counts().sort_index())

    test_mask = np.isin(years, testyears)
    val_mask = np.isin(years, valyears)
    train_indices = indices[~(test_mask | val_mask)]
    val_indices = indices[val_mask]
    test_indices = indices[test_mask]

    train_loader = torch.utils.data.DataLoader(
        dataset, 
        sampler=torch.utils.data.SubsetRandomSampler(
            train_indices, 
            generator=torch.Generator().manual_seed(0), # fixed example order across experiments
        ),
        batch_size=batchsize,
        collate_fn=collate_custom_batch,
        num_workers=workers,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, 
        sampler=torch.utils.data.SubsetRandomSampler(
            val_indices, 
            generator=torch.Generator().manual_seed(1),
        ),
        batch_size=batchsize,
        collate_fn=collate_custom_batch,
        num_workers=workers,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, 
        sampler=torch.utils.data.SubsetRandomSampler(
            test_indices, 
            generator=torch.Generator().manual_seed(2),
        ),
        batch_size=batchsize,
        collate_fn=collate_custom_batch,
        num_workers=workers,
        drop_last=False,
    )

    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("test batches:", len(test_loader))

    return train_loader, val_loader, test_loader




###
### UTILS
###

    

def county_aggregate_predictions(pred_fractions, pixel_customers_scaled, valid_pixel_mask, lowerbound):
    """
    inputs, all torch.Tensor:
        pred_fractions: (B, pixels, 1) predicted pixel level outage fractions
        pixel_customers_scaled: (B, pixels)
        valid_pixel_mask: (B, pixels)
    returns:
        predictions (B,): county-level aggregated predicted population out
    """
    # get predicted total out
    pred_fractions = pred_fractions.squeeze(dim=-1) # (B, pixels)

    # convert fractions to customers
    pred_cust_out = pred_fractions * pixel_customers_scaled # (B, pixels)

    # filter invalid pixels (landscan=0 or missing nightlight)
    # also seemlessly handles pixels added during zero-padding of input
    pred_cust_out = torch.where(valid_pixel_mask > 0, pred_cust_out, 0)
    
    # sum over county pixels
    pred_cust_out = torch.sum(pred_cust_out, dim=1) # (B,)

    print_nans(
        pred_cust_out=pred_cust_out,
        raiseit=True,
    )
    
    # clip to lowerbound to avoid log(0) issues, and ignore differences of <1 customer
    # leaky relu instead of hard clip to preserve some gradient signal for predictions that are very close to the lowerbound
    # pred_cust_out = torch.nn.functional.leaky_relu(pred_cust_out - lowerbound, negative_slope=0.1) + lowerbound
    return pred_cust_out



def compute_metrics(all_losses, all_targets, all_predictions, target_col, lowerbound):
    """
    args: lists of loss, target, prediction for each sample
    """
    all_losses = np.array(all_losses)
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    if target_col.startswith("log"):
        # lowerbound
        all_predictions = np.maximum(all_predictions, np.log10(lowerbound))
        log_targets = all_targets
        log_predictions = all_predictions
        # de-log-ify the results
        all_targets = 10 ** all_targets
        all_predictions = 10 ** all_predictions
    else:
        # lowerbound
        all_predictions = np.maximum(all_predictions, lowerbound)
        # pred and target are both clipped to lowerbound, so no log(0) issues
        log_targets = np.log10(all_targets)
        log_predictions = np.log10(all_predictions)

    loss = np.mean(all_losses)
    metrics = {
        # losses
        "loss": float(loss),
        "sqrt(loss)": float(np.sqrt(loss)),
        # correlations
        "r": float(scipy.stats.pearsonr(all_predictions, all_targets).statistic),
        "r_log": float(scipy.stats.pearsonr(log_predictions, log_targets).statistic),
        "spearman": float(scipy.stats.spearmanr(all_predictions, all_targets).statistic),
        # R-squared
        "R2": float(sklearn.metrics.r2_score(all_targets, all_predictions)),
        "R2_log": float(sklearn.metrics.r2_score(log_targets, log_predictions)),
    }
    if np.isnan(metrics["r"]):
        print("NaN r")
        print("targ all eq", np.all(all_targets == all_targets[0]), all_targets[0])
        print("pred all eq", np.all(all_predictions == all_predictions[0]), all_predictions[0])
    data = {
        "targets": all_targets,
        "predictions": all_predictions,
    }
    return metrics, data


def evaluate(model, val_loader, device, target_col, lowerbound, name="Val"):
    """
    validate/test model on a held-out set
    """
    model.eval()

    all_targets = []
    all_predictions = []
    all_losses = []
    all_fips_codes = []
    all_dates = []
    with torch.inference_mode(), torch.no_grad(), tqdm(total=len(val_loader), desc=name, ncols=120) as pbar:
        for batch in val_loader:
            X, Y, meta, attributes = batch
            X, Y, meta = (x.to(device) for x in batch[:3])

            # predict pixels independently
            pred_pixels = model(X) # (B, pixels, 1)

            # aggregate pixel-level predictions to the county level
            pred_counties = county_aggregate_predictions(
                pred_pixels,
                pixel_customers_scaled=meta[:, :, META_COLS.index("pixel_customers_scaled")],
                valid_pixel_mask=meta[:, :, META_COLS.index("valid_pixel_mask")],
                lowerbound=lowerbound,
            ) # (B,)
            # if targets are in log space, convert prediction to log space
            if target_col.startswith("log"):
                pred_counties = torch.log10(pred_counties)
            
            # mean squared error
            losses = (pred_counties - Y)**2

            all_losses += losses.tolist()
            all_targets += Y.tolist()
            all_predictions += pred_counties.tolist()
            
            # save attributes (list of dicts)
            for attrset in attributes:
                all_fips_codes += [attrset["fips_code"]] * len(attrset["date"]) # scalar, repeated for each date
                all_dates += list(attrset["date"]) # array same shape as Y

            pbar.update(1)
    
    assert len(all_targets) == len(all_fips_codes) == len(all_dates)
    
    metrics, data = compute_metrics(all_losses, all_targets, all_predictions, target_col=target_col, lowerbound=lowerbound)
    
    attributes = {
        "fips_code": np.array(all_fips_codes),
        "date": np.array(all_dates),
    }
    return metrics, data, attributes



###
### Model
###



class ConvBlock(nn.Module):
    """
    1x1 conv along pixels (ie, process each pixel independently)

    input:  shape (batch, in_features, pixels)
    output: shape (batch, out_features, pixels)
    """

    def __init__(self, in_features, out_features, activation="relu", normalization="batchnorm"):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1)
        # norm
        if normalization is None:
            self.norm = nn.Identity()
        elif normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(out_features)
        else:
            raise ValueError(f"unknown norm {normalization}")
        # activation
        if activation is None:
            self.activation = nn.Identity()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU(approximate="tanh")
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class NL2OutageModel(nn.Module):
    """
    inputs: (batch, pixels, channels)
    internally transposes to torch-preferred (batch, channels, pixels)
    outputs: (batch, pixels, channels)
    """
    
    def __init__(self, in_features, out_features, dropout=0.0,
                 activation="relu", normalization=None,
                 hidden_sizes=[32, 128, 32], weight_init="default"):
        super().__init__()
        self.weight_init = weight_init

        # build network
        self.layers = nn.Sequential()
        in_size = in_features
        for out_size in hidden_sizes:
            self.layers.append(
                ConvBlock(in_size, out_size, activation=activation, normalization=normalization)
            )
            in_size = out_size
        # output layer
        if dropout > 0:
            self.layers.append(
                nn.Dropout(dropout)
            )
        self.layers.append(
            ConvBlock(in_size, out_features, activation=None, normalization=None)
        )

        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if self.weight_init == "default":
                pass
            elif self.weight_init == "trunc_normal":
                # borrowed from ConvNext2D scheme
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif self.weight_init == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif self.weight_init == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu", mode="fan_in")
            elif self.weight_init == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif self.weight_init == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu", mode="fan_in")
            else:
                raise ValueError(f"unknown weight init: {self.weight_init}")

    def forward(self, x):
        # x is shape (B, pixels, features)
        x = x.transpose(2, 1) # (B, features, pixels)
        x = self.layers(x) # (B, 1, pixels)
        # limit to [0-1] fraction of population out in each pixel
        x = F.sigmoid(x) # (B, 1, pixels)
        x = x.transpose(2, 1) # (B, pixels, 1)
        return x



def train_nl2outage_model(ARGS, eval_testset=True, prune_condition=None):
    """
    train a pixel-to-pixel outage model

    args:
        ARGS: namespace with model hyperparameters
        prune_condition: callable that returns True when this run should be stopped early (seperate from normal early stopping)
    """
    pprint(vars(ARGS))
    
    SCRIPT_START_TIME_STR = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    EXPERIMENT_DIR = OUTPUT_DIR / f"nl2outage/models/{ARGS.run_name}/{SCRIPT_START_TIME_STR}/"
    WEIGHTS_PATH = EXPERIMENT_DIR / "best_model.pytorch"
    os.makedirs(EXPERIMENT_DIR, exist_ok=False)

    print("experiment dir:", EXPERIMENT_DIR)

    # write metadata
    with open(EXPERIMENT_DIR / "meta.json", "w") as f:
        json.dump(dict(vars(ARGS)), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader, test_loader = build_dataloaders(
        predictors=ARGS.predictors,
        target=ARGS.target,
        valyears=ARGS.valyears,
        testyears=ARGS.testyears,
        batchsize=ARGS.batchsize,
        min_pop_cov_frac=ARGS.min_pop_cov_frac,
        lowerbound=ARGS.lowerbound,
        workers=ARGS.workers,
        recache=ARGS.recache_data,
    )

    print("Building model")
    descending_sizes = [int(ARGS.max_width / (ARGS.width_factor ** i)) for i in range(math.ceil(ARGS.depth/2))]
    if ARGS.depth % 2 == 0:
        hidden_sizes = descending_sizes[::-1] + descending_sizes
    else:
        hidden_sizes = descending_sizes[::-1][:-1] + descending_sizes
    print("hidden sizes:", hidden_sizes)

    model = NL2OutageModel(
        in_features=len(ARGS.predictors),
        out_features=1,
        hidden_sizes=hidden_sizes,
        dropout=ARGS.dropout,
        activation=ARGS.activation,
        normalization=ARGS.norm,
    )
    model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=ARGS.lr, 
        fused=True,
        # weight_decay=CONFIG.weight_decay,
    )
    # lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     factor=0.1,
    #     patience=5,
    # )
    lrscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=ARGS.epochs,
        eta_min=ARGS.lr / 1e3, # three orders of magnitude smaller than the original lr
    )



    ###
    ### Train Loop
    ###

    val_loss_history = []
    for epoch in range(ARGS.epochs):
        print(f"\n***** EPOCH {epoch} *****")

        model.train()
        train_losses = []
        all_targets = []
        all_predictions = []
        with tqdm(total=len(train_loader), desc="Train", ncols=120) as pbar:
            for batch in train_loader:
                X, Y, meta = (x.to(device) for x in batch[:3]) # attributes not needed for training
                print_nans(X=X, Y=Y, meta=meta)
                
                optimizer.zero_grad()

                # predict pixels independently
                pred_pixels = model(X) # (B, pixels, 1)
                
                # aggregate pixel-level predictions to the county level
                pred_counties = county_aggregate_predictions(
                    pred_pixels,
                    pixel_customers_scaled=meta[:, :, META_COLS.index("pixel_customers_scaled")],
                    valid_pixel_mask=meta[:, :, META_COLS.index("valid_pixel_mask")],
                    lowerbound=ARGS.lowerbound,
                ) # (B,)
                # if targets are in log space, convert prediction to log space
                if ARGS.target.startswith("log"):
                    pred_counties = torch.log10(pred_counties)
                
                # mean squared error
                losses = (pred_counties - Y)**2
                loss = torch.mean(losses)
                # save preds and targets
                all_targets += Y.tolist()
                all_predictions += pred_counties.tolist()
                
                loss.backward()
                # save loss elements
                train_losses += losses.tolist()
    
                # clip gradients, then optimizer step
                torch.nn.utils.clip_grad_value_(model.parameters(), 10)
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
        
        # metrics
        train_metrics, train_data = compute_metrics(
            train_losses, 
            all_targets, 
            all_predictions, 
            target_col=ARGS.target, 
            lowerbound=ARGS.lowerbound,
        )
        print("Train metrics:")
        pprint(train_metrics)
        print("LR:", lrscheduler.get_last_lr())
        
        val_metrics, val_data, val_attributes = evaluate(
            model, 
            val_loader, 
            device=device, 
            target_col=ARGS.target, 
            lowerbound=ARGS.lowerbound, 
            name="Val"
        )
        val_loss = val_metrics["loss"]
        print("Val metrics:")
        pprint(val_metrics)

        lrscheduler.step()

        # save best model
        if (epoch == 0) or (val_loss < np.min(val_loss_history)):
            # save weights
            print("saving best model to", WEIGHTS_PATH)
            torch.save(model.state_dict(), WEIGHTS_PATH)

            # save metrics
            with open(EXPERIMENT_DIR / f"best_val_metrics.json", "w") as f:
                json.dump(val_metrics, f)
            with open(EXPERIMENT_DIR / "best_val_data.pickle", "wb") as f:
                pickle.dump(val_data, f)

        val_loss_history.append(val_loss)

        # early stopping
        if epoch - np.argmin(val_loss_history) >= ARGS.earlystop:
            print("Early stopping")
            break

        # pruning
        if prune_condition is not None:
            if prune_condition(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics):
                print("Pruning condition met")
                with open(EXPERIMENT_DIR / f"pruned.txt", "w") as f:
                    f.write(f"pruned after epoch {epoch}")
                break

        # mem usage
        free, total = torch.cuda.mem_get_info(device)
        mem_used = (total - free) / (1024 ** 3)
        print(f"  gpu mem usage: {mem_used}")

    print("\n***** Training complete *****\n")
    print("Evaluating best model (by val loss)")
    state_dict = torch.load(WEIGHTS_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Train metrics (in eval mode, i.e. no dropout etc), for best-val-loss version of model
    train_metrics, train_data, train_attributes = evaluate(
        model, 
        train_loader, 
        device=device, 
        target_col=ARGS.target, 
        lowerbound=ARGS.lowerbound, 
        name="Train",
    )
    print("Train metrics:")
    pprint(train_metrics)
    # save metrics
    with open(EXPERIMENT_DIR / f"train_metrics.json", "w") as f:
        json.dump(train_metrics, f)
    with open(EXPERIMENT_DIR / "train_data.pickle", "wb") as f:
        pickle.dump(train_data, f)
    with open(EXPERIMENT_DIR / "train_attributes.pickle", "wb") as f:
        pickle.dump(train_attributes, f)

    # Validation set. best metrics were already saved at end of best epoch
    val_metrics, val_data, val_attributes = evaluate(
        model, 
        val_loader, 
        device=device, 
        target_col=ARGS.target, 
        lowerbound=ARGS.lowerbound, 
        name="Val",
    )
    print("Val metrics:")
    pprint(val_metrics)

    # Testing set
    if eval_testset:
        test_metrics, test_data, test_attributes = evaluate(
            model, 
            test_loader, 
            device=device, 
            target_col=ARGS.target, 
            lowerbound=ARGS.lowerbound, 
            name="Test",
        )
        print("Test metrics:")
        pprint(test_metrics)
        # save metrics
        with open(EXPERIMENT_DIR / f"test_metrics.json", "w") as f:
            json.dump(test_metrics, f)
        with open(EXPERIMENT_DIR / "test_data.pickle", "wb") as f:
            pickle.dump(test_data, f)
        with open(EXPERIMENT_DIR / "test_attributes.pickle", "wb") as f:
            pickle.dump(test_attributes, f)
    
    # return val metrics for use in hyperopt
    return train_metrics, val_metrics



if __name__ == "__main__":
    DEFAULT_X_COLS = [
        "log_landscan",
        "log_normalized_frac",
        "normalized_zscore",
        "log_annual_median",
        "log_annual_median_dev",
    ]
    DEFAULT_Y_COL = "customers_out_atviewtime"


    parser = argparse.ArgumentParser()
    # metadata
    parser.add_argument("--run-name", required=True, help="distinguishing name for this run")
    # model
    parser.add_argument("--max-width", default=64, type=int, help="size of the largest hidden layer")
    parser.add_argument("--width-factor", default=1.75, type=float, help="multiplicative factor determining the size increase from one hidden layer to the next")
    parser.add_argument("--depth", default=1, type=int, help="number of layers")
    parser.add_argument("--norm", default=None)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--dropout", type=float, default=0.25)
    # predictors/target
    parser.add_argument("--predictors", default=DEFAULT_X_COLS, nargs="+")
    parser.add_argument("--target", default=DEFAULT_Y_COL, choices=[
        "customers_out_atviewtime", 
        "customers_out_nightly_mean",
        "log_customers_out_atviewtime",
        "log_customers_out_nightly_mean",
    ])
    parser.add_argument("--lowerbound", default=1, type=float, help="clip county-level customers_out to not be below this value")
    # dataset
    parser.add_argument("--valyears", default=[2023], nargs="+", type=int, help="years to use for validation")
    parser.add_argument("--testyears", default=[2024], nargs="+", type=int, help="years to use for testing")
    parser.add_argument("--min-pop-cov-frac", default=0.99, type=float, help="minimum population coverage fraction for a date to be included in the dataset")
    # training
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("--epochs", default=25, type=int, help="max epochs")
    parser.add_argument("--earlystop", default=10, type=int, help="n epochs with no improvement")
    # misc
    parser.add_argument("--workers", default=2, type=int, help="dataset worker processes")
    parser.add_argument("--recache-data", action="store_true")
    ARGS = parser.parse_args()

    train_nl2outage_model(ARGS)

