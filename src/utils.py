import functools
import time

import numpy as np
import numba


def timed(f):
    """
    decorator which times the execution and writes to logger.info, and 
    prints it too
    """
    @functools.wraps(f)
    def wrapper(*args, **kwds):
        start = time.perf_counter()
        result = f(*args, **kwds)
        elapsed = time.perf_counter() - start
        # output
        print(f"{f.__name__} took {elapsed:.02f} sec")
        return result

    return wrapper


def downsampling_2d(data, x_step, y_step=None, agg_f=np.nanmean,
        allow_truncate=False, min_valid_frac=None, min_valid_count=None):
    """
    split a 2D grid into (x_step, y_step) sized blocks, and average each block (by default ignoring NaNs with np.nanmean)
    to create each new pixel in the downsampled array. All units in pixels
    args:
        data: shape (x, y)
        agg: a numpy reduction function that accepts an array and an `axis=(-2,-1)` keyword arg.
        x_step: int in pixels
        y_step (defaults to x_step if None)
        allow_truncate: drop the leftover data that does not fit in a block (if False, will throw an error if data must be dropped)
        min_valid_frac: each block must be at least this fraction valid (not-None), or the result in that block will be set to NaN
        min_valid_count: same as above, but absolute count. Supply at most one of these two args
    returns:
        array, shape (x//x_step, y//y_step)
    adapted from https://github.com/ilastik/lazyflow/blob/master/lazyflow/utility/blockwise_view.py
    """
    # at least one of these args must be None
    assert (min_valid_count is None) or (min_valid_frac is None)

    if y_step is None:
        y_step = x_step

    if not allow_truncate:
        x, y = data.shape
        assert x % x_step == 0
        assert y % y_step == 0

    if not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data)
    blockshape = (y_step, x_step)
    outershape = tuple(np.array(data.shape) // blockshape)
    view_shape = outershape + blockshape

    # inner strides: strides within each block (same as original array)
    intra_block_strides = data.strides
    # outer strides: strides from one block to another
    inter_block_strides = tuple(data.strides * np.array(blockshape))

    view = np.lib.stride_tricks.as_strided(data, shape=view_shape, strides=(inter_block_strides + intra_block_strides))

    result = agg_f(view, axis=(-2,-1))
    # filter out blocks with too few samples
    if min_valid_frac is not None:
        # compute valid count from valid frac
        blocksize = x_step * y_step
        min_valid_count = blocksize * min_valid_frac
    if min_valid_count is not None:
        counts = np.sum(~np.isnan(view), axis=(-2,-1))
        result[counts < min_valid_count] = np.nan
    return result