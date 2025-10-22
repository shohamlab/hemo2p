# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 15:53:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-05-03 12:17:14

''' Collection of generic utilities. '''

import numpy as np
import pandas as pd
from functools import wraps
from string import ascii_lowercase
import re

from logger import logger

IND_LETTERS = list(ascii_lowercase[8:])  # generic index letters


def is_iterable(x):
    ''' Check if an object is iterbale (i.e. a list, tuple or numpy array) '''
    for t in [list, tuple, np.ndarray, pd.Series]:
        if isinstance(x, t):
            return True
    return False


def as_iterable(x):
    ''' Return an iterable of an object if it is not already iterable '''
    return x if is_iterable(x) else [x]


def float_to_uint8(arr):
    ''' Transform a floating point (0 to 1) array to an 8-bit unsigned integer (0 to 255) array. '''
    return (arr * 255).astype(np.uint8)


def moving_average(x, n=3):
    '''
    Apply moving average on first axis of a n-dimensional array
    
    :param x: n-dimensional array
    :param n: moving average window size (in number of frames)
    :return: smoothed array with exact same dimensions as x
    '''
    # Logging string
    s = f'smoothing {x.shape} array with {n} samples moving average'
    if x.ndim > 1:
        s = f'{s} along axis 0'
    logger.info(s)
    # Pad input array on both sides
    if n % 2 == 0:
        n -= 1
    w = n // 2
    wvec = [(0, 0)] * x.ndim
    wvec[0] = (w, w)
    xpad = np.pad(x, wvec, mode='symmetric')
    # Apply moving average along first axis
    ret = np.cumsum(xpad, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    # Return output
    return ret[n - 1:] / n


def apply_rolling_window(x, w, func=None, warn_oversize=True, pad=True):
    '''
    Generate a rolling window over an array an apply a specific function to the result.
    Defaults to a moving average.
    
    :param x: input array
    :param w: window size (number of array samples used to apply the function)
    :param func (optional): function to apply to the rolling window result
    :return: output array of equal size to the input array, with the rolling window and function applied.
    '''
    # If more than 1 dimension -> reshape to 2D, apply on each row, and reshape back to original shape
    if x.ndim > 1:
        dims = x.shape
        x = x.reshape(-1, dims[-1])
        x = np.array([apply_rolling_window(xx, w, func=func, warn_oversize=False) for xx in x])
        return x.reshape(*dims)
    # Check that window size is valid
    if w % 2 == 0:
        raise ValueError('window size must be an odd number')
    if w > x.size and warn_oversize:
        logger.warning(f'window size ({w}) is larger than array length ({x.size})')
    # If function not provided, apply mean by default
    if func is None:
        func = lambda x: x.mean()
    # Pad input array on both sides
    if pad:
        x = np.pad(x, w // 2, mode='symmetric')
    # Generate rolling window over array
    roll = pd.Series(x).rolling(w, center=True)
    # Apply function over rolling window object, drop NaNs and extract output array 
    # return np.array([func(r) for r in roll])
    return func(roll).dropna().values


def array_to_dataframe(arr, name, dim_names=None):
    '''
    Convert a multidimensional array into a multi-index linearized dataframe.
    
    :param arr: multi-dimensional array
    :param name: name of the variable stored in the array
    :param dim_names (optional): names of the dimensions of the array
    :return: multi-index dataframe with linearized array as the only non-index column
    '''
    if dim_names is None:
        dim_names = IND_LETTERS[:arr.ndim]
    else:
        if len(dim_names) != arr.ndim:
            raise ValueError(f'number of dimensions names {len(dim_names)} do not match number of array dimensions ({arr.shape})')
    index = pd.MultiIndex.from_product([np.arange(x) for x in arr.shape], names=dim_names)
    return pd.DataFrame(data=arr.flatten(), columns=[name], index=index)


def arrays_to_dataframe(arrs_dict, **kwargs):
    '''
    Convert a dictionary of multidimensional arrays into a multi-index linearized dataframe.
    
    :param arrs_dict: dictionary of multi-dimensional arrays
    :return: multi-index dataframe with linearized arrays in different columns
    '''
    names, arrs = zip(*arrs_dict.items())
    assert all(x.shape == arrs[0].shape for x in arrs), 'inconsistent array shapes'
    df = array_to_dataframe(arrs[0], names[0], **kwargs)
    for name, arr in zip(names[1:], arrs[1:]):
        df[name] = arr.flatten()
    return df


def describe_dataframe_index(df, join_str=' x '):
    ''' Describe dataframe index '''
    d = {}
    if hasattr(df, 'index'):
        mux = df.index
    else:
        mux = df
    for k in mux.names:
        l = len(mux.unique(level=k))
        key = k
        if l > 1:
            key = f'{key}s'
        d[key] = l
    return join_str.join([f'{v} {k}' for k, v in d.items()])


def idx_format(idxs):
    ''' 
    Format a list of indexes as a range string (if possible)

    :param idxs: list of indexes
    :return: range string, or original list if not possible
    '''
    # If input is scalar, return corresponding string
    if isinstance(idxs, (int, np.int64)):
        return str(idxs)
    
    # Cast input as numpy array
    idxs = np.asarray(idxs)

    # If input is contiguous, return corresponding range string
    if idxs.data.contiguous:
        return f'{idxs[0]} - {idxs[-1]}'
    
    # Otherwise, return original list
    return str(idxs)


def is_rectilinear(s):
    '''
    Check that the index of the given pandas Series/DataFrame is rectilinear, i.e. that the
    index is a Cartesian product of the indices of each level.
    '''
    dims = [len(s.index.unique(level=k)) for k in s.index.names]
    return np.prod(dims) == len(s)


def mux_series_to_array(s):
    ''' 
    Convert a multi-indexed series to a multi-dimensional numpy array.
    
    :param s: multi-indexed series
    :return: multi-dimensional numpy array in which the dimensions are ordered
        according to the order of the index levels in the series
    '''
    # Check that input is a rectilinear multi-indexed series
    if not isinstance(s, pd.Series):
        raise ValueError('input is not a series')
    if not isinstance(s.index, pd.MultiIndex):
        raise ValueError(f'{s.name} is not a multi-indexed series')
    if not is_rectilinear(s):
        raise ValueError(f'{s.name} series index is not rectilinear')
    
    # Sort the series by the index levels
    s = s.sort_index()
    
    # Get the size of each index dimension
    shape = {k: len(s.index.unique(level=k)) for k in s.index.names}

    # Extract the values and reshape
    return s.values.reshape([shape[k] for k in s.index.names])


def normalize_stack(x, bounds=(0, 1000)):
    '''
    Normalize stack to a given interval
    
    :param x: (nframe, Ly, Lx) stack array
    :param bounds (optional): bounds for the intensity interval
    :return rescaled stack array
    '''
    # Get input data type and related bounds
    dtype = x.dtype
    if str(dtype).startswith('int'):
        dinfo = np.iinfo(dtype)
    else:
        dinfo = np.finfo(dtype)
    dbounds = (dinfo.min, dinfo.max)
    # Make sure output bounds are within data type limits
    if bounds[0] < dbounds[0] or bounds[1] > dbounds[1]:
        raise ValueError(f'rescaling interval {bounds} exceeds possible {dtype} values')
    # Get input bounds (recasting as float to make ensure correct downstream computations)
    input_bounds = (x.min().astype(float), x.max().astype(float))
    # Get normalization factor
    input_ptp = input_bounds[1] - input_bounds[0]
    output_ptp = bounds[1] - bounds[0]
    norm_factor = input_ptp / output_ptp
    # Compute normalized array
    y = x / norm_factor - input_bounds[0] / norm_factor
    # Cast as input type and return
    return y.astype(dtype)


def nan_proof(func):
    '''
    Wrapper around cost function that makes it NaN-proof
    
    :param func: function taking a input a pandas Series and outputing a pandas Series
    :return: modified, NaN-proof function object
    '''
    @wraps(func)
    def wrapper(s, *args, **kwargs):
        # Remove NaN values
        s2 = s.dropna()
        # Call function on cleaned input series
        out = func(s2, *args, **kwargs)
        # If output is of the same size as the cleaned input, add it to original input to retain same dimensions
        if (is_iterable(out) or isinstance(out, pd.Series)) and len(out) == s2.size:
            s[s2.index] = out
            return s
        # Otherwise return output as is
        else:
            return out
    return wrapper


def pandas_proof(func):
    '''
    Wrapper around function that makes it pandas-proof
    
    :param func: processing function
    :return: modified, pandas-proof function object
    '''
    @wraps(func)
    def wrapper(y, *args, **kwargs):
        yout = func(y)
        if isinstance(y, pd.Series):
            return pd.Series(data=yout, index=y.index)
        else:
            return yout
    return wrapper


def bounds(x):
    ''' Extract minimum and maximum of array simultaneously '''
    if isinstance(x, slice):
        return np.array([x.start, x.stop - 1])    
    return np.array([min(x), max(x)])


def is_within(x, bounds):
    ''' Determine if value is within defined bounds '''
    return np.logical_and(x >= bounds[0], x <= bounds[1])


def rsquared(y, ypred):
    '''
    Compute the R-squared coefficient between two 1D arrays
    
    :param x: reference (i.e. data) array
    :param xpred: predictor array
    :return: R2 of predictor
    '''
    # Compute SS of residuals
    residuals = y - ypred
    ss_res = np.sum(residuals**2)
    # Compute total SS
    ss_tot = np.sum((y - np.mean(y))**2)
    # Compute and return R2
    return 1 - (ss_res / ss_tot)


def relative_error(y, yref):
    '''
    Return relative error between 2 arrays
    
    :param y: evaluated array
    :param yref: reference array
    :return: relative error
    '''
    return np.mean(np.abs((y - yref) / yref))


def sigmoid(x, x0=0, sigma=1., A=1, y0=0):
    ''' 
    Apply sigmoid function with specific center and width
    
    :param x: input signal
    :param x0: sigmoid center (i.e. inflection point)
    :param sigma: sigmoid width
    :param A: sigmoid min-to-max amplitude
    :param y0: sigmoid vertical offset
    :return sigmoid function output
    '''
    norm_sig = 1 / (1 + np.exp(-(x - x0) / sigma))
    return A * norm_sig + y0
