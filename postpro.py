# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-05-03 12:25:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-12-15 21:41:10

''' Post-processing utilities '''

import numpy as np
import pandas as pd
from tifffile import imread
import os

from logger import logger
from utils import apply_rolling_window


def extract_traces_df_from_stack(stack, masks=None, aggfunc='mean', nmax=None, npertrial=None):
    '''
    Extract traces from a stack of images and return a pandas DataFrame.

    :param stack: 3D numpy array (nframes x nrows x ncols)
    :param masks (optional): 2D numpy array (nrows x ncols) where each ROI is labeled with a unique integer
    :param aggfunc (optional): function to compute the trace for each ROI (default: 'mean')
    :param nmax (optional): maximum number of frames to process
    :param npertrial (optional): number of frames per trial
    :return: pandas series/dataframe with traces
    '''
    # If input stack is a dictionary, call function recursively to extract traces
    if isinstance(stack, dict):
        tracesdict = {}
        for k, s in stack.items():
            logger.info(f'processing {k} stack...')
            tracesdict[k] = extract_traces_df_from_stack(s, masks=masks, aggfunc=aggfunc, nmax=nmax, npertrial=npertrial)
        return pd.concat(tracesdict, axis=0, names=['kind'])
        
    # Convert aggregation function string to function object
    aggfuncobj = {
        'mean': np.mean,
        'median': np.median
    }[aggfunc]

    # If input stack is a file path, read the image stack
    if isinstance(stack, str) and os.path.isfile(stack):
        stack = imread(stack)
    
    # Create index vector
    nframes = stack.shape[0]
    if nmax is not None:
        nframes = min(nframes, nmax)
        stack = stack[:nframes]
    frameidx = pd.Index(np.arange(nframes), name='frame')

    # If npertrial provided, add extra multi-index level for trial index and within-trial frame index
    if npertrial is not None:
        itrial = frameidx // npertrial
        within_trial_frame = frameidx % npertrial
        frameidx = pd.MultiIndex.from_arrays(
            [frameidx.values, itrial, within_trial_frame], 
            names=['frame', 'trial', 'within_trial_frame']
        )

    # Create series with FOV spatial aggregate trace
    logger.info(f'computing FOV-{aggfunc} trace over {nframes} frames')
    s = pd.Series(
        data=aggfuncobj(stack, axis=(1, 2)),
        index=frameidx, 
        name='FOV'
    )

    # If ROI masks not provided, return series 
    if masks is None:
        return s

    # Extract mean trace for each ROI using mask information 
    nrois = masks.max()
    logger.info(f'computing mean traces for {nrois} ROIs')
    ROIlabels = [f'ROI{i+1}' for i in range(nrois)]
    yROIs = np.array([stack[:, masks == i].mean(axis=1) for i in range(1, nrois + 1)]).T

    # Assemble into DataFrame
    df = pd.DataFrame(index=frameidx, columns=ROIlabels, data=yROIs)

    # Merge with FOV average, and return
    return pd.concat([s.to_frame(), df], axis=1)


def compute_dff(F, basefunc='median', wroll=None):
    '''
    Compute dF/F for each trace in a DataFrame.

    :param F: DataFrame with traces
    :param basefunc (optional): function to compute baseline value for each trace (default: 'median')
    :param wroll (optional): window size for rolling baseline calculation
    :return: DataFrame with dF/F traces
    '''
    # If multi-index, compute dF/F separately for each group across extra dimension(s)
    if isinstance(F.index, pd.MultiIndex) and len(F.index.names) > 1:
        gby = [k for k in F.index.names if k not in ('frame', 'within_trial_frame', 'trial')]
        if len(gby) > 0:
            logger.info(f'computing dFF across {gby}')
            return (F
                .groupby(gby, sort=False)
                .apply(lambda x: compute_dff(x.droplevel(gby), basefunc=basefunc, wroll=wroll))
            )
        
    # If rolling window specified, compute rolling baseline and correct signals
    if wroll is not None:
        logger.info(f'computing time-varying F0 with rolling {wroll} frames window using {basefunc.__name__} function')
        # print(apply_rolling_window(F['FOV'].values, wroll, func=basefunc))
        F0 = F.transform(lambda x: apply_rolling_window(x.values, wroll, func=basefunc), axis=0)
        logger.info('correcting F traces to compensate for time variations in baseline')
        F0mean = F0.mean(axis=0)
        F = F - F0 + F0mean

    # Compute baseline value for each trace as median
    logger.info(f'computing static F0 using {basefunc.__name__} function')
    F0 = F.agg(basefunc, axis=0)

    # Compute dF/F for each trace
    logger.info('computing dF/F traces')
    dFF = (F - F0) / F0

    # Return
    return dFF