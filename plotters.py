# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-05-03 12:15:01
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-06-12 15:50:46

''' Plotting utilities '''

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
from skimage.measure import find_contours
from itertools import chain
from matplotlib.collections import PathCollection
from matplotlib.path import Path

from logger import logger
from utils import is_iterable, as_iterable


def plot_traces(df, iROIs=None, ROIavg=False, istimbounds=None, ax=None, 
                ylabel='intensity', xbounds=None, height=2, aspect=5, legend='auto'):
    ''' 
    Plot time traces of FOV average and a few random ROIs.
    
    :param df: DataFrame with traces
    :param iROIs (optional): indexes of ROIs to plot
    :param ROIavg (optional): whether to plot ROI average traces (default: False)
    :param istimbounds (optional): start and end indexes of stimulus interval
    :param ax: axis to plot on (default: None)
    :param ylabel: y-axis label (default: 'intensity')
    :param xbounds: x-axis bounds (default: None)
    :param height: figure height (default: 3)
    :param aspect: aspect ratio (default: 5)
    :return: figure
    '''
    # Convefrt input to DataFrame if Series
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    # If input is a dictionary, call function recursively to plot
    # each trace in a separate axis
    if isinstance(df, dict):
        nrows = len(df)
        if ax is not None:
            if not is_iterable(ax) or len(ax) != nrows:
                raise ValueError('Number of axes must match number of traces')
            axes = ax
            fig = axes[0].get_figure()
        else:
            fig, axes = plt.subplots(
                nrows, 1, figsize=(height * aspect, height * nrows), sharex=True)
            axes = np.atleast_1d(axes)
        for ax, (k, subdf) in zip(axes, df.items()):
            plot_traces(subdf, iROIs=iROIs, ROIavg=ROIavg, istimbounds=istimbounds, 
                        ax=ax, ylabel=k, xbounds=xbounds, legend=legend)
            legend = False
        return fig

    # Create/retrieve figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(aspect * height, height))
    else:
        fig = ax.get_figure()
    sns.despine(ax=ax)

    # If additional index dimension, set as "style" variable
    has_extra_dims = isinstance(df.index, pd.MultiIndex) and len(df.index.names) > 1
    if has_extra_dims:
        style = df.index.names[0]
        style_order = df.index.unique(level=style)
    else:
        style = None
        style_order = None
    
    # Initialize list of columns to plot with FOV
    cols = ['FOV']

    # Identify available ROI columns
    ROIcols = df.columns[1:]

    # If specific ROI indexes specified, select corresponding columns 
    if iROIs is not None:
        # If 'all' is provided, plot all ROIs
        if isinstance(iROIs, str):
            if iROIs == 'all':
                iROIs = np.arange(len(ROIcols))
            else:
                raise ValueError(f'Invalid iROIs value: {iROIs}')
        iROIs = as_iterable(iROIs)
        ROIcols = [ROIcols[i] for i in iROIs]

    # If ROI-averaging requested, compute and add ROI-average column
    if ROIavg:
        df['ROIavg'] = df[ROIcols].mean(axis=1)
        cols.append('ROIavg')
    
    # Otherwise, add explicited ROI columns to list of columns to plot
    elif iROIs is not None:
        cols = cols + ROIcols

    # Check total number of columns to plot
    if len(cols) > 10:
        raise ValueError('Number of traces to plot must be <= 10')
    
    logger.info(f'plotting {ylabel} traces for columns {cols}')

    # Select columns of interest
    dfplot = df[cols]  
    dfplot.columns.name = 'variable'

    # Prepare data for plotting
    dfplot = (
        dfplot.stack()
        .swaplevel()
        .sort_index()
        .rename(ylabel)
    )

    # Plot traces
    colors = ['k'] + sns.color_palette('tab10', n_colors=len(cols) - 1)
    palette = dict(zip(cols, colors))
    sns.lineplot(
        data=dfplot.reset_index(),
        x='frame',
        y=ylabel,
        ax=ax, 
        hue='variable',
        palette=palette,
        style=style,
        style_order=style_order,
        legend=legend
    )

    # Move legend
    if legend:
        sns.move_legend(
            ax, 
            loc='center left', 
            bbox_to_anchor=(1, 0.5), 
            frameon=False
        )

    # Mark stimulus interval, if provided
    if istimbounds is not None:
        ax.axvspan(*istimbounds, color='silver', alpha=0.5)

    # Set x-axis bounds, if provided
    if xbounds is not None:
        ax.set_xlim(xbounds)

    # Return figure
    return fig


def get_cells_mplobjs(ROI_masks, dims, mode='contour', color='k', alpha=1):
    '''
    Get matplotlib objects to plot the spatial distribution of cells in the field of view.
    
    :param ROI_masks (optional): ROI-indexed dataframe of (x, y) coordinates and weights
    :param dims: (Ly, Lx) dimensions of the image
    :param mode (optional): 'contour' or 'mask'
    :param color (optional): color of the mask
    :param alpha (optional): transparency of the mask
    :return: list of [matplotlib objects to plot] for each hue level
    '''
    # Check inputs
    if mode not in ['contour', 'mask']:
        raise ValueError(f'unknown mode {mode} (can be "contour" or "mask")')

    # Get number of ROIs
    iROIs = ROI_masks.index.unique(level='ROI')
    nROIs = len(iROIs)

    # Initialize empty 3D mask-per-cell matrix
    Z = np.zeros((nROIs, *dims), dtype=np.float32)
    
    # Compute mask per ROI
    for i, (_, ROI_mask) in enumerate(ROI_masks.groupby('ROI')):
        Z[i, ROI_mask['ypix'], ROI_mask['xpix']] = 1
    
    # Mask "contour" mode
    if mode == 'contour':
        # Extract contours
        contours = list(chain.from_iterable(map(find_contours, Z)))
        
        # Invert x and y coordinates for compatibility with imshow
        contours = [c[:, ::-1] for c in contours]
        
        # Return contours
        return contours

    # Full "mask" mode
    else:
        # Stack Z matrices along ROIs to get 1 mask matrix
        mask = Z.max(axis=0)

        # Assign color and transparency to each mask
        rgbmask = np.zeros((*mask.shape, 4))

        if isinstance(color, str):
            color = to_rgb(color)
        rgbmask[mask == 1] = [*color, alpha]
        
        # Return RGBA mask
        return rgbmask


def get_ROI_contours(*args, color='k', lw=1, **kwargs):
    contours = get_cells_mplobjs(*args, mode='contour', **kwargs)
    return PathCollection(
        [Path(ctr) for ctr in contours], 
        fc='none', ec=color, lw=lw)


def save_figs_book(input_dir, figs, suffix=None):
    ''' Save figures dictionary as consecutive pages in single PDF document. '''
    if not os.path.isdir(input_dir):
        raise ValueError(f'input directory {input_dir} does not exist')
    fcode = 'figs'
    if suffix is not None:
        fcode = f'{fcode}_{suffix}'
    fname = f'{fcode}.pdf'
    fpath = os.path.join(input_dir, fname)
    file = pdf_backend.PdfPages(fpath)
    logger.info(f'saving figures in {fpath}:')
    for v in tqdm(figs.values()):
        file.savefig(v, transparent=True, bbox_inches='tight')
    file.close()