# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-12-01 14:15:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-12-02 10:18:03

import os
import numpy as np
from tifffile import imread, imwrite
from scipy.io import loadmat

from logger import logger
from utils import is_iterable


def load_run_data(input_dir, projfunc=None):
    '''
    Load data from acquisition run(s)

    :param: input_dir: input data folder. Can also be a list of folder, in which
        case data from each folder is loaded and returned as a folder-indexed dictionary
    :param projfunc: stack projection function (defaults to mean)
    :return: 4-tuple with filepath to fluorescence stack, number of frames, FOV projection image, and ROI masks
    '''
    # If input is a list of directories, call function recursively to load data from each directory
    if is_iterable(input_dir):
        stackfpath_dict = {}
        nframes_dict = {}
        FOV_dict = {}
        masks_dict = {}
        for idir in input_dir:
            key = os.path.basename(os.path.normpath(idir))
            (stackfpath_dict[key],
             nframes_dict[key],
             FOV_dict[key],
             masks_dict[key]) = load_run_data(idir, projfunc=projfunc)
        return stackfpath_dict, nframes_dict, FOV_dict, masks_dict
    
    logger.info(f'loading data from "{input_dir}"')

    # Create stacks directory if it does not exist
    stacksdir = os.path.join(input_dir, 'stacks')
    if not os.path.exists(stacksdir):
        os.makedirs(stacksdir)
        logger.info(f'created stacks directory: {stacksdir}')

    # Load TIF stack, either from existing file or by assembling from individual TIF files
    stackfpath = os.path.join(stacksdir, 'stack_original.tif')
    if os.path.exists(stackfpath):        
        logger.info('loading stack from existing file')
        stack = imread(stackfpath)
    else:
        tif_files = [f for f in os.listdir(input_dir) if f.endswith('.ome.tif')]
        tif_files.sort()
        logger.info(f'assembling stack from {len(tif_files)} TIF files...')
        stack = np.array([imread(os.path.join(input_dir, file)) for file in tif_files])
        logger.info(f'saving stack to file: {stackfpath}')
        imwrite(stackfpath, stack)
    nframes, nx, ny = stack.shape
    logger.info(f'loaded {nframes}-frames stack with shape {nx}x{ny}')

    # Extract FOV from stack using projection function
    if projfunc is None:
        projfunc = np.mean
    logger.info(f'extracting FOV using {projfunc.__name__} projection')
    FOV = projfunc(stack, axis=0)

    # Load ROI mask
    mask_fpath = os.path.join(input_dir, 'masks', 'labelimg.mat')
    masks = loadmat(mask_fpath)['labelimg']
    nrois = masks.max()
    logger.info(f'loaded {nrois} ROIs mask')

    return stackfpath, nframes, FOV, masks