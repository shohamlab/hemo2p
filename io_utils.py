# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-12-01 14:15:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-12-01 14:15:55

import os
import numpy as np
from tifffile import imread, imwrite
from scipy.io import loadmat

from logger import logger


def load_stack_and_masks(input_dir):
    '''
    Load stack and masks from input data directory

    :param: input_dir: input data directory
    :return: 2-tuple with fluorescence stack and ROI masks
    '''
    logger.info(f'input data directory: {input_dir}')

    # Create stacks directory if it does not exist
    stacksdir = os.path.join(input_dir, 'stacks')
    if not os.path.exists(stacksdir):
        os.makedirs(stacksdir)
        logger.info(f'created stacks directory: {stacksdir}')

    # List all sorted TIF files in the data directory
    logger.info('listing TIF files...')
    tif_files = [f for f in os.listdir(input_dir) if f.endswith('.ome.tif')]
    tif_files.sort()

    # Load TIFs and assemble stack
    logger.info('loading stack...')
    stackfpath = os.path.join(stacksdir, 'stack_original.tif')
    if os.path.exists(stackfpath):        
        stack = imread(stackfpath)
    else:
        stack = np.array([imread(os.path.join(input_dir, file)) for file in tif_files])
        imwrite(stackfpath, stack)
    nframes, nx, ny = stack.shape
    logger.info(f'loaded {nframes}-frames stack with shape {nx}x{ny}')

    # Load ROI mask
    mask_fpath = os.path.join(input_dir, 'masks', 'labelimg.mat')
    masks = loadmat(mask_fpath)['labelimg']
    nrois = masks.max()
    logger.info(f'loaded {nrois} ROIs mask')

    return stack, masks    