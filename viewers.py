# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-05 17:56:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-05-15 17:04:34

''' Notebook image viewing utilities. '''

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy_image_widget as niw
from ipywidgets import Label, IntSlider, Play, VBox, HBox, HTML, Button, Output, jslink
from IPython.display import display
from tifffile import TiffFile
import imageio as iio

from logger import logger
from imlabel import add_label_to_image
from utils import float_to_uint8, is_iterable, idx_format


class StackViewer:
    '''
    Simple stack viewer, inspired from Robert Haase's stackview package
    (https://github.com/haesleinhuepf/stackview/).
    '''

    npix_label = 10 # number of pixels used for upper-right labeling
    
    def __init__(self, fpaths, headers, title=None, continuous_update=True, display_size=350, 
                 scaling_factor=1., verbose=True):
        '''
        Initialization.

        :param fpaths: list of path(s) to the file(s) containing the image stack(s)
        :param headers: list of header(s) associated to each stack file
        :param title: optional title to render above the image(s)
        :param continuous_update: update the image while dragging the mouse (default: True)
        :param display_size: size of the render (in pixels per axis)
        '''
        logfunc = logger.info if verbose else logger.debug
        self.fpaths = fpaths
        self.scaling_factor = scaling_factor
        logfunc('initializing stack viewer')
        self.fobjs = [self.get_fileobj(fp) for fp in self.fpaths]
        self.headers = headers
        self.title = title
        nframes = [self.get_nframes(x) for x in self.fobjs]
        shapes = [self.get_frame_shape(x) for x in self.fobjs]
        if not len(list(set(nframes))) == 1:
            raise ValueError(f'Inconsistent number of frames: {nframes}')
        if not len(list(set(shapes))) == 1:
            raise ValueError(f'Inconsistent frame shapes: {shapes}')
        self.nframes = nframes[0]
        self.shape = shapes[0]
        logfunc(f'stack size: {(self.nframes, *self.shape)}')

        # Initialize other attributes
        self.display_width = self.display_height = display_size
        self.continuous_update = continuous_update

    def get_fileobj(self, fpath):
        ''' Get the binary file object corresponding to a file path '''
        if isinstance(fpath, np.ndarray):
            return fpath
        else:
            return TiffFile(fpath)

    def get_nframes(self, fobj):
        ''' Get the number of frames in a stack '''
        if isinstance(fobj, np.ndarray):
            return fobj.shape[0]
        else:
            return len(fobj.pages)

    def get_frame_shape(self, fobj, i=0):
        ''' Get the shape of a frame in a stack '''
        if isinstance(fobj, np.ndarray):
            return fobj.shape[1:]
        else:
            return fobj.pages[i].shape

    def get_frame(self, fobj, i):
        '''Get a particular frame in the stack '''
        if isinstance(fobj, np.ndarray):
            return fobj[i]
        else:
            return fobj.pages[i].asarray()

    # def __del__(self):
    #     ''' Making sure to close all open binary file objects upon deletion. '''
    #     for sobj in self.fobjs:
    #         sobj.close()

    def get_frame_index_range(self, fbounds):
        ''' Get a range of frames indexes given specified index bounds '''
        if fbounds is None:
            fbounds = [0, self.nframes - 1]
        else:
            logger.info(f'frame frange: {fbounds}')
        return range(fbounds[0], fbounds[1] + 1)
    
    def get_dynamic_range(self, i, frange):
        ''' 
        Get dynamic range of pixel values across a stack.
        
        :param i: stack index
        :param frange: frame index range
        :return: lower and upper bounds of pixel values across the stack
        '''
        logger.info(f'computing stack dynamic range across frames {frange.start} - {frange.stop -1}...')
        # Get vectors of min and max intensity values across frames
        Imin, Imax = self.get_frame_metric_evolution(
            self.fobjs[i], frange, func=lambda x: (x.min(), x.max())).T
        # Get min and max values across the stack
        Imin, Imax = np.min(Imin), np.max(Imax)
        # Log and return
        logger.info(f'stack dynamic range range: {Imin} - {Imax}')
        return (Imin, Imax)
    
    def slider_format(self, val):
        ''' Format slider readout value '''
        if self.fps is not None:
            return f'{val / self.fps:.2f} s'
        else:
            return f'{val}'

    def set_frame_slider(self, frange):
        ''' Set slider control to change frame. '''
        # Define default slider parameters
        slider_params = dict(
            value=frange.start,
            min=frange.start,
            max=frange.stop - 1,
            step=1,
        )

        # Create play widget
        playback_interval = 100. if self.fps is None else 1000 / self.fps
        playback_interval /= self.playback_speed
        self.play = Play(
            interval=playback_interval, 
            **slider_params
        )
        
        # Create slider object
        self.frame_slider = IntSlider(
            continuous_update=self.continuous_update,
            readout=False,
            **slider_params
        )
        
        # Link frame slider and play widgets
        jslink((self.play, 'value'), (self.frame_slider, 'value'))

        # Create slider label
        self.frame_slider_label = Label(value=self.slider_format(self.frame_slider.value))
    
    def get_header(self, text):
        ''' Get header text component '''
        return HTML(value=f'<center>{text}</center>')

    def iter_frames(self, fobj, frange, func=None):
        '''
        Iterate through frames and apply func

        :param fobj: file object
        :param frange: frame index range
        :param func: function applied on the frame at each iteration within the range
        '''
        if func is None:
            func = lambda x: None
        out = []
        for k in tqdm(list(frange)):
            frame = self.get_frame(fobj, k)
            out.append(func(k, frame))
        return out

    def get_frame_metric_evolution(self, fobj, frange, func=None):
        ''' Compute the time course of a given frame metrics across a specific range of frames. '''
        if func is None:
            func = lambda x: x.mean()
        xlist = self.iter_frames(fobj, frange, func=lambda _, x: func(x))
        return np.array(xlist)

    def transform(self, arr):
        ''' Transform a grayscale intensity image to a colored image using a specific colormap '''
        # Transform array through colormap to get a (0, 1) float RGB image
        img = self.cmap(arr)[:, :, :-1]
        # Convert to uint8 and return
        return float_to_uint8(img)
    
    def label(self, img, label):
        ''' Add label to top-right corner of a frame. '''
        return add_label_to_image(img, label, color='red')

    def process_frame(self, arr, norm, i):
        '''
        Process frame.
        
        :param arr: frame array
        :param norm: normalizer object
        :return: processed frame (i.e. normalized, transformed and labeled) 
        '''
        if norm is not None:
            arr = norm(arr)
        arr = self.transform(arr)
        if self.labels[i]:
            arr = self.label(arr, self.labels[i])
        return arr

    def init_view(self):
        ''' Initialize view with random data.'''
        view = niw.NumpyImage(self.transform(np.random.rand(*self.shape)))
        if self.display_width is not None:
            view.width_display = self.display_width
        if self.display_height is not None:
            view.height_display = self.display_height
        return view
    
    def update(self, event):
        ''' Event handler: update view(s) upon change in slider index. '''
        # Get current frame index from slider value
        iframe = self.frame_slider.value
        # Update slider readout
        self.frame_slider_label.value = self.slider_format(iframe - self.frange.start)
        # Update view(s) with current frame
        for i in range(len(self.fpaths)):
            arr = self.get_frame(self.fobjs[i], iframe)
            self.views[i].data = self.process_frame(arr, self.norms[i], iframe)
    
    def get_vbounds(self, x, rel_vbounds):
        ''' Get value bounds from relative value bounds. '''
        vrange = x[1] - x[0]
        lb = x[0] + rel_vbounds[0] * vrange
        ub = x[0] + rel_vbounds[1] * vrange
        return (lb, ub)

    def init_render(self, fps=None, norm='stack', rel_vbounds=None, cmap='viridis', fbounds=None, ilabels=None, playback_speed=1.):
        '''
        Initialize stacks rendering.
        
        :param fps (optional): frame rate (in Hz) used to convert frame index to time (in s)
        :param norm: method used to normalize frame intensity prior to rendering (default: 'stack'):
            - 'frame': render each frame indepdendently using its full dynamic range
            - 'stack': render each frame using the dynamic range of the entire stack 
            - 'all': render each frame using the dynamic range of all stacks
        :param rel_vbounds (optional): relative value bounds used to normalize the stack data.
        :param cmap (optional): colormap used to display grayscale image. If none, a gray colormap is used by default.
        :param fbounds (optional): boundary frame indexes. If none, the entire stack is rendered.
        :param ilabels (optional): dictionary of (label:frame indexes) pairs.
        :param playback_speed (optional): playback speed factor (default = 1.)
        '''
        # Initialize fps and playback speed
        self.fps = fps
        self.playback_speed = playback_speed
        # Get frame range
        self.frange = self.get_frame_index_range(fbounds)
        # Initialize label arrray
        self.labels = [''] * self.nframes
        if ilabels is not None:
            for label, iframes in ilabels.items():
                for iframe in iframes:
                    self.labels[iframe] = label
        self.labels = np.array(self.labels)
        # Initialize colormap
        if cmap is None:
            cmap = 'gray'
        self.cmap = plt.get_cmap(cmap)
        # Initialize normalizers
        if norm in ('stack', 'all'):
            # Get limits for each stack
            lims = [self.get_dynamic_range(i, self.frange) for i in range(len(self.fobjs))]
            # Adapt limits to relative value bounds if provided
            if rel_vbounds is not None:
                lims = [self.get_vbounds(x, rel_vbounds) for x in lims]
            # If norm is 'stack', use different limits for each stack
            if norm == 'stack':
                self.norms = [plt.Normalize(vmin=x[0], vmax=x[1]) for x in lims]
            # If norm is 'all', use the same limits for all stacks 
            else:
                vmin, vmax = np.min([x[0] for x in lims]), np.max([x[1] for x in lims])
                self.norms = [plt.Normalize(vmin=vmin, vmax=vmax)] * len(self.fobjs)
        else:
            self.norms = [None] * len(self.fobjs)
            
    def render(self, *args, **kwargs):
        '''
        Render stacks
        
        :param args: positional arguments passed on to init_render method
        :param kwargs: keyword arguments passed on to init_render method
        :return: VBox ipywidget to render the stacks interactively
        '''
        self.init_render(*args, **kwargs)
        logger.info('rendering stack view...')
        # Initialize views - DO NOT USE LIST COMPREHENSION HERE TO PRESERVE CLOSURE!!!
        self.views = [] 
        for i in range(len(self.fpaths)):
            self.views.append(self.init_view())
        # Set up slider control to change frame
        self.set_frame_slider(self.frange)
        # Connect slider to image view and update view with first frame in range
        self.frame_slider.observe(self.update)
        self.update(None)
        # Return ipywidget
        container = HBox([VBox([self.get_header(h), v]) for h, v in zip(self.headers, self.views)])
        if self.title is not None:
            container = HBox([VBox([self.get_header(self.title), container])])
        return VBox([container, HBox([self.play, self.frame_slider, self.frame_slider_label])])

    def save_as_gif(self, outdir, fps):
        ''' Save stack(s) as GIF(s) '''
        # For each stack object
        for header, fpath, fobj, norm in zip(self.headers, self.fpaths, self.fobjs, self.norms):
            # Fetch output file path from header (if any) or from input file path
            if header is None:
                fcode = os.path.splitext(os.path.basename(fpath))[0]
            else:
                fcode = header
            new_fpath = os.path.join(os.path.abspath(outdir), f'{fcode}.gif')
            # Export stack to gif, frame by frame
            logger.info(f'exporting "{fcode}.gif"...')
            with iio.get_writer(new_fpath, mode='I', duration=1 / fps) as writer:
                def add_frame(iframe, x):
                    writer.append_data(float_to_uint8(self.process_frame(x, norm, iframe)))
                self.iter_frames(fobj, self.frange, add_frame)


def is_s2p_dict(d):
    ''' Determine if dictionary is a suite2p output option dictionary '''
    return isinstance(d, dict) and 'reg_file' in d


def get_stack_viewer(fpaths, *args, **kwargs):
    ''' Interface function to instanciate stack(s) viewers. '''
    # If fpaths if a dictionary but not a suite2p options dictionary
    if isinstance(fpaths, dict) and not is_s2p_dict(fpaths):
        # Extract headers and filepaths from dictionary items
        headers, fpaths = zip(*fpaths.items())
        # Get title from input arguments
        title = kwargs.pop('title', None)
        kwargs['display_size'] = 800 / len(fpaths)
    # If fpaths is a single file instance or a suite2p output options dictionary
    else:
        # Extract header from title and populate single filepaths list
        headers, fpaths = [kwargs.pop('title', None)], [fpaths]
        # Set no title (because single view does not need it)
        title = None
    # Return stack viewer initialized with appropriate fpaths, headers and title
    return StackViewer(fpaths, headers, *args, title=title, **kwargs)


def view_stack(*args, **kwargs):
    ''' Interface function to view stacks '''
    if args[0] is None:
        return
    cmap = kwargs.pop('cmap', 'viridis')
    norm = kwargs.pop('norm', 'stack')
    rel_vbounds = kwargs.pop('rel_vbounds', None)
    fbounds = kwargs.pop('fbounds', None)
    fps = kwargs.pop('fps', None)
    ilabels = kwargs.pop('ilabels', None)
    playback_speed = kwargs.pop('playback_speed', 1.)
    return get_stack_viewer(*args, **kwargs).render(
        fps=fps, norm=norm, rel_vbounds=rel_vbounds, cmap=cmap, fbounds=fbounds, 
        ilabels=ilabels, playback_speed=playback_speed)
