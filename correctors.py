# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 11:59:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-05-01 15:58:33

''' Collection of image stacking utilities. '''

import numpy as np
from scipy.stats import skew
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from tqdm import tqdm
# from scipy.optimize import curve_fit

from logger import logger
from utils import sigmoid, rsquared

from tqdm import tqdm


def mylinregress(x, y, robust=False, intercept=True, return_model=False):
    '''
    Perform robust or standard linear regression between 2 1D arrays

    :param x: independent variable
    :param y: dependent variable
    :param robust: whether to perform robust linear regression
    :param intercept: whether to fit with or without intercept
    :param return_model: whether to return the model object 
        in addition to fit output (default = False)
    :return: fit output as a pandas Series, and optionally the model object
    '''
    # If intercept requested, add constant to input vector
    if intercept:
        x = sm.add_constant(x)
    
    # Construct OLS or RLM linear regression model, depending on "robust" flag
    if robust:
        model = sm.RLM(y, x)
    else:
        model = sm.OLS(y, x)

    # Fit model
    fit = model.fit()

    # Create fit output series
    fit_output = pd.Series()

    # Extract fit parameters (slope and intercept)
    slopeidx = 0
    if intercept:
        fit_output['intercept'] = fit.params[0]
        slopeidx = 1
    else:
        fit_output['intercept'] = 0.
    fit_output['slope'] = fit.params[slopeidx]

    # Extract associated p-value for the slope
    fit_output['pval'] = fit.pvalues[slopeidx]

    # If OLM, extract R-squared value
    if not robust:
        fit_output['r2'] = fit.rsquared
    # Otherwise, compute R-squared value manually
    else:
        fit_output['r2'] = rsquared(y, fit.predict(x))

    # If specified, return fit output and model object
    if return_model:
        return fit_output, fit
    # Otherwise, return fit output
    else:
        return fit_output



class Corrector:

    @property
    def rootcode(self):
        return 'corrected'

    def run(self, stack):
        '''
        Correct image stack.

        :param stack: input image stack
        :return: processed image stack
        '''
        # For multi-channel stack, process each channel separately
        if stack.ndim > 3:
            outstack = []
            for i in range(stack.shape[1]):
                logger.info(f'working on channel {i + 1}...')
                outstack.append(self.run(stack[:, i]))
            outstack = np.stack(outstack)
            return np.swapaxes(outstack, 0, 1)
        
        # Apply correction function, and return
        return self.correct(stack)
    
    def subtract_vector(self, stack, y):
        ''' Detrend stack using a frame-average subtraction vector '''
        assert y.size == stack.shape[0], f'inputs incompatibility:{y.size} and {stack.shape}'
        # Subtract mean from vector
        y -= y.mean()
        # Subtract mean-corrected vector from stack
        return (stack.T - y).T

    def plot(self, y, yfit):
        fig, ax = plt.subplots()
        ax.plot(y, label='data')
        ax.plot(yfit, label='fit')
        ax.plot(y - yfit + yfit.mean(), label='detrended')
        ax.legend()
        for sk in ['top', 'right']:
            ax.spines[sk].set_visible(False)
        plt.show()
    
    @staticmethod
    def get_dtype_bounds(dtype):
        '''
        Get numerical bounds of data type
        
        :param dtype: data type
        :return: minimum and maximum allowed numerical values for data type
        '''
        info_func = np.finfo if np.issubdtype(dtype, np.floating) else np.iinfo
        dinfo = info_func(dtype)
        return dinfo.min, dinfo.max
    
    @classmethod
    def adapt_stack_range(cls, stack, dtype):
        '''
        Adapt stack data range to fit within numerical bounds of reference data type.

        :param stack: image stack array
        :param dtype: reference data type
        :return: adapted stack
        '''
        # Get input data type bounds
        dmin, dmax = cls.get_dtype_bounds(dtype)

        # If values lower than lower bound are found, offset stack accordingly
        if stack.min() < dmin:
            logger.warning(f'values lower than {dmin} found in corrected stack -> offseting')
            stack = stack - stack.min() + dmin + 1

        # If values higher than input data type maximum are found, 
        # rescale stack within bounds
        if stack.max() > dmax:
            logger.warning(f'values higher than {dmax} found in corrected stack -> rescaling')
            ratio = 0.5 * dmax / stack.max()
            stack = stack * ratio

        # Return adapted stack        
        return stack
    
    @classmethod
    def check_stack_range(cls, stack, dtype):
        '''
        Check that stack data range fits within numerical bounds of reference data type.

        :param stack: image stack array
        :param dtype: reference data type
        '''
        # Get input data type numerical bounds
        dmin, dmax = cls.get_dtype_bounds(dtype)
        # Get stack value bounds
        vbounds = stack.min(), stack.max()
        # Check that value bounds fit within data type bounds, raise error otherwise
        for v in vbounds:
            if not dmin <= v <= dmax:
                raise ValueError(f'stack data range {vbounds} is outside of {dtype} data type bounds: {dmin, dmax}') 



class LinRegCorrector(Corrector):

    def __init__(self, robust=False, intercept=True, iref=None, qmin=0, qmax=1, custom=False, wc=None, **kwargs):
        '''
        Initialization

        :param robust: whether or not to use robust linear regression
        :param intercept: whether or not to compute intercept during linear regression fit
        :param iref (optional): frame index range to use to compute reference image
        :param qmin (optional): minimum quantile to use for pixel selection (0-1 float, default: 0)
        :param qmax (optional): maximum quantile to use for pixel selection (0-1 float or "adaptive", default: 1)
        :param custom (optional): whether or not to use custom regressor
        :param wc (optional): normalized cutoff frequency for temporal low-pass filtering of regression parameters
        '''
        # Assign input arguments as attributes
        self.robust = robust
        self.intercept = intercept
        self.iref = iref
        self.qmin = qmin
        self.qmax = qmax
        self.custom = custom
        self.adaptive_qmax = None
        self.wc = wc

        # Initialize empty dictionary of cached reference images
        self.refimg_cache = {}
        
        # Call parent constructor
        super().__init__(**kwargs)
    
    @classmethod
    def from_string(cls, s):
        ''' Instantiate class from string code '''

        # Check if string code is compatible with class
        if not s.startswith('linreg'):
            raise ValueError(f'invalid {cls.__name__} code: "{s}" does not start with "linreg"')
        
        # Split code by underscores
        s = s.split('_')[1:]

        # Define parameters dictionary
        params = {}

        # Extract parameters from code
        for item in s:
            if item == 'robust':
                params['robust'] = True
            elif item == 'nointercept':
                params['intercept'] = False
            elif item.startswith('iref'):
                params['iref'] = range(*[int(i) for i in item[4:].split('-')])
            elif item.startswith('qmin'):
                params['qmin'] = float(item[4:])
            elif item.startswith('qmax'):
                params['qmax'] = item[4:]
                if params['qmax'] != 'adaptive':
                    params['qmax'] = float(params['qmax'])
            elif item == 'custom':
                params['custom'] = True
            elif item.startswith('wc'):
                params['wc'] = float(item[2:])
            else:
                raise ValueError(f'unknown parameter: "{item}"')
        
        # Instantiate class with extracted parameters
        return cls(**params)
        
    @property
    def robust(self):
        return self._robust
    
    @robust.setter
    def robust(self, value):
        if not isinstance(value, bool):
            raise ValueError('robust must be a boolean')
        self._robust = value
    
    @property
    def intercept(self):
        return self._intercept
    
    @intercept.setter
    def intercept(self, value):
        if not isinstance(value, bool):
            raise ValueError('intercept must be a boolean')
        self._intercept = value

    @property
    def iref(self):
        return self._iref
    
    @iref.setter
    def iref(self, value):
        if value is not None and not isinstance(value, range):
            raise ValueError('iref must be a range object')
        self._iref = value
    
    @property
    def qmin(self):
        return self._qmin
    
    @qmin.setter
    def qmin(self, value):
        if not 0 <= value < 1:
            raise ValueError('qmin must be between 0 and 1')
        if hasattr(self, 'qmax') and isinstance(self.qmax, float) and value >= self.qmax:
            raise ValueError(f'qmin must be smaller than qmax ({self.qmax})')    
        self._qmin = value
    
    @property
    def qmax(self):
        return self._qmax
    
    @qmax.setter
    def qmax(self, value):
        if isinstance(value, str):
            if value != 'adaptive':
                raise ValueError(f'invalid qmax string code: "{value}"')
        else:
            if not 0 < value <= 1:
                raise ValueError('qmax must be between 0 and 1')
            if hasattr(self, 'qmin') and value <= self.qmin:
                raise ValueError(f'qmax must be larger than qmin ({self.qmin})')
        self._qmax = value
    
    @property
    def custom(self):
        return self._custom
    
    @custom.setter
    def custom(self, value):
        if not isinstance(value, bool):
            raise ValueError('custom must be a boolean')
        self._custom = value
    
    @property
    def wc(self):
        return self._wc

    @wc.setter
    def wc(self, value):
        if value is not None and (value <= 0 or value >= 1):
            raise ValueError('normalized cutoff frequency must be between 0 and 1')
        self._wc = value
        
    def __repr__(self) -> str:
        plist = [f'robust={self.robust}']
        if not self.intercept:
            plist.append('no intercept')
        if self.iref is not None:
            plist.append(f'iref={self.iref}')
        if self.qmin > 0:
            plist.append(f'qmin={self.qmin}')
        if self.qmax == 'adaptive' or self.qmax < 1:
            plist.append(f'qmax={self.qmax}')
        if self.custom:
            plist.append('custom')
        if self.wc is not None:
            plist.append(f'wc={self.wc}')
        pstr = ', '.join(plist)
        return f'{self.__class__.__name__}({pstr})'
        
    @property
    def code(self):
        clist = []
        if self.robust:
            clist.append('robust')
        if not self.intercept:
            clist.append('nointercept')
        if self.iref is not None:
            clist.append(f'iref_{self.iref.start}_{self.iref.stop - 1}')
        if self.qmin > 0:
            clist.append(f'qmin{self.qmin:.2f}')
        if self.qmax == 'adaptive':
            clist.append('qmaxadaptive')
        elif self.qmax < 1:
            clist.append(f'qmax{self.qmax:.2f}')
        if self.custom:
            clist.append('custom')
        if self.wc is not None:
            clist.append(f'wc{self.wc:.2f}')
        s = 'linreg'
        if len(clist) > 0:
            cstr = '_'.join(clist)
            s = f'{s}_{cstr}'
        return s
    
    def get_reference_frame(self, stack):
        ''' Get reference frame from stack '''
        # If stack ID found in cache, return corresponding reference image
        if id(stack) in self.refimg_cache:
            return self.refimg_cache[id(stack)]
        
        # Otherwise, compute reference image and add it to cache
        if self.iref is not None:
            ibounds = (self.iref.start, self.iref.stop - 1)
            stack = stack[self.iref]
        else:
            ibounds = (0, stack.shape[0] - 1)
        logger.info(
            f'computing ref. image as median of frames {ibounds[0]} - {ibounds[1]}')
        refimg = np.median(stack, axis=0)
        s = skew(refimg.ravel())
        logger.info(f'ref. image skewness: {s:.2f}')
        self.refimg_cache[id(stack)] = refimg

        # Return reference image
        return refimg

    @staticmethod
    def skew_to_qmax(s, zcrit=2, q0=.01, qinf=.99, sigma=1):
        '''
        Function mapping a distribution skewness value to a maximum selection quantile
        
        :param s: distribution skewness value.
        :param zcrit: critical skewness value (i.e. inflexion point of sigmoid)
        :param q0: selection quantile for zero skewness.
        :param qinf: selection quantile for infinite skewness.
        :param sigma: sigmoid steepness parameter.
        :return: maximum selection quantile.
        '''
        return sigmoid(s, x0=zcrit, sigma=sigma, A=qinf - q0, y0=q0)

    def get_qmax(self, frame):
        '''
        Parse qmax value to determine maximum quantile to use for pixel selection

        :param frame: image 2D array
        :return: maximum quantile to use for pixel selection
        '''
        if self.qmax == 'adaptive':
            # If adaptive qmax provided, use it 
            if self.adaptive_qmax is not None:
                return self.adaptive_qmax
            # Otherwise, compute it from reference image skewness
            return self.skew_to_qmax(skew(frame.ravel()))
        else:
            return self.qmax

    def get_pixel_mask(self, img):
        ''' 
        Get selection mask for pixels within quantile range of interest in input image
        
        :param img: image 2D array
        :return: boolean mask of selected pixels that can be used to select pixels
            from an image array by simply using img[mask]
        '''
        # Compute bounding values corresponding to input quantiles 
        qmax = self.get_qmax(img)
        vbounds = np.quantile(img, [self.qmin, qmax])
        
        # Create boolean mask of pixels within quantile range
        mask = np.logical_and(img >= vbounds[0], img <= vbounds[1])
        
        # Log
        logger.info(f'selecting {mask.sum()}/{mask.size} pixels within quantile range {self.qmin:.3f} - {qmax:.3f}')
        
        # Return mask
        return mask
        
    def plot_frame(self, frame, ax=None, mode='img', **kwargs):
        ''' 
        Plot frame image / distribution
        
        :param img: image 2D array
        :param ax: axis to use for plotting (optional)
        :param mode: type of plot to use:
            - "img" for the frame image
            - "dist" for its distribution
            - "all" for both
        :return: figure handle
        '''
        # If mode is "all", create figure with two axes, and 
        # plot both image and its distribution
        if mode == 'all':
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for mode, ax in zip(['img', 'dist'], axes):
                self.plot_frame(frame, ax=ax, mode=mode, **kwargs)
            fig.tight_layout()
            return fig
        
        # Create/retrieve figure and axis 
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        sns.despine(ax=ax)

        # Plot image or distribution
        if mode == 'dist':
            # Flatten image array and convert to pandas DataFrame
            dist = pd.Series(frame.ravel(), name='intensity').to_frame()
            dist['selected'] = True
            hue, hue_order, palette = None, None, None
            # If quantiles are provided, materialize them on the histogram distribution
            if self.qmin > 0:
                vmin = np.quantile(frame, self.qmin)
                dist['selected'] = np.logical_and(dist['selected'], dist['intensity'] >= vmin)
                hue, hue_order = 'selected', [True, False]
                ax.axvline(vmin, color='k', ls='--')
            qmax = self.get_qmax(frame)
            if qmax < 1:
                vmax = np.quantile(frame, qmax)
                dist['selected'] = np.logical_and(dist['selected'], dist['intensity'] <= vmax)
                ax.axvline(vmax, color='k', ls='--')
            if 'selected' in dist:
                hue, hue_order, palette = 'selected', [True, False], {True: 'g', False: 'r'}
            # Plot histogram
            sns.histplot(
                ax=ax, 
                data=dist, 
                x='intensity', 
                hue=hue, 
                hue_order=hue_order,
                palette=palette, 
                bins=100,
                **kwargs
            )
        elif mode == 'img':
            # Plot image
            ax.imshow(frame, cmap='viridis', **kwargs)
            # If quantiles are provided, materialize corresponding selected pixels
            # on the image by making the others slightly transparent
            if self.qmin > 0 or self.get_qmax(frame) < 1:
                mask = self.get_pixel_mask(frame)
                masked = np.ma.masked_where(mask, mask)
                ax.imshow(masked, alpha=.5, cmap='gray_r')
        else:
            raise ValueError(f'unknown plotting mode: {mode}')

        # Return figure handle
        return fig    

    def get_legend_handle(self, kind, color, label):
        if kind == 'scatter':
            return Line2D(
                [0], [0], 
                label=label, 
                linestyle='',
                marker='o', 
                markersize=10, 
                markerfacecolor=color, 
                markeredgecolor='none',
            )
        elif kind == 'hist':
            return mpatches.Patch(color=color, label=label)
        else: 
            raise ValueError(f'unknown plotting kind: {kind}')
    
    def plot_codist(self, refimg, img, ax=None, kind='hist', marginals=False, regres=None,
                    color=None, label=None, height=4, qmax=None, verbose=True):
        ''' 
        Plot co-distribution of pixel intensity in reference and current image
        
        :param refimg: reference image 2D array
        :param img: current image 2D array
        :param ax: axis to use for plotting (optional)
        :param kind: type of plot to use ("hist" or "scatter")
        :param marginals: whether to plot marginal distributions (optional)
        :param regres: linear regression parameters (optional)
        :param height: height of figure (optional). Only used if ax is None.
        :param qmax: maximum quantile to use for plot limits (optional)
        :param verbose: whether or not to log progress (optional)
        :return: figure handle
        '''
        # If single axis provided and marginals are required, raise error
        if marginals and ax is not None:
            raise ValueError('cannot plot marginals on provided axis')
        
        # Create dataframe from flattened images
        df = pd.DataFrame({'reference frame': refimg.ravel(), 'current frame': img.ravel()})

        # Log, if required
        if verbose:
            logger.info(f'plotting {kind} intensity co-distribution of {len(df)} pixels')

        # Create/retrieve figure and ax(es)
        if ax is None:
            # If marginals, create joint grid and extract axes
            if marginals:
                g = sns.JointGrid(height=height)
                ax = g.ax_joint
                axmargx = g.ax_marg_x
                axmargy = g.ax_marg_y
            # Otherwise, create facet grid and extract single axis
            else:
                g = sns.FacetGrid(data=df, height=height)
                ax = g.ax
            # Retrieve figure handle
            fig = g.fig
        else:
            fig = ax.get_figure()
        
        # Determine plotting function
        pltkwargs = dict(color=color)
        if kind == 'hist':
            pltfunc = sns.histplot
        elif kind == 'scatter':
            pltfunc = sns.scatterplot
            pltkwargs.update(dict(s=1, alpha=0.1))
        else:
            raise ValueError(f'unknown plotting kind: {kind}')
        
        # Plot co-distribution
        pltfunc(x=df['reference frame'], y=df['current frame'], ax=ax, **pltkwargs)
        xref = df['reference frame'].mean()

        # Add unit diagonal line
        ax.axline((xref, xref), slope=1, color='k', ls='--')

        # Create and append legend handle, if provided
        if label is not None:
            handle = self.get_legend_handle(kind, color, label)
            if hasattr(ax, 'custom_legend_handles'):
                ax.custom_legend_handles.append(handle)
                ax.legend(handles=ax.custom_legend_handles)
            else:
                ax.custom_legend_handles = [handle]

        # Plot marginals, if required
        if marginals:
            sns.histplot(x=df['reference frame'], ax=axmargx, color=color)
            sns.histplot(y=df['current frame'], ax=axmargy, color=color)

        # Plot linear regression, if provided
        if regres is not None:
            yref = xref * regres['slope'] + regres['intercept']
            ax.axline((xref, yref), slope=regres['slope'], color=color)
        
        # # Make sure to include (0, 0) in plot limits
        # ax.set_xlim(left=0)
        # ax.set_ylim(bottom=0)

        # If maximum quantile is provided, set plot limits accordingly
        if qmax is not None:
            xmax = np.quantile(df['reference frame'], qmax)
            ymax = np.quantile(df['current frame'], qmax)
            ax.set_xlim(right=xmax)
            ax.set_ylim(top=ymax)
        
        # Add grid
        ax.grid(True)
                
        # Return figure handle
        return fig

    def plot_codists(self, stack, iframes, regres=None, height=3, col_wrap=4, axes=None, **kwargs):
        ''' 
        Plot co-distributions of pixel intensity of several stack frames
        with the stack reference image

        :param stack: input image stack
        :param iframes: indices of frames for which to plot co-distributions
        :param regres: linear regression parameters dataframe (optional)
        :return: figure handle
        '''
        # Get reference frame from stack
        refimg = self.get_reference_frame(stack)

        # Select subset of pixels to use for plotting as mask
        mask = self.get_pixel_mask(refimg)

        # Apply mask to reference image
        refimg = refimg[mask]

        # If frame index provied as range object, convert to list
        if isinstance(iframes, range):
            iframes = list(iframes)
        
        # Create/extract figure and axes
        newfig = axes is None
        if axes is None:
            fig = sns.FacetGrid(
                pd.DataFrame({'frame': iframes}), 
                height=height, 
                col='frame', 
                col_wrap=col_wrap
            ).fig
            axes = fig.axes
        else:
            if len(axes) != len(iframes):
                raise ValueError(f'number of provided axes ({len(axes)}) must match number of evaluated frames ({len(iframes)})')
            fig = axes[0].get_figure()

        # Plot co-distributions for each frame of interest
        logger.info(f'plotting intensity co-distribution of {len(iframes)} frames')
        for i, (ax, iframe) in enumerate(zip(axes, tqdm(iframes))):
            self.plot_codist(
                refimg, 
                stack[iframe][mask], 
                ax=ax, 
                regres=None if regres is None else regres.loc[iframe],
                verbose=False,
                label=self.code if i == len(iframes) - 1 else None,
                **kwargs
            )

        # If not new figure, move legend
        if not newfig:
            sns.move_legend(axes[-1], bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add global title 
        fig.suptitle('intensity co-distributions', y=1.05)
        
        # Return figure handle
        return fig
    
    def custom_fit(self, x, y):
        '''
        Fit custom regression model between two vectors

        :param x: reference vector
        :param y: current vector
        :return: regression fit parameters as pandas Series 
        '''
        # slope = ratio of standard deviations
        alpha = y.std() / x.std()
        # intercept = offset between means of rescaled reference frame and current frame 
        beta = y.mean() - alpha * x.mean()
        return pd.Series({
            'slope': alpha, 
            'intercept': beta
        })

    def fit_frame(self, frame, ref_frame, idxs=None):
        '''
        Fit linear regression model between a frame and a reference frame
        
        :param frame: frame 2D array
        :param ref_frame: reference frame 2D array
        :param idxs: serialized indices of pixels to use for regression (optional)
        :return: linear fit parameters as pandas Series
        '''
        x, y = ref_frame.ravel(), frame.ravel()
        if idxs is not None:
            x, y = x[idxs], y[idxs]
        if self.custom:
            return self.custom_fit(x, y)
        else:
            return mylinregress(x, y, robust=self.robust, intercept=self.intercept)
    
    def fit(self, stack, ref_frame=None, npix=None):
        ''' 
        Fit linear regression parameters w.r.t. reference frame for each frame in the stack
        
        :param stack: input image stack
        :param ref_frame: reference frame to use for regression (optional).
        :param npix: number of pixels to use for regression (optional). 
            If None, all pixels are used.
        :return: dataframe of fitted linear regression parameters
        '''
        # Get reference frame from stack, if not provided
        if ref_frame is None:
            ref_frame = self.get_reference_frame(stack)

        # Select subset of pixels to use for regression as mask
        mask = self.get_pixel_mask(ref_frame)

        # If required, select random subset of pixels to use for regression 
        if npix is not None:
            logger.info(f'selecting random {npix} pixels for regression')
            idxs = np.random.choice(ref_frame.size, npix, replace=False)
        else:
            idxs = None
        logger.info(f'performing {"robust " if self.robust else ""}linear regression on {stack.shape[0]} frames')
        
        # Perform fit for each frame
        res = []
        for frame in tqdm(stack):
            res.append(self.fit_frame(
                frame[mask], ref_frame[mask], idxs=idxs))
        
        # Concatenate results into dataframe
        df = pd.concat(res, axis=1).T
        
        # If required, apply temporal low-pass filtering to fit parameters
        if self.wc is not None:
            sos = butter(2, self.wc, btype='low', output='sos')
            for k in df:
                df[k] = sosfiltfilt(sos, df[k], axis=0)
        
        # Return dataframe
        return df
    
    def plot_fit(self, stack, params=None, keys=None, axes=None, periodicity=None, 
                           fps=None, delimiters=None, color=None, height=None, width=None):
        ''' 
        Plot linear regression parameters (along with median frame intensity) over time
        
        :param stack: input image stack
        :param params: dataframe of linear regression parameters (optional)
        :param keys: list of parameters to plot (optional)
        :param axes: list of axes to use for plotting (optional)
        :param periodicity: periodicity index used to aggregate data before plotting (optional)
        :param fps: frame rate (optional)
        :param delimiters: list indices to highlight (optional)
        '''
        # Adjust width and height based on periodicity flag
        if width is None:
            width = 8 if periodicity is None else 5
        if height is None:
            height = 2 if periodicity is None else 1
        
        # If regression parameters not provided, compute them. Otherwise, copy them
        if params is None:
            df = self.fit(stack)
        else:
            df = params.copy()
        
        # If keys provided, select corresponding columns
        if keys is not None:
            df = df[keys]
        
        # Compute median frame (or frame subset) intensity over time 
        if self.qmin > 0 or self.qmax == 'adaptive' or self.qmax < 1:
            mask = self.get_pixel_mask(self.get_reference_frame(stack))
            substack = np.array([frame[mask] for frame in stack])
            ymed = np.median(substack, axis=1)
        else:
            ymed = np.median(stack, axis=(1, 2))
        
        # Add median frame intensity to dataframe
        df.insert(0, 'med. I', ymed)
        
        # Create/retrieve figure and axes
        keys = df.columns
        naxes = len(keys)
        newfig = axes is None
        if axes is None:
            fig, axes = plt.subplots(naxes, 1, figsize=(width, naxes * height))
            sns.despine(fig=fig)
        else:
            if len(axes) != naxes:
                raise ValueError(f'number of axes must match number of parameters + 1 {naxes}')
            fig = axes[0].get_figure()

        # Create index vector
        df['frame'] = np.arange(len(params))
        xlabel = 'frame'

        # Wrap index around periodicity, if provided
        if periodicity is not None:
            df['frame'] = df['frame'] % periodicity
        
        # If frame rate provided, convert index to time 
        if fps is not None:
            df['time (s)'] = df['frame'] / fps
            xlabel = 'time (s)'

        # Plot each timeseries over time 
        for i, (k, ax) in enumerate(zip(keys, axes)):
            sns.lineplot(
                ax=ax, 
                data=df, 
                x=xlabel, 
                y=k, 
                color=color,
                label=self.code if i==0 else None
            )
        
        # Set x-axis label on last axis
        axes[-1].set_xlabel(xlabel)

        # Highlight delimiters, if provided
        if delimiters is not None:
            for ax in axes:
                for d in delimiters:
                    ax.axvline(d, color='k', ls='--')
        
        # If not new figure, create legend on last axis
        if not newfig:
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout
        fig.tight_layout()

        # Add global title
        fig.suptitle('linear regression parameters', y=1.05)

        # Return figure handle
        return fig
    
    def correct(self, stack, regparams=None):
        ''' Correct image stack with linear regresion to reference frame '''
        # Save input data type and cast as float64 for increased precision
        ref_dtype = stack.dtype
        stack = stack.astype(np.float64)

        # Compute linear regression parameters over time, if not provided
        if regparams is None:
            regparams = self.fit(stack)
        else:
            if len(regparams) != stack.shape[0]:
                raise ValueError(
                    f'number of provided regression parameters ({len(regparams)}) does not match stack size ({stack.shape[0]})')
        
        # Extract slopes and intercepts, and reshape to 3D
        slopes = regparams['slope'].values[:, np.newaxis, np.newaxis]
        intercepts = regparams['intercept'].values[:, np.newaxis, np.newaxis]
        
        # Correct stack
        logger.info('correcting stack with linear regression parameters')
        corrected_stack = (stack - intercepts) / slopes

        # If negative values are found, offset stack to obtain only positive values
        if corrected_stack.min() < 0:
            logger.warning('negative values found in corrected stack -> offseting to 1')
            corrected_stack = corrected_stack - corrected_stack.min() + 1

        # Adapt stack range to fit within input data type numerical bounds
        corrected_stack = self.adapt_stack_range(corrected_stack, ref_dtype)

        # Check that corrected stack is within input data type range
        self.check_stack_range(corrected_stack, ref_dtype)

        # If input was integer-typed, round corrected stack to nearest integers 
        if not np.issubdtype(ref_dtype, np.floating):
            logger.info(f'rounding corrected stack')
            corrected_stack = np.round(corrected_stack)
        
        # Cast back to input data type
        corrected_stack = corrected_stack.astype(ref_dtype)

        # Return
        return corrected_stack
