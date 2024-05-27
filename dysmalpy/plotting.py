# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Functions for plotting DYSMALPY kinematic model fit results


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging
import copy
from collections import OrderedDict

import os
import datetime

# Third party imports
import numpy as np
import astropy.units as u
import matplotlib as mpl

# Check if there is a display for plotting, or if there is an SSH/TMUX session.
# If no display, or if SSH/TMUX, use the matplotlib "agg" backend for plotting.
havedisplay = "DISPLAY" in os.environ
if havedisplay:
    exitval = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
    skipconds = (("SSH_CLIENT" in os.environ) | ("TMUX" in os.environ) | ("SSH_CONNECTION" in os.environ) | (os.environ["TERM"].lower().strip()=='screen') | (exitval != 0))
    havedisplay = not skipconds
if not havedisplay:
    mpl.use('agg')


import astropy.modeling as apy_mod
import astropy.io.fits as fits
from astropy.wcs import WCS

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

import matplotlib.colors as mplcolors
from mpl_toolkits.axes_grid1 import ImageGrid, AxesGrid

from matplotlib import colorbar

import corner
from spectral_cube import SpectralCube, BooleanArrayMask


# Package imports
from .utils import calc_pix_position, apply_smoothing_3D, gaus_fit_sp_opt_leastsq, gaus_fit_apy_mod_fitter
from .utils_io import create_vel_profile_files_obs1d, create_vel_profile_files_intrinsic, \
                      read_bestfit_1d_obs_file, read_model_intrinsic_profile
from .observation import Observation
from .aperture_classes import CircApertures
from .data_classes import Data1D, Data2D
from .extern.altered_colormaps import new_diverging_cmap
#from .config import Config_create_model_data, Config_simulate_cube

try:
    from dysmalpy.utils_least_chi_squares_1d_fitter import LeastChiSquares1D
    _loaded_LeastChiSquares1D = True
except:
    _loaded_LeastChiSquares1D = False


# New colormap:
new_diverging_cmap('RdBu_r', diverge = 0.5,
            gamma_lower=1.5, gamma_upper=1.5,
            name_new='RdBu_r_stretch')


# Base colormaps:
try:
    cmap_viridis = mpl.colormap['viridis']
    cmap_spectral_r = mpl.colormap['Spectral_r']
    cmap_greys = mpl.colormaps['Greys']
    cmap_plasma = mpl.colormaps['plasma']
    cmap_rdbu_r = mpl.colormaps["RdBu_r_stretch"]
    cmap_seismic = mpl.colormaps['seismic']
except:
    import matplotlib.cm as cm
    cmap_viridis = cm.viridis
    cmap_spectral_r = mpl.colormaps["Spectral_r"]
    cmap_greys = cm.Greys
    cmap_plasma = cm.plasma
    cmap_rdbu_r = mpl.colormaps["RdBu_r_stretch"]
    cmap_seismic = cm.seismic

    

# Default settings for contours on 2D maps:
_kwargs_contour_defaults = { 'colors_cont': 'black',
                            'alpha_cont': 1.,
                            'ls_cont': 'solid',
                            'lw_cont': 0.75,
                            'delta_cont_v': 25.,
                            'delta_cont_disp': 25.,
                            'delta_cont_flux': 20., #5.,
                            ####
                            'lw_cont_minor': 0.2,
                            'alpha_cont_minor': 1.,
                            'colors_cont_minor': 'black',
                            'ls_cont_minor': '-',
                            'delta_cont_v_minor': None,
                            'delta_cont_disp_minor': None,
                            'delta_cont_flux_minor': None,
                            ####
                            'lw_cont_minor_resid': 0.15,
                            'colors_cont_minor_resid': 'grey',
                            'alpha_cont_minor_resid' : 1.,
                            'ls_cont_minor_resid' : ':',
                            'delta_cont_v_minor_resid': None,
                            'delta_cont_disp_minor_resid': None,
                            'delta_cont_flux_minor_resid': None
                            }



__all__ = ['plot_trace', 'plot_corner', 'plot_bestfit']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

def plot_bestfit(mcmcResults, gal,
                 show_1d_apers=False,
                 fileout=None,
                 vcrop=False,
                 vcrop_value=800.,
                 remove_shift=False,
                 overwrite=False,
                 fill_mask=False,
                 **plot_kwargs):
    """
    Plot data, bestfit model, and residuals from the MCMC fitting.
    """
    plot_data_model_comparison(gal, theta = mcmcResults.bestfit_parameters,
            fileout=fileout,
            vcrop=vcrop, vcrop_value=vcrop_value, show_1d_apers=show_1d_apers,
            remove_shift=remove_shift, fill_mask=fill_mask,
            overwrite=overwrite, **plot_kwargs)

    return None


def plot_trace(bayesianResults, fileout=None, overwrite=False):
    """
    Plot trace of Bayesian samples
    """
    if bayesianResults.fit_method.lower() == 'mcmc':
        plot_trace_mcmc(bayesianResults, fileout=fileout, overwrite=overwrite)
    elif bayesianResults.fit_method.lower() == 'nested':
        plot_trace_nested(bayesianResults, fileout=fileout, overwrite=overwrite)
    else:
        raise ValueError("plot_trace() not supported for fit method: {}".format(bayesianResults.fit_method))

def plot_run(bayesianResults, fileout=None, overwrite=False):

    if bayesianResults.fit_method.lower() == 'nested':
        plot_run_nested(bayesianResults, fileout=fileout, overwrite=overwrite)
    else:
        raise ValueError("plot_run() not supported for fit method: {}".format(bayesianResults.fit_method))


def plot_corner(bayesianResults, gal=None, fileout=None, 
                step_slice=None, blob_name=None, overwrite=False):
    """
    Plot corner plot of Bayesian result posterior distributions.
    Optional:
    step slice: 2 element tuple/array with beginning and end step number to use
    """

    if bayesianResults.fit_method.lower() == 'mcmc':
        plot_corner_mcmc(bayesianResults, gal=gal, fileout=fileout, 
                         step_slice=step_slice, blob_name=blob_name, 
                         overwrite=overwrite)
    elif bayesianResults.fit_method.lower() == 'nested':
        plot_corner_nested(bayesianResults, fileout=fileout, overwrite=overwrite)
    else:
        raise ValueError("plot_corner() not supported for fit method: {}".format(bayesianResults.fit_method))


def plot_trace_mcmc(mcmcResults, fileout=None, overwrite=False):
    """
    Plot trace of MCMC walkers
    """
    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    names = make_clean_bayesian_plot_names(mcmcResults)

    ######################################
    # Setup plot:
    f = plt.figure()
    scale = 1.75
    nRows = len(names)
    nWalkers = mcmcResults.sampler_results['nWalkers']

    f.set_size_inches(4.*scale, nRows*scale)
    gs = gridspec.GridSpec(nRows, 1, hspace=0.2)

    axes = []
    alpha = max(0.01, min(10./nWalkers, 1.))

    # Define random color inds for tracking some walkers:
    nTraceWalkers = 5
    cmap = cmap_viridis
    alphaTrace = 0.8
    lwTrace = 1.5
    trace_inds = np.random.randint(0,nWalkers, size=nTraceWalkers)
    trace_colors = []
    for i in range(nTraceWalkers):
        trace_colors.append(cmap(1./float(nTraceWalkers)*i))

    norm_inds = np.setdiff1d(range(nWalkers), trace_inds)


    for k in range(nRows):
        axes.append(plt.subplot(gs[k,0]))

        axes[k].plot(mcmcResults.sampler_results['chain'][norm_inds,:,k].T, '-', 
                     color='black', alpha=alpha, rasterized=True)

        for j in range(nTraceWalkers):
            axes[k].plot(mcmcResults.sampler_results['chain'][trace_inds[j],:,k].T, '-',
                    color=trace_colors[j], lw=lwTrace, alpha=alphaTrace)


        axes[k].set_ylabel(names[k])

        if k == nRows-1:
            axes[k].set_xlabel('Step number')
        else:
            axes[k].set_xticks([])

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


    return None



def plot_trace_nested(bayesianResults, fileout=None, overwrite=False):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    from dynesty import plotting as dyplot

    names = make_clean_bayesian_plot_names(bayesianResults)

    scale = 1.5
    ndim = bayesianResults.nparams_free
    figsize = (4*scale, ndim*scale)

    # Plot dynesty trace plot:
    f, axes = dyplot.traceplot(bayesianResults.sampler_results, 
                               truths=bayesianResults.bestfit_parameters,
                               labels=names, show_titles=True, 
                               trace_cmap='viridis',
                               title_kwargs={'fontsize': 15, 'y': 1.05}, 
                               quantiles=None,
                               fig=plt.subplots(ndim, 2, figsize=figsize))
    f.tight_layout()

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

    return None


def plot_run_nested(bayesianResults, fileout=None, overwrite=False):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    from dynesty import plotting as dyplot

    f, axes = dyplot.runplot(bayesianResults.sampler_results)
    
    scale = 1.5
    nRows = bayesianResults.nparams_free
    f.set_size_inches(4*scale, nRows*scale)
    f.tight_layout()

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

    return None


def plot_corner_mcmc(mcmcResults, gal=None, fileout=None, step_slice=None, 
                     blob_name=None, overwrite=False):
    """
    Plot corner plot of MCMC result posterior distributions.
    Optional:
            step slice: 2 element tuple/array with beginning and end step number to use
    """
    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    names = make_clean_bayesian_plot_names(mcmcResults)

    if step_slice is None:
        sampler_chain = mcmcResults.sampler_results['flatchain']
    else:
        sampler_chain = mcmcResults.sampler_results['chain'][:,step_slice[0]:step_slice[1],:].reshape((-1, mcmcResults.sampler_results['nParam']))

    truths = mcmcResults.bestfit_parameters

    truths_l68 = mcmcResults.bestfit_parameters_l68_err
    truths_u68 = mcmcResults.bestfit_parameters_u68_err

    try:
        truths_l68_percentile = mcmcResults.bestfit_parameters_l68_err_percentile
        truths_u68_percentile = mcmcResults.bestfit_parameters_u68_err_percentile
    except:
        truths_l68_percentile = None
        truths_u68_percentile = None

    if gal is not None:
        # Get prior locations:
        priors = []

        pfree_dict = gal.model.get_free_parameter_keys()
        comps_names = pfree_dict.keys()
        for compn in comps_names:
            comp = gal.model.components.__getitem__(compn)
            params_names = pfree_dict[compn].keys()
            for paramn in params_names:
                if pfree_dict[compn][paramn] >= 0:
                    # Free parameter:
                    # check if uniform or prior
                    if 'center' in comp.__getattribute__(paramn).prior.__dict__.keys():
                        priors.append(comp.__getattribute__(paramn).prior.center)
                    else:
                        priors.append(None)
    else:
        priors = None

    ###############
    if blob_name is not None:

        names_nice = names[:]

        if isinstance(blob_name, str):
            blob_names = [blob_name]
        else:
            blob_names = blob_name[:]

        for blobn in blob_names:
            blob_true = mcmcResults.__dict__['bestfit_{}'.format(blobn)]
            blob_l68_err = mcmcResults.__dict__['bestfit_{}_l68_err'.format(blobn)]
            blob_u68_err = mcmcResults.__dict__['bestfit_{}_u68_err'.format(blobn)]

            if blobn.lower() == 'fdm':
                names.append('Blob: fDM(RE)')
                names_nice.append(r'$f_{\mathrm{DM}}(R_E)$')
            elif blobn.lower() == 'alpha':
                names.append('Blob: alpha')
                names_nice.append(r'$\alpha$')
            elif blobn.lower() == 'mvirial':
                names.append('Blob: Mvirial')
                names_nice.append(r'$\log_{10}(M_{\rm vir})$')
            elif blobn.lower() == 'rb':
                names.append('Blob: rB')
                names_nice.append(r'$R_B$')
            else:
                names.append(blobn)
                names_nice.append(blobn)

            if isinstance(blob_name, str):
                if step_slice is None:
                    blobs = mcmcResults.sampler_results['flatblobs']
                else:
                    blobs = mcmcResults.sampler_results['blobs'][step_slice[0]:step_slice[1],:,:].reshape((-1, 1))
            else:
                indv = blob_names.index(blobn)
                if step_slice is None:
                    blobs = mcmcResults.sampler_results['flatblobs'][:,indv]
                else:
                    blobs = mcmcResults.sampler_results['blobs'][step_slice[0]:step_slice[1],:,:].reshape((-1, mcmcResults.sampler_results['blobs'].shape[2]))[:,indv]

            sampler_chain = np.concatenate( (sampler_chain, np.array([blobs]).T ), axis=1)

            truths = np.append(truths, blob_true)
            truths_l68 = np.append(truths_l68, blob_l68_err)
            truths_u68 = np.append(truths_u68, blob_u68_err )

            if priors is not None:
                priors.append(None)

            if truths_l68_percentile is not None:
                truths_l68_percentile = np.append(truths_l68_percentile, mcmcResults.__dict__['bestfit_{}_l68_err_percentile'.format(blobn)])
                truths_u68_percentile = np.append(truths_u68_percentile, mcmcResults.__dict__['bestfit_{}_u68_err_percentile'.format(blobn)])



    title_kwargs = {'horizontalalignment': 'left', 'x': 0.}
    fig = corner.corner(sampler_chain,
                            labels=names,
                            title_quantiles=[0.16,0.5,0.84], # Required by corner, will be overwritten later with self-calculated truth/l/u68
                            quantiles= [.02275, 0.15865, 0.84135, .97725], # Plot raw upper, lower 1 and 2 sigma quantiles
                            truths=truths,
                            plot_datapoints=False,
                            show_titles=True,
                            bins=40,
                            plot_contours=True,
                            verbose=False,
                            title_kwargs=title_kwargs)

    axes = fig.axes
    nFreeParam = len(truths)
    for i in range(nFreeParam):
        ax = axes[i*nFreeParam + i]
        # Format the quantile display.
        best = truths[i]
        q_m = truths_l68[i]
        q_p = truths_u68[i]
        title_fmt=".2f"
        fmt = "{{0:{0}}}".format(title_fmt).format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(best), fmt(q_m), fmt(q_p))

        # Add in the column name if it's given.
        if names is not None:
            title = "{0} = {1}".format(names[i], title)
        ax.set_title(title, **title_kwargs)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if truths_l68_percentile is not None:
            if (truths_l68_percentile[i] != q_m) | (truths_u68_percentile[i] != q_p):
                ax.axvline(best-q_m, ls='--', color='#9467bd')   # purple
                ax.axvline(best+q_p, ls='--', color='#9467bd')   # purple

        if priors is not None:
            if priors[i] is not None:
                ax.axvline(priors[i], ls=':', color='#ff7f0e')   # orange

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if priors is not None:
        for i in range(nFreeParam):
            for j in range(nFreeParam):
                # need the off-diagonals:
                if j >= i:
                    pass
                else:
                    ax = axes[i*nFreeParam + j]
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    if priors[i] is not None:
                        ax.axhline(priors[i], ls=':', color='#ff7f0e') # orange
                    if priors[j] is not None:
                        ax.axvline(priors[j], ls=':', color='#ff7f0e') # orange
                    if (priors[i] is not None) & (priors[j] is not None):
                        ax.scatter([priors[j]], [priors[i]], marker='s', edgecolor='#ff7f0e', facecolor='None') # orange
                    #
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return None



def plot_corner_nested(bayesianResults, fileout=None, overwrite=False):

    from dynesty import plotting as dyplot

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    MAP_vals = bayesianResults.bestfit_parameters

    ndim = bayesianResults.nparams_free
    names = make_clean_bayesian_plot_names(bayesianResults, short=True)

    # Plot dynesty corner plot:
    # initialize figure
    scale = 1.5
    f, axes = plt.subplots(ndim, ndim, figsize=(scale*ndim,scale*ndim))
    axes = axes.reshape((ndim, ndim))

    fg, ax = dyplot.cornerplot(bayesianResults.sampler_results, color='blue',
                               truths=MAP_vals,
                               #span=[(0,1), (0,1), (0,1), (0,1)],
                               labels=names,
                               show_titles=True, title_kwargs={'y': 1.05},
                               quantiles=None, fig=(f, axes[:, :]))

    title_fmt = '.2f'
    for i in range(ndim):
        axi = ax[i,i]
        qm = MAP_vals[i]

        ql = bayesianResults.bestfit_parameters_l68[i]
        qh = bayesianResults.bestfit_parameters_u68[i]

        q_minus, q_plus = qm - ql, qh - qm
        fmt = "{{0:{0}}}".format(title_fmt).format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
        title = "{0} = {1}".format(names[i], title)
        axi.set_title(title)

    #############################################################
    # Save to file:

    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()

    return None



# -------------------------------------------------------------

def plot_data_model_comparison(gal,theta=None,
                               fileout=None,
                               vcrop=False,
                               show_1d_apers=False,
                               vcrop_value=800.,
                               remove_shift=False,
                               overwrite=False,
                               fill_mask=False,
                               show_contours=False,
                               show_ruler=True,
                               ruler_loc='lowerleft',
                               **plot_kwargs):


    dummy_gal = copy.deepcopy(gal)

    if remove_shift:
        dummy_gal.model.geometries[obs.name].xshift = 0
        dummy_gal.model.geometries[obs.name].yshift = 0

    # Update model parameters and create new set of model data if
    # theta array provided
    if theta is not None:
        dummy_gal.model.update_parameters(theta)
        dummy_gal.create_model_data()

    # Plot data and model for each observation
    for obs_name in dummy_gal.observations:
        obs = dummy_gal.observations[obs_name]

        plot_single_obs_data_model_comparison(obs, dummy_gal.model,
                                       fileout=fileout,
                                       dscale=dummy_gal.dscale,
                                       vcrop=vcrop,
                                       show_1d_apers=show_1d_apers,
                                       vcrop_value=vcrop_value,
                                       overwrite=overwrite,
                                       fill_mask=fill_mask,
                                       show_contours=show_contours,
                                       show_ruler=show_ruler,
                                       ruler_loc=ruler_loc,
                                       **plot_kwargs)


def plot_single_obs_data_model_comparison(obs, model, theta = None,
                               fileout=None,
                               dscale=None,
                               vcrop=False,
                               show_1d_apers=False,
                               vcrop_value=800.,
                               remove_shift=False,
                               overwrite=False,
                               fill_mask=False,
                               show_contours=False,
                               show_ruler=True,
                               ruler_loc='lowerleft',
                               **plot_kwargs):
    """
    Plot data, model, and residuals between the data and this model.
    """

    dummy_obs = copy.deepcopy(obs)
    dummy_model = copy.deepcopy(model)

    if remove_shift:
        dummy_model.geometries[obs.name].xshift = 0
        dummy_model.geometries[obs.name].yshift = 0

    if fileout is not None:
        fileout_in = fileout
        f_r = fileout.split('.')
        f_base = ".".join(f_r[:-1])
        fileout = "{}_{}.{}".format(f_base, obs.name, f_r[-1])

        if (not overwrite) & os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))

    if dummy_obs.instrument.ndim == 1:
        plot_data_model_comparison_1D(dummy_obs, fileout=fileout,
                    overwrite=overwrite,
                    **plot_kwargs)
    elif dummy_obs.instrument.ndim == 2:
        plot_data_model_comparison_2D(dummy_obs, dummy_model,
                    fileout=fileout,
                    show_contours=show_contours,
                    show_ruler=show_ruler,
                    ruler_loc=ruler_loc,
                    overwrite=overwrite,
                    **plot_kwargs)
    elif dummy_obs.instrument.ndim == 3:
        plot_data_model_comparison_3D(dummy_obs, dummy_model,
                    show_1d_apers=show_1d_apers,
                    fileout=fileout,
                    dscale=dscale,
                    vcrop=vcrop,
                    vcrop_value=vcrop_value,
                    overwrite=overwrite,
                    fill_mask=fill_mask,
                    show_contours=show_contours,
                    show_ruler=show_ruler,
                    ruler_loc=ruler_loc,
                    **plot_kwargs)

    elif dummy_obs.instrument.ndim == 0:
        plot_data_model_comparison_0D(dummy_obs, fileout=fileout,
                                      overwrite=overwrite,
                                      **plot_kwargs)
    else:
        logger.warning("nDim="+str(dummy_obs.instrument.ndim)+" not supported!")
        raise ValueError("nDim="+str(dummy_obs.instrument.ndim)+" not supported!")

    return None


def plot_data_model_comparison_0D(obs, fileout=None,
        overwrite=False):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None


    data = obs.data
    model_data = obs.model_data

    ######################################
    # Setup plot:
    scale = 3.5
    f = plt.figure(figsize=(2.2 * scale, scale))
    ax = f.add_subplot(111)

    # Plot the observed spectrum with error shading
    ax.plot(data.x, data.data, color='black', lw=1.5)
    ax.fill_between(data.x, data.data - data.error, data.data + data.error, color='black', alpha=0.2)

    # Plot the model spectrum
    ax.plot(model_data.x, model_data.data, color='red', lw=1.5)

    # Plot the residuals
    ax.plot(model_data.x, data.data - model_data.data, color='blue', lw=1.0)

    ax.set_ylabel('Normalized Flux')
    ax.set_xlabel(obs.instrument.spec_type.capitalize() + ' [' + obs.instrument.spec_step.unit.name + ']')

    f.suptitle(obs.name)

    # Save to file:
    if fileout is not None:
        f.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close(f)
    else:
        plt.show()


def plot_data_model_comparison_1D(obs, fileout=None, overwrite=False):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    # Default: fit in 1D, compare to 1D data:
    data = obs.data
    model_data = obs.model_data

    # Correct model for instrument dispersion if the data is instrument corrected:

    if 'inst_corr' in data.data.keys():
        inst_corr = data.data['inst_corr']


    if inst_corr:
            model_data.data['dispersion'] = \
                np.sqrt( model_data.data['dispersion']**2 - \
                    obs.instrument.lsf.dispersion.to(u.km/u.s).value**2 )


    ######################################
    # Setup plot:
    f = plt.figure()
    scale = 3.5
    ncols = 0
    for cond in [obs.fit_options.fit_flux, obs.fit_options.fit_velocity, obs.fit_options.fit_dispersion]:
        if cond:
            ncols += 1
    nrows = 2
    f.set_size_inches(1.1*ncols*scale, nrows*scale)
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.35, hspace=0.2)


    keyxtitle = r'$r$ [arcsec]'
    keyyarr, keyytitlearr, keyyresidtitlearr = ([] for _ in range(3))
    if obs.fit_options.fit_flux:
        keyyarr.append('flux')
        keyytitlearr.append(r'Flux [arb]')
        keyyresidtitlearr.append(r'$\mathrm{Flux_{data} - Flux_{model}}$ [arb]')
    if obs.fit_options.fit_velocity:
        keyyarr.append('velocity')
        keyytitlearr.append(r'$V$ [km/s]')
        keyyresidtitlearr.append(r'$V_{\mathrm{data}} - V_{\mathrm{model}}$ [km/s]')
    if obs.fit_options.fit_dispersion:
        keyyarr.append('dispersion')
        keyytitlearr.append(r'$\sigma$ [km/s]')
        keyyresidtitlearr.append(r'$\sigma_{\mathrm{data}} - \sigma_{\mathrm{model}}$ [km/s]')

    errbar_lw = 0.5
    errbar_cap = 1.5

    axes = []
    k = -1
    for j in range(ncols):
        # Comparison:
        axes.append(plt.subplot(gs[0,j]))
        k += 1

        if keyyarr[j] == 'velocity':
            if hasattr(obs.data, 'mask_velocity'):
                if obs.data.mask_velocity is not None:
                    msk = obs.data.mask_velocity
                else:
                    msk = obs.data.mask
            else:
                msk = obs.data.mask
        elif keyyarr[j] == 'dispersion':
            if hasattr(obs.data, 'mask_vel_disp'):
                if obs.data.mask_vel_disp is not None:
                    msk = obs.data.mask_vel_disp
                else:
                    msk = obs.data.mask
            else:
                msk = obs.data.mask
        elif keyyarr[j] == 'flux':
            msk = obs.data.mask
        else:
            msk = np.array(np.ones(len(data.rarr)), dtype=bool)

        # Masked points
        if data.error[keyyarr[j]] is not None:
            axes[k].errorbar( data.rarr[~msk], data.data[keyyarr[j]][~msk],
                    xerr=None, yerr=data.error[keyyarr[j]][~msk],
                    marker=None, ls='None', ecolor='darkgrey', zorder=-1.,
                    lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None, alpha=0.5 )
        if np.any(~msk):
            lbl_mask = 'Masked data'
        else:
            lbl_mask = None
        axes[k].scatter( data.rarr[~msk], data.data[keyyarr[j]][~msk],
            c='darkgrey', marker='o', s=25, lw=1, label=lbl_mask, alpha=0.5)

        # Unmasked points
        if data.error[keyyarr[j]] is not None:
            axes[k].errorbar( data.rarr[msk], data.data[keyyarr[j]][msk],
                    xerr=None, yerr=data.error[keyyarr[j]][msk],
                    marker=None, ls='None', ecolor='k', zorder=-1.,
                    lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )

            # Weights: effective errorbars show in blue, for reference
            if hasattr(data, 'weight'):
                if obs.data.weight is not None:
                    wgt = obs.data.weight
                    axes[k].errorbar( data.rarr[msk], data.data[keyyarr[j]][msk],
                            xerr=None, yerr=data.error[keyyarr[j]][msk]/np.sqrt(wgt[msk]),
                            marker=None, ls='None', ecolor='blue', zorder=-1.,
                            lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )

        axes[k].scatter( data.rarr[msk], data.data[keyyarr[j]][msk],
            c='black', marker='o', s=25, lw=1, label='Data')


        # Masked points
        axes[k].scatter( model_data.rarr[~msk], model_data.data[keyyarr[j]][~msk],
            c='lightsalmon', marker='s', s=25, lw=1, label=None, alpha=0.3)

        # Unmasked points
        axes[k].scatter( model_data.rarr[msk], model_data.data[keyyarr[j]][msk],
            c='red', marker='s', s=25, lw=1, label='Model')


        axes[k].set_xlabel(keyxtitle)
        axes[k].set_ylabel(keyytitlearr[j])
        axes[k].axhline(y=0, ls='--', color='k', zorder=-10.)

        if k == 0:
            handles, lbls = axes[k].get_legend_handles_labels()
            if len(lbls) > 2:
                ind_reord = [1,0,2]
                labels_arr = []
                handles_arr = []
                for ir in ind_reord:
                    labels_arr.append(lbls[ir])
                    handles_arr.append(handles[ir])
            else:
                labels_arr = lbls
                handles_arr = handles
            frameon = True
            borderpad = 0.25
            markerscale = 0.8 #1.
            labelspacing= 0.25
            handletextpad = 0.2
            handlelength = 1.
            fontsize_leg= 7.5
            legend = axes[k].legend(handles_arr, labels_arr,
                labelspacing=labelspacing, borderpad=borderpad,
                markerscale=markerscale,
                handletextpad=handletextpad,
                handlelength=handlelength,
                loc='lower right',
                frameon=frameon, numpoints=1,
                scatterpoints=1,
                fontsize=fontsize_leg)


        # Residuals:
        axes.append(plt.subplot(gs[1,j]))
        k += 1

        # Masked points
        if data.error[keyyarr[j]] is not None:
            axes[k].errorbar( data.rarr[~msk], data.data[keyyarr[j]][~msk]-model_data.data[keyyarr[j]][~msk],
                    xerr=None, yerr = data.error[keyyarr[j]][~msk],
                    marker=None, ls='None', ecolor='darkgrey', zorder=-1.,
                    lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None, alpha=0.5 )
        axes[k].scatter( data.rarr[~msk], data.data[keyyarr[j]][~msk]-model_data.data[keyyarr[j]][~msk],
            c='lightsalmon', marker='s', s=25, lw=1, label=None, alpha=0.3)

        # Unmasked points:
        if data.error[keyyarr[j]] is not None:
            axes[k].errorbar( data.rarr[msk], data.data[keyyarr[j]][msk]-model_data.data[keyyarr[j]][msk],
                    xerr=None, yerr = data.error[keyyarr[j]][msk],
                    marker=None, ls='None', ecolor='k', zorder=-1.,
                    lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )

            # Weights: effective errorbars show in blue, for reference
            if hasattr(data, 'weight'):
                if obs.data.weight is not None:
                    wgt = obs.data.weight
                    axes[k].errorbar( data.rarr[msk], data.data[keyyarr[j]][msk]-model_data.data[keyyarr[j]][msk],
                            xerr=None, yerr = data.error[keyyarr[j]][msk]/np.sqrt(wgt[msk]),
                            marker=None, ls='None', ecolor='blue', zorder=-1.,
                            lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
        #
        axes[k].scatter( data.rarr[msk], data.data[keyyarr[j]][msk]-model_data.data[keyyarr[j]][msk],
            c='red', marker='s', s=25, lw=1, label=None)


        axes[k].axhline(y=0, ls='--', color='k', zorder=-10.)
        axes[k].set_xlabel(keyxtitle)
        axes[k].set_ylabel(keyyresidtitlearr[j])

    # Figure title is the observation name
    f.suptitle(obs.name)

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_data_model_comparison_2D(obs, model,
            fileout=None,
            symmetric_residuals=True,
            max_residual=100.,
            max_residual_flux=None,
            overwrite=False,
            show_contours=False,
            show_ruler=True,
            ruler_loc='lowerleft',
            **plot_kwargs):

    if show_contours:
        # Set contour defaults, if not specifed:
        for key in _kwargs_contour_defaults.keys():
            if key not in plot_kwargs.keys():
                plot_kwargs[key] = _kwargs_contour_defaults[key]

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    try:
        if 'inst_corr' in obs.data.data.keys():
            inst_corr = obs.data.data['inst_corr']
    except:
        inst_corr = False

    ######################################
    # Setup plot:
    f = plt.figure(figsize=(9.5, 6))
    scale = 3.5

    # Plot settings:
    nrows_ncols=(1, 3)
    direction="row"
    axes_pad=0.5
    label_mode="1"
    share_all=True
    cbar_location="right"
    cbar_mode="each"
    cbar_size="5%"
    cbar_pad="1%"

    nrows = 0
    for cond in [obs.fit_options.fit_flux, obs.fit_options.fit_velocity, obs.fit_options.fit_dispersion]:
        if cond:
            nrows += 1

    cntr = 0
    if obs.fit_options.fit_flux:
        cntr += 1
        grid_flux = ImageGrid(f, 100*nrows+10+cntr,
                             nrows_ncols=nrows_ncols,
                             direction=direction,
                             axes_pad=axes_pad,
                             label_mode=label_mode,
                             share_all=share_all,
                             cbar_location=cbar_location,
                             cbar_mode=cbar_mode,
                             cbar_size=cbar_size,
                             cbar_pad=cbar_pad)
    if obs.fit_options.fit_velocity:
        cntr += 1
        grid_vel = ImageGrid(f, 100*nrows+10+cntr,
                             nrows_ncols=nrows_ncols,
                             direction=direction,
                             axes_pad=axes_pad,
                             label_mode=label_mode,
                             share_all=share_all,
                             cbar_location=cbar_location,
                             cbar_mode=cbar_mode,
                             cbar_size=cbar_size,
                             cbar_pad=cbar_pad)
    if obs.fit_options.fit_dispersion:
        cntr += 1
        grid_disp = ImageGrid(f, 100*nrows+10+cntr,
                             nrows_ncols=nrows_ncols,
                             direction=direction,
                             axes_pad=axes_pad,
                             label_mode=label_mode,
                             share_all=share_all,
                             cbar_location=cbar_location,
                             cbar_mode=cbar_mode,
                             cbar_size=cbar_size,
                             cbar_pad=cbar_pad)


    keyxarr = ['data', 'model', 'residual']
    keyxtitlearr = ['Data', 'Model', 'Residual']
    keyyarr, keyytitlearr, grid_arr, annstr_arr = ([] for _ in range(4))
    if obs.fit_options.fit_flux:
        keyyarr.append('flux')
        keyytitlearr.append(r'Flux')
        grid_arr.append(grid_flux)
        annstr_arr.append('f')

        if obs.data is not None:
            flux_vmin = obs.data.data['flux'][obs.data.mask].min()
            flux_vmax = obs.data.data['flux'][obs.data.mask].max()
            if flux_vmin == flux_vmax:
                flux_vmin = obs.model_data.data['flux'].min()
                flux_vmax = obs.model_data.data['flux'].max()
        else:
            flux_vmin = obs.model_data.data['flux'].min()
            flux_vmax = obs.model_data.data['flux'].max()
        if max_residual_flux is None:
            max_residual_flux = np.max(np.abs(obs.data.data['flux'][obs.data.mask]))

    if obs.fit_options.fit_velocity:
        keyyarr.append('velocity')
        keyytitlearr.append(r'$V$')
        grid_arr.append(grid_vel)
        annstr_arr.append('V')

        vel_vmin = obs.data.data['velocity'][obs.data.mask].min()
        vel_vmax = obs.data.data['velocity'][obs.data.mask].max()

        try:
            vel_shift = model.geometries[obs.name].vel_shift.value
        except:
            vel_shift = 0

        vel_vmin -= vel_shift
        vel_vmax -= vel_shift

    if obs.fit_options.fit_dispersion:
        keyyarr.append('dispersion')
        keyytitlearr.append(r'$\sigma$')
        grid_arr.append(grid_disp)
        annstr_arr.append('\sigma')

        if obs.data is not None:
            disp_vmin = obs.data.data['dispersion'][obs.data.mask].min()
            disp_vmax = obs.data.data['dispersion'][obs.data.mask].max()
        else:
            disp_vmin = obs.model_data.data['dispersion'].min()
            disp_vmax = obs.model_data.data['dispersion'].max()



    int_mode = "nearest"
    origin = 'lower'
    cmap = copy.copy(cmap_spectral_r)

    # color_bad = 'black'
    # color_annotate = 'white'
    color_bad = 'white'
    color_annotate = 'black'

    cmap.set_bad(color=color_bad)

    cmap_resid = copy.copy(cmap_rdbu_r)
    cmap_resid.set_bad(color=color_bad)



    for j in range(len(keyyarr)):
        grid = grid_arr[j]

        for ax, k, xt in zip(grid, keyxarr, keyxtitlearr):

            if k == 'data':
                im = obs.data.data[keyyarr[j]].copy()

                if keyyarr[j] == 'velocity':
                    im -= vel_shift
                    vmin = vel_vmin
                    vmax = vel_vmax
                elif keyyarr[j] == 'dispersion':
                    vmin = disp_vmin
                    vmax = disp_vmax
                elif keyyarr[j] == 'flux':
                    vmin = flux_vmin
                    vmax = flux_vmax

                cmaptmp = cmap
            elif k == 'model':
                im = obs.model_data.data[keyyarr[j]].copy()
                if keyyarr[j] == 'velocity':
                    im -= vel_shift
                    vmin = vel_vmin
                    vmax = vel_vmax
                elif keyyarr[j] == 'dispersion':
                    if inst_corr:
                        im = np.sqrt(im ** 2 - obs.instrument.lsf.dispersion.to(
                                     u.km / u.s).value ** 2)
                    vmin = disp_vmin
                    vmax = disp_vmax
                elif keyyarr[j] == 'flux':
                    vmin = flux_vmin
                    vmax = flux_vmax

                cmaptmp = cmap
            elif k == 'residual':
                im_data = obs.data.data[keyyarr[j]].copy()
                im_model = obs.model_data.data[keyyarr[j]].copy()
                if keyyarr[j] == 'dispersion':
                    if inst_corr:
                        im_model = np.sqrt(im_model ** 2 -
                                       obs.instrument.lsf.dispersion.to( u.km / u.s).value ** 2)
                im = im_data - im_model
                if symmetric_residuals:
                    if keyyarr[j] == 'flux':
                        vmin = -max_residual_flux
                        vmax = max_residual_flux
                    else:
                        vmin = -max_residual
                        vmax = max_residual

                cmaptmp = cmap_resid
            else:
                raise ValueError("key not supported.")



            # Mask image:
            im[~obs.data.mask] = np.nan

            imax = ax.imshow(im, cmap=cmaptmp, interpolation=int_mode,
                             vmin=vmin, vmax=vmax, origin=origin)
            if len(model.geometries) > 0:
                ax = plot_major_minor_axes_2D(ax, obs, model, im, obs.data.mask)
            if show_ruler:
                pixscale = obs.instrument.pixscale.value
                ax = plot_ruler_arcsec_2D(ax, pixscale, len_arcsec=1.,
                                      ruler_loc=ruler_loc, color=color_annotate)

            if show_contours:
                ax = plot_contours_2D_multitype(im, ax=ax, mapname=keyyarr[j], plottype=k,
                            vmin=vmin, vmax=vmax, kwargs=plot_kwargs)


            if k == 'data':
                ax.set_ylabel(keyytitlearr[j])
            for pos in ['top', 'bottom', 'left', 'right']:
                ax.spines[pos].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            if j == 0:
                ax.set_title(xt)

            ####
            if k == 'residual':
                med = np.median(im[obs.data.mask])
                rms = np.std(im[obs.data.mask])
                median_str  = r"${}".format(annstr_arr[j])+r"_{med}="+r"{:0.1f}".format(med)+r"$"
                scatter_str = r"${}".format(annstr_arr[j])+r"_{rms}="+r"{:0.1f}".format(rms)+r"$"
                ax.annotate(median_str,
                    (0.01,-0.05), xycoords='axes fraction',
                    ha='left', va='top', fontsize=8)
                ax.annotate(scatter_str,
                    (0.99,-0.05), xycoords='axes fraction',
                    ha='right', va='top', fontsize=8)

            cbar = ax.cax.colorbar(imax)
            cbar.ax.tick_params(labelsize=8)


    # Figure title is the observation name
    f.suptitle(obs.name)

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()


def plot_data_model_comparison_3D(obs, model,
            theta = None,
            fileout=None,
            dscale=None,
            symmetric_residuals=True,
            show_1d_apers = False,
            max_residual=100.,
            inst_corr = True,
            vcrop=False,
            vcrop_value=800.,
            overwrite=False,
            moment=False,
            remove_shift=False,
            fill_mask=False,
            show_contours=False,
            show_ruler=True,
            ruler_loc='lowerleft',
            **plot_kwargs):

    if fileout is not None:
        fileout_in = fileout
        f_r = fileout.split('.')
        f_base = ".".join(f_r[:-1])
        fileout_aperture = "{}_apertures.{}".format(f_base, f_r[-1])
        fileout_spaxel = "{}_spaxels.{}".format(f_base, f_r[-1])
        fileout_channel = "{}_channels.{}".format(f_base, f_r[-1])
    else:
        fileout_aperture = fileout_spaxel = fileout_channel = None

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        for f in [fileout, fileout_aperture, fileout_spaxel, fileout_channel]:
            if os.path.isfile(f):
                logger.warning("overwrite={} & File already exists! Will not save file: {} \n ".format(overwrite, f))
                return None

    plot_3D_extracted_to_1D_2D(obs, model, fileout=fileout, dscale=dscale,
                symmetric_residuals=symmetric_residuals,
                max_residual=max_residual, show_1d_apers=show_1d_apers,
                inst_corr=True, remove_shift=False,
                vcrop=vcrop, vcrop_value=vcrop_value,
                overwrite=overwrite, fill_mask=fill_mask,
                show_ruler=show_ruler, ruler_loc=ruler_loc,
                show_contours=show_contours, **plot_kwargs)

    plot_spaxel_compare_3D_cubes(obs, fileout=fileout_spaxel, typ='all',
                                 show_model=True, overwrite=overwrite)

    plot_aperture_compare_3D_cubes(obs, model, fileout=fileout_aperture,
                                   fill_mask=fill_mask, overwrite=overwrite)

    plot_channel_maps_3D_cube(obs, model, show_data=True, show_model=True,
                              show_residual=True, fileout=fileout_channel,
                              vbounds = [-450., 450.], delv = 100.,
                              vbounds_shift=True, cmap=cmap_greys,
                              overwrite=overwrite)



    return None


#############################################################
#############################################################


def plot_3D_extracted_to_1D_2D(obs_in, model_in,
            fileout=None,
            dscale=None,
            symmetric_residuals=True, max_residual=100.,
            show_1d_apers=False, inst_corr=None,
            xshift = None,
            yshift = None,
            vcrop=False,
            vcrop_value=800.,
            remove_shift=False,
            overwrite=False,
            fill_mask=False,
            show_contours=False,
            show_ruler=True,
            ruler_loc='lowerleft',
            **plot_kwargs):


    obs = copy.deepcopy(obs_in)
    model = copy.deepcopy(model_in)

    print("plot_model_multid: ndim=3: moment={}".format(obs.instrument.moment))
    obs_extract, model = extract_1D_2D_obs_from_cube(obs, model, inst_corr=True,
                                               fill_mask=fill_mask)


    if fileout is not None:
        f_r = fileout.split('.')
        f_base = ".".join(f_r[:-1])
        fileout_1D = "{}_{}.{}".format(f_base, "extract_1D", f_r[-1])
        fileout_2D = "{}_{}.{}".format(f_base, "extract_2D", f_r[-1])
    else:
        fileout_1D = fileout_2D = None

    # Data haven't actually been corrected for instrument LSF yet
    # (Note: 1D/2D *models* will be corrected for LSF during plotting,
    #        based on the data['inst_corr'] setting)
    if obs_extract['extract_1D'].data.data['inst_corr']:
        inst_corr_sigma = obs_extract['extract_1D'].instrument.lsf.dispersion.to(u.km/u.s).value
        disp_prof_1D = np.sqrt(obs_extract['extract_1D'].data.data['dispersion']**2 - inst_corr_sigma**2 )
        disp_prof_1D[~np.isfinite(disp_prof_1D)] = 0.
        obs_extract['extract_1D'].data.data['dispersion'] = disp_prof_1D

        if 'filled_mask_data' in obs_extract['extract_1D'].data.__dict__.keys():
            disp_prof_1D = np.sqrt(obs_extract['extract_1D'].data.filled_mask_data.data['dispersion']**2 - inst_corr_sigma**2 )
            disp_prof_1D[~np.isfinite(disp_prof_1D)] = 0.
            obs_extract['extract_1D'].data.filled_mask_data.data['dispersion'] = disp_prof_1D


    if obs_extract['extract_2D'].data.data['inst_corr']:
        inst_corr_sigma = obs_extract['extract_2D'].instrument.lsf.dispersion.to(u.km/u.s).value
        im = obs_extract['extract_2D'].data.data['dispersion'].copy()
        im = np.sqrt(im ** 2 - inst_corr_sigma ** 2)
        im[~np.isfinite(im)] = 0.
        obs_extract['extract_2D'].data.data['dispersion'] = im


    plot_data_model_comparison_1D(obs_extract['extract_1D'], fileout=fileout_1D,
                                  overwrite=overwrite, **plot_kwargs)


    plot_data_model_comparison_2D(obs_extract['extract_2D'], model,
                                  fileout=fileout_2D, show_contours=show_contours,
                                  show_ruler=show_ruler, ruler_loc=ruler_loc,
                                  overwrite=overwrite, **plot_kwargs)

    return None




#############################################################

def plot_aperture_compare_3D_cubes(obs, model, datacube=None, errcube=None,
                                   modelcube=None, mask=None,
                                   fileout=None,
                                   slit_width=None, slit_pa=None,
                                   aper_dist=None, fill_mask=False,
                                   skip_fits=True, overwrite=False):


    #############################################################

    if datacube is None:
        datacube = obs.data.data
    if errcube is None:
        errcube = obs.data.error
    if modelcube is None:
        modelcube = obs.model_data.data
    if mask is None:
        mask = obs.data.mask

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    ######################################

    if slit_width is None:
        try:
            slit_width = obs.instrument.beam.major.to(u.arcsec).value
        except:
            slit_width = obs.instrument.beam.major_fwhm.to(u.arcsec).value
    if slit_pa is None:
        slit_pa = model.geometries[obs.name].pa.value

    if mask is None:
        mask = obs.data.mask.copy()

    pixscale = obs.instrument.pixscale.value

    rpix = slit_width/pixscale/2.

    # Aper centers: pick roughly number fitting into size:
    nx = datacube.shape[2]
    ny = datacube.shape[1]
    try:
        center_pixel = (obs.mod_options.xcenter + model.geometries[obs.name].xshift.value,
                        obs.mod_options.ycenter + model.geometries[obs.name].yshift.value)
    except:
        #center_pixel = (int(nx / 2.) + model.geometries[obs.name].geometry.xshift,
        #                int(ny / 2.) + model.geometries[obs.name].yshift)
        center_pixel = (int(nx / 2.) + model.geometries[obs.name].xshift.value,
                        int(ny / 2.) + model.geometries[obs.name].yshift.value)


    aper_centers_arcsec = aper_centers_arcsec_from_cube(datacube, obs, model,
                mask=mask, slit_width=slit_width, slit_pa=slit_pa,
                aper_dist=aper_dist, fill_mask=fill_mask)


    #############################################################

    specarr = datacube.spectral_axis.to(u.km/u.s).value

    apertures = CircApertures(rarr=aper_centers_arcsec, slit_PA=slit_pa, rpix=rpix,
             nx=nx, ny=ny, center_pixel=center_pixel, pixscale=pixscale, moment=False)

    if not skip_fits:
        data_scaled = datacube.unmasked_data[:].value
        if errcube is not None:
            ecube =  errcube.unmasked_data[:].value * mask
        else:
            ecube = None
        model_scaled = modelcube.unmasked_data[:].value

        aper_centers, flux1d, vel1d, disp1d = apertures.extract_1d_kinematics(spec_arr=specarr,
                        cube=data_scaled, mask=mask, err=ecube,
                        center_pixel = center_pixel, pixscale=pixscale)

        #####
        aper_centers_mod, flux1d_mod, vel1d_mod, disp1d_mod = apertures.extract_1d_kinematics(spec_arr=specarr,
                        cube=model_scaled, mask=mask, err=None,
                        center_pixel = center_pixel, pixscale=pixscale)

        apertures_mom = CircApertures(rarr=aper_centers_arcsec, slit_PA=slit_pa, rpix=rpix,
                 nx=nx, ny=ny, center_pixel=center_pixel, pixscale=pixscale, moment=True)
        aper_centers_mod_mom, flux1d_mod_mom, vel1d_mod_mom, disp1d_mod_mom = apertures_mom.extract_1d_kinematics(spec_arr=specarr,
                        cube=model_scaled, mask=mask, err=None,
                        center_pixel = center_pixel, pixscale=pixscale)

        aper_centers2, flux1d2, vel1d2, disp1d2 = apertures_mom.extract_1d_kinematics(spec_arr=specarr,
                        cube=data_scaled, mask=mask, err=ecube,
                        center_pixel = center_pixel, pixscale=pixscale)

    ######################################
    # Setup plot:

    nrows = int(np.round(np.sqrt(len(aper_centers_arcsec))))
    ncols = int(np.ceil(len(aper_centers_arcsec)/(1.*nrows)))

    padx = 0.25
    pady = 0.25

    xextra = 0.15
    yextra = 0.

    scale = 2.5

    f = plt.figure()
    f.set_size_inches((ncols+(ncols-1)*padx+xextra)*scale, (nrows+pady+yextra)*scale)


    suptitle = '{}: ndim={}'.format(obs.name,obs.instrument.ndim)


    gs = gridspec.GridSpec(nrows, ncols, wspace=padx, hspace=pady)

    axes = []


    for i in range(nrows):
        for j in range(ncols):
            axes.append(plt.subplot(gs[i,j]))



    #############################################################
    # for each ap:

    for k in range(len(axes)):
        ax = axes[k]
        if k < len(aper_centers_arcsec):
            _, datarr, errarr = apertures.apertures[k].extract_aper_spec(spec_arr=specarr,
                    cube=datacube, mask=mask, err=errcube, skip_specmask=True)
            _, modarr, _ = apertures.apertures[k].extract_aper_spec(spec_arr=specarr,
                    cube=modelcube, mask=mask, skip_specmask=True)
            _, maskarr, _ = apertures.apertures[k].extract_aper_spec(spec_arr=specarr,
                    cube=mask, skip_specmask=True)
            maskarr[maskarr>0] = 1.

            if not skip_fits:
                gmod_flux2=flux1d_mod_mom[k]
                gmod_vel2=vel1d_mod_mom[k]
                gmod_disp2=disp1d_mod_mom[k]

                gmod_flux2 = gmod_vel2 = gmod_disp2 = None

                gdata_flux2=flux1d2[k]
                gdata_vel2=vel1d2[k]
                gdata_disp2=disp1d2[k]

                gdata_flux=flux1d[k]
                gdata_vel=vel1d[k]
                gdata_disp=disp1d[k]
                gmod_flux=flux1d_mod[k]
                gmod_vel=vel1d_mod[k]
                gmod_disp=disp1d_mod[k]

            else:
                gdata_flux = gdata_vel = gdata_disp = None
                gdata_flux2 = gdata_vel2 = gdata_disp2 = None
                gmod_flux = gmod_vel = gmod_disp = None
                gmod_flux2 = gmod_vel2 = gmod_disp2 = None

            ax = plot_spaxel_fit(specarr, datarr, maskarr, err=errarr,
                gdata_flux=gdata_flux, gdata_vel=gdata_vel, gdata_disp=gdata_disp,
                gdata_flux2=gdata_flux2, gdata_vel2=gdata_vel2, gdata_disp2=gdata_disp2,
                model=modarr, gmod_flux=gmod_flux, gmod_vel=gmod_vel, gmod_disp=gmod_disp,
                gmod_flux2=gmod_flux2, gmod_vel2=gmod_vel2, gmod_disp2=gmod_disp2,
                ax=ax)

            ax.annotate('Ap {}'.format(k),
                    (0.02,0.98), xycoords='axes fraction',
                    ha='left', va='top', fontsize=8)
        else:
            ax.set_axis_off()

    #############################################################
    # Save to file:
    f.suptitle(suptitle, fontsize=16, y=0.925)

    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()

    return None



def plot_spaxel_compare_3D_cubes(obs, datacube=None, errcube=None,
                                 modelcube=None, mask=None,
                                 fileout=None,
                                 typ='all',
                                 show_model=True,
                                 skip_masked=False,
                                 skip_fits=True,
                                 overwrite=False):

    if typ.strip().lower() not in ['all', 'diag']:
        raise ValueError("typ={} not recognized for `plot_spaxel_compare_3D_cubes`!".format(typ))
    # Clean up:
    typ = typ.strip().lower()

    if datacube is None:
        datacube = obs.data.data
    if errcube is None:
        errcube = obs.data.error
    if mask is None:
        mask = np.array(obs.data.mask, dtype=float)
    if show_model:
        if modelcube is None:
            try:
                modelcube = obs.model_data.data
            except:
                # Skip showing model
                show_model = False

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None


    # Aper centers: pick roughly number fitting into size:
    nx = datacube.shape[2]
    ny = datacube.shape[1]
    npix = np.max([nx,ny])

    specarr = datacube.spectral_axis.to(u.km/u.s).value

    ######################################
    # Setup plot:

    if typ == 'all':
        rowinds = np.where(np.sum(np.sum(mask,axis=0),axis=1)>0)[0]
        colinds = np.where(np.sum(np.sum(mask,axis=0),axis=0)>0)[0]

        nrows = len(rowinds)
        ncols = len(colinds)
    elif typ == 'diag':
        nrows = int(np.round(np.sqrt(npix)))
        ncols = int(np.ceil(npix/(1.*nrows)))
        rowinds = np.arange(nrows*ncols)
        colinds = np.arange(nrows*ncols)



    padx = 0.25
    pady = 0.25

    xextra = 0.15
    yextra = 0.

    scale = 2.5

    f = plt.figure()
    figsize = ((ncols+(ncols-1)*padx+xextra)*scale,
               (nrows+(nrows-1)*pady+yextra)*scale)
    f.set_size_inches((ncols+(ncols-1)*padx+xextra)*scale,
                      (nrows+(nrows-1)*pady+yextra)*scale)


    suptitle = '{}: ndim={}'.format(obs.name, obs.instrument.ndim)

    gs = gridspec.GridSpec(nrows, ncols, wspace=padx, hspace=pady)

    axes = []

    for i in range(nrows):
        for j in range(ncols):
            if typ == 'all':
                # invert rows:
                ii = nrows - 1 - i
            else:
                ii = i
            axes.append(plt.subplot(gs[ii,j]))


    #############################################################
    if not skip_fits:
        if show_model:
            mom0_mod = modelcube.moment0().to(u.km/u.s).value
            mom1_mod = modelcube.moment1().to(u.km/u.s).value
            mom2_mod = modelcube.linewidth_sigma().to(u.km/u.s).value

        maskbool = np.array(mask, dtype=bool)
        datacube_masked = datacube.with_mask(maskbool)
        mom0_dat = datacube_masked.moment0().to(u.km/u.s).value
        mom1_dat = datacube_masked.moment1().to(u.km/u.s).value
        mom2_dat = datacube_masked.linewidth_sigma().to(u.km/u.s).value

    # for each spax:
    k = -1
    for i in rowinds:
        for j in colinds:
            skip = False
            if typ == 'diag':
                if (i == j):
                    k += 1
                else:
                    skip = True
            elif typ == 'all':
                k += 1

            if (k < len(axes)) & (not skip) & (i < npix):
                ax = axes[k]

                datarr = datacube[:,i,j].value
                maskarr = mask[:,i,j]
                errarr = errcube[:,i,j].value
                if show_model:
                    modarr = modelcube[:,i,j].value
                else:
                    modarr = None

                do_plot = True
                if skip_masked:
                    if (maskarr.max() <= 0):
                        do_plot = False

                if ((typ == 'all') & (do_plot)) | (typ == 'diag'):
                    if not skip_fits:
                        if show_model:
                            flux1d_mod_mom = mom0_mod[i,j]
                            vel1d_mod_mom = mom1_mod[i,j]
                            disp1d_mod_mom = mom2_mod[i,j]

                            best_fit = gaus_fit_sp_opt_leastsq(specarr, modarr, mom0_mod[i,j],
                                            mom1_mod[i,j], mom2_mod[i,j])
                            flux1d_mod = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                            vel1d_mod = best_fit[1]
                            disp1d_mod = best_fit[2]
                        else:
                            flux1d_mod_mom = None
                            vel1d_mod_mom = None
                            disp1d_mod_mom = None
                            flux1d_mod = None
                            vel1d_mod = None
                            disp1d_mod = None

                        maskarr_bool = np.array(maskarr, dtype=bool)

                        flux1d2 = mom0_dat[i,j]
                        vel1d2 = mom1_dat[i,j]
                        disp1d2 = mom2_dat[i,j]
                        try:
                            best_fit = gaus_fit_apy_mod_fitter(specarr[maskarr_bool], datarr[maskarr_bool],
                                            mom0_dat[i,j], mom1_dat[i,j], mom2_dat[i,j], yerr=errarr[maskarr_bool])
                        except:
                            best_fit = [np.NaN, np.NaN, np.NaN]
                        flux1d = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                        vel1d = best_fit[1]
                        disp1d = best_fit[2]

                        gmod_flux2=flux1d_mod_mom
                        gmod_vel2=vel1d_mod_mom
                        gmod_disp2=disp1d_mod_mom

                    else:
                        flux1d = vel1d = disp1d = None
                        flux1d2 = vel1d2 = disp1d2 = None
                        flux1d_mod = vel1d_mod = disp1d_mod = None
                        gmod_flux2 = gmod_vel2 = gmod_disp2 = None

                    ax = plot_spaxel_fit(specarr, datarr, maskarr, err=errarr,
                        gdata_flux=flux1d, gdata_vel=vel1d, gdata_disp=disp1d,
                        gdata_flux2=flux1d2, gdata_vel2=vel1d2, gdata_disp2=disp1d2,
                        model=modarr, gmod_flux=flux1d_mod, gmod_vel=vel1d_mod,
                        gmod_disp=disp1d_mod,
                        gmod_flux2=gmod_flux2, gmod_vel2=gmod_vel2,
                        gmod_disp2=gmod_disp2,
                        ax=ax)

                    ax.annotate('Pix ({},{})'.format(j,i),
                            (0.02,0.98), xycoords='axes fraction',
                            ha='left', va='top', fontsize=8)

                    if (maskarr.max() <= 0):
                        ax.set_facecolor('#f0f0f0')
                else:
                    ax.set_axis_off()
            elif (k < len(axes)) & (not skip) & (k >= npix):
                ax = axes[k]
                ax.set_axis_off()

    #############################################################
    # Save to file:
    if suptitle is not None:
        yoff = 0.105 - 0.01*(36.875-f.get_size_inches()[1])/18.75
        ytitlepos = 1.-yoff
        if f.get_size_inches()[1] < 20.:
            ytitlefontsize = 20
        else:
            ytitlefontsize = 30
        f.suptitle(suptitle, fontsize=ytitlefontsize, y=ytitlepos)

    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()

    return None


#############################################################

def plot_channel_maps_3D_cube(obs, model, show_data=True,
                              show_model=True, show_residual=True,
                              vbounds = [-450., 450.], delv = 100.,
                              vbounds_shift=True,
                              cmap=cmap_greys, cmap_resid=cmap_seismic, 
                              fileout=None, overwrite=False):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    if (not show_data) | (not show_model):
        show_residual = False

    vbounds = np.array(vbounds)
    if vbounds_shift:
        vbounds += model.geometries[obs.name].vel_shift.value

    v_slice_lims_arr = np.arange(vbounds[0], vbounds[1]+delv, delv)


    #################################################
    # center slice: flux limits:
    ind = int(np.round((len(v_slice_lims_arr)-2)/2.))
    v_slice_lims = v_slice_lims_arr[ind:ind+2]
    if show_data:
        subcube = obs.data.data.spectral_slab(v_slice_lims[0]*u.km/u.s, v_slice_lims[1]*u.km/u.s)
        im = subcube.moment0().value * obs.data.mask
    else:
        subcube = obs.model_data.data.spectral_slab(v_slice_lims[0]*u.km/u.s, v_slice_lims[1]*u.km/u.s)
        im = subcube.moment0().value * obs.model_data.mask
    flux_lims = [im.min(), im.max()]
    fac = 1.
    immax = np.max(np.abs(im))
    flux_lims_resid = [-fac*immax, fac*immax]
    #################################################

    f = plt.figure()
    scale = 3.

    show_multi = True
    if show_residual:
        n_cols = 3
        n_rows = len(v_slice_lims_arr)-1
        ind_dat = 0
        ind_mod = 1
        ind_resid = 2
    elif (show_data) & (show_model):
        n_cols = 2
        n_rows = len(v_slice_lims_arr)-1
        ind_dat = 0
        ind_mod = 1
        ind_resid = None
    else:
        n_cols = int(np.ceil(np.sqrt(len(v_slice_lims_arr)-1)))
        n_rows = int(np.ceil((len(v_slice_lims_arr)-1.)/(1.*n_cols)))
        show_multi = False
        if show_data:
            ind_dat = 0
            ind_mod = ind_resid = None
        else:
            ind_mod = 0
            ind_dat = ind_resid = None


    wspace_outer = 0.
    wspace = hspace = padfac = 0.1
    fac = 1.
    f.set_size_inches(fac*scale*n_cols+(n_cols-1)*scale*padfac,scale*n_rows+(n_rows-1)*scale*padfac)

    gs =  gridspec.GridSpec(n_rows,n_cols, wspace=wspace, hspace=hspace)


    if show_multi:
        if show_data:
            axes_dat = []
        if show_model:
            axes_mod = []
        if show_residual:
            axes_resid = []
        for j in range(n_rows):
            if show_data:
                axes_dat.append(plt.subplot(gs[j,ind_dat]))
            if show_model:
                axes_mod.append(plt.subplot(gs[j,ind_mod]))
            if show_residual:
                axes_resid.append(plt.subplot(gs[j,ind_resid]))
    else:
        if show_data:
            axes_dat = []
        if show_model:
            axes_mod = []
        for j in range(n_rows):
            for i in range(n_cols):
                if show_data:
                    axes_dat.append(plt.subplot(gs[j,i]))
                if show_model:
                    axes_mod.append(plt.subplot(gs[j,i]))

    types = []
    axes_stack = []
    if show_data:
        types.append('data')
        axes_stack.append(axes_dat)
    if show_model:
        types.append('model')
        axes_stack.append(axes_mod)
    if show_residual:
        types.append('residual')
        axes_stack.append(axes_resid)


    center = np.array([(obs.model_data.data.shape[2]-1.)/2., (obs.model_data.data.shape[1]-1.)/2.])
    center[0] += model.geometries[obs.name].xshift.value
    center[1] += model.geometries[obs.name].yshift.value

    k = -1
    for ii in range(n_rows):
        if show_multi:
            k += 1
            v_slice_lims = v_slice_lims_arr[k:k+2]
        for j in range(n_cols):
            if show_multi:
                ind_stack = j
            else:
                k += 1
                ind_stack = 0
                v_slice_lims = v_slice_lims_arr[k:k+2]

            ##
            ax = axes_stack[ind_stack][k]

            typ = types[ind_stack]
            flims = flux_lims
            cmap_tmp = cmap
            residual=False
            if typ == 'data':
                cube = obs.data.data*obs.data.mask
                color_contours='blue'
            elif typ == 'model':
                cube = obs.model_data.data*obs.model_data.mask
                color_contours='red'
            elif typ == 'residual':
                cube = obs.data.data*obs.data.mask - obs.model_data.data*obs.model_data.mask
                flims = flux_lims_resid
                cmap_tmp = cmap_resid
                color_contours='black'
                residual=True

            ax = plot_channel_slice(ax=ax,speccube=cube, center=center,
                        v_slice_lims=v_slice_lims, flux_lims=flims,
                        cmap=cmap_tmp,  color_contours=color_contours,
                        residual=residual)


            ###########################
            label_str = None
            if (show_multi & (ii == 0)):
                if (j==0):
                    label_str = "{}: {}".format(obs.name, typ.capitalize())
                else:
                    label_str = typ.capitalize()
            elif ((not show_multi) & (k == 0)):
                label_str = "{}: {}".format(obs.name, typ.capitalize())
            if label_str is not None:
                ax.set_title(label_str, fontsize=14)
            ###########################

    #############################################################
    # Save to file:

    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()

    return None


#############################################################

def plot_spaxels_cube(cube=None, hdr=None, mask=None,
                fname=None,
                typ='all',
                skip_masked=False,
                fileout=None,
                overwrite=False):

    if typ.strip().lower() not in ['all', 'diag']:
        raise ValueError("typ={} not recognized for `plot_spaxels_cube`!".format(typ))
    # Clean up:
    typ = typ.strip().lower()


    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    if ((cube is None) | (hdr is None)) and (fname is None):
        raise ValueError("Either 'fname' must be specified,"
                         "or both 'cube' and 'hdr' must be set!")


    import astropy.units as u
    from astropy.wcs import WCS
    from spectral_cube import SpectralCube

    if ((cube is None) | (hdr is None)):
        datcube = SpectralCube.read(fname)
    else:
        if (hdr['SPEC_TYPE'] == 'VOPT'):
            spec_unit = u.km/u.s
        elif (hdr['SPEC_TYPE'] == 'WAVE'):
            spec_unit = u.Angstrom
        w = WCS(header=hdr)
        datcube = SpectralCube(data=cube, wcs=w, mask=mask).with_spectral_unit(spec_unit)

    #################################################

    # Aper centers: pick roughly number fitting into size:
    nx = datcube.shape[2]
    ny = datcube.shape[1]
    npix = np.max([nx,ny])

    specarr = datcube.spectral_axis.to(u.km/u.s).value

    ######################################
    # Setup plot:

    if typ == 'all':
        if mask is not None:
            rowinds = np.where(np.sum(np.sum(mask,axis=0),axis=1)>0)[0]
            colinds = np.where(np.sum(np.sum(mask,axis=0),axis=0)>0)[0]
        else:
            rowinds = np.where(np.sum(np.sum(np.abs(datcube),axis=0),axis=1)>0)[0]
            colinds = np.where(np.sum(np.sum(np.abs(datcube),axis=0),axis=0)>0)[0]

        nrows = len(rowinds)
        ncols = len(colinds)
    elif typ == 'diag':
        nrows = int(np.round(np.sqrt(npix)))
        ncols = int(np.ceil(npix/(1.*nrows)))
        rowinds = np.arange(nrows*ncols)
        colinds = np.arange(nrows*ncols)



    padx = 0.25
    pady = 0.25

    xextra = 0.15
    yextra = 0.

    scale = 2.5

    f = plt.figure()
    figsize = ((ncols+(ncols-1)*padx+xextra)*scale,
               (nrows+(nrows-1)*pady+yextra)*scale)
    f.set_size_inches((ncols+(ncols-1)*padx+xextra)*scale,
                      (nrows+(nrows-1)*pady+yextra)*scale)


    suptitle = None

    gs = gridspec.GridSpec(nrows, ncols, wspace=padx, hspace=pady)

    axes = []

    for i in range(nrows):
        for j in range(ncols):
            if typ == 'all':
                # invert rows:
                ii = nrows - 1 - i
            else:
                ii = i
            axes.append(plt.subplot(gs[ii,j]))


    #############################################################

    # for each spax:
    k = -1
    for i in rowinds:
        for j in colinds:
            skip = False
            if typ == 'diag':
                if (i == j):
                    k += 1
                else:
                    skip = True
            elif typ == 'all':
                k += 1


            if (k < len(axes)) & (not skip) & (i < npix):
                ax = axes[k]

                datarr = datcube[:,i,j].value
                if mask is not None:
                    maskarr = mask[:,i,j]
                else:
                    maskarr = specarr * 0. + 1.

                do_plot = True
                if skip_masked:
                    if (maskarr.max() <= 0):
                        do_plot = False

                if ((typ == 'all') & (do_plot)) | (typ == 'diag'):
                    ax.plot(specarr, datarr, color='black', ls='-', lw=1., alpha=0.5, zorder=1.)
                    ax.plot(specarr, datarr*maskarr, color='black',lw=1.5, zorder=1.)

                    ax.axhline(y=0., ls='--', color='grey', alpha=0.5, zorder=-20.)

                    xlim = ax.get_xlim()
                    xrange = xlim[1]-xlim[0]
                    if xrange >= 1000.:
                        xmajloc = 500.
                        xminloc = 100.
                    elif xrange >= 500.:
                        xmajloc = 200.
                        xminloc = 50.
                    elif xrange >= 250.:
                        xmajloc = 100.
                        xminloc = 20.
                    elif xrange >= 100.:
                        xmajloc = 50.
                        xminloc = 10.
                    elif xrange >= 50.:
                        xmajloc = 10.
                        xminloc = 2.
                    elif xrange >= 10.:
                        xmajloc = 2.
                        xminloc = 0.5
                    else:
                        xmajloc = None
                        xminloc = None

                    if xmajloc is not None:
                        ax.xaxis.set_major_locator(MultipleLocator(xmajloc))
                        ax.xaxis.set_minor_locator(MultipleLocator(xminloc))

                    ax.annotate('Pix ({},{})'.format(j,i),
                            (0.02,0.98), xycoords='axes fraction',
                            ha='left', va='top', fontsize=8)

                    if (maskarr.max() <= 0):
                        ax.set_facecolor('#f0f0f0')
                else:
                    ax.set_axis_off()
            elif (k < len(axes)) & (not skip) & (k >= npix):
                ax = axes[k]
                ax.set_axis_off()

    #############################################################
    # Save to file:
    if suptitle is not None:
        yoff = 0.105 - 0.01*(36.875-f.get_size_inches()[1])/18.75
        ytitlepos = 1.-yoff
        if f.get_size_inches()[1] < 20.:
            ytitlefontsize = 20
        else:
            ytitlefontsize = 30
        f.suptitle(suptitle, fontsize=ytitlefontsize, y=ytitlepos)

    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()

    return None

#############################################################

def plot_spaxel_fit(specarr, data, mask, err=None,
        gdata_flux=None, gdata_vel=None, gdata_disp=None,
        gdata_flux2=None, gdata_vel2=None, gdata_disp2=None,
        model=None,
        gmod_flux=None, gmod_vel=None, gmod_disp=None,
        gmod_flux2=None, gmod_vel2=None, gmod_disp2=None,
        ax=None):
    returnax = True
    if ax is None:
        returnax = False
        ax = plt.subplot(111)


    ax.plot(specarr, data*mask, color='black', marker='o', ms=4., zorder=1.)
    if (mask.max() > 0):
        ylim = ax.get_ylim()
    ax.plot(specarr, data, color='black', marker='o', ms=4., mfc='None', ls='None', alpha=0.5, zorder=0.)
    if (mask.max() > 0):
        ax.set_ylim(ylim)

    if gdata_flux is not None:
        try:
            gdata_A = gdata_flux / ( np.sqrt(2 * np.pi) * gdata_disp)
            ax.plot(specarr, gdata_A*np.exp(-((specarr-gdata_vel)**2/(2.*gdata_disp**2))),
                    color='turquoise',zorder=10., lw=0.5)
        except:
            pass

    if gdata_flux2 is not None:
        try:
            gdata_A2 = gdata_flux2 / ( np.sqrt(2 * np.pi) * gdata_disp2)
            ax.plot(specarr, gdata_A2*np.exp(-((specarr-gdata_vel2)**2/(2.*gdata_disp2**2))),
                    color='tab:green', ls='--', zorder=5., lw=0.5)
        except:
            pass

    if model is not None:
        ax.plot(specarr, model, color='red', ls='-', lw=1., alpha=0.5, zorder=1.)
        ax.plot(specarr, model*mask, color='red',lw=1.5, zorder=1.)

        if gmod_flux is not None:
            try:
                gmod_A = gmod_flux / ( np.sqrt(2 * np.pi) * gmod_disp)
                ax.plot(specarr, gmod_A*np.exp(-((specarr-gmod_vel)**2/(2.*gmod_disp**2))),
                            color='orange', ls='--', zorder=10., lw=0.5)
            except:
                pass

        if gmod_flux2 is not None:
            try:
                gmod_A2 = gmod_flux2 / ( np.sqrt(2 * np.pi) * gmod_disp2)
                ax.plot(specarr, gmod_A2*np.exp(-((specarr-gmod_vel2)**2/(2.*gmod_disp2**2))),
                        color='purple', ls=':', zorder=10., lw=0.5)
            except:
                pass

    if err is not None:
        ylim = ax.get_ylim()
        ax.errorbar(specarr, data, xerr=None, yerr=err, alpha=0.25, capsize=0.,
                                marker=None, ls='None', ecolor='k', zorder=-1.)
        ax.set_ylim(ylim)

    ax.axhline(y=0., ls='--', color='grey', alpha=0.5, zorder=-20.)

    xlim = ax.get_xlim()
    xrange = xlim[1]-xlim[0]
    if xrange >= 1000.:
        xmajloc = 500.
        xminloc = 100.
    elif xrange >= 500.:
        xmajloc = 200.
        xminloc = 50.
    elif xrange >= 250.:
        xmajloc = 100.
        xminloc = 20.
    elif xrange >= 100.:
        xmajloc = 50.
        xminloc = 10.
    elif xrange >= 50.:
        xmajloc = 10.
        xminloc = 2.
    elif xrange >= 10.:
        xmajloc = 2.
        xminloc = 0.5
    else:
        xmajloc = None
        xminloc = None

    if xmajloc is not None:
        ax.xaxis.set_major_locator(MultipleLocator(xmajloc))
        ax.xaxis.set_minor_locator(MultipleLocator(xminloc))

    if returnax:
        return ax
    else:
        return None


#############################################################

def plot_channel_maps_cube(cube=None, hdr=None, mask=None,
            fname=None,
            vbounds = [-450., 450.],
            delv = 100.,
            cmap=cmap_greys, 
            fileout=None,
            overwrite=False):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    if ((cube is None) | (hdr is None)) and (fname is None):
        raise ValueError("Either 'fname' must be specified,"
                         "or both 'cube' and 'hdr' must be set!")

    import astropy.units as u
    from astropy.wcs import WCS
    from spectral_cube import SpectralCube

    if ((cube is None) | (hdr is None)):
        datcube = SpectralCube.read(fname)
    else:
        if (hdr['SPEC_TYPE'] == 'VOPT'):
            spec_unit = u.km/u.s
        elif (hdr['SPEC_TYPE'] == 'WAVE'):
            spec_unit = u.Angstrom
        w = WCS(header=hdr)
        datcube = SpectralCube(data=cube, wcs=w, mask=mask).with_spectral_unit(spec_unit)

    vbounds = np.array(vbounds)

    v_slice_lims_arr = np.arange(vbounds[0], vbounds[1]+delv, delv)


    #################################################
    # center slice: flux limits:
    ind = int(np.round((len(v_slice_lims_arr)-2)/2.))
    v_slice_lims = v_slice_lims_arr[ind:ind+2]
    subcube = datcube.spectral_slab(v_slice_lims[0]*u.km/u.s, v_slice_lims[1]*u.km/u.s)
    im = subcube.moment0().value
    flux_lims = [im.min(), im.max()]
    fac = 1.
    immax = np.max(np.abs(im))
    flux_lims_resid = [-fac*immax, fac*immax]
    #################################################

    f = plt.figure()
    scale = 3.

    n_cols = int(np.ceil(np.sqrt(len(v_slice_lims_arr)-1)))
    n_rows = int(np.ceil((len(v_slice_lims_arr)-1.)/(1.*n_cols)))

    wspace_outer = 0.
    wspace = hspace = padfac = 0.1
    fac = 1.
    f.set_size_inches(fac*scale*n_cols+(n_cols-1)*scale*padfac,scale*n_rows+(n_rows-1)*scale*padfac)

    gs =  gridspec.GridSpec(n_rows,n_cols, wspace=wspace, hspace=hspace)

    axes = []
    for j in range(n_rows):
        for i in range(n_cols):
            axes.append(plt.subplot(gs[j,i]))


    center = np.array([(datcube.shape[2]-1.)/2., (datcube.shape[1]-1.)/2.])

    k = -1
    for ii in range(n_rows):
        for j in range(n_cols):
            k += 1
            ax = axes[k]
            if k > (len(v_slice_lims_arr)-2):
                ax.set_axis_off()
            else:
                v_slice_lims = v_slice_lims_arr[k:k+2]
                flims = flux_lims
                cmap_tmp = cmap
                color_contours='red'

                ax = plot_channel_slice(ax=ax,speccube=datcube, center=center,
                            v_slice_lims=v_slice_lims, flux_lims=flims,
                            cmap=cmap_tmp,  color_contours=color_contours,
                            residual=False)


    #############################################################
    # Save to file:

    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()

    return None


def plot_channel_slice(ax=None, speccube=None, v_slice_lims=None, flux_lims=None,
                      center=None, show_pix_coords=False, 
                      cmap=cmap_greys,  
                      color_contours='red', color_center='cyan',
                      residual=False):

    if ax is None:
        ax = plt.gca()
    if v_slice_lims is None:
        v_slice_lims = [-50., 50.]

    subcube = speccube.spectral_slab(v_slice_lims[0]*u.km/u.s, v_slice_lims[1]*u.km/u.s)

    im = subcube.moment0().value

    if center is None:
        center = ((im.shape[1]-1.)/2., (im.shape[0]-1.)/2.)



    int_mode = "nearest"
    origin = 'lower'
    if flux_lims is not None:
        vmin = flux_lims[0]
        vmax = flux_lims[1]
    else:
        vmin = None
        vmax = None
    imax = ax.imshow(im, cmap=cmap, interpolation=int_mode,
                        vmin=vmin, vmax=vmax, origin=origin)



    sigma_levels = [1.,2.,3.]
    levels = 1.0 - np.exp(-0.5 * np.array(sigma_levels) ** 2)
    lw_contour = 1.5
    rgba_color = mplcolors.colorConverter.to_rgba(color_contours)
    color_levels = [list(rgba_color) for l in levels]
    lw_arr = []
    for ii, l in enumerate(levels):
        color_levels[ii][-1] *= float(ii+1) / (len(levels))
        lw_arr.append(lw_contour * float(ii+1+len(levels)*0.5) / (len(levels)*1.5))
    contour_kwargs = {'colors': color_levels, 'linewidths': lw_arr, 'linestyles': '-'}


    #################################################
    # Syntax taken from corner/core.py
    imflat = im.flatten()
    imtmp = im.copy()
    if residual:
        imflat = np.abs(imflat)
        imtmp = np.abs(imtmp)
    inds = np.argsort(imflat)[::-1]
    imflat = imflat[inds]
    sm = np.cumsum(imflat)
    sm /= sm[-1]
    contour_levels = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            contour_levels[i] = imflat[sm<=v0][-1]
        except:
            contour_levels[i] = imflat[0]
    contour_levels.sort()
    m = np.diff(contour_levels) == 0
    if np.any(m):
        print("Too few points to create valid contours")
    while np.any(m):
        contour_levels[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(contour_levels) == 0
    contour_levels.sort()

    ax.contour(imtmp, contour_levels, **contour_kwargs)

    #################################################

    ax.plot(center[0], center[1], '+', mew=1., ms=10., c=color_center)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ybase_offset = 0.035
    x_base = xlim[0] + (xlim[1]-xlim[0])*0.075
    y_base = ylim[0] + (ylim[1]-ylim[0])*(ybase_offset+0.075)
    string = r'$[{:0.1f},{:0.1f}]$'.format(v_slice_lims[0], v_slice_lims[1])
    ax.annotate(string, xy=(x_base, y_base),
                xycoords="data",
                xytext=(0,0),
                color='black',
                textcoords="offset points", ha="left", va="center",
                fontsize=8)

    if not show_pix_coords:
        ax.set_xticks([])
        ax.set_yticks([])

    return ax


#############################################################

def plot_channel_maps_cube_overlay(obs, model,
                              vbounds = [-450., 450.], delv = 100.,
                              vbounds_shift=True,
                              cmap=cmap_greys, cmap_resid=cmap_seismic, 
                              fileout=None, overwrite=False):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None


    vbounds = np.array(vbounds)
    if vbounds_shift:
        vbounds += model.geometries[obs.name].vel_shift.value

    v_slice_lims_arr = np.arange(vbounds[0], vbounds[1]+delv, delv)


    #################################################
    # center slice: flux limits:
    ind = int(np.round((len(v_slice_lims_arr)-2)/2.))
    v_slice_lims = v_slice_lims_arr[ind:ind+2]
    subcube = obs.data.data.spectral_slab(v_slice_lims[0]*u.km/u.s, v_slice_lims[1]*u.km/u.s)
    im = subcube.moment0().value * obs.data.mask
    flux_lims = [im.min(), im.max()]
    fac = 1.
    immax = np.max(np.abs(im))
    flux_lims_resid = [-fac*immax, fac*immax]
    #################################################

    f = plt.figure()
    scale = 2. #3.

    n_cols = int(np.ceil(np.sqrt(len(v_slice_lims_arr)-1)))
    n_rows = int(np.ceil((len(v_slice_lims_arr)-1.)/(1.*n_cols)))


    wspace_outer = 0.
    wspace = hspace = padfac = 0.1
    fac = 1.
    f.set_size_inches(fac*scale*n_cols+(n_cols-1)*scale*padfac,scale*n_rows+(n_rows-1)*scale*padfac)

    gs =  gridspec.GridSpec(n_rows,n_cols, wspace=wspace, hspace=hspace)


    axes = []
    for j in range(n_rows):
        for i in range(n_cols):
            axes.append(plt.subplot(gs[j,i]))


    center = np.array([(obs.model_data.data.shape[2]-1.)/2., (obs.model_data.data.shape[1]-1.)/2.])
    center[0] += model.geometries[obs.name].xshift.value
    center[1] += model.geometries[obs.name].yshift.value

    k = -1
    for ii in range(n_rows):
        for j in range(n_cols):
            k += 1
            v_slice_lims = v_slice_lims_arr[k:k+2]

            ##
            ax = axes[k]

            flims = flux_lims
            cmap_tmp = cmap
            cube_dat = obs.data.data*obs.data.mask
            cube_mod = obs.model_data.data*obs.model_data.mask
            color_contours='red'

            ax = plot_channel_slice_diff_contours(ax=ax,speccube=cube_dat,
                        modcube=cube_mod, center=center,
                        v_slice_lims=v_slice_lims, flux_lims=flims,
                        cmap=cmap_tmp,  color_contours=color_contours)


            # ###########################
            # label_str = None
            # if (k == 0):
            #     label_str = "{}: {}".format(obs.name, typ.capitalize())
            # if label_str is not None:
            #     ax.set_title(label_str, fontsize=14)
            ###########################

    #############################################################
    # Save to file:

    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()

    return None

def plot_channel_slice_diff_contours(ax=None, speccube=None, modcube=None,
                      v_slice_lims=None, flux_lims=None,
                      center=None, show_pix_coords=False, 
                      cmap=cmap_greys,  
                      color_contours='red', color_center='cyan'):

    if ax is None:
        ax = plt.gca()
    if v_slice_lims is None:
        v_slice_lims = [-50., 50.]

    subcube = speccube.spectral_slab(v_slice_lims[0]*u.km/u.s, v_slice_lims[1]*u.km/u.s)
    modsubcube = modcube.spectral_slab(v_slice_lims[0]*u.km/u.s, v_slice_lims[1]*u.km/u.s)

    im = subcube.moment0().value
    immod = modsubcube.moment0().value

    if center is None:
        center = ((im.shape[1]-1.)/2., (im.shape[0]-1.)/2.)



    int_mode = "nearest"
    origin = 'lower'
    if flux_lims is not None:
        vmin = flux_lims[0]
        vmax = flux_lims[1]
    else:
        vmin = None
        vmax = None
    imax = ax.imshow(im, cmap=cmap, interpolation=int_mode,
                        vmin=vmin, vmax=vmax, origin=origin)



    sigma_levels = [1.,2.,3.]
    levels = 1.0 - np.exp(-0.5 * np.array(sigma_levels) ** 2)
    lw_contour = 1.5
    rgba_color = mplcolors.colorConverter.to_rgba(color_contours)
    color_levels = [list(rgba_color) for l in levels]
    lw_arr = []
    for ii, l in enumerate(levels):
        color_levels[ii][-1] *= float(ii+1) / (len(levels))
        lw_arr.append(lw_contour * float(ii+1+len(levels)*0.5) / (len(levels)*1.5))
    contour_kwargs = {'colors': color_levels, 'linewidths': lw_arr, 'linestyles': '-'}


    #################################################
    # Syntax taken from corner/core.py
    imflat = immod.flatten()
    imtmp = immod.copy()
    inds = np.argsort(imflat)[::-1]
    imflat = imflat[inds]
    sm = np.cumsum(imflat)
    sm /= sm[-1]
    contour_levels = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            contour_levels[i] = imflat[sm<=v0][-1]
        except:
            contour_levels[i] = imflat[0]
    contour_levels.sort()
    m = np.diff(contour_levels) == 0
    if np.any(m):
        print("Too few points to create valid contours")
    while np.any(m):
        contour_levels[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(contour_levels) == 0
    contour_levels.sort()

    ax.contour(imtmp, contour_levels, **contour_kwargs)

    #################################################

    ax.plot(center[0], center[1], '+', mew=1., ms=10., c=color_center)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ybase_offset = 0.035
    x_base = xlim[0] + (xlim[1]-xlim[0])*0.075
    y_base = ylim[0] + (ylim[1]-ylim[0])*(ybase_offset+0.075)
    string = r'$[{:0.1f},{:0.1f}]$'.format(v_slice_lims[0], v_slice_lims[1])
    ax.annotate(string, xy=(x_base, y_base),
                xycoords="data",
                xytext=(0,0),
                color='black',
                textcoords="offset points", ha="left", va="center",
                fontsize=8)

    if not show_pix_coords:
        ax.set_xticks([])
        ax.set_yticks([])

    return ax

def plot_3D_data_automask_info(obs, mask_dict, axes=None):
    if axes is None:
        # Setup plotting
        return_axes = False

        nrows = 1; ncols = 5
        padx = pady = 0.2
        xextra = 0.15; yextra = 0.
        scale = 2.5
        fig = plt.figure()
        fig.set_size_inches((ncols+(ncols-1)*padx+xextra)*scale,
                            (nrows+(nrows-1)*pady+yextra)*scale )


        gs = gridspec.GridSpec(nrows, ncols, hspace=pady, wspace=padx)
        axes = []

        for jj in range(nrows):
            for mm in range(ncols):
                axes.append(plt.subplot(gs[jj,mm]))

    else:
        return_axes = True

    int_mode = "nearest"; origin = 'lower'; cmap=cmap_viridis

    xcenter = obs.mod_options.xcenter
    ycenter = obs.mod_options.ycenter
    if xcenter is None:
        xcenter =(obs.data.data.shape[2]-1)/2.
    if ycenter is None:
        ycenter =(obs.data.data.shape[1]-1)/2.


    titles = ['Collapsed flux', 'Collapsed err', 'Segm', 'Mask2D', 'Masked flux']
    for i, im in enumerate([mask_dict['fmap_cube_sn'],
                            mask_dict['emap_cube_sn'],
                            mask_dict['segm'], mask_dict['mask2D'],
                            mask_dict['mask2D']*mask_dict['fmap_cube_sn']]):
        ax = axes[i]
        imax = ax.imshow(im, cmap=cmap, interpolation=int_mode,
                                origin=origin, vmin=None, vmax=None)
        ax.scatter(xcenter, ycenter, color='magenta', marker='+', s=30)
        cax, kw = colorbar.make_axes_gridspec(ax, pad=0.01,
                fraction=5./101., aspect=20.)
        cbar = plt.colorbar(imax, cax=cax)
        cbar.ax.tick_params(labelsize=8)
        ax.set_title(titles[i], fontsize=12)

        axes[i] = ax

    if return_axes:
        return axes
    else:
        return None



#############################################################

def plot_model_1D(gal,
                  best_dispersion=None,
                  inst_corr=True,
                  fileout_base=None):

    for obs_name in gal.observations:
        obs = gal.observations[obs_name]
        if obs.instrument.ndim == 1:
            if fileout_base is not None:
                fileout_obs= "{}_{}.pdf".format(fileout_base, obs.name)
            else:
                fileout_obs = None

            plot_single_obs_model_1D(obs,
                best_dispersion=best_dispersion,
                inst_corr=inst_corr,
                fileout=fileout_obs)


def plot_model_2D(gal,
            fileout_base=None,
            symmetric_residuals=True,
            max_residual=100.,
            inst_corr=True,
            show_contours=True,
            show_ruler=True,
            len_ruler=None,
            ruler_unit='arcsec',
            apply_mask=True,
            ruler_loc='lowerleft',
            color_annotate='black',
            color_bad='white',
            **plot_kwargs):


    for obs_name in gal.observations:
        obs = gal.observations[obs_name]
        if obs.instrument.ndim == 2:
            if fileout_base is not None:
                fileout_obs= "{}_{}.pdf".format(fileout_base, obs.name)
            else:
                fileout_obs = None

            plot_single_obs_model_2D(obs, gal.model,
                        dscale=gal.dscale,
                        fileout=fileout_obs,
                        symmetric_residuals=symmetric_residuals,
                        max_residual=max_residual,
                        inst_corr=inst_corr,
                        show_contours=show_contours,
                        show_ruler=show_ruler,
                        len_ruler=len_ruler,
                        ruler_unit=ruler_unit,
                        apply_mask=apply_mask,
                        ruler_loc=ruler_loc,
                        color_annotate=color_annotate,
                        color_bad=color_bad,
                        **plot_kwargs)


#############################################################

def plot_single_obs_model_1D(obs,
            # fitvelocity=True,
            # fitdispersion=True,
            # fitflux=False,
            best_dispersion=None,
            inst_corr=True,
            fileout=None):

    ######################################
    # Setup data/model comparison: if this isn't the fit dimension
    #   data/model comparison (eg, fit in 2D, showing 1D comparison)

    model_data = obs.model_data

    if inst_corr:
            model_data.data['dispersion'] = \
                np.sqrt( model_data.data['dispersion']**2 - \
                    obs.instrument.lsf.dispersion.to(u.km/u.s).value**2 )


    ######################################
    # Setup plot:
    f = plt.figure()
    scale = 3.5
    ncols = 0
    nrows = 1

    keyxtitle = r'$r$ [arcsec]'
    keyyarr = []
    keyytitlearr = []
    keyyresidtitlearr = []

    if obs.fit_options.fit_flux:
        ncols += 1
        keyyarr.append('flux')
        keyytitlearr.append('Flux [arb]')
        keyyresidtitlearr.append(r'$\mathrm{Flux_{data} - Flux_{model}}$ [arb]')
    if obs.fit_options.fit_velocity:
        ncols += 1
        keyyarr.append('velocity')
        keyytitlearr.append(r'$V$ [km/s]')
        keyyresidtitlearr.append(r'$V_{\mathrm{data}} - V_{\mathrm{model}}$ [km/s]')
    if obs.fit_options.fit_dispersion:
        ncols += 1
        keyyarr.append('dispersion')
        keyytitlearr.append(r'$\sigma$ [km/s]')
        keyyresidtitlearr.append(r'$\sigma_{\mathrm{data}} - \sigma_{\mathrm{model}}$ [km/s]')


    f.set_size_inches(1.1*ncols*scale, nrows*scale)
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.35, hspace=0.2)


    errbar_lw = 0.5
    errbar_cap = 1.5

    axes = []
    k = -1
    for j in range(ncols):
        # Comparison:
        axes.append(plt.subplot(gs[0,j]))
        k += 1

        axes[k].scatter( model_data.rarr, model_data.data[keyyarr[j]],
            c='red', marker='s', s=25, lw=1, label=None)

        if keyyarr[j] == 'dispersion':
            if best_dispersion:
                axes[k].axhline(y=best_dispersion, ls='--', color='red', zorder=-1.)

        axes[k].set_xlabel(keyxtitle)
        axes[k].set_ylabel(keyytitlearr[j])
        axes[k].axhline(y=0, ls='--', color='k', zorder=-10.)

    f.suptitle(obs.name)

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()



def plot_single_obs_model_2D(obs, model,
            dscale=None,
            fileout=None,
            symmetric_residuals=True,
            max_residual=100.,
            inst_corr=True,
            show_contours=True,
            show_ruler=True,
            len_ruler=None,
            ruler_unit='arcsec',
            apply_mask=True,
            ruler_loc='lowerleft',
            color_annotate='black',
            color_bad='white',
            **plot_kwargs):

    if show_contours:
        # Set contour defaults, if not specifed:
        for key in _kwargs_contour_defaults.keys():
            if key not in plot_kwargs.keys():
                plot_kwargs[key] = _kwargs_contour_defaults[key]

    ######################################
    # Setup plot:

    # f = plt.figure(figsize=(9.5, 6))
    # scale = 3.5

    f = plt.figure()
    scale = 3.5
    nrows = 1

    ncols = 0
    for cond in [obs.fit_options.fit_flux, obs.fit_options.fit_velocity,
                 obs.fit_options.fit_dispersion]:
        if cond:
            ncols += 1

    f.set_size_inches(1.1*ncols*scale, nrows*scale)

    cntr = 0
    if obs.fit_options.fit_flux:
        cntr += 1
        grid_flux = ImageGrid(f, 100+ncols*10+cntr,
                              nrows_ncols=(1, 1),
                              direction="row",
                              axes_pad=0.5,
                              label_mode="1",
                              share_all=True,
                              cbar_location="right",
                              cbar_mode="each",
                              cbar_size="5%",
                              cbar_pad="1%",
                              )

    if obs.fit_options.fit_velocity:
        cntr += 1
        grid_vel = ImageGrid(f, 100+ncols*10+cntr,
                             nrows_ncols=(1, 1),
                             direction="row",
                             axes_pad=0.5,
                             label_mode="1",
                             share_all=True,
                             cbar_location="right",
                             cbar_mode="each",
                             cbar_size="5%",
                             cbar_pad="1%",
                             )
    if obs.fit_options.fit_dispersion:
        cntr += 1
        grid_disp = ImageGrid(f, 100+ncols*10+cntr,
                              nrows_ncols=(1, 1),
                              direction="row",
                              axes_pad=0.5,
                              label_mode="1",
                              share_all=True,
                              cbar_location="right",
                              cbar_mode="each",
                              cbar_size="5%",
                              cbar_pad="1%",
                              )


    keyxarr = ['model']
    keyxtitlearr = ['Model']

    keyyarr, keyytitlearr, grid_arr = ([] for _ in range(3))
    if obs.fit_options.fit_flux:
        keyyarr.append('flux')
        keyytitlearr.append(r'Flux')
        grid_arr.append(grid_flux)

        msk = np.isfinite(obs.model_data.data['flux'])
        flux_vmin = obs.model_data.data['flux'][msk].min()
        flux_vmax = obs.model_data.data['flux'][msk].max()

    if obs.fit_options.fit_velocity:
        keyyarr.append('velocity')
        keyytitlearr.append(r'$V$')
        grid_arr.append(grid_vel)

        msk = np.isfinite(obs.model_data.data['velocity'])
        vel_vmin = obs.model_data.data['velocity'][msk].min()
        vel_vmax = obs.model_data.data['velocity'][msk].max()
        if np.abs(vel_vmax) > 400.:
            vel_vmax = 400.
        if np.abs(vel_vmin) > 400.:
            vel_vmin = -400.

    if obs.fit_options.fit_dispersion:
        keyyarr.append('dispersion')
        keyytitlearr.append(r'$\sigma$')
        grid_arr.append(grid_disp)

        msk = np.isfinite(obs.model_data.data['dispersion'])
        disp_vmin = obs.model_data.data['dispersion'][msk].min()
        disp_vmax = obs.model_data.data['dispersion'][msk].max()

        if np.abs(disp_vmax) > 500:
            disp_vmax = 500.
        if np.abs(disp_vmin) > 500:
            disp_vmin = 0.

    int_mode = "nearest"
    origin = 'lower'
    cmap = copy.copy(cmap_spectral_r)

    cmap.set_bad(color=color_bad)

    for j in range(len(keyyarr)):
        msk = np.isfinite(obs.model_data.data[keyyarr[j]])
        # Also use mask if defined:
        msk[~obs.model_data.mask] = False
        grid = grid_arr[j]

        for ax, k in zip(grid, keyxarr):
            im = obs.model_data.data[keyyarr[j]].copy()
            if apply_mask:
                im[~msk] = np.NaN
            if keyyarr[j] == 'flux':
                vmin = flux_vmin
                vmax = flux_vmax
            elif keyyarr[j] == 'velocity':
                vel_shift = model.geometries[obs.name].vel_shift.value
                im -= vel_shift

                vel_vmin -= vel_shift
                vel_vmax -= vel_shift

                vmin = vel_vmin
                vmax = vel_vmax

            elif keyyarr[j] == 'dispersion':
                if inst_corr:
                    im = np.sqrt(im ** 2 - obs.instrument.lsf.dispersion.to(
                                 u.km / u.s).value ** 2)

                    disp_vmin = max(0, np.sqrt(disp_vmin**2 - obs.instrument.lsf.dispersion.to(u.km / u.s).value ** 2))
                    disp_vmax = np.sqrt(disp_vmax**2 - obs.instrument.lsf.dispersion.to(u.km / u.s).value ** 2)

                vmin = disp_vmin
                vmax = disp_vmax


            imax = ax.imshow(im, cmap=cmap, interpolation=int_mode,
                             vmin=vmin, vmax=vmax, origin=origin)

            ax.set_ylabel(keyytitlearr[j])

            for pos in ['top', 'bottom', 'left', 'right']:
                ax.spines[pos].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            if show_contours:
                ax = plot_contours_2D_multitype(im, ax=ax, mapname=keyyarr[j], plottype=k,
                            vmin=vmin, vmax=vmax, kwargs=plot_kwargs)

            ax = plot_major_minor_axes_2D(ax, obs, model, im, obs.model_data.mask)
            if show_ruler:
                pixscale = obs.instrument.pixscale.value
                if (len_ruler is None):
                    len_arcsec = 1.
                elif (len_ruler is not None):
                    if ruler_unit.lower() == 'arcsec':
                        len_arcsec = len_ruler
                    elif ruler_unit.lower() == 'kpc':
                        if dscale is not None:
                            len_arcsec = len_ruler * dscale
                        else:
                            logger.warning("No 'dscale' provided! Using 1arcsec ruler")
                            len_arcsec = 1.
                            ruler_unit = 'arcsec'
                ax = plot_ruler_arcsec_2D(ax, pixscale, len_arcsec=len_arcsec,
                                            ruler_loc=ruler_loc,  color=color_annotate,
                                            ruler_unit=ruler_unit, dscale=dscale)

            cbar = ax.cax.colorbar(imax)
            cbar.ax.tick_params(labelsize=8)

    f.suptitle(obs.name)


    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()




#############################################################

def plot_model_comparison_2D(obs1=None, obs2=None,
        model1=None, model2=None,
        show_models=True,
        label_gal1='Gal1',
        label_gal2='Gal2',
        label_residuals='Residuals: Gal2-Gal1',
        symmetric_residuals=True,
        max_residual=100.,
        fileout=None,
        vcrop = False,
        vcrop_value = 800.,
        inst_corr=True,
        show_contours=True,
        apply_mask=True,
        **kwargs):

    # Set contour defaults, if not specifed:
    for key in _kwargs_contour_defaults.keys():
        if key not in kwargs.keys():
            kwargs[key] = _kwargs_contour_defaults[key]


    ######################################
    # Setup plot:

    ncols = 3
    if show_models:
        nrows = 3
    else:
        nrows = 1


    padx = pady = 0.25

    xextra = 0.25 #0.15
    yextra = 0.25

    scale = 2.5

    f = plt.figure()
    f.set_size_inches((ncols+(ncols-1)*padx+xextra)*scale, (nrows+(nrows-1)*pady+yextra)*scale)


    padx = 0.2
    pady = 0.1
    gs02 = gridspec.GridSpec(nrows, ncols, wspace=padx, hspace=pady)
    grid_2D = []
    for jj in range(nrows):
        for mm in range(ncols):
            grid_2D.append(plt.subplot(gs02[jj,mm]))



    if inst_corr:
        inst_corr_sigma = obs1.instrument.lsf.dispersion.to(u.km/u.s).value
        inst_corr_sigma2 = obs2.instrument.lsf.dispersion.to(u.km/u.s).value
        # Check values are equivalent:
        if inst_corr_sigma != inst_corr_sigma2:
            raise ValueError
    else:
        inst_corr_sigma = 0.


    # ----------------------------------------------------------------------
    # 2D plotting

    pixscale = obs1.instrument.pixscale.value


    if show_models:
        keyyarr = ['gal1', 'gal2', 'residual']
        keyytitlearr = [label_gal1, label_gal2, label_residuals]
    else:
        keyyarr = ['residual']
        keyytitlearr = [label_residuals]
    keyxarr = ['flux', 'velocity', 'dispersion']
    keyxtitlearr = ['Flux', r'$V$', r'$\sigma$']

    int_mode = "nearest"
    origin = 'lower'
    cmap = copy.copy(cmap_spectral_r)
    bad_color = 'white'
    color_annotate = 'black'
    # bad_color = 'black'
    # color_annotate = 'white'
    cmap.set_bad(color=bad_color)


    cmap_resid = copy.copy(cmap_rdbu_r)
    cmap_resid.set_bad(color=bad_color)
    cmap_resid.set_over(color='magenta')
    cmap_resid.set_under(color='blueviolet')


    # -----------------------
    if show_models:
        vel_vmin = disp_vmin = flux_vmin = 999.
        vel_vmax = disp_vmax = flux_vmax = -999.
        for obs in [obs1, obs2]:
            vel_vmin = np.min([vel_vmin, obs.model_data.data['velocity'][obs.model_data.mask].min()])
            vel_vmax = np.max([vel_vmin, obs.model_data.data['velocity'][obs.model_data.mask].max()])

            if inst_corr:
                im = obs.model_data.data['dispersion'].copy()
                im = np.sqrt(im ** 2 - inst_corr_sigma ** 2)
                msk = obs.model_data.mask.copy()
                msk[~np.isfinite(im)] = False
                disp_vmin = np.min([disp_vmin, im[msk].min()])
                disp_vmax = np.max([disp_vmax, im[msk].max()])
            else:
                disp_vmin = np.min([disp_vmin, obs.model_data.data['dispersion'][obs.model_data.mask].min()])
                disp_vmax = np.max([disp_vmax, obs.model_data.data['dispersion'][obs.model_data.mask].max()])


            flux_vmin = np.min([flux_vmin, obs.model_data.data['flux'][obs.model_data.mask].min()])
            flux_vmax = np.max([flux_vmax, obs.model_data.data['flux'][obs.model_data.mask].max()])

        # Apply vel shift from model:
        vel_shift = model1.geometries[obs1.name].vel_shift.value
        vel_vmin -= vel_shift
        vel_vmax -= vel_shift

        # Check for not too crazy:
        if vcrop:
            if vel_vmin < -vcrop_value:
                vel_vmin = -vcrop_value
            if vel_vmax > vcrop_value:
                vel_vmax = vcrop_value

            if disp_vmin < 0:
                disp_vmin = 0
            if disp_vmax > vcrop_value:
                disp_vmax = vcrop_value


    alpha_unmasked = 1.
    alpha_masked = 0.5
    alpha_bkgd = 1.
    alpha_aper = 0.8


    for mm in range(len(keyyarr)):
        for j in range(len(keyxarr)):
            kk = mm*len(keyyarr) + j

            k = keyyarr[mm]

            ax = grid_2D[kk]

            xt = keyxtitlearr[j]
            yt = keyytitlearr[mm]

            # -----------------------------------
            if (k == 'gal1') | (k == 'gal2'):
                if (k == 'gal1'):
                    obs = obs1
                    model = model1
                elif (k == 'gal2'):
                    obs = obs2
                    model = model2
                if keyxarr[j] == 'velocity':
                    im = obs.model_data.data['velocity'].copy()
                    im -= model.geometries[obs.name].vel_shift.value
                    vmin = vel_vmin
                    vmax = vel_vmax
                elif keyxarr[j] == 'dispersion':
                    im = obs.model_data.data['dispersion'].copy()
                    im = np.sqrt(im ** 2 - inst_corr_sigma ** 2)

                    vmin = disp_vmin
                    vmax = disp_vmax

                elif keyxarr[j] == 'flux':
                    im = obs.model_data.data['flux'].copy()

                    vmin = flux_vmin
                    vmax = flux_vmax

                mask = obs.model_data.mask
                cmaptmp = cmap
                gal = None

            elif k == 'residual':
                if keyxarr[j] == 'velocity':
                    im = obs2.model_data.data['velocity'].copy() - obs1.model_data.data['velocity'].copy()
                    im -= model2.geometries[obs2.name].vel_shift.value - model1.geometries[obs1.name].vel_shift.value
                    if symmetric_residuals:
                        vmin = -max_residual
                        vmax = max_residual
                elif keyxarr[j] == 'dispersion':
                    im_model1 = obs1.model_data.data['dispersion'].copy()
                    im_model1 = np.sqrt(im_model1 ** 2 - inst_corr_sigma ** 2)

                    im_model2 = obs2.model_data.data['dispersion'].copy()
                    im_model2 = np.sqrt(im_model2 ** 2 - inst_corr_sigma ** 2)

                    im = im_model2 - im_model1

                    if symmetric_residuals:
                        vmin = -max_residual
                        vmax = max_residual
                elif keyxarr[j] == 'flux':
                    im = obs2.model_data.data['flux'].copy() - obs1.model_data.data['flux'].copy()

                    if symmetric_residuals:
                        if show_models:
                            fabsmax = np.max(np.abs([flux_vmin, flux_vmax]))
                        else:
                            fabsmax = np.max(np.abs(im[np.isfinite(im)]))
                        vmin = -fabsmax
                        vmax = fabsmax
                if not symmetric_residuals:
                    vmin = im[np.isfinite(im)].min()
                    vmax = im[np.isfinite(im)].max()

                mask = obs1.model_data.mask
                cmaptmp = cmap_resid

            else:
                raise ValueError("key not supported.")

            if apply_mask:
                im[~mask] = np.NaN
            imax = ax.imshow(im, cmap=cmaptmp, interpolation=int_mode,
                             vmin=vmin, vmax=vmax, origin=origin)

            if show_contours:
                ax = plot_contours_2D_multitype(im, ax=ax, mapname=keyxarr[j], plottype=k,
                            vmin=vmin, vmax=vmax, kwargs=kwargs)

            ####################################
            # Show a 1arcsec line:
            ax = plot_ruler_arcsec_2D(ax, pixscale, len_arcsec=1.,
                                      ruler_loc='lowerright', color=color_annotate)
            ####################################

            ax = plot_major_minor_axes_2D(ax, obs1, model1, im, obs1.model_data.mask)

            if j == 0:
                ax.set_ylabel(yt)

            for pos in ['top', 'bottom', 'left', 'right']:
                ax.spines[pos].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            if mm == 0:
                ax.set_title(xt)

            #########
            cax, kw = colorbar.make_axes_gridspec(ax, pad=0.01,
                    fraction=4.75/101., aspect=20.)
            cbar = plt.colorbar(imax, cax=cax, **kw)
            cbar.ax.tick_params(labelsize=8)




    ################

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()

    return None


#############################################################


def plot_rotcurve_components(gal,
            overwrite=False,
            overwrite_curve_files=False,
            outpath = None,
            plotfile = None,
            fname_model_matchdata = None,
            fname_model_finer = None,
            fname_intrinsic = None,
            plot_type='pdf',
            **plot_kwargs):

    if (plotfile is None) & (outpath is None):
        raise ValueError

    if fname_intrinsic is None:
        fname_intrinsic = '{}{}_vcirc_tot_bary_dm.dat'.format(outpath, gal.name)


    # Check if the rot curves are done:
    if overwrite_curve_files:
        curve_files_exist_int = False
    else:
        curve_files_exist_int = os.path.isfile(fname_intrinsic)

    if not curve_files_exist_int:
        # *_vcirc_tot_bary_dm.dat
        create_vel_profile_files_intrinsic(gal=gal, outpath=outpath,
                    fname_intrinsic=fname_intrinsic,
                    overwrite=overwrite_curve_files)

    for obs_name in gal.observations:
        obs = gal.observations[obs_name]
        if obs.instrument.ndim == 1:
            plot_single_obs_rotcurve_components(obs, gal.model,
                        gal_name=gal.name,  overwrite=overwrite,
                        dscale=gal.dscale, z=gal.z,
                        overwrite_curve_files=overwrite_curve_files,
                        outpath = outpath, plotfile = plotfile,
                        fname_model_matchdata = fname_model_matchdata,
                        fname_model_finer = fname_model_finer,
                        fname_intrinsic = fname_intrinsic,
                        plot_type=plot_type, **plot_kwargs)

    return None

def plot_single_obs_rotcurve_components(obs, model,
            dscale=None, z=None,
            gal_name = None,
            overwrite=False, overwrite_curve_files=False,
            outpath = None,
            plotfile = None,
            fname_model_matchdata = None,
            fname_model_finer = None,
            fname_intrinsic = None,
            plot_type='pdf',
            **plot_kwargs):

    if (plotfile is None) & (outpath is None):
        raise ValueError
    fbase = '{}{}_{}'.format(outpath, gal_name, obs.name)
    if plotfile is None:
        plotfile = '{}_rot_components.{}'.format(fbase, plot_type)
    if fname_model_matchdata is None:
        fname_model_matchdata = '{}_out-1dplots.txt'.format(fbase)
    if fname_model_finer is None:
        fname_model_finer = '{}_out-1dplots_finer_sampling.txt'.format(fbase)

    # check if the file exists:
    if overwrite:
        file_exists = False
    else:
        file_exists = os.path.isfile(plotfile)

    # Check if the rot curves are done:
    if overwrite_curve_files:
        # curve_files_exist_int = False
        curve_files_exist_obs1d = False
        file_exists = False
    else:
        # curve_files_exist_int = os.path.isfile(fname_intrinsic)
        curve_files_exist_obs1d = os.path.isfile(fname_model_finer)

    if not curve_files_exist_obs1d:
        # *_out-1dplots_finer_sampling.txt, *_out-1dplots.txt
        create_vel_profile_files_obs1d(obs=obs, model=model, dscale=dscale,
                    gal_name=gal_name, outpath=outpath,
                    fname_finer=fname_model_finer,
                    fname_model_matchdata=fname_model_matchdata,
                    overwrite=overwrite_curve_files)


    if not file_exists:
        # ---------------------------------------------------------------------------
        # Read in stuff:
        model_obs = read_bestfit_1d_obs_file(filename=fname_model_finer)
        model_int = read_model_intrinsic_profile(filename=fname_intrinsic)

        deg2rad = np.pi/180.
        sini = np.sin(model.geometries[obs.name].inc.value*deg2rad)

        vel_asymm_drift_sq = model.kinematic_options.get_asymm_drift_profile(model_int.rarr,
                                                    model, tracer=obs.tracer)
        vsq = model_int.data['vcirc_tot'] ** 2 - vel_asymm_drift_sq
        vsq[vsq<0] = 0.

        model_int.data['vrot'] = np.sqrt(vsq)

        model_int.data['vrot_sini'] = model_int.data['vrot']*sini

        sini_l = np.sin(np.max([model.geometries[obs.name].inc.value - 5., 0.])*deg2rad)
        sini_u = np.sin(np.min([model.geometries[obs.name].inc.value + 5., 90.])*deg2rad)

        model_int.data['vcirc_tot_linc'] = np.sqrt((model_int.data['vrot_sini']/sini_l)**2 + vel_asymm_drift_sq )
        model_int.data['vcirc_tot_uinc'] = np.sqrt((model_int.data['vrot_sini']/sini_u)**2 + vel_asymm_drift_sq )


        ######################################
        # Setup plot:
        f = plt.figure()
        scale = 3.5

        ncols = 2
        nrows = 1

        wspace = 0.2
        hspace = 0.2
        f.set_size_inches(1.1*ncols*scale, nrows*scale)
        gs = gridspec.GridSpec(nrows, ncols, wspace=wspace, hspace=hspace)
        axes = []
        for i in range(nrows):
            for j in range(ncols):
                # Comparison:
                axes.append(plt.subplot(gs[i,j]))


        keyxtitle = r'Radius [arcsec]'
        keyxtitle_alt = r'Radius [kpc]'

        keyytitle = r'Velocity [km/s]'
        keyytitle_fdm = r'DM Fraction'

        errbar_lw = 0.5
        errbar_cap = 1.5
        lw = 1.5

        fontsize_ticks = 9.
        fontsize_label = 10.
        fontsize_ann = 8.
        fontsize_title = 10.
        fontsize_leg= 7.5

        color_arr = ['mediumblue', 'mediumturquoise', 'orange', 'red', 'blueviolet', 'dimgrey']

        # ++++++++++++++++++++++++++++++++++++
        ax = axes[0]


        xlim = [-0.05, np.max([np.max(np.abs(obs.data.rarr)) + 0.5, 2.0])]
        xlim2 = np.array(xlim) / dscale
        ylim = [0., np.max(model_int.data['vcirc_tot'])*1.15]



        ax.plot( model_obs.rarr, model_obs.data['velocity'],
            c='red', lw=lw, zorder=3., label=r'$V_{\mathrm{rot}} \sin(i)$ observed')

        ax.axhline(y=model.dispersions[obs.tracer].sigma0.value, ls='--', color='blueviolet',
                zorder=-20., label=r'Intrinsic $\sigma_0$')

        msk = obs.data.mask
        if hasattr(obs.data, 'mask_velocity'):
            if obs.data.mask_velocity is not None:
                msk = obs.data.mask_velocity

        # Masked points
        ax.errorbar( np.abs(obs.data.rarr[~msk]), np.abs(obs.data.data['velocity'][~msk]),
                xerr=None, yerr = obs.data.error['velocity'][~msk],
                marker=None, ls='None', ecolor='lightgrey', zorder=4.,
                alpha=0.75, lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
        ax.scatter( np.abs(obs.data.rarr[~msk]), np.abs(obs.data.data['velocity'][~msk]),
                    edgecolor='lightgrey', facecolor='whitesmoke', marker='s', s=25, lw=1,
                    zorder=5., label=None)

        # Unmasked points
        ax.errorbar( np.abs(obs.data.rarr[msk]), np.abs(obs.data.data['velocity'][msk]),
                xerr=None, yerr = obs.data.error['velocity'][msk],
                marker=None, ls='None', ecolor='dimgrey', zorder=4.,
                alpha=0.75, lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
        ax.scatter( np.abs(obs.data.rarr[msk]), np.abs(obs.data.data['velocity'][msk]),
            edgecolor='dimgrey', facecolor='white', marker='s', s=25, lw=1, zorder=5., label='Data')


        ax2 = ax.twiny()

        ax2.plot(model_int.rarr, model_int.data['vrot_sini'],
            c='orange', lw=lw, label=r'$V_{\mathrm{rot}} \sin(i)$ intrinsic', zorder=1.)
        #
        ax2.plot(model_int.rarr, model_int.data['vrot'],
            c='mediumturquoise', lw=lw, label=r'$V_{\mathrm{rot}}$ intrinsic', zorder=0.)
        ####
        ax2.fill_between(model_int.rarr, model_int.data['vcirc_tot_linc'], model_int.data['vcirc_tot_uinc'],
            color='mediumblue', alpha=0.1, lw=0, label=None, zorder=-1.)
        ####
        ax2.plot(model_int.rarr, model_int.data['vcirc_tot'],
            c='mediumblue', lw=lw, label=r'$V_{\mathrm{c}}$ intrinsic', zorder=2.)



        ax.annotate(r'{}: {} $z={:0.1f}$'.format(gal_name, obs.name, z), (0.5, 0.96),
                xycoords='axes fraction', ha='center', va='top', fontsize=fontsize_title)

        ax.set_xlabel(keyxtitle, fontsize=fontsize_label)
        ax.set_ylabel(keyytitle, fontsize=fontsize_label)
        ax2.set_xlabel(keyxtitle_alt, fontsize=fontsize_label)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax2.set_xlim(xlim2)
        ax.tick_params(labelsize=fontsize_ticks)
        ax2.tick_params(labelsize=fontsize_ticks)

        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(50.))
        ax.yaxis.set_minor_locator(MultipleLocator(10.))

        ax2.xaxis.set_major_locator(MultipleLocator(5.))
        ax2.xaxis.set_minor_locator(MultipleLocator(1.))

        ax.set_zorder(ax2.get_zorder()+1)
        ax.patch.set_visible(False)



        handles, lbls = ax.get_legend_handles_labels()
        handles2, lbls2 = ax2.get_legend_handles_labels()
        handles_arr = handles2[::-1]
        labels_arr = lbls2[::-1]
        handles_arr.extend(handles)
        labels_arr.extend(lbls)

        frameon = True
        borderpad = 0.25
        markerscale = 1.
        labelspacing= 0.25
        handletextpad= 0.05
        handlelength = 0.
        legend = ax.legend(handles_arr, labels_arr,
            labelspacing=labelspacing, borderpad=borderpad,
            markerscale=markerscale,
            handletextpad=handletextpad,
            handlelength=handlelength,
            loc='lower right',
            frameon=frameon, numpoints=1,
            scatterpoints=1,
            fontsize=fontsize_leg)


        for ind,text in enumerate(legend.get_texts()):
            legend.legend_handles[ind].set_visible(False)
            #text.set_color(color_arr[ind])
            try:
                text.set_color(legend.legend_handles[ind]._color)
            except:
                text.set_color(legend.legend_handles[ind]._original_edgecolor)

        # ++++++++++++++++++++++++++++++++++++
        ax = axes[1]

        color_arr = ['mediumblue', 'limegreen', 'purple']


        xlim2 = [xlim2[0], np.max(model_int.rarr)+0.1]
        xlim = np.array(xlim2) * dscale

        ax2 = ax.twiny()

        ax2.plot(model_int.rarr, model_int.data['vcirc_tot'],
                c='mediumblue', lw=lw, label=r'$V_{\mathrm{c}}$ intrinsic', zorder=2.)

        ax2.plot(model_int.rarr, model_int.data['vcirc_bar'],
                c='limegreen', lw=lw, label=r'$V_{\mathrm{bar}}$ intrinsic', zorder=1.)

        ax2.plot(model_int.rarr, model_int.data['vcirc_dm'],
                c='purple', lw=lw, label=r'$V_{\mathrm{DM}}$ intrinsic', zorder=0.)
        ###


        if 'disk+bulge' in model.components.keys():
            ax2.axvline(x=model.components['disk+bulge'].r_eff_disk.value, ls='--', color='dimgrey', zorder=-10.)
            ax2.annotate(r'$R_{\mathrm{eff}}$',
                (model.components['disk+bulge'].r_eff_disk.value + 0.05*(xlim2[1]-xlim2[0]), 0.025*(ylim[1]-ylim[0])), # 0.05
                xycoords='data', ha='left', va='bottom', color='dimgrey', fontsize=fontsize_ann)

        ###
        ax3 = ax2.twinx()
        fdm = model_int.data['vcirc_dm']**2/model_int.data['vcirc_tot']**2
        ax3.plot(model_int.rarr, fdm, ls='-', lw=1,
                    color='grey', alpha=0.8)
        #

        ax2.set_zorder(ax3.get_zorder()+1+ax.get_zorder())
        ax2.patch.set_visible(False)
        ax.patch.set_visible(False)

        ax.set_xlabel(keyxtitle, fontsize=fontsize_label)
        ax.set_ylabel(keyytitle, fontsize=fontsize_label)
        ax2.set_xlabel(keyxtitle_alt, fontsize=fontsize_label)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax2.set_xlim(xlim2)
        ax.tick_params(labelsize=fontsize_ticks)
        ax2.tick_params(labelsize=fontsize_ticks)

        ax.yaxis.set_major_locator(MultipleLocator(50.))
        ax.yaxis.set_minor_locator(MultipleLocator(10.))

        ax.xaxis.set_major_locator(MultipleLocator(1.))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax2.xaxis.set_major_locator(MultipleLocator(10.))
        ax2.xaxis.set_minor_locator(MultipleLocator(2.))

        ax3.set_ylabel(keyytitle_fdm, fontsize=fontsize_label-2, color='grey')
        ax3.tick_params(axis='y', direction='in', color='black', labelsize=fontsize_ticks-2, colors='grey')
        ax3.set_ylim([0.,1.])

        ax.yaxis.set_ticks_position('left')
        ax2.yaxis.set_ticks_position('right')

        ax3.yaxis.set_major_locator(MultipleLocator(0.2))
        ax3.yaxis.set_minor_locator(MultipleLocator(0.05))

        handles2, lbls2 = ax2.get_legend_handles_labels()
        handles_arr = handles2
        labels_arr = lbls2
        #
        legend = ax2.legend(handles_arr, labels_arr,
            labelspacing=labelspacing, borderpad=borderpad,
            markerscale=markerscale,
            handletextpad=handletextpad,
            handlelength=handlelength,
            loc='lower right',
            frameon=frameon, numpoints=1,
            scatterpoints=1,
            fontsize=fontsize_leg)
        #
        for ind,text in enumerate(legend.get_texts()):
            legend.legend_handles[ind].set_visible(False)
            text.set_color(color_arr[ind])


        #############################################################
        # Save to file:

        plt.savefig(plotfile, bbox_inches='tight', dpi=300)
        plt.close()

    return None

def make_clean_bayesian_plot_names(bayesianResults, short=False):
    names = []
    for key in bayesianResults.free_param_names.keys():
        for i in range(len(bayesianResults.free_param_names[key])):
            param = bayesianResults.free_param_names[key][i]
            key_nice = " ".join(key.split("_"))
            param_nice = " ".join(param.split("_"))
            if short:
                names.append(param_nice)
            else:
                names.append(key_nice+': '+param_nice)

    return names


#############################################################

def extract_1D_2D_obs_from_cube(obs, model, slit_width=None, slit_pa=None,
                                aper_dist=None, inst_corr=True, fill_mask=False):

    obs_extract = OrderedDict()

    obs_extract['extract_2D'] = extract_2D_from_cube(obs.data.data, obs,
                                      errcube=obs.data.error,
                                      modcube=obs.model_data.data,
                                      inst_corr=inst_corr)

    obs_extract['extract_1D'] = extract_1D_from_cube(obs.data.data, obs, model,
                                      errcube=obs.data.error,
                                      modcube=obs.model_data.data,
                                      slit_width=slit_width, slit_pa=slit_pa,
                                      aper_dist=aper_dist, inst_corr=inst_corr,
                                      fill_mask=fill_mask)

    # Add geometries for these obs to model:
    geom1d = copy.deepcopy(model.geometries[obs.name])
    geom1d.name = "{}_1D".format(model.geometries[obs.name].name)
    geom1d.obs_name = obs_extract['extract_1D'].name

    geom2d = copy.deepcopy(model.geometries[obs.name])
    geom2d.name = "{}_2D".format(model.geometries[obs.name].name)
    geom2d.obs_name = obs_extract['extract_2D'].name

    model.add_component(geom1d)
    model.add_component(geom2d)

    return obs_extract, model


def aper_centers_arcsec_from_cube(datacube, obs, model, mask=None,
                                  slit_width=None, slit_pa=None,
                                  aper_dist=None, fill_mask=False):

    if slit_width is None:
        try:
            slit_width = obs.instrument.beam.major.to(u.arcsec).value
        except:
            slit_width = obs.instrument.beam.major_fwhm.to(u.arcsec).value
    if slit_pa is None:
        slit_pa = model.geometries[obs.name].pa.value


    if mask is None:
        mask = obs.data.mask.copy()

    pixscale = obs.instrument.pixscale.value

    rpix = slit_width/pixscale/2.

    #############################


    if aper_dist is None:
        # # EVERY PIXEL
        aper_dist_pix = 1. #pixscale #rstep

    else:
        aper_dist_pix = aper_dist/pixscale

    # Aper centers: pick roughly number fitting into size:
    nx = datacube.shape[2]
    ny = datacube.shape[1]
    try:
        center_pixel = (obs.mod_options.xcenter + model.geometries[obs.name].xshift.value,
                        obs.mod_options.ycenter + model.geometries[obs.name].yshift.value)
    except:
        center_pixel = (int(nx / 2.) + model.geometries[obs.name].xshift,
                        int(ny / 2.) + model.geometries[obs.name].yshift)

    cPA = np.cos(slit_pa * np.pi/180.)
    sPA = np.sin(slit_pa * np.pi/180.)
    ###################
    if fill_mask:
        diag_step = False
        if np.abs(cPA) >= np.abs(sPA):
            nmax = ny / np.abs(cPA)
            if diag_step & (aper_dist_pix == 1.):
                aper_dist_pix *= 1. / np.abs(cPA)
        else:
            nmax = nx / np.abs(sPA)
            if diag_step & (aper_dist_pix == 1.):
                aper_dist_pix *= 1. / np.abs(sPA)
        rMA_arr = None
    else:
        # Just use unmasked range:
        maskflat = np.sum(mask, axis=0)
        maskflat[maskflat>0] = 1
        mask2D = np.array(maskflat, dtype=bool)
        rstep_A = 0.25

        rMA_tmp = 0
        rMA_arr = []
        # PA is to Blue; rMA_arr is [Blue (neg), Red (pos)]
        # but for PA definition blue will be pos step; invert at the end
        for fac in [1.,-1.]:
            ended_MA = False
            while not ended_MA:
                rMA_tmp += fac * rstep_A
                xtmp = rMA_tmp * -sPA + center_pixel[0]
                ytmp = rMA_tmp * cPA  + center_pixel[1]
                if (xtmp < 0) | (xtmp > mask2D.shape[1]-1) | (ytmp < 0) | (ytmp > mask2D.shape[0]-1):
                    rMA_arr.append(-1.*(rMA_tmp - fac*rstep_A))  # switch sign: pos / blue for calc becomes neg
                    rMA_tmp = 0
                    ended_MA = True
                elif not mask2D[int(np.round(ytmp)), int(np.round(xtmp))]:
                    rMA_arr.append(-1.*rMA_tmp)  # switch sign: pos / blue for calc becomes neg
                    rMA_tmp = 0
                    ended_MA = True
        nmax = None
        rMA_arr = np.array(rMA_arr)

    if rMA_arr is not None:
        aper_centers_pix = np.arange(np.sign(rMA_arr[0])*np.floor(np.abs(rMA_arr[0])),
                                     np.sign(rMA_arr[1])*np.floor(np.abs(rMA_arr[1]))+1., 1.)
    else:
        nap = int(np.floor(nmax/aper_dist_pix))  # If aper_dist_pix = 1, than nmax apertures.
                                                    # Otherwise, fewer and more spaced out
        # Make odd
        if nap % 2 == 0.:
            if fill_mask:
                nap -= 1
            else:
                nap += 1
        aper_centers_pix = np.linspace(0.,nap-1, num=nap) - int(nap / 2.)

    ######################################

    aper_centers_arcsec = aper_centers_pix*aper_dist_pix*pixscale
    return aper_centers_arcsec



###################################################################
###################################################################
###################################################################

def extract_1D_from_cube(datacube, obs, model, errcube=None, modcube=None, mask=None,
                         slit_width=None, slit_pa=None, aper_dist=None,
                         inst_corr=True, fill_mask=False):

    # ############################################

    # Set up the observation
    obs1d = Observation(name="{}_1d_extract".format(obs.name),
                        tracer=obs.tracer)
    for key in ['flux', 'velocity', 'dispersion']:
        obs1d.fit_options.__dict__['fit_{}'.format(key)] = True
    inst1d = copy.deepcopy(obs.instrument)
    inst1d.ndim = 1

    if slit_width is None:
        try:
            slit_width = obs.instrument.beam.major.to(u.arcsec).value
        except:
            slit_width = obs.instrument.beam.major_fwhm.to(u.arcsec).value
    if slit_pa is None:
        slit_pa = model.geometries[obs.name].pa.value

    if mask is None:
        mask = obs.data.mask.copy()

    #pixscale = obs.instrument.pixscale.value

    inst1d.slit_width = slit_width
    inst1d.slit_pa = slit_pa


    obs1d.mod_options.xcenter = obs.mod_options.xcenter
    obs1d.mod_options.ycenter = obs.mod_options.ycenter


    data1d, inst1d = extract_1D_from_cube_general(datacube, obs, model, inst1d,
                                     errcube=errcube, mask=mask,
                                     slit_width=slit_width, slit_pa=slit_pa,
                                     aper_dist=aper_dist,
                                     inst_corr=inst_corr, fill_mask=fill_mask)

    obs1d.instrument = inst1d
    obs1d.data = data1d

    if modcube is not None:
        mod1d, _ = extract_1D_from_cube_general(modcube, obs, model, copy.deepcopy(inst1d),
                                         errcube=None, mask=mask,
                                         slit_width=slit_width, slit_pa=slit_pa,
                                         aper_dist=aper_dist,
                                         inst_corr=inst_corr, fill_mask=fill_mask)
        obs1d.model_data = mod1d


    return obs1d

def extract_1D_from_cube_general(datacube, obs, model, inst1d, errcube=None, mask=None,
                                 slit_width=None, slit_pa=None, aper_dist=None,
                                 inst_corr=True, fill_mask=False):

    # ############################################


    pixscale = inst1d.pixscale.value
    slit_width = inst1d.slit_width
    slit_pa = inst1d.slit_pa


    rpix = slit_width/pixscale/2.

    # Aper centers: pick roughly number fitting into size:
    nx = datacube.shape[2]
    ny = datacube.shape[1]
    try:
        center_pixel = (obs.mod_options.xcenter + model.geometries[obs.name].xshift.value,
                        obs.mod_options.ycenter + model.geometries[obs.name].yshift.value)
    except:
        center_pixel = (int(nx / 2.) + model.geometries[obs.name].xshift,
                        int(ny / 2.) + model.geometries[obs.name].yshift)

    aper_centers_arcsec = aper_centers_arcsec_from_cube(datacube, obs, model,
                mask=mask, slit_width=slit_width, slit_pa=slit_pa,
                aper_dist=aper_dist, fill_mask=fill_mask)


    #######

    vel_arr = datacube.spectral_axis.to(u.km/u.s).value

    apertures = CircApertures(rarr=aper_centers_arcsec, slit_PA=slit_pa, rpix=rpix,
             nx=nx, ny=ny, center_pixel=center_pixel, pixscale=pixscale,
             moment=obs.instrument.moment)

    data_scaled = datacube.unmasked_data[:].value


    if errcube is not None:
        ecube = errcube.unmasked_data[:].value * mask
    else:
        ecube = None

    aper_centers, flux1d, vel1d, disp1d = apertures.extract_1d_kinematics(spec_arr=vel_arr,
                    cube=data_scaled, mask=mask, err=ecube,
                    center_pixel = center_pixel, pixscale=pixscale)


    if not fill_mask:
        # Remove points where the fit was bad
        ind = np.isfinite(vel1d) & np.isfinite(disp1d)

        data1d = Data1D(r=aper_centers[ind], velocity=vel1d[ind],
                        vel_disp=disp1d[ind], flux=flux1d[ind],
                        inst_corr=inst_corr)
        apertures_redo = CircApertures(rarr=aper_centers_arcsec[ind], slit_PA=slit_pa, rpix=rpix,
                            nx=nx, ny=ny, center_pixel=center_pixel, pixscale=pixscale,
                            moment=obs.instrument.moment)
        inst1d.apertures = apertures_redo
    else:
        data1d = Data1D(r=aper_centers, velocity=vel1d,vel_disp=disp1d, flux=flux1d,
                        inst_corr=inst_corr)

        inst1d.apertures = apertures

    data1d.profile1d_type = 'circ_ap_cube'

    if fill_mask:
        # Unmask any fully masked bits, and fill with the other mask:
        mask2d = np.sum(mask, axis=0)
        whzero = np.where(mask2d == 0)
        maskspec = np.sum(np.sum(mask, axis=2), axis=1)
        maskspec[maskspec>0] = 1
        mask_filled = np.tile(maskspec.reshape((maskspec.shape[0],1,1)), (1, data_scaled.shape[1], data_scaled.shape[2]))
        mask[:, whzero[0], whzero[1]] = mask_filled[:, whzero[0], whzero[1]]

        if errcube is not None:
            ecube = errcube.unmasked_data[:].value * mask
        else:
            ecube = None
        aper_centers, flux1d, vel1d, disp1d = apertures.extract_1d_kinematics(spec_arr=vel_arr,
                        cube=data_scaled, mask=mask, err=ecube,
                        center_pixel = center_pixel, pixscale=pixscale)
        data1d.filled_mask_data = Data1D(r=aper_centers, velocity=vel1d,
                                         vel_disp=disp1d, flux=flux1d,
                                         inst_corr=inst_corr)


    return data1d, inst1d

def extract_2D_from_cube(cube, obs, errcube=None, modcube=None, inst_corr=True):

    # Set up the observation
    obs2d = Observation(name="{}_2d_extract".format(obs.name),
                        tracer=obs.tracer)
    for key in ['flux', 'velocity', 'dispersion']:
        obs2d.fit_options.__dict__['fit_{}'.format(key)] = True
    inst2d = copy.deepcopy(obs.instrument)
    inst2d.ndim = 2

    # cube must be SpectralCube instance!
    data2d = extract_2D_from_cube_general(cube, err=errcube, mask=obs.data.mask,
                                          inst_corr=inst_corr,
                                          directly_correct_LSF=False,
                                          LSF_disp_kms=0,
                                          moment=obs.instrument.moment,
                                          pixscale=obs.instrument.pixscale.value,
                                          gauss_extract_with_c=obs.mod_options.gauss_extract_with_c)

    obs2d.instrument = inst2d
    obs2d.data = data2d

    if modcube is not None:
        mod2d = extract_2D_from_cube_general(modcube, err=None, mask=obs.data.mask,
                                              inst_corr=inst_corr,
                                              directly_correct_LSF=False,
                                              LSF_disp_kms=0,
                                              moment=obs.instrument.moment,
                                              pixscale=obs.instrument.pixscale.value,
                                              gauss_extract_with_c=obs.mod_options.gauss_extract_with_c)

        obs2d.model_data = mod2d

    return obs2d

def extract_2D_from_cube_general(cube, err=None, mask=None,
                                 inst_corr=True, pixscale=None, moment=False,
                                 directly_correct_LSF=False, LSF_disp_kms=0,
                                 gauss_extract_with_c=True):
    # cube must be SpectralCube instance!
    orig_mask = copy.deepcopy(mask)
    # mask = BooleanArrayMask(mask= np.array(mask, dtype=bool),
    #                         wcs=cube.wcs)
    if orig_mask is None:
        mask_start = np.ones(cube.shape)
    else:
        mask_start = copy.deepcopy(mask)

    mask = BooleanArrayMask(mask= np.array(mask_start, dtype=bool),
                            wcs=cube.wcs)


    datacube = SpectralCube(data=cube.unmasked_data[:].value, mask=mask, wcs=cube.wcs)
    if err is not None:
        err_cube =  SpectralCube(data=err.unmasked_data[:].value, mask=mask, wcs=err.wcs)

    if moment:
        extrac_type = 'moment'
    else:
        extrac_type = 'gauss'

    if extrac_type == 'moment':
        flux = datacube.moment0().to(u.km/u.s).value
        vel = datacube.moment1().to(u.km/u.s).value
        disp = datacube.linewidth_sigma().to(u.km/u.s).value
    elif extrac_type == 'gauss':
        mom0 = datacube.moment0().to(u.km/u.s).value
        mom1 = datacube.moment1().to(u.km/u.s).value
        mom2 = datacube.linewidth_sigma().to(u.km/u.s).value

        # Clean up NaNs in moms for initial guesses:
        mom0[~np.isfinite(mom0)] = 0.0
        mom1[~np.isfinite(mom1)] = 0.0
        mom2[~np.isfinite(mom2)] = 20.0

        flux = np.zeros(mom0.shape)
        vel = np.zeros(mom0.shape)
        disp = np.zeros(mom0.shape)
        # ++++++++++
        my_least_chi_squares_1d_fitter = None
        if (gauss_extract_with_c) & (_loaded_LeastChiSquares1D):
            if gauss_extract_with_c is not None and \
                gauss_extract_with_c is not False:

                # we will use the C++ LeastChiSquares1D to run the 1d spectral fitting
                # but note that if a spectrum has data all too close to zero, it will fail.
                # try to prevent this by excluding too low data

                # CLEAN DATA
                data_cleaned = datacube.unmasked_data[:,:,:].value

                if err is not None:
                    dataerr = err_cube.unmasked_data[:,:,:].value
                    dataerr[dataerr==0.] = 99.
                    dataerr[dataerr==99.] = dataerr.min()
                else:
                    dataerr = None

                # data_cleaned = copy.deepcopy(datacube.unmasked_data[:,:,:].value)
                # this_fitting_mask = 'auto'
                this_fitting_mask = None
                if mask_start is not None:
                    this_fitting_mask = copy.copy(mask_start)

                # # Only do verbose if logging level is DEBUG or lower
                # if logger.level <= logging.DEBUG:
                #     this_fitting_verbose = True
                # else:
                #     this_fitting_verbose = False
                ## Force non-verbose, because multiprocessing pool
                ##    resets logging to logger.level = 0....
                this_fitting_verbose = False


                # do the least chisquares fitting
                my_least_chi_squares_1d_fitter = LeastChiSquares1D(\
                        x = datacube.spectral_axis.to(u.km/u.s).value,
                        data = data_cleaned,
                        dataerr = dataerr,
                        datamask = this_fitting_mask,
                        initparams = np.array([mom0 / np.sqrt(2 * np.pi) / np.abs(mom2), mom1, mom2]),
                        nthread = 4,
                        verbose = this_fitting_verbose)
        if my_least_chi_squares_1d_fitter is not None:
            logger.debug('my_least_chi_squares_1d_fitter '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
            my_least_chi_squares_1d_fitter.runFitting()
            flux = my_least_chi_squares_1d_fitter.outparams[0,:,:] * np.sqrt(2 * np.pi) * my_least_chi_squares_1d_fitter.outparams[2,:,:]
            vel = my_least_chi_squares_1d_fitter.outparams[1,:,:]
            disp = my_least_chi_squares_1d_fitter.outparams[2,:,:]
            flux[np.isnan(flux)] = 0.0 #<DZLIU><DEBUG># 20210809 fixing this bug
            logger.debug('my_least_chi_squares_1d_fitter '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#

            if np.sum(np.isfinite(vel)) == 0:
                alt_fit = True
            else:
                alt_fit = False
        else:
            alt_fit = True

        if alt_fit:
            # HAS ERROR:
            if err is not None:
                # print("Doing alt fit: astropy model!")
                wgt_cube = np.array(datacube.mask._mask, dtype=np.int64)  # use mask as weights to get "weighted" solution
                try:
                    ecube = err_cube.filled_data[:].value
                    ecube[ecube==99.] = ecube.min()
                    wgt_cube = 1./(ecube)
                except:
                   pass

                for i in range(mom0.shape[0]):
                    for j in range(mom0.shape[1]):
                        mod = apy_mod.models.Gaussian1D(amplitude=mom0[i,j] / np.sqrt(2 * np.pi * mom2[i,j]**2),
                                                mean=mom1[i,j],
                                                stddev=np.abs(mom2[i,j]))
                        mod.amplitude.bounds = (0, None)
                        mod.stddev.bounds = (0, None)
                        mod.mean.bounds = (datacube.spectral_axis.to(u.km/u.s).value.min(), datacube.spectral_axis.to(u.km/u.s).value.max())

                        fitter = apy_mod.fitting.LevMarLSQFitter()

                        wgts = wgt_cube[:,i,j]
                        if (np.max(np.abs(wgts)) == 0):
                            wgts = None

                        ########################
                        # Masked fit:
                        spec_arr = datacube.spectral_axis.to(u.km/u.s).value
                        flux_arr = datacube.filled_data[:,i,j].value

                        if np.isfinite(flux_arr).sum() >= len(mod._parameters):
                            if wgts is not None:
                                whgood = (np.isfinite(flux_arr) & np.isfinite(wgts))
                            else:
                                whgood = (np.isfinite(flux_arr))
                            spec_arr = spec_arr[whgood]
                            if wgts is not None:
                                wgts = wgts[whgood]
                            flux_arr = flux_arr[whgood]


                            ########################
                            try:
                                best_fit = fitter(mod, spec_arr, flux_arr, weights=wgts)

                                flux[i,j] = np.sqrt( 2. * np.pi) * best_fit.stddev.value * best_fit.amplitude
                                vel[i,j] = best_fit.mean.value
                                disp[i,j] = best_fit.stddev.value
                            except:
                                flux[i, j] = vel[i,j] = disp[i,j] = np.nan

                        else:
                            flux[i, j] = vel[i,j] = disp[i,j] = np.nan

            # NO ERROR:
            else:
                for i in range(mom0.shape[0]):
                    for j in range(mom0.shape[1]):
                        if i==0 and j==0:
                            logger.debug('gaus_fit_sp_opt_leastsq '+str(mom0.shape[0])+'x'+str(mom0.shape[1])+' '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
                        best_fit = gaus_fit_sp_opt_leastsq(datacube.spectral_axis.to(u.km/u.s).value,
                                            datacube.unmasked_data[:,i,j].value,
                                            mom0[i,j], mom1[i,j], mom2[i,j])
                        flux[i,j] = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                        vel[i,j] = best_fit[1]
                        disp[i,j] = best_fit[2]
                        if i==mom0.shape[0]-1 and j==mom0.shape[1]-1:
                            logger.debug('gaus_fit_sp_opt_leastsq '+str(mom0.shape[0])+'x'+str(mom0.shape[1])+' '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#

        # ----------


    ###########################
    # Flatten mask: only mask fully masked spaxels:
    msk3d_coll = np.sum(mask_start, axis=0)
    whmsk = np.where(msk3d_coll == 0)
    mask = np.ones((mask.shape[1], mask.shape[2]))
    mask[whmsk] = 0


    ## Check about instrument correction:
    if inst_corr & directly_correct_LSF:
        raise ValueError("Should not pre-correct for LSF (`directly_correct_LSF=True`) if setting `inst_corr=True` !!")

    if directly_correct_LSF & (LSF_disp_kms > 0.):
        disp = np.sqrt(disp**2 - LSF_disp_kms**2)
        disp[~np.isfinite(disp)] = 0   # Set the dispersion to zero when its below
                                       # below the instrumental dispersion


    # Artificially mask the bad stuff:
    mask[~np.isfinite(flux)] = 0
    mask[~np.isfinite(vel)] = 0
    mask[~np.isfinite(disp)] = 0
    flux[~np.isfinite(flux)] = 0
    vel[~np.isfinite(vel)] = 0
    disp[~np.isfinite(disp)] = 0

    if np.sum(mask) == 0:
        raise ValueError

    data2d = Data2D(pixscale=pixscale, velocity=vel, vel_disp=disp, mask=mask,
                    flux=flux, vel_err=None, vel_disp_err=None, flux_err=None,
                    inst_corr=inst_corr)

    return data2d



def extract_2D_from_cube_file(fname=None, fname_err=None, fname_mask=None,
                              moment=False, inst_corr=False,
                              directly_correct_LSF=False, LSF_disp_kms=0,
                              xcenter=None, ycenter=None,
                              gauss_extract_with_c=True):

    # LOAD STUFF: load cube into SpectralCube instance!

    cube_raw = fits.getdata(fname)
    hdr = fits.getheader(fname)

    # Should be square:
    if np.abs(hdr['CDELT1']) != np.abs(hdr['CDELT2']):
        raise ValueError

    if hdr['CUNIT1'].strip().upper() in ['DEGREE', 'DEG']:
        pixscale = np.abs(hdr['CDELT1']) * 3600.    # convert from deg CDELT1 to arcsec
    elif hdr['CUNIT1'].strip().upper() in ['ARCSEC']:
        pixscale = np.abs(hdr['CDELT1'])

    # Define WCS:
    w = WCS(hdr)
    spec_unit = u.km/u.s

    cube = SpectralCube(data=cube_raw, wcs=w).with_spectral_unit(spec_unit)

    if fname_err is not None:
        err_raw = fits.getdata(fname_err)
        err = SpectralCube(data=err_raw, wcs=w).with_spectral_unit(spec_unit)
    else:
        err = None
    if fname_mask is not None:
        mask = fits.getdata(fname_mask)
    else:
        mask = None


    data2d = extract_2D_from_cube_general(cube, err=err, mask=mask,
                                     moment=moment, inst_corr=inst_corr,
                                     directly_correct_LSF=directly_correct_LSF,
                                     LSF_disp_kms=LSF_disp_kms, pixscale=pixscale,
                                     gauss_extract_with_c=gauss_extract_with_c)

    data2d.xcenter = xcenter
    data2d.ycenter = ycenter

    return data2d

def plot_axes_flux_vel_disp(flux, vel, disp, axes=None,
                            mask=None, pixscale=None,
                            center_pixel_kin=None, PA_deg=None,
                            plottype='data',
                            label='',
                            show_contours=True,
                            apply_mask=True,
                            vrange_dict=None,
                            show_xlabels=True,
                            cmap=None,
                            **kwargs):


    ims = [flux, vel, disp]
    keyxarr = ['flux', 'velocity', 'dispersion']
    keyxtitlearr = ['Flux', r'$V$', r'$\sigma$']

    int_mode = "nearest"
    origin = 'lower'
    bad_color = 'white'
    color_annotate = 'black'
    if cmap is None:
        cmap = copy.copy(cmap_spectral_r)
        cmap.set_bad(color=bad_color)


    for i, im in enumerate(ims):
        ax = axes[i]

        xt = keyxtitlearr[i]
        yt = label

        if apply_mask:
            im[~mask] = np.NaN

        if vrange_dict is not None:
            vmin = vrange_dict[keyxarr[i]][0]
            vmax = vrange_dict[keyxarr[i]][1]
        else:
            vmin = None
            vmax = None
        imax = ax.imshow(im, cmap=cmap, interpolation=int_mode,
                         vmin=vmin, vmax=vmax, origin=origin)

        ####################################
        if show_contours:
            ax = plot_contours_2D_multitype(im, ax=ax, mapname=keyxarr[i],
                                            plottype=plottype,
                                            vmin=vmin, vmax=vmax,
                                            kwargs=kwargs)

        # Show a 1arcsec line:
        if pixscale is not None:
            ax = plot_ruler_arcsec_2D(ax, pixscale, len_arcsec=1.,
                                      ruler_loc='lowerright',
                                      color=color_annotate)

        if PA_deg is not None:
            ax = plot_major_minor_axes_2D_general(ax, im, mask,
                        center_pixel_kin=center_pixel_kin, PA_deg=PA_deg)

        ####################################

        if i == 0:
            ax.set_ylabel(yt)

        for pos in ['top', 'bottom', 'left', 'right']:
            ax.spines[pos].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        if show_xlabels:
            ax.set_title(xt)

        #########
        cax, kw = colorbar.make_axes_gridspec(ax, pad=0.01,
                fraction=4.75/101.,
                aspect=20.)
        cbar = plt.colorbar(imax, cax=cax, **kw)
        cbar.ax.tick_params(labelsize=8)

        axes[i] = ax

    return axes


def plot_2D_from_cube_files(fileout=None,
                            fname_cube=None, fname_cube2=None,
                            fname_err=None, fname_err2=None,
                            fname_mask=None, fname_mask2=None,
                            show_residual=True,
                            directly_correct_LSF=False, LSF_disp_kms=0,
                            directly_correct_LSF2=False, LSF_disp_kms2=0,
                            label='Gal', label2='Gal2',
                            moment=False, xcenter=None, ycenter=None,
                            PA_deg=None,
                            show_contours=True,
                            symmetric_residuals=True,
                            max_residual=100.,
                            match_ranges=True,
                            **kwargs):

    # Load data
    data1 = extract_2D_from_cube_file(fname=fname_cube, fname_err=fname_err,
                                      fname_mask=fname_mask,
                                      inst_corr=False,
                                      directly_correct_LSF=directly_correct_LSF,
                                      LSF_disp_kms=LSF_disp_kms, moment=moment,
                                      xcenter=xcenter, ycenter=ycenter)
    if fname_cube2 is not None:
        data2 = extract_2D_from_cube_file(fname=fname_cube2, fname_err=fname_err2,
                                          fname_mask=fname_mask2,
                                          inst_corr=False,
                                          directly_correct_LSF=directly_correct_LSF2,
                                          LSF_disp_kms=LSF_disp_kms2, moment=moment,
                                          xcenter=xcenter, ycenter=ycenter)


    # Setup plotting
    # Set contour defaults, if not specifed:
    for key in _kwargs_contour_defaults.keys():
        if key not in kwargs.keys():
            kwargs[key] = _kwargs_contour_defaults[key]

    cmap = copy.copy(cmap_spectral_r)
    bad_color = 'white'
    cmap.set_bad(color=bad_color)

    cmap_resid = copy.copy(cmap_rdbu_r)
    cmap_resid.set_bad(color=bad_color)
    cmap_resid.set_over(color='magenta')
    cmap_resid.set_under(color='blueviolet')



    # Setup plot structures:
    if xcenter is None:
        xcenter = (data1.data['flux'].shape[1]-1.)/ 2.
    if ycenter is None:
        ycenter = (data1.data['flux'].shape[0]-1.)/ 2.
    vrange_dict = {'flux': (None,None),
                   'velocity': (None,None),
                   'dispersion': (None,None)}
    cube1_dict = {'plottype': 'data',
                  'cmap': cmap,
                  'flux': data1.data['flux'],
                  'vel': data1.data['velocity'],
                  'disp': data1.data['dispersion'],
                  'mask': data1.mask,
                  'center_pixel_kin': (xcenter, ycenter),
                  'PA_deg': PA_deg,
                  'pixscale': data1.pixscale,
                  'label': label,
                  'vrange_dict': vrange_dict}

    cubes_list = [cube1_dict]
    if fname_cube2 is not None:
        if match_ranges:
            mins = []
            maxs = []
            keys = ['flux', 'velocity', 'dispersion']
            for key in keys:
                min = np.min([data1.data[key][np.isfinite(data1.data[key])].min(),
                              data2.data[key][np.isfinite(data2.data[key])].min()])
                max = np.max([data1.data[key][np.isfinite(data1.data[key])].max(),
                              data2.data[key][np.isfinite(data2.data[key])].max()])
                mins.append(min)
                maxs.append(max)
            vrange_dict = {'flux': (mins[0],maxs[0]),
                           'velocity': (mins[1],maxs[1]),
                           'dispersion': (mins[2],maxs[2])}
            cube1_dict['vrange_dict'] = vrange_dict
            cubes_list = [cube1_dict]
        cube2_dict = {'plottype': 'data',
                      'cmap': cmap,
                      'flux': data2.data['flux'],
                      'vel': data2.data['velocity'],
                      'disp': data2.data['dispersion'],
                      'mask': data2.mask,
                      'center_pixel_kin': (xcenter, ycenter),
                      'PA_deg': PA_deg,
                      'pixscale': data2.pixscale,
                      'label': label2,
                      'vrange_dict': vrange_dict}

        cubes_list.append(cube2_dict)
        if show_residual:
            if symmetric_residuals:
                vmin = -max_residual
                vmax = max_residual
                im = cube2_dict['flux']-cube1_dict['flux']
                fabsmax = np.max(np.abs(im[np.isfinite(im)]))
                fmin = -fabsmax
                fmax = fabsmax
            else:
                vmin = None
                vmax = None
                fmin = None
                fmax = None
            vrange_dict_resid = {'flux': (fmin,fmax),
                           'velocity': (vmin,vmax),
                           'dispersion': (vmin,vmax)}
            resid_dict = {'plottype': 'residual',
                          'cmap': cmap_resid,
                          'flux': cube2_dict['flux']-cube1_dict['flux'],
                          'vel': cube2_dict['vel']-cube1_dict['vel'],
                          'disp': cube2_dict['disp']-cube1_dict['disp'],
                          'mask': cube1_dict['mask'],
                          'center_pixel_kin': (xcenter, ycenter),
                          'PA_deg': PA_deg,
                          'pixscale': cube1_dict['pixscale'],
                          'label': 'Residuals: {}-{}'.format(label2,label),
                          'vrange_dict': vrange_dict_resid}
            cubes_list.append(resid_dict)



    ######################################
    # Setup axes:

    ncols = 3
    nrows = len(cubes_list)

    padx = pady = 0.25
    xextra = 0.25 #0.15
    yextra = 0.25
    scale = 2.5

    f = plt.figure()
    f.set_size_inches((ncols+(ncols-1)*padx+xextra)*scale,
                      (nrows+(nrows-1)*pady+yextra)*scale)

    padx = 0.2
    pady = 0.1
    gs02 = gridspec.GridSpec(nrows, ncols, wspace=padx, hspace=pady)
    axes = []
    for jj in range(nrows):
        for mm in range(ncols):
            axes.append(plt.subplot(gs02[jj,mm]))


    for i, cdict in enumerate(cubes_list):
        if i == 0:
            show_xlabels = True
        else:
            show_xlabels = False

        axes[ncols*i:(ncols)*(i+1)] = plot_axes_flux_vel_disp(cdict['flux'], cdict['vel'], cdict['disp'],
                                            axes=axes[ncols*i:(ncols)*(i+1)],
                                            mask=cdict['mask'],
                                            pixscale=cdict['pixscale'],
                                            center_pixel_kin=cdict['center_pixel_kin'],
                                            PA_deg=cdict['PA_deg'],
                                            plottype=cdict['plottype'],
                                            label=cdict['label'],
                                            vrange_dict=cdict['vrange_dict'],
                                            cmap=cdict['cmap'],
                                            show_xlabels=show_xlabels,
                                            show_contours=show_contours, **kwargs)



    #############################################################
    # Save plot to file or display directly:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()

    return None



###################################################################
###################################################################
###################################################################


#############################################################
#############################################################
#############################################################
# UTILITY FUNCTIONS
#############################################################


def show_1d_apers_plot(ax, obs, data1d, data2d, model=None, obsorig=None, alpha_aper=0.8, remove_shift=True):

    aper_centers = data1d.rarr
    slit_width = data1d.slit_width
    slit_pa = data1d.slit_pa
    rstep = obs.instrument.pixscale.value
    try:
        rstep1d = obs.instrument1d.pixscale.value
    except:
        rstep1d = rstep
    rpix = slit_width/rstep/2.

    aper_centers_pix = aper_centers/rstep

    if aper_centers[0] <= 0:
        # Starts from neg -> pos:
        pa = slit_pa
    else:
        # Goes "backwards": pos -> neg [but the aper shapes are ALSO stored this way]
        pa = slit_pa + 180

    print(" ndim={}:  xshift={}, yshift={}, vsys2d={}".format(obsorig.data.ndim,
                                    model.geometries[obs.name].xshift.value,
                                    model.geometries[obs.name].yshift.value,
                                    model.geometries[obs.name].vel_shift.value))



    nx = data2d.data['velocity'].shape[1]
    ny = data2d.data['velocity'].shape[0]

    try:
        center_pixel_kin = (obs.mod_options.xcenter + model.geometries[obs.name].xshift.value*rstep/rstep1d,
                            obs.mod_options.ycenter + model.geometries[obs.name].yshift.value*rstep/rstep1d)
    except:
        center_pixel_kin = (int(nx / 2.) + model.geometries[obs.name].xshift.value*rstep/rstep1d,
                            int(ny / 2.) + model.geometries[obs.name].yshift.value*rstep/rstep1d)

    if not remove_shift:
        try:
            center_pixel = (obs.mod_options.xcenter, obs.mod_options.ycenter)
        except:
            center_pixel = None
    else:
        center_pixel = center_pixel_kin



    if center_pixel is None:
        center_pixel = (int(nx / 2.) + model.geometries[obs.name].xshift.value*rstep/rstep1d,
                        int(ny / 2.) + model.geometries[obs.name].yshift.value*rstep/rstep1d)

    # +++++++++++++++++

    pyoff = 0.
    ax.scatter(center_pixel[0], center_pixel[1], color='magenta', marker='+')
    ax.scatter(center_pixel_kin[0], center_pixel_kin[1], color='cyan', marker='+')
    ax.scatter(int(nx / 2), int(ny / 2), color='lime', marker='+')

    # +++++++++++++++++

    # First determine the centers of all the apertures that fit within the cube
    xaps, yaps = calc_pix_position(aper_centers_pix, pa, center_pixel[0], center_pixel[1])


    cmstar = cmap_plasma
    cNorm = mplcolors.Normalize(vmin=0, vmax=len(xaps)-1)
    cmapscale = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmstar)

    for mm, (rap, xap, yap) in enumerate(zip(aper_centers, xaps, yaps)):
        circle = plt.Circle((xap+pyoff, yap+pyoff), rpix, color=cmapscale.to_rgba(mm, alpha=alpha_aper), fill=False)
        ax.add_artist(circle)
        if (mm == 0):
            ax.scatter(xap+pyoff, yap+pyoff, color=cmapscale.to_rgba(mm), marker='.')

    return ax


def plot_major_minor_axes_2D(ax, obs, model, im, mask, finer_step=True,
    lw_major = 3., lw_minor = 2.25, fac2 = 0.66, fac_len_minor_marker = 1./20.,
    color_kin_axes = 'black', color_kin_axes2 = 'white'):
    ####################################
    # Show MAJOR AXIS line, center:
    try:
        center_pixel_kin = (obs.mod_options.xcenter + model.geometries[obs.name].xshift.value,
                            obs.mod_options.ycenter + model.geometries[obs.name].yshift.value)
    except:
        center_pixel_kin = ((im.shape[1]-1.)/ 2. + model.geometries[obs.name].xshift.value,
                            (im.shape[0]-1.)/ 2. + model.geometries[obs.name].yshift.value)

    return plot_major_minor_axes_2D_general(ax, im, mask,
                        center_pixel_kin=center_pixel_kin,
                        PA_deg=model.geometries[obs.name].pa.value,
                        finer_step=finer_step,
                        lw_major=lw_major, lw_minor=lw_minor,
                        fac2=fac2, fac_len_minor_marker=fac_len_minor_marker,
                        color_kin_axes=color_kin_axes,
                        color_kin_axes2=color_kin_axes2)



def plot_major_minor_axes_2D_general(ax, im, mask,
    center_pixel_kin=None, PA_deg=None,
    finer_step=True,
    lw_major = 3., lw_minor = 2.25, fac2 = 0.66, fac_len_minor_marker = 1./20.,
    color_kin_axes = 'black', color_kin_axes2 = 'white'):
    ####################################
    # Show MAJOR AXIS line, center:
    if center_pixel_kin is None:
        center_pixel_kin = ((im.shape[1]-1.)/ 2., (im.shape[0]-1.)/ 2.)

    # Start going to neg, pos of center, at PA, and check if mask True/not
    #   in steps of pix, then rounding. if False: stop, and set 1 less as the end.

    cPA = np.cos(PA_deg * np.pi/180.)
    sPA = np.sin(PA_deg * np.pi/180.)

    A_xs = []
    A_ys = []
    if finer_step:
        rstep_A = 0.25
    else:
        rstep_A = 1.

    rMA_tmp = 0
    rMA_arr = []
    # PA is to Blue; rMA_arr is [Blue (neg), Red (pos)]
    # but for PA definition blue will be pos step; invert at the end
    for fac in [1.,-1.]:
        ended_MA = False
        while not ended_MA:
            rMA_tmp += fac * rstep_A
            xtmp = rMA_tmp * -sPA + center_pixel_kin[0]
            ytmp = rMA_tmp * cPA  + center_pixel_kin[1]
            if (xtmp < 0) | (xtmp > mask.shape[1]-1) | (ytmp < 0) | (ytmp > mask.shape[0]-1):
                A_xs.append((rMA_tmp - fac*rstep_A) * -sPA + center_pixel_kin[0])
                A_ys.append((rMA_tmp - fac*rstep_A) * cPA  + center_pixel_kin[1])
                rMA_arr.append(-1.*(rMA_tmp - fac*rstep_A))  # switch sign: pos / blue for calc becomes neg
                rMA_tmp = 0
                ended_MA = True
            elif not mask[int(np.round(ytmp)), int(np.round(xtmp))]:
                A_xs.append((rMA_tmp) * -sPA + center_pixel_kin[0])
                A_ys.append((rMA_tmp) * cPA  + center_pixel_kin[1])
                rMA_arr.append(-1.*rMA_tmp)  # switch sign: pos / blue for calc becomes neg
                rMA_tmp = 0
                ended_MA = True

    B_xs = []
    B_ys = []
    len_minor_marker = fac_len_minor_marker * im.shape[0]   # CONSTANT SIZE REL TO AXIS
    for fac in [-1.,1.]:
        B_xs.append(fac*len_minor_marker*0.5 * cPA  + center_pixel_kin[0])
        B_ys.append(fac*len_minor_marker*0.5 * sPA + center_pixel_kin[1])


    ax.plot(A_xs, A_ys, color=color_kin_axes, lw=lw_major, ls='-')
    ax.plot(B_xs, B_ys,color=color_kin_axes, lw=lw_minor, ls='-')

    ax.plot(A_xs, A_ys, color=color_kin_axes2, lw=lw_major*fac2, ls='-')
    ax.plot(B_xs, B_ys,color=color_kin_axes2, lw=lw_minor*fac2, ls='-')

    return ax


def plot_ruler_arcsec_2D(ax, pixscale, len_arcsec=0.5,
        ruler_unit='arcsec', dscale=None,
        ruler_loc='lowerright', color='black', ybase_offset=0.02,
        delx=0.075, dely=0.075, dely_text=0.06):
    ####################################
    # Show a ruler line:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    #len_line_angular = 0.5/(pixscale)
    len_line_angular = len_arcsec/(pixscale)
    if ruler_unit.lower() == 'arcsec':
        if len_arcsec % 1. == 0.:
            string = r'{}"'.format(int(len_arcsec))
        else:
            intpart = str(len_arcsec).split('.')[0]
            decpart = str(len_arcsec).split('.')[1]
            string = r'{}."{}'.format(intpart, decpart)
    elif ruler_unit.lower() == 'kpc':
        if len_arcsec/dscale % 1. == 0.:
            string = r'{:0.0f} kpc'.format(int(len_arcsec/dscale))
        else:
            string = r'{:0.1f} kpc'.format(int(len_arcsec/dscale))


    #ybase_offset = 0.02 #0.035 #0.065
    if 'left' in ruler_loc:
        x_base = xlim[0] + (xlim[1]-xlim[0])*delx
        sign_x = 1.
        ha = 'left'
    elif 'right' in ruler_loc:
        x_base = xlim[1] - (xlim[1]-xlim[0])*delx
        sign_x = -1.
        ha = 'right'
    if 'upper' in ruler_loc:
        y_base = ylim[1] - (ylim[1]-ylim[0])*(ybase_offset+dely)
        y_text = y_base - (ylim[1]-ylim[0])*(dely_text)
        va = 'center'
    elif 'lower' in ruler_loc:
        y_base = ylim[0] + (ylim[1]-ylim[0])*(ybase_offset+dely)
        y_text = y_base + (ylim[1]-ylim[0])*(dely_text)
        va = 'center'

    ax.plot([x_base+sign_x*len_line_angular, x_base], [y_base, y_base],
                c=color, ls='-',lw=2, solid_capstyle='butt')

    ax.annotate(string, xy=(x_base, y_text), xycoords="data", xytext=(0,0),
                color=color, textcoords="offset points", ha=ha, va=va, fontsize=9)

    return ax

def plot_contours_2D_multitype(im, ax=None, mapname='velocity', plottype='data',
            vmin=None, vmax=None, kwargs=None):
    # mapname: 'flux', 'velocity', 'dispersion'
    # plottype: 'data', 'model', 'gal1', 'gal2', etc -- and then 'residual' handled separately

    if mapname == 'velocity':
        delta_cont = kwargs['delta_cont_v']
        delta_cont_minor = kwargs['delta_cont_v_minor']
        delta_cont_minor_resid = kwargs['delta_cont_v_minor_resid']
    elif mapname == 'dispersion':
        delta_cont = kwargs['delta_cont_disp']
        delta_cont_minor = kwargs['delta_cont_disp_minor']
        delta_cont_minor_resid = kwargs['delta_cont_disp_minor_resid']
    elif mapname == 'flux':
        delta_cont = kwargs['delta_cont_flux']
        delta_cont_minor = kwargs['delta_cont_flux_minor']
        delta_cont_minor_resid = kwargs['delta_cont_flux_minor_resid']
    else:
        raise ValueError

    if vmin is None:
        vmin = im[np.isfinite(im)].min()
    if vmax is None:
        vmax = im[np.isfinite(im)].max()

    if (plottype != 'residual'):
        #####
        if delta_cont_minor is not None:
            lo_minor = vmin - (vmin%delta_cont_minor) +delta_cont_minor
            hi_minor = vmax -(vmax%delta_cont_minor)
            contour_levels_tmp2_minor = np.arange(lo_minor, hi_minor+delta_cont_minor,
                                delta_cont_minor)
            ax.contour(im,levels=contour_levels_tmp2_minor,
                        colors=kwargs['colors_cont_minor'],
                        alpha=kwargs['alpha_cont_minor'],
                        linestyles=kwargs['ls_cont_minor'],
                        linewidths=kwargs['lw_cont_minor'])
        #####
        if delta_cont is not None:
            lo = vmin - (vmin%delta_cont) +delta_cont
            hi = vmax -(vmax%delta_cont)
            contour_levels_tmp2 = np.arange(lo, hi+delta_cont, delta_cont)
            ax.contour(im,levels=contour_levels_tmp2,
                        colors=kwargs['colors_cont'], alpha=kwargs['alpha_cont'],
                        linestyles=kwargs['ls_cont'], linewidths=kwargs['lw_cont'])
    elif plottype == 'residual':
        # Check that the residual isn't all basically 0:
        if delta_cont_minor is not None:
            compval = np.min([delta_cont_minor / 10., 0.01])
        else:
            compval = 0.01

        if ((im[np.isfinite(im)].max() - im[np.isfinite(im)].min()) >= compval):
            #####
            if delta_cont_minor_resid is not None:
                # Minor minor contours:
                lo_mminor = vmin - (vmin%delta_cont_minor_resid) +delta_cont_minor_resid
                hi_mminor = vmax -(vmax%delta_cont_minor_resid)
                contour_levels_tmp2_mminor = np.arange(lo_mminor,
                                    hi_mminor+delta_cont_minor_resid,
                                    delta_cont_minor_resid)
                ax.contour(im,levels=contour_levels_tmp2_mminor,
                            colors=kwargs['colors_cont_minor_resid'],
                            alpha=kwargs['alpha_cont_minor_resid'],
                            linestyles=kwargs['ls_cont_minor_resid'],
                            linewidths=kwargs['lw_cont_minor_resid'])
            if delta_cont_minor is not None:
                lo_minor = vmin - (vmin%delta_cont_minor) +delta_cont_minor
                hi_minor = vmax -(vmax%delta_cont_minor)
                contour_levels_tmp2_minor = np.arange(lo_minor, hi_minor+delta_cont_minor,
                                    delta_cont_minor)
                ax.contour(im,levels=contour_levels_tmp2_minor,
                            colors=kwargs['colors_cont_minor'],
                            alpha=kwargs['alpha_cont_minor'],
                            linestyles=kwargs['ls_cont_minor'],
                            linewidths=kwargs['lw_cont_minor'])
            #####
            if delta_cont is not None:
                lo = vmin - (vmin%delta_cont) +delta_cont
                hi = vmax -(vmax%delta_cont)
                contour_levels_tmp2 = np.arange(lo, hi+delta_cont, delta_cont)
                ax.contour(im,levels=contour_levels_tmp2,
                            colors=kwargs['colors_cont'], alpha=kwargs['alpha_cont'],
                            linestyles=kwargs['ls_cont'], linewidths=kwargs['lw_cont'])

    return ax


############

# def _count_nObs_ndim(gal, ndim):
#     nObsnD = 0
#     for obs_name in gal.observations:
#         obs = gal.observations[obs_name]#
#         if obs.instrument.ndim == ndim:#
#             nObsnD += 1
#     return nObsnD
