# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Functions for plotting DYSMALPY kinematic model fit results


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging
import copy

import os

# Third party imports
import numpy as np
import six
import astropy.units as u
import matplotlib as mpl
mpl.use('agg')


import astropy.modeling as apy_mod

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, FixedFormatter

import matplotlib.colors as mplcolors
from mpl_toolkits.axes_grid1 import ImageGrid, AxesGrid

from matplotlib import colorbar

import corner
from spectral_cube import SpectralCube, BooleanArrayMask


# Package imports
from .utils import calc_pix_position, apply_smoothing_3D, gaus_fit_sp_opt_leastsq, gaus_fit_apy_mod_fitter
from .utils_io import create_vel_profile_files_obs1d, create_vel_profile_files_intrinsic, \
                      read_bestfit_1d_obs_file, read_model_intrinsic_profile
from .aperture_classes import CircApertures
from .data_classes import Data1D, Data2D
from .extern.altered_colormaps import new_diverging_cmap
from .config import Config_create_model_data, Config_simulate_cube

__all__ = ['plot_trace', 'plot_corner', 'plot_bestfit']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

def plot_bestfit(mcmcResults, gal,
                 fitdispersion=True,
                 fitflux=False,
                 show_1d_apers=False,
                 fileout=None,
                 fileout_aperture=None,
                 fileout_spaxel=None,
                 fileout_channel=None,
                 vcrop=False,
                 vcrop_value=800.,
                 remove_shift=False,
                 overwrite=False,
                 moment=False,
                 fill_mask=False,
                 **kwargs_galmodel):
    """
    Plot data, bestfit model, and residuals from the MCMC fitting.
    """
    plot_data_model_comparison(gal, theta = mcmcResults.bestfit_parameters,
            fitdispersion=fitdispersion, fitflux=fitflux, fileout=fileout,
            fileout_aperture=fileout_aperture, fileout_spaxel=fileout_spaxel,
            fileout_channel=fileout_channel,
            vcrop=vcrop, vcrop_value=vcrop_value, show_1d_apers=show_1d_apers,
            remove_shift=remove_shift, moment=moment, fill_mask=fill_mask,
            overwrite=overwrite, **kwargs_galmodel)

    return None


def plot_trace(mcmcResults, fileout=None, overwrite=False):
    """
    Plot trace of MCMC walkers
    """
    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    names = make_clean_mcmc_plot_names(mcmcResults)

    ######################################
    # Setup plot:
    f = plt.figure()
    scale = 1.75
    nRows = len(names)
    nWalkers = mcmcResults.sampler['nWalkers']

    f.set_size_inches(4.*scale, nRows*scale)
    gs = gridspec.GridSpec(nRows, 1, hspace=0.2)

    axes = []
    alpha = max(0.01, min(10./nWalkers, 1.))

    # Define random color inds for tracking some walkers:
    nTraceWalkers = 5
    cmap = cm.viridis
    alphaTrace = 0.8
    lwTrace = 1.5
    trace_inds = np.random.randint(0,nWalkers, size=nTraceWalkers)
    trace_colors = []
    for i in six.moves.xrange(nTraceWalkers):
        trace_colors.append(cmap(1./np.float(nTraceWalkers)*i))

    norm_inds = np.setdiff1d(range(nWalkers), trace_inds)


    for k in six.moves.xrange(nRows):
        axes.append(plt.subplot(gs[k,0]))

        axes[k].plot(mcmcResults.sampler['chain'][norm_inds,:,k].T, '-', color='black', alpha=alpha)

        for j in six.moves.xrange(nTraceWalkers):
            axes[k].plot(mcmcResults.sampler['chain'][trace_inds[j],:,k].T, '-',
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


def plot_corner(mcmcResults, gal=None, fileout=None, step_slice=None, blob_name=None, overwrite=False):
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

    names = make_clean_mcmc_plot_names(mcmcResults)

    if step_slice is None:
        sampler_chain = mcmcResults.sampler['flatchain']
    else:
        sampler_chain = mcmcResults.sampler['chain'][:,step_slice[0]:step_slice[1],:].reshape((-1, mcmcResults.sampler['nParam']))

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
                    blobs = mcmcResults.sampler['flatblobs']
                else:
                    blobs = mcmcResults.sampler['blobs'][step_slice[0]:step_slice[1],:,:].reshape((-1, 1))
            else:
                indv = blob_names.index(blobn)
                if step_slice is None:
                    blobs = mcmcResults.sampler['flatblobs'][:,indv]
                else:
                    blobs = mcmcResults.sampler['blobs'][step_slice[0]:step_slice[1],:,:].reshape((-1, mcmcResults.sampler['blobs'].shape[2]))[:,indv]

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
                            quantiles= [.02275, 0.15865, 0.84135, .97725],
                            truths=truths,
                            plot_datapoints=False,
                            show_titles=True,
                            bins=40,
                            plot_contours=True,
                            verbose=False,
                            title_kwargs=title_kwargs)

    axes = fig.axes
    nFreeParam = len(truths)
    for i in six.moves.xrange(nFreeParam):
        ax = axes[i*nFreeParam + i]
        # Format the quantile display.
        best = truths[i] #fit_results.bestfit_parameters[i]
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
        for i in six.moves.xrange(nFreeParam):
            for j in six.moves.xrange(nFreeParam):
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
        plt.savefig(fileout, bbox_inches='tight')#, dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return None




def plot_data_model_comparison(gal,theta = None,
                               fitdispersion=True,
                               fitflux=False,
                               fileout=None,
                               fileout_aperture=None,
                               fileout_spaxel=None,
                               fileout_channel=None,
                               show_multid=True,
                               show_apertures=True,
                               show_all_spax=True,
                               show_channel=True,
                               vcrop=False,
                               show_1d_apers=False,
                               vcrop_value=800.,
                               remove_shift=False,
                               overwrite=False,
                               moment=False,
                               fill_mask=False,
                               **kwargs_galmodel):
    """
    Plot data, model, and residuals between the data and this model.
    """

    dummy_gal = copy.deepcopy(gal)

    if remove_shift:
        dummy_gal.data.aper_center_pix_shift = (0,0)
        dummy_gal.model.geometry.xshift = 0
        dummy_gal.model.geometry.yshift = 0

    # if (theta is not None) & (gal.data.ndim < 3):
    #     dummy_gal.model.update_parameters(theta)     # Update the parameters
    #     dummy_gal.create_model_data(**kwargs_galmodel)

    if (theta is not None) & (gal.data.ndim == 3):
        dummy_gal.model.update_parameters(theta)     # Update the parameters
        dummy_gal.create_model_data(**kwargs_galmodel)


    if gal.data.ndim == 1:
        plot_data_model_comparison_1D(dummy_gal,
                    data = None,
                    theta = theta,
                    fitdispersion=fitdispersion,
                    fitflux=fitflux,
                    fileout=fileout,
                    overwrite=overwrite,
                    **kwargs_galmodel)
    elif gal.data.ndim == 2:
        plot_data_model_comparison_2D(dummy_gal,
                    theta = theta,
                    fitdispersion=fitdispersion,
                    fitflux=fitflux,
                    fileout=fileout,
                    overwrite=overwrite,
                    **kwargs_galmodel)
    elif gal.data.ndim == 3:
        plot_data_model_comparison_3D(dummy_gal,
                    theta = theta,
                    show_1d_apers=show_1d_apers,
                    fileout=fileout,
                    fileout_aperture=fileout_aperture,
                    fileout_spaxel=fileout_spaxel,
                    fileout_channel=fileout_channel,
                    vcrop=vcrop,
                    vcrop_value=vcrop_value,
                    overwrite=overwrite,
                    moment=moment,
                    show_multid=show_multid,
                    show_apertures=show_apertures,
                    show_all_spax=show_all_spax,
                    show_channel=show_channel,
                    fill_mask=fill_mask,
                    **kwargs_galmodel)

    elif gal.data.ndim == 0:
        plot_data_model_comparison_0D(dummy_gal,
                                      fileout=fileout,
                                      overwrite=overwrite,
                                      **kwargs_galmodel)
    else:
        logger.warning("nDim="+str(gal.data.ndim)+" not supported!")
        raise ValueError("nDim="+str(gal.data.ndim)+" not supported!")

    return None


def plot_data_model_comparison_0D(gal, data = None, fileout=None,
        overwrite=False, **kwargs_galmodel):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    ######################################
    # Setup data/model comparison: if this isn't the fit dimension
    #   data/model comparison (eg, fit in 2D, showing 1D comparison)
    if data is not None:

        # Setup the model with the correct dimensionality:
        galnew = copy.deepcopy(gal)
        galnew.data = data
        galnew.create_model_data(**kwargs_galmodel)
        model_data = galnew.model_data

    else:
        # Default: fit in 1D, compare to 1D data:
        data = gal.data
        model_data = gal.model_data


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
    ax.set_xlabel(gal.instrument.spec_type.capitalize() + ' [' + gal.instrument.spec_step.unit.name + ']')

    # Save to file:
    if fileout is not None:
        f.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close(f)
    else:
        plt.show()


def plot_data_model_comparison_1D(gal,
            data = None,
            theta = None,
            fitdispersion=True,
            fitflux=False,
            fileout=None, overwrite=False,
            **kwargs_galmodel):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    ######################################
    # Setup data/model comparison: if this isn't the fit dimension
    #   data/model comparison (eg, fit in 2D, showing 1D comparison)
    if data is not None:

        # Setup the model with the correct dimensionality:
        galnew = copy.deepcopy(gal)
        galnew.data = data
        galnew.create_model_data(**kwargs_galmodel)
        model_data = galnew.model_data

    else:
        # Default: fit in 1D, compare to 1D data:
        data = gal.data
        model_data = gal.model_data

    # Correct model for instrument dispersion if the data is instrument corrected:

    if 'inst_corr' in data.data.keys():
        inst_corr = data.data['inst_corr']


    if inst_corr:
            model_data.data['dispersion'] = \
                np.sqrt( model_data.data['dispersion']**2 - \
                    gal.instrument.lsf.dispersion.to(u.km/u.s).value**2 )


    ######################################
    # Setup plot:
    f = plt.figure()
    scale = 3.5
    ncols = 1
    if fitdispersion:
        ncols += 1
    if fitflux:
        ncols += 1
    nrows = 2
    f.set_size_inches(1.1*ncols*scale, nrows*scale)
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.35, hspace=0.2)

    keyxtitle = r'$r$ [arcsec]'
    keyyarr = ['velocity', 'dispersion', 'flux']
    keyytitlearr = [r'$V$ [km/s]', r'$\sigma$ [km/s]', r'Flux [arb]']
    keyyresidtitlearr = [r'$V_{\mathrm{data}} - V_{\mathrm{model}}$ [km/s]',
                    r'$\sigma_{\mathrm{data}} - \sigma_{\mathrm{model}}$ [km/s]',
                    r'$\mathrm{Flux_{data} - Flux_{model}}$ [arb]']

    errbar_lw = 0.5
    errbar_cap = 1.5

    axes = []
    k = -1
    for j in six.moves.xrange(ncols):
        # Comparison:
        axes.append(plt.subplot(gs[0,j]))
        k += 1

        # msk = data.data.mask
        if keyyarr[j] == 'velocity':
            if hasattr(gal.data, 'mask_velocity'):
                if gal.data.mask_velocity is not None:
                    msk = gal.data.mask_velocity
                else:
                    msk = gal.data.mask
            else:
                msk = gal.data.mask
        elif keyyarr[j] == 'dispersion':
            if hasattr(gal.data, 'mask_vel_disp'):
                if gal.data.mask_vel_disp is not None:
                    msk = gal.data.mask_vel_disp
                else:
                    msk = gal.data.mask
            else:
                msk = gal.data.mask
        elif keyyarr[j] == 'flux':
            msk = gal.data.mask
        else:
            msk = np.array(np.ones(len(data.rarr)), dtype=np.bool)

        # Masked points
        axes[k].errorbar( data.rarr[~msk], data.data[keyyarr[j]][~msk],
                xerr=None, yerr = data.error[keyyarr[j]][~msk],
                marker=None, ls='None', ecolor='darkgrey', zorder=-1.,
                lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None, alpha=0.5 )
        if np.any(~msk):
            lbl_mask = 'Masked data'
        else:
            lbl_mask = None
        axes[k].scatter( data.rarr[~msk], data.data[keyyarr[j]][~msk],
            c='darkgrey', marker='o', s=25, lw=1, label=lbl_mask, alpha=0.5)

        # Unmasked points
        axes[k].errorbar( data.rarr[msk], data.data[keyyarr[j]][msk],
                xerr=None, yerr = data.error[keyyarr[j]][msk],
                marker=None, ls='None', ecolor='k', zorder=-1.,
                lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )

        # Weights: effective errorbars show in blue, for reference
        if hasattr(data, 'weight'):
            if gal.data.weight is not None:
                wgt = gal.data.weight
                axes[k].errorbar( data.rarr[msk], data.data[keyyarr[j]][msk],
                        xerr=None, yerr = data.error[keyyarr[j]][msk]/np.sqrt(wgt[msk]),
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
        axes[k].errorbar( data.rarr[~msk], data.data[keyyarr[j]][~msk]-model_data.data[keyyarr[j]][~msk],
                xerr=None, yerr = data.error[keyyarr[j]][~msk],
                marker=None, ls='None', ecolor='darkgrey', zorder=-1.,
                lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None, alpha=0.5 )
        axes[k].scatter( data.rarr[~msk], data.data[keyyarr[j]][~msk]-model_data.data[keyyarr[j]][~msk],
            c='lightsalmon', marker='s', s=25, lw=1, label=None, alpha=0.3)

        # Unmasked points:
        axes[k].errorbar( data.rarr[msk], data.data[keyyarr[j]][msk]-model_data.data[keyyarr[j]][msk],
                xerr=None, yerr = data.error[keyyarr[j]][msk],
                marker=None, ls='None', ecolor='k', zorder=-1.,
                lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )

        # Weights: effective errorbars show in blue, for reference
        if hasattr(data, 'weight'):
            if gal.data.weight is not None:
                wgt = gal.data.weight
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
    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_data_model_comparison_2D(gal,
            theta = None,
            fitdispersion=True,
            fitflux=False,
            fileout=None,
            symmetric_residuals=True,
            max_residual=100.,
            max_residual_flux=None,
            overwrite=False,
            **kwargs_galmodel):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    try:
        if 'inst_corr' in gal.data.data.keys():
            inst_corr = gal.data.data['inst_corr']
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
    add_all=True,
    label_mode="1"
    share_all=True
    cbar_location="right"
    cbar_mode="each"
    cbar_size="5%"
    cbar_pad="1%"
    if fitdispersion:
        if fitflux:
            pos_gridvel = 311
            pos_griddisp = 312
            pos_gridflux = 313
        else:
            pos_gridvel = 211
            pos_griddisp = 212
    else:
        if fitflux:
            pos_gridvel = 211
            pos_gridflux = 212
        else:
            pos_gridvel = 111


    grid_vel = ImageGrid(f, pos_gridvel,
                         nrows_ncols=nrows_ncols,
                         direction=direction,
                         axes_pad=axes_pad,
                         add_all=add_all,
                         label_mode=label_mode,
                         share_all=share_all,
                         cbar_location=cbar_location,
                         cbar_mode=cbar_mode,
                         cbar_size=cbar_size,
                         cbar_pad=cbar_pad)
    if fitdispersion:
        grid_disp = ImageGrid(f, pos_griddisp,
                             nrows_ncols=nrows_ncols,
                             direction=direction,
                             axes_pad=axes_pad,
                             add_all=add_all,
                             label_mode=label_mode,
                             share_all=share_all,
                             cbar_location=cbar_location,
                             cbar_mode=cbar_mode,
                             cbar_size=cbar_size,
                             cbar_pad=cbar_pad)

    if fitflux:
        grid_flux = ImageGrid(f, pos_gridflux,
                             nrows_ncols=nrows_ncols,
                             direction=direction,
                             axes_pad=axes_pad,
                             add_all=add_all,
                             label_mode=label_mode,
                             share_all=share_all,
                             cbar_location=cbar_location,
                             cbar_mode=cbar_mode,
                             cbar_size=cbar_size,
                             cbar_pad=cbar_pad)
    #
    keyxarr = ['data', 'model', 'residual']
    keyyarr = ['velocity', 'dispersion', 'flux']
    keyxtitlearr = ['Data', 'Model', 'Residual']
    keyytitlearr = [r'$V$', r'$\sigma$', r'Flux']


    int_mode = "nearest"
    origin = 'lower'
    cmap =  cm.Spectral_r #cm.nipy_spectral
    cmap.set_bad(color='k')


    gamma = 1.5
    cmap_resid = new_diverging_cmap('RdBu_r', diverge = 0.5,
                gamma_lower=gamma, gamma_upper=gamma,
                name_new='RdBu_r_stretch')


    vel_vmin = gal.data.data['velocity'][gal.data.mask].min()
    vel_vmax = gal.data.data['velocity'][gal.data.mask].max()

    try:
        vel_shift = gal.model.geometry.vel_shift.value
    except:
        vel_shift = 0

    #
    vel_vmin -= vel_shift
    vel_vmax -= vel_shift

    for ax, k, xt in zip(grid_vel, keyxarr, keyxtitlearr):
        if k == 'data':
            im = gal.data.data['velocity'].copy()
            im -= vel_shift
            im[~gal.data.mask] = np.nan
            cmaptmp = cmap
        elif k == 'model':
            im = gal.model_data.data['velocity'].copy()
            im -= vel_shift
            im[~gal.data.mask] = np.nan
            cmaptmp = cmap
        elif k == 'residual':
            im = gal.data.data['velocity'] - gal.model_data.data['velocity']
            im[~gal.data.mask] = np.nan
            if symmetric_residuals:
                vel_vmin = -max_residual
                vel_vmax = max_residual
            cmaptmp = cmap_resid
        else:
            raise ValueError("key not supported.")

        imax = ax.imshow(im, cmap=cmaptmp, interpolation=int_mode,
                         vmin=vel_vmin, vmax=vel_vmax, origin=origin)



        ax = plot_major_minor_axes_2D(ax, gal, im, gal.data.mask)

        if k == 'data':
            ax.set_ylabel(keyytitlearr[0])
            for pos in ['top', 'bottom', 'left', 'right']:
                ax.spines[pos].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # #ax.set_axis_off()
            for pos in ['top', 'bottom', 'left', 'right']:
                ax.spines[pos].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        #
        if k == 'residual':
            med = np.median(im[gal.data.mask])
            rms = np.std(im[gal.data.mask])
            median_str = r"$V_{med}="+r"{:0.1f}".format(med)+r"$"
            scatter_str = r"$V_{rms}="+r"{:0.1f}".format(rms)+r"$"
            ax.annotate(median_str,
                (0.01,-0.05), xycoords='axes fraction',
                ha='left', va='top', fontsize=8)
            ax.annotate(scatter_str,
                (0.99,-0.05), xycoords='axes fraction',
                ha='right', va='top', fontsize=8)

        ax.set_title(xt)

        cbar = ax.cax.colorbar(imax)
        cbar.ax.tick_params(labelsize=8)

    #####
    if fitdispersion:

        if gal.data is not None:
            disp_vmin = gal.data.data['dispersion'][gal.data.mask].min()
            disp_vmax = gal.data.data['dispersion'][gal.data.mask].max()
        else:
            disp_vmin = gal.model_data.data['dispersion'].min()
            disp_vmax = gal.model_data.data['dispersion'].max()

        for ax, k in zip(grid_disp, keyxarr):
            if k == 'data':
                im = gal.data.data['dispersion'].copy()
                im[~gal.data.mask] = np.nan
                cmaptmp = cmap
            elif k == 'model':
                im = gal.model_data.data['dispersion'].copy()

                im[~gal.data.mask] = np.nan

                # Correct model for instrument dispersion
                # if the data is instrument corrected:
                if inst_corr:
                    im = np.sqrt(im ** 2 - gal.instrument.lsf.dispersion.to(
                                 u.km / u.s).value ** 2)
                cmaptmp = cmap
            elif k == 'residual':

                im_model = gal.model_data.data['dispersion'].copy()
                if inst_corr:
                    im_model = np.sqrt(im_model ** 2 -
                                       gal.instrument.lsf.dispersion.to( u.km / u.s).value ** 2)


                im = gal.data.data['dispersion'] - im_model
                im[~gal.data.mask] = np.nan

                if symmetric_residuals:
                    disp_vmin = -max_residual
                    disp_vmax = max_residual
                cmaptmp = cmap_resid
            else:
                raise ValueError("key not supported.")

            imax = ax.imshow(im, cmap=cmaptmp, interpolation=int_mode,
                             vmin=disp_vmin, vmax=disp_vmax, origin=origin)


            ax = plot_major_minor_axes_2D(ax, gal, im, gal.data.mask)

            if k == 'data':
                ax.set_ylabel(keyytitlearr[1])
                for pos in ['top', 'bottom', 'left', 'right']:
                    ax.spines[pos].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # #ax.set_axis_off()
                for pos in ['top', 'bottom', 'left', 'right']:
                    ax.spines[pos].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])

            #
            if k == 'residual':
                med = np.median(im[gal.data.mask])
                rms = np.std(im[gal.data.mask])
                median_str = r"$\sigma_{med}="+r"{:0.1f}".format(med)+r"$"
                scatter_str = r"$\sigma_{rms}="+r"{:0.1f}".format(rms)+r"$"
                ax.annotate(median_str,
                    (0.01,-0.05), xycoords='axes fraction',
                    ha='left', va='top', fontsize=8)
                ax.annotate(scatter_str,
                    (0.99,-0.05), xycoords='axes fraction',
                    ha='right', va='top', fontsize=8)

            cbar = ax.cax.colorbar(imax)
            cbar.ax.tick_params(labelsize=8)

    ######
    if fitflux:
        if gal.data is not None:
            flux_vmin = gal.data.data['flux'][gal.data.mask].min()
            flux_vmax = gal.data.data['flux'][gal.data.mask].max()
            if flux_vmin == flux_vmax:
                flux_vmin = gal.model_data.data['flux'].min()
                flux_vmax = gal.model_data.data['flux'].max()
        else:
            flux_vmin = gal.model_data.data['flux'].min()
            flux_vmax = gal.model_data.data['flux'].max()


        for ax, k in zip(grid_flux, keyxarr):
            if k == 'data':
                im = gal.data.data['flux'].copy()
                im[~gal.data.mask] = np.nan
                cmaptmp = cmap
            elif k == 'model':
                im = gal.model_data.data['flux'].copy()

                im[~gal.data.mask] = np.nan

                cmaptmp = cmap
            elif k == 'residual':
                im = gal.data.data['flux'].copy() - gal.model_data.data['flux'].copy()
                im[~gal.data.mask] = np.nan

                if symmetric_residuals:
                    if max_residual_flux is None:
                        max_residual_flux = np.max(np.abs(im[gal.data.mask]))
                    flux_vmin = -max_residual_flux
                    flux_vmax = max_residual_flux
                cmaptmp = cmap_resid
            else:
                raise ValueError("key not supported.")

            imax = ax.imshow(im, cmap=cmaptmp, interpolation=int_mode,
                             vmin=flux_vmin, vmax=flux_vmax, origin=origin)

            ax = plot_major_minor_axes_2D(ax, gal, im, gal.data.mask)

            if k == 'data':
                ax.set_ylabel(keyytitlearr[2])
                for pos in ['top', 'bottom', 'left', 'right']:
                    ax.spines[pos].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # #ax.set_axis_off()
                for pos in ['top', 'bottom', 'left', 'right']:
                    ax.spines[pos].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])

            ####
            if k == 'residual':
                med = np.median(im[gal.data.mask])
                rms = np.std(im[gal.data.mask])
                median_str = r"$f_{med}="+r"{:0.1f}".format(med)+r"$"
                scatter_str = r"$f{rms}="+r"{:0.1f}".format(rms)+r"$"
                ax.annotate(median_str,
                    (0.01,-0.05), xycoords='axes fraction',
                    ha='left', va='top', fontsize=8)
                ax.annotate(scatter_str,
                    (0.99,-0.05), xycoords='axes fraction',
                    ha='right', va='top', fontsize=8)

            cbar = ax.cax.colorbar(imax)
            cbar.ax.tick_params(labelsize=8)

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()


def plot_data_model_comparison_3D(gal,
            theta = None,
            fileout=None,
            fileout_aperture=None,
            fileout_spaxel=None,
            fileout_channel=None,
            symmetric_residuals=True,
            show_1d_apers = False,
            max_residual=100.,
            inst_corr = True,
            vcrop=False,
            vcrop_value=800.,
            overwrite=False,
            moment=False,
            show_multid=True,
            show_apertures=True,
            show_all_spax=True,
            show_channel=True,
            remove_shift=False,
            fill_mask=False,
            **kwargs_galmodel):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    if show_multid:
        plot_model_multid(gal, theta=theta,
                    fileout=fileout,
                    symmetric_residuals=symmetric_residuals, max_residual=max_residual,
                    show_1d_apers=show_1d_apers,
                    fitdispersion=True, fitflux=True,
                    inst_corr=True,
                    vcrop=vcrop,
                    vcrop_value=vcrop_value,
                    overwrite=overwrite,
                    moment=moment,
                    remove_shift=False,   # TRY THIS
                    fill_mask=fill_mask,
                    **kwargs_galmodel)

    if show_all_spax:
        #if fileout_spaxel is not None:
        plot_spaxel_compare_3D_cubes(gal, fileout=fileout_spaxel,
                        typ='all', show_model=True, overwrite=overwrite)

    if show_apertures:
        #if fileout_aperture is not None:
        plot_aperture_compare_3D_cubes(gal, fileout=fileout_aperture,
                            fill_mask=fill_mask, overwrite=overwrite)

    if show_channel:
        #if fileout_channel is not None:
        plot_channel_maps_3D_cube(gal,
                    show_data=True, show_model=True,show_residual=True,
                    vbounds = [-450., 450.], delv = 100.,
                    vbounds_shift=True, cmap=cm.Greys,
                    fileout=fileout_channel, overwrite=overwrite)



    return None


#############################################################
#############################################################

def plot_model_multid(gal, theta=None, fitdispersion=True, fitflux=False,
            fileout=None,
            symmetric_residuals=True, max_residual=100.,
            show_1d_apers=False, inst_corr=None,
            xshift = None,
            yshift = None,
            vcrop=False,
            vcrop_value=800.,
            remove_shift=True,
            overwrite=False,
            moment=False,
            fill_mask=False,
            **kwargs_galmodel):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    if gal.data.ndim == 1:
        plot_model_multid_base(gal, data1d=gal.data, data2d=gal.data2d,
                    theta=theta,fitdispersion=fitdispersion, fitflux=fitflux,
                    symmetric_residuals=symmetric_residuals, max_residual=max_residual,
                    fileout=fileout,
                    xshift = xshift,
                    yshift = yshift,
                    show_1d_apers=show_1d_apers,
                    remove_shift=True,
                    overwrite=overwrite, **kwargs_galmodel)
    elif gal.data.ndim == 2:
        plot_model_multid_base(gal, data1d=gal.data1d, data2d=gal.data,
                    theta=theta,fitdispersion=fitdispersion, fitflux=fitflux,
                    symmetric_residuals=symmetric_residuals,  max_residual=max_residual,
                    fileout=fileout,
                    show_1d_apers=show_1d_apers,
                    remove_shift=remove_shift,
                    overwrite=overwrite, **kwargs_galmodel)

    elif gal.data.ndim == 3:
        print("plot_model_multid: ndim=3: moment={}".format(moment))
        if moment:
            gal = extract_1D_2D_data_moments_from_cube(gal, inst_corr=True, fill_mask=fill_mask)
        else:
            gal = extract_1D_2D_data_gausfit_from_cube(gal, inst_corr=True, fill_mask=fill_mask)
        # saves in gal.data1d, gal.data2d

        # Data haven't actually been corrected for instrument LSF yet
        if gal.data1d.data['inst_corr']:
            inst_corr_sigma = gal.instrument.lsf.dispersion.to(u.km/u.s).value
            disp_prof_1D = np.sqrt(gal.data1d.data['dispersion']**2 - inst_corr_sigma**2 )
            disp_prof_1D[~np.isfinite(disp_prof_1D)] = 0.
            gal.data1d.data['dispersion'] = disp_prof_1D

        if gal.data2d.data['inst_corr']:
            inst_corr_sigma = gal.instrument.lsf.dispersion.to(u.km/u.s).value
            im = gal.data2d.data['dispersion'].copy()
            im = np.sqrt(im ** 2 - inst_corr_sigma ** 2)
            im[~np.isfinite(im)] = 0.
            gal.data2d.data['dispersion'] = im

        if 'filled_mask_data' in gal.data1d.__dict__.keys():
            if gal.data1d.filled_mask_data.data['inst_corr']:
                inst_corr_sigma = gal.instrument.lsf.dispersion.to(u.km/u.s).value
                disp_prof_1D = np.sqrt(gal.data1d.filled_mask_data.data['dispersion']**2 - inst_corr_sigma**2 )
                disp_prof_1D[~np.isfinite(disp_prof_1D)] = 0.
                gal.data1d.filled_mask_data.data['dispersion'] = disp_prof_1D


        plot_model_multid_base(gal, data1d=gal.data1d, data2d=gal.data2d,
                    theta=theta,fitdispersion=fitdispersion, fitflux=fitflux,
                    symmetric_residuals=symmetric_residuals,  max_residual=max_residual,
                    fileout=fileout,
                    show_1d_apers=show_1d_apers, inst_corr=inst_corr,
                    vcrop=vcrop, vcrop_value=vcrop_value,
                    remove_shift=remove_shift, moment=moment, fill_mask=fill_mask,
                    overwrite=overwrite, **kwargs_galmodel)


    return None


#############################################################

def plot_model_multid_base(gal,
            data1d=None, data2d=None,
            theta=None,
            fitdispersion=True, fitflux=False,
            symmetric_residuals=True,
            max_residual=100.,
            xshift = None,
            yshift = None,
            fileout=None,
            vcrop = False,
            vcrop_value = 800.,
            show_1d_apers=False,
            remove_shift = True,
            inst_corr=None,
            moment=True,
            fill_mask=False,
            overwrite=False,
            **kwargs_galmodel):

    #
    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    ######################################
    # Setup plot:

    nrows = 1
    if fitdispersion:
        nrows += 1
    if fitflux:
        nrows += 1


    ncols = 5

    padx = 0.25
    pady = 0.25

    xextra = 0.15 #0.2
    yextra = 0. #0.75

    scale = 2.5 #3.5

    f = plt.figure()
    f.set_size_inches((ncols+(ncols-1)*padx+xextra)*scale, (nrows+pady+yextra)*scale)


    suptitle = '{}: Fitting dim: n={}'.format(gal.name, gal.data.ndim)


    padx = 0.1
    gs_outer = gridspec.GridSpec(1, 2, wspace=padx, width_ratios=[3.35, 4.75])


    padx = 0.35
    gs01 = gridspec.GridSpecFromSubplotSpec(nrows, 2, subplot_spec=gs_outer[0], wspace=padx)
    indr = 0
    grid_1D = [plt.subplot(gs01[indr,0]), plt.subplot(gs01[indr,1])]
    if fitdispersion:
        indr += 1
        grid_1D.append(plt.subplot(gs01[indr,0]))
        grid_1D.append(plt.subplot(gs01[indr,1]))
    if fitflux:
        indr += 1
        grid_1D.append(plt.subplot(gs01[indr,0]))
        grid_1D.append(plt.subplot(gs01[indr,1]))


    padx = 0.2 #0.4
    pady = 0.1 #0.35
    gs02 = gridspec.GridSpecFromSubplotSpec(nrows, 3, subplot_spec=gs_outer[1],
            wspace=padx, hspace=pady)
    grid_2D = []
    # grid_2D_cax = []
    for jj in six.moves.xrange(nrows):
        for mm in six.moves.xrange(3):
            grid_2D.append(plt.subplot(gs02[jj,mm]))


    gal_input = copy.deepcopy(gal)
    if theta is not None:
        gal.model.update_parameters(theta)     # Update the parameters
        #gal.create_model_data(**kwargs_galmodel)

    #raise ValueError
    #
    inst_corr_1d = inst_corr_2d = None
    if inst_corr is None:
        if 'inst_corr' in data1d.data.keys():
            inst_corr_1d = data1d.data['inst_corr']
        if 'inst_corr' in data2d.data.keys():
            inst_corr_2d = data2d.data['inst_corr']
    else:
        inst_corr_1d = inst_corr
        inst_corr_2d = inst_corr

    if (inst_corr_1d):
        inst_corr_sigma = gal.instrument.lsf.dispersion.to(u.km/u.s).value
    else:
        inst_corr_sigma = 0.

    galorig = copy.deepcopy(gal)
    instorig = copy.deepcopy(gal.instrument)
    try:
        instorig2d = copy.deepcopy(gal.instrument2d)
    except:
        instorig2d = copy.deepcopy(gal.instrument)

    #
    try:
        instorig1d = copy.deepcopy(gal.instrument1d)
    except:
        instorig1d = copy.deepcopy(gal.instrument)

    # In case missing / set to None:
    if instorig2d is None:
        instorig2d = copy.deepcopy(gal.instrument)
    if instorig1d is None:
        instorig1d = copy.deepcopy(gal.instrument)
    # ----------------------------------------------------------------------
    # 2D plotting

    if data2d is None:
        for ax in grid_2D:
            ax.set_axis_off()

    else:
        gal = copy.deepcopy(galorig)
        if gal.data.ndim == 1:
            apply_shift = True
        else:
            apply_shift = False

        # if theta is not None:
        #     gal.model.update_parameters(theta)

        if apply_shift:
            if xshift is not None:
                gal.model.geometry.xshift = xshift
            if yshift is not None:
                gal.model.geometry.yshift = yshift



        gal.instrument = copy.deepcopy(instorig2d)
        pixscale = gal.instrument.pixscale.value
        kwargs_galmodel_2d = kwargs_galmodel.copy()
        kwargs_galmodel_2d['from_data'] = True
        if galorig.data.ndim <= 2:
            kwargs_galmodel_2d['ndim_final'] = 2
            gal.data = copy.deepcopy(data2d)
        elif galorig.data.ndim == 3:
            kwargs_galmodel_2d['ndim_final'] = 3
        gal.model_data = None
        gal.create_model_data(**kwargs_galmodel_2d)
        # gal.model.update_parameters(theta)
        # gal.create_model_data(**kwargs_galmodel)

        if galorig.data.ndim == 3:
            #EXTRACT 2D MAPS HERE! (don't pre-do the dispersion correction)
            #SET EQUAL TO MODEL_DATA!
            gal.model_data_3D = copy.deepcopy(gal.model_data)
            if moment:
                gal.model_data = extract_2D_moments_from_cube(gal.model_data_3D.data, gal, inst_corr=inst_corr_2d)
            else:
                gal.model_data = extract_2D_gausfit_from_cube(gal.model_data_3D.data, gal, inst_corr=inst_corr_2d)

            gal.data = copy.deepcopy(data2d)


            #raise ValueError

        keyxarr = ['data', 'model', 'residual']
        keyxtitlearr = ['Data', 'Model', 'Residual']
        # keyyarr = ['velocity', 'dispersion', 'flux']
        # keyytitlearr = [r'$V$', r'$\sigma$', 'Flux']
        keyyarr = ['velocity']
        keyytitlearr = [r'$V$']
        if fitdispersion:
            keyyarr.append('dispersion')
            keyytitlearr.append(r'$\sigma$')
        if fitflux:
            keyyarr.append('flux')
            keyytitlearr.append('Flux')

        int_mode = "nearest"
        origin = 'lower'
        cmap =  cm.Spectral_r
        cmap.set_bad(color='k')

        gamma = 1.5
        cmap_resid = new_diverging_cmap('RdBu_r', diverge = 0.5,
                    gamma_lower=gamma, gamma_upper=gamma,
                    name_new='RdBu_r_stretch')

        cmap.set_bad(color='k')

        color_annotate = 'white'

        # -----------------------
        vel_vmin = gal.data.data['velocity'][gal.data.mask].min()
        vel_vmax = gal.data.data['velocity'][gal.data.mask].max()

        # Check for not too crazy:
        if vcrop:
            if vel_vmin < -vcrop_value:
                vel_vmin = -vcrop_value
            if vel_vmax > vcrop_value:
                vel_vmax = vcrop_value

        vel_shift = gal.model.geometry.vel_shift.value

        vel_vmin -= vel_shift
        vel_vmax -= vel_shift

        # ++++++++++++++
        if fitdispersion:
            if (gal.data.data['dispersion'] is not None):
                disp_vmin = gal.data.data['dispersion'][gal.data.mask].min()
                disp_vmax = gal.data.data['dispersion'][gal.data.mask].max()

                # Check for not too crazy:
                if vcrop:
                    if disp_vmin < 0:
                        disp_vmin = 0
                    if disp_vmax > vcrop_value:
                        disp_vmax = vcrop_value
            else:
                disp_vmin = None
                disp_vmax = None
        else:
            disp_vmin = None
            disp_vmax = None

        # ++++++++++++++
        if fitflux:
            if (gal.data.data['flux'] is not None):
                flux_vmin = gal.data.data['flux'][gal.data.mask].min()
                flux_vmax = gal.data.data['flux'][gal.data.mask].max()
            else:
                flux_vmin = None
                flux_vmax = None
        else:
            flux_vmin = None
            flux_vmax = None

        # ++++++++++++++
        alpha_unmasked = 1. #0.7 #0.6
        alpha_masked = 0.5   # 0.
        alpha_bkgd = 1. #0.5 #1. #0.5
        alpha_aper = 0.8

        vmin_2d = []
        vmax_2d = []
        vmin_2d_resid = []
        vmax_2d_resid = []

        #for ax, k, xt in zip(grid_2D, keyyarr, keyytitlearr):
        for j in six.moves.xrange(len(keyyarr)):
            for mm in six.moves.xrange(len(keyxarr)):
                kk = j*len(keyxarr) + mm

                k = keyxarr[mm]

                ax = grid_2D[kk]

                xt = keyxtitlearr[mm]
                yt = keyytitlearr[j]

                # -----------------------------------

                if k == 'data':
                    if keyyarr[j] == 'velocity':
                        im = gal.data.data['velocity'].copy()
                        im -= vel_shift
                        #im[~gal.data.mask] = np.nan
                        vmin = vel_vmin
                        vmax = vel_vmax
                    elif keyyarr[j] == 'dispersion':
                        im = gal.data.data['dispersion'].copy()
                        #im[~gal.data.mask] = np.nan
                        vmin = disp_vmin
                        vmax = disp_vmax
                    elif keyyarr[j] == 'flux':
                        im = gal.data.data['flux'].copy()
                        #im[~gal.data.mask] = np.nan
                        vmin = flux_vmin
                        vmax = flux_vmax

                    cmaptmp = cmap
                    vmin_2d.append(vmin)
                    vmax_2d.append(vmax)
                elif k == 'model':
                    if keyyarr[j] == 'velocity':
                        im = gal.model_data.data['velocity'].copy()
                        im -= vel_shift
                        vmin = vel_vmin
                        vmax = vel_vmax
                        #im[~gal.data.mask] = np.nan
                    elif keyyarr[j] == 'dispersion':
                        im = gal.model_data.data['dispersion'].copy()
                        #im[~gal.data.mask] = np.nan
                        vmin = disp_vmin
                        vmax = disp_vmax

                        # Correct model for instrument dispersion
                        # if the data is instrument corrected:
                        if inst_corr_2d:
                            im = np.sqrt(im ** 2 - inst_corr_sigma ** 2)
                    elif keyyarr[j] == 'flux':
                        im = gal.model_data.data['flux'].copy()
                        vmin = flux_vmin
                        vmax = flux_vmax

                    cmaptmp = cmap
                elif k == 'residual':
                    if keyyarr[j] == 'velocity':
                        im = gal.data.data['velocity'] - gal.model_data.data['velocity']
                        im[~gal.data.mask] = np.nan

                        if symmetric_residuals:
                            vmin = -max_residual
                            vmax = max_residual
                    elif keyyarr[j] == 'dispersion':
                        im_model = gal.model_data.data['dispersion'].copy()
                        im_model = np.sqrt(im_model ** 2 - inst_corr_sigma ** 2)

                        im = gal.data.data['dispersion'] - im_model
                        im[~gal.data.mask] = np.nan

                        if symmetric_residuals:
                            vmin = -max_residual
                            vmax = max_residual

                    elif keyyarr[j] == 'flux':
                        im = gal.data.data['flux'].copy() - gal.model_data.data['flux'].copy()
                        im[~gal.data.mask] = np.nan

                        if symmetric_residuals:
                            fabsmax = np.max(np.abs([flux_vmin, flux_vmax]))
                            vmin = -fabsmax
                            vmax = fabsmax

                    cmaptmp = cmap_resid
                    vmin_2d_resid.append(vmin)
                    vmax_2d_resid.append(vmax)
                else:
                    raise ValueError("key not supported.")


                imax = ax.imshow(im, cmap=cmaptmp, interpolation=int_mode,
                                 vmin=vmin, vmax=vmax, origin=origin)


                # ++++++++++++++++++++++++++
                show_alpha_mask = True
                if show_alpha_mask:
                    imtmp = im.copy()
                    imtmp[gal.data.mask] = vmax
                    imtmp[~gal.data.mask] = np.nan

                    # Create an alpha channel of linearly increasing values moving to the right.
                    alphas = np.ones(im.shape)
                    alphas[~gal.data.mask] = alpha_masked
                    alphas[gal.data.mask] = 1.-alpha_unmasked # 0.
                    # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
                    imtmpalph = mplcolors.Normalize(vmin, vmax, clip=True)(imtmp)
                    imtmpalph = cm.Greys_r(imtmpalph)
                    # Now set the alpha channel to the one we created above
                    imtmpalph[..., -1] = alphas

                    immask = ax.imshow(imtmpalph, interpolation=int_mode, origin=origin)
                # ++++++++++++++++++++++++++

                if (show_1d_apers) & (data1d is not None):
                    ax = show_1d_apers_plot(ax, gal, data1d, data2d,
                                    galorig=galorig, alpha_aper=alpha_aper,
                                    remove_shift=remove_shift)



                ax = plot_major_minor_axes_2D(ax, gal, im, gal.data.mask)

                ####################################
                # Show a 1arcsec line:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                #
                ybase_offset = 0.035 #0.065
                x_base = xlim[0] + (xlim[1]-xlim[0])*0.075 # 0.1
                y_base = ylim[0] + (ylim[1]-ylim[0])*(ybase_offset+0.075) #(ybase_offset + 0.06)
                len_line_angular = 1./(pixscale)

                ax.plot([x_base, x_base+len_line_angular], [y_base, y_base],
                            c=color_annotate, ls='-',lw=2, solid_capstyle='butt')
                string = '1"'
                y_text = y_base
                ax.annotate(string, xy=(x_base+len_line_angular*1.25, y_text),
                                xycoords="data",
                                xytext=(0,0),
                                color=color_annotate,
                                textcoords="offset points", ha="left", va="center",
                                fontsize=8)
                ####################################

                if k == 'data':
                    ax.set_ylabel(yt)

                    for pos in ['top', 'bottom', 'left', 'right']:
                        ax.spines[pos].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    for pos in ['top', 'bottom', 'left', 'right']:
                        ax.spines[pos].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                if (j == 0):
                    ax.set_title(xt)

                #########
                cax, kw = colorbar.make_axes_gridspec(ax, pad=0.01,
                        fraction=5./101., aspect=20.)
                cbar = plt.colorbar(imax, cax=cax, **kw)
                cbar.ax.tick_params(labelsize=8)


                if k == 'residual':
                    med = np.median(im[gal.data.mask])
                    rms = np.std(im[gal.data.mask])
                    if keyyarr[j] == 'velocity':
                        median_str = r"$V_{med}="+r"{:0.1f}".format(med)+r"$"
                        scatter_str = r"$V_{rms}="+r"{:0.1f}".format(rms)+r"$"
                    elif keyyarr[j] == 'dispersion':
                        median_str = r"$\sigma_{med}="+r"{:0.1f}".format(med)+r"$"
                        scatter_str = r"$\sigma_{rms}="+r"{:0.1f}".format(rms)+r"$"
                    elif keyyarr[j] == 'flux':
                        median_str = r"$f_{med}="+r"{:0.1f}".format(med)+r"$"
                        scatter_str = r"$f_{rms}="+r"{:0.1f}".format(rms)+r"$"
                    ax.annotate(median_str, (0.01,-0.05), xycoords='axes fraction',
                        ha='left', va='top', fontsize=8)
                    ax.annotate(scatter_str, (0.99,-0.05), xycoords='axes fraction',
                        ha='right', va='top', fontsize=8)


            # -----------------------------------

    # ----------------------------------------------------------------------
    # 1D plotting

    # gal = copy.deepcopy(galorig)
    # gal.data = data2d

    if data1d is None:
        for ax in grid_1D:
            ax.set_axis_off()
    else:

        gal = copy.deepcopy(galorig)
        #
        gal.data = copy.deepcopy(data1d)
        gal.instrument = copy.deepcopy(instorig1d)

        if 'profile1d_type' in data1d.__dict__.keys():
            if data1d.profile1d_type is not None:
                profile1d_type = data1d.profile1d_type

        if galorig.data.ndim >= 2:
            if remove_shift:
                # Should not be shifted here:
                gal.model.geometry.vel_shift = 0

                gal.model.geometry.xshift = 0
                gal.model.geometry.yshift = 0
                # Need to also set the central aperture in the data to (0,0)
                gal.data.aper_center_pix_shift = (0,0)

            else:
                # Testing with Emily's models -- no shifts applied from Hannah
                pass
                #gal.model.geometry.vel_shift = 0

        elif galorig.data.ndim == 1:
            if remove_shift:
                # Should not be shifted here:
                gal.model.geometry.xshift = 0
                gal.model.geometry.yshift = 0
                gal.data.aper_center_pix_shift = (0,0)
                gal.model.geometry.vel_shift = 0


        kwargs_galmodel_1d = kwargs_galmodel.copy()
        if galorig.data.ndim <= 2:
            kwargs_galmodel_1d['ndim_final'] = 1
        elif galorig.data.ndim == 3:
            kwargs_galmodel_1d['ndim_final'] = 3
            datatmp = copy.deepcopy(gal.data)
            gal.data = copy.deepcopy(galorig.data)

        gal.create_model_data(**kwargs_galmodel_1d)

        if galorig.data.ndim == 3:
            #EXTRACT 1D PROFILES HERE! (don't pre-do the dispersion correction)
            #SET EQUAL TO MODEL_DATA!

            gal.model_data_3D = copy.deepcopy(gal.model_data)

            gal.model_data = extract_1D_from_cube(gal.model_data_3D.data, gal, moment=moment, inst_corr=inst_corr)
            gal.data = copy.deepcopy(datatmp)

        galnew = copy.deepcopy(gal)
        model_data = galnew.model_data
        data = data1d #galnew.data
        if (inst_corr_1d):
            model_data.data['dispersion'] = \
                np.sqrt( model_data.data['dispersion']**2 - inst_corr_sigma**2 )


        # COMPARE TO EXISTING PROFILES:
        if 'data1d_2' in galnew.__dict__.keys():
            data1d_2 = galnew.data1d_2.copy()
        else:
            data1d_2 = None

        ######################################

        keyxtitle = r'$r$ [arcsec]'
        keyyarr = ['velocity', 'dispersion', 'flux']
        plottype = ['data', 'residual']
        keyytitlearr = [r'$V$ [km/s]', r'$\sigma$ [km/s]', 'Flux']
        keyytitlearrresid = [r'$V_{\mathrm{data}}-V_{\mathrm{model}}$ [km/s]',
                        r'$\sigma_{\mathrm{data}}-\sigma_{\mathrm{model}}$ [km/s]',
                        r'$f_{\mathrm{data}}-f_{\mathrm{model}}$ [km/s]']

        errbar_lw = 0.5
        errbar_cap = 1.5

        k = -1
        for j in six.moves.xrange(nrows):
            for mm in six.moves.xrange(2):
                # Comparison:
                k += 1
                ax = grid_1D[k]
                padfacxlim = 0.05
                rrange = data.rarr.max() - data.rarr.min()
                xlim = [data.rarr.min() - padfacxlim*rrange, data.rarr.max() + padfacxlim*rrange]
                if plottype[mm] == 'data':
                    try:
                        ax.errorbar( data.rarr, data.data[keyyarr[j]],
                                xerr=None, yerr = data.error[keyyarr[j]],
                                marker=None, ls='None', ecolor='k', zorder=-1.,
                                lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
                        ax.scatter( data.rarr, data.data[keyyarr[j]],
                            c='black', marker='o', s=25, lw=1, label='Data')
                    except:
                        pass


                    if fill_mask:
                        ax.scatter( model_data.rarr, model_data.data[keyyarr[j]],
                            edgecolors='red', facecolors='none', marker='s', s=25, lw=1, zorder=-10., label='Model')
                    else:
                        ax.scatter( model_data.rarr, model_data.data[keyyarr[j]],
                            c='red', marker='s', s=25, lw=1, label='Model')

                    if data1d_2 is not None:
                        if data1d_2.data[keyyarr[j]] is not None:
                            ax.errorbar( data1d_2.rarr, data1d_2.data[keyyarr[j]],
                                    xerr=None, yerr = data1d_2.error[keyyarr[j]],
                                    marker=None, ls='None', ecolor='blue', zorder=-5.,
                                    lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
                            ax.scatter( data1d_2.rarr, data1d_2.data[keyyarr[j]],
                                edgecolors='blue', facecolors='none', marker='o', s=25, lw=1, #zorder=-0.5,
                                label='Data2')
                            rrange = data1d_2.rarr.max() - data1d_2.rarr.min()
                            xlim = [data1d_2.rarr.min() - padfacxlim*rrange, data1d_2.rarr.max() + padfacxlim*rrange]


                    ylim = ax.get_ylim()

                    if fill_mask:
                        if 'filled_mask_data' in data.__dict__.keys():
                            #ylim = [data.filled_mask_data.data[keyyarr[j]].min(), data.filled_mask_data.data[keyyarr[j]].max()]
                            ax.errorbar( data.filled_mask_data.rarr, data.filled_mask_data.data[keyyarr[j]],
                                    xerr=None, yerr = data.filled_mask_data.error[keyyarr[j]],
                                    marker=None, ls='None', ecolor='grey', zorder=-1.,
                                    lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
                            ax.scatter( data.filled_mask_data.rarr, data.filled_mask_data.data[keyyarr[j]],
                                edgecolors='grey', facecolors='none', marker='o', s=25, lw=1, label='Unmasked data')

                    ax.set_xlabel(keyxtitle)
                    ax.set_ylabel(keyytitlearr[j])
                    ax.axhline(y=0, ls='--', color='k', zorder=-10.)

                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

                    if ((show_1d_apers) & (data2d is not None)):
                        # Color gradient background:
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        nstp = 20
                        yrange = ylim[1]-ylim[0]
                        Xtmp = []
                        for nn in six.moves.xrange(nstp+1):
                            ytmp = ylim[1] - nn/(1.*nstp)*yrange
                            Xtmp.append([ytmp, ytmp])

                        ax.imshow(Xtmp, interpolation='bicubic', cmap=cmap,
                                    extent=(xlim[0], xlim[1], ylim[0], ylim[1]), alpha=alpha_bkgd,
                                    zorder=-100., aspect='auto',
                                    vmin=vmin_2d[j], vmax=vmax_2d[j])


                    if k == 0:
                        handles, lbls = ax.get_legend_handles_labels()
                        frameon = True
                        borderpad = 0.25
                        markerscale = 0.8 #1.
                        labelspacing= 0.25
                        handletextpad = 0.2
                        handlelength = 1.
                        fontsize_leg= 7.5
                        legend = ax.legend(handles, lbls,
                            labelspacing=labelspacing, borderpad=borderpad,
                            markerscale=markerscale,
                            handletextpad=handletextpad,
                            handlelength=handlelength,
                            loc='lower right',
                            frameon=frameon, numpoints=1,
                            scatterpoints=1,
                            fontsize=fontsize_leg)

                elif plottype[mm] == 'residual':
                    try:
                        ax.errorbar( data.rarr, data.data[keyyarr[j]]-model_data.data[keyyarr[j]],
                                xerr=None, yerr = data.error[keyyarr[j]],
                                marker=None, ls='None', ecolor='k', zorder=-1.,
                                lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
                        ax.scatter( model_data.rarr, data.data[keyyarr[j]]-model_data.data[keyyarr[j]],
                            c='red', marker='s', s=25, lw=1, label=None)
                    except:
                        pass


                    ax.set_xlabel(keyxtitle)
                    ax.set_ylabel(keyytitlearrresid[j])
                    ax.axhline(y=0, ls='--', color='k', zorder=-10.)

                    ax.set_xlim(xlim)

                    if ((show_1d_apers) & (data2d is not None)):
                        # Color gradient background:
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        nstp = 20
                        yrange = ylim[1]-ylim[0]
                        Xtmp = []
                        for nn in six.moves.xrange(nstp+1):
                            ytmp = ylim[1] - nn/(1.*nstp)*yrange
                            Xtmp.append([ytmp, ytmp])

                        ax.imshow(Xtmp, interpolation='bicubic', cmap=cmap_resid,
                                    extent=(xlim[0], xlim[1], ylim[0], ylim[1]), alpha=alpha_bkgd,
                                    zorder=-100., aspect='auto',
                                    vmin=vmin_2d_resid[j], vmax=vmax_2d_resid[j])

    ######################################


    ################

    f.suptitle(suptitle, fontsize=16, y=0.95)

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()


    #raise ValueError

    return None




#############################################################

def plot_aperture_compare_3D_cubes(gal,
                datacube=None, errcube=None, modelcube=None, mask=None,
                fileout=None,
                slit_width=None, slit_pa=None,
                aper_dist=None,
                fill_mask=False,
                skip_fits=True,
                overwrite=False):
    #
    # if datacube is None:
    #     datacube = gal.data.data
    # if errcube is None:
    #     errcube = gal.data.error
    # if modelcube is None:
    #     modelcube = gal.model_data.data
    # if mask is None:
    #     mask = gal.data.mask
    #
    # # Check for existing file:
    # if (not overwrite) and (fileout is not None):
    #     if os.path.isfile(fileout):
    #         logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
    #         return None
    #
    # ######################################
    # if slit_width is None:
    #     try:
    #         slit_width = gal.instrument.beam.major.to(u.arcsec).value
    #     except:
    #         slit_width = gal.instrument.beam.major_fwhm.to(u.arcsec).value
    # if slit_pa is None:
    #     slit_pa = gal.model.geometry.pa.value
    #
    #
    # pixscale = gal.instrument.pixscale.value
    #
    #
    # rpix = slit_width/pixscale/2.
    #
    # if aper_dist is None:
    #     aper_dist_pix = 1. # pixscale #rstep * 2.
    # else:
    #     #aper_dist_pix = aper_dist/rstep
    #     aper_dist_pix = aper_dist
    #
    #
    # # Aper centers: pick roughly number fitting into size:
    # nx = datacube.shape[2]
    # ny = datacube.shape[1]
    # center_pixel = (np.int(nx / 2.) + gal.model.geometry.xshift,
    #                 np.int(ny / 2.) + gal.model.geometry.yshift)
    #
    # #aper_centers = np.linspace(0.,nx-1, num=nx) - np.int(nx / 2.)
    #
    # nap = np.int(np.floor((nx/rpix)))
    # # Make odd
    # if nap % 2 == 0.:
    #     nap -= 1
    #
    # ###
    # # nap = nx
    #
    # aper_centers_pix = np.linspace(0.,nap-1, num=nap) - np.int(nap / 2.)
    # aper_centers_arcsec = aper_centers*aper_dist_pix*pixscale      # /rstep


    #############################################################

    if datacube is None:
        datacube = gal.data.data
    if errcube is None:
        errcube = gal.data.error
    if modelcube is None:
        modelcube = gal.model_data.data
    if mask is None:
        mask = gal.data.mask

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    ######################################

    if slit_width is None:
        try:
            slit_width = gal.instrument.beam.major.to(u.arcsec).value
        except:
            slit_width = gal.instrument.beam.major_fwhm.to(u.arcsec).value
    if slit_pa is None:
        slit_pa = gal.model.geometry.pa.value

    if mask is None:
        mask = gal.data.mask.copy()

    pixscale = gal.instrument.pixscale.value

    rpix = slit_width/pixscale/2.

    # Aper centers: pick roughly number fitting into size:
    nx = datacube.shape[2]
    ny = datacube.shape[1]
    try:
        center_pixel = (gal.data.xcenter + gal.model.geometry.xshift.value,
                        gal.data.ycenter + gal.model.geometry.yshift.value)
    except:
        center_pixel = (np.int(nx / 2.) + gal.model.geometry.xshift,
                        np.int(ny / 2.) + gal.model.geometry.yshift)


    aper_centers_arcsec = aper_centers_arcsec_from_cube(datacube, gal, mask=mask,
                slit_width=slit_width, slit_pa=slit_pa,
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
                        center_pixel = center_pixel, pixscale=gal.instrument.pixscale.value)

        ## UNWEIGHTED GAUS FIT TO DATA
        # aper_centers2, flux1d2, vel1d2, disp1d2 = apertures.extract_1d_kinematics(spec_arr=specarr,
        #                 cube=data_scaled, mask=mask, err=None,
        #                 center_pixel = center_pixel, pixscale=gal.instrument.pixscale.value)

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

    nrows = np.int(np.round(np.sqrt(len(aper_centers_arcsec))))
    ncols = np.int(np.ceil(len(aper_centers_arcsec)/(1.*nrows)))

    padx = 0.25
    pady = 0.25

    xextra = 0.15 #0.2
    yextra = 0. #0.75

    scale = 2.5 #3.5

    f = plt.figure()
    f.set_size_inches((ncols+(ncols-1)*padx+xextra)*scale, (nrows+pady+yextra)*scale)


    suptitle = '{}: Fitting dim: n={}'.format(gal.name, gal.data.ndim)


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
                # gdata_flux2=None
                # gdata_vel2=None
                # gdata_disp2=None

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
    f.suptitle(suptitle, fontsize=16, y=0.925) #0.95)

    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()

    return None



def plot_spaxel_compare_3D_cubes(gal,
                datacube=None, errcube=None,
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
        datacube = gal.data.data
    if errcube is None:
        errcube = gal.data.error
    if mask is None:
        mask = np.array(gal.data.mask, dtype=np.float)
    if show_model:
        if modelcube is None:
            try:
                modelcube = gal.model_data.data
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
        nrows = np.int(np.round(np.sqrt(npix)))
        ncols = np.int(np.ceil(npix/(1.*nrows)))
        rowinds = np.arange(nrows*ncols)
        colinds = np.arange(nrows*ncols)
        #print("npix={}, nrows={}, ncols={}".format(npix, ncols, nrows))



    padx = 0.25
    pady = 0.25

    xextra = 0.15 #0.2
    yextra = 0. #0.75

    scale = 2.5 #3.5

    f = plt.figure()
    figsize = ((ncols+(ncols-1)*padx+xextra)*scale,
               (nrows+(nrows-1)*pady+yextra)*scale)
    #print("figsize={}; nrows={}, ncols={}".format(figsize, nrows, ncols))
    f.set_size_inches((ncols+(ncols-1)*padx+xextra)*scale,
                      (nrows+(nrows-1)*pady+yextra)*scale)
    #                 (nrows+pady+yextra)*scale)


    suptitle = '{}: Fitting dim: n={}'.format(gal.name, gal.data.ndim)
    #suptitle = None

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

        maskbool = np.array(mask, dtype=np.bool)
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

                        maskarr_bool = np.array(maskarr, dtype=np.bool)

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

                        # UNWEIGHTED:
                        # best_fit = gaus_fit_sp_opt_leastsq(specarr[maskarr_bool], datarr[maskarr_bool],
                        #                 mom0[i,j], mom1[i,j], mom2[i,j])
                        # flux1d2 = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                        # vel1d2 = best_fit[1]
                        # disp1d2 = best_fit[2]
                        # # flux1d2 = None
                        # # vel1d2 = None
                        # # disp1d2 = None

                        gmod_flux2=flux1d_mod_mom
                        gmod_vel2=vel1d_mod_mom
                        gmod_disp2=disp1d_mod_mom

                        # gmod_flux2 = None
                        # gmod_vel2 = None
                        # gmod_disp2 = None

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
                        ax.set_facecolor('#f0f0f0') #'#e3e3e3')
                else:
                    ax.set_axis_off()
            elif (k < len(axes)) & (not skip) & (k >= npix):
                #print("TURNING OFF: typ={}, i={}, j={}, k={}".format(typ, i, j, k))
                ax = axes[k]
                ax.set_axis_off()

    #############################################################
    # Save to file:
    if suptitle is not None:
        #ytitlepos=0.895  # 36.875
        #ytitlepos=0.905  # 18.125
        yoff = 0.105 - 0.01*(36.875-f.get_size_inches()[1])/18.75
        ytitlepos = 1.-yoff
        if f.get_size_inches()[1] < 20.:
            ytitlefontsize = 20
        else:
            ytitlefontsize = 30
        f.suptitle(suptitle, fontsize=ytitlefontsize, y=ytitlepos) # color='red',

    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()

    return None


#############################################################

def plot_channel_maps_3D_cube(gal,
            show_data=True,
            show_model=True,
            show_residual=True,
            vbounds = [-450., 450.],
            delv = 100.,
            vbounds_shift=True,
            cmap=cm.Greys,
            cmap_resid=cm.seismic,
            fileout=None,
            overwrite=False):

    # Check for existing file:
    if (not overwrite) and (fileout is not None):
        if os.path.isfile(fileout):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
            return None

    if (not show_data) | (not show_model):
        show_residual = False

    vbounds = np.array(vbounds)
    if vbounds_shift:
        vbounds += gal.model.geometry.vel_shift.value

    v_slice_lims_arr = np.arange(vbounds[0], vbounds[1]+delv, delv)


    #################################################
    # center slice: flux limits:
    ind = np.int(np.round((len(v_slice_lims_arr)-2)/2.))
    v_slice_lims = v_slice_lims_arr[ind:ind+2]
    subcube = gal.data.data.spectral_slab(v_slice_lims[0]*u.km/u.s, v_slice_lims[1]*u.km/u.s)
    im = subcube.moment0().value * gal.data.mask
    flux_lims = [im.min(), im.max()]
    fac = 1. # 0.33
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
        n_cols = np.int(np.ceil(np.sqrt(len(v_slice_lims_arr)-1)))
        n_rows = np.int(np.ceil((len(v_slice_lims_arr)-1.)/(1.*n_cols)))
        show_multi = False
        if show_data:
            ind_dat = 0
            ind_mod = ind_resid = None
        else:
            ind_mod = 0
            ind_dat = ind_resid = None


    wspace_outer = 0.
    wspace = 0.1 #0.25
    hspace = 0.1 #0.25
    padfac = 0.1 #0.15
    fac = 1.
    f.set_size_inches(fac*scale*n_cols+(n_cols-1)*scale*padfac,scale*n_rows+(n_rows-1)*scale*padfac)

    gs =  gridspec.GridSpec(n_rows,n_cols, wspace=wspace, hspace=hspace)
    # width_ratios=np.ones(n_cols), height_ratios=np.ones(n_rows),


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


    center = np.array([(gal.model_data.data.shape[2]-1.)/2., (gal.model_data.data.shape[1]-1.)/2.])
    #print(center)
    center[0] += gal.model.geometry.xshift.value
    center[1] += gal.model.geometry.yshift.value

    #for k in range(n_rows):
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
            #color_contours='red'
            residual=False
            if typ == 'data':
                cube = gal.data.data*gal.data.mask
                color_contours='blue'
            elif typ == 'model':
                cube = gal.model_data.data*gal.model_data.mask
                color_contours='red'
            elif typ == 'residual':
                cube = gal.data.data*gal.data.mask - gal.model_data.data*gal.model_data.mask
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
                    label_str = "{}: {}".format(gal.name, typ.capitalize())
                else:
                    label_str = typ.capitalize()
            elif ((not show_multi) & (k == 0)):
                label_str = "{}: {}".format(typ.capitalize())
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

def plot_channel_slice(ax=None, speccube=None, v_slice_lims=None, flux_lims=None,
                      center=None, show_pix_coords=False, cmap=cm.Greys,
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
    if np.any(m): #and not quiet:
        print("Too few points to create valid contours")
    while np.any(m):
        contour_levels[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(contour_levels) == 0
    contour_levels.sort()

    ax.contour(imtmp, contour_levels, **contour_kwargs)

    #################################################

    ax.plot(center[0], center[1], '+', mew=1., ms=10., c=color_center) #'cyan') #'white')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ybase_offset = 0.035 #0.065
    x_base = xlim[0] + (xlim[1]-xlim[0])*0.075 # 0.1
    y_base = ylim[0] + (ylim[1]-ylim[0])*(ybase_offset+0.075) #(ybase_offset + 0.06)
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


def plot_3D_data_automask_info(gal, mask_dict, axes=None):
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

    int_mode = "nearest"; origin = 'lower'; cmap=cm.viridis

    xcenter = gal.data.xcenter
    ycenter = gal.data.ycenter
    if xcenter is None:
        xcenter =(gal.data.data.shape[2]-1)/2.
    if ycenter is None:
        ycenter =(gal.data.data.shape[1]-1)/2.


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

#############################################################

def plot_model_1D(gal,
            fitdispersion=True,
            fitflux=False,
            best_dispersion=None,
            inst_corr=True,
            fileout=None,
            **kwargs_galmodel):
    ######################################
    # Setup data/model comparison: if this isn't the fit dimension
    #   data/model comparison (eg, fit in 2D, showing 1D comparison)

    model_data = gal.model_data

    if inst_corr:
            model_data.data['dispersion'] = \
                np.sqrt( model_data.data['dispersion']**2 - \
                    gal.instrument.lsf.dispersion.to(u.km/u.s).value**2 )


    ######################################
    # Setup plot:
    f = plt.figure()
    scale = 3.5
    ncols = 1
    if fitdispersion:
        ncols += 1
    if fitflux:
        ncols += 1
    nrows = 1

    f.set_size_inches(1.1*ncols*scale, nrows*scale)
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.35, hspace=0.2)

    keyxtitle = r'$r$ [arcsec]'
    keyyarr = ['velocity', 'dispersion', 'flux']
    keyytitlearr = [r'$V$ [km/s]', r'$\sigma$ [km/s]', 'Flux [arb]']
    # keyyresidtitlearr = [r'$V_{\mathrm{model}} - V_{\mathrm{data}}$ [km/s]',
    #                 r'$\sigma_{\mathrm{model}} - \sigma_{\mathrm{data}}$ [km/s]']
    keyyresidtitlearr = [r'$V_{\mathrm{data}} - V_{\mathrm{model}}$ [km/s]',
                    r'$\sigma_{\mathrm{data}} - \sigma_{\mathrm{model}}$ [km/s]',
                    r'$\mathrm{Flux_{data} - Flux_{model}}$ [arb]']

    errbar_lw = 0.5
    errbar_cap = 1.5

    axes = []
    k = -1
    for j in six.moves.xrange(ncols):
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

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_model_2D(gal,
            fitdispersion=True,
            fitflux=False,
            fileout=None,
            symmetric_residuals=True,
            max_residual=100.,
            inst_corr=True,
            **kwargs_galmodel):
    #

    ######################################
    # Setup plot:
    f = plt.figure(figsize=(9.5, 6))
    scale = 3.5
    ncols = 1
    if fitdispersion:
        ncols += 1
    if fitflux:
        ncols += 1

    #
    cntr = 1
    grid_vel = ImageGrid(f, '{}'.format(100+ncols*10+cntr),
                         nrows_ncols=(1, 1),
                         direction="row",
                         axes_pad=0.5,
                         add_all=True,
                         label_mode="1",
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="each",
                         cbar_size="5%",
                         cbar_pad="1%",
                         )
    if fitdispersion:
        cntr += 1
        grid_disp = ImageGrid(f, '{}'.format(100+ncols*10+cntr),
                              nrows_ncols=(1, 1),
                              direction="row",
                              axes_pad=0.5,
                              add_all=True,
                              label_mode="1",
                              share_all=True,
                              cbar_location="right",
                              cbar_mode="each",
                              cbar_size="5%",
                              cbar_pad="1%",
                              )

    if fitflux:
        cntr += 1
        grid_flux = ImageGrid(f, '{}'.format(100+ncols*10+cntr),
                              nrows_ncols=(1, 1),
                              direction="row",
                              axes_pad=0.5,
                              add_all=True,
                              label_mode="1",
                              share_all=True,
                              cbar_location="right",
                              cbar_mode="each",
                              cbar_size="5%",
                              cbar_pad="1%",
                              )

    #
    keyxarr = ['model']
    keyyarr = ['velocity', 'dispersion', 'flux']
    keyxtitlearr = ['Model']
    keyytitlearr = [r'$V$', r'$\sigma$', r'Flux']


    int_mode = "nearest"
    origin = 'lower'
    cmap =  cm.Spectral_r  #cm.nipy_spectral
    cmap.set_bad(color='k')

    msk = np.isfinite(gal.model_data.data['velocity'])
    vel_vmin = gal.model_data.data['velocity'][msk].min()
    vel_vmax = gal.model_data.data['velocity'][msk].max()
    if np.abs(vel_vmax) > 400.:
        vel_vmax = 400.
    if np.abs(vel_vmin) > 400.:
        vel_vmin = -400.

    vel_shift = gal.model.geometry.vel_shift.value
    #
    vel_vmin -= vel_shift
    vel_vmax -= vel_shift

    for ax, k, xt in zip(grid_vel, keyxarr, keyxtitlearr):
        im = gal.model_data.data['velocity'].copy()
        im -= vel_shift

        imax = ax.imshow(im, cmap=cmap, interpolation=int_mode,
                         vmin=vel_vmin, vmax=vel_vmax, origin=origin)

        for pos in ['top', 'bottom', 'left', 'right']:
            ax.spines[pos].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_ylabel(keyytitlearr[0])

        ax = plot_major_minor_axes_2D(ax, gal, im, gal.model_data.mask)

        cbar = ax.cax.colorbar(imax)
        cbar.ax.tick_params(labelsize=8)

    if fitdispersion:
        msk = np.isfinite(gal.model_data.data['dispersion'])
        disp_vmin = gal.model_data.data['dispersion'][msk].min()
        disp_vmax = gal.model_data.data['dispersion'][msk].max()

        if np.abs(disp_vmax) > 500:
            disp_vmax = 500.
        if np.abs(disp_vmin) > 500:
            disp_vmin = 0.

        for ax, k in zip(grid_disp, keyxarr):
            im = gal.model_data.data['dispersion'].copy()
            if inst_corr:
                im = np.sqrt(im ** 2 - gal.instrument.lsf.dispersion.to(
                             u.km / u.s).value ** 2)

                disp_vmin = max(0, np.sqrt(disp_vmin**2 - gal.instrument.lsf.dispersion.to(u.km / u.s).value ** 2))
                disp_vmax = np.sqrt(disp_vmax**2 - gal.instrument.lsf.dispersion.to(u.km / u.s).value ** 2)


            imax = ax.imshow(im, cmap=cmap, interpolation=int_mode,
                             vmin=disp_vmin, vmax=disp_vmax, origin=origin)

            ax.set_ylabel(keyytitlearr[1])

            for pos in ['top', 'bottom', 'left', 'right']:
                ax.spines[pos].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = plot_major_minor_axes_2D(ax, gal, im, gal.model_data.mask)

            cbar = ax.cax.colorbar(imax)
            cbar.ax.tick_params(labelsize=8)

    if fitflux:
        msk = np.isfinite(gal.model_data.data['flux'])
        flux_vmin = gal.model_data.data['flux'][msk].min()
        flux_vmax = gal.model_data.data['flux'][msk].max()


        for ax, k in zip(grid_flux, keyxarr):
            im = gal.model_data.data['flux'].copy()

            imax = ax.imshow(im, cmap=cmap, interpolation=int_mode,
                             vmin=flux_vmin, vmax=flux_vmax, origin=origin)

            ax.set_ylabel(keyytitlearr[2])

            for pos in ['top', 'bottom', 'left', 'right']:
                ax.spines[pos].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = plot_major_minor_axes_2D(ax, gal, im, gal.model_data.mask)

            cbar = ax.cax.colorbar(imax)
            cbar.ax.tick_params(labelsize=8)

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()




#############################################################

def plot_model_2D_residual(gal,
            data1d=None,
            data2d=None,
            theta=None,
            fitdispersion=True,
            fitflux=False,
            symmetric_residuals=True,
            max_residual=100.,
            xshift = None,
            yshift = None,
            fileout=None,
            vcrop = False,
            vcrop_value = 800.,
            show_1d_apers=False,
            remove_shift = True,
            inst_corr=None,
            **kwargs_galmodel):

    #
    ######################################
    # Setup plot:

    ncols = 1
    if fitdispersion:
        ncols += 1
    if fitflux:
        ncols += 1

    nrows = 1


    padx = 0.25
    pady = 0.25

    xextra = 0.15 #0.2
    yextra = 0. #0.75

    scale = 2.5 #3.5

    f = plt.figure()
    f.set_size_inches((ncols+(ncols-1)*padx+xextra)*scale, (nrows+pady+yextra)*scale)


    suptitle = '{}: Fitting dim: n={}'.format(gal.name, gal.data.ndim)


    padx = 0.2 #0.4
    pady = 0.1 #0.35
    gs02 = gridspec.GridSpec(nrows, ncols, wspace=padx, hspace=pady)
    grid_2D = []
    # grid_2D_cax = []
    for jj in six.moves.xrange(nrows):
        for mm in six.moves.xrange(ncols):
            grid_2D.append(plt.subplot(gs02[jj,mm]))



    if theta is not None:
        gal.model.update_parameters(theta)     # Update the parameters

    #
    inst_corr_2d = None
    if inst_corr is None:
        if 'inst_corr' in data2d.data.keys():
            inst_corr_2d = data2d.data['inst_corr']
    else:
        inst_corr_2d = inst_corr


    if inst_corr_2d:
        inst_corr_sigma = gal.instrument.lsf.dispersion.to(u.km/u.s).value
    else:
        inst_corr_sigma = 0.

    galorig = copy.deepcopy(gal)
    instorig = copy.deepcopy(gal.instrument)
    try:
        instorig2d = copy.deepcopy(gal.instrument2d)
    except:
        instorig2d = copy.deepcopy(gal.instrument)


    # In case missing / set to None:
    if instorig2d is None:
        instorig2d = copy.deepcopy(gal.instrument)
    # ----------------------------------------------------------------------
    # 2D plotting

    if data2d is None:
        for ax in grid_2D:
            ax.set_axis_off()

    else:
        gal = copy.deepcopy(galorig)
        if gal.data.ndim == 1:
            apply_shift = True
        else:
            apply_shift = False


        gal.data = copy.deepcopy(data2d)
        gal.instrument = copy.deepcopy(instorig2d)
        pixscale = gal.instrument.pixscale.value

        gal.model.update_parameters(theta)

        if apply_shift:
            if xshift is not None:
                gal.model.geometry.xshift = xshift
            if yshift is not None:
                gal.model.geometry.yshift = yshift

        #
        kwargs_galmodel_2d = kwargs_galmodel.copy()
        kwargs_galmodel_2d['ndim_final'] = 2
        kwargs_galmodel_2d['from_data'] = True
        gal.create_model_data(**kwargs_galmodel_2d)


        keyyarr = ['residual']
        keyxarr = ['velocity', 'dispersion', 'flux']
        keyytitlearr = ['Residual']
        keyxtitlearr = [r'$V$', r'$\sigma$', 'Flux']

        int_mode = "nearest"
        origin = 'lower'
        cmap =  cm.Spectral_r
        cmap.set_bad(color='k')


        gamma = 1.5
        cmap_resid = new_diverging_cmap('RdBu_r', diverge = 0.5,
                    gamma_lower=gamma, gamma_upper=gamma,
                    name_new='RdBu_r_stretch')


        cmap.set_bad(color='k')

        color_annotate = 'white'


        # -----------------------
        vel_vmin = gal.data.data['velocity'][gal.data.mask].min()
        vel_vmax = gal.data.data['velocity'][gal.data.mask].max()


        # Check for not too crazy:
        if vcrop:
            if vel_vmin < -vcrop_value:
                vel_vmin = -vcrop_value
            if vel_vmax > vcrop_value:
                vel_vmax = vcrop_value


        vel_shift = gal.model.geometry.vel_shift.value

        #
        vel_vmin -= vel_shift
        vel_vmax -= vel_shift

        disp_vmin = gal.data.data['dispersion'][gal.data.mask].min()
        disp_vmax = gal.data.data['dispersion'][gal.data.mask].max()

        # Check for not too crazy:
        if vcrop:
            if disp_vmin < 0:
                disp_vmin = 0
            if disp_vmax > vcrop_value:
                disp_vmax = vcrop_value

        flux_vmin = gal.data.data['flux'][gal.data.mask].min()
        flux_vmax = gal.data.data['flux'][gal.data.mask].max()

        alpha_unmasked = 1. #0.7 #0.6
        alpha_masked = 0.5   # 0.
        alpha_bkgd = 1. #0.5 #1. #0.5
        alpha_aper = 0.8

        vmin_2d = []
        vmax_2d = []
        vmin_2d_resid = []
        vmax_2d_resid = []

        #for ax, k, xt in zip(grid_2D, keyyarr, keyytitlearr):
        for j in six.moves.xrange(len(keyxarr)):
            for mm in six.moves.xrange(len(keyyarr)):
                kk = j*len(keyyarr) + mm

                k = keyyarr[mm]

                ax = grid_2D[kk]

                xt = keyxtitlearr[j]
                yt = keyytitlearr[mm]

                print("plot_model_2D_residual: doing j={}: {} // mm={}: {}".format(j, keyxarr[j], mm, k))
                # -----------------------------------

                if k == 'residual':
                    if keyxarr[j] == 'velocity':
                        im = gal.data.data['velocity'] - gal.model_data.data['velocity']
                        im[~gal.data.mask] = np.nan
                        if symmetric_residuals:
                            vmin = -max_residual
                            vmax = max_residual
                    elif keyxarr[j] == 'dispersion':
                        im_model = gal.model_data.data['dispersion'].copy()
                        im_model = np.sqrt(im_model ** 2 - inst_corr_sigma ** 2)

                        im = gal.data.data['dispersion'] - im_model
                        im[~gal.data.mask] = np.nan

                        if symmetric_residuals:
                            vmin = -max_residual
                            vmax = max_residual
                    elif keyxarr[j] == 'flux':
                        im = gal.data.data['flux'].copy() - gal.model_data.data['flux'].copy()
                        im[~gal.data.mask] = np.nan

                        if symmetric_residuals:
                            fabsmax = np.max(np.abs([flux_vmin, flux_vmax]))
                            vmin = -fabsmax
                            vmax = fabsmax

                    cmaptmp = cmap_resid
                    vmin_2d_resid.append(vmin)
                    vmax_2d_resid.append(vmax)

                else:
                    raise ValueError("key not supported.")


                imax = ax.imshow(im, cmap=cmaptmp, interpolation=int_mode,
                                 vmin=vmin, vmax=vmax, origin=origin)

                # ++++++++++++++++++++++++++
                imtmp = im.copy()
                imtmp[gal.data.mask] = vel_vmax
                imtmp[~gal.data.mask] = np.nan

                # Create an alpha channel of linearly increasing values moving to the right.
                alphas = np.ones(im.shape)
                alphas[~gal.data.mask] = alpha_masked
                alphas[gal.data.mask] = 1.-alpha_unmasked # 0.
                # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
                imtmpalph = mplcolors.Normalize(vmin, vmax, clip=True)(imtmp)
                imtmpalph = cm.Greys_r(imtmpalph)
                # Now set the alpha channel to the one we created above
                imtmpalph[..., -1] = alphas


                immask = ax.imshow(imtmpalph, interpolation=int_mode, origin=origin)
                # ++++++++++++++++++++++++++

                if (show_1d_apers) & (data1d is not None):

                    ax = show_1d_apers_plot(ax, gal, data1d, data2d,
                                    galorig=galorig, alpha_aper=alpha_aper,
                                    remove_shift=remove_shift)


                ####################################
                # Show a 1arcsec line:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                #
                ybase_offset = 0.035 #0.065
                x_base = xlim[0] + (xlim[1]-xlim[0])*0.075 # 0.1
                y_base = ylim[0] + (ylim[1]-ylim[0])*(ybase_offset+0.075) #(ybase_offset + 0.06)
                len_line_angular = 1./(pixscale)

                ax.plot([x_base, x_base+len_line_angular], [y_base, y_base],
                            c=color_annotate, ls='-',lw=2, solid_capstyle='butt')
                string = '1"'
                y_text = y_base
                ax.annotate(string, xy=(x_base+len_line_angular*1.25, y_text),
                                xycoords="data",
                                xytext=(0,0),
                                color=color_annotate,
                                textcoords="offset points", ha="left", va="center",
                                fontsize=8)
                ####################################


                ax = plot_major_minor_axes_2D(ax, gal, im, gal.data.mask)

                if j == 0:
                    ax.set_ylabel(yt)

                for pos in ['top', 'bottom', 'left', 'right']:
                    ax.spines[pos].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                print("ytitle={}".format(yt))

                if mm == 0:
                    ax.set_title(xt)

                #########
                cax, kw = colorbar.make_axes_gridspec(ax, pad=0.01,
                        fraction=5./101., aspect=20.)
                cbar = plt.colorbar(imax, cax=cax, **kw)
                cbar.ax.tick_params(labelsize=8)


                if k == 'residual':
                    med = np.median(im[gal.data.mask])
                    rms = np.std(im[gal.data.mask])
                    if keyxarr[j] == 'velocity':
                        median_str = r"$V_{med}="+r"{:0.1f}".format(med)+r"$"
                        scatter_str = r"$V_{rms}="+r"{:0.1f}".format(rms)+r"$"
                    elif keyxarr[j] == 'dispersion':
                        median_str = r"$\sigma_{med}="+r"{:0.1f}".format(med)+r"$"
                        scatter_str = r"$\sigma_{rms}="+r"{:0.1f}".format(rms)+r"$"
                    elif keyxarr[j] == 'flux':
                        median_str = r"$f_{med}="+r"{:0.1f}".format(med)+r"$"
                        scatter_str = r"$f_{rms}="+r"{:0.1f}".format(rms)+r"$"
                    ax.annotate(median_str,
                        (0.01,-0.05), xycoords='axes fraction',
                        ha='left', va='top', fontsize=8)
                    ax.annotate(scatter_str,
                        (0.99,-0.05), xycoords='axes fraction',
                        ha='right', va='top', fontsize=8)


    ################

    f.suptitle(suptitle, fontsize=16, y=0.95)

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()

    return None




def plot_rotcurve_components(gal=None, overwrite=False, overwrite_curve_files=False,
            outpath = None,
            plotfile = None,
            fname_model_matchdata = None,
            fname_model_finer = None,
            fname_intrinsic = None,
            moment=False,
            partial_weight=True,
            plot_type='pdf',
            **kwargs_galmodel):

    if (plotfile is None) & (outpath is None):
        raise ValueError
    if plotfile is None:
        plotfile = '{}{}_rot_components.{}'.format(outpath, gal.name, plot_type)
    if fname_model_matchdata is None:
        fname_model_matchdata = '{}{}_out-1dplots.txt'.format(outpath, gal.name)
    if fname_model_finer is None:
        fname_model_finer = '{}{}_out-1dplots_finer_sampling.txt'.format(outpath, gal.name)
    if fname_intrinsic is None:
        fname_intrinsic = '{}{}_vcirc_tot_bary_dm.dat'.format(outpath, gal.name)

    # check if the file exists:
    if overwrite:
        file_exists = False
    else:
        file_exists = os.path.isfile(plotfile)

    # Check if the rot curves are done:
    if overwrite_curve_files:
        curve_files_exist_int = False
        curve_files_exist_obs1d = False
        file_exists = False
    else:
        curve_files_exist_int = os.path.isfile(fname_intrinsic)
        curve_files_exist_obs1d = os.path.isfile(fname_model_finer)

    if not curve_files_exist_obs1d:
        # *_out-1dplots_finer_sampling.txt, *_out-1dplots.txt
        create_vel_profile_files_obs1d(gal=gal, outpath=outpath,
                    fname_finer=fname_model_finer,
                    fname_model_matchdata=fname_model_matchdata,
                    moment=moment,
                    partial_weight=partial_weight,
                    overwrite=overwrite_curve_files,
                    **kwargs_galmodel)
    if not curve_files_exist_int:
        # *_vcirc_tot_bary_dm.dat
        create_vel_profile_files_intrinsic(gal=gal, outpath=outpath,
                    fname_intrinsic=fname_intrinsic,
                    overwrite=overwrite_curve_files,
                    **kwargs_galmodel)


    if not file_exists:
        # ---------------------------------------------------------------------------
        # Read in stuff:
        model_obs = read_bestfit_1d_obs_file(filename=fname_model_finer)
        model_int = read_model_intrinsic_profile(filename=fname_intrinsic)

        deg2rad = np.pi/180.
        sini = np.sin(gal.model.components['geom'].inc.value*deg2rad)

        vel_asymm_drift = gal.model.kinematic_options.get_asymm_drift_profile(model_int.rarr, gal.model)
        vsq = model_int.data['vcirc_tot'] ** 2 - vel_asymm_drift**2
        vsq[vsq<0] = 0.

        model_int.data['vrot'] = np.sqrt(vsq)

        model_int.data['vrot_sini'] = model_int.data['vrot']*sini

        sini_l = np.sin(np.max([gal.model.components['geom'].inc.value - 5., 0.])*deg2rad)
        sini_u = np.sin(np.min([gal.model.components['geom'].inc.value + 5., 90.])*deg2rad)

        model_int.data['vcirc_tot_linc'] = np.sqrt((model_int.data['vrot_sini']/sini_l)**2 + vel_asymm_drift**2 )
        model_int.data['vcirc_tot_uinc'] = np.sqrt((model_int.data['vrot_sini']/sini_u)**2 + vel_asymm_drift**2 )

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
        for i in six.moves.xrange(nrows):
            for j in six.moves.xrange(ncols):
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
        fontsize_leg= 7.5 #8.

        color_arr = ['mediumblue', 'mediumturquoise', 'orange', 'red', 'blueviolet', 'dimgrey']

        # ++++++++++++++++++++++++++++++++++++
        ax = axes[0]


        xlim = [-0.05, np.max([np.max(np.abs(gal.data.rarr)) + 0.5, 2.0])]
        xlim2 = np.array(xlim) / gal.dscale
        ylim = [0., np.max(model_int.data['vcirc_tot'])*1.15]

        # msk = data.data.mask
        if hasattr(gal.data, 'mask_velocity'):
            if gal.data.mask_velocity is not None:
                msk = gal.data.mask_velocity
            else:
                msk = gal.data.mask
        else:
            msk = gal.data.mask

        # Masked points
        ax.errorbar( np.abs(gal.data.rarr[~msk]), np.abs(gal.data.data['velocity'][~msk]),
                xerr=None, yerr = gal.data.error['velocity'][~msk],
                marker=None, ls='None', ecolor='lightgrey', zorder=4.,
                alpha=0.75, lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
        ax.scatter( np.abs(gal.data.rarr[~msk]), np.abs(gal.data.data['velocity'][~msk]),
            edgecolor='lightgrey', facecolor='whitesmoke', marker='s', s=25, lw=1, zorder=5., label=None)

        # Unmasked points
        ax.errorbar( np.abs(gal.data.rarr[msk]), np.abs(gal.data.data['velocity'][msk]),
                xerr=None, yerr = gal.data.error['velocity'][msk],
                marker=None, ls='None', ecolor='dimgrey', zorder=4.,
                alpha=0.75, lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
        ax.scatter( np.abs(gal.data.rarr[msk]), np.abs(gal.data.data['velocity'][msk]),
            edgecolor='dimgrey', facecolor='white', marker='s', s=25, lw=1, zorder=5., label='Data')


        ax.plot( model_obs.rarr, model_obs.data['velocity'],
            c='red', lw=lw, zorder=3., label=r'$V_{\mathrm{rot}} \sin(i)$ observed')

        ax.axhline(y=gal.model.components['dispprof'].sigma0.value, ls='--', color='blueviolet',
                zorder=-20., label=r'Intrinsic $\sigma_0$')

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



        ax.annotate(r'{} $z={:0.1f}$'.format(gal.name, gal.z), (0.5, 0.96),
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

        frameon = True #False
        borderpad = 0.25 #0
        markerscale = 1.#0.8
        labelspacing= 0.25 #0.15
        handletextpad= 0.05 #0.2
        handlelength = 0.
        #legend_properties = {'weight':'bold'}
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
            legend.legendHandles[ind].set_visible(False)
            text.set_color(color_arr[ind])


        # ++++++++++++++++++++++++++++++++++++
        ax = axes[1]

        color_arr = ['mediumblue', 'limegreen', 'purple']


        xlim2 = [xlim2[0], np.max(model_int.rarr)+0.1]
        xlim = np.array(xlim2) * gal.dscale

        ax2 = ax.twiny()

        ax2.plot(model_int.rarr, model_int.data['vcirc_tot'],
                c='mediumblue', lw=lw, label=r'$V_{\mathrm{c}}$ intrinsic', zorder=2.)

        ax2.plot(model_int.rarr, model_int.data['vcirc_bar'],
                c='limegreen', lw=lw, label=r'$V_{\mathrm{bar}}$ intrinsic', zorder=1.)

        ax2.plot(model_int.rarr, model_int.data['vcirc_dm'],
                c='purple', lw=lw, label=r'$V_{\mathrm{DM}}$ intrinsic', zorder=0.)
        ###


        if 'disk+bulge' in gal.model.components.keys():
            ax2.axvline(x=gal.model.components['disk+bulge'].r_eff_disk.value, ls='--', color='dimgrey', zorder=-10.)
            ax2.annotate(r'$R_{\mathrm{eff}}$',
                (gal.model.components['disk+bulge'].r_eff_disk.value + 0.05*(xlim2[1]-xlim2[0]), 0.025*(ylim[1]-ylim[0])), # 0.05
                xycoords='data', ha='left', va='bottom', color='dimgrey', fontsize=fontsize_ann)

            # ax2.axvline(x=gal.model.components['disk+bulge'].r_eff_disk.value*6./1.678, ls='--', color='dimgrey', zorder=-10.)
            # ax2.annotate(r'$6r_d}$',
            #     (gal.model.components['disk+bulge'].r_eff_disk.value*6./1.678 - 0.05*(xlim2[1]-xlim2[0]), 0.975*(ylim[1]-ylim[0])),
            #     xycoords='data', ha='right', va='top', color='dimgrey', fontsize=fontsize_ann)

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
            legend.legendHandles[ind].set_visible(False)
            text.set_color(color_arr[ind])


        #############################################################
        # Save to file:

        plt.savefig(plotfile, bbox_inches='tight', dpi=300)
        plt.close()

    return None

def make_clean_mcmc_plot_names(mcmcResults):
    names = []
    for key in mcmcResults.free_param_names.keys():
        for i in six.moves.xrange(len(mcmcResults.free_param_names[key])):
            param = mcmcResults.free_param_names[key][i]
            key_nice = " ".join(key.split("_"))
            param_nice = " ".join(param.split("_"))
            names.append(key_nice+': '+param_nice)

    return names


#############################################################

def extract_1D_2D_data_gausfit_from_cube(gal,
            slit_width=None, slit_pa=None,
            aper_dist=None, inst_corr=True,
            fill_mask=False):
    try:
        if gal.data2d is not None:
            extract = False
        else:
            extract = True
    except:
        extract = True


    if extract:
        gal.data2d = extract_2D_gausfit_from_cube(gal.data.data, gal,
                        errcube=gal.data.error, inst_corr=inst_corr)


    #
    try:
        if gal.data1d is not None:
            extract = False
        else:
            extract = True
    except:
        extract = True

    if extract:
        gal.data1d = extract_1D_from_cube(gal.data.data, gal,
                errcube=gal.data.error,
                slit_width=slit_width, slit_pa=slit_pa, aper_dist=aper_dist,
                moment=False, inst_corr=inst_corr, fill_mask=fill_mask)


    return gal

#
##################################################

def extract_1D_2D_data_moments_from_cube(gal,
            slit_width=None, slit_pa=None,
            aper_dist=None, inst_corr=True,
            fill_mask=False):
    try:
        if gal.data2d is not None:
            extract = False
        else:
            extract = True
    except:
        extract = True


    if extract:
        gal.data2d = extract_2D_moments_from_cube(gal.data.data, gal, inst_corr=inst_corr)


    try:
        if gal.data1d is not None:
            extract = False
        else:
            extract = True
    except:
        extract = True

    if extract:
        gal.data1d = extract_1D_from_cube(gal.data.data, gal, slit_width=slit_width,
                slit_pa=slit_pa, aper_dist=aper_dist, moment=True, inst_corr=inst_corr, fill_mask=fill_mask)


    return gal

def aper_centers_arcsec_from_cube(data_cube, gal, mask=None,
            slit_width=None, slit_pa=None,
            aper_dist=None, fill_mask=False):

    if slit_width is None:
        try:
            slit_width = gal.instrument.beam.major.to(u.arcsec).value
        except:
            slit_width = gal.instrument.beam.major_fwhm.to(u.arcsec).value
    if slit_pa is None:
        slit_pa = gal.model.geometry.pa.value


    if mask is None:
        mask = gal.data.mask.copy()

    pixscale = gal.instrument.pixscale.value

    rpix = slit_width/pixscale/2.

    #############################


    if aper_dist is None:
        # # EVERY PIXEL
        aper_dist_pix = 1. #pixscale #rstep

        # # EVERY 0.5 PSF FWHM
        # aper_dist_pix = slit_width/2./ pixscale
    else:
        aper_dist_pix = aper_dist/pixscale
        #aper_dist_pix = aper_dist

    # Aper centers: pick roughly number fitting into size:
    nx = data_cube.shape[2]
    ny = data_cube.shape[1]
    try:
        center_pixel = (gal.data.xcenter + gal.model.geometry.xshift.value,
                            gal.data.ycenter + gal.model.geometry.yshift.value)
    except:
        center_pixel = (np.int(nx / 2.) + gal.model.geometry.xshift,
                        np.int(ny / 2.) + gal.model.geometry.yshift)

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
        mask2D = np.array(maskflat, dtype=np.bool)
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
                elif not mask2D[np.int(np.round(ytmp)), np.int(np.round(xtmp))]:
                    rMA_arr.append(-1.*rMA_tmp)  # switch sign: pos / blue for calc becomes neg
                    rMA_tmp = 0
                    ended_MA = True
        #nmax = 2. * np.max(np.abs(np.array(rMA_arr)))
        nmax = None
        rMA_arr = np.array(rMA_arr)

    if rMA_arr is not None:
        aper_centers_pix = np.arange(np.sign(rMA_arr[0])*np.floor(np.abs(rMA_arr[0])),
                                     np.sign(rMA_arr[1])*np.floor(np.abs(rMA_arr[1]))+1., 1.)
    else:
        nap = np.int(np.floor(nmax/aper_dist_pix))  # If aper_dist_pix = 1, than nmax apertures.
                                                    # Otherwise, fewer and more spaced out
        # Make odd
        if nap % 2 == 0.:
            if fill_mask:
                nap -= 1
            else:
                nap += 1
        aper_centers_pix = np.linspace(0.,nap-1, num=nap) - np.int(nap / 2.)

    ######################################
    ## ORIG: 13 ap, -0.75 to 0.75
    # if aper_dist is None:
    #     aper_dist_pix = 1. # pixscale
    # else:
    #     aper_dist_pix = aper_dist/pixscale
    #     #aper_dist_pix = aper_dist
    #
    # # Aper centers: pick roughly number fitting into size:
    # nx = data_cube.shape[2]
    # ny = data_cube.shape[1]
    # try:
    #     center_pixel = (gal.data.xcenter + gal.model.geometry.xshift.value,
    #                         gal.data.ycenter + gal.model.geometry.yshift.value)
    # except:
    #     center_pixel = (np.int(nx / 2.) + gal.model.geometry.xshift,
    #                     np.int(ny / 2.) + gal.model.geometry.yshift)
    #
    # nap = np.int(np.floor((nx/rpix)))
    # # Make odd
    # if nap % 2 == 0.:
    #     nap -= 1
    #
    #
    # aper_centers_pix = np.linspace(0.,nap-1, num=nap) - np.int(nap / 2.)
    # aper_centers_arcsec = aper_centers_pix*aper_dist_pix*pixscale

    aper_centers_arcsec = aper_centers_pix*aper_dist_pix*pixscale
    return aper_centers_arcsec


def extract_1D_from_cube(data_cube, gal, errcube=None, mask=None,
            slit_width=None, slit_pa=None,
            aper_dist=None,
            moment=False, inst_corr=True,
            fill_mask=False):

    # if slit_width is None:
    #     try:
    #         slit_width = gal.instrument.beam.major.to(u.arcsec).value
    #     except:
    #         slit_width = gal.instrument.beam.major_fwhm.to(u.arcsec).value
    # if slit_pa is None:
    #     slit_pa = gal.model.geometry.pa.value
    #
    #
    # if mask is None:
    #     mask = gal.data.mask.copy()
    #
    # pixscale = gal.instrument.pixscale.value
    #
    # rpix = slit_width/pixscale/2.
    #
    #
    #
    # #############################
    #
    #
    # if aper_dist is None:
    #     # # EVERY PIXEL
    #     aper_dist_pix = 1. #pixscale #rstep
    #
    #     # # EVERY 0.5 PSF FWHM
    #     # aper_dist_pix = slit_width/2./ pixscale
    # else:
    #     aper_dist_pix = aper_dist/pixscale
    #     #aper_dist_pix = aper_dist
    #
    # # Aper centers: pick roughly number fitting into size:
    # nx = data_cube.shape[2]
    # ny = data_cube.shape[1]
    # try:
    #     center_pixel = (gal.data.xcenter + gal.model.geometry.xshift.value,
    #                         gal.data.ycenter + gal.model.geometry.yshift.value)
    # except:
    #     center_pixel = (np.int(nx / 2.) + gal.model.geometry.xshift,
    #                     np.int(ny / 2.) + gal.model.geometry.yshift)
    #
    # cPA = np.cos(slit_pa * np.pi/180.)
    # sPA = np.sin(slit_pa * np.pi/180.)
    # ###################
    # if fill_mask:
    #     diag_step = False
    #     if np.abs(cPA) >= np.abs(sPA):
    #         nmax = ny / np.abs(cPA)
    #         if diag_step & (aper_dist_pix == 1.):
    #             aper_dist_pix *= 1. / np.abs(cPA)
    #     else:
    #         nmax = nx / np.abs(sPA)
    #         if diag_step & (aper_dist_pix == 1.):
    #             aper_dist_pix *= 1. / np.abs(sPA)
    #     rMA_arr = None
    # else:
    #     # Just use unmasked range:
    #     maskflat = np.sum(mask, axis=0)
    #     maskflat[maskflat>0] = 1
    #     mask2D = np.array(maskflat, dtype=np.bool)
    #     rstep_A = 0.25
    #
    #     rMA_tmp = 0
    #     rMA_arr = []
    #     # PA is to Blue; rMA_arr is [Blue (neg), Red (pos)]
    #     # but for PA definition blue will be pos step; invert at the end
    #     for fac in [1.,-1.]:
    #         ended_MA = False
    #         while not ended_MA:
    #             rMA_tmp += fac * rstep_A
    #             xtmp = rMA_tmp * -sPA + center_pixel[0]
    #             ytmp = rMA_tmp * cPA  + center_pixel[1]
    #             if (xtmp < 0) | (xtmp > mask2D.shape[1]-1) | (ytmp < 0) | (ytmp > mask2D.shape[0]-1):
    #                 rMA_arr.append(-1.*(rMA_tmp - fac*rstep_A))  # switch sign: pos / blue for calc becomes neg
    #                 rMA_tmp = 0
    #                 ended_MA = True
    #             elif not mask2D[np.int(np.round(ytmp)), np.int(np.round(xtmp))]:
    #                 rMA_arr.append(-1.*rMA_tmp)  # switch sign: pos / blue for calc becomes neg
    #                 rMA_tmp = 0
    #                 ended_MA = True
    #     #nmax = 2. * np.max(np.abs(np.array(rMA_arr)))
    #     nmax = None
    #     rMA_arr = np.array(rMA_arr)
    #
    # if rMA_arr is not None:
    #     aper_centers_pix = np.arange(np.sign(rMA_arr[0])*np.floor(np.abs(rMA_arr[0])),
    #                                  np.sign(rMA_arr[1])*np.floor(np.abs(rMA_arr[1]))+1., 1.)
    # else:
    #     nap = np.int(np.floor(nmax/aper_dist_pix))  # If aper_dist_pix = 1, than nmax apertures.
    #                                                 # Otherwise, fewer and more spaced out
    #     # Make odd
    #     if nap % 2 == 0.:
    #         if fill_mask:
    #             nap -= 1
    #         else:
    #             nap += 1
    #     aper_centers_pix = np.linspace(0.,nap-1, num=nap) - np.int(nap / 2.)
    #
    # ######################################
    # ## ORIG: 13 ap, -0.75 to 0.75
    # # if aper_dist is None:
    # #     aper_dist_pix = 1. # pixscale
    # # else:
    # #     aper_dist_pix = aper_dist/pixscale
    # #     #aper_dist_pix = aper_dist
    # #
    # # # Aper centers: pick roughly number fitting into size:
    # # nx = data_cube.shape[2]
    # # ny = data_cube.shape[1]
    # # try:
    # #     center_pixel = (gal.data.xcenter + gal.model.geometry.xshift.value,
    # #                         gal.data.ycenter + gal.model.geometry.yshift.value)
    # # except:
    # #     center_pixel = (np.int(nx / 2.) + gal.model.geometry.xshift,
    # #                     np.int(ny / 2.) + gal.model.geometry.yshift)
    # #
    # # nap = np.int(np.floor((nx/rpix)))
    # # # Make odd
    # # if nap % 2 == 0.:
    # #     nap -= 1
    # #
    # #
    # # aper_centers_pix = np.linspace(0.,nap-1, num=nap) - np.int(nap / 2.)
    # # aper_centers_arcsec = aper_centers_pix*aper_dist_pix*pixscale
    #
    # aper_centers_arcsec = aper_centers_pix*aper_dist_pix*pixscale
    #
    # ############################################

    if slit_width is None:
        try:
            slit_width = gal.instrument.beam.major.to(u.arcsec).value
        except:
            slit_width = gal.instrument.beam.major_fwhm.to(u.arcsec).value
    if slit_pa is None:
        slit_pa = gal.model.geometry.pa.value

    if mask is None:
        mask = gal.data.mask.copy()

    pixscale = gal.instrument.pixscale.value

    rpix = slit_width/pixscale/2.

    # Aper centers: pick roughly number fitting into size:
    nx = data_cube.shape[2]
    ny = data_cube.shape[1]
    try:
        center_pixel = (gal.data.xcenter + gal.model.geometry.xshift.value,
                        gal.data.ycenter + gal.model.geometry.yshift.value)
    except:
        center_pixel = (np.int(nx / 2.) + gal.model.geometry.xshift,
                        np.int(ny / 2.) + gal.model.geometry.yshift)

    aper_centers_arcsec = aper_centers_arcsec_from_cube(data_cube, gal, mask=mask,
                slit_width=slit_width, slit_pa=slit_pa,
                aper_dist=aper_dist, fill_mask=fill_mask)


    #######

    vel_arr = data_cube.spectral_axis.to(u.km/u.s).value

    apertures = CircApertures(rarr=aper_centers_arcsec, slit_PA=slit_pa, rpix=rpix,
             nx=nx, ny=ny, center_pixel=center_pixel, pixscale=pixscale, moment=moment)

    data_scaled = data_cube.unmasked_data[:].value


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
                                 slit_width=slit_width, slit_pa=slit_pa, inst_corr=inst_corr)
        apertures_redo = CircApertures(rarr=aper_centers_arcsec[ind], slit_PA=slit_pa, rpix=rpix,
                            nx=nx, ny=ny, center_pixel=center_pixel, pixscale=pixscale, moment=moment)
        data1d.apertures = apertures_redo
    else:
        data1d = Data1D(r=aper_centers, velocity=vel1d,vel_disp=disp1d, flux=flux1d,
                            slit_width=slit_width, slit_pa=slit_pa, inst_corr=inst_corr)
        data1d.apertures = apertures

    data1d.profile1d_type = 'circ_ap_cube'
    data1d.xcenter = gal.data.xcenter
    data1d.ycenter = gal.data.ycenter

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
        data1d.filled_mask_data = Data1D(r=aper_centers, velocity=vel1d,vel_disp=disp1d, flux=flux1d,
                            slit_width=slit_width, slit_pa=slit_pa, inst_corr=inst_corr)


    return data1d


def extract_2D_gausfit_from_cube(cubein, gal, errcube=None, inst_corr=True):
    # cubein must be SpectralCube instance!

    mask = BooleanArrayMask(mask= np.array(gal.data.mask, dtype=np.bool), wcs=gal.data.data.wcs)

    data_cube = SpectralCube(data=cubein.unmasked_data[:].value, mask=mask, wcs=cubein.wcs)

    #
    if gal.data.smoothing_type is not None:
        data_cube = apply_smoothing_3D(data_cube,
                    smoothing_type=gal.data.smoothing_type,
                    smoothing_npix=gal.data.smoothing_npix)
        smoothing_type=gal.data.smoothing_type
        smoothing_npix=gal.data.smoothing_npix
    else:
        smoothing_type = None
        smoothing_npix = None


    ## GAUSFIT:
    mom0 = data_cube.moment0().to(u.km/u.s).value
    mom1 = data_cube.moment1().to(u.km/u.s).value
    mom2 = data_cube.linewidth_sigma().to(u.km/u.s).value

    flux = np.zeros(mom0.shape)
    vel = np.zeros(mom0.shape)
    disp = np.zeros(mom0.shape)


    wgt_cube = np.array(data_cube.mask._mask, dtype=np.int64)  # use mask as weights to get "weighted" solution
    try:
        err_cube = gal.data.error.unmasked_data[:].value
        err_cube[err_cube==99.] = err_cube.min()
        err_cube = err_cube / np.abs(data_unscaled[np.isfinite(data_unscaled)]).max()

        #wgt_cube = wgt_cube * 1./err_cube

        wgt_cube = 1./(err_cube)

        # # Do a crude flux cut too:
        # wgt_cube[data_scaled<0.05*np.percentile(data_scaled, 95)] = 0


    except:
       pass

    for i in range(mom0.shape[0]):
        for j in range(mom0.shape[1]):
            mod = apy_mod.models.Gaussian1D(amplitude=mom0[i,j] / np.sqrt(2 * np.pi * mom2[i,j]**2),
                                    mean=mom1[i,j],
                                    stddev=np.abs(mom2[i,j]))
            mod.amplitude.bounds = (0, None)
            mod.stddev.bounds = (0, None)
            mod.mean.bounds = (data_cube.spectral_axis.to(u.km/u.s).value.min(), data_cube.spectral_axis.to(u.km/u.s).value.max())


            fitter = apy_mod.fitting.LevMarLSQFitter()

            # wgts = None
            wgts = wgt_cube[:,i,j]
            if (np.max(np.abs(wgts)) == 0):
                wgts = None

            # ########################
            # # Unmasked fit:
            # best_fit = fitter(mod, data_cube.spectral_axis.to(u.km/u.s).value,
            #             data_cube.unmasked_data[:,i,j].value, weights=wgts)
            # ########################

            ########################
            # Masked fit:
            spec_arr = data_cube.spectral_axis.to(u.km/u.s).value
            flux_arr = data_cube.filled_data[:,i,j].value

            if np.isfinite(flux_arr).sum() >= len(mod._parameters):
                spec_arr = spec_arr[np.isfinite(flux_arr)]
                if wgts is not None:
                    wgts = wgts[np.isfinite(flux_arr)]
                flux_arr = flux_arr[np.isfinite(flux_arr)]


            best_fit = fitter(mod, spec_arr, flux_arr, weights=wgts)
            ########################

            flux[i,j] = np.sqrt( 2. * np.pi)  * best_fit.stddev.value * best_fit.amplitude
            vel[i,j] = best_fit.mean.value
            disp[i,j] = best_fit.stddev.value



    #raise ValueError

    ###########################
    # Flatten mask: only mask fully masked spaxels:
    msk3d_coll = np.sum(gal.data.mask, axis=0)
    whmsk = np.where(msk3d_coll == 0)
    mask = np.ones((gal.data.mask.shape[1], gal.data.mask.shape[2]))
    mask[whmsk] = 0


    # Artificially mask the bad stuff:
    flux[~np.isfinite(flux)] = 0
    vel[~np.isfinite(vel)] = 0
    disp[~np.isfinite(disp)] = 0
    mask[~np.isfinite(vel)] = 0
    mask[~np.isfinite(disp)] = 0


    # # setup data2d:
    # try:
    #     pixscale = gal.instrument.pixscale.value
    # except:
    #     pixscale = None
    data2d = Data2D(pixscale=gal.instrument.pixscale.value, velocity=vel, vel_disp=disp, mask=mask,
                        flux=flux, vel_err=None, vel_disp_err=None, flux_err=None,
                        smoothing_type=smoothing_type, smoothing_npix=smoothing_npix,
                        inst_corr = inst_corr, moment=False,
                        xcenter=gal.data.xcenter, ycenter=gal.data.ycenter)




    return data2d




def extract_2D_moments_from_cube(cubein, gal, inst_corr=True):
    # cubein must be SpectralCube instance!

    mask = BooleanArrayMask(mask= np.array(gal.data.mask, dtype=np.bool), wcs=cubein.wcs)

    data_cube = SpectralCube(data=cubein.unmasked_data[:].value, mask=mask, wcs=cubein.wcs)

    # data_unscaled = cubein.unmasked_data[:].value
    # data_scaled = data_unscaled / np.abs(data_unscaled[np.isfinite(data_unscaled)]).max()
    # # Some extra arbitrary scaling:
    # data_scaled *= 100.
    # data_cube = SpectralCube(data=data_scaled, mask=mask, wcs=cubein.wcs)

    #print("gal.data.smoothing_type={}".format(gal.data.smoothing_type))
    if gal.data.smoothing_type is not None:
        data_cube = apply_smoothing_3D(data_cube,
                    smoothing_type=gal.data.smoothing_type,
                    smoothing_npix=gal.data.smoothing_npix)
        smoothing_type=gal.data.smoothing_type
        smoothing_npix=gal.data.smoothing_npix
    else:
        smoothing_type = None
        smoothing_npix = None

    vel = data_cube.moment1().to(u.km/u.s).value
    disp = data_cube.linewidth_sigma().to(u.km/u.s).value
    flux = data_cube.moment0().to(u.km/u.s).value

    msk3d_coll = np.sum(gal.data.mask, axis=0)
    whmsk = np.where(msk3d_coll == 0)
    mask = np.ones((gal.data.mask.shape[1], gal.data.mask.shape[2]))
    mask[whmsk] = 0


    # Artificially mask the bad stuff:
    flux[~np.isfinite(flux)] = 0
    vel[~np.isfinite(vel)] = 0
    disp[~np.isfinite(disp)] = 0
    mask[~np.isfinite(vel)] = 0
    mask[~np.isfinite(disp)] = 0


    # setup data2d:
    data2d = Data2D(pixscale=gal.instrument.pixscale.value, velocity=vel, vel_disp=disp, mask=mask,
                        flux=flux, vel_err=None, vel_disp_err=None, flux_err=None,
                        smoothing_type=smoothing_type, smoothing_npix=smoothing_npix,
                        inst_corr = inst_corr, moment=True,
                        xcenter=gal.data.xcenter, ycenter=gal.data.ycenter)

    return data2d





#############################################################
def show_1d_apers_plot(ax, gal, data1d, data2d, galorig=None, alpha_aper=0.8, remove_shift=True):

    aper_centers = data1d.rarr
    slit_width = data1d.slit_width
    slit_pa = data1d.slit_pa
    rstep = gal.instrument.pixscale.value
    try:
        rstep1d = gal.instrument1d.pixscale.value
    except:
        rstep1d = rstep
    rpix = slit_width/rstep/2.
    # aper_dist_pix = 2*rpix
    aper_centers_pix = aper_centers/rstep#1d

    if aper_centers[0] <= 0:
        # Starts from neg -> pos:
        pa = slit_pa
    else:
        # Goes "backwards": pos -> neg [but the aper shapes are ALSO stored this way]
        pa = slit_pa + 180

    print(" ndim={}:  xshift={}, yshift={}, vsys2d={}".format(galorig.data.ndim,
                                    gal.model.geometry.xshift.value,
                                    gal.model.geometry.yshift.value,
                                    gal.model.geometry.vel_shift.value))



    nx = data2d.data['velocity'].shape[1]
    ny = data2d.data['velocity'].shape[0]

    try:
        center_pixel_kin = (gal.data.xcenter + gal.model.geometry.xshift.value*rstep/rstep1d,
                            gal.data.ycenter + gal.model.geometry.yshift.value*rstep/rstep1d)
    except:
        center_pixel_kin = (np.int(nx / 2.) + gal.model.geometry.xshift.value*rstep/rstep1d,
                            np.int(ny / 2.) + gal.model.geometry.yshift.value*rstep/rstep1d)

    if not remove_shift:
        if data1d.aper_center_pix_shift is not None:
            try:
                center_pixel = (gal.data.xcenter + data1d.aper_center_pix_shift[0]*rstep/rstep1d,
                                gal.data.ycenter + data1d.aper_center_pix_shift[1]*rstep/rstep1d)
            except:
                center_pixel = (np.int(nx / 2.) + data1d.aper_center_pix_shift[0]*rstep/rstep1d,
                                np.int(ny / 2.) + data1d.aper_center_pix_shift[1]*rstep/rstep1d)
        else:
            try:
                center_pixel = (gal.data.xcenter, gal.data.ycenter)
            except:
                center_pixel = None
    else:
        center_pixel = center_pixel_kin



    if center_pixel is None:
        center_pixel = (np.int(nx / 2.) + gal.model.geometry.xshift.value*rstep/rstep1d,
                        np.int(ny / 2.) + gal.model.geometry.yshift.value*rstep/rstep1d)

    # +++++++++++++++++

    pyoff = 0.
    ax.scatter(center_pixel[0], center_pixel[1], color='magenta', marker='+')
    ax.scatter(center_pixel_kin[0], center_pixel_kin[1], color='cyan', marker='+')
    ax.scatter(np.int(nx / 2), np.int(ny / 2), color='lime', marker='+')

    # +++++++++++++++++

    # First determine the centers of all the apertures that fit within the cube
    xaps, yaps = calc_pix_position(aper_centers_pix, pa, center_pixel[0], center_pixel[1])


    cmstar = cm.plasma
    cNorm = mplcolors.Normalize(vmin=0, vmax=len(xaps)-1)
    cmapscale = cm.ScalarMappable(norm=cNorm, cmap=cmstar)

    for mm, (rap, xap, yap) in enumerate(zip(aper_centers, xaps, yaps)):
        #print("mm={}:  rap={}, xap, yap=({}, {}), rpix={}".format(mm, rap, xap, yap, rpix))
        circle = plt.Circle((xap+pyoff, yap+pyoff), rpix, color=cmapscale.to_rgba(mm, alpha=alpha_aper), fill=False)
        ax.add_artist(circle)
        if (mm == 0):
            ax.scatter(xap+pyoff, yap+pyoff, color=cmapscale.to_rgba(mm), marker='.')

    return ax


def plot_major_minor_axes_2D(ax, gal, im, mask, finer_step=True):
    ####################################
    # Show MAJOR AXIS line, center:
    try:
        center_pixel_kin = (gal.data.xcenter + gal.model.geometry.xshift.value,
                            gal.data.ycenter + gal.model.geometry.yshift.value)
    except:
        center_pixel_kin = ((im.shape[1]-1.)/ 2. + gal.model.geometry.xshift.value,
                            (im.shape[0]-1.)/ 2. + gal.model.geometry.yshift.value)

    # Start going to neg, pos of center, at PA, and check if mask True/not
    #   in steps of pix, then rounding. if False: stop, and set 1 less as the end.

    cPA = np.cos(gal.model.components['geom'].pa.value * np.pi/180.)
    sPA = np.sin(gal.model.components['geom'].pa.value * np.pi/180.)

    A_xs = []
    A_ys = []
    if finer_step:
        rstep_A = 0.25
    else:
        rstep_A = 1.
    #rstep_A = 0.25

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
            elif not mask[np.int(np.round(ytmp)), np.int(np.round(xtmp))]:
                A_xs.append((rMA_tmp) * -sPA + center_pixel_kin[0])
                A_ys.append((rMA_tmp) * cPA  + center_pixel_kin[1])
                rMA_arr.append(-1.*rMA_tmp)  # switch sign: pos / blue for calc becomes neg
                rMA_tmp = 0
                ended_MA = True

    B_xs = []
    B_ys = []
    len_minor_marker = 1./20. * im.shape[0]   # CONSTANT SIZE REL TO AXIS
    for fac in [-1.,1.]:
        B_xs.append(fac*len_minor_marker*0.5 * cPA  + center_pixel_kin[0])
        B_ys.append(fac*len_minor_marker*0.5 * sPA + center_pixel_kin[1])

    lw_major = 3.
    lw_minor = 2.25
    color_kin_axes = 'black'
    ax.plot(A_xs , A_ys, color=color_kin_axes, lw=lw_major, ls='-')
    ax.plot(B_xs, B_ys,color=color_kin_axes, lw=lw_minor, ls='-')

    color_kin_axes2 = 'white'
    fac2 = 0.66
    ax.plot(A_xs , A_ys, color=color_kin_axes2, lw=lw_major*fac2, ls='-')
    ax.plot(B_xs, B_ys,color=color_kin_axes2, lw=lw_minor*fac2, ls='-')

    return ax
