# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Functions for plotting DYSMALPY kinematic model fit results


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np
from astropy.extern import six
# import matplotlib
# matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import corner

__all__ = ['plot_trace', 'plot_corner', 'plot_bestfit']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


def plot_trace(mcmcResults, fileout=None):
    names = make_clean_mcmc_plot_names(mcmcResults)

    ######################################
    # Setup plot:
    f = plt.figure()
    scale = 1.75
    n_rows = len(names)
    f.set_size_inches(4.*scale, n_rows*scale)
    gs = gridspec.GridSpec(n_rows, 1, hspace=0.2)

    axes = []
    alpha = max(0.01, 1./mcmcResults.sampler['nWalkers'])

    for k in six.moves.xrange(n_rows):
        axes.append(plt.subplot(gs[k,0]))

        axes[k].plot(mcmcResults.sampler['chain'][:,:,k].T, '-', color='black', alpha=alpha)
        axes[k].set_ylabel(names[k])

        if k == n_rows-1:
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


def plot_corner(mcmcResults, fileout=None):
    names = make_clean_mcmc_plot_names(mcmcResults)
    
    title_kwargs = {'horizontalalignment': 'left', 'x': 0.}
    fig = corner.corner(mcmcResults.sampler['flatchain'],
                            labels=names,
                            quantiles= [.02275, 0.15865, 0.84135, .97725],
                            truths=mcmcResults.bestfit_parameters,
                            plot_datapoints=False,
                            show_titles=True,
                            bins=40,
                            plot_contours=True,
                            verbose=False,
                            title_kwargs=title_kwargs)
                            
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight')#, dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return None

def plot_bestfit(mcmcResults, gal, 
            oversample=1,
            fitdispersion=True, 
            fileout=None):
            
    gal.model.update_parameters(mcmcResults.bestfit_parameters)     # Update the parameters
    gal.create_model_data(oversample=oversample,
                          line_center=gal.model.line_center)
    
    if gal.data.ndim == 1:
        ######################################
        # Setup plot:
        f = plt.figure()
        scale = 3.5
        if fitdispersion:
            ncols = 2 
        else:
            ncols = 1
        nrows = 2
        f.set_size_inches(1.1*ncols*scale, nrows*scale)
        gs = gridspec.GridSpec(nrows, ncols, wspace=0.35, hspace=0.2)
        
        keyxtitle = r'$r$ [arcsec]'
        keyyarr = ['velocity', 'dispersion']
        keyytitlearr = [r'$V$ [km/s]', r'$\sigma$ [km/s]']
        keyyresidtitlearr = [r'$V_{\mathrm{model}} - V_{\mathrm{data}}$ [km/s]', 
                        r'$\sigma_{\mathrm{model}} - \sigma_{\mathrm{data}}$ [km/s]']
        
        errbar_lw = 0.5
        errbar_cap = 1.5 
        
        axes = []
        k = -1
        for j in six.moves.xrange(ncols):
            # Comparison:
            axes.append(plt.subplot(gs[0,j]))
            k += 1
            axes[k].errorbar( gal.data.rarr, gal.data.data[keyyarr[j]], 
                    xerr=None, yerr = gal.data.error[keyyarr[j]], 
                    marker=None, ls='None', ecolor='k', zorder=-1.,
                    lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
            axes[k].scatter( gal.data.rarr, gal.data.data[keyyarr[j]], 
                c='black', marker='o', s=25, lw=1, label=None)
            axes[k].scatter( gal.data.rarr, gal.model_data.data[keyyarr[j]], 
                c='red', marker='s', s=25, lw=1, label=None)
            axes[k].set_xlabel(keyxtitle)
            axes[k].set_ylabel(keyytitlearr[j])
            # Residuals:
            axes.append(plt.subplot(gs[1,j]))
            k += 1
            axes[k].errorbar( gal.data.rarr, gal.model_data.data[keyyarr[j]]-gal.data.data[keyyarr[j]], 
                    xerr=None, yerr = gal.data.error[keyyarr[j]], 
                    marker=None, ls='None', ecolor='k', zorder=-1.,
                    lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
            axes[k].scatter( gal.data.rarr, gal.model_data.data[keyyarr[j]]-gal.data.data[keyyarr[j]], 
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
    elif gal.data.ndim == 2:
        logger.warning("Need to implement fitting plot_bestfit for 2D *AFTER* Dysmalpy datastructure finalized!")
    elif gal.data.ndim == 3:
        logger.warning("Need to implement fitting plot_bestfit for 3D *AFTER* Dysmalpy datastructure finalized!")
    else:
        logger.warning("nDim="+str(gal.data.ndim)+" not supported!")
        raise ValueError
    

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


