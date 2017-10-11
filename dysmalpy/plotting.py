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
    # possible: , title_kwargs={"fontsize": 12}

    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight')#, dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return None

def plot_bestfit(mcmcResults, fileout=None):
    logger.warning("Need to implement fitting plot_bestfit *AFTER* Dysmalpy datastructure finalized!")

    #raise ValueError

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


