# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Functions for plotting DYSMALPY kinematic model fit results


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging
import copy

# Third party imports
import numpy as np
from astropy.extern import six
import astropy.units as u
import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import matplotlib.colors as mplcolors
from mpl_toolkits.axes_grid1 import ImageGrid, AxesGrid

from matplotlib import colorbar

import corner

from .utils import calc_pix_position
from .utils import extract_1D_2D_data_moments_from_cube

__all__ = ['plot_trace', 'plot_corner', 'plot_bestfit']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


def plot_trace(mcmcResults, fileout=None):
    """
    Plot trace of MCMC walkers
    """
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
    alpha = max(0.01, 1./nWalkers)
    
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


def plot_corner(mcmcResults, fileout=None, step_slice=None):
    """
    Plot corner plot of MCMC result posterior distributions.
    Optional:
            step slice: 2 element tuple/array with beginning and end step number to use
    """
    names = make_clean_mcmc_plot_names(mcmcResults)
    
    if step_slice is None:
        sampler_chain = mcmcResults.sampler['flatchain']
    else:
        sampler_chain = mcmcResults.sampler['chain'][:,step_slice[0]:step_slice[1],:].reshape((-1, mcmcResults.sampler['nParam']))

    title_kwargs = {'horizontalalignment': 'left', 'x': 0.}
    fig = corner.corner(sampler_chain,
                            labels=names,
                            quantiles= [.02275, 0.15865, 0.84135, .97725],
                            truths=mcmcResults.bestfit_parameters,
                            plot_datapoints=False,
                            show_titles=True,
                            bins=40,
                            plot_contours=True,
                            verbose=False,
                            title_kwargs=title_kwargs)
                            
    axes = fig.axes
    nFreeParam = len(mcmcResults.bestfit_parameters)
    for i in six.moves.xrange(nFreeParam):
        ax = axes[i*nFreeParam + i]
        # Format the quantile display.
        best = mcmcResults.bestfit_parameters[i]
        q_m = mcmcResults.bestfit_parameters_l68_err[i]
        q_p = mcmcResults.bestfit_parameters_u68_err[i]
        title_fmt=".2f"
        fmt = "{{0:{0}}}".format(title_fmt).format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(best), fmt(q_m), fmt(q_p))
        
        # Add in the column name if it's given.
        if names is not None:
            title = "{0} = {1}".format(names[i], title)
        ax.set_title(title, **title_kwargs)

    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight')#, dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return None
    
def plot_data_model_comparison_1D(gal, 
            data = None,
            theta = None, 
            oversample=1,
            oversize=1,
            fitdispersion=True, 
            fileout=None):
    ######################################
    # Setup data/model comparison: if this isn't the fit dimension 
    #   data/model comparison (eg, fit in 2D, showing 1D comparison)
    if data is not None:

        # Setup the model with the correct dimensionality:
        galnew = copy.deepcopy(gal)
        galnew.data = data
        galnew.create_model_data(oversample=oversample, oversize=oversize,
                              line_center=galnew.model.line_center)
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
    # keyyresidtitlearr = [r'$V_{\mathrm{model}} - V_{\mathrm{data}}$ [km/s]',
    #                 r'$\sigma_{\mathrm{model}} - \sigma_{\mathrm{data}}$ [km/s]']
    keyyresidtitlearr = [r'$V_{\mathrm{data}} - V_{\mathrm{model}}$ [km/s]',
                    r'$\sigma_{\mathrm{data}} - \sigma_{\mathrm{model}}$ [km/s]']

    errbar_lw = 0.5
    errbar_cap = 1.5

    axes = []
    k = -1
    for j in six.moves.xrange(ncols):
        # Comparison:
        axes.append(plt.subplot(gs[0,j]))
        k += 1
        axes[k].errorbar( data.rarr, data.data[keyyarr[j]],
                xerr=None, yerr = data.error[keyyarr[j]],
                marker=None, ls='None', ecolor='k', zorder=-1.,
                lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
        axes[k].scatter( data.rarr, data.data[keyyarr[j]],
            c='black', marker='o', s=25, lw=1, label=None)
            
            
        axes[k].scatter( model_data.rarr, model_data.data[keyyarr[j]],
            c='red', marker='s', s=25, lw=1, label=None)
        axes[k].set_xlabel(keyxtitle)
        axes[k].set_ylabel(keyytitlearr[j])
        axes[k].axhline(y=0, ls='--', color='k', zorder=-10.)
        # Residuals:
        axes.append(plt.subplot(gs[1,j]))
        k += 1
        axes[k].errorbar( data.rarr, data.data[keyyarr[j]]-model_data.data[keyyarr[j]],
                xerr=None, yerr = data.error[keyyarr[j]],
                marker=None, ls='None', ecolor='k', zorder=-1.,
                lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
        axes[k].scatter( data.rarr, data.data[keyyarr[j]]-model_data.data[keyyarr[j]],
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
            oversample=1,
            oversize=1,
            fitdispersion=True, 
            fileout=None,
            symmetric_residuals=True,
            max_residual=100.):
    #
    try:
        if 'inst_corr' in gal.data.data.keys():
            inst_corr = gal.data.data['inst_corr']
    except:
        inst_corr = False
        
    ######################################
    # Setup plot:
    f = plt.figure(figsize=(9.5, 6))
    scale = 3.5
    if fitdispersion:
        grid_vel = ImageGrid(f, 211,
                             nrows_ncols=(1, 3),
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

        grid_disp = ImageGrid(f, 212,
                              nrows_ncols=(1, 3),
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

    else:
        grid_vel = ImageGrid(f, 111,
                             nrows_ncols=(1, 3),
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
    keyxarr = ['data', 'model', 'residual']
    keyyarr = ['velocity', 'dispersion']
    keyxtitlearr = ['Data', 'Model', 'Residual']
    keyytitlearr = [r'$V$', r'$\sigma$']

    #f.set_size_inches(1.1*ncols*scale, nrows*scale)
    #gs = gridspec.GridSpec(nrows, ncols, wspace=0.05, hspace=0.05)



    int_mode = "nearest"
    origin = 'lower'
    cmap =  cm.Spectral_r #cm.nipy_spectral
    cmap.set_bad(color='k')
    
    vel_vmin = gal.data.data['velocity'][gal.data.mask].min()
    vel_vmax = gal.data.data['velocity'][gal.data.mask].max()
    
    # try:
    #     vel_shift = gal.model.get_vel_shift(model_key_vel_shift=model_key_vel_shift)
    # except:
    #     vel_shift = 0
    vel_shift = gal.model.geometry.vel_shift.value
    
    #
    vel_vmin -= vel_shift
    vel_vmax -= vel_shift
    
    for ax, k, xt in zip(grid_vel, keyxarr, keyxtitlearr):
        if k == 'data':
            im = gal.data.data['velocity'].copy()
            im -= vel_shift
            im[~gal.data.mask] = np.nan
        elif k == 'model':
            im = gal.model_data.data['velocity'].copy()
            im -= vel_shift
            im[~gal.data.mask] = np.nan
        elif k == 'residual':
            im = gal.data.data['velocity'] - gal.model_data.data['velocity']
            im[~gal.data.mask] = np.nan
            if symmetric_residuals:
                vel_vmin = -max_residual
                vel_vmax = max_residual
        else:
            raise ValueError("key not supported.")

        imax = ax.imshow(im, cmap=cmap, interpolation=int_mode,
                         vmin=vel_vmin, vmax=vel_vmax, origin=origin)

        if k == 'data':
            ax.set_ylabel(keyytitlearr[0])
            ax.tick_params(which='both', top='off', bottom='off',
                           left='off', right='off', labelbottom='off',
                           labelleft='off')
            for sp in ax.spines.values():
                sp.set_visible(False)
        else:
            #ax.set_axis_off()
            ax.tick_params(which='both', top='off', bottom='off',
                           left='off', right='off', labelbottom='off',
                           labelleft='off')
            for sp in ax.spines.values():
                sp.set_visible(False)
            
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
            elif k == 'model':
                im = gal.model_data.data['dispersion'].copy()
                
                im[~gal.data.mask] = np.nan

                # Correct model for instrument dispersion
                # if the data is instrument corrected:
                if inst_corr:
                    im = np.sqrt(im ** 2 - gal.instrument.lsf.dispersion.to(
                                 u.km / u.s).value ** 2)

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

            else:
                raise ValueError("key not supported.")

            imax = ax.imshow(im, cmap=cmap, interpolation=int_mode,
                             vmin=disp_vmin, vmax=disp_vmax, origin=origin)

            if k == 'data':
                ax.set_ylabel(keyytitlearr[1])
                ax.tick_params(which='both', top='off', bottom='off',
                               left='off', right='off', labelbottom='off',
                               labelleft='off')
                for sp in ax.spines.values():
                    sp.set_visible(False)
            else:
                #ax.set_axis_off()
                ax.tick_params(which='both', top='off', bottom='off',
                               left='off', right='off', labelbottom='off',
                               labelleft='off')
                for sp in ax.spines.values():
                    sp.set_visible(False)
                
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

    #############################################################
    # Save to file:
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.draw()
        plt.show()
        
#
def plot_data_model_comparison_3D(gal, 
            theta = None, 
            oversample=1,
            oversize=1,
            fitdispersion=True, 
            fileout=None,
            symmetric_residuals=True,
            show_1d_apers = False, 
            max_residual=100.,
            inst_corr = True,
            vcrop=False, 
            vcrop_value=800.):
            
    plot_model_multid(gal, theta=theta, fitdispersion=fitdispersion, 
                oversample=oversample, oversize=oversize, fileout=fileout, 
                symmetric_residuals=symmetric_residuals, max_residual=max_residual,
                show_1d_apers=show_1d_apers,
                inst_corr=inst_corr,
                vcrop=vcrop, 
                vcrop_value=vcrop_value)
                
    return None



def plot_model_1D(gal, 
            oversample=1,
            oversize=1,
            fitdispersion=True, 
            best_dispersion=None, 
            inst_corr=True, 
            fileout=None):
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
    if fitdispersion:
        ncols = 2
    else:
        ncols = 1
    nrows = 1
    
    f.set_size_inches(1.1*ncols*scale, nrows*scale)
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.35, hspace=0.2)

    keyxtitle = r'$r$ [arcsec]'
    keyyarr = ['velocity', 'dispersion']
    keyytitlearr = [r'$V$ [km/s]', r'$\sigma$ [km/s]']
    # keyyresidtitlearr = [r'$V_{\mathrm{model}} - V_{\mathrm{data}}$ [km/s]',
    #                 r'$\sigma_{\mathrm{model}} - \sigma_{\mathrm{data}}$ [km/s]']
    keyyresidtitlearr = [r'$V_{\mathrm{data}} - V_{\mathrm{model}}$ [km/s]',
                    r'$\sigma_{\mathrm{data}} - \sigma_{\mathrm{model}}$ [km/s]']

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
            oversample=1,
            oversize=1,
            fitdispersion=True, 
            fileout=None,
            symmetric_residuals=True,
            max_residual=100.,
            inst_corr=True):
    #
        
    ######################################
    # Setup plot:
    f = plt.figure(figsize=(9.5, 6))
    scale = 3.5
    if fitdispersion:
        grid_vel = ImageGrid(f, 121,
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

        grid_disp = ImageGrid(f, 122,
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

    else:
        grid_vel = ImageGrid(f, 111,
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
    keyyarr = ['velocity', 'dispersion']
    keyxtitlearr = ['Model']
    keyytitlearr = [r'$V$', r'$\sigma$']

    #f.set_size_inches(1.1*ncols*scale, nrows*scale)
    #gs = gridspec.GridSpec(nrows, ncols, wspace=0.05, hspace=0.05)



    int_mode = "nearest"
    origin = 'lower'
    cmap =  cm.Spectral_r  #cm.nipy_spectral
    cmap.set_bad(color='k')
    
    vel_vmin = gal.model_data.data['velocity'].min()
    vel_vmax = gal.model_data.data['velocity'].max()
    if np.abs(vel_vmax) > 400.:
        vel_vmax = 400.
    if np.abs(vel_vmin) > 400.:
        vel_vmin = -400.
    
    # try:
    #     vel_shift = gal.model.get_vel_shift(model_key_vel_shift=model_key_vel_shift)
    # except:
    #     vel_shift = 0
    vel_shift = gal.model.geometry.vel_shift.value
    #
    vel_vmin -= vel_shift
    vel_vmax -= vel_shift
    
    for ax, k, xt in zip(grid_vel, keyxarr, keyxtitlearr):
        im = gal.model_data.data['velocity'].copy()
        im -= vel_shift
            
        imax = ax.imshow(im, cmap=cmap, interpolation=int_mode,
                         vmin=vel_vmin, vmax=vel_vmax, origin=origin)

        ax.set_ylabel(keyytitlearr[0])
        ax.tick_params(which='both', top='off', bottom='off',
                       left='off', right='off', labelbottom='off',
                       labelleft='off')
        for sp in ax.spines.values():
            sp.set_visible(False)
        
        ax.set_ylabel(keyytitlearr[0])

        cbar = ax.cax.colorbar(imax)
        cbar.ax.tick_params(labelsize=8)

    if fitdispersion:

        disp_vmin = gal.model_data.data['dispersion'].min()
        disp_vmax = gal.model_data.data['dispersion'].max()
        
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
            ax.tick_params(which='both', top='off', bottom='off',
                           left='off', right='off', labelbottom='off',
                           labelleft='off')
            for sp in ax.spines.values():
                sp.set_visible(False)

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
        
  

def plot_model_multid(gal, theta=None, fitdispersion=True, 
            oversample=1, oversize=1, fileout=None, 
            symmetric_residuals=True, max_residual=100.,
            show_1d_apers=False, inst_corr=None,
            xshift = None,
            yshift = None,
            vcrop=False, 
            vcrop_value=800.,
            remove_shift=True):
            
    if gal.data.ndim == 1:
        plot_model_multid_base(gal, data1d=gal.data, data2d=gal.data2d, 
                    theta=theta,fitdispersion=fitdispersion, 
                    symmetric_residuals=symmetric_residuals, max_residual=max_residual, 
                    oversample=oversample, oversize=oversize, fileout=fileout, 
                    xshift = xshift,
                    yshift = yshift,
                    show_1d_apers=show_1d_apers,
                    remove_shift=True)
    elif gal.data.ndim == 2:
        plot_model_multid_base(gal, data1d=gal.data1d, data2d=gal.data, 
                    theta=theta,fitdispersion=fitdispersion, 
                    symmetric_residuals=symmetric_residuals,  max_residual=max_residual, 
                    oversample=oversample, oversize=oversize, fileout=fileout,
                    show_1d_apers=show_1d_apers, 
                    remove_shift=remove_shift)
        
    elif gal.data.ndim == 3:
        
        gal = extract_1D_2D_data_moments_from_cube(gal)
        # saves in gal.data1d, gal.data2d
        
        plot_model_multid_base(gal, data1d=gal.data1d, data2d=gal.data2d, 
                    theta=theta,fitdispersion=fitdispersion, 
                    symmetric_residuals=symmetric_residuals,  max_residual=max_residual, 
                    oversample=oversample, oversize=oversize, fileout=fileout,
                    show_1d_apers=show_1d_apers, inst_corr=inst_corr, 
                    vcrop=vcrop, vcrop_value=vcrop_value, 
                    remove_shift=remove_shift)
        
        # raise ValueError("Not implemented yet!")
        
    return None

def plot_model_multid_base(gal, 
            data1d=None, 
            data2d=None, 
            theta=None,
            fitdispersion=True,  
            symmetric_residuals=True, 
            max_residual=100.,
            xshift = None,
            yshift = None, 
            oversample=1, 
            oversize=1, 
            fileout=None,
            vcrop = False, 
            vcrop_value = 800., 
            show_1d_apers=False, 
            remove_shift = True,
            inst_corr=None):
        
    #
    ######################################
    # Setup plot:
    
    if fitdispersion:
        nrows = 2
    else:
        nrows = 1
        
    
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
    grid_1D = [plt.subplot(gs01[0,0]), plt.subplot(gs01[0,1])]
    if fitdispersion:
        grid_1D.append(plt.subplot(gs01[1,0]))
        grid_1D.append(plt.subplot(gs01[1,1]))
    
    
    padx = 0.2 #0.4
    pady = 0.1 #0.35
    gs02 = gridspec.GridSpecFromSubplotSpec(nrows, 3, subplot_spec=gs_outer[1], 
            wspace=padx, hspace=pady)
    grid_2D = []
    # grid_2D_cax = []
    for jj in six.moves.xrange(nrows):
        for mm in six.moves.xrange(3):
            grid_2D.append(plt.subplot(gs02[jj,mm]))
            
            
    
    if theta is not None:
        gal.model.update_parameters(theta)     # Update the parameters
        
    #
    if inst_corr is None:
        if 'inst_corr' in gal.data.data.keys():
            inst_corr = gal.data.data['inst_corr']
    if (inst_corr):
        inst_corr_sigma = gal.instrument.lsf.dispersion.to(u.km/u.s).value
    else:
        inst_corr_sigma = 0.
            
    galorig = copy.deepcopy(gal)
    instorig = copy.deepcopy(gal.instrument)
    
    
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
            
            
        
        gal.data = data2d
        #gal.instrument = instorig
    
        gal.model.update_parameters(theta) 
        
        if apply_shift:
            if xshift is not None:
                gal.model.geometry.xshift = xshift
            if yshift is not None:
                gal.model.geometry.yshift = yshift
    
        gal.create_model_data(oversample=oversample, oversize=oversize,
                                  line_center=gal.model.line_center, ndim_final=2, 
                                  from_data=True)
                              
    
        keyxarr = ['data', 'model', 'residual']
        keyyarr = ['velocity', 'dispersion']
        keyxtitlearr = ['Data', 'Model', 'Residual']
        keyytitlearr = [r'$V$', r'$\sigma$']
    
        int_mode = "nearest"
        origin = 'lower'
        cmap =  cm.Spectral_r #cm.nipy_spectral
        cmap.set_bad(color='k')
        
        
        # cmap_resid = cm.RdBu_r
        
        gamma = 1.5 #1.2 # 1. # 1.5 # 3. # 2. 
        cmap_resid = new_diverging_cmap('RdBu_r', diverge = 0.5, 
                    gamma_lower=gamma, gamma_upper=gamma, 
                    name_new='RdBu_r_stretch')
        
        
        cmap.set_bad(color='k')
        
    
    
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
        
        
        alpha_masked = 0.7 #0.6
        alpha_bkgd = 0.5 #1. #0.5
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
                if keyyarr[j] == 'velocity':
                    if k == 'data':
                        im = gal.data.data['velocity'].copy()
                        im -= vel_shift
                        #im[~gal.data.mask] = np.nan
                        cmaptmp = cmap
                        
                        vmin_2d.append(vel_vmin)
                        vmax_2d.append(vel_vmax)
                    elif k == 'model':
                        im = gal.model_data.data['velocity'].copy()
                        im -= vel_shift
                        #im[~gal.data.mask] = np.nan
                        cmaptmp = cmap
                    elif k == 'residual':
                        im = gal.data.data['velocity'] - gal.model_data.data['velocity']
                        im[~gal.data.mask] = np.nan
                        if symmetric_residuals:
                            vel_vmin = -max_residual
                            vel_vmax = max_residual
                            
                        cmaptmp = cmap_resid
                        
                        vmin_2d_resid.append(vel_vmin)
                        vmax_2d_resid.append(vel_vmax)
                    else:
                        raise ValueError("key not supported.")

                    imax = ax.imshow(im, cmap=cmaptmp, interpolation=int_mode,
                                     vmin=vel_vmin, vmax=vel_vmax, origin=origin)
                                     
                    
                    
                    # ++++++++++++++++++++++++++
                    imtmp = im.copy()
                    imtmp[gal.data.mask] = vel_vmax
                    imtmp[~gal.data.mask] = np.nan
                    
                    # Create an alpha channel of linearly increasing values moving to the right.
                    alphas = np.ones(im.shape)
                    alphas[~gal.data.mask] = alpha_masked
                    alphas[gal.data.mask] = 0.
                    # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
                    imtmpalph = mplcolors.Normalize(vel_vmin, vel_vmax, clip=True)(imtmp)
                    imtmpalph = cm.Greys_r(imtmpalph)
                    # Now set the alpha channel to the one we created above
                    imtmpalph[..., -1] = alphas
                    
                    
                    immask = ax.imshow(imtmpalph, interpolation=int_mode, origin=origin)
                    # ++++++++++++++++++++++++++
                                     
                    if (show_1d_apers) & (data1d is not None):
                        
                        ax = show_1d_apers_plot(ax, gal, data1d, data2d, 
                                        galorig=galorig, alpha_aper=alpha_aper,
                                        remove_shift=remove_shift)
                        
                        
                    #ax.set_ylabel(yt)
                    if k == 'data':
                        ax.set_ylabel(yt)
                        ax.tick_params(which='both', top='off', bottom='off',
                                       left='off', right='off', labelbottom='off',
                                       labelleft='off')
                        for sp in ax.spines.values():
                            sp.set_visible(False)
                        #
                        print("ytitle={}".format(yt))
                    else:
                        #ax.set_axis_off()
                        ax.tick_params(which='both', top='off', bottom='off',
                                       left='off', right='off', labelbottom='off',
                                       labelleft='off')
                        for sp in ax.spines.values():
                            sp.set_visible(False)

                    ax.set_title(xt)
                    
                    #########
                    cax, kw = colorbar.make_axes_gridspec(ax, pad=0.01, 
                            fraction=5./101., aspect=20.)
                    cbar = plt.colorbar(imax, cax=cax, **kw)
                    cbar.ax.tick_params(labelsize=8)
                    
                    
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
                            
                    
                # -----------------------------------
                if keyyarr[j] == 'dispersion':
                    if k == 'data':
                        im = gal.data.data['dispersion'].copy()
                        #im[~gal.data.mask] = np.nan
                        cmaptmp = cmap
                        
                        vmin_2d.append(disp_vmin)
                        vmax_2d.append(disp_vmax)
                        
                    elif k == 'model':
                        im = gal.model_data.data['dispersion'].copy()
                        cmaptmp = cmap
                        #im[~gal.data.mask] = np.nan

                        # Correct model for instrument dispersion
                        # if the data is instrument corrected:
                        if inst_corr:
                            im = np.sqrt(im ** 2 - inst_corr_sigma ** 2)
                            
                            
                        # -------------------------------------------
                        if (show_1d_apers) & (data1d is not None):
                            
                            ax = show_1d_apers_plot(ax, gal, data1d, data2d, 
                                        galorig=galorig, alpha_aper=alpha_aper,
                                        remove_shift=remove_shift)
                            
                        # -------------------------------------------

                    elif k == 'residual':

                        im_model = gal.model_data.data['dispersion'].copy()
                        im_model = np.sqrt(im_model ** 2 - inst_corr_sigma ** 2)


                        im = gal.data.data['dispersion'] - im_model
                        im[~gal.data.mask] = np.nan

                        if symmetric_residuals:
                            disp_vmin = -max_residual
                            disp_vmax = max_residual
                        cmaptmp = cmap_resid
                        
                        vmin_2d_resid.append(disp_vmin)
                        vmax_2d_resid.append(disp_vmax)
                        
                    else:
                        raise ValueError("key not supported.")

                    imax = ax.imshow(im, cmap=cmaptmp, interpolation=int_mode,
                                     vmin=disp_vmin, vmax=disp_vmax, origin=origin)
                    #
                    
                    # ++++++++++++++++++++++++++
                    imtmp = im.copy()
                    imtmp[gal.data.mask] = disp_vmax
                    imtmp[~gal.data.mask] = np.nan
                    
                    # Create an alpha channel of linearly increasing values moving to the right.
                    alphas = np.ones(im.shape)
                    alphas[~gal.data.mask] = alpha_masked
                    alphas[gal.data.mask] = 0.
                    # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
                    imtmpalph = mplcolors.Normalize(disp_vmin, disp_vmax, clip=True)(imtmp)
                    imtmpalph = cm.Greys_r(imtmpalph)
                    # Now set the alpha channel to the one we created above
                    imtmpalph[..., -1] = alphas
                    
                    
                    immask = ax.imshow(imtmpalph, interpolation=int_mode, origin=origin)
                    # ++++++++++++++++++++++++++
                    
                    
                    
                    if k == 'data':
                        ax.set_ylabel(yt)
                        ax.tick_params(which='both', top='off', bottom='off',
                                       left='off', right='off', labelbottom='off',
                                       labelleft='off')
                        for sp in ax.spines.values():
                            sp.set_visible(False)
                    else:
                        #ax.set_axis_off()
                        ax.tick_params(which='both', top='off', bottom='off',
                                       left='off', right='off', labelbottom='off',
                                       labelleft='off')
                        for sp in ax.spines.values():
                            sp.set_visible(False)
                        
                    
                    #########
                    cax, kw = colorbar.make_axes_gridspec(ax, pad=0.01, 
                            fraction=5./101., aspect=20.)
                    cbar = plt.colorbar(imax, cax=cax, **kw)
                    cbar.ax.tick_params(labelsize=8)
                    
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
                            
    #   
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
        gal.data = data1d
        
        
        if galorig.data.ndim == 2:
            if remove_shift:
                # Should not be shifted here:
                gal.model.geometry.vel_shift = 0
                
                gal.model.geometry.xshift = 0
                gal.model.geometry.yshift = 0
                # Need to also set the central aperture in the data to (0,0)  
                gal.data.aper_center_pix_shift = (0,0)
            
            else:
                # Testing with Emily's models -- no shifts applied from Hannah
                #pass
                gal.model.geometry.vel_shift = 0
                
        elif galorig.data.ndim == 1:
            if remove_shift:
                # Should not be shifted here:
                gal.model.geometry.xshift = 0
                gal.model.geometry.yshift = 0
                gal.data.aper_center_pix_shift = (0,0)
    
        try:
            gal.create_model_data(oversample=oversample, oversize=oversize,
                                  line_center=gal.model.line_center, 
                                  ndim_final=1)
        except:
            gal.create_model_data(oversample=oversample, oversize=oversize,
                                  line_center=gal.model.line_center, 
                                  ndim_final=1, from_data=False)
                                  
        galnew = copy.deepcopy(gal)
        model_data = galnew.model_data
        data = data1d #galnew.data
        if (inst_corr):
            model_data.data['dispersion'] = \
                np.sqrt( model_data.data['dispersion']**2 - inst_corr_sigma**2 )
    
    
        ######################################
    
        keyxtitle = r'$r$ [arcsec]'
        keyyarr = ['velocity', 'dispersion']
        plottype = ['data', 'residual']
        keyytitlearr = [r'$V$ [km/s]', r'$\sigma$ [km/s]']
        keyytitlearrresid = [r'$V_{\mathrm{model}}-V_{\mathrm{data}}$ [km/s]', 
                        r'$\sigma_{\mathrm{model}}-\sigma_{\mathrm{data}}$ [km/s]']
    
        errbar_lw = 0.5
        errbar_cap = 1.5
    
        k = -1
        for j in six.moves.xrange(nrows):
            for mm in six.moves.xrange(2):
                # Comparison:
                k += 1
                ax = grid_1D[k]
            
                if plottype[mm] == 'data':
                    try:
                        #
                        ax.errorbar( data.rarr, data.data[keyyarr[j]],
                                xerr=None, yerr = data.error[keyyarr[j]],
                                marker=None, ls='None', ecolor='k', zorder=-1.,
                                lw = errbar_lw,capthick= errbar_lw,capsize=errbar_cap,label=None )
                        ax.scatter( data.rarr, data.data[keyyarr[j]],
                            c='black', marker='o', s=25, lw=1, label=None)
                    except:
                        pass
        
                    ax.scatter( model_data.rarr, model_data.data[keyyarr[j]],
                        c='red', marker='s', s=25, lw=1, label=None)
                    ax.set_xlabel(keyxtitle)
                    ax.set_ylabel(keyytitlearr[j])
                    ax.axhline(y=0, ls='--', color='k', zorder=-10.)
                    
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
                        #Xtmp = [[ylim[1], ylim[1]], [ylim[0], ylim[0]]]
                        print("ylim={}".format(ylim))
                        print("vmin_2d[j]={}".format(vmin_2d[j]))
                        print("vmax_2d[j]={}".format(vmax_2d[j]))
                        ax.imshow(Xtmp, interpolation='bicubic', cmap=cmap, 
                                    extent=(xlim[0], xlim[1], ylim[0], ylim[1]), alpha=alpha_bkgd, 
                                    zorder=-100., aspect='auto', 
                                    vmin=vmin_2d[j], vmax=vmax_2d[j])
                        
                    
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
                    #
                    ax.set_xlabel(keyxtitle)
                    ax.set_ylabel(keyytitlearrresid[j])
                    ax.axhline(y=0, ls='--', color='k', zorder=-10.)
                    
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
                        #Xtmp = [[ylim[1], ylim[1]], [ylim[0], ylim[0]]]
                        #Xtmp = [[ylim[1], ylim[1]], [ylim[0], ylim[0]]]
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
    
    return None
    
    
        
def plot_data_model_comparison(gal, 
                               theta = None,
                               oversample=1,
                               oversize=1,
                               fitdispersion=True,
                               fileout=None,
                               vcrop=False,
                               show_1d_apers=False,
                               vcrop_value=800.,
                               profile1d_type='circ_ap_cube',
                               remove_shift=False):
    """
    Plot data, model, and residuals between the data and this model.
    """

    dummy_gal = gal.copy()

    if remove_shift:
        dummy_gal.data.aper_center_pix_shift = (0,0)

    if theta is not None:
        dummy_gal.model.update_parameters(theta)     # Update the parameters
        dummy_gal.create_model_data(oversample=oversample, oversize=oversize,
                              line_center=gal.model.line_center,
                              profile1d_type=profile1d_type)

    if gal.data.ndim == 1:
        plot_data_model_comparison_1D(dummy_gal,
                    data = None,
                    theta = theta, 
                    oversample=oversample,
                    oversize=oversize,
                    fitdispersion=fitdispersion, 
                    fileout=fileout)
    elif gal.data.ndim == 2:
        plot_data_model_comparison_2D(dummy_gal,
                    theta = theta, 
                    oversample=oversample,
                    oversize=oversize,
                    fitdispersion=fitdispersion, 
                    fileout=fileout)
    elif gal.data.ndim == 3:
        plot_data_model_comparison_3D(dummy_gal,
                    theta = theta, 
                    oversample=oversample,
                    oversize=oversize,
                    fitdispersion=fitdispersion, 
                    show_1d_apers=show_1d_apers, 
                    fileout=fileout, 
                    vcrop=vcrop, 
                    vcrop_value=vcrop_value)
                    
        
        # logger.warning("Need to implement fitting plot_bestfit for 3D *AFTER* Dysmalpy datastructure finalized!")
    else:
        logger.warning("nDim="+str(gal.data.ndim)+" not supported!")
        raise ValueError
    
    return None

def plot_bestfit(mcmcResults, gal,
                 oversample=1,
                 oversize=1,
                 fitdispersion=True,
                 show_1d_apers=False,
                 fileout=None,
                 vcrop=False,
                 vcrop_value=800.,
                 remove_shift=False):
    """
    Plot data, bestfit model, and residuals from the MCMC fitting.
    """
    plot_data_model_comparison(gal, theta = mcmcResults.bestfit_parameters, 
            oversample=oversample, oversize=oversize, fitdispersion=fitdispersion, fileout=fileout,
            vcrop=vcrop, vcrop_value=vcrop_value, show_1d_apers=show_1d_apers, remove_shift=remove_shift)
                
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


#
def new_diverging_cmap(name_original, diverge=0.5, gamma_lower=1.0,
    gamma_upper=1.0, excise_middle=False, bad=None, over=None, under=None,
    name_new=None):
  """
  Provides functions for altering colormaps to make them more suitable for
  particular use cases.

  From Drummond Fielding and Chris White GSPS talk 2016-17
  
  ++++++++++++++++
  
  Creates a recentered and/or stretched and/or sharper version of the given
  colormap.

  Inputs:

    name_original: String naming existing colormap, which should be diverging
    and must have an anchor point at 0.5.

    diverge: Location of new center from which colors diverge. Defaults to 0.5.

    gamma_lower, gamma_upper: Stretch parameters for values below and above the
    diverging point. Must be positive. Values greater than 1 compress colors
    near the diverging point, providing more color resolution there. Values less
    than 1 do the same at the extremes of the range. Default to no stretching.

    excise_middle: Flag indicating the middle point should be removed, with the
    two color ranges joined sharply instead. Defaults to False.

    bad, over, under: Colors to be used for invalid values, values above the
    upper limit, and values below the lower limit. Default to values from
    original map.

    name_new: String under which new colormap will be registered. Defaults to
    prepending 'New' to original name.

  Returns new colormap.
  
  
  
  """

  # Get original colormap
  cmap_original_data = cm.datad[name_original]
  cmap_original = cm.get_cmap(name_original)

  # Define new colormap data
  cmap_new_data = {}
  for color in ('red', 'green', 'blue'):

    # Get original definition
    new_data = np.array(cmap_original._segmentdata[color])
    midpoint = np.where(new_data[:,0] == 0.5)[0][0]

    # Excise middle value if desired
    if excise_middle:
      anchor_lower = new_data[midpoint-1,0]
      anchors_lower = new_data[:midpoint,0]
      anchors_lower = 1.0/(2.0*anchor_lower) * anchors_lower
      anchor_upper = new_data[midpoint+1,0]
      anchors_upper = new_data[midpoint+1:,0]
      anchors_upper = 1.0/(2.0-2.0*anchor_upper) * (anchors_upper-1.0) + 1.0
      anchors = np.concatenate((anchors_lower[:-1], [0.5], anchors_upper[1:]))
      vals_below = \
          np.concatenate((new_data[:midpoint,1], new_data[midpoint+2:,1]))
      vals_above = \
          np.concatenate((new_data[:midpoint-1,2], new_data[midpoint+1:,2]))
      new_data = np.vstack((anchors, vals_below, vals_above)).T
      midpoint -= 1

    # Apply shift and stretch if desired
    anchors_lower = new_data[:midpoint,0]
    if diverge != 0.5 or gamma_lower != 1.0:
      anchors_lower = diverge * (1.0 - (1.0-2.0*anchors_lower) ** gamma_lower)
    anchors_upper = new_data[midpoint+1:,0]
    if diverge != 0.5 or gamma_upper != 1.0:
      anchors_upper = \
          diverge + (1.0-diverge) * (2.0*anchors_upper-1.0) ** gamma_upper
    anchors = np.concatenate((anchors_lower, [diverge], anchors_upper))
    anchors[0] = 0.0
    anchors[-1] = 1.0
    new_data[:,0] = anchors

    # Record changes
    cmap_new_data[color] = new_data

  # Create new colormap
  if name_new is None:
    name_new = 'New' + name_original
  cmap_new = mplcolors.LinearSegmentedColormap(name_new, cmap_new_data)
  bad = cmap_original(np.nan) if bad is None else bad
  over = cmap_original(np.inf) if over is None else over
  under = cmap_original(-np.inf) if under is None else under
  cmap_new.set_bad(bad)
  cmap_new.set_over(over)
  cmap_new.set_under(under)

  # Register and return new colormap
  cm.register_cmap(name=name_new, cmap=cmap_new)
  return cmap_new

#
def show_1d_apers_plot(ax, gal, data1d, data2d, galorig=None, alpha_aper=0.8, remove_shift=True):

    aper_centers = data1d.rarr
    slit_width = data1d.slit_width
    slit_pa = data1d.slit_pa
    rstep = gal.instrument.pixscale.value
    rpix = slit_width/rstep/2.
    aper_dist_pix = 2*rpix
    aper_centers_pix = aper_centers/rstep

    pa = slit_pa


    print(" ndim={}:  xshift={}, yshift={}, vsys2d={}".format(galorig.data.ndim, 
                                    gal.model.geometry.xshift.value, 
                                    gal.model.geometry.yshift.value, 
                                    gal.model.geometry.vel_shift.value))



    nx = data2d.data['velocity'].shape[1]
    ny = data2d.data['velocity'].shape[0]
    
    
    if not remove_shift:
        if data1d.aper_center_pix_shift is not None:
            center_pixel = (np.int(nx / 2) + data1d.aper_center_pix_shift[0], 
                            np.int(ny / 2) + data1d.aper_center_pix_shift[1])
        else:
            center_pixel = None
    else:
        # remove shift:
        center_pixel = None
    
    
    print("center_pixel w/ NO REMOVE shift:")
    print("center_pixel={}".format(center_pixel))
    print("aper_center_pix_shift={}".format(data1d.aper_center_pix_shift))
    
    center_pixel_kin = (np.int(nx / 2) + gal.model.geometry.xshift.value, 
                    np.int(ny / 2) + gal.model.geometry.yshift.value)
    
    
    if center_pixel is None:
        #center_pixel = (np.int(nx / 2), np.int(ny / 2))
        center_pixel = (np.int(nx / 2) + gal.model.geometry.xshift.value, 
                        np.int(ny / 2) + gal.model.geometry.yshift.value)

    #

    #if (gal.data.ndim == 2):
    ax.scatter(center_pixel[0], center_pixel[1], color='cyan', marker='+')
    ax.scatter(center_pixel_kin[0], center_pixel_kin[1], color='magenta', marker='+')
    ax.scatter(np.int(nx / 2), np.int(ny / 2), color='lime', marker='+')

    # Assume equal distance between successive apertures equal to diameter of aperture
    dr = aper_dist_pix

    # First determine the centers of all the apertures that fit within the cube
    xaps, yaps = calc_pix_position(aper_centers_pix, pa, center_pixel[0], center_pixel[1])

    pyoff = 0.

    cmstar = cm.plasma
    cNorm = mplcolors.Normalize(vmin=0, vmax=len(xaps)-1)
    cmapscale = cm.ScalarMappable(norm=cNorm, cmap=cmstar)

    for mm, (rap, xap, yap) in enumerate(zip(aper_centers, xaps, yaps)):
        #print("mm={}:  rap={}".format(mm, rap))
        circle = plt.Circle((xap+pyoff, yap+pyoff), rpix, color=cmapscale.to_rgba(mm, alpha=alpha_aper), fill=False)
        ax.add_artist(circle)
        if (mm == 0):
            ax.scatter(xap+pyoff, yap+pyoff, color=cmapscale.to_rgba(mm), marker='.')
    
    return ax

