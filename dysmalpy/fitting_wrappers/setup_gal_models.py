# Methods for setting up galaxies / models for fitting fitting_wrappers

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys


import numpy as np
import astropy.units as u
import astropy.constants as apy_con

from dysmalpy import parameters
from dysmalpy import fitting
from dysmalpy import models

from dysmalpy import aperture_classes

import emcee


def setup_gal_model_base():

    raise ValueError("Need to implement this!")

    return none


def setup_fit_dict(params=None, ndim_data=None):

    if params['fit_method'] == 'mcmc':

        fit_dict = setup_mcmc_dict(params=params, ndim_data=ndim_data)

    elif params['fit_method'] == 'mpfit':

        fit_dict = setup_mpfit_dict(params=params, ndim_data=ndim_data)

    return fit_dict


def setup_mcmc_dict(params=None, ndim_data=None):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Parameters for the MCMC fitting + output filenames

    fitting.ensure_dir(params['outdir'])

    outdir = params['outdir']
    galID = params['galID']

    if 'plot_type' in params.keys():
        plot_type = params['plot_type']
    else:
        plot_type = 'pdf'

    # All in one directory:
    f_plot_trace_burnin = outdir+'{}_mcmc_burnin_trace.{}'.format(galID, plot_type)
    f_plot_trace = outdir+'{}_mcmc_trace.{}'.format(galID, plot_type)
    f_model = outdir+'{}_galaxy_model.pickle'.format(galID)


    lkskljlsksf
    if ndim_data == 1:
        f_model_bestfit =
            # if gal.data.ndim == 1:
            #     f_model_bestfit = outdir+'galaxy_out-1dplots.txt'
            # elif gal.data.ndim == 2:
            #     f_model_bestfit = outdir+'galaxy_out-velmaps.fits'
            # elif gal.data.ndim == 3:
            #     f_model_bestfit = outdir+'galaxy_out-cube.fits'
            # elif gal.data.ndim == 0:
            #     f_model_bestfit = outdir+'galaxy_out-0d.txt'

    f_cube = outdir+'{}_mcmc_bestfit_model_cube.fits'.format(galID)

    if np.int(emcee.__version__[0]) >= 3:
        ftype_sampler = 'h5'
    else:
        ftype_sampler = 'pickle'
    f_sampler = outdir+'{}_mcmc_sampler.{}'.format(galID, ftype_sampler)
    f_burn_sampler = outdir+'{}_mcmc_burn_sampler.{}'.format(galID, ftype_sampler)

    f_plot_param_corner = outdir+'{}_mcmc_param_corner.{}'.format(galID, plot_type)
    f_plot_bestfit = outdir+'{}_mcmc_best_fit.{}'.format(galID, plot_type)
    f_plot_bestfit_multid = outdir+'{}_mcmc_best_fit_multid.{}'.format(galID, plot_type)
    f_mcmc_results = outdir+'{}_mcmc_results.pickle'.format(galID)
    f_chain_ascii = outdir+'{}_mcmc_chain_blobs.dat'.format(galID)
    f_vel_ascii = outdir+'{}_galaxy_bestfit_vel_profile.dat'.format(galID)
    f_log = outdir+'{}_info.log'.format(galID)

    mcmc_dict = {'outdir': outdir,
                'f_plot_trace_burnin':  f_plot_trace_burnin,
                'f_plot_trace':  f_plot_trace,
                'f_model': f_model,
                'f_model_bestfit': f_model_bestfit,
                'f_cube': f_cube,
                'f_sampler':  f_sampler,
                'f_burn_sampler':  f_burn_sampler,
                'f_plot_param_corner':  f_plot_param_corner,
                'f_plot_bestfit':  f_plot_bestfit,
                'f_plot_bestfit_multid': f_plot_bestfit_multid,
                'f_mcmc_results':  f_mcmc_results,
                'f_chain_ascii': f_chain_ascii,
                'f_vel_ascii': f_vel_ascii,
                'f_log': f_log,
                'do_plotting': True}

    for key in params.keys():
        # Copy over all various fitting options
        mcmc_dict[key] = params[key]

    # #
    if 'linked_posteriors' in mcmc_dict.keys():
        if mcmc_dict['linked_posteriors'] is not None:
            linked_post_arr = []
            for lpost in mcmc_dict['linked_posteriors']:
                if lpost.strip().lower() == 'total_mass':
                    linked_post_arr.append(['disk+bulge', 'total_mass'])
                elif lpost.strip().lower() == 'mvirial':
                    linked_post_arr.append(['halo', 'mvirial'])
                elif lpost.strip().lower() == 'fdm':
                    linked_post_arr.append(['halo', 'fdm'])
                elif lpost.strip().lower() == 'alpha':
                    linked_post_arr.append(['halo', 'alpha'])
                elif lpost.strip().lower() == 'rb':
                    linked_post_arr.append(['halo', 'rB'])
                elif lpost.strip().lower() == 'r_eff_disk':
                    linked_post_arr.append(['disk+bulge', 'r_eff_disk'])
                elif lpost.strip().lower() == 'bt':
                    linked_post_arr.append(['disk+bulge', 'bt'])
                elif lpost.strip().lower() == 'sigma0':
                    linked_post_arr.append(['dispprof', 'sigma0'])
                elif lpost.strip().lower() == 'inc':
                    linked_post_arr.append(['geom', 'inc'])
                elif lpost.strip().lower() == 'pa':
                    linked_post_arr.append(['geom', 'pa'])
                elif lpost.strip().lower() == 'xshift':
                    linked_post_arr.append(['geom', 'xshift'])
                elif lpost.strip().lower() == 'yshift':
                    linked_post_arr.append(['geom', 'yshift'])
                elif lpost.strip().lower() == 'vel_shift':
                    linked_post_arr.append(['geom', 'vel_shift'])
                else:
                    raise ValueError("linked posterior for {} not currently implemented!".format(lpost))

            # "Bundle of linked posteriors"
            linked_posterior_names = [ linked_post_arr ]
            mcmc_dict['linked_posterior_names'] = linked_posterior_names
        else:
            mcmc_dict['linked_posterior_names'] = None
    else:
        mcmc_dict['linked_posterior_names'] = None


    #
    mcmc_dict['model_key_re'] = ['disk+bulge', 'r_eff_disk']
    mcmc_dict['model_key_halo'] = ['halo']


    if 'continue_steps' not in mcmc_dict.keys():
        mcmc_dict['continue_steps'] = False

    return mcmc_dict


def setup_mpfit_dict(params=None, ndim_data=None):
    if 'plot_type' in params.keys():
        plot_type = params['plot_type']
    else:
        plot_type = 'pdf'

    fitting.ensure_dir(params['outdir'])
    outdir = params['outdir']
    galID = params['galID']
    f_model = outdir+'{}_galaxy_model.pickle'.format(galID)
    f_model_bestfit =
    f_cube = outdir+'{}_mpfit_bestfit_model_cube.fits'.format(galID)
    f_plot_bestfit = outdir+'{}_mpfit_best_fit.{}'.format(galID, plot_type)
    f_results = outdir+'{}_mpfit_results.pickle'.format(galID)
    f_plot_bestfit_multid = outdir+'{}_mpfit_best_fit_multid.{}'.format(galID, plot_type)
    f_vel_ascii = outdir+'{}_galaxy_bestfit_vel_profile.dat'.format(galID)
    f_log = outdir+'{}_info.log'.format(galID)

    mpfit_dict = {'outdir': outdir,
                  'f_model': f_model,
                  'f_model_bestfit': f_model_bestfit,
                  'f_cube': f_cube,
                  'f_plot_bestfit':  f_plot_bestfit,
                  'f_plot_bestfit_multid': f_plot_bestfit_multid,
                  'f_results':  f_results,
                  'f_vel_ascii': f_vel_ascii,
                  'f_log': f_log,
                  'do_plotting': True}

    for key in params.keys():
        # Copy over all various fitting options
        mpfit_dict[key] = params[key]

    return mpfit_dict


def setup_basic_aperture_types(gal=None, params=None):

    if ('aperture_radius' in params.keys()):
        aperture_radius=params['aperture_radius']
    else:
        aperture_radius = None

    if ('pix_perp' in params.keys()):
        pix_perp=params['pix_perp']
    else:
        pix_perp = None

    if ('pix_parallel' in params.keys()):
        pix_parallel=params['pix_parallel']
    else:
        pix_parallel = None

    if ('pix_length' in params.keys()):
        pix_length=params['pix_length']
    else:
        pix_length = None


    if 'partial_weight' in params.keys()):
        partial_weight = params['partial_weight']
    else:
        # # Preserve previous default behavior
        # partial_weight = False

        ## NEW default behavior: always use partial_weight:
        partial_weight = True

    if ('moment_calc' in params.keys()):
        moment_calc = params['moment_calc']
    else:
        moment_calc = False

    apertures = aperture_classes.setup_aperture_types(gal=gal,
                profile1d_type=params['profile1d_type'],
                aperture_radius=aperture_radius,
                pix_perp=pix_perp, pix_parallel=pix_parallel,
                pix_length=pix_length, from_data=True,
                partial_weight=partial_weight,
                moment=moment_calc)


    return apertures

def setup_data_weighting_method(method='UNSET', r=None):
    if r is not None:
        rmax = np.abs(np.max(r))
    else:
        rmax = None

    if method == 'UNSET':
        raise ValueError("Must set method if setting data point weighting!")
    elif (method is None):
        weight = None
    elif ((method.strip().lower() == 'none') | (method.strip().lower() == 'uniform')):
        weight = None
        #weight = np.ones(len(r), dtype=np.float)
    # exp[ A * (r/rmax) ]  // exponential or more general power-law
    elif method.strip().lower() == 'radius_rmax':
        # value at 0: 1 // value at rmax: 2.718
        weight = np.exp( np.abs(r)/ rmax )
    elif method.strip().lower() == 'radius_rmax_a5':
        # value at 0: 1 // value at rmax: 5.
        weight = np.exp( np.log(5.) * (np.abs(r)/ rmax)  )
    elif method.strip().lower() == 'radius_rmax_a10':
        # value at 0: 1 // value at rmax: 10.
        weight = np.exp( np.log(10.) * (np.abs(r)/ rmax)  )
    # exp[ A * (r/rmax)^2 ]  // exponential or more general power-law
    elif method.strip().lower() == 'radius_rmax2':
        # value at 0: 1 // value at rmax: 2.718
        weight = np.exp((np.abs(r)/ rmax)**2 )
    elif method.strip().lower() == 'radius_2rmax2':
        # value at 0: 1 // value at rmax: 7.389
        weight = np.exp( 2. * (np.abs(r)/ rmax)**2 )
    elif method.strip().lower() == 'radius_rmax2_a5':
        # value at 0: 1 // value at rmax: 5.
        weight = np.exp( np.log(5.) * (np.abs(r)/ rmax)**2 )
    elif method.strip().lower() == 'radius_rmax2_a10':
        # value at 0: 1 // value at rmax: 10.
        weight = np.exp( np.log(10.) * (np.abs(r)/ rmax)**2 )
    else:
        raise ValueError("Weighting method not implmented yet!: {}".format(method))

    return weight


def set_comp_param_prior(comp=None, param_name=None, params=None):
    if params['{}_fixed'.format(param_name)] is False:
        if '{}_prior'.format(param_name) in list(params.keys()):
            # Default to using pre-set value!
            try:
                try:
                    center = comp.prior[param_name].center
                except:
                    center = params[param_name]
            except:
                # eg, UniformPrior
                center = None

            # Default to using pre-set value, if already specified!!!
            try:
                try:
                    stddev = comp.prior[param_name].stddev
                except:
                    stddev = params['{}_stddev'.format(param_name)]
            except:
                stddev = None

            if params['{}_prior'.format(param_name)].lower() == 'flat':
                comp.__getattribute__(param_name).prior = parameters.UniformPrior()
            elif params['{}_prior'.format(param_name)].lower() == 'flat_linear':
                comp.__getattribute__(param_name).prior = parameters.UniformLinearPrior()
            elif params['{}_prior'.format(param_name)].lower() == 'gaussian':
                comp.__getattribute__(param_name).prior = parameters.BoundedGaussianPrior(center=center, stddev=stddev)
            elif params['{}_prior'.format(param_name)].lower() == 'sine_gaussian':
                comp.__getattribute__(param_name).prior = parameters.BoundedSineGaussianPrior(center=center, stddev=stddev)
            elif params['{}_prior'.format(param_name)].lower() == 'gaussian_linear':
                comp.__getattribute__(param_name).prior = parameters.BoundedGaussianLinearPrior(center=center, stddev=stddev)
            elif params['{}_prior'.format(param_name)].lower() == 'tied_flat_lowtrunc':
                comp.__getattribute__(param_name).prior = TiedUniformPriorLowerTrunc(compn='disk+bulge', paramn='total_mass')
            elif params['{}_prior'.format(param_name)].lower() == 'tied_gaussian_lowtrunc':
                comp.__getattribute__(param_name).prior = TiedBoundedGaussianPriorLowerTrunc(center=center, stddev=stddev,
                                                            compn='disk+bulge', paramn='total_mass')
            else:
                print(" CAUTION: {}: {} prior is not currently supported. Defaulting to 'flat'".format(param_name,
                                    params['{}_prior'.format(param_name)]))
                pass

    return comp
