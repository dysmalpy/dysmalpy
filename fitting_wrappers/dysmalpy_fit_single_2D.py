# Script to fit single object in 2D with Dysmalpy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from contextlib import contextmanager
import sys
import shutil

import matplotlib
matplotlib.use('agg')

from dysmalpy import galaxy
from dysmalpy import models
from dysmalpy import fitting
from dysmalpy import instrument
from dysmalpy import parameters
from dysmalpy import plotting

from dysmalpy.instrument import DoubleBeam, Moffat, GaussianBeam

import copy
import numpy as np
import astropy.units as u

try:
    import utils_io
    import plotting as fw_plotting
except:
    from . import utils_io
    from . import plotting as fw_plotting




def dysmalpy_fit_single_2D(param_filename=None, data=None):
    
    # Read in the parameters from param_filename:
    params = utils_io.read_fitting_params(fname=param_filename)
    
    # Setup some paths:
    outdir = utils_io.ensure_path_trailing_slash(params['outdir'])
    params['outdir'] = outdir
    
    fitting.ensure_dir(params['outdir'])

    # Check if fitting already done:
    if params['fit_method'] == 'mcmc':

        fit_exists = os.path.isfile(outdir+'{}_mcmc_results.pickle'.format(params['galID']))

    elif params['fit_method'] == 'mpfit':

        fit_exists = os.path.isfile(outdir + '{}_mpfit_results.pickle'.format(params['galID']))

    else:

        raise ValueError(
            '{} not accepted as a fitting method. Please only use "mcmc" or "mpfit"'.format(
                params['fit_method']))

    
    if fit_exists: 
        print('------------------------------------------------------------------')
        print(' Fitting already complete for: {}'.format(params['galID']))
        print("   make new output folder or remove previous fitting files")
        print('------------------------------------------------------------------')
        print(" ")
    else:

        # Copy paramfile that is OS independent
        shutil.copy(param_filename, outdir)
        
        #######################
        # Setup
        gal, fit_dict = setup_single_object_2D(params=params, data=data)

        # Clean up existing log file:
        if os.path.isfile(fit_dict['f_log']):
            os.remove(fit_dict['f_log'])

        # Fit
        if fit_dict['fit_method'] == 'mcmc':
            results = fitting.fit(gal, nWalkers=fit_dict['nWalkers'],
                                       nCPUs=fit_dict['nCPUs'],
                                       scale_param_a=fit_dict['scale_param_a'],
                                       nBurn=fit_dict['nBurn'],
                                       nSteps=fit_dict['nSteps'],
                                       minAF=fit_dict['minAF'],
                                       maxAF=fit_dict['maxAF'],
                                       nEff=fit_dict['nEff'],
                                       do_plotting=fit_dict['do_plotting'],
                                       red_chisq=fit_dict['red_chisq'],
                                       oversampled_chisq=fit_dict['oversampled_chisq'],
                                       oversample=fit_dict['oversample'],
                                       fitdispersion=fit_dict['fitdispersion'],
                                       blob_name=fit_dict['blob_name'], 
                                       outdir=fit_dict['outdir'],
                                       save_bestfit_cube=False,
                                       linked_posterior_names=fit_dict['linked_posterior_names'], 
                                       model_key_re=fit_dict['model_key_re'], 
                                       model_key_halo=fit_dict['model_key_halo'], 
                                       f_plot_trace_burnin=fit_dict['f_plot_trace_burnin'],
                                       f_plot_trace=fit_dict['f_plot_trace'],
                                       f_model=fit_dict['f_model'],
                                       f_sampler=fit_dict['f_sampler'],
                                       f_burn_sampler=fit_dict['f_burn_sampler'],
                                       f_plot_param_corner=fit_dict['f_plot_param_corner'],
                                       f_plot_bestfit=fit_dict['f_plot_bestfit'],
                                       f_mcmc_results=fit_dict['f_mcmc_results'],
                                       f_chain_ascii=fit_dict['f_chain_ascii'],
                                       f_vel_ascii=fit_dict['f_vel_ascii'],
                                       f_log=fit_dict['f_log'])

        elif fit_dict['fit_method'] == 'mpfit':

            results = fitting.fit_mpfit(gal, oversample=fit_dict['oversample'],
                                        oversize=fit_dict['oversize'],
                                        fitdispersion=fit_dict['fitdispersion'],
                                        maxiter=fit_dict['maxiter'],
                                        do_plotting=fit_dict['do_plotting'],
                                        outdir=fit_dict['outdir'],
                                        f_model=fit_dict['f_model'],
                                        f_cube=fit_dict['f_cube'],
                                        f_plot_bestfit=fit_dict['f_plot_bestfit'],
                                        f_results=fit_dict['f_results'],
                                        f_vel_ascii=fit_dict['f_vel_ascii'],
                                        f_log=fit_dict['f_log'],
                                        blob_name=fit_dict['blob_name'])

        # Save results
        utils_io.save_results_ascii_files(fit_results=results, gal=gal, params=params)
        
        
        # Plot multid, if enabled:
        if 'fdata_1d' in params.keys():
            fw_plotting.plot_results_multid(param_filename=param_filename, fit_ndim=2, show_1d_apers=True, remove_shift=True)
    
    return None
    
    
def dysmalpy_reanalyze_single_2D(param_filename=None, data=None):
    
    # Read in the parameters from param_filename:
    params = utils_io.read_fitting_params(fname=param_filename)
    
    # Setup some paths:
    outdir = utils_io.ensure_path_trailing_slash(params['outdir'])
    params['outdir'] = outdir
    
    fitting.ensure_dir(params['outdir'])

    # Check if fitting already done:
    if params['fit_method'] == 'mcmc':
        # Copy paramfile that is OS independent
        shutil.copy(param_filename, outdir)
        
        # Reload the results, etc
        #######################
        # Reload stuff
        galtmp, fit_dict = setup_single_object_2D(params=params, data=data)
        
        
        try:
            gal, results = fitting.reload_all_fitting(filename_galmodel=fit_dict['f_model'], 
                                    filename_mcmc_results=fit_dict['f_mcmc_results'])
        except:
            # Something went wrong after sampler was saved
            gal = copy.deepcopy(galtmp)
            
            # +++++++++++++++++++++++
            # Setup for oversampled_chisq:
            gal = fitting.setup_oversampled_chisq(gal)
            # +++++++++++++++++++++++
            
            
            sampler_dict = fitting.load_pickle(fit_dict['f_sampler'])
            results = fitting.MCMCResults(model=gal.model, sampler=sampler_dict,
                                      f_plot_trace_burnin = fit_dict['f_plot_trace_burnin'],
                                      f_plot_trace = fit_dict['f_plot_trace'],
                                      f_sampler = fit_dict['f_sampler'], 
                                      f_plot_param_corner = fit_dict['f_plot_param_corner'],
                                      f_plot_bestfit = fit_dict['f_plot_bestfit'],
                                      f_results= fit_dict['f_mcmc_results'],
                                      f_chain_ascii = fit_dict['f_chain_ascii'])
            if fit_dict['oversampled_chisq']:
                results.oversample_factor_chisq = gal.data.oversample_factor_chisq
        
        # Do all analysis, plotting, saving:
        results.analyze_plot_save_results(gal,                           
                      blob_name=fit_dict['blob_name'], 
                      linked_posterior_names=fit_dict['linked_posterior_names'], 
                      model_key_re=fit_dict['model_key_re'], 
                      model_key_halo=fit_dict['model_key_halo'], 
                      oversample=fit_dict['oversample'], 
                      oversize=fit_dict['oversize'], 
                      fitdispersion=fit_dict['fitdispersion'], 
                      f_model=fit_dict['f_model'], 
                      f_vel_ascii = fit_dict['f_vel_ascii'], 
                      save_data=True, 
                      save_bestfit_cube=False,
                      do_plotting = True)
        
        # Reload fitting stuff to get the updated gal object
        gal, results = fitting.reload_all_fitting(filename_galmodel=fit_dict['f_model'], 
                                    filename_mcmc_results=fit_dict['f_mcmc_results'])
                                    
        # Save results
        utils_io.save_results_ascii_files(fit_results=results, gal=gal, params=params)
                                   
    elif params['fit_method'] == 'mpfit':
        galtmp, fit_dict = setup_single_object_2D(params=params, data=data)
        
        # reload results:
        gal, results = fitting.reload_all_fitting_mpfit(filename_galmodel=fit_dict['f_model'], 
                                    filename_results=fit_dict['f_results'])
        # Don't reanalyze anything...
    else:
        raise ValueError(
            '{} not accepted as a fitting method. Please only use "mcmc" or "mpfit"'.format(
                params['fit_method']))
    
    
    # Plot multid, if enabled:
    if 'fdata_1d' in params.keys():
        fw_plotting.plot_results_multid(param_filename=param_filename, fit_ndim=2, show_1d_apers=True, remove_shift=True)

    return None

    
    
def setup_single_object_2D(params=None, data=None):
    # ------------------------------------------------------------
    # Setup galaxy, instrument, model:
    
    gal = setup_gal_inst_mod_2D(params=params)
    
    # ------------------------------------------------------------
    # Load data:
    if data is None:
        gal.data = utils_io.load_single_object_2D_data(params=params)
    else:
        gal.data = data
    
    # ------------------------------------------------------------
    # Setup fitting dict:
    fit_dict = utils_io.setup_fit_dict(params=params)
    
    return gal, fit_dict
    
    
    
def setup_gal_inst_mod_2D(params=None):
    # ------------------------------------------------------------
    # Initialize the Galaxy, Instrument, and Model Set
    gal = galaxy.Galaxy(z=params['z'], name=params['galID'])
    mod_set = models.ModelSet()
    inst = instrument.Instrument()
    
    # ------------------------------------------------------------
    # Baryonic Component: Combined Disk+Bulge
    total_mass =  params['total_mass']        # log M_sun
    bt =          params['bt']                # Bulge-Total ratio
    r_eff_disk =  params['r_eff_disk']        # kpc
    n_disk =      params['n_disk']                
    invq_disk =   params['invq_disk']         # 1/q0, disk
    r_eff_bulge = params['r_eff_bulge']       # kpc
    n_bulge =     params['n_bulge']           
    invq_bulge =  params['invq_bulge']        
    noord_flat =  params['noord_flat']        # Switch for applying Noordermeer flattening
    
    # Fix components
    bary_fixed = {'total_mass': params['total_mass_fixed'],
                  'r_eff_disk': params['r_eff_disk_fixed'],
                  'n_disk': params['n_disk_fixed'],
                  'r_eff_bulge': params['r_eff_bulge_fixed'],
                  'n_bulge': params['n_bulge_fixed'],
                  'bt': params['bt_fixed']}
    
    # Set bounds
    bary_bounds = {'total_mass': (params['total_mass_bounds'][0], params['total_mass_bounds'][1]),
                   'r_eff_disk': (params['r_eff_disk_bounds'][0], params['r_eff_disk_bounds'][1]),
                   'n_disk':     (params['n_disk_bounds'][0], params['n_disk_bounds'][1]),
                   'r_eff_bulge': (params['r_eff_bulge_bounds'][0], params['r_eff_bulge_bounds'][1]),
                   'n_bulge': (params['n_bulge_bounds'][0], params['n_bulge_bounds'][1]),
                   'bt':         (params['bt_bounds'][0], params['bt_bounds'][1])}
                   
    
    bary = models.DiskBulge(total_mass=total_mass, bt=bt,
                            r_eff_disk=r_eff_disk, n_disk=n_disk,
                            invq_disk=invq_disk,
                            r_eff_bulge=r_eff_bulge, n_bulge=n_bulge,
                            invq_bulge=invq_bulge,
                            noord_flat=noord_flat,
                            name='disk+bulge',
                            fixed=bary_fixed, bounds=bary_bounds)
                            
                            
    bary = utils_io.set_comp_param_prior(comp=bary, param_name='total_mass', params=params)
    bary = utils_io.set_comp_param_prior(comp=bary, param_name='bt', params=params)
    bary = utils_io.set_comp_param_prior(comp=bary, param_name='r_eff_disk', params=params)
    bary = utils_io.set_comp_param_prior(comp=bary, param_name='n_disk', params=params)
    bary = utils_io.set_comp_param_prior(comp=bary, param_name='r_eff_bulge', params=params)
    bary = utils_io.set_comp_param_prior(comp=bary, param_name='n_bulge', params=params)
    
    # ------------------------------------------------------------
    # Halo Component: (if added)
    # ------------------------------------------------------------
    if params['include_halo']:
        # Halo component
        if (params['halo_profile_type'].strip().upper() == 'NFW'):

            # NFW halo fit:
            mvirial =                   params['mvirial']  
            conc =                      params['halo_conc']
            fdm =                       params['fdm']

            halo_fixed = {'mvirial':    params['mvirial_fixed'], 
                          'conc':       params['halo_conc_fixed'], 
                          'fdm':        params['fdm_fixed']}

            halo_bounds = {'mvirial':   (params['mvirial_bounds'][0], params['mvirial_bounds'][1]),
                           'conc':      (params['halo_conc_bounds'][0], params['halo_conc_bounds'][1]), 
                           'fdm':       (params['fdm_bounds'][0], params['fdm_bounds'][1])}

            halo = models.NFW(mvirial=mvirial, conc=conc, fdm=fdm, z=gal.z, 
                              fixed=halo_fixed, bounds=halo_bounds, name='halo')

            halo = utils_io.set_comp_param_prior(comp=halo, param_name='mvirial', params=params)
            halo = utils_io.set_comp_param_prior(comp=halo, param_name='halo_conc', params=params)

            if (params['fdm_fixed'] is False):
                # Tie the virial mass to fDM
                halo.mvirial.tied = utils_io.tie_lmvirial_NFW
                halo.mvirial.fixed = False
                halo = utils_io.set_comp_param_prior(comp=halo, param_name='fdm', params=params)
            else:
                if params['mvirial_fixed'] is False:
                    # Tie fDM to the virial mass
                    halo.fdm.tied = utils_io.tie_fdm
                    halo.fdm.fixed = False
            #
            if 'fdm_tied' in params.keys():
                if params['fdm_tied']:
                    # Tie fDM to the virial mass
                    halo.fdm.tied = utils_io.tie_fdm
                    halo.fdm.fixed = False
            #
            if 'mvirial_tied' in params.keys():
                if params['mvirial_tied']:
                    # Tie the virial mass to fDM
                    halo.mvirial.tied = utils_io.tie_lmvirial_NFW
                    halo.mvirial.fixed = False
                    
        elif (params['halo_profile_type'].strip().upper() == 'TWOPOWERHALO'):
            # Two-power halo fit:
            
            # Add values needed:
            bary.lmstar = params['lmstar']
            bary.fgas =  params['fgas']
            bary.mhalo_relation = params['mhalo_relation']
            bary.truncate_lmstar_halo = params['truncate_lmstar_halo']

            # Setup parameters:
            mvirial =  params['mvirial']
            conc =     params['halo_conc']
            alpha =    params['alpha']
            beta =     params['beta']
            fdm =      params['fdm']

            halo_fixed = {'mvirial':    params['mvirial_fixed'],
                          'conc':       params['halo_conc_fixed'],
                          'alpha':      params['alpha_fixed'],
                          'beta':       params['beta_fixed'],
                          'fdm':        params['fdm_fixed']}

            halo_bounds = {'mvirial':   (params['mvirial_bounds'][0], params['mvirial_bounds'][1]), 
                           'conc':      (params['halo_conc_bounds'][0], params['halo_conc_bounds'][1]), 
                           'alpha':     (params['alpha_bounds'][0], params['alpha_bounds'][1]), 
                           'beta':      (params['beta_bounds'][0], params['beta_bounds'][1]),
                           'fdm':       (params['fdm_bounds'][0], params['fdm_bounds'][1]) } 

            halo = models.TwoPowerHalo(mvirial=mvirial, conc=conc, 
                                alpha=alpha, beta=beta, fdm=fdm, z=gal.z, 
                                fixed=halo_fixed, bounds=halo_bounds, name='halo')

            # Tie the virial mass to Mstar
            if params['mvirial_tied']:
                halo.mvirial.tied = utils_io.tied_mhalo_mstar

            halo = utils_io.set_comp_param_prior(comp=halo, param_name='mvirial', params=params)
            halo = utils_io.set_comp_param_prior(comp=halo, param_name='halo_conc', params=params)
            halo = utils_io.set_comp_param_prior(comp=halo, param_name='alpha', params=params)
            halo = utils_io.set_comp_param_prior(comp=halo, param_name='beta', params=params)

            if params['fdm_fixed'] is False:
                # Tie the virial mass to fDM
                halo.alpha.tied = utils_io.tie_alpha_TwoPower
                halo = utils_io.set_comp_param_prior(comp=halo, param_name='fdm', params=params)

        elif (params['halo_profile_type'].strip().upper() == 'BURKERT'):
            # Burkert halo profile:
            # Add values needed:
            bary.lmstar = params['lmstar']
            bary.fgas =  params['fgas']
            bary.mhalo_relation = params['mhalo_relation']
            bary.truncate_lmstar_halo = params['truncate_lmstar_halo']
            
            # Setup parameters:
            mvirial =  params['mvirial']
            rB =       params['rB']
            fdm =      params['fdm']

            halo_fixed = {'mvirial':    params['mvirial_fixed'],
                          'rB':         params['rB_fixed'],
                          'fdm':        params['fdm_fixed']}

            halo_bounds = {'mvirial':   (params['mvirial_bounds'][0], params['mvirial_bounds'][1]), 
                           'rB':        (params['rB_bounds'][0], params['rB_bounds'][1]),
                           'fdm':       (params['fdm_bounds'][0], params['fdm_bounds'][1]) } 

            halo = models.Burkert(mvirial=mvirial, rB=rB, fdm=fdm, z=gal.z, 
                              fixed=halo_fixed, bounds=halo_bounds, name='halo')

            # Tie the virial mass to Mstar
            if params['mvirial_tied']:
                halo.mvirial.tied = utils_io.tied_mhalo_mstar

            halo = utils_io.set_comp_param_prior(comp=halo, param_name='mvirial', params=params)
            halo = utils_io.set_comp_param_prior(comp=halo, param_name='rB', params=params)

            if params['fdm_fixed'] is False:
                # Tie the virial mass to fDM
                halo.rB.tied = utils_io.tie_rB_Burkert
                halo = utils_io.set_comp_param_prior(comp=halo, param_name='fdm', params=params)
                
        elif (params['halo_profile_type'].strip().upper() == 'EINASTO'):
            # Einastro halo profile:
            # Add values needed:
            bary.lmstar = params['lmstar']
            bary.fgas =  params['fgas']
            bary.mhalo_relation = params['mhalo_relation']
            bary.truncate_lmstar_halo = params['truncate_lmstar_halo']
            
            # Setup parameters:
            mvirial =           params['mvirial']
            fdm =               params['fdm']
            conc =              params['conc']
            
            halo_fixed = {'mvirial':        params['mvirial_fixed'],
                          'conc':           params['halo_conc_fixed'],
                          'fdm':            params['fdm_fixed']}
                          
            halo_bounds = {'mvirial':       (params['mvirial_bounds'][0], params['mvirial_bounds'][1]), 
                           'conc':          (params['halo_conc_bounds'][0], params['halo_conc_bounds'][1]),
                           'fdm':           (params['fdm_bounds'][0], params['fdm_bounds'][1]) }
            
            if 'alphaEinasto' in params.keys():
                alphaEinasto =                  params['alphaEinasto']
                halo_fixed['alphaEinasto'] =    params['alphaEinasto_fixed']
                halo_bounds['alphaEinasto'] =   (params['alphaEinasto_bounds'][0], 
                                                 params['alphaEinasto_bounds'][1])
                halo = models.Einasto(mvirial=mvirial, alphaEinasto=alphaEinasto, conc=conc, fdm=fdm, z=gal.z, 
                              fixed=halo_fixed, bounds=halo_bounds, name='halo')
            elif 'nEinasto' in params.keys():
                nEinasto =                  params['nEinasto']
                halo_fixed['nEinasto'] =    params['nEinasto_fixed']
                halo_bounds['nEinasto'] =   (params['nEinasto_bounds'][0], params['nEinasto_bounds'][1])
                halo = models.Einasto(mvirial=mvirial, nEinasto=nEinasto, conc=conc, fdm=fdm, z=gal.z, 
                              fixed=halo_fixed, bounds=halo_bounds, name='halo')
                              
            # Tie the virial mass to Mstar
            if params['mvirial_tied']:
                halo.mvirial.tied = utils_io.tied_mhalo_mstar
            
            halo = utils_io.set_comp_param_prior(comp=halo, param_name='mvirial', params=params)
            halo = utils_io.set_comp_param_prior(comp=halo, param_name='rB', params=params)
            
            if params['fdm_fixed'] is False:
                # Tie the virial mass to fDM
                if 'alphaEinasto' in params.keys():
                    halo.alphaEinasto.tied = utils_io.tie_alphaEinasto_Einasto
                elif 'nEinasto' in params.keys():
                    halo.alphaEinasto.tied = utils_io.tie_nEinasto_Einasto
                halo = utils_io.set_comp_param_prior(comp=halo, param_name='fdm', params=params)
            
        else:
            raise ValueError("{} halo profile type not recognized!".format(params['halo_profile_type']))
    # ------------------------------------------------------------
        
        
    # ------------------------------------------------------------
    # Dispersion profile
    sigma0 = params['sigma0']       # km/s
    disp_fixed = {'sigma0': params['sigma0_fixed']}
    disp_bounds = {'sigma0': (params['sigma0_bounds'][0], params['sigma0_bounds'][1])}   
    
    disp_prof = models.DispersionConst(sigma0=sigma0, fixed=disp_fixed, bounds=disp_bounds, name='dispprof')
                                              
    disp_prof = utils_io.set_comp_param_prior(comp=disp_prof, param_name='sigma0', params=params)
    
    # ------------------------------------------------------------
    # z-height profile
    sigmaz = params['sigmaz']      # kpc
    zheight_fixed = {'sigmaz': params['sigmaz_fixed']}
    zheight_bounds = {'sigmaz': (params['sigmaz_bounds'][0], params['sigmaz_bounds'][1])}   
    
    zheight_prof = models.ZHeightGauss(sigmaz=sigmaz, name='zheightgaus',
                                       fixed=zheight_fixed, bounds=zheight_bounds)
    if params['zheight_tied']:
        zheight_prof.sigmaz.tied = utils_io.tie_sigz_reff
    else:
        # Do prior changes away from default flat prior, if so specified:
        zheight_prof = utils_io.set_comp_param_prior(comp=zheight_prof, param_name='sigmaz', params=params)
        
    # --------------------------------------
    # Geometry
    
    inc = params['inc']                # degrees
    pa =  params['pa']                 # default convention; neg r is blue side
        
    xshift = params['xshift']          # pixels from center
    yshift = params['yshift']          # pixels from center
    vel_shift = params['vel_shift']    # km/s ; systemic vel
    
    geom_fixed = {'inc': params['inc_fixed'],
                  'pa': params['pa_fixed'],
                  'xshift': params['xshift_fixed'],
                  'yshift': params['yshift_fixed'], 
                  'vel_shift': params['vel_shift_fixed']}
                  
    geom_bounds = {'inc':  (params['inc_bounds'][0], params['inc_bounds'][1]),
                   'pa':  (params['pa_bounds'][0], params['pa_bounds'][1]),
                   'xshift':  (params['xshift_bounds'][0], params['xshift_bounds'][1]),
                   'yshift':  (params['yshift_bounds'][0], params['yshift_bounds'][1]), 
                   'vel_shift': (params['vel_shift_bounds'][0], params['vel_shift_bounds'][1])}
    
    geom = models.Geometry(inc=inc, pa=pa, xshift=xshift, yshift=yshift, vel_shift=vel_shift, 
                           fixed=geom_fixed, bounds=geom_bounds, name='geom')
    geom = utils_io.set_comp_param_prior(comp=geom, param_name='inc', params=params)
    geom = utils_io.set_comp_param_prior(comp=geom, param_name='pa', params=params)
    geom = utils_io.set_comp_param_prior(comp=geom, param_name='xshift', params=params)
    geom = utils_io.set_comp_param_prior(comp=geom, param_name='yshift', params=params)
    geom = utils_io.set_comp_param_prior(comp=geom, param_name='vel_shift', params=params)
                               
    
    
    # --------------------------------------
    # Add all of the model components to the ModelSet
    mod_set.add_component(bary, light=True)
    if params['include_halo']:
        mod_set.add_component(halo)
    mod_set.add_component(disp_prof)
    mod_set.add_component(zheight_prof)
    mod_set.add_component(geom)
    
    # --------------------------------------
    # Set some kinematic options for calculating the velocity profile
    mod_set.kinematic_options.adiabatic_contract = params['adiabatic_contract']
    mod_set.kinematic_options.pressure_support = params['pressure_support']
    
    
    # --------------------------------------
    # Set up the instrument
    pixscale = params['pixscale']*u.arcsec                # arcsec/pixel
    fov = [params['fov_npix'], params['fov_npix']]        # (nx, ny) pixels
    spec_type = params['spec_type']                       # 'velocity' or 'wavelength'
    if spec_type.strip().lower() == 'velocity':
        spec_start = params['spec_start']*u.km/u.s        # Starting value of spectrum
        spec_step = params['spec_step']*u.km/u.s          # Spectral step
    else:
        raise ValueError("not implemented for wavelength yet!")
    nspec = params['nspec']                               # Number of spectral pixels

    if params['psf_type'].lower().strip() == 'gaussian':
        if 'psf_major' in params.keys():
            
            beam_major = params['psf_major']*u.arcsec              # FWHM of beam, Gaussian
            try:
                beam_minor  = params['psf_minor']*u.arcsec
            except:
                beam_minor = beam_major

            try:
                beam_pa = params['psf_pa']*u.deg
            except:
                beam_pa = 0*u.deg
        else:
            beam_major = params['psf_fwhm']*u.arcsec              # FWHM of beam, Gaussian
            beam_minor = beam_major
            beam_pa = 0.*u.deg

        beam = instrument.GaussianBeam(major=beam_major, minor=beam_minor, pa=beam_pa)

    elif params['psf_type'].lower().strip() == 'moffat':
        beamsize = params['psf_fwhm']*u.arcsec              # FWHM of beam, Moffat
        beta = params['psf_beta']
        beam = instrument.Moffat(major_fwhm=beamsize, beta=beta)
    elif params['psf_type'].lower().strip() == 'doublegaussian':
        # Kernel of both components multipled by: self._scaleN / np.sum(kernelN.array)
        #    -- eg, scaleN controls the relative amount of flux in each component.
        
        beamsize1 = params['psf_fwhm1']*u.arcsec              # FWHM of beam, Gaussian
        beamsize2 = params['psf_fwhm2']*u.arcsec              # FWHM of beam, Gaussian
        
        try:
            scale1 = params['psf_scale1']                     # Flux scaling of component 1
        except:
            scale1 = 1.                                       # If ommitted, assume scale2 is rel to scale1=1.
        scale2 = params['psf_scale2']                         # Flux scaling of component 2

        try:
            theta1 = params['psf_theta1']
        except:
            theta1 = 0*u.deg

        try:
            theta2 = params['psf_theta2']
        except:
            theta2 = theta1
        
        beam = instrument.DoubleBeam(major1=beamsize1, major2=beamsize2, 
                        scale1=scale1, scale2=scale2, pa1=theta1, pa2=theta2)
    
    else:
        raise ValueError("PSF type {} not recognized!".format(params['psf_type']))

    if params['use_lsf']:
        sig_inst = params['sig_inst_res']*u.km/u.s          # Instrumental spectral resolution  [km/s]
        lsf = instrument.LSF(sig_inst)
        inst.lsf = lsf
        inst.set_lsf_kernel()

    inst.beam = beam
    inst.pixscale = pixscale
    inst.fov = fov
    inst.spec_type = spec_type
    inst.spec_step = spec_step
    inst.spec_start = spec_start
    inst.nspec = nspec
    
    # Set the beam kernel so it doesn't have to be calculated every step
    inst.set_beam_kernel(support_scaling=12.)   # ORIGINAL: support_scaling=8.

    
    # Add the model set and instrument to the Galaxy
    gal.model = mod_set
    gal.instrument = inst
    
    
    
    return gal
    
    


if __name__ == "__main__":
    
    param_filename = sys.argv[1]
    
    try:
        if sys.argv[2].strip().lower() == 'reanalyze':
            reanalyze = True
        else:
            reanalyze = False
    except:
        reanalyze = False
        
    if reanalyze:
        dysmalpy_reanalyze_single_2D(param_filename=param_filename)
    else:
        dysmalpy_fit_single_2D(param_filename=param_filename)
    
    




