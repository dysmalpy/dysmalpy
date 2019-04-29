# Script to fit single object in 3D with Dysmalpy

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

import copy
import numpy as np
import pandas as pd
import astropy.units as u

import utils_io



def user_specific_load_3D_data(param_filename=None):
    # EDIT THIS FILE TO HAVE SPECIFIC LOADING OF DATA!
    
    params = utils_io.read_fitting_params(fname=param_filename)
    
    
    # Recommended to trim cube to around the relevant line only, 
    # both for speed of computation and to avoid noisy spectral resolution elements.
    
    FNAME_CUBE = None
    FNAME_ERR = None
    FNAME_MASK = None
    FNAME_MASK_SKY = None       # sky plane mask -- eg, mask areas away from galaxy.
    FNAME_MASK_SPEC = None      # spectral dim masking -- eg, mask a skyline. 
                                #  ** When trimming cube mind that masks need to be appropriately trimmed too.
    
    # Optional: set RA/Dec of reference pixel in the cube: mind trimming....
    ref_pixel = None
    ra = None       
    dec = None      
    
    pixscale=params['pixscale']
    
    # +++++++++++++++++++++++++++++++++++++++++++
    # Upload the data set to be fit
    cube = fits.getdata(FNAME_CUBE)
    err_cube = fits.getdata(FNAME_ERR)
    mask_sky = fits.getdata(FNAME_MASK_SKY)
    mask_spec = fits.getdata(FNAME_MASK_SPEC)
    
    spec_type = 'velocity'    # either 'velocity' or 'wavelength'
    spec_arr = None           # Needs to be array of vel / wavelength for the spectral dim of cube. 
                              #  1D arr. Length must be length of spectral dim of cube.
    spec_unit = u.km/u.s      # EXAMPLE! set as needed.

    # Auto mask some bad data
    if automask: 
        # Add settings here: S/N ??
        
        pass
        
    data3d = data_classes.Data3D(cube, pixscale, spec_type, spec_arr, 
                            err_cube = err_cube, mask_sky=mask_sky, mask_spec=mask_spec, 
                            estimate_err=False, error_frac=0.2, ra=ra, dec=dec,
                             ref_pixel=ref_pixel, spec_unit=spec_unit)

    return data3d


#
def dysmalpy_fit_single_3D_wrapper(param_filename=None):
    
    data3d = user_specific_load_3D_data(param_filename=param_filename)
    
    dysmalpy_fit_single_3D(param_filename=param_filename, data=data3d)
    
    return None

def dysmalpy_fit_single_3D(param_filename=None, data=None):
    
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
        # Copy paramfile into outdir for posterity:
        #os.system('cp {} {}'.format(param_filename, outdir))

        # Copy paramfile that is OS independent
        shutil.copy(param_filename, outdir)
        
        #######################
        # Setup
        gal, fit_dict = setup_single_object_3D(params=params, data=data)

        # Clean up existing log file:
        if os.path.isfile(fit_dict['f_log']):
            os.remove(fit_dict['f_log'])

        # Fit
        if fit_dict['fit_method'] == 'mcmc':
            results = fitting.fit(gal, nWalkers=fit_dict['nWalkers'], nCPUs=fit_dict['nCPUs'],
                                  scale_param_a=fit_dict['scale_param_a'], nBurn=fit_dict['nBurn'],
                                  nSteps=fit_dict['nSteps'], minAF=fit_dict['minAF'],
                                  maxAF=fit_dict['maxAF'],
                                  nEff=fit_dict['nEff'], do_plotting=fit_dict['do_plotting'],
                                  red_chisq=fit_dict['red_chisq'],
                                  oversample=fit_dict['oversample'],
                                  fitdispersion=fit_dict['fitdispersion'],
                                  compute_dm=fit_dict['compute_dm'],
                                  linked_posterior_names=fit_dict['linked_posterior_names'],
                                  outdir=fit_dict['outdir'],
                                  f_plot_trace_burnin=fit_dict['f_plot_trace_burnin'],
                                  f_plot_trace=fit_dict['f_plot_trace'],
                                  f_model=fit_dict['f_model'],
                                  f_cube=fit_dict['f_cube'],
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
                                        f_log=fit_dict['f_log'])

        # Save results
        utils_io.save_results_ascii_files(fit_results=results, gal=gal, params=params)
    
    return None
    
    
def setup_single_object_3D(params=None, data=None):
    # ------------------------------------------------------------
    # Setup galaxy, instrument, model:
    
    gal = setup_gal_inst_mod_3D(params=params)
    
    # ------------------------------------------------------------
    # Load data:
    if data is None:
        gal.data = utils_io.load_single_object_3D_data(params=params)
    else:
        gal.data = data
    
    # ------------------------------------------------------------
    # Setup fitting dict:
    fit_dict = utils_io.setup_fit_dict(params=params)
    
    return gal, fit_dict
    
    
    
def setup_gal_inst_mod_3D(params=None):
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
    
    if params['include_halo']:
        # Halo component
        mvirial = params['mvirial']     # log Msun
        conc = params['halo_conc']      
        
        halo_fixed = {'mvirial': params['mvirial_fixed'],       
                      'conc': params['halo_conc_fixed']}       
                      
        halo_bounds = {'mvirial': (params['mvirial_bounds'][0], params['mvirial_bounds'][1]),    
                       'conc': (params['halo_conc_bounds'][0], params['halo_conc_bounds'][1])}  
                       
        halo = models.NFW(mvirial=mvirial, conc=conc, z=gal.z,
                          fixed=halo_fixed, bounds=halo_bounds, name='halo')
        
        halo = utils_io.set_comp_param_prior(comp=halo, param_name='mvirial', params=params)
        halo = utils_io.set_comp_param_prior(comp=halo, param_name='halo_conc', params=params)
        
        
        
        
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
    
    geom = models.Geometry(inc=inc, pa=pa, xshift=xshift, yshift=yshift,
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
        beam_major = params['psf_major']*u.arcsec              # FWHM of beam, Gaussian
        try:
            beam_minor  = params['psf_minor']*u.arcsec
        except:
            beam_minor = beam_major

        try:
            beam_pa = params['psf_pa']*u.deg
        except:
            beam_pa = 0*u.deg

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
    
    dysmalpy_fit_single_3D_wrapper(param_filename=param_filename)
    
    




