# Script to fit single object in 1D with Dysmalpy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from contextlib import contextmanager

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

from . import utils_io



def dysmalpy_fit_single_1D(param_filename=None, data=None):
    
    # Read in the parameters from param_filename:
    params = utils_io.read_fitting_params(fname=param_filename)
    
    # Setup some paths:
    outdir = utils_io.ensure_path_trailing_slash(params['outdir'])
    params['outdir'] = outdir
    
    fitting.ensure_dir(params['outdir'])
    
    # Copy paramfile into outdir for posterity:
    os.system('cp {} {}'.format(param_filename, outdir))
    
    # Check if fitting already done:
    fit_exists = os.path.isfile(outdir+'{}_mcmc_results.pickle'.format(params['galID']))
    
    if fit_exists: 
        print('------------------------------------------------------------------')
        print(' Fitting already complete for: {}'.format(params['galID']))
        print("   make new output folder or remove previous fitting files")
        print('------------------------------------------------------------------')
        print(" ")
    else:
        #######################
        # Setup
        gal, mcmc_dict = setup_single_object_1D(params=params, data=data)
    
        # Clean up existing log file:
        if os.path.isfile(mcmc_dict['f_log']):
            os.remove(mcmc_dict['f_log'])
        
        # Fit
        mcmc_results = fitting.fit(gal, nWalkers=mcmc_dict['nWalkers'], nCPUs=mcmc_dict['nCPUs'],
                                    scale_param_a=mcmc_dict['scale_param_a'], nBurn=mcmc_dict['nBurn'],
                                    nSteps=mcmc_dict['nSteps'], minAF=mcmc_dict['minAF'], maxAF=mcmc_dict['maxAF'],
                                    nEff=mcmc_dict['nEff'], do_plotting=mcmc_dict['do_plotting'],
                                    red_chisq=mcmc_dict['red_chisq'],
                                    profile1d_type=mcmc_dict['profile1d_type'],
                                    oversample=mcmc_dict['oversample'], 
                                    fitdispersion=mcmc_dict['fitdispersion'], 
                                    out_dir=mcmc_dict['outdir'],
                                    f_plot_trace_burnin=mcmc_dict['f_plot_trace_burnin'],
                                    f_plot_trace=mcmc_dict['f_plot_trace'],
                                    f_model=mcmc_dict['f_model'],
                                    f_sampler=mcmc_dict['f_sampler'],
                                    f_burn_sampler=mcmc_dict['f_burn_sampler'],
                                    f_plot_param_corner=mcmc_dict['f_plot_param_corner'],
                                    f_plot_bestfit=mcmc_dict['f_plot_bestfit'],
                                    f_mcmc_results=mcmc_dict['f_mcmc_results'],
                                    f_chain_ascii=mcmc_dict['f_chain_ascii'], 
                                    f_vel_ascii=mcmc_dict['f_vel_ascii'], 
                                    f_log=mcmc_dict['f_log'])
    
    
        # Save results
        utils_io.save_results_ascii_files(mcmc_results=mcmc_results, gal=gal, params=params)
    
    
    return None
    
    
def setup_single_object_1D(params=None, data=None):
    # ------------------------------------------------------------
    # Setup galaxy, instrument, model:
    
    if (params['profile1d_type'].lower() != 'circ_ap_cube'):
        raise ValueError(" Are you really sure the data is extracted as {}? Everything tested for 'circ_ap_cube'.".format(params['profile1d_type']))
    
    gal = setup_gal_inst_mod_1D(params=params)
    
    # ------------------------------------------------------------
    # Load data:
    if data is None:
        gal.data = utils_io.load_single_object_1D_data(fdata=params['fdata'], params=params)
        gal.data.filename_velocity = params['fdata']
    else:
        gal.data = data
    
    # ------------------------------------------------------------
    # Setup fitting dict:
    mcmc_dict = utils_io.setup_mcmc_dict(params=params)
    
    return gal, mcmc_dict
    
    
    
def setup_gal_inst_mod_1D(params=None):
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
    
    inc = params['inc']       # degrees
    
    if 'pa' in list(params.keys()):
        pa =  params['pa']
    else:
        pa = params['slit_pa']  # default convention; neg r is blue side
        
    xshift = 0                  # pixels from center
    yshift = 0                  # pixels from center
    
    geom_fixed = {'inc': params['inc_fixed'],
                  'pa': True,
                  'xshift': True,
                  'yshift': True}
                  
    geom_bounds = {'inc':  (params['inc_bounds'][0], params['inc_bounds'][1]),
                   'pa':  (-180., 180.),
                   'xshift':  (-1., 1.),
                   'yshift':  (-1., 1.)}
    
    geom = models.Geometry(inc=inc, pa=pa, xshift=xshift, yshift=yshift,
                           fixed=geom_fixed, bounds=geom_bounds, name='geom')
    geom = utils_io.set_comp_param_prior(comp=geom, param_name='inc', params=params)
    geom = utils_io.set_comp_param_prior(comp=geom, param_name='pa', params=params)
    geom = utils_io.set_comp_param_prior(comp=geom, param_name='xshift', params=params)
    geom = utils_io.set_comp_param_prior(comp=geom, param_name='yshift', params=params)
                               
    
    
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
    sig_inst = params['sig_inst_res']*u.km/u.s          # Instrumental spectral resolution  [km/s]
    
    
    if params['psf_type'].lower().strip() == 'gaussian':
        beamsize = params['psf_fwhm']*u.arcsec              # FWHM of beam, Gaussian
        beam = instrument.GaussianBeam(major=beamsize)
    elif params['psf_type'].lower().strip() == 'moffat':
        beamsize = params['psf_fwhm']*u.arcsec              # FWHM of beam, Moffat
        beta = params['psf_beta']
        beam = instrument.Moffat(major_fwhm=beamsize, beta=beta)
    else:
        raise ValueError("PSF type {} not recognized!".format(params['psf_type']))
    
    lsf = instrument.LSF(sig_inst)
    
    inst.beam = beam
    inst.lsf = lsf
    inst.pixscale = pixscale
    inst.fov = fov
    inst.spec_type = spec_type
    inst.spec_step = spec_step
    inst.spec_start = spec_start
    inst.nspec = nspec
    
    # Set the beam kernel so it doesn't have to be calculated every step
    inst.set_beam_kernel(support_scaling=12.)   # ORIGINAL: support_scaling=8.
    inst.set_lsf_kernel()
    
    # Add the model set and instrument to the Galaxy
    gal.model = mod_set
    gal.instrument = inst
    
    
    
    return gal
    
    


if __name__ == "__main__":
    
    param_filename = sys.argv[1]
    
    dysmalpy_fit_single_1D(param_filename=param_filename)
    
    




