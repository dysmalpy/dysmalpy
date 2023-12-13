from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import dysmalpy
from dysmalpy import galaxy
from dysmalpy import models
from dysmalpy import fitting
from dysmalpy import observation
from dysmalpy import instrument
from dysmalpy import data_classes
from dysmalpy import parameters
# from dysmalpy import plotting
from dysmalpy import aperture_classes
from dysmalpy.fitting_wrappers import tied_functions 
from dysmalpy import config

import os
# import copy

import numpy as np
import astropy.units as u
# import astropy.io.fits as fits




import logging
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)


# #### Set data, output paths ####

# Data directory
### CHANGE TO SPECIFICS FOR YOUR TEST:
# data_dir = '/afs/mpe.mpg.de/home/sedona/dysmalpy_example_data/'
# outdir = '/afs/mpe.mpg.de/home/sedona/JUPYTER_EXAMPLES/JUPYTER_OUTPUT_1D/'

# Data directory
dir_path = os.path.abspath(fitting.__path__[0])
data_dir = os.sep.join([os.sep.join(dir_path.split(os.sep)[:-1]),'tests', 'test_data', ''])
#'/YOUR/DATA/PATH/'   
outdir = '/Users/sedona/data/dysmalpy_test_examples/JUPYTER_OUTPUT_1D/'

# ----------


def run_dynesty_full_1D():
    # ## Initialize galaxy, model set, instrument ##
    
    gal = galaxy.Galaxy(z=1.613, name='GS4_43501')
    mod_set = models.ModelSet()
    
    
    # ### Baryonic component: Combined Disk+Bulge ###
    
    total_mass = 11.0    # M_sun
    bt = 0.3             # Bulge-Total ratio
    r_eff_disk = 5.0     # kpc
    n_disk = 1.0
    invq_disk = 5.0
    r_eff_bulge = 1.0    # kpc
    n_bulge = 4.0
    invq_bulge = 1.0
    noord_flat = True    # Switch for applying Noordermeer flattening

    # Fix components
    bary_fixed = {'total_mass': False,
                  'r_eff_disk': False, #True,
                  'n_disk': True,
                  'r_eff_bulge': True,
                  'n_bulge': True,
                  'bt': True}

    # Set bounds
    bary_bounds = {'total_mass': (10, 13),
                   'r_eff_disk': (1.0, 30.0),
                   'n_disk': (1, 8),
                   'r_eff_bulge': (1, 5),
                   'n_bulge': (1, 8),
                   'bt': (0, 1)}

    bary = models.DiskBulge(total_mass=total_mass, bt=bt,
                            r_eff_disk=r_eff_disk, n_disk=n_disk,
                            invq_disk=invq_disk,
                            r_eff_bulge=r_eff_bulge, n_bulge=n_bulge,
                            invq_bulge=invq_bulge,
                            noord_flat=noord_flat,
                            name='disk+bulge',
                            fixed=bary_fixed, bounds=bary_bounds)

    bary.r_eff_disk.prior = parameters.BoundedGaussianPrior(center=5.0, stddev=1.0)


    # ### Halo component ###
    
    mvirial = 12.0
    conc = 5.0
    fdm = 0.5

    halo_fixed = {'mvirial': False,
                  'conc': True, 
                  'fdm':  False}
    # Mvirial will be tied -- so must set 'fixed=False' for Mvirial...

    halo_bounds = {'mvirial': (10, 13),
                   'conc': (1, 20),
                   'fdm': (0, 1)}

    halo = models.NFW(mvirial=mvirial, conc=conc, fdm=fdm, z=gal.z,
                      fixed=halo_fixed, bounds=halo_bounds, name='halo')


    halo.mvirial.tied = tied_functions.tie_lmvirial_NFW


    # ### Dispersion profile ###
    
    sigma0 = 39.   # km/s
    disp_fixed = {'sigma0': False}
    disp_bounds = {'sigma0': (5, 300)}

    disp_prof = models.DispersionConst(sigma0=sigma0, fixed=disp_fixed,
                                       bounds=disp_bounds, name='dispprof', tracer='halpha')


    # ### z-height profile ###
    
    sigmaz = 0.9   # kpc
    zheight_fixed = {'sigmaz': False}

    zheight_prof = models.ZHeightGauss(sigmaz=sigmaz, name='zheightgaus',
                                       fixed=zheight_fixed)
    zheight_prof.sigmaz.tied = tied_functions.tie_sigz_reff


    # ### Geometry ###
    
    inc = 62.     # degrees
    pa = 142.     # degrees, blue-shifted side CCW from north
    xshift = 0    # pixels from center
    yshift = 0    # pixels from center

    geom_fixed = {'inc': True,
                  'pa': True,
                  'xshift': True,
                  'yshift': True}

    geom_bounds = {'inc': (0, 90),
                   'pa': (90, 180),
                   'xshift': (0, 4),
                   'yshift': (-10, -4)}

    geom = models.Geometry(inc=inc, pa=pa, xshift=xshift, yshift=yshift,
                           fixed=geom_fixed, bounds=geom_bounds, name='geom', 
                           obs_name='halpha_1D')


    # ## Add all model components to ModelSet ##

    # Add all of the model components to the ModelSet
    mod_set.add_component(bary, light=True)
    mod_set.add_component(halo)
    mod_set.add_component(disp_prof)
    mod_set.add_component(zheight_prof)
    mod_set.add_component(geom)


    # ### Set kinematic options for calculating velocity profile ###
    
    mod_set.kinematic_options.adiabatic_contract = False
    mod_set.kinematic_options.pressure_support = True


    # ### Set up the observation and instrument ###
    obs = observation.Observation(name='halpha_1D', tracer='halpha')
    inst = instrument.Instrument()

    beamsize = 0.55*u.arcsec                 # FWHM of beam
    sig_inst = 45*u.km/u.s                   # Instrumental spectral resolution

    beam = instrument.GaussianBeam(major=beamsize)
    lsf = instrument.LSF(sig_inst)

    inst.beam = beam
    inst.lsf = lsf
    inst.pixscale = 0.125*u.arcsec           # arcsec/pixel
    inst.fov = [33, 33]                      # (nx, ny) pixels
    inst.spec_type = 'velocity'              # 'velocity' or 'wavelength'
    inst.spec_step = 10*u.km/u.s             # Spectral step
    inst.spec_start = -1000*u.km/u.s         # Starting value of spectrum
    inst.nspec = 201                         # Number of spectral pixels

    # Extraction information
    inst.ndim = 1                            # Dimensionality of data
    inst.moment = False                      # For 1D/2D data, if True then velocities and dispersion calculated from moments
                                             # Default is False, meaning Gaussian extraction used

    # Set the beam kernel so it doesn't have to be calculated every step
    inst.set_beam_kernel()
    inst.set_lsf_kernel()


    # Add instrument to observation
    obs.instrument = inst



    # ## Load data ##

    # * Load the data from file:
    #   - *1D velocity, dispersion profiles and error*
    #   - *A mask can be loaded / created as well*
    #   
    # * Put data in `Data1D` class
    # 
    # * Add data to Galaxy object

    f_data_1d = data_dir+'GS4_43501.obs_prof.txt'
    dat_arr = np.loadtxt(f_data_1d)
    gs4_r = dat_arr[:,0]
    gs4_vel = dat_arr[:,1]
    gs4_disp = dat_arr[:,3]
    err_vel = dat_arr[:,2]
    err_disp = dat_arr[:,4]
    inst_corr = True                  # Flag for if the measured dispersion has been
                                      # corrected for instrumental resolution
    
    # Put data in Data1D data class: 
    data1d = data_classes.Data1D(r=gs4_r, velocity=gs4_vel,
                                    vel_disp=gs4_disp, vel_err=err_vel,
                                    vel_disp_err=err_disp, inst_corr=inst_corr, 
                                    filename_velocity=f_data_1d)


    # Add data to Observation:
    obs.data = data1d

    # #### Setup apertures:
    # Setup apertures: circular apertures placed on the cube for GS4_43501.

    profile1d_type = 'circ_ap_cube'    # Extraction in circular apertures placed on the cube

    aperture_radius = 0.5 * obs.instrument.beam.major.value

    obs.instrument.apertures = aperture_classes.setup_aperture_types(obs=obs, 
                    profile1d_type=profile1d_type, 
                    aper_centers=obs.data.rarr,
                    slit_pa=pa, slit_width=beamsize.value,
                    aperture_radius=aperture_radius, 
                    partial_weight=True)

    # Define model, fit options:
    obs.mod_options.oversample = 1  
    # Factor by which to oversample model (eg, subpixels)

    obs.fit_options.fit = True             # Include this observation in the fit (T/F)
    obs.fit_options.fit_velocity = True    # 1D/2D: Fit velocity of observation (T/F)
    obs.fit_options.fit_dispersion = True  # 1D/2D: Fit dispersion of observation (T/F)
    obs.fit_options.fit_flux = False       # 1D/2D: Fit flux of observation (T/F)



    # ## Add the model set, observation  to the Galaxy ##
    gal.model = mod_set
    gal.add_observation(obs)
    

    # -----------------

    # # DYNESTY Fitting #

    # ### DYNESTY fitting parameters ###

    # Set parameters for fitting: 
    #    - Passing options to `emcee`
    #    - Other calculation options
    
    # Options passed to Nested Sampler
    ## FULL TEST
    sample = 'rwalk'
    maxiter = None
    nlive_init = 1000
    nlive_batch = 1000
    # nCPUs = 190
    nCPUs = 8

    # Other options
    blob_name = 'mvirial'    # Also save 'blob' values of Mvirial, calculated at every chain step
    
    # Output directory
    outdir_dynesty_full = outdir + 'Dynesty_full_run_maxiter{}_nliveinit{}_nlivebatch{}/'.format( 
                str(maxiter), nlive_init, nlive_batch)


    # Output options: 
    do_plotting = True  
    plot_type = 'pdf'
    overwrite = True

    output_options = config.OutputOptions(outdir=outdir_dynesty_full, 
                                        do_plotting=do_plotting, 
                                        plot_type=plot_type,
                                        overwrite=overwrite)

    # -------

    # ## Run `Dysmalpy` fitting: DYNESTY ##

    fitter = fitting.NestedFitter(sample=sample, maxiter=maxiter, 
                               nlive_init=nlive_init, nlive_batch=nlive_batch, 
                               nCPUs=nCPUs, blob_name=blob_name)
    
    # output_options.set_output_options(gal, fitter)
    # raise ValueError

    dynesty_results = fitter.fit(gal, output_options)

    # dynesty_results
    return None

if __name__ == "__main__":
    run_dynesty_full_1D()
    
    
