from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from dysmalpy import galaxy
from dysmalpy import models
from dysmalpy import fitting
from dysmalpy import instrument
from dysmalpy import data_classes
from dysmalpy import parameters
# from dysmalpy import plotting
from dysmalpy import aperture_classes
from dysmalpy.fitting_wrappers import utils_io

# import os
# import copy

import numpy as np
import astropy.units as u
# import astropy.io.fits as fits



# A check for compatibility:
import emcee
if np.int(emcee.__version__[0]) >= 3:
    ftype_sampler = 'h5'
else:
    ftype_sampler = 'pickle' 

import logging
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)


# #### Set data, output paths ####

# Data directory
### CHANGE TO SPECIFICS FOR YOUR TEST:
data_dir = '/afs/mpe.mpg.de/home/sedona/dysmalpy_example_data/'
outdir = '/afs/mpe.mpg.de/home/sedona/JUPYTER_EXAMPLES/JUPYTER_OUTPUT_1D/'


# -------------------

# # ##### Set function to tie scale height relative to effective radius #####
# 
# def tie_sigz_reff(model_set):
#  
#     reff = model_set.components['disk+bulge'].r_eff_disk.value
#     invq = model_set.components['disk+bulge'].invq_disk
#     sigz = 2.0*reff/invq/2.35482
# 
#     return sigz
# 
# 
# # ##### Set function to tie Mvirial to $f_{DM}(R_e)$
# 
# def tie_lmvirial_NFW(model_set):
#     comp_halo = model_set.components.__getitem__('halo')
#     comp_baryons = model_set.components.__getitem__('disk+bulge')
#     r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
#     mvirial = comp_halo.calc_mvirial_from_fdm(comp_baryons, r_fdm, 
#                     adiabatic_contract=model_set.kinematic_options.adiabatic_contract)
#     return mvirial
# 
# # - Also see **fitting_wrappers.utils_io** for more tied functions

# ----------


def run_mcmc_full_1D():
    # ## Initialize galaxy, model set, instrument ##
    
    gal = galaxy.Galaxy(z=1.613, name='GS4_43501')
    mod_set = models.ModelSet()
    inst = instrument.Instrument()
    
    
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


    halo.mvirial.tied = utils_io.tie_lmvirial_NFW


    # ### Dispersion profile ###
    
    sigma0 = 39.   # km/s
    disp_fixed = {'sigma0': False}
    disp_bounds = {'sigma0': (5, 300)}

    disp_prof = models.DispersionConst(sigma0=sigma0, fixed=disp_fixed,
                                              bounds=disp_bounds, name='dispprof')


    # ### z-height profile ###
    
    sigmaz = 0.9   # kpc
    zheight_fixed = {'sigmaz': False}

    zheight_prof = models.ZHeightGauss(sigmaz=sigmaz, name='zheightgaus',
                                       fixed=zheight_fixed)
    zheight_prof.sigmaz.tied = utils_io.tie_sigz_reff


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
                           fixed=geom_fixed, bounds=geom_bounds, name='geom')


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


    # ### Set up the instrument ###
    
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

    # Set the beam kernel so it doesn't have to be calculated every step
    inst.set_beam_kernel()
    inst.set_lsf_kernel()


    # ## Add the model set, instrument to the Galaxy ##
    
    gal.model = mod_set
    gal.instrument = inst


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
    #    ** specifies slit width, slit PA as well **
    data1d = data_classes.Data1D(r=gs4_r, velocity=gs4_vel,
                                      vel_disp=gs4_disp, vel_err=err_vel,
                                      vel_disp_err=err_disp, slit_width=beamsize.value,
                                      slit_pa=pa, inst_corr=inst_corr, 
                                      filename_velocity=f_data_1d)

    # Add data to Galaxy object:
    gal.data = data1d


    # #### Setup apertures:
    # Setup apertures: circular apertures placed on the cube for GS4_43501.

    profile1d_type = 'circ_ap_cube'    # Extraction in circular apertures placed on the cube

    aperture_radius = 0.5 * gal.instrument.beam.major.value

    moment_calc  = False    # 1D data was extracted using Gaussian fits

    gal.data.apertures = aperture_classes.setup_aperture_types(gal=gal, 
                    profile1d_type=profile1d_type, 
                    aperture_radius=aperture_radius, 
                    from_data=True, 
                    partial_weight=True,
                    moment=moment_calc)

    # Add profile1d_type to data:
    gal.data.profile1d_type = profile1d_type


    # -----------------

    # # MCMC Fitting #

    # ### MCMC fitting parameters ###

    # Set parameters for fitting: 
    #    - Passing options to `emcee`
    #    - Other calculation options
    
    # Options passed to emcee
    ## FULL TEST
    ncpus = 190
    scale_param_a = 5 #3
    nwalkers = 1000
    nburn = 50
    nsteps = 200
    minaf = None
    maxaf = None
    neff = 10

    # Other options
    do_plotting = True       # Plot bestfit, corner, trace or not
    oversample = 1           # Factor by which to oversample model (eg, subpixels)
    fitdispersion = True     # Fit dispersion profile in addition to velocity

    blob_name = 'mvirial'    # Also save 'blob' values of Mvirial, calculated at every chain step
    
    if scale_param_a != 3:
        extra = '_a{}'.format(scale_param_a)
    else:
        extra = ''
    outdir_mcmc_full = outdir_mcmc = outdir + 'MCMC_full_run_nw{}_ns{}{}/'.format(nwalkers, nsteps, extra)

    f_log = outdir_mcmc_full + 'mcmc_run.log'

    # Choose plot filetype:
    plot_type = 'pdf'


    # -------

    # ## Run `Dysmalpy` fitting: MCMC ##

    mcmc_results = fitting.fit_mcmc(gal, nWalkers=nwalkers, nCPUs=ncpus,
                                   scale_param_a=scale_param_a, nBurn=nburn,
                                   nSteps=nsteps, minAF=minaf, maxAF=maxaf,
                                   nEff=neff, do_plotting=do_plotting,
                                   oversample=oversample, outdir=outdir_mcmc_full,
                                   fitdispersion=fitdispersion,
                                   f_log=f_log, 
                                   profile1d_type=gal.data.profile1d_type,
                                   blob_name=blob_name, 
                                   plot_type=plot_type, overwrite=False)
    return None

if __name__ == "__main__":
    run_mcmc_full_1D()
    
    
