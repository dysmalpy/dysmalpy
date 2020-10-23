from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from dysmalpy import galaxy
from dysmalpy import models
from dysmalpy import fitting
from dysmalpy import instrument
from dysmalpy import data_classes
from dysmalpy import parameters
from dysmalpy import plotting
from dysmalpy import aperture_classes

from fitting_wrappers import utils_io

import os
import copy

import numpy as np
import astropy.units as u
import astropy.io.fits as fits



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
outdir = '/afs/mpe.mpg.de/home/sedona/JUPYTER_EXAMPLES/JUPYTER_OUTPUT_2D/'

# -------------------

# ##### Set function to tie scale height relative to effective radius #####

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


# - Also see **fitting_wrappers.utils_io** for more tied functions

# ----------

def run_mcmc_full_2D():
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
    vel_shift = 0 # velocity shift at center ; km/s
    geom_fixed = {'inc': False,
                  'pa':  False,    # True,
                  'xshift': False,
                  'yshift': False,
                  'vel_shift': False}

    geom_bounds = {'inc': (52, 72),
                   'pa': (132, 152),
                   'xshift': (-2.5, 2.5),
                   'yshift': (-2.5, 2.5),
                   'vel_shift': (-100, 100)}

    geom = models.Geometry(inc=inc, pa=pa, xshift=xshift, yshift=yshift, vel_shift=vel_shift, 
                           fixed=geom_fixed, bounds=geom_bounds, name='geom')


    geom.inc.prior = parameters.BoundedSineGaussianPrior(center=62, stddev=0.1)
    
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
    inst.fov = [27, 27]                      # (nx, ny) pixels
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
    #   - *2D velocity, dispersion maps and error*
    #   - *A mask can be loaded / created as well*
    #   
    # * Put data in `Data2D` class
    # 
    # * Add data to Galaxy object


    gal_vel = fits.getdata(data_dir+'GS4_43501_Ha_vm.fits')
    gal_disp = fits.getdata(data_dir+'GS4_43501_Ha_dm.fits')

    err_vel = fits.getdata(data_dir+'GS4_43501_Ha_vm_err.fits')
    err_disp = fits.getdata(data_dir+'GS4_43501_Ha_dm_err.fits')

    mask = fits.getdata(data_dir+'GS4_43501_Ha_m.fits')

    #gal_disp[(gal_disp > 1000.) | (~np.isfinite(gal_disp))] = -1e6
    #mask[(gs4_disp < 0)] = 0

    inst_corr = True                  # Flag for if the measured dispersion has been
                                      # corrected for instrumental resolution

    # Mask NaNs:
    mask[~np.isfinite(gal_vel)] = 0
    gal_vel[~np.isfinite(gal_vel)] = 0.

    mask[~np.isfinite(err_vel)] = 0
    err_vel[~np.isfinite(err_vel)] = 0.

    mask[~np.isfinite(gal_disp)] = 0
    gal_disp[~np.isfinite(gal_disp)] = 0.

    mask[~np.isfinite(err_disp)] = 0
    err_disp[~np.isfinite(err_disp)] = 0.
    

    # Put data in Data2D data class: 
    #    ** specifies data pixscale as well **
    data2d = data_classes.Data2D(pixscale=inst.pixscale.value, velocity=gal_vel,
                                      vel_disp=gal_disp, vel_err=err_vel,
                                      vel_disp_err=err_disp, mask=mask, 
                                      inst_corr=inst_corr)

    # Use moment for fitting -- on single pixel scale it does not 
    #     appear to strongly impact the results, and it's faster
    data2d.moment = True
    
    # Add data to Galaxy object:
    gal.data = data2d
    

    # -----------------

    # # MCMC Fitting #

    # ### MCMC fitting parameters ###

    # Options passed to emcee
    ## FULL TEST
    nwalkers = 1000
    ncpus = 190
    scale_param_a = 5 #3
    nburn = 175
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
    
    mcmc_results = fitting.fit_mcmc(gal, nWalkers=nwalkers, nCPUs=ncpus,
                                   scale_param_a=scale_param_a, nBurn=nburn,
                                   nSteps=nsteps, minAF=minaf, maxAF=maxaf,
                                   nEff=neff, do_plotting=do_plotting,
                                   oversample=oversample, outdir=outdir_mcmc_full,
                                   fitdispersion=fitdispersion,
                                   blob_name=blob_name, 
                                   f_log=f_log, 
                                   plot_type=plot_type, overwrite=False)


    return None

if __name__ == "__main__":
    run_mcmc_full_2D()
    
    
