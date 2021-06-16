# Script to test 2D fitting on data from KMOS3D object GS4_43501

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

from dysmalpy import galaxy
from dysmalpy import models
from dysmalpy import fitting
from dysmalpy import instrument
from dysmalpy import data_classes
from dysmalpy import parameters

from fitting_wrappers import utils_io

import numpy as np
import astropy.units as u
import astropy.io.fits as fits


# Function to tie the scale height to the effective radius
def tie_sigz_reff(model_set):
 
    reff = model_set.components['disk+bulge'].r_eff_disk.value
    invq = model_set.components['disk+bulge'].invq_disk
    sigz = 2.0*reff/invq/2.35482

    return sigz
    
def setup_gal(data_dir=None, out_dir=None):

    # Initialize the Galaxy, Instrument, and Model Set
    gal = galaxy.Galaxy(z=1.613, name='GS4_43501')
    mod_set = models.ModelSet()
    inst = instrument.Instrument()

    # Baryonic Component: Combined Disk+Bulge
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
                  'r_eff_disk': False,
                  'n_disk': True,
                  'r_eff_bulge': True,
                  'n_bulge': True,
                  'bt': False}

    # Set bounds
    bary_bounds = {'total_mass': (10, 13),
                   'r_eff_disk': (0.5, 50.0),
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

    #bary.r_eff_disk.prior = parameters.GaussianPrior(center=5.0, stddev=1.0)

    # Halo component
    mvirial = 12.0
    conc = 5.0
    fdm = 0.5

    halo_fixed = {'mvirial': False,
                  'conc': True,
                  'fdm': False}

    halo_bounds = {'mvirial': (10, 13),
                   'conc': (1, 20),
                   'fdm': (0., 1.)}

    halo = models.NFW(mvirial=mvirial, conc=conc, fdm=fdm, z=gal.z,
                      fixed=halo_fixed, bounds=halo_bounds, name='halo')
    #
    
    halo.fdm.tied = utils_io.tie_fdm
    halo.mvirial.prior = parameters.BoundedGaussianPrior(center=11.5, stddev=0.5)
    
    
    # Dispersion profile
    sigma0 = 39.   # km/s
    disp_fixed = {'sigma0': False}
    disp_bounds = {'sigma0': (10, 200)}

    disp_prof = models.DispersionConst(sigma0=sigma0, fixed=disp_fixed,
                                              bounds=disp_bounds, name='dispprof')

    # z-height profile
    sigmaz = 0.9   # kpc
    zheight_fixed = {'sigmaz': False}

    zheight_prof = models.ZHeightGauss(sigmaz=sigmaz, name='zheightgaus',
                                       fixed=zheight_fixed)
    zheight_prof.sigmaz.tied = tie_sigz_reff

    # Geometry
    inc = 62.     # degrees
    pa = 142.     # degrees, blue-shifted side CCW from north
    xshift = 2    # pixels from center
    yshift = -6    # pixels from center
    vel_shift = 0.

    geom_fixed = {'inc': False,
                  'pa': False,
                  'xshift': False,
                  'yshift': False,
                  'vel_shift': False}

    geom_bounds = {'inc': (0, 90),
                   'pa': (90, 180),
                   'xshift': (0, 4),
                   'yshift': (-10, -4),
                   'vel_shift': (-100., 100.)}

    geom = models.Geometry(inc=inc, pa=pa, xshift=xshift, yshift=yshift, vel_shift=vel_shift, 
                           fixed=geom_fixed, bounds=geom_bounds, name='geom')
    #
    geom.inc.prior = parameters.BoundedSineGaussianPrior(center=inc, stddev=0.1)

    # Add all of the model components to the ModelSet
    mod_set.add_component(bary, light=True)
    mod_set.add_component(halo)
    mod_set.add_component(disp_prof)
    mod_set.add_component(zheight_prof)
    mod_set.add_component(geom)

    # Set some kinematic options for calculating the velocity profile
    adiabatic_contract = False
    pressure_support = True
    mod_set.kinematic_options.adiabatic_contract = adiabatic_contract
    mod_set.kinematic_options.pressure_support = pressure_support


    # # Set the line central wavelength that is being modeled
    # mod_set.line_center = 6550.
    # 
    # # Set up the instrument
    # pixscale = 0.125*u.arcsec                # arcsec/pixel
    # fov = [33, 33]                           # (nx, ny) pixels
    # beamsize = 0.55*u.arcsec                 # FWHM of beam
    # wave_start = 6528.15155*u.Angstrom       # Starting wavelength of spectrum
    # wave_step = 0.655*u.Angstrom             # Spectral step
    # nwave = 67                               # Number of spectral pixels
    # sig_inst = 45*u.km/u.s                   # Instrumental spectral resolution
    
    pixscale = 0.125*u.arcsec                # arcsec/pixel
    fov = [33, 33]                           # (nx, ny) pixels
    beamsize = 0.55*u.arcsec                 # FWHM of beam
    spec_type = 'velocity'                   # 'velocity' or 'wavelength'
    spec_start = -1000*u.km/u.s              # Starting value of spectrum
    spec_step = 10*u.km/u.s                  # Spectral step
    nspec = 201                              # Number of spectral pixels
    sig_inst = 45*u.km/u.s                   # Instrumental spectral resolution

    beam = instrument.GaussianBeam(major=beamsize)
    lsf = instrument.LSF(sig_inst)

    inst.beam = beam
    inst.lsf = lsf
    inst.pixscale = pixscale
    inst.fov = fov
    # inst.wave_step = wave_step
    # inst.wave_start = wave_start
    # inst.nwave = nwave
    inst.spec_type = spec_type
    inst.spec_step = spec_step
    inst.spec_start = spec_start
    inst.nspec = nspec

    # Set the beam kernel so it doesn't have to be calculated every step
    inst.set_beam_kernel()
    inst.set_lsf_kernel()

    # Add the model set and instrument to the Galaxy
    gal.model = mod_set
    gal.instrument = inst

    # Upload the data set to be fit
    gs4_vel = fits.getdata(data_dir+'GS4_43501-vel.fits')
    gs4_disp = fits.getdata(data_dir+'GS4_43501-disp.fits')
    gs4_disp[(gs4_disp > 1000.) | (~np.isfinite(gs4_disp))] = -1e6
    #mask = np.ones(gs4_vel.shape)
    mask = fits.getdata(data_dir+'GS4_43501-mask1.fits')
    mask[(gs4_disp < 0)] = 0
    err_vel = np.ones(gs4_vel.shape)*15.
    err_disp = np.ones(gs4_vel.shape)*15.
    # # SHP test: messy / missing errors in masked parts:
    # err_vel[mask==0] = 0.
    # err_disp[mask==0] = 0.

    test_data2d = data_classes.Data2D(pixscale=pixscale.value, velocity=gs4_vel,
                                      vel_disp=gs4_disp, vel_err=err_vel,
                                      vel_disp_err=err_disp, mask=mask)

    gal.data = test_data2d

    # # Parameters for the MCMC fitting
    # nwalkers = 500
    # ncpus = 8
    # scale_param_a = 2
    # nburn = 200
    # nsteps = 1000
    # minaf = None
    # maxaf = None
    # neff = 10
    # do_plotting = True
    # oversample = 1

    params = {'outdir': out_dir, 
              'galID': 'GS4_43501',
              'fit_method': 'mcmc',
              'fit_module': fitting.emcee, 
              'fdata': data_dir+'GS4_43501.obs_prof.txt',
              'profile1d_type': 'circ_ap_cube',
              'moment_calc': True, #False, 
              'partial_weight': False, 
              'param_filename': None,
              'nWalkers': 20,
              'nCPUs': 2,
              'scale_param_a': 2,
              'nBurn': 5,
              'nSteps': 20,
              'minAF': None,
              'maxAF': None,
              'nEff': 10,
              'do_plotting': True,
              'oversample': 1,
              'do_plotting': True,
              'fitdispersion': True,
              'oversampled_chisq': True}
          
    #
    gal.data.moment=params['moment_calc']
    
    return gal, params


#
def run_2d_test(data_dir=None, out_dir=None):
    
    gal, params = setup_gal(data_dir=data_dir, out_dir=out_dir)
    
    mcmc_results = fitting.fit(gal, nWalkers=params['nWalkers'], nCPUs=params['nCPUs'],
                               scale_param_a=params['scale_param_a'], nBurn=params['nBurn'],
                               nSteps=params['nSteps'], minAF=params['minAF'], maxAF=params['maxAF'],
                               nEff=params['nEff'], do_plotting=params['do_plotting'],
                               oversample=params['oversample'], outdir=params['outdir'],
                               fitdispersion=params['fitdispersion'], 
                               oversampled_chisq=params['oversampled_chisq'])
                               
    #
    utils_io.save_results_ascii_files(fit_results=mcmc_results, gal=gal, params=params)

#
if __name__ == "__main__":
    
    try:
        data_dir = sys.argv[1]
        out_dir  = sys.argv[2]
    except:
        #data_dir = '/Users/sedona/Dropbox/RCOut_Reinhard/rc_2019_analysis/2D_maps/'
        data_dir = '/Users/sedona/data/mpe_ir/gs4_43501_dysmalpy_fitting/GS4_43501_H250/'
        out_dir  = '/Users/sedona/data/dysmalpy_tests/test2D/'
    
    #
    fitting.ensure_dir(out_dir)
    
    run_2d_test(data_dir=data_dir, out_dir=out_dir)
