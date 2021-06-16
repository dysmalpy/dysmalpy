# Script to test 3D fitting on data from KMOS3D object GS4_43501

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

# Directory where the data lives
data_dir = '/data/dysmalpy/test_data/EGS4_24985/'
#data_dir = '/Users/sedona/data/kmos3d/dysmalpy_tests/'

# Directory where to save output files
out_dir = '/data/dysmalpy/3D_tests/EGS4_24985/first_test/200walkers/'
#out_dir = '/Users/sedona/data/kmos3d/dysmalpy_tests/3D_tests/quick_test/'

# Function to tie the scale height to the effective radius
def tie_sigz_reff(model_set):

    reff = model_set.components['disk+bulge'].r_eff_disk.value
    invq = model_set.components['disk+bulge'].invq_disk
    sigz = 2.0*reff/invq/2.35482

    return sigz

def setup_gal(data_dir=None, out_dir=None):

    # Initialize the Galaxy, Instrument, and Model Set
    gal = galaxy.Galaxy(z=1.3965, name='EGS4_24985')
    mod_set = models.ModelSet()
    inst = instrument.Instrument()

    # Baryonic Component: Combined Disk+Bulge
    total_mass = 11.146    # M_sun
    bt = 0.2             # Bulge-Total ratio
    r_eff_disk = 4.383     # kpc
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

    bary.r_eff_disk.prior = parameters.GaussianPrior(center=4.383, stddev=0.843)
    bary.total_mass.prior = parameters.GaussianPrior(center=np.log10(1.4e11), stddev=0.22)
    bary.bt.prior = parameters.GaussianPrior(center=0.20, stddev=0.15)


    # Halo component
    mvirial = np.log10(4.24e12)
    conc = 4.4

    halo_fixed = {'mvirial': False,
                  'conc': True}

    halo_bounds = {'mvirial': (10, 13),
                   'conc': (1, 20)}

    halo = models.NFW(mvirial=mvirial, conc=conc, z=gal.z,
                      fixed=halo_fixed, bounds=halo_bounds, name='halo')
    halo.mvirial.prior = parameters.GaussianPrior(center=np.log10(4.24e12), stddev=0.92)

    # Dispersion profile
    sigma0 = 30.   # km/s
    disp_fixed = {'sigma0': False}
    #disp_bounds = {'sigma0': (10, 100)}

    disp_prof = models.DispersionConst(sigma0=sigma0, fixed=disp_fixed, name='dispprof')
    disp_prof.sigma0.prior = parameters.GaussianPrior(center=30., stddev=10.)

    # z-height profile
    #sigmaz = 2./2.35482   # kpc
    sigmaz = 0.9
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


    # Set the line central wavelength that is being modeled
    #mod_set.line_center = 6550.

    # Set up the instrument
    pixscale = 0.06*u.arcsec                # arcsec/pixel
    fov = [61, 61]                           # (nx, ny) pixels
    beamsize = 0.75*u.arcsec                 # FWHM of beam
    #wave_start = 6528.15155*u.Angstrom       # Starting wavelength of spectrum
    #wave_step = 0.655*u.Angstrom             # Spectral step
    #nwave = 153                               # Number of spectral pixels
    spec_type = 'velocity'                   # 'velocity' or 'wavelength'
    spec_start = -1000*u.km/u.s              # Starting value of spectrum
    spec_step = 10*u.km/u.s                  # Spectral step
    nspec = 201                              # Number of spectral pixels

    sig_inst = 38*u.km/u.s                   # Instrumental spectral resolution

    beam = instrument.GaussianBeam(major=beamsize)
    lsf = instrument.LSF(sig_inst)

    inst.beam = beam
    inst.lsf = lsf
    inst.pixscale = pixscale
    inst.fov = fov
    #inst.wave_step = wave_step
    #inst.wave_start = wave_start
    #inst.nwave = nwave
    inst.spec_type = spec_type
    inst.spec_step = spec_step
    inst.spec_start = spec_start
    inst.nspec = nspec

    # Set the beam kernel so it doesn't have to be calculated every step
    inst.set_beam_kernel()

    # Add the model set and instrument to the Galaxy
    gal.model = mod_set
    gal.instrument = inst

    # Upload the data set to be fit
    cube = fits.getdata(data_dir+'egs24985-co3-2-un-e-0.06-0.61arcsec.fits')
    header = fits.getheader(data_dir+'egs24985-co3-2-un-e-0.06-0.61arcsec.fits')
    #mask = fits.getdata(data_dir+'GS4_43501-mask1.fits')
    err_cube = np.ones(cube.shape)*0.000143

    # # SHP test: messy / missing errors in masked parts:
    # mask_3d = np.tile(mask, (err_cube.shape[0], 1, 1))
    # err_cube[mask_3d==0] = 0.

    spec_arr = (np.arange(cube.shape[0]) - header['CRPIX3'])*header['CDELT3']/1000.
    pscale = pixscale.value
    gal.instrument.set_lsf_kernel(spec_type='velocity', spec_step=header['CDELT3']/1000.*u.km/u.s)

    test_data3d = data_classes.Data3D(cube, pixscale=pscale, spec_type='velocity', spec_arr=spec_arr,
                                      err_cube=err_cube, mask_sky=None, mask_spec=None,
                                      estimate_err=False, spec_unit=u.km/u.s)

    gal.data = test_data3d

    # # Parameters for the MCMC fitting
    # nwalkers = 200
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
              'moment_calc': False,
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

    return gal, params

def run_3d_test(data_dir=None, out_dir=None):

    #** FIX THIS WITH UPDATED GALAXY SELECTION, AND APPROPRIATE PARAM BOUNDS + INITIAL VALUES ***

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

    return mcmc_results

if __name__ == "__main__":

    try:
        data_dir = sys.argv[1]
        out_dir  = sys.argv[2]
    except:
        #data_dir = '/Users/sedona/Dropbox/RCOut_Reinhard/rc_2019_analysis/2D_maps/'
        data_dir = '/Users/sedona/data/mpe_ir/outer_rc_2019/RC40/'
        out_dir  = '/Users/sedona/data/dysmalpy_tests/test3D/'


    fitting.ensure_dir(out_dir)

    run_3d_test(data_dir=data_dir, out_dir=out_dir)
