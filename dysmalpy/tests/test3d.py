# Script to test 3D fitting on data from KMOS3D object GS4_43501

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
sys.path.append('..')

from .. import galaxy
from .. import models
from .. import fitting
from .. import instrument
from .. import data_classes
from .. import parameters

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
sigmaz = 2./2.35482   # kpc
zheight_fixed = {'sigmaz': False}

zheight_prof = models.ZHeightGauss(sigmaz=sigmaz, name='zheightgaus',
                                   fixed=zheight_fixed)
zheight_prof.sigmaz.tied = tie_sigz_reff

# Geometry
inc = 55.     # degrees
pa = 23.     # degrees, blue-shifted side CCW from north
xshift = -2    # pixels from center
yshift = 5    # pixels from center

geom_fixed = {'inc': True,
              'pa': True,
              'xshift': False,
              'yshift': False}

geom_bounds = {'inc': (0, 90),
               'pa': (0, 90),
               'xshift': (-6, 0),
               'yshift': (3, 9)}

geom = models.Geometry(inc=inc, pa=pa, xshift=xshift, yshift=yshift,
                       fixed=geom_fixed, bounds=geom_bounds, name='geom')


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
mod_set.line_center = 6550.

# Set up the instrument
pixscale = 0.06*u.arcsec                # arcsec/pixel
fov = [61, 61]                           # (nx, ny) pixels
beamsize = 0.75*u.arcsec                 # FWHM of beam
wave_start = 6528.15155*u.Angstrom       # Starting wavelength of spectrum
wave_step = 0.655*u.Angstrom             # Spectral step
nwave = 153                               # Number of spectral pixels
sig_inst = 38*u.km/u.s                   # Instrumental spectral resolution

beam = instrument.Beam(major=beamsize)
lsf = instrument.LSF(sig_inst)

inst.beam = beam
inst.lsf = lsf
inst.pixscale = pixscale
inst.fov = fov
inst.wave_step = wave_step
inst.wave_start = wave_start
inst.nwave = nwave

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

# Parameters for the MCMC fitting
nwalkers = 200
ncpus = 8
scale_param_a = 2
nburn = 200
nsteps = 1000
minaf = None
maxaf = None
neff = 10
do_plotting = True
oversample = 1

def run3d_test():
    mcmc_results = fitting.fit(gal, nWalkers=nwalkers, nCPUs=ncpus,
                               scale_param_a=scale_param_a, nBurn=nburn,
                               nSteps=nsteps, minAF=minaf, maxAF=maxaf,
                               nEff=neff, do_plotting=do_plotting,
                               oversample=oversample, outdir=out_dir)
                               
    return mcmc_results

if __name__ == "__main__":
    run3d_test()
