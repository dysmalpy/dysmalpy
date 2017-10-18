# Script to test 1D fitting on data from KMOS3D object GS4_43501

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

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
data_dir = '/data/dysmalpy/test_data/GS4_43501/'

# Directory where to save output files
out_dir = '/data/dysmalpy/1D_tests/GS4_43501/'

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

halo_fixed = {'mvirial': False,
              'conc': True}

halo_bounds = {'mvirial': (10, 13),
               'conc': (1, 20)}

halo = models.NFW(mvirial=mvirial, conc=conc, z=gal.z,
                  fixed=halo_fixed, bounds=halo_bounds, name='halo')

# Dispersion profile
sigma0 = 39.   # km/s
disp_fixed = {'sigma0': False}
disp_bounds = {'sigma0': (10, 200)}

disp_prof = models.DispersionConst(sigma0=sigma0, fixed=disp_fixed,
                                          bounds=disp_bounds, name='dispprof')

# z-height profile
sigmaz = 0.9   # kpc
zheight_fixed = {'sigmaz': True}

zheight_prof = models.ZHeightGauss(sigmaz=sigmaz, name='zheightgaus',
                                   fixed=zheight_fixed)

# Geometry
inc = 62.     # degrees
pa = 142.     # degrees, blue-shifted side CCW from north
xshift = 0    # pixels from center
yshift = 0    # pixels from center

geom_fixed = {'inc': False,
              'pa': True,
              'xshift': True,
              'yshift': True}

geom_bounds = {'inc': (0, 90),
               'pa': (90, 180),
               'xshift': (0, 4),
               'yshift': (-10, -4)}

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
pixscale = 0.125*u.arcsec                # arcsec/pixel
fov = [33, 33]                           # (nx, ny) pixels
beamsize = 0.55*u.arcsec                 # FWHM of beam
wave_start = 6528.15155*u.Angstrom       # Starting wavelength of spectrum
wave_step = 0.655*u.Angstrom             # Spectral step
nwave = 67                               # Number of spectral pixels
sig_inst = 45*u.km/u.s                   # Instrumental spectral resolution

beam = instrument.Beam(major=beamsize)
lsf = instrument.LSF(sig_inst)

inst.beam = beam
#inst.lsf = lsf
inst.pixscale = pixscale
inst.fov = fov
inst.wave_step = wave_step
inst.wave_start = wave_start
inst.nwave = nwave

# Set the beam kernel so it doesn't have to be calculated every step
inst.set_beam_kernel()
#inst.set_lsf_kernel(spec_type='wavelength', spec_center=mod_set.line_center*u.Angstrom)

# Add the model set and instrument to the Galaxy
gal.model = mod_set
gal.instrument = inst

# Upload the data set to be fit
dat_arr = np.loadtxt(data_dir+'GS4_43501.obs_prof.txt')
gs4_r = dat_arr[:,0]
gs4_vel = dat_arr[:,1]
gs4_disp = dat_arr[:,3]
err_vel = dat_arr[:,2]
err_disp = dat_arr[:,4]

test_data1d = data_classes.Data1D(r=gs4_r, velocity=gs4_vel,
                                  vel_disp=gs4_disp, vel_err=err_vel,
                                  vel_disp_err=err_disp, slit_width=0.22,
								  slit_pa=-37)

gal.data = test_data1d

# Parameters for the MCMC fitting
nwalkers = 500
ncpus = 8
scale_param_a = 2
nburn = 200
nsteps = 1000
minaf = None
maxaf = None
neff = 10
do_plotting = True
oversample = 1

mcmc_results = fitting.fit(gal, nWalkers=nwalkers, nCPUs=ncpus,
                           scale_param_a=scale_param_a, nBurn=nburn,
                           nSteps=nsteps, minAF=minaf, maxAF=maxaf,
                           nEff=neff, do_plotting=do_plotting,
                           oversample=oversample, out_dir=out_dir)
