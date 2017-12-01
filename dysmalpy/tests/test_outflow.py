# Script to test 3D fitting on data from KMOS3D object GS4_43501

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from dysmalpy import galaxy
from dysmalpy import models
from dysmalpy import fitting
from dysmalpy import instrument
from dysmalpy import data_classes
from dysmalpy import parameters

import numpy as np
import astropy.units as u
import astropy.io.fits as fits

# Directory where the data lives
data_dir = '/Users/ttshimiz/Dropbox/Research/LLAMA/dysmal/input/data/GS4_43501/'

# Directory where to save output files
# If it doesn't exist, Dysmalpy will create the directory
out_dir = '.'

# Initialize the Galaxy, Instrument, and Model Set
gal = galaxy.Galaxy(z=1.613, name='GS4_43501')
mod_set = models.ModelSet()
inst = instrument.Instrument()

# Baryonic Component: Combined Disk+Bulge
total_mass = 11.0  # M_sun
bt = 0.3  # Bulge-Total ratio
r_eff_disk = 5.0  # kpc
n_disk = 1.0
invq_disk = 5.0   # Effective radius/scale height
r_eff_bulge = 1.0  # kpc
n_bulge = 4.0
invq_bulge = 1.0
noord_flat = True  # Switch for applying Noordermeer flattening

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

bary.r_eff_disk.prior = parameters.GaussianPrior(center=5.0, stddev=1.0)

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
sigma0 = 39.  # km/s
disp_fixed = {'sigma0': False}
disp_bounds = {'sigma0': (10, 200)}

disp_prof = models.DispersionConst(sigma0=sigma0, fixed=disp_fixed,
                                   bounds=disp_bounds, name='dispprof')

# z-height profile
sigmaz = 0.9  # kpc
zheight_fixed = {'sigmaz': True}

zheight_prof = models.ZHeightGauss(sigmaz=sigmaz, name='zheightgaus',
                                   fixed=zheight_fixed)

# Geometry
inc = 62.  # degrees
pa = 142.  # degrees, blue-shifted side CCW from north
xshift = 2  # pixels from center
yshift = -6  # pixels from center

geom_fixed = {'inc': False,
              'pa': False,
              'xshift': False,
              'yshift': False}

geom_bounds = {'inc': (0, 90),
               'pa': (90, 180),
               'xshift': (0, 4),
               'yshift': (-10, -4)}

geom = models.Geometry(inc=inc, pa=pa, xshift=xshift, yshift=yshift,
                       fixed=geom_fixed, bounds=geom_bounds, name='geom')

# Outflow component
inner_angle = 0.           # Inner opening angle in degrees
outer_angle = 90.          # Outer opening angle in degrees
index = 0                # Power-law index that controls the radial dependence of
                           # the outflow velocity
vmax = 2000.               # Maximum velocity in km/s
rend = 1.0                 # The radial extent of the outflow in kpc
profile_type = 'decrease'  # The velocity radial profile type.
                           # Can be 'increase', 'decrease', or 'both'.
rturn = 0.5                # The turnover radius in kpc for the 'both' profile
norm_flux = 0.5            # Normalization for the flux profile which follows
                           # an exponential
tau = 0                  # How fast the flux falls off with radius

outflow_fixed = {'thetain': True,
                 'thetaout': True,
                 'n': True,
                 'vmax': True,
                 'rturn': True,
                 'rend': True}

outflow_bounds = {'thetain': (0, 90),
                 'thetaout': (0, 90),
                 'n': (0, 5),
                 'vmax': (200, 2000),
                 'rturn': (0.5, 5.0),
                 'rend': (2.0, 5.0)}

outflow = models.BiconicalOutflow(thetain=inner_angle, thetaout=outer_angle,
                                  n=index, vmax=vmax, rend=rend,
                                  profile_type=profile_type, rturn=rturn,
                                  norm_flux=norm_flux, tau_flux=tau,
                                  fixed=outflow_fixed, bounds=outflow_bounds,
                                  name='outflow')

# Outflow geometry
out_pa = 135.           # Position anlge in degrees with 0 corresponding to West
out_inc = 45.           # Inclination with 0 corresponding to face-on
out_xshift = 2         # X pixel position of outflow center
out_yshift = -6          # Y pixel position of outflow center

out_geom_fixed = {'inc': False,
                  'pa': False,
                  'xshift': False,
                  'yshift': False}

out_geom_bounds = {'inc': (-90, 90),
                   'pa': (0, 180),
                   'xshift': (0, 4),
                   'yshift': (-10, -4)}

outflow_geom = models.Geometry(inc=out_inc, pa=out_pa,
                               xshift=out_xshift, yshift=out_yshift,
                               fixed=out_geom_fixed, bounds=out_geom_bounds,
                               name='outflow_geom')

# Outflow dispersion
out_sigma0 = 10.     # Intrinsic dispersion at each radius in km/s
out_disp_fixed = {'sigma0': True}
out_disp_bounds = {'sigma0': (0, 100)}

out_disp_prof = models.DispersionConst(sigma0=sigma0, fixed=out_disp_fixed,
                                       bounds=out_disp_bounds, name='out_dispprof')

# Add all of the model components to the ModelSet
mod_set.add_component(bary, light=True)
mod_set.add_component(halo)
mod_set.add_component(disp_prof)
mod_set.add_component(zheight_prof)
mod_set.add_component(geom)
mod_set.add_component(outflow)
mod_set.add_component(outflow_geom, geom_type='outflow')
mod_set.add_component(out_disp_prof, disp_type='outflow')


# Set some kinematic options for calculating the velocity profile
adiabatic_contract = False
pressure_support = True
mod_set.kinematic_options.adiabatic_contract = adiabatic_contract
mod_set.kinematic_options.pressure_support = pressure_support

# Set the line central wavelength that is being modeled
mod_set.line_center = 6550.

# Set up the instrument
pixscale = 0.125 * u.arcsec  # arcsec/pixel
fov = [41, 41]  # (nx, ny) pixels
beamsize = 0.55 * u.arcsec  # FWHM of beam
wave_start = 6528.15155 * u.Angstrom  # Starting wavelength of spectrum
wave_step = 0.655 * u.Angstrom  # Spectral step
nwave = 67  # Number of spectral pixels
sig_inst = 45 * u.km / u.s  # Instrumental spectral resolution

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
cube = fits.getdata(data_dir + 'GS4-43501-combo-22h-mccc-s2-v2.175.fits')
header = fits.getheader(data_dir + 'GS4-43501-combo-22h-mccc-s2-v2.175.fits')
mask = fits.getdata(data_dir + 'GS4_43501-mask1.fits')
err_cube = np.ones(cube.shape) * 0.1067

# # SHP test: messy / missing errors in masked parts:
# mask_3d = np.tile(mask, (err_cube.shape[0], 1, 1))
# err_cube[mask_3d==0] = 0.

spec_arr = (np.arange(cube.shape[0]) - header['CRPIX3']) * header['CDELT3']
pscale = 0.125
inst.set_lsf_kernel(spec_type='velocity',
                    spec_step=header['CDELT3'] * u.km / u.s)

test_data3d = data_classes.Data3D(cube, pixscale=pscale, spec_type='velocity',
                                  spec_arr=spec_arr,
                                  err_cube=err_cube, mask_sky=mask,
                                  mask_spec=None,
                                  estimate_err=False, spec_unit=u.km / u.s)

gal.data = test_data3d

# Parameters for the MCMC fitting
nwalkers = 20
ncpus = 8
scale_param_a = 2
nburn = 10
nsteps = 10
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
	                           oversample=oversample, out_dir=out_dir)

    return mcmc_results


if __name__ == "__main__":
    run3d_test()
