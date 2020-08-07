# Script to test 1D fitting on data from KMOS3D object GS4_43501

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from dysmalpy import galaxy
from dysmalpy import models
from dysmalpy import fitting
from dysmalpy import aperture_classes
from dysmalpy import instrument
from dysmalpy import data_classes
from dysmalpy import parameters
from fitting_wrappers import utils_io

import numpy as np
import astropy.units as u
import astropy.io.fits as fits

# Directory where the data lives
#data_dir = '/Users/ttshimiz/Dropbox/Research/LLAMA/dysmal/input/obs_prof/'
data_dir = '/Users/sedona/Dropbox/RCOut_Reinhard/rc_2019_analysis/1D_profiles/'

# Directory where to save output files
#out_dir = '/Users/ttshimiz/Dropbox/Research/LLAMA/dysmal/testing/1D_tests/fix_conc5_gaussian_prior_reffANDmvirial_nodispersion/'
out_dir = '/Users/sedona/Desktop/mpfit_1d_test/cap_test/'
#out_dir = '/Users/sedona/Desktop/mpfit_1d_test/mpfit_test/'

# Function to tie the scale height to the effective radius
def tie_sigz_reff(model_set):
 
    reff = model_set.components['disk+bulge'].r_eff_disk.value
    invq = model_set.components['disk+bulge'].invq_disk
    sigz = 2.0*reff/invq/2.35482

    return sigz

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

# Halo component
mvirial = 12.0
conc = 5.0

halo_fixed = {'mvirial': False,
              'conc': True}

halo_bounds = {'mvirial': (10, 13),
               'conc': (1, 20)}

halo = models.NFW(mvirial=mvirial, conc=conc, z=gal.z,
                  fixed=halo_fixed, bounds=halo_bounds, name='halo')

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
xshift = 0    # pixels from center
yshift = 0    # pixels from center

geom_fixed = {'inc': True, #False,
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
#mod_set.line_center = 6550.

# Set up the instrument
pixscale = 0.125*u.arcsec                # arcsec/pixel
fov = [33, 33]                           # (nx, ny) pixels
beamsize = 0.55*u.arcsec                 # FWHM of beam
spec_type = 'velocity'                   # 'velocity' or 'wavelength'
spec_start = -1000*u.km/u.s              # Starting value of spectrum
spec_step = 10*u.km/u.s                  # Spectral step
nspec = 201                              # Number of spectral pixels
sig_inst = 45*u.km/u.s                   # Instrumental spectral resolution

beam = instrument.Beam(major=beamsize)
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
inst.set_beam_kernel()
inst.set_lsf_kernel()

# Add the model set and instrument to the Galaxy
gal.model = mod_set
gal.instrument = inst

# Upload the data set to be fit
dat_arr = np.loadtxt(data_dir+'GS4_43501.obs_prof.txt')
gs4_r = dat_arr[:,0]
gs4_vel = dat_arr[:,1]
gs4_disp = np.sqrt(dat_arr[:,3]**2 + sig_inst.value**2)
err_vel = dat_arr[:,2]
err_disp = dat_arr[:,4]
inst_corr = True                  # Flag for if the measured dispersion has been
                                  # corrected for instrumental resolution

test_data1d = data_classes.Data1D(r=gs4_r, velocity=gs4_vel,
                                  vel_disp=gs4_disp, vel_err=err_vel,
                                  vel_disp_err=err_disp, slit_width=0.55, #0.22,
                                  slit_pa=pa, inst_corr=inst_corr)

gal.data = test_data1d
gal.data.apertures = aperture_classes.setup_aperture_types(gal=gal, 
            profile1d_type='circ_ap_cube', 
            from_data=True, 
            partial_weight=False,
            moment=False)
gal.data.profile1d_type = 'circ_ap_cube'


# Parameters for the MPFIT fitting
maxiter = 200
oversample = 1
do_plotting = True
fitdispersion = True


params = {'outdir': out_dir, 
          'galID': 'GS4_43501',
          'fit_method': 'mpfit',
          'fit_module': fitting.mpfit.__module__, 
          'fdata': data_dir+'GS4_43501.obs_prof.txt',
          'profile1d_type': 'circ_ap_cube',
          'moment_calc': False, 
          'partial_weight': False, 
          'param_filename': None}

              
def run_1d_test():
    mpfit_results = fitting.fit_mpfit(gal, maxiter=maxiter, 
                               do_plotting=do_plotting,
                               oversample=oversample, outdir=out_dir,
                               fitdispersion=fitdispersion)
                               
    # Save results
    utils_io.save_results_ascii_files(fit_results=mpfit_results, gal=gal, params=params)

def reload_1d_test():
    
    # For compatibility with Python 2.7:
    mod_in = copy.deepcopy(gal.model)
    gal.model = mod_in
    
    # Initialize a basic dummy results class
    mpfit_results = fitting.MPFITResults(model=gal.model)
    # Set what the names are for reloading
    fresults = out_dir+'mpfit_results.pickle'
    
    mpfit_results.reload_results(filename=fresults)
    
    return mpfit_results
    
    
    
    


if __name__ == "__main__":
    run_1d_test()
