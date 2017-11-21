# Script to test 2D fitting on data from KMOS3D object GS4_43501

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import sys
sys.path.append('.')

from dysmalpy import galaxy
from dysmalpy import models
from dysmalpy import fitting
from dysmalpy import instrument
from dysmalpy import data_classes
from dysmalpy import parameters

import numpy as np
import astropy.units as u
import astropy.io.fits as fits

from gs4_43501_test_base import setup_dysmalpy_fitting

basedir = os.getenv("HOME")+'/data/mpe_ir/gs4_43501_dysmalpy_fitting/'

# Directory where the data lives
data_dir = basedir+'GS4_43501_H250/'

# Directory where to save output files
out_dir = basedir+'fitting_output/3D/'


# Data files:
# 3D:
# Cube file:
file3d = data_dir+'gs4-43501_h250_19h10.fits.gz'
# error file:
fileerr3d = data_dir+'noise_gs4-43501_h250_19h10.fits.gz'

filepsf = data_dir+'psf_gs4-43501_h250_19h10.fits.gz'

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Do 3D fitting:
gal, mcmc_settings_kwargs = setup_dysmalpy_fitting(ndim=3)

# Upload the data set to be fit
cube = fits.getdata(file3d)
header = fits.getheader(file3d)
mask = np.ones((header['NAXIS1'], header['NAXIS2']))
err_cube = fits.getdata(fileerr3d)
unit = u.um


# RESET PSF TO EMPIRICAL AT SOME POINT

spec_arr = (np.arange(cube.shape[0]) - header['CRPIX3'])*header['CDELT3'] + header['CRVAL3']
pscale = 0.125
inst.set_lsf_kernel(spec_type='wavelength', spec_step=header['CDELT3']*unit, 
                    spec_center=header['CRVAL3']*unit)

test_data3d = data_classes.Data3D(cube, pixscale=pscale, spec_type='wavelength', spec_arr=spec_arr,
                                  err_cube=err_cube, mask_sky=mask, mask_spec=None,
                                  estimate_err=False, spec_unit=unit)

gal.data = test_data3d


def run_3d_test():
    mcmc_results = fitting.fit(gal, out_dir=out_dir, **mcmc_settings_kwargs)

if __name__ == "__main__":
        run_2d_test()
