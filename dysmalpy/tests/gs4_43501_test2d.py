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
out_dir = basedir+'fitting_output/2D/'


# Data files:
# 2D:
# Vel file: 
velfile2D = data_dir+'GS4_43501-vel.fits'
dispfile2D = data_dir+'GS4_43501-disp.fits'

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Do 2D fitting:

gal, mcmc_settings_kwargs = setup_dysmalpy_fitting(ndim=2)

# Upload the data set to be fit
gs4_vel = fits.getdata(velfile2D)
gs4_disp = fits.getdata(dispfile2D)
# Mask bad data:
gs4_disp[(gs4_disp > 1000.) | (~np.isfinite(gs4_disp))] = -1e6
mask = np.ones(gs4_vel.shape)
# Create mask of bad data
mask[(gs4_disp < 0)] = 0
err_vel = np.ones(gs4_vel.shape)*15.
err_disp = np.ones(gs4_vel.shape)*15.

# SINFONI pixscale for this obs: 0.125
test_data2d = data_classes.Data2D(pixscale=0.125, velocity=gs4_vel,
                                  vel_disp=gs4_disp, vel_err=err_vel,
                                  vel_disp_err=err_disp, mask=mask)

gal.data = test_data2d


def run_2d_test():
    mcmc_results = fitting.fit(gal, out_dir=out_dir, **mcmc_settings_kwargs)

if __name__ == "__main__":
        run_2d_test()
