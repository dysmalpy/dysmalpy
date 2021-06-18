# Script to fit single object in 3D with Dysmalpy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import platform
from contextlib import contextmanager
import sys
import shutil

import matplotlib
matplotlib.use('agg')

from dysmalpy import galaxy
from dysmalpy import models
from dysmalpy import fitting
from dysmalpy import instrument
from dysmalpy import parameters
from dysmalpy import plotting
from dysmalpy import config

import copy
import numpy as np
import astropy.units as u

try:
    import utils_io
    from dysmalpy_fit_single import dysmalpy_fit_single
except ImportError:
    from . import utils_io
    from .dysmalpy_fit_single import dysmalpy_fit_single

# Backwards compatibility
def dysmalpy_fit_single_3D(param_filename=None, data=None, datadir=None,
        outdir=None, plot_type='pdf', overwrite=False):
    return dysmalpy_fit_single(param_filename=param_filename, data=data, datadir=datadir,
                outdir=outdir, plot_type=plot_type, overwrite=overwrite)



def user_specific_load_3D_data(param_filename=None, datadir=None):
    # EDIT THIS FILE TO HAVE SPECIFIC LOADING OF DATA!

    params = utils_io.read_fitting_params(fname=param_filename)

    # Recommended to trim cube to around the relevant line only,
    # both for speed of computation and to avoid noisy spectral resolution elements.

    FNAME_CUBE = None
    FNAME_ERR = None
    FNAME_MASK = None
    FNAME_MASK_SKY = None       # sky plane mask -- eg, mask areas away from galaxy.
    FNAME_MASK_SPEC = None      # spectral dim masking -- eg, mask a skyline.
                                #  ** When trimming cube mind that masks need to be appropriately trimmed too.

    # Optional: set RA/Dec of reference pixel in the cube: mind trimming....
    ref_pixel = None
    ra = None
    dec = None

    pixscale=params['pixscale']

    # +++++++++++++++++++++++++++++++++++++++++++
    # Upload the data set to be fit
    cube = fits.getdata(FNAME_CUBE)
    err_cube = fits.getdata(FNAME_ERR)
    mask_sky = fits.getdata(FNAME_MASK_SKY)
    mask_spec = fits.getdata(FNAME_MASK_SPEC)

    spec_type = 'velocity'    # either 'velocity' or 'wavelength'
    spec_arr = None           # Needs to be array of vel / wavelength for the spectral dim of cube.
                              #  1D arr. Length must be length of spectral dim of cube.
    spec_unit = u.km/u.s      # EXAMPLE! set as needed.

    # Auto mask some bad data
    if automask:
        # Add settings here: S/N ??

        pass

    data3d = data_classes.Data3D(cube, pixscale, spec_type, spec_arr,
                            err_cube = err_cube, mask_cube=mask_cube,
                            mask_sky=mask_sky, mask_spec=mask_spec,
                            ra=ra, dec=dec,
                             ref_pixel=ref_pixel, spec_unit=spec_unit)

    return data3d

def dysmalpy_fit_single_3D_wrapper(param_filename=None, datadir=None, default_load_data=True, overwrite=False):

    if default_load_data:
        params = utils_io.read_fitting_params(fname=param_filename)
        data3d = utils_io.load_single_object_3D_data(params=params, datadir=datadir)
    else:
        data3d = user_specific_load_3D_data(param_filename=param_filename, datadir=datadir)

    dysmalpy_fit_single_3D(param_filename=param_filename, data=data3d, overwrite=overwrite)

    return None



if __name__ == "__main__":

    param_filename = sys.argv[1]

    try:
        if sys.argv[2].strip().lower() != 'reanalyze':
            datadir = sys.argv[2]
        else:
            datadir = None
    except:
        datadir = None


    dysmalpy_fit_single_3D_wrapper(param_filename=param_filename, datadir=datadir)
