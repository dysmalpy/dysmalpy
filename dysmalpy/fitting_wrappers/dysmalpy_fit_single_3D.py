# Script to fit single object in 3D with Dysmalpy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys

import matplotlib
# Check if there is a display for plotting, or if there is an SSH/TMUX session.
# If no display, or if SSH/TMUX, use the matplotlib "agg" backend for plotting.
havedisplay = "DISPLAY" in os.environ
if havedisplay:
    exitval = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
    skipconds = (("SSH_CLIENT" in os.environ) | ("TMUX" in os.environ) | ("SSH_CONNECTION" in os.environ) | (os.environ["TERM"].lower().strip()=='screen') | (exitval != 0))
    havedisplay = not skipconds
if not havedisplay:
    matplotlib.use('agg')

import astropy.units as u

try:
    from dysmalpy_fit_single import dysmalpy_fit_single
except ImportError:
    from .dysmalpy_fit_single import dysmalpy_fit_single

# Backwards compatibility
def dysmalpy_fit_single_3D(param_filename=None, data_loader=None, datadir=None,
        outdir=None, plot_type='pdf', overwrite=False):
    return dysmalpy_fit_single(param_filename=param_filename,
                data_loader=data_loader, datadir=datadir,
                outdir=outdir, plot_type=plot_type, overwrite=overwrite)



# def user_specific_load_3D_data(params=None, datadir=None):

#     # CRUDE, INCOMPLETE EXAMPLE OF HOW TO MAKE A CUSTOM LOADER. 
#     # WRITE OWN FUNCTION LIKE THIS TO HAVE SPECIFIC LOADING OF DATA!

#     # Recommended to trim cube to around the relevant line only,
#     # both for speed of computation and to avoid noisy spectral resolution elements.

#     FNAME_CUBE = None
#     FNAME_ERR = None
#     FNAME_MASK = None
#     FNAME_MASK_SKY = None       # sky plane mask -- eg, mask areas away from galaxy.
#     FNAME_MASK_SPEC = None      # spectral dim masking -- eg, mask a skyline.
#                                 #  ** When trimming cube mind that masks need to be appropriately trimmed too.

#     # Optional: set RA/Dec of reference pixel in the cube: mind trimming....
#     ref_pixel = None
#     ra = None
#     dec = None

#     pixscale=params['pixscale']

#     # +++++++++++++++++++++++++++++++++++++++++++
#     # Upload the data set to be fit
#     cube = fits.getdata(FNAME_CUBE)
#     err_cube = fits.getdata(FNAME_ERR)
#     mask_cube = fits.getdata(FNAME_MASK)
#     mask_sky = fits.getdata(FNAME_MASK_SKY)
#     mask_spec = fits.getdata(FNAME_MASK_SPEC)

#     spec_type = 'velocity'    # either 'velocity' or 'wavelength'
#     spec_arr = None           # Needs to be array of vel / wavelength for the spectral dim of cube.
#                               #  1D arr. Length must be length of spectral dim of cube.
#     spec_unit = u.km/u.s      # EXAMPLE! set as needed.

#     # Auto mask some bad data
#     if automask:
#         # Add settings here: S/N ??

#         pass

#     data3d = data_classes.Data3D(cube, pixscale, spec_type, spec_arr,
#                             err_cube = err_cube, mask_cube=mask_cube,
#                             mask_sky=mask_sky, mask_spec=mask_spec,
#                             ra=ra, dec=dec,
#                              ref_pixel=ref_pixel, spec_unit=spec_unit)

#     return data3d

def dysmalpy_fit_single_3D_wrapper(param_filename=None, datadir=None,
                                   default_load_data=True, overwrite=False):

    # if default_load_data:
    #     data_loader=None
    # else:
    #     data_loader=user_specific_load_3D_data

    data_loader=None
    dysmalpy_fit_single_3D(param_filename=param_filename, data_loader=data_loader,
                           overwrite=overwrite)

    return None



if __name__ == "__main__":

    param_filename = sys.argv[1]

    # try:
    #     if sys.argv[2].strip().lower() != 'reanalyze':
    #         datadir = sys.argv[2]
    #     else:
    #         datadir = None
    # except:
    #     datadir = None


    # dysmalpy_fit_single_3D_wrapper(param_filename=param_filename, datadir=datadir)


    try:
        datadir = sys.argv[2]
    except:
        datadir = None


    dysmalpy_fit_single_3D_wrapper(param_filename=param_filename, datadir=datadir)
