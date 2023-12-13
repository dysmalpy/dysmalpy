# Script to plot kin

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from dysmalpy import galaxy, plotting

from dysmalpy.fitting_wrappers import utils_io


# ----------------------------------------------------------------------
def plot_1D_rotcurve_components(gal=None, output_options=None):

    # Reload bestfit case
    if gal is None:
        gal = galaxy.load_galaxy_object(filename=output_options.f_model)

    plotting.plot_rotcurve_components(gal=gal, outpath=output_options.outdir,
            plot_type=output_options.plot_type,
            overwrite=output_options.overwrite,
            overwrite_curve_files=output_options.overwrite)


    return None

# ----------------------------------------------------------------------

def plot_curve_components_overview(fname_gal=None, fname_results=None,
        param_filename=None,
        overwrite = False,
        overwrite_curve_files=False,
        outpath=None):

    # Reload the galaxy:
    gal = galaxy.load_galaxy_object(filename=fname_gal)

    params = utils_io.read_fitting_params(fname=param_filename)

    if 'aperture_radius' not in params.keys():
        params['aperture_radius'] = -99.


    plotting.plot_rotcurve_components(gal=gal, overwrite=overwrite,
                                overwrite_curve_files=overwrite_curve_files,
                                outpath = outpath)


    return None
