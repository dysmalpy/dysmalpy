# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Module containing some useful utility functions

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library

# Third party imports
import numpy as np
import astropy.modeling as apy_mod
import matplotlib.pyplot as plt
import scipy.signal as sp_sig


def calc_pixel_distance(nx, ny, center_coord):
    """
    Function to calculate the distance of each pixel
    from a specific pixel as well as the position angle
    :param nx:
    :param ny:
    :param center_coord:
    :return:
    """

    xx, yy = np.meshgrid(range(nx), range(ny))

    dx = xx - center_coord[0]
    dy = yy - center_coord[1]

    # Calculate the separation
    seps = np.sqrt(dx**2 + dy**2)

    pa = -np.arctan(dx/dy)*180./np.pi

    return seps, pa


def create_aperture_mask(nx, ny, center_coord, dr):
    """

    :param nx:
    :param ny:
    :param center_coord:
    :param dr:
    :return:
    """

    seps, pa = calc_pixel_distance(nx, ny, center_coord)

    return seps <= dr


def determine_aperture_centers(nx, ny, center_coord, pa, dr):
    """
    Determine the centers of the apertures that span an image/cube along a line with position
    angle pa and goes through center_coord. Each aperture is dr away from each other.
    :param nx:
    :param ny:
    :param center_coord:
    :param pa:
    :param dr:
    :return:
    """

    pa_rad = -np.pi/180. * pa

    # Calculate the intersection of the line defined by PA and center_coord with the edges
    # of the image/cube
    xcenter = center_coord[0]
    ycenter = center_coord[1]
    count = 0
    edge_points = []

    # Check the x = 0 border
    y_x0 = -xcenter/np.tan(pa_rad) + ycenter
    if (y_x0 >= 0) & (y_x0 <= ny):
        count += 1
        edge_points.append([0, y_x0])

    # Check the y = 0 border
    x_y0 = -ycenter*np.tan(pa_rad) + xcenter
    if (x_y0 >= 0) & (x_y0 <= nx):
        count += 1
        edge_points.append([x_y0, 0])

    # Check the x = nx border
    y_nx = (nx - xcenter)/np.tan(pa_rad) + ycenter
    if (y_nx >= 0) & (y_nx <= ny):
        count += 1
        edge_points.append([nx, y_nx])

    # Check the y = ny border
    x_ny = (ny - ycenter) * np.tan(pa_rad) + xcenter
    if (x_ny >= 0) & (x_ny <= nx):
        count += 1
        edge_points.append([x_ny, ny])

    # Make sure there are only two intersections
    if count != 2:
        raise(ValueError, 'Number of intersections is not 2, something wrong happened!')

    # Calculate the start and end radii for the aperture centers based on the border
    # intersections. If the intersection is below ycenter the radius is negative
    r1 = np.sqrt((edge_points[0][0] - xcenter)**2 + (edge_points[0][1] - ycenter)**2)
    if edge_points[0][1] < ycenter:
        r1 = -r1

    r2 = np.sqrt((edge_points[1][0] - xcenter) ** 2 + (edge_points[1][1] - ycenter) ** 2)
    if edge_points[1][1] < ycenter:
        r2 = -r2

    # Setup the radii for the aperture centers with r = 0 always occurring
    if r1 > r2:
        r_neg = np.sort(-np.arange(0, -r2, dr))
        r_pos = np.arange(0, r1, dr)
        r_centers = np.concatenate((r_neg, r_pos[1:]))

    else:
        r_neg = np.sort(-np.arange(0, -r1, dr))
        r_pos = np.arange(0, r2, dr)
        r_centers = np.concatenate((r_neg, r_pos[1:]))

    # Get the pixel positions for each radii
    xaps, yaps = calc_pix_position(r_centers, pa, xcenter, ycenter)

    return xaps, yaps, r_centers


def calc_pix_position(r, pa, xcenter, ycenter):
    """
    Simple function to determine the pixel that is r away from (xcenter, ycenter) along
    a line with position angle, pa
    :param r:
    :param pa:
    :param xcenter:
    :param ycenter:
    :return:
    """
    
    pa_rad = np.pi/180. * pa
    #signfac = np.sign(np.cos(pa_rad))
    #xnew = -r*np.sin(pa_rad)*signfac + xcenter
    #signfac = np.sign(np.sin(pa_rad))
    signfac = -1
    xnew = -r*np.sin(pa_rad)*signfac + xcenter
    ynew = r*np.cos(pa_rad)*signfac + ycenter

    return xnew, ynew


def measure_1d_profile_apertures(cube, rap, pa, spec_arr, dr=None, center_pixel=None,
                                 ap_centers=None, spec_mask=None, estimate_err=False, nmc=100,
                                 profile_direction='positive', debug=False):
    """
    Measure the 1D rotation curve using equally spaced apertures along a defined axis
    :param cube: Cube to measure the 1D profile on
    :param rap: Radius of the circular apertures in pixels
    :param dr: Distance between the circular apertures in pixels
    :param center_pixel: Central pixel that defines r = 0
    :param pa: Position angle of the line that the circular apertures lie on
    :param spec_arr: The spectral array (i.e. wavelengths, frequencies, velocities, etc)
    :param spec_mask: Boolean mask to apply to the spectrum to exclude from fitting
    :param estimate_err: True or False to use Monte Carlo to estimate the errors on the fits
    :param nmc: The number of trials in the Monte Carlo analysis to use.
    :returns: centers: The radial offset from the central pixel in pixels
    :returns: flux: The integrated best fit "flux" for each aperture
    :returns: mean: The best fit mean of the Gaussian profile in the same units as spec_arr
    :returns: disp: The best fit dispersion of the Gaussian profile in the same units as spec_arr

    Note: flux, mean, and disp will be Nap x 2 arrays if estimate_err = True where Nap is the number
          of apertures that are fit. The first row will be best fit values and the second row will
          contain the errors on those parameters.
    """
    # profile_direction = 'negative'
    
    ny = cube.shape[1]
    nx = cube.shape[2]

    # Assume the default central pixel is the center of the cube
    if center_pixel is None:
        center_pixel = (np.int(nx / 2), np.int(ny / 2))

    # Assume equal distance between successive apertures equal to diameter of aperture
    if dr is None:
        dr = 2*rap

    # First determine the centers of all the apertures that fit within the cube
    if ap_centers is None:
        xaps, yaps, ap_centers = determine_aperture_centers(nx, ny, center_pixel, pa, dr)
    else:
        xaps, yaps = calc_pix_position(ap_centers, pa, center_pixel[0], center_pixel[1])
    
    ### SHOULD SORT THIS OUT FROM DATA by correctly setting model geom PA + slit_pa
    # if (profile_direction == 'negative') & (np.abs(pa) < 90.0):
    #     ap_centers = -ap_centers

    # Setup up the arrays to hold the results
    naps = len(xaps)
    flux = np.zeros(naps)
    mean = np.zeros(naps)
    disp = np.zeros(naps)

    if estimate_err:
        flux_err = np.zeros(naps)
        mean_err = np.zeros(naps)
        disp_err = np.zeros(naps)

    # For each aperture, sum up the spaxels within the aperture and fit a Gaussian line profile
    for i in range(naps):

        mask_ap = create_aperture_mask(nx, ny, (xaps[i], yaps[i]), rap)
        mask_cube = np.tile(mask_ap, (cube.shape[0], 1, 1))
        spec = np.nansum(np.nansum(cube*mask_cube, axis=1), axis=1)

        if spec_mask is not None:
            spec_fit = spec[spec_mask]
            spec_arr_fit = spec_arr[spec_mask]
        else:
            spec_fit = spec
            spec_arr_fit = spec_arr

        # Use the first and second moment as a guess of the line parameters
        mom0 = np.sum(spec_fit)
        mom1 = np.sum(spec_fit * spec_arr_fit) / mom0
        mom2 = np.sum(spec_fit * (spec_arr_fit - mom1) ** 2) / mom0

        mod = apy_mod.models.Gaussian1D(amplitude=mom0 / np.sqrt(2 * np.pi * np.abs(mom2)),
                                        mean=mom1,
                                        stddev=np.sqrt(np.abs(mom2)))
        mod.amplitude.bounds = (0, None)
        mod.stddev.bounds = (0, None)
        fitter = apy_mod.fitting.LevMarLSQFitter()
        best_fit = fitter(mod, spec_arr_fit, spec_fit)

        mean[i] = best_fit.mean.value
        disp[i] = best_fit.stddev.value
        flux[i] = best_fit.amplitude.value * np.sqrt(2 * np.pi) * disp[i]

        if debug:
            plt.figure()
            plt.plot(spec_arr_fit, spec_fit)
            plt.plot(spec_arr_fit, best_fit(spec_arr_fit))
            plt.title('r = {0}'.format(ap_centers[i]))

        if estimate_err:
            residual = spec - best_fit(spec)
            rms = np.std(residual)
            flux_mc = np.zeros(nmc)
            mean_mc = np.zeros(nmc)
            disp_mc = np.zeros(nmc)

            for j in range(nmc):
                rand_spec = np.random.randn(len(spec_fit)) * rms + spec_fit
                best_fit_mc = fitter(mod, spec_arr_fit, rand_spec)
                mean_mc[j] = best_fit_mc.mean.value
                disp_mc[j] = best_fit_mc.stddev.value
                flux_mc[j] = best_fit_mc.amplitude.value * np.sqrt(2 * np.pi) * disp_mc[j]

            flux_err[i] = np.nanstd(flux_mc)
            mean_err[i] = np.nanstd(mean_mc)
            disp_err[i] = np.nanstd(disp_mc)

    if estimate_err:
        return ap_centers, np.vstack([flux, flux_err]), np.vstack([mean, mean_err]), np.vstack([disp, disp_err])

    else:
        return ap_centers, flux, mean, disp



def apply_smoothing_2D(vel, disp, smoothing_type=None, smoothing_npix=1):
    if smoothing_type is None:
        return vel, disp
    else:
        if (smoothing_type.lower() == 'median'):
            vel = sp_sig.medfilt(vel, kernel_size=smoothing_npix)
            disp = sp_sig.medfilt(disp, kernel_size=smoothing_npix)
        else:
            message("Smoothing type={} not supported".format(smoothing_type))
            
        return vel, disp
        