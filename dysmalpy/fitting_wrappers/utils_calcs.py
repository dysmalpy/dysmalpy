# Utility calculation methods for fitting wrappers


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys

import datetime

import numpy as np
import astropy.units as u
import astropy.constants as apy_con

try:
    import photutils
    from astropy.convolution import Gaussian2DKernel
    loaded_photutils = True
except:
    loaded_photutils = False


def _auto_gen_3D_mask_simple(cube=None, err=None, snr_thresh=3.):
    # Crude first-pass on auto-generated 3D cube mask, based on S/N:

    snr_cube = np.abs(cube)/np.abs(err)
    # Set NaNs / 0 err to SNR=0.
    snr_cube[~np.isfinite(snr_cube)] = 0.

    mask = np.ones(cube.shape)
    mask[snr_cube < snr_thresh] = 0


    return mask

def auto_gen_3D_mask(cube=None, err=None, sig_thresh=1.5, npix_min=5, snr_thresh=3., snr_thresh_1=3.):

    ## Crude first-pass masking by pixel S/N:
    mask_sn_pix = _auto_gen_3D_mask_simple(cube=cube, err=err, snr_thresh=snr_thresh_1)

    #mask = mask_sn_pix.copy()

    #cube_m = cube.copy() * mask_sn_pix.copy()
    #ecube_m = err.copy() * mask_sn_pix.copy()


    # TEST:
    mask = np.ones(cube.shape)

    cube_m = cube.copy()
    ecube_m = err.copy()

    ####################################
    # Mask NaNs:
    mask[~np.isfinite(cube_m)] = 0
    cube_m[~np.isfinite(cube_m)] = -99.

    mask[~np.isfinite(cube_m)] = 0
    ecube_m[~np.isfinite(ecube_m)] = -99.

    # Clean up 0s in error, if it's masked
    ecube_m[mask == 0] = 99.

    ####################################
    fmap_cube_sn = np.sum(cube_m, axis=0)
    emap_cube_sn = np.sqrt(np.sum(ecube_m**2, axis=0))



    # Do segmap on mask2D?????
    if loaded_photutils:

        bkg = photutils.Background2D(fmap_cube_sn, fmap_cube_sn.shape, filter_size=(3,3))

        thresh = sig_thresh * bkg.background_rms

        #kernel = Gaussian2DKernel(2. /(2. *np.sqrt(2.*np.log(2.))), x_size=3, y_size=3)   # Gaussian of FWHM 2 pix
        kernel = Gaussian2DKernel(3. /(2. *np.sqrt(2.*np.log(2.))), x_size=5, y_size=5)   # Gaussian of FWHM 3 pix
        segm = photutils.detect_sources(fmap_cube_sn, thresh, npixels=npix_min, filter_kernel=kernel)


        mask2D = segm._data.copy()
        mask2D[mask2D>0] = 1
    else:
        # TRY JUST S/N cut on 2D?
        sn_map_cube_sn = fmap_cube_sn / emap_cube_sn

        mask2D = np.ones(sn_map_cube_sn.shape)
        mask2D[sn_map_cube_sn < snr_thresh] = 0



    # Apply mask2D to mask:
    mask_cube = np.tile(mask2D, (cube_m.shape[0], 1, 1))

    #mask = mask * mask_cube

    mask = mask * mask_sn_pix


    mask = mask * mask_cube

    return mask

def _auto_truncate_crop_cube(cube, params=None,
            pixscale=None, spec_type='velocity', spec_arr=None,
            err_cube=None, mask_cube=None, mask_sky=None, mask_spec=None,
            spec_unit=u.km/u.s,weight=None, xcenter=None, ycenter=None):

    # First truncate by spec:
    if 'spec_vel_trim' in params.keys():
        whin = np.where((spec_arr >= params['spec_vel_trim'][0]) & (spec_arr <= params['spec_vel_trim'][1]))[0]
        spec_arr = spec_arr[whin]
        cube = cube[whin, :, :]
        err_cube = err_cube[whin, :, :]

        if mask_cube is not None:
            mask_cube = mask_cube[whin, :, :]
        if mask_sky is not None:
            mask_sky = mask_sky[whin, :, :]
        if mask_spec is not None:
            mask_spec = mask_spec[whin, :, :]
        if weight is not None:
            weight = weight[whin, :, :]


    # Then truncate area:
    if 'spatial_crop_trim' in params.keys():
        # left right bottom top
        sp_trm = np.array(params['spatial_crop_trim'], dtype=np.int32)
        cube = cube[:, sp_trm[2]:sp_trm[3], sp_trm[0]:sp_trm[1]]
        err_cube = err_cube[:, sp_trm[2]:sp_trm[3], sp_trm[0]:sp_trm[1]]
        if mask_cube is not None:
            mask_cube = mask_cube[:, sp_trm[2]:sp_trm[3], sp_trm[0]:sp_trm[1]]
        if mask_sky is not None:
            mask_sky = mask_sky[:, sp_trm[2]:sp_trm[3], sp_trm[0]:sp_trm[1]]
        if mask_spec is not None:
            mask_spec = mask_spec[:, sp_trm[2]:sp_trm[3], sp_trm[0]:sp_trm[1]]
        if weight is not None:
            weight = weight[:, sp_trm[2]:sp_trm[3], sp_trm[0]:sp_trm[1]]
        if xcenter is not None:
            xcenter -= sp_trm[0]
        if ycenter is not None:
            ycenter -= sp_trm[2]


    ##############
    # Then check for first / last non-masked parts:
    if mask_cube is not None:
        mcube = cube.copy()*mask_cube.copy()
    else:
        mcube = cube.copy()

    mcube[~np.isfinite(mcube)] = 0.
    mcube=np.abs(mcube)
    c_sum_spec = mcube.sum(axis=(1,2))
    c_spec_up = np.cumsum(c_sum_spec)
    c_spec_down = np.cumsum(c_sum_spec[::-1])

    wh_l = np.where(c_spec_up > 0.)[0][0]
    wh_r = np.where(c_spec_down > 0.)[0][0]
    if (wh_l > 0) | (wh_r > 0):
        if wh_r == 0:
            v_wh_r = len(c_spec_down)
        else:
            v_wh_r = -wh_r

        spec_arr = spec_arr[wh_l:v_wh_r]
        cube = cube[wh_l:v_wh_r, :, :]
        err_cube = err_cube[wh_l:v_wh_r, :, :]

        if mask_cube is not None:
            mask_cube = mask_cube[wh_l:v_wh_r, :, :]
        if mask_sky is not None:
            mask_sky = mask_sky[wh_l:v_wh_r, :, :]
        if mask_spec is not None:
            mask_spec = mask_spec[wh_l:v_wh_r, :, :]
        if weight is not None:
            weight = weight[wh_l:v_wh_r, :, :]


    #####
    return cube, err_cube, mask_cube, mask_sky, mask_spec, weight, spec_arr, xcenter, ycenter
