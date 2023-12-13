# Utility calculation methods for fitting wrappers


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u

try:
    # import photutils
    try:
        from photutils.background import Background2D
        from photutils.segmentation import detect_sources
    except:
        from photutils import Background2D, detect_sources
    from astropy.convolution import Gaussian2DKernel, convolve
    _loaded_photutils = True
except:
    _loaded_photutils = False



def auto_gen_3D_mask(cube=None, err=None,
            sig_segmap_thresh=1.5, npix_segmap_min=5,
            snr_int_flux_thresh=3.,
            snr_thresh_pixel=None,
            sky_var_thresh=3.,
            apply_skymask_first=True):
    """
    Generate a mask for a 3D cube, based on thresholds / other automated detections.

    Input:
        cube:               3D cube (numpy ndarray)
        err:                3D error cube (numpy ndarray)

    Optional input:
        sig_segmap_thresh:  If photutils is available, use segmentation map on the
                            integrated 2D spaxel flux map to determine the spatial masking region.
                            Default: sig_segmap_thresh = 1.5
        npix_segmap_min:    Minimum number of pixels in each segmap grouping.

        snr_int_flux_thresh:If photutils is unavailable, do a SNR threshold cut on the
                            integrated 2D spaxel flux map to determine the spatial masking region.
                            Default: snr_int_flux_thresh = 3.

        snr_thresh_pixel:   Flat SNR threshold to apply to all pixels (eg, each point in x,y,spectral space).
                            USE CAUTION IF SETTING, as can bias fits.
                            Default: snr_thresh_pixel = None (no per-pixel SNR masking).

        sky_var_thresh:     Threshold for determining skylines from variance spectrum.
                            NaNs are masked in error cube, then variance spectrum is
                            determined by spatial sum in quadrature of variance (error^2).
                            Spectral regions where variance > sky_var_thresh * median(variance)
                            are masked as skyline regions.
                            Typical values: 3.0 for K/Y, 2.0 for H/J. (Tune as necessary.)

        apply_skymask_first:    Apply skyline mask before doing segmentation/integrated flux masking? Default: True

    Output:
            mask (3D cube), mask_dict (info about generated mask)
    """

    # snr_int_flux_thresh=3., snr_thresh_pixel=3.,


    mask_dict = {}
    mask_dict['sig_segmap_thresh'] =    sig_segmap_thresh
    mask_dict['npix_segmap_min'] =      npix_segmap_min
    mask_dict['snr_int_flux_thresh'] =  snr_int_flux_thresh
    mask_dict['snr_thresh_pixel'] =     snr_thresh_pixel
    mask_dict['sky_var_thresh'] =       sky_var_thresh
    mask_dict['apply_skymask_first'] =  apply_skymask_first

    ## Crude first-pass masking by pixel S/N:
    if snr_thresh_pixel is not None:
        mask_sn_pix = _auto_gen_3D_mask_pixel_SNR(cube=cube, err=err, snr_thresh=snr_thresh_pixel)
    else:
        mask_sn_pix = None

    mask_dict['mask_sn_pix'] = mask_sn_pix


    mask = np.ones(cube.shape)

    cube_m = cube.copy()
    ecube_m = err.copy()

    ####################################
    # Mask NaNs:
    mask[~np.isfinite(cube_m)] = 0.
    cube_m[~np.isfinite(cube_m)] = -99.

    mask[~np.isfinite(ecube_m)] = 0.
    ecube_m[~np.isfinite(ecube_m)] = -99.

    # Clean up 0s in error, if it's masked
    ecube_m[mask == 0] = 99.



    if sky_var_thresh is not None:
        # Mask skylines from variance:
        mask_sky = _auto_gen_3D_mask_sky(ecube_m, mask=mask, sky_var_thresh=sky_var_thresh)
    else:
        mask_sky = None

    mask_dict['mask_sky'] = mask_sky

    ####################################
    # Integrated 2D flux masking: either segmentation map, or SNR cut

    # Premask sky:
    if (apply_skymask_first & (mask_sky is not None)):
        mask *= mask_sky

    do_int_flux_SNR_mask = False
    mask2D = None
    fmap_cube_sn = np.sum(cube_m*mask, axis=0)
    emap_cube_sn = np.sqrt(np.sum(ecube_m**2*mask, axis=0))
    mask_dict['fmap_cube_sn'] = fmap_cube_sn
    mask_dict['emap_cube_sn'] = emap_cube_sn
    if sig_segmap_thresh is not None:
        # Do segmap on mask2D?????
        if _loaded_photutils:
            bkg = None
            exclude_percentile=0.
            ex_perctl_increment = 5.
            while ((bkg is None) & (exclude_percentile < 100.)):
                exclude_percentile += ex_perctl_increment
                try:
                    bkg = Background2D(fmap_cube_sn, fmap_cube_sn.shape, filter_size=(3,3),
                                    exclude_percentile=exclude_percentile)
                except:
                    pass
            if bkg is None:
                raise ValueError
            else:
                if exclude_percentile > 10.:
                    print(" Masking segmap: used exclude_percentile={}".format(exclude_percentile))

            thresh = sig_segmap_thresh * bkg.background_rms

            kernel = Gaussian2DKernel(3. /(2. *np.sqrt(2.*np.log(2.))), x_size=5, y_size=5)   # Gaussian of FWHM 3 pix
            fmap_cube_sn_filtered = convolve(fmap_cube_sn.copy(), kernel)
            segm = detect_sources(fmap_cube_sn_filtered, thresh, npixels=npix_segmap_min)

            mask_dict['exclude_percentile'] = exclude_percentile
            mask_dict['thresh'] = thresh
            mask_dict['segm'] = segm

            # Find the max flux seg region:
            segfluxmax = 0.
            for seg in segm.segments:
                mseg = seg._segment_data.copy()
                mseg[mseg>0] = 1
                mseg_flux = fmap_cube_sn.copy() * mseg
                if mseg_flux.sum() > segfluxmax:
                    segfluxmax = mseg_flux.sum()
                    mask2D = mseg


            #mask2D = segm._data.copy()
            #mask2D[mask2D>0] = 1
        else:
            do_int_flux_SNR_mask = True


    elif snr_int_flux_thresh is not None:
        do_int_flux_SNR_mask = True

    if do_int_flux_SNR_mask:
        # TRY JUST S/N cut on 2D integrated flux?
        sn_map_cube_sn = fmap_cube_sn / emap_cube_sn
        # Handle nans / pre-masked
        sn_map_cube_sn[~np.isfinite(sn_map_cube_sn)] = 0

        mask2D = np.ones(sn_map_cube_sn.shape)
        mask2D[sn_map_cube_sn < snr_int_flux_thresh] = 0

    if mask2D is not None:
        # Apply mask2D to mask:
        mask_cube = np.tile(mask2D, (cube_m.shape[0], 1, 1))
    else:
        mask_cube = None


    mask_dict['mask2D'] = mask2D
    mask_dict['mask_cube'] = mask_cube

    if mask_sn_pix is not None:
        mask *= mask_sn_pix

    if mask_cube is not None:
        mask *= mask_cube

    if mask_sky is not None:
        mask *= mask_sky

    return mask, mask_dict

def _auto_gen_3D_mask_pixel_SNR(cube=None, err=None, snr_thresh=3.):
    # Crude first-pass on auto-generated 3D cube mask, based on S/N:

    snr_cube = np.abs(cube)/np.abs(err)
    # Set NaNs / 0 err to SNR=0.
    snr_cube[~np.isfinite(snr_cube)] = 0.

    mask = np.ones(cube.shape)
    mask[snr_cube < snr_thresh] = 0


    return mask


def _auto_gen_3D_mask_sky(ecube, mask=None, sky_var_thresh=None):
    varspec = np.sum(np.sum(ecube**2 * mask, axis=2), axis=1)

    whmask = np.where(varspec > sky_var_thresh*np.median(varspec))[0]
    mask_sky1d = np.ones(len(varspec))
    mask_sky1d[whmask] = 0.
    mask_sky = np.tile(mask_sky1d.reshape((mask_sky1d.shape[0],1,1)), (1, ecube.shape[1], ecube.shape[2]))

    return mask_sky


def _auto_truncate_crop_cube(cube, params=None,
            spec_type='velocity', spec_arr=None,
            err_cube=None, mask_cube=None, mask_sky=None, mask_spec=None,
            spec_unit=u.km/u.s,weight=None, xcenter=None, ycenter=None):


    cube_orig = cube.copy()

    # First truncate by spec:
    wh_spec_keep = None
    if 'spec_vel_trim' in params.keys():
        wh_spec_keep = np.where((spec_arr >= params['spec_vel_trim'][0]) & (spec_arr <= params['spec_vel_trim'][1]))[0]
        spec_arr = spec_arr[wh_spec_keep]
        cube = cube[wh_spec_keep, :, :]
        err_cube = err_cube[wh_spec_keep, :, :]

        if mask_cube is not None:
            mask_cube = mask_cube[wh_spec_keep, :, :]
        if mask_sky is not None:
            mask_sky = mask_sky[wh_spec_keep, :, :]
        if mask_spec is not None:
            mask_spec = mask_spec[wh_spec_keep, :, :]
        if weight is not None:
            weight = weight[wh_spec_keep, :, :]

    # Then truncate area:
    sp_trm = None
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
    wh_ends_trim = None
    if 'auto_trim_ends' in params.keys():
        if params['auto_trim_ends']:
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

                wh_ends_trim = [wh_l, v_wh_r]

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
    return cube, err_cube, mask_cube, mask_sky, mask_spec, weight, \
                    spec_arr, xcenter, ycenter, wh_spec_keep, wh_ends_trim, sp_trm



def pad_3D_mask_to_uncropped_size(obs=None, mask=None):
    spec_unit=u.Unit(obs.data.data.wcs.wcs.cunit[2].to_string())
    spec_arr = obs.data.data.spectral_axis.to(spec_unit).value

    if obs.data.cube_precrop_sh is not None:
        mask_cube = np.zeros(obs.data.cube_precrop_sh)
        sp_trm = obs.data.sp_trm
        if sp_trm is None:
            sp_trm = [0,mask_cube.shape[2],0,mask_cube.shape[1]]

        if obs.data.wh_spec_keep is not None:
            spc_trm = [obs.data.wh_spec_keep[0], obs.data.wh_spec_keep[-1]+1]
        else:
            spc_trm = [0, mask_cube.shape[0]]

        if obs.data.wh_ends_trim is not None:
            spc_trm[0] += obs.data.wh_ends_trim[0]
            spc_trm[1] += obs.data.wh_ends_trim[0]
            if obs.data.wh_ends_trim[1]< 0:
                spc_trm[1] += obs.data.wh_ends_trim[1]

        mask_cube[spc_trm[0]:spc_trm[1], sp_trm[2]:sp_trm[3], sp_trm[0]:sp_trm[1]] = mask.copy()

        spec_step = obs.data.data.wcs.wcs.cdelt[2]
        spec_start = obs.data.data.wcs.wcs.crval[2] - (spc_trm[0]+1-obs.data.data.wcs.wcs.crpix[2])
        spec_arr = np.arange(obs.data.cube_precrop_sh[0]) * spec_step + spec_start

    else:
        mask_cube = mask.copy()
    return mask_cube, spec_arr
