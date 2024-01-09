# Methods for loading data for fitting wrappers

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as apy_con

try:
    import tkinter_io
except ImportError:
    from . import tkinter_io

from dysmalpy import data_classes
from dysmalpy import utils as dpy_utils

import astropy.io.fits as fits

from dysmalpy.fitting_wrappers.utils_calcs import auto_gen_3D_mask, _auto_truncate_crop_cube, pad_3D_mask_to_uncropped_size
from dysmalpy.fitting_wrappers.setup_gal_models import setup_data_weighting_method

def read_fitting_params_input(fname=None):
    params = {}

    columns = ['keys', 'values']
    df = pd.read_csv(fname, sep=',', comment='#', names=columns, skipinitialspace=True).copy()

    for j, key in enumerate(df['keys'].values):
        if key is np.NaN:
            pass
        else:
            valset = False
            try:
                tmpval = df['values'][j].split('#')[0].strip()
            except:
                try:
                    if np.isnan(df['values'][j]):
                        tmpval = None
                    else:
                        tmpval = df['values'][j].strip()
                except:
                    print("param key: {}".format(key))
                    print("param line: {}".format(df['values'][j]))
                    raise ValueError
            try:
                tmparr = tmpval.split(' ')
                tmparrnew = []
                if len(tmparr) > 1:
                    tmparrnew = []
                    for ta in tmparr:
                        if len(ta) > 0:
                            tv = ta.strip()
                            try:
                                tvn = float(tv)
                            except:
                                tvn = tv
                            tmparrnew.append(tvn)
                    tmpval = tmparrnew
                    valset = True

            except:
                pass

            if not valset:
                strtmpval = str(tmpval).strip()
                if strtmpval == 'True':
                    tmpval = True
                elif strtmpval == 'False':
                    tmpval = False
                elif strtmpval == 'None':
                    tmpval = None
                elif strtmpval.lower() == 'inf':
                    tmpval = np.inf
                else:
                    try:
                        fltval = float(tmpval)
                        if (fltval % 1) == 0.:
                            tmpval = int(fltval)
                        else:
                            tmpval = fltval
                    except:
                        tmpval = strtmpval.strip()

            params[key] = tmpval

    return params


def read_fitting_params(fname=None):
    if fname is None:
        raise ValueError("parameter filename {} not found!".format(fname))

    # READ FILE HERE!
    params = read_fitting_params_input(fname=fname)

    # Set some defaults if not otherwise specified
    params_wrapper_specific = {'nObs': 1,
                               'obs_1_name': 'OBS',
                               'obs_1_tracer': 'LINE',
                               'overwrite': False,
                               'oversample': 1,
                               'include_halo': False,
                               'halo_profile_type': 'NFW',
                               'weighting_method': None,
                               'slit_width': None,
                               'slit_pa': None,
                               'integrate_cube': True,
                               'smoothing_type': None,
                               'smoothing_npix': 1,
                               # Fitting parameters:
                               'fitvelocity': True,
                               'fitdispersion': True,
                               'fitflux': False
                               }
    if 'fit_method' in list(params.keys()):
        fit_method = params['fit_method'].strip().lower()
        if 'blob_name' not in params:
            params['blob_name'] = None
    else:
        fit_method = None

    ## Add other defaults if not specified:
    for key in params_wrapper_specific.keys():
        if key not in params.keys():
            params[key] = params_wrapper_specific[key]

    # param_filename
    fname_split = fname.split(os.sep)
    params['param_filename'] = fname_split[-1]

    # Clean up outdir, datadir: ensure separators are os.sep:
    for key in ['outdir', 'datadir']:
        if key in params.keys():
            if params[key] is not None:
                params[key] = os.sep.join(os.sep.join(params[key].split("/")).split("\\"))

    # Catch depreciated case:
    if 'halo_inner_slope_fit' in params.keys():
        if params['halo_inner_slope_fit']:
            if params['halo_profile_type'].upper() == 'NFW':
                print("using depreciated param setting 'halo_inner_slope_fit=True'.")
                print("Assuming 'halo_profile_type=TwoPowerHalo' halo form.")
                params['halo_profile_type'] = 'TwoPowerHalo'

    # Catch other cases:
    if 'components_list' in params.keys():
        if 'halo' in params['components_list']:
            params['include_halo'] = True

    if params['include_halo']:
        if (fit_method is not None):
            if params['blob_name'] is None:
                if 'fdm_fixed' in params.keys():
                    if not params['fdm_fixed']:
                        # fdm is free
                        if params['halo_profile_type'].upper() == 'NFW':
                            params['blob_name'] = 'mvirial'
                        elif params['halo_profile_type'].lower() == 'twopowerhalo':
                            params['blob_name'] = ['alpha', 'mvirial']
                        elif params['halo_profile_type'].lower() == 'burkert':
                            params['blob_name'] = ['rb', 'mvirial']
                    else:
                        if params['halo_profile_type'].upper() == 'NFW':
                            if params['halo_conc_fixed'] is False:
                                params['blob_name'] = ['fdm', 'mvirial']
                            else:
                                params['blob_name'] = 'fdm'
                        else:
                            params['blob_name'] = ['fdm', 'mvirial']

            # ONLY SET THESE IF FITTING, FOR NOW
            if ('fdm_fixed' not in params.keys()) | ('fdm' not in params.keys()):
                if params['mvirial_fixed'] is True:
                    params['fdm'] = 0.5
                    params['fdm_fixed'] = False
                    params['fdm_bounds'] = [0, 1]
                    params['blob_name'] = 'mvirial'
                else:
                    params['fdm'] = -99.9
                    params['fdm_fixed'] = True
                    params['fdm_bounds'] = [0, 1]
                    params['blob_name'] = 'fdm'

            # Put a default, if missing
            if ('mvirial_tied' not in params.keys()):
                if params['halo_profile_type'].upper() == 'NFW':
                    params['mvirial_tied'] = False
                else:
                    # Default to the "old" behavior
                    params['mvirial_tied'] = True

        # Put a default, if missing:
        if ('mhalo_relation' not in params.keys()):
            # Default to MISSING
            params['mhalo_relation'] = None

        if ('truncate_lmstar_halo' not in params.keys()):
            # Default to MISSING
            params['truncate_lmstar_halo'] = None

    return params


def read_results_ascii_file(fname=None):


    names = ['component', 'param_name', 'fixed', 'best_value', 'l68_err', 'u68_err']

    data = pd.read_csv(
        fname, sep=' ', comment='#', names=names, skipinitialspace=True,
        index_col=False
    )


    return data





def load_single_obs_1D_data(fdata=None, fdata_mask=None, params=None, datadir=None, extra=''):


    # Setup datadir, if set. If not set (so datadir=None), fdata must be the full path.
    if datadir is None:
        # If datadir not passed directly, look for entry in params file:
        if 'datadir'+extra in params.keys():
            datadir = params['datadir'+extra]
        # If not passed directly as kwarg, and missing from params file, set to empty -- filenames must be full path.
        if datadir is None:
            datadir = ''

    # Load the data set to be fit
    dat_arr =   np.loadtxt(datadir+fdata)
    gal_r =     dat_arr[:,0]
    gal_vel =   dat_arr[:,1]
    gal_disp =  dat_arr[:,3]
    err_vel =   dat_arr[:,2]
    err_disp =  dat_arr[:,4]

    try:
        gal_flux = dat_arr[:,5]
        err_flux = dat_arr[:,6]
    except:
        gal_flux = None
        err_flux = None

    if 'r_shift'+extra in params.keys():
        # Apply a shift to the radius, if specified:
        if params['r_shift'+extra] is not None:
            gal_r += params['r_shift'+extra]


    if 'v_shift'+extra in params.keys():
        # Apply a shift to the radius, if specified:
        if params['v_shift'+extra] is not None:
            gal_vel += params['v_shift'+extra]

    if fdata_mask is not None:
        if os.path.isfile(datadir+fdata_mask):
            msk_arr =   np.loadtxt(datadir+fdata_mask)
            msk_r =     msk_arr[:,0]
            msk_vel =   msk_arr[:,1]
            msk_disp =  msk_arr[:,2]
        else:
            msk_vel = None
            msk_disp = None
    else:
        msk_vel = None
        msk_disp = None
    #####
    # Apply symmetrization if wanted:
    try:
        if params['symmetrize_data'+extra]:
            gal_r_new, gal_vel, err_vel = dpy_utils.symmetrize_1D_profile(gal_r, gal_vel, err_vel, sym=1)
            gal_r, gal_disp, err_disp = dpy_utils.symmetrize_1D_profile(gal_r, gal_disp, err_disp, sym=2)
            if gal_flux is not None:
                gal_r, gal_flux, err_flux = dpy_utils.symmetrize_1D_profile(gal_r, gal_flux, err_flux, sym=2)
    except:
        pass


    if 'weighting_method'+extra in params.keys():
        gal_weight = setup_data_weighting_method(method=params['weighting_method'+extra], r=gal_r)
    else:
        gal_weight = None


    # MOVE THESE TO INST AS WELL
    if ('xcenter'+extra in params.keys()):
        xcenter = params['xcenter'+extra]
    else:
        xcenter = None
    #
    if ('ycenter'+extra in params.keys()):
        ycenter = params['ycenter'+extra]
    else:
        ycenter = None

    data1d = data_classes.Data1D(r=gal_r, velocity=gal_vel,vel_disp=gal_disp,
                                vel_err=err_vel, vel_disp_err=err_disp,
                                flux=gal_flux, flux_err=err_flux,
                                weight=gal_weight,
                                mask_velocity=msk_vel, mask_vel_disp=msk_disp,
                                inst_corr=params['data_inst_corr'+extra],
                                xcenter=xcenter, ycenter=ycenter)

    return data1d

def load_single_obs_2D_data(params=None, adjust_error=False,
            automask=True, vmax=500., dispmax=600.,
            skip_crop=False, datadir=None, extra=''):

    # +++++++++++++++++++++++++++++++++++++++++++
    # Load the data set to be fit


    # Setup datadir, if set. If not set (so datadir=None), fdata must be the full path.
    if datadir is None:
        # If datadir not passed directly, look for entry in params file:
        if 'datadir'+extra in params.keys():
            datadir = params['datadir'+extra]
        elif 'datadir' in params.keys():
            datadir = params['datadir']
        # If not passed directly as kwarg, and missing from params file, set to empty -- filenames must be full path.
        if datadir is None:
            datadir = ''


    gal_vel = fits.getdata(datadir+params['fdata_vel'+extra])
    err_vel = fits.getdata(datadir+params['fdata_verr'+extra])
    if params['fitdispersion'+extra]:
        gal_disp = fits.getdata(datadir+params['fdata_disp'+extra])
        err_disp = fits.getdata(datadir+params['fdata_derr'+extra])
    if params['fitflux'+extra]:
        gal_flux = fits.getdata(datadir+params['fdata_flux'+extra])
        err_flux = fits.getdata(datadir+params['fdata_ferr'+extra])

    mask = fits.getdata(datadir+params['fdata_mask'+extra])


    # Mask NaNs:
    mask[~np.isfinite(gal_vel)] = 0
    gal_vel[~np.isfinite(gal_vel)] = 0.

    mask[~np.isfinite(err_vel)] = 0
    err_vel[~np.isfinite(err_vel)] = 0.

    if params['fitdispersion'+extra]:
        mask[~np.isfinite(gal_disp)] = 0
        gal_disp[~np.isfinite(gal_disp)] = 0.

        mask[~np.isfinite(err_disp)] = 0
        err_disp[~np.isfinite(err_disp)] = 0.
    if params['fitflux'+extra]:
        mask[~np.isfinite(gal_flux)] = 0
        gal_flux[~np.isfinite(gal_flux)] = 0.

        mask[~np.isfinite(err_disp)] = 0
        err_flux[~np.isfinite(err_flux)] = 0.

    # Auto mask som bad data
    if automask:
        indtmp = (gal_disp > dispmax) | (np.abs(gal_vel) > vmax)
        mask[indtmp] = 0



    if adjust_error:
        # Mask > 1sig lower error outliers:
        errv_l68 = np.percentile(err_vel, 15.865)
        indv = (err_vel < errv_l68)
        err_vel[indv] = errv_l68
        if params['fitdispersion'+extra]:
            errd_l68 = np.percentile(err_disp, 15.865)
            indd = (err_disp < errd_l68)
            err_disp[indd] = errd_l68
        if params['fitflux'+extra]:
            errf_l68 = np.percentile(err_flux, 15.865)
            indf = (err_flux < errf_l68)
            err_flux[indf] = errf_l68


    # Mask pixels with zero error for vel/disp:
    mask[(err_vel == 0)] = 0
    if params['fitdispersion'+extra]:
        mask[(err_disp == 0)] = 0
    if params['fitflux'+extra]:
        mask[(err_flux == 0)] = 0

    #####
    # Apply symmetrization if wanted:
    try:
        if params['symmetrize_data'+extra]:
            ybin, xbin = np.indices(gal_vel.shape, dtype=float)
            ybin = ybin.flatten()
            xbin = xbin.flatten()
            xbin -= (gal_vel.shape[1]-1.)/2.
            ybin -= (gal_vel.shape[0]-1.)/2.
            xbin -= params['xshift'+extra]
            ybin -= params['yshift'+extra]

            bool_mask = np.array(mask.copy(), dtype=bool)
            bool_mask_flat = np.array(mask.copy(), dtype=bool).flatten()

            gal_vel_flat_in = gal_vel.flatten()
            err_vel_flat_in = err_vel.flatten()
            gal_vel_flat_in[~bool_mask_flat] = np.NaN
            err_vel_flat_in[~bool_mask_flat] = np.NaN

            gal_vel_flat, err_vel_flat = dpy_utils.symmetrize_velfield(xbin, ybin,
                                gal_vel_flat_in, err_vel_flat_in,
                                sym=1, pa=params['pa'])

            gal_vel[bool_mask] = gal_vel_flat[bool_mask_flat]
            err_vel[bool_mask] = err_vel_flat[bool_mask_flat]

            if params['fitdispersion'+extra]:
                gal_disp_flat_in = gal_disp.flatten()
                err_disp_flat_in = err_disp.flatten()
                gal_disp_flat_in[~bool_mask_flat] = np.NaN
                err_disp_flat_in[~bool_mask_flat] = np.NaN
                gal_disp_flat, err_disp_flat = dpy_utils.symmetrize_velfield(xbin, ybin,
                                    gal_disp_flat_in, err_disp_flat_in,
                                    sym=2, pa=params['pa'+extra])

                gal_disp[bool_mask] = gal_disp_flat[bool_mask_flat]
                err_disp[bool_mask] = err_disp_flat[bool_mask_flat]
            if params['fitflux'+extra]:
                gal_flux_flat_in = gal_flux.flatten()
                err_flux_flat_in = err_flux.flatten()
                gal_flux_flat_in[~bool_mask_flat] = np.NaN
                err_flux_flat_in[~bool_mask_flat] = np.NaN
                gal_flux_flat, err_flux_flat = dpy_utils.symmetrize_velfield(xbin, ybin,
                                    gal_flux_flat_in, err_flux_flat_in,
                                    sym=2, pa=params['pa'+extra])

                gal_flux[bool_mask] = gal_flux_flat[bool_mask_flat]
                err_flux[bool_mask] = err_flux_flat[bool_mask_flat]

    except:
        pass

    #
    if 'weighting_method'+extra in params.keys():
        gal_weight = setup_data_weighting_method(method=params['weighting_method'+extra], r=None)
    else:
        gal_weight = None

    # #
    # if ('moment_calc'+extra in params.keys()):
    #     moment_calc = params['moment_calc'+extra]
    # else:
    #     moment_calc = False

    if ('xcenter'+extra in params.keys()):
        xcenter = params['xcenter'+extra]
    else:
        xcenter = None
    if ('ycenter'+extra in params.keys()):
        ycenter = params['ycenter'+extra]
    else:
        ycenter = None

    if params['fitdispersion'+extra]:
        file_disp = datadir+params['fdata_disp'+extra]
    else:
        file_disp = None
        gal_disp = None
        err_disp = None
    if params['fitflux'+extra]:
        try:
            file_flux = datadir+params['fdata_flux'+extra]
        except:
            file_flux = None
    else:
        file_flux = None
        gal_flux = None
        err_flux = None


    # Crop, if desired
    if not skip_crop:
        if 'cropbox'+extra in params.keys():
            if params['cropbox'+extra] is not None:
                crp = params['cropbox'+extra]
                # cropbox: l r b t
                mask = mask[crp[2]:crp[3], crp[0]:crp[1]]
                gal_vel = gal_vel[crp[2]:crp[3], crp[0]:crp[1]]
                err_vel = err_vel[crp[2]:crp[3], crp[0]:crp[1]]
                if params['fitdispersion'+extra]:
                    gal_disp = gal_disp[crp[2]:crp[3], crp[0]:crp[1]]
                    err_disp = err_disp[crp[2]:crp[3], crp[0]:crp[1]]
                if params['fitflux'+extra]:
                    gal_flux = gal_flux[crp[2]:crp[3], crp[0]:crp[1]]
                    err_flux = err_flux[crp[2]:crp[3], crp[0]:crp[1]]
                if gal_weight is not None:
                    gal_weight = gal_weight[crp[2]:crp[3], crp[0]:crp[1]]
                if xcenter is not None:
                    xcenter -= crp[0]
                    ycenter -= crp[2]
        elif params['fov_npix'+extra] < min(gal_vel.shape):
            crp_x = int(np.round((gal_vel.shape[1] - params['fov_npix'+extra])/2.))
            crp_y = int(np.round((gal_vel.shape[0] - params['fov_npix'+extra])/2.))
            gal_vel = gal_vel[crp_y:params['fov_npix'+extra]+crp_y, crp_x:params['fov_npix'+extra]+crp_x]
            err_vel = err_vel[crp_y:params['fov_npix'+extra]+crp_y, crp_x:params['fov_npix'+extra]+crp_x]
            if params['fitdispersion'+extra]:
                gal_disp = gal_disp[crp_y:params['fov_npix'+extra]+crp_y, crp_x:params['fov_npix'+extra]+crp_x]
                err_disp = err_disp[crp_y:params['fov_npix'+extra]+crp_y, crp_x:params['fov_npix'+extra]+crp_x]
            if params['fitflux'+extra]:
                gal_flux = gal_flux[crp_y:params['fov_npix'+extra]+crp_y, crp_x:params['fov_npix'+extra]+crp_x]
                err_flux = err_flux[crp_y:params['fov_npix'+extra]+crp_y, crp_x:params['fov_npix'+extra]+crp_x]

            mask = mask[crp_y:params['fov_npix'+extra]+crp_y, crp_x:params['fov_npix'+extra]+crp_x]
            if gal_weight is not None:
                gal_weight = gal_weight[crp_y:params['fov_npix'+extra]+crp_y, crp_x:params['fov_npix'+extra]+crp_x]


    data2d = data_classes.Data2D(pixscale=params['pixscale'+extra], velocity=gal_vel,
                                      vel_disp=gal_disp, vel_err=err_vel,
                                      vel_disp_err=err_disp,
                                      flux=gal_flux, flux_err=err_flux,
                                      mask=mask,
                                      weight=gal_weight,
                                      filename_velocity=datadir+params['fdata_vel'+extra],
                                      filename_dispersion=file_disp,
                                      filename_flux=file_flux,
                                      smoothing_type=params['smoothing_type'+extra],
                                      smoothing_npix=params['smoothing_npix'+extra],
                                      inst_corr=params['data_inst_corr'+extra],
                                      # moment=moment_calc,
                                      xcenter=xcenter, ycenter=ycenter)


    return data2d


def load_single_obs_3D_data(params=None, datadir=None,
            skip_mask=False, skip_automask=False,
            skip_auto_truncate_crop=False, return_crop_info=False, extra=''):
    # +++++++++++++++++++++++++++++++++++++++++++
    # Load the data set to be fit

    # Setup datadir, if set. If not set (so datadir=None), fdata must be the full path.
    if datadir is None:
        # If datadir not passed directly, look for entry in params file:
        if 'datadir'+extra in params.keys():
            datadir = params['datadir'+extra]
        elif 'datadir' in params.keys():
            datadir = params['datadir']
        # If not passed directly as kwarg, and missing from params file, set to empty -- filenames must be full path.
        if datadir is None:
            datadir = ''

    cube = fits.getdata(datadir+params['fdata_cube'+extra])
    err_cube = fits.getdata(datadir+params['fdata_err'+extra])
    header = fits.getheader(datadir+params['fdata_cube'+extra])

    mask = None
    mask_sky=None
    mask_spec=None
    if not skip_mask:
        if 'fdata_mask'+extra in params.keys():
            if params['fdata_mask'+extra] is not None:
                # Check if it's full path:
                fdata_mask = params['fdata_mask'+extra]
                if not os.path.isfile(fdata_mask):
                    # Otherwise try datadir:
                    fdata_mask = datadir+params['fdata_mask'+extra]
                if os.path.isfile(fdata_mask):
                    mask = fits.getdata(fdata_mask)
                    # Crop cube: first check if masks match the cubes already or not -- otherwise load later
                    if mask.shape != cube.shape:
                        #mask = None
                        raise ValueError

        if 'fdata_mask_sky'+extra in params.keys():
            if params['fdata_mask_sky'+extra] is not None:
                # Check if it's full path:
                fdata_mask_sky = params['fdata_mask_sky'+extra]
                if not os.path.isfile(fdata_mask_sky):
                    fdata_mask_sky = datadir+params['fdata_mask_sky'+extra]
                if os.path.isfile(fdata_mask_sky):
                    mask_sky = fits.getdata(fdata_mask_sky)
                    # Crop cube: first check if masks match the cubes already or not -- otherwise load later
                    if mask_sky.shape != cube.shape[1:3]:
                        #mask_sky = None
                        raise ValueError



        if 'fdata_mask_spec'+extra in params.keys():
            if params['fdata_mask_spec'+extra] is not None:
                # Check if it's full path:
                fdata_mask_spec = params['fdata_mask_spec'+extra]
                if not os.path.isfile(fdata_mask_spec):
                    fdata_mask_spec = datadir+params['fdata_mask_spec'+extra]
                if os.path.isfile(fdata_mask_spec):
                    mask_spec = fits.getdata(fdata_mask_spec)
                    # Crop cube: first check if masks match the cubes already or not -- otherwise load later
                    if mask_spec.shape[0] != cube.shape[0]:
                        #mask_sky = None
                        raise ValueError
                    

    if 'weighting_method'+extra in params.keys():
        gal_weight = setup_data_weighting_method(method=params['weighting_method'+extra], r=None)
    else:
        gal_weight = None

    if ('xcenter'+extra in params.keys()):
        xcenter = params['xcenter'+extra]
    else:
        xcenter = None
    #
    if ('ycenter'+extra in params.keys()):
        ycenter = params['ycenter'+extra]
    else:
        ycenter = None


    ####################################
    # Convert spectrum:
    if 'spec_orig_type'+extra in params.keys():
        spec_arr_orig = (np.arange(cube.shape[0])+1 - header['CRPIX3'])*header['CDELT3'] + header['CRVAL3']
        spec_line = params['spec_line_rest'+extra] * (1.+params['z'])
        if params['spec_orig_type'+extra].strip().upper() == 'WAVE':
            if params['spec_line_rest_unit'+extra].strip().upper() == 'ANGSTROM':
                l0 = spec_line * u.angstrom
            elif (params['spec_line_rest_unit'+extra].strip().upper() == 'MICRON') | (params['spec_line_rest_unit'+extra].strip().upper() == 'UM'):
                l0 = spec_line * u.micrometer
            #
            if (header['CUNIT3'].strip().upper() == 'MICRON') | (header['CUNIT3'].strip().upper() == 'UM'):
                spec_arr_unit = u.micrometer
            elif (header['CUNIT3'].strip().upper() == 'ANGSTROM'):
                spec_arr_unit = u.angstrom
            elif (header['CUNIT3'].strip().upper() == 'M'):
                spec_arr_unit = u.meter
            elif (header['CUNIT3'].strip().upper() == 'CM'):
                spec_arr_unit = u.centimeter

            spec_arr_wave = spec_arr_orig * spec_arr_unit

            c_kms = apy_con.c.cgs.to(u.km/u.s)

            spec_arr_tmp = (spec_arr_wave - l0.to(spec_arr_unit))/l0.to(spec_arr_unit) * c_kms
            spec_arr = spec_arr_tmp.value
            spec_unit = u.km/u.s


        elif params['spec_orig_type'+extra].strip().upper() == 'VELOCITY':
            spec_arr = spec_arr_orig
            spec_unit = u.km/u.s

            if header['CUNIT3'].strip().upper() == 'M/S':
                spec_arr /= 1000.


    else:
        # ASSUME IN KM/S
        spec_arr = (np.arange(cube.shape[0])+1 - header['CRPIX3'])*header['CDELT3'] + header['CRVAL3']
        spec_unit = u.km/u.s

        if header['CUNIT3'].strip().upper() == 'M/S':
            spec_arr /= 1000.
        elif header['CUNIT3'].strip().upper() == 'MICRON':
            raise ValueError('Assumed unit was km/s -- but does not match the cube header! CUNIT3={}'.format(header['CUNIT3']))

    if header['CUNIT1'].strip().upper() in ['DEGREE', 'DEG']:
        pscale = np.abs(header['CDELT1']) * 3600.    # convert from deg CDELT1 to arcsec
    elif header['CUNIT1'].strip().upper() in ['ARCSEC']:
        pscale = np.abs(header['CDELT1'])

    ####################################

    cube_precrop_sh = cube.shape
    wh_spec_keep = wh_ends_trim = sp_trm = None
    if (not skip_auto_truncate_crop):
        cube, err_cube, mask, mask_sky, mask_spec, \
            gal_weight, spec_arr, xcenter, ycenter, \
            wh_spec_keep, wh_ends_trim, sp_trm = _auto_truncate_crop_cube(cube,
                                                params=params,
                                                spec_type='velocity', spec_arr=spec_arr,
                                                err_cube=err_cube, mask_cube=mask,
                                                mask_sky=mask_sky, mask_spec=mask_spec,
                                                spec_unit=spec_unit,weight=gal_weight,
                                                xcenter=xcenter, ycenter=ycenter)


    ####################################
    if (mask is None) & ('auto_gen_3D_mask'+extra in params.keys()) & (not skip_automask):
        if params['auto_gen_3D_mask'+extra]:
            # if 'fdata_mask'+extra in params.keys():
            if 'fdata_mask'+extra in params.keys():
                if params['fdata_mask'+extra] is not None:
                    # Check if it's full path:
                    fdata_mask = params['fdata_mask'+extra]
                    if not os.path.isfile(fdata_mask):
                        # Otherwise try datadir:
                        fdata_mask = datadir+params['fdata_mask'+extra]
                if not os.path.isfile(fdata_mask):
                    print("Can't load mask from 'fdata_mask{}'={}, ".format(extra,fdata_mask))
                    print("  but 'auto_gen_3D_mask'={}, ".format(auto_gen_3D_mask))
                    print("  so automatically generating mask")

            mask, mask_dict = generate_3D_mask(cube=cube, err=err_cube, params=params,
                                               extra=extra)


    # Catch final cases: skip_automask=True or 'auto_gen_3D_mask' not in params.keys():
    if (mask is None):
        mask = np.ones(cube.shape)

    ####################################
    # Mask NaNs:
    mask[~np.isfinite(cube)] = 0
    cube[~np.isfinite(cube)] = -99.

    mask[~np.isfinite(cube)] = 0
    err_cube[~np.isfinite(err_cube)] = -99.

    # # Clean up 0s in error, if it's masked
    # err_cube[mask == 0] = 99.

    ####################################
    if 'smoothing_type'+extra in params.keys():
        smoothing_type=params['smoothing_type'+extra]
    else:
        smoothing_type = None
    if 'smoothing_npix'+extra in params.keys():
        smoothing_npix=params['smoothing_npix'+extra]
    else:
        smoothing_npix = 1



    data3d = data_classes.Data3D(cube, pixscale=pscale, spec_type='velocity', spec_arr=spec_arr,
                                      err_cube=err_cube, mask_cube=mask,
                                      mask_sky=mask_sky, mask_spec=mask_spec,
                                      spec_unit=u.km/u.s,
                                      weight=gal_weight,
                                      smoothing_type=smoothing_type,
                                      smoothing_npix=smoothing_npix,
                                      xcenter=xcenter, ycenter=ycenter)


    if return_crop_info:
        data3d.cube_precrop_sh = cube_precrop_sh
        data3d.wh_spec_keep = wh_spec_keep
        data3d.wh_ends_trim = wh_ends_trim
        data3d.sp_trm = sp_trm

    return data3d


def generate_3D_mask(obs=None, cube=None, err=None, params=None,
            sig_segmap_thresh=None, npix_segmap_min=None,
            snr_int_flux_thresh=None, snr_thresh_pixel=None,
            sky_var_thresh=None,
            apply_skymask_first=None,
            extra=''):

    """
    Generate a mask for a 3D cube, based on thresholds / other automated detections.

    If values are not set, they are taken from the params settings.

    See `~dysmalpy.fitting_wrappers.utils_calcs.auto_gen_3D_mask` for parameter explanations.

    Input:
        cube:               3D cube (numpy ndarray)
        err:                3D error cube (numpy ndarray)

        params:             Parameters setting dictionary

    Optional input:
        obs:                Observation instance. If cube or err is not set, will use
                            the obs.data.data, obs.data.error cubes

    Output:
            mask (3D cube), mask_dict (info about generated mask)
    """
    if cube is None:
        cube = obs.data.data.unmasked_data[:].value * obs.data.mask
    if err is None:
        err = obs.data.error.unmasked_data[:].value

    if snr_thresh_pixel is None:
        if 'auto_gen_mask_snr_thresh_pixel'+extra not in params.keys():
            params['auto_gen_mask_snr_thresh_pixel'+extra] = None
        snr_thresh_pixel = params['auto_gen_mask_snr_thresh_pixel'+extra]

    if sky_var_thresh is None:
        if 'auto_gen_mask_sky_var_thresh'+extra not in params.keys():
            params['auto_gen_mask_sky_var_thresh'+extra] = 3.
        sky_var_thresh = params['auto_gen_mask_sky_var_thresh'+extra]

    if snr_int_flux_thresh is None:
        if 'auto_gen_mask_snr_int_flux_thresh'+extra not in params.keys():
            params['auto_gen_mask_snr_int_flux_thresh'+extra] = 3.
        snr_int_flux_thresh = params['auto_gen_mask_snr_int_flux_thresh'+extra]

    if sig_segmap_thresh is None:
        if 'auto_gen_mask_sig_segmap_thresh'+extra not in params.keys():
            params['auto_gen_mask_sig_segmap_thresh'+extra] = 1.5
        sig_segmap_thresh = params['auto_gen_mask_sig_segmap_thresh'+extra]

    if npix_segmap_min is None:
        if 'auto_gen_mask_npix_segmap_min'+extra not in params.keys():
            params['auto_gen_mask_npix_segmap_min'+extra] = 5
        npix_segmap_min = params['auto_gen_mask_npix_segmap_min'+extra]

    if apply_skymask_first is None:
        if 'auto_gen_mask_apply_skymask_first'+extra not in params.keys():
            params['auto_gen_mask_apply_skymask_first'+extra] = True
        apply_skymask_first = params['auto_gen_mask_apply_skymask_first'+extra]


    msg =  "++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    msg += "  Creating 3D auto mask with the following settings: \n"
    msg += "        sig_segmap_thresh   = {} \n".format(sig_segmap_thresh)
    msg += "        npix_segmap_min     = {} \n".format(npix_segmap_min)
    msg += "        snr_int_flux_thresh = {} \n".format(snr_int_flux_thresh)
    msg += "        snr_thresh_pixel    = {} \n".format(snr_thresh_pixel)
    msg += "        sky_var_thresh      = {} \n".format(sky_var_thresh)
    msg += "        apply_skymask_first = {} \n".format(apply_skymask_first)
    msg += "++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    print(msg)

    return auto_gen_3D_mask(cube=cube, err=err,
                    sig_segmap_thresh=sig_segmap_thresh, npix_segmap_min=npix_segmap_min,
                    snr_int_flux_thresh=snr_int_flux_thresh, snr_thresh_pixel = snr_thresh_pixel,
                    sky_var_thresh=sky_var_thresh, apply_skymask_first=apply_skymask_first)




def save_3D_mask(obs=None, mask=None, filename=None,
                 overwrite=False, save_uncropped_size=True):

    """
    Generate a mask for a 3D cube, based on thresholds / other automated detections.

    If values are not set, they are taken from the params settings.

    See `~dysmalpy.fitting_wrappers.utils_calcs.auto_gen_3D_mask` for parameter explanations.

    Input:
        cube:               3D cube (numpy ndarray)
        err:                3D error cube (numpy ndarray)

        params:             Parameters setting dictionary

    Optional input:
        obs:                Observation instance. If cube or err is not set, will use
                            the obs.data.data, obs.data.error cubes

        save_uncropped_size:  If obs.data has info about pre-cropped size, will save the
                                mask w/ zero padding to match the original data cube size (eg, for saving)

    Output:
            mask (3D cube)
    """
    mask_cube = mask.copy()
    if obs is not None:
        if obs.data.data.wcs.wcs.cunit[1].to_string().upper().strip() in ['DEGREE', 'DEG']:
            pscale = np.abs(obs.data.data.wcs.wcs.cdelt[0]) * 3600.    # convert from deg CDELT1 to arcsec
        elif obs.data.data.wcs.wcs.cunit[1].to_string().upper().strip() in ['ARCSEC']:
            pscale = np.abs(obs.data.data.wcs.wcs.cdelt[0])
        pixscale=pscale
        if obs.data.data.wcs.wcs.ctype[2] == 'VOPT':
            spec_type = 'velocity'
        elif obs.data.data.wcs.wcs.ctype[2] == 'WAVE':
            spec_type = 'wavelength'

        spec_unit=u.Unit(obs.data.data.wcs.wcs.cunit[2].to_string())
        spec_arr = obs.data.data.spectral_axis.to(spec_unit).value

        if save_uncropped_size:
            mask_cube, spec_arr  = pad_3D_mask_to_uncropped_size(obs=obs, mask=mask)

    mask_cube = data_classes.Data3D(cube=mask_cube, pixscale=pixscale,
                        spec_type=spec_type, spec_arr=spec_arr, spec_unit=spec_unit)
    mask_cube.data.write(filename, overwrite=overwrite)

    return None


####
def get_ndim_fit_from_paramfile(obs_ind, params=None, param_filename=None):
    if obs_ind>0:
        ndim_fit = _get_ndim_fit_from_paramfile_with_extra(extra="_{}".format(int(obs_ind+1)), params=params,
                        param_filename=param_filename)
    else:
        try:
            ndim_fit = _get_ndim_fit_from_paramfile_with_extra(extra="", params=params,
                            param_filename=param_filename)
        except:
            ndim_fit = _get_ndim_fit_from_paramfile_with_extra(extra="_{}".format(int(obs_ind+1)), params=params,
                            param_filename=param_filename)
    return ndim_fit


def _get_ndim_fit_from_paramfile_with_extra(extra="", params=None, param_filename=None):
    if params is None:
        params = read_fitting_params(fname=param_filename)

    ndim_fit = None

    if 'fdata'+extra in params.keys():
        ndim_fit = 1
    elif 'fdata_vel'+extra in params.keys():
        ndim_fit = 2
    elif 'fdata_cube'+extra in params.keys():
        ndim_fit = 3

    if ndim_fit is None:
        # Try a final thing:
        if 'ndim'+extra in params.keys():
            ndim_fit = params['ndim'+extra]

    if ndim_fit is None:
        msg = "Could not determine fit dimension from data filenames!\n"
        msg += "  1D: params['fdata']\n"
        msg += "  2D: params['fdata_vel']\n"
        msg += "  1D: params['fdata_cube']\n"
        msg += "   OR SET params['ndim']"
        raise ValueError(msg)

    return ndim_fit

def stub_paramfile_dir(param_filename):
    try:
        delim = os.sep
        #delim = '/'
        # Strip dir from param_filename
        pf_arr = param_filename.split(delim)
        if len(pf_arr) > 1:
            param_dir = delim.join(pf_arr[:-1]) + delim
        else:
            param_dir = os.getcwd() + delim
    except:
        raise ValueError("Problem getting directory of paramfile={}".format(param_filename))

    return param_dir

def check_outdir_specified(params, outdir, param_filename=None):
    try:
        try:
            if os.path.isabs(outdir):
                stub_paramfilepath = False
            else:
                stub_paramfilepath = True
        except:
            print("Performing string splitting")
            delim = os.sep
            od_arr = outdir.split(delim)
            od_arr_nonempt = []
            for od_d in od_arr:
                if len(od_d) > 0:
                    od_arr_nonempt.append(od_d)

            # If only a SINGLE relative path specified, prepend the param directory
            if len(od_arr_nonempt) == 1:
                stub_paramfilepath = True
            else:
                stub_paramfilepath = False

        if stub_paramfilepath:
            # Strip dir from param_filename
            param_dir = stub_paramfile_dir(param_filename)
            outdir = param_dir+outdir
            params['outdir'] = outdir
    except:
        raise ValueError("Directory {} not found! Couldn't get outdir.".format(outdir))

    return outdir, params

def check_datadir_specified(params, datadir, ndim=None, param_filename=None):
    if ndim is None:
        raise ValueError("Must specify 'ndim'!")
    if ndim == 1:
        fdata_orig = params['fdata']
    elif ndim == 2:
        fdata_orig = params['fdata_vel']
    elif ndim == 3:
        fdata_orig = params['fdata_cube']

    if datadir is not None:
        fdata = "{}{}".format(datadir, fdata_orig)
    else:
        # Try case of absolute path for filenames
        fdata = fdata_orig
        datadir = None

    if not os.path.isfile(fdata):
        # Try relative WRT current dir
        datadir = os.getcwd() + os.sep
        fdata = "{}{}".format(datadir, fdata_orig)

    if not os.path.isfile(fdata):
        # Strip dir from param_filename
        datadir = stub_paramfile_dir(param_filename)
        fdata = "{}{}".format(datadir, fdata_orig)

        if os.path.isfile(fdata):
            params['datadir'] = datadir
        else:
            try:
                datadir = tkinter_io.get_datadir_tkinter(ndim=ndim)
                params['datadir'] = datadir
            except:
                raise ValueError("Data file {} not found! Couldn't get datadir from dialog window.".format(fdata))

    return datadir, params


def preserve_param_file(param_filename, params=None, datadir=None, outdir=None):
    # Copy paramfile that is OS independent
    param_filename_nopath = param_filename.split(os.sep)[-1].split('/')[-1]
    # FORCE, as python will often use "/" even for windows!
    galID_strp = "".join(params['galID'].strip().split("_"))
    galID_strp = "".join(galID_strp.split("-"))
    galID_strp = "".join(galID_strp.split(" "))
    paramfile_strp = "".join(param_filename_nopath.strip().split("_"))
    paramfile_strp = "".join(paramfile_strp.split("-"))
    paramfile_strp = "".join(paramfile_strp.split(" "))


    if galID_strp.strip().lower() in paramfile_strp.strip().lower():
        # Already has galID in param filename:
        fout_name = outdir+param_filename_nopath
    else:
        # Copy, prepending galID
        fout_name = outdir+"{}_{}".format(params['galID'], param_filename_nopath)


    # Check if file already exists in output directory:
    if not os.path.isfile(fout_name):
        # Replace datadir, outdir:
        with open(param_filename, 'r') as f:
            lines = f.readlines()

        for i,l in enumerate(lines):
            #if 'datadir' in l:
            ll = l.split('#')[0]
            if 'datadir' in ll:
                larr = ll.split(',')
                lines[i] = l.replace(larr[1].strip(), "{}".format(datadir))
            if 'outdir' in ll:
                larr = ll.split(',')
                lines[i] = l.replace(larr[1].strip(), "{}".format(outdir))

        with open(fout_name, 'w') as fnew:
            fnew.writelines(lines)

