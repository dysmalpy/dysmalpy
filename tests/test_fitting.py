# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Testing of DYSMALPY fitting calculations
#    + Primarily using the FITTING_WRAPPER functionality, as a shortcut

import os
import shutil

import math

import numpy as np

from dysmalpy.fitting_wrappers import dysmalpy_fit_single
from dysmalpy.fitting_wrappers import utils_io as fw_utils_io
from dysmalpy import fitting


# TESTING DIRECTORY
path = os.path.abspath(__file__)
_dir_tests = os.path.dirname(path) + os.sep
_dir_tests_data = _dir_tests+'test_data' + os.sep


def read_params(param_filename=None):
    param_filename_full=_dir_tests_data+param_filename
    params = fw_utils_io.read_fitting_params(fname=param_filename_full)
    return params

def run_fit(param_filename=None):
    param_filename_full=_dir_tests_data+param_filename

    # Delete existing folder:
    params = fw_utils_io.read_fitting_params(fname=param_filename_full)
    outdir = _dir_tests_data+params['outdir']
    if os.path.isdir(outdir):
        shutil.rmtree(outdir, ignore_errors=True)

    # Run fit
    dysmalpy_fit_single.dysmalpy_fit_single(param_filename=param_filename_full)

def check_output_files(outdir, list_files):
    outfiles = os.listdir(outdir)
    for lf in list_files:
        # Assert all files in the expected list are found in the output directory
        assert lf in outfiles


def check_bestfit_values(results, dict_bf_values, fit_method=None, ndim=None, ftol=5.e-3):
    for compn in dict_bf_values.keys():
        for paramn in dict_bf_values[compn].keys():
            whm = np.where((results['component'].str.strip() == compn) & \
                           (results['param_name'].str.strip() == paramn))[0]
            if len(whm) == 1:
                # Assert best-fit values are the same
                assert math.isclose(float(results['best_value'].iloc[whm[0]]),
                            dict_bf_values[compn][paramn], rel_tol=ftol), \
                            '{}, ndim={}, {}:{}'.format(fit_method,ndim,compn,paramn)
            else:
                raise ValueError('{}, ndim={}, {}:{}'.format(fit_method,ndim,compn,paramn))


def expected_output_files_base(galID, param_filename=None, fit_method=None, ndim=None):
    fit_method = fit_method.lower().strip()
    list_files = ['{}_{}'.format(galID, param_filename),
                  '{}_{}.log'.format(galID, fit_method),
                  '{}_model.pickle'.format(galID),
                  '{}_{}_results.pickle'.format(galID, fit_method),
                  '{}_{}_bestfit_results.dat'.format(galID, fit_method),
                  '{}_{}_bestfit_results_report.info'.format(galID, fit_method),
                  '{}_OBS_bestfit_cube.fits'.format(galID),
                  '{}_bestfit_menc.dat'.format(galID),
                  '{}_bestfit_vcirc.dat'.format(galID),
                  '{}_LINE_bestfit_velprofile.dat'.format(galID)]
    if ndim < 3:
        list_files.append('{}_{}_bestfit_OBS.pdf'.format(galID, fit_method))

    if fit_method == 'mpfit':
        pass

    elif fit_method == 'mcmc':
        list_files.append('{}_{}_burnin_trace.pdf'.format(galID, fit_method))
        list_files.append('{}_{}_trace.pdf'.format(galID, fit_method))
        list_files.append('{}_{}_param_corner.pdf'.format(galID, fit_method))
        list_files.append('{}_{}_sampler.h5'.format(galID, fit_method))
        list_files.append('{}_{}_chain_blobs.dat'.format(galID, fit_method))

    return list_files

def expected_output_files_1D(galID, param_filename=None, fit_method=None):
    list_files = expected_output_files_base(galID, param_filename=param_filename, fit_method=fit_method, ndim=1)

    list_files.append('{}_OBS_out-1dplots.txt'.format(galID))
    list_files.append('{}_OBS_out-1dplots_finer_sampling.txt'.format(galID))
    list_files.append('{}_OBS_rot_components.pdf'.format(galID))
    list_files.append('{}_menc_tot_bary_dm.dat'.format(galID))
    list_files.append('{}_vcirc_tot_bary_dm.dat'.format(galID))

    return list_files

def expected_output_files_2D(galID, param_filename=None, fit_method=None):
    list_files = expected_output_files_base(galID, param_filename=param_filename, fit_method=fit_method, ndim=2)

    list_files.append('{}_OBS_out-velmaps.fits'.format(galID))
    return list_files

def expected_output_files_3D(galID, param_filename=None, fit_method=None):
    list_files = expected_output_files_base(galID, param_filename=param_filename, fit_method=fit_method, ndim=3)

    list_files.append('{}_OBS_out-cube.fits'.format(galID))
    list_files.append('{}_{}_bestfit_OBS_apertures.pdf'.format(galID, fit_method))
    list_files.append('{}_{}_bestfit_OBS_channels.pdf'.format(galID, fit_method))
    list_files.append('{}_{}_bestfit_OBS_spaxels.pdf'.format(galID, fit_method))
    list_files.append('{}_{}_bestfit_OBS_extract_1D.pdf'.format(galID, fit_method))
    list_files.append('{}_{}_bestfit_OBS_extract_2D.pdf'.format(galID, fit_method))

    return list_files

class TestFittingWrappers:

    def test_1D_mpfit(self):
        param_filename = 'fitting_1D_mpfit.params'
        params = read_params(param_filename=param_filename)
        outdir_full = _dir_tests_data+params['outdir']

        run_fit(param_filename=param_filename)

        # Make sure all files exist:
        list_files = expected_output_files_1D(params['galID'], param_filename=param_filename,
                            fit_method=params['fit_method'])
        check_output_files(outdir_full, list_files)

        # Load output, check results
        f_ascii_machine = outdir_full+'{}_{}_bestfit_results.dat'.format(params['galID'],
                                    params['fit_method'].strip().lower())
        results = fw_utils_io.read_results_ascii_file(fname=f_ascii_machine)

        dict_bf_values = {'disk+bulge': {'total_mass': 10.9137,
                                         'r_eff_disk': 3.4011,
                                         'bt': 0.0102},
                          'halo': {'fdm': 0.1245},
                          'dispprof_LINE': {'sigma0': 35.8929}}

        # Check that best-fit values are the same
        check_bestfit_values(results, dict_bf_values, fit_method=params['fit_method'], ndim=1)



    def test_1D_mcmc(self):
        param_filename = 'fitting_1D_mcmc.params'
        params = read_params(param_filename=param_filename)
        outdir_full = _dir_tests_data+params['outdir']

        run_fit(param_filename=param_filename)

        # Make sure all files exist:
        list_files = expected_output_files_1D(params['galID'], param_filename=param_filename,
                            fit_method=params['fit_method'])
        check_output_files(outdir_full, list_files)

        # Load output, check results: DID WALKERS MOVE?
        f_galmodel = outdir_full+'{}_model.pickle'.format(params['galID'])
        f_results = outdir_full+'{}_{}_results.pickle'.format(params['galID'],
                            params['fit_method'].strip().lower())
        f_sampler = outdir_full+'{}_{}_sampler.h5'.format(params['galID'],
                            params['fit_method'].strip().lower())
        gal, results = fitting.reload_all_fitting(filename_galmodel=f_galmodel,
                            filename_results=f_results, fit_method=params['fit_method'])
        results.reload_sampler(filename=f_sampler)

        # Assert lnprob values are all finite:
        assert np.sum(np.isfinite(results.sampler['flatlnprobability'])) == results.sampler['flatchain'].shape[0]

        for i in range(results.sampler['nParam']):
            # Assert at least one walker moved at least once for parameter i
            assert len(np.unique(results.sampler['flatchain'][:,i])) > results.sampler['nWalkers']
            # Assert all values of parameter i are finite:
            assert np.sum(np.isfinite(results.sampler['flatchain'][:,i])) == results.sampler['flatchain'].shape[0]


    def test_2D_mpfit(self):
        param_filename = 'fitting_2D_mpfit.params'
        params = read_params(param_filename=param_filename)
        outdir_full = _dir_tests_data+params['outdir']

        run_fit(param_filename=param_filename)

        # Make sure all files exist:
        list_files = expected_output_files_2D(params['galID'], param_filename=param_filename,
                            fit_method=params['fit_method'])
        check_output_files(outdir_full, list_files)

        # Load output, check results
        f_ascii_machine = outdir_full+'{}_{}_bestfit_results.dat'.format(params['galID'],
                                    params['fit_method'].strip().lower())
        results = fw_utils_io.read_results_ascii_file(fname=f_ascii_machine)

        dict_bf_values = {'disk+bulge': {'total_mass': 10.7086,
                                         'r_eff_disk': 3.6318,
                                         'bt': 0.6920},
                          'halo': {'fdm': 0.2760},
                          'dispprof_LINE': {'sigma0': 35.4497}}

        # Check that best-fit values are the same
        check_bestfit_values(results, dict_bf_values, fit_method=params['fit_method'], ndim=2)


    def test_3D_mpfit(self):
        param_filename = 'fitting_3D_mpfit.params'
        params = read_params(param_filename=param_filename)
        outdir_full = _dir_tests_data+params['outdir']
        #print("gauss_extract_with_c = {}",format(params['gauss_extract_with_c']))
        run_fit(param_filename=param_filename)

        # Make sure all files exist:
        list_files = expected_output_files_3D(params['galID'], param_filename=param_filename,
                            fit_method=params['fit_method'])
        check_output_files(outdir_full, list_files)

        # Load output, check results
        f_ascii_machine = outdir_full+'{}_{}_bestfit_results.dat'.format(params['galID'],
                                    params['fit_method'].strip().lower())
        results = fw_utils_io.read_results_ascii_file(fname=f_ascii_machine)

        dict_bf_values = {'disk+bulge': {'total_mass': 10.7764,
                                         'r_eff_disk': 2.7922,
                                         'bt': 0.1196},
                          'halo': {'fdm': 0.2338},
                          'dispprof_LINE': {'sigma0': 67.7606}}

        # Check that best-fit values are the same
        check_bestfit_values(results, dict_bf_values, fit_method=params['fit_method'], ndim=3)
