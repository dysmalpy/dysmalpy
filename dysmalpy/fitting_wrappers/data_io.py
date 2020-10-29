# Methods for loading data for fitting wrappers

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys

import datetime

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as apy_con

from dysmalpy import data_classes
from dysmalpy import utils as dysmalpy_utils

import astropy.io.fits as fits

from dysmalpy.fitting_wrappers.utils_calcs import auto_gen_3D_mask
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
                                tvn = np.float(tv)
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
                else:
                    try:
                        fltval = np.float(tmpval)
                        if (fltval % 1) == 0.:
                            tmpval = np.int(fltval)
                        else:
                            tmpval = fltval
                    except:
                        tmpval = strtmpval.strip()

            params[key] = tmpval

    return params
    #

def read_fitting_params(fname=None):
    if fname is None:
        raise ValueError("parameter filename {} not found!".format(fname))

    # READ FILE HERE!
    param_input = read_fitting_params_input(fname=fname)

    # Set some defaults if not otherwise specified
    params = {'nWalkers': 20,
              'nCPUs': 4,
              'scale_param_a': 2,
              'nBurn': 10,
              'nSteps': 10,
              'minAF': None,
              'maxAF': None,
              'nEff': 10,
              'do_plotting': True,
              'oversample': 1,
              'fitdispersion': True,
              'include_halo': False,
              'halo_profile_type': 'NFW',
              'blob_name': None,
              'red_chisq': False,
              'oversampled_chisq': True,
              'linked_posteriors': None,
              'weighting_method': None}

    # param_filename
    fname_split = fname.split('/')
    params['param_filename'] = fname_split[-1]

    for key in param_input.keys():
        params[key] = param_input[key]

    # Catch depreciated case:
    if 'halo_inner_slope_fit' in params.keys():
        if params['halo_inner_slope_fit']:
            if params['halo_profile_type'].upper() == 'NFW':
                print("using depreciated param setting 'halo_inner_slope_fit=True'.")
                print("Assuming 'halo_profile_type=TwoPowerHalo' halo form.")
                params['halo_profile_type'] = 'TwoPowerHalo'

    # Catch other cases:
    if params['include_halo']:
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
            elif ((params['halo_profile_type'].lower() == 'twopowerhalo') | \
                        (params['halo_profile_type'].lower() == 'burkert')):
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

def save_results_ascii_files(fit_results=None, gal=None, params=None):

    outdir = params['outdir']
    galID = params['galID']

    if params['fit_method'] == 'mcmc':

        save_results_ascii_files_mcmc(fit_results=fit_results, gal=gal, params=params, outdir=outdir, galID=galID)



    elif params['fit_method'] == 'mpfit':
        save_results_ascii_files_mpfit(fit_results=fit_results, gal=gal, params=params, outdir=outdir, galID=galID)



    return None

#
def save_results_ascii_files_mcmc(fit_results=None, gal=None, params=None, outdir=None, galID=None):

    f_ascii_machine = outdir+'{}_mcmc_best_fit_results.dat'.format(galID)

    f_ascii_pretty = outdir+'{}_mcmc_best_fit_results.info'.format(galID)


    # --------------------------------------------
    if 'blob_name' in params.keys():
        blob_name = params['blob_name']
        if isinstance(blob_name, str):
            blob_names = [blob_name]
        else:
            blob_names = blob_name[:]

    # --------------------------------------------


    with open(f_ascii_machine, 'w') as f:
        namestr = '# component    param_name    fixed    best_value   l68_err   u68_err'
        f.write(namestr+'\n')

        for cmp_n in gal.model.param_names.keys():
            for param_n in gal.model.param_names[cmp_n]:

                if '{}:{}'.format(cmp_n,param_n) in fit_results.chain_param_names:
                    whparam = np.where(fit_results.chain_param_names == '{}:{}'.format(cmp_n, param_n))[0][0]
                    best = fit_results.bestfit_parameters[whparam]
                    l68 = fit_results.bestfit_parameters_l68_err[whparam]
                    u68 = fit_results.bestfit_parameters_u68_err[whparam]
                else:
                    best = getattr(gal.model.components[cmp_n], param_n).value
                    l68 = -99.
                    u68 = -99.

                datstr = '{: <12}   {: <11}   {: <5}   {:9.4f}   {:9.4f}   {:9.4f}'.format(cmp_n, param_n,
                            "{}".format(gal.model.fixed[cmp_n][param_n]), best, l68, u68)
                f.write(datstr+'\n')

        ###

        if 'blob_name' in params.keys():
            for blobn in blob_names:
                blob_best = fit_results.__dict__['bestfit_{}'.format(blobn)]
                l68_blob = fit_results.__dict__['bestfit_{}_l68_err'.format(blobn)]
                u68_blob = fit_results.__dict__['bestfit_{}_u68_err'.format(blobn)]
                datstr = '{: <12}   {: <11}   {: <5}   {:9.4f}   {:9.4f}   {:9.4f}'.format(blobn, '-----',
                            '-----', blob_best, l68_blob, u68_blob)
                f.write(datstr+'\n')


        ###
        datstr = '{: <12}   {: <11}   {: <5}   {}   {:9.4f}   {:9.4f}'.format('adiab_contr', '-----',
                    '-----', gal.model.kinematic_options.adiabatic_contract, -99, -99)
        f.write(datstr+'\n')

        datstr = '{: <12}   {: <11}   {: <5}   {:9.4f}   {:9.4f}   {:9.4f}'.format('redchisq', '-----',
                    '-----', fit_results.bestfit_redchisq, -99, -99)
        f.write(datstr+'\n')

        if 'profile1d_type' in params.keys():
            datstr = '{: <12}   {: <11}   {: <5}   {: <20}   {:9.4f}   {:9.4f}'.format('profile1d_type', '-----',
                        '-----', params['profile1d_type'], -99, -99)
            f.write(datstr+'\n')

        #
        if 'weighting_method' in params.keys():
            if params['weighting_method'] is not None:
                datstr = '{: <12}   {: <11}   {: <5}   {: <20}   {:9.4f}   {:9.4f}'.format('weighting_method', '-----',
                            '-----', params['weighting_method'], -99, -99)
                f.write(datstr+'\n')

        if 'moment_calc' in params.keys():
            datstr = '{: <12}   {: <11}   {: <5}   {: <20}   {:9.4f}   {:9.4f}'.format('moment_calc', '-----',
                        '-----', params['moment_calc'], -99, -99)
            f.write(datstr+'\n')

        #
        if 'partial_weight' in params.keys():
            datstr = '{: <12}   {: <11}   {: <5}   {: <20}   {:9.4f}   {:9.4f}'.format('partial_weight', '-----',
                        '-----', params['partial_weight'], -99, -99)
            f.write(datstr+'\n')
        #
        # INFO on pressure support type:
        datstr = '{: <12}   {: <11}   {: <5}   {: <20}   {:9.4f}   {:9.4f}'.format('pressure_support_type', '-----',
                    '-----', gal.model.kinematic_options.pressure_support_type, -99, -99)
        f.write(datstr+'\n')

    #
    with open(f_ascii_pretty, 'w') as f:
        f.write('###############################'+'\n')
        f.write(' Fitting for {}'.format(params['galID'])+'\n')
        f.write('\n')

        f.write("Date: {}".format(datetime.datetime.now())+'\n')
        f.write('\n')

        try:
            f.write('Datafile: {}'.format(params['fdata'])+'\n')
        except:
            try:
                f.write('Datafiles:\n')
                f.write(' vel:  {}'.format(params['fdata_vel'])+'\n')
                f.write(' verr: {}'.format(params['fdata_verr'])+'\n')
                f.write(' disp: {}'.format(params['fdata_disp'])+'\n')
                f.write(' derr: {}'.format(params['fdata_derr'])+'\n')
                try:
                    f.write(' mask: {}'.format(params['fdata_mask'])+'\n')
                except:
                    pass
            except:
                pass
        f.write('Paramfile: {}'.format(params['param_filename'])+'\n')

        f.write('\n')
        f.write('Fitting method: {}'.format(params['fit_method'].upper()))
        f.write('\n')
        if 'fit_module' in params.keys():
            f.write('   fit_module: {}'.format(params['fit_module']))
            f.write('\n')
        f.write('\n')
        # --------------------------------------
        if 'profile1d_type' in params.keys():
            f.write('profile1d_type: {}'.format(params['profile1d_type']))
            f.write('\n')
        if 'weighting_method' in params.keys():
            if params['weighting_method'] is not None:
                f.write('weighting_method: {}'.format(params['weighting_method']))
                f.write('\n')
        if 'moment_calc' in params.keys():
            f.write('moment_calc: {}'.format(params['moment_calc']))
            f.write('\n')
        if 'partial_weight' in params.keys():
            f.write('partial_weight: {}'.format(params['partial_weight']))
            f.write('\n')
        #
        # INFO on pressure support type:
        f.write('pressure_support_type: {}'.format(gal.model.kinematic_options.pressure_support_type))
        f.write('\n')
        # --------------------------------------
        f.write('\n')
        f.write('###############################'+'\n')
        f.write(' Fitting results'+'\n')

        for cmp_n in gal.model.param_names.keys():
            f.write('-----------'+'\n')
            f.write(' {}'.format(cmp_n)+'\n')

            nfree = 0
            nfixedtied = 0

            for param_n in gal.model.param_names[cmp_n]:

                if '{}:{}'.format(cmp_n,param_n) in fit_results.chain_param_names:
                    nfree += 1
                    whparam = np.where(fit_results.chain_param_names == '{}:{}'.format(cmp_n, param_n))[0][0]
                    best = fit_results.bestfit_parameters[whparam]
                    l68 = fit_results.bestfit_parameters_l68_err[whparam]
                    u68 = fit_results.bestfit_parameters_u68_err[whparam]


                    datstr = '    {: <11}    {:9.4f}  -{:9.4f} +{:9.4f}'.format(param_n, best, l68, u68)
                    f.write(datstr+'\n')
                else:
                    nfixedtied += 1
            #
            if (nfree > 0) & (nfixedtied > 0):
                f.write('\n')
            #
            for param_n in gal.model.param_names[cmp_n]:

                if '{}:{}'.format(cmp_n,param_n) not in fit_results.chain_param_names:
                    best = getattr(gal.model.components[cmp_n], param_n).value

                    if not getattr(gal.model.components[cmp_n], param_n).tied:
                        if getattr(gal.model.components[cmp_n], param_n).fixed:
                            fix_tie = '[FIXED]'
                        else:
                            fix_tie = '[UNKNOWN]'
                    else:
                        fix_tie = '[TIED]'

                    datstr = '    {: <11}    {:9.4f}  {}'.format(param_n, best, fix_tie)
                    f.write(datstr+'\n')


        ####
        if 'blob_name' in params.keys():
            blob_name = params['blob_name']
            if isinstance(blob_name, str):
                blob_names = [blob_name]
            else:
                blob_names = blob_name[:]

            f.write('\n')
            f.write('-----------'+'\n')
            for blobn in blob_names:
                blob_best = fit_results.__dict__['bestfit_{}'.format(blobn)]
                l68_blob = fit_results.__dict__['bestfit_{}_l68_err'.format(blobn)]
                u68_blob = fit_results.__dict__['bestfit_{}_u68_err'.format(blobn)]
                datstr = '    {: <11}    {:9.4f}  -{:9.4f} +{:9.4f}'.format(blobn, blob_best, l68_blob, u68_blob)
                f.write(datstr+'\n')


        ####
        f.write('\n')
        f.write('-----------'+'\n')
        datstr = 'Adiabatic contraction: {}'.format(gal.model.kinematic_options.adiabatic_contract)
        f.write(datstr+'\n')

        f.write('\n')
        f.write('-----------'+'\n')
        datstr = 'Red. chisq: {:0.4f}'.format(fit_results.bestfit_redchisq)
        f.write(datstr+'\n')

        try:
            Routmax2D = calc_Rout_max_2D(gal=gal, fit_results=fit_results)
            f.write('\n')
            f.write('-----------'+'\n')
            datstr = 'Rout,max,2D: {:0.4f}'.format(Routmax2D)
            f.write(datstr+'\n')
        except:
            pass




        f.write('\n')


    return None

#
def save_results_ascii_files_mpfit(fit_results=None, gal=None, params=None, outdir=None, galID=None):
    f_ascii_machine = outdir + '{}_mpfit_best_fit_results.dat'.format(galID)

    f_ascii_pretty = outdir + '{}_mpfit_best_fit_results.info'.format(galID)

    with open(f_ascii_machine, 'w') as f:
        namestr = '# component    param_name    fixed    best_value   error'
        f.write(namestr + '\n')

        for cmp_n in gal.model.param_names.keys():
            for param_n in gal.model.param_names[cmp_n]:

                if '{}:{}'.format(cmp_n, param_n) in fit_results.chain_param_names:
                    whparam = \
                    np.where(fit_results.chain_param_names == '{}:{}'.format(cmp_n, param_n))[
                        0][0]
                    best = fit_results.bestfit_parameters[whparam]
                    err = fit_results.bestfit_parameters_err[whparam]
                else:
                    best = getattr(gal.model.components[cmp_n], param_n).value
                    err = -99

                datstr = '{: <12}   {: <11}   {: <5}   {:9.4f}   {:9.4f}'.format(
                    cmp_n, param_n,
                    "{}".format(gal.model.fixed[cmp_n][param_n]), best, err)
                f.write(datstr + '\n')

        #
        datstr = '{: <12}   {: <11}   {: <5}   {:9.4f}   {:9.4f}'.format('redchisq',
                                                                                   '-----',
                                                                                   '-----',
                                                                                   fit_results.bestfit_redchisq,
                                                                                   -99)
        f.write(datstr + '\n')

    #
    with open(f_ascii_pretty, 'w') as f:
        f.write('###############################' + '\n')
        f.write(' Fitting for {}'.format(params['galID']) + '\n')
        f.write('\n')

        f.write("Date: {}".format(datetime.datetime.now()) + '\n')
        f.write('\n')

        try:
            f.write('Datafile: {}'.format(params['fdata']) + '\n')
        except:
            try:
                f.write('Datafiles:\n')
                f.write(' vel:  {}'.format(params['fdata_vel']) + '\n')
                f.write(' verr: {}'.format(params['fdata_verr']) + '\n')
                f.write(' disp: {}'.format(params['fdata_disp']) + '\n')
                f.write(' derr: {}'.format(params['fdata_derr']) + '\n')
                try:
                    f.write(' mask: {}'.format(params['fdata_mask']) + '\n')
                except:
                    pass
            except:
                pass
        f.write('Paramfile: {}'.format(params['param_filename']) + '\n')

        f.write('\n')
        f.write('Fitting method: {}'.format(params['fit_method'].upper()))
        f.write('\n')
        if 'fit_module' in params.keys():
            f.write('   fit_module: {}'.format(params['fit_module']))
            f.write('\n')
        f.write('\n')
        # --------------------------------------
        if 'profile1d_type' in params.keys():
            f.write('profile1d_type: {}'.format(params['profile1d_type']))
            f.write('\n')
        if 'weighting_method' in params.keys():
            if params['weighting_method'] is not None:
                f.write('weighting_method: {}'.format(params['weighting_method']))
                f.write('\n')
        if 'moment_calc' in params.keys():
            f.write('moment_calc: {}'.format(params['moment_calc']))
            f.write('\n')
        if 'partial_weight' in params.keys():
            f.write('partial_weight: {}'.format(params['partial_weight']))
            f.write('\n')

        # INFO on pressure support type:
        #if 'pressure_support_type' in params.keys():
        f.write('pressure_support_type: {}'.format(gal.model.kinematic_options.pressure_support_type))
        f.write('\n')

        # --------------------------------------
        f.write('\n')
        f.write('###############################' + '\n')
        f.write(' Fitting results' + '\n')

        for cmp_n in gal.model.param_names.keys():
            f.write('-----------' + '\n')
            f.write(' {}'.format(cmp_n) + '\n')

            nfree = 0
            nfixedtied = 0

            for param_n in gal.model.param_names[cmp_n]:

                if '{}:{}'.format(cmp_n, param_n) in fit_results.chain_param_names:
                    nfree += 1
                    whparam = \
                    np.where(fit_results.chain_param_names == '{}:{}'.format(cmp_n, param_n))[
                        0][0]
                    best = fit_results.bestfit_parameters[whparam]
                    err = fit_results.bestfit_parameters_err[whparam]

                    datstr = '    {: <11}    {:9.4f}  +/-{:9.4f}'.format(param_n, best,
                                                                                err)
                    f.write(datstr + '\n')
                else:
                    nfixedtied += 1

            #
            if (nfree > 0) & (nfixedtied > 0):
                f.write('\n')
            #
            for param_n in gal.model.param_names[cmp_n]:

                if '{}:{}'.format(cmp_n, param_n) not in fit_results.chain_param_names:
                    best = getattr(gal.model.components[cmp_n], param_n).value

                    if not getattr(gal.model.components[cmp_n], param_n).tied:
                        if getattr(gal.model.components[cmp_n], param_n).fixed:
                            fix_tie = '[FIXED]'
                        else:
                            fix_tie = '[UNKNOWN]'
                    else:
                        fix_tie = '[TIED]'

                    datstr = '    {: <11}    {:9.4f}  {}'.format(param_n, best, fix_tie)
                    f.write(datstr + '\n')

        #
        f.write('\n')
        f.write('-----------' + '\n')
        datstr = 'Red. chisq: {:0.4f}'.format(fit_results.bestfit_redchisq)
        f.write(datstr + '\n')

        try:
            Routmax2D = calc_Rout_max_2D(gal=gal, fit_results=fit_results)
            f.write('\n')
            f.write('-----------'+'\n')
            datstr = 'Rout,max,2D: {:0.4f}'.format(Routmax2D)
            f.write(datstr+'\n')
        except:
            pass

        f.write('\n')

    return None



def read_results_ascii_file(fname=None):


    names = ['component', 'param_name', 'fixed', 'best_value', 'l68_err', 'u68_err']

    data = pd.read_csv(fname, sep=' ', comment='#', names=names, skipinitialspace=True,
                    index_col=False)


    return data


def write_bestfit_1d_obs_file(gal=None, fname=None):
    """
    Short function to save *observed* space 1D obs profile for a galaxy (eg, for plotting, etc)
    Follows form of H.Ü. example.
    """
    model_r = gal.model_data.rarr
    model_flux = gal.model_data.data['flux']
    model_vel = gal.model_data.data['velocity']
    model_disp = gal.model_data.data['dispersion']

    # Write 1D circular aperture plots to text file
    np.savetxt(fname, np.transpose([model_r, model_flux, model_vel, model_disp]),
               fmt='%2.4f\t%2.4f\t%5.4f\t%5.4f',
               header='r [arcsec], flux [...], vel [km/s], disp [km/s]')


    return None

def read_bestfit_1d_obs_file(fname=None, mirror=False):
    """
    Short function to save load space 1D obs profile for a galaxy (eg, for plotting, etc)
    Follows form of H.Ü. example.
    """

    # Load the model file
    dat_arr =   np.loadtxt(fname)
    gal_r =     dat_arr[:,0]
    gal_flux =  dat_arr[:,1]
    gal_vel =   dat_arr[:,2]
    gal_disp =  dat_arr[:,3]

    slit_width = None
    slit_pa = None

    if mirror:
        gal_r = np.append(-1.*gal_r[::-1][:-1], gal_r)
        gal_flux = np.append(1.*gal_flux[::-1][:-1], gal_flux)
        gal_vel = np.append(-1.*gal_vel[::-1][:-1], gal_vel)
        gal_disp = np.append(1.*gal_disp[::-1][:-1], gal_disp)

    #
    model_data = data_classes.Data1D(r=gal_r, velocity=gal_vel,
                             vel_disp=gal_disp, flux=gal_flux,
                             slit_width=slit_width,
                             slit_pa=slit_pa)
    model_data.apertures = None

    return model_data


def make_catalog_row_entry(ascii_data=None, galID=None):

    params = ['total_mass', 'r_eff_disk', 'bt', 'mvirial', 'conc', 'sigma0']
    extra_params = ['f_DM_RE']

    data = pd.DataFrame({'galID': galID},
                         index=[0])

    for par in params:
        whrow = np.where((ascii_data['param_name'].str.strip()==par))[0][0]
        data[par] = ascii_data['best_value'].iloc[whrow]
        data[par+"_u68_err"] = ascii_data['u68_err'].iloc[whrow]
        data[par+"_l68_err"] = ascii_data['l68_err'].iloc[whrow]

    for par in extra_params:
        whrow = np.where((ascii_data['component'].str.strip()==par))[0][0]
        data[par] = ascii_data['best_value'].iloc[whrow]
        data[par+"_u68_err"] = ascii_data['u68_err'].iloc[whrow]
        data[par+"_l68_err"] = ascii_data['l68_err'].iloc[whrow]




    return data





def load_single_object_1D_data(fdata=None, fdata_mask=None, params=None, datadir=None):

    # Setup datadir, if set. If not set (so datadir=None), fdata must be the full path.
    if datadir is None:
        # If datadir not passed directly, look for entry in params file:
        if 'datadir' in params.keys():
            datadir = params['datadir']
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
        if params['symmetrize_data']:
            gal_r_new, gal_vel, err_vel = dysmalpy_utils.symmetrize_1D_profile(gal_r, gal_vel, err_vel, sym=1)
            gal_r, gal_disp, err_disp = dysmalpy_utils.symmetrize_1D_profile(gal_r, gal_disp, err_disp, sym=2)
    except:
        pass


    if 'weighting_method' in params.keys():
        gal_weight = setup_data_weighting_method(method=params['weighting_method'], r=gal_r)
    else:
        gal_weight = None
    #
    if ('xcenter' in params.keys()):
        xcenter = params['xcenter']
    else:
        xcenter = None
    #
    if ('ycenter' in params.keys()):
        ycenter = params['ycenter']
    else:
        ycenter = None

    data1d = data_classes.Data1D(r=gal_r, velocity=gal_vel,vel_disp=gal_disp,
                                vel_err=err_vel, vel_disp_err=err_disp,
                                weight=gal_weight,
                                mask_velocity=msk_vel, mask_vel_disp=msk_disp,
                                slit_width=params['slit_width'],
                                slit_pa=params['slit_pa'], inst_corr=params['data_inst_corr'],
                                xcenter=xcenter, ycenter=ycenter)

    return data1d

def load_single_object_2D_data(params=None, adjust_error=True,
            automask=True, vmax=500., dispmax=600.,
            skip_crop=False, datadir=None):

    # +++++++++++++++++++++++++++++++++++++++++++
    # Upload the data set to be fit


    # Setup datadir, if set. If not set (so datadir=None), fdata must be the full path.
    if datadir is None:
        # If datadir not passed directly, look for entry in params file:
        if 'datadir' in params.keys():
            datadir = params['datadir']
        # If not passed directly as kwarg, and missing from params file, set to empty -- filenames must be full path.
        if datadir is None:
            datadir = ''


    gal_vel = fits.getdata(datadir+params['fdata_vel'])
    err_vel = fits.getdata(datadir+params['fdata_verr'])
    gal_disp = fits.getdata(datadir+params['fdata_disp'])
    err_disp = fits.getdata(datadir+params['fdata_derr'])
    mask = fits.getdata(datadir+params['fdata_mask'])


    # Mask NaNs:
    mask[~np.isfinite(gal_vel)] = 0
    gal_vel[~np.isfinite(gal_vel)] = 0.

    mask[~np.isfinite(err_vel)] = 0
    err_vel[~np.isfinite(err_vel)] = 0.

    mask[~np.isfinite(gal_disp)] = 0
    gal_disp[~np.isfinite(gal_disp)] = 0.

    mask[~np.isfinite(err_disp)] = 0
    err_disp[~np.isfinite(err_disp)] = 0.


    # Crop, if desired
    if not skip_crop:
        if params['fov_npix'] < min(gal_vel.shape):
            crp_x = np.int64(np.round((gal_vel.shape[1] - params['fov_npix'])/2.))
            crp_y = np.int64(np.round((gal_vel.shape[0] - params['fov_npix'])/2.))
            gal_vel = gal_vel[crp_y:params['fov_npix']+crp_y, crp_x:params['fov_npix']+crp_x]
            err_vel = err_vel[crp_y:params['fov_npix']+crp_y, crp_x:params['fov_npix']+crp_x]
            gal_disp = gal_disp[crp_y:params['fov_npix']+crp_y, crp_x:params['fov_npix']+crp_x]
            err_disp = err_disp[crp_y:params['fov_npix']+crp_y, crp_x:params['fov_npix']+crp_x]
            mask = mask[crp_y:params['fov_npix']+crp_y, crp_x:params['fov_npix']+crp_x]

    # Auto mask som bad data
    if automask:
        indtmp = (gal_disp > dispmax) | (np.abs(gal_vel) > vmax)
        mask[indtmp] = 0



    if adjust_error:
        # Mask > 1sig lower error outliers:
        errv_l68 = np.percentile(err_vel, 15.865)
        errd_l68 = np.percentile(err_disp, 15.865)

        indv = (err_vel < errv_l68)
        err_vel[indv] = errv_l68

        indd = (err_disp < errd_l68)
        err_disp[indd] = errd_l68



    # Mask pixels with zero error for vel/disp:
    mask[(err_vel == 0)] = 0
    mask[(err_disp == 0)] = 0


    #####
    # Apply symmetrization if wanted:
    try:
        if params['symmetrize_data']:
            ybin, xbin = np.indices(gal_vel.shape, dtype=np.float64)
            ybin = ybin.flatten()
            xbin = xbin.flatten()
            xbin -= (gal_vel.shape[1]-1.)/2.
            ybin -= (gal_vel.shape[0]-1.)/2.
            xbin -= params['xshift']
            ybin -= params['yshift']

            bool_mask = np.array(mask.copy(), dtype=bool)
            bool_mask_flat = np.array(mask.copy(), dtype=bool).flatten()

            gal_vel_flat_in = gal_vel.flatten()
            err_vel_flat_in = err_vel.flatten()
            gal_vel_flat_in[~bool_mask_flat] = np.NaN
            err_vel_flat_in[~bool_mask_flat] = np.NaN

            gal_vel_flat, err_vel_flat = dysmalpy_utils.symmetrize_velfield(xbin, ybin,
                                gal_vel_flat_in, err_vel_flat_in,
                                sym=1, pa=params['pa'])

            gal_disp_flat_in = gal_disp.flatten()
            err_disp_flat_in = err_disp.flatten()
            gal_disp_flat_in[~bool_mask_flat] = np.NaN
            err_disp_flat_in[~bool_mask_flat] = np.NaN
            gal_disp_flat, err_disp_flat = dysmalpy_utils.symmetrize_velfield(xbin, ybin,
                                gal_disp_flat_in, err_disp_flat_in,
                                sym=2, pa=params['pa'])

            #
            gal_vel[bool_mask] = gal_vel_flat[bool_mask_flat]
            err_vel[bool_mask] = err_vel_flat[bool_mask_flat]
            gal_disp[bool_mask] = gal_disp_flat[bool_mask_flat]
            err_disp[bool_mask] = err_disp_flat[bool_mask_flat]
    except:
        pass

    #
    if 'weighting_method' in params.keys():
        gal_weight = setup_data_weighting_method(method=params['weighting_method'], r=None)
    else:
        gal_weight = None

    #
    if ('moment_calc' in params.keys()):
        moment_calc = params['moment_calc']
    else:
        moment_calc = False
    if ('xcenter' in params.keys()):
        xcenter = params['xcenter']
    else:
        xcenter = None
    if ('ycenter' in params.keys()):
        ycenter = params['ycenter']
    else:
        ycenter = None

    data2d = data_classes.Data2D(pixscale=params['pixscale'], velocity=gal_vel,
                                      vel_disp=gal_disp, vel_err=err_vel,
                                      vel_disp_err=err_disp, mask=mask,
                                      weight=gal_weight,
                                      filename_velocity=datadir+params['fdata_vel'],
                                      filename_dispersion=datadir+params['fdata_disp'],
                                      smoothing_type=params['smoothing_type'],
                                      smoothing_npix=params['smoothing_npix'],
                                      inst_corr=params['data_inst_corr'],
                                      moment=moment_calc,
                                      xcenter=xcenter, ycenter=ycenter)


    return data2d

#
def load_single_object_3D_data(params=None, datadir=None):


    # +++++++++++++++++++++++++++++++++++++++++++
    # Upload the data set to be fit

    # Setup datadir, if set. If not set (so datadir=None), fdata must be the full path.
    if datadir is None:
        # If datadir not passed directly, look for entry in params file:
        if 'datadir' in params.keys():
            datadir = params['datadir']
        # If not passed directly as kwarg, and missing from params file, set to empty -- filenames must be full path.
        if datadir is None:
            datadir = ''


    cube = fits.getdata(datadir+params['fdata_cube'])
    err_cube = fits.getdata(datadir+params['fdata_err'])
    header = fits.getheader(datadir+params['fdata_cube'])

    mask = None
    if 'fdata_mask' in params.keys():
        if params['fdata_mask'] is not None:
            mask = fits.getdata(datadir+params['fdata_mask'])

    #
    mask_sky=None
    if 'fdata_mask_sky' in params.keys():
        if params['fdata_mask_sky'] is not None:
            mask_sky = fits.getdata(datadir+params['fdata_mask_sky'])
    #
    mask_spec=None
    if 'fdata_mask_spec' in params.keys():
        if params['fdata_mask_spec'] is not None:
            mask_spec = fits.getdata(datadir+params['fdata_mask_spec'])


    if 'weighting_method' in params.keys():
        gal_weight = setup_data_weighting_method(method=params['weighting_method'], r=None)
    else:
        gal_weight = None




    ####################################
    # Convert spectrum:
    if 'spec_orig_type' in params.keys():
        spec_arr_orig = (np.arange(cube.shape[0])+1 - header['CRPIX3'])*header['CDELT3'] + header['CRVAL3']
        spec_line = params['spec_line_rest'] * (1.+params['z'])
        if params['spec_orig_type'].strip().upper() == 'WAVE':
            if params['spec_line_rest_unit'].strip().upper() == 'ANGSTROM':
                l0 = spec_line * u.angstrom
            elif (params['spec_line_rest_unit'].strip().upper() == 'MICRON') | (params['spec_line_rest_unit'].strip().upper() == 'UM'):
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


        elif params['spec_orig_type'].strip().upper() == 'VELOCITY':
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




    pscale = np.abs(header['CDELT1']) * 3600.    # convert from deg CDELT1 to arcsec

    ####################################


    cube, err_cube, mask, mask_sky, mask_spec, gal_weight, spec_arr = _auto_truncate_crop_cube(cube,
                                            params=params,
                                            pixscale=pscale,
                                            spec_type='velocity', spec_arr=spec_arr,
                                            err_cube=err_cube, mask_cube=mask,
                                            mask_sky=mask_sky, mask_spec=mask_spec,
                                            spec_unit=spec_unit,weight=gal_weight)


    ####################################
    if (mask is None) & ('auto_gen_3D_mask' in params.keys()):
        if params['auto_gen_3D_mask']:
            if 'auto_gen_mask_snr_thresh_1' not in params.keys():
                params['auto_gen_mask_snr_thresh_1'] = params['auto_gen_mask_snr_thresh']
            #mask = _auto_gen_3D_mask_simple(cube=cube, err=err_cube, snr_thresh=params['auto_gen_mask_snr_thresh'],
            #        npix_min=params['auto_gen_mask_npix_min'])
            mask = auto_gen_3D_mask(cube=cube, err=err_cube,
                    sig_thresh=params['auto_gen_mask_sig_thresh'],
                    #snr_thresh=params['auto_gen_mask_snr_thresh'],
                    #snr_thresh_1 = params['auto_gen_mask_snr_thresh_1'],
                    npix_min=params['auto_gen_mask_npix_min'])

        else:
            mask = np.ones(cube.shape)

    ####################################
    # Mask NaNs:
    mask[~np.isfinite(cube)] = 0
    cube[~np.isfinite(cube)] = -99.

    mask[~np.isfinite(cube)] = 0
    err_cube[~np.isfinite(err_cube)] = -99.

    # Clean up 0s in error, if it's masked
    err_cube[mask == 0] = 99.

    ####################################
    data3d = data_classes.Data3D(cube, pixscale=pscale, spec_type='velocity', spec_arr=spec_arr,
                                      err_cube=err_cube, mask_cube=mask,
                                      mask_sky=mask_sky, mask_spec=mask_spec,
                                      spec_unit=u.km/u.s,
                                      weight=gal_weight)

    return data3d




def ensure_path_trailing_slash(path):
    if (path[-1] != '/'):
        path += '/'
    return path
