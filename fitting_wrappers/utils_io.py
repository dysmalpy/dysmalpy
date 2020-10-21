# Script to fit kinematics

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys
from contextlib import contextmanager


import datetime

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as apy_con

from dysmalpy import data_classes
from dysmalpy import parameters
from dysmalpy import fitting
from dysmalpy import models
from dysmalpy import utils as dysmalpy_utils

from dysmalpy import aperture_classes

import scipy.optimize as scp_opt

from astropy.table import Table

import astropy.io.fits as fits

try:
    import photutils
    from astropy.convolution import Gaussian2DKernel
    loaded_photutils = True
except:
    loaded_photutils = False

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

    
def setup_fit_dict(params=None):

    if params['fit_method'] == 'mcmc':

        fit_dict = setup_mcmc_dict(params=params)

    elif params['fit_method'] == 'mpfit':

        fit_dict = setup_mpfit_dict(params=params)

    return fit_dict


def setup_mcmc_dict(params=None):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Parameters for the MCMC fitting + output filenames
    
    fitting.ensure_dir(params['outdir'])
    
    outdir = params['outdir']
    galID = params['galID']
    
    if 'plot_type' in params.keys():
        plot_type = params['plot_type']
    else:
        plot_type = 'pdf'
    
    # All in one directory:
    f_plot_trace_burnin = outdir+'{}_mcmc_burnin_trace.{}'.format(galID, plot_type)
    f_plot_trace = outdir+'{}_mcmc_trace.{}'.format(galID, plot_type)
    f_model = outdir+'{}_galaxy_model.pickle'.format(galID)
    f_cube = outdir+'{}_mcmc_bestfit_model_cube.fits'.format(galID)
    f_sampler = outdir+'{}_mcmc_sampler.pickle'.format(galID)
    f_burn_sampler = outdir+'{}_mcmc_burn_sampler.pickle'.format(galID)
    f_plot_param_corner = outdir+'{}_mcmc_param_corner.{}'.format(galID, plot_type)
    f_plot_bestfit = outdir+'{}_mcmc_best_fit.{}'.format(galID, plot_type)
    f_plot_bestfit_multid = outdir+'{}_mcmc_best_fit_multid.{}'.format(galID, plot_type)
    f_mcmc_results = outdir+'{}_mcmc_results.pickle'.format(galID)
    f_chain_ascii = outdir+'{}_mcmc_chain_blobs.dat'.format(galID)
    f_vel_ascii = outdir+'{}_galaxy_bestfit_vel_profile.dat'.format(galID)
    f_log = outdir+'{}_info.log'.format(galID)
    
    mcmc_dict = {'outdir': outdir, 
                'f_plot_trace_burnin':  f_plot_trace_burnin,
                'f_plot_trace':  f_plot_trace,
                'f_model': f_model,
                'f_cube': f_cube,
                'f_sampler':  f_sampler,
                'f_burn_sampler':  f_burn_sampler,
                'f_plot_param_corner':  f_plot_param_corner,
                'f_plot_bestfit':  f_plot_bestfit,
                'f_plot_bestfit_multid': f_plot_bestfit_multid, 
                'f_mcmc_results':  f_mcmc_results, 
                'f_chain_ascii': f_chain_ascii,
                'f_vel_ascii': f_vel_ascii, 
                'f_log': f_log, 
                'do_plotting': True}
                
    for key in params.keys():
        # Copy over all various fitting options
        mcmc_dict[key] = params[key]
        
    # #
    if 'linked_posteriors' in mcmc_dict.keys():
        if mcmc_dict['linked_posteriors'] is not None:
            linked_post_arr = []
            for lpost in mcmc_dict['linked_posteriors']:
                if lpost.strip().lower() == 'total_mass':
                    linked_post_arr.append(['disk+bulge', 'total_mass'])
                elif lpost.strip().lower() == 'mvirial':
                    linked_post_arr.append(['halo', 'mvirial'])
                elif lpost.strip().lower() == 'fdm':
                    linked_post_arr.append(['halo', 'fdm'])
                elif lpost.strip().lower() == 'alpha':
                    linked_post_arr.append(['halo', 'alpha'])
                elif lpost.strip().lower() == 'rb':
                    linked_post_arr.append(['halo', 'rB'])
                elif lpost.strip().lower() == 'r_eff_disk':
                    linked_post_arr.append(['disk+bulge', 'r_eff_disk'])
                elif lpost.strip().lower() == 'bt':
                    linked_post_arr.append(['disk+bulge', 'bt'])
                elif lpost.strip().lower() == 'sigma0':
                    linked_post_arr.append(['dispprof', 'sigma0'])
                elif lpost.strip().lower() == 'inc':
                    linked_post_arr.append(['geom', 'inc'])
                elif lpost.strip().lower() == 'pa':
                    linked_post_arr.append(['geom', 'pa'])
                elif lpost.strip().lower() == 'xshift':
                    linked_post_arr.append(['geom', 'xshift'])
                elif lpost.strip().lower() == 'yshift':
                    linked_post_arr.append(['geom', 'yshift'])
                elif lpost.strip().lower() == 'vel_shift':
                    linked_post_arr.append(['geom', 'vel_shift'])
                else:
                    raise ValueError("linked posterior for {} not currently implemented!".format(lpost))
            
            # "Bundle of linked posteriors"
            linked_posterior_names = [ linked_post_arr ] 
            mcmc_dict['linked_posterior_names'] = linked_posterior_names
        else:
            mcmc_dict['linked_posterior_names'] = None
    else:
        mcmc_dict['linked_posterior_names'] = None
        
        
    #
    mcmc_dict['model_key_re'] = ['disk+bulge', 'r_eff_disk']
    mcmc_dict['model_key_halo'] = ['halo']
    
    
    if 'continue_steps' not in mcmc_dict.keys():
        mcmc_dict['continue_steps'] = False
    
    return mcmc_dict


def setup_mpfit_dict(params=None):
    if 'plot_type' in params.keys():
        plot_type = params['plot_type']
    else:
        plot_type = 'pdf'
        
    fitting.ensure_dir(params['outdir'])
    outdir = params['outdir']
    galID = params['galID']
    f_model = outdir+'{}_galaxy_model.pickle'.format(galID)
    f_cube = outdir+'{}_mpfit_bestfit_model_cube.fits'.format(galID)
    f_plot_bestfit = outdir+'{}_mpfit_best_fit.{}'.format(galID, plot_type)
    f_results = outdir+'{}_mpfit_results.pickle'.format(galID)
    f_plot_bestfit_multid = outdir+'{}_mpfit_best_fit_multid.{}'.format(galID, plot_type)
    f_vel_ascii = outdir+'{}_galaxy_bestfit_vel_profile.dat'.format(galID)
    f_log = outdir+'{}_info.log'.format(galID)

    mpfit_dict = {'outdir': outdir,
                  'f_model': f_model,
                  'f_cube': f_cube,
                  'f_plot_bestfit':  f_plot_bestfit,
                  'f_plot_bestfit_multid': f_plot_bestfit_multid,
                  'f_results':  f_results,
                  'f_vel_ascii': f_vel_ascii,
                  'f_log': f_log,
                  'do_plotting': True}

    for key in params.keys():
        # Copy over all various fitting options
        mpfit_dict[key] = params[key]

    return mpfit_dict
    
    
def setup_basic_aperture_types(gal=None, params=None):
    
    if ('aperture_radius' in params.keys()):
        aperture_radius=params['aperture_radius']
    else:
        aperture_radius = None
    #
    if ('pix_perp' in params.keys()):
        pix_perp=params['pix_perp']
    else:
        pix_perp = None
    #
    if ('pix_parallel' in params.keys()):
        pix_parallel=params['pix_parallel']
    else:
        pix_parallel = None
    #
    if ('pix_length' in params.keys()):
        pix_length=params['pix_length']
    else:
        pix_length = None
        
    #
    if ('partial_weight' in params.keys()):
        partial_weight = params['partial_weight']
    else:
        # # Preserve previous default behavior
        # partial_weight = False
        
        ## NEW default behavior: always use partial_weight:
        partial_weight = True
        
    if ('moment_calc' in params.keys()):
        moment_calc = params['moment_calc']
    else:
        moment_calc = False
    
    apertures = aperture_classes.setup_aperture_types(gal=gal, 
                profile1d_type=params['profile1d_type'], 
                aperture_radius=aperture_radius, 
                pix_perp=pix_perp, pix_parallel=pix_parallel,
                pix_length=pix_length, from_data=True, 
                partial_weight=partial_weight,
                moment=moment_calc)
                
    
    return apertures

    
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
    
def setup_data_weighting_method(method='UNSET', r=None):
    if r is not None:
        rmax = np.abs(np.max(r))
    else:
        rmax = None
        
    if method == 'UNSET':
        raise ValueError("Must set method if setting data point weighting!")
    elif (method is None): 
        weight = None
    elif ((method.strip().lower() == 'none') | (method.strip().lower() == 'uniform')):
        weight = None
        #weight = np.ones(len(r), dtype=np.float)
    # exp[ A * (r/rmax) ]  // exponential or more general power-law
    elif method.strip().lower() == 'radius_rmax':
        # value at 0: 1 // value at rmax: 2.718
        weight = np.exp( np.abs(r)/ rmax )
    elif method.strip().lower() == 'radius_rmax_a5':
        # value at 0: 1 // value at rmax: 5.
        weight = np.exp( np.log(5.) * (np.abs(r)/ rmax)  )
    elif method.strip().lower() == 'radius_rmax_a10':
        # value at 0: 1 // value at rmax: 10.
        weight = np.exp( np.log(10.) * (np.abs(r)/ rmax)  )
    # exp[ A * (r/rmax)^2 ]  // exponential or more general power-law
    elif method.strip().lower() == 'radius_rmax2':
        # value at 0: 1 // value at rmax: 2.718
        weight = np.exp((np.abs(r)/ rmax)**2 )
    elif method.strip().lower() == 'radius_2rmax2':
        # value at 0: 1 // value at rmax: 7.389
        weight = np.exp( 2. * (np.abs(r)/ rmax)**2 )
    elif method.strip().lower() == 'radius_rmax2_a5':
        # value at 0: 1 // value at rmax: 5.
        weight = np.exp( np.log(5.) * (np.abs(r)/ rmax)**2 )
    elif method.strip().lower() == 'radius_rmax2_a10':
        # value at 0: 1 // value at rmax: 10.
        weight = np.exp( np.log(10.) * (np.abs(r)/ rmax)**2 )
    else:
        raise ValueError("Weighting method not implmented yet!: {}".format(method))
    
    return weight
    
    
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
    
    #raise ValueError
    
    return mask
    
def _auto_truncate_crop_cube(cube, params=None, 
            pixscale=None, spec_type='velocity', spec_arr=None,
                                            err_cube=None, mask_cube=None, 
                                            mask_sky=None, mask_spec=None,
                                            spec_unit=u.km/u.s,weight=None):
    
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
    return cube, err_cube, mask_cube, mask_sky, mask_spec, weight, spec_arr
    

def set_comp_param_prior(comp=None, param_name=None, params=None):
    if params['{}_fixed'.format(param_name)] is False:
        if '{}_prior'.format(param_name) in list(params.keys()):
            # Default to using pre-set value!
            try:
                try:
                    center = comp.prior[param_name].center
                except:
                    center = params[param_name] 
            except:
                # eg, UniformPrior
                center = None
            
            # Default to using pre-set value, if already specified!!!
            try:
                try:
                    stddev = comp.prior[param_name].stddev
                except:
                    stddev = params['{}_stddev'.format(param_name)]
            except:
                stddev = None
            
            if params['{}_prior'.format(param_name)].lower() == 'flat':
                comp.__getattribute__(param_name).prior = parameters.UniformPrior()
            elif params['{}_prior'.format(param_name)].lower() == 'flat_linear':
                comp.__getattribute__(param_name).prior = parameters.UniformLinearPrior()
            elif params['{}_prior'.format(param_name)].lower() == 'gaussian':
                comp.__getattribute__(param_name).prior = parameters.BoundedGaussianPrior(center=center, stddev=stddev)
            elif params['{}_prior'.format(param_name)].lower() == 'sine_gaussian':
                comp.__getattribute__(param_name).prior = parameters.BoundedSineGaussianPrior(center=center, stddev=stddev)
            elif params['{}_prior'.format(param_name)].lower() == 'gaussian_linear':
                comp.__getattribute__(param_name).prior = parameters.BoundedGaussianLinearPrior(center=center, stddev=stddev)
            elif params['{}_prior'.format(param_name)].lower() == 'tied_flat_lowtrunc':
                comp.__getattribute__(param_name).prior = TiedUniformPriorLowerTrunc(compn='disk+bulge', paramn='total_mass')
            elif params['{}_prior'.format(param_name)].lower() == 'tied_gaussian_lowtrunc':
                comp.__getattribute__(param_name).prior = TiedBoundedGaussianPriorLowerTrunc(center=center, stddev=stddev, 
                                                            compn='disk+bulge', paramn='total_mass')
            else:
                print(" CAUTION: {}: {} prior is not currently supported. Defaulting to 'flat'".format(param_name, 
                                    params['{}_prior'.format(param_name)]))
                pass
    
    return comp


def tie_sigz_reff(model_set):
 
    reff = model_set.components['disk+bulge'].r_eff_disk.value
    invq = model_set.components['disk+bulge'].invq_disk
    sigz = 2.0*reff/invq/2.35482

    return sigz



def m13_lshfm(lMh, z):
    # Moster + 2013, MNRAS 428, 3121
    # Table 1, Best fit
    M10 = 11.590
    M11 = 1.195
    N10 = 0.0351
    N11 = -0.0247
    b10 = 1.376
    b11 = -0.826
    g10 = 0.608
    g11 = 0.329
    
    zarg = z/(1.+z)
    M1 = np.power(10., M10 + M11*zarg)
    Nz = N10 + N11*zarg
    bz = b10 + b11*zarg
    gz = g10 + g11*zarg
    
    Mh = np.power(10., lMh)
    lmSMh = np.log10(2*Nz) - np.log10(np.power(Mh/M1 , -bz)  + np.power( Mh/M1, gz))
    
    return lmSMh
    
def lmstar_num_solver_moster(lMh, z, lmass):
    
    return m13_lshfm(lMh, z) - lmass + lMh
    

def moster13_halo_mass_num_solve(z=None, lmass=None, truncate_lmstar_halo=None):
    # Do zpt solver to get lmhalo given lmass:
    
    if truncate_lmstar_halo is None:
        raise ValueError
        
    if truncate_lmstar_halo:
        lmstar = min(lmass, 11.2)
    else:
        lmstar = lmass
        
    lmhalo = scp_opt.newton(lmstar_num_solver_moster, lmstar + 2.,
                        args=(z, lmstar),
                        maxiter=200)
    
    return lmhalo
    
    
def behroozi13_halo_mass(z=None, lmass=None):
    # From the inverted relation fit by Omri Ginzburg (A Dekel student; email from A Dekel 2020-05-15)
    # Valid for lM* = 10-12; z=0.5-3 at better than 0.5% accuracy 
    # ** NO TRUNCATION **
    
    A0, A1, A2, A3, A4 = 13.3402, -1.8671, 1.3010, -0.4037, 0.0439
    B0, B1, B2, B3, B4 = -0.1814, 0.1791, -0.1020, 0.0270, -2.85e-3
    C0, C1, C2, C3 = 0.7361, 0.6427, -0.2737, 0.0280
    D0, D1, D2, D3 = 5.3744, 6.2722, -2.6661, 0.2503
    
    A_OGAD_z = A0 + A1*z + A2*(z**2) + A3*(z**3) + A4*(z**4)
    B_OGAD_z = B0 + B1*z + B2*(z**2) + B3*(z**3) + B4*(z**4)
    C_OGAD_z = C0 + C1*z + C2*(z**2) + C3*(z**3)
    D_OGAD_z = D0 + D1*z + D2*(z**2) + D3*(z**3)
    
    lmhalo = A_OGAD_z + B_OGAD_z*lmass * np.sin(C_OGAD_z * lmass - D_OGAD_z)
    
    return lmhalo
    
def moster13_halo_mass(z=None, lmass=None):
    # From the fitting relation from Moster, Naab & White 2013; from email from Thorsten Naab on 2020-05-21
    # ** NO TRUNCATION **
    
    log_m1 = 10.485 + 1.099 * (z/(z+1.))
    n  = np.power( 10., (1.425 + 0.328 * (z/(z+1.)) - 1.174 * ((z/(z+1.))**2)) )
    b  = -0.569 + 0.132 * (z/(z+1.))
    g  = 1.023 + 0.295 * (z/(z+1.)) - 2.768 * ((z/(z+1.))**2)
    
    lmhalo = lmass + np.log10(0.5) + np.log10(n) + np.log10( np.power( np.power(10., (lmass-log_m1)), b ) + np.power( np.power(10., (lmass-log_m1)), g ) )
    
    return lmhalo
    
def moster18_halo_mass(z=None, lmass=None):
    # From the updated fitting relation from Moster, Naab & White 2018; from email from Thorsten Naab on 2020-05-21
    # From stellar mass binned fitting result (avoids divergance at high lMstar)
    # ** NO TRUNCATION **
    
    log_m1 = 10.6
    n = np.power(10., (1.507 - 0.124 * (z/(z+1.)) ) )
    b = -0.621 - 0.059 * (z/(z+1.))
    g = 1.055 + 0.838 * (z/(z+1)) - 3.083 * ( ((z/(z+1)))**2 )
    
    lmhalo = lmass + np.log10(0.5) + np.log10(n) + np.log10( np.power( np.power(10., (lmass-log_m1)), b ) + np.power( np.power(10., (lmass-log_m1)), g ) )
    
    return lmhalo
    
def tied_mhalo_mstar(model_set):
    # Uses constant fgas to go from lMbar to the stellar mass for the moster calculation
    z = model_set.components['halo'].z
    
    lmbar = model_set.components['disk+bulge'].total_mass.value
    fgas = model_set.components['disk+bulge'].fgas
    Mbar = np.power(10., lmbar)
    Mstar = (1.-fgas)*Mbar
    
    try:
        mhalo_relation = model_set.components['disk+bulge'].mhalo_relation
    except:
        
        print("Missing mhalo_relation! setting mhalo_relation='Moster18' ! [options: 'Moster18', 'Behroozi13', 'Moster13']")
        mhalo_relation = 'Moster18'
        
    ########    
        
    if mhalo_relation.lower().strip() == 'behroozi13':
        lmhalo = behroozi13_halo_mass(z=z, lmass=np.log10(Mstar))
        
    elif mhalo_relation.lower().strip() == 'moster18':
        lmhalo = moster18_halo_mass(z=z, lmass=np.log10(Mstar))
        
    elif mhalo_relation.lower().strip() == 'moster13':
        raise ValueError
        
        ## OLD VERSION, NUMERICAL SOLUTION TO MOSTER13
        try:
            truncate_lmstar_halo = model_set.components['disk+bulge'].truncate_lmstar_halo
        except:
            print("Missing truncate_lmstar_halo! setting truncate_lmstar_halo=True")
            truncate_lmstar_halo = True
        
        lmhalo = moster13_halo_mass_num_solve(z=z, lmass=np.log10(Mstar), 
                                truncate_lmstar_halo=truncate_lmstar_halo)
    ####
    return lmhalo
    
############################################################################
# Tied functions for halo fitting:
def tie_lmvirial_NFW(model_set):
    comp_halo = model_set.components.__getitem__('halo')
    comp_baryons = model_set.components.__getitem__('disk+bulge')
    r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    mvirial = comp_halo.calc_mvirial_from_fdm(comp_baryons, r_fdm, 
                    adiabatic_contract=model_set.kinematic_options.adiabatic_contract)
    return mvirial

def tie_alpha_TwoPower(model_set):
    comp_halo = model_set.components.__getitem__('halo')
    comp_baryons = model_set.components.__getitem__('disk+bulge')
    r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    alpha = comp_halo.calc_alpha_from_fdm(comp_baryons, r_fdm)
    return alpha

def tie_rB_Burkert(model_set):
    comp_halo = model_set.components.__getitem__('halo')
    comp_baryons = model_set.components.__getitem__('disk+bulge')
    r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    rB = comp_halo.calc_rB_from_fdm(comp_baryons, r_fdm)
    return rB
    
    
def tie_alphaEinasto_Einasto(model_set):
    comp_halo = model_set.components.__getitem__('halo')
    comp_baryons = model_set.components.__getitem__('disk+bulge')
    r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    alphaEinasto = comp_halo.calc_alphaEinasto_from_fdm(comp_baryons, r_fdm)
    return alphaEinasto
    
def tie_nEinasto_Einasto(model_set):
    comp_halo = model_set.components.__getitem__('halo')
    comp_baryons = model_set.components.__getitem__('disk+bulge')
    r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    nEinasto = comp_halo.calc_nEinasto_from_fdm(comp_baryons, r_fdm)
    return nEinasto
    
def tie_fdm(model_set):
    r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    fdm = model_set.get_dm_aper(r_fdm)
    return fdm
    
############################################################################



class TiedUniformPriorLowerTrunc(parameters.UniformPrior):
    def __init__(self, compn='disk+bulge', paramn='total_mass'):
        self.compn = compn
        self.paramn = paramn
        
        super(TiedUniformPriorLowerTrunc, self).__init__()
    def log_prior(self, param, modelset=None, **kwargs):
        
        pmin = modelset.components[self.compn].__getattribute__(self.paramn).value  

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (param.value >= pmin) & (param.value <= pmax):
            return 0.
        else:
            return -np.inf


class TiedBoundedGaussianPriorLowerTrunc(parameters.BoundedGaussianPrior):
    def __init__(self, compn='disk+bulge', paramn='total_mass', center=0, stddev=1.0):
        self.compn = compn
        self.paramn = paramn
        
        super(TiedBoundedGaussianPriorLowerTrunc, self).__init__(center=center, stddev=stddev)

    def log_prior(self, param, modelset=None, **kwargs):

        pmin = modelset.components[self.compn].__getattribute__(self.paramn).value  

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (param.value >= pmin) & (param.value <= pmax):
            return np.log(parameters.norm.pdf(param.value, loc=self.center, scale=self.stddev))
        else:
            return -np.inf


def ensure_path_trailing_slash(path):
    if (path[-1] != '/'):
        path += '/'
    return path
    
    

############################################################################

def calc_Rout_max_2D(gal=None, fit_results=None):
    gal.model.update_parameters(fit_results.bestfit_parameters)
    inc_gal = gal.model.geometry.inc.value
    
    
    
    ###############
    # Get grid of data coords:
    nx_sky = gal.data.data['velocity'].shape[1]
    ny_sky = gal.data.data['velocity'].shape[0]
    nz_sky = 1 #np.int(np.max([nx_sky, ny_sky]))
    rstep = gal.data.pixscale
    
    
    xcenter = gal.data.xcenter 
    ycenter = gal.data.ycenter 
    
        
    if xcenter is None:
        xcenter = (nx_sky - 1) / 2.
    if ycenter is None:
        ycenter = (ny_sky - 1) / 2.
        
        
    #
    sh = (nz_sky, ny_sky, nx_sky)
    zsky, ysky, xsky = np.indices(sh)
    zsky = zsky - (nz_sky - 1) / 2.
    ysky = ysky - ycenter 
    xsky = xsky - xcenter 
    
    # Apply the geometric transformation to get galactic coordinates
    xgal, ygal, zgal = gal.model.geometry(xsky, ysky, zsky)
    
    # Get the 4 corners sets:
    gal.model.geometry.inc = 0
    xskyp_ur, yskyp_ur, zskyp_ur = gal.model.geometry(xsky+0.5, ysky+0.5, zsky)
    xskyp_ll, yskyp_ll, zskyp_ll = gal.model.geometry(xsky-0.5, ysky-0.5, zsky)
    xskyp_lr, yskyp_lr, zskyp_lr = gal.model.geometry(xsky+0.5, ysky-0.5, zsky)
    xskyp_ul, yskyp_ul, zskyp_ul = gal.model.geometry(xsky-0.5, ysky+0.5, zsky)
    
    #Reset:
    gal.model.geometry.inc = inc_gal
    
    
    yskyp_ur_flat = yskyp_ur[0,:,:]
    yskyp_ll_flat = yskyp_ll[0,:,:]
    yskyp_lr_flat = yskyp_lr[0,:,:]
    yskyp_ul_flat = yskyp_ul[0,:,:]
    
    val_sgns = np.zeros(yskyp_ur_flat.shape)
    val_sgns += np.sign(yskyp_ur_flat)
    val_sgns += np.sign(yskyp_ll_flat)
    val_sgns += np.sign(yskyp_lr_flat)
    val_sgns += np.sign(yskyp_ul_flat)
    
    whgood = np.where( ( np.abs(val_sgns) < 4. ) & (gal.data.mask) )
    
    xgal_flat = xgal[0,:,:]
    ygal_flat = ygal[0,:,:]
    xgal_list = xgal_flat[whgood]
    ygal_list = ygal_flat[whgood]
    
    
    # The circular velocity at each position only depends on the radius
    # Convert to kpc
    rgal = np.sqrt(xgal_list ** 2 + ygal_list ** 2) * rstep / gal.dscale
    
    Routmax2D = np.max(rgal.flatten())
    
    
    return Routmax2D 
    
    
    


