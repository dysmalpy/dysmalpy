# Script to fit KMOS3D kinematics

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys
from contextlib import contextmanager


import datetime

import numpy as np
import pandas as pd
import astropy.units as u

from dysmalpy import data_classes
from dysmalpy import parameters
from dysmalpy import fitting

from astropy.table import Table

import astropy.io.fits as fits

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
                tmpval = df['values'][j].strip()
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
              'red_chisq': True }

    # param_filename
    fname_split = fname.split('/')
    params['param_filename'] = fname_split[-1]

    for key in param_input.keys():
        params[key] = param_input[key]

    return params
    
def save_results_ascii_files(mcmc_results=None, gal=None, params=None):
    
    outdir = params['outdir']
    galID = params['galID']
    
    f_ascii_machine = outdir+'{}_mcmc_best_fit_results.dat'.format(galID)
    
    f_ascii_pretty = outdir+'{}_mcmc_best_fit_results.info'.format(galID)
    
    
    with open(f_ascii_machine, 'w') as f:
        namestr = '# component    param_name    fixed    best_value   l68_err   u68_err'
        f.write(namestr+'\n')
        
        
        for cmp_n in gal.model.param_names.keys():
            for param_n in gal.model.param_names[cmp_n]:
                
                if '{}:{}'.format(cmp_n,param_n) in mcmc_results.chain_param_names:
                    whparam = np.where(mcmc_results.chain_param_names == '{}:{}'.format(cmp_n,param_n))[0][0]
                    best = mcmc_results.bestfit_parameters[whparam]
                    l68 = mcmc_results.bestfit_parameters_l68_err[whparam]
                    u68 = mcmc_results.bestfit_parameters_u68_err[whparam]
                else:
                    best = getattr(gal.model.components[cmp_n], param_n).value
                    l68 = -99.
                    u68 = -99.
                
                datstr = '{: <12}   {: <11}   {: <5}   {:9.4f}   {:9.4f}   {:9.4f}'.format(cmp_n, param_n, 
                            "{}".format(gal.model.fixed[cmp_n][param_n]), best, l68, u68)
                f.write(datstr+'\n')
                
        #
        datstr = '{: <12}   {: <11}   {: <5}   {:9.4f}   {:9.4f}   {:9.4f}'.format('redchisq', '-----', 
                    '-----', mcmc_results.bestfit_redchisq, -99, -99)
        f.write(datstr+'\n')
        
    #
    with open(f_ascii_pretty, 'w') as f:
        f.write('###############################'+'\n')
        f.write(' Fitting for {}'.format(params['galID'])+'\n')
        f.write('\n')
        
        f.write("Date: {}".format(datetime.datetime.now())+'\n')
        f.write('\n')
        
        f.write('Datafile: {}'.format(params['fdata'])+'\n')
        f.write('Paramfile: {}'.format(params['param_filename'])+'\n')
        
        f.write('\n')
        f.write('###############################'+'\n')
        f.write(' Fitting results'+'\n')
        
        for cmp_n in gal.model.param_names.keys():
            f.write('-----------'+'\n')
            f.write(' {}'.format(cmp_n)+'\n')
            
            for param_n in gal.model.param_names[cmp_n]:
                
                if '{}:{}'.format(cmp_n,param_n) in mcmc_results.chain_param_names:
                    whparam = np.where(mcmc_results.chain_param_names == '{}:{}'.format(cmp_n,param_n))[0][0]
                    best = mcmc_results.bestfit_parameters[whparam]
                    l68 = mcmc_results.bestfit_parameters_l68_err[whparam]
                    u68 = mcmc_results.bestfit_parameters_u68_err[whparam]
                    
                    
                    datstr = '    {: <11}    {:9.4f}  -{:9.4f} +{:9.4f}'.format(param_n, best, l68, u68)
                    f.write(datstr+'\n')
            #
            f.write('\n')
            #
            for param_n in gal.model.param_names[cmp_n]:
                
                if '{}:{}'.format(cmp_n,param_n) not in mcmc_results.chain_param_names:
                    best = getattr(gal.model.components[cmp_n], param_n).value
                    
                    datstr = '    {: <11}    {:9.4f}  [FIXED]'.format(param_n, best)
                    f.write(datstr+'\n')
                    
        #
        f.write('\n')
        f.write('-----------'+'\n')
        datstr = 'Red. chisq: {:0.4f}'.format(mcmc_results.bestfit_redchisq)
        f.write(datstr+'\n')
        
        f.write('\n')
    
    
    return None

    
#
def setup_mcmc_dict(params=None):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Parameters for the MCMC fitting + output filenames
    
    fitting.ensure_dir(params['outdir'])
    
    outdir = params['outdir']
    galID = params['galID']
    
    # All in one directory:
    f_plot_trace_burnin = outdir+'{}_mcmc_burnin_trace.pdf'.format(galID)
    f_plot_trace = outdir+'{}_mcmc_trace.pdf'.format(galID)
    f_model = outdir+'{}_galaxy_model.pickle'.format(galID)
    f_sampler = outdir+'{}_mcmc_sampler.pickle'.format(galID)
    f_burn_sampler = outdir+'{}_mcmc_burn_sampler.pickle'.format(galID)
    f_plot_param_corner = outdir+'{}_mcmc_param_corner.pdf'.format(galID)
    f_plot_bestfit = outdir+'{}_mcmc_best_fit.pdf'.format(galID)
    f_plot_bestfit_multid = outdir+'{}_mcmc_best_fit_multid.pdf'.format(galID)
    f_mcmc_results = outdir+'{}_mcmc_results.pickle'.format(galID)
    f_chain_ascii = outdir+'{}_mcmc_chain_blobs.dat'.format(galID)
    f_vel_ascii = outdir+'{}_galaxy_bestfit_vel_profile.dat'.format(galID)
    f_log = outdir+'{}_info.log'.format(galID)
    
    mcmc_dict = {'outdir': outdir, 
                'f_plot_trace_burnin':  f_plot_trace_burnin,
                'f_plot_trace':  f_plot_trace,
                'f_model': f_model, 
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
        
    return mcmc_dict
    
    
def load_single_object_1D_data(fdata=None, params=None):
    
    # Load the data set to be fit
    dat_arr =   np.loadtxt(fdata)
    gal_r =     dat_arr[:,0]
    gal_vel =   dat_arr[:,1]
    gal_disp =  dat_arr[:,3]
    err_vel =   dat_arr[:,2]
    err_disp =  dat_arr[:,4]
    # inst_corr = params['data_inst_corr']                  # Flag for if the measured dispersion has been
    #                                                       # corrected for instrumental resolution
    
    #####
    # Apply symmetrization if wanted:
    try:
        if params['symmetrize_data']:
            gal_r_new, gal_vel, err_vel = dysmalpy_utils.symmetrize_1D_profile(gal_r, gal_vel, err_vel, sym=1)
            gal_r, gal_disp, err_disp = dysmalpy_utils.symmetrize_1D_profile(gal_r, gal_disp, err_disp, sym=2)
    except:
        pass
    
    
    data1d = data_classes.Data1D(r=gal_r, velocity=gal_vel,
                                      vel_disp=gal_disp, vel_err=err_vel,
                                      vel_disp_err=err_disp, slit_width=params['slit_width'],
                                      slit_pa=params['slit_pa'], inst_corr=params['data_inst_corr'] )
                                      
    return data1d
    
def load_single_object_2D_data(params=None, adjust_error=True, 
            automask=True, vmax=500., dispmax=600.):
    
    # +++++++++++++++++++++++++++++++++++++++++++
    # Upload the data set to be fit
    gal_vel = fits.getdata(params['fdata_vel'])
    err_vel = fits.getdata(params['fdata_verr'])
    gal_disp = fits.getdata(params['fdata_disp'])
    err_disp = fits.getdata(params['fdata_derr'])
    mask = fits.getdata(params['fdata_mask'])
    
    inst_corr = params['data_inst_corr']
    
    
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
    
    
    data2d = data_classes.Data2D(pixscale=pixscale, velocity=gal_vel,
                                      vel_disp=gal_disp, vel_err=err_vel,
                                      vel_disp_err=err_disp, mask=mask,
                                      filename_velocity=params['fdata_vel'],
                                      filename_dispersion=params['fdata_disp'],
                                      smoothing_type=params['smoothing_type'],
                                      smoothing_npix=params['smoothing_npix'],
                                      inst_corr=inst_corr)
                                      
            
    return data2d
    
#
def set_comp_param_prior(comp=None, param_name=None, params=None):
    
    if params['{}_fixed'.format(param_name)] is False:
        if '{}_prior'.format(param_name) in list(params.keys()):
            if params['{}_prior'.format(param_name)].lower() == 'flat':
                pass
            elif params['{}_prior'.format(param_name)].lower() == 'gaussian':
                comp.prior[param_name] = parameters.BoundedGaussianPrior(center=params[param_name],
                                                                        stddev=params['{}_stddev'.format(param_name)])
                                                                        
            elif params['{}_prior'.format(param_name)].lower() == 'sine_gaussian':
                comp.prior[param_name] = parameters.BoundedSineGaussianPrior(center=params[param_name],
                                                                        stddev=params['{}_stddev'.format(param_name)])
            else:
                print(" CAUTION: {}: {} prior is not currently supported. Defaulting to 'flat'".format(param_name, 
                                    params['{}_prior'.format(param_name)]))
                pass
    
    
    return comp

    
#
def tie_sigz_reff(model_set):
 
    reff = model_set.components['disk+bulge'].r_eff_disk.value
    invq = model_set.components['disk+bulge'].invq_disk
    sigz = 2.0*reff/invq/2.35482

    return sigz
    
    
def ensure_path_trailing_slash(path):
    if (path[-1] != '/'):
        path += '/'
    return path
    
    
    
    
