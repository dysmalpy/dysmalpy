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
from dysmalpy import utils as dysmalpy_utils

from dysmalpy import aperture_classes

from astropy.table import Table

import astropy.io.fits as fits

try:
    import utils_io
except:
    from . import utils_io



def read_gal_list_from_file(f_gallist=None):
    
    names = ['galID']
    
    data = pd.read_csv(f_gallist, sep=' ', comment='#', names=names, skipinitialspace=True,
                    index_col=False)
    
    galIDs = data['galID'].values
    
    
    return galIDs

def gather_catalog(f_gallist=None, cat_outname_base=None, cat_outpath=None,
            fitting_path=None, galdirect_base=None, truncate=False):
    
    
    galIDs = read_gal_list_from_file(f_gallist=f_gallist)
    
    cat = pd.DataFrame({})
    
    for galID in galIDs:
        
        aper_types = ['flared_rect', 'circ']
        data = None
        
        for aper_type in aper_types:
            if data is None:
                try:
                    galdirect = "{}_{}_{}_aps".format(galID, galdirect_base, aper_type)
                    galfilename = "{}_mcmc_best_fit_results.dat".format(galID)
        
                    if not truncate:
                        fname = fitting_path+'/'+galdirect+'/'+galfilename
                    else:
                        fname = fitting_path+'/'+galfilename
        
                    ascii_data = utils_io.read_results_ascii_file(fname=fname)
                    data = utils_io.make_catalog_row_entry(ascii_data=ascii_data, galID=galID)
                except:
                    pass
                
        
        cat = cat.append(data, ignore_index = True) 
        
        
    # Save to file:
    # outdir
    fitting.ensure_dir(cat_outpath)
    f_FITS = cat_outpath+'/'+cat_outname_base+'.fits'
    f_CSV = cat_outpath+'/'+cat_outname_base+'.csv'
    
    
    cat.to_csv(f_CSV)
    
    t = Table.from_pandas(cat)
    t.write(f_FITS, format='fits', overwrite=True)
    
    
    
    return None
    
    
#

if __name__ == "__main__":
    
    f_gallist = sys.argv[1]
    cat_outpath = sys.argv[2]
    cat_outname_base = sys.argv[3]
    fitting_path = sys.argv[4]
    galdirect_base = sys.argv[5]
    
    try:
        trunc_tmp = sys.argv[6]
        if trunc_tmp.strip().lower() == 'true':
            truncate=True
        elif trunc_tmp.strip().lower() == 'false':
            truncate=False
    except:
        truncate=False

    gather_catalog(f_gallist=f_gallist, cat_outpath=cat_outpath, 
                    cat_outname_base=cat_outname_base, 
                    fitting_path=fitting_path, galdirect_base=galdirect_base,
                    truncate=truncate)
    
    