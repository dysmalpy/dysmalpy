# -*- coding: utf-8 -*-
"""
    This is a Python code to implement 1D least squares fitting in DysmalPy. 
    This code uses the C++ library "leastChiSquares1D.cpp". 
    
    Last updates: 
        2021-08-05, finished first version, Daizhong Liu, MPE. 
    
"""

# <DZLIU><20210805> ++++++++++

import os, sys, copy, timeit
import logging
logger = logging.getLogger(__name__) # here we do not setLevel so that it inherits its caller logging level. 
import ctypes
from ctypes import cdll, c_void_p, c_char_p, c_double, c_long, c_int, POINTER
import numpy as np
from distutils.sysconfig import get_config_var
#mylib = cdll.LoadLibrary(os.path.abspath(os.path.dirname(__file__))+os.sep+"libLeastChiSquares1D.so")
mylibfile = os.path.abspath(os.path.dirname(__file__))+os.sep+"leastChiSquares1D"+get_config_var('EXT_SUFFIX')
mylib = cdll.LoadLibrary(mylibfile)

class LeastChiSquares1D(object):
    """docstring for LeastChiSquares1D
    
    Args:
        `x`: 1D numpy array for the spectral axis.
        `data`: 3D numpy array for the data cube.
        `dataerr`: 3D numpy array for the data error cube.
        `verbose`: Boolean. The verbose level for this Python class.
        `c_verbose`: Integer. The verbose level for the C program.
    """
    def __init__(
            self, 
            x, 
            data, 
            dataerr, 
            initparams, 
            nthread = 4, 
            maxniter = 1000, 
            verbose = True, 
            c_verbose = 0,
        ):
        # 
        self.logger = logging.getLogger('LensingTransformer')
        self.logger.setLevel(logging.getLevelName(logging.getLogger(__name__).level)) # self.logger.setLevel(logging.INFO)
        if verbose:
            self.printLibInfo()
        # 
        self.x = x
        self.data = data
        self.dataerr = dataerr
        self.initparams = initparams
        self.nchan, self.ny, self.nx = self.data.shape
        # 
        # Prepare initial guesses
        #scube = SpectralCube(data=data, header=header, wcs=WCS(header))
        #flux_guess = scube.moment0().to(u.km/u.s).value
        #mean_guess = scube.moment1().to(u.km/u.s).value
        #stddev_guess = scube.linewidth_sigma().to(u.km/u.s).value
        ##flux_guess = np.array([0.003, 0.003])
        ##mean_guess = np.array([0.0, 0.0])
        ##stddev_guess = np.array([100., 100.])
        ##flux_guess = flux_guess * np.sqrt(2*np.pi*(stddev_guess**2))
        #initparams = np.array([flux_guess/np.sqrt(2*np.pi*(stddev_guess**2)), mean_guess, np.abs(stddev_guess)])


        # Create carrays
        if sys.byteorder == 'little':
            self.data = self.data.astype('<f8')
            if self.dataerr is not None:
                self.dataerr = self.dataerr.astype('<f8')
            self.x = self.x.astype('<f8')
            self.initparams = self.initparams.astype('<f8')
        else:
            self.data = self.data.astype('>f8')
            if self.dataerr is not None:
                self.dataerr = self.dataerr.astype('>f8')
            self.x = self.x.astype('>f8')
            self.initparams = self.initparams.astype('>f8')
        
        self.cdata = self.data.ctypes.data_as(POINTER(c_double))
        if self.dataerr is not None:
            self.cdataerr = self.dataerr.ctypes.data_as(POINTER(c_double))
        else:
            self.cdataerr = POINTER(c_double)()
        self.cx = self.x.ctypes.data_as(POINTER(c_double))
        self.cinitparams = self.initparams.ctypes.data_as(POINTER(c_double))
        
        # Other variables
        self.nparams = 3 # 1D Gaussian has 3 parameters
        self.maxniter = maxniter
        self.verbose = verbose
        self.c_verbose = c_verbose
        self.nthread = nthread
    
    def printLibInfo(self):
        self.logger.info('mylibfile %r'%(mylibfile))
        self.logger.info('mylib %s'%(mylib))
    
    def runFitting(self):
        """Run the fitting by calling the C library functions.
        """
        if self.verbose:
            print('running least chi-squares 1d fitting to data cube with dimension %d x %d x %d'%(self.nx, self.ny, self.nchan))
        
        time_begin = timeit.default_timer()
        #print('mylib.fitLeastChiSquares1DForDataCubeWithMultiThread', mylib.fitLeastChiSquares1DForDataCubeWithMultiThread)
        """
        In "leastChiSquares1D.hpp": 

        double *fitLeastChiSquares1DForDataCubeWithMultiThread(\
            double *x,
            double *data,
            double *dataerr,
            long nx,
            long ny,
            long nchan,
            double *initparamsall,
            long nparams,
            int maxniter = 1000,
            int verbose = 1,
            int nthread = 4);
        """
        
        mylib.fitLeastChiSquares1DForDataCubeWithMultiThread.argtypes = [\
            POINTER(c_double), POINTER(c_double), POINTER(c_double), 
            c_long, c_long, c_long, 
            POINTER(c_double), c_long, 
            c_int, c_int, c_int]
        
        mylib.fitLeastChiSquares1DForDataCubeWithMultiThread.restype = \
            ctypes.POINTER(ctypes.c_double * (self.nparams*2 + self.nchan*2 + 1) * self.ny * self.nx)
        
        outcdata = mylib.fitLeastChiSquares1DForDataCubeWithMultiThread(\
            self.cx, self.cdata, self.cdataerr, 
            self.nx, self.ny, self.nchan, 
            self.cinitparams, self.nparams, 
            self.maxniter, self.c_verbose, self.nthread)
        #print('dir(outcdata)', dir(outcdata))
        #print('ctypes.addressof(outcdata)', hex(ctypes.addressof(outcdata)))
        #print('ctypes.addressof(outcdata.contents)', hex(ctypes.addressof(outcdata.contents)))
        
        outdata = np.ctypeslib.as_array(\
                        (ctypes.c_double * (self.nparams*2 + self.nchan*2 + 1) * self.ny * self.nx\
                        ).from_address(ctypes.addressof(outcdata.contents))\
                    )
        
        outdata = outdata.reshape([(self.nparams*2 + self.nchan*2 + 1), self.ny, self.nx])
        
        time_finish = timeit.default_timer()
        
        if self.verbose:
            print('elasped %s seconds'%(time_finish - time_begin))
        
        #print('outdata.shape', outdata.shape)
        #print('np.count_nonzero(np.any(np.isnan(outdata),axis=0))', np.count_nonzero(np.any(np.isnan(outdata),axis=0)))
        #print('np.count_nonzero(np.any(np.isnan(data),axis=0))', np.count_nonzero(np.any(np.isnan(data),axis=0)))
        #print('np.count_nonzero(np.any(np.isnan(dataerr),axis=0))', np.count_nonzero(np.any(np.isnan(dataerr),axis=0)))
        #print('np.count_nonzero(np.any(np.isnan(initparamsall),axis=0))', np.count_nonzero(np.any(np.isnan(initparamsall),axis=0)))
        
        nparams = self.nparams
        nchan = self.nchan
        self.outparams = outdata[0:nparams, :, :]
        self.outparamerrs = outdata[nparams:nparams*2, :, :]
        self.outyfitted = outdata[nparams*2:nparams*2+nchan, :, :]
        self.outyresidual = outdata[nparams*2+nchan:nparams*2+nchan*2, :, :]
        self.outchisq = outdata[nparams*2+nchan*2:nparams*2+nchan*2+1, :, :]
        
        #print('outparams.shape', outparams.shape)
        #print('outparamerrs.shape', outparamerrs.shape)
        #print('outyfitted.shape', outyfitted.shape)
        #print('outyresidual.shape', outyresidual.shape)
        #print('outchisq.shape', outchisq.shape)




# <DZLIU><20210805> ----------

