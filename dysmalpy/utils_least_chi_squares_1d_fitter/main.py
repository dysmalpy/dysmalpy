#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This is a Python program testing the "libLeastChiSquares1D.so" functions.
    
    Last update: 2021-08-05, Daizhong Liu, MPE. 
    
"""
import os, sys, copy, time, timeit, binascii, struct
import ctypes
from ctypes import cdll, c_void_p, c_char_p, c_double, c_long, c_int, POINTER
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube


# Load C library
mylib = cdll.LoadLibrary("libLeastChiSquares1D.so")


# Prepare data
data, header = fits.getdata('/tmp/test_data_cube.fits.gz', header=True)
#data = data[:, 66:68, 67:68]
#header['NAXIS1'] = 68-67
#header['NAXIS2'] = 68-66

print('data.shape', data.shape)
print('data mean %s max %s min %s median %s stddev %s'%(np.nanmean(data), np.nanmax(data), np.nanmin(data), np.nanmedian(data), np.nanstd(data)))
dataerr = np.full(data.shape, fill_value = np.nanmax(data) * 0.001)
dataerr = None

nchan, ny, nx = data.shape

x = ((np.arange(nchan)+1.0) - header['CRPIX3']) * header['CDELT3'] + header['CRVAL3']


# Prepare initial guesses
scube = SpectralCube(data=data, header=header, wcs=WCS(header))
flux_guess = scube.moment0().to(u.km/u.s).value
mean_guess = scube.moment1().to(u.km/u.s).value
stddev_guess = scube.linewidth_sigma().to(u.km/u.s).value
#flux_guess = np.array([0.003, 0.003])
#mean_guess = np.array([0.0, 0.0])
#stddev_guess = np.array([100., 100.])
#flux_guess = flux_guess * np.sqrt(2*np.pi*(stddev_guess**2))
initparamsall = np.array([flux_guess/np.sqrt(2*np.pi*(stddev_guess**2)), mean_guess, np.abs(stddev_guess)])


# Create carrays
if sys.byteorder == 'little':
    data = data.astype('<f8')
    if dataerr is not None:
        dataerr = dataerr.astype('<f8')
    x = x.astype('<f8')
    initparamsall = initparamsall.astype('<f8')
else:
    data = data.astype('>f8')
    if dataerr is not None:
        dataerr = dataerr.astype('>f8')
    x = x.astype('>f8')
    initparamsall = initparamsall.astype('>f8')

cdata = data.ctypes.data_as(POINTER(c_double))
if dataerr is not None:
    cdataerr = dataerr.ctypes.data_as(POINTER(c_double))
else:
    cdataerr = POINTER(c_double)()
cx = x.ctypes.data_as(POINTER(c_double))
cinitparamsall = initparamsall.ctypes.data_as(POINTER(c_double))


# Other variables
nparams = 3 # 1D Gaussian has 3 parameters
maxniter = 1000
verbose = 1 # 2
nthread = 1



#print('mylib.GlobalDebug', mylib.GlobalDebug)


time_begin = timeit.default_timer()



print('mylib.fitLeastChiSquares1DForDataCubeWithMultiThread', mylib.fitLeastChiSquares1DForDataCubeWithMultiThread)
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
    ctypes.POINTER(ctypes.c_double * (nparams*2 + nchan*2 + 1) * ny * nx)

outcdata = mylib.fitLeastChiSquares1DForDataCubeWithMultiThread(\
    cx, cdata, cdataerr, 
    nx, ny, nchan, 
    cinitparamsall, nparams, 
    maxniter, verbose, nthread)
#print('dir(outcdata)', dir(outcdata))
#print('ctypes.addressof(outcdata)', hex(ctypes.addressof(outcdata)))
#print('ctypes.addressof(outcdata.contents)', hex(ctypes.addressof(outcdata.contents)))


outdata = np.ctypeslib.as_array(\
                (ctypes.c_double * (nparams*2 + nchan*2 + 1) * ny * nx).from_address(ctypes.addressof(outcdata.contents))\
            )

outdata = outdata.reshape([(nparams*2 + nchan*2 + 1), ny, nx])

time_finish = timeit.default_timer()

print('elasped %s seconds'%(time_finish - time_begin))

print('outdata.shape', outdata.shape)
print('np.count_nonzero(np.any(np.isnan(outdata),axis=0))', np.count_nonzero(np.any(np.isnan(outdata),axis=0)))
print('np.count_nonzero(np.any(np.isnan(data),axis=0))', np.count_nonzero(np.any(np.isnan(data),axis=0)))
if dataerr:
    print('np.count_nonzero(np.any(np.isnan(dataerr),axis=0))', np.count_nonzero(np.any(np.isnan(dataerr),axis=0)))
print('np.count_nonzero(np.any(np.isnan(initparamsall),axis=0))', np.count_nonzero(np.any(np.isnan(initparamsall),axis=0)))

outparams = outdata[0:nparams, :, :]
outparamerrs = outdata[nparams:nparams*2, :, :]
outyfitted = outdata[nparams*2:nparams*2+nchan, :, :]
outyresidual = outdata[nparams*2+nchan:nparams*2+nchan*2, :, :]
outchisq = outdata[nparams*2+nchan*2:nparams*2+nchan*2+1, :, :]

print('outparams.shape', outparams.shape)
print('outparamerrs.shape', outparamerrs.shape)
print('outyfitted.shape', outyfitted.shape)
print('outyresidual.shape', outyresidual.shape)
print('outchisq.shape', outchisq.shape)

outheader = copy.deepcopy(header)
outhdu = fits.PrimaryHDU(data=outyfitted, header=outheader)
print('Writing to "%s"'%("outyfitted.fits"))
outhdu.writeto("outyfitted.fits", overwrite=True)
outhdu = fits.PrimaryHDU(data=outyresidual, header=outheader)
print('Writing to "%s"'%("outyresidual.fits"))
outhdu.writeto("outyresidual.fits", overwrite=True)

outheader = copy.deepcopy(header)
outheader['NAXIS3'] = 1
outheader['CDELT3'] = 1
outheader['CUNIT3'] = ''
outheader['CTYPE3'] = 'CHISQ'
outheader['CRVAL3'] = 1
outheader['CRPIX3'] = 1
outhdu = fits.PrimaryHDU(data=outchisq, header=outheader)
print('Writing to "%s"'%("outchisq.fits"))
outhdu.writeto("outchisq.fits", overwrite=True)

outheader = copy.deepcopy(header)
outheader['NAXIS3'] = nparams
outheader['CDELT3'] = 1
outheader['CUNIT3'] = ''
outheader['CTYPE3'] = 'PARAMS'
outheader['CRVAL3'] = 1
outheader['CRPIX3'] = 1
outhdu = fits.PrimaryHDU(data=outparams, header=outheader)
print('Writing to "%s"'%("outparams.fits"))
outhdu.writeto("outparams.fits", overwrite=True)
outhdu = fits.PrimaryHDU(data=outparamerrs, header=outheader)
print('Writing to "%s"'%("outparamerrs.fits"))
outhdu.writeto("outparamerrs.fits", overwrite=True)




