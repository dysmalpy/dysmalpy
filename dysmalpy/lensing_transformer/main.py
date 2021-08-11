#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This is a Python program testing the "libLensingTransformer.so" functions.
    
    Last update: 2021-08-03, Daizhong Liu, MPE. 
    
"""
import os, sys, time, binascii, struct
import ctypes
from ctypes import cdll, c_void_p, c_char_p, c_double, c_long, c_int, POINTER
import numpy as np
import astropy.io.fits as fits

#mylib = ctypes.CDLL("libLensingTransformer.so", mode = ctypes.RTLD_GLOBAL)
#mylib = ctypes.CDLL("libLensingTransformer.so", mode = os.RTLD_LAZY)
#print('dir(mylib)', dir(mylib))

mylib = cdll.LoadLibrary("libLensingTransformer.so")
#print('dir(mylib)', dir(mylib))
#print('mylib.AllLensingTransformerInstances', mylib.AllLensingTransformerInstances)


mesh_file = "data/mesh.dat"
mesh_ra = 135.3434883
mesh_dec = 18.2418031

data, header = fits.getdata("data/model_cube.fits", header=True)
nx = header['NAXIS1']
ny = header['NAXIS2']
nchan = header['NAXIS3']
cenra = 135.3434883
cendec = 18.2418031
cenx = (nx+1.0)/2.0
ceny = (ny+1.0)/2.0
pixsc = 0.02 # model cube pixel size

imsizex = 600 # for image plane
imsizey = 600 # for image plane
imcenra = 135.3434883
imcendec = 18.2418031
imcenx = (imsizex+1.0)/2.0
imceny = (imsizey+1.0)/2.0
impixsc = 0.04

verbose = 1


#myobj = mylib.createLensingTransformer() # this will cause SIGSEGV (Address boundary error), use the three lines below
#createLensingTransformer = mylib.createLensingTransformer
#createLensingTransformer.argtypes = [c_char_p, c_double, c_double, POINTER(c_double), c_long, c_long, c_long, c_double, c_double, c_double]
#createLensingTransformer.restype = c_void_p
##myobj = c_void_p(createLensingTransformer("data/mesh.dat", 135.3434883, 18.2418031, data, nx, ny, nchan, 135.3434883, 18.2418031, 0.02))
#myobj = c_void_p(createLensingTransformer("data/mesh.dat", 135.3434883, 18.2418031, data, nx, ny, nchan, 135.3434883, 18.2418031, 0.02))
#print('dir(myobj)', dir(myobj))


#data[0,0,0] = 1.5 # testing cdata endian, it seems cdata is big endian when passing to C for some reason.
if sys.byteorder == 'little':
    data = data.astype('<f8')
else:
    data = data.astype('>f8')

print('mylib.createLensingTransformer', mylib.createLensingTransformer)
#print('c_char_p("data/mesh.dat")', c_char_p(mesh_file.encode('utf-8')))
#print('data.dtype', data.dtype)
cdata = data.ctypes.data_as(POINTER(c_double))
#print('ctypes.addressof(cdata)', hex(ctypes.addressof(cdata)))
#print('ctypes.addressof(cdata.contents)', hex(ctypes.addressof(cdata.contents)))
#cdata_first_eight_bytes = (ctypes.c_char*8).from_address(ctypes.addressof(cdata.contents))
#print('cdata_first_eight_bytes.value', cdata_first_eight_bytes.value)
#print('binascii.hexlify(cdata_first_eight_bytes.raw)', binascii.hexlify(cdata_first_eight_bytes.raw))
#print('data[0,0,0]', data[0,0,0])
##IsLittleEndian = (hex(struct.unpack('=L', struct.pack('=I', 0x01234567))[0]) == '0x67452301')
##IsLittleEndian = sys.byteorder
#print("hex(struct.unpack('=L', struct.pack('=I', 0x01234567))[0])", hex(struct.unpack('=L', struct.pack('=I', 0x01234567))[0]))
#print("hex(struct.unpack('=Q', struct.pack('=d', data[0,0,0]))[0])", hex(struct.unpack('=Q', struct.pack('=d', data[0,0,0]))[0]), 'native')
#print("hex(struct.unpack('=Q', struct.pack('>d', data[0,0,0]))[0])", hex(struct.unpack('=Q', struct.pack('>d', data[0,0,0]))[0]), 'big-endian')
#print("hex(struct.unpack('=Q', struct.pack('<d', data[0,0,0]))[0])", hex(struct.unpack('=Q', struct.pack('<d', data[0,0,0]))[0]), 'little-endian')
#                                  args :  mesh_file, mesh_ra, mesh_dec, 
#                                          source_plane_data_cube, 
#                                          source_plane_data_nx, source_plane_data_ny, source_plane_data_nchan,  
#                                          source_plane_ra, source_plane_dec, source_plane_pixelsize, 
#                                          source_plane_cenx=nan, source_plane_ceny=nan, verbose=1
mylib.createLensingTransformer.argtypes = [c_char_p, c_double, c_double, 
                                           POINTER(c_double), 
                                           c_long, c_long, c_long, 
                                           c_double, c_double, c_double, 
                                           c_double, c_double, c_int]
mylib.createLensingTransformer.restype = c_void_p
myobj = c_void_p(mylib.createLensingTransformer(c_char_p(mesh_file.encode('utf-8')), mesh_ra, mesh_dec, 
                                                cdata, 
                                                nx, ny, nchan, 
                                                cenra, cendec, pixsc, 
                                                cenx, ceny, verbose))
print('dir(myobj)', dir(myobj))


print('mylib.performLensingTransformation', mylib.performLensingTransformation)
#                                      args :  ptr, 
#                                              image_plane_ra, image_plane_dec, image_plane_pixelsize, 
#                                              image_plane_sizex, image_plane_sizey, 
#                                              image_plane_cenx=nan, image_plane_ceny=nan, verbose=1
mylib.performLensingTransformation.argtypes = [c_void_p, 
                                               c_double, c_double, c_double, 
                                               c_long, c_long, 
                                               c_double, c_double, c_int]
mylib.performLensingTransformation.restype = ctypes.POINTER(ctypes.c_double * imsizex * imsizey * nchan)
outcdata = mylib.performLensingTransformation(myobj, 
                                                        imcenra, imcendec, impixsc, 
                                                        imsizex, imsizey, 
                                                        imcenx, imceny, verbose)
print('dir(outcdata)', dir(outcdata))
print('ctypes.addressof(outcdata)', hex(ctypes.addressof(outcdata)))
print('ctypes.addressof(outcdata.contents)', hex(ctypes.addressof(outcdata.contents)))
#outdata = np.frombuffer(outcdata.contents, dtype=np.dtype(np.double))
#outdata = outdata.reshape((nchan, imsizey, imsizex))
#outdata = np.ctypeslib.as_array(outcdata)
#outdata = np.frombuffer(np.core.multiarray.int_asbuffer(\
#                        ctypes.addressof(outcdata.contents), 
#                        nchan*imsizey*imsizex*np.dtype(np.double).itemsize))
outdata = np.ctypeslib.as_array(\
                (ctypes.c_double * imsizex * imsizey * nchan).from_address(ctypes.addressof(outcdata.contents))\
            )
#outcdatacast = ctypes.cast(outcdata, ctypes.POINTER(ctypes.c_double * imsizex * imsizey * nchan))
#outdata = np.frombuffer(outcdatacast.contents)
print('outdata', outdata)

outheader = fits.Header()
outheader['NAXIS'] = 3
outheader['NAXIS1'] = imsizex
outheader['NAXIS2'] = imsizey
outheader['NAXIS3'] = nchan
outheader['RADESYS'] = 'FK5'
outheader['SPECSYS'] = 'TOPOCENT'
outheader['EQUINOX'] = 2000.0
outheader['CTYPE1'] = 'RA---TAN'
outheader['CTYPE2'] = 'DEC--TAN'
outheader['CTYPE3'] = ''
outheader['CUNIT1'] = 'deg'
outheader['CUNIT2'] = 'deg'
outheader['CUNIT3'] = ''
outheader['CRPIX1'] = imcenx
outheader['CRPIX2'] = imceny
outheader['CRPIX3'] = 1.0 # (nchan+1.0)/2.0
outheader['CRVAL1'] = imcenra
outheader['CRVAL2'] = imcendec
outheader['CRVAL3'] = 1.0 # 0.0
outheader['CDELT1'] = -impixsc/3600.0
outheader['CDELT2'] = impixsc/3600.0
outheader['CDELT3'] = 1.0

outhdu = fits.PrimaryHDU(data=outdata, header=outheader)
print('Writing to "%s"'%("data/image_plane_cube_from_python.fits"))
outhdu.writeto("data/image_plane_cube_from_python.fits", overwrite=True)

time.sleep(4)

mylib.destroyLensingTransformer(myobj)



