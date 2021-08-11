# -*- coding: utf-8 -*-
"""
    This is a Python code to implement lensing tranformation in DysmalPy. 
    This code uses the C++ library "libLensingTransformer.so". 
    
    Last updates: 
        2021-08-03, finished first version, Daizhong Liu, MPE. 
        2021-08-04, added self.image_plane_data_cube and self.image_plane_data_info, set logging, Daizhong Liu, MPE. 
    
"""

# <DZLIU><20210726> ++++++++++

import os, sys
import logging
logger = logging.getLogger(__name__) # here we do not setLevel so that it inherits its caller logging level. 
import ctypes
from ctypes import cdll, c_void_p, c_char_p, c_double, c_long, c_int, POINTER
import numpy as np
from distutils.sysconfig import get_config_var
#mylib = cdll.LoadLibrary(os.path.abspath(os.path.dirname(__file__))+os.sep+"libLensingTransformer.so")
mylibfile = os.path.abspath(os.path.dirname(__file__))+os.sep+"lensingTransformer"+get_config_var('EXT_SUFFIX')
mylib = cdll.LoadLibrary(mylibfile)

class LensingTransformer(object):
    """docstring for LensingTransformer
    
    Args:
        `mesh_file`: String. The "mesh.dat" file from the lensing modeling using Glafic software.
        `verbose`: Boolean. The verbose level for this Python class.
        `c_verbose`: Integer. The verbose level for the C program.
    """
    def __init__(
            self, 
            mesh_file, 
            mesh_ra, 
            mesh_dec, 
            source_plane_data_cube = None, 
            source_plane_nx = None, 
            source_plane_ny = None, 
            source_plane_nchan = None, 
            source_plane_cenra = None, 
            source_plane_cendec = None, 
            source_plane_pixsc = None, 
            source_plane_cenx = None, 
            source_plane_ceny = None, 
            image_plane_cenra = None, 
            image_plane_cendec = None, 
            image_plane_pixsc = None, 
            image_plane_sizex = None, 
            image_plane_sizey = None, 
            image_plane_cenx = None, 
            image_plane_ceny = None, 
            verbose = True, 
            c_verbose = 0,
        ):
        # 
        self.logger = logging.getLogger('LensingTransformer')
        self.logger.setLevel(logging.getLevelName(logging.getLogger(__name__).level)) # self.logger.setLevel(logging.INFO)
        if verbose:
            self.printLibInfo()
        self.myobj = None
        self.mesh_file = mesh_file
        self.mesh_ra = mesh_ra
        self.mesh_dec = mesh_dec
        self.source_plane_data_cube = None
        self.source_plane_data_info = None
        self.source_plane_nx = None
        self.source_plane_ny = None
        self.source_plane_nchan = None
        self.source_plane_cenra = None
        self.source_plane_cendec = None
        self.source_plane_pixsc = None
        self.source_plane_cenx = None
        self.source_plane_ceny = None
        self.image_plane_cenra = None
        self.image_plane_cendec = None
        self.image_plane_pixsc = None
        self.image_plane_sizex = None
        self.image_plane_sizey = None
        self.image_plane_cenx = None
        self.image_plane_ceny = None
        self.image_plane_data_cube = None
        self.image_plane_data_info = None
        if source_plane_nx is not None:
            self.source_plane_nx = source_plane_nx
        if source_plane_ny is not None:
            self.source_plane_ny = source_plane_ny
        if source_plane_nchan is not None:
            self.source_plane_nchan = source_plane_nchan
        if source_plane_cenra is not None:
            self.source_plane_cenra = source_plane_cenra
        if source_plane_cendec is not None:
            self.source_plane_cendec = source_plane_cendec
        if source_plane_pixsc is not None:
            self.source_plane_pixsc = source_plane_pixsc
        if source_plane_cenx is not None:
            self.source_plane_cenx = source_plane_cenx
        if source_plane_ceny is not None:
            self.source_plane_ceny = source_plane_ceny
        if image_plane_cenra is not None:
            self.image_plane_cenra = image_plane_cenra
        if image_plane_cendec is not None:
            self.image_plane_cendec = image_plane_cendec
        if image_plane_pixsc is not None:
            self.image_plane_pixsc = image_plane_pixsc
        if image_plane_sizex is not None:
            self.image_plane_sizex = image_plane_sizex
        if image_plane_sizey is not None:
            self.image_plane_sizey = image_plane_sizey
        if image_plane_cenx is not None:
            self.image_plane_cenx = image_plane_cenx
        if image_plane_ceny is not None:
            self.image_plane_ceny = image_plane_ceny
        if source_plane_data_cube is not None:
            if np.any([t is None for t in [\
                source_plane_cenra, 
                source_plane_cendec, 
                source_plane_pixsc]]):
                raise Exception('Error! source_plane_data_cube is not None but one of source_plane_cenra/cendec/pixsc is None! Please input all of them.')
            # 
            self.source_plane_nx = source_plane_data_cube.shape[2]
            self.source_plane_ny = source_plane_data_cube.shape[1]
            self.source_plane_nchan = source_plane_data_cube.shape[0]
            self.setSourcePlaneDataCube(\
                    source_plane_data_cube, 
                    source_plane_cenra, 
                    source_plane_cendec, 
                    source_plane_pixsc, 
                    source_plane_cenx, 
                    source_plane_ceny, 
                    verbose, 
                )
            if self.myobj is None:
                raise Exception('Error! Could not set source plane data cube!')
    
    def __del__(self):
        if self.myobj is not None:
            mylib.destroyLensingTransformer(self.myobj)
    
    def printLibInfo(self):
        self.logger.info('mylibfile %r'%(mylibfile))
        self.logger.info('mylib %s'%(mylib))
    
    def setSourcePlaneDataCube(
            self, 
            source_plane_data_cube, 
            source_plane_cenra = None, 
            source_plane_cendec = None, 
            source_plane_pixsc = None, 
            source_plane_cenx = None, 
            source_plane_ceny = None, 
            verbose = True, 
            c_verbose = 0, 
        ):
        # 
        self.source_plane_data_cube = source_plane_data_cube
        if len(self.source_plane_data_cube.shape) != 3:
            raise Exception('Error! The input data cube should have 3 dimensions!')
        # 
        self.source_plane_nx = source_plane_data_cube.shape[2]
        self.source_plane_ny = source_plane_data_cube.shape[1]
        self.source_plane_nchan = source_plane_data_cube.shape[0]
        # 
        if source_plane_cenra is None:
            source_plane_cenra = self.source_plane_cenra
        if source_plane_cendec is None:
            source_plane_cendec = self.source_plane_cendec
        if source_plane_pixsc is None:
            source_plane_pixsc = self.source_plane_pixsc
        if source_plane_cenx is None:
            if self.source_plane_cenx is not None:
                source_plane_cenx = self.source_plane_cenx
        if source_plane_ceny is None:
            if self.source_plane_ceny is not None:
                source_plane_ceny = self.source_plane_ceny
        # 
        data = source_plane_data_cube
        nx = source_plane_data_cube.shape[2]
        ny = source_plane_data_cube.shape[1]
        nchan = source_plane_data_cube.shape[0]
        cenra = source_plane_cenra
        cendec = source_plane_cendec
        pixsc = source_plane_pixsc
        if source_plane_cenx is None:
            cenx = (nx+1.0)/2.0
        else:
            cenx = float(source_plane_cenx)
        if source_plane_ceny is None:
            ceny = (ny+1.0)/2.0
        else:
            ceny = float(source_plane_ceny)
        self.source_plane_data_info = {'NAXIS':3, 'NAXIS1':nx, 'NAXIS2':ny, 'NAXIS3':nchan, 'RADESYS':'ICRS', 'SPECSYS':'TOPOCENT', 'EQUINOX':2000.0, 
                'CTYPE1':'RA---TAN', 'CTYPE2':'DEC--TAN', 'CTYPE3':'CHANNEL', 'CUNIT1':'deg', 'CUNIT2':'deg', 'CUNIT2':'', 
                'CRPIX1':cenx, 'CRPIX2':ceny, 'CRPIX3':1.0, 'CRVAL1':cenra, 'CRVAL2':cendec, 'CRVAL3':1.0, 
                'CDELT1':-pixsc/3600.0, 'CDELT2':pixsc/3600.0, 'CDELT3':1.0, 
            }
        # 
        if sys.byteorder == 'little':
            data = data.astype('<f8')
        else:
            data = data.astype('>f8')

        # createLensingTransformer
        if verbose:
            self.logger.info('mylib.createLensingTransformer %s is called'%(mylib.createLensingTransformer))
        cdata = data.ctypes.data_as(POINTER(c_double))

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
        self.myobj = c_void_p(mylib.createLensingTransformer(\
                                                    c_char_p(self.mesh_file.encode('utf-8')), self.mesh_ra, self.mesh_dec, 
                                                    cdata, 
                                                    nx, ny, nchan, 
                                                    cenra, cendec, pixsc, 
                                                    cenx, ceny, c_verbose))
        
        if verbose:
            self.logger.info('mylib.createLensingTransformer %s finished'%(mylib.createLensingTransformer))
    
    def updateSourcePlaneDataCube(
            self, 
            source_plane_data_cube, 
            verbose = True, 
            c_verbose = 0, 
        ):
        
        # 
        if source_plane_data_cube.shape != self.source_plane_data_cube.shape:
            raise Exception('Error! Wrong input data dimension! Must input a data cube with the same shape as before.')
        
        # 
        data = source_plane_data_cube
        # 
        if sys.byteorder == 'little':
            data = data.astype('<f8')
        else:
            data = data.astype('>f8')
        
        # updateSourcePlaneDataCube
        if verbose:
            self.logger.info('mylib.updateSourcePlaneDataCube %s is called'%(mylib.updateSourcePlaneDataCube))
        cdata = data.ctypes.data_as(POINTER(c_double))
        
        #                                      args :  ptr, 
        #                                              source_plane_data_cube, 
        #                                              verbose=1
        mylib.updateSourcePlaneDataCube.argtypes = [c_void_p, 
                                                    POINTER(c_double), 
                                                    c_int]
        mylib.updateSourcePlaneDataCube(self.myobj, 
                                        cdata, 
                                        c_verbose)
        
        if verbose:
            self.logger.info('mylib.updateSourcePlaneDataCube %s finished'%(mylib.updateSourcePlaneDataCube))
    
    def performLensingTransformation(
            self, 
            imcenra = None, 
            imcendec = None, 
            impixsc = None, 
            imsizex = None, 
            imsizey = None, 
            imcenx = None, 
            imceny = None, 
            verbose = True, 
            c_verbose = 0, 
        ):
        
        # 
        if imcenra is None:
            if self.image_plane_cenra is not None:
                imcenra = self.image_plane_cenra
        if imcendec is None:
            if self.image_plane_cendec is not None:
                imcendec = self.image_plane_cendec
        if impixsc is None:
            if self.image_plane_pixsc is not None:
                impixsc = self.image_plane_pixsc
        if imsizex is None:
            if self.image_plane_sizex is not None:
                imsizex = self.image_plane_sizex
        if imsizey is None:
            if self.image_plane_sizey is not None:
                imsizey = self.image_plane_sizey
        if imcenx is None:
            if self.image_plane_cenx is not None:
                imcenx = self.image_plane_cenx
        if imceny is None:
            if self.image_plane_ceny is not None:
                imceny = self.image_plane_ceny
        # 
        if np.any([t is None for t in [imcenra, imcendec, impixsc, imsizex, imsizey]]):
            self.error('Error! Incorrect input to performLensingTransformation. Please check imcenra, imcendec, impixsc, imsizex, imsizey. Retunning None.')
            return None
        # 
        nchan = self.source_plane_data_cube.shape[0]
        if imcenx is None:
            imcenx = (imsizex+1.0)/2.0
        else:
            imcenx = float(imcenx)
        if imceny is None:
            imceny = (imsizey+1.0)/2.0
        else:
            imceny = float(imceny)
        
        # performLensingTransformation
        if verbose:
            self.logger.info('mylib.performLensingTransformation %s is called'%(mylib.performLensingTransformation))
        #                                      args :  ptr, 
        #                                              image_plane_ra, image_plane_dec, image_plane_pixelsize, 
        #                                              image_plane_sizex, image_plane_sizey, 
        #                                              image_plane_cenx=nan, image_plane_ceny=nan, verbose=1
        mylib.performLensingTransformation.argtypes = [c_void_p, 
                                                       c_double, c_double, c_double, 
                                                       c_long, c_long, 
                                                       c_double, c_double, c_int]
        mylib.performLensingTransformation.restype = ctypes.POINTER(ctypes.c_double * imsizex * imsizey * nchan)
        outcdata = mylib.performLensingTransformation(self.myobj, 
                                                      imcenra, imcendec, impixsc, 
                                                      imsizex, imsizey, 
                                                      imcenx, imceny, c_verbose)
        outdata = np.ctypeslib.as_array(\
                        (ctypes.c_double * imsizex * imsizey * nchan).from_address(ctypes.addressof(outcdata.contents))\
                    )
        
        self.image_plane_data_cube = outdata
        self.image_plane_data_info = {'NAXIS':3, 'NAXIS1':imsizex, 'NAXIS2':imsizey, 'NAXIS3':nchan, 'RADESYS':'ICRS', 'SPECSYS':'TOPOCENT', 'EQUINOX':2000.0, 
                'CTYPE1':'RA---TAN', 'CTYPE2':'DEC--TAN', 'CTYPE3':'CHANNEL', 'CUNIT1':'deg', 'CUNIT2':'deg', 'CUNIT2':'', 
                'CRPIX1':imcenx, 'CRPIX2':imceny, 'CRPIX3':1.0, 'CRVAL1':imcenra, 'CRVAL2':imcendec, 'CRVAL3':1.0, 
                'CDELT1':-impixsc/3600.0, 'CDELT2':impixsc/3600.0, 'CDELT3':1.0, 
            }
        
        if verbose:
            self.logger.info('mylib.performLensingTransformation %s finished'%(mylib.performLensingTransformation))
        
        return outdata



# <DZLIU><20210726> ----------
