# -*- coding: utf-8 -*-
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
"""
    This is a Python code to implement lensing tranformation in DysmalPy.
    This code uses the C++ library `libLensingTransformer.so`.
    
    .. Last updates:
    ..     2021-08-03, finished first version, Daizhong Liu, MPE.
    ..     2021-08-04, added self.image_plane_data_cube and self.image_plane_data_info, set logging, Daizhong Liu, MPE.
    ..     2022-05-24, change to take ObsLensingOptions instance as input vs params dict.
    ..     2022-06-18, change back to simple params dict and do not use ObsLensingOptions here.

"""

__all__=['LensingTransformer']

import os, sys, datetime, timeit, glob
import logging
logger = logging.getLogger(__name__) # here we do not setLevel so that it inherits its caller logging level.
if '__main__' in logging.Logger.manager.loggerDict: # pragma: no cover
    logger.setLevel(logging.getLogger('__main__').level)
import ctypes
from ctypes import cdll, c_void_p, c_char_p, c_double, c_long, c_int, POINTER
import numpy as np
import multiprocessing, threading
from distutils.sysconfig import get_config_var

mylibfile = os.path.abspath(os.path.dirname(__file__))+os.sep+"lensingTransformer"+get_config_var('EXT_SUFFIX')

# ++++++++++++
if not os.path.isfile(mylibfile):
    mylibfile = os.path.abspath(os.path.dirname(__file__))+os.sep+"lensingTransformer*.so"
    mylibfile = glob.glob(mylibfile)[0]
# ++++++++++++
mylib = cdll.LoadLibrary(mylibfile)
cached_lensing_transformer_dict = {'0': None}

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
        global mylib
        self.mylib = mylib
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
            if self.myobj is None: # pragma: no cover
                raise Exception('Error! Could not set source plane data cube!')

    def __del__(self):
        if self.myobj is not None and self.mylib is not None:
            self.mylib.destroyLensingTransformer(self.myobj)

    def logMessage(self, text):
        """Print message with logger.info() or logger.debug() depending on current logging level.
        """
        if self.logger.level == logging.DEBUG: # pragma: no cover
            self.logger.debug(text)
        else:
            self.logger.info(text)

    def printLibInfo(self):
        self.logMessage('mylibfile %r'%(mylibfile))
        self.logMessage('mylib %s'%(mylib))

    def setDebugLevel(self, level=1):
        if self.myobj is not None and self.mylib is not None:
            self.mylib.setLensingTransformerDebugLevel(self.myobj, level)

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
                'CTYPE1':'RA---TAN', 'CTYPE2':'DEC--TAN', 'CTYPE3':'CHANNEL', 'CUNIT1':'deg', 'CUNIT2':'deg', 'CUNIT3':'',
                'CRPIX1':cenx, 'CRPIX2':ceny, 'CRPIX3':1.0, 'CRVAL1':cenra, 'CRVAL2':cendec, 'CRVAL3':1.0,
                'CDELT1':-pixsc/3600.0, 'CDELT2':pixsc/3600.0, 'CDELT3':1.0,
            }
        #
        if sys.byteorder == 'little':
            data = data.astype('<f8')
        else: # pragma: no cover
            data = data.astype('>f8')

        # createLensingTransformer
        if verbose:
            self.logMessage('mylib.createLensingTransformer %s is called'%(mylib.createLensingTransformer))
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
            self.logMessage('mylib.createLensingTransformer %s finished'%(mylib.createLensingTransformer))

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
        else: # pragma: no cover
            data = data.astype('>f8')

        # updateSourcePlaneDataCube
        if verbose:
            self.logMessage('mylib.updateSourcePlaneDataCube %s is called'%(mylib.updateSourcePlaneDataCube))
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
            self.logMessage('mylib.updateSourcePlaneDataCube %s finished'%(mylib.updateSourcePlaneDataCube))

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
        if np.any([t is None for t in [imcenra, imcendec, impixsc, imsizex, imsizey]]): # pragma: no cover
            self.logger.error('Error! Incorrect input to performLensingTransformation.' +
                              'Please check imcenra, imcendec, impixsc, imsizex, imsizey.' +
                              'Retunning None.')
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
            self.logMessage('mylib.performLensingTransformation %s is called'%(mylib.performLensingTransformation))
            self.logMessage('running lensing transformation at '+str(datetime.datetime.now()))
            time_begin = timeit.default_timer()
        #                                      args :  ptr,
        #                                              image_plane_ra, image_plane_dec, image_plane_pixelsize,
        #                                              image_plane_sizex, image_plane_sizey,
        #                                              image_plane_cenx=nan, image_plane_ceny=nan, verbose=1
        if verbose and logger.level == logging.DEBUG:
            self.setDebugLevel(2) #DBEUGGING# 20211221
            mylib.setGlobalDebugLevel(2)
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
                'CTYPE1':'RA---TAN', 'CTYPE2':'DEC--TAN', 'CTYPE3':'CHANNEL', 'CUNIT1':'deg', 'CUNIT2':'deg', 'CUNIT3':'',
                'CRPIX1':imcenx, 'CRPIX2':imceny, 'CRPIX3':1.0, 'CRVAL1':imcenra, 'CRVAL2':imcendec, 'CRVAL3':1.0,
                'CDELT1':-impixsc/3600.0, 'CDELT2':impixsc/3600.0, 'CDELT3':1.0,
            }

        if verbose:
            time_finish = timeit.default_timer()
            self.logMessage('finished lensing transformation at '+str(datetime.datetime.now()))
            self.logMessage('elapsed %s seconds'%(time_finish - time_begin))
            self.logMessage('mylib.performLensingTransformation %s finished'%(mylib.performLensingTransformation))

        return outdata



def has_lensing_transform_keys_in_params(
        params,
    ):
    """Check if the input dict has lensing* keys
    """
    for key in params:
        if key.startswith('lensing'):
            return True
    return False # pragma: no cover



def setup_lensing_transformer_from_params(
        params = None,
		mesh_dir = None, 
        mesh_file = None,
        mesh_ra = None,
        mesh_dec = None,
        source_plane_nx = None,
        source_plane_ny = None,
        source_plane_nchan = None,
        source_plane_cenra = None,
        source_plane_cendec = None,
        source_plane_pixsc = None,
        image_plane_cenra = None,
        image_plane_cendec = None,
        image_plane_pixsc = None,
        image_plane_sizex = None,
        image_plane_sizey = None,
        reuse_lensing_transformer = None,
        cache_lensing_transformer = True,
        reuse_cached_lensing_transformer = True,
        verbose = True,
        **kwargs,
    ):
    """A utility function to return a LensingTransformer instance from the input parameters.

    One can either provide a params dict with following madatory keys:
        - `lensing_ra`
        - `lensing_dec`
        - `lensing_ssizex`
        - `lensing_ssizey`
        - `lensing_sra`
        - `lensing_sdec`
        - `lensing_spixsc`
        - `lensing_imra`
        - `lensing_imdec`
        - `pixscale`
        - `fov_npix`
        - `nspec`

    or individual parameters as arguments.

    Note that the individual parameter inputs overrides the use of the keys in the params dict.

    If one inputs a `reuse_lensing_transformer`, then we will assume it is a LensingTransformer
    object and try to reuse it if all parameters are matched.

    """

    if mesh_dir is None:
        if params is not None:
            if 'lensing_datadir' in params:
                mesh_dir = params['lensing_datadir']
            elif 'datadir' in params:
                mesh_dir = params['datadir']
    if mesh_file is None:
        if params is not None:
            if 'lensing_mesh' in params:
                mesh_file = params['lensing_mesh']
    if mesh_ra is None:
        if params is not None:
            if 'lensing_ra' in params:
                mesh_ra = params['lensing_ra']
    if mesh_dec is None:
        if params is not None:
            if 'lensing_dec' in params:
                mesh_dec = params['lensing_dec']
    if source_plane_nx is None:
        if params is not None:
            if 'lensing_ssizex' in params:
                source_plane_nx = params['lensing_ssizex']
    if source_plane_ny is None:
        if params is not None:
            if 'lensing_ssizey' in params:
                source_plane_ny = params['lensing_ssizey']
    if source_plane_nchan is None:
        if params is not None:
            if 'nspec' in params:
                source_plane_nchan = params['nspec']
    if source_plane_cenra is None:
        if params is not None:
            if 'lensing_sra' in params:
                source_plane_cenra = params['lensing_sra']
    if source_plane_cendec is None:
        if params is not None:
            if 'lensing_sdec' in params:
                source_plane_cendec = params['lensing_sdec']
    if source_plane_pixsc is None:
        if params is not None:
            if 'lensing_spixsc' in params:
                source_plane_pixsc = params['lensing_spixsc']
    if image_plane_cenra is None:
        if params is not None:
            if 'lensing_imra' in params:
                image_plane_cenra = params['lensing_imra']
    if image_plane_cendec is None:
        if params is not None:
            if 'lensing_imdec' in params:
                image_plane_cendec = params['lensing_imdec']
    if image_plane_pixsc is None:
        if params is not None:
            if 'pixscale' in params:
                image_plane_pixsc = params['pixscale']
    if image_plane_sizex is None:
        if params is not None:
            if 'nx_sky' in params:
                image_plane_sizex = params['nx_sky']
            elif 'fov_npix' in params:
                image_plane_sizex = params['fov_npix']
    if image_plane_sizey is None:
        if params is not None:
            if 'ny_sky' in params:
                image_plane_sizey = params['ny_sky']
            elif 'fov_npix' in params:
                image_plane_sizey = params['fov_npix']

    # return None if everything is None
    # this means that the input params dict does not contain a lensing model
    if params is not None and \
        mesh_file is None and \
        mesh_ra is None and \
        mesh_dec is None: # pragma: no cover
        return None

    # check error if the input params dict does not contain all mandatory lensing transformation keys
    has_error = False
    if mesh_file is None:
        has_error = True
        logger.error('Error! The input mesh_file is invalid or key \'lensing_mesh\' is not in the input params dict!')
    if mesh_ra is None:
        has_error = True
        logger.error('Error! The input mesh_ra is invalid or key \'lensing_ra\' is not in the input params dict!')
    if mesh_dec is None:
        has_error = True
        logger.error('Error! The input mesh_dec is invalid or key \'lensing_dec\' is not in the input params dict!')
    if source_plane_nx is None:
        has_error = True
        logger.error('Error! The input source_plane_nx is invalid or key \'lensing_ssizex\' is not in the input params dict!')
    if source_plane_ny is None:
        has_error = True
        logger.error('Error! The input source_plane_ny is invalid or key \'lensing_ssizey\' is not in the input params dict!')
    if source_plane_nchan is None:
        has_error = True
        logger.error('Error! The input source_plane_nchan is invalid or key \'nspec\' is not in the input params dict!')
    if source_plane_cenra is None:
        has_error = True
        logger.error('Error! The input source_plane_cenra is invalid or key \'lensing_sra\' is not in the input params dict!')
    if source_plane_cendec is None:
        has_error = True
        logger.error('Error! The input source_plane_cendec is invalid or key \'lensing_sdec\' is not in the input params dict!')
    if source_plane_pixsc is None:
        has_error = True
        logger.error('Error! The input source_plane_pixsc is invalid or key \'lensing_spixsc\' is not in the input params dict!')
    if image_plane_cenra is None:
        has_error = True
        logger.error('Error! The input image_plane_cenra is invalid or key \'lensing_imra\' is not in the input params dict!')
    if image_plane_cendec is None:
        has_error = True
        logger.error('Error! The input image_plane_cendec is invalid or key \'lensing_imdec\' is not in the input params dict!')
    if image_plane_pixsc is None:
        has_error = True
        logger.error('Error! The input image_plane_pixsc is invalid or key \'pixscale\' is not in the input params dict!')
    if image_plane_sizex is None:
        has_error = True
        logger.error('Error! The input image_plane_sizex is invalid or key \'fov_npix\' is not in the input params dict!')
    if image_plane_sizey is None:
        has_error = True
        logger.error('Error! The input image_plane_sizey is invalid or key \'fov_npix\' is not in the input params dict!')

    # raise exception for the error
    if has_error:
        raise Exception('Error occurred! Please check error messages above.')
        # return None

    # prepend lensing_datadir path to the mesh file
    if mesh_dir is not None and mesh_dir != '':
        mesh_file = os.path.join(mesh_dir, mesh_file)
	
    # unzip
    if mesh_file.endswith('.gz'):
        import gzip
        import shutil
        unzip_mesh_file = mesh_file.rstrip('.gz')
        with gzip.open(mesh_file, 'rb') as f_in:
            with open(unzip_mesh_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        mesh_file = unzip_mesh_file


    # define logMessage function
    if logger.level == logging.DEBUG: # pragma: no cover
        logMessage = logger.debug
    else:
        logMessage = logger.info

    # try to reuse a LensingTransformer from cache
    global cached_lensing_transformer_dict
    if reuse_lensing_transformer is None:
        if reuse_cached_lensing_transformer:
            if cached_lensing_transformer_dict['0'] is not None:
                reuse_lensing_transformer = cached_lensing_transformer_dict['0']

    # try to reuse a LensingTransformer if one has an input
    # <TODO> note that we have some tolerance for reusing a lensing transfomer
    #        e.g., 0.01 arcsec in RA Dec, 0.001 arcsec in pixsize
    if reuse_lensing_transformer is not None:
        reuse_check_ok = True
        if reuse_lensing_transformer.mesh_file != mesh_file:
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for mesh_file')
        elif not np.isclose(reuse_lensing_transformer.mesh_ra, mesh_ra, rtol=1e-6, atol=0.01/3600.):
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for mesh_ra')
        elif not np.isclose(reuse_lensing_transformer.mesh_dec, mesh_dec, rtol=1e-6, atol=0.01/3600.):
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for mesh_dec')
        elif reuse_lensing_transformer.source_plane_nx != source_plane_nx:
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for source_plane_nx')
        elif reuse_lensing_transformer.source_plane_ny != source_plane_ny:
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for source_plane_ny')
        elif reuse_lensing_transformer.source_plane_nchan != source_plane_nchan:
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for source_plane_nchan')
        elif not np.isclose(reuse_lensing_transformer.source_plane_cenra, source_plane_cenra, rtol=1e-7, atol=0.01/3600.):
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for source_plane_cenra')
        elif not np.isclose(reuse_lensing_transformer.source_plane_cendec, source_plane_cendec, rtol=1e-7, atol=0.01/3600.):
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for source_plane_cendec')
        elif not np.isclose(reuse_lensing_transformer.source_plane_pixsc, source_plane_pixsc, rtol=1e-4, atol=0.001):
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for source_plane_pixsc')
        elif not np.isclose(reuse_lensing_transformer.image_plane_cenra, image_plane_cenra, rtol=1e-7, atol=0.01/3600.):
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for image_plane_cenra')
        elif not np.isclose(reuse_lensing_transformer.image_plane_cendec, image_plane_cendec, rtol=1e-7, atol=0.01/3600.):
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for image_plane_cendec')
        elif not np.isclose(reuse_lensing_transformer.image_plane_pixsc, image_plane_pixsc, rtol=1e-4, atol=0.001):
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for image_plane_pixsc')
        elif reuse_lensing_transformer.image_plane_sizex != image_plane_sizex:
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for image_plane_sizex')
        elif reuse_lensing_transformer.image_plane_sizey != image_plane_sizey:
            reuse_check_ok = False
            logger.debug('reusing lensing transformer check failed for image_plane_sizey')
        #
        if reuse_check_ok:
            if verbose or (logger.level == logging.DEBUG):
                logMessage('reusing lensing transformer '+str(hex(id(reuse_lensing_transformer)))+
                           ' in thread '+str(multiprocessing.current_process().pid)+' '+str(hex(threading.currentThread().ident))
                          )
            return reuse_lensing_transformer
        else:
            del reuse_lensing_transformer
            reuse_lensing_transformer = None

    # create LensingTransformer
    if ((cache_lensing_transformer and cached_lensing_transformer_dict['0'] is None)) or \
       (logger.level == logging.DEBUG) or \
       (verbose == True):
        # print message when
        #   this is the first time creating this object and the cache is empty
        #   debug mode
        #   or verbose
        logMessage('creating lensing transformer with:\n'+
                   '    mesh_file = %r, \n'%(mesh_file)+
                   '    mesh_ra = %s, \n'%(mesh_ra)+
                   '    mesh_dec = %s, \n'%(mesh_dec)+
                   '    source_plane_nx = %s, \n'%(source_plane_nx)+
                   '    source_plane_ny = %s, \n'%(source_plane_ny)+
                   '    source_plane_nchan = %s, \n'%(source_plane_nchan)+
                   '    source_plane_cenra = %s, \n'%(source_plane_cenra)+
                   '    source_plane_cendec = %s, \n'%(source_plane_cendec)+
                   '    source_plane_pixsc = %s, \n'%(source_plane_pixsc)+
                   '    image_plane_cenra = %s, \n'%(image_plane_cenra)+
                   '    image_plane_cendec = %s, \n'%(image_plane_cendec)+
                   '    image_plane_pixsc = %s, \n'%(image_plane_pixsc)+
                   '    image_plane_sizex = %s, \n'%(image_plane_sizex)+
                   '    image_plane_sizey = %s\n'%(image_plane_sizey)+
                   '  in thread '+str(multiprocessing.current_process().pid)+' '+str(hex(threading.currentThread().ident))
                  )
    lensing_transformer = LensingTransformer(\
            mesh_file = mesh_file,
            mesh_ra = mesh_ra,
            mesh_dec = mesh_dec,
            source_plane_nx = source_plane_nx,
            source_plane_ny = source_plane_ny,
            source_plane_nchan = source_plane_nchan,
            source_plane_cenra = source_plane_cenra,
            source_plane_cendec = source_plane_cendec,
            source_plane_pixsc = source_plane_pixsc,
            image_plane_cenra = image_plane_cenra,
            image_plane_cendec = image_plane_cendec,
            image_plane_pixsc = image_plane_pixsc,
            image_plane_sizex = image_plane_sizex,
            image_plane_sizey = image_plane_sizey,
            verbose = verbose,
        )
    if verbose:
        logMessage('created lensing transformer '+str(hex(id(lensing_transformer)))+
                   ' in thread '+str(multiprocessing.current_process().pid)+' '+str(hex(threading.currentThread().ident))
                  )

    # global cached_lensing_transformer_dict
    if cache_lensing_transformer:
        cached_lensing_transformer_dict['0'] = lensing_transformer
    else:
        cached_lensing_transformer_dict['0'] = None

    # return
    return lensing_transformer


def _get_cached_lensing_transformer():
    global cached_lensing_transformer_dict
    return cached_lensing_transformer_dict['0']



