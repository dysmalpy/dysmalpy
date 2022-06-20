# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Testing of DYSMALPY model (component) calculations
# Note that the tests will create a "PYTEST_OUTPUT"

import copy
import datetime
import math
import numpy as np
import os
import pytest
import shutil
import time
from collections import OrderedDict

from astropy.io import fits
import astropy.units as u
from spectral_cube import SpectralCube

import logging
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.DEBUG)

from dysmalpy import fitting
from dysmalpy.fitting_wrappers import utils_io as fw_utils_io
from dysmalpy import lensing
from dysmalpy.lensing import has_lensing_transform_keys_in_params
from dysmalpy.lensing import setup_lensing_transformer_from_params
from dysmalpy.lensing import get_cached_lensing_transformer
from dysmalpy.lensing import LensingTransformer


# TESTING DIRECTORY
path = os.path.abspath(__file__)
_dir_tests = os.path.dirname(path) + os.sep
_dir_tests_data = _dir_tests+'test_data_lensing' + os.sep



class TestLensing():

    def _unzip_lens_mesh(self, mesh_file, outdir=None):
        # unzip
        if mesh_file.endswith('.gz'):
            import gzip
            import shutil
            if outdir is None:
                unzip_mesh_file = mesh_file.rstrip('.gz')
            else:
                unzip_mesh_file = os.path.join(outdir, os.path.basename(mesh_file).rstrip('.gz'))
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
            with gzip.open(mesh_file, 'rb') as f_in:
                with open(unzip_mesh_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return unzip_mesh_file
        else:
            if outdir is None:
                return mesh_file
            else:
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
                copied_mesh_file = os.path.join(outdir, os.path.basename(mesh_file))
                shutil.copy2(mesh_file, copied_mesh_file)
                return copied_mesh_file
    

    def test_read_params(self):

        # load params
        self.params = fw_utils_io.read_fitting_params(fname = os.path.join(_dir_tests_data, 'fitting_3D_mpfit.params'))
        assert self.params is not None
        assert 'outdir' in self.params
        assert 'lensing_mesh' in self.params
        assert 'lensing_ra' in self.params
        assert 'lensing_dec' in self.params
        assert 'lensing_sra' in self.params
        assert 'lensing_sdec' in self.params
        assert 'lensing_ssizex' in self.params
        assert 'lensing_ssizey' in self.params
        assert 'lensing_spixsc' in self.params
        assert 'lensing_imra' in self.params
        assert 'lensing_imdec' in self.params
        assert has_lensing_transform_keys_in_params(self.params)
        self.outdir = os.path.join(_dir_tests_data, self.params['outdir'])
        fitting.ensure_dir(self.outdir)
        assert os.path.isdir(self.outdir)

        _lens_mesh_file = os.path.join(_dir_tests_data, self.params['lensing_mesh'])
        assert os.path.isfile(_lens_mesh_file)
        _lens_mesh_file = self._unzip_lens_mesh(_lens_mesh_file, outdir=self.outdir) # copy the mesh.dat file to output dir
        assert os.path.isfile(_lens_mesh_file)
        self.params['lensing_datadir'] = self.outdir
        self.params['lensing_mesh'] = os.path.basename(_lens_mesh_file)


    def test_read_data(self):
        self.test_read_params()

        # load data cube
        self.data_cube = SpectralCube.read(os.path.join(_dir_tests_data, 'fdata_cube.fits.gz'))
        self.data_mask = SpectralCube.read(os.path.join(_dir_tests_data, 'fdata_mask3D.fits.gz'))

        print('self.data_cube', self.data_cube)
        print('self.data_mask', self.data_mask)
        assert self.data_cube is not None
        assert self.data_mask is not None

        if self.params['datadir'] is None:
            self.params['datadir'] = _dir_tests_data

        #gal, fit_dict = fw_utils_io.setup_single_object_3D(params=self.params)
        gal, output_options = fw_utils_io.setup_single_galaxy(params=self.params)

        print('gal', gal)
        assert gal is not None
        assert hasattr(gal, 'model')
        assert hasattr(gal.model, 'simulate_cube')

        self.gal = gal


    def test_object_construction(self):
        self.test_read_params()

        kwargs = self.params
        nspec = kwargs['nspec']
        nx_sky = kwargs['fov_npix']
        ny_sky = kwargs['fov_npix']
        rstep = kwargs['pixscale']
        oversample = kwargs['oversample']
        oversize = kwargs['oversize']

        this_lensing_transformer = setup_lensing_transformer_from_params(\
                params = kwargs,
                source_plane_nchan = nspec,
                image_plane_sizex = nx_sky * oversample * oversize,
                image_plane_sizey = ny_sky * oversample * oversize,
                image_plane_pixsc = rstep / oversample,
                reuse_lensing_transformer = None,
                cache_lensing_transformer = False,
                reuse_cached_lensing_transformer = False,
                verbose = True,
            )

        assert this_lensing_transformer is not None


    def test_object_construction_with_params_alone(self):
        self.test_read_params()

        kwargs = self.params
        kwargs['nx_sky'] = kwargs['fov_npix']
        kwargs['ny_sky'] = kwargs['fov_npix']

        this_lensing_transformer = setup_lensing_transformer_from_params(\
                params = kwargs,
                reuse_lensing_transformer = None,
                cache_lensing_transformer = False,
                reuse_cached_lensing_transformer = False,
                verbose = True,
            )

        assert this_lensing_transformer is not None


    def test_object_construction_with_valid_inputs(self):
        self.test_read_params()

        mesh_file = os.path.join(_dir_tests_data, os.path.basename(self.params['lensing_mesh']))
        mesh_ra = self.params['lensing_ra']
        mesh_dec = self.params['lensing_dec']

        kwargs = self.params

        this_lensing_transformer = LensingTransformer(\
                mesh_file,
                mesh_ra,
                mesh_dec,
                source_plane_data_cube = np.full(
                        [self.params['nspec'],
                         self.params['lensing_ssizey'],
                         self.params['lensing_ssizex']],
                        fill_value = 0.0,
                    ),
                source_plane_cenra = self.params['lensing_sra'],
                source_plane_cendec = self.params['lensing_sdec'],
                source_plane_ceny = (self.params['lensing_ssizey'] + 1.0) / 2.0,
                source_plane_cenx = (self.params['lensing_ssizex'] + 1.0) / 2.0,
                source_plane_pixsc = self.params['lensing_spixsc'],
                image_plane_ceny = (self.params['fov_npix'] + 1.0) / 2.0,
                image_plane_cenx = (self.params['fov_npix'] + 1.0) / 2.0,
                image_plane_pixsc = self.params['pixscale'],
                verbose = True,
            )

        assert this_lensing_transformer is not None


    def test_object_construction_with_incomplete_inputs(self):
        self.test_read_params()

        mesh_file = os.path.join(_dir_tests_data, os.path.basename(self.params['lensing_mesh']))
        mesh_ra = self.params['lensing_ra']
        mesh_dec = self.params['lensing_dec']

        kwargs = self.params

        with pytest.raises(Exception) as e:
            this_lensing_transformer = LensingTransformer(\
                    mesh_file,
                    mesh_ra,
                    mesh_dec,
                    source_plane_data_cube = np.full(
                            [self.params['nspec'],
                             self.params['lensing_ssizey'],
                             self.params['lensing_ssizex']],
                            fill_value = 0.0,
                        ),
                    source_plane_ceny = (self.params['lensing_ssizey'] + 1.0) / 2.0,
                    source_plane_cenx = (self.params['lensing_ssizex'] + 1.0) / 2.0,
                    image_plane_ceny = (self.params['fov_npix'] + 1.0) / 2.0,
                    image_plane_cenx = (self.params['fov_npix'] + 1.0) / 2.0,
                    verbose = True,
                )

        print('str(e.value)', str(e.value))
        assert str(e.value) != ''


    def test_object_construction_update_a_wrong_input(self):
        self.test_read_params()

        kwargs = self.params

        with pytest.raises(Exception) as e:
            this_lensing_transformer = setup_lensing_transformer_from_params(mesh_file = None)

        assert str(e.value) == 'Error occurred! Please check error messages above.'

        this_lensing_transformer = setup_lensing_transformer_from_params(\
                params = kwargs,
                reuse_lensing_transformer = None,
                cache_lensing_transformer = False,
                reuse_cached_lensing_transformer = False,
                verbose = True,
            )

        assert this_lensing_transformer is not None

        with pytest.raises(Exception) as e:
            this_lensing_transformer.setSourcePlaneDataCube(np.zeros((100,100)))

        assert str(e.value) == 'Error! The input data cube should have 3 dimensions!'

        this_lensing_transformer.source_plane_cenx = (self.params['lensing_ssizex'] + 1.0) / 2.0
        this_lensing_transformer.source_plane_ceny = (self.params['lensing_ssizey'] + 1.0) / 2.0
        this_lensing_transformer.setSourcePlaneDataCube(
                source_plane_data_cube = np.full(
                        [self.params['nspec'],
                         self.params['lensing_ssizey'],
                         self.params['lensing_ssizex']],
                        fill_value = 0.0,
                    ),
                source_plane_ceny = None,
                source_plane_cenx = None,
            )

        assert this_lensing_transformer is not None

        with pytest.raises(Exception) as e:
            this_lensing_transformer.updateSourcePlaneDataCube(np.zeros((100,100,100)))

        print('str(e.value)', str(e.value))
        assert str(e.value) != ''


    def test_reusing_a_lensing_transformer_for_changed_params(self):
        self.test_read_params()

        kwargs = self.params

        this_lensing_transformer_1 = setup_lensing_transformer_from_params(\
                params = kwargs,
                reuse_lensing_transformer = None,
                cache_lensing_transformer = False,
                reuse_cached_lensing_transformer = False,
                verbose = True,
            )

        # first reusing looks good
        this_lensing_transformer_2 = setup_lensing_transformer_from_params(\
                params = kwargs,
                reuse_lensing_transformer = this_lensing_transformer_1,
                cache_lensing_transformer = False,
                reuse_cached_lensing_transformer = False,
                verbose = True,
            )

        assert this_lensing_transformer_1 is this_lensing_transformer_2

        # then change params and reuse
        params_changed = OrderedDict()
        #params_changed['mesh_file'] = os.path.join(self.outdir, 'lensing_mesh_copy.dat') # 20211118 now the mesh_file has already been copied to outdir in self.test_read_params()
        params_changed['mesh_ra'] = kwargs['lensing_ra']*1.05
        params_changed['mesh_dec'] = kwargs['lensing_dec']*1.05
        params_changed['source_plane_cenra'] = kwargs['lensing_sra']*1.05
        params_changed['source_plane_cendec'] = kwargs['lensing_sdec']*1.05
        params_changed['source_plane_nx'] = kwargs['lensing_ssizex']+5
        params_changed['source_plane_ny'] = kwargs['lensing_ssizey']+5
        params_changed['source_plane_nchan'] = kwargs['nspec']+5
        params_changed['source_plane_pixsc'] = kwargs['lensing_spixsc']*1.05
        params_changed['image_plane_cenra'] = kwargs['lensing_imra']*1.05
        params_changed['image_plane_cendec'] = kwargs['lensing_imdec']*1.05
        params_changed['image_plane_pixsc'] = kwargs['pixscale']*1.05
        params_changed['image_plane_sizex'] = kwargs['fov_npix']+5
        params_changed['image_plane_sizey'] = kwargs['fov_npix']+5
        #shutil.copy2(os.path.join(_dir_tests_data, kwargs['lensing_mesh']),
        #             params_changed['mesh_file']) # 20211118 now the mesh_file has already been copied to outdir in self.test_read_params()
        for key in params_changed:
            # test changing each lensing key
            print('testing params changed: ' + str(key))
            params_changed_3 = {key: params_changed[key]}
            this_lensing_transformer_3 = setup_lensing_transformer_from_params(\
                    params = kwargs,
                    **params_changed_3,
                    reuse_lensing_transformer = this_lensing_transformer_2,
                    cache_lensing_transformer = False,
                    reuse_cached_lensing_transformer = False,
                    verbose = True,
                )

            assert this_lensing_transformer_2 is not None
            assert this_lensing_transformer_3 is not None
            assert this_lensing_transformer_3 is not this_lensing_transformer_2


    def test_reusing_a_lensing_transformer_with_module_cache(self):
        self.test_read_params()

        kwargs = self.params

        this_lensing_transformer_1 = setup_lensing_transformer_from_params(\
                params = kwargs,
                reuse_lensing_transformer = None,
                cache_lensing_transformer = True,
                reuse_cached_lensing_transformer = False,
                verbose = True,
            )

        this_lensing_transformer_2 = setup_lensing_transformer_from_params(\
                params = kwargs,
                reuse_lensing_transformer = None,
                cache_lensing_transformer = True,
                reuse_cached_lensing_transformer = True,
                verbose = True,
            )

        assert this_lensing_transformer_1 is this_lensing_transformer_2



    def test_lensing_transformation(self):
        self.test_read_params()
        self.test_read_data()

        #obs = self.gal.observations.items()[0]
        #obs.create_single_obs_model_data()
        self.gal.create_model_data() # this calls obs.create_single_obs_model_data()
        
        # obs.create_single_obs_model_data() calls 

        logger.debug('self.gal.observations: {}'.format(self.gal.observations))
        obs0key = list(self.gal.observations.keys())[0]
        obs0 = self.gal.observations[obs0key]
        logger.debug('obs0.name: {}'.format(obs0.name))
        assert obs0.model_data is not None
        
        logger.debug('type(obs0.model_data): {}'.format(type(obs0.model_data)))
        fits.PrimaryHDU(data=obs0.model_data.data).writeto('obs0.model_data.fits', overwrite=True)
        
        this_lensing_transformer = get_cached_lensing_transformer()
        fits.PrimaryHDU(data=this_lensing_transformer.image_plane_data_cube).writeto('obs0.image_plane_data_cube.fits', overwrite=True)
        fits.PrimaryHDU(data=this_lensing_transformer.source_plane_data_cube).writeto('obs0.source_plane_data_cube.fits', overwrite=True)


    # def test_lensing_transformation_in_a_wrong_way(self):
    #     self.test_read_params()
    #     self.test_read_data()

    #     kwargs = self.params

    #     this_lensing_transformer = setup_lensing_transformer_from_params(\
    #             params = kwargs,
    #         )

    #     assert this_lensing_transformer is not None

    #     this_lensing_transformer.image_plane_cenx = (kwargs['fov_npix'] + 1.0) / 2.0
    #     this_lensing_transformer.image_plane_ceny = (kwargs['fov_npix'] + 1.0) / 2.0
    #     this_lensing_transformer.performLensingTransformation()



# pragma: no cover

if __name__ == '__main__':

    TestLensing().test_lensing_transformation()

    #TestLensing().test_reusing_a_lensing_transformer_for_changed_params()

    print('All done!')
