# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Testing of DYSMALPY model (component) calculations

import datetime
import math
import numpy as np
import os
import pytest
import shutil

import astropy.io as fits
import astropy.units as u
from spectral_cube import SpectralCube

from dysmalpy.utils_least_chi_squares_1d_fitter import LeastChiSquares1D
from dysmalpy.utils import gaus_fit_sp_opt_leastsq

import logging
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.DEBUG)

# TESTING DIRECTORY
path = os.path.abspath(__file__)
_dir_tests = os.path.dirname(path) + os.sep
_dir_tests_data = _dir_tests+'test_data_lensing' + os.sep



class TestUtilsLeastChiSquares():

    def test_read_data(self):

        # load data cube
        self.data_cube = SpectralCube.read(os.path.join(_dir_tests_data, 'fdata_cube.fits.gz'))
        self.data_mask = SpectralCube.read(os.path.join(_dir_tests_data, 'fdata_mask3D.fits.gz'))

        assert self.data_cube is not None
        assert self.data_mask is not None

        # compute moment maps as initial guess
        self.mom0 = self.data_cube.moment0().to(u.km/u.s).value
        self.mom1 = self.data_cube.moment1().to(u.km/u.s).value
        self.mom2 = self.data_cube.linewidth_sigma().to(u.km/u.s).value
        self.initparams = np.array([self.mom0 / np.sqrt(2 * np.pi) / np.abs(self.mom2), self.mom1, self.mom2])


    def test_object_construction_with_mask_3d(self):
        self.test_read_data()
        this_fitting_mask = self.data_mask.unmasked_data[:,:,:].value
        this_fitting_verbose = True

        my_least_chi_squares_1d_fitter = LeastChiSquares1D(\
                        x = self.data_cube.spectral_axis.to(u.km/u.s).value,
                        data = self.data_cube.unmasked_data[:,:,:].value,
                        dataerr = None,
                        datamask = this_fitting_mask,
                        initparams = self.initparams,
                        nthread = 4,
                        verbose = this_fitting_verbose)

        assert my_least_chi_squares_1d_fitter is not None


    def test_object_construction_with_mask_2d(self):
        self.test_read_data()
        this_fitting_mask = np.all(self.data_mask.unmasked_data[:,:,:].value > 0, axis=0).astype(int)
        this_fitting_verbose = True

        my_least_chi_squares_1d_fitter = LeastChiSquares1D(\
                        x = self.data_cube.spectral_axis.to(u.km/u.s).value,
                        data = self.data_cube.unmasked_data[:,:,:].value,
                        dataerr = None,
                        datamask = this_fitting_mask,
                        initparams = self.initparams,
                        nthread = 4,
                        verbose = this_fitting_verbose)

        assert my_least_chi_squares_1d_fitter is not None


    def test_object_construction_with_invalid_x(self):
        self.test_read_data()
        with pytest.raises(Exception) as e:
            my_least_chi_squares_1d_fitter = LeastChiSquares1D(\
                            x = None,
                            data = self.data_cube.unmasked_data[:,:,:].value,
                            dataerr = None,
                            datamask = 'auto',
                            initparams = self.initparams,
                            nthread = 4,
                            verbose = True)

        assert str(e.value) == 'Please input x.'


    def test_object_construction_with_invalid_data(self):
        self.test_read_data()
        with pytest.raises(Exception) as e:
            my_least_chi_squares_1d_fitter = LeastChiSquares1D(\
                            x = self.data_cube.spectral_axis.to(u.km/u.s).value,
                            data = None,
                            dataerr = None,
                            datamask = 'auto',
                            initparams = self.initparams,
                            nthread = 4,
                            verbose = True)

        assert str(e.value) == 'Please input data.'


    def test_object_construction_with_invalid_mask(self):
        self.test_read_data()
        with pytest.raises(Exception) as e:
            my_least_chi_squares_1d_fitter = LeastChiSquares1D(\
                            x = self.data_cube.spectral_axis.to(u.km/u.s).value,
                            data = self.data_cube.unmasked_data[:,:,:].value,
                            dataerr = None,
                            datamask = np.array([1,1,1]),
                            initparams = self.initparams,
                            nthread = 4,
                            verbose = True)

        assert str(e.value) == 'Error! datamask should be a 2D or 3D array!'


    def test_object_construction_with_dataerr(self):
        self.test_read_data()
        this_fitting_dataerr = np.full(self.data_cube.shape, fill_value = np.nanmax(self.data_cube) * 0.05)
        this_fitting_mask = 'auto'
        this_fitting_verbose = True

        my_least_chi_squares_1d_fitter = LeastChiSquares1D(\
                        x = self.data_cube.spectral_axis.to(u.km/u.s).value,
                        data = self.data_cube.unmasked_data[:,:,:].value,
                        dataerr = this_fitting_dataerr,
                        datamask = this_fitting_mask,
                        initparams = self.initparams,
                        nthread = 4,
                        verbose = this_fitting_verbose)

        assert my_least_chi_squares_1d_fitter is not None


    def test_object_construction_with_neagtive_cube(self):
        self.test_read_data()
        this_fitting_data = -np.abs(self.data_cube.unmasked_data[:,:,:].value)
        this_fitting_mask = 'auto'
        this_fitting_verbose = True

        my_least_chi_squares_1d_fitter = LeastChiSquares1D(\
                        x = self.data_cube.spectral_axis.to(u.km/u.s).value,
                        data = this_fitting_data,
                        dataerr = None,
                        datamask = this_fitting_mask,
                        initparams = self.initparams,
                        nthread = 4,
                        verbose = this_fitting_verbose)

        assert my_least_chi_squares_1d_fitter is not None


    def test_object_construction_with_big_endian(self):
        self.test_read_data()
        # TODO
        pass


    def test_utils_least_chi_squares_1d_fitter(self):
        self.test_read_data()

        # prepare method 1 output array
        flux = np.zeros(self.mom0.shape)
        vel = np.zeros(self.mom0.shape)
        disp = np.zeros(self.mom0.shape)

        # run method 1
        this_fitting_mask = 'auto'
        this_fitting_verbose = True

        my_least_chi_squares_1d_fitter = LeastChiSquares1D(\
                        x = self.data_cube.spectral_axis.to(u.km/u.s).value,
                        data = self.data_cube.unmasked_data[:,:,:].value,
                        dataerr = None,
                        datamask = this_fitting_mask,
                        initparams = self.initparams,
                        nthread = 4,
                        verbose = this_fitting_verbose)

        assert my_least_chi_squares_1d_fitter is not None

        my_least_chi_squares_1d_fitter.printLibInfo()

        print('my_least_chi_squares_1d_fitter '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
        my_least_chi_squares_1d_fitter.runFitting()
        flux = my_least_chi_squares_1d_fitter.outparams[0,:,:] * np.sqrt(2 * np.pi) * my_least_chi_squares_1d_fitter.outparams[2,:,:]
        vel = my_least_chi_squares_1d_fitter.outparams[1,:,:]
        disp = my_least_chi_squares_1d_fitter.outparams[2,:,:]
        flux[np.isnan(flux)] = 0.0 #<DZLIU><DEBUG># 20210809 fixing this bug
        print('my_least_chi_squares_1d_fitter '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#


        # prepare method 2 output array
        flux2 = np.zeros(self.mom0.shape)
        vel2 = np.zeros(self.mom0.shape)
        disp2 = np.zeros(self.mom0.shape)

        # run method 2 to compare the results
        for i in range(self.mom0.shape[0]):
            for j in range(self.mom0.shape[1]):
                if i==0 and j==0:
                    print('gaus_fit_sp_opt_leastsq '+str(self.mom0.shape[0])+'x'+str(self.mom0.shape[1])+' '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
                best_fit = gaus_fit_sp_opt_leastsq(self.data_cube.spectral_axis.to(u.km/u.s).value,
                                                   self.data_cube.unmasked_data[:,i,j].value,
                                                   self.mom0[i,j], self.mom1[i,j], self.mom2[i,j])
                flux2[i,j] = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                vel2[i,j] = best_fit[1]
                disp2[i,j] = best_fit[2]
                if i==self.mom0.shape[0]-1 and j==self.mom0.shape[1]-1:
                    print('gaus_fit_sp_opt_leastsq '+str(self.mom0.shape[0])+'x'+str(self.mom0.shape[1])+' '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#

        # compare
        mask = np.logical_and(np.isfinite(vel), np.isfinite(vel2))
        assert np.all(np.isclose(vel[mask], vel2[mask], rtol=0.05))




# pragma: no cover

if __name__ == '__main__':

    TestUtilsLeastChiSquares().test_object_construction_with_mask_3d()

    #TestUtilsLeastChiSquares().test_utils_least_chi_squares_1d_fitter()

    print('All done!')
