# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Testing of DYSMALPY model (component) calculations

import pytest

import numpy as np
import astropy.units as u

from dysmalpy.fitting_wrappers import utils_io as fw_utils_io

from dysmalpy import galaxy, models, parameters, instrument, observation

import logging
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.DEBUG)

class HelperSetups(object):

    def __init__(self):
        self.z = 1.613
        self.name = 'GS4_43501'

    def setup_diskbulge(self):
        # Baryonic Component: Combined Disk+Bulge
        total_mass =    11.0    # M_sun
        bt =            0.3     # Bulge-Total ratio
        r_eff_disk =    5.0     # kpc
        n_disk =        1.0
        invq_disk =     5.0
        r_eff_bulge =   1.0     # kpc
        n_bulge =       4.0
        invq_bulge =    1.0
        noord_flat =    True    # Switch for applying Noordermeer flattening

        # Fix components
        bary_fixed = {'total_mass': False,
                      'r_eff_disk': False,
                      'n_disk': True,
                      'r_eff_bulge': True,
                      'n_bulge': True,
                      'bt': False}

        # Set bounds
        bary_bounds = {'total_mass': (10, 13),
                       'r_eff_disk': (1.0, 30.0),
                       'n_disk': (1, 8),
                       'r_eff_bulge': (1, 5),
                       'n_bulge': (1, 8),
                       'bt': (0, 1)}

        bary = models.DiskBulge(total_mass=total_mass, bt=bt,
                                r_eff_disk=r_eff_disk, n_disk=n_disk,
                                invq_disk=invq_disk,
                                r_eff_bulge=r_eff_bulge, n_bulge=n_bulge,
                                invq_bulge=invq_bulge,
                                noord_flat=noord_flat,
                                name='disk+bulge',
                                fixed=bary_fixed, bounds=bary_bounds,
                                gas_component='total')

        bary.r_eff_disk.prior = parameters.BoundedGaussianPrior(center=5.0, stddev=1.0)

        return bary

    def setup_NFW(self):
        # NFW Halo component
        mvirial = 12.0
        conc = 5.0
        fdm = 0.5
        halo_fixed = {'mvirial': False,
                      'conc': True,
                      'fdm': False}

        halo_bounds = {'mvirial': (10, 13),
                       'conc': (1, 20),
                       'fdm': (0., 1.)}

        halo = models.NFW(mvirial=mvirial, conc=conc, fdm=fdm,z=self.z,
                          fixed=halo_fixed, bounds=halo_bounds, name='halo')

        halo.fdm.tied = fw_utils_io.tie_fdm
        halo.mvirial.prior = parameters.BoundedGaussianPrior(center=11.5, stddev=0.5)

        return halo

    def setup_const_dispprof(self):
        # Dispersion profile
        sigma0 = 39.   # km/s
        disp_fixed = {'sigma0': False}
        disp_bounds = {'sigma0': (10, 200)}

        disp_prof = models.DispersionConst(sigma0=sigma0, fixed=disp_fixed,
                                                  bounds=disp_bounds, name='dispprof',
                                                  tracer='halpha')

        return disp_prof

    def setup_zheight_prof(self):
        # z-height profile
        sigmaz = 0.9   # kpc
        zheight_fixed = {'sigmaz': False}

        zheight_prof = models.ZHeightGauss(sigmaz=sigmaz, name='zheightgaus', fixed=zheight_fixed)
        zheight_prof.sigmaz.tied = fw_utils_io.tie_sigz_reff

        return zheight_prof

    def setup_geom(self):
        # Geometry
        inc = 62.     # degrees
        pa = 142.     # degrees, blue-shifted side CCW from north
        xshift = 0    # pixels from center
        yshift = 0    # pixels from center

        geom_fixed = {'inc': False,
                      'pa': True,
                      'xshift': True,
                      'yshift': True}

        geom_bounds = {'inc': (0, 90),
                       'pa': (90, 180),
                       'xshift': (0, 4),
                       'yshift': (-10, -4)}

        geom = models.Geometry(inc=inc, pa=pa, xshift=xshift, yshift=yshift,
                               fixed=geom_fixed, bounds=geom_bounds, name='geom',
                               obs_name='halpha_1D')

        return geom

    def setup_fullmodel(self, adiabatic_contract=False,
                pressure_support=True, pressure_support_type=1, instrument=None):
        # Initialize the Galaxy, Observation, Instrument, and Model Set
        gal = galaxy.Galaxy(z=self.z, name=self.name)
        obs = observation.Observation(name='halpha_1D', tracer='halpha')
        obs.mod_options.oversample = 3
        obs.mod_options.zcalc_truncate = True

        mod_set = models.ModelSet()

        bary = self.setup_diskbulge()
        halo = self.setup_NFW()
        disp_prof = self.setup_const_dispprof()
        zheight_prof = self.setup_zheight_prof()
        geom = self.setup_geom()
        dimming = self.setup_constant_dimming()

        # Add all of the model components to the ModelSet
        mod_set.add_component(bary, light=True)
        mod_set.add_component(halo)
        mod_set.add_component(disp_prof)
        mod_set.add_component(zheight_prof)
        mod_set.add_component(geom)

        ## Set some kinematic options for calculating the velocity profile
        # pressure_support_type: 1 / Exponential, self-grav [Burkert+10]
        #                        2 / Exact nSersic, self-grav
        #                        3 / Pressure gradient
        mod_set.kinematic_options.adiabatic_contract = adiabatic_contract
        mod_set.kinematic_options.pressure_support = pressure_support
        mod_set.kinematic_options.pressure_support_type = pressure_support_type

        mod_set.dimming = dimming

        # Add the model set and instrument to the Galaxy
        gal.model = mod_set

        if instrument:
            obs.instrument = instrument

        # Add the observation to the Galaxy
        gal.add_observation(obs)

        return gal


    def setup_instrument(self, beam=None, lsf=None):
        inst = instrument.Instrument()

        # Set up the instrument
        pixscale = 0.125*u.arcsec                # arcsec/pixel
        fov = [33, 33]                           # (nx, ny) pixels
        spec_type = 'velocity'                   # 'velocity' or 'wavelength'
        spec_start = -1000*u.km/u.s              # Starting value of spectrum
        spec_step = 10*u.km/u.s                  # Spectral step
        nspec = 201                              # Number of spectral pixels

        inst.pixscale = pixscale
        inst.fov = fov
        inst.spec_type = spec_type
        inst.spec_step = spec_step
        inst.spec_start = spec_start
        inst.nspec = nspec

        # Extraction information
        inst.ndim = 3                            # Dimensionality of data
        inst.moment = False                      # For 1D/2D data, if True then velocities and dispersion calculated from moments
                                                 # Default is False, meaning Gaussian extraction used

        # Set the beam kernel so it doesn't have to be calculated every step
        if beam is not None:
            inst.beam = beam
            inst.set_beam_kernel()
        if lsf is not None:
            inst.lsf = lsf
            inst.set_lsf_kernel()

        return inst
    
    def setup_lsf(self, sig_inst = 45*u.km/u.s):
        # Instrumental spectral resolution
        return instrument.LSF(sig_inst)
    
    def setup_gaussian_beam(self, beamsize = 0.55*u.arcsec):
        # FWHM of beam
        return instrument.GaussianBeam(major=beamsize)

    def setup_double_beam(
        self,
        major1 = 0.55*u.arcsec, 
        major2=0.25*u.arcsec,
        scale2 = 0.2
    ):
        # FWHM of beam
        return instrument.DoubleBeam(
            major1=major1,
            major2=major2,
            scale2=scale2
        )
    
    def setup_moffat_beam(self, major_fwhm=0.55*u.arcsec, beta=3):
        return instrument.Moffat(
             major_fwhm=major_fwhm,
             beta=beta,
        )
    
    def setup_empirical_beam(self, pixscale=0.125*u.arcsec):
        gaus = self.setup_gaussian_beam()
        # Upscale to check normalization
        kernel = gaus.as_kernel(pixscale).array * 5

        return instrument.EmpiricalPSF(
            psf_array=kernel,
            pixscale=pixscale
        )


class TestInstrument:
    helper = HelperSetups()

    def test_gaussian_beam(self):
        beam = self.helper.setup_gaussian_beam()
        lsf = self.helper.setup_lsf()

        inst = self.helper.setup_instrument(beam=beam, lsf=lsf)
        
        assert inst._beam_kernel is not None
        assert inst._lsf_kernel is not None

        assert np.isclose(np.sum(inst._beam_kernel), 1)

    def test_double_beam(self):
        beam = self.helper.setup_double_beam()
        lsf = self.helper.setup_lsf()

        inst = self.helper.setup_instrument(beam=beam, lsf=lsf)
        
        assert inst._beam_kernel is not None
        assert inst._lsf_kernel is not None

        assert np.isclose(np.sum(inst._beam_kernel), 1)

    def test_moffat_beam(self):
        beam = self.helper.setup_moffat_beam()
        lsf = self.helper.setup_lsf()

        inst = self.helper.setup_instrument(beam=beam, lsf=lsf)
        
        assert inst._beam_kernel is not None
        assert inst._lsf_kernel is not None

        assert np.isclose(np.sum(inst._beam_kernel), 1)

    def test_empirical_beam(self):
        beam = self.helper.setup_empirical_beam()
        lsf = self.helper.setup_lsf()

        inst = self.helper.setup_instrument(beam=beam, lsf=lsf)
        
        assert inst._beam_kernel is not None
        assert inst._lsf_kernel is not None

        assert np.isclose(np.sum(inst._beam_kernel), 1)

    def test_empirical_beam_pixscale_mismatch(self):
        beam = self.helper.setup_empirical_beam(pixscale=0.1*u.arcsec)
        lsf = self.helper.setup_lsf()

        with pytest.raises(ValueError) as excinfo:
            _ = self.helper.setup_instrument(beam=beam, lsf=lsf)
        estr = "Cannot set kernel, as EmpiricalPSF beam "
        assert estr in str(excinfo.value)
        

    def test_no_lsf(self):
        beam = self.helper.setup_gaussian_beam()

        inst = self.helper.setup_instrument(beam=beam, lsf=None)
        
        assert inst._beam_kernel is not None
        assert inst._lsf_kernel is None

    def test_no_beam(self):
        lsf = self.helper.setup_lsf()
        inst = self.helper.setup_instrument(beam=None, lsf=lsf)
        
        assert inst._beam_kernel is None
        assert inst._lsf_kernel is not None

