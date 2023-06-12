# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Testing of DYSMALPY model (component) calculations

import os
import shutil

import math

import numpy as np
import astropy.io.fits as fits
import astropy.units as u

from dysmalpy.fitting_wrappers import dysmalpy_make_model
from dysmalpy.fitting_wrappers import utils_io as fw_utils_io

from dysmalpy import galaxy, models, parameters, instrument, config, observation

import logging
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.DEBUG)

# TESTING DIRECTORY
path = os.path.abspath(__file__)
_dir_tests = os.path.dirname(path) + os.sep
_dir_tests_data = _dir_tests+'test_data' + os.sep


# MASSIVE RING DIRECTORIES:
_dir_gaussian_ring_tables = os.getenv('GAUSSIAN_RING_PROFILE_DATADIR', None)


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

    def setup_sersic(self, noord_flat=False):
        # Baryonic Component: Combined Disk+Bulge
        total_mass =    11.0    # M_sun
        r_eff =         5.0     # kpc
        n =             1.0
        invq =          5.0

        # Fix components
        sersic_fixed = {'total_mass': False,
                      'r_eff': False,
                      'n': True}

        # Set bounds
        sersic_bounds = {'total_mass': (10, 13),
                       'r_eff': (1.0, 30.0),
                       'n': (1, 8)}

        sersic = models.Sersic(total_mass=total_mass,r_eff=r_eff, n=n,invq=invq,
                                noord_flat=noord_flat,name='sersic',
                                fixed=sersic_fixed, bounds=sersic_bounds)

        sersic.r_eff.prior = parameters.BoundedGaussianPrior(center=5.0, stddev=1.0)

        return sersic

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

    def setup_TPH(self):
        # TPH Halo component
        mvirial = 12.0
        conc = 5.0
        alpha = 0.
        beta = 3.
        fdm = 0.5

        halo_fixed = {'mvirial': False,
                      'conc': True,
                      'alpha': False,
                      'beta': True,
                      'fdm': False}

        halo_bounds = {'mvirial': (10, 13),
                       'conc': (1, 20),
                       'alpha': (0, 3),
                       'beta': (1,4),
                       'fdm': (0., 1.)}

        halo = models.TwoPowerHalo(mvirial=mvirial, conc=conc,
                            alpha=alpha, beta=beta, fdm=fdm, z=self.z,
                            fixed=halo_fixed, bounds=halo_bounds, name='halo')

        halo.fdm.tied = fw_utils_io.tie_fdm
        halo.fdm.fixed = False

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

    def setup_biconical_outflow(self):
        bicone = models.BiconicalOutflow(n=0.5, vmax=300., rturn=0.5, thetain=30, dtheta=20.,
                                         rend=1., norm_flux=11., tau_flux=5., name='bicone')
        # To fully match old tests, set norm_flux = 0.
        bicone_geom = models.Geometry(inc=10., pa=30., xshift=0., yshift=0., name='outflow_geom',
                                      obs_name='halpha_1d')
        bicone_disp = models.DispersionConst(sigma0=250., name='outflow_dispprof', tracer='halpha')
        return bicone, bicone_geom, bicone_disp

    def setup_uniform_inflow(self):
        # Negative vr is inflow
        inflow = models.UniformRadialFlow(vr=-90, name='inflow')
        return inflow

    def setup_bar_inflow(self):
        # Negative vbar is inflow; phi=90 is along galaxy minor axis
        bar = models.UniformBarFlow(vbar=-90., phi=90., bar_width=2., name='bar')
        return bar

    def setup_wedge_inflow(self):
        # Negative vbar is inflow; phi=90 is along galaxy minor axis
        wedge = models.UniformWedgeFlow(vr=-90., phi=90., theta=60, name='wedge')
        return wedge

    def setup_spiral_density_waves_flatVrot(self):
        def constV(R):
            return R*0. + 100.

        def constrho(R):
            return R*0. + 1.

        def dVrot_dR(R):
            return R*0.

        def f_spiral(R, m, cs, Om_p, Vrot):
            _amp = np.sqrt(m**2 -2.) * Vrot(R) / cs
            return _amp * np.log(R)

        def k_spiral(R, m, cs, Om_p, Vrot):
            _amp = np.sqrt(m**2 -2.) * Vrot(R) / cs
            # k = df/dR
            return _amp/R

        spiral = models.SpiralDensityWave(m=2, phi0=0., cs=50., epsilon=1.0,
                                               Om_p=0., Vrot=constV, rho0=constrho,
                                               f=f_spiral, k=k_spiral, dVrot_dR=dVrot_dR,
                                               name='spiral')
        return spiral


    def setup_massive_gaussian_ring(self):
        gausring = models.GaussianRing(total_mass=10., R_peak=5., FWHM=2.5, name='gausring')
        return gausring

    def setup_constant_dimming(self):
        dimming = models.ConstantDimming(amp_lumtoflux=1.e-10)
        # To fully match old test cases, set this to 1.e-11 / 0.7
        return dimming

    def setup_fullmodel(self, adiabatic_contract=False,
                pressure_support=True, pressure_support_type=1, instrument=True):
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
            inst = self.setup_instrument()
            obs.instrument = inst

        # Add the observation to the Galaxy
        gal.add_observation(obs)

        return gal


    def setup_instrument(self):
        inst = instrument.Instrument()

        # Set up the instrument
        pixscale = 0.125*u.arcsec                # arcsec/pixel
        fov = [33, 33]                           # (nx, ny) pixels
        beamsize = 0.55*u.arcsec                 # FWHM of beam
        spec_type = 'velocity'                   # 'velocity' or 'wavelength'
        spec_start = -1000*u.km/u.s              # Starting value of spectrum
        spec_step = 10*u.km/u.s                  # Spectral step
        nspec = 201                              # Number of spectral pixels
        sig_inst = 45*u.km/u.s                   # Instrumental spectral resolution

        beam = instrument.GaussianBeam(major=beamsize)
        lsf = instrument.LSF(sig_inst)

        inst.beam = beam
        inst.lsf = lsf
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
        inst.set_beam_kernel()
        inst.set_lsf_kernel()

        return inst


class TestModels:
    helper = HelperSetups()


    def test_diskbulge(self):
        bary = self.helper.setup_diskbulge()

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.])   # kpc
        vcirc = np.array([0., 234.5537625461696, 232.5422746300449,
                          222.81686610765897, 207.87067673318512]) #km/s
        menc = np.array([0.0, 33608286555.18022, 59533241580.68471, 
                         76951223276.7583, 87374973250.0867]) # Msun

        dlnrho_dlnr = np.array([-0.748634562849152, -1.4711285560530236, -2.179833936965688,
                                -3.000041227680322, -3.8337218490100025])
        rho = np.array([13508748522957.645, 400003289.8613462, 117861773.23589979,
                        42118714.10378093, 15952739.398213904]) # Msun/kpc^3

        for i, r in enumerate(rarr):
            # Assert vcirc, menc, density, dlnrho_dlnr values are the same
            assert math.isclose(bary.circular_velocity(r), vcirc[i], rel_tol=ftol)
            assert math.isclose(bary.enclosed_mass(r), menc[i], rel_tol=ftol)
            assert math.isclose(bary.rhogas(r), rho[i], rel_tol=ftol)
            assert math.isclose(bary.dlnrhogas_dlnr(r), dlnrho_dlnr[i], rel_tol=ftol)


    def test_sersic(self):
        sersic = self.helper.setup_sersic(noord_flat=False)

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.])   # kpc
        vcirc = np.array([0.0, 187.95808079510437, 207.38652969925448,
                            202.6707348023267, 190.9947720259013]) #km/s
        menc = np.array([0.0, 20535293937.195515, 50000000000.0,
                            71627906617.42969, 84816797558.70425]) # Msun

        for i, r in enumerate(rarr):
            # Assert vcirc, menc values are the same
            assert math.isclose(sersic.circular_velocity(r), vcirc[i], rel_tol=ftol)
            assert math.isclose(sersic.enclosed_mass(r), menc[i], rel_tol=ftol)

        dlnrho_dlnr = np.array([0.0, -1.6783469900166612, -3.3566939800333224,
                                -5.035040970049984, -6.713387960066645])
        rho = np.array([1793261526.5567722, 774809992.0335385, 334770202.1509947,
                        144643318.23351952, 62495674.272009104]) # Msun/kpc^3
        for i, r in enumerate(rarr):
            # Assert density, dlnrho_dlnr values are the same
            assert math.isclose(sersic.rhogas(r), rho[i], rel_tol=ftol)
            assert math.isclose(sersic.dlnrhogas_dlnr(r), dlnrho_dlnr[i], rel_tol=ftol)


    def test_sersic_noord_flat(self):
        sersic = self.helper.setup_sersic(noord_flat=True)

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.])   # kpc
        vcirc = np.array([0.0, 169.35690975584475, 214.89133818431344,
                          220.37288364585189, 210.5840270430537]) #km/s
        menc = np.array([0.0, 19005659064.41927, 48971130611.22708, 
                         70961906665.46979, 84403844577.93872]) # Msun

        dlnrho_dlnr = np.array([0., -1.2608181791170565, -2.1284950291483833,
                                -2.9805332449789246, -3.827214472961034])
        rho = np.array([35133994466.11217, 511500615.0259837, 162957719.186185,
                        59053333.61277823, 22454637.866798732])  # Msun/kpc^3 

        for i, r in enumerate(rarr):
            # Assert vcirc, menc, density, dlnrho_dlnr values are the same
            assert math.isclose(sersic.circular_velocity(r), vcirc[i], rel_tol=ftol)
            assert math.isclose(sersic.enclosed_mass(r), menc[i], rel_tol=ftol)
            assert math.isclose(sersic.rhogas(r), rho[i], rel_tol=ftol)
            assert math.isclose(sersic.dlnrhogas_dlnr(r), dlnrho_dlnr[i], rel_tol=ftol)


    def test_NFW(self):
        halo = self.helper.setup_NFW()

        ftol = 1.e-9
        # Assert Rvir is the same
        assert math.isclose(halo.calc_rvir(), 113.19184480200144, rel_tol=ftol)

        rarr = np.array([0.,2.5,5.,7.5,10.,50.])   # kpc
        vcirc = np.array([0.0, 97.53274745638375, 129.37952931721014,
                        149.39249515561673, 163.34037609257453, 207.0167394246318]) #km/s
        menc = np.array([0.0, 5529423277.0931425, 19459875132.71848,
                        38918647245.552315, 62033461205.42702, 498218492834.53705]) # Msun

        for i, r in enumerate(rarr):
            # Assert vcirc, menc values are the same
            assert math.isclose(halo.circular_velocity(r), vcirc[i], rel_tol=ftol)
            assert math.isclose(halo.enclosed_mass(r), menc[i], rel_tol=ftol)


    def test_TPH(self):
        halo = self.helper.setup_TPH()

        ftol = 1.e-9
        # Assert Rvir is the same
        assert math.isclose(halo.calc_rvir(), 113.19184480200144, rel_tol=ftol)

        rarr = np.array([0.,2.5,5.,7.5,10.,50.])   # kpc
        vcirc = np.array([0.0, 31.585366510730005, 56.73315065922273,
                            77.10747504831834, 93.85345758098778, 184.0132403616831]) #km/s
        menc = np.array([0.0, 579896865.582416, 3741818525.6905646,
                        10367955836.483051, 20480448580.59536, 393647104816.9644]) # Msun

        for i, r in enumerate(rarr):
            # Assert vcirc, menc values are the same
            assert math.isclose(halo.circular_velocity(r), vcirc[i], rel_tol=ftol)
            assert math.isclose(halo.enclosed_mass(r), menc[i], rel_tol=ftol)

    def test_massive_gaussian_ring(self):
        if _dir_gaussian_ring_tables is not None:
            gausring = self.helper.setup_massive_gaussian_ring()

            ftol = 1.e-9
            rarr = np.array([0.,2.5,4., 5., 6., 7., 8., 9.,10.])   # kpc
            vcirc = np.array([0., np.NaN, np.NaN, 55.28490315178013,
                              115.61430265260948, 116.36168966861332,
                              97.409470113934, 83.55004344269108, 75.312359412538]) #km/s
            menc = np.array([0.0, 39716658.10417131, 1187568328.9910522,
                             4152924363.1541986, 7725273046.944063, 9558434405.20822,
                             9960786376.52532, 9998475800.392485, 9999974666.918144]) # Msun

            dlnrho_dlnr = np.array([0.0, 5.545177444479562, 3.5489135644669196,
                                    -0.0, -5.323370346700379, -12.42119747563422,
                                    -21.293481386801517, -31.940222080202275, -44.3614195558365])
            rho = np.array([1825.1474473944347, 7475803.944527601, 76757123.1000771,
                            119612863.11244161, 76757123.1000771, 20283415.964593038,
                            2207217.3991932594, 98907.84290890688, 1825.1474473944347])  # Msun/kpc^3

            for i, r in enumerate(rarr):
                # Assert vcirc, menc, density, dlnrho_dlnr values are the same
                if np.isfinite(vcirc[i]):
                    assert math.isclose(gausring.circular_velocity(r), vcirc[i], rel_tol=ftol)
                else:
                    assert ~np.isfinite(gausring.circular_velocity(r))
                assert math.isclose(gausring.enclosed_mass(r), menc[i], rel_tol=ftol)
                assert math.isclose(gausring.rhogas(r), rho[i], rel_tol=ftol)
                assert math.isclose(gausring.dlnrhogas_dlnr(r), dlnrho_dlnr[i], rel_tol=ftol)

    def test_asymm_drift_pressuregradient(self):
        gal = self.helper.setup_fullmodel(pressure_support_type=3)

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.])   # kpc
        vrot = np.array([0.0, 249.58068398249594, 259.8065524159829,
                         259.6197425533726, 253.09920145099804]) # km/s

        for i, r in enumerate(rarr):
            # Assert vrot values are the same
            assert math.isclose(gal.model.velocity_profile(r, tracer='halpha'), vrot[i], rel_tol=ftol)


    def test_asymm_drift_exactsersic(self):
        gal = self.helper.setup_fullmodel(pressure_support_type=2, instrument=False)
        gal.model.set_parameter_value('disk+bulge', 'n_disk', 0.5)

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.])   # kpc
        vrot = np.array([0.0, 232.88505971810986, 254.77986217020114,
                         262.0585360537673, 243.36586952376553]) #km/s

        for i, r in enumerate(rarr):
            # Assert vrot values are the same
            assert math.isclose(gal.model.velocity_profile(r, tracer='halpha'), vrot[i], rel_tol=ftol)


    def test_asymm_drift_selfgrav(self):
        gal = self.helper.setup_fullmodel(pressure_support_type=1, instrument=False)

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.])   # kpc
        vrot = np.array([0.0, 248.94341596219644, 256.3287188287798,
                         253.57372385714814, 244.27275064459627]) #km/s

        for i, r in enumerate(rarr):
            # Assert vrot values are the same
            assert math.isclose(gal.model.velocity_profile(r, tracer='halpha'), vrot[i], rel_tol=ftol)


    def test_composite_model(self):
        gal_noAC = self.helper.setup_fullmodel(adiabatic_contract=False, instrument=False)

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.,50.])   # kpc
        vcirc_noAC = np.array([0.0, 254.02382634494577, 266.1108267177487,
                               268.26381312204825, 264.3677300796697, 227.0647554258678]) #km/s
        menc_noAC = np.array([0.0, 39137709832.30013, 78993116713.49243, 115869870522.4792, 
                              149408434455.76968, 598184965008.3972]) # Msun

        for i, r in enumerate(rarr):
            # Assert vcirc, menc values are the same
            assert math.isclose(gal_noAC.model.circular_velocity(r), vcirc_noAC[i], rel_tol=ftol)
            assert math.isclose(gal_noAC.model.enclosed_mass(r), menc_noAC[i], rel_tol=ftol)


    def test_adiabatic_contraction(self):
        gal_AC = self.helper.setup_fullmodel(adiabatic_contract=True, instrument=False)

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.,50.])   # kpc
        vcirc_AC = np.array([45.59127186551518, 271.17987202928936, 284.2764274637112,
                             285.19266202687686, 279.3433093489505, 226.9501065655811]) #km/s
        menc_AC = np.array([0.0, 44375208066.49577, 90616352511.10013, 132208328905.18863, 
                            168340181474.65088, 597579834948.7798]) # Msun

        for i, r in enumerate(rarr):
            # Assert vcirc, menc values are the same
            assert math.isclose(gal_AC.model.circular_velocity(r), vcirc_AC[i], rel_tol=ftol)
            assert math.isclose(gal_AC.model.enclosed_mass(r), menc_AC[i], rel_tol=ftol)


    def test_simulate_cube(self):
        gal = self.helper.setup_fullmodel(instrument=True)

        ##################
        # Create cube:

        # Make model
        gal.create_model_data()

        # Get cube:
        cube = gal.get_observation('halpha_1D').model_cube.data.unmasked_data[:].value

        ##################
        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.025385795785447536],
                            [0,0,0, -4.291633599421788e-21],
                            [100,18,0, 2.1864570449974476e-06],
                            [50,18,18, 1.8379212571075407e-08],
                            [95,10,10, 0.0017331850380921273],
                            [100,5,5, 2.59806284303019e-05],
                            [150,18,18, 4.196387384525213e-07],
                            [100,15,15, 0.048575246695046544],
                            [100,15,21, 0.01571651134386143],
                            [90,15,15, 0.07109102928581162],
                            [90,15,21, 0.0038996448540969905]]


        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)


    def test_uniform_inflow(self):
        gal_inflow = self.helper.setup_fullmodel(instrument=True)
        inflow = self.helper.setup_uniform_inflow()
        gal_inflow.model.add_component(inflow)

        # Make model
        gal_inflow.create_model_data()

        # Get cube:
        cube = gal_inflow.get_observation('halpha_1D').model_cube.data.unmasked_data[:].value

        ##################
        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value

        arr_pix_values =   [[100,18,18, 0.024107090069722958],
                            [0,0,0, 2.2587545260114675e-22],
                            [100,18,0, 2.836532901062861e-07],
                            [50,18,18, 1.296264105031993e-07],
                            [95,10,10, 0.0014847101563940928],
                            [100,5,5, 1.3391806466223583e-05],
                            [150,18,18, 1.7878573518278115e-06],
                            [100,15,15, 0.04249165716318261],
                            [100,15,21, 0.00608171807364393],
                            [90,15,15, 0.062040129665900806],
                            [90,15,21, 0.0013105590579818582]]


        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)

    def test_bar_inflow(self):
        gal_bar = self.helper.setup_fullmodel(instrument=True)
        bar = self.helper.setup_bar_inflow()
        gal_bar.model.add_component(bar)

        # Make model
        gal_bar.create_model_data()

        # Get cube:
        cube = gal_bar.get_observation('halpha_1D').model_cube.data.unmasked_data[:].value

        ##################
        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value

        arr_pix_values =   [[100,18,18, 0.020557222934201248],
                            [0,0,0, 1.807003620809174e-21],
                            [100,18,0, 2.1861796543816574e-06],
                            [50,18,18, 1.460015377540332e-06],
                            [95,10,10, 0.0017317593164739437],
                            [100,5,5, 2.5980628430081473e-05],
                            [150,18,18, 4.3054634022043904e-06],
                            [100,15,15, 0.03792157115110738],
                            [100,15,21, 0.011708927308800302],
                            [90,15,15, 0.07144121102983325],
                            [90,15,21, 0.002261938326454337]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)

    def test_wedge_inflow(self):
        gal_wedge = self.helper.setup_fullmodel(instrument=True)
        wedge = self.helper.setup_wedge_inflow()
        gal_wedge.model.add_component(wedge)

        # Make model
        gal_wedge.create_model_data()

        # Get cube:
        cube = gal_wedge.get_observation('halpha_1D').model_cube.data.unmasked_data[:].value

        ##################
        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value

        arr_pix_values =   [[100,18,18, 0.01952842802384748],
                            [0,0,0, -1.5811281682080273e-21],
                            [100,18,0, 4.475289469761097e-07],
                            [50,18,18, 1.846058642502683e-08],
                            [95,10,10, 0.0016539411986031265],
                            [100,5,5, 2.5979264215177718e-05],
                            [150,18,18, 4.2038218150593684e-07],
                            [100,15,15, 0.03638019111119462],
                            [100,15,21, 0.006001767379924278],
                            [90,15,15, 0.06596075151464298],
                            [90,15,21, 0.0011865074902281455]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)


    def test_spiral_density_waves_flatVrot(self):
        gal_spiral = self.helper.setup_fullmodel(instrument=True)
        spiral = self.helper.setup_spiral_density_waves_flatVrot()
        gal_spiral.model.add_component(spiral)

        # Make model
        gal_spiral.create_model_data()

        # Get cube:
        cube = gal_spiral.get_observation('halpha_1D').model_cube.data.unmasked_data[:].value

        ##################
        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value

        arr_pix_values =   [[100,18,18, 0.023993424077871684],
                            [0,0,0, 6.324512672832109e-21],
                            [100,18,0, 2.2714689873138346e-06],
                            [50,18,18, 7.315591860369378e-08],
                            [95,10,10, 0.0023635192298107136],
                            [100,5,5, 0.00010984016795835791],
                            [150,18,18, 2.5374320642886547e-06],
                            [100,15,15, 0.04640212010747527],
                            [100,15,21, 0.015740611383889422],
                            [90,15,15, 0.0690862364702349],
                            [90,15,21, 0.006386863833688872]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)



    def test_biconical_outflow(self):
        gal_bicone = self.helper.setup_fullmodel(instrument=True)
        bicone, bicone_geom, bicone_disp = self.helper.setup_biconical_outflow()
        gal_bicone.model.add_component(bicone)
        gal_bicone.model.add_component(bicone_geom, geom_type=bicone.name)
        gal_bicone.model.add_component(bicone_disp, disp_type=bicone.name)

        # Make model
        gal_bicone.create_model_data()

        # Get cube:
        cube = gal_bicone.get_observation('halpha_1D').model_cube.data.unmasked_data[:].value

        ##################
        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.027077067596332466],
                            [0,0,0, -3.3881317890172014e-21],
                            [100,18,0, 2.1864570449973455e-06],
                            [50,18,18, 0.0005320338878855436],
                            [95,10,10, 0.001733374835297935],
                            [100,5,5, 2.5980628430308753e-05],
                            [150,18,18, 0.000531098618269112],
                            [100,15,15, 0.05254364932460036],
                            [100,15,21, 0.015847892702694284],
                            [90,15,15, 0.07490801854057066],
                            [90,15,21, 0.004025998769499641]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)


class TestModelsFittingWrappers:
    def test_fitting_wrapper_model(self):
        param_filename = 'make_model_3Dcube.params'
        param_filename_full=_dir_tests_data+param_filename

        # Delete existing folder:
        params = fw_utils_io.read_fitting_params(fname=param_filename_full)
        outdir = _dir_tests_data+params['outdir']
        if os.path.isdir(outdir):
            shutil.rmtree(outdir, ignore_errors=True)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.062096325861100865],
                            [0,0,0, 1.3385212005993883e-21],
                            [100,18,0, 2.5253432241441e-06],
                            [50,18,18, 5.698384148161866e-07],
                            [95,10,10, 0.0016377058627328684],
                            [100,5,5, 0.00010039456527216781],
                            [150,18,18, 5.698384148120747e-07],
                            [100,15,15, 0.010942160099466264],
                            [100,15,21, 0.02704783322800287],
                            [90,15,15, 0.04528310698904516],
                            [90,15,21, 0.020552348141567225]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)


    def test_fitting_wrapper_model_radial_flow(self):
        param_filename = 'make_model_3Dcube_radial_flow.params'
        param_filename_full=_dir_tests_data+param_filename

        # Delete existing folder:
        params = fw_utils_io.read_fitting_params(fname=param_filename_full)
        outdir = _dir_tests_data+params['outdir']
        if os.path.isdir(outdir):
            shutil.rmtree(outdir, ignore_errors=True)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.05274746107036724],
                            [0,0,0, 2.6770424011987766e-21],
                            [100,18,0, 6.149085755387803e-07],
                            [50,18,18, 2.6768273685509387e-06],
                            [95,10,10, 0.0011851378117069761],
                            [100,5,5, 6.198276696294433e-05],
                            [150,18,18, 2.676827368554137e-06],
                            [100,15,15, 0.011320861657201713],
                            [100,15,21, 0.022009103331619515],
                            [90,15,15, 0.03757150698720647],
                            [90,15,21, 0.010338830606177003]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)


    def test_fitting_wrapper_model_planar_radial_flow(self):
        param_filename = 'make_model_3Dcube_planar_radial_flow.params'
        param_filename_full=_dir_tests_data+param_filename

        # Delete existing folder:
        params = fw_utils_io.read_fitting_params(fname=param_filename_full)
        outdir = _dir_tests_data+params['outdir']
        if os.path.isdir(outdir):
            shutil.rmtree(outdir, ignore_errors=True)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.052652695795088286],
                            [0,0,0, 3.5693898682650355e-21],
                            [100,18,0, 5.896054073241489e-07],
                            [50,18,18, 1.6954768341167763e-06],
                            [95,10,10, 0.0011658817894668796],
                            [100,5,5, 6.13056469185842e-05],
                            [150,18,18, 1.6954768341291121e-06],
                            [100,15,15, 0.01112988318401541],
                            [100,15,21, 0.02176983522399486],
                            [90,15,15, 0.03726923786824762],
                            [90,15,21, 0.010078226699562426]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)


    def test_fitting_wrapper_model_uniform_bar_flow(self):
        param_filename = 'make_model_3Dcube_uniform_bar_flow.params'
        param_filename_full=_dir_tests_data+param_filename

        # Delete existing folder:
        params = fw_utils_io.read_fitting_params(fname=param_filename_full)
        outdir = _dir_tests_data+params['outdir']
        if os.path.isdir(outdir):
            shutil.rmtree(outdir, ignore_errors=True)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.04927368010973892],
                            [0,0,0, 1.3385212005993883e-21],
                            [100,18,0, 2.5253430077721125e-06],
                            [50,18,18, 1.4329957667203139e-05],
                            [95,10,10, 0.0016377049977153518],
                            [100,5,5, 0.00010039456527216798],
                            [150,18,18, 1.4329957667196742e-05],
                            [100,15,15, 0.009732331095492985],
                            [100,15,21, 0.020623913146535198],
                            [90,15,15, 0.04503083931308861],
                            [90,15,21, 0.015578674251922076]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)



    def test_fitting_wrapper_model_uniform_wedge_flow(self):
        param_filename = 'make_model_3Dcube_uniform_wedge_flow.params'
        param_filename_full=_dir_tests_data+param_filename

        # Delete existing folder:
        params = fw_utils_io.read_fitting_params(fname=param_filename_full)
        outdir = _dir_tests_data+params['outdir']
        if os.path.isdir(outdir):
            shutil.rmtree(outdir, ignore_errors=True)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.04625110437599818],
                            [0,0,0, 2.2308686676656474e-21],
                            [100,18,0, 1.3070541911134733e-06],
                            [50,18,18, 5.733893197456777e-07],
                            [95,10,10, 0.001635687883516719],
                            [100,5,5, 0.00010039455380341816],
                            [150,18,18, 5.733893197461346e-07],
                            [100,15,15, 0.008701545542027213],
                            [100,15,21, 0.02001730309988766],
                            [90,15,15, 0.04273073210594165],
                            [90,15,21, 0.009754801538534088]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)



    def test_fitting_wrapper_model_unresolved_outflow(self):
        param_filename = 'make_model_3Dcube_unresolved_outflow.params'
        param_filename_full=_dir_tests_data+param_filename

        # Delete existing folder:
        params = fw_utils_io.read_fitting_params(fname=param_filename_full)
        outdir = _dir_tests_data+params['outdir']
        if os.path.isdir(outdir):
            shutil.rmtree(outdir, ignore_errors=True)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.14971321657588857],
                            [0,0,0, 7.852657710183078e-20],
                            [100,18,0, 2.525343224145588e-06],
                            [50,18,18, 0.04424249966019471],
                            [95,10,10, 0.0016377068144249323],
                            [100,5,5, 0.00010039456527216733],
                            [150,18,18, 0.044242499660194716],
                            [100,15,15, 0.01759573109762244],
                            [100,15,21, 0.033701404226159054],
                            [90,15,15, 0.05175728516818657],
                            [90,15,21, 0.027026526320708646]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)


    def test_fitting_wrapper_model_biconical_outflow(self):
        param_filename = 'make_model_3Dcube_biconical_outflow.params'
        param_filename_full=_dir_tests_data+param_filename

        # Delete existing folder:
        params = fw_utils_io.read_fitting_params(fname=param_filename_full)
        outdir = _dir_tests_data+params['outdir']
        if os.path.isdir(outdir):
            shutil.rmtree(outdir, ignore_errors=True)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.06623011996908379],
                            [0,0,0, 2.2308686676656474e-21],
                            [100,18,0, 2.544396329727811e-06],
                            [50,18,18, 0.0025079171449907366],
                            [95,10,10, 0.0016486289015841358],
                            [100,5,5, 0.0001003945662387093],
                            [150,18,18, 0.0025079171449907374],
                            [100,15,15, 0.012753172002366388],
                            [100,15,21, 0.03600768607557391],
                            [90,15,15, 0.0479565993649698],
                            [90,15,21, 0.028101759132787885]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)


    def test_fitting_wrapper_model_variable_bar_flow(self):
        param_filename = 'make_model_3Dcube_variable_bar_flow.params'
        param_filename_full=_dir_tests_data+param_filename

        # Delete existing folder:
        params = fw_utils_io.read_fitting_params(fname=param_filename_full)
        outdir = _dir_tests_data+params['outdir']
        if os.path.isdir(outdir):
            shutil.rmtree(outdir, ignore_errors=True)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.058823972468395366],
                            [0,0,0, 8.923474670662589e-22],
                            [100,18,0, 2.525343223136356e-06],
                            [50,18,18, 1.2647106027087264e-05],
                            [95,10,10, 0.001637705637702002],
                            [100,5,5, 0.0001003945652721678],
                            [150,18,18, 1.2647106027088177e-05],
                            [100,15,15, 0.010633342579011853],
                            [100,15,21, 0.0261541595394654],
                            [90,15,15, 0.04514075021269858],
                            [90,15,21, 0.01786060517580677]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)


    def test_fitting_wrapper_model_azimuthal_planar_radial_flow(self):
        param_filename = 'make_model_3Dcube_azimuthal_planar_radial_flow.params'
        param_filename_full=_dir_tests_data+param_filename

        # Delete existing folder:
        params = fw_utils_io.read_fitting_params(fname=param_filename_full)
        outdir = _dir_tests_data+params['outdir']
        if os.path.isdir(outdir):
            shutil.rmtree(outdir, ignore_errors=True)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.0565965229237657],
                            [0,0,0, 4.461737335331294e-22],
                            [100,18,0, 2.537954068057636e-06],
                            [50,18,18, 7.419933553374344e-07],
                            [95,10,10, 0.0016088622346838795],
                            [100,5,5, 9.963441075490312e-05],
                            [150,18,18, 7.419933553296674e-07],
                            [100,15,15, 0.010995456056668813],
                            [100,15,21, 0.025039792811775576],
                            [90,15,15, 0.04454043313102052],
                            [90,15,21, 0.02465183936170711]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)

    def test_fitting_wrapper_model_spiral_flow(self):
        param_filename = 'make_model_3Dcube_spiral_flow.params'
        param_filename_full=_dir_tests_data+param_filename

        # Delete existing folder:
        params = fw_utils_io.read_fitting_params(fname=param_filename_full)
        outdir = _dir_tests_data+params['outdir']
        if os.path.isdir(outdir):
            shutil.rmtree(outdir, ignore_errors=True)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.061932705904893304],
                            [0,0,0, 1.7846949341325177e-21],
                            [100,18,0, 2.4244854910987994e-06],
                            [50,18,18, 6.021356235187983e-07],
                            [95,10,10, 0.0016470620217776321],
                            [100,5,5, 0.00010209511831378836],
                            [150,18,18, 6.021356235128589e-07],
                            [100,15,15, 0.010992460769702379],
                            [100,15,21, 0.026926089713230416],
                            [90,15,15, 0.04521738711777384],
                            [90,15,21, 0.0205481672058022]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)



# # GETTING TEST CUBE NUMBERS:
# for arr in arr_pix_values:
#     print("[{},{},{}, {}],".format(arr[0],arr[1],arr[2],cube[arr[0],arr[1],arr[2]]))
