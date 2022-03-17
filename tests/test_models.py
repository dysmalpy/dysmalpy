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

from dysmalpy import galaxy, models, parameters, instrument, config


# TESTING DIRECTORY
path = os.path.abspath(__file__)
_dir_tests = os.path.dirname(path) + os.sep
_dir_tests_data = _dir_tests+'test_data' + os.sep



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
                                                  bounds=disp_bounds, name='dispprof')

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
                               fixed=geom_fixed, bounds=geom_bounds, name='geom')

        return geom

    def setup_biconical_outflow(self):
        bicone = models.BiconicalOutflow(n=0.5, vmax=300., rturn=0.5, thetain=30, dtheta=20.,
                                         rend=1., norm_flux=11., tau_flux=5., name='bicone')
        # To fully match old tests, set norm_flux = 0.
        bicone_geom = models.Geometry(inc=10., pa=30., xshift=0., yshift=0., name='outflow_geom')
        bicone_disp = models.DispersionConst(sigma0=250., name='outflow_dispprof')
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
        # Initialize the Galaxy, Instrument, and Model Set
        gal = galaxy.Galaxy(z=self.z, name=self.name)
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
            gal.instrument = inst

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

        # Set the beam kernel so it doesn't have to be calculated every step
        inst.set_beam_kernel()
        inst.set_lsf_kernel()

        return inst

    def setup_3Dcube_kwargs(self):
        param_filename = 'make_model_3Dcube.params'
        param_filename_full=_dir_tests_data+param_filename
        params = fw_utils_io.read_fitting_params(fname=param_filename_full)

        config_c_m_data = config.Config_create_model_data(**params)
        config_sim_cube = config.Config_simulate_cube(**params)
        kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

        # Additional settings:
        kwargs_galmodel['from_data'] = False
        kwargs_galmodel['ndim_final'] = 3

        return kwargs_galmodel


class TestModels:
    helper = HelperSetups()


    def test_diskbulge(self):
        bary = self.helper.setup_diskbulge()

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.])   # kpc
        vcirc = np.array([0., 233.84762112, 231.63051349, 222.14143224, 207.24934609]) #km/s
        menc = np.array([0., 3.17866553e+10, 6.23735490e+10, 8.60516713e+10, 9.98677462e+10]) # Msun

        dlnrho_dlnr = np.array([-0.748634562849152, -1.4711285560530236, -2.179833936965688,
                                -3.000041227680322, -3.8337218490100025])
        rho = np.array([13508748522957.645, 400003289.8613462, 117861773.23589979,
                        42118714.10378093, 15952739.398213904]) # msun/kpc^3 ??

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
                        144643318.23351952, 62495674.272009104]) # msun/kpc^3 ??
        for i, r in enumerate(rarr):
            # Assert density, dlnrho_dlnr values are the same
            assert math.isclose(sersic.rhogas(r), rho[i], rel_tol=ftol)
            assert math.isclose(sersic.dlnrhogas_dlnr(r), dlnrho_dlnr[i], rel_tol=ftol)


    def test_sersic_noord_flat(self):
        sersic = self.helper.setup_sersic(noord_flat=True)

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.])   # kpc
        vcirc = np.array([0.0, 168.75231977473914, 213.96601851622563,
                            219.70437621918154, 209.92527768889798]) #km/s
        menc = np.array([0.0, 16553065102.83822, 53222898982.82668,
                            84173927151.13939, 102463310604.53976]) # Msun

        dlnrho_dlnr = np.array([0.0, -1.2608181791170565, -2.1284950291483833,
                                -2.9805332449789246, -3.827214472961034])
        rho = np.array([35133994466.11217, 511500615.0259837, 162957719.186185,
                        59053333.61277823, 22454637.866798732])  # msun/kpc^3 ??

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
                        2207217.3991932594, 98907.84290890688, 1825.1474473944347])  # msun/kpc^3 ??

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
        vrot = np.array([0.0, 248.91717537419922, 258.9907912770804,
                         259.0402880219702, 252.58915056627905]) # km/s

        for i, r in enumerate(rarr):
            # Assert vrot values are the same
            assert math.isclose(gal.model.velocity_profile(r), vrot[i], rel_tol=ftol)


    def test_asymm_drift_exactsersic(self):
        gal = self.helper.setup_fullmodel(pressure_support_type=2, instrument=False)
        gal.model.set_parameter_value('disk+bulge', 'n_disk', 0.5)

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.])   # kpc
        vrot = np.array([0.0, 232.03861252444398, 253.47823945210072,
                        261.186198203435, 242.75798891697548]) #km/s

        for i, r in enumerate(rarr):
            # Assert vrot values are the same
            assert math.isclose(gal.model.velocity_profile(r), vrot[i], rel_tol=ftol)


    def test_asymm_drift_selfgrav(self):
        gal = self.helper.setup_fullmodel(pressure_support_type=1, instrument=False)

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.])   # kpc
        vrot = np.array([0.0, 248.27820429923966, 255.50185397469704,
                252.9804212303498, 243.74423052912974]) #km/s

        for i, r in enumerate(rarr):
            # Assert vrot values are the same
            assert math.isclose(gal.model.velocity_profile(r), vrot[i], rel_tol=ftol)


    def test_composite_model(self):
        gal_noAC = self.helper.setup_fullmodel(adiabatic_contract=False, instrument=False)

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.,50.])   # kpc
        vcirc_noAC = np.array([0.0, 253.37195332170248, 265.3144500107512,
                        267.70306969828573, 263.8794609594266, 226.93164583634422]) #km/s
        menc_noAC = np.array([0.0, 37316078582.12215, 81833424086.48811,
                        124970318584.0155, 161901207448.951, 598685915680.8903]) # Msun

        for i, r in enumerate(rarr):
            # Assert vcirc, menc values are the same
            assert math.isclose(gal_noAC.model.circular_velocity(r), vcirc_noAC[i], rel_tol=ftol)
            assert math.isclose(gal_noAC.model.enclosed_mass(r), menc_noAC[i], rel_tol=ftol)


    def test_adiabatic_contraction(self):
        gal_AC = self.helper.setup_fullmodel(adiabatic_contract=True, instrument=False)

        ftol = 1.e-9
        rarr = np.array([0.,2.5,5.,7.5,10.,50.])   # kpc
        vcirc_AC = np.array([45.52585221837478, 270.5127652416235, 283.47161746884086,
                        284.6186917220378, 278.84200286773034, 226.8176250265685]) #km/s
        menc_AC = np.array([0.0, 42535784556.8831, 93417465234.21672,
                        141262539926.01163, 180782046435.46255, 598084452596.9691]) # Msun

        for i, r in enumerate(rarr):
            # Assert vcirc, menc values are the same
            assert math.isclose(gal_AC.model.circular_velocity(r), vcirc_AC[i], rel_tol=ftol)
            assert math.isclose(gal_AC.model.enclosed_mass(r), menc_AC[i], rel_tol=ftol)


    def test_biconical_outflow(self):
        gal_bicone = self.helper.setup_fullmodel(instrument=True)
        bicone, bicone_geom, bicone_disp = self.helper.setup_biconical_outflow()
        gal_bicone.model.add_component(bicone)
        gal_bicone.model.add_component(bicone_geom, geom_type=bicone.name)
        gal_bicone.model.add_component(bicone_disp, disp_type=bicone.name)

        ##################
        # Create cube:
        kwargs_galmodel = self.helper.setup_3Dcube_kwargs()

        # Make model
        gal_bicone.create_model_data(**kwargs_galmodel)

        # Get cube:
        cube = gal_bicone.model_cube.data.unmasked_data[:].value

        ##################
        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.027185579072496628],
                            [0,0,0, 2.7105054312137612e-21],
                            [100,18,0, 2.1887814203259354e-06],
                            [50,18,18, 0.0005320325936246136],
                            [95,10,10, 0.001758091539590106],
                            [100,5,5, 2.6361844098141783e-05],
                            [150,18,18, 0.0005310729496857921],
                            [100,15,15, 0.052710956556674675],
                            [100,15,21, 0.01588655546256837],
                            [90,15,15, 0.07522552630778426],
                            [90,15,21, 0.004030068174217149]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)


    def test_uniform_inflow(self):
        gal_inflow = self.helper.setup_fullmodel(instrument=True)
        inflow = self.helper.setup_uniform_inflow()
        gal_inflow.model.add_component(inflow)

        ##################
        # Create cube:
        kwargs_galmodel = self.helper.setup_3Dcube_kwargs()

        # Make model
        gal_inflow.create_model_data(**kwargs_galmodel)

        # Get cube:
        cube = gal_inflow.model_cube.data.unmasked_data[:].value

        ##################
        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value

        arr_pix_values =   [[100,18,18, 0.024201244000354585],
                            [0,0,0, -2.0328790734103207e-21],
                            [100,18,0, 2.842332467196115e-07],
                            [50,18,18, 1.2200523993773544e-07],
                            [95,10,10, 0.0015049702372115622],
                            [100,5,5, 1.3623785523327364e-05],
                            [150,18,18, 1.6933163357361428e-06],
                            [100,15,15, 0.042615056815275054],
                            [100,15,21, 0.0060938800208993026],
                            [90,15,15, 0.06229601362739992],
                            [90,15,21, 0.0013108842080778876]]


        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)

    def test_bar_inflow(self):
        gal_bar = self.helper.setup_fullmodel(instrument=True)
        bar = self.helper.setup_bar_inflow()
        gal_bar.model.add_component(bar)

        ##################
        # Create cube:
        kwargs_galmodel = self.helper.setup_3Dcube_kwargs()

        # Make model
        gal_bar.create_model_data(**kwargs_galmodel)

        # Get cube:
        cube = gal_bar.model_cube.data.unmasked_data[:].value

        ##################
        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value

        arr_pix_values =   [[100,18,18, 0.02064790492316384],
                            [0,0,0, -6.776263578034403e-22],
                            [100,18,0, 2.1885040552975126e-06],
                            [50,18,18, 1.3853082438536335e-06],
                            [95,10,10, 0.0017564723153457415],
                            [100,5,5, 2.6361844097914003e-05],
                            [150,18,18, 4.082901162771879e-06],
                            [100,15,15, 0.038049267735887],
                            [100,15,21, 0.01174313004385695],
                            [90,15,15, 0.07175454405079977],
                            [90,15,21, 0.0022651705774065253]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)

    def test_wedge_inflow(self):
        gal_wedge = self.helper.setup_fullmodel(instrument=True)
        wedge = self.helper.setup_wedge_inflow()
        gal_wedge.model.add_component(wedge)

        ##################
        # Create cube:
        kwargs_galmodel = self.helper.setup_3Dcube_kwargs()

        # Make model
        gal_wedge.create_model_data(**kwargs_galmodel)

        # Get cube:
        cube = gal_wedge.model_cube.data.unmasked_data[:].value

        ##################
        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value

        arr_pix_values =   [[100,18,18, 0.019586006862063132],
                            [0,0,0, 1.1293772630057337e-21],
                            [100,18,0, 4.4833603035803543e-07],
                            [50,18,18, 1.7163234226902357e-08],
                            [95,10,10, 0.0016784517511998372],
                            [100,5,5, 2.636047616113313e-05],
                            [150,18,18, 3.9468783100682355e-07],
                            [100,15,15, 0.03645396056094084],
                            [100,15,21, 0.006016831537326521],
                            [90,15,15, 0.06634179926804772],
                            [90,15,21, 0.001189154821866105]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)


    def test_spiral_density_waves_flatVrot(self):
        gal_spiral = self.helper.setup_fullmodel(instrument=True)
        spiral = self.helper.setup_spiral_density_waves_flatVrot()
        gal_spiral.model.add_component(spiral)

        ##################
        # Create cube:
        kwargs_galmodel = self.helper.setup_3Dcube_kwargs()

        # Make model
        gal_spiral.create_model_data(**kwargs_galmodel)

        # Get cube:
        cube = gal_spiral.model_cube.data.unmasked_data[:].value

        ##################
        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value

        arr_pix_values =   [[100,18,18, 0.0240910407123745],
                            [0,0,0, -1.807003620809174e-21],
                            [100,18,0, 2.274254720591604e-06],
                            [50,18,18, 6.869583540925853e-08],
                            [95,10,10, 0.002387161656309826],
                            [100,5,5, 0.00011124286936046525],
                            [150,18,18, 2.4073925094865906e-06],
                            [100,15,15, 0.04656194667451479],
                            [100,15,21, 0.015767707488431298],
                            [90,15,15, 0.06939214762985071],
                            [90,15,21, 0.006395139680842508]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)

    def test_simulate_cube(self):
        gal = self.helper.setup_fullmodel(instrument=True)

        ##################
        # Create cube:
        kwargs_galmodel = self.helper.setup_3Dcube_kwargs()

        # Make model
        gal.create_model_data(**kwargs_galmodel)

        # Get cube:
        cube = gal.model_cube.data.unmasked_data[:].value

        ##################
        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.025494307261611702],
                            [0,0,0, 6.776263578034403e-22],
                            [100,18,0, 2.188781420324916e-06],
                            [50,18,18, 1.7084951645559194e-08],
                            [95,10,10, 0.0017579017423842986],
                            [100,5,5, 2.6361844098134272e-05],
                            [150,18,18, 3.939701551269881e-07],
                            [100,15,15, 0.04874255392712087],
                            [100,15,21, 0.015755174103735517],
                            [90,15,15, 0.07140853705302519],
                            [90,15,21, 0.003903714258814497]]

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
        arr_pix_values =   [[100,18,18, 0.062306424425167345],
                            [0,0,0, 1.7846949341325177e-21],
                            [100,18,0, 2.5143654319214826e-06],
                            [50,18,18, 5.345480239025723e-07],
                            [95,10,10, 0.0016590598295565717],
                            [100,5,5, 0.0001009414287366288],
                            [150,18,18, 5.345480239012016e-07],
                            [100,15,15, 0.011018604410484396],
                            [100,15,21, 0.027103610119410277],
                            [90,15,15, 0.045620894103943446],
                            [90,15,21, 0.02057779281245439]]

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
            shutil.rmtree(outdir)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.05289229590921311],
                            [0,0,0, 0.0],
                            [100,18,0, 6.08081831144031e-07],
                            [50,18,18, 2.5394868500877555e-06],
                            [95,10,10, 0.0012018147554400322],
                            [100,5,5, 6.240538925948083e-05],
                            [150,18,18, 2.5394868500927814e-06],
                            [100,15,15, 0.011393115571699731],
                            [100,15,21, 0.02203820364233378],
                            [90,15,15, 0.03784075512013737],
                            [90,15,21, 0.010338744289985281]]

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
            shutil.rmtree(outdir)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.05279108769198229],
                            [0,0,0, -8.923474670662589e-22],
                            [100,18,0, 5.829535971686665e-07],
                            [50,18,18, 1.6023518969472919e-06],
                            [95,10,10, 0.001182408713692744],
                            [100,5,5, 6.172078700674826e-05],
                            [150,18,18, 1.6023518969472919e-06],
                            [100,15,15, 0.011200194989420681],
                            [100,15,21, 0.021796598886909838],
                            [90,15,15, 0.037539158849834955],
                            [90,15,21, 0.010076756380715115]]

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
            shutil.rmtree(outdir)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.049436951836480725],
                            [0,0,0, 1.7846949341325177e-21],
                            [100,18,0, 2.514365215564211e-06],
                            [50,18,18, 1.3672043804203988e-05],
                            [95,10,10, 0.001659058962060482],
                            [100,5,5, 0.00010094142873662862],
                            [150,18,18, 1.3672043804205359e-05],
                            [100,15,15, 0.009804466102837077],
                            [100,15,21, 0.020668291051880523],
                            [90,15,15, 0.045366403861279395],
                            [90,15,21, 0.015600750987785348]]

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
            shutil.rmtree(outdir)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.046355793187690814],
                            [0,0,0, 1.7846949341325177e-21],
                            [100,18,0, 1.2981907321834523e-06],
                            [50,18,18, 5.379775661284546e-07],
                            [95,10,10, 0.0016570420573892048],
                            [100,5,5, 0.00010094141728171345],
                            [150,18,18, 5.379775661279977e-07],
                            [100,15,15, 0.008759122487890474],
                            [100,15,21, 0.020035373406315067],
                            [90,15,15, 0.04308626032574506],
                            [90,15,21, 0.009778660629345865]]

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
            shutil.rmtree(outdir)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.14992331513995502],
                            [0,0,0, 8.120361950302955e-20],
                            [100,18,0, 2.51436543192309e-06],
                            [50,18,18, 0.0442424643698038],
                            [95,10,10, 0.0016590607812486358],
                            [100,5,5, 0.00010094142873662856],
                            [150,18,18, 0.04424246436980381],
                            [100,15,15, 0.017672175408640557],
                            [100,15,21, 0.033757181117566454],
                            [90,15,15, 0.052095072283084846],
                            [90,15,21, 0.027051970991595804]]

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
            shutil.rmtree(outdir)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.06644021853315027],
                            [0,0,0, 8.923474670662589e-22],
                            [100,18,0, 2.533418537505829e-06],
                            [50,18,18, 0.002507881854599829],
                            [95,10,10, 0.0016699828684078382],
                            [100,5,5, 0.00010094142970317003],
                            [150,18,18, 0.002507881854599829],
                            [100,15,15, 0.012829616313384517],
                            [100,15,21, 0.03606346296698132],
                            [90,15,15, 0.048294386479868084],
                            [90,15,21, 0.028127203803675047]]

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
            shutil.rmtree(outdir)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.05901650542176798],
                            [0,0,0, 1.7846949341325177e-21],
                            [100,18,0, 2.5143654309156026e-06],
                            [50,18,18, 1.205654170222069e-05],
                            [95,10,10, 0.0016590596033040968],
                            [100,5,5, 0.00010094142873662871],
                            [150,18,18, 1.2056541702219777e-05],
                            [100,15,15, 0.01070841016380301],
                            [100,15,21, 0.02620604457113305],
                            [90,15,15, 0.04547514330653823],
                            [90,15,21, 0.017880950006750315]]

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
            shutil.rmtree(outdir)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.056778986180900354],
                            [0,0,0, 1.7846949341325177e-21],
                            [100,18,0, 2.5270060135964004e-06],
                            [50,18,18, 6.962378894060093e-07],
                            [95,10,10, 0.001629922739873721],
                            [100,5,5, 0.00010017612936480282],
                            [150,18,18, 6.962378894046386e-07],
                            [100,15,15, 0.011069858082284945],
                            [100,15,21, 0.025089408659842063],
                            [90,15,15, 0.04487368036335876],
                            [90,15,21, 0.02469265352009432]]

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
            shutil.rmtree(outdir)

        # Make model
        dysmalpy_make_model.dysmalpy_make_model(param_filename=param_filename_full)

        # Load cube:
        f_cube = outdir+'{}_model_cube.fits'.format(params['galID'])
        cube = fits.getdata(f_cube)

        # Check some pix points:
        atol = 1.e-9
        # array: ind0,ind1,ind2, value
        arr_pix_values =   [[100,18,18, 0.06214084407576038],
                            [0,0,0, 1.7846949341325177e-21],
                            [100,18,0, 2.413427431561494e-06],
                            [50,18,18, 5.649122109342417e-07],
                            [95,10,10, 0.0016684203374615159],
                            [100,5,5, 0.00010265137792981161],
                            [150,18,18, 5.649122109383537e-07],
                            [100,15,15, 0.011069093580124412],
                            [100,15,21, 0.026981243904103936],
                            [90,15,15, 0.04555353748217607],
                            [90,15,21, 0.02057394395279147]]

        for arr in arr_pix_values:
            # Assert pixel values are the same
            assert math.isclose(cube[arr[0],arr[1],arr[2]], arr[3], abs_tol=atol)



# # GETTING TEST CUBE NUMBERS:
# for arr in arr_pix_values:
#     print("[{},{},{}, {}],".format(arr[0],arr[1],arr[2],cube[arr[0],arr[1],arr[2]]))
