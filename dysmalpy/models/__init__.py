# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Submodule for handling of DysmalPy ModelSets (and Models) to use build the galaxy model

from dysmalpy.models.base import MassModel, LightModel, \
                                 HigherOrderKinematicsSeparate, HigherOrderKinematicsPerturbation, \
                                 v_circular, sersic_mr, truncate_sersic_mr, \
                                menc_from_vcirc
from dysmalpy.models.baryons import Sersic, DiskBulge, LinearDiskBulge, \
                                    ExpDisk, BlackHole, GaussianRing, \
                                    surf_dens_exp_disk, menc_exp_disk, vcirc_exp_disk, \
                                    sersic_menc_2D_proj, \
                                    mass_comp_conditional_ring, \
                                    NoordFlat, InfThinMassiveGaussianRing
from dysmalpy.models.halos import NFW, TwoPowerHalo, Burkert, \
                                  Einasto, DekelZhao, LinearNFW
from dysmalpy.models.higher_order_kinematics import BiconicalOutflow, UnresolvedOutflow, \
                                                    UniformRadialFlow, PlanarUniformRadialFlow, \
                                                    AzimuthalPlanarRadialFlow, \
                                                    UniformBarFlow, VariableXBarFlow, \
                                                    UniformWedgeFlow, \
                                                    SpiralDensityWave
from dysmalpy.models.zheight import ZHeightGauss, ZHeightExp
from dysmalpy.models.dispersion_profiles import DispersionConst
from dysmalpy.models.light_distributions import LightTruncateSersic, LightGaussianRing, \
                                                LightClump, LightGaussianRingAzimuthal
from dysmalpy.models.extinction import ThinCentralPlaneDustExtinction, \
                                       ForegroundConstantExtinction, \
                                       ForegroundExponentialExtinction
from dysmalpy.models.dimming import ConstantDimming, CosmologicalDimming
from dysmalpy.models.kinematic_options import KinematicOptions
from dysmalpy.models.geometry import Geometry
from dysmalpy.models.model_set import ModelSet


__all__ = ['ModelSet',
           # Baryons
           'Sersic', 'DiskBulge', 'LinearDiskBulge', 'ExpDisk', 'BlackHole',
           'GaussianRing',
           # Halos
           'NFW', 'LinearNFW', 'TwoPowerHalo', 'Burkert', 'Einasto', 'DekelZhao',
           # Higher-order components
           'BiconicalOutflow', 'UnresolvedOutflow',
           'UniformRadialFlow', 'PlanarUniformRadialFlow',
           'AzimuthalPlanarRadialFlow',
           'UniformBarFlow', 'VariableXBarFlow',
           'UniformWedgeFlow',
           'SpiralDensityWave',
           # Light profiles
           'LightTruncateSersic', 'LightGaussianRing',
           'LightClump', 'LightGaussianRingAzimuthal',
           # Dispersion
           'DispersionConst',
           # Extinction
           'ThinCentralPlaneDustExtinction', 'ForegroundConstantExtinction',
           'ForegroundExponentialExtinction',
           # Dimming
           'ConstantDimming', 'CosmologicalDimming',
           # Geometry and optoins
           'Geometry', 'KinematicOptions',
           # Zheight profiles
           'ZHeightGauss', 'ZHeightExp',
           # Functions
           'surf_dens_exp_disk', 'menc_exp_disk', 'vcirc_exp_disk',
           'sersic_mr', 'sersic_menc_2D_proj', 'v_circular', 'menc_from_vcirc',
           'NoordFlat', 'InfThinMassiveGaussianRing']
