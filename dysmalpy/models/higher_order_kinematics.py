# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Higher-order kinematics models for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np
import scipy.misc as scp_misc
import scipy.integrate as scp_integrate

# Local imports
from .base import _DysmalFittable3DModel, HigherOrderKinematicsSeparate, \
                  HigherOrderKinematicsPerturbation
from dysmalpy.parameters import DysmalParameter

try:
    import utils
except:
    from . import utils

__all__ = ['BiconicalOutflow', 'UnresolvedOutflow', 'UniformRadialFlow',
           'UniformBarFlow', 'SpiralDensityWave']

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

np.warnings.filterwarnings('ignore')


class BiconicalOutflow(HigherOrderKinematicsSeparate, _DysmalFittable3DModel):
    r"""
    Model for a biconical outflow

    Parameters
    ----------
    n : float
        Power law index of the outflow velocity profile

    vmax : float
        Maximum velocity of the outflow in km/s

    rturn : float
        Turn-over radius in kpc of the velocty profile

    thetain : float
        Half inner opening angle in degrees. Measured from the bicone
        axis

    dtheta : float
        Difference between inner and outer opening angle in degrees

    rend : float
        Maximum radius of the outflow in kpc

    norm_flux : float
        Log flux amplitude of the outflow at r = 0

    tau_flux : float
        Exponential decay rate of the flux

    profile_type : {'both', 'increase', 'decrease', 'constant'}
        Type of velocity profile

    Notes
    -----
    This biconical outflow model is based on the model presented in Bae et al. (2016) [1]_.
    It consists of two symmetric cones joined at their apexes. `thetain` and `dtheta` control
    how hollow the cones are. `thetain` = 0 therefore would produce a fully filled cone.

    Within the cone, the velocity radial profile of the gas follows a power law with index `n`.
    Four different profile types can be selected. The simplest is 'constant' in which case
    the velocity of the gas is `vmax` at all radii.

    For a `profile_type` = 'increase':

        .. math::

            v = v_{\rm max}\left(\frac{r}{r_{\rm end}}\right)^n

    For a `profile_type` = 'decrease':

        .. math::

            v = v_{\rm max}\left(1 - \left(\frac{r}{r_{\rm end}}\right)^n\right)

    For a `profile_type` = 'both' the velocity first increases up to the turnover radius, `rturn`,
    then decreases back to 0 at 2 `rturn`.

    For :math:`r < r_{\rm turn}`:

        .. math::

            v =  v_{\rm max}\left(\frac{r}{r_{\rm turn}}\right)^n

    For :math:`r > r_{\rm turn}`:

        .. math::

            v = v_{\rm max}\left(2 - \frac{r}{r_{\rm turn}}\right)^n

    The flux radial profile of the outflow is described by a decreasing exponential:

        .. math::

            F = A\exp\left\{\frac{-\tau r}{r_{\rm end}}\right\}

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2016ApJ...828...97B/abstract

    """

    n = DysmalParameter(default=0.5, fixed=True)
    vmax = DysmalParameter(min=0)
    rturn = DysmalParameter(default=0.5, min=0)
    thetain = DysmalParameter(bounds=(0, 90))
    dtheta = DysmalParameter(default=20.0, bounds=(0, 90))
    rend = DysmalParameter(default=1.0, min=0)
    norm_flux = DysmalParameter(default=0.0, fixed=True)
    tau_flux = DysmalParameter(default=5.0, fixed=True)

    _spatial_type = 'resolved'
    _multicoord_velocity = False
    _native_geometry = 'spherical'
    outputs = ('vout',)

    def __init__(self, profile_type='both', **kwargs):

        valid_profiles = ['increase', 'decrease', 'both', 'constant']

        if profile_type in valid_profiles:
            self.profile_type = profile_type
        else:
            logger.error("Invalid profile type. Must be one of 'increase',"
                         "'decrease', 'constant', or 'both.'")

        super(BiconicalOutflow, self).__init__(**kwargs)

    def __setstate__(self, state):
        # Compatibility hack, to handle the change to generalized
        #    higher order components in ModelSet.simulate_cube().
        self.__dict__ = state

        # Change '_type' from 'outflow' to 'higher_order':
        if self._type == 'outflow':
            self._type = 'higher_order'


    def evaluate(self, x, y, z, n, vmax, rturn, thetain, dtheta, rend, norm_flux, tau_flux):
        """Evaluate the outflow velocity as a function of position x, y, z"""

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.abs(z)/r)*180./np.pi
        theta[r == 0] = 0.
        vel = np.zeros(r.shape)

        if self.profile_type == 'increase':

            amp = vmax/rend**n
            vel[r <= rend] = amp*r[r <= rend]**n
            vel[r == 0] = 0

        elif self.profile_type == 'decrease':

            amp = -vmax/rend**n
            vel[r <= rend] = vmax + amp*r[r <= rend]** n


        elif self.profile_type == 'both':

            vel[r <= rturn] = vmax*(r[r <= rturn]/rturn)**n
            ind = (r > rturn) & (r <= 2*rturn)
            vel[ind] = vmax*(2 - r[ind]/rturn)**n

        elif self.profile_type == 'constant':

            vel[r <= rend] = vmax

        thetaout = np.min([thetain+dtheta, 90.])
        ind_zero = (theta < thetain) | (theta > thetaout) | (vel < 0)
        vel[ind_zero] = 0.

        return vel

    def velocity(self, x, y, z):
        """Return the velocity as a function of x, y, z"""
        return self.evaluate(x, y, z, self.n, self.vmax, self.rturn, self.thetain,
                             self.dtheta, self.rend, self.norm_flux, self.tau_flux)

    def light_profile(self, x, y, z):
        """Evaluate the outflow line flux as a function of position x, y, z"""

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.abs(z) / r) * 180. / np.pi
        theta[r == 0] = 0.
        flux = 10**self.norm_flux*np.exp(-self.tau_flux*(r/self.rend))
        thetaout = np.min([self.thetain + self.dtheta, 90.])
        ind_zero = ((theta < self.thetain) |
                    (theta > thetaout) |
                    (r > self.rend))
        flux[ind_zero] = 0.

        return flux

    def vel_direction_emitframe(self, x, y, z):
        r"""
        Method to return the velocity direction in the outflow Cartesian frame.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the outflow reference frame.

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyz).

            For biconical outflows, this is the +rhat direction, in spherical coordinates
            (r,phi,theta).
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2 )
        vel_dir_unit_vector = [ x/r, y/r, z/r ]
        return vel_dir_unit_vector



class UnresolvedOutflow(HigherOrderKinematicsSeparate, _DysmalFittable3DModel):
    """
    Model for an unresolved outflow component described by a Gaussian

    Parameters
    ----------
    vcenter : float
        Central velocity of the Gaussian in km/s

    fwhm : float
        FWHM of the Gaussian in km/s

    amplitude : float
        Amplitude of the Gaussian

    Notes
    -----
    This model simply produces a broad Gaussian spectrum that will be placed in the
    central spectrum of the galaxy.
    """

    vcenter = DysmalParameter(default=0)
    fwhm = DysmalParameter(default=1000.0, bounds=(0, None))
    amplitude = DysmalParameter(default=1.0, bounds=(0, None))

    _spatial_type = 'unresolved'
    _multicoord_velocity = False
    _native_geometry = 'cartesian'
    outputs = ('vout',)

    def __setstate__(self, state):
        # Compatibility hack, to handle the change to generalized
        #    higher order components in ModelSet.simulate_cube().
        self.__dict__ = state

        # Change '_type' from 'outflow' to 'higher_order':
        if self._type == 'outflow':
            self._type = 'higher_order'

    @staticmethod
    def evaluate(x, y, z, vcenter, fwhm, amplitude):
        return np.ones(x.shape)*vcenter

    def velocity(self, x, y, z):
        """Return the velocity as a function of x, y, z"""
        return self.evaluate(x, y, z, self.vcenter, self.fwhm, self.amplitude)

    def dispersion_profile(self, x, y, z, fwhm=None):
        """Dispersion profile for the outflow"""
        if fwhm is None: fwhm = self.fwhm
        return np.ones(x.shape)*(fwhm/2.35482)

    def light_profile(self, x, y, z):
        """Evaluate the outflow line flux as a function of position x, y, z
           All the light will be deposited at the center pixel."""

        # The coordinates where the unresolved outflow is placed needs to be
        # an integer pixel so for now we round to nearest integer.

        r = np.sqrt(x**2 + y**2 + z**2)
        ind_min = r.argmin()
        flux = x*0.
        flux[ind_min] = self.amplitude

        return flux

    def vel_direction_emitframe(self, x, y, z):
        r"""Method to return the velocity direction in the output geometry Cartesian frame.
        Undefined for `UnresolvedOutflow`"""
        return None



class UniformRadialFlow(HigherOrderKinematicsPerturbation, _DysmalFittable3DModel):
    """
    Model for a uniform radial flow.

    Parameters
    ----------
    vr : float
        Radial velocity in km/s. vr > 0 for outflow, vr < 0 for inflow

    Notes
    -----
    This model simply adds a constant radial velocity component
    to all of the positions in the galaxy.
    """
    vr = DysmalParameter(default=30.)

    _spatial_type = 'resolved'
    _multicoord_velocity = False
    _native_geometry = 'spherical'
    outputs = ('vrad',)

    def __init__(self, **kwargs):

        super(UniformRadialFlow, self).__init__(**kwargs)

    def __setstate__(self, state):
        # Compatibility hack, to handle the change to generalized
        #    higher order components in ModelSet.simulate_cube().
        self.__dict__ = state

        # Change '_type' from 'flow' to 'higher_order':
        if self._type == 'flow':
            self._type = 'higher_order'

    @staticmethod
    def evaluate(x, y, z, vr):
        """Evaluate the radial velocity as a function of position x, y, z"""

        vel = np.ones(x.shape) * (vr)

        return vel

    def velocity(self, x, y, z):
        """Return the velocity as a function of x, y, z"""
        return self.evaluate(x, y, z, self.vr)


    def vel_direction_emitframe(self, x, y, z):
        r"""
        Method to return the velocity direction in the galaxy Cartesian frame.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the radial flow reference frame.

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyz).

            For a uniform radial flow, this is the +rhat direction, in spherical coordinates
            (r,phi,theta).
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2 )
        vel_dir_unit_vector = [ x/r, y/r, z/r ]
        return vel_dir_unit_vector


class UniformBarFlow(HigherOrderKinematicsPerturbation, _DysmalFittable3DModel):
    """
    Model for a uniform flow along a bar, along some axis in the galaxy midplane.

    Parameters
    ----------
    vbar : float
        Bar streaming velocity in km/s. vbar > 0 for outflow, vbar < 0 for inflow

    phi : float
        Azimuthal angle of bar, counter-clockwise from blue major axis. In degrees.
        Default is 90 (eg, along galaxy minor axis)

    bar_width : float
        Width of the bar, in kpc.
        So bar velocity only is nonzero between -bar_width/2 < ygal  < bar_width/2.

    Notes
    -----
    This model simply adds a constant radial velocity component
    to all of the positions in the galaxy.
    """

    vbar = DysmalParameter(default=30.)
    phi = DysmalParameter(default=90.)
    bar_width = DysmalParameter(default=2.)

    _spatial_type = 'resolved'
    _multicoord_velocity = False
    _native_geometry = 'cartesian'
    outputs = ('vflow',)

    def __init__(self, **kwargs):
        super(UniformBarFlow, self).__init__(**kwargs)

    @staticmethod
    def evaluate(x, y, z, vbar, phi, bar_width):
        """Evaluate the bar velocity amplitude as a function of position x, y, z"""

        phi_rad = phi * np.pi / 180.
        # Rotate by phi_rad:
        ybar = - np.sin(phi_rad) * x + np.cos(phi_rad) * y

        vel = np.ones(ybar.shape) * (vbar)
        if len(ybar.shape) > 0:
            # Array-like inputs
            vel[np.abs(ybar) > 0.5*bar_width] = 0.
        else:
            # Float inputs:
            if np.abs(ybar) > 0.5*bar_width:
                vel = 0.

        return vel

    def velocity(self, x, y, z):
        """Return the velocity as a function of x, y, z"""
        return self.evaluate(x, y, z, self.vbar, self.phi, self.bar_width)

    def vel_direction_emitframe(self, x, y, z):
        r"""
        Method to return the velocity direction in the galaxy Cartesian frame.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the galaxy/bar reference frame.

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyz).

            For a bar uniform flow, this is the (xbar, -xbar) direction, in cartesian coordinates,
            for xbar <, > 0.
        """
        # Rotate by phi_rad:
        phi_rad = self.phi * np.pi / 180.
        xbar = np.cos(phi_rad) * x + np.sin(phi_rad) * y
        vel_dir_unit_vector = [ np.cos(phi_rad)*np.sign(xbar), -np.sin(phi_rad)*np.sign(xbar), z*0.]

        return vel_dir_unit_vector



class SpiralDensityWave(HigherOrderKinematicsPerturbation, _DysmalFittable3DModel):
    """
    Model for a spiral density wave, assumed in the galaxy midplane.

    Parameters
    ----------

    Notes
    -----
    """

    m = DysmalParameter(default=2., bounds=(0, None))       # Number of photometric arms
    phi0 = DysmalParameter(default=0., bounds=(0, 360))     # Angle offset of arm winding, in degrees
    cs = DysmalParameter(default=50., bounds=(0, None))     # Speed of sound, in km/s
    epsilon = DysmalParameter(default=0.1, bounds=(0,None)) # Density contrast of perturbation
    Om_p = DysmalParameter(default=0.)                      # Driving angular speed


    _spatial_type = 'resolved'
    _multicoord_velocity = True
    _native_geometry = 'cylindrical'
    outputs = ('vr','vphi','vz',)

    def __init__(self, Vrot=None, rho=None, **kwargs):

        # Set functions: MUST TAKE R as input!
        self.Vrot = Vrot
        self.rho0 = rho0
        # Functions of R (midplane): rho0, f, k, Vrot,
        # -> k and f are actually derived....

        super(SpiralDensityWave, self).__init__(**kwargs)

    def ep_freq_sq(self, R):
        """ Return kappa^2, square of the epicyclic frequency """
        dx=1.e-5
        order=3

        V = self.Vrot(R)
        Om = V / R
        shR = np.shape(R)
        if len(shR) == 0:
            dVrot_dR = scp_misc.derivative(self.Vrot, R, dx=dx, n=1, order=order)
        else:
            dVrot_dR = R * 0.
            for i in range(shR[0]):
                if len(shR) == 1:
                    dVrot_dR[i] = scp_misc.derivative(self.Vrot, R[i], dx=dx, n=1, order=order)
                else:
                    for j in range(shR[1]):
                        if len(shR) == 2:
                            dVrot_dR[i,j] = scp_misc.derivative(self.Vrot, R[i,j],
                                                                  dx=dx, n=1, order=order)
                        else:
                            for k in range(shR[2]):
                                dVrot_dR[i,j,k] = scp_misc.derivative(self.Vrot, R[i,j,k],
                                                                      dx=dx, n=1, order=order)


        kappasq = 2*Om * ( R * (1./R * dVrot_dR - V/R**2 ) + 2*Om )

        return kappasq

    def k(self, R):
        """Calculate wavenumber"""

        Om = self.Vrot(R) / R
        kappasq = self.ep_freq_sq(R)
        eta = np.sqrt(1. - kappasq / (self.m**2 * (Om-self.Om_p)**2))

        wavenum = (1.-self.Om_p / Om) * eta * self.m * Om / self.cs

        return wavenum

    def f(self, R):
        """Shape of spiral arms, with f=m*phi = Int_0^R(k dR) """
        intgrl, abserr = scp_integrate.quad(self.k, 0, R, args=(R))

        return intgrl


    def rho_perturb(self, x, y, z):
        """
        Return the density perturbation -- consider only the midplane
        Given by Eq. A9, Davies et al. 2009, ApJ, 702, 114
        """

        R = np.sqrt(x**2 + y**2)
        phi0_rad = self.phi0 * np.pi / 180.
        phi = utils.get_geom_phi_rad_polar(x, y) - phi0_rad

        fvals = self.f(R)
        rho0vals = self.rho0(R)

        rho1 = self.eps * rho0vals * np.cos( self.m*phi - fvals )

        return rho1

    def vr_perturb(self, x, y, z):
        """
        Return the radial velocity perturbation
        Given by Eq. A10, Davies et al. 2009, ApJ, 702, 114

        Here inflow is NEGATIVE, outflow is POSITIVE.
        """

        R = np.sqrt(x**2 + y**2)
        Om = self.Vrot(R) / R
        phi0_rad = self.phi0 * np.pi / 180.
        phi = utils.get_geom_phi_rad_polar(x, y) - phi0_rad

        fvals = self.f(R)
        kvals = self.k(R)

        #vr1 = -self.eps * self.m*(Om-self.Om_p)/kvals * np.cos( self.m*phi - fvals )

        # Inflow is neg for this vector definition
        vr1 = self.eps * self.m*(Om-self.Om_p)/kvals * np.cos( self.m*phi - fvals )

        return vr1


    def vphi_perturb(self, x, y, z):
        """
        Return the phi-direction velocity perturbation
        Given by Eq. A11, Davies et al. 2009, ApJ, 702, 114
        """

        R = np.sqrt(x**2 + y**2)
        Om = Vrot(R) / R
        phi = calc_phi(x, y, phi0)

        #fvals = f(R)
        #kvals = k(R)
        fvals = f(R, m, cs, Om_p, Vrot)
        kvals = k(R, m, cs, Om_p, Vrot)

        kappasq = ep_freq_sq(R, Vrot)

        vphi1 = eps * kappasq/(2.*kvals*Om) * np.sin( m*phi - fvals )

        return vphi1

    def vLOS_perturb(self, x, y, z):
        """
        Return the projected LOS velocity combining both radial and phi components.
        For use in visualizing.
        Handled via matrix multiplication & projection for model_set.simulate_cube()

        Uses NEGATIVE for inflow, POSITIVE for outflow
        """

        R = np.sqrt(x ** 2 + y ** 2)
        vr1 = self.vr_perturb(x, y, z)
        vphi1 = self.vphi_perturb(x, y, z)

        return vr1*(-y/R) + vphi1*(x/R)


    def velocity(self, x, y, z):
        """
        Evaluate the spiral density velocity amplitude as a function of position x, y, z,
        for the different radial and phi components.
        Return a tuple of (vr, vphi, vz).
        """

        vr1 = self.vr_perturb(x, y, z)
        vphi1 = self.vphi_perturb(x, y, z)

        # Tuple in the native cylindrical coordinates (really polar, ignoring z)
        return (vr1, vphi1, z*0.)


    def light_profile(self, x, y, z):
        R = np.sqrt(x ** 2 + y ** 2)

        phi_rad = phi * np.pi / 180.
        phi_geom_rad = utils.get_geom_phi_rad_polar(x, y)

        return self.rho_perturb(x, y, z)

    def vel_direction_emitframe(self, x, y, z):
        r"""
        Method to return the velocity matrix in the output Cartesian frame.

        As the native geometry is cylindrical (ignoring z dir), and the velocity is multicoordinate,
        we need a more complex output than a single vector for a dotproduct.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the output reference frame.

        Returns
        -------
        vel_dir_matrix : 3x3-element matrix
            Transform of the velocity from the native coordinates to the output cartesian frame.

            vel = (R, phi, z).
            Need matmul(vel_dir_matrix, vel) = vel in (x,y,z). So:
            vel_dir_matrix = [[Rtox, phitox, ztox],
                              [Rtoy, phitoy, ztoy],
                              [Rtoz, phitoz, ztoz]]

        """
        vel_dir_matrix = np.array([[x/R, -y/R, 0.*z],
                                   [y/R,  x/R, 0.*z],
                                   [0.*z, 0.*z, 0.*z]])

        # As zsky_unit_vector = [ x*0., y*0. + np.sin(inc), z*0. - np.cos(inc) ]
        # The dot is:
        # vr :  y/R*sin(i) = sin(phi) * sin(i)
        # vphi: x/R*sin(i) = cos(phi) * sin(i)
        # So really only picking out the y-cartesian coord

        return vel_dir_matrix
