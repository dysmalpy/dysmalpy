#!/Users/ttshimiz/anaconda/bin/python

"""
A direct line-by-line translation of DYSMAL 2.19 which was written in IDL. This is just a test to make
sure we understand all components of the original code and form a base for DYSMAL in Python.

History
-------
25-07-2017: First write -- TTS
"""

# Module imports
import numpy as np
import scipy.special as scp_spec
import scipy.interpolate as scp_interp
import scipy.io as scp_io
import scipy.optimize as scp_opt
import astropy.constants as apy_con
import astropy.units as u
import astropy.cosmology as apy_cosmo
import astropy.convolution as apy_conv
import astropy.io.fits as fits

# Set the cosmology that will be assumed throughout
cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)

# Necessary constants
G = apy_con.G.value
Msun = apy_con.M_sun.value
pc = apy_con.pc.value


# For testing right now
def run_test():
    # Function to test DYSMALPY with inputs for a known galaxy that we can test against the IDL version
    uniqID = "GS4_43501"
    redshift = 1.613
    inc = 62.
    theta = 143.
    maxr = 2.  # [arcsec]
    rstep = 0.125
    dscale = 1. / (cosmo.arcsec_per_kpc_proper(redshift).to(u.arcsec / u.pc)).value
    mscale = 190837543014.914
    beam = [0.55]
    velmax = 1000.
    vstep = 30.
    sigma0 = 39.  # [km/s]
    turb = sigma0 * 2.35482
    sang = None
    sth = None
    auto = None
    erad = np.array([0.58, 0.58, 0.118031])
    emin = np.array([0., 0., 0., 0.])
    emax = np.array([5., 5., 5., 5.])
    noise = 0
    cubename = 'test_py4'
    enser = np.array([1., 1., 4.])
    Nrvir = np.array([12.141565])
    Nconcentration = np.array([5.0])
    xshift = 0.
    yshift = 0.
    ethick = np.array([0.0985213, 0.0985213, 0.100246])
    elight = np.array([0., 1., 0.])
    eratio = np.array([9.9196854e+10, 9.919686e+02, 2.0340846e+14])
    Nthick = np.array([0.0985213])
    Nlight = np.array([0.0])
    Nratio = np.array([9.9196854e+10])
    msc_arr = np.array([4.40585910e+10, 0.00000000e+00, 2.9610287e+10, 1.1674034e+11])
    do_Psupport = True
    Psupport_Re = erad[0]
    type_fit = "testpy_taro"
    dir_dysm = '/Users/ttshimiz/Dropbox/Research/LLAMA/dysmal/out_dysm/'
    out_dir = dir_dysm + uniqID + '/' + type_fit + '/'
    do_adiabatic_con = False
    use_noordermeer = True

    test = dysmalpy(inc, theta, maxr, rstep, dscale, mscale, beam, velmax, vstep, turb, sang, sth, auto,
                    erad, emin, emax, enser, Nrvir, Nconcentration, xshift=xshift, yshift=yshift,
                    ethick=ethick, elight=elight, eratio=eratio, Nthick=Nthick, Nlight=Nlight, Nratio=Nratio,
                    msc_arr=msc_arr, do_Psupport=do_Psupport, Psupport_Re=Psupport_Re, out_dir=out_dir,
                    use_noordermeer=use_noordermeer, do_adiabatic_con=do_adiabatic_con, noise=noise,
                    cubename=cubename)

    return test


def adiabatic(rprime, r_adi, adia_v_DM, adia_x_DM, adia_v_disk):
    if rprime < 0.:
        rprime = 0.1
    if rprime < adia_x_DM[1]:
        rprime = adia_x_DM[1]
    rprime_interp = scp_interp.interp1d(adia_x_DM, adia_v_DM, fill_value="extrapolate")
    result = r_adi + r_adi * ((r_adi*adia_v_disk**2)/(rprime*(rprime_interp(rprime))**2)) - rprime
    return result


def rotation_matrix(xang, yang, zang, inverse=False):

    Rz = np.array([[np.cos(zang), -np.sin(zang), 0.], [np.sin(zang), np.cos(zang), 0.], [0., 0., 1.]])
    Ry = np.array([[np.cos(yang), 0., np.sin(yang)], [0., 1., 0.], [-np.sin(yang), 0., np.cos(yang)]])
    Rx = np.array([[1., 0., 0.], [0., np.cos(xang), -np.sin(xang)], [0., np.sin(xang), np.cos(xang)]])

    if not inverse:
        return np.dot(Rx, np.dot(Ry, Rz))
    else:
        return np.dot(Rz, np.dot(Ry, Rx))


def create_beam_kernel(size, pxs, bmaj, bmin, bang):

    bpr = np.zeros((size, size))
    xycen = np.int(size/2)

    for ix in range(size):
        for iy in range(size):
            xprime = (ix - xycen) * np.cos(bang) - (iy - xycen) * np.sin(bang)
            xfac = (pxs * xprime / (bmin / 2.355))**2.
            yprime = (ix - xycen) * np.sin(bang) + (iy - xycen) * np.cos(bang)
            yfac = (pxs * yprime / (bmaj / 2.355))**2.
            bpr[iy, ix] = 10. * np.exp(-0.5 * (xfac + yfac))

    return bpr


# Main routine
def dysmalpy(inc, theta, maxr, rstep, dscale, mscale, beam, velmax, vstep, turb,
             sang, sth, auto, erad, emin, emax, enser, Nrvir, Nconcentration,
             xshift=0., yshift=0., vshift=0, vexp=0,
             ethick=np.array([]), elight=np.array([]), eratio=np.array([]),
             Nthick=np.array([]), Nlight=np.array([]), Nratio=np.array([]),
             msc_arr=None, bhmass=0, noise=0, do_Psupport=True, Psupport_Re=None,
             out_dir='./', do_circaper=True,
             cubename='velcube', use_noordermeer=True, do_adiabatic_con=True):

    # Set the directory where the Noordermeer+2008 velocity curves live
    # if using them
    if use_noordermeer:
        dir_noordermeer = '/Users/ttshimiz/Dropbox/Research/LLAMA/dysmal/noordermeer/'

    # Use the array of mass scalings if one is given
    if msc_arr is not None:
        mscale = np.sum(msc_arr)

    # !!!! This all needs to be rewritten and better handled !!!!
    if (len(ethick) > 0) | (len(Nthick) > 0):
        thick = np.hstack([[0.], ethick, Nthick])
    else:
        thick = np.array([0.])

    if len(thick) > 1:
        thick = thick[1:]

    if (len(elight) == 0) & (len(eratio) > 0):
        elight = eratio

    if (len(Nlight) == 0) & (len(Nratio) > 0):
        Nlight = Nratio

    if (len(eratio) == 0) & (len(elight) > 0):
        eratio = elight

    if (len(Nratio) == 0) & (len(Nlight) > 0):
        Nratio = Nlight

    if (len(eratio) > 0) | (len(Nratio) > 0):
        light = np.hstack([[0.], elight / eratio, Nlight / Nratio])
    else:
        light = np.array([0.])

    if len(light) > 1:
        light = light[1:]

    inc = np.pi/180.*inc
    # What is the point is the next two lines?
    pxs = rstep/1.
    vstep = vstep/1.

    # Setup the model cube
    oversize = 1.5*(1.+np.max(np.abs(np.array([xshift, yshift])/maxr)))
    xsize = np.int((maxr*oversize)/pxs*2.+0.5)
    if np.mod(xsize, 2) < 0.5:
        xsize = xsize + 1
    maxr_y = np.max(np.array([maxr*oversize, np.min(np.hstack([maxr*oversize/np.cos(inc), maxr*5.]))]))
    ysize = np.int(maxr_y/pxs*2.+0.5)
    if np.mod(ysize, 2) < 0.5:
        ysize = ysize + 1
    maxr_z = np.min(np.array([maxr_y, np.max(np.hstack([thick * 2., maxr_y * np.sin(inc)]))]))
    zsize_full = np.max([3, np.int(maxr_z / pxs * 2. + 0.5)])
    if np.mod(zsize_full, 2) < 0.5:
        zsize_full = zsize_full + 1
    zsize = np.max([3, np.int(np.max(thick * 2.) / pxs * 2. + 0.5)])
    if np.mod(zsize, 2) < 0.5:
        zsize = zsize + 1

    # Calculate the 1-D profiles. Use steps in radius 1/3 the size of the
    # requested radial step for the model
    incr = pxs/3.
    # rmaxes = np.array([maxr, maxr_y, maxr_z])
    rmax = np.sqrt((oversize*maxr)**2 + (5*maxr)**2)
    r1d = np.arange(0, rmax+2*incr, incr)
    rsize = len(r1d)

    n_cpts = len(eratio) + len(Nratio)
    type_cpt = []
    cpt = np.zeros((rsize, n_cpts))

    # Sersic Profiles *** Can probably use the Astropy.modeling Serscic1D model
    if (len(eratio) > 0) & (len(enser) == 0):
        enser = np.ones(len(eratio))

    for i in range(len(eratio)):
        bna = np.arange(2000)*0.01
        left = scp_spec.gamma(2*enser[i])
        right = 2*scp_spec.gammainc(2*enser[i], bna)*scp_spec.gamma(2*enser[i])
        if (left < right[1]):
            fint = scp_interp.interp1d(right, bna, kind='linear')
        else:
            fint = scp_interp.interp1d(right, bna, kind='quadratic')
        bn = fint(left)
        etmp = eratio[i]*np.exp(-bn*((r1d/erad[i])**(1./enser[i])))
        etmp[r1d < emin[i]] = 0
        cpt[:, i] = etmp
        type_cpt.append('Sersic')

    # NFW haloes
    for i in range(len(Nratio)):
        # rho0 = 1.0
        Ntmp = Nratio[i]*2*Nrvir[i]/(Nconcentration[i]*(1+Nconcentration[i]*r1d/Nrvir[i])**2)
        cpt[:, len(eratio)+i] = Ntmp
        type_cpt.append('NFW')

    # Flux and mass profile for each z-layer
    ir = np.zeros((np.int(zsize/2)+1, rsize))
    mr = np.zeros((np.int(zsize/2)+1, rsize))
    mcpt = np.zeros((rsize, n_cpts))
    for zp in range(np.int(zsize/2.)+1):
        for i in range(n_cpts):
            mcpt[:, i] = mcpt[:, i] + cpt[:, i]*np.exp(-0.5*(pxs*zp/thick[i])**2)
            mr[zp, :] = mr[zp, :] + cpt[:, i]*np.exp(-0.5*(pxs*zp/thick[i])**2)
            ir[zp, :] = ir[zp, :] + light[i]*cpt[:, i]*np.exp(-0.5*(pxs*zp/thick[i])**2)

    # Total flux and mass profiles
    mprof = np.zeros(rsize)
    iprof = np.zeros(rsize)
    for ii in range(rsize):
        mprof[ii] = np.mean(mr[:, ii])
        iprof[ii] = np.mean(ir[:, ii])

    # Radial points at which v is calculated
    rv = r1d[0:-1]

    # Calculate the 1D mass profile
    msum = np.zeros(rsize-1)                 # Sum of the enclosed mass profiles
    msum_int = np.zeros((rsize-1, n_cpts))   # Integrated (enclosed) mass profiles

    for i in np.arange(1, rsize-1):
        for j in range(n_cpts):
            msum_int[i, j] = (msum_int[i-1, j] +
                              np.pi*(r1d[i]**2 - r1d[i-1]**2)*mcpt[i, j])

        msum[i] = np.sum(msum_int[i, :])

    # Mass scaling in terms of mass enclosed within maxr
    if msc_arr is None:
        msum_int = msum_int*mscale / np.max(msum[r1d < maxr])
    else:
        for i in range(n_cpts):
            msum_int[:, i] = msum_int[:, i] * msc_arr[i] / np.max(msum_int[rv <= maxr, i])

        msum = np.sum(msum_int, axis=1)

    msum = msum*mscale / np.max(msum[rv <= maxr])

    # Add in the black hole mass in the center
    msum = msum + bhmass
    mprof[0] = mprof[0] + bhmass

    # Calculate the surface mass density
    msurf = np.zeros(rsize-1)
    msurf[0] = 0
    for i in range(1, rsize-1):
        msurf[i] = mprof[i] * 2*np.pi*r1d[i] * (r1d[i+1] - r1d[i-1])/2.
    msurf = msurf * np.max(msum)/np.sum(msurf)
    for i in range(1, rsize-1):
        msurf[i] = msurf[i] / (2*np.pi*r1d[i] * (r1d[i+1] - r1d[i-1])*dscale/2.)

    # Calculate the 1D velocity profile
    # Account for flattening using the Noordermeer+2008 profiles
    # NOTE: Currently only implemented for a combination of Sersic, BH, and NFW profiles
    v_baryon = np.zeros(msum.shape)
    vel = np.zeros(len(rv))
    ncomp = len(msc_arr)

    # GM/r for NFW halo
    if len(Nratio) > 0:
        vel = np.sqrt((G*msum_int[:, (ncomp-1)]*Msun/(rv*dscale*pc)) / 1000.**2)

    v_halo = vel.copy()

    # !!!! Need to make sure code works without requiring the flattening from Noordermeer+2008 !!!!
    for cnt_comp in range(len(enser)):

        # Read the Noordermeer rotation curve for the nearest Sersic n and thickness q
        noordermeer_n = np.arange(0.5, 8.1, 0.1)   # Sersic indices
        noordermeer_invq = np.array([1, 2, 3, 4, 5, 6, 8, 10, 20, 100])   # 1:1, 1:2, 1:3, ...flattening
        invthickratio = (erad[cnt_comp]/ethick[cnt_comp])*(2./2.35482)
        nearest_n = noordermeer_n[np.argmin(np.abs(noordermeer_n - enser[cnt_comp]))]
        nearest_q = noordermeer_invq[np.argmin(np.abs(noordermeer_invq - invthickratio))]

        print('Using Noordermeer RCs...')
        file_noord = dir_noordermeer+'VC_n{0:3.1f}_invq{1}.save'.format(nearest_n, nearest_q)
        restNVC = scp_io.readsav(file_noord)
        N2008_vcirc = restNVC.N2008_vcirc
        N2008_rad = restNVC.N2008_rad
        N2008_Re = restNVC.N2008_Re
        N2008_mass = restNVC.N2008_mass

        if msc_arr is not None:
            mass_comp = msc_arr[cnt_comp]
        else:
            mass_comp = np.max(msum_int[:, cnt_comp])

        v_interp = scp_interp.interp1d(N2008_rad, N2008_vcirc, fill_value="extrapolate")
        v_sersiccomp = (v_interp(rv / erad[cnt_comp] * N2008_Re) * np.sqrt(mass_comp / N2008_mass) *
                        np.sqrt(N2008_Re / (erad[cnt_comp] * dscale / 1000.)))
        v_baryon = np.sqrt(v_baryon**2 + v_sersiccomp**2)
        vel = np.sqrt(vel**2 + v_sersiccomp**2)

    vel[0] = 0.
    v_circ = vel.copy()

    # !!!! This whole section I think can be really cleaned up. Lots of repetitive things

    if len(Nratio) > 0:
        v_halo_adi = v_halo.copy()
    m_adi = msum_int[:, np.array(type_cpt) == 'NFW'][:, 0]
    m_halo_nocon = m_adi.copy()
    sel_disk = 0
    mass_baryons = msum_int[:, sel_disk]
    if len(enser) > 2:
        mass_baryons = mass_baryons + msum_int[:, 2]

    # Perform adiabatic contraction
    if do_adiabatic_con:
        print('Applying adiabatic contraction...')
        rprime_all = np.zeros(rv.shape)
        # check_all = np.zeros(rv.shape)
        for i in range(1, len(rv)):
            result = scp_opt.newton(adiabatic, rv[i]+1., args=(rv[i], v_halo, rv, v_baryon[i]))
            # !!!! Why aren't nom and denom used here??? !!!!
            # nom = rv[i]*v_baryon[i]**2
            # denom_interp = scp_interp.interp1d(rv, v_halo, fill_value='extrapolate')
            # denom = result*denom_interp(result)**2
            rprime_all[i] = result

            # Need to see if there is same "check" for scp_opt.newton on whether its a local or global minimum

        v_halo_adi_interp = scp_interp.interp1d(rv, v_halo, fill_value='extrapolate')
        v_halo_adi = v_halo_adi_interp(rprime_all)
        vel = np.sqrt(v_halo_adi**2 + v_baryon**2)
        v_circ = vel.copy()

        # Recalculate the enclosed mass profile
        m_halo = (v_halo_adi*1000.)**2 * (rv*dscale*pc)/(G*Msun)
        m_halo[0] = 0.
        sel_halo = np.array(type_cpt) == 'NFW'
        msum_int[0:len(m_halo), sel_halo] = m_halo
        m_adi = m_halo

        if msc_arr is not None:
            msum = np.sum(msum_int, axis=0)

    if len(Nratio) == 0:
        v_halo = np.zeros(len(v_circ))
        v_halo_adi = v_halo.copy()

    # Apply scaling for pressure support if requested
    if do_Psupport:
        if Psupport_Re is None:
            # Use the Re of the first Sersic component if Psupport_Re is not given
            Psupport_Re = erad[0]
        print('Scaling for pressure support...')
        # Assume constant sigma(r) = sigma0
        sigma0 = turb/2.35842
        vel_squared = (vel**2 - 3.36*(rv/Psupport_Re)*sigma0**2)
        vel_squared[vel_squared < 0] = 0.
        vel = np.sqrt(vel_squared)

    # Save 1D mass and velocity profiles
    np.savetxt(out_dir + "v_intrinsic.txt", np.transpose([rv, msum, v_circ, vel, v_halo, v_halo_adi, v_baryon]),
               fmt="%2.4f\t%1.3e\t%3.3f\t%3.3f\t%3.3f\t%3.3f\t%3.3f",
               header=("radius [arcsec], total mass [Msun], v_circ [km/s], v_rot [km/s], v_DM [km/s],"
                       "v_DM contracted [km/s], v_baryons [km/s]"))

    np.savetxt(out_dir + "mass_intrins.txt", np.transpose([rv, msum, m_halo_nocon, m_adi, mass_baryons]),
               fmt="%2.4f\t%1.3e\t%1.3e\t%1.3e\t%1.3e",
               header=("radius [arcsec], total mass [Msun], DM mass [Msun], DM mass contracted [Msun],"
                       "baryonic mass [Msun]"))

    # Take a mass-weighted mean of the thick values of the components
    # NFW halo is assumed to be spherical in computing v^2 = GM/R
    # !!!! This section doesn't seem to actually be used anywhere else? !!!
    """
    if msc_arr is not None:
        if len(Nthick) == 0:
            idx = np.arange(len(thick))
        else:
            idx = np.arange(len(thick) - len(Nthick))

        
        wmean_thick = np.sum(msc_arr[idx]*thick[idx]/np.sum(msc_arr[idx]))

    else:
        wmean_thick = np.mean(thick)
    """

    # No intrinsic velocity dispersion?
    sigz = np.zeros(len(vel))

    # Print out to standard output and an ASCII file the total mass of each component
    file_mass = open(out_dir+'out-masses.txt', 'w')
    cptlist = np.zeros(n_cpts, dtype=int)
    for i in range(n_cpts):
            if type_cpt[i] == 'Sersic':
                cptlist[i] = i+1
            elif type_cpt[i] == 'NFW':
                cptlist[i] = i+1-len(eratio)

            txt = '# {0:9.3g} Msun is mass of {1} cpt {2} out to {3:4.1f} arcsec'.format(np.max(msum_int[rv <= maxr, i]),
                                                                                         type_cpt[i], cptlist[i], maxr)
            print(txt)
            file_mass.write(txt+'\n')
    txt = '# {0:9.3g} Msun is total mass out to {1:4.1f} arcsec'.format(np.max(msum[rv <= maxr]), maxr)
    print(txt)
    file_mass.write(txt+'\n')
    cpt_names = [type_cpt[i]+str(cptlist[i]) for i in range(n_cpts)]
    hdr = 'Radius     '+" ".join(['{:11s}'.format(j) for j in cpt_names])+'\n'
    file_mass.write(hdr)
    for i in range(rsize-1):
        mout = msum_int[i, :]
        line = '{:8.4f}'.format(rv[i])+"  "+"  ".join(['{:9.4g}'.format(j) for j in mout])+'\n'
        file_mass.write(line)
    file_mass.close()

    # Print out the model surface mass density, surface intensity, enclosed mass, and velocity
    mod_data = np.vstack([rv, mprof[0:-1], iprof[0:-1], msum, vel]).transpose()
    np.savetxt(out_dir+'out-model.txt', mod_data, header='# Radius  Surface-Mass-Density  Surface-Intensity'
                                                         '  Enclosed-Mass  Velocity', fmt='%9.4g')
    print('Written 1D radial profiles to out-model.txt')

    # Make the intensity and velocity maps
    print('Creating spectrally convolved intensity and velocity model maps...')

    # linx = np.arange(xsize) - np.int(xsize / 2.)
    # liny = np.arange(ysize) - np.int(ysize / 2.)
    # linz = np.arange(zsize) - np.int(zsize / 2.)
    x0 = 0.
    y0 = 0.
    z0 = 0.
    theta = 90. - theta
    i_cutoff = -1
    # radcutoff = np.max(r1d[ir[0,:] > i_cutoff])
    # w_rad = [0., np.max(r1d)*2.]                # for now no warps

    # w_ixyz will be the intensity cube of each x,y position in each z plane
    # w_vxyz will be the velocity cube adjusted for inclination
    # v_c will be the final velocity cube
    # im will be the image
    # w_vs2 is the square velocity dispersion out of the plane in each x,y position convolved with
    #       the instrumental resolution
    vx = np.arange(np.int(2*velmax/vstep+1))*vstep - velmax
    vc = np.zeros((xsize, xsize, len(vx)))

    w_ir = ir.copy()
    w_inc = inc
    w_theta = theta

    # Each ring is created face-on and then rotated about the horizontal axis before being added
    # to the full cube
    # We assume at this point that the disk is inclined about the horizontal axis
    # 'rad' is the actual radial position in the galaxy plane for each point being calculated

    # Initial calculations to save time during loops
    sin_inc = np.sin(w_inc)
    cos_inc = np.cos(w_inc)
    xx0, yy0, zz0 = np.meshgrid(np.arange(xsize, dtype=float), np.arange(ysize, dtype=float),
                                np.arange(zsize, dtype=float))
    xx0 -= np.int(xsize/2.) + x0
    yy0 -= np.int(ysize/2.) + y0
    zz0 -= np.int(zsize/2.) + z0
    xpos = pxs*xx0
    ypos = pxs*yy0
    xpos2 = xpos**2
    ypos2 = ypos**2

    w_ixyz_orig = np.zeros((ysize, xsize, zsize))
    w_vxyz_orig = np.zeros((ysize, xsize, zsize))
    w_vs2_orig = np.zeros((ysize, xsize, zsize))

    # Create an array that contains the radial distance from galaxy center for each z-plane
    rad_array = np.sqrt(xpos2 + ypos2)
    rad_array2d = rad_array[:, :, 0]

    # Interpolate the model velocity and dispersion onto the cube radii
    raduniq = np.unique(rad_array2d)
    print('Interpolating {} of {} intensities and velocities...'.format(len(raduniq), ysize*xsize))
    w_vinter_interp = scp_interp.interp1d(rv, vel, kind='quadratic')
    w_vinter = w_vinter_interp(raduniq) / raduniq * sin_inc
    w_vexp = vexp / raduniq * sin_inc
    w_vexp[0] = np.nan
    w_vs2inter_interp = scp_interp.interp1d(rv, sigz, kind='quadratic')
    w_vs2inter = w_vs2inter_interp(raduniq)**2                          # Isotropic, not sure why this is necessary
                                                                        # when sigz = 0
    w_iinter = np.zeros((np.int(zsize / 2.) + 1, len(raduniq)))
    for zp in range(np.int(zsize / 2) + 1):
        w_iinter_interp = scp_interp.interp1d(r1d, w_ir[zp, :], kind='quadratic')
        w_iinter[zp, :] = w_iinter_interp(raduniq)

    # NB: for w_Vinter, dividing by raduniq & multiplying by lin_xp is
    # equivalent to multiplying by cos(phi) where phi is angle from x-axis
    # similarly for w_Vexp, where it is cos(phi) from y-axis

    # Insert interpolated data into the appropriate arrays
    print('Copying interpolated values into arrays...')
    for xp in range(xsize):
        for yp in range(ysize):
            rpos = raduniq == rad_array2d[yp, xp]
            w_vxyz_orig[yp, xp, :] = w_vinter[rpos] * xpos[0, xp, 0] + w_vexp[rpos] * ypos[yp, 0, 0]
            w_vs2_orig[yp, xp, :] = w_vs2inter[rpos]
            for zp in range(int(zsize / 2) + 1):
                w_ixyz_orig[yp, xp, np.int(zsize / 2) - zp] = w_iinter[zp, rpos]
                w_ixyz_orig[yp, xp, np.int(zsize / 2) + zp] = w_iinter[zp, rpos]

    w_vxyz_orig[np.int(ysize/2), np.int(xsize/2), :] = 0

    w_ixyz = np.zeros((xsize, xsize, zsize_full))
    w_vxyz = np.zeros((xsize, xsize, zsize_full))
    w_vs2 = np.zeros((xsize, xsize, zsize_full))

    # w_inc_deg = w_inc * 180./np.pi

    print('Calculating 3D rotation for inclination and position angle...')
    # First incline the disk, then rotate to position angle
    rot_mat = rotation_matrix(-w_inc, 0.0, w_theta*np.pi/180.)

    # There must be a better way to do the rotation in python, but I can't figure it out right now...
    origai = np.where(w_ixyz_orig > i_cutoff)
    # Switch the columns around so that x coordinate is first column
    origai = np.vstack([origai[1], origai[0], origai[2]]).transpose()
    newpos = np.dot(origai, rot_mat)

    # Not entirely sure what is going on here. Why are we subtracting by the mean of the column???
    # Otherwise it looks like we're moving (0,0,0) to the center of the final cube
    newpos[:, 0] = newpos[:, 0] - np.mean(newpos[:, 0]) + np.int(xsize / 2)
    newpos[:, 1] = newpos[:, 1] - np.mean(newpos[:, 1]) + np.int(xsize / 2)
    newpos[:, 2] = newpos[:, 2] - np.mean(newpos[:, 2]) + np.int(zsize_full / 2)

    # Get all points that are actually in the final cube
    validpts = ((newpos[:, 0] >= 0) & (newpos[:, 0] <= (xsize - 1)) &
                (newpos[:, 1] >= 0) & (newpos[:, 1] <= (xsize - 1)) &
                (newpos[:, 2] >= 0) & (newpos[:, 2] <= (zsize_full - 1)))

    newpos = np.array(np.round(newpos[validpts, :]), dtype=np.int)

    # Move the columns back so its [y, x, z] i.e [row, column, z]
    # newpos = np.vstack([newpos[:, 1], newpos[:, 0], newpos[:, 2]]).transpose()

    # Computing the linear index for each of the indices in newpos?
    # No I think its getting the linear index for all newpos plus the 27 points around them.
    ai_index = np.array([newpos[0, 0] + newpos[0, 1]*xsize + newpos[0, 2]*xsize**2])
    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            for iz in [-1, 0, 1]:
                # dummy_newpos = np.vstack([newpos[:, 0] + ix,
                #                          newpos[:, 1] + iy,
                #                          newpos[:, 2] + iz])
                # ai_index = np.hstack([ai_index, np.ravel_multi_index(dummy_newpos, w_ixyz.shape, mode='clip')])
                ai_index = np.hstack([ai_index, ((newpos[:, 0] + ix) +
                                                 (newpos[:, 1] + iy)*xsize +
                                                 (newpos[:, 2] + iz)*xsize**2)])

    ai_index = ai_index[(ai_index >= 0) & (ai_index < (xsize*xsize*zsize_full))]
    ai_index = np.unique(ai_index)
    ai = np.unravel_index(ai_index, w_ixyz.shape, order='F')
    ai = np.vstack(ai).transpose()

    # Create inverse transformation to get positions within the original uninclined and unrotated cube
    rot_mat_inv = rotation_matrix(w_inc, 0, -w_theta*np.pi/180., inverse=True)
    origpos = np.dot(ai, rot_mat_inv)
    origpos[:, 0] = origpos[:, 0] - np.mean(origpos[:, 0]) + np.int(xsize/2)
    origpos[:, 1] = origpos[:, 1] - np.mean(origpos[:, 1]) + np.int(ysize/2)
    origpos[:, 2] = origpos[:, 2] - np.mean(origpos[:, 2]) + np.int(zsize/2)

    # Interpolate intensities to this grid of irregular original positions
    # and hence to the final regularly gridded positions
    validpts = ((origpos[:, 0] >= -0.5) & (origpos[:, 0] < (xsize - 0.5)) &
                (origpos[:, 1] >= -0.5) & (origpos[:, 1] < (ysize - 0.5)) &
                (origpos[:, 2] >= -0.5) & (origpos[:, 2] < (zsize - 0.5)))
    # origpos = np.array(np.round(origpos[validpts, :]), dtype=np.int)
    # ai = ai[validpts, :]
    # validpts = (w_ixyz_orig[origpos[:, 1], origpos[:, 0], origpos[:, 2]] > i_cutoff)

    print('Calculating {} of {} values for rotation'.format(np.sum(validpts), len(w_ixyz.flatten())))
    origpos = origpos[validpts, :]
    ai = ai[validpts, :]

    xi = np.vstack([origpos[:, 1], origpos[:, 0], origpos[:, 2]]).transpose()
    orig_grid = ((np.arange(ysize)), (np.arange(xsize)), (np.arange(zsize)))
    new_i = scp_interp.interpn(orig_grid, w_ixyz_orig, xi, bounds_error=False, fill_value=0.0)
    w_ixyz[ai[:, 1], ai[:, 0], ai[:, 2]] = new_i
    new_v = scp_interp.interpn(orig_grid, w_vxyz_orig, xi, bounds_error=False, fill_value=0.0)
    w_vxyz[ai[:, 1], ai[:, 0], ai[:, 2]] = new_v
    new_vs2 = scp_interp.interpn(orig_grid, w_vs2_orig, xi, bounds_error=False, fill_value=0.0)
    w_vs2[ai[:, 1], ai[:, 0], ai[:, 2]] = new_vs2

    # Set to 0 any pixels with flux < 0 or dispersion < 0
    tmp = (w_ixyz < 0)
    w_ixyz[tmp] = 0
    tmp = (w_vs2 < 0)
    w_vs2[tmp] = 0

    # Add to velocity cube
    print('Creating velocity cube...')
    for i in range(ai.shape[0]):
        xp = ai[i, 0]
        yp = ai[i, 1]
        zp = ai[i, 2]
        tmpgauss = np.exp(-0.5*(vx - w_vxyz[yp, xp, zp] - vshift)**2 / ((turb/2.355)**2. + w_vs2[yp, xp, zp]))
        newvel = tmpgauss / np.sum(tmpgauss)*100. * w_ixyz[yp, xp, zp]
        vc[yp, xp, :] = vc[yp, xp, :] + newvel

    # Convolve with the spatial beam
    print('Convolving with spatial beam profile...')

    # Create the kernel to convolve the cube with
    if len(beam) < 4:
        if len(beam) == 1:
            b_maj = beam[0]
            b_min = beam[0]
            b_ang = 0.
        elif len(beam) == 3:
            b_maj = beam[0]
            b_min = beam[1]
            b_ang = -beam[2]*np.pi/180.

        bpr = create_beam_kernel(xsize, pxs, b_maj, b_min, b_ang)

    elif len(beam) == 4:
        b_maj1 = beam[0]
        b_min1 = beam[0]
        b_scale1 = beam[1]
        b_maj2 = beam[2]
        b_min2 = beam[2]
        b_scale2 = beam[3]
        b_ang = 0.

        bpr1 = create_beam_kernel(xsize, pxs, b_maj1, b_min1, b_ang)
        bpr2 = create_beam_kernel(xsize, pxs, b_maj2, b_min2, b_ang)

        bpr = 10. * (bpr1 * b_scale1 / np.sum(bpr1) + bpr2 * b_scale2 / np.sum(bpr2))

    # In future use the astropy.spectralcube package to handle creation of final cube
    cvc = np.zeros((len(vx), xsize, xsize))    # Python is weird and flips axes when saving to FITS
    for i in range(len(vx)):

        cvc[i, :, :] = apy_conv.convolve_fft(vc[:, :, i], bpr)

    # Create final cube section
    linx = np.arange(xsize) - np.int(xsize/2)
    label_range = ((np.abs(linx) * pxs) <= maxr)
    x_range = ((linx * pxs) >= (-maxr - xshift)) & ((linx * pxs) <= (maxr - xshift))
    y_range = ((linx * pxs) >= (-maxr - yshift)) & ((linx * pxs) <= (maxr - yshift))
    xy_vec = linx[label_range] * pxs
    cvc_sec = cvc[np.ix_(np.ones(len(vx), dtype=np.bool), x_range, y_range)]

    # Normalize
    normalisation = np.mean(cvc_sec)
    cvc_sec = cvc_sec / normalisation

    # Add noise if desired
    if noise > 0:
        print('Adding noise...')
        cvc_shape = cvc_sec.shape
        noisecube = np.random.randn(cvc_shape[0], cvc_shape[1], cvc_shape[2])*noise
        cvc_sec = cvc_sec + noisecube

    # Make the FITS header and write the FITS cube
    if cubename[-5:] != '.fits':
        cubename = cubename + '.fits'
    print('Creating FITS velocity cube called {}'.format(cubename))
    hdu = fits.PrimaryHDU(data=cvc_sec)
    hdr = hdu.header
    hdr['CDELT1'] = pxs/3600.
    hdr['CRPIX1'] = 1
    hdr['CRVAL1'] = xy_vec[0]/3600.
    hdr['CDELT2'] = pxs/3600.
    hdr['CRPIX2'] = 1
    hdr['CRVAL2'] = xy_vec[0]/3600.
    hdr['CDELT3'] = vstep
    hdr['CRPIX3'] = 1
    hdr['CRVAL3'] = vx[0]

    hdu.header = hdr
    hdulist = fits.HDUList(hdu)
    hdulist.writeto(out_dir + cubename, overwrite=True)

    return cvc_sec


if __name__ == '__main__':

    run_test()
