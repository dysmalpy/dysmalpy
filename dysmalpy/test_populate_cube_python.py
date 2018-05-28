

import numpy as np

def populate_cube(flux, vel, sigma, vspec):

    nz_sky_samp, ny_sky_samp, nx_sky_samp = flux.shape
    cube_final = np.zeros((vspec.shape[0], ny_sky_samp, nx_sky_samp))
    nspec = vspec.shape[0]
    velcube = np.tile(np.resize(vspec, (nspec, 1, 1)), (1, ny_sky_samp, nx_sky_samp))

    for zz in range(nz_sky_samp):
        f_cube = np.tile(flux[zz, :, :], (nspec, 1, 1))
        vobs_cube = np.tile(vel[zz, :, :], (nspec, 1, 1))
        sig_cube = np.tile(sigma[zz, :, :], (nspec, 1, 1))
        tmp_cube = np.exp(
            -0.5 * ((velcube - vobs_cube) / sig_cube) ** 2)
        cube_sum = np.nansum(tmp_cube, 0)
        cube_final += tmp_cube / cube_sum * f_cube

    return cube_final