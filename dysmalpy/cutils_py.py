#


import numpy as np

from numba import jit


DTYPE_t = np.float64

# def populate_cube(flux, vel, sigma, vspec):
#     # init: flux, vel, sigma: XYZ
#     #       vspec: SPEC
#     
#     print("HI I'M THE NEW ONE!")
#     
#     nwave = len(vspec)
#     #vspectile = np.tile(vspec, (1,flux.shape[0],flux.shape[1], flux.shape[2]))
#     vspectile = np.repeat(vspec,flux.shape[0]*flux.shape[1]*flux.shape[2]).reshape((nwave, 
#             flux.shape[0],flux.shape[1], flux.shape[2]))
#     
#     amp = flux / np.sqrt(2. * np.pi * sigma)
#     
#     
#     gaus = amp*np.exp(-0.5 * ((vel-vspectile)/sigma)**2)
#     
#     delt_z = 1.   # if we ever care about spacing between z datapoints for normalization, etc...
#     
#     result_np = np.sum(gaus, axis=1) * delt_z
# 
#     return result_np



@jit
def populate_cube(flux, vel, sigma, vspec):
    # init: flux, vel, sigma: XYZ
    #       vspec: SPEC

    print("HI I'M THE NEWER ONE!")
    
    result_np = np.zeros([len(vspec), flux.shape[1], flux.shape[2]], dtype=DTYPE_t)
    
    for z in range(flux.shape[0]):
        v = vel[z, :, :]
        sig = sigma[z, :, :]
        f = flux[z, :, :]
        amp = f / np.sqrt(2.0 * np.pi * sig)

        for s in range(vspec.shape[0]):

            result_np[s, :, :] += amp * np.exp(-0.5 * ((vspec[s] - v) / sig) **2)

    return result_np
