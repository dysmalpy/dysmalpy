#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# Module containing Cython functions for speed optimization

# from math import exp, sqrt, pi
from libc.math cimport exp, sqrt, pi
import numpy as np

#cdef extern from "vfastexp.h":
#    double exp_approx "EXP" (double)

DTYPE_t = np.float64

def populate_cube(double [:, :, :] flux,
                  double [:, :, :] vel,
                  double [:, :, :] sigma,
                  double [:] vspec):

    cdef Py_ssize_t s, x, y, z
    cdef double amp, v, sig, f

    result_np = np.zeros([len(vspec), flux.shape[1], flux.shape[2]], dtype=DTYPE_t)
    cdef double [:, :, :] result = result_np

    for x in range(flux.shape[2]):
        for y in range(flux.shape[1]):
            for z in range(flux.shape[0]):

                v = vel[z, y, x]
                sig = sigma[z, y, x]
                f = flux[z, y, x]
                amp = f / sqrt(2.0 * pi * sig)

                for s in range(vspec.shape[0]):

                    result[s, y, x] += amp * exp(-0.5 * ((vspec[s] - v) / sig) **2)

    return result_np


    
def populate_cube_ais(double [:, :, :] flux,
                  double [:, :, :] vel,
                  double [:, :, :] sigma,
                  double [:] vspec,
                  long [:, :] ai):

    cdef Py_ssize_t s, x, y, z, i
    cdef double amp, v, sig, f

    result_np = np.zeros([len(vspec), flux.shape[1], flux.shape[2]], dtype=DTYPE_t)
    cdef double [:, :, :] result = result_np

    for i in range(ai.shape[1]):
        x = ai[0, i]
        y = ai[1, i]
        z = ai[2, i]

        v = vel[z, y, x]
        sig = sigma[z, y, x]
        f = flux[z, y, x]
        amp = f / sqrt(2.0 * pi * sig)

        for s in range(vspec.shape[0]):

            result[s, y, x] += amp * exp(-0.5 * ((vspec[s] - v) / sig) **2)

    return result_np
