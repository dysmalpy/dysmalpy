#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# Module containing Cython functions for speed optimization

# from math import exp, sqrt, pi
from libc.math cimport exp, sqrt, pi
import numpy as np

cdef extern from "vfastexp.h":
    double exp_approx "EXP" (double)

DTYPE_t = np.float64

def populate_cube(double [:, :, :] flux,
                  double [:, :, :] vel,
                  double [:, :, :] sigma,
                  double [:] vspec):

    cdef Py_ssize_t s, x, y, z
    cdef double amp, v, sig, f

    result_np = np.zeros([len(vspec), flux.shape[0], flux.shape[1]], dtype=DTYPE_t)
    cdef double [:, :, :] result = result_np

    for z in range(flux.shape[2]):
        for y in range(flux.shape[1]):
            for x in range(flux.shape[0]):

                v = vel[z, y, x]
                sig = sigma[z, y, x]
                f = flux[z, y, x]
                amp = f / sqrt(2.0 * pi * sig)

                for s in range(vspec.shape[0]):

                    result[s, y, x] += amp * exp_approx(-0.5 * ((vspec[s] - v) / sig) **2)

    return result_np
