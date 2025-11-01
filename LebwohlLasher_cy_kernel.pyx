# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, infer_types=True, initializedcheck=False
# cython: embedsignature=True

# GPT recommended these headers 

from cython cimport boundscheck, wraparound, cdivision
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport cos, exp
from cython.parallel cimport prange

ctypedef np.float64_t f64

@cython.cfunc
@cython.inline
cdef double P2_from_cos(double c) nogil:
    return 0.5 * (3.0 * c * c - 1.0)

@cython.cfunc
@cython.inline
cdef Py_ssize_t pmod(Py_ssize_t i, Py_ssize_t n) nogil:
    if i >= n:
        return i - n
    if i < 0:
        return i + n
    return i

@cython.cfunc
cdef double _local_delta_energy(double[:, ::1] arr,
                                Py_ssize_t ix,
                                Py_ssize_t iy,
                                double dtheta,
                                Py_ssize_t nmax) nogil:
    cdef Py_ssize_t ixp = pmod(ix + 1, nmax)
    cdef Py_ssize_t ixm = pmod(ix - 1, nmax)
    cdef Py_ssize_t iyp = pmod(iy + 1, nmax)
    cdef Py_ssize_t iym = pmod(iy - 1, nmax)

    cdef double t0 = arr[ix, iy]
    cdef double t1 = t0 + dtheta
    cdef double nb, d0, d1, de = 0.0

    nb = arr[ixp, iy]; d0 = t0 - nb; d1 = t1 - nb
    de += (1.0 - 3.0 * (cos(d1) * cos(d1))) - (1.0 - 3.0 * (cos(d0) * cos(d0)))
    nb = arr[ixm, iy]; d0 = t0 - nb; d1 = t1 - nb
    de += (1.0 - 3.0 * (cos(d1) * cos(d1))) - (1.0 - 3.0 * (cos(d0) * cos(d0)))
    nb = arr[ix, iyp]; d0 = t0 - nb; d1 = t1 - nb
    de += (1.0 - 3.0 * (cos(d1) * cos(d1))) - (1.0 - 3.0 * (cos(d0) * cos(d0)))
    nb = arr[ix, iym]; d0 = t0 - nb; d1 = t1 - nb
    de += (1.0 - 3.0 * (cos(d1) * cos(d1))) - (1.0 - 3.0 * (cos(d0) * cos(d0)))

    return 0.5 * de  # parity with your Python energy

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mc_step_checkerboard(double[:, ::1] arr, double Ts):
    """
    OpenMP chequerboard sweep (periodic BCs).
    Red/black updates touch disjoint sites ⇒ safe to update arr in parallel.
    Reductions are auto-inferred (no 'reduction=' kwarg in Cython 3.1).
    """
    cdef Py_ssize_t nmax = arr.shape[0]
    if arr.shape[1] != nmax:
        raise ValueError("arr must be square (n x n)")

    cdef double scale = 0.1 + Ts

    # RNG with GIL
    dtheta_r = np.random.normal(0.0, scale, (nmax, nmax)).astype(np.float64, copy=False)
    urand_r  = np.random.random((nmax, nmax)).astype(np.float64, copy=False)
    dtheta_b = np.random.normal(0.0, scale, (nmax, nmax)).astype(np.float64, copy=False)
    urand_b  = np.random.random((nmax, nmax)).astype(np.float64, copy=False)

    cdef double[:, ::1] dtr = dtheta_r
    cdef double[:, ::1] urr = urand_r
    cdef double[:, ::1] dtb = dtheta_b
    cdef double[:, ::1] ubb = urand_b

    cdef Py_ssize_t i, j, jstart
    cdef double dth, dE
    cdef double accepts_r = 0.0
    cdef double accepts_b = 0.0

    # Red sublattice (i+j even)
    for i in prange(nmax, schedule='static', nogil=True):
        jstart = 0 if (i & 1) == 0 else 1
        for j in range(jstart, nmax, 2):
            dth = dtr[i, j]
            dE  = _local_delta_energy(arr, i, j, dth, nmax)
            if dE <= 0.0 or exp(-dE / Ts) >= urr[i, j]:
                arr[i, j] += dth
                accepts_r += 1.0  # auto-reduction

    # Black sublattice (i+j odd)
    for i in prange(nmax, schedule='static', nogil=True):
        jstart = 1 if (i & 1) == 0 else 0
        for j in range(jstart, nmax, 2):
            dth = dtb[i, j]
            dE  = _local_delta_energy(arr, i, j, dth, nmax)
            if dE <= 0.0 or exp(-dE / Ts) >= ubb[i, j]:
                arr[i, j] += dth
                accepts_b += 1.0  # auto-reduction

    return (accepts_r + accepts_b) / (nmax * nmax)

cpdef double total_energy(double[:, ::1] arr):
    """
    OpenMP total energy with scalar auto-reduction.
    """
    cdef Py_ssize_t nmax = arr.shape[0]
    if arr.shape[1] != nmax:
        raise ValueError("arr must be square (n x n)")

    cdef Py_ssize_t i, j, ip, im, jp, jm
    cdef double t, d
    cdef double e = 0.0

    for i in prange(nmax, schedule='static', nogil=True):
        ip = pmod(i + 1, nmax)
        im = pmod(i - 1, nmax)
        for j in range(nmax):
            jp = pmod(j + 1, nmax)
            jm = pmod(j - 1, nmax)
            t = arr[i, j]
            d = t - arr[ip, j]; e += P2_from_cos(cos(d))
            d = t - arr[im, j]; e += P2_from_cos(cos(d))
            d = t - arr[i, jp]; e += P2_from_cos(cos(d))
            d = t - arr[i, jm]; e += P2_from_cos(cos(d))

    return -0.5 * e

cpdef double get_order(np.ndarray[f64, ndim=2] arr):
    """
    Nematic order parameter S (largest eigenvalue of Q).
    """
    cdef Py_ssize_t nmax = arr.shape[0]
    if arr.shape[1] != nmax:
        raise ValueError("arr must be square (n x n)")

    cdef double[:, ::1] mv = arr
    cdef double sxx = 0.0
    cdef double sxy = 0.0
    cdef double syy = 0.0
    cdef double lx, ly
    cdef Py_ssize_t i, j

    with cython.nogil:
        for i in range(nmax):
            for j in range(nmax):
                lx = cos(mv[i, j])
                ly = cos(mv[i, j] - 1.5707963267948966)
                sxx += lx * lx
                sxy += lx * ly
                syy += ly * ly

    cdef double N2 = <double>(nmax * nmax)
    Q = np.zeros((3, 3), dtype=np.float64)
    Q[0, 0] = 3.0 * sxx
    Q[0, 1] = 3.0 * sxy
    Q[1, 0] = 3.0 * sxy
    Q[1, 1] = 3.0 * syy
    Q[0, 0] -= N2; Q[1, 1] -= N2; Q[2, 2] -= N2
    Q /= (2.0 * N2)

    evals = np.linalg.eigvalsh(Q)
    return float(evals.max())
