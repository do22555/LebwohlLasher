# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, infer_types=True, initializedcheck=False
# cython: embedsignature=True

# Minimal, robust Cython kernels (no OpenMP). RNG is generated under the GIL;
# the update/energy loops run 'nogil' using C-level math & memoryviews.

from cython cimport boundscheck, wraparound, cdivision
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport cos, exp

ctypedef np.float64_t f64

@cython.cfunc
@cython.inline
cdef double P2_from_cos(double c) nogil:
    # P2(cos θ) = 0.5 * (3 cos^2 θ - 1)
    return 0.5 * (3.0 * c * c - 1.0)

@cython.cfunc
@cython.inline
cdef Py_ssize_t pmod(Py_ssize_t i, Py_ssize_t n) nogil:
    # Periodic index assuming n > 0
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
    """ΔE for a proposed rotation at (ix, iy) with 4 neighbours, periodic BCs."""
    cdef Py_ssize_t ixp = pmod(ix + 1, nmax)
    cdef Py_ssize_t ixm = pmod(ix - 1, nmax)
    cdef Py_ssize_t iyp = pmod(iy + 1, nmax)
    cdef Py_ssize_t iym = pmod(iy - 1, nmax)

    cdef double t0 = arr[ix, iy]
    cdef double t1 = t0 + dtheta
    cdef double nb, d0, d1, de = 0.0

    # (+x)
    nb = arr[ixp, iy]
    d0 = t0 - nb; d1 = t1 - nb
    de += (1.0 - 3.0 * (cos(d1) * cos(d1))) - (1.0 - 3.0 * (cos(d0) * cos(d0)))
    # (-x)
    nb = arr[ixm, iy]
    d0 = t0 - nb; d1 = t1 - nb
    de += (1.0 - 3.0 * (cos(d1) * cos(d1))) - (1.0 - 3.0 * (cos(d0) * cos(d0)))
    # (+y)
    nb = arr[ix, iyp]
    d0 = t0 - nb; d1 = t1 - nb
    de += (1.0 - 3.0 * (cos(d1) * cos(d1))) - (1.0 - 3.0 * (cos(d0) * cos(d0)))
    # (-y)
    nb = arr[ix, iym]
    d0 = t0 - nb; d1 = t1 - nb
    de += (1.0 - 3.0 * (cos(d1) * cos(d1))) - (1.0 - 3.0 * (cos(d0) * cos(d0)))

    # Keep parity with your Python total energy (overall 0.5 factor per site)
    return 0.5 * de


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mc_step_checkerboard(double[:, ::1] arr, double Ts):
    """
    One (serial) chequerboard Metropolis sweep with periodic BCs.

    - RNG arrays are created with the GIL in NumPy (fast C backend).
    - The update loops run 'nogil' in C for speed (no Python overhead).
    - Returns acceptance ratio in [0, 1].
    """
    cdef Py_ssize_t nmax = arr.shape[0]
    if arr.shape[1] != nmax:
        raise ValueError("arr must be square (n x n)")

    cdef double scale = 0.1 + Ts

    # --- Generate proposals & uniforms (GIL held) ---
    dtheta_r = np.random.normal(0.0, scale, (nmax, nmax)).astype(np.float64, copy=False)
    urand_r  = np.random.random((nmax, nmax)).astype(np.float64, copy=False)
    dtheta_b = np.random.normal(0.0, scale, (nmax, nmax)).astype(np.float64, copy=False)
    urand_b  = np.random.random((nmax, nmax)).astype(np.float64, copy=False)

    # C-typed views for nogil region
    cdef double[:, ::1] dtr = dtheta_r
    cdef double[:, ::1] urr = urand_r
    cdef double[:, ::1] dtb = dtheta_b
    cdef double[:, ::1] ubb = urand_b

    cdef Py_ssize_t i, j, jstart
    cdef double dth, dE
    cdef double accepts = 0.0

    # ---- Red sites (i+j even) ----
    with cython.nogil:
        for i in range(nmax):
            jstart = 0 if (i & 1) == 0 else 1
            for j in range(jstart, nmax, 2):
                dth = dtr[i, j]
                dE  = _local_delta_energy(arr, i, j, dth, nmax)
                if dE <= 0.0 or exp(-dE / Ts) >= urr[i, j]:
                    arr[i, j] += dth
                    accepts += 1.0

        # ---- Black sites (i+j odd) ----
        for i in range(nmax):
            jstart = 1 if (i & 1) == 0 else 0
            for j in range(jstart, nmax, 2):
                dth = dtb[i, j]
                dE  = _local_delta_energy(arr, i, j, dth, nmax)
                if dE <= 0.0 or exp(-dE / Ts) >= ubb[i, j]:
                    arr[i, j] += dth
                    accepts += 1.0

    return accepts / (nmax * nmax)


cpdef double total_energy(double[:, ::1] arr):
    """
    Cache-friendly total energy with periodic boundaries (serial, nogil).
    Mirrors your Python formula: 0.5 * sum over four bonds per site.
    """
    cdef Py_ssize_t nmax = arr.shape[0]
    if arr.shape[1] != nmax:
        raise ValueError("arr must be square (n x n)")

    cdef Py_ssize_t i, j, ip, im, jp, jm
    cdef double t, d, e = 0.0

    with cython.nogil:
        for i in range(nmax):
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

    # Python energy used: 0.5*((1 - 3 cos^2)...), which equals -P2.
    # Each site contributes four bonds and an overall 0.5 factor.
    return -0.5 * e


cpdef double get_order(np.ndarray[f64, ndim=2] arr):
    """
    Nematic order parameter S = largest eigenvalue of Q (3x3).
    We compute the l_a l_b sums in a nogil loop, then build Q and
    use NumPy's eigvalsh under the GIL for robustness.
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
                # sin(θ) = cos(θ - π/2)
                ly = cos(mv[i, j] - 1.5707963267948966)
                sxx += lx * lx
                sxy += lx * ly
                syy += ly * ly

    cdef double N2 = <double>(nmax * nmax)

    # Back to Python/NumPy for the tiny 3x3 eigensolve
    Q = np.zeros((3, 3), dtype=np.float64)
    Q[0, 0] = 3.0 * sxx
    Q[0, 1] = 3.0 * sxy
    Q[1, 0] = 3.0 * sxy
    Q[1, 1] = 3.0 * syy
    Q[0, 0] -= N2; Q[1, 1] -= N2; Q[2, 2] -= N2
    Q /= (2.0 * N2)

    evals = np.linalg.eigvalsh(Q)
    return float(evals.max())
