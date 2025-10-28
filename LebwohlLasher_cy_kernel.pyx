# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: language = c

import cython
from cython.parallel cimport prange, atomic
cimport numpy as cnp
from libc.math cimport cos, exp
cimport openmp


@cython.cfunc
cdef double _local_delta_energy(
    double[:, ::1] arr,
    Py_ssize_t ix, Py_ssize_t iy,
    double dtheta, Py_ssize_t nmax
) nogil:
    """
    Compute local ΔE for a proposed rotation at (ix, iy) with periodic BCs.
    """
    cdef Py_ssize_t ixp = (ix + 1) % nmax
    cdef Py_ssize_t ixm = (ix - 1 + nmax) % nmax
    cdef Py_ssize_t iyp = (iy + 1) % nmax
    cdef Py_ssize_t iym = (iy - 1 + nmax) % nmax

    cdef double t0 = arr[ix, iy]
    cdef double t1 = t0 + dtheta
    cdef double en = 0.0
    cdef double nb, d0, d1

    # (ix+1, iy)
    nb = arr[ixp, iy]; d0 = t0 - nb; d1 = t1 - nb
    en += 0.5 * ((1.0 - 3.0 * cos(d1) * cos(d1)) - (1.0 - 3.0 * cos(d0) * cos(d0)))

    # (ix-1, iy)
    nb = arr[ixm, iy]; d0 = t0 - nb; d1 = t1 - nb
    en += 0.5 * ((1.0 - 3.0 * cos(d1) * cos(d1)) - (1.0 - 3.0 * cos(d0) * cos(d0)))

    # (ix, iy+1)
    nb = arr[ix, iyp]; d0 = t0 - nb; d1 = t1 - nb
    en += 0.5 * ((1.0 - 3.0 * cos(d1) * cos(d1)) - (1.0 - 3.0 * cos(d0) * cos(d0)))

    # (ix, iy-1)
    nb = arr[ix, iym]; d0 = t0 - nb; d1 = t1 - nb
    en += 0.5 * ((1.0 - 3.0 * cos(d1) * cos(d1)) - (1.0 - 3.0 * cos(d0) * cos(d0)))

    return en


@cython.boundscheck(False)
@cython.wraparound(False)
def mc_step_checkerboard(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] arr,
    double Ts,
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] dtheta_r,
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] urand_r,
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] dtheta_b,
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] urand_b,
):
    """
    One chequerboard Metropolis sweep, OpenMP-parallel.
    Returns acceptance ratio ∈ [0, 1].
    """
    cdef Py_ssize_t nmax = arr.shape[0]
    cdef Py_ssize_t i, j, j_start
    cdef double dE, dth
    cdef long acc_r = 0
    cdef long acc_b = 0
    cdef long local_acc, local_acc2

    cdef double[:, ::1] A  = arr
    cdef double[:, ::1] DR = dtheta_r
    cdef double[:, ::1] UR = urand_r
    cdef double[:, ::1] DB = dtheta_b
    cdef double[:, ::1] UB = urand_b

 # --- Red pass (i + j even) ---
    with nogil:
        for i in prange(nmax, schedule='static'):
            local_acc = 0
            j_start = 0 if (i % 2 == 0) else 1
            for j in range(j_start, nmax, 2):
                dth = DR[i, j]
                dE  = _local_delta_energy(A, i, j, dth, nmax)
                if dE <= 0.0 or exp(-dE / Ts) >= UR[i, j]:
                    A[i, j] += dth
                    local_acc += 1
            with cython.parallel.atomic():
                acc_r += local_acc

    # --- Black pass (i + j odd) ---
    with nogil:
        for i in prange(nmax, schedule='static'):
            local_acc2 = 0
            j_start = 1 if (i % 2 == 0) else 0
            for j in range(j_start, nmax, 2):
                dth = DB[i, j]
                dE  = _local_delta_energy(A, i, j, dth, nmax)
                if dE <= 0.0 or exp(-dE / Ts) >= UB[i, j]:
                    A[i, j] += dth
                    local_acc2 += 1
            with cython.parallel.atomic():
                acc_b += local_acc2
                
    return (acc_r + acc_b) / float(nmax * nmax)


def set_omp_threads(int n):
    """Set OpenMP thread count programmatically (no env var needed)."""
    if n > 0:
        openmp.omp_set_num_threads(n)
    return openmp.omp_get_max_threads()