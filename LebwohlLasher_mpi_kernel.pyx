# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import cython
import numpy as np
cimport numpy as np
from libc.math cimport cos, exp

DTYPE = np.float64
ctypedef np.float64_t f64

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _P2_from_cos(double c) nogil:
    return 0.5 * (3.0 * c * c - 1.0)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _local_energy(double[:, :] arr, int i, int j, int nmax) nogil:
    cdef int ip = i + 1
    cdef int im = i - 1
    cdef int jp = (j + 1) % nmax
    cdef int jm = (j - 1) % nmax
    cdef double t = arr[i, j]
    cdef double d, e = 0.0
    d = t - arr[ip, j]; e += 0.5 * (1.0 - 3.0 * cos(d) * cos(d))
    d = t - arr[im, j]; e += 0.5 * (1.0 - 3.0 * cos(d) * cos(d))
    d = t - arr[i, jp]; e += 0.5 * (1.0 - 3.0 * cos(d) * cos(d))
    d = t - arr[i, jm]; e += 0.5 * (1.0 - 3.0 * cos(d) * cos(d))
    return e

@cython.boundscheck(False)
@cython.wraparound(False)
def mc_step_local(
    np.ndarray[f64, ndim=2] arr,
    np.ndarray[f64, ndim=2] dtheta_r,
    np.ndarray[f64, ndim=2] urand_r,
    np.ndarray[f64, ndim=2] dtheta_b,
    np.ndarray[f64, ndim=2] urand_b,
    double Ts,
    int nmax,
    int i0
):
    """
    One local Monte Carlo step (chequerboard).
    arr includes halos in X (shape (local_nx+2, nmax)).
    """
    cdef int local_nx = arr.shape[0] - 2
    cdef int i, j, gi
    cdef double en0, en1, ang
    cdef long accepts = 0

    # Red sites
    with cython.nogil:
        for i in range(1, local_nx + 1):
            gi = i0 + (i - 1)
            for j in range(nmax):
                if ((gi + j) & 1) == 0:
                    en0 = _local_energy(arr, i, j, nmax)
                    ang = dtheta_r[i - 1, j]
                    arr[i, j] += ang
                    en1 = _local_energy(arr, i, j, nmax)
                    if en1 <= en0:
                        accepts += 1
                    elif exp(-(en1 - en0) / Ts) >= urand_r[i - 1, j]:
                        accepts += 1
                    else:
                        arr[i, j] -= ang

    # Black sites
    with cython.nogil:
        for i in range(1, local_nx + 1):
            gi = i0 + (i - 1)
            for j in range(nmax):
                if ((gi + j) & 1) == 1:
                    en0 = _local_energy(arr, i, j, nmax)
                    ang = dtheta_b[i - 1, j]
                    arr[i, j] += ang
                    en1 = _local_energy(arr, i, j, nmax)
                    if en1 <= en0:
                        accepts += 1
                    elif exp(-(en1 - en0) / Ts) >= urand_b[i - 1, j]:
                        accepts += 1
                    else:
                        arr[i, j] -= ang

    return accepts
