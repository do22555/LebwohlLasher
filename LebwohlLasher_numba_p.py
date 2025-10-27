"""
VECTORISED, JIT-COMPILED, AND PARALLEL Python Lebwohl–Lasher code.
Based on P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426–429 (1972).
"""

import sys
import time
import datetime
import numpy as np

# --- Numba imports for compilation and threading ---
from numba import njit, prange, set_num_threads, get_num_threads

# --- Lazy matplotlib import ---
def _lazy_import_matplotlib():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    return plt, mpl

# ============================================================
#  NumPy vectorised helpers
# ============================================================
def initdat(nmax):
    # float64 by default; JIT operates fastest on C-contiguous arrays
    return np.random.random_sample((nmax, nmax)).astype(np.float64) * 2.0 * np.pi

def total_energy(arr):
    """Vectorised total energy with periodic boundaries."""
    ang_xp = arr - np.roll(arr, -1, axis=0)
    ang_xm = arr - np.roll(arr,  1, axis=0)
    ang_yp = arr - np.roll(arr, -1, axis=1)
    ang_ym = arr - np.roll(arr,  1, axis=1)
    en = 0.5 * ((1 - 3 * np.cos(ang_xp)**2) +
                (1 - 3 * np.cos(ang_xm)**2) +
                (1 - 3 * np.cos(ang_yp)**2) +
                (1 - 3 * np.cos(ang_ym)**2))
    return float(np.sum(en))

def get_order(arr):
    """Vectorised nematic order parameter (correct normalisation: subtract N^2 * I, then / (2 N^2))."""
    nmax = arr.shape[0]
    lab = np.stack((np.cos(arr), np.sin(arr), np.zeros_like(arr)), axis=0)  # shape: (3, n, n)
    Q = 3.0 * np.einsum('aij,bij->ab', lab, lab)          # 3 * Σ l_a l_b over lattice
    Q -= (nmax * nmax) * np.eye(3)                        # subtract δ_ab once per site -> N^2 * I
    Q /= (2.0 * nmax * nmax)                              # divide by 2 N^2
    evals = np.linalg.eigvalsh(Q)
    return float(evals.max())

def plotdat(arr, pflag, nmax):
    if pflag == 0:
        return
    plt, mpl = _lazy_import_matplotlib()
    u, v = np.cos(arr), np.sin(arr)
    x, y = np.arange(nmax), np.arange(nmax)
    if pflag == 1:
        mpl.rc('image', cmap='rainbow')
        ang_xp = arr - np.roll(arr, -1, axis=0)
        ang_xm = arr - np.roll(arr,  1, axis=0)
        ang_yp = arr - np.roll(arr, -1, axis=1)
        ang_ym = arr - np.roll(arr,  1, axis=1)
        cols = 0.5 * ((1 - 3 * np.cos(ang_xp)**2) +
                      (1 - 3 * np.cos(ang_xm)**2) +
                      (1 - 3 * np.cos(ang_yp)**2) +
                      (1 - 3 * np.cos(ang_ym)**2))
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag == 2:
        mpl.rc('image', cmap='hsv')
        cols = arr % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)
    fig, ax = plt.subplots()
    ax.quiver(x, y, u, v, cols, norm=norm,
              headlength=0, pivot='middle', headwidth=1, scale=1.1 * nmax)
    ax.set_aspect('equal')
    plt.show()

def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = f"DATA/LL-Output-{current_datetime}.txt"
    with open(filename, "w") as f:
        print("#=====================================================", file=f)
        print(f"# File created:        {current_datetime}", file=f)
        print(f"# Size of lattice:     {nmax}x{nmax}", file=f)
        print(f"# Number of MC steps:  {nsteps}", file=f)
        print(f"# Reduced temperature: {Ts:5.3f}", file=f)
        print(f"# Run time (s):        {runtime:8.6f}", file=f)
        print("#=====================================================", file=f)
        print("# MC step:  Ratio:     Energy:   Order:", file=f)
        print("#=====================================================", file=f)
        for i in range(nsteps + 1):
            print(f"   {i:05d}    {ratio[i]:6.4f} {energy[i]:12.4f}  {order[i]:6.4f}", file=f)

# ============================================================
#   JIT + PARALLEL SECTION
# ============================================================
@njit(fastmath=True, inline='always')
def _local_delta_energy(arr, ix, iy, dtheta, nmax):
    """Compute ΔE for a proposed rotation at (ix, iy) with periodic BCs."""
    ixp = (ix + 1) % nmax
    ixm = (ix - 1 + nmax) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1 + nmax) % nmax

    t0 = arr[ix, iy]
    t1 = t0 + dtheta

    en = 0.0

    # neighbour (ix+1, iy)
    nb = arr[ixp, iy]
    d0 = t0 - nb; d1 = t1 - nb
    en += 0.5 * ((1.0 - 3.0 * np.cos(d1)**2) - (1.0 - 3.0 * np.cos(d0)**2))

    # neighbour (ix-1, iy)
    nb = arr[ixm, iy]
    d0 = t0 - nb; d1 = t1 - nb
    en += 0.5 * ((1.0 - 3.0 * np.cos(d1)**2) - (1.0 - 3.0 * np.cos(d0)**2))

    # neighbour (ix, iy+1)
    nb = arr[ix, iyp]
    d0 = t0 - nb; d1 = t1 - nb
    en += 0.5 * ((1.0 - 3.0 * np.cos(d1)**2) - (1.0 - 3.0 * np.cos(d0)**2))

    # neighbour (ix, iy-1)
    nb = arr[ix, iym]
    d0 = t0 - nb; d1 = t1 - nb
    en += 0.5 * ((1.0 - 3.0 * np.cos(d1)**2) - (1.0 - 3.0 * np.cos(d0)**2))

    return en

@njit(parallel=True, fastmath=True)
def MC_step_checkerboard_parallel(arr, Ts):
    """
    Checkerboard-parallelised Metropolis sweep.
    Red sites: (i + j) even. Black sites: (i + j) odd.
    Thread-safe via per-row accumulators.
    """
    nmax = arr.shape[0]
    scale = 0.1 + Ts
    thread_acc = np.zeros(nmax, dtype=np.int64)  # per-row accumulator

    # --- Red sites (i+j even) ---
    dtheta_r = np.random.normal(0.0, scale, (nmax, nmax))
    urand_r  = np.random.random((nmax, nmax))
    for i in prange(nmax):
        local_acc = 0
        j_start = 0 if (i % 2 == 0) else 1
        for j in range(j_start, nmax, 2):
            dth = dtheta_r[i, j]
            dE  = _local_delta_energy(arr, i, j, dth, nmax)
            if dE <= 0.0 or np.exp(-dE / Ts) >= urand_r[i, j]:
                arr[i, j] += dth
                local_acc += 1
        thread_acc[i] = local_acc
    acc_r = np.sum(thread_acc)

    # --- Black sites (i+j odd) ---
    dtheta_b = np.random.normal(0.0, scale, (nmax, nmax))
    urand_b  = np.random.random((nmax, nmax))
    thread_acc.fill(0)
    for i in prange(nmax):
        local_acc = 0
        j_start = 1 if (i % 2 == 0) else 0
        for j in range(j_start, nmax, 2):
            dth = dtheta_b[i, j]
            dE  = _local_delta_energy(arr, i, j, dth, nmax)
            if dE <= 0.0 or np.exp(-dE / Ts) >= urand_b[i, j]:
                arr[i, j] += dth
                local_acc += 1
        thread_acc[i] = local_acc
    acc_b = np.sum(thread_acc)

    return (acc_r + acc_b) / (nmax * nmax)

# ============================================================
#   Main driver
# ============================================================
def main(program, nsteps, nmax, temp, pflag):
    set_num_threads(10)  # use 10 threads (match your machine)
    print(f"[Numba] using {get_num_threads()} threads")

    # Ensure C-contiguous for JIT kernels
    lattice = np.ascontiguousarray(initdat(nmax))
    plotdat(lattice, pflag, nmax)

    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio  = np.zeros(nsteps + 1, dtype=np.float64)
    order  = np.zeros(nsteps + 1, dtype=np.float64)

    energy[0] = total_energy(lattice)
    ratio[0]  = 0.5
    order[0]  = get_order(lattice)

    # Compile once (excluded from runtime)
    t_compile_start = time.time()
    MC_step_checkerboard_parallel(lattice, temp)
    t_compile = time.time() - t_compile_start
    print(f"Compilation complete in {t_compile:.3f} s")

    # Simulation timing
    t0 = time.time()
    for it in range(1, nsteps + 1):
        ratio[it]  = MC_step_checkerboard_parallel(lattice, temp)
        energy[it] = total_energy(lattice)
        order[it]  = get_order(lattice)
    runtime = time.time() - t0

    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:5.3f}: "
          f"Order: {order[-1]:5.3f}, Time: {runtime:8.6f} s")

    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)
    plotdat(lattice, pflag, nmax)

# ============================================================
#   CLI
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) == 5:
        PROGNAME    = sys.argv[0]
        ITERATIONS  = int(sys.argv[1])
        SIZE        = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG    = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
