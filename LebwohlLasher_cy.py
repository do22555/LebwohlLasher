"""
Cython/OpenMP-parallelised Lebwohl–Lasher simulation using LebwohlLasher_cy_kernel.pyx.

Run:
  python LebwohlLasher_cy.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
"""

import sys
import time
import datetime
import numpy as np

from LebwohlLasher_cy_kernel import mc_step_checkerboard, set_omp_threads  # compiled Cython module


# ========================= NumPy helpers =========================
def initdat(nmax):
    return np.random.random_sample((nmax, nmax)).astype(np.float64) * 2.0 * np.pi

def total_energy(arr):
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
    nmax = arr.shape[0]
    lab = np.stack((np.cos(arr), np.sin(arr), np.zeros_like(arr)), axis=0)
    Q = 3.0 * np.einsum('aij,bij->ab', lab, lab)
    Q -= (nmax * nmax) * np.eye(3)
    Q /= (2.0 * nmax * nmax)
    return float(np.linalg.eigvalsh(Q).max())

def plotdat(arr, pflag, nmax):
    if pflag == 0:
        return
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    u, v = np.cos(arr), np.sin(arr)
    x, y = np.arange(nmax), np.arange(nmax)
    if pflag == 1:
        mpl.rc('image', cmap='rainbow')
        ang_xp = arr - np.roll(arr, -1, axis=0)
        ang_xm = arr - np.roll(arr, 1, axis=0)
        ang_yp = arr - np.roll(arr, -1, axis=1)
        ang_ym = arr - np.roll(arr, 1, axis=1)
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
    ax.quiver(x, y, u, v, cols, norm=norm, headlength=0, pivot='middle', headwidth=1, scale=1.1 * nmax)
    ax.set_aspect('equal')
    plt.show()

def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = f"DATA/LL-Output-{current_datetime}.txt"
    with open(filename, "w") as f:
        print("#=====================================================", file=f)
        print(f"# File created:        {current_datetime}", file=f)
        print(f"# Lattice size:        {nmax}×{nmax}", file=f)
        print(f"# Monte Carlo steps:   {nsteps}", file=f)
        print(f"# Reduced temperature: {Ts:5.3f}", file=f)
        print(f"# Runtime (s):         {runtime:8.6f}", file=f)
        print("#=====================================================", file=f)
        print("# Step | Ratio | Energy | Order", file=f)
        print("#=====================================================", file=f)
        for i in range(nsteps + 1):
            print(f"{i:05d}  {ratio[i]:6.4f}  {energy[i]:12.4f}  {order[i]:6.4f}", file=f)


# ============================ MC wrapper ============================
def mc_step_cython(arr: np.ndarray, Ts: float) -> float:
    """Generate proposals in NumPy, then call the OpenMP kernel."""
    nmax = arr.shape[0]
    scale = 0.1 + Ts
    dtheta_r = np.random.normal(0.0, scale, (nmax, nmax)).astype(np.float64, copy=False)
    urand_r  = np.random.random((nmax, nmax)).astype(np.float64, copy=False)
    dtheta_b = np.random.normal(0.0, scale, (nmax, nmax)).astype(np.float64, copy=False)
    urand_b  = np.random.random((nmax, nmax)).astype(np.float64, copy=False)
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return mc_step_checkerboard(arr, float(Ts), dtheta_r, urand_r, dtheta_b, urand_b)


# ============================== Driver ==============================
def main(program, nsteps, nmax, temp, pflag):
    # Set OpenMP thread count programmatically here:
    threads = 10  # change to your core count
    used = set_omp_threads(threads)
    print(f"[OpenMP] requested {threads} threads, runtime reports max = {used}")

    lattice = np.ascontiguousarray(initdat(nmax))
    plotdat(lattice, pflag, nmax)

    energy = np.zeros(nsteps + 1)
    ratio  = np.zeros(nsteps + 1)
    order  = np.zeros(nsteps + 1)

    energy[0] = total_energy(lattice)
    ratio[0]  = 0.5
    order[0]  = get_order(lattice)

    t0 = time.time()
    for it in range(1, nsteps + 1):
        ratio[it]  = mc_step_cython(lattice, temp)
        energy[it] = total_energy(lattice)
        order[it]  = get_order(lattice)
    runtime = time.time() - t0

    print(f"{program}: Size={nmax}, Steps={nsteps}, T*={temp:5.3f}, "
          f"Order={order[-1]:5.3f}, Time={runtime:8.6f}s")

    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)
    plotdat(lattice, pflag, nmax)


# ================================ CLI ===============================
if len(sys.argv) == 5:
    PROGNAME    = sys.argv[0]
    ITERATIONS  = int(sys.argv[1])
    SIZE        = int(sys.argv[2])
    TEMPERATURE = float(sys.argv[3])
    PLOTFLAG    = int(sys.argv[4])
    main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
else:
    print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
