"""
VECTORISED AND JIT COMPILED Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""
import sys
import time
import datetime
import numpy as np

# Only import matplotlib when needed; eg. if plotting is requested (saves start-up time)
def _lazy_import_matplotlib():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    return plt, mpl

# --- Vectorised helpers (stay in NumPy) --------------------------------
def initdat(nmax):
    return np.random.random_sample((nmax, nmax)) * 2.0 * np.pi

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
    return np.sum(en)

def get_order(arr):
    """Vectorised order parameter"""
    nmax = arr.shape[0]
    lab = np.stack((np.cos(arr), np.sin(arr), np.zeros_like(arr)), axis=0)
    # Contract over lattice indices i,j
    Q = 3.0 * np.einsum('aij,bij->ab', lab, lab)
    # Subtract trace(δ_ab) = 3 → same as 3*I
    Q -= 3.0 * np.eye(3)
    Q /= (2.0 * nmax * nmax)
    # Symmetric by construction; eigvalsh is stable & fast
    evals = np.linalg.eigvalsh(Q)
    return evals.max()

def plotdat(arr, pflag, nmax):
    if pflag == 0:
        return
    plt, mpl = _lazy_import_matplotlib()
    u = np.cos(arr); v = np.sin(arr)
    x = np.arange(nmax); y = np.arange(nmax)
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
    quiveropts = dict(headlength=0, pivot='middle', headwidth=1, scale=1.1 * nmax)
    fig, ax = plt.subplots()
    ax.quiver(x, y, u, v, cols, norm=norm, **quiveropts)
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

# --- Numba-accelerated --------------------------------
from numba import njit

@njit(fastmath=True)
def local_delta_energy(arr, ix, iy, dtheta, nmax):
    """
    Compute ΔE for rotating site (ix,iy) by dtheta,
    considering only its four neighbours (periodic BC).
    Same interaction: 0.5 * (1 - 3*cos(Δ)^2)
    """
    ixp = (ix + 1) % nmax
    ixm = (ix - 1 + nmax) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1 + nmax) % nmax

    t0 = arr[ix, iy]
    t1 = t0 + dtheta

    en = 0.0
    # neighbour 1 (ix+1, iy)
    nb = arr[ixp, iy]
    d0 = t0 - nb
    d1 = t1 - nb
    en += 0.5 * ((1.0 - 3.0 * np.cos(d1)**2) - (1.0 - 3.0 * np.cos(d0)**2))
    # neighbour 2 (ix-1, iy)
    nb = arr[ixm, iy]
    d0 = t0 - nb
    d1 = t1 - nb
    en += 0.5 * ((1.0 - 3.0 * np.cos(d1)**2) - (1.0 - 3.0 * np.cos(d0)**2))
    # neighbour 3 (ix, iy+1)
    nb = arr[ix, iyp]
    d0 = t0 - nb
    d1 = t1 - nb
    en += 0.5 * ((1.0 - 3.0 * np.cos(d1)**2) - (1.0 - 3.0 * np.cos(d0)**2))
    # neighbour 4 (ix, iy-1)
    nb = arr[ix, iym]
    d0 = t0 - nb
    d1 = t1 - nb
    en += 0.5 * ((1.0 - 3.0 * np.cos(d1)**2) - (1.0 - 3.0 * np.cos(d0)**2))

    return en

# fastmath can be turned off for stricter mathematics; in my testing this wasn't necessary
@njit(fastmath=True)  
def MC_step_numba(arr, Ts, nmax):
    """
    One NON-PARALLEL MC step (asynchronous / sequential).
    JIT removes Python-call overhead. RNGs are Numba-supported.
    """
    scale = 0.1 + Ts
    accept = 0
    total_sites = nmax * nmax

    for _ in range(total_sites):
        ix = np.random.randint(0, nmax)
        iy = np.random.randint(0, nmax)
        dtheta = np.random.normal(0.0, scale)

        dE = local_delta_energy(arr, ix, iy, dtheta, nmax)

        if dE <= 0.0:
            arr[ix, iy] += dtheta
            accept += 1
        else:
            #  test
            if np.exp(-dE / Ts) >= np.random.random():
                arr[ix, iy] += dtheta
                accept += 1
            # else: reject (do nothing)

    return accept / total_sites

# --- Driver ------------------------------------------------------------
def main(program, nsteps, nmax, temp, pflag):
    lattice = initdat(nmax)
    plotdat(lattice, pflag, nmax)

    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio  = np.zeros(nsteps + 1, dtype=np.float64)
    order  = np.zeros(nsteps + 1, dtype=np.float64)

    energy[0] = total_energy(lattice)
    ratio[0]  = 0.5
    order[0]  = get_order(lattice)

    t0 = time.time()
    # First call includes JIT compile time; subsequent steps are fast
    for it in range(1, nsteps + 1):
        ratio[it]  = MC_step_numba(lattice, temp, nmax)
        energy[it] = total_energy(lattice)
        order[it]  = get_order(lattice)
    runtime = time.time() - t0

    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:5.3f}: "
          f"Order: {order[-1]:5.3f}, Time: {runtime:8.6f} s")

    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)
    plotdat(lattice, pflag, nmax)

# --- CLI ---------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) == 5:
        PROGNAME    = sys.argv[0]
        ITERATIONS  = int(sys.argv[1])
        SIZE        = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG    = int(sys.argv[4])
        # Defer heavy imports until needed inside plotdat
        import datetime
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")