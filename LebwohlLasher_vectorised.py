"""
VECTORISED Python Lebwohl-Lasher code.  Based on the paper 
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
import matplotlib.pyplot as plt
import matplotlib as mpl


# ======================================================================
def initdat(nmax):
    """Initialise lattice with random orientations in [0, 2pi]."""
    return np.random.random_sample((nmax, nmax)) * 2.0 * np.pi


# ======================================================================
def total_energy(arr):
    """Vectorised total lattice energy with periodic boundaries."""
    ang_xp = arr - np.roll(arr, -1, axis=0)
    ang_xm = arr - np.roll(arr,  1, axis=0)
    ang_yp = arr - np.roll(arr, -1, axis=1)
    ang_ym = arr - np.roll(arr,  1, axis=1)

    en = 0.5 * ((1 - 3 * np.cos(ang_xp)**2) +
                (1 - 3 * np.cos(ang_xm)**2) +
                (1 - 3 * np.cos(ang_yp)**2) +
                (1 - 3 * np.cos(ang_ym)**2))
    return np.sum(en)


# ======================================================================
def get_order(arr):
    """Vectorised computation of the nematic (unordered) order parameter (corrected normalisation, see sect 5)."""
    nmax = arr.shape[0]

    # Construct local orientation vectors (3 * n * n)
    lab = np.stack((np.cos(arr), np.sin(arr), np.zeros_like(arr)), axis=0)

    # Perform tensor contraction over all lattice sites
    # Equivalent to: Σ_ij [3 l_a l_b - δ_ab]
    Qab = 3.0 * np.einsum('aij,bij->ab', lab, lab)

    # Subtract δ_ab once per site → N^2 * I
    Qab -= (nmax * nmax) * np.eye(3)

    # Divide by 2N^2 (as per original Lebwohl–Lasher definition)
    Qab /= (2.0 * nmax * nmax)

    # Largest eigenvalue is the scalar order parameter
    eigenvalues = np.linalg.eigvalsh(Qab)
    return float(eigenvalues.max())



# ======================================================================
def plotdat(arr, pflag, nmax):
    """Plot the lattice configuration using quiver arrows."""
    if pflag == 0:
        return

    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax, nmax))

    if pflag == 1:
        mpl.rc('image', cmap='rainbow')
        en = np.zeros_like(arr)
        ang_xp = arr - np.roll(arr, -1, axis=0)
        ang_xm = arr - np.roll(arr, 1, axis=0)
        ang_yp = arr - np.roll(arr, -1, axis=1)
        ang_ym = arr - np.roll(arr, 1, axis=1)
        en = 0.5 * ((1 - 3 * np.cos(ang_xp)**2) +
                    (1 - 3 * np.cos(ang_xm)**2) +
                    (1 - 3 * np.cos(ang_yp)**2) +
                    (1 - 3 * np.cos(ang_ym)**2))
        cols = en
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


# ======================================================================
def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
    """Save energy, order, and acceptance ratio per MCS to file."""
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


# ======================================================================
def one_energy(arr, ix, iy, nmax):
    """Compute local energy (unchanged, used for MC updates)."""
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax
    en = 0.0
    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1 - 3 * np.cos(ang)**2)
    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1 - 3 * np.cos(ang)**2)
    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1 - 3 * np.cos(ang)**2)
    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1 - 3 * np.cos(ang)**2)
    return en


# ======================================================================
def MC_step(arr, Ts, nmax):
    """Perform one Monte Carlo step (partially vectorised)."""
    scale = 0.1 + Ts
    xran = np.random.randint(0, nmax, size=(nmax, nmax))
    yran = np.random.randint(0, nmax, size=(nmax, nmax))
    aran = np.random.normal(scale=scale, size=(nmax, nmax))

    accept = 0
    for ix, iy, ang in zip(xran.flat, yran.flat, aran.flat):
        en0 = one_energy(arr, ix, iy, nmax)
        arr[ix, iy] += ang
        en1 = one_energy(arr, ix, iy, nmax)
        dE = en1 - en0
        if dE <= 0 or np.exp(-dE / Ts) >= np.random.rand():
            accept += 1
        else:
            arr[ix, iy] -= ang
    return accept / (nmax * nmax)


# ======================================================================
def main(program, nsteps, nmax, temp, pflag):
    """Main simulation loop."""
    lattice = initdat(nmax)
    plotdat(lattice, pflag, nmax)

    energy = np.zeros(nsteps + 1)
    ratio = np.zeros(nsteps + 1)
    order = np.zeros(nsteps + 1)

    energy[0] = total_energy(lattice)
    ratio[0] = 0.5
    order[0] = get_order(lattice)

    t0 = time.time()
    for it in range(1, nsteps + 1):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = total_energy(lattice)
        order[it] = get_order(lattice)
    runtime = time.time() - t0

    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:5.3f}: "
          f"Order: {order[-1]:5.3f}, Time: {runtime:8.6f} s")

    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)
    plotdat(lattice, pflag, nmax)


# ======================================================================
if __name__ == '__main__':
    if len(sys.argv) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")