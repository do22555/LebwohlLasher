
import os
os.environ["OMP_NUM_THREADS"] = "12"
import sys
import time
import datetime
import numpy as np

### This whole thing is just a wrapper. Logic is the same as prev scripts.
from LebwohlLasher_cy_kernel import (
    mc_step_checkerboard as cy_mc_step,
    total_energy as cy_total_energy,
    get_order as cy_get_order,
)

def _lazy_import_matplotlib():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    return plt, mpl

def initdat(nmax):
    return (np.random.random_sample((nmax, nmax)).astype(np.float64) * 2.0 * np.pi)

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
    import os
    os.makedirs("DATA", exist_ok=True)
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

def main(program, nsteps, nmax, temp, pflag):
    lattice = np.ascontiguousarray(initdat(nmax))
    plotdat(lattice, pflag, nmax)

    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio  = np.zeros(nsteps + 1, dtype=np.float64)
    order  = np.zeros(nsteps + 1, dtype=np.float64)

    energy[0] = cy_total_energy(lattice)
    ratio[0]  = 0.5
    order[0]  = cy_get_order(lattice)

    _ = cy_mc_step(lattice, temp)

    t0 = time.time()
    for it in range(1, nsteps + 1):
        ratio[it]  = cy_mc_step(lattice, temp)
        energy[it] = cy_total_energy(lattice)
        order[it]  = cy_get_order(lattice)
    runtime = time.time() - t0

    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:5.3f}: "
          f"Order: {order[-1]:5.3f}, Time: {runtime:8.6f} s")

    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)
    plotdat(lattice, pflag, nmax)

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
