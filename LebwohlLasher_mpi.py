#!/usr/bin/env python3
"""
Lebwohl–Lasher (2D) with MPI domain decomposition by columns.

Run:
    mpiexec -n <PROCESSES> python LebwohlLasher_mpi.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

This preserves the physics of the serial code:
- Global red–black (chequerboard) updates.
- Halo exchanges between ranks after each colour sweep.
- Periodic boundaries in both directions.
- Global reductions for energy, order, and acceptance ratio.

"""

import sys
import time
import datetime
import numpy as np
from mpi4py import MPI
from mpi4py import MPI
from LebwohlLasher_mpi_kernel import mc_step_local

# ---------- Optional plotting (rank 0 only, lazy import) ----------
def _lazy_import_matplotlib():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    return plt, mpl

# ================================================================
# MPI helpers
# ================================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def decompose_columns(nmax, size, rank):
    """Return (local_nx, i0) where local_nx columns starting at global column i0 belong to this rank."""
    base = nmax // size
    rem  = nmax % size  # first 'rem' ranks get one extra
    if rank < rem:
        local_nx = base + 1
        i0 = rank * (base + 1)
    else:
        local_nx = base
        i0 = rem * (base + 1) + (rank - rem) * base
    return local_nx, i0

def halo_exchange_x(field):
    """
    Exchange left/right halo columns for 'field' of shape (local_nx+2, nmax).
    field[1:-1, :] are owned columns; field[0, :] is left halo; field[-1, :] is right halo.
    Periodic neighbours: left_rank = (rank-1)%size, right_rank = (rank+1)%size.
    """
    left_rank  = (rank - 1) % size
    right_rank = (rank + 1) % size

    # Non-blocking to avoid deadlocks
    reqs = []
    # Send left interior (col 1) to left neighbour's right halo; receive right halo from right neighbour
    reqs.append(comm.Irecv(field[-1, :], source=right_rank, tag=10))
    reqs.append(comm.Isend(field[1, :].copy(), dest=left_rank, tag=10))

    # Send right interior (col -2) to right neighbour's left halo; receive left halo from left neighbour
    reqs.append(comm.Irecv(field[0, :], source=left_rank, tag=11))
    reqs.append(comm.Isend(field[-2, :].copy(), dest=right_rank, tag=11))

    MPI.Request.Waitall(reqs)

# ================================================================
# Physics helpers (identical formulas to your serial code)
# ================================================================
def one_energy_local(arr_with_halo, il, j, nmax):
    """
    Energy of local cell (il,j) where il is local interior index (1..local_nx),
    j is 0..nmax-1. Uses halo in X and periodic in Y.
    """
    # x neighbours via halos
    ixp = il + 1
    ixm = il - 1
    # y periodic neighbours (within row)
    jp = (j + 1) % nmax
    jm = (j - 1) % nmax

    t = arr_with_halo[il, j]
    en = 0.0
    ang = t - arr_with_halo[ixp, j]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang)**2)
    ang = t - arr_with_halo[ixm, j]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang)**2)
    ang = t - arr_with_halo[il, jp]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang)**2)
    ang = t - arr_with_halo[il, jm]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang)**2)
    return en

def all_energy_global(arr_with_halo, nmax):
    """Global energy via local sum + Allreduce."""
    local_nx = arr_with_halo.shape[0] - 2
    # Ensure halos are up to date before computing energy
    halo_exchange_x(arr_with_halo)
    en_local = 0.0
    for il in range(1, local_nx + 1):
        for j in range(nmax):
            en_local += one_energy_local(arr_with_halo, il, j, nmax)
    en_global = comm.allreduce(en_local, op=MPI.SUM)
    return en_global

def get_order_global(arr_with_halo, nmax):
    """
    Nematic order parameter S via 3x3 Q tensor:
      Q_ab = (1/(2N^2)) * sum_{sites} (3 l_a l_b - delta_ab)
    with l = (cos θ, sin θ, 0).
    We accumulate sxx, sxy, syy locally, then reduce.
    """
    local_nx = arr_with_halo.shape[0] - 2
    # halos not strictly needed here unless you use neighbours; we don't.
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for il in range(1, local_nx + 1):
        th = arr_with_halo[il, :]
        lx = np.cos(th)
        ly = np.sin(th)
        sxx += np.sum(lx * lx)
        sxy += np.sum(lx * ly)
        syy += np.sum(ly * ly)

    Sxx = comm.allreduce(sxx, op=MPI.SUM)
    Sxy = comm.allreduce(sxy, op=MPI.SUM)
    Syy = comm.allreduce(syy, op=MPI.SUM)

    N2 = float(nmax * nmax)
    Q = np.zeros((3, 3), dtype=np.float64)
    Q[0, 0] = 3.0 * Sxx - N2
    Q[0, 1] = 3.0 * Sxy
    Q[1, 0] = 3.0 * Sxy
    Q[1, 1] = 3.0 * Syy - N2
    Q[2, 2] = -N2
    Q /= (2.0 * N2)

    evals = np.linalg.eigvalsh(Q)
    return float(evals.max())

# ================================================================
# MPI-aware initialisation, plotting, saving
# ================================================================
def init_local(nmax, seed_base=12345):
    """Initialise local block with random angles in [0, 2π). Add halos (uninitialised until first exchange)."""
    local_nx, i0 = decompose_columns(nmax, size, rank)
    rng = np.random.default_rng(seed_base + rank)
    core = rng.random((local_nx, nmax)) * 2.0 * np.pi
    # add 2 halo columns
    arr = np.zeros((local_nx + 2, nmax), dtype=np.float64)
    arr[1:-1, :] = core
    # fill halos from neighbours
    halo_exchange_x(arr)
    return arr, local_nx, i0

def plotdat_global(arr_with_halo, nmax, pflag, i0):
    """Gather full lattice to rank 0 to plot. Avoid for large n."""
    if pflag == 0:
        return
    local_core = arr_with_halo[1:-1, :]
    # gather sizes and displacements
    counts = comm.gather(local_core.shape[0], root=0)
    if rank == 0:
        recv = np.empty((nmax, nmax), dtype=np.float64)  # will fill by columns
        displs = np.cumsum([0] + counts[:-1])
    else:
        recv = None
        displs = None
    # Gather columns as a list to rank 0
    gathered = comm.gather(local_core, root=0)
    if rank == 0:
        col = 0
        for blk in gathered:
            w = blk.shape[0]
            recv[col:col+w, :] = blk
            col += w
        # transpose to match your original plotting (x=0..n-1, y=0..n-1 with quiver using arr[i,j])
        arr_full = recv
        # --- plotting ---
        plt, mpl = _lazy_import_matplotlib()
        u, v = np.cos(arr_full), np.sin(arr_full)
        x = np.arange(nmax)
        y = np.arange(nmax)
        if pflag == 1:
            mpl.rc('image', cmap='rainbow')
            # local energy per site (expensive for big n)
            cols = np.zeros((nmax, nmax))
            # quick vectorised stencil using rolls on full array (periodic)
            ang_xp = arr_full - np.roll(arr_full, -1, axis=0)
            ang_xm = arr_full - np.roll(arr_full,  1, axis=0)
            ang_yp = arr_full - np.roll(arr_full, -1, axis=1)
            ang_ym = arr_full - np.roll(arr_full,  1, axis=1)
            cols = 0.5 * ((1 - 3 * np.cos(ang_xp)**2) +
                          (1 - 3 * np.cos(ang_xm)**2) +
                          (1 - 3 * np.cos(ang_yp)**2) +
                          (1 - 3 * np.cos(ang_ym)**2))
            norm = plt.Normalize(cols.min(), cols.max())
        elif pflag == 2:
            mpl.rc('image', cmap='hsv')
            cols = arr_full % np.pi
            norm = plt.Normalize(vmin=0, vmax=np.pi)
        else:
            mpl.rc('image', cmap='gist_gray')
            cols = np.zeros_like(arr_full)
            norm = plt.Normalize(vmin=0, vmax=1)
        fig, ax = plt.subplots()
        ax.quiver(x, y, u, v, cols, norm=norm,
                  headlength=0, pivot='middle', headwidth=1, scale=1.1 * nmax)
        ax.set_aspect('equal')
        plt.show()

def savedat_rank0(nsteps, Ts, runtime, ratio, energy, order, nmax):
    if rank != 0:
        return
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = f"DATA/LL-Output-{current_datetime}.txt"
    # make sure DATA exists
    import os
    os.makedirs("DATA", exist_ok=True)
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
            print(f"   {i:05d}    {ratio[i]:6.4f} {energy[i]:12.4f}  {order[i]:6.4f} ", file=f)

# ================================================================
# Monte Carlo step (global chequerboard with halo exchanges)
# ================================================================
def mc_step_checkerboard(arr_with_halo, Ts, nmax, i0):
    """
    One global MCS:
      1) Update all red sites (global parity) using current halos.
      2) Exchange halos.
      3) Update all black sites.
      4) Exchange halos.
    Returns: global acceptance ratio.
    """
    local_nx = arr_with_halo.shape[0] - 2
    scale = 0.1 + Ts

    # Proposals (per local interior site)
    rng = np.random.default_rng(0xA5A5_0000 + rank)  # fixed per step is fine; change if you want different each step
    dtheta_r = rng.normal(loc=0.0, scale=scale, size=(local_nx, nmax))
    urand_r  = rng.random(size=(local_nx, nmax))
    dtheta_b = rng.normal(loc=0.0, scale=scale, size=(local_nx, nmax))
    urand_b  = rng.random(size=(local_nx, nmax))

    accepts_local = 0

    # ---- RED ----
    halo_exchange_x(arr_with_halo)  # ensure neighbours are current before red
    for il in range(1, local_nx + 1):
        gi = i0 + (il - 1)  # global column index
        for j in range(nmax):
            if ((gi + j) & 1) == 0:  # red
                en0 = one_energy_local(arr_with_halo, il, j, nmax)
                ang = dtheta_r[il - 1, j]
                arr_with_halo[il, j] += ang
                en1 = one_energy_local(arr_with_halo, il, j, nmax)
                if en1 <= en0:
                    accepts_local += 1
                else:
                    if np.exp(-(en1 - en0) / Ts) >= urand_r[il - 1, j]:
                        accepts_local += 1
                    else:
                        arr_with_halo[il, j] -= ang
    halo_exchange_x(arr_with_halo)

    # ---- BLACK ----
    for il in range(1, local_nx + 1):
        gi = i0 + (il - 1)
        for j in range(nmax):
            if ((gi + j) & 1) == 1:  # black
                en0 = one_energy_local(arr_with_halo, il, j, nmax)
                ang = dtheta_b[il - 1, j]
                arr_with_halo[il, j] += ang
                en1 = one_energy_local(arr_with_halo, il, j, nmax)
                if en1 <= en0:
                    accepts_local += 1
                else:
                    if np.exp(-(en1 - en0) / Ts) >= urand_b[il - 1, j]:
                        accepts_local += 1
                    else:
                        arr_with_halo[il, j] -= ang
    halo_exchange_x(arr_with_halo)

    # Global acceptance ratio
    accepts_global = comm.allreduce(accepts_local, op=MPI.SUM)
    return accepts_global / float(nmax * nmax)

# ================================================================
# Main
# ================================================================
def main(program, nsteps, nmax, temp, pflag):
    arr, local_nx, i0 = init_local(nmax)
    # Initial diagnostics (global)
    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio  = np.zeros(nsteps + 1, dtype=np.float64)
    order  = np.zeros(nsteps + 1, dtype=np.float64)

    energy[0] = all_energy_global(arr, nmax)
    ratio[0]  = 0.5
    order[0]  = get_order_global(arr, nmax)

    # Warm-up halo (already done) and RNG warm call
    _ = mc_step_checkerboard(arr, temp, nmax, i0)

    t0 = time.time()
    for it in range(1, nsteps + 1):
        ratio[it]  = mc_step_checkerboard(arr, temp, nmax, i0)
        energy[it] = all_energy_global(arr, nmax)
        order[it]  = get_order_global(arr, nmax)
    runtime = time.time() - t0

    if rank == 0:
        print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:5.3f}: "
              f"Order: {order[-1]:5.3f}, Time: {runtime:8.6f} s")

    savedat_rank0(nsteps, temp, runtime, ratio, energy, order, nmax)
    if rank == 0:
        plotdat_global(arr, nmax, pflag, i0)

# ================================================================
if __name__ == '__main__':
    if len(sys.argv) == 5:
        PROGNAME    = sys.argv[0]
        ITERATIONS  = int(sys.argv[1])
        SIZE        = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG    = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        if rank == 0:
            print(f"Usage: mpiexec -n <PROCS> python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
