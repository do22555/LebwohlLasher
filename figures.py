import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# File paths
files = [
    "DATA/LL-Output-Fri-31-Oct-2025-at-03-20-29PM.txt",
    "DATA/LL-Output-Fri-31-Oct-2025-at-03-20-49PM.txt",
    "DATA/LL-Output-Fri-31-Oct-2025-at-03-21-05PM.txt",
    "DATA/LL-Output-Fri-31-Oct-2025-at-03-21-16PM.txt",
    "DATA/LL-Output-Fri-31-Oct-2025-at-03-21-42PM.txt",
    "DATA/LL-Output-Fri-31-Oct-2025-at-03-30-28PM.txt",
]


labels = ["Raw", "Vectorised", "Numba", "Threaded Numba", "Cython", "MPI"]
colors = plt.cm.tab10.colors


plt.figure(figsize=(10, 6))
for file, label, col in zip(files, labels, colors):
    df = pd.read_csv(file, comment="#", delim_whitespace=True, names=["MC_step", "Ratio", "Energy", "Order"])
    plt.plot(df["MC_step"], df["Energy"], label=label, color=col)
plt.xlabel("Monte Carlo Step")
plt.ylabel("Total Energy")
plt.legend()
plt.grid(True)
plt.show()

# Load and plot order parameter vs MC step (overlay all)
plt.figure(figsize=(10, 6))
for file, label, col in zip(files, labels, colors):
    df = pd.read_csv(file, comment="#", delim_whitespace=True, names=["MC_step", "Ratio", "Energy", "Order"])
    plt.plot(df["MC_step"], df["Order"], label=label, color=col)
plt.xlabel("Monte Carlo Step")
plt.ylabel("Order Parameter (S)")
plt.legend()
plt.grid(True)
plt.show()

# Data extracted from the image
implementations = ["Raw", "Vectorised", "Numba", "Threaded Numba", "Cython", "MPI"]
runtime_50 = [2.999688, 0.99825, 2.609752, 2.024596, 0.038707, 0.7]
runtime_512 = [3010.9, 1079.5, 140, 20.231, 13.461785, 2000]

# Convert to numpy arrays for log plotting
runtime_50 = np.array(runtime_50)
runtime_512 = np.array(runtime_512)
mc_steps = [50, 512]

plt.figure(figsize=(10, 6))

# Plot each implementation
for i, label in enumerate(implementations):
    plt.plot(mc_steps, [runtime_50[i], runtime_512[i]], marker='o', label=label)

# Log scale
plt.yscale('log')
plt.xlabel("Scale (N and L)")
plt.ylabel("Runtime (s)")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.show()