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

# Load and clean
path = 'DATA/Runtimes.csv'
df = pd.read_csv(path)

# Build two clean tables
df_L = df.iloc[0:6].copy()
df_N = df.iloc[8:14].copy()

# Ensure numeric
for sub in (df_L, df_N):
    for col in ['L=50', 'L=100', 'L=256', 'L=512']:
        sub[col] = pd.to_numeric(sub[col], errors='coerce')

# Set index
df_L = df_L.set_index('Script Name')
df_N = df_N.set_index('Script Name')

#display_dataframe_to_user("Runtime vs L (numbers used)", df_L.reset_index())
#isplay_dataframe_to_user("Runtime vs N (numbers used)", df_N.reset_index())

# Sanity check: print Simple Python and Cython rows used for L plot
print("L-scaling — Simple Python:", df_L.loc['Simple Python', ['L=50','L=100','L=256','L=512']].to_list())
print("L-scaling — Cython (2 Th.):", df_L.loc['Cython (2 Th.)', ['L=50','L=100','L=256','L=512']].to_list())

# Plot 1: Runtime vs L (N fixed)
L_vals = [50, 100, 256, 512]
plt.figure(figsize=(10,6))
for script in df_L.index:
    y = df_L.loc[script, ['L=50','L=100','L=256','L=512']].values.astype(float)
    plt.plot(L_vals, y, marker='o', label=script)
plt.yscale('log')
plt.xlabel('L')
plt.ylabel('Runtime (s)')
#plt.title('Runtime vs L (N fixed)')
plt.grid(True, which='both')
plt.legend()
plt.show()

# Plot 2: Runtime vs N (L fixed)
N_vals = [50, 100, 256, 512]
plt.figure(figsize=(10,6))
for script in df_N.index:
    y = df_N.loc[script, ['L=50','L=100','L=256','L=512']].values.astype(float)
    plt.plot(N_vals, y, marker='o', label=script)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Runtime (s)')
#plt.title('Runtime vs N (L fixed)')
plt.grid(True, which='both')
plt.legend()
plt.show()