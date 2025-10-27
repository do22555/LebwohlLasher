import pandas as pd
import matplotlib.pyplot as plt
import re

# File paths
files = {
    "Simple Python": "LL-Output-Thu-23-Oct-2025-at-12-17-40PM.txt",
    "Vectorised": "LL-Output-Mon-27-Oct-2025-at-04-31-50PM.txt",
    "Series Numba": "LL-Output-Mon-27-Oct-2025-at-12-11-45PM.txt",
    "Parallel Numba": "LL-Output-Mon-27-Oct-2025-at-01-15-16PM.txt"
    

}

# Function to load file and extract relevant columns
def load_data(filepath):
    data = []
    pattern = re.compile(r"^\s*\d+\s+([\d.]+)\s+([-.\d]+)\s+([\d.]+)")
    with open(filepath, "r") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                ratio, energy, order = map(float, match.groups())
                data.append((ratio, energy, order))
    df = pd.DataFrame(data, columns=["Ratio", "Energy", "Order"])
    df.index.name = "MC Step"
    return df

# Load all files
dfs = {label: load_data(path) for label, path in files.items()}

# Plot energy vs MC step
plt.figure(figsize=(10, 6))
for label, df in dfs.items():
    plt.plot(df.index, df["Energy"], label=f"{label}")
plt.xlabel("Monte Carlo Step")
plt.ylabel("Energy")
plt.legend()
plt.grid(True)
plt.show()

# Plot order vs MC step
plt.figure(figsize=(10, 6))
for label, df in dfs.items():
    plt.plot(df.index, df["Order"], label=f"{label}")
plt.xlabel("Monte Carlo Step")
plt.ylabel("Order Parameter")
plt.legend()
plt.grid(True)
plt.show()
