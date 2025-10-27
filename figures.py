import matplotlib.pyplot as plt

# Data for NUMBA (FIG 1)
threads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
runtime_ms = [30.3, 23.1, 22.3, 23.0, 30.3, 23.3, 25.1, 22.8, 23.4, 22.2, 23.1, 21.0]

# Plot
plt.figure(figsize=(8, 5))
plt.bar(threads, runtime_ms)
plt.xlabel("Thread Count")
plt.ylabel("Runtime (ms)")

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()


# Data for TOTAL PERFORMANCE
category = ["Simple","Vectorised","S.Numba","P.Numba (1Th)","P.Numba (12Th)"]
runtime_ms = [3000, 989, 610, 30.3, 21]

# Plot
plt.figure(figsize=(8, 5))
plt.bar(category, runtime_ms)
plt.xlabel("Thread Count")
plt.ylabel("Runtime (ms)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
              
plt.show()
