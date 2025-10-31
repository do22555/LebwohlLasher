from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "LebwohlLasher_mpi_kernel",
    ["LebwohlLasher_mpi_kernel.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-ffast-math", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(
    name="LebwohlLasher_mpi_kernel",
    ext_modules=cythonize([ext], language_level="3"),
)