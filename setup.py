from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="LebwohlLasher_cy_kernel",
    sources=["LebwohlLasher_cy_kernel.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-fopenmp"],  # remove -ffast-math for now
    extra_link_args=["-fopenmp"],
    libraries=["m"],  # <--- important
    language="c",
)

setup(
    name="LebwohlLasher_cy_kernel",
    ext_modules=cythonize([ext], language_level="3"),
)
