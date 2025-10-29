from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

extra_compile_args = ["-O3", "-ffast-math"]
extra_link_args = []

if sys.platform.startswith("win"):
    extra_compile_args = ["/O2", "/fp:fast"]

ext_modules = [
    Extension(
        name="LebwohlLasher_cy_kernel",
        sources=["LebwohlLasher_cy_kernel.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=["m"],   # 👈 add this line to link libm (exp, cos, etc.)
        language="c",
    )
]

setup(
    name="LebwohlLasher_cy",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "initializedcheck": False,
            "infer_types": True,
            "embedsignature": True,
        },
    ),
)