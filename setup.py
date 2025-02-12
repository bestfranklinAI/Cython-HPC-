#setup.py

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='cython_gauss_seidel',
    ext_modules=cythonize("gauss.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)
