from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='rvchallenge',
    version="0.0.1",
    description='An attempt at competing in the RVChallenge.',
    long_description='See https://rv-challenge.wikispaces.com',

    author='Will M. Farr',
    author_email='w.farr@bham.ac.uk',

    license='MIT',

    packages=['rvchallenge'],

    ext_modules = cythonize([Extension('kepler', ['rvchallenge/kepler.pyx'])]),
    include_dirs = [np.get_include()]
)
