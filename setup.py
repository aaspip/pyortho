#!/usr/bin/env python
# -*- encoding: utf8 -*-
import io
import os

from setuptools import setup
from distutils.core import Extension
import numpy


long_description = """
Source code: https://github.com/aaspip/pyortho""".strip() 


def read(*names, **kwargs):
    return io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")).read()

orthoc_module = Extension('orthocfun', sources=['pyortho/src/orthocfuns.c'], 
										include_dirs=[numpy.get_include()])

setup(
    name="pyortho",
    version="0.0.4",
    license='GNU General Public License, Version 3 (GPLv3)',
    description="A python package for local signal-and-noise orthogonalization and local similarity calculation",
    long_description=long_description,
    author="pyortho developing team",
    author_email="chenyk2016@gmail.com",
    url="https://github.com/aaspip/pyortho",
    ext_modules=[orthoc_module],
    packages=['pyortho'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    keywords=[
        "seismology", "earthquake seismology", "exploration seismology", "array seismology", "denoising", "science", "engineering", "structure", "local slope", "filtering"
    ],
    install_requires=[
        "numpy", "scipy", "matplotlib"
    ],
    extras_require={
        "docs": ["sphinx", "ipython", "runipy"]
    }
)
