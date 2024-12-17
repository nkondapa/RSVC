from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from pathlib import Path
from os import path

extensions = [
    Extension("src.nmf_torch.nmf.cylib.nnls_bpp_utils", ["src/nmf_torch/ext_modules/nnls_bpp_utils.pyx"]),
]

setup(ext_modules=cythonize(extensions))

# python3 setup.py build_ext --inplace