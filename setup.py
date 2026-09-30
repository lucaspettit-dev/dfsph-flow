"""Build script: cythonizes the DFSPH core extension."""
import platform

from Cython.Build import cythonize
from setuptools import Extension, setup

compile_args = []
if platform.system() != "Windows":
    compile_args = ["-O3", "-ffast-math"]
else:
    compile_args = ["/O2"]

extensions = [
    Extension(
        "dfsph_flow._core",
        ["src/dfsph_flow/_core.pyx"],
        extra_compile_args=compile_args,
    )
]

setup(ext_modules=cythonize(extensions, language_level="3"))
