from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "pyxdm.core.exchange_hole_cpp",
        ["pyxdm/cpp/exchange_hole.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++"
    )
]

setup(
    name="pyxdm",
    ext_modules=ext_modules,
)