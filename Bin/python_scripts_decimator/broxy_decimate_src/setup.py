from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).resolve().parent

sources = [
    "broxy_bindings.cpp",
    "Decimator/Decimator.cpp",
    "Decimator/SolveQP.cpp",
    "Decimator.cpp",
]

include_dirs = [
    str(ROOT),
    str(ROOT / "vcglib"),
    str(ROOT / "eigen"),
    str(ROOT / "Common"),
    str(ROOT / "Decimator"),
]

ext_modules = [
    Pybind11Extension(
        "broxy_decimate",
        sources=sources,
        include_dirs=include_dirs,
        cxx_std=14,
        extra_link_args=[
            "/NODEFAULTLIB:FortranRuntime.static.lib",
            "/NODEFAULTLIB:FortranDecimal.static.lib",
            "/DEFAULTLIB:FortranRuntime.dynamic.lib",
            "/DEFAULTLIB:FortranDecimal.dynamic.lib",
            "/DEFAULTLIB:FortranCommon.lib",
        ]
        if os.name == "nt"
        else ["-lgfortran"],
    )
]

class BuildWithFortran(build_ext):
    def build_extension(self, ext):
        # Compile Fortran source to an object file and link it in.
        fc = os.environ.get("FC")
        if not fc:
            fc = (
                shutil.which("flang-new")
                or shutil.which("gfortran")
                or shutil.which("ifx")
                or shutil.which("ifort")
            )
        if not fc:
            raise RuntimeError("No Fortran compiler found. Set FC or install one.")

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        fortran_src = ROOT / "Decimator" / "ql0001.f"
        obj_suffix = ".obj" if os.name == "nt" else ".o"
        fortran_obj = build_temp / f"ql0001{obj_suffix}"

        fortran_flags = os.environ.get("FFLAGS", "").split()
        # Try to align the Fortran runtime with MSVC /MD on Windows.
        if os.name == "nt" and "flang" in Path(fc).name.lower():
            fortran_flags += ["-fms-runtime-lib=md"]

        compile_cmd = [fc, "-c", str(fortran_src), "-o", str(fortran_obj), *fortran_flags]
        subprocess.check_call(compile_cmd)

        ext.extra_objects = list(ext.extra_objects or []) + [str(fortran_obj)]
        super().build_extension(ext)

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildWithFortran},
)
