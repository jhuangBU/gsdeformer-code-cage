"""Pre-load libboost_thread so pymesh can find it. Import before pymesh."""
import os
import ctypes

_conda = os.environ.get("CONDA_PREFIX", "")
if _conda:
    _lib = os.path.join(_conda, "lib")
    os.environ["LD_LIBRARY_PATH"] = _lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    ctypes.CDLL(os.path.join(_lib, "libboost_thread.so.1.65.1"))
