"""Task-scoped compatibility for TMS-preloaded CANN forkservers."""
import os
from pathlib import Path
import signal
import sys

# TMS installs a native SIGINT handler before Python starts, represented as None.
# Python's forkserver saves it then fails restoring None in every forked child.
if (os.environ.get("GLM53_RAY_SIGINT_NONE_COMPAT") == "1"
        and any("from multiprocessing.forkserver import main" in arg
                for arg in sys.orig_argv)
        and Path(os.environ.get("GLM53_NATIVE_ROOT", "")).resolve()
            == Path(__file__).resolve().parents[5]
        and signal.getsignal(signal.SIGINT) is None):
    signal.signal(signal.SIGINT, signal.default_int_handler)
