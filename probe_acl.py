import ctypes
import os
import sys

lib = ctypes.CDLL("libascendcl.so")
lib.aclInit.argtypes = [ctypes.c_char_p]
lib.aclGetRecentErrMsg.restype = ctypes.c_char_p
lib.aclrtSetDevice.argtypes = [ctypes.c_int]
init = lib.aclInit(None)
print("ASCEND_HOME_PATH", os.environ.get("ASCEND_HOME_PATH"), "aclInit", init, flush=True)
if init:
    print(lib.aclGetRecentErrMsg().decode("utf-8", "replace"), flush=True)
    sys.exit(1)
status = lib.aclrtSetDevice(0)
print("visible_devices", os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), "aclrtSetDevice", status, flush=True)
if status:
    print(lib.aclGetRecentErrMsg().decode("utf-8", "replace"), flush=True)
sys.exit(bool(status))
