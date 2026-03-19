import sys

collect_ignore = []

# test_gpu_setup imports utils.gpu_utils which clashes with the root utils.py
# module when running under --import-mode=importlib
try:
    from utils.gpu_utils import GPUManager  # noqa: F401
except (ImportError, ModuleNotFoundError):
    collect_ignore.append("test_gpu_setup.py")
