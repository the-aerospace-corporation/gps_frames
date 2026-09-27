import os
import sys
from pathlib import Path
import pytest

# Add project root to sys.path so tests can import examples if needed
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Disable Numba JIT for tests to ensure full coverage measurement
if "NUMBA_DISABLE_JIT" not in os.environ:
    os.environ["NUMBA_DISABLE_JIT"] = "1"
