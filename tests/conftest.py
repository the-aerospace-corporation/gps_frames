import os
import pytest

# Disable Numba JIT for tests to ensure full coverage measurement
os.environ['NUMBA_DISABLE_JIT'] = '1'
