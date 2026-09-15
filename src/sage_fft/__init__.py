# Author: even
"""SAGE: hierarchical FFT realization and compiler-constrained model control."""

from .fft_ir import Contract, execute, import_cuda

__all__ = ["Contract", "execute", "import_cuda"]
__version__ = "0.3.0"
__author__ = "even"
