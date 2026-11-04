"""pytest hook: make the repo root importable (drill.load) and, without a CUDA GPU, run any
Triton code in the CPU interpreter (TRITON_INTERPRET must be set before triton is imported)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
try:
    import torch
except ImportError:
    torch = None
if torch is not None and not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
