"""T2 · Layout reasoning: re-derive Triton's coalesce pass, then map lanes to elements — your implementation (run ./check.sh t2).

Triton's `tritongpu-coalesce` pass picks a BlockedEncoding for every load/store. Two pieces of
real code decide it, and this drill reimplements both in Python:

  1. lib/Dialect/TritonGPU/Transforms/Utility.cpp  getNumElementsPerThread()
        perThread = min(divisibility_bytes / elem_bytes, contiguity, shape[order[0]], 128 / elem_bits)
     then CoalesceUtils.cpp also caps it by  max(numElems / numThreads, 1).
  2. include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td  BlockedEncodingAttr builder:
        walk dims from the fastest (order[0]) and hand out lanes, then warps; the slowest dim
        takes whatever is left.

With the layout in hand we answer the T2 questions: which lane owns which coordinates, is the
first load of a warp coalesced, and does a ttg.convert_layout stay in registers, need warp
shuffles, or go through shared memory (lib/Analysis/Utility.cpp: cvtReordersRegisters /
cvtNeedsWarpShuffle / cvtNeedsSharedMemory).
"""
import re
from dataclasses import dataclass
from itertools import product
from math import prod


@dataclass(frozen=True)
class Blocked:
    size_per_thread: tuple
    threads_per_warp: tuple
    warps_per_cta: tuple
    order: tuple

    def __str__(self):
        f = lambda v: "[" + ", ".join(map(str, v)) + "]"
        return (f"#ttg.blocked<{{sizePerThread = {f(self.size_per_thread)}, threadsPerWarp = "
                f"{f(self.threads_per_warp)}, warpsPerCTA = {f(self.warps_per_cta)}, order = {f(self.order)}}}>")


def parse_blocked(text):
    """Parse '#ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], ...}>'."""
    def grab(key):
        m = re.search(key + r"\s*=\s*\[([^\]]*)\]", text)
        if not m:
            raise ValueError(f"missing {key}")
        return tuple(int(x) for x in m.group(1).split(","))
    return Blocked(grab("sizePerThread"), grab("threadsPerWarp"), grab("warpsPerCTA"), grab("order"))


def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def build_blocked(shape, size_per_thread, order, num_warps, threads_per_warp=32):
    """Port of the BlockedEncodingAttr::get(shape, sizePerThread, order, numWarps, numThreadsPerWarp) builder."""
    raise NotImplementedError("build_blocked")


def elements_per_thread(shape, order, elem_bits, contiguity, divisibility_bytes, num_warps, threads_per_warp=32):
    """getNumElementsPerThread + the numElems/numThreads cap from CoalesceUtils.cpp."""
    raise NotImplementedError("elements_per_thread")


def coalesced_layout(shape, order, elem_bits, contiguity, divisibility_bytes=16, num_warps=4):
    """The layout the coalesce pass gives a load/store whose pointer has this contiguity along order[0]."""
    raise NotImplementedError("coalesced_layout")


# ---------------------------------------------------------------- who owns what
def _delinearize(idx, sizes, order):
    """Split a linear index into per-dim coordinates; order[0] varies fastest."""
    c = [0] * len(sizes)
    for d in order:
        c[d] = idx % sizes[d]
        idx //= sizes[d]
    return c


def tile(L: Blocked):
    return tuple(s * t * w for s, t, w in zip(L.size_per_thread, L.threads_per_warp, L.warps_per_cta))


def thread_elements(L: Blocked, shape, warp, lane):
    """Coordinates held by (warp, lane), in register order (inside a chunk first, then repetitions,
    both linearized along `order`)."""
    raise NotImplementedError("thread_elements")


def owners(L: Blocked, shape):
    """coord -> set of (warp, lane) holding it (more than one = broadcast / duplication)."""
    raise NotImplementedError("owners")


def lane_table(L: Blocked, shape, warp=0):
    """[(lane, [coords...])] for one warp — the T2 deliverable."""
    raise NotImplementedError("lane_table")


# ---------------------------------------------------------------- coalescing
def first_load_sectors(L: Blocked, shape, elem_bytes, strides=None):
    """Simulate warp 0's first vectorized load instruction on a tensor with element strides
    `strides` (row-major contiguous by default). Each lane loads one vector of
    vec = min(spt[order[0]], 16 // elem_bytes) contiguous elements along order[0].
    Returns (bytes_requested, sectors_touched, efficiency = bytes / (sectors * 32))."""
    raise NotImplementedError("first_load_sectors")


# ---------------------------------------------------------------- convert_layout cost
def classify_conversion(src: Blocked, dst: Blocked, shape):
    """'registers'      every element stays in the same thread(s): only registers are renamed
                         (Triton: cvtReordersRegisters)
       'warp_shuffle'   every element stays in the same warp: lanes exchange with shfl
                         (Triton: cvtNeedsWarpShuffle — the real check also limits the permutation)
       'shared_memory'  some element must cross warps: write to smem, barrier, read back
                         (Triton: cvtNeedsSharedMemory)"""
    raise NotImplementedError("classify_conversion")
