"""T2 · Layout reasoning: re-derive Triton's coalesce pass, then map lanes to elements — reference solution.

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
    rank = len(size_per_thread)
    tpw, wpc = [0] * rank, [0] * rank
    rem_lanes, rem_threads, rem_warps = threads_per_warp, num_warps * threads_per_warp, num_warps
    prev_lanes = prev_warps = 1
    for d in range(rank - 1):
        i = order[d]
        threads_cta = clamp(rem_threads, 1, max(1, shape[i] // size_per_thread[i]))
        tpw[i] = clamp(threads_cta, 1, rem_lanes)
        wpc[i] = clamp(threads_cta // tpw[i], 1, rem_warps)
        rem_warps //= wpc[i]
        rem_lanes //= tpw[i]
        rem_threads //= threads_cta
        prev_lanes *= tpw[i]
        prev_warps *= wpc[i]
    tpw[order[rank - 1]] = threads_per_warp // prev_lanes
    wpc[order[rank - 1]] = num_warps // prev_warps
    return Blocked(tuple(size_per_thread), tuple(tpw), tuple(wpc), tuple(order))


def elements_per_thread(shape, order, elem_bits, contiguity, divisibility_bytes, num_warps, threads_per_warp=32):
    """getNumElementsPerThread + the numElems/numThreads cap from CoalesceUtils.cpp."""
    elem_bytes = max(elem_bits // 8, 1)
    max_multiple = max(divisibility_bytes // elem_bytes, 1)
    max_contig = min(contiguity, shape[order[0]])
    alignment = min(max_multiple, max_contig)
    per_thread = min(alignment, 128 // elem_bits)
    return min(per_thread, max(prod(shape) // (num_warps * threads_per_warp), 1))


def coalesced_layout(shape, order, elem_bits, contiguity, divisibility_bytes=16, num_warps=4):
    """The layout the coalesce pass gives a load/store whose pointer has this contiguity along order[0]."""
    per = elements_per_thread(shape, order, elem_bits, contiguity, divisibility_bytes, num_warps)
    spt = [1] * len(shape)
    spt[order[0]] = per
    return build_blocked(shape, spt, order, num_warps)


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
    t = tile(L)
    reps = [max(1, -(-s // td)) for s, td in zip(shape, t)]
    lc = _delinearize(lane, L.threads_per_warp, L.order)
    wc = _delinearize(warp, L.warps_per_cta, L.order)
    out = []
    for r in range(prod(reps)):
        rc = _delinearize(r, reps, L.order)
        for s in range(prod(L.size_per_thread)):
            sc = _delinearize(s, L.size_per_thread, L.order)
            coord = tuple((rc[d] * t[d] + wc[d] * L.size_per_thread[d] * L.threads_per_warp[d]
                           + lc[d] * L.size_per_thread[d] + sc[d]) % shape[d] for d in range(len(shape)))
            out.append(coord)
    return out


def owners(L: Blocked, shape):
    """coord -> set of (warp, lane) holding it (more than one = broadcast / duplication)."""
    num_warps = prod(L.warps_per_cta)
    lanes = prod(L.threads_per_warp)
    own = {}
    for w in range(num_warps):
        for ln in range(lanes):
            for c in thread_elements(L, shape, w, ln):
                own.setdefault(c, set()).add((w, ln))
    return own


def lane_table(L: Blocked, shape, warp=0):
    """[(lane, [coords...])] for one warp — the T2 deliverable."""
    return [(ln, thread_elements(L, shape, warp, ln)) for ln in range(prod(L.threads_per_warp))]


# ---------------------------------------------------------------- coalescing
def first_load_sectors(L: Blocked, shape, elem_bytes, strides=None):
    """Simulate warp 0's first vectorized load instruction on a tensor with element strides
    `strides` (row-major contiguous by default). Each lane loads one vector of
    vec = min(spt[order[0]], 16 // elem_bytes) contiguous elements along order[0].
    Returns (bytes_requested, sectors_touched, efficiency = bytes / (sectors * 32))."""
    if strides is None:
        strides = [prod(shape[d + 1:]) for d in range(len(shape))]
    vec = min(L.size_per_thread[L.order[0]], max(16 // elem_bytes, 1))
    sectors = set()
    nbytes = 0
    for ln in range(prod(L.threads_per_warp)):
        first = thread_elements(L, shape, 0, ln)[:vec]
        for c in first:
            addr = sum(ci * si for ci, si in zip(c, strides)) * elem_bytes
            sectors.add(addr // 32)
            nbytes += elem_bytes
    return nbytes, len(sectors), nbytes / (32 * len(sectors))


# ---------------------------------------------------------------- convert_layout cost
def classify_conversion(src: Blocked, dst: Blocked, shape):
    """'registers'      every element stays in the same thread(s): only registers are renamed
                         (Triton: cvtReordersRegisters)
       'warp_shuffle'   every element stays in the same warp: lanes exchange with shfl
                         (Triton: cvtNeedsWarpShuffle — the real check also limits the permutation)
       'shared_memory'  some element must cross warps: write to smem, barrier, read back
                         (Triton: cvtNeedsSharedMemory)"""
    a, b = owners(src, shape), owners(dst, shape)
    if all(a[c] == b[c] for c in a):
        return "registers"
    warps = lambda s: {w for w, _ in s}
    if all(warps(a[c]) == warps(b[c]) for c in a):
        return "warp_shuffle"
    return "shared_memory"
