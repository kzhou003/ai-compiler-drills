"""T1 · IR dump lab: one kernel, four representations — your implementation (run ./check.sh t1).

Triton compiles a @triton.jit function through a fixed list of stages (see
third_party/nvidia/backend/compiler.py: make_ttir -> make_ttgir -> make_llir -> make_ptx -> make_cubin).
`kernel[grid](...)` returns a CompiledKernel whose `.asm` dict holds the text of each stage:

    k = add_kernel[grid](x, y, out, n, BLOCK=1024)
    k.asm["ttir"], k.asm["ttgir"], k.asm["llir"], k.asm["ptx"]

This drill is the *reading* half: given those texts (fixtures/t1/*), tell which stage a dump is,
count ops per dialect, pull out layout encodings and layout conversions, list the tensor-core
instructions in PTX, and diff which concepts appear / disappear between stages.
Runs anywhere — on a GPU box, `dump_stages()` produces the real texts.
"""
import re
from collections import Counter

# ---------------------------------------------------------------- stage detection
def detect_stage(text):
    """'ptx' | 'llir' | 'ttgir' | 'ttir' (checked in that order: lower stages are unambiguous)."""
    raise NotImplementedError("detect_stage")


# ---------------------------------------------------------------- MLIR ops
_OP = re.compile(r"^\s*(?:%[\w#:]+(?:\s*,\s*%[\w#:]+)*\s*=\s*)?\"?([a-z_]+\.[a-z_][\w.]*)\"?(?=[\s({<]|$)")


def mlir_ops(text):
    """Counter of fully qualified op names ('tt.load', 'ttg.convert_layout', 'scf.for', ...).
    One op per line in printed MLIR; attribute alias lines (#x = ...) and comments are skipped."""
    raise NotImplementedError("mlir_ops")


def dialect_histogram(text):
    """Counter dialect -> number of ops ('tt', 'ttg', 'arith', 'scf', ...)."""
    raise NotImplementedError("dialect_histogram")


# ---------------------------------------------------------------- layouts
def parse_attr_body(body):
    """'sizePerThread = [1, 4], order = [1, 0], vec = 8, parent = #blocked' -> dict.
    Lists become lists of ints, ints become ints, anything else stays a string."""
    raise NotImplementedError("parse_attr_body")


def layout_aliases(text):
    """{'#blocked': {'kind': 'blocked', 'sizePerThread': [...], ...}, '#mma': {'kind': 'nvidia_mma', ...}}"""
    raise NotImplementedError("layout_aliases")




def convert_layouts(text):
    """[(src_encoding, dst_encoding)] for every ttg.convert_layout."""
    raise NotImplementedError("convert_layouts")


# ---------------------------------------------------------------- PTX / LLVM
_PTX_KINDS = {
    "mma": r"\b((?:mma|wgmma\.mma_async)\.[\w.]+)",
    "ldmatrix": r"\b(ldmatrix\.[\w.]+)",
    "cp_async": r"\b(cp\.async\.[\w.]+|cp\.async\.bulk[\w.]*)",
    "barrier": r"\b(bar\.sync|barrier\.sync|mbarrier\.[\w.]+)",
    "global_vec": r"\b((?:ld|st)\.global(?:\.\w+)*\.v\d\.\w+)",
}


def ptx_instructions(text):
    """{'mma': Counter({'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32': 2}), 'ldmatrix': ..., ...}
    Works on PTX and on LLVM IR (Triton emits these as inline asm strings there)."""
    raise NotImplementedError("ptx_instructions")


def mma_shape(instr):
    """'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32' -> (16, 8, 16)."""
    raise NotImplementedError("mma_shape")


# ---------------------------------------------------------------- concepts per stage
CONCEPTS = {
    "program_id": r"tt\.get_program_id|ctaid",
    "block_tensor": r"tensor<",
    "pointer_tensor": r"tensor<[^>]*!tt\.ptr",
    "layout_encoding": r"#ttg\.(blocked|nvidia_mma|slice|dot_op)|, #\w+>",
    "shared_memory": r"ttg\.local_alloc|memdesc|addrspace\(3\)|\.shared",
    "async_copy": r"async_copy|cp\.async",
    "mma_instruction": r"mma\.sync|wgmma|ldmatrix",
    "thread_id": r"tid\.x|%tid",
    "virtual_registers": r"\.reg\s+\.",
    "high_level_dot": r"tt\.dot\b",
}


def concepts(text):
    raise NotImplementedError("concepts")


def concept_diff(stages):
    """stages: ordered list of (name, text). Returns [(from, to, appeared, disappeared)]."""
    raise NotImplementedError("concept_diff")


# ---------------------------------------------------------------- the GPU half
def dump_stages(compiled_kernel):
    """On a CUDA machine: compiled = kernel[grid](...); dump_stages(compiled) -> {stage: text}."""
    return {k: compiled_kernel.asm[k] for k in ("ttir", "ttgir", "llir", "ptx") if k in compiled_kernel.asm}
