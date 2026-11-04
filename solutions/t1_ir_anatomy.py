"""T1 · IR dump lab: one kernel, four representations — reference solution.

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
    if re.search(r"^\s*\.version\s+\d", text, re.M) and re.search(r"^\s*\.target\s+sm_", text, re.M):
        return "ptx"
    if re.search(r"^\s*define\s", text, re.M) and "nvptx" in text:
        return "llir"
    if "#ttg." in text or '"ttg.num-warps"' in text:
        return "ttgir"
    if "tt.func" in text:
        return "ttir"
    raise ValueError("not a Triton stage dump")


# ---------------------------------------------------------------- MLIR ops
_OP = re.compile(r"^\s*(?:%[\w#:]+(?:\s*,\s*%[\w#:]+)*\s*=\s*)?\"?([a-z_]+\.[a-z_][\w.]*)\"?(?=[\s({<]|$)")


def mlir_ops(text):
    """Counter of fully qualified op names ('tt.load', 'ttg.convert_layout', 'scf.for', ...).
    One op per line in printed MLIR; attribute alias lines (#x = ...) and comments are skipped."""
    c = Counter()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("//", "#", "}")):
            continue
        m = _OP.match(line)
        if m:
            c[m.group(1)] += 1
    return c


def dialect_histogram(text):
    """Counter dialect -> number of ops ('tt', 'ttg', 'arith', 'scf', ...)."""
    c = Counter()
    for op, n in mlir_ops(text).items():
        c[op.split(".")[0]] += n
    return c


# ---------------------------------------------------------------- layouts
def parse_attr_body(body):
    """'sizePerThread = [1, 4], order = [1, 0], vec = 8, parent = #blocked' -> dict.
    Lists become lists of ints, ints become ints, anything else stays a string."""
    out = {}
    for m in re.finditer(r"(\w+)\s*=\s*(\[[^\]]*\]|#[\w.]+(?:<\{.*?\}>)?|[^,]+)", body):
        k, v = m.group(1), m.group(2).strip()
        if v.startswith("["):
            out[k] = [int(x) for x in v[1:-1].split(",") if x.strip()]
        elif re.fullmatch(r"-?\d+", v):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def layout_aliases(text):
    """{'#blocked': {'kind': 'blocked', 'sizePerThread': [...], ...}, '#mma': {'kind': 'nvidia_mma', ...}}"""
    out = {}
    for m in re.finditer(r"^(#[\w]+)\s*=\s*#ttg\.(\w+)(?:<\{(.*)\}>)?\s*$", text, re.M):
        d = {"kind": m.group(2)}
        if m.group(3):
            d.update(parse_attr_body(m.group(3)))
        out[m.group(1)] = d
    return out


def _encoding_of(tensor_type):
    """'tensor<64x64xf16, #mma>' -> '#mma'; nested '#ttg.dot_op<{opIdx = 0, parent = #mma, ...}>' kept whole."""
    inner = tensor_type[tensor_type.index("<") + 1: tensor_type.rindex(">")]
    parts = inner.split(",", 1)
    return parts[1].strip() if len(parts) > 1 else None


def convert_layouts(text):
    """[(src_encoding, dst_encoding)] for every ttg.convert_layout."""
    out = []
    for m in re.finditer(r"ttg\.convert_layout\s+%[\w#]+\s*:\s*(tensor<.*?>)\s*->\s*(tensor<.*>)\s*$", text, re.M):
        out.append((_encoding_of(m.group(1)), _encoding_of(m.group(2))))
    return out


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
    out = {}
    for kind, pat in _PTX_KINDS.items():
        out[kind] = Counter(m.group(1).rstrip(".") for m in re.finditer(pat, text))
    return out


def mma_shape(instr):
    """'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32' -> (16, 8, 16)."""
    m = re.search(r"\.m(\d+)n(\d+)k(\d+)", instr)
    return tuple(int(x) for x in m.groups()) if m else None


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
    return {k for k, pat in CONCEPTS.items() if re.search(pat, text)}


def concept_diff(stages):
    """stages: ordered list of (name, text). Returns [(from, to, appeared, disappeared)]."""
    out = []
    for (a, ta), (b, tb) in zip(stages, stages[1:]):
        ca, cb = concepts(ta), concepts(tb)
        out.append((a, b, sorted(cb - ca), sorted(ca - cb)))
    return out


# ---------------------------------------------------------------- the GPU half
def dump_stages(compiled_kernel):
    """On a CUDA machine: compiled = kernel[grid](...); dump_stages(compiled) -> {stage: text}."""
    return {k: compiled_kernel.asm[k] for k in ("ttir", "ttgir", "llir", "ptx") if k in compiled_kernel.asm}
