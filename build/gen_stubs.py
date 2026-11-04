"""Generate mine/<drill>.py stubs from solutions/<drill>.py.

    python build/gen_stubs.py            # writes mine/*.py that do not exist yet
    python build/gen_stubs.py --force    # overwrite (destroys your work!)

Every function / method keeps its signature and docstring; its body becomes
`raise NotImplementedError`. Names listed in GIVEN stay exactly as in the solution (helpers,
data, fixtures). Private helpers (leading underscore) not in GIVEN are dropped: structure your
own code as you like.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GIVEN = {
    "k1_gemm_model": {"GPU", "TileConfig", "RTX3070"},
    "k2_bench_harness": {"FakeClock"},
    "k3_numerics": {"EPS", "to_bf16"},
    "k4_fusion": {"Stage", "chain_mlp_epilogue", "groups_of"},
    "t1_ir_anatomy": {"_OP", "_PTX_KINDS", "CONCEPTS", "dump_stages"},
    "t2_coalesce_layout": {"Blocked", "parse_blocked", "clamp", "_delinearize", "tile"},
    "t3_autotune_sweep": {"config_space", "fake_measure", "to_csv"},
    "t4_minimize": {"DTYPE_ORDER", "random_case", "NP", "make_input", "row_sum_reference", "buggy_row_sum"},
    "t5_filecheck": {"DIRECTIVE", "CheckFailure", "split_input_file"},
    "l1_graph": {"PRIMITIVES", "MOVEMENT", "Node", "Graph.__init__", "Graph.add", "Graph.tensor", "Graph.constant",
                 "Tensor.shape", "Tensor.dtype", "Tensor.output", "Tensor.expand", "Tensor.expand_to",
                 "Tensor.permute", "Tensor._bin", "Tensor.__add__", "Tensor.__mul__", "Tensor.lt",
                 "Tensor.select", "Tensor.sum", "Tensor.exp2"},
    "l2_egraph": {"EGraph.__init__", "EGraph.find", "EGraph.canon", "EGraph.add_enode", "EGraph.add",
                  "EGraph.num_classes", "EGraph.num_nodes", "COMM_ASSOC", "ALGEBRA", "ast_size", "op_cost", "evaluate"},
    "l3_search": {"space_size", "all_genomes", "fingerprint", "random_genome", "mutate", "crossover", "mlp_space",
                  "MATMUL_NS", "FUSE_SAVING", "mlp_measure", "mlp_proxy", "mlp_validate"},
    "l4_plan_trace": {"SlabBuffer", "aligned", "mlp_plan", "PHASES"},
    "l5_differential": {"STORAGE", "store", "reference_softmax", "device_softmax", "make_input", "RUST_DTYPE"},
}


def stub(name):
    src = (ROOT / "solutions" / f"{name}.py").read_text()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    given = GIVEN[name]
    edits = []          # (start_line0, end_line0_exclusive, replacement_lines)

    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, ast.ClassDef):
                if node.name in given:
                    continue
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                full = prefix + node.name
                if full in given:
                    continue
                start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
                if node.name.startswith("_") and not node.name.startswith("__"):
                    edits.append((start, node.end_lineno, []))
                    continue
                first = node.body[0]
                has_doc = isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str)
                keep_to = (first.end_lineno if has_doc else node.body[0].lineno - 1)
                indent = " " * node.body[0].col_offset
                body_start = keep_to if has_doc else first.lineno - 1
                edits.append((body_start, node.end_lineno, [f'{indent}raise NotImplementedError("{full}")\n']))

    visit(tree.body)
    for s, e, rep in sorted(edits, reverse=True):
        lines[s:e] = rep
    out = "".join(lines).replace("— reference solution.", "— your implementation (run ./check.sh " +
                                 name.split("_")[0] + ").", 1)
    ast.parse(out)
    return out


if __name__ == "__main__":
    force = "--force" in sys.argv
    for name in GIVEN:
        p = ROOT / "mine" / f"{name}.py"
        if p.exists() and not force:
            print("skip", p.name)
            continue
        p.write_text(stub(name))
        print("wrote", p.name)
