"""T5 · A mini FileCheck: how Triton's lit tests decide pass / fail — your implementation (run ./check.sh t5).

Every file under triton/test/ starts with a RUN line such as
    // RUN: triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s
lit runs the pass, pipes the output to LLVM's FileCheck, and FileCheck matches the `// CHECK...`
comments of the same file against that output. Supported here (the subset Triton tests use most):

  CHECK:        match somewhere at or after the current position
  CHECK-NEXT:   match on the line right after the previous match
  CHECK-SAME:   match on the same line as the previous match, after it
  CHECK-NOT:    must not appear between the previous match and the next positive match
  CHECK-LABEL:  splits input into blocks first; checks never cross a label
  CHECK-DAG:    a group of consecutive DAG checks may match in any order
  CHECK-COUNT-n: match n times in a row (each after the previous)

Patterns: literal text, except {{regex}} and [[VAR:regex]] (capture) / [[VAR]] (use), with
the `$` prefix ([[$VAR]]) meaning "global": with --enable-var-scope only those survive a CHECK-LABEL. Whitespace runs in the pattern
match any whitespace run in the input (FileCheck's default canonicalization).
"""
import re

DIRECTIVE = re.compile(r"(?://|;|#)\s*(CHECK(?:-NEXT|-SAME|-NOT|-LABEL|-DAG|-COUNT-\d+)?):(.*)$")


class CheckFailure(AssertionError):
    pass


def parse_checks(source, prefix="CHECK"):
    """[(kind, pattern, line_no)] from a test file. kind: 'CHECK', 'NEXT', 'SAME', 'NOT', 'LABEL', 'DAG', 'COUNT'."""
    raise NotImplementedError("parse_checks")


def compile_pattern(pat, env):
    """Turn a FileCheck pattern into a Python regex. Returns (regex, [capture variable names])."""
    raise NotImplementedError("compile_pattern")










def filecheck(check_source, output, prefix="CHECK", enable_var_scope=False):
    """Raise CheckFailure (an AssertionError) if `output` does not satisfy the CHECKs in check_source."""
    raise NotImplementedError("filecheck")


def split_input_file(text):
    """triton-opt -split-input-file: chunks separated by a line `// -----`."""
    return [c.strip("\n") for c in re.split(r"^// -----\s*$", text, flags=re.M)]
