"""T5 · A mini FileCheck: how Triton's lit tests decide pass / fail — reference solution.

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
    out = []
    rx = re.compile(DIRECTIVE.pattern.replace("CHECK", re.escape(prefix)))
    for i, line in enumerate(source.splitlines(), 1):
        m = rx.search(line)
        if not m:
            continue
        tag, pat = m.group(1), m.group(2).strip()
        kind = tag[len(prefix):].lstrip("-") or "CHECK"
        if kind.startswith("COUNT-"):
            out.extend([("CHECK", pat, i)] * int(kind[6:]))
        else:
            out.append((kind, pat, i))
    return out


def compile_pattern(pat, env):
    """Turn a FileCheck pattern into a Python regex. Returns (regex, [capture variable names])."""
    rx, names, i = "", [], 0
    while i < len(pat):
        if pat.startswith("{{", i):
            j = pat.index("}}", i + 2)
            rx += "(?:" + pat[i + 2:j] + ")"
            i = j + 2
        elif pat.startswith("[[", i):
            j = pat.index("]]", i + 2)
            body = pat[i + 2:j]
            if ":" in body:
                name, sub = body.split(":", 1)
                key = re.sub(r"\W", "_", name)
                rx += f"(?P<{key}>{sub})"
                names.append(name)
            else:
                if body not in env:
                    raise CheckFailure(f"undefined variable [[{body}]]")
                rx += re.escape(env[body])
            i = j + 2
        elif pat[i].isspace():
            while i < len(pat) and pat[i].isspace():
                i += 1
            rx += r"\s+"
        else:
            rx += re.escape(pat[i])
            i += 1
    return re.compile(rx), names


def _search(pat, text, pos, end, env):
    rx, names = compile_pattern(pat, env)
    m = rx.search(text, pos, end)
    if m:
        for n in names:
            env[n] = m.group(re.sub(r"\W", "_", n))
    return m


def _line_bounds(text, pos):
    s = text.rfind("\n", 0, pos) + 1
    e = text.find("\n", pos)
    return s, (len(text) if e < 0 else e)


def _check_block(checks, text, start, end, env):
    pos = start
    last = None             # (match_start, match_end) of the previous positive match
    pending_not = []
    i = 0
    while i < len(checks):
        kind, pat, ln = checks[i]
        if kind == "NOT":
            pending_not.append((pat, ln))
            i += 1
            continue
        if kind == "DAG":
            group = []
            while i < len(checks) and checks[i][0] == "DAG":
                group.append(checks[i])
                i += 1
            used, hi = [], pos
            for _, p, l in group:
                rx, names = compile_pattern(p, env)
                found = None
                for m in rx.finditer(text, pos, end):
                    if all(m.end() <= a or m.start() >= b for a, b in used):
                        found = m
                        break
                if not found:
                    raise CheckFailure(f"line {l}: CHECK-DAG: {p!r} not found")
                for n in names:
                    env[n] = found.group(re.sub(r"\W", "_", n))
                used.append((found.start(), found.end()))
                hi = max(hi, found.end())
            _check_nots(pending_not, text, pos, min(a for a, _ in used), env)
            pending_not, pos, last = [], hi, (min(a for a, _ in used), hi)
            continue
        if kind == "SAME":
            if last is None:
                raise CheckFailure(f"line {ln}: CHECK-SAME with no previous match")
            _, le = _line_bounds(text, last[1] - 1 if last[1] > last[0] else last[1])
            m = _search(pat, text, last[1], le, env)
            if not m:
                raise CheckFailure(f"line {ln}: CHECK-SAME: {pat!r} not on the same line")
        elif kind == "NEXT":
            if last is None:
                raise CheckFailure(f"line {ln}: CHECK-NEXT with no previous match")
            _, le = _line_bounds(text, max(last[1] - 1, last[0]))
            if le >= end:
                raise CheckFailure(f"line {ln}: CHECK-NEXT: no next line")
            ns, ne = _line_bounds(text, le + 1)
            m = _search(pat, text, ns, ne, env)
            if not m:
                raise CheckFailure(f"line {ln}: CHECK-NEXT: {pat!r} is not on the next line")
        else:
            m = _search(pat, text, pos, end, env)
            if not m:
                raise CheckFailure(f"line {ln}: CHECK: {pat!r} not found")
        _check_nots(pending_not, text, pos, m.start(), env)
        pending_not = []
        last, pos = (m.start(), m.end()), m.end()
        i += 1
    _check_nots(pending_not, text, pos, end, env)


def _check_nots(nots, text, lo, hi, env):
    for pat, ln in nots:
        rx, _ = compile_pattern(pat, env)
        m = rx.search(text, lo, hi)
        if m:
            raise CheckFailure(f"line {ln}: CHECK-NOT: {pat!r} found: {m.group(0)!r}")


def filecheck(check_source, output, prefix="CHECK", enable_var_scope=False):
    """Raise CheckFailure (an AssertionError) if `output` does not satisfy the CHECKs in check_source."""
    checks = parse_checks(check_source, prefix)
    if not checks:
        raise CheckFailure("no check strings found")
    env = {}
    # 1. labels cut the input into blocks
    labels = [(i, c) for i, c in enumerate(checks) if c[0] == "LABEL"]
    if not labels:
        _check_block(checks, output, 0, len(output), env)
        return True
    starts = []
    pos = 0
    for _, (_, pat, ln) in labels:
        m = _search(pat, output, pos, len(output), env)
        if not m:
            raise CheckFailure(f"line {ln}: CHECK-LABEL: {pat!r} not found")
        starts.append((m.start(), m.end()))
        pos = m.end()
    idx = [i for i, _ in labels] + [len(checks)]
    if labels[0][0] > 0:
        _check_block(checks[:labels[0][0]], output, 0, starts[0][0], env)
    for b, (s, e) in enumerate(starts):
        block_end = starts[b + 1][0] if b + 1 < len(starts) else len(output)
        if enable_var_scope:     # FileCheck --enable-var-scope: only [[$VAR]] survives a label
            env = {k: v for k, v in env.items() if k.startswith("$")}
        _check_block(checks[idx[b] + 1: idx[b + 1]], output, e, block_end, env)
    return True


def split_input_file(text):
    """triton-opt -split-input-file: chunks separated by a line `// -----`."""
    return [c.strip("\n") for c in re.split(r"^// -----\s*$", text, flags=re.M)]
