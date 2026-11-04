"""Build index.html (one self-contained page) from content/*.md + the code in mine/, tests/, solutions/.

    python build/build.py        # needs markdown-it-py

Markdown extensions (fenced containers, nesting allowed; fenced code is skipped when scanning):
    ::: hint 1 标题        L1/L2/L3 collapsible hint
    ::: solution 标题      collapsible reference solution (red, "write yours first")
    ::: answer 标题        click-to-reveal answer
    ::: callout info 标题  boxed note; kinds: info | warn | ok | idea
    ::: raw                emitted verbatim
    :::                    closes the innermost container
Line macro:
    {{code: path/to/file [title]}}   embeds the file as a highlighted code block with a copy button
Each content file starts with `lede: one sentence`.
"""
import html
import json
import re
import shlex
from pathlib import Path

from markdown_it import MarkdownIt

import sources as SRC

LINK_ERRORS: list[str] = []

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"

# (id, file stem or None for a lab page, title, minutes, difficulty, track)
DRILLS = [
    ("K1", "k1_gemm_model", "GEMM tile 账本：FLOPs、搬运量、smem / 寄存器、occupancy、wave", 45, "易", "K"),
    ("K2", "k2_bench_harness", "可信的 benchmark：warmup、同步、中位数、compile 与 run 分离", 30, "易", "K"),
    ("K3", "k3_numerics", "数值正确性：ULP、容差、重结合、边界形状", 45, "中", "K"),
    ("K4", "k4_fusion", "Fusion 的账：launch、中间结果流量与寄存器压力", 30, "中", "K"),
    ("T1", "t1_ir_anatomy", "IR dump lab：同一个 kernel 的 TTIR / TTGIR / LLVM IR / PTX", 60, "易", "T"),
    ("T2", "t2_coalesce_layout", "Layout reasoning：复刻 coalesce pass，lane → 坐标，convert_layout 的代价", 75, "难", "T"),
    ("T3", "t3_autotune_sweep", "GEMM tile sweep：prune → perf model → do_bench → 选最优", 45, "中", "T"),
    ("T4", "t4_minimize", "Correctness bug minimization：fuzz → 失败 → 缩小 → 回归测试", 45, "中", "T"),
    ("T5", "t5_filecheck", "lit / FileCheck：自己写一个 FileCheck，读懂 Triton 的编译器测试", 60, "难", "T"),
    ("L1", "l1_graph", "Graph semantics：把 relu(x @ w + b) 拆成 primitive ops", 40, "易", "L"),
    ("L2", "l2_egraph", "E-graph 从零写：union-find、rebuild、rewrite、saturation、extraction", 75, "难", "L"),
    ("L3", "l3_search", "Search：genome、GA、实测排名、finalists 与代理成本", 50, "中", "L"),
    ("L4", "l4_plan_trace", "Backend trace：选中计划 → launch 列表 → arena 分配（plan_slab）", 45, "中", "L"),
    ("L5", "l5_differential", "Differential test：Reference vs CUDA，带理由的容差", 45, "中", "L"),
    ("X1", None, "共同实验：同一个 fused MLP，在 Triton 和 Luminal 两边量一遍", 120, "中", "X"),
    ("P1", None, "贡献实战：issue 筛选、#370 复现、PR 模板、review 节奏", 180, "中", "P"),
]
# track -> (title, description, tier)
TRACKS = {
    "K": ("Track K · 共同知识主线", "GEMM 资源账、benchmark 方法、数值正确性、fusion 取舍：两个项目都要用的基本功。", "共同基础"),
    "T": ("Track T · Triton（kernel compiler）", "从 @triton.jit 到 PTX：IR 各层、layout 与 coalescing、autotune、最小复现、lit/FileCheck 测试。", "双主线"),
    "L": ("Track L · Luminal（graph compiler）", "primitive graph、e-graph 与 equality saturation、实测驱动的搜索、执行计划与 arena、Reference vs CUDA 对拍。", "双主线"),
    "X": ("Track X · 互补实验", "同一个 workload 两边各走一遍：kernel 怎样生成和跑快，graph 怎样选对实现。", "实战"),
    "P": ("Track P · 第一笔贡献", "从 issue 到 merged PR：复现、定位、最小改动、测试、benchmark 证据。", "实战"),
}
BASICS = ["intro", "basics_env", "basics_gpu", "basics_triton", "basics_luminal", "basics_compare"]
HINT_NOTES = {"1": "方向（轻度剧透）", "2": "拆解与常见坑（中度剧透）", "3": "几乎是答案（重度剧透）"}

MD = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable(["table", "strikethrough"])
OPEN_RE = re.compile(r"^:::\s+(\w+)\s*(.*)$")
CLOSE_RE = re.compile(r"^:::\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
CODE_RE = re.compile(r"^\{\{code:\s*(\S+)\s*(.*?)\}\}\s*$")
LANG = {".py": "python", ".cu": "cpp", ".sh": "bash", ".mlir": "plaintext", ".ptx": "plaintext", ".ll": "llvm", ".rs": "rust"}


def split_blocks(lines):
    out, i, in_fence = [], 0, False
    while i < len(lines):
        line = lines[i]
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        m = None if in_fence else OPEN_RE.match(line)
        if not m:
            out.append(line)
            i += 1
            continue
        depth, j, inner, f = 1, i + 1, [], False
        while j < len(lines):
            ln = lines[j]
            if FENCE_RE.match(ln):
                f = not f
            elif not f and OPEN_RE.match(ln):
                depth += 1
            elif not f and CLOSE_RE.match(ln):
                depth -= 1
                if depth == 0:
                    break
            inner.append(ln)
            j += 1
        if depth:
            raise SystemExit(f"unclosed ::: {m.group(1)} near {line!r}")
        out.append((m.group(1), m.group(2).strip(), inner))
        i = j + 1
    return out


def code_block(path: str, title: str = "") -> str:
    p = ROOT / path
    text = p.read_text(encoding="utf-8").rstrip("\n")
    lang = LANG.get(p.suffix, "plaintext")
    label = html.escape(title or path)
    n = text.count("\n") + 1
    return (f'<div class="codebox"><div class="code-head"><span class="path">{label}</span>'
            f'<span class="muted small">{n} 行</span><button type="button" class="btn copy">复制</button></div>'
            f'<pre><code class="language-{lang}">{html.escape(text)}</code></pre></div>')



def parse_flow(title, lines):
    """Tiny diagram DSL -> JSON for the React-Flow-style renderer in template.html.

        cfg cw=200 rh=90 nw=160               column width, row height, default node width (px)
        n id col row "title" ["sub"] [k=acc] [w=1.4]
        e a[:side] -> b[:side] ["label"] [k=acc] [anim] [dash]     side: t b l r (auto if omitted)
        g col0 row0 col1 row1 "label" [k=acc]                       group box around a grid range
        cap: caption text (markdown inline allowed)
    """
    spec = {"title": title, "nodes": [], "edges": [], "groups": [], "cfg": {}}
    caption = ""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("cap:"):
            caption = line[4:].strip()
            continue
        tok = shlex.split(line)
        kind, rest = tok[0], tok[1:]
        is_opt = lambda t: re.fullmatch(r"(k|w|cw|rh|nw)=\S+", t) is not None   # only real options, not "i = 2·256"
        opts = {k: v for k, _, v in (t.partition("=") for t in rest if is_opt(t))}
        flags = {t for t in rest if not is_opt(t)}
        if kind == "cfg":
            spec["cfg"] = {k: float(v) for k, v in opts.items()}
        elif kind == "n":
            nid, c, r = rest[0], float(rest[1]), float(rest[2])
            texts = [t for t in rest[3:] if not is_opt(t)]
            spec["nodes"].append({"id": nid, "c": c, "r": r, "t": texts[0] if texts else nid,
                                  "s": texts[1] if len(texts) > 1 else "", "k": opts.get("k", ""),
                                  "w": float(opts.get("w", 1))})
        elif kind == "e":
            a, arrow, b = rest[0], rest[1], rest[2]
            assert arrow == "->", line
            texts = [t for t in rest[3:] if not is_opt(t) and t not in ("anim", "dash")]
            spec["edges"].append({"a": a, "b": b, "t": texts[0] if texts else "", "k": opts.get("k", ""),
                                  "anim": "anim" in flags, "dash": "dash" in flags})
        elif kind == "g":
            c0, r0, c1, r1 = map(float, rest[:4])
            texts = [t for t in rest[4:] if not is_opt(t)]
            spec["groups"].append({"c0": c0, "r0": r0, "c1": c1, "r1": r1, "t": texts[0] if texts else "",
                                   "k": opts.get("k", "")})
        else:
            raise SystemExit(f"flow: unknown line {line!r}")
    ids = {n["id"] for n in spec["nodes"]}
    cw, rh, nw = spec["cfg"].get("cw", 200), spec["cfg"].get("rh", 88), spec["cfg"].get("nw", 160)
    boxes = [(n["id"], n["c"] * cw, n["r"] * rh, n["c"] * cw + nw * n["w"], n["r"] * rh + (52 if n["s"] else 38)) for n in spec["nodes"]]
    for i, A in enumerate(boxes):                       # layout lint: nodes must not overlap (12 px gap)
        for B in boxes[i + 1:]:
            if A[1] < B[3] + 12 and B[1] < A[3] + 12 and A[2] < B[4] + 8 and B[2] < A[4] + 8:
                raise SystemExit(f"flow {title!r}: nodes {A[0]!r} and {B[0]!r} overlap")
    for e in spec["edges"]:
        for end in (e["a"], e["b"]):
            if end.split(":")[0] not in ids:
                raise SystemExit(f"flow {title!r}: edge references unknown node {end!r}")
    cap = MD.renderInline(SRC.render(caption, LINK_ERRORS)) if caption else ""
    return (f'<figure class="flow"><div class="flow-head">{html.escape(title)}<span class="flow-hint">悬停节点查看连接</span></div>'
            f'<div class="flow-body" data-flow="{html.escape(json.dumps(spec, ensure_ascii=False))}"></div>'
            + (f"<figcaption>{cap}</figcaption>" if cap else "") + "</figure>")


def md(text: str) -> str:
    out = MD.render(SRC.render(text, LINK_ERRORS))
    out = out.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    return out


def render(lines) -> str:
    parts, buf = [], []

    def flush():
        if buf:
            parts.append(md("\n".join(buf)))
            buf.clear()

    for b in split_blocks(lines):
        if isinstance(b, str):
            m = CODE_RE.match(b)
            if m:
                flush()
                parts.append(code_block(m.group(1), m.group(2)))
            else:
                buf.append(b)
            continue
        flush()
        name, args, inner = b
        if name == "raw":
            parts.append("\n".join(inner))
        elif name == "hint":
            lvl, _, title = args.partition(" ")
            parts.append(f'<details class="hint l{lvl}"><summary><span class="lvl">L{lvl}</span>{html.escape(title)}'
                         f'<span class="spoiler-note">{HINT_NOTES.get(lvl, "")}</span></summary>'
                         f'<div class="body">{render(inner)}</div></details>')
        elif name == "solution":
            parts.append(f'<details class="solution"><summary><span class="lvl">解</span>{html.escape(args or "参考解")}'
                         f'<span class="spoiler-note">先自己写完、跑过测试再看</span></summary>'
                         f'<div class="body">{render(inner)}</div></details>')
        elif name == "answer":
            parts.append(f'<details class="answer"><summary>{html.escape(args or "显示答案")}</summary>'
                         f'<div class="body">{render(inner)}</div></details>')
        elif name == "flow":
            parts.append(parse_flow(args, inner))
        elif name == "callout":
            kind, _, title = args.partition(" ")
            label = "类比 · " if kind == "analogy" else ""
            t = f'<div class="callout-title">{label}{html.escape(title)}</div>' if (title or label) else ""
            parts.append(f'<div class="callout {kind}">{t}{render(inner)}</div>')
        else:
            raise SystemExit(f"unknown container ::: {name}")
    flush()
    return "\n".join(parts)


def read_src(name):
    lines = (CONTENT / f"{name}.md").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("lede:"), f"{name}.md must start with 'lede:'"
    return lines[0][5:].strip(), lines[1:]


def with_ids(body: str, prefix: str):
    toc = []

    def sub(m):
        toc.append((f"{prefix}-{len(toc) + 1}", html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))))
        return f'<h3 id="{toc[-1][0]}">{m.group(1)}</h3>'

    return re.sub(r"<h2>(.*?)</h2>", sub, body), toc


def basics_section(name, idx):
    lede, lines = read_src(name)
    body, toc = with_ids(render(lines), name)
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("# ") else name
    if lines and lines[0].startswith("# "):
        body = body.split("</h1>", 1)[1] if "</h1>" in body else body
    sub = "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in toc)
    return (f'<section class="part" id="{name}"><div class="kicker">Part 0 · 基础 {idx}</div>'
            f'<h2>{html.escape(title)}</h2><p class="lede">{html.escape(lede)}</p>'
            + (f'<nav class="mini-toc"><ol>{sub}</ol></nav>' if toc else "") + body + "</section>"), title, toc


def hint_badge(did, stem):
    """Link to hints/<stem>.html (or hints/<id>_*.html for lab pages) when it exists."""
    hits = [ROOT / "hints" / f"{stem}.html"] if stem else sorted((ROOT / "hints").glob(f"{did.lower()}_*.html"))
    hits = [h for h in hits if h.exists()]
    return f'<span class="badge">📖 <a href="hints/{hits[0].name}">单独讲解页</a></span>' if hits else ""


def drill_section(k):
    did, stem, title, mins, diff, track = DRILLS[k]
    lede, lines = read_src(did.lower())
    body, toc = with_ids(render(lines), did.lower())
    ext = ".py"
    sub = "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in toc)
    if stem:
        files = (f'<span class="badge">你写 <code>mine/{stem}{ext}</code></span>'
                 f'<span class="badge">验收 <code>./check.sh {did.lower()}</code></span>'
                 f'<span class="badge">参考解 <code>solutions/{stem}{ext}</code></span>')
    else:
        files = '<span class="badge">实践 · 在 Triton / Luminal 仓库里动手，没有 check.sh</span>'
    head = f"""<div class="drill-head">
<div class="kicker">{TRACKS[track][2]} · {TRACKS[track][0]} · 第 {k + 1} 题</div>
<h2><span class="did">{did}</span>{html.escape(title)}</h2>
<p class="lede">{html.escape(lede)}</p>
<div class="badges">
<span class="badge">⏱ <b>{mins} min</b></span>
<span class="badge diff-{diff}">难度 <b>{diff}</b></span>
{files}
{hint_badge(did, stem)}
<label class="badge done-toggle"><input type="checkbox" class="done-box" data-id="{did}">标记完成</label>
</div>
<nav class="mini-toc"><ol>{sub}</ol></nav>
</div>"""
    return f'<section class="drill" id="{did.lower()}" data-id="{did}">{head}{body}</section>', toc


def page():
    parts, nav = [], []
    # Part 0
    nav.append('<div class="nav-tier">Part 0</div><div class="nav-group">基础与全景</div>')
    for i, name in enumerate(BASICS):
        html_, title, _ = basics_section(name, i)
        parts.append(html_)
        nav.append(f'<a class="nav-item" href="#{name}">{html.escape(title)}</a>')
    # drills
    tier_seen = set()
    for t, (tname, tdesc, tier) in TRACKS.items():
        if tier not in tier_seen:
            tier_seen.add(tier)
            nav.append(f'<div class="nav-tier">{tier}</div>')
            parts.append(f'<div class="tier-banner" id="tier-{tier}">{tier}</div>')
        nav.append(f'<div class="nav-group">{html.escape(tname)}</div>')
        parts.append(f'<div class="track-banner" id="track-{t}"><h2>{html.escape(tname)}</h2><p>{html.escape(tdesc)}</p></div>')
        for k, d in enumerate(DRILLS):
            if d[5] != t:
                continue
            sec, _ = drill_section(k)
            parts.append(sec)
            nav.append(f'<a class="nav-item drill-link" href="#{d[0].lower()}" data-id="{d[0]}">'
                       f'<span class="nid">{d[0]}</span>{html.escape(d[2].split("：")[0])}<span class="nmin">{d[3]}′</span></a>')
    html_, title, _ = basics_section("plan14", "")
    parts.append(html_.replace("Part 0 · 基础 ", "计划"))
    nav.append(f'<div class="nav-tier">收尾</div><a class="nav-item" href="#plan14">{html.escape(title)}</a>')
    lede, lines = read_src("appendix")
    app_body, _ = with_ids(render(lines), "appendix")
    parts.append(f'<section class="part" id="appendix"><div class="kicker">附录</div><h2>附录：资源、PR 模板与参考资料</h2>'
                 f'<p class="lede">{html.escape(lede)}</p>{app_body}</section>')
    nav.append('<a class="nav-item" href="#appendix">附录：资源与参考资料</a>')

    total = sum(d[3] for d in DRILLS)
    template = (ROOT / "build" / "template.html").read_text(encoding="utf-8")
    return (template.replace("{{NAV}}", "\n".join(nav))
            .replace("{{BODY}}", "\n".join(parts))
            .replace("{{N_DRILLS}}", str(len(DRILLS)))
            .replace("{{TOTAL_H}}", f"{total // 60} 小时 {total % 60} 分"))


if __name__ == "__main__":
    out = ROOT / "index.html"
    html_out = page()
    if LINK_ERRORS:
        raise SystemExit("unresolved source links:\n  " + "\n  ".join(sorted(set(LINK_ERRORS))))
    out.write_text(html_out, encoding="utf-8")
    print("built", out, f"{out.stat().st_size // 1024} KB")
