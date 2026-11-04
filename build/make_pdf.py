"""Export index.html to ai-compiler-drills.pdf with headless Chrome.

    python3 build/make_pdf.py

Every <details> (hints, solutions, answers) is opened, the light theme is forced, and the
diagrams are rendered before printing. Set CHROME=/path/to/chrome if it is not in the default
macOS location.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

PRINT_PREP = """<style>
@page { size: A4; margin: 12mm 10mm; }
@media print {
  .progress, #prog-text, #prog-reset, .done-toggle, .flow-hint { display: none !important; }
  .page { max-width: none !important; padding: 0 !important; }
  figure.flow, details, table, pre { break-inside: avoid; }
  section.drill, section.part, .track-banner { break-before: page; }
  a.src::after { content: none; }
}
</style>
<script>
document.documentElement.setAttribute("data-theme", "light");
window.addEventListener("load", function () {
  document.querySelectorAll("details").forEach(function (d) { d.open = true; });
});
</script>
</body>"""


def main():
    src = ROOT / "index.html"
    tmp = ROOT / ".print_index.html"
    out = ROOT / "ai-compiler-drills.pdf"
    tmp.write_text(src.read_text(encoding="utf-8").replace("</body>", PRINT_PREP, 1), encoding="utf-8")
    try:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=8000", f"--print-to-pdf={out}", tmp.as_uri()],
                       check=True, capture_output=True)
    finally:
        tmp.unlink(missing_ok=True)
    print("wrote", out, f"{out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
