"""Render AGENIX-完整测试报告.md (v9-only) into a self-contained, browser-openable HTML.

Variant of make_html_report.py pointing at the v9-only full report. Same approach:
- Charts are base64-embedded (data: URIs) so the file is portable (no broken image paths).
- GFM tables via the `markdown` package.
- MathJax from CDN as progressive enhancement; UTF-8 charset (Chinese).

The source MD / output HTML are overridable via argv or env (REPORT_MD / REPORT_HTML)
so this does not touch the original AGENIX-评测报告 pipeline.
"""
from __future__ import annotations

import base64
import os
import re
import sys

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = (
    (sys.argv[1] if len(sys.argv) > 1 else None)
    or os.environ.get("REPORT_MD")
    or os.path.join(ROOT, "AGENIX-完整测试报告.md")
)
HTML_PATH = (
    (sys.argv[2] if len(sys.argv) > 2 else None)
    or os.environ.get("REPORT_HTML")
    or os.path.splitext(MD_PATH)[0] + ".html"
)

IMG_RE = re.compile(r"!\[([^\]]*)\]\((engine/results/figs/[^)]+\.png)\)")


def embed_images(md_text: str) -> tuple[str, int]:
    n = 0

    def repl(m):
        nonlocal n
        alt, rel = m.group(1), m.group(2)
        png = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.exists(png):
            return m.group(0)  # leave as-is if missing
        with open(png, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        n += 1
        return f"![{alt}](data:image/png;base64,{b64})"

    return IMG_RE.sub(repl, md_text), n


CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Segoe UI", "Microsoft YaHei", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.65; color: #1f2328; background: #ffffff;
  max-width: 920px; margin: 0 auto; padding: 40px 24px 80px;
}
h1 { font-size: 26px; line-height: 1.3; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }
h2 { font-size: 21px; margin-top: 38px; border-bottom: 1px solid #eef2f6; padding-bottom: 6px; }
h3 { font-size: 17px; margin-top: 26px; }
h4 { font-size: 15px; margin-top: 20px; color: #334155; }
p, li { font-size: 15px; }
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
img { max-width: 100%; height: auto; display: block; margin: 16px auto; border: 1px solid #eef2f6; border-radius: 6px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; display: block; overflow-x: auto; }
th, td { border: 1px solid #e2e8f0; padding: 7px 11px; text-align: left; vertical-align: top; }
th { background: #f8fafc; font-weight: 600; }
tr:nth-child(even) td { background: #fbfcfe; }
blockquote { margin: 14px 0; padding: 8px 16px; border-left: 4px solid #cbd5e1; background: #f8fafc; color: #475569; }
code { background: #f1f5f9; padding: 1px 6px; border-radius: 4px; font-size: 13px;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }
pre { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px 14px; overflow-x: auto; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 32px 0; }
strong { color: #0f172a; }
@media (prefers-color-scheme: dark) {
  body { color: #d6dee6; background: #0d1117; }
  h1, strong { color: #f0f6fc; } h4 { color: #aeb9c5; }
  h1 { border-bottom-color: #232b34; } h2 { border-bottom-color: #1b222a; }
  th { background: #161b22; } td, th { border-color: #232b34; }
  tr:nth-child(even) td { background: #11161d; }
  blockquote { background: #11161d; border-left-color: #30363d; color: #9aa7b3; }
  code, pre { background: #161b22; border-color: #232b34; }
}
"""

MATHJAX = """
<script>
window.MathJax = { tex: { inlineMath: [['$','$'], ['\\\\(','\\\\)']],
  displayMath: [['$$','$$'], ['\\\\[','\\\\]']] }, options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] } };
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""


def main():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()
    md_text, n_img = embed_images(md_text)
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        output_format="html5",
    )
    html = (
        "<!DOCTYPE html>\n<html lang=\"zh\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>AGENIX 完整测试报告 (v9) — seed (doubao-seed-evolving)</title>\n"
        f"<style>{CSS}</style>\n{MATHJAX}</head>\n<body>\n{body}\n</body>\n</html>\n"
    )
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {HTML_PATH}")
    print(f"embedded {n_img} chart image(s) as base64")
    print(f"html size: {os.path.getsize(HTML_PATH)/1024:.1f} KB")


if __name__ == "__main__":
    main()
