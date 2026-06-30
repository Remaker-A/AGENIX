"""Render AGENIX-评测报告.md into a fully self-contained, browser-openable HTML.

- Charts are base64-embedded (data: URIs) so the file is portable and has no
  broken relative-path images.
- Uses the stdlib-adjacent `markdown` package (confirmed available) with the
  GFM tables extension.
- MathJax is loaded from CDN as progressive enhancement: online → formulas
  render; offline → they degrade to readable raw TeX. Charset is UTF-8 (Chinese).
"""
from __future__ import annotations

import base64
import html as _html
import os
import re

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(ROOT, "AGENIX-评测报告.md")
HTML_PATH = os.path.join(ROOT, "AGENIX-评测报告.html")

IMG_RE = re.compile(r"!\[([^\]]*)\]\((engine/results/figs/[^)]+\.png)\)")

# Cursor 代码引用块 ```startLine:endLine:filepath 的 info string 含冒号，python-markdown
# 的 fenced_code 正则不认（lang 只允许 [\w#.+-]），会导致围栏错配、吞掉后续 mermaid 块。
# 这些引用在 Cursor 的 .md 预览里渲染良好；仅在生成独立 HTML 时把它规整成合法 ```json + 标题。
CITATION_RE = re.compile(r"^```(\d+):(\d+):(\S+?)\s*$", re.MULTILINE)


def normalize_citations(md_text: str) -> tuple[str, int]:
    n = 0

    def repl(m):
        nonlocal n
        n += 1
        start, end, path = m.group(1), m.group(2), m.group(3)
        return "*摘自 `%s`（L%s–%s）：*\n\n```json" % (path, start, end)

    return CITATION_RE.sub(repl, md_text), n


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


# python-markdown(fenced_code) 把 ```mermaid 渲染成 <pre><code class="language-mermaid">…，
# mermaid.js 默认找 <div class="mermaid">；故把这类代码块转成 div 并反转义实体（-->、引号等）。
MERMAID_RE = re.compile(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>', re.DOTALL)


def convert_mermaid(html_body: str) -> tuple[str, int]:
    n = 0

    def repl(m):
        nonlocal n
        n += 1
        src = _html.unescape(m.group(1))
        return '<div class="mermaid">\n%s\n</div>' % src

    return MERMAID_RE.sub(repl, html_body), n


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

# 在线加载 mermaid 渲染树状图/流程图；离线则 .mermaid 块退化为可读文本。深色模式自适配。
MERMAID_JS = """
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
mermaid.initialize({ startOnLoad: true, securityLevel: 'loose', theme: dark ? 'dark' : 'default' });
</script>
"""

MERMAID_CSS = """
.mermaid { background: transparent; margin: 18px auto; text-align: center; overflow-x: auto; }
"""


def main():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()
    md_text, n_img = embed_images(md_text)
    md_text, n_cite = normalize_citations(md_text)
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        output_format="html5",
    )
    body, n_mmd = convert_mermaid(body)
    html = (
        "<!DOCTYPE html>\n<html lang=\"zh\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>AGENIX 评测报告 — seed (doubao-seed-evolving)</title>\n"
        f"<style>{CSS}{MERMAID_CSS}</style>\n{MATHJAX}{MERMAID_JS}</head>\n<body>\n{body}\n</body>\n</html>\n"
    )
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {HTML_PATH}")
    print(f"embedded {n_img} chart image(s) as base64")
    print(f"normalized {n_cite} code-citation block(s)")
    print(f"converted {n_mmd} mermaid diagram(s)")
    print(f"html size: {os.path.getsize(HTML_PATH)/1024:.1f} KB")


if __name__ == "__main__":
    main()
