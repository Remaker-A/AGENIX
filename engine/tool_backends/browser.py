"""Static browser backend.

This backend intentionally does not require Playwright. It loads deterministic
HTML pages from task state and returns a compact DOM snapshot, text extract,
links, and simple selector matches.
"""
from __future__ import annotations

import hashlib
import html
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from schema import StateDiff
from tool_backends.base import BackendContext, BackendResult, ToolBackend, canonical_json, register_backend


def _extra(obj: Any, name: str, default: Any = None) -> Any:
    val = getattr(obj, name, default)
    if val is not default:
        return val
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name, default)
    return default


def _state_path(path: str) -> str:
    return path[len("state."):] if path.startswith("state.") else path


class _DomParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.nodes: List[Dict[str, Any]] = []
        self.text_parts: List[str] = []
        self.links: List[Dict[str, str]] = []
        self._stack: List[int] = []

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        attr = {str(k): str(v or "") for k, v in attrs}
        node = {"tag": tag.lower(), "attrs": attr, "text": ""}
        self.nodes.append(node)
        self._stack.append(len(self.nodes) - 1)
        if tag.lower() == "a" and attr.get("href"):
            self.links.append({"href": attr.get("href", ""), "text": ""})

    def handle_endtag(self, tag: str) -> None:
        if self._stack:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        text = " ".join((data or "").split())
        if not text:
            return
        self.text_parts.append(text)
        if self._stack:
            node = self.nodes[self._stack[-1]]
            node["text"] = (node.get("text", "") + " " + text).strip()
        if self.links:
            self.links[-1]["text"] = (self.links[-1].get("text", "") + " " + text).strip()


def _parse(html_text: str) -> _DomParser:
    parser = _DomParser()
    parser.feed(html_text or "")
    return parser


def _title(html_text: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text or "", flags=re.I | re.S)
    return html.unescape(" ".join(m.group(1).split())) if m else ""


def _matches_selector(node: Dict[str, Any], selector: str) -> bool:
    selector = (selector or "").strip()
    if not selector:
        return False
    attrs = node.get("attrs", {})
    if selector.startswith("#"):
        return attrs.get("id") == selector[1:]
    if selector.startswith("."):
        return selector[1:] in (attrs.get("class", "").split())
    return node.get("tag") == selector.lower()


class BrowserBackend(ToolBackend):
    """Return deterministic snapshots for static HTML pages."""

    name = "browser"
    deterministic = True
    requires_workdir = False

    def _pages(self, ctx: BackendContext) -> Dict[str, str]:
        for key in ("browser", "dom"):
            node = ctx.state.get(key)
            if isinstance(node, dict) and isinstance(node.get("pages"), dict):
                return {str(k): str(v) for k, v in node["pages"].items()}
        pages = _extra(ctx.tool, "pages", {})
        return {str(k): str(v) for k, v in pages.items()} if isinstance(pages, dict) else {}

    def _observable_path(self, ctx: BackendContext) -> str:
        paths = list(getattr(ctx.tool, "observable_result_paths", []) or [])
        return _state_path(paths[0]) if paths else "observations.%s" % ctx.tool.name

    def _write_state(self, ctx: BackendContext, path: str, value: Any,
                     diffs: List[StateDiff], source: str) -> None:
        if ctx.set_path is None:
            raise RuntimeError("browser backend requires ctx.set_path")
        norm = _state_path(path)
        ctx.set_path(ctx.state, norm, value)
        diffs.append(StateDiff(path=norm, new_value=value, provenance=source))

    def _apply_signed_provenance(self, ctx: BackendContext, diffs: List[StateDiff],
                                 extra: Any) -> str:
        signature = self.sign(ctx, diffs, extra=extra)
        signed = "%s#%s" % (ctx.source, signature)
        for diff in diffs:
            diff.provenance = signed
        return signature

    def execute(self, ctx: BackendContext) -> BackendResult:
        try:
            url = str(ctx.args.get("url") or ctx.args.get("path") or _extra(ctx.tool, "url", "") or "/")
            pages = self._pages(ctx)
            html_text = pages.get(url)
            if html_text is None and url.startswith("/"):
                html_text = pages.get(url[1:])
            if html_text is None:
                raise FileNotFoundError("no static page for url: %s" % url)
            parser = _parse(html_text)
            selector = ctx.args.get("selector") or _extra(ctx.tool, "selector", None)
            matches = []
            if selector:
                for node in parser.nodes:
                    if _matches_selector(node, str(selector)):
                        matches.append({"tag": node["tag"], "attrs": node["attrs"], "text": node.get("text", "")})
            text = " ".join(parser.text_parts)
            obs = {
                "backend": self.name,
                "op": str(ctx.args.get("op") or _extra(ctx.tool, "op", "snapshot")),
                "url": url,
                "title": _title(html_text),
                "text": text[:2000],
                "links": parser.links[:50],
                "selector": selector,
                "matches": matches[:50],
                "dom_hash": hashlib.sha256(canonical_json({
                    "url": url,
                    "title": _title(html_text),
                    "text": text,
                    "links": parser.links,
                    "matches": matches,
                }).encode("utf-8")).hexdigest()[:16],
            }
            diffs: List[StateDiff] = []
            self._write_state(ctx, self._observable_path(ctx), obs, diffs, ctx.source)
            signature = self._apply_signed_provenance(ctx, diffs, obs)
            return BackendResult(diffs=diffs, observation=obs, event_signature=signature,
                                 raw_response=obs, replay_key=ctx.replay_key())
        except Exception as exc:  # noqa: BLE001
            obs = {"backend": self.name, "status": "error", "error": str(exc)}
            return BackendResult(diffs=[], observation=obs, status="error", error=str(exc),
                                 event_signature=self.sign(ctx, [], extra=obs))


register_backend("browser", BrowserBackend, override=True)
