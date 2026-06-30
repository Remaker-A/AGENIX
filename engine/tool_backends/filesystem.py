"""Restricted filesystem tool backend.

The backend seeds a small file tree from ``state["fs"]["files"]`` (or
``state["filesystem"]["files"]``), executes read/write/list/diff/verify
operations under an isolated root, and writes structured observable results
back into state with signed provenance.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from schema import StateDiff
from tool_backends.base import BackendContext, BackendResult, ToolBackend, register_backend


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


def _safe_rel(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("filesystem path must be a non-empty relative string")
    path = path.replace("\\", "/").lstrip("/")
    norm = os.path.normpath(path).replace("\\", "/")
    if norm in ("", ".") or norm.startswith("../") or norm == ".." or os.path.isabs(norm):
        raise ValueError("path escapes filesystem root: %r" % path)
    return norm


def _jsonable_content(content: str) -> Any:
    try:
        return json.loads(content)
    except Exception:  # noqa: BLE001 - plain text is a valid file payload
        return content


class FilesystemBackend(ToolBackend):
    """Read and write files below a sandbox-owned root directory."""

    name = "filesystem"
    deterministic = True
    requires_workdir = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._root: Optional[str] = None
        self._owns_root = False
        self._seeded = False

    def teardown(self) -> None:
        if self._owns_root and self._root and os.path.isdir(self._root):
            shutil.rmtree(self._root, ignore_errors=True)

    def _ensure_root(self, ctx: BackendContext) -> str:
        if self._root is None:
            if self.workdir:
                self._root = os.path.join(self.workdir, "filesystem")
                os.makedirs(self._root, exist_ok=True)
            else:
                self._root = tempfile.mkdtemp(prefix="agenix-fs-")
                self._owns_root = True
        if not self._seeded:
            files = self._seed_files(ctx)
            for rel, content in sorted(files.items()):
                safe = _safe_rel(rel)
                abs_path = os.path.join(self._root, safe)
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, "w", encoding="utf-8", newline="") as f:
                    f.write(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
            self._seeded = True
        return self._root

    def _seed_files(self, ctx: BackendContext) -> Dict[str, Any]:
        for key in ("fs", "filesystem"):
            node = ctx.state.get(key)
            if isinstance(node, dict) and isinstance(node.get("files"), dict):
                return dict(node["files"])
        seed = _extra(ctx.tool, "seed_files", {})
        return dict(seed) if isinstance(seed, dict) else {}

    def _snapshot_files(self, root: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for base, _dirs, files in os.walk(root):
            for fn in sorted(files):
                abs_path = os.path.join(base, fn)
                rel = os.path.relpath(abs_path, root).replace("\\", "/")
                with open(abs_path, "r", encoding="utf-8") as f:
                    out[rel] = f.read()
        return out

    def _write_state(self, ctx: BackendContext, path: str, value: Any,
                     diffs: List[StateDiff], source: str) -> None:
        norm = _state_path(path)
        if ctx.set_path is None:
            raise RuntimeError("filesystem backend requires ctx.set_path")
        ctx.set_path(ctx.state, norm, value)
        diffs.append(StateDiff(path=norm, new_value=value, provenance=source))

    def _observable_path(self, ctx: BackendContext) -> str:
        paths = list(getattr(ctx.tool, "observable_result_paths", []) or [])
        return _state_path(paths[0]) if paths else "observations.%s" % ctx.tool.name

    def _apply_signed_provenance(self, ctx: BackendContext, diffs: List[StateDiff],
                                 extra: Any) -> str:
        signature = self.sign(ctx, diffs, extra=extra)
        signed = "%s#%s" % (ctx.source, signature)
        for diff in diffs:
            diff.provenance = signed
        return signature

    def execute(self, ctx: BackendContext) -> BackendResult:
        try:
            root = self._ensure_root(ctx)
            op = str(ctx.args.get("op") or _extra(ctx.tool, "op", "") or "").lower()
            if not op:
                op = "write" if "content" in ctx.args else "read"
            if op == "read":
                return self._read(ctx, root)
            if op == "list":
                return self._list(ctx, root)
            if op == "write":
                return self._write(ctx, root)
            if op == "diff":
                return self._diff(ctx, root)
            if op == "verify":
                return self._verify(ctx, root)
            raise ValueError("unsupported filesystem op: %s" % op)
        except Exception as exc:  # noqa: BLE001
            obs = {"backend": self.name, "status": "error", "error": str(exc)}
            return BackendResult(diffs=[], observation=obs, status="error", error=str(exc),
                                 event_signature=self.sign(ctx, [], extra=obs))

    def _read(self, ctx: BackendContext, root: str) -> BackendResult:
        rel = _safe_rel(ctx.args.get("path") or _extra(ctx.tool, "path", ""))
        abs_path = os.path.join(root, rel)
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        obs = {
            "backend": self.name,
            "op": "read",
            "path": rel,
            "content": content,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "size": len(content),
        }
        diffs: List[StateDiff] = []
        self._write_state(ctx, self._observable_path(ctx), obs, diffs, ctx.source)
        signature = self._apply_signed_provenance(ctx, diffs, obs)
        return BackendResult(diffs=diffs, observation=obs, event_signature=signature,
                             raw_response=obs, replay_key=ctx.replay_key())

    def _list(self, ctx: BackendContext, root: str) -> BackendResult:
        prefix = ctx.args.get("path") or _extra(ctx.tool, "path", ".")
        safe_prefix = "." if prefix in ("", ".") else _safe_rel(prefix)
        base = root if safe_prefix == "." else os.path.join(root, safe_prefix)
        entries: List[str] = []
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                entries.append((safe_prefix.rstrip("/") + "/" + name).lstrip("./"))
        obs = {"backend": self.name, "op": "list", "path": safe_prefix, "entries": entries}
        diffs: List[StateDiff] = []
        self._write_state(ctx, self._observable_path(ctx), obs, diffs, ctx.source)
        signature = self._apply_signed_provenance(ctx, diffs, obs)
        return BackendResult(diffs=diffs, observation=obs, event_signature=signature,
                             raw_response=obs, replay_key=ctx.replay_key())

    def _write(self, ctx: BackendContext, root: str) -> BackendResult:
        if not any(p in (ctx.tool.permissions or []) for p in ("fs:write", "filesystem:write", "write")):
            raise PermissionError("tool lacks fs:write permission")
        rel = _safe_rel(ctx.args.get("path") or _extra(ctx.tool, "path", ""))
        content = ctx.args.get("content")
        if content is None:
            content = ctx.args.get("value", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        abs_path = os.path.join(root, rel)
        before = ""
        if os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8") as f:
                before = f.read()
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        patch = "\n".join(difflib.unified_diff(before.splitlines(), content.splitlines(),
                                               fromfile=rel + ":before",
                                               tofile=rel + ":after",
                                               lineterm=""))
        obs = {
            "backend": self.name,
            "op": "write",
            "path": rel,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "diff": patch,
        }
        diffs: List[StateDiff] = []
        self._write_state(ctx, "fs.files", self._snapshot_files(root), diffs, ctx.source)
        self._write_state(ctx, self._observable_path(ctx), obs, diffs, ctx.source)
        state_path = ctx.args.get("state_path") or _extra(ctx.tool, "state_path", None)
        if state_path:
            value = ctx.args.get("state_value", _jsonable_content(content))
            self._write_state(ctx, state_path, value, diffs, ctx.source)
        signature = self._apply_signed_provenance(ctx, diffs, obs)
        return BackendResult(diffs=diffs, observation=obs, event_signature=signature,
                             raw_response=obs, replay_key=ctx.replay_key())

    def _diff(self, ctx: BackendContext, root: str) -> BackendResult:
        rel = _safe_rel(ctx.args.get("path") or _extra(ctx.tool, "path", ""))
        expected = ctx.args.get("expected", _extra(ctx.tool, "expected", ""))
        with open(os.path.join(root, rel), "r", encoding="utf-8") as f:
            current = f.read()
        patch = "\n".join(difflib.unified_diff(str(expected).splitlines(), current.splitlines(),
                                               fromfile=rel + ":expected",
                                               tofile=rel + ":actual",
                                               lineterm=""))
        obs = {"backend": self.name, "op": "diff", "path": rel, "diff": patch,
               "matches": str(expected) == current}
        diffs: List[StateDiff] = []
        self._write_state(ctx, self._observable_path(ctx), obs, diffs, ctx.source)
        signature = self._apply_signed_provenance(ctx, diffs, obs)
        return BackendResult(diffs=diffs, observation=obs, event_signature=signature,
                             raw_response=obs, replay_key=ctx.replay_key())

    def _verify(self, ctx: BackendContext, root: str) -> BackendResult:
        rel = _safe_rel(ctx.args.get("path") or _extra(ctx.tool, "path", ""))
        with open(os.path.join(root, rel), "r", encoding="utf-8") as f:
            current = f.read()
        expected_hash = ctx.args.get("sha256") or _extra(ctx.tool, "sha256", None)
        expected_content = ctx.args.get("content") or _extra(ctx.tool, "content", None)
        actual_hash = hashlib.sha256(current.encode("utf-8")).hexdigest()
        ok = True
        if expected_hash:
            ok = ok and actual_hash == expected_hash
        if expected_content is not None:
            ok = ok and current == str(expected_content)
        obs = {"backend": self.name, "op": "verify", "path": rel, "ok": bool(ok),
               "sha256": actual_hash}
        diffs: List[StateDiff] = []
        self._write_state(ctx, self._observable_path(ctx), obs, diffs, ctx.source)
        signature = self._apply_signed_provenance(ctx, diffs, obs)
        return BackendResult(diffs=diffs, observation=obs, event_signature=signature,
                             raw_response=obs, replay_key=ctx.replay_key())


register_backend("filesystem", FilesystemBackend, override=True)
