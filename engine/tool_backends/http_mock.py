"""Record/replay HTTP mock backend.

No real network calls are made. Routes are seeded from task state and resolved
through the shared RecordReplayStore in record/replay modes.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

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


class HttpMockBackend(ToolBackend):
    """Serve deterministic mock HTTP responses and log requests."""

    name = "http_mock"
    deterministic = True
    requires_workdir = False

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._calls: List[Dict[str, Any]] = []

    def _routes(self, ctx: BackendContext) -> Dict[str, Any]:
        node = ctx.state.get("http")
        if isinstance(node, dict) and isinstance(node.get("routes"), dict):
            return dict(node["routes"])
        routes = _extra(ctx.tool, "routes", {})
        return dict(routes) if isinstance(routes, dict) else {}

    def _request(self, ctx: BackendContext) -> Tuple[str, str, Any]:
        method = str(ctx.args.get("method") or _extra(ctx.tool, "method", "GET")).upper()
        url = str(ctx.args.get("url") or ctx.args.get("path") or _extra(ctx.tool, "url", "/"))
        body = ctx.args.get("body")
        return method, url, body

    def _lookup(self, ctx: BackendContext, method: str, url: str) -> Dict[str, Any]:
        routes = self._routes(ctx)
        keys = ["%s %s" % (method, url), "%s:%s" % (method, url), url]
        if url.startswith("/"):
            keys.append(url[1:])
        for key in keys:
            if key in routes:
                resp = routes[key]
                if isinstance(resp, dict):
                    return dict(resp)
                return {"status": 200, "json": resp}
        return {"status": 404, "json": {"error": "mock route not found", "url": url}}

    def _observable_path(self, ctx: BackendContext) -> str:
        paths = list(getattr(ctx.tool, "observable_result_paths", []) or [])
        return _state_path(paths[0]) if paths else "observations.%s" % ctx.tool.name

    def _write_state(self, ctx: BackendContext, path: str, value: Any,
                     diffs: List[StateDiff], source: str) -> None:
        if ctx.set_path is None:
            raise RuntimeError("http_mock backend requires ctx.set_path")
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
            method, url, body = self._request(ctx)
            key = canonical_json({
                "backend": self.name,
                "method": method,
                "url": url,
                "body": body,
                "seed": ctx.seed,
            })

            def live() -> Dict[str, Any]:
                return self._lookup(ctx, method, url)

            response = self.resolve(key, live)
            if not isinstance(response, dict):
                response = {"status": 200, "json": response}
            call = {
                "method": method,
                "url": url,
                "body_hash": hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()[:16],
                "status": int(response.get("status", 200)),
            }
            self._calls.append(call)
            obs = {
                "backend": self.name,
                "op": "request",
                "method": method,
                "url": url,
                "status": int(response.get("status", 200)),
                "headers": dict(response.get("headers") or {}),
                "json": response.get("json"),
                "text": response.get("text"),
                "response_hash": hashlib.sha256(canonical_json(response).encode("utf-8")).hexdigest()[:16],
                "replay_key": key,
            }
            diffs: List[StateDiff] = []
            self._write_state(ctx, self._observable_path(ctx), obs, diffs, ctx.source)
            self._write_state(ctx, "http.calls", list(self._calls), diffs, ctx.source)
            signature = self._apply_signed_provenance(ctx, diffs, obs)
            return BackendResult(diffs=diffs, observation=obs, event_signature=signature,
                                 raw_response=response, replay_key=key)
        except Exception as exc:  # noqa: BLE001
            obs = {"backend": self.name, "status": "error", "error": str(exc)}
            return BackendResult(diffs=[], observation=obs, status="error", error=str(exc),
                                 event_signature=self.sign(ctx, [], extra=obs))


register_backend("http_mock", HttpMockBackend, override=True)
