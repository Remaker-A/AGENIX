"""SQLite tool backend with deterministic seed data and transaction logs."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Dict, List, Optional, Sequence

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


def _quote_ident(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("invalid sqlite identifier")
    return '"' + name.replace('"', '""') + '"'


def _sqlite_type(value: Any) -> str:
    if isinstance(value, bool) or isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


class SQLiteBackend(ToolBackend):
    """Execute SQL against a seed database and expose rows/logs as StateDiffs."""

    name = "sqlite"
    deterministic = True
    requires_workdir = False

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._conn: Optional[sqlite3.Connection] = None
        self._seeded = False
        self._tx_log: List[Dict[str, Any]] = []

    def teardown(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_conn(self, ctx: BackendContext) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(":memory:")
            self._conn.row_factory = sqlite3.Row
        if not self._seeded:
            self._seed(ctx)
            self._seeded = True
        return self._conn

    def _seed_tables(self, ctx: BackendContext) -> Dict[str, List[Dict[str, Any]]]:
        for key in ("sqlite", "db"):
            node = ctx.state.get(key)
            if isinstance(node, dict) and isinstance(node.get("tables"), dict):
                return {str(k): list(v or []) for k, v in node["tables"].items()}
        seed = _extra(ctx.tool, "seed_tables", {})
        if isinstance(seed, dict):
            return {str(k): list(v or []) for k, v in seed.items()}
        return {}

    def _seed(self, ctx: BackendContext) -> None:
        conn = self._conn
        assert conn is not None
        for table, rows in sorted(self._seed_tables(ctx).items()):
            if not rows:
                continue
            columns: List[str] = []
            for row in rows:
                if isinstance(row, dict):
                    for key in row:
                        if key not in columns:
                            columns.append(str(key))
            if not columns:
                continue
            types = {c: "TEXT" for c in columns}
            for row in rows:
                for c in columns:
                    if c in row:
                        types[c] = _sqlite_type(row[c])
                        break
            ddl = "CREATE TABLE %s (%s)" % (
                _quote_ident(table),
                ", ".join("%s %s" % (_quote_ident(c), types[c]) for c in columns),
            )
            conn.execute(ddl)
            placeholders = ", ".join("?" for _ in columns)
            insert = "INSERT INTO %s (%s) VALUES (%s)" % (
                _quote_ident(table),
                ", ".join(_quote_ident(c) for c in columns),
                placeholders,
            )
            for row in rows:
                values = [row.get(c) for c in columns]
                conn.execute(insert, values)
        conn.commit()

    def _observable_path(self, ctx: BackendContext) -> str:
        paths = list(getattr(ctx.tool, "observable_result_paths", []) or [])
        return _state_path(paths[0]) if paths else "observations.%s" % ctx.tool.name

    def _write_state(self, ctx: BackendContext, path: str, value: Any,
                     diffs: List[StateDiff], source: str) -> None:
        if ctx.set_path is None:
            raise RuntimeError("sqlite backend requires ctx.set_path")
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

    def _params(self, ctx: BackendContext) -> Any:
        params = ctx.args.get("params", _extra(ctx.tool, "params", []))
        if isinstance(params, (list, tuple, dict)):
            return params
        raise ValueError("sqlite params must be a list, tuple, or dict")

    def execute(self, ctx: BackendContext) -> BackendResult:
        try:
            conn = self._ensure_conn(ctx)
            sql = str(ctx.args.get("sql") or _extra(ctx.tool, "sql", "")).strip()
            if not sql:
                raise ValueError("sqlite sql is required")
            op = str(ctx.args.get("op") or _extra(ctx.tool, "op", "") or "").lower()
            is_select = sql.lstrip().lower().startswith(("select", "with", "pragma"))
            if not op:
                op = "query" if is_select else "execute"
            if op == "query":
                return self._query(ctx, conn, sql)
            if op == "execute":
                return self._execute_sql(ctx, conn, sql)
            raise ValueError("unsupported sqlite op: %s" % op)
        except Exception as exc:  # noqa: BLE001
            obs = {"backend": self.name, "status": "error", "error": str(exc)}
            return BackendResult(diffs=[], observation=obs, status="error", error=str(exc),
                                 event_signature=self.sign(ctx, [], extra=obs))

    def _query(self, ctx: BackendContext, conn: sqlite3.Connection, sql: str) -> BackendResult:
        cur = conn.execute(sql, self._params(ctx))
        rows = [dict(row) for row in cur.fetchall()]
        columns = [d[0] for d in (cur.description or [])]
        obs = {
            "backend": self.name,
            "op": "query",
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "result_hash": hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()[:16],
        }
        diffs: List[StateDiff] = []
        self._write_state(ctx, self._observable_path(ctx), obs, diffs, ctx.source)
        signature = self._apply_signed_provenance(ctx, diffs, obs)
        return BackendResult(diffs=diffs, observation=obs, event_signature=signature,
                             raw_response=obs, replay_key=ctx.replay_key())

    def _execute_sql(self, ctx: BackendContext, conn: sqlite3.Connection, sql: str) -> BackendResult:
        if not any(p in (ctx.tool.permissions or []) for p in ("db:write", "sqlite:write", "write")):
            raise PermissionError("tool lacks db:write permission")
        cur = conn.execute(sql, self._params(ctx))
        conn.commit()
        rec = {
            "sql_hash": hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16],
            "rowcount": cur.rowcount,
        }
        self._tx_log.append(rec)
        obs = {
            "backend": self.name,
            "op": "execute",
            "sql": sql,
            "rowcount": cur.rowcount,
            "transaction_log": list(self._tx_log),
            "transaction_log_hash": hashlib.sha256(canonical_json(self._tx_log).encode("utf-8")).hexdigest()[:16],
        }
        diffs: List[StateDiff] = []
        self._write_state(ctx, "sqlite.transaction_log", list(self._tx_log), diffs, ctx.source)
        self._write_state(ctx, self._observable_path(ctx), obs, diffs, ctx.source)
        signature = self._apply_signed_provenance(ctx, diffs, obs)
        return BackendResult(diffs=diffs, observation=obs, event_signature=signature,
                             raw_response=obs, replay_key=ctx.replay_key())


register_backend("sqlite", SQLiteBackend, override=True)
