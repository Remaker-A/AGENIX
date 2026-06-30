"""
Real-media ingest skeleton for U3 grounding assets.

The importer normalizes a hand-labeled manifest into `assets/real/manifest.json`,
optionally copies licensed media into `assets/real/media/`, and emits one typed-GT
annotation file per asset under `assets/real/annotations/`.

It intentionally does not scrape, download, or infer annotations. Promotion to a
trusted real track should happen only after the manifest carries licensed media,
human typed GT, and verifier calibration evidence.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ALLOWED_MEDIA_TYPES = {"image", "document", "table", "webpage", "video"}
DEFAULT_SPLIT = "pilot"
DEFAULT_LICENSE = "unknown"


class RealAssetError(ValueError):
    """Raised when a real-asset manifest cannot be trusted enough to ingest."""


def _repo_engine_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _slug(value: str) -> str:
    out = []
    for ch in str(value).strip().lower():
        out.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(out).strip("_") or "asset"


def load_manifest(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RealAssetError("manifest must be a JSON object")
    return data


def _asset_entries(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    assets = manifest.get("assets", [])
    if not isinstance(assets, list):
        raise RealAssetError("manifest.assets must be a list")
    return assets


def validate_asset(entry: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise RealAssetError("asset entry must be an object")
    aid = _slug(entry.get("id", ""))
    media_type = str(entry.get("media_type", "")).strip().lower()
    if not aid:
        raise RealAssetError("asset entry missing id")
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise RealAssetError("%s has unsupported media_type %r" % (aid, media_type))
    if not entry.get("asset_uri"):
        raise RealAssetError("%s missing asset_uri" % aid)
    typed_gt = entry.get("typed_gt")
    if not isinstance(typed_gt, dict) or not typed_gt:
        raise RealAssetError("%s missing non-empty typed_gt" % aid)

    normalized = dict(entry)
    normalized["id"] = aid
    normalized["media_type"] = media_type
    normalized["split"] = str(entry.get("split") or DEFAULT_SPLIT)
    normalized["license"] = str(entry.get("license") or DEFAULT_LICENSE)
    normalized["annotation_source"] = str(entry.get("annotation_source") or "human")
    normalized["real_trusted"] = bool(entry.get("real_trusted", False))
    return normalized


def validate_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    assets = [validate_asset(entry) for entry in _asset_entries(manifest)]
    ids = [entry["id"] for entry in assets]
    if len(ids) != len(set(ids)):
        raise RealAssetError("asset ids must be unique")
    return {
        "schema_version": str(manifest.get("schema_version") or "1.0"),
        "description": manifest.get("description", ""),
        "assets": assets,
    }


def _copy_media(asset: Dict[str, Any], source_base: Path, media_dir: Path) -> str:
    src = Path(str(asset["asset_uri"]))
    if not src.is_absolute():
        src = source_base / src
    if not src.exists():
        raise RealAssetError("%s asset_uri does not exist: %s" % (asset["id"], src))
    media_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        dst = media_dir / asset["id"]
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst = media_dir / ("%s%s" % (asset["id"], src.suffix))
        shutil.copy2(src, dst)
    return str(dst)


def _write_annotation(asset: Dict[str, Any], annotation_dir: Path) -> str:
    annotation_dir.mkdir(parents=True, exist_ok=True)
    path = annotation_dir / ("%s.json" % asset["id"])
    payload = {
        "asset_id": asset["id"],
        "media_type": asset["media_type"],
        "annotation_source": asset.get("annotation_source", "human"),
        "typed_gt": asset["typed_gt"],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return str(path)


def ingest_real_assets(
    source_manifest: Path,
    out_manifest: Optional[Path] = None,
    assets_root: Optional[Path] = None,
    copy_assets: bool = False,
) -> Dict[str, Any]:
    """Normalize a real-media manifest and write annotation sidecars.

    `copy_assets=False` keeps `asset_uri` as supplied, which is useful for early
    pilot manifests where media lives outside the repository.
    """
    source_manifest = Path(source_manifest)
    engine_root = _repo_engine_root()
    assets_root = Path(assets_root) if assets_root else engine_root / "assets" / "real"
    out_manifest = Path(out_manifest) if out_manifest else assets_root / "manifest.json"
    source_base = source_manifest.parent

    normalized = validate_manifest(load_manifest(source_manifest))
    annotation_dir = assets_root / "annotations"
    media_dir = assets_root / "media"

    emitted: List[Dict[str, Any]] = []
    for asset in normalized["assets"]:
        asset = dict(asset)
        if copy_assets:
            asset["asset_uri"] = _copy_media(asset, source_base, media_dir)
        asset["annotation_uri"] = _write_annotation(asset, annotation_dir)
        # Keep top-level manifest compact; annotation_uri is the durable GT source.
        asset.pop("typed_gt", None)
        emitted.append(asset)

    out = dict(normalized)
    out["assets"] = emitted
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Normalize U3 real-media typed-GT manifests.")
    p.add_argument("source_manifest", type=Path)
    p.add_argument("--out", type=Path, default=None, help="Output manifest path")
    p.add_argument("--assets-root", type=Path, default=None, help="Root for media/annotations")
    p.add_argument("--copy-assets", action="store_true", help="Copy local media into assets/real/media")
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    out = ingest_real_assets(
        args.source_manifest,
        out_manifest=args.out,
        assets_root=args.assets_root,
        copy_assets=args.copy_assets,
    )
    print("ingested %d real assets -> %s" % (len(out["assets"]), args.out or "assets/real/manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
