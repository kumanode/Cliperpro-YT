"""Storage manager: inspect and safely clean cached source videos, Whisper models, and sessions."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import local_whisper


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def _prune_video_history(output_root: Path) -> None:
    """Drop entries from ``database/video_history.json`` whose files no longer exist."""
    db_path = output_root / "database" / "video_history.json"
    if not db_path.exists():
        return
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
    except Exception:
        return

    kept: dict[str, Any] = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        file_path = entry.get("file_path") or entry.get("path")
        if file_path and Path(file_path).is_file():
            kept[key] = entry

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(kept, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


def get_storage_summary(output_dir: str | Path) -> dict[str, Any]:
    """Return disk usage breakdown for source videos, Whisper models, and sessions."""
    output_root = Path(output_dir)
    source_dir = output_root / "source_videos"
    sessions_dir = output_root / "sessions"
    now = time.time()

    source_items: list[dict[str, Any]] = []
    source_bytes = 0
    if source_dir.is_dir():
        for child in source_dir.iterdir():
            try:
                if not child.is_file():
                    continue
                st = child.stat()
                source_bytes += st.st_size
                age_days = max(0.0, (now - st.st_mtime) / 86400.0)
                source_items.append({
                    "filename": child.name,
                    "size_bytes": st.st_size,
                    "size_mb": round(st.st_size / (1024 * 1024), 1),
                    "age_days": round(age_days, 1),
                    "modified_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                })
            except OSError:
                continue
    source_items.sort(key=lambda x: x["size_bytes"], reverse=True)

    sessions_count = 0
    if sessions_dir.is_dir():
        try:
            sessions_count = sum(1 for c in sessions_dir.iterdir() if c.is_dir())
        except OSError:
            sessions_count = 0
    sessions_bytes = _dir_size_bytes(sessions_dir)

    whisper_models = local_whisper.list_cached_models()
    whisper_bytes = sum(int(m.get("size_bytes", 0)) for m in whisper_models)

    total_bytes = source_bytes + sessions_bytes + whisper_bytes
    return {
        "output_dir": str(output_root),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 1),
        "source_videos": {
            "count": len(source_items),
            "size_bytes": source_bytes,
            "size_mb": round(source_bytes / (1024 * 1024), 1),
            "items": source_items,
        },
        "sessions": {
            "count": sessions_count,
            "size_bytes": sessions_bytes,
            "size_mb": round(sessions_bytes / (1024 * 1024), 1),
        },
        "whisper_models": {
            "count": len(whisper_models),
            "size_bytes": whisper_bytes,
            "size_mb": round(whisper_bytes / (1024 * 1024), 1),
            "items": whisper_models,
        },
    }


def cleanup_storage(
    output_dir: str | Path,
    *,
    action: str,
    older_than_days: int = 7,
    filename: str | None = None,
) -> dict[str, Any]:
    """Delete cached source videos safely and synchronize ``video_history.json``."""
    output_root = Path(output_dir)
    source_dir = output_root / "source_videos"
    deleted_count = 0
    freed_bytes = 0

    if action == "source_video_file":
        raw = (filename or "").strip()
        if not raw or "/" in raw or "\\" in raw or ".." in raw:
            raise ValueError(f"Nama file tidak valid: {filename!r}")
        if not source_dir.is_dir():
            return {"deleted_count": 0, "freed_bytes": 0, "freed_mb": 0.0}
        target = (source_dir / raw).resolve()
        if target.parent != source_dir.resolve() or not target.is_file():
            raise FileNotFoundError(f"Source video tidak ditemukan: {raw}")
        freed_bytes = target.stat().st_size
        target.unlink()
        deleted_count = 1
        _prune_video_history(output_root)
    elif action == "source_videos_older_than":
        cutoff = time.time() - max(0, int(older_than_days)) * 86400
        if source_dir.is_dir():
            for child in list(source_dir.iterdir()):
                try:
                    if not child.is_file():
                        continue
                    st = child.stat()
                    if st.st_mtime <= cutoff:
                        freed_bytes += st.st_size
                        child.unlink()
                        deleted_count += 1
                except OSError:
                    continue
        _prune_video_history(output_root)
    elif action == "all_source_videos":
        if source_dir.is_dir():
            for child in list(source_dir.iterdir()):
                try:
                    if not child.is_file():
                        continue
                    freed_bytes += child.stat().st_size
                    child.unlink()
                    deleted_count += 1
                except OSError:
                    continue
        _prune_video_history(output_root)
    else:
        raise ValueError(f"Aksi cleanup tidak dikenal: {action!r}")

    return {
        "deleted_count": deleted_count,
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 1),
    }
