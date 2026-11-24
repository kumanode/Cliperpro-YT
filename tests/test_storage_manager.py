"""Unit tests for yt_short_clipper_core.storage_manager."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from unittest import mock

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yt_short_clipper_core import local_whisper, storage_manager


def test_storage_summary_and_cleanup(tmp_path):
    output_dir = tmp_path / "output"
    src_dir = output_dir / "source_videos"
    db_dir = output_dir / "database"
    sess_dir = output_dir / "sessions" / "20261124_100000"
    src_dir.mkdir(parents=True)
    db_dir.mkdir(parents=True)
    sess_dir.mkdir(parents=True)

    old_vid = src_dir / "old_video_720p.mp4"
    new_vid = src_dir / "new_video_720p.mp4"
    old_vid.write_bytes(b"a" * 4096)
    new_vid.write_bytes(b"b" * 2048)
    (sess_dir / "session_data.json").write_text("{}", encoding="utf-8")

    ten_days_ago = time.time() - 10 * 86400
    os.utime(old_vid, (ten_days_ago, ten_days_ago))

    db_path = db_dir / "video_history.json"
    db_path.write_text(
        json.dumps(
            {
                "old_720p": {"file_path": str(old_vid)},
                "new_720p": {"file_path": str(new_vid)},
            }
        ),
        encoding="utf-8",
    )

    with mock.patch.object(local_whisper, "list_cached_models", return_value=[]):
        summary = storage_manager.get_storage_summary(output_dir)
        assert summary["source_videos"]["count"] == 2
        assert summary["source_videos"]["size_bytes"] == 6144
        assert summary["sessions"]["count"] == 1

        # Clean older than 7 days -> removes old_vid, keeps new_vid
        res = storage_manager.cleanup_storage(
            output_dir, action="source_videos_older_than", older_than_days=7
        )
        assert res["deleted_count"] == 1
        assert res["freed_bytes"] == 4096
        assert not old_vid.exists()
        assert new_vid.exists()

        # History index pruned
        history = json.loads(db_path.read_text(encoding="utf-8"))
        assert list(history.keys()) == ["new_720p"]

        # Reject path traversal
        with pytest.raises(ValueError):
            storage_manager.cleanup_storage(
                output_dir, action="source_video_file", filename="../escape.mp4"
            )

        # Delete specific file
        res2 = storage_manager.cleanup_storage(
            output_dir, action="source_video_file", filename="new_video_720p.mp4"
        )
        assert res2["deleted_count"] == 1
        assert not new_vid.exists()
        assert json.loads(db_path.read_text(encoding="utf-8")) == {}
