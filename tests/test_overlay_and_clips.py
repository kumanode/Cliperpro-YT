"""Unit tests for overlay_compositor, clip_processor helpers, and job_control."""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yt_short_clipper_core.clip_processor import _apply_caption_override, sanitize_clip_filename
from yt_short_clipper_core.job_control import (
    JobCancelledError,
    check_cancelled,
    request_cancel,
    reset_cancel,
)
from yt_short_clipper_core.overlay_compositor import build_single_pass_graph


def test_build_single_pass_graph_combines_all_layers(tmp_path):
    video = tmp_path / "in.mp4"
    hook = tmp_path / "hook.png"
    ass = tmp_path / "subs.ass"
    logo = tmp_path / "logo.png"
    bgm = tmp_path / "music.mp3"
    for f in (video, hook, ass, logo, bgm):
        f.write_bytes(b"0")

    inputs, graph, v_map, a_map = build_single_pass_graph(
        input_video_path=str(video),
        hook_png_path=str(hook),
        hook_duration=4.5,
        ass_path=str(ass),
        watermark={"image_path": str(logo), "position_x": 0.85, "position_y": 0.05, "opacity": 0.8, "scale": 0.15},
        credit_watermark={"text": "Source: Demo", "color": "#FFFFFF", "font_size": 20, "opacity": 0.7, "position_x": 0.5, "position_y": 0.3},
        bgm_path=str(bgm),
        bgm_volume=0.25,
    )

    assert "-stream_loop" in inputs
    assert "overlay=0:0:enable='lte(t,4.5)'" in graph
    assert "ass='" in graph
    assert "drawtext=" in graph
    assert "sidechaincompress=" in graph
    assert v_map.startswith("[v") and v_map.endswith("]")
    assert a_map == "[aout]"


def test_apply_caption_override_preserves_timestamps_on_matching_length():
    orig = [
        {"word": "HALO", "start": 1.0, "end": 1.5},
        {"word": "DUNIA", "start": 1.6, "end": 2.2},
    ]
    out = _apply_caption_override(orig, "Halo Semua", 5.0)
    assert out == [
        {"word": "Halo", "start": 1.0, "end": 1.5},
        {"word": "Semua", "start": 1.6, "end": 2.2},
    ]


def test_apply_caption_override_redistributes_when_word_count_changes():
    orig = [
        {"word": "SATU", "start": 1.0, "end": 2.0},
        {"word": "DUA", "start": 2.0, "end": 3.0},
    ]
    out = _apply_caption_override(orig, "satu dua tiga empat", 5.0)
    assert len(out) == 4
    assert out[0]["start"] == pytest.approx(1.0)
    assert out[-1]["end"] == pytest.approx(3.0)


def test_sanitize_clip_filename():
    assert sanitize_clip_filename('Judul: "Mantap" <1>?') == "Judul- -Mantap- -1--"
    assert sanitize_clip_filename("   ...  ") == "Untitled"


def test_job_control_cancel_and_reset():
    reset_cancel()
    check_cancelled()
    request_cancel()
    with pytest.raises(JobCancelledError):
        check_cancelled()
    reset_cancel()
    check_cancelled()
