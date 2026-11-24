"""Single-pass FFmpeg overlay compositor: hook + captions + watermark + BGM.

Replaces 3-4 sequential FFmpeg encode passes (hook -> caption -> watermark -> bgm)
with ONE filter_complex pass so clips render ~2-3x faster with zero extra
generation loss.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .caption_generator import _create_ass_subtitle
from .gpu import build_video_enc_args
from .helpers import get_ffmpeg_path
from .hook_generator import (
    DEFAULT_HOOK_DURATION,
    DEFAULT_HOOK_STYLE,
    HOOK_STYLE_PRESETS,
    _find_system_font_bold,
    _hex_to_rgb,
)
from .job_control import check_cancelled, run_cancellable_subprocess
from .watermark import _escape_filter_path, _find_text_font

LogFn = Callable[[str], None]
_SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _probe_dimensions(ffmpeg_path: str, video_path: str) -> tuple[int, int]:
    """Read (width, height) from ``ffmpeg -i`` stderr; defaults to 1080x1920."""
    try:
        res = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-i", video_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            creationflags=_SUBPROCESS_FLAGS,
        )
        m = re.search(r"(\d{3,4})x(\d{3,4})", res.stderr)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return 1080, 1920


def _render_hook_png(
    hook_text: str,
    width: int,
    height: int,
    hook_style: dict[str, Any] | None,
    temp_dir: Path,
) -> tuple[str, float]:
    """Render the transparent hook title card PNG and return ``(png_path, duration)``."""
    from PIL import Image, ImageDraw, ImageFont

    style_name = (hook_style or {}).get("style_name", "")
    if style_name == "AI Future Poster" or style_name not in HOOK_STYLE_PRESETS:
        style_name = "Clean White Box"
    incoming_style = dict(hook_style or {})
    incoming_style.pop("brush", None)
    incoming_style.pop("shadow", None)
    style = {**DEFAULT_HOOK_STYLE, **incoming_style}
    if style_name in HOOK_STYLE_PRESETS:
        style.update(HOOK_STYLE_PRESETS[style_name])

    try:
        duration = float(style.get("duration_seconds", DEFAULT_HOOK_DURATION))
    except (TypeError, ValueError):
        duration = DEFAULT_HOOK_DURATION
    if duration <= 0:
        duration = DEFAULT_HOOK_DURATION

    font_size_frac = float(style.get("font_size", 0.054))
    font_color_rgb = _hex_to_rgb(style.get("font_color", "#FFD700"))
    bg_color_rgb = _hex_to_rgb(style.get("bg_color", "#FFFFFF"))
    corner_radius = int(style.get("corner_radius", 0))
    pos_x = float(style.get("position_x", 0.5))
    pos_y = float(style.get("position_y", 0.333))
    user_font_path = style.get("font_path") or ""

    font_px = max(20, int(font_size_frac * width))
    pil_font = None
    for candidate in [user_font_path, _find_system_font_bold()]:
        if not candidate or not Path(candidate).exists():
            continue
        try:
            pil_font = ImageFont.truetype(candidate, font_px)
            break
        except Exception:
            pass
    if pil_font is None:
        pil_font = ImageFont.load_default()

    words = hook_text.upper().split()
    lines: list[str] = []
    current_line: list[str] = []
    for w in words:
        current_line.append(w)
        if len(current_line) >= 3:
            lines.append(" ".join(current_line))
            current_line = []
    if current_line:
        lines.append(" ".join(current_line))

    padding = max(10, int(font_px * 0.22))
    line_spacing = max(6, int(font_px * 0.25))
    line_metrics = []
    for line in lines:
        try:
            bbox = pil_font.getbbox(line)
        except AttributeError:
            w_t, h_t = pil_font.getsize(line)
            bbox = (0, 0, w_t, h_t)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        line_metrics.append({
            "text": line,
            "bbox": bbox,
            "box_w": text_w + padding * 2,
            "box_h": text_h + padding * 2,
        })

    total_h = sum(m["box_h"] for m in line_metrics)
    if len(line_metrics) > 1:
        total_h += line_spacing * (len(line_metrics) - 1)

    center_x = int(pos_x * width)
    center_y = int(pos_y * height)
    block_top = center_y - total_h // 2

    overlay_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_img)
    cur_y = block_top
    for m in line_metrics:
        box_w, box_h = m["box_w"], m["box_h"]
        box_x1 = center_x - box_w // 2
        box_y1 = cur_y
        box_x2 = box_x1 + box_w
        box_y2 = box_y1 + box_h

        if corner_radius > 0 and hasattr(draw, "rounded_rectangle"):
            r = min(corner_radius, box_w // 2, box_h // 2)
            draw.rounded_rectangle(
                [box_x1, box_y1, box_x2, box_y2],
                radius=r,
                fill=(*bg_color_rgb, 255),
            )
        else:
            draw.rectangle(
                [box_x1, box_y1, box_x2, box_y2],
                fill=(*bg_color_rgb, 255),
            )

        text_x = box_x1 + padding - m["bbox"][0]
        text_y = box_y1 + padding - m["bbox"][1]
        draw.text(
            (text_x, text_y),
            m["text"],
            font=pil_font,
            fill=(*font_color_rgb, 255),
        )
        cur_y = box_y2 + line_spacing

    overlay_png = str(temp_dir / f"hook_overlay_{int(time.time() * 1000)}.png")
    overlay_img.save(overlay_png, "PNG")
    return overlay_png, duration


def build_single_pass_graph(
    *,
    input_video_path: str,
    hook_png_path: str | None = None,
    hook_duration: float = DEFAULT_HOOK_DURATION,
    ass_path: str | None = None,
    watermark: dict[str, Any] | None = None,
    credit_watermark: dict[str, Any] | None = None,
    bgm_path: str | None = None,
    bgm_volume: float = 0.2,
    font_path: str | None = None,
) -> tuple[list[str], str, str, str]:
    """Build ``(inputs, filter_complex, video_map, audio_map)`` for one FFmpeg pass.

    Pure builder function so it can be unit-tested without invoking FFmpeg.
    """
    inputs: list[str] = ["-i", input_video_path]
    next_idx = 1
    filter_parts: list[str] = []
    v_label = "0:v"
    stage = 0

    def _next_v() -> str:
        nonlocal stage
        stage += 1
        return f"v{stage}"

    # 1. Hook overlay (first N seconds)
    if hook_png_path:
        hook_idx = next_idx
        next_idx += 1
        inputs.extend(["-i", hook_png_path])
        out_v = _next_v()
        filter_parts.append(
            f"[{v_label}][{hook_idx}:v]overlay=0:0:enable='lte(t,{hook_duration})'[{out_v}]"
        )
        v_label = out_v

    # 2. Captions (ASS word-by-word)
    if ass_path:
        out_v = _next_v()
        escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
        filter_parts.append(f"[{v_label}]ass='{escaped_ass}'[{out_v}]")
        v_label = out_v

    # 3. Logo watermark
    has_logo = bool(watermark and watermark.get("image_path") and Path(watermark["image_path"]).exists())
    if has_logo and watermark is not None:
        logo_idx = next_idx
        next_idx += 1
        inputs.extend(["-i", str(watermark["image_path"])])
        pos_x = watermark.get("position_x", 0.85)
        pos_y = watermark.get("position_y", 0.05)
        opacity = watermark.get("opacity", 0.8)
        scale = watermark.get("scale", 0.15)
        out_v = _next_v()
        filter_parts.append(
            f"[{logo_idx}:v]format=rgba,colorchannelmixer=aa={opacity},"
            f"scale=iw*{scale}:-1[logo];"
            f"[{v_label}][logo]overlay=W*{pos_x}-overlay_w/2:H*{pos_y}-overlay_h/2[{out_v}]"
        )
        v_label = out_v

    # 4. Credit text drawtext
    has_credit = bool(credit_watermark and credit_watermark.get("text"))
    if has_credit and credit_watermark is not None:
        text = str(credit_watermark["text"])
        color = credit_watermark.get("color", "#FFFFFF")
        font_size = credit_watermark.get("font_size", 24)
        opacity = credit_watermark.get("opacity", 0.7)
        pos_x = credit_watermark.get("position_x", 0.5)
        pos_y = credit_watermark.get("position_y", 0.3)
        if pos_y > 0.75:
            pos_y = 0.72
        ff_color = color.lstrip("#")
        alpha_hex = format(int(opacity * 255), "02x")
        max_len = 80
        display_text = text if len(text) <= max_len else text[: max_len - 3] + "..."
        escaped_display = display_text.replace("'", "\\'").replace(":", "\\:")
        fontfile_part = f":fontfile={_escape_filter_path(font_path)}" if font_path else ""
        fontsize_expr = f"({font_size}*main_w/270)"
        out_v = _next_v()
        filter_parts.append(
            f"[{v_label}]drawtext="
            f"text='{escaped_display}':"
            f"fontsize={fontsize_expr}:"
            f"fontcolor=0x{ff_color}{alpha_hex}{fontfile_part}:"
            f"x=(main_w*{pos_x}-text_w/2):y=(main_h*{pos_y}-text_h/2):"
            f"box=1:boxcolor=black@0.5:boxborderw=5:"
            f"shadowcolor=black@0.5:shadowx=1:shadowy=1[{out_v}]"
        )
        v_label = out_v

    # 5. Optional Background Music (BGM) with sidechaincompress auto-ducking
    a_map = "0:a?"
    if bgm_path and Path(bgm_path).exists():
        bgm_idx = next_idx
        next_idx += 1
        vol = max(0.01, min(1.0, float(bgm_volume)))
        inputs.extend(["-stream_loop", "-1", "-i", bgm_path])
        filter_parts.append(
            f"[0:a]asplit=2[voice_main][voice_sc];"
            f"[{bgm_idx}:a]volume={vol:.3f}[bgm_vol];"
            f"[bgm_vol][voice_sc]sidechaincompress="
            f"threshold=0.03:ratio=6:attack=20:release=350[bgm_ducked];"
            f"[voice_main][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        a_map = "[aout]"

    video_map = f"[{v_label}]" if v_label != "0:v" else "0:v"
    return inputs, ";".join(filter_parts), video_map, a_map


def compose_clip_overlays(
    *,
    input_video_path: str,
    output_path: str,
    hook_text: str | None = None,
    hook_style: dict[str, Any] | None = None,
    caption_words: list[dict[str, Any]] | None = None,
    caption_style: str = "Creator Pop",
    watermark: dict[str, Any] | None = None,
    credit_watermark: dict[str, Any] | None = None,
    bgm_path: str | None = None,
    bgm_volume: float = 0.2,
    log: LogFn | None = None,
    gpu_config: dict[str, Any] | None = None,
) -> str:
    """Render hook + captions + watermark + optional BGM in ONE FFmpeg pass."""
    log = log or (lambda _m: None)
    check_cancelled()

    has_hook = bool(hook_text and hook_text.strip())
    has_captions = bool(caption_words)
    has_logo = bool(watermark and watermark.get("image_path") and Path(watermark["image_path"]).exists())
    has_credit = bool(credit_watermark and credit_watermark.get("text"))
    has_bgm = bool(bgm_path and Path(bgm_path).exists())

    if not (has_hook or has_captions or has_logo or has_credit or has_bgm):
        shutil.copy2(input_video_path, output_path)
        return output_path

    ffmpeg_path = get_ffmpeg_path()
    temp_dir = Path(output_path).parent
    cleanup_paths: list[Path] = []

    try:
        hook_png: str | None = None
        hook_duration = DEFAULT_HOOK_DURATION
        if has_hook and hook_text:
            w, h = _probe_dimensions(ffmpeg_path, input_video_path)
            hook_png, hook_duration = _render_hook_png(
                hook_text, w, h, hook_style, temp_dir
            )
            cleanup_paths.append(Path(hook_png))

        ass_path: str | None = None
        if has_captions and caption_words:
            ass_file = temp_dir / f"captions_{int(time.time() * 1000)}.ass"
            transcript = SimpleNamespace(
                words=[
                    SimpleNamespace(
                        word=str(w["word"]),
                        start=float(w["start"]),
                        end=float(w["end"]),
                    )
                    for w in caption_words
                ],
                segments=[],
            )
            _create_ass_subtitle(transcript, str(ass_file), style_name=caption_style)
            ass_path = str(ass_file)
            cleanup_paths.append(ass_file)

        font_path = _find_text_font() if has_credit else None
        inputs, filter_complex, v_map, a_map = build_single_pass_graph(
            input_video_path=input_video_path,
            hook_png_path=hook_png,
            hook_duration=hook_duration,
            ass_path=ass_path,
            watermark=watermark if has_logo else None,
            credit_watermark=credit_watermark if has_credit else None,
            bgm_path=bgm_path if has_bgm else None,
            bgm_volume=bgm_volume,
            font_path=font_path,
        )

        video_enc_args = build_video_enc_args(gpu_config)
        audio_enc_args = ["-c:a", "aac", "-b:a", "192k"] if has_bgm else ["-c:a", "copy"]

        active_layers = [
            name
            for name, flag in (
                ("hook", has_hook),
                ("captions", has_captions),
                ("logo", has_logo),
                ("credit", has_credit),
                ("bgm", has_bgm),
            )
            if flag
        ]
        log(f"Single-pass FFmpeg compose ({' + '.join(active_layers)})...")

        cmd = [
            ffmpeg_path,
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            v_map,
            "-map",
            a_map,
            *video_enc_args,
            "-pix_fmt",
            "yuv420p",
            *audio_enc_args,
            output_path,
        ]
        result = run_cancellable_subprocess(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"Single-pass FFmpeg compose failed: {result.stderr[-600:]}"
            )
        return output_path
    finally:
        for p in cleanup_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
