"""Local transcription with faster-whisper — no API, no per-request cost.

Why this exists
---------------
The cloud path (OpenAI / Groq / Hugging Face) has three failure modes that
have each cost us a release cycle: a ~25 MB upload cap, a provider edge that
disconnects on long audio, and provider hosts that get retired underneath a
hard-coded URL. Running Whisper in-process removes all three — there is no
upload, no cap and no remote endpoint involved. It trades them for CPU time,
which is the honest cost the user accepts by picking this option.

Design notes
------------
- **Import is lazy.** ``faster_whisper`` drags in ctranslate2 (~60 MB), av
  (~32 MB) and tokenizers (~12 MB). Importing it at module scope would make
  every session pay the cost even when transcribing via the cloud, so it is
  resolved on first *use*.
- **onnxruntime stays excluded from the frozen build.** faster-whisper imports
  it lazily, inside ``vad.py``'s VAD class, and only when ``vad_filter=True``.
  We never enable VAD, so the ~67 MB dependency is unnecessary. If a future
  faster-whisper release hoists that import to module level, the user gets a
  clear error instead of a crash — see ``_import_error_message``.
- **The model is cached on disk** under the Tauri app-data dir and downloaded
  from Hugging Face on first use. It is *not* bundled: ``turbo`` alone is
  ~1.6 GB.
- **The loaded model is cached per process.** Loading large-v3 costs 10–20 s,
  and one session transcribes many chunks; reloading per chunk would dominate
  the runtime.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

LogFn = Callable[[str], None]

#: Sentinel written into ``transcription_base_url`` so the rest of the app
#: keeps using the existing settings fields instead of a parallel config path.
LOCAL_SCHEME = "local://faster-whisper"

#: faster-whisper model ids, smallest first. The trade-off is stark on CPU:
#: ``tiny`` is roughly real-time, ``turbo`` is 3–5x slower than real-time on a
#: low-power laptop CPU. The UI explains this; we only enforce validity here.
KNOWN_MODELS = ("tiny", "tiny.en", "base", "base.en", "small", "small.en",
                "medium", "large-v2", "large-v3", "turbo", "distil-large-v3")

#: Rough on-disk size, so the UI can warn before a 1.6 GB surprise download.
MODEL_SIZES_MB = {
    "tiny": 75, "tiny.en": 75,
    "base": 145, "base.en": 145,
    "small": 465, "small.en": 465,
    "medium": 1500,
    "large-v2": 3000, "large-v3": 3100,
    "distil-large-v3": 1500,
    "turbo": 1600,
}

_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def is_local_base_url(url: str | None) -> bool:
    """True when the configured transcription base URL means 'in-process'."""
    return (url or "").strip().lower().startswith("local://")


def _model_cache_dir() -> Path:
    """Where downloaded Whisper weights live.

    Prefers the Tauri app-data dir so uninstalling the app can be understood
    by users, and falls back to the temp dir on a bare sidecar run.
    """
    try:
        from .helpers import _tauri_app_data_dir

        base = _tauri_app_data_dir()
        if base is not None:
            d = base / "whisper-models"
            d.mkdir(parents=True, exist_ok=True)
            return d
    except Exception:
        pass
    import tempfile

    d = Path(tempfile.gettempdir()) / "cliperpro-whisper-models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _point_hf_at_our_cache() -> None:
    """Make huggingface_hub download into our cache dir.

    Must run before the first ``from faster_whisper import ...``. Otherwise
    HF defaults to ``~/.cache/huggingface``, which a user clearing app data
    would not expect to be holding gigabytes.
    """
    cache = _model_cache_dir()
    os.environ.setdefault("HF_HOME", str(cache))
    # Older hub versions read these separately.
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "hub"))


def _looks_like_local_path(raw: str) -> bool:
    """True when *raw* looks like a filesystem path rather than a model id."""
    s = (raw or "").strip()
    if not s:
        return False
    # Windows drive letter (C:\ or C:/), UNC (\\server), absolute POSIX (/),
    # or explicit relative path (./ or ../)
    if (
        (len(s) >= 2 and s[1] == ":" and s[0].isalpha())
        or s.startswith(("\\\\", "/", "./", "../", ".\\", "..\\"))
        or "\\" in s
    ):
        return True
    try:
        return Path(s).exists()
    except OSError:
        return False


def resolve_local_model_path(raw_path: str) -> Path:
    """Resolve and validate a user-supplied local faster-whisper model folder.

    Accepts:
      - A CTranslate2 model directory containing ``model.bin``
      - A direct path to ``model.bin`` (uses its parent directory)
      - A Hugging Face cache repo dir containing ``snapshots/<rev>/model.bin``
      - A wrapper directory containing a single CTranslate2 subfolder
    """
    raw = (raw_path or "").strip().strip('"').strip("'")
    if not raw:
        raise ValueError("Path folder model lokal masih kosong.")

    p = Path(raw).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Folder/file model lokal tidak ditemukan: {raw}")

    if p.is_file():
        low_name = p.name.lower()
        if low_name == "model.bin":
            p = p.parent
        elif low_name.endswith((".pt", ".pth")) or low_name.startswith("ggml"):
            raise ValueError(
                f"File '{p.name}' adalah format PyTorch/whisper.cpp, bukan format "
                "CTranslate2 (faster-whisper). Pilih folder model faster-whisper "
                "yang berisi 'model.bin' dan 'config.json'."
            )
        else:
            raise ValueError(
                f"Path '{p.name}' bukan folder model faster-whisper. "
                "Pilih folder yang berisi 'model.bin' dan 'config.json'."
            )

    # 1. Direct CTranslate2 model directory
    if (p / "model.bin").is_file():
        return p.resolve()

    # 2. Hugging Face cache layout: models--.../snapshots/<hash>/model.bin
    snapshots_dir = p / "snapshots"
    if snapshots_dir.is_dir():
        candidates = [
            sub for sub in snapshots_dir.iterdir()
            if sub.is_dir() and (sub / "model.bin").is_file()
        ]
        if candidates:
            candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            return candidates[0].resolve()

    # 3. Single nested subfolder containing model.bin
    try:
        nested = [
            sub for sub in p.iterdir()
            if sub.is_dir() and (sub / "model.bin").is_file()
        ]
        if len(nested) == 1:
            return nested[0].resolve()
    except OSError:
        pass

    # Check if user pointed to a folder with .pt or ggml .bin files to give a helpful hint
    try:
        wrong_files = [
            f.name for f in p.iterdir()
            if f.is_file() and (
                f.name.lower().endswith((".pt", ".pth"))
                or f.name.lower().startswith("ggml")
            )
        ]
    except OSError:
        wrong_files = []
    if wrong_files:
        raise ValueError(
            f"Folder ini berisi model non-CTranslate2 ({wrong_files[0]}). "
            "faster-whisper membutuhkan folder model format CTranslate2 yang "
            "berisi 'model.bin', 'config.json', dan 'tokenizer.json'/'vocabulary.json'."
        )

    raise ValueError(
        "File 'model.bin' tidak ditemukan di folder tersebut. Pastikan folder "
        "berisi model faster-whisper (format CTranslate2: model.bin, config.json, "
        "dan tokenizer.json atau vocabulary.json)."
    )


def validate_local_model_path(raw_path: str) -> dict[str, Any]:
    """Validate a manual local model folder for the UI without loading weights."""
    try:
        resolved = resolve_local_model_path(raw_path)
    except Exception as exc:
        return {
            "valid": False,
            "resolved_path": None,
            "name": None,
            "size_mb": 0.0,
            "error": str(exc),
        }

    size_bytes = _dir_size_bytes(resolved)
    # If resolved inside .../models--Org--Name/snapshots/<hash>, use friendly parent name
    display_name = resolved.name
    if resolved.parent.name == "snapshots" and resolved.parent.parent.name.startswith("models--"):
        display_name = resolved.parent.parent.name[len("models--"):].replace("--", "/")
    return {
        "valid": True,
        "resolved_path": str(resolved),
        "name": display_name,
        "size_mb": round(size_bytes / (1024 * 1024), 1),
        "error": None,
    }


def normalise_model(name: str | None) -> str:
    """Map a user-entered model onto a known faster-whisper id.

    Users arrive from the cloud world and type things like
    ``openai/whisper-large-v3-turbo``. Accepting those verbatim would send
    huggingface_hub off to fetch a *different, much larger* repo, so we strip
    the provider prefix and map the leftovers.

    Local filesystem paths (manual pre-downloaded models) are preserved verbatim.
    """
    raw = (name or "").strip()
    if not raw:
        return "turbo"
    if _looks_like_local_path(raw):
        return raw
    low = raw.lower()
    # Drop a provider prefix: "openai/whisper-large-v3-turbo" -> "large-v3-turbo"
    if "/" in low:
        low = low.rsplit("/", 1)[-1]
    # HF spells the distilled distil model "distil-whisper/distil-large-v3".
    if low.startswith("whisper-"):
        low = low[len("whisper-"):]
    if low in KNOWN_MODELS:
        return low
    # "large-v3-turbo" and "turbo" both mean the turbo distillation.
    for cand in ("turbo",):
        if low.endswith(cand) or low == cand:
            return cand
    # Accept the cloud spelling of large-v3 ("whisper-large-v3" -> "large-v3").
    if low.startswith("large-v3"):
        return "large-v3"
    if low.startswith("large-v2"):
        return "large-v2"
    if low.startswith("distil-large-v3"):
        return "distil-large-v3"
    # Unknown id: hand it to faster-whisper, which supports HF repo ids too.
    return raw


def _import_faster_whisper():
    """Import WhisperModel, translating any failure into actionable text."""
    _point_hf_at_our_cache()
    try:
        from faster_whisper import WhisperModel  # type: ignore

        return WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Mesin transkripsi lokal (faster-whisper) tidak tersedia di build "
            "ini. Install dengan 'pip install faster-whisper' atau pakai "
            "provider cloud (Hugging Face / Groq / OpenAI). "
            f"Detail: {exc}"
        ) from exc
    except Exception as exc:  # native lib load failures land here
        raise RuntimeError(
            "Gagal memuat pustaka native faster-whisper (ctranslate2). "
            "Biasanya berarti build sidecar tidak lengkap — pasang ulang "
            f"versi portable terbaru. Detail: {exc}"
        ) from exc


_CUDA_ERROR_MARKERS = (
    "cublas",
    "cudnn",
    "cudart",
    "libcublas",
    "libcudnn",
    "cuda driver version is insufficient",
    "no cuda-capable device",
    "cuda failed",
)


def _is_cuda_runtime_error(exc: Exception) -> bool:
    """True when *exc* indicates missing/unusable CUDA runtime libraries."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _CUDA_ERROR_MARKERS)


def _cuda_runtime_ready() -> tuple[bool, str]:
    """Check whether CUDA 12 runtime libraries required by CTranslate2 can load.

    ``ctranslate2.get_cuda_device_count()`` only checks for an NVIDIA GPU and
    display driver (``nvcuda.dll``). CTranslate2 loads ``cublas64_12.dll``
    lazily on the first ``encode()`` call, so without this probe a Windows
    machine with an NVIDIA GPU but no CUDA 12 Toolkit crashes mid-transcription.
    """
    if sys.platform == "win32":
        required_libs = ("cublas64_12.dll", "cublasLt64_12.dll")
        loader = getattr(ctypes, "WinDLL", ctypes.CDLL)
    elif sys.platform.startswith("linux"):
        required_libs = ("libcublas.so.12",)
        loader = ctypes.CDLL
    else:
        return False, f"CUDA tidak didukung di platform {sys.platform}"

    for lib in required_libs:
        try:
            loader(lib)
        except OSError as exc:
            return False, f"{lib} tidak ditemukan ({exc})"
    return True, ""


def _pick_device(log: LogFn | None = None) -> tuple[str, str]:
    """Choose (device, compute_type) for this machine.

    CUDA when a usable GPU AND CUDA 12 runtime libraries are present,
    otherwise CPU + int8. int8 is the right default for CPU: it is markedly
    faster than float32 at negligible accuracy cost for speech, and it
    roughly halves the memory footprint.
    """
    try:
        import ctranslate2  # type: ignore

        count = ctranslate2.get_cuda_device_count()
        if count and count > 0:
            ok, reason = _cuda_runtime_ready()
            if ok:
                return "cuda", "float16"
            if log is not None:
                log(
                    "GPU NVIDIA terdeteksi, tetapi runtime CUDA 12 tidak "
                    f"tersedia ({reason}). Beralih otomatis ke CPU/int8."
                )
    except Exception:
        pass
    return "cpu", "int8"


def _load_model(model: str, device: str | None, compute: str | None,
                log: LogFn):
    """Return a cached WhisperModel, downloading the weights on first use."""
    key = (model, device or "auto", compute or "auto")
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    WhisperModel = _import_faster_whisper()

    auto_device_requested = device in (None, "", "auto")
    if auto_device_requested or compute in (None, "", "auto"):
        auto_device, auto_compute = _pick_device(log=log)
        device = device if not auto_device_requested else auto_device
        compute = compute if compute not in (None, "", "auto") else auto_compute

    is_manual_local = _looks_like_local_path(model)
    model_arg = model
    if is_manual_local:
        resolved = resolve_local_model_path(model)
        model_arg = str(resolved)
        log(
            f"Memuat model Whisper lokal dari folder manual '{model_arg}' "
            f"di {device}/{compute} (offline, tanpa unduh)."
        )
    else:
        size_mb = MODEL_SIZES_MB.get(model)
        size_txt = f" (~{size_mb} MB)" if size_mb else ""
        log(f"Memuat model Whisper lokal '{model}'{size_txt} di {device}/{compute}. "
            "Kesabaran, unduhan pertama cukup besar dan butuh koneksi internet.")

    load_kwargs: dict[str, Any] = {"device": device, "compute_type": compute}
    if is_manual_local:
        load_kwargs["local_files_only"] = True

    try:
        loaded = WhisperModel(model_arg, **load_kwargs)
    except Exception as exc:
        if auto_device_requested and device == "cuda" and _is_cuda_runtime_error(exc):
            log(
                f"Gagal memuat model di CUDA ({exc}). "
                "Beralih otomatis ke CPU/int8..."
            )
            try:
                cpu_kwargs: dict[str, Any] = {"device": "cpu", "compute_type": "int8"}
                if is_manual_local:
                    cpu_kwargs["local_files_only"] = True
                loaded = WhisperModel(model_arg, **cpu_kwargs)
            except Exception as cpu_exc:
                raise RuntimeError(
                    f"Gagal memuat model Whisper '{model_arg}' di CPU setelah "
                    f"fallback dari CUDA. Detail: {cpu_exc}"
                ) from cpu_exc
        elif is_manual_local:
            raise RuntimeError(
                f"Gagal memuat model Whisper dari folder manual '{model_arg}'. "
                "Pastikan folder berisi model CTranslate2 yang lengkap "
                f"(model.bin, config.json, tokenizer.json/vocabulary.json). Detail: {exc}"
            ) from exc
        else:
            raise RuntimeError(
                f"Gagal memuat model Whisper '{model}'. Kalau ini model yang "
                f"belum pernah diunduh, pastikan ada koneksi internet "
                f"(diunduh dari Hugging Face). Detail: {exc}"
            ) from exc

    _MODEL_CACHE[key] = loaded
    return loaded


def _format_eta(seconds: float) -> str:
    """Human-readable short ETA string."""
    sec = max(0, int(round(seconds)))
    if sec >= 3600:
        return f"{sec // 3600}j {(sec % 3600) // 60}m"
    if sec >= 60:
        return f"{sec // 60}m {sec % 60}d"
    return f"{sec}d"


def _decode_segments(
    loaded: Any,
    audio_path: Path,
    language: str | None,
    emit: LogFn,
    *,
    word_timestamps: bool = False,
    cancel_event: threading.Event | None = None,
    progress_cb: Callable[[float, float | None, int], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, int]:
    """Run faster-whisper decode generator and collect timestamped segments + words."""
    # vad_filter stays off deliberately: it needs onnxruntime, which the
    # frozen build excludes, and for speech-dense Indonesian video the filter
    # is not needed.
    #
    # NOTE: faster_whisper returns a GENERATOR, not a list — decoding happens
    # lazily while we iterate. So we cannot ask for a total up front, and any
    # decode error surfaces mid-loop, not at the call. That generator contract
    # is easy to get wrong: calling len() on it raises TypeError, which is
    # exactly the bug the first real end-to-end run caught.
    kwargs: dict[str, Any] = {
        "language": (language or None),
        "vad_filter": False,
        "beam_size": 5,
    }
    if word_timestamps:
        kwargs["word_timestamps"] = True

    segments, info = loaded.transcribe(str(audio_path), **kwargs)

    dur = getattr(info, "duration", None)
    total_dur = float(dur) if isinstance(dur, (int, float)) and dur > 0 else 0.0
    t0 = time.monotonic()
    last_emit_t = t0
    last_emit_pct = -10.0

    out: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    seen = 0
    for seg in segments:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Transkripsi lokal dibatalkan oleh pengguna.")
        text = (getattr(seg, "text", "") or "").strip()
        seen += 1
        seg_start = float(getattr(seg, "start", 0.0) or 0.0)
        seg_end = float(getattr(seg, "end", 0.0) or 0.0)
        if text:
            out.append({
                "start": seg_start,
                "end": seg_end,
                "text": text,
            })
        if word_timestamps:
            seg_words = getattr(seg, "words", None) or []
            for w in seg_words:
                w_text = (getattr(w, "word", "") or "").strip()
                if not w_text:
                    continue
                w_start = float(getattr(w, "start", seg_start) or 0.0)
                w_end = float(getattr(w, "end", seg_end) or w_start)
                if w_end < w_start:
                    w_end = w_start
                words.append({"word": w_text, "start": w_start, "end": w_end})

        now = time.monotonic()
        if total_dur > 0:
            pct = min(100.0, max(0.0, (seg_end / total_dur) * 100.0))
            if (pct - last_emit_pct >= 2.0) or (now - last_emit_t >= 1.5):
                elapsed = max(0.001, now - t0)
                eta = (elapsed * (100.0 - pct) / pct) if pct > 1.0 else None
                eta_txt = f", ETA ~{_format_eta(eta)}" if eta is not None else ""
                emit(f"  transkripsi lokal: {pct:.0f}% ({len(out)} segmen{eta_txt})")
                if progress_cb is not None:
                    progress_cb(pct, eta, len(out))
                last_emit_pct = pct
                last_emit_t = now
        elif seen % 25 == 0:
            emit(f"  transkripsi lokal: {len(out)} segmen")
    return out, words, info, seen


def transcribe_with_words(
    audio_path: Path,
    model: str = "turbo",
    language: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    log: LogFn | None = None,
    *,
    word_timestamps: bool = True,
    cancel_event: threading.Event | None = None,
    progress_cb: Callable[[float, float | None, int], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transcribe one audio/video file locally and return ``(segments, words)``.

    Each segment is ``{"start", "end", "text"}`` and each word (when
    ``word_timestamps=True``) is ``{"word", "start", "end"}`` in seconds.
    """
    emit: LogFn = log or (lambda _m: None)
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio tidak ditemukan: {audio_path}")

    mdl = normalise_model(model)
    auto_mode = device in (None, "", "auto")
    loaded = _load_model(mdl, device, compute_type, emit)

    out: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    try:
        out, words, info, _seen = _decode_segments(
            loaded,
            audio_path,
            language,
            emit,
            word_timestamps=word_timestamps,
            cancel_event=cancel_event,
            progress_cb=progress_cb,
        )
    except InterruptedError as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        # Safety net: if CUDA runtime libraries (e.g. cublas64_12.dll / cudnn)
        # fail lazily on the first encode() call while in auto mode, drop the
        # broken cached CUDA instance and retry cleanly on CPU/int8.
        if auto_mode and _is_cuda_runtime_error(exc):
            emit(
                f"CUDA gagal saat mulai transkripsi ({exc}). "
                "Beralih otomatis ke CPU/int8..."
            )
            _MODEL_CACHE.pop((mdl, device or "auto", compute_type or "auto"), None)
            cpu_loaded = _load_model(mdl, "cpu", "int8", emit)
            try:
                out, words, info, _seen = _decode_segments(
                    cpu_loaded,
                    audio_path,
                    language,
                    emit,
                    word_timestamps=word_timestamps,
                    cancel_event=cancel_event,
                    progress_cb=progress_cb,
                )
            except InterruptedError as c_exc:
                raise RuntimeError(str(c_exc)) from c_exc
            except Exception as cpu_exc:
                raise RuntimeError(
                    f"Mesin transkripsi lokal gagal di tengah pemrosesan setelah "
                    f"{len(out)} segmen (CPU fallback). Kalau errornya soal "
                    f"memori atau lambat, coba model yang lebih kecil "
                    f"('small' atau 'base'). Detail: {cpu_exc}"
                ) from cpu_exc
        elif _is_cuda_runtime_error(exc):
            raise RuntimeError(
                "Mesin transkripsi lokal gagal memuat pustaka CUDA 12 "
                "(mis. cublas64_12.dll). Install CUDA Toolkit 12 atau "
                f"gunakan mode CPU/auto. Detail: {exc}"
            ) from exc
        else:
            # No bare `except RuntimeError: raise` here, unlike session.py: this
            # module has no control-flow exceptions to preserve, and ctranslate2
            # signals out-of-memory as a plain RuntimeError. Letting that through
            # unwrapped would surface "ct2 out of memory" with no hint that the
            # fix is a smaller model or a bigger compute budget.
            raise RuntimeError(
                f"Mesin transkripsi lokal gagal di tengah pemrosesan setelah "
                f"{len(out)} segmen. Kalau errornya soal memori atau lambat, "
                f"coba model yang lebih kecil ('small' atau 'base'). "
                f"Detail: {exc}"
            ) from exc

    lang = getattr(info, "language", None) or language or "?"
    dur = getattr(info, "duration", None)
    words_txt = f", {len(words)} kata" if words else ""
    emit(f"Transkripsi lokal selesai: {len(out)} segmen{words_txt} "
         f"(bahasa {lang}"
         + (f", audio {dur:.0f} dtk" if isinstance(dur, (int, float)) else "")
         + ").")
    return out, words


def transcribe(
    audio_path: Path,
    model: str = "turbo",
    language: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    log: LogFn | None = None,
    *,
    cancel_event: threading.Event | None = None,
    progress_cb: Callable[[float, float | None, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Transcribe one audio/video file locally.

    Returns a list of ``{"start", "end", "text"}`` dicts in seconds, ordered
    by start time. Timestamps are real, so callers build a far more accurate
    SRT than the cloud path can produce from a single blob of text.
    """
    segments, _words = transcribe_with_words(
        audio_path,
        model=model,
        language=language,
        device=device,
        compute_type=compute_type,
        log=log,
        word_timestamps=False,
        cancel_event=cancel_event,
        progress_cb=progress_cb,
    )
    return segments


def _dir_size_bytes(path: Path) -> int:
    total = 0
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


def list_cached_models() -> list[dict[str, Any]]:
    """List downloaded Hugging Face faster-whisper model directories and sizes."""
    hub_dir = _model_cache_dir() / "hub"
    if not hub_dir.is_dir():
        return []
    items: list[dict[str, Any]] = []
    try:
        for child in hub_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("models--"):
                continue
            size_bytes = _dir_size_bytes(child)
            # "models--Systran--faster-whisper-small" -> "Systran/faster-whisper-small"
            friendly = child.name[len("models--"):].replace("--", "/")
            items.append({
                "dir_name": child.name,
                "name": friendly,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 1),
            })
    except OSError:
        return []
    items.sort(key=lambda x: x["size_bytes"], reverse=True)
    return items


def delete_cached_model(dir_name: str) -> dict[str, Any]:
    """Safely delete one cached Whisper model directory under ``hub/``."""
    raw = (dir_name or "").strip()
    if (
        not raw
        or not raw.startswith("models--")
        or "/" in raw
        or "\\" in raw
        or ".." in raw
    ):
        raise ValueError(f"Nama direktori model tidak valid: {dir_name!r}")

    hub_dir = (_model_cache_dir() / "hub").resolve()
    target = (hub_dir / raw).resolve()
    if target.parent != hub_dir or not target.is_dir():
        raise FileNotFoundError(f"Cache model tidak ditemukan: {raw}")

    freed = _dir_size_bytes(target)
    shutil.rmtree(target)
    # Clear in-memory model cache so a deleted model isn't served stale.
    _MODEL_CACHE.clear()
    return {
        "deleted": raw,
        "freed_bytes": freed,
        "freed_mb": round(freed / (1024 * 1024), 1),
    }


def probe() -> dict[str, Any]:
    """Report local-engine status without loading any weights.

    Used by the UI so the "Lokal" option can say *why* it is unavailable
    instead of failing at the moment the user submits a job.
    """
    cached = list_cached_models()
    cache_total_mb = round(sum(m["size_bytes"] for m in cached) / (1024 * 1024), 1)
    try:
        _point_hf_at_our_cache()
        import faster_whisper  # type: ignore

        # `faster_whisper.version` is a MODULE, not a string — reading the
        # attribute off the package yields a module object. Report something
        # a human can read instead.
        try:
            from importlib.metadata import version as _pkg_version

            version = _pkg_version("faster-whisper")
        except Exception:
            ver_mod = getattr(faster_whisper, "version", None)
            version = getattr(ver_mod, "__version__", "unknown")
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "version": None,
            "cache_dir": str(_model_cache_dir()),
            "cached_models": cached,
            "cache_total_mb": cache_total_mb,
        }
    cuda_ready, cuda_reason = _cuda_runtime_ready()
    device, compute = _pick_device()
    return {
        "available": True,
        "version": version,
        "device": device,
        "compute_type": compute,
        "cuda_ready": cuda_ready,
        "cuda_reason": cuda_reason or None,
        "cache_dir": str(_model_cache_dir()),
        "cached_models": cached,
        "cache_total_mb": cache_total_mb,
        "models": list(KNOWN_MODELS),
    }
