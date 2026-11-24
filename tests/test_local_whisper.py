"""Unit tests for yt_short_clipper_core.local_whisper."""

from __future__ import annotations

import pathlib
import sys
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yt_short_clipper_core import local_whisper as lw


def test_is_local_base_url():
    assert lw.is_local_base_url("local://faster-whisper")
    assert lw.is_local_base_url("  LOCAL://Faster-Whisper  ")
    assert not lw.is_local_base_url("https://api.openai.com/v1")
    assert not lw.is_local_base_url("")
    assert not lw.is_local_base_url(None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("turbo", "turbo"),
        ("openai/whisper-large-v3-turbo", "turbo"),
        ("whisper-large-v3", "large-v3"),
        ("distil-whisper/distil-large-v3", "distil-large-v3"),
        ("base.en", "base.en"),
        ("", "turbo"),
        (None, "turbo"),
    ],
)
def test_normalise_model(raw, expected):
    assert lw.normalise_model(raw) == expected


def test_pick_device_falls_back_to_cpu_when_cublas_missing():
    logs: list[str] = []
    fake_ct2 = SimpleNamespace(get_cuda_device_count=lambda: 1)
    with (
        mock.patch.dict(sys.modules, {"ctranslate2": fake_ct2}),
        mock.patch.object(
            lw, "_cuda_runtime_ready", return_value=(False, "cublas64_12.dll tidak ditemukan")
        ),
    ):
        dev, comp = lw._pick_device(log=logs.append)
    assert (dev, comp) == ("cpu", "int8")
    assert any("CUDA 12" in m for m in logs)


def test_pick_device_uses_cuda_when_runtime_ready():
    fake_ct2 = SimpleNamespace(get_cuda_device_count=lambda: 1)
    with (
        mock.patch.dict(sys.modules, {"ctranslate2": fake_ct2}),
        mock.patch.object(lw, "_cuda_runtime_ready", return_value=(True, "")),
    ):
        dev, comp = lw._pick_device()
    assert (dev, comp) == ("cuda", "float16")


def test_transcribe_with_words_and_progress(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")

    w1 = SimpleNamespace(word=" halo ", start=0.2, end=0.7)
    w2 = SimpleNamespace(word=" dunia", start=0.8, end=1.4)
    seg = SimpleNamespace(start=0.2, end=5.0, text="  halo dunia  ", words=[w1, w2])
    info = SimpleNamespace(language="id", duration=10.0)

    fake_model = mock.Mock()
    fake_model.transcribe.return_value = (iter([seg]), info)
    progress_events: list[tuple[float, float | None, int]] = []

    with mock.patch.object(lw, "_load_model", return_value=fake_model):
        segments, words = lw.transcribe_with_words(
            audio,
            model="tiny",
            word_timestamps=True,
            progress_cb=lambda pct, eta, count: progress_events.append((pct, eta, count)),
        )

    assert segments == [{"start": 0.2, "end": 5.0, "text": "halo dunia"}]
    assert words == [
        {"word": "halo", "start": 0.2, "end": 0.7},
        {"word": "dunia", "start": 0.8, "end": 1.4},
    ]
    assert fake_model.transcribe.call_args.kwargs.get("vad_filter") is False
    assert fake_model.transcribe.call_args.kwargs.get("word_timestamps") is True
    assert progress_events and progress_events[0][0] == pytest.approx(50.0)


def test_transcribe_lazy_cuda_fallback_to_cpu(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    info = SimpleNamespace(language="id", duration=5.0)
    seg = SimpleNamespace(start=0.0, end=2.0, text="berhasil cpu", words=[])

    def _cublas_boom():
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        yield  # pragma: no cover

    cuda_model = mock.Mock()
    cuda_model.transcribe.return_value = (_cublas_boom(), info)
    cpu_model = mock.Mock()
    cpu_model.transcribe.return_value = (iter([seg]), info)

    calls: list[tuple[str, str | None, str | None]] = []

    def _fake_load(mdl, dev, comp, log):
        calls.append((mdl, dev, comp))
        return cpu_model if dev == "cpu" else cuda_model

    with mock.patch.object(lw, "_load_model", side_effect=_fake_load):
        out = lw.transcribe(audio, model="tiny")

    assert out == [{"start": 0.0, "end": 2.0, "text": "berhasil cpu"}]
    assert calls == [("tiny", None, None), ("tiny", "cpu", "int8")]


def test_transcribe_explicit_cuda_error_is_specific(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    info = SimpleNamespace(language="id", duration=5.0)

    def _cublas_boom():
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        yield  # pragma: no cover

    cuda_model = mock.Mock()
    cuda_model.transcribe.return_value = (_cublas_boom(), info)

    with mock.patch.object(lw, "_load_model", return_value=cuda_model):
        with pytest.raises(RuntimeError) as exc_info:
            lw.transcribe(audio, model="tiny", device="cuda")

    assert "CUDA 12" in str(exc_info.value)
    assert "model yang lebih kecil" not in str(exc_info.value)


def test_transcribe_oom_preserves_smaller_model_guidance(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    info = SimpleNamespace(language="id", duration=5.0)

    def _oom_boom():
        raise RuntimeError("ct2 out of memory")
        yield  # pragma: no cover

    oom_model = mock.Mock()
    oom_model.transcribe.return_value = (_oom_boom(), info)

    with mock.patch.object(lw, "_load_model", return_value=oom_model):
        with pytest.raises(RuntimeError) as exc_info:
            lw.transcribe(audio, model="tiny")

    assert "gagal di tengah pemrosesan" in str(exc_info.value)
    assert "model yang lebih kecil" in str(exc_info.value)


def test_transcribe_respects_cancel_event(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    info = SimpleNamespace(language="id", duration=5.0)
    seg = SimpleNamespace(start=0.0, end=1.0, text="tes")
    fake_model = mock.Mock()
    fake_model.transcribe.return_value = (iter([seg]), info)

    cancel = threading.Event()
    cancel.set()

    with mock.patch.object(lw, "_load_model", return_value=fake_model):
        with pytest.raises(RuntimeError, match="dibatalkan"):
            lw.transcribe(audio, model="tiny", cancel_event=cancel)


def test_cached_model_list_and_delete(tmp_path):
    hub = tmp_path / "hub"
    model_dir = hub / "models--Systran--faster-whisper-tiny"
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_bytes(b"x" * 2048)

    with mock.patch.object(lw, "_model_cache_dir", return_value=tmp_path):
        listed = lw.list_cached_models()
        assert len(listed) == 1
        assert listed[0]["dir_name"] == "models--Systran--faster-whisper-tiny"
        assert listed[0]["size_bytes"] == 2048

        with pytest.raises(ValueError):
            lw.delete_cached_model("../escape")

        res = lw.delete_cached_model("models--Systran--faster-whisper-tiny")
        assert res["freed_bytes"] == 2048
        assert lw.list_cached_models() == []


def test_normalise_model_preserves_local_paths():
    assert lw.normalise_model(r"D:\Models\faster-whisper-small") == r"D:\Models\faster-whisper-small"
    assert lw.normalise_model("/mnt/models/faster-whisper-turbo") == "/mnt/models/faster-whisper-turbo"


def test_resolve_and_validate_manual_local_model_paths(tmp_path):
    # 1. Direct CTranslate2 folder
    direct = tmp_path / "faster-whisper-small"
    direct.mkdir()
    (direct / "model.bin").write_bytes(b"1" * 1024)
    (direct / "config.json").write_text("{}", encoding="utf-8")

    assert lw.resolve_local_model_path(str(direct)) == direct.resolve()
    assert lw.resolve_local_model_path(str(direct / "model.bin")) == direct.resolve()
    val = lw.validate_local_model_path(str(direct))
    assert val["valid"] is True
    assert val["name"] == "faster-whisper-small"

    # 2. Hugging Face cache layout: models--Org--Name/snapshots/<hash>/model.bin
    hf_repo = tmp_path / "models--Systran--faster-whisper-turbo"
    snap = hf_repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"2" * 2048)

    assert lw.resolve_local_model_path(str(hf_repo)) == snap.resolve()
    val_hf = lw.validate_local_model_path(str(hf_repo))
    assert val_hf["valid"] is True
    assert val_hf["name"] == "Systran/faster-whisper-turbo"
    assert val_hf["resolved_path"] == str(snap.resolve())

    # 3. Rejects PyTorch .pt / whisper.cpp ggml*.bin with clear CTranslate2 guidance
    pt_file = tmp_path / "medium.pt"
    pt_file.write_bytes(b"pt")
    val_pt = lw.validate_local_model_path(str(pt_file))
    assert val_pt["valid"] is False
    assert "CTranslate2" in (val_pt["error"] or "")

    # 4. Rejects empty folder without model.bin
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    val_empty = lw.validate_local_model_path(str(empty))
    assert val_empty["valid"] is False
    assert "model.bin" in (val_empty["error"] or "")


def test_load_model_manual_path_uses_local_files_only(tmp_path):
    direct = tmp_path / "my-whisper"
    direct.mkdir()
    (direct / "model.bin").write_bytes(b"x" * 16)

    fake_cls = mock.Mock(return_value="loaded-instance")
    lw._MODEL_CACHE.clear()
    try:
        with (
            mock.patch.object(lw, "_import_faster_whisper", return_value=fake_cls),
            mock.patch.object(lw, "_pick_device", return_value=("cpu", "int8")),
        ):
            res = lw._load_model(str(direct), "cpu", "int8", lambda _m: None)
        assert res == "loaded-instance"
        fake_cls.assert_called_once_with(
            str(direct.resolve()),
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )
    finally:
        lw._MODEL_CACHE.clear()
