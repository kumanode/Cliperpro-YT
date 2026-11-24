# Cliperpro-YT — Improvement Plan

Dokumen ini merangkum rencana peningkatan bertahap untuk **Cliperpro-YT** (`yt-short-clipper-v2`), mulai dari stabilitas & UX transkripsi lokal, percepatan pipeline render FFmpeg, kontrol editing sebelum render, manajemen storage, hingga maintainability & automated testing.

---

## Status Saat Ini (Phase 0 — Selesai)

- [x] **Fix CUDA 12 DLL crash (`cublas64_12.dll`) pada transkripsi lokal**
  - File: `yt_short_clipper_core/local_whisper.py`
  - Validasi ketersediaan `cublas64_12.dll` & `cublasLt64_12.dll` sebelum memilih `cuda/float16`.
  - Auto-fallback ke `cpu/int8` jika CUDA gagal saat load maupun secara *lazy* pada `encode()` pertama.
  - Pesan error eksplisit saat device dipaksa ke `cuda`.

---

## Phase 1 — Transkripsi Lokal (`faster-whisper`): Akurasi, Progres & Manajemen Model

- [x] **1.1 Word-Level Timestamps untuk Caption Karaoke**
  - `local_whisper.py`: `transcribe_with_words(..., word_timestamps=True)` mengekstrak `seg.words`.
  - `session.py`: `_transcribe_audio_local()` menyimpan `words.json` dan menandai `caption_available: True` di `session_data.json`.
- [x] **1.2 Real-Time Progress (%) & ETA**
  - `local_whisper.py`: menghitung `(seg.end / info.duration) * 100` + ETA dengan throttling (tiap >= 2% atau >= 1.5s).
  - `ProcessingPage.tsx`: menampilkan progress bar & ETA transkripsi lokal secara real-time.
- [x] **1.3 Panel Status Device & Cache Model di Settings**
  - `sidecar.py`, `src-tauri/src/commands/mod.rs`, `src-tauri/src/lib.rs`, `src/hooks/appConfig.ts`: command `probe_local_whisper` & `delete_local_whisper_model`.
  - `AIModelsPage.tsx` & `CreatePage.tsx`: dropdown `Auto` / `CPU` / `CUDA`, status engine aktif + info fallback CUDA, daftar model terunduh + ukuran MB & tombol hapus cache.

---

## Phase 2 — Kecepatan Render & Kualitas Pipeline Video

- [x] **2.1 Single-Pass FFmpeg Filter Graph untuk Overlay**
  - Modul baru: `yt_short_clipper_core/overlay_compositor.py` (`build_single_pass_graph` & `compose_clip_overlays`).
  - `clip_processor.py`: menggabungkan `hook` + `captions` ASS + `logo watermark` + `credit text` + opsional `BGM` ke dalam satu pass FFmpeg (dengan fallback otomatis ke sequential passes jika gagal).
- [x] **2.2 Clean Job Cancellation**
  - Modul baru: `yt_short_clipper_core/job_control.py` (`cancel_event`, `request_cancel`, `run_cancellable_subprocess`).
  - `src-tauri/src/commands/mod.rs` & `lib.rs`: pelacakan `ACTIVE_SIDECAR_PID` + command `cancel_active_job` (`taskkill /PID <pid> /T /F` di Windows agar `ffmpeg.exe` tidak yatim).
  - `ProcessingPage.tsx` & `ProcessingClipsPage.tsx`: tombol **Cancel** aktif selama proses berjalan.

---

## Phase 3 — Kontrol & Review Sebelum Render

- [x] **3.1 Fine-Tune Trim (`start`/`end`)**
  - `sessionStore.ts` & `HighlightSelectionPage.tsx`: tombol nudge `-5s / -1s / +1s / +5s` untuk awal & akhir setiap highlight sebelum render.
- [x] **3.2 Quick Caption Editor Sebelum Burn-In**
  - `sessionStore.ts`, `HighlightSelectionPage.tsx`, `clip_processor.py` (`_apply_caption_override`): edit teks caption per highlight sebelum burn-in tanpa transkripsi ulang.
- [x] **3.3 Background Music (BGM) dengan Auto-Ducking**
  - `ProcessConfirmDialog.tsx`, `processClips.ts`, `clip_processor.py`, `overlay_compositor.py`: pilih file audio BGM + slider volume (`5%–60%`), diproses dengan filter FFmpeg `sidechaincompress` dalam single-pass.

---

## Phase 4 — Storage Manager

- [x] **Storage Manager Backend & UI**
  - Modul baru: `yt_short_clipper_core/storage_manager.py` (`get_storage_summary`, `cleanup_storage`, sinkronisasi `database/video_history.json`).
  - `sidecar.py`, `src-tauri/src/commands/mod.rs`, `src-tauri/src/lib.rs`, `src/hooks/appConfig.ts`: command `get_storage_summary` & `cleanup_storage`.
  - `SettingsPage.tsx`: panel **Storage Manager** dengan ringkasan disk (`source_videos`, `sessions`, `whisper_models`), tombol bersih-bersih `> 7 hari`, hapus per file, dan hapus semua cache source video.

---

## Phase 5 — Stabilitas, Refactor & Automated Testing

- [x] **5.1 Konsolidasi Test Suite ke `tests/` (`pytest`) & CI**
  - `tests/test_local_whisper.py`, `tests/test_overlay_and_clips.py`, `tests/test_storage_manager.py` (22 unit test, path relatif lintas OS).
  - `.github/workflows/python-tests.yml`: CI otomatis menjalankan `pytest tests -v`.
- [x] **5.2 Modularisasi Backend**
  - Tanggung jawab baru dipisah ke modul terfokus: `job_control.py`, `overlay_compositor.py`, dan `storage_manager.py`.

---

## Langkah Selanjutnya (Build Rilis)

- [ ] Jalankan `scripts/build-sidecar.ps1` lalu `npm run tauri build` (atau `npm run release`) untuk memaketkan binary sidecar & aplikasi Tauri terbaru.
