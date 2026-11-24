#!/usr/bin/env bash
# Downloads bundled macOS runtime dependencies (static ffmpeg, deno, and
# MediaPipe face_landmarker.task) into src-tauri/binaries so Tauri can bundle
# them inside Cliperpro.app/Contents/Resources.
#
# Usage:
#   ./scripts/fetch-deps-macos.sh [--force] [--arch arm64|x86_64]

set -euo pipefail

FORCE=0
ARCH_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force|-f)
      FORCE=1
      shift
      ;;
    --arch)
      ARCH_OVERRIDE="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

HOST_ARCH="${ARCH_OVERRIDE:-$(uname -m)}"
case "$HOST_ARCH" in
  arm64|aarch64)
    FFMPEG_ARCH="arm64"
    DENO_TRIPLE="aarch64-apple-darwin"
    ;;
  x86_64|amd64)
    FFMPEG_ARCH="amd64"
    DENO_TRIPLE="x86_64-apple-darwin"
    ;;
  *)
    echo "Unsupported macOS architecture: $HOST_ARCH" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINARIES_DIR="$ROOT/src-tauri/binaries"
FFMPEG_DIR="$BINARIES_DIR/ffmpeg"
BIN_DIR="$BINARIES_DIR/bin"
MODELS_DIR="$BINARIES_DIR/models"
TEMP_DIR="$ROOT/build/deps-temp-macos"

mkdir -p "$FFMPEG_DIR" "$BIN_DIR" "$MODELS_DIR" "$TEMP_DIR"
trap 'rm -rf "$TEMP_DIR"' EXIT

# ---------------------------------------------------------------------------
# 1. Static FFmpeg (martin-riedl.de — GPL static macOS build with libass,
#    libx264, freetype, and VideoToolbox hardware encoding)
# ---------------------------------------------------------------------------
FFMPEG_BIN="$FFMPEG_DIR/ffmpeg"
if [[ -x "$FFMPEG_BIN" && "$FORCE" -eq 0 ]]; then
  echo "[fetch-deps-macos] ffmpeg already present, skipping."
else
  echo "[fetch-deps-macos] Downloading static ffmpeg (macos/$FFMPEG_ARCH)..."
  FFMPEG_URL="https://ffmpeg.martin-riedl.de/redirect/latest/macos/${FFMPEG_ARCH}/release/ffmpeg.zip"
  curl -fSL --retry 3 --retry-delay 2 "$FFMPEG_URL" -o "$TEMP_DIR/ffmpeg.zip"
  unzip -qo "$TEMP_DIR/ffmpeg.zip" -d "$TEMP_DIR/ffmpeg-extract"
  FOUND_FFMPEG="$(find "$TEMP_DIR/ffmpeg-extract" -type f -name "ffmpeg" | head -n 1)"
  if [[ -z "$FOUND_FFMPEG" ]]; then
    echo "ERROR: ffmpeg binary not found in archive" >&2
    exit 1
  fi
  cp "$FOUND_FFMPEG" "$FFMPEG_BIN"
  chmod +x "$FFMPEG_BIN"
  echo "[fetch-deps-macos] ffmpeg ready: $FFMPEG_BIN"
fi

# ---------------------------------------------------------------------------
# 2. Deno (for yt-dlp remote-components / YouTube JS challenges)
# ---------------------------------------------------------------------------
DENO_BIN="$BIN_DIR/deno"
if [[ -x "$DENO_BIN" && "$FORCE" -eq 0 ]]; then
  echo "[fetch-deps-macos] deno already present, skipping."
else
  echo "[fetch-deps-macos] Downloading deno ($DENO_TRIPLE)..."
  DENO_URL="https://github.com/denoland/deno/releases/latest/download/deno-${DENO_TRIPLE}.zip"
  curl -fSL --retry 3 --retry-delay 2 "$DENO_URL" -o "$TEMP_DIR/deno.zip"
  unzip -qo "$TEMP_DIR/deno.zip" -d "$TEMP_DIR/deno-extract"
  FOUND_DENO="$(find "$TEMP_DIR/deno-extract" -type f -name "deno" | head -n 1)"
  if [[ -z "$FOUND_DENO" ]]; then
    echo "ERROR: deno binary not found in archive" >&2
    exit 1
  fi
  cp "$FOUND_DENO" "$DENO_BIN"
  chmod +x "$DENO_BIN"
  echo "[fetch-deps-macos] deno ready: $DENO_BIN"
fi

# ---------------------------------------------------------------------------
# 3. MediaPipe face_landmarker model
# ---------------------------------------------------------------------------
MODEL_FILE="$MODELS_DIR/face_landmarker.task"
MODEL_SIZE=0
if [[ -f "$MODEL_FILE" ]]; then
  MODEL_SIZE="$(wc -c < "$MODEL_FILE" | tr -d ' ')"
fi
if [[ "$MODEL_SIZE" -ge 1000000 && "$FORCE" -eq 0 ]]; then
  echo "[fetch-deps-macos] face_landmarker.task already present, skipping."
else
  echo "[fetch-deps-macos] Downloading MediaPipe face_landmarker.task..."
  curl -fSL --retry 3 --retry-delay 2 \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
    -o "$MODEL_FILE"
  echo "[fetch-deps-macos] model ready: $MODEL_FILE"
fi

echo "[fetch-deps-macos] All macOS dependencies ready in $BINARIES_DIR"
