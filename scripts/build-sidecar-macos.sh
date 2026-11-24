#!/usr/bin/env bash
# Builds the standalone Python sidecar binary for macOS via PyInstaller.
#
# Usage:
#   ./scripts/build-sidecar-macos.sh [--arch arm64|x86_64]
#   PYTHON=/path/to/python3 ./scripts/build-sidecar-macos.sh

set -euo pipefail

ARCH_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
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
    TARGET_TRIPLE="aarch64-apple-darwin"
    ;;
  x86_64|amd64)
    TARGET_TRIPLE="x86_64-apple-darwin"
    ;;
  *)
    echo "Unsupported macOS architecture: $HOST_ARCH" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPEC="$ROOT/build/spec/sidecar.spec"
BINARY_DIR="$ROOT/src-tauri/binaries"
WORK_DIR="$ROOT/build/sidecar-macos"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p "$BINARY_DIR" "$WORK_DIR"

echo "[build-sidecar-macos] Target triple: $TARGET_TRIPLE"
echo "[build-sidecar-macos] Python: $PYTHON_BIN"

SIDECAR_TARGET_TRIPLE="$TARGET_TRIPLE" "$PYTHON_BIN" -m PyInstaller \
  --clean \
  --noconfirm \
  --distpath "$BINARY_DIR" \
  --workpath "$WORK_DIR" \
  "$SPEC"

OUT="$BINARY_DIR/ytclip-sidecar-$TARGET_TRIPLE"
if [[ ! -f "$OUT" ]]; then
  echo "ERROR: Sidecar binary was not created: $OUT" >&2
  exit 1
fi

chmod +x "$OUT"
echo "[build-sidecar-macos] Built: $OUT ($(du -h "$OUT" | cut -f1))"
