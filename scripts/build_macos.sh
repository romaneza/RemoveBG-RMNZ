#!/bin/sh
# Build RemoveBG-RMNZ.app + RemoveBG-RMNZ.dmg (jalankan di macOS).
# Pemakaian: sh scripts/build_macos.sh
set -eu

APP_NAME="RemoveBG-RMNZ"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "venv tidak ditemukan. Buat dulu: python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

VERSION="$("$PY" -c 'from app import __version__; print(__version__)')"

echo "==> Bersihkan build lama"
rm -rf build "dist/$APP_NAME" "dist/$APP_NAME.app" "dist/$APP_NAME.dmg" dist/dmg-staging

echo "==> PyInstaller ($APP_NAME $VERSION)"
"$PY" -m PyInstaller run.py \
  --name "$APP_NAME" \
  --windowed \
  --noconfirm \
  --clean \
  --osx-bundle-identifier "com.rmnz.removebg" \
  --collect-binaries onnxruntime \
  --exclude-module PySide6.QtWebEngineCore \
  --exclude-module PySide6.QtWebEngineWidgets \
  --exclude-module PySide6.Qt3DCore \
  --exclude-module PySide6.QtQuick \
  --exclude-module PySide6.QtQml \
  --exclude-module PySide6.QtMultimedia \
  --exclude-module tkinter \
  --exclude-module pytest

# Versi di Info.plist
PLIST="dist/$APP_NAME.app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$PLIST" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :NSHighResolutionCapable bool true" "$PLIST" 2>/dev/null || true

echo "==> Buat DMG"
STAGING="dist/dmg-staging"
mkdir -p "$STAGING"
cp -R "dist/$APP_NAME.app" "$STAGING/"
ln -s /Applications "$STAGING/Applications"
hdiutil create \
  -volname "$APP_NAME" \
  -srcfolder "$STAGING" \
  -ov -format UDZO \
  "dist/$APP_NAME.dmg"
rm -rf "$STAGING"

echo "==> Selesai: dist/$APP_NAME.app dan dist/$APP_NAME.dmg"