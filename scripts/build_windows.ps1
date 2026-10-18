# Build RemoveBG-RMNZ.exe (jalankan di Windows, PowerShell).
# Pemakaian:  powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
# Hasil:
#   dist\RemoveBG-RMNZ\RemoveBG-RMNZ.exe     (aplikasi portable, satu folder)
#   dist\RemoveBG-RMNZ-windows.zip           (folder di atas di-zip)
#   dist\RemoveBG-RMNZ-Setup.exe             (installer, jika Inno Setup terpasang)
$ErrorActionPreference = "Stop"

$AppName = "RemoveBG-RMNZ"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "==> Membuat venv (Python 3.12)"
    py -3.12 -m venv .venv
    & $Py -m pip install --upgrade pip
    & $Py -m pip install -r requirements.txt
}

$Version = & $Py -c "from app import __version__; print(__version__)"

Write-Host "==> Bersihkan build lama"
Remove-Item -Recurse -Force build, "dist\$AppName", "dist\$AppName-windows.zip", "dist\$AppName-Setup.exe" -ErrorAction SilentlyContinue

Write-Host "==> PyInstaller ($AppName $Version)"
& $Py -m PyInstaller run.py `
    --name $AppName `
    --windowed `
    --noconfirm `
    --clean `
    --collect-binaries onnxruntime `
    --exclude-module PySide6.QtWebEngineCore `
    --exclude-module PySide6.QtWebEngineWidgets `
    --exclude-module PySide6.Qt3DCore `
    --exclude-module PySide6.QtQuick `
    --exclude-module PySide6.QtQml `
    --exclude-module PySide6.QtMultimedia `
    --exclude-module tkinter `
    --exclude-module pytest
if ($LASTEXITCODE -ne 0) { throw "PyInstaller gagal" }

Write-Host "==> ZIP portable"
Compress-Archive -Path "dist\$AppName" -DestinationPath "dist\$AppName-windows.zip" -Force

# Installer opsional via Inno Setup (https://jrsoftware.org/isinfo.php)
$Iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($Iscc) {
    Write-Host "==> Installer (Inno Setup)"
    & $Iscc "/DAppVersion=$Version" "scripts\installer.iss"
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup gagal" }
} else {
    Write-Host "Inno Setup tidak ditemukan, installer dilewati (ZIP portable tetap dibuat)."
}

Write-Host "==> Selesai. Lihat folder dist\"