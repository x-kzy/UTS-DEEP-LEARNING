@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title UTS DEEP LEARNING - Upload Repository
where git >nul 2>&1
if errorlevel 1 (
 echo ERROR: Git for Windows belum ditemukan.
 echo Install dari https://git-scm.com/download/win
 pause
 exit /b 1
)
where powershell >nul 2>&1
if errorlevel 1 (
 echo ERROR: PowerShell Windows tidak ditemukan.
 pause
 exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\push_to_github.ps1"
set "result=%errorlevel%"
if not "%result%"=="0" echo Proses tidak berhasil. Periksa pesan di atas.
pause
exit /b %result%
