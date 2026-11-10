@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem  copa3 114M  (A133 tablet) deployment via adb
rem
rem  Usage:
rem    deploy_a133.bat                 push Q4_K_M + bench.sh
rem    deploy_a133.bat Q8              push Q8_0 instead
rem    deploy_a133.bat both            push Q4 + Q8
rem    deploy_a133.bat --fetch         copy GGUFs from 175 first
rem    deploy_a133.bat --bench         run llama-bench on device
rem    deploy_a133.bat --fetch --bench combined
rem
rem  Files are resolved next to this script (Z:\SDK\paper\indo02b\deploy\).
rem ============================================================

set "HERE=%~dp0"
if not defined ADB set "ADB=adb"
set "SSH_HOST=USER@TRAIN_HOST"
set "SRC_DIR=/path/to/data/deploy"
set "DEV_DIR=/data/local/tmp/copa3"
set "LLAMA_DIR=/data/local/tmp/llama.cpp"
set "THREADS=4"

set "WHICH=Q4"
set "DO_FETCH=0"
set "DO_BENCH=0"

:parse
if "%~1"=="" goto after_parse
if /i "%~1"=="--fetch" (set "DO_FETCH=1" & shift & goto parse)
if /i "%~1"=="--bench" (set "DO_BENCH=1" & shift & goto parse)
if /i "%~1"=="Q4"      (set "WHICH=Q4"   & shift & goto parse)
if /i "%~1"=="Q8"      (set "WHICH=Q8"   & shift & goto parse)
if /i "%~1"=="both"    (set "WHICH=BOTH" & shift & goto parse)
echo [ERROR] unknown argument: %~1
goto usage
:after_parse

if /i "%WHICH%"=="Q4" set "MODELS=copa3_Q4_K_M.gguf"
if /i "%WHICH%"=="Q8" set "MODELS=copa3_Q8_0.gguf"
if /i "%WHICH%"=="BOTH" set "MODELS=copa3_Q4_K_M.gguf copa3_Q8_0.gguf"

if "%DO_FETCH%"=="1" (
    call :fetch || goto fail
)

if /i "%ADB%"=="adb" (
    where adb >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] adb not found in PATH. Install platform-tools and retry.
        goto fail
    )
) else (
    if not exist "%ADB%" (
        echo [ERROR] adb not found: %ADB%
        goto fail
    )
)

"%ADB%" start-server >nul 2>&1
set "NDEV=0"
for /f "skip=1 tokens=1,2" %%a in ('"%ADB%" devices') do (
    if /i "%%b"=="device" set "NDEV=1"
)
if "!NDEV!"=="0" (
    echo [ERROR] no device in state "device". Check USB debugging / authorization:
    "%ADB%" devices
    goto fail
)

echo [info] device:
"%ADB%" shell "getprop ro.product.model; getprop ro.build.version.release; grep MemTotal /proc/meminfo"

"%ADB%" shell "mkdir -p %DEV_DIR%" || goto fail

for %%M in (%MODELS%) do (
    call :push "%HERE%%%M" || goto fail
)
call :push "%HERE%bench.sh" || goto fail

if "%DO_BENCH%"=="1" (
    call :bench || goto fail
)

echo.
echo [done] files in %DEV_DIR%:
"%ADB%" shell "ls -l %DEV_DIR%"
echo.
echo Next steps (see deploy\README.md):
echo   RSS: run llama-completion on device, then: adb shell "ps -A -o RSS,NAME ^| findstr llama"
goto :eof

rem ------------------------------------------------------------
:push
set "SRC=%~1"
if not exist "%SRC%" (
    echo [ERROR] missing file: %SRC%
    echo         use --fetch to copy it from %SSH_HOST%:%SRC_DIR%
    exit /b 1
)
for %%F in ("%SRC%") do (
    set "NAME=%%~nxF"
    set "SIZE=%%~zF"
)
echo [push] !NAME! (!SIZE! bytes) -> !DEV_DIR!
"%ADB%" push "%SRC%" "!DEV_DIR!/!NAME!" || exit /b 1

set "RSIZE="
for /f "tokens=1" %%s in ('"%ADB%" shell "wc -c ^< !DEV_DIR!/!NAME!" 2^>nul') do set "RSIZE=%%s"
if not "!RSIZE!"=="!SIZE!" (
    echo [ERROR] size mismatch: local=!SIZE! device=!RSIZE!
    exit /b 1
)
echo [ok] size verified

set "LH="
for /f "skip=1 delims=" %%h in ('certutil -hashfile "%SRC%" SHA256 2^>nul') do (
    if not defined LH set "LH=%%h"
)
set "LH=!LH: =!"
set "RH="
for /f "tokens=1" %%h in ('"%ADB%" shell "sha256sum !DEV_DIR!/!NAME! 2^>nul"') do set "RH=%%h"
if defined RH (
    if /i "!LH!"=="!RH!" (
        echo [ok] sha256 match
    ) else (
        echo [warn] sha256 differs: local=!LH! device=!RH!
    )
) else (
    echo [warn] sha256sum unavailable on device, skipped
)
exit /b 0

rem ------------------------------------------------------------
:fetch
where scp >nul 2>&1
if errorlevel 1 (
    echo [ERROR] scp not found in PATH.
    exit /b 1
)
echo [fetch] from %SSH_HOST%:%SRC_DIR%
if not exist "%HERE%copa3_Q4_K_M.gguf" (
    scp "%SSH_HOST%:%SRC_DIR%/copa3_Q4_K_M.gguf" "%HERE%copa3_Q4_K_M.gguf" || exit /b 1
)
if not exist "%HERE%copa3_Q8_0.gguf" (
    scp "%SSH_HOST%:%SRC_DIR%/copa3_Q8_0.gguf" "%HERE%copa3_Q8_0.gguf" || exit /b 1
)
if not exist "%HERE%bench.sh" (
    scp "%SSH_HOST%:%SRC_DIR%/bench.sh" "%HERE%bench.sh" || exit /b 1
)
exit /b 0

rem ------------------------------------------------------------
:bench
if /i "%WHICH%"=="BOTH" (
    set "BMODEL=copa3_Q4_K_M.gguf"
) else (
    set "BMODEL=!MODELS!"
)
echo [bench] llama-bench %BMODEL% (threads=%THREADS%)
"%ADB%" shell "ls %DEV_DIR%/llama/llama-bench" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] %DEV_DIR%/llama/llama-bench not found on device (run run_a133_all.bat first).
    echo         Build it first, or set LLAMA_DIR at the top of this script.
    exit /b 1
)
"%ADB%" shell "cd %DEV_DIR% && BIN=%DEV_DIR%/llama/llama-bench sh ./bench.sh %BMODEL% %THREADS%"
exit /b 0

rem ------------------------------------------------------------
:usage
echo Usage: deploy_a133.bat [Q4^|Q8^|both] [--fetch] [--bench]
:fail
exit /b 1
