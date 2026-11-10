@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem  One-shot A133 pipeline: push -> bench -> smoke gen -> RSS
rem  -> pull logs to deploy/logs/<timestamp>
rem
rem  Usage:
rem    run_a133_all.bat                 Q4_K_M, 4 threads
rem    run_a133_all.bat Q8              Q8_0
rem    run_a133_all.bat both            Q4 + Q8
rem    run_a133_all.bat --threads 4
rem    run_a133_all.bat --upload       also copy logs to a remote host
rem ============================================================

set "HERE=%~dp0"

rem Capture everything to a transcript: a double-clicked window then still shows errors.
if defined COPA3_IN_WRAPPER goto :main
set "COPA3_IN_WRAPPER=1"
if not exist "%HERE%logs" mkdir "%HERE%logs" >NUL 2>&1
set "TS0="
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss" 2^>NUL') do set "TS0=%%i"
if not defined TS0 set "TS0=%RANDOM%%RANDOM%"
set "RUNLOG=%HERE%logs\console_%TS0%.txt"
echo [info] transcript: %RUNLOG%
echo [info] running... (GGUF push may take a few minutes; output shown at end)
cmd /c call "%~f0" %* > "%RUNLOG%" 2>&1
echo.
echo ==== transcript: %RUNLOG% ====
for %%A in ("%RUNLOG%") do echo [info] transcript bytes=%%~zA
type "%RUNLOG%"
echo.
echo [done] press any key to close
pause >NUL
exit /b

:main
if defined COPA3_IN_WRAPPER echo on
echo [info] %date% %time%  args=[%*]  cwd=%CD%
pushd "%~dp0" >nul 2>&1
set "ADB=adb"
set "SSH_HOST=USER@TRAIN_HOST"
set "SRC_DIR=/path/to/data/deploy"
set "DEV_DIR=/data/local/tmp/copa3"
set "DEV_LLAMA=%DEV_DIR%/llama"
set "LLAMA_NAME=armv7-a53"
set "LLAMA_PKG=%HERE%llama-android-armv7"
set "THREADS=4"
set "WHICH=Q4"
set "DO_UPLOAD=0"

:parse
if "%~1"=="" goto after_parse
if /i "%~1"=="--upload" goto arg_upload
if /i "%~1"=="--threads" goto arg_threads
if /i "%~1"=="Q4" goto arg_q4
if /i "%~1"=="Q8" goto arg_q8
if /i "%~1"=="both" goto arg_both
echo [ERROR] unknown argument: %~1
goto usage

:arg_upload
set "DO_UPLOAD=1"
shift
goto parse

:arg_threads
set "THREADS=%~2"
shift
shift
goto parse

:arg_q4
set "WHICH=Q4"
shift
goto parse

:arg_q8
set "WHICH=Q8"
shift
goto parse

:arg_both
set "WHICH=BOTH"
shift
goto parse
:after_parse

if /i "%WHICH%"=="Q4" set "MODELS=copa3_Q4_K_M.gguf"
if /i "%WHICH%"=="Q8" set "MODELS=copa3_Q8_0.gguf"
if /i "%WHICH%"=="BOTH" set "MODELS=copa3_Q4_K_M.gguf copa3_Q8_0.gguf"
echo [trace] WHICH=%WHICH% MODELS=%MODELS%

rem ---- local assets -------------------------------------------------
set "NEED_FETCH=0"
for %%M in (%MODELS%) do (
    if not exist "%%M" set "NEED_FETCH=1"
)
echo [trace] NEED_FETCH=!NEED_FETCH! cwd=%CD%
set "HAVE_PKG=0"
if exist "%LLAMA_PKG%\llama-completion" set "HAVE_PKG=1"
if "!HAVE_PKG!"=="0" (
    echo [ERROR] missing %LLAMA_PKG%\llama-completion
    echo         build it on a Linux host: bash deploy/build_android_armv7.sh
    goto fail
)
if "!NEED_FETCH!"=="1" (
    echo [fetch] model files missing, pulling from %SSH_HOST%
    call "%HERE%deploy_a133.bat" --fetch || goto fail
)

rem ---- adb / device -------------------------------------------------
where adb >NUL 2>&1
if not errorlevel 1 (
    set "ADB=adb"
) else if exist "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" (
    set "ADB=%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"
) else if exist "C:\platform-tools\adb.exe" (
    set "ADB=C:\platform-tools\adb.exe"
) else if exist "%USERPROFILE%\platform-tools\adb.exe" (
    set "ADB=%USERPROFILE%\platform-tools\adb.exe"
) else (
    echo [ERROR] adb not found in PATH, LOCALAPPDATA, C:\platform-tools or %%USERPROFILE%%\platform-tools
    goto fail
)
echo [info] adb = %ADB%
"%ADB%" start-server >nul 2>&1
set "NDEV=0"
for /f "skip=1 tokens=1,2" %%a in ('"%ADB%" devices') do (
    if /i "%%b"=="device" set "NDEV=1"
)
if "!NDEV!"=="0" (
    echo [ERROR] no adb device. Check USB debugging / authorization.
    "%ADB%" devices
    goto fail
)

if defined COPA3_IN_WRAPPER echo off
echo [trace] adb device ok, pushing
"%ADB%" shell "mkdir -p %DEV_DIR%" || goto fail

rem ---- push models (checksums inside deploy_a133.bat) ---------------
echo [step 1/5] push models
call "%HERE%deploy_a133.bat" %WHICH% || goto fail

rem ---- push llama.cpp android build once ----------------------------
echo [step 2/5] push llama.cpp (%LLAMA_NAME%)
set "HAVE="
for /f "tokens=*" %%v in ('"%ADB%" shell "cat %DEV_LLAMA%/BUILD 2^>nul"') do set "HAVE=%%v"
if /i not "!HAVE!"=="%LLAMA_NAME%" (
    "%ADB%" shell "rm -rf %DEV_LLAMA% && mkdir -p %DEV_LLAMA%" || goto fail
    "%ADB%" push "%LLAMA_PKG%\." "%DEV_LLAMA%/" || goto fail
    "%ADB%" shell "echo %LLAMA_NAME% > %DEV_LLAMA%/BUILD" || goto fail
) else (
    echo [ok] already on device
)

"%ADB%" push "%HERE%a133_run_all.sh" "%DEV_DIR%/a133_run_all.sh" || goto fail

rem ---- run on device -----------------------------------------------
echo [step 3/5] run tests (threads=%THREADS%, models=%MODELS%)
"%ADB%" shell "cd %DEV_DIR% && sh ./a133_run_all.sh %THREADS% %DEV_LLAMA% %MODELS%"
if errorlevel 1 (
    echo [ERROR] device pipeline failed, pulling logs anyway
)

rem ---- pull logs ----------------------------------------------------
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss" 2^>nul') do set "TS=%%i"
if not defined TS set "TS=%RANDOM%%RANDOM%"
set "LOGDIR=%HERE%logs\%TS%"
echo [step 4/5] pull logs to %LOGDIR%
"%ADB%" pull "%DEV_DIR%/logs" "%LOGDIR%" || goto fail

rem ---- summary ------------------------------------------------------
echo.
echo ==== SUMMARY (%LOGDIR%) ====
type "%LOGDIR%\sysinfo.txt" 2>nul
findstr /i /c:"t/s" "%LOGDIR%\bench_*.txt" 2>nul
findstr /i /c:"peak_rss_kb" "%LOGDIR%\gen_*.txt" 2>nul

rem ---- optional upload ----------------------------------------------
if "%DO_UPLOAD%"=="1" (
    echo [step 5/5] upload logs to %SSH_HOST%
    where scp >nul 2>&1
    if errorlevel 1 (
        echo [warn] scp not found, skip upload
    ) else (
        "%ADB%" shell "true"
        ssh "%SSH_HOST%" "mkdir -p %SRC_DIR%/logs" || goto fail
        scp -r "%LOGDIR%" "%SSH_HOST%:%SRC_DIR%/logs/" || goto fail
        echo [ok] uploaded to %SSH_HOST%:%SRC_DIR%/logs/%TS%
    )
)

echo.
echo [done] logs: %LOGDIR%
goto :eof

:usage
echo Usage: run_a133_all.bat [Q4^|Q8^|both] [--threads N] [--upload]
:fail
echo [fail] exiting with code 1
if not defined COPA3_IN_WRAPPER pause >NUL
exit /b 1
