@echo off
setlocal enabledelayedexpansion

rem ============================================================================
rem Unified launcher for Falcor render demos (Windows / D3D12 backend).
rem Interactive menus mirroring run.sh.
rem
rem Usage:
rem   run.bat                          interactive: pick script + scene
rem   run.bat <script> [scene] [anim]  direct mode (same as run.sh)
rem   run.bat ncr living-room anim
rem   run.bat ptnrd kitchen
rem   Options: --width/-w N   --height/-h N   --device/-d d3d12|vulkan   --gpu N
rem   Set NCR_RUN_DRYRUN=1 to print the command without launching.
rem   Set NCR_BUILD_DIR=build\windows-vs2022-vk to launch a different build tree.
rem ============================================================================

set "ROOT=%~dp0"
if not defined NCR_BUILD_DIR set "NCR_BUILD_DIR=build\windows-vs2022"
set "BIN=%ROOT%%NCR_BUILD_DIR%\bin\Release\Mogwai.exe"
set "SCENE_ROOT=%ROOT%media\scenes"

if not exist "%BIN%" (
    echo Error: Mogwai.exe not found at %BIN%
    echo Build first: cmake --build build\windows-vs2022 --config Release
    exit /b 1
)

set WIDTH=1280
set HEIGHT=720
set "DEVICE_ARGS="
set "SCRIPT_KEY="
set "SCENE_NAME="
set "SCENE_SUFFIX="
set "SCENE_SUFFIX_SET=0"

rem ---------------------------------------------------------------------------
rem Script registry (alphabetical, same set as run.sh).
rem ---------------------------------------------------------------------------
set SCRIPT_COUNT=10
set "SCRIPT_KEY_1=bsdf"    & set "SCRIPT_FILE_1=scripts\BSDFViewer.py"
set "SCRIPT_KEY_2=debug"   & set "SCRIPT_FILE_2=scripts\SceneDebugger.py"
set "SCRIPT_KEY_3=gbuffer" & set "SCRIPT_FILE_3=scripts\GBuffer.py"
set "SCRIPT_KEY_4=mpt"     & set "SCRIPT_FILE_4=scripts\MinimalPathTracer.py"
set "SCRIPT_KEY_5=ncr"     & set "SCRIPT_FILE_5=scripts\NeuralRadiosity.py"
set "SCRIPT_KEY_6=pt"      & set "SCRIPT_FILE_6=scripts\PathTracer.py"
set "SCRIPT_KEY_7=ptdn"    & set "SCRIPT_FILE_7=scripts\OptixDenoiser.py"
set "SCRIPT_KEY_8=ptnrd"   & set "SCRIPT_FILE_8=scripts\PathTracerNRD_VK.py"
set "SCRIPT_KEY_9=rtxdi"   & set "SCRIPT_FILE_9=scripts\RTXDI.py"
set "SCRIPT_KEY_10=ward"   & set "SCRIPT_FILE_10=scripts\WARDiffPathTracer.py"

rem ---------------------------------------------------------------------------
rem Argument parsing.
rem ---------------------------------------------------------------------------
:argloop
if "%~1"=="" goto argdone
set "A=%~1"
if /i "%A%"=="anim"       ( set "SCENE_SUFFIX=anim" & set "SCENE_SUFFIX_SET=1" & shift & goto argloop )
if /i "%A%"=="animation"  ( set "SCENE_SUFFIX=anim" & set "SCENE_SUFFIX_SET=1" & shift & goto argloop )
if /i "%A%"=="static"     ( set "SCENE_SUFFIX=static" & set "SCENE_SUFFIX_SET=1" & shift & goto argloop )
if /i "%A%"=="v4"         ( set "SCENE_SUFFIX=static" & set "SCENE_SUFFIX_SET=1" & shift & goto argloop )
if /i "%A%"=="--width"    ( set "WIDTH=%~2" & shift & shift & goto argloop )
if /i "%A%"=="-w"         ( set "WIDTH=%~2" & shift & shift & goto argloop )
if /i "%A%"=="--height"   ( set "HEIGHT=%~2" & shift & shift & goto argloop )
if /i "%A%"=="-h"         ( set "HEIGHT=%~2" & shift & shift & goto argloop )
if /i "%A%"=="--device"   ( set "DEVICE_ARGS=!DEVICE_ARGS! --device-type=%~2" & shift & shift & goto argloop )
if /i "%A%"=="-d"         ( set "DEVICE_ARGS=!DEVICE_ARGS! --device-type=%~2" & shift & shift & goto argloop )
if /i "%A%"=="--gpu"      ( set "DEVICE_ARGS=!DEVICE_ARGS! --gpu=%~2" & shift & shift & goto argloop )
if not defined SCRIPT_KEY ( set "SCRIPT_KEY=%A%" & shift & goto argloop )
if not defined SCENE_NAME ( set "SCENE_NAME=%A%" & shift & goto argloop )
echo Unknown argument: %A%
exit /b 1
:argdone

rem ---------------------------------------------------------------------------
rem Script selection (interactive menu when not given on the command line).
rem ---------------------------------------------------------------------------
if not defined SCRIPT_KEY (
    echo Available scripts:
    for /l %%i in (1,1,%SCRIPT_COUNT%) do echo   [%%i] !SCRIPT_KEY_%%i!
    echo.
    set "SCRIPT_CHOICE="
    set /p SCRIPT_CHOICE="Select script [1]: "
)

if not defined SCRIPT_CHOICE goto script_resolved
if defined SCRIPT_CHOICE set "SCRIPT_CHOICE=!SCRIPT_CHOICE: =!"
if not defined SCRIPT_CHOICE goto script_resolved
set "NCR_NUMCHK=!SCRIPT_CHOICE!"
for %%d in (0 1 2 3 4 5 6 7 8 9) do if defined NCR_NUMCHK set "NCR_NUMCHK=!NCR_NUMCHK:%%d=!"
if not defined NCR_NUMCHK (
    rem numeric index -> lookup via for (runtime-safe dynamic variable name)
    if !SCRIPT_CHOICE! GEQ 1 if !SCRIPT_CHOICE! LEQ !SCRIPT_COUNT! (
        for %%i in (!SCRIPT_CHOICE!) do set "SCRIPT_KEY=!SCRIPT_KEY_%%i!"
    ) else (
        set "SCRIPT_KEY=!SCRIPT_CHOICE!"
    )
) else (
    rem name given directly
    set "SCRIPT_KEY=!SCRIPT_CHOICE!"
)
:script_resolved

if not defined SCRIPT_KEY set "SCRIPT_KEY=bsdf"
rem resolve registry key -> file (fall back to treating it as a direct path)
set "SCRIPT_PATH="
for /l %%i in (1,1,%SCRIPT_COUNT%) do (
    if /i "!SCRIPT_KEY!"=="!SCRIPT_KEY_%%i!" set "SCRIPT_PATH=!SCRIPT_FILE_%%i!"
)
if not defined SCRIPT_PATH (
    if exist "%ROOT%!SCRIPT_KEY!" set "SCRIPT_PATH=!SCRIPT_KEY!"
)
if not defined SCRIPT_PATH (
    echo Error: unknown script '%SCRIPT_KEY%'
    exit /b 1
)
if not exist "%ROOT%!SCRIPT_PATH!" (
    echo Error: script not found: %ROOT%!SCRIPT_PATH!
    exit /b 1
)

rem ---------------------------------------------------------------------------
rem Scene discovery.
rem ---------------------------------------------------------------------------
set SCENE_COUNT=0
if not exist "%SCENE_ROOT%" (
    echo Error: scene directory not found: %SCENE_ROOT%
    exit /b 1
)
for /d %%D in ("%SCENE_ROOT%\*") do (
    set /a SCENE_COUNT+=1
    set "SCENE_!SCENE_COUNT!=%%~nxD"
)
if %SCENE_COUNT% EQU 0 (
    echo Error: no scenes found under %SCENE_ROOT%
    exit /b 1
)

rem ---------------------------------------------------------------------------
rem Scene selection (interactive menu when not given on the command line).
rem ---------------------------------------------------------------------------
if not defined SCENE_NAME (
    echo Available scenes:
    for /l %%i in (1,1,%SCENE_COUNT%) do (
        set "HAS_ANIM="
        set "HAS_STATIC="
        if exist "%SCENE_ROOT%\!SCENE_%%i!\scene-animation.pyscene" set "HAS_ANIM=1"
        if exist "%SCENE_ROOT%\!SCENE_%%i!\scene-v4.pyscene" set "HAS_STATIC=1"
        set "MARK="
        if defined HAS_ANIM if defined HAS_STATIC set "MARK=(anim/static)"
        if defined HAS_ANIM if not defined HAS_STATIC set "MARK=(anim)"
        if not defined HAS_ANIM if defined HAS_STATIC set "MARK=(static)"
        echo   [%%i] !SCENE_%%i! !MARK!
    )
    echo.
    set "SCENE_CHOICE="
    set /p SCENE_CHOICE="Select scene [living-room]: "
)

if not defined SCENE_CHOICE goto scene_resolved
if defined SCENE_CHOICE set "SCENE_CHOICE=!SCENE_CHOICE: =!"
if not defined SCENE_CHOICE goto scene_resolved
set "NCR_NUMCHK=!SCENE_CHOICE!"
for %%d in (0 1 2 3 4 5 6 7 8 9) do if defined NCR_NUMCHK set "NCR_NUMCHK=!NCR_NUMCHK:%%d=!"
if not defined NCR_NUMCHK (
    if !SCENE_CHOICE! GEQ 1 if !SCENE_CHOICE! LEQ !SCENE_COUNT! (
        for %%i in (!SCENE_CHOICE!) do set "SCENE_NAME=!SCENE_%%i!"
    ) else (
        set "SCENE_NAME=!SCENE_CHOICE!"
    )
) else (
    set "SCENE_NAME=!SCENE_CHOICE!"
)
:scene_resolved

if not defined SCENE_NAME set "SCENE_NAME=living-room"

rem pick variant interactively when both exist and none was requested
if "!SCENE_SUFFIX_SET!"=="0" (
    set "HAS_ANIM="
    set "HAS_STATIC="
    if exist "%SCENE_ROOT%\!SCENE_NAME!\scene-animation.pyscene" set "HAS_ANIM=1"
    if exist "%SCENE_ROOT%\!SCENE_NAME!\scene-v4.pyscene" set "HAS_STATIC=1"
    if defined HAS_ANIM if defined HAS_STATIC (
        set "VARIANT_CHOICE="
        rem only prompt when the scene was picked interactively (run.sh parity)
        if defined SCENE_CHOICE set /p VARIANT_CHOICE="Variant: [1] static (scene-v4)  [2] anim (scene-animation) [1]: "
        if defined VARIANT_CHOICE set "VARIANT_CHOICE=!VARIANT_CHOICE: =!"
        if "!VARIANT_CHOICE!"=="2" ( set "SCENE_SUFFIX=anim" ) else ( set "SCENE_SUFFIX=static" )
    ) else if defined HAS_ANIM (
        set "SCENE_SUFFIX=anim"
    )
)

set "SCENE_BASE=%SCENE_ROOT%\%SCENE_NAME%"
set "SCENE_FILE="
if not "%SCENE_SUFFIX%"=="anim" (
    if exist "%SCENE_BASE%\scene-v4.pyscene" set "SCENE_FILE=%SCENE_BASE%\scene-v4.pyscene"
)
if not defined SCENE_FILE (
    if exist "%SCENE_BASE%\scene-animation.pyscene" set "SCENE_FILE=%SCENE_BASE%\scene-animation.pyscene"
)
if not defined SCENE_FILE (
    echo Error: no pyscene found in %SCENE_BASE%
    exit /b 1
)

rem ---------------------------------------------------------------------------
rem Summary + launch.
rem ---------------------------------------------------------------------------
echo ============================================
echo  Script : %SCRIPT_KEY% (!SCRIPT_PATH!)
echo  Scene  : %SCENE_NAME% (%SCENE_SUFFIX%)
echo  File   : %SCENE_FILE%
echo  Window : %WIDTH%x%HEIGHT%
echo  Binary : %BIN%
if defined DEVICE_ARGS echo  Device :%DEVICE_ARGS%
echo ============================================

if defined NCR_RUN_DRYRUN (
    echo [dryrun] "%BIN%"%DEVICE_ARGS% --width %WIDTH% --height %HEIGHT% --script "%ROOT%!SCRIPT_PATH!" --scene "%SCENE_FILE%"
    exit /b 0
)

"%BIN%"%DEVICE_ARGS% --width %WIDTH% --height %HEIGHT% --script "%ROOT%!SCRIPT_PATH!" --scene "%SCENE_FILE%"
