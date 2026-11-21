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
rem            --camera-anim on|off     the cameras follow their animation path (default on)
rem            --mirror-albedo on|off   NeuralRadiosity's Mirror Albedo (ncr only, default on)
rem            --train-schedule on|off  NeuralRadiosity's online training schedule: learning rate and
rem                                     EMA decay follow the heat (ncr only, default on; off keeps
rem                                     the hot values)
rem   Interactive mode asks for them after the scene variant.
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
set "SCENE_INTERACTIVE="
set "CAMERA_ANIM="
set "MIRROR_ALBEDO="
set "TRAIN_SCHEDULE="
rem Options for Mogwai's scripts (scripts\run_launch.py, scripts\NeuralRadiosity.py) come from this
rem launcher only: values left in the calling shell must not leak in.
set "NCR_GRAPH_SCRIPT="
set "NCR_CAMERA_ANIMATED="
set "NCR_MIRROR_ALBEDO="
set "NCR_TRAIN_SCHEDULE="

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
if /i "%A%"=="--camera-anim"   ( set "CAMERA_ANIM=%~2" & shift & shift & goto argloop )
if /i "%A%"=="--mirror-albedo" ( set "MIRROR_ALBEDO=%~2" & shift & shift & goto argloop )
if /i "%A%"=="--train-schedule" ( set "TRAIN_SCHEDULE=%~2" & shift & shift & goto argloop )
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
    set "SCENE_INTERACTIVE=1"
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

rem ---------------------------------------------------------------------------
rem Camera animation, Mirror Albedo and the training schedule: asked after the variant when the scene
rem was picked from the menu, unless given on the command line. Mirror Albedo and the training
rem schedule belong to NeuralRadiosity (ncr) only.
rem ---------------------------------------------------------------------------
if defined SCENE_INTERACTIVE (
    if not defined CAMERA_ANIM (
        set "CAM_CHOICE="
        set /p CAM_CHOICE="Camera animation: [1] on  [2] off [1]: "
        if defined CAM_CHOICE set "CAM_CHOICE=!CAM_CHOICE: =!"
        if "!CAM_CHOICE!"=="2" ( set "CAMERA_ANIM=off" ) else ( set "CAMERA_ANIM=on" )
    )
    if /i "!SCRIPT_KEY!"=="ncr" if not defined MIRROR_ALBEDO (
        set "MIRROR_CHOICE="
        set /p MIRROR_CHOICE="Mirror Albedo: [1] on  [2] off [1]: "
        if defined MIRROR_CHOICE set "MIRROR_CHOICE=!MIRROR_CHOICE: =!"
        if "!MIRROR_CHOICE!"=="2" ( set "MIRROR_ALBEDO=off" ) else ( set "MIRROR_ALBEDO=on" )
    )
    if /i "!SCRIPT_KEY!"=="ncr" if not defined TRAIN_SCHEDULE (
        set "SCHEDULE_CHOICE="
        set /p SCHEDULE_CHOICE="Training schedule (LR / EMA heat): [1] on  [2] off [1]: "
        if defined SCHEDULE_CHOICE set "SCHEDULE_CHOICE=!SCHEDULE_CHOICE: =!"
        if "!SCHEDULE_CHOICE!"=="2" ( set "TRAIN_SCHEDULE=off" ) else ( set "TRAIN_SCHEDULE=on" )
    )
)
call :onoff CAMERA_ANIM --camera-anim || exit /b 1
call :onoff MIRROR_ALBEDO --mirror-albedo || exit /b 1
call :onoff TRAIN_SCHEDULE --train-schedule || exit /b 1
if defined MIRROR_ALBEDO if /i not "!SCRIPT_KEY!"=="ncr" (
    echo Note: --mirror-albedo belongs to the ncr script; ignored for !SCRIPT_KEY!
    set "MIRROR_ALBEDO="
)
if defined TRAIN_SCHEDULE if /i not "!SCRIPT_KEY!"=="ncr" (
    echo Note: --train-schedule belongs to the ncr script; ignored for !SCRIPT_KEY!
    set "TRAIN_SCHEDULE="
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
if defined CAMERA_ANIM echo  Camera : animation !CAMERA_ANIM!
if defined MIRROR_ALBEDO echo  Mirror : albedo !MIRROR_ALBEDO!
if defined TRAIN_SCHEDULE echo  Train  : schedule !TRAIN_SCHEDULE!
echo  Window : %WIDTH%x%HEIGHT%
echo  Binary : %BIN%
if defined DEVICE_ARGS echo  Device :%DEVICE_ARGS%
echo ============================================

set "LAUNCH_SCRIPT=%ROOT%!SCRIPT_PATH!"
set "LAUNCH_ARGS="
if /i "!CAMERA_ANIM!"=="off" (
    rem The camera's flag lives in the scene, which Mogwai loads after --script. scripts\run_launch.py,
    rem run after the scene (--deferred), sets it and then runs the chosen script.
    set "NCR_GRAPH_SCRIPT=%ROOT%!SCRIPT_PATH!"
    set "NCR_CAMERA_ANIMATED=0"
    set "LAUNCH_SCRIPT=%ROOT%scripts\run_launch.py"
    set "LAUNCH_ARGS= --deferred"
)
if /i "!MIRROR_ALBEDO!"=="on" set "NCR_MIRROR_ALBEDO=1"
if /i "!MIRROR_ALBEDO!"=="off" set "NCR_MIRROR_ALBEDO=0"
if /i "!TRAIN_SCHEDULE!"=="on" set "NCR_TRAIN_SCHEDULE=1"
if /i "!TRAIN_SCHEDULE!"=="off" set "NCR_TRAIN_SCHEDULE=0"

if defined NCR_RUN_DRYRUN (
    if defined NCR_GRAPH_SCRIPT echo [dryrun] set NCR_GRAPH_SCRIPT=!NCR_GRAPH_SCRIPT!
    if defined NCR_CAMERA_ANIMATED echo [dryrun] set NCR_CAMERA_ANIMATED=!NCR_CAMERA_ANIMATED!
    if defined NCR_MIRROR_ALBEDO echo [dryrun] set NCR_MIRROR_ALBEDO=!NCR_MIRROR_ALBEDO!
    if defined NCR_TRAIN_SCHEDULE echo [dryrun] set NCR_TRAIN_SCHEDULE=!NCR_TRAIN_SCHEDULE!
    echo [dryrun] "%BIN%"%DEVICE_ARGS%!LAUNCH_ARGS! --width %WIDTH% --height %HEIGHT% --script "!LAUNCH_SCRIPT!" --scene "%SCENE_FILE%"
    exit /b 0
)

"%BIN%"%DEVICE_ARGS%!LAUNCH_ARGS! --width %WIDTH% --height %HEIGHT% --script "!LAUNCH_SCRIPT!" --scene "%SCENE_FILE%"
exit /b !ERRORLEVEL!

rem ---------------------------------------------------------------------------
rem :onoff VAR OPTION  normalises an on/off option in place (1/true/yes, 0/false/no); unset stays unset.
rem ---------------------------------------------------------------------------
:onoff
if not defined %1 exit /b 0
set "V=!%1!"
if /i "!V!"=="1" set "%1=on"
if /i "!V!"=="true" set "%1=on"
if /i "!V!"=="yes" set "%1=on"
if /i "!V!"=="0" set "%1=off"
if /i "!V!"=="false" set "%1=off"
if /i "!V!"=="no" set "%1=off"
set "V=!%1!"
if /i "!V!"=="on" exit /b 0
if /i "!V!"=="off" exit /b 0
echo Error: %2 takes on or off, not '!V!'
exit /b 1
