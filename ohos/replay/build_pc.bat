@echo off
rem Builds the replayer for Windows with MSVC: ohos\_build\replay\ncr_replay.exe, and its own
rem shaders next to it (ohos\_build\replay\shaders\*.spv, compiled by Falcor's slangc).
rem Vulkan headers come from the repository, the import library from build\vulkan-implib
rem (generated from the driver's vulkan-1.dll, see CLAUDE.md), so no Vulkan SDK is needed.
setlocal
set REPO=%~dp0..\..
set OUT=%REPO%\ohos\_build\replay
set SLANGC=%REPO%\external\packman\slang\bin\slangc.exe
if not exist "%OUT%\shaders" mkdir "%OUT%\shaders"

set SLANG_ARGS=-I "%~dp0shaders" -target spirv -entry main -stage compute
"%SLANGC%" "%~dp0shaders\EncodeInputs.cs.slang" -DSPECULAR=0 %SLANG_ARGS% -o "%OUT%\shaders\EncodeDiff.spv" || exit /b 1
"%SLANGC%" "%~dp0shaders\EncodeInputs.cs.slang" -DSPECULAR=1 %SLANG_ARGS% -o "%OUT%\shaders\EncodeSpec.spv" || exit /b 1
"%SLANGC%" "%~dp0shaders\DecodeOutputs.cs.slang" %SLANG_ARGS% -o "%OUT%\shaders\DecodeOutputs.spv" || exit /b 1

if not defined VSCMD_VER call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" > nul
cl /nologo /std:c++17 /O2 /EHsc /W3 /I "%REPO%\external\vulkan-headers\include" "%~dp0ncr_replay.cpp" ^
   /Fe:"%OUT%\ncr_replay.exe" /Fo:"%OUT%\\" /link "%REPO%\build\vulkan-implib\vulkan-1.lib"
exit /b %ERRORLEVEL%
