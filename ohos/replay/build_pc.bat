@echo off
rem Builds the replayer for Windows with MSVC: ohos\_build\replay\ncr_replay.exe, and the runtime's own
rem shaders next to it (ohos\_build\replay\shaders\*.spv, compiled by Falcor's slangc, see compile_shaders.py).
rem Vulkan headers come from the repository, the import library from build\vulkan-implib
rem (generated from the driver's vulkan-1.dll, see CLAUDE.md), so no Vulkan SDK is needed.
setlocal
set REPO=%~dp0..\..
set OUT=%REPO%\ohos\_build\replay
python "%~dp0compile_shaders.py" "%OUT%\shaders" || exit /b 1

if not defined VSCMD_VER call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" > nul
cl /nologo /std:c++17 /O2 /EHsc /W3 /I "%REPO%\external\vulkan-headers\include" "%~dp0ncr_replay.cpp" "%~dp0ncr_runtime.cpp" ^
   /Fe:"%OUT%\ncr_replay.exe" /Fo:"%OUT%\\" /link "%REPO%\build\vulkan-implib\vulkan-1.lib"
exit /b %ERRORLEVEL%
