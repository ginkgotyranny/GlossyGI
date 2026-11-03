@echo off
rem ============================================================================
rem Launch Mogwai on the Vulkan backend (Windows).
rem Thin wrapper around run.bat: same arguments, same script/scene menus.
rem
rem   run_vulkan.bat                      interactive: pick script + scene
rem   run_vulkan.bat ncr bathroom anim    NeuralRadiosity, animated scene
rem   run_vulkan.bat ptnrd cornell-box    PathTracer + NRD (Vulkan backend)
rem
rem Uses the Vulkan-enabled build tree (build\windows-vs2022-vk) and GPU 0
rem (the RTX; GPU 1 is the AMD iGPU). Override with NCR_BUILD_DIR / NCR_GPU.
rem ============================================================================
setlocal
if not defined NCR_BUILD_DIR set "NCR_BUILD_DIR=build\windows-vs2022-vk"
if not defined NCR_GPU set "NCR_GPU=0"
call "%~dp0run.bat" %* -d vulkan --gpu %NCR_GPU%
endlocal
