# libentry.so: the app's native side (app.cpp) and the NCR runtime it drives (ohos/replay).
# One shared library: the NAPI module registers itself from a static constructor, which a static
# library would lose at link time.
cmake_minimum_required(VERSION 3.5.0)
project(ncr_app CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ohos/app/entry/src/main/cpp -> ohos/replay
set(NCR_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../replay)

add_library(entry SHARED
    app.cpp
    ${NCR_RUNTIME_DIR}/ncr_runtime.cpp)
target_include_directories(entry PRIVATE ${NCR_RUNTIME_DIR})
target_compile_options(entry PRIVATE -O2 -fexceptions -frtti -Wall -Wno-unused-parameter -Wno-unused-function)
# libhiai_foundation.so is not linked: the NPU backend opens it at run time (replay_nnrt.h).
target_link_libraries(entry PUBLIC
    libace_napi.z.so libace_ndk.z.so libhilog_ndk.z.so librawfile.z.so
    libnative_window.so libnative_buffer.so
    libneural_network_core.so libqos.so libvulkan.so)
