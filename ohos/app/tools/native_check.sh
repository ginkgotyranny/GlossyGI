#!/usr/bin/env bash
# Compiles libentry.so with the OpenHarmony NDK's CMake and ninja, without hvigor: the quickest check
# of the native code. Output in ohos/_build/app_native.
#
#   bash ohos/app/tools/native_check.sh [Release|Debug]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$HERE/.."
SDK="${OHOS_NATIVE_SDK:-/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/native}"
OUT="$APP/../_build/app_native"
CONFIG="${1:-Release}"
CMAKE="$SDK/build-tools/cmake/bin/cmake.exe"
"$CMAKE" -S "$APP/entry/src/main/cpp" -B "$OUT" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$SDK/build-tools/cmake/bin/ninja.exe" \
  -DCMAKE_TOOLCHAIN_FILE="$SDK/build/cmake/ohos.toolchain.cmake" \
  -DOHOS_ARCH=arm64-v8a -DOHOS_PLATFORM=OHOS -DOHOS_STL=c++_shared -DCMAKE_BUILD_TYPE="$CONFIG" > "$OUT.configure.log" 2>&1 ||
  { cat "$OUT.configure.log"; exit 1; }
"$CMAKE" --build "$OUT"
ls -la "$OUT/libentry.so"
