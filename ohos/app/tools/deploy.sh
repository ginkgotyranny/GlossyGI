#!/usr/bin/env bash
# Builds, installs and starts the app on the phone, waits, then brings back its log and a screenshot.
#
#   bash ohos/app/tools/deploy.sh [--no-build] [--release] [--args "<launch arguments>"] [--pull-out] [seconds]
#
#   --args      the app's launch arguments, e.g. "--save-at=60 --loss=30" (see entry/src/main/cpp/app.cpp)
#   --pull-out  also bring back what the app saved (--save-at) in its files folder, as ohos/_build/app_out
#   seconds     how long to let it run before the screenshot (default 30)
# Results: ohos/_build/app.log (the app's complete log), ohos/_build/app_screenshot.jpeg.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$(cd "$HERE/.." && pwd)"
OUT="$APP/../_build"
HDC="${HDC:-/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe}"
BUILD=1
MODE=debug
ARGS=""
PULL=0
WAIT=30
while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) BUILD=0 ;;
    --release) MODE=release ;;
    --args) ARGS="$2"; shift ;;
    --pull-out) PULL=1 ;;
    *) WAIT="$1" ;;
  esac
  shift
done

if [ "$BUILD" = 1 ]; then bash "$HERE/build_hap.sh" "$MODE"; fi   # stops here if the build fails
HAP="$(ls "$APP"/entry/build/default/outputs/default/entry-default-signed.hap 2>/dev/null || true)"
[ -n "$HAP" ] || { echo "no signed HAP: set up the signing (python ohos/app/tools/gen_build_profile.py --help)"; exit 1; }
BUNDLE="$(python -c "import re,sys; print(re.search(r'\"bundleName\"\s*:\s*\"([^\"]+)\"', open(sys.argv[1]).read()).group(1))" "$APP/AppScope/app.json5")"
FILES="/data/app/el2/100/base/$BUNDLE/haps/entry/files"

export MSYS_NO_PATHCONV=1
"$HDC" shell "aa force-stop $BUNDLE" > /dev/null || true
INSTALL="$("$HDC" install -r "$(cygpath -w "$HAP")" 2>&1 | tr -d '\r')"
echo "$INSTALL" | tail -2
echo "$INSTALL" | grep -q "install bundle successfully" || { echo "install failed"; exit 1; }
# A log or results left from an earlier run must not pass for this one's.
"$HDC" shell "rm -rf $FILES/ncr.log $FILES/out; hilog -r" > /dev/null || true
# aa rejects a --ps value that looks like one of its own options: the leading space avoids that.
START="$("$HDC" shell "aa start -a EntryAbility -b $BUNDLE --ps args \" $ARGS\"" 2>&1 | tr -d '\r')"
echo "$START"
echo "$START" | grep -q "start ability successfully" || exit 1
echo "running for $WAIT s"
sleep "$WAIT"

"$HDC" shell "snapshot_display -f /data/local/tmp/ncr_screenshot.jpeg" > /dev/null
(cd "$OUT" && "$HDC" file recv /data/local/tmp/ncr_screenshot.jpeg app_screenshot.jpeg > /dev/null)
if (cd "$OUT" && "$HDC" file recv "$FILES/ncr.log" app.log > /dev/null 2>&1) && [ -s "$OUT/app.log" ]; then
  echo "log: $OUT/app.log"
  tail -25 "$OUT/app.log"
else
  echo "no log file; hilog:"
  "$HDC" shell "hilog -x -T NCR" | tail -40
fi
if [ "$PULL" = 1 ]; then
  rm -rf "$OUT/app_out" "$OUT/app_out.tar"
  "$HDC" shell "cd $FILES && tar -cf /data/local/tmp/ncr_app_out.tar out" > /dev/null
  (cd "$OUT" && "$HDC" file recv /data/local/tmp/ncr_app_out.tar app_out.tar > /dev/null && tar -xf app_out.tar && mv out app_out && rm app_out.tar)
  echo "results: $OUT/app_out"
fi
echo "screenshot: $OUT/app_screenshot.jpeg"
