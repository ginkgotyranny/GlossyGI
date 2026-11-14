#!/usr/bin/env bash
# Builds the app's HAP: assets into rawfile, the signing into build-profile.json5, then hvigor.
#
#   bash ohos/app/tools/build_hap.sh [debug|release] [--no-sync]
#
# Uses DevEco Studio's own hvigor, node, ohpm and SDK (DEVECO_HOME overrides where it is installed).
# Output: ohos/app/entry/build/default/outputs/default/entry-default-signed.hap (or -unsigned without a
# signing configuration, see tools/gen_build_profile.py). hvigor's output: ohos/_build/app_build.log.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$(cd "$HERE/.." && pwd)"
LOG="$APP/../_build/app_build.log"
MODE=debug
SYNC=1
for arg in "$@"; do
  case "$arg" in
    debug|release) MODE="$arg" ;;
    --no-sync) SYNC=0 ;;
    *) echo "unknown argument $arg"; exit 2 ;;
  esac
done
DEVECO="${DEVECO_HOME:-/c/Program Files/Huawei/DevEco Studio}"
mkdir -p "$APP/../_build"

if [ "$SYNC" = 1 ]; then python "$HERE/sync_assets.py"; fi
python "$HERE/gen_build_profile.py"

# hvigor wants Windows paths in these two, and node and ohpm on the PATH.
export DEVECO_SDK_HOME="$(cygpath -w "$DEVECO/sdk")"
export NODE_HOME="$(cygpath -w "$DEVECO/tools/node")"
export PATH="$DEVECO/tools/node:$DEVECO/tools/ohpm/bin:$PATH"
cd "$APP"
if [ ! -d "$APP/oh_modules" ] || [ ! -d "$APP/entry/oh_modules" ]; then
  "$DEVECO/tools/ohpm/bin/ohpm.bat" install --all > "$LOG" 2>&1 || { tail -30 "$LOG"; exit 1; }
fi
if ! "$DEVECO/tools/hvigor/bin/hvigorw.bat" --mode module -p product=default -p buildMode="$MODE" -p module=entry@default assembleHap --no-daemon > "$LOG" 2>&1; then
  grep -iE "error|fail" "$LOG" | head -40
  echo "BUILD FAILED, see $LOG"
  exit 1
fi
grep -E "BUILD SUCCESSFUL|WARN.*sign" "$LOG" | tail -3
ls -la "$APP"/entry/build/default/outputs/default/*.hap
