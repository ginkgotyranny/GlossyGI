#!/bin/bash
# Unified launcher for Falcor render demos.
# Usage:
#   ./run.sh                     # interactive mode: pick script + scene
#   ./run.sh <script> [scene]    # direct mode
#
# Examples:
#   ./run.sh ncr living-room
#   ./run.sh ncr living-room anim
#   ./run.sh ptnrd kitchen
#   ./run.sh ptdn cornell-box
#   ./run.sh pt
#
# Scene suffix:
#   (default)  -> scene-v4.pyscene        (static)
#   anim       -> scene-animation.pyscene  (animated)

set -e

# --- paths ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/build/linux-gcc/bin/Release/Mogwai"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libpng16.so.16

# --- script registry: short name -> script path ---
declare -A SCRIPTS=(
    [ncr]="scripts/NeuralRadiosity.py"
    [ptnrd]="scripts/PathTracerNRD_VK.py"
    [ptdn]="scripts/OptixDenoiser.py"
    [pt]="scripts/PathTracer.py"
    [mpt]="scripts/MinimalPathTracer.py"
    [gbuffer]="scripts/GBuffer.py"
    [rtxdi]="scripts/RTXDI.py"
    [bsdf]="scripts/BSDFViewer.py"
    [debug]="scripts/SceneDebugger.py"
    [ward]="scripts/WARDiffPathTracer.py"
)

# --- discover scenes ---
SCENE_DIR="$ROOT/media/scenes"
SCENES=()
while IFS= read -r d; do
    SCENES+=("$(basename "$d")")
done < <(find "$SCENE_DIR" -maxdepth 1 -mindepth 1 -type d | sort)

resolve_scene() {
    local name="$1"
    local suffix="${2:-static}"
    local base="$SCENE_DIR/$name"
    local file
    if [[ "$suffix" == "anim" || "$suffix" == "animation" ]]; then
        file="$base/scene-animation.pyscene"
    else
        file="$base/scene-v4.pyscene"
    fi
    if [[ ! -f "$file" ]]; then
        # fall back to whatever exists (pyscene first, then pbrt)
        if [[ -f "$base/scene-animation.pyscene" ]]; then
            file="$base/scene-animation.pyscene"
        elif [[ -f "$base/scene-v4.pyscene" ]]; then
            file="$base/scene-v4.pyscene"
        elif [[ -f "$base/scene-v4.pbrt" ]]; then
            file="$base/scene-v4.pbrt"
        else
            echo "Error: no scene file found in $base" >&2
            return 1
        fi
    fi
    echo "$file"
}

# --- parse args ---
WIDTH=1280
HEIGHT=720
SCRIPT_KEY=""
SCENE_NAME=""
SCENE_SUFFIX="static"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --width|-w)  WIDTH="$2";  shift 2 ;;
        --height|-h) HEIGHT="$2"; shift 2 ;;
        anim|animation) SCENE_SUFFIX="anim"; shift ;;
        *)
            if [[ -z "$SCRIPT_KEY" ]]; then
                SCRIPT_KEY="$1"
            elif [[ -z "$SCENE_NAME" ]]; then
                SCENE_NAME="$1"
            else
                echo "Unknown argument: $1" >&2; exit 1
            fi
            shift ;;
    esac
done

# --- pick script ---
if [[ -z "$SCRIPT_KEY" ]]; then
    echo "Available scripts:"
    local_keys=("${!SCRIPTS[@]}")
    sorted_keys=($(printf '%s\n' "${local_keys[@]}" | sort))
    for i in "${!sorted_keys[@]}"; do
        printf "  [%d] %-10s %s\n" "$((i+1))" "${sorted_keys[$i]}" "${SCRIPTS[${sorted_keys[$i]}]}"
    done
    echo
    read -rp "Select script [1]: " SCRIPT_KEY
    SCRIPT_KEY="${SCRIPT_KEY:-1}"
    # allow numeric index
    if [[ "$SCRIPT_KEY" =~ ^[0-9]+$ ]]; then
        SCRIPT_KEY="${sorted_keys[$((SCRIPT_KEY-1))]}"
    fi
fi

SCRIPT_PATH="${SCRIPTS[$SCRIPT_KEY]}"
if [[ -z "$SCRIPT_PATH" ]]; then
    # allow passing a direct path
    if [[ -f "$ROOT/$SCRIPT_KEY" ]]; then
        SCRIPT_PATH="$SCRIPT_KEY"
    else
        echo "Unknown script: $SCRIPT_KEY" >&2
        exit 1
    fi
fi
SCRIPT_FULL="$ROOT/$SCRIPT_PATH"
if [[ ! -f "$SCRIPT_FULL" ]]; then
    echo "Script not found: $SCRIPT_FULL" >&2
    exit 1
fi

# --- pick scene ---
if [[ -z "$SCENE_NAME" ]]; then
    echo "Available scenes:"
    for i in "${!SCENES[@]}"; do
        printf "  [%d] %s" "$((i+1))" "${SCENES[$i]}"
        # mark which variants exist
        has_anim=""
        has_static=""
        [[ -f "$SCENE_DIR/${SCENES[$i]}/scene-animation.pyscene" ]] && has_anim="anim"
        { [[ -f "$SCENE_DIR/${SCENES[$i]}/scene-v4.pyscene" ]] || [[ -f "$SCENE_DIR/${SCENES[$i]}/scene-v4.pbrt" ]]; } && has_static="static"
        printf " (%s)\n" "$(echo "$has_anim $has_static" | tr ' ' '/' | sed 's|//*|/|; s|^/||; s|/$||')"
    done
    echo
    read -rp "Select scene [living-room]: " SCENE_NAME
    SCENE_NAME="${SCENE_NAME:-living-room}"
    # allow numeric index
    if [[ "$SCENE_NAME" =~ ^[0-9]+$ ]]; then
        SCENE_NAME="${SCENES[$((SCENE_NAME-1))]}"
    fi

    # pick variant if both exist
    if [[ -z "$SCENE_SUFFIX_SET" ]]; then
        has_anim=""
        has_static=""
        [[ -f "$SCENE_DIR/$SCENE_NAME/scene-animation.pyscene" ]] && has_anim="1"
        { [[ -f "$SCENE_DIR/$SCENE_NAME/scene-v4.pyscene" ]] || [[ -f "$SCENE_DIR/$SCENE_NAME/scene-v4.pbrt" ]]; } && has_static="1"
        if [[ -n "$has_anim" && -n "$has_static" ]]; then
            read -rp "Variant: [1] static (scene-v4)  [2] anim (scene-animation) [1]: " VARIANT_CHOICE
            case "${VARIANT_CHOICE:-1}" in
                2|anim|animation) SCENE_SUFFIX="anim" ;;
                *) SCENE_SUFFIX="static" ;;
            esac
        elif [[ -n "$has_anim" && -z "$has_static" ]]; then
            SCENE_SUFFIX="anim"
        fi
    fi
fi

# resolve suffix if user passed it as part of scene name (e.g. "living-room anim")
read -r SCENE_NAME_ONLY SCENE_SUFFIX_EXTRA <<< "$SCENE_NAME"
if [[ -n "$SCENE_SUFFIX_EXTRA" ]]; then
    SCENE_NAME="$SCENE_NAME_ONLY"
    case "$SCENE_SUFFIX_EXTRA" in
        anim|animation) SCENE_SUFFIX="anim" ;;
        static|v4)      SCENE_SUFFIX="static" ;;
    esac
fi

SCENE_FILE="$(resolve_scene "$SCENE_NAME" "$SCENE_SUFFIX")" || exit 1

# --- summary ---
echo "============================================"
echo " Script : $SCRIPT_KEY ($SCRIPT_PATH)"
echo " Scene  : $SCENE_NAME ($SCENE_SUFFIX)"
echo " File   : $SCENE_FILE"
echo " Window : ${WIDTH}x${HEIGHT}"
echo "============================================"

# --- launch ---
exec "$BIN" \
    --width "$WIDTH" --height "$HEIGHT" \
    --script "$SCRIPT_FULL" \
    --scene "$SCENE_FILE"
