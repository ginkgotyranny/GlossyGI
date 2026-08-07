#!/bin/bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libpng16.so.16 ./build/linux-gcc/bin/Release/Mogwai \
    --width 1280 --height 720 \
    --script scripts/PathTracerNRD.py \
    --scene media/scenes/living-room/scene-animation.pyscene
