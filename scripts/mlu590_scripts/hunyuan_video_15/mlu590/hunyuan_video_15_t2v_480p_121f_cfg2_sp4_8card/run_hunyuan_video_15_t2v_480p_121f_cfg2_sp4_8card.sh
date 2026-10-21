#!/bin/bash
set -e

# Set these paths for your checkout and model weights.
lightx2v_path=/data/LightX2V
platform_run_path=/data/Lightx2v-platform-run
model_path=/data/models/HunyuanVideo-1.5

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=${MLU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NEUWARE_HOME=${NEUWARE_HOME:-/usr/local/neuware}
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${NEUWARE_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None

source "${lightx2v_path}/scripts/base/base.sh"

prompt="A close-up shot captures a scene on a polished, light-colored granite kitchen counter, \
illuminated by soft natural light from an unseen window. Initially, the frame focuses on \
a tall, clear glass filled with golden, translucent apple juice standing next to a \
single, shiny red apple with a green leaf still attached to its stem. The camera moves \
horizontally to the right. As the shot progresses, a white ceramic plate smoothly enters \
the frame, revealing a fresh arrangement of about seven or eight more apples, a mix of \
vibrant reds and greens, piled neatly upon it. A shallow depth of field keeps the focus \
sharply on the fruit and glass, while the kitchen backsplash in the background remains \
softly blurred. The scene is in a realistic style."

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
    --model_cls hunyuan_video_1.5 \
    --task t2v \
    --model_path "${model_path}" \
    --config_json "${platform_run_path}/scripts/mlu590_scripts/hunyuan_video_15/mlu590/hunyuan_video_15_t2v_480p_121f_cfg2_sp4_8card/config.json" \
    --prompt "${prompt}" \
    --seed 123 \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_video_15_t2v_480p_121f_cfg2_sp4_8card.mp4" \
    "$@"
