#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/HunyuanVideo-1.5
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=cuda DEVICE_TYPE=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export INFER_PROFILE_LEVEL=2

source "${lightx2v_path}/scripts/base/base.sh"

prompt="A close-up shot captures a scene on a polished, light-colored granite kitchen counter, \
illuminated by soft natural light from an unseen window. Initially, the frame focuses on a tall,\
 clear glass filled with golden, translucent apple juice standing next to a single, shiny red \
apple with a green leaf still attached to its stem. The camera moves horizontally to the right. \
As the shot progresses, a white ceramic plate smoothly enters the frame, revealing a fresh \
arrangement of about seven or eight more apples, a mix of vibrant reds and greens, piled neatly \
upon it. A shallow depth of field keeps the focus sharply on the fruit and glass, while the \
kitchen backsplash in the background remains softly blurred. The scene is in a realistic style."
negative_prompt=""

torchrun --nproc_per_node=8 -m lightx2v.infer \
  --model_cls hunyuan_video_1.5 \
  --task t2v \
  --model_path "${model_path}" \
  --config_json "${platform_run_path}/scripts/A100_scripts/hunyuan_video_15/nvidia/hunyuan_video_15_t2v_720p_121f_cfg2_sp4_8card/config.json" \
  --seed 123 \
  --prompt "${prompt}" \
  --negative_prompt "${negative_prompt}" \
  --save_result_path "${lightx2v_path}/save_results/hunyuan_video_15_t2v_720p_121f_cfg2_sp4_8card.mp4" \
  "$@"
