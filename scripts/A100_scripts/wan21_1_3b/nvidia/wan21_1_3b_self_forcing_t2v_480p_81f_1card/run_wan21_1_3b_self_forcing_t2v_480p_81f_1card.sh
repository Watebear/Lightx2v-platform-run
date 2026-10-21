#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/Wan2.1-T2V-1.3B
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=cuda DEVICE_TYPE=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export INFER_PROFILE_LEVEL=2

source "${lightx2v_path}/scripts/base/base.sh"

prompt="A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated \
city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a \
flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched \
on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp \
and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. \
Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic \
medium shot with the woman walking slightly to one side, highlighting her graceful strides."
negative_prompt=""

python -m lightx2v.infer \
  --model_cls wan2.1_sf \
  --task t2v \
  --model_path "${model_path}" \
  --config_json "${platform_run_path}/scripts/A100_scripts/wan21_1_3b/nvidia/wan21_1_3b_self_forcing_t2v_480p_81f_1card/config.json" \
  --seed 42 \
  --prompt "${prompt}" \
  --negative_prompt "${negative_prompt}" \
  --save_result_path "${lightx2v_path}/save_results/wan21_1_3b_self_forcing_t2v_480p_81f_1card.mp4" \
  "$@"
